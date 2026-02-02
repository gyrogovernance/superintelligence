"""
GyroSpectacles: Topology-Guided Attention Optimization

Tests whether transformer attention heads can be replaced with
horizon-bucket retrieval using the CGM kernel atlas.

This script:
1. Routes tokens through the kernel to assign topological coordinates (horizon, vertex)
2. Captures real attention outputs from OLMo
3. Computes what horizon-bucket attention WOULD produce
4. Measures approximation error and theoretical speedup
5. Optionally runs with actual replacement to measure real speedup

No training. No weight modification. Just inference-time interception.

Metrics:
- Attention approximation error (L2, cosine similarity)
- Output divergence (logit KL, top-k agreement)
- Theoretical FLOPs saved
- Actual timing comparison

Usage:
    python gyrospectacles.py --model-path data/models/OLMo-3-7B-Instruct
    python gyrospectacles.py --model-path data/models/OLMo-3-7B-Instruct --replace
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding


# =============================================================================
# Horizon Labeler: Routes tokens through kernel to get topological coordinates
# =============================================================================

@dataclass
class TopologicalLabels:
    """Per-position topological coordinates from kernel routing."""
    horizons: list[int]          # h ∈ {0..255} per position
    vertices: list[int]          # χ ∈ {0..3} per position
    phases: list[int]            # p ∈ {0..3} per position
    gammas: list[float]          # cumulative gamma per token
    token_ids: list[int]         # original token IDs
    
    def __len__(self) -> int:
        return len(self.horizons)
    
    def horizon_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.horizons, dtype=torch.long, device=device)
    
    def vertex_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.vertices, dtype=torch.long, device=device)


class HorizonLabeler:
    """
    Routes tokens through the kernel to assign topological coordinates.
    
    Uses SemanticTokenCodec for token->bytes, then steps kernel to get
    (horizon, vertex, phase) per position.
    """
    
    def __init__(self, atlas_dir: Path, codec_path: Path):
        from src.router.kernel import RouterKernel
        from src.agent.adapters import SemanticTokenCodec
        
        self.kernel = RouterKernel(atlas_dir)
        self.codec = SemanticTokenCodec.load(codec_path)
        
        # Cache gamma table for scoring
        self.gamma_table = self.kernel.gamma_table
    
    def label_sequence(self, token_ids: list[int]) -> TopologicalLabels:
        """
        Route token sequence through kernel, return per-position labels.
        
        For each token:
        1. Encode to 4 bytes via codec
        2. Step kernel through those bytes
        3. Record final (horizon, vertex, phase) and cumulative gamma
        """
        self.kernel.reset()
        
        horizons = []
        vertices = []
        phases = []
        gammas = []
        
        for tid in token_ids:
            bs = self.codec.encode(tid)
            
            # Track gamma through the 4 bytes
            gamma_sum = 0.0
            chi_prev = self.kernel.current_vertex
            
            for b in bs:
                chi_curr = self.kernel.peek_next_vertex(b)
                w = int(self.kernel.byte_weight[b])
                gamma = float(self.gamma_table[chi_prev, chi_curr, w])
                gamma_sum += gamma
                
                self.kernel.step_byte(b)
                chi_prev = chi_curr
            
            # Record final state for this token
            horizons.append(self.kernel.current_horizon)
            vertices.append(self.kernel.current_vertex)
            phases.append(self.kernel.current_phase)
            gammas.append(gamma_sum)
        
        return TopologicalLabels(
            horizons=horizons,
            vertices=vertices,
            phases=phases,
            gammas=gammas,
            token_ids=token_ids,
        )


# =============================================================================
# Bucket Attention: O(n) horizon-indexed retrieval
# =============================================================================

class BucketAttention:
    """
    Computes attention output via horizon-bucket pooling.
    
    Instead of O(n²) pairwise attention:
    1. Pool V vectors by horizon bucket: V_bucket[h] = mean(V[positions with horizon h])
    2. For each query position, retrieve V_bucket[h_query]
    
    This is O(n) in sequence length.
    """
    
    def __init__(self, n_buckets: int = 256, use_gamma_weighting: bool = False):
        self.n_buckets = n_buckets
        self.use_gamma_weighting = use_gamma_weighting
    
    def compute(
        self,
        V: torch.Tensor,                    # [batch, heads, seq, head_dim]
        query_horizons: torch.Tensor,       # [seq] horizon index per query position
        key_horizons: torch.Tensor,         # [seq] horizon index per key position
        gamma_table: Optional[torch.Tensor] = None,
        query_vertices: Optional[torch.Tensor] = None,
        key_vertices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute bucket-attention output.
        
        Returns: [batch, heads, seq, head_dim] approximating attention output
        """
        batch, heads, seq_len, head_dim = V.shape
        device = V.device
        
        # Pool V by horizon bucket
        # V_pooled[batch, heads, bucket, head_dim]
        V_pooled = torch.zeros(batch, heads, self.n_buckets, head_dim, device=device, dtype=V.dtype)
        bucket_counts = torch.zeros(batch, self.n_buckets, device=device, dtype=torch.float32)
        
        # Accumulate V into buckets
        for pos in range(seq_len):
            h = int(key_horizons[pos].item())
            V_pooled[:, :, h, :] += V[:, :, pos, :]
            bucket_counts[:, h] += 1.0
        
        # Average (avoid div by zero)
        bucket_counts = bucket_counts.clamp(min=1.0)
        for h in range(self.n_buckets):
            V_pooled[:, :, h, :] /= bucket_counts[:, h].unsqueeze(1).unsqueeze(-1)
        
        # Retrieve for each query position
        output = torch.zeros_like(V)
        
        if self.use_gamma_weighting and gamma_table is not None and query_vertices is not None and key_vertices is not None:
            # Weighted retrieval using gamma
            output = self._gamma_weighted_retrieval(
                V_pooled, query_horizons, key_horizons, 
                gamma_table, query_vertices, key_vertices, bucket_counts
            )
        else:
            # Simple retrieval: just take the bucket matching query horizon
            for pos in range(seq_len):
                h = int(query_horizons[pos].item())
                output[:, :, pos, :] = V_pooled[:, :, h, :]
        
        return output
    
    def _gamma_weighted_retrieval(
        self,
        V_pooled: torch.Tensor,
        query_horizons: torch.Tensor,
        key_horizons: torch.Tensor,
        gamma_table: torch.Tensor,
        query_vertices: torch.Tensor,
        key_vertices: torch.Tensor,
        bucket_counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted retrieval: blend buckets using gamma as soft attention.
        
        For each query, compute gamma-weight for each bucket and blend.
        This is still O(n * n_buckets) = O(n) since n_buckets is fixed at 256.
        """
        batch, heads, n_buckets, head_dim = V_pooled.shape
        seq_len = query_horizons.shape[0]
        device = V_pooled.device
        
        output = torch.zeros(batch, heads, seq_len, head_dim, device=device, dtype=V_pooled.dtype)
        
        # Precompute bucket-level vertex (mode of vertices in each bucket)
        # For simplicity, use the vertex of first position in each bucket
        bucket_vertex = torch.zeros(n_buckets, dtype=torch.long, device=device)
        for pos in range(seq_len):
            h = int(key_horizons[pos].item())
            if bucket_counts[0, h] == 1:  # First occurrence
                bucket_vertex[h] = key_vertices[pos]
        
        for pos in range(seq_len):
            h_q = int(query_horizons[pos].item())
            chi_q = int(query_vertices[pos].item())
            
            # Compute gamma weights for all buckets
            weights = torch.zeros(n_buckets, device=device)
            for h_k in range(n_buckets):
                if bucket_counts[0, h_k] > 0:
                    chi_k = int(bucket_vertex[h_k].item())
                    # Use weight=6 (middle value) as proxy
                    gamma = float(gamma_table[chi_q, chi_k, 6])
                    # Boost same-horizon bucket
                    if h_k == h_q:
                        gamma += 2.0
                    weights[h_k] = gamma
            
            # Softmax over non-empty buckets
            mask = bucket_counts[0] > 0
            weights[~mask] = float('-inf')
            weights = F.softmax(weights, dim=0)
            
            # Weighted sum of V_pooled
            for h_k in range(n_buckets):
                if mask[h_k]:
                    output[:, :, pos, :] += weights[h_k] * V_pooled[:, :, h_k, :]
        
        return output
    
    def theoretical_flops_saved(self, seq_len: int, head_dim: int, n_heads: int) -> dict[str, Any]:
        """
        Compute theoretical FLOPs saved by using bucket attention.
        
        Standard attention per head:
        - Q @ K.T: seq × seq × head_dim (2 * seq² * head_dim FLOPs)
        - softmax: seq × seq
        - weights @ V: seq × seq × head_dim (2 * seq² * head_dim FLOPs)
        Total: ~4 * seq² * head_dim per head
        
        Bucket attention per head:
        - Pool V into 256 buckets: seq * head_dim
        - Retrieve for each position: seq * head_dim (or seq * 256 * head_dim with gamma)
        Total: ~2 * seq * head_dim (simple) or ~512 * seq * head_dim (gamma-weighted)
        """
        standard_flops = 4 * seq_len * seq_len * head_dim * n_heads
        bucket_simple_flops = 2 * seq_len * head_dim * n_heads
        bucket_gamma_flops = 512 * seq_len * head_dim * n_heads
        
        return {
            "standard_attention_flops": standard_flops,
            "bucket_simple_flops": bucket_simple_flops,
            "bucket_gamma_flops": bucket_gamma_flops,
            "speedup_simple": standard_flops / bucket_simple_flops,
            "speedup_gamma": standard_flops / bucket_gamma_flops,
            "seq_len": seq_len,
            "crossover_seq_len": 256 * 2,  # Where bucket becomes faster
        }


# =============================================================================
# Attention Capture: Hooks to intercept real attention outputs
# =============================================================================

@dataclass
class CapturedAttention:
    """Captured attention data from a forward pass."""
    layer: int
    head: int
    attention_weights: torch.Tensor  # [batch, heads, seq, seq] or per-head slice
    attention_output: torch.Tensor   # [batch, seq, hidden] before output projection
    V: torch.Tensor                  # [batch, heads, seq, head_dim] value vectors
    Q: Optional[torch.Tensor] = None
    K: Optional[torch.Tensor] = None


class AttentionCaptureHooks:
    """
    Registers hooks to capture attention internals during forward pass.
    """
    
    def __init__(self, model: torch.nn.Module, target_layers: list[int]):
        self.model = model
        self.target_layers = target_layers
        self.captures: dict[int, dict[str, torch.Tensor]] = {}
        self.hooks: list[Any] = []
        
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        for layer_idx in self.target_layers:
            layer = self.model.model.layers[layer_idx]
            
            # Hook into self_attn to capture Q, K, V and outputs
            hook = layer.self_attn.register_forward_hook(
                self._make_capture_hook(layer_idx)
            )
            self.hooks.append(hook)
    
    def _make_capture_hook(self, layer_idx: int) -> Any:
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], outputs: Any) -> None:
            # outputs is typically (attn_output, attn_weights, past_key_value)
            # or just attn_output depending on config
            
            hidden_states = inputs[0]  # Input to attention
            
            # Get Q, K, V by accessing the projections
            # This is model-specific; for OLMo:
            batch, seq_len, hidden = hidden_states.shape
            
            Q = module.q_proj(hidden_states)
            K = module.k_proj(hidden_states)
            V = module.v_proj(hidden_states)
            
            # Reshape to [batch, heads, seq, head_dim]
            num_heads = module.num_heads
            head_dim = hidden // num_heads
            
            Q = Q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
            K = K.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
            V = V.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
            
            # Compute attention weights manually for analysis
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            # Store captures
            self.captures[layer_idx] = {
                "Q": Q.detach(),
                "K": K.detach(),
                "V": V.detach(),
                "attn_weights": attn_weights.detach(),
                "attn_output": outputs[0].detach() if isinstance(outputs, tuple) else outputs.detach(),
            }
        
        return hook
    
    def clear(self) -> None:
        self.captures.clear()
    
    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


# =============================================================================
# Spectacles Replacement: Actually substitute bucket attention
# =============================================================================

class SpectaclesModule(torch.nn.Module):
    """
    Wrapper that replaces attention computation for specified heads.
    
    For replaced heads: use bucket attention
    For other heads: use original attention
    """
    
    def __init__(
        self,
        original_attn: torch.nn.Module,
        replaced_heads: list[int],
        bucket_attention: BucketAttention,
        labels: TopologicalLabels,
        gamma_table: torch.Tensor,
    ):
        super().__init__()
        self.original_attn = original_attn
        self.replaced_heads = set(replaced_heads)
        self.bucket_attention = bucket_attention
        self.labels = labels
        self.gamma_table = gamma_table
        
        self.num_heads = original_attn.num_heads
        self.head_dim = original_attn.head_dim
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, ...]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, Optional[tuple[torch.Tensor, ...]]]:
        batch, seq_len, hidden = hidden_states.shape
        device = hidden_states.device
        
        # Get Q, K, V from original projections
        Q = self.original_attn.q_proj(hidden_states)
        K = self.original_attn.k_proj(hidden_states)
        V = self.original_attn.v_proj(hidden_states)
        
        # Reshape
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings if present (model-specific)
        if hasattr(self.original_attn, 'rotary_emb') and position_ids is not None:
            cos, sin = self.original_attn.rotary_emb(V, position_ids)
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        
        # Compute attention output per head
        scale = self.head_dim ** -0.5
        attn_outputs = []
        
        horizons = self.labels.horizon_tensor(device)
        vertices = self.labels.vertex_tensor(device)
        
        for head_idx in range(self.num_heads):
            if head_idx in self.replaced_heads:
                # Use bucket attention for this head
                V_head = V[:, head_idx:head_idx+1, :, :]  # [batch, 1, seq, head_dim]
                
                bucket_out = self.bucket_attention.compute(
                    V=V_head,
                    query_horizons=horizons,
                    key_horizons=horizons,
                    gamma_table=self.gamma_table,
                    query_vertices=vertices,
                    key_vertices=vertices,
                )
                attn_outputs.append(bucket_out)
            else:
                # Use standard attention for this head
                Q_head = Q[:, head_idx, :, :]  # [batch, seq, head_dim]
                K_head = K[:, head_idx, :, :]
                V_head = V[:, head_idx, :, :]
                
                scores = torch.matmul(Q_head, K_head.transpose(-2, -1)) * scale
                
                if attention_mask is not None:
                    scores = scores + attention_mask
                
                weights = F.softmax(scores, dim=-1)
                out = torch.matmul(weights, V_head)
                attn_outputs.append(out.unsqueeze(1))
        
        # Concatenate heads
        attn_output = torch.cat(attn_outputs, dim=1)  # [batch, heads, seq, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        
        # Output projection
        attn_output = self.original_attn.o_proj(attn_output)
        
        return (attn_output, None, past_key_value)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings (simplified)."""
    # This is a simplified version; real implementation depends on model
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# =============================================================================
# Main Experiment
# =============================================================================

@dataclass
class ExperimentConfig:
    model_path: Path
    atlas_dir: Path = Path("data/atlas")
    codec_path: Optional[Path] = None
    target_layers: list[int] = field(default_factory=lambda: [17])
    target_heads: list[int] = field(default_factory=lambda: [27])
    test_prompts: list[str] = field(default_factory=lambda: [
        "The fundamental principles of quantum mechanics state that",
        "In the year 2050, artificial intelligence had become",
        "The mathematical proof begins by assuming that",
    ])
    run_replacement: bool = False
    use_gamma_weighting: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ExperimentResults:
    # Approximation quality
    attention_l2_error: dict[str, float] = field(default_factory=dict)
    attention_cosine_sim: dict[str, float] = field(default_factory=dict)
    output_l2_error: dict[str, float] = field(default_factory=dict)
    
    # Output divergence
    logit_kl_divergence: float = 0.0
    top1_agreement: float = 0.0
    top5_agreement: float = 0.0
    
    # Performance
    baseline_time_ms: float = 0.0
    spectacles_time_ms: float = 0.0
    speedup_ratio: float = 0.0
    theoretical_speedup: float = 0.0
    
    # Topological stats
    gamma_stats: dict[str, float] = field(default_factory=dict)
    horizon_coverage: int = 0
    vertex_distribution: list[int] = field(default_factory=list)


class GyroSpectaclesExperiment:
    """
    Main experiment class for GyroSpectacles.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print("=" * 60)
        print("GyroSpectacles: Topology-Guided Attention Optimization")
        print("=" * 60)
        
        # Load model
        print(f"\nLoading model from {config.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16,
            device_map=config.device,
        )
        self.model.eval()
        print(f"  Model loaded: {self.model.config.num_hidden_layers} layers, "
              f"{self.model.config.num_attention_heads} heads")
        
        # Load kernel and codec
        print(f"\nLoading kernel from {config.atlas_dir}...")
        codec_path = config.codec_path or (config.atlas_dir / "semantic_codec.npz")
        if not codec_path.exists():
            raise FileNotFoundError(
                f"Semantic codec not found at {codec_path}. "
                "Build it first with: python -m src.agent.build_codec"
            )
        
        self.labeler = HorizonLabeler(config.atlas_dir, codec_path)
        print("  Kernel loaded: 65536 states, 256 horizons")
        
        # Initialize bucket attention
        self.bucket_attention = BucketAttention(
            n_buckets=256,
            use_gamma_weighting=config.use_gamma_weighting,
        )
        
        # Prepare gamma table as tensor
        self.gamma_table = torch.from_numpy(
            self.labeler.kernel.gamma_table
        ).to(self.device)
        
        print("\nTarget heads for replacement:")
        for layer in config.target_layers:
            for head in config.target_heads:
                print(f"  Layer {layer}, Head {head}")
    
    def run(self) -> ExperimentResults:
        """Run the full experiment."""
        results = ExperimentResults()
        
        # Temporary storage for aggregation
        l2_error_lists: dict[str, list[float]] = {}
        cosine_sim_lists: dict[str, list[float]] = {}
        
        for prompt_idx, prompt in enumerate(self.config.test_prompts):
            print(f"\n{'='*60}")
            print(f"Prompt {prompt_idx + 1}/{len(self.config.test_prompts)}")
            print(f"{'='*60}")
            print(f"Text: {prompt[:50]}...")
            
            prompt_results = self._run_single_prompt(prompt)
            
            # Aggregate results
            for key, value in prompt_results.attention_l2_error.items():
                if key not in l2_error_lists:
                    l2_error_lists[key] = []
                l2_error_lists[key].append(value)
            
            for key, value in prompt_results.attention_cosine_sim.items():
                if key not in cosine_sim_lists:
                    cosine_sim_lists[key] = []
                cosine_sim_lists[key].append(value)
        
        # Average across prompts
        for key, values in l2_error_lists.items():
            results.attention_l2_error[key] = sum(values) / len(values)
        
        for key, values in cosine_sim_lists.items():
            results.attention_cosine_sim[key] = sum(values) / len(values)
        
        self._print_summary(results)
        
        return results
    
    def _run_single_prompt(self, prompt: str) -> ExperimentResults:
        """Run experiment on a single prompt."""
        results = ExperimentResults()
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        token_ids = inputs.input_ids[0].tolist()
        seq_len = len(token_ids)
        
        print(f"\n  Sequence length: {seq_len} tokens")
        
        # Get topological labels
        labels = self.labeler.label_sequence(token_ids)
        
        # Topological stats
        results.horizon_coverage = len(set(labels.horizons))
        results.vertex_distribution = [
            sum(1 for v in labels.vertices if v == i) for i in range(4)
        ]
        results.gamma_stats = {
            "mean": float(np.mean(labels.gammas)),
            "std": float(np.std(labels.gammas)),
            "min": float(np.min(labels.gammas)),
            "max": float(np.max(labels.gammas)),
        }
        
        print(f"  Horizon coverage: {results.horizon_coverage}/256")
        print(f"  Vertex distribution: {results.vertex_distribution}")
        print(f"  Gamma: mean={results.gamma_stats['mean']:.3f}, "
              f"std={results.gamma_stats['std']:.3f}")
        
        # Register hooks to capture attention
        hooks = AttentionCaptureHooks(self.model, self.config.target_layers)
        
        # Baseline forward pass
        print("\n  Running baseline forward pass...")
        t0 = time.perf_counter()
        
        with torch.no_grad():
            baseline_outputs = self.model(**inputs)
        
        results.baseline_time_ms = (time.perf_counter() - t0) * 1000
        print(f"  Baseline time: {results.baseline_time_ms:.2f} ms")
        
        # Analyze captured attention
        for layer_idx in self.config.target_layers:
            if layer_idx not in hooks.captures:
                print(f"  Warning: Layer {layer_idx} not captured")
                continue
            
            capture = hooks.captures[layer_idx]
            V = capture["V"]
            attn_weights = capture["attn_weights"]
            
            # Compute bucket attention output
            horizons = labels.horizon_tensor(self.device)
            vertices = labels.vertex_tensor(self.device)
            
            for head_idx in self.config.target_heads:
                key = f"L{layer_idx}H{head_idx}"
                
                # Get real attention output for this head
                V_head = V[:, head_idx:head_idx+1, :, :]
                real_weights = attn_weights[:, head_idx, :, :]
                real_output = torch.matmul(real_weights, V[:, head_idx, :, :])
                
                # Get bucket attention output
                bucket_output = self.bucket_attention.compute(
                    V=V_head,
                    query_horizons=horizons,
                    key_horizons=horizons,
                    gamma_table=self.gamma_table if self.config.use_gamma_weighting else None,
                    query_vertices=vertices if self.config.use_gamma_weighting else None,
                    key_vertices=vertices if self.config.use_gamma_weighting else None,
                )
                bucket_output = bucket_output.squeeze(1)  # [batch, seq, head_dim]
                
                # Compute errors
                l2_error = torch.norm(real_output - bucket_output).item()
                l2_norm = torch.norm(real_output).item()
                relative_l2 = l2_error / (l2_norm + 1e-8)
                
                cosine_sim = F.cosine_similarity(
                    real_output.flatten(),
                    bucket_output.flatten(),
                    dim=0
                ).item()
                
                results.attention_l2_error[key] = relative_l2
                results.attention_cosine_sim[key] = cosine_sim
                
                print(f"\n  {key}:")
                print(f"    Relative L2 error: {relative_l2:.4f}")
                print(f"    Cosine similarity: {cosine_sim:.4f}")
                
                # Analyze attention pattern vs horizon match
                self._analyze_attention_pattern(
                    attn_weights[:, head_idx, :, :],
                    horizons,
                    key
                )
        
        # Theoretical speedup
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        n_replaced = len(self.config.target_layers) * len(self.config.target_heads)
        
        flops_info = self.bucket_attention.theoretical_flops_saved(
            seq_len=seq_len,
            head_dim=head_dim,
            n_heads=n_replaced,
        )
        
        results.theoretical_speedup = flops_info["speedup_simple"]
        print(f"\n  Theoretical speedup (simple bucket): {results.theoretical_speedup:.1f}x")
        print(f"  (at seq_len={seq_len}, replacing {n_replaced} heads)")
        
        # Run with actual replacement if requested
        if self.config.run_replacement:
            print("\n  Running with spectacles replacement...")
            results = self._run_with_replacement(inputs, labels, baseline_outputs, results)
        
        hooks.remove_hooks()
        hooks.clear()
        
        return results
    
    def _analyze_attention_pattern(
        self,
        attn_weights: torch.Tensor,
        horizons: torch.Tensor,
        key: str,
    ) -> None:
        """Analyze how well attention aligns with horizon structure."""
        seq_len = attn_weights.shape[-1]
        
        # For each query, what fraction of attention goes to same-horizon keys?
        same_horizon_mass = 0.0
        
        for q in range(seq_len):
            h_q = horizons[q].item()
            
            # Mask for same horizon
            same_h_mask = (horizons == h_q).float()
            
            # Attention mass on same horizon
            mass = (attn_weights[0, q, :] * same_h_mask).sum().item()
            same_horizon_mass += mass
        
        avg_same_horizon = same_horizon_mass / seq_len
        
        # Expected if uniform random
        horizon_counts = torch.bincount(horizons, minlength=256)  # type: ignore[attr-defined]
        expected_random = (horizon_counts.float() / seq_len).sum().item() / 256
        
        enrichment = avg_same_horizon / (expected_random + 1e-8)
        
        print(f"    Same-horizon attention mass: {avg_same_horizon:.4f}")
        print(f"    Expected (random): {expected_random:.4f}")
        print(f"    Enrichment: {enrichment:.2f}x")
    
    def _run_with_replacement(
        self,
        inputs: BatchEncoding,
        labels: TopologicalLabels,
        baseline_outputs: Any,
        results: ExperimentResults,
    ) -> ExperimentResults:
        """Run forward pass with actual attention replacement."""
        
        # Save original attention modules
        original_attns: dict[int, torch.nn.Module] = {}
        
        for layer_idx in self.config.target_layers:
            layer = self.model.model.layers[layer_idx]
            original_attns[layer_idx] = layer.self_attn
            
            # Replace with spectacles module
            layer.self_attn = SpectaclesModule(
                original_attn=layer.self_attn,
                replaced_heads=self.config.target_heads,
                bucket_attention=self.bucket_attention,
                labels=labels,
                gamma_table=self.gamma_table,
            )
        
        # Forward pass with replacement
        t0 = time.perf_counter()
        
        with torch.no_grad():
            spectacles_outputs = self.model(**inputs)
        
        results.spectacles_time_ms = (time.perf_counter() - t0) * 1000
        
        # Restore original modules
        for layer_idx in self.config.target_layers:
            self.model.model.layers[layer_idx].self_attn = original_attns[layer_idx]
        
        # Compare outputs
        baseline_logits = baseline_outputs.logits
        spectacles_logits = spectacles_outputs.logits
        
        if baseline_logits is None or spectacles_logits is None:
            print("  Warning: Could not compare logits (None returned)")
            return results
        
        # KL divergence
        baseline_probs = F.softmax(baseline_logits, dim=-1)
        spectacles_log_probs = F.log_softmax(spectacles_logits, dim=-1)
        kl_div = F.kl_div(spectacles_log_probs, baseline_probs, reduction='batchmean').item()
        results.logit_kl_divergence = kl_div
        
        # Top-k agreement
        top1_baseline = baseline_logits.argmax(dim=-1)
        top1_spectacles = spectacles_logits.argmax(dim=-1)
        results.top1_agreement = (top1_baseline == top1_spectacles).float().mean().item()
        
        _, top5_baseline = baseline_logits.topk(5, dim=-1)
        
        # Check if top-1 of spectacles is in top-5 of baseline
        top5_agreement = 0.0
        for i in range(top5_baseline.shape[1]):
            if top1_spectacles[0, i] in top5_baseline[0, i]:
                top5_agreement += 1
        results.top5_agreement = top5_agreement / top5_baseline.shape[1]
        
        results.speedup_ratio = results.baseline_time_ms / results.spectacles_time_ms
        
        print(f"\n  Spectacles time: {results.spectacles_time_ms:.2f} ms")
        print(f"  Actual speedup: {results.speedup_ratio:.2f}x")
        print(f"  Logit KL divergence: {kl_div:.6f}")
        print(f"  Top-1 agreement: {results.top1_agreement:.2%}")
        print(f"  Top-5 agreement: {results.top5_agreement:.2%}")
        
        return results
    
    def _print_summary(self, results: ExperimentResults) -> None:
        """Print final summary."""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        print("\nApproximation Quality (averaged across prompts):")
        for key in sorted(results.attention_l2_error.keys()):
            l2 = results.attention_l2_error[key]
            cos = results.attention_cosine_sim[key]
            print(f"  {key}: L2={l2:.4f}, cosine={cos:.4f}")
        
        if results.spectacles_time_ms > 0:
            print("\nPerformance:")
            print(f"  Baseline: {results.baseline_time_ms:.2f} ms")
            print(f"  Spectacles: {results.spectacles_time_ms:.2f} ms")
            print(f"  Speedup: {results.speedup_ratio:.2f}x")
            
            print("\nOutput Fidelity:")
            print(f"  KL divergence: {results.logit_kl_divergence:.6f}")
            print(f"  Top-1 agreement: {results.top1_agreement:.2%}")
            print(f"  Top-5 agreement: {results.top5_agreement:.2%}")
        
        print(f"\nTheoretical speedup: {results.theoretical_speedup:.1f}x")
        print("  (speedup grows with sequence length)")


# =============================================================================
# Entry Point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GyroSpectacles: Topology-Guided Attention Optimization"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/models/OLMo-3-7B-Instruct"),
        help="Path to OLMo model",
    )
    parser.add_argument(
        "--atlas-dir",
        type=Path,
        default=Path("data/atlas"),
        help="Path to kernel atlas directory",
    )
    parser.add_argument(
        "--codec-path",
        type=Path,
        default=None,
        help="Path to semantic codec (default: atlas_dir/semantic_codec.npz)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[17],
        help="Target layers for replacement",
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=[27],
        help="Target heads for replacement",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Actually replace attention and measure speedup",
    )
    parser.add_argument(
        "--gamma-weighting",
        action="store_true",
        help="Use gamma-weighted bucket retrieval",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Custom test prompts",
    )
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        model_path=args.model_path,
        atlas_dir=args.atlas_dir,
        codec_path=args.codec_path,
        target_layers=args.layers,
        target_heads=args.heads,
        run_replacement=args.replace,
        use_gamma_weighting=args.gamma_weighting,
    )
    
    if args.prompts:
        config.test_prompts = args.prompts
    
    experiment = GyroSpectaclesExperiment(config)
    results = experiment.run()
    
    # Final verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    avg_cosine = float(np.mean(list(results.attention_cosine_sim.values()))) if results.attention_cosine_sim else 0.0
    avg_l2 = float(np.mean(list(results.attention_l2_error.values()))) if results.attention_l2_error else 1.0
    
    if avg_cosine > 0.9 and avg_l2 < 0.2:
        print("✓ VIABLE: Bucket attention closely approximates real attention")
        print("  Proceed to full replacement and scaling tests")
    elif avg_cosine > 0.7:
        print("◐ PROMISING: Moderate approximation quality")
        print("  Consider gamma-weighting or targeting different heads")
    else:
        print("✗ NOT VIABLE: Poor approximation quality")
        print("  These heads may not be topology-addressable")


if __name__ == "__main__":
    main()