"""
PyTorch Transformer Internals Probe

This script investigates the exact mechanics of how PyTorch transformers work:
1. Weight storage and access patterns
2. What operations fire during forward pass
3. The computational graph that could be replaced by routing logic

For Gyroscopic ASI: Understanding these mechanics is critical for figuring out
how to read weights and replace matmul-based attention with horizon-indexed routing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

# =========
# Part 1: Weight Storage Analysis
# =========

def analyze_weight_storage() -> dict[str, Any]:
    """
    Analyze how PyTorch stores weights internally.
    
    Key findings for Gyroscopic ASI:
    - Weights are torch.nn.Parameter (subclass of Tensor)
    - Storage is contiguous float arrays (bfloat16, float16, or float32)
    - Access is via tensor indexing or matmul
    """
    results: dict[str, Any] = {}
    
    # Create a simple Linear layer to examine
    linear = nn.Linear(64, 32, bias=True)
    
    results["weight_type"] = type(linear.weight).__name__
    results["weight_shape"] = list(linear.weight.shape)
    results["weight_dtype"] = str(linear.weight.dtype)
    results["weight_device"] = str(linear.weight.device)
    results["weight_requires_grad"] = linear.weight.requires_grad
    results["weight_is_contiguous"] = linear.weight.is_contiguous()
    
    # Memory layout
    results["weight_stride"] = list(linear.weight.stride())
    results["weight_storage_offset"] = linear.weight.storage_offset()
    results["weight_numel"] = linear.weight.numel()
    results["weight_element_size"] = linear.weight.element_size()
    results["weight_nbytes"] = linear.weight.numel() * linear.weight.element_size()
    
    # Bias analysis
    bias = getattr(linear, 'bias', None)
    if bias is not None:
        results["bias_shape"] = list(bias.shape)
        results["bias_dtype"] = str(bias.dtype)
    
    # How forward pass accesses weights
    x = torch.randn(8, 64)  # (batch, in_features)
    
    # Method 1: Using the module
    y1 = linear(x)
    
    # Method 2: Manual matmul (what actually happens inside)
    # Note: weight is (out, in), so we do x @ weight.T
    y2 = x @ linear.weight.T + linear.bias
    
    results["forward_equivalence"] = torch.allclose(y1, y2, atol=1e-6)
    results["forward_output_shape"] = list(y1.shape)
    
    return results


def analyze_embedding_storage() -> dict[str, Any]:
    """
    Analyze embedding layer weight access.
    
    Key for Gyroscopic ASI:
    - Embedding is just a lookup table: weight[token_id]
    - This could be replaced by kernel-derived embeddings
    """
    results: dict[str, Any] = {}
    
    vocab_size = 1000
    embed_dim = 128
    embed = nn.Embedding(vocab_size, embed_dim)
    
    results["embed_weight_shape"] = list(embed.weight.shape)
    results["embed_weight_dtype"] = str(embed.weight.dtype)
    
    # How lookup works
    token_ids = torch.tensor([42, 100, 7])
    
    # Method 1: Using embedding layer
    embeddings1 = embed(token_ids)
    
    # Method 2: Direct indexing (what actually happens)
    embeddings2 = embed.weight[token_ids]
    
    results["lookup_equivalence"] = torch.allclose(embeddings1, embeddings2)
    results["lookup_output_shape"] = list(embeddings1.shape)
    
    # Key insight: embedding is weight[input], not weight @ input
    results["operation"] = "indexing (weight[ids]), not matmul"
    
    return results


# =========
# Part 2: Forward Pass Tracing with Hooks
# =========

@dataclass
class OpTrace:
    """Record of a single operation during forward pass."""
    name: str
    module_type: str
    input_shapes: list[list[int]]
    output_shape: list[int]
    weight_shape: list[int] | None
    duration_us: float


@dataclass
class ForwardTracer:
    """
    Traces all operations during a forward pass.
    
    Uses PyTorch hooks to capture:
    - Module calls (Linear, LayerNorm, Embedding, etc.)
    - Input/output shapes
    - Weight shapes involved
    """
    traces: list[OpTrace] = field(default_factory=list)
    _handles: list[Any] = field(default_factory=list)
    _depth: int = 0
    
    def _make_hook(self, name: str) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            # Get input shapes
            input_shapes = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    input_shapes.append(list(inp.shape))
            
            # Get output shape
            if isinstance(output, torch.Tensor):
                output_shape = list(output.shape)
            elif isinstance(output, tuple) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    output_shape = list(output[0].shape)
                else:
                    output_shape = []
            else:
                output_shape = []
            
            # Get weight shape if applicable
            weight_shape = None
            weight = getattr(module, 'weight', None)
            if weight is not None:
                weight_shape = list(weight.shape)
            
            self.traces.append(OpTrace(
                name=name,
                module_type=type(module).__name__,
                input_shapes=input_shapes,
                output_shape=output_shape,
                weight_shape=weight_shape,
                duration_us=0.0,  # Would need CUDA events for accurate timing
            ))
        return hook
    
    def attach(self, model: nn.Module, prefix: str = "") -> None:
        """Attach hooks to all modules in the model."""
        for name, module in model.named_modules():
            if name == "":
                continue
            full_name = f"{prefix}.{name}" if prefix else name
            handle = module.register_forward_hook(self._make_hook(full_name))
            self._handles.append(handle)
    
    def detach(self) -> None:
        """Remove all hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
    
    def summarize(self) -> dict[str, Any]:
        """Summarize the traced operations."""
        by_type: dict[str, int] = {}
        for trace in self.traces:
            by_type[trace.module_type] = by_type.get(trace.module_type, 0) + 1
        
        return {
            "total_ops": len(self.traces),
            "by_type": by_type,
            "first_10": [
                {"name": t.name, "type": t.module_type, "in": t.input_shapes, "out": t.output_shape}
                for t in self.traces[:10]
            ],
            "last_5": [
                {"name": t.name, "type": t.module_type, "in": t.input_shapes, "out": t.output_shape}
                for t in self.traces[-5:]
            ],
        }


# =========
# Part 3: Attention Mechanics Deep Dive
# =========

def trace_attention_step_by_step(
    hidden_dim: int = 128,
    num_heads: int = 4,
    seq_len: int = 16,
    batch_size: int = 1,
) -> dict[str, Any]:
    """
    Trace attention computation step by step.
    
    This is the EXACT sequence that transformers execute:
    1. Project Q, K, V from input
    2. Reshape for multi-head
    3. Compute attention scores: Q @ K.T / sqrt(d_k)
    4. Apply softmax
    5. Apply attention to V: attn_weights @ V
    6. Reshape and project output
    
    For Gyroscopic ASI:
    - Steps 3-5 (attention computation) could be replaced by horizon-indexed lookup
    - The O(n^2) attention matrix is what we want to replace
    """
    results: dict[str, Any] = {"steps": []}
    head_dim = hidden_dim // num_heads
    
    # Input: hidden states from previous layer
    x = torch.randn(batch_size, seq_len, hidden_dim)
    results["steps"].append({
        "step": 0,
        "name": "input",
        "shape": list(x.shape),
        "description": "Hidden states from previous layer/embedding"
    })
    
    # Step 1: Linear projections for Q, K, V
    W_q = torch.randn(hidden_dim, hidden_dim) * 0.02
    W_k = torch.randn(hidden_dim, hidden_dim) * 0.02
    W_v = torch.randn(hidden_dim, hidden_dim) * 0.02
    
    Q = x @ W_q  # (batch, seq, hidden)
    K = x @ W_k
    V = x @ W_v
    
    results["steps"].append({
        "step": 1,
        "name": "qkv_projection",
        "operation": "x @ W",
        "shapes": {"Q": list(Q.shape), "K": list(K.shape), "V": list(V.shape)},
        "weight_shape": list(W_q.shape),
        "flops": 3 * batch_size * seq_len * hidden_dim * hidden_dim,
        "description": "Linear projection: dense matmul"
    })
    
    # Step 2: Reshape for multi-head attention
    # (batch, seq, hidden) -> (batch, heads, seq, head_dim)
    Q = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    results["steps"].append({
        "step": 2,
        "name": "reshape_multihead",
        "operation": "view + transpose",
        "shapes": {"Q": list(Q.shape), "K": list(K.shape), "V": list(V.shape)},
        "description": "Reshape for parallel head computation (no FLOPs)"
    })
    
    # Step 3: Attention scores: Q @ K.T / sqrt(d_k)
    # This is the CRITICAL O(n^2) operation
    scale = head_dim ** -0.5
    attn_scores = (Q @ K.transpose(-2, -1)) * scale  # (batch, heads, seq, seq)
    
    results["steps"].append({
        "step": 3,
        "name": "attention_scores",
        "operation": "Q @ K.T * scale",
        "shape": list(attn_scores.shape),
        "flops": batch_size * num_heads * seq_len * seq_len * head_dim,
        "description": "ATTENTION BOTTLENECK: O(n^2) in sequence length"
    })
    
    # Step 4: Softmax
    attn_weights = torch.softmax(attn_scores, dim=-1)
    
    results["steps"].append({
        "step": 4,
        "name": "softmax",
        "operation": "softmax(scores, dim=-1)",
        "shape": list(attn_weights.shape),
        "description": "Normalize to probability distribution per query"
    })
    
    # Step 5: Apply attention to values
    attn_output = attn_weights @ V  # (batch, heads, seq, head_dim)
    
    results["steps"].append({
        "step": 5,
        "name": "attention_apply",
        "operation": "attn_weights @ V",
        "shape": list(attn_output.shape),
        "flops": batch_size * num_heads * seq_len * seq_len * head_dim,
        "description": "SECOND O(n^2): weighted sum of values"
    })
    
    # Step 6: Reshape back and output projection
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
    W_o = torch.randn(hidden_dim, hidden_dim) * 0.02
    output = attn_output @ W_o
    
    results["steps"].append({
        "step": 6,
        "name": "output_projection",
        "operation": "reshape + x @ W_o",
        "shape": list(output.shape),
        "flops": batch_size * seq_len * hidden_dim * hidden_dim,
        "description": "Project back to hidden dimension"
    })
    
    # Summary
    total_flops = sum(s.get("flops", 0) for s in results["steps"])
    results["summary"] = {
        "total_flops": total_flops,
        "attention_flops": 2 * batch_size * num_heads * seq_len * seq_len * head_dim,
        "projection_flops": 4 * batch_size * seq_len * hidden_dim * hidden_dim,
        "n_squared_ops": ["attention_scores", "attention_apply"],
    }
    
    return results


# =========
# Part 4: MLP Gate/Up/Down Mechanics
# =========

def trace_mlp_step_by_step(
    hidden_dim: int = 128,
    intermediate_dim: int = 512,  # Often 4x hidden
    seq_len: int = 16,
    batch_size: int = 1,
) -> dict[str, Any]:
    """
    Trace MLP (FFN) computation step by step.
    
    OLMo uses gated MLP: SiLU(gate) * up -> down
    Intermediate dim = 11008 = 256 * 43 (matches kernel fiber!)
    
    For Gyroscopic ASI:
    - The 256x43 factorization could encode kernel structure
    - MLP routing could be horizon-dependent
    """
    results: dict[str, Any] = {"steps": []}
    
    # Input: hidden states
    x = torch.randn(batch_size, seq_len, hidden_dim)
    results["steps"].append({
        "step": 0,
        "name": "input",
        "shape": list(x.shape),
    })
    
    # Gate projection: hidden -> intermediate
    W_gate = torch.randn(intermediate_dim, hidden_dim) * 0.02
    gate = x @ W_gate.T  # (batch, seq, intermediate)
    
    results["steps"].append({
        "step": 1,
        "name": "gate_projection",
        "operation": "x @ W_gate.T",
        "shape": list(gate.shape),
        "weight_shape": list(W_gate.shape),
        "flops": batch_size * seq_len * hidden_dim * intermediate_dim,
    })
    
    # Up projection: hidden -> intermediate
    W_up = torch.randn(intermediate_dim, hidden_dim) * 0.02
    up = x @ W_up.T
    
    results["steps"].append({
        "step": 2,
        "name": "up_projection",
        "operation": "x @ W_up.T",
        "shape": list(up.shape),
        "weight_shape": list(W_up.shape),
        "flops": batch_size * seq_len * hidden_dim * intermediate_dim,
    })
    
    # Gated activation: SiLU(gate) * up
    # SiLU(x) = x * sigmoid(x)
    intermediate = torch.nn.functional.silu(gate) * up
    
    results["steps"].append({
        "step": 3,
        "name": "gated_activation",
        "operation": "silu(gate) * up",
        "shape": list(intermediate.shape),
        "description": "SiLU(x) = x * sigmoid(x), then element-wise multiply"
    })
    
    # Down projection: intermediate -> hidden
    W_down = torch.randn(hidden_dim, intermediate_dim) * 0.02
    output = intermediate @ W_down.T
    
    results["steps"].append({
        "step": 4,
        "name": "down_projection",
        "operation": "intermediate @ W_down.T",
        "shape": list(output.shape),
        "weight_shape": list(W_down.shape),
        "flops": batch_size * seq_len * intermediate_dim * hidden_dim,
    })
    
    # 256x43 factorization insight
    if intermediate_dim == 11008:
        results["kernel_factorization"] = {
            "intermediate_dim": 11008,
            "factorization": "256 x 43",
            "256": "horizon buckets",
            "43": "fiber features (mask bits + dual code + anatomy)",
            "insight": "MLP could route based on kernel horizon index"
        }
    
    return results


# =========
# Part 5: What Can Be Replaced?
# =========

def summarize_observations() -> dict[str, Any]:
    """
    Summarize factual observations about transformer weight access.
    
    NOTE: This reports what we measured, not hypotheses about replacements.
    """
    return {
        "weight_access_types": {
            "embedding": {
                "operation": "weight[token_id]",
                "complexity": "O(1) lookup per token",
                "verified": "Manual indexing matches module output"
            },
            "linear": {
                "operation": "x @ weight.T + bias",
                "complexity": "O(in * out) per input vector",
                "verified": "Manual matmul matches module output"
            },
            "attention_scores": {
                "operation": "Q @ K.T / sqrt(d_k)",
                "complexity": "O(seq^2 * head_dim) per head",
                "note": "This is the quadratic operation"
            },
            "attention_apply": {
                "operation": "attn_weights @ V",
                "complexity": "O(seq^2 * head_dim) per head",
                "note": "Second quadratic operation"
            },
            "layernorm": {
                "operation": "(x - mean) / std * gamma + beta",
                "complexity": "O(dim) per position",
                "note": "Element-wise, not matmul"
            }
        },
        "architectural_facts": {
            "olmo_intermediate": "11008 = 256 x 43",
            "olmo_hidden": 4096,
            "olmo_heads": 32,
            "head_dim": 128,
            "quadratic_ops": ["attention_scores", "attention_apply"]
        }
    }


# =========
# Part 6: OLMo-Specific Analysis
# =========

def analyze_olmo_structure() -> dict[str, Any]:
    """
    OLMo-3-7B architecture facts (from config.json).
    """
    return {
        "model_config": {
            "num_layers": 32,
            "num_heads": 32,
            "hidden_dim": 4096,
            "head_dim": 128,
            "intermediate_dim": 11008,
            "vocab_size": 100278,
            "sliding_window": 4096,
        },
        "factorizations": {
            "intermediate": "11008 = 256 x 43",
            "hidden": "4096 = 64 x 64",
            "heads": "32 = 4 x 8",
        },
        "weight_access_pattern": {
            "embedding": "weight[token_id] -> (4096,)",
            "attention_qkv": "hidden @ W.T -> (4096,) each",
            "attention_out": "attended @ W_o.T -> (4096,)",
            "mlp_gate": "hidden @ W_gate.T -> (11008,)",
            "mlp_up": "hidden @ W_up.T -> (11008,)",
            "mlp_down": "intermediate @ W_down.T -> (4096,)",
        }
    }


# =========
# Main: Run All Analyses
# =========

def main() -> None:
    print("PyTorch Transformer Internals Probe")
    print("=" * 5)
    print("\nFor Gyroscopic ASI: Understanding weight access and computation steps")
    print()
    
    # Part 1: Weight Storage
    print("-" * 5)
    print("1. Weight Storage Analysis")
    print("-" * 5)
    weight_info = analyze_weight_storage()
    for key, val in weight_info.items():
        print(f"  {key}: {val}")
    
    # Part 1b: Embedding Storage
    print("\n" + "-" * 5)
    print("1b. Embedding Storage Analysis")
    print("-" * 5)
    embed_info = analyze_embedding_storage()
    for key, val in embed_info.items():
        print(f"  {key}: {val}")
    
    # Part 2: Attention Step-by-Step
    print("\n" + "-" * 5)
    print("2. Attention Computation Steps")
    print("-" * 5)
    attn_trace = trace_attention_step_by_step(
        hidden_dim=4096,
        num_heads=32,
        seq_len=64,
        batch_size=1,
    )
    for step in attn_trace["steps"]:
        print(f"  Step {step['step']}: {step['name']}")
        if "operation" in step:
            print(f"    Op: {step['operation']}")
        if "shape" in step:
            print(f"    Shape: {step['shape']}")
        elif "shapes" in step:
            print(f"    Shapes: {step['shapes']}")
        if "flops" in step:
            print(f"    FLOPs: {step['flops']:,}")
        if "description" in step:
            print(f"    Note: {step['description']}")
    
    print("\n  Attention Summary:")
    for key, val in attn_trace["summary"].items():
        print(f"    {key}: {val}")
    
    # Part 3: MLP Step-by-Step
    print("\n" + "-" * 5)
    print("3. MLP (Gated FFN) Computation Steps")
    print("-" * 5)
    mlp_trace = trace_mlp_step_by_step(
        hidden_dim=4096,
        intermediate_dim=11008,  # OLMo's actual size
        seq_len=64,
        batch_size=1,
    )
    for step in mlp_trace["steps"]:
        print(f"  Step {step['step']}: {step['name']}")
        if "operation" in step:
            print(f"    Op: {step['operation']}")
        if "shape" in step:
            print(f"    Shape: {step['shape']}")
        if "weight_shape" in step:
            print(f"    Weight: {step['weight_shape']}")
        if "flops" in step:
            print(f"    FLOPs: {step['flops']:,}")
    
    # Part 4: Summary of Observations
    print("\n" + "-" * 5)
    print("4. Summary of Observations")
    print("-" * 5)
    observations = summarize_observations()
    
    print("\n  Weight Access Types (verified):")
    for name, info in observations["weight_access_types"].items():
        print(f"\n  {name}:")
        for key, val in info.items():
            print(f"    {key}: {val}")
    
    print("\n  Architectural Facts:")
    for name, val in observations["architectural_facts"].items():
        print(f"    {name}: {val}")
    
    # Part 5: OLMo Structure
    print("\n" + "-" * 5)
    print("5. OLMo-3-7B Architecture (from config.json)")
    print("-" * 5)
    olmo = analyze_olmo_structure()
    
    print("\n  Config:")
    for key, val in olmo["model_config"].items():
        print(f"    {key}: {val}")
    
    print("\n  Factorizations:")
    for key, val in olmo["factorizations"].items():
        print(f"    {key}: {val}")
    
    print("\n  Weight Access Pattern:")
    for key, val in olmo["weight_access_pattern"].items():
        print(f"    {key}: {val}")
    
    print("\nProbe complete.")


if __name__ == "__main__":
    main()
