# src/agent/chat.py - Gyroscopic ASI with coupled semantic-geometric scoring

"""
Gyroscopic ASI + OLMo Hybrid Chat.

Key insight: Kernel geometry COUPLES with embedding semantics.
- Kernel provides STRUCTURE: 256-horizon field, phase-aware memory M (K=43)
- Embeddings provide SEMANTICS: what words mean, how they relate
- M accumulates trajectory-dependent biases that shape semantic selection

Architecture:
1. Semantic byte features: mean embedding of tokens per first-code-byte
2. Geometric byte features: kernel's byte_features (256, 43) per CGM-1
3. M field: phase-aware memory M[h, p, :] with shape (256, 4, 43)
4. Hybrid scoring: 0.3 * semantic + 0.7 * geometric (CGM-G3: geometry dominates)

CGM Alignment (per Mech_Interp_Report):
- K=43 matches OLMo's MLP 256x43 factorization (CGM-1: confirmed)
- Semantic prototypes demoted (CGM-G3: saturates at ~8% accuracy)
- Kernel K4 used as constitutional (CGM-D3: OLMo K4 is skewed auxiliary)
- Future adapter should use hidden_L4 (CGM-C/J: peak horizon alignment)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import safe_open
from transformers import AutoTokenizer

from src.agent.adapters import SemanticTokenCodec
from src.agent.inference import InferenceFunction, InferenceState
from src.router.kernel import RouterKernel
from src.router.constants import mask12_for_byte, popcount


def _configure_stdout_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass


IM_END_TOKEN = 100265
ENDOFTEXT_TOKEN = 100257
EOS_TOKENS = {IM_END_TOKEN, ENDOFTEXT_TOKEN}


def format_chat_prompt(user_message: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful AI assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_message}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@dataclass
class OlmoManifold:
    """Simplified manifold: just embeddings for semantic scoring."""
    tokenizer: Any
    embed_tokens: torch.Tensor  # [vocab, 4096]
    vocab_size: int = 100278
    embed_dim: int = 4096


def load_olmo_manifold(model_dir: Path) -> OlmoManifold:
    """Load tokenizer and embeddings."""
    print("  Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    vocab_size = int(cfg.get("vocab_size", 100278))
    
    with open(model_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    
    shard_name = weight_map["model.embed_tokens.weight"]
    shard_path = model_dir / shard_name
    
    print("  Loading embeddings...")
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        embed_tokens = f.get_tensor("model.embed_tokens.weight")
    
    return OlmoManifold(
        tokenizer=tok,
        embed_tokens=embed_tokens,
        vocab_size=vocab_size,
    )


class GyroscopicChat:
    """
    Gyroscopic chat with COUPLED semantic-geometric scoring.
    
    The kernel provides structure: 256-horizon field, phase-aware memory M.
    Embeddings provide semantics: byte scoring via similarity to context.
    M accumulates trajectory-dependent biases via Hebbian learning.
    
    Hybrid scoring: semantic_score + geometric_score (M-modulated).
    This couples the two worlds: semantics provide signal, geometry shapes it.
    """
    
    CONTEXT_DECAY = 0.9
    
    # Hybrid scoring weights (CGM-G3: prototypes saturate at ~8%, geometry must dominate)
    SEMANTIC_WEIGHT = 0.3  # Weight for embedding-based semantic scores (demoted per CGM-G3)
    GEOMETRIC_WEIGHT = 0.7  # Weight for M-modulated geometric scores (promoted per CGM-1)
    
    # Prompt anchor: prevents self-poisoning from random early tokens
    PROMPT_ANCHOR_LAMBDA_START = 0.05
    PROMPT_ANCHOR_LAMBDA_GROWTH = 0.02
    
    def __init__(
        self,
        model_dir: Path,
        atlas_dir: Path,
        codec_path: Path | None = None,
        deterministic: bool = False,
        debug_mode: bool = False,
    ):
        self.debug_mode = debug_mode
        self._model_dir = model_dir
        
        print("Loading OLMo manifold (embeddings only)...")
        self.manifold = load_olmo_manifold(model_dir)
        self.tok = self.manifold.tokenizer
        
        print("Initializing kernel...")
        self.kernel = RouterKernel(atlas_dir)
        
        self.deterministic = deterministic
        
        # Track kernel state
        self._last_mask = mask12_for_byte(self.kernel.last_byte)
        self._last_vertex = self.kernel.current_vertex
        
        # Context: separate prompt vs generated to prevent self-poisoning
        # _context_prompt: frozen after process_input (anchors to direct query)
        # _context_gen: accumulated from generated tokens
        # Mixing controlled by PROMPT_ANCHOR_LAMBDA
        self._context_prompt: torch.Tensor = torch.zeros(4096, dtype=torch.float32)
        self._context_gen: torch.Tensor = torch.zeros(4096, dtype=torch.float32)
        self._gen_token_count: int = 0  # Tracks tokens generated for lambda growth
        
        # Load or build semantic codec
        if codec_path is not None and codec_path.exists():
            print(f"Loading codec from {codec_path}...")
            self.codec = SemanticTokenCodec.load(codec_path)
        else:
            print("Building semantic codec...")
            self.codec = SemanticTokenCodec.build(
                embed_tokens=self.manifold.embed_tokens,
                vocab_size=self.manifold.vocab_size,
            )
            if codec_path is not None:
                codec_path.parent.mkdir(parents=True, exist_ok=True)
                self.codec.save(codec_path)
        
        # Build semantic byte features: mean embedding per first-code-byte
        print("Building semantic byte features...")
        self.semantic_byte_features = self._build_semantic_byte_features()
        
        # Initialize InferenceFunction with K=43 (CGM-1 confirms MLP 256x43 structure)
        # OLMo embeddings are 4096 = 256*16, but MLP intermediate is 11008 = 256*43
        # We use K=43 as the canonical gyroscopic configuration per CGM physics
        print("Initializing InferenceFunction (K=43)...")
        self.inference = InferenceFunction(K=43, eta=0.00117)
        
        # Load geometric byte features from phenomenology
        phen_path = atlas_dir / "phenomenology.npz"
        with np.load(phen_path) as phen:
            byte_features_K43 = phen["features_K43"]
            gamma_table = phen["gamma_table"]
        
        # Set kernel tables for inference (kernel K4 is constitutional per CGM-D3)
        self.inference.set_kernel_tables(
            byte_weight=self.kernel.byte_weight,
            byte_charge=self.kernel.byte_charge,  # Router's K4, not OLMo's skewed K4
            byte_features=byte_features_K43,
            gamma_table=gamma_table,
        )
        
        # Create inference state (M field accumulator)
        self.inf_state = InferenceState.create(K=43)
        print(f"  M field shape: {self.inf_state.M.shape}")  # (256, 4, 43)
        
        # NOTE: Future adapter should use hidden_L4 as input (CGM-C/J: peak horizon info)
        # Placeholder path for offline-precomputed L4-based token fields:
        # self._adapter_L4_path = atlas_dir / "adapter_L4_to_gyro.npz"
        
        # Analyze feature resolution if debug mode
        if self.debug_mode:
            self.analyze_feature_resolution()
            self.analyze_prefix_resolution(prefix_len=2)
            self.analyze_prefix_resolution(prefix_len=3)
        
        self.genealogy: list[int] = []
        self._decode_stats = {"exact": 0, "3byte": 0, "2byte": 0, "1byte": 0, "fallback": 0}
        
        # Kernel geometry counters (debug mode)
        self._vertex_counts = np.zeros(4, dtype=int)
        self._phase_counts = np.zeros(4, dtype=int)
        self._horizon_sample: list[int] = []
        
        # Precompute mask-driven bonus (translational DOF from CGM 6DOF)
        # Prefers "short translations" (low-weight masks)
        self._mask_bonus = 0.1 * (1.0 - np.array(
            [popcount(mask12_for_byte(b)) / 12.0 for b in range(256)],
            dtype=np.float32
        ))
        
        print(f"  vocab_size: {self.manifold.vocab_size}")
        print(f"  context_decay: {self.CONTEXT_DECAY}")
        print(f"  hybrid_weights: semantic={self.SEMANTIC_WEIGHT}, geometric={self.GEOMETRIC_WEIGHT}")
    
    def _build_semantic_byte_features(self) -> torch.Tensor:
        """
        Build semantic embedding for each byte value (0-255).
        
        semantic_byte_features[b] = mean embedding of all tokens whose
        4-byte semantic code starts with byte b.
        
        This allows scoring bytes by: semantic_byte_features @ context
        """
        features = torch.zeros(256, 4096, dtype=torch.float32)
        
        for b in range(256):
            # Get all tokens whose code starts with this byte
            candidates = self.codec._prefix_index[1].get((b,), [])
            if candidates:
                # Mean embedding of these tokens
                embeds = self.manifold.embed_tokens[candidates].float()
                features[b] = embeds.mean(dim=0)
        
        # Normalize for stable dot products
        norms = features.norm(dim=1, keepdim=True)
        features = features / (norms + 1e-8)
        
        print(f"  Built semantic features for {(norms.squeeze() > 0).sum().item()} bytes")  # type: ignore[attr-defined]
        return features
    
    def _build_semantic_position_features(self, prefix: list[int]) -> torch.Tensor:
        """
        Build semantic features for the NEXT byte given a prefix.
        
        For prefix of length k, returns features[b] = mean embedding of tokens
        whose code has prefix + [b] as first k+1 bytes.
        """
        features = torch.zeros(256, 4096, dtype=torch.float32)
        prefix_len = len(prefix) + 1
        
        for b in range(256):
            key = tuple(prefix + [b])
            candidates = self.codec._prefix_index.get(prefix_len, {}).get(key, [])
            if candidates:
                embeds = self.manifold.embed_tokens[candidates].float()
                features[b] = embeds.mean(dim=0)
        
        norms = features.norm(dim=1, keepdim=True)
        features = features / (norms + 1e-8)
        return features
    
    def _embed_to_field_K43(self, embed: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Adapt 4096-dim embedding to (256, 43) field for K=43 inference.
        
        OLMo embeddings are 4096 = 256*16, but the canonical gyroscopic structure
        is 256*43 (matching MLP intermediate confirmed by CGM-1).
        
        Current adapter: zero-pad from K=16 to K=43.
        Future: Replace with learned adapter from hidden_L4 (CGM-C/J).
        """
        if isinstance(embed, torch.Tensor):
            embed = embed.numpy()
        embed = np.asarray(embed, dtype=np.float32)
        
        # Reshape to (256, 16) from embedding space
        X_raw = embed.reshape(256, 16)
        
        # Pad to (256, 43) - interim adapter until L4-based adapter is built
        # Zeros preserve semantic signal while matching geometric structure
        X = np.pad(X_raw, ((0, 0), (0, 27)), mode='constant', constant_values=0)
        
        return X  # (256, 43)
    
    def analyze_feature_resolution(self) -> None:
        """
        Measure variance within each byte bucket.
        
        If disparate tokens map to the same byte prefix, the average might be
        a meaningless gray vector near zero. High Ratio = noisy buckets.
        """
        print("\n[DEBUG] Analyzing Semantic Feature Resolution...")
        variances = []
        norms = []
        bucket_sizes = []
        
        for b in range(256):
            candidates = self.codec._prefix_index[1].get((b,), [])
            if len(candidates) > 1:
                embeds = self.manifold.embed_tokens[candidates].float()
                # Calculate standard deviation of embeddings in this bucket
                std_dev = torch.std(embeds, dim=0).mean().item()  # type: ignore[attr-defined]
                mean_norm = torch.norm(embeds.mean(dim=0)).item()
                variances.append(std_dev)
                norms.append(mean_norm)
                bucket_sizes.append(len(candidates))
        
        if not variances:
            print("  No multi-token buckets found!")
            return
        
        avg_std = sum(variances) / len(variances)
        avg_norm = sum(norms) / len(norms)
        avg_bucket_size = sum(bucket_sizes) / len(bucket_sizes)
        ratio = avg_std / avg_norm if avg_norm > 1e-8 else float('inf')
        
        print(f"  Multi-token buckets: {len(variances)}/256")
        print(f"  Avg bucket size: {avg_bucket_size:.1f} tokens")
        print(f"  Avg Bucket Std Dev: {avg_std:.4f}")
        print(f"  Avg Centroid Norm:  {avg_norm:.4f}")
        print(f"  Ratio (Noise/Signal): {ratio:.2f}")
        
        if ratio > 1.0:
            print("  WARNING: Ratio > 1.0 -> byte buckets are too noisy!")
            print("  Mean embeddings may represent conflicting semantics.")
        elif ratio > 0.5:
            print("  CAUTION: Ratio > 0.5 -> moderate noise in buckets.")
        else:
            print("  OK: Ratio < 0.5 -> byte buckets have reasonable coherence.")
        
        # Find worst buckets for diagnostics
        worst_idx = np.argsort(variances)[-5:][::-1]
        print("\n  Highest variance buckets:")
        bucket_list = list(self.codec._prefix_index[1].items())
        for i in worst_idx:
            b_key = bucket_list[i][0] if i < len(bucket_list) else (i,)
            print(f"    Byte 0x{b_key[0]:02x}: std={variances[i]:.4f}, "
                  f"size={bucket_sizes[i]}")
    
    def analyze_prefix_resolution(self, prefix_len: int = 2) -> None:
        """
        Analyze resolution/noise for deeper prefixes (2 or 3 bytes).
        
        Shows whether semantic resolution improves as prefix grows,
        or if buckets remain large and noisy.
        """
        print(f"\n[DEBUG] Analyzing prefix_len={prefix_len} feature resolution...")
        index = self.codec._prefix_index.get(prefix_len, {})
        variances = []
        norms = []
        sizes = []
        
        for _prefix, ids in index.items():
            if len(ids) > 1:
                embeds = self.manifold.embed_tokens[ids].float()
                std_dev = torch.std(embeds, dim=0).mean().item()  # type: ignore[attr-defined]
                mean_norm = torch.norm(embeds.mean(dim=0)).item()
                variances.append(std_dev)
                norms.append(mean_norm)
                sizes.append(len(ids))
        
        if not variances:
            print("  No multi-token buckets at this prefix length.")
            return
        
        avg_std = sum(variances) / len(variances)
        avg_norm = sum(norms) / len(norms)
        ratio = avg_std / avg_norm if avg_norm > 1e-8 else float('inf')
        avg_size = sum(sizes) / len(sizes)
        
        print(f"  Multi-token buckets: {len(variances)}/{len(index)}")
        print(f"  Avg bucket size: {avg_size:.1f} tokens")
        print(f"  Avg Std Dev: {avg_std:.4f}")
        print(f"  Avg Centroid Norm: {avg_norm:.4f}")
        print(f"  Noise/Signal: {ratio:.2f}")
        
        if ratio > 1.0:
            print("  WARNING: Noisy buckets at this prefix depth.")
        elif ratio > 0.5:
            print("  CAUTION: Moderate noise.")
        else:
            print("  OK: Reasonable coherence.")
    
    def _update_prompt_context(self, embedding: torch.Tensor) -> None:
        """Update prompt context (decayed accumulation during input processing)."""
        self._context_prompt = self.CONTEXT_DECAY * self._context_prompt + embedding
    
    def _update_gen_context(self, embedding: torch.Tensor) -> None:
        """Update generation context (decayed accumulation during generation)."""
        self._context_gen = self.CONTEXT_DECAY * self._context_gen + embedding
        self._gen_token_count += 1
    
    def _get_mixed_context(self) -> torch.Tensor:
        """
        Get mixed context for scoring: anchored to prompt, slowly incorporating gen.
        
        context = (1 - lambda) * prompt + lambda * gen
        lambda grows from PROMPT_ANCHOR_LAMBDA_START as more tokens are generated.
        """
        # Lambda grows but caps at 0.5 to always keep prompt influence
        lam = min(
            0.5,
            self.PROMPT_ANCHOR_LAMBDA_START + 
            self.PROMPT_ANCHOR_LAMBDA_GROWTH * self._gen_token_count
        )
        
        # Mix prompt and gen contexts
        mixed = (1.0 - lam) * self._context_prompt + lam * self._context_gen
        return mixed
    
    def process_input(self, token_ids: list[int]) -> None:
        """Process input tokens: build prompt context, step kernel, and update M."""
        for t in token_ids:
            tid = t if 0 <= t < self.manifold.vocab_size else 0
            e_t = self.manifold.embed_tokens[tid].float()
            
            # Update PROMPT context with this token's embedding
            self._update_prompt_context(e_t)
            
            # Get field X from embedding (adapt to 256x43 for K=43 inference)
            X = self._embed_to_field_K43(e_t)
            
            # Step kernel through the 4-byte code and update M
            bs = self.codec.encode(tid)
            for b in bs:
                # Get current kernel state
                h_curr = int(self.kernel.state_horizon[self.kernel.state_index])
                p_curr = int(self.kernel.phase[self.kernel.state_index, self.kernel.last_byte])
                chi_prev = self.kernel.current_vertex
                
                # Extract local activation at current horizon
                a_curr = X[h_curr, :].copy()
                
                # Step kernel
                prev_vertex = self.kernel.current_vertex
                self.kernel.step_byte(b)
                self.genealogy.append(b)
                
                # Update M via inference function
                delta_mask = mask12_for_byte(b)
                chi_curr = self.kernel.current_vertex
                self.inference.update(
                    state=self.inf_state,
                    h_curr=h_curr,
                    p_curr=p_curr,
                    a_curr=a_curr,
                    delta_mask=delta_mask,
                    chi_prev=chi_prev,
                    chi_curr=chi_curr,
                )
                
                self._last_mask = delta_mask
                self._last_vertex = prev_vertex
        
        # Initialize gen context from prompt (so first tokens have some signal)
        self._context_gen = self._context_prompt.clone()
        
        if self.debug_mode:
            m_norm = np.linalg.norm(self.inf_state.M)
            print(f"  M field norm after input: {m_norm:.4f}")
    
    def generate_token(self, context_token: int) -> tuple[int, float]:
        """
        Generate one token using HYBRID semantic-geometric scoring.
        
        For each byte position:
        1. Compute semantic scores: position_features @ context
        2. Compute geometric scores: inference.score_bytes() using M field
        3. Combine: SEMANTIC_WEIGHT * semantic + GEOMETRIC_WEIGHT * geometric
        4. Mask to valid prefixes, sample or argmax
        
        Args:
            context_token: Last token for context update
            
        Returns:
            (token_id, p0_entropy): Generated token and Position-0 entropy (bits)
        """
        V = self.manifold.vocab_size
        
        # Get mixed context (anchored to prompt, slowly incorporating gen)
        mixed_context = self._get_mixed_context()
        
        # Normalize context for stable scoring
        context_norm = mixed_context / (mixed_context.norm() + 1e-8)
        
        # Get field X from context (adapt to 256x43 for K=43 inference)
        X = self._embed_to_field_K43(mixed_context)
        
        # Ephemeral kernel state for planning
        idx = int(self.kernel.state_index)
        eph_last_byte = int(self.kernel.last_byte)
        
        planned: list[int] = []
        
        # Debug stats collectors (entropy, rank, valid_count)
        # P0 entropy is tracked as a first-class metric per CGM physics
        _debug_p0 = (0.0, 0, 0)  # (entropy_bits, rank, valid_count)
        _debug_p1 = (0.0, 0, 0)
        p0_entropy = 0.0  # Position-0 entropy in bits (CGM metric)
        
        for pos in range(4):
            # Get kernel observables
            h = int(self.kernel.state_horizon[idx])
            chi = int(self.kernel.state_vertex[idx])
            p = int(self.kernel.phase[idx, eph_last_byte])
            
            # Track kernel geometry (debug mode)
            if self.debug_mode:
                self._vertex_counts[chi] += 1
                self._phase_counts[p] += 1
                if len(self._horizon_sample) < 100:
                    self._horizon_sample.append(h)
            
            # Get allowed bytes for this prefix
            allowed_mask = self.codec.allowed_mask_for_prefix(planned)
            
            # --- SEMANTIC SCORES ---
            # Build position-specific semantic features
            if pos == 0:
                sem_features = self.semantic_byte_features
            else:
                sem_features = self._build_semantic_position_features(planned)
            
            # Semantic scores: how similar is each byte's semantic neighborhood to context?
            semantic_scores = (sem_features @ context_norm).numpy()
            
            # --- GEOMETRIC SCORES (M-modulated) ---
            # Extract local activation at current horizon
            a_curr = X[h, :].copy()
            
            # Get geometric scores from inference function
            geometric_scores = self.inference.score_bytes(
                state=self.inf_state,
                h_curr=h,
                p_curr=p,
                a_curr=a_curr,
                chi_curr=chi,
            )
            
            # Add translational DOF bonus (prefer short translations)
            geometric_scores = geometric_scores + self._mask_bonus
            
            # --- HYBRID COMBINATION ---
            # Normalize both score vectors to comparable scales
            sem_valid = semantic_scores[allowed_mask]
            geo_valid = geometric_scores[allowed_mask]
            
            if len(sem_valid) > 1:
                sem_std = np.std(sem_valid)
                sem_mean = np.mean(sem_valid)
                if sem_std > 1e-8:
                    semantic_normed = (semantic_scores - sem_mean) / sem_std
                else:
                    semantic_normed = semantic_scores - sem_mean
                
                geo_std = np.std(geo_valid)
                geo_mean = np.mean(geo_valid)
                if geo_std > 1e-8:
                    geometric_normed = (geometric_scores - geo_mean) / geo_std
                else:
                    geometric_normed = geometric_scores - geo_mean
            else:
                semantic_normed = semantic_scores
                geometric_normed = geometric_scores
            
            # Combine with weights
            raw_scores = (
                self.SEMANTIC_WEIGHT * semantic_normed + 
                self.GEOMETRIC_WEIGHT * geometric_normed
            )
            
            # Final normalization
            valid_raw = raw_scores[allowed_mask]
            if len(valid_raw) > 1:
                score_std = np.std(valid_raw)
                score_mean = np.mean(valid_raw)
                if score_std > 1e-8:
                    scores = (raw_scores - score_mean) / score_std
                else:
                    scores = raw_scores - score_mean
            else:
                scores = raw_scores
            
            # Mask invalid bytes
            scores[~allowed_mask] = -np.inf
            
            # Check we have valid options
            valid_count = np.sum(np.isfinite(scores))
            if valid_count == 0:
                # Fallback: pick any allowed byte
                allowed_indices = np.where(allowed_mask)[0]
                if len(allowed_indices) == 0:
                    b = 0
                else:
                    b = int(allowed_indices[0])
            elif self.deterministic:
                b = int(np.argmax(scores))
            else:
                # Softmax sampling
                valid_mask = np.isfinite(scores)
                exp_scores = np.zeros(256)
                exp_scores[valid_mask] = np.exp(scores[valid_mask] - scores[valid_mask].max())
                probs = exp_scores / exp_scores.sum()
                b = int(np.random.choice(256, p=probs))
            
            # DIAGNOSTIC: Collect entropy and rank for compact output
            if self.debug_mode:
                scores_copy = scores.copy()
                scores_copy[~np.isfinite(scores_copy)] = -np.inf
                sorted_indices = np.argsort(scores_copy)[::-1]
                
                rank_arr = np.where(sorted_indices == b)[0]
                chosen_rank = int(rank_arr[0]) if len(rank_arr) > 0 else 256
                
                valid = np.isfinite(scores_copy)
                if valid.any():
                    exp_s = np.exp(scores_copy[valid] - scores_copy[valid].max())
                    probs_ent = exp_s / exp_s.sum()
                    entropy = -np.sum(probs_ent * np.log(probs_ent + 1e-10))
                else:
                    entropy = 0.0
                
                # Store for compact summary (only Pos0 and Pos1 matter)
                if pos == 0:
                    _debug_p0 = (entropy, chosen_rank, int(valid.sum()))
                    p0_entropy = entropy  # Track as CGM metric
                elif pos == 1:
                    _debug_p1 = (entropy, chosen_rank, int(valid.sum()))
            
            planned.append(b)
            
            # Advance ephemeral kernel state
            idx = int(self.kernel.epistemology[idx, b])
            eph_last_byte = b
        
        # Commit: step real kernel and update M field
        for b in planned:
            # Get current kernel state before stepping
            h_curr = int(self.kernel.state_horizon[self.kernel.state_index])
            p_curr = int(self.kernel.phase[self.kernel.state_index, self.kernel.last_byte])
            chi_prev = self.kernel.current_vertex
            
            # Extract local activation
            a_curr = X[h_curr, :].copy()
            
            # Step kernel
            prev_vertex = self.kernel.current_vertex
            self.kernel.step_byte(b)
            self.genealogy.append(b)
            
            # Update M field via inference function
            delta_mask = mask12_for_byte(b)
            chi_curr = self.kernel.current_vertex
            self.inference.update(
                state=self.inf_state,
                h_curr=h_curr,
                p_curr=p_curr,
                a_curr=a_curr,
                delta_mask=delta_mask,
                chi_prev=chi_prev,
                chi_curr=chi_curr,
            )
            
            self._last_mask = delta_mask
            self._last_vertex = prev_vertex
        
        # Decode bytes to token
        token_id = self.codec.decode(
            planned, 
            context_norm, 
            self.manifold.embed_tokens, 
            stats=self._decode_stats
        )
        
        # Update context with generated token
        if 0 <= token_id < V:
            e_new = self.manifold.embed_tokens[token_id].float()
            
            # DIAGNOSTIC: Compact one-line summary per token
            if self.debug_mode:
                e_new_norm = e_new / (e_new.norm() + 1e-8)
                cosine_sim = torch.nn.functional.cosine_similarity(  # type: ignore[attr-defined]
                    context_norm.unsqueeze(0),
                    e_new_norm.unsqueeze(0)
                ).item()
                
                # Nearest-neighbor comparison (smaller sample for speed)
                subset_size = 1000
                rand_idx = torch.randint(0, V, (subset_size,))  # type: ignore[attr-defined]
                cand_embeds = self.manifold.embed_tokens[rand_idx].float()
                cand_norm = torch.nn.functional.normalize(cand_embeds, dim=1)
                sims = cand_norm @ context_norm
                best_sim = float(sims.max().item())  # type: ignore[attr-defined]
                
                # M field norm
                m_norm = np.linalg.norm(self.inf_state.M)
                
                # Get token string for context
                tok_str = self.tok.decode([token_id])
                tok_str_short = repr(tok_str[:12]) if len(tok_str) > 12 else repr(tok_str)
                
                # Compact summary: P0[e=X.XX r=N] P1[e=X.XX r=N] | sim=X.XX/X.XX | M=X.XXXX
                print(f"  P0[e={_debug_p0[0]:.2f} r={_debug_p0[1]:3d}/{_debug_p0[2]:3d}] "
                      f"P1[e={_debug_p1[0]:.2f} r={_debug_p1[1]:3d}/{_debug_p1[2]:3d}] | "
                      f"sim={cosine_sim:.2f}/{best_sim:.2f} | M={m_norm:.4f} | {tok_str_short}")
            
            # Update GENERATION context (not prompt) to prevent self-poisoning
            self._update_gen_context(e_new)
        
        return token_id, p0_entropy
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate response to prompt."""
        formatted = format_chat_prompt(prompt)
        input_ids = self.tok.encode(formatted, add_special_tokens=False)
        
        print(f"Input: {len(input_ids)} tokens")
        
        self.process_input(input_ids)
        
        print(f"Kernel state after input: {self.kernel.state_index}")
        print(f"Prompt context norm: {self._context_prompt.norm().item():.4f}")
        
        output_ids: list[int] = []
        context = input_ids[-1] if input_ids else 0
        
        # P0 entropy accumulator (CGM metric: target is < LM entropy)
        p0_entropies: list[float] = []
        
        # Print header for debug output
        if self.debug_mode:
            print("\n[DEBUG] Token generation (P0=Pos0, P1=Pos1, e=entropy, r=rank/valid, sim=chosen/best):")
        
        for i in range(max_tokens):
            if context in EOS_TOKENS:
                break
            
            if self.debug_mode:
                print(f"  T{i+1:2d}:", end="")
            
            token_id, p0_ent = self.generate_token(context)
            p0_entropies.append(p0_ent)
            
            if token_id >= self.manifold.vocab_size:
                print(f"Warning: token {token_id} out of range")
                break
            
            output_ids.append(token_id)
            context = token_id
            
            if not self.debug_mode and (i + 1) % 10 == 0:
                print(f"  Generated {i + 1} tokens...")
        
        # DIAGNOSTIC: Compact summary with CGM P0 entropy metric
        if self.debug_mode:
            m_final = np.linalg.norm(self.inf_state.M)
            mean_p0_ent = np.mean(p0_entropies) if p0_entropies else 0.0
            print(f"\n[SUMMARY] {len(output_ids)} tokens | M_final={m_final:.4f} | "
                  f"vertices={self._vertex_counts.tolist()} | phases={self._phase_counts.tolist()}")
            # CGM metric: P0 entropy vs LM entropy (target: P0 < LM)
            # LM entropy for this prompt class is ~0.82 bits (from MI report)
            print(f"[METRIC] Mean P0 entropy: {mean_p0_ent:.4f} bits "
                  f"(target: < 1.0 bits for LM-competitive decisiveness)")
        
        return self.tok.decode(output_ids, skip_special_tokens=True)


def run_test() -> None:
    """Run chat test."""
    _configure_stdout_utf8()
    
    MODEL_DIR = Path("data/models/Olmo-3-7B-Instruct")
    ATLAS_DIR = Path("data/atlas")
    CODEC_PATH = MODEL_DIR / "gyro_codebook.npz"
    
    # Check for debug flag
    debug_mode = "--debug" in sys.argv or "-d" in sys.argv
    
    print("==========")
    if debug_mode:
        print("Gyro Chat (Semantic Scoring) - DEBUG MODE")
        print("Rail/Entropy/Alignment diagnostics enabled")
    else:
        print("Gyro Chat (Semantic Scoring)")
    print("==========")
    
    chat = GyroscopicChat(
        model_dir=MODEL_DIR,
        atlas_dir=ATLAS_DIR,
        codec_path=CODEC_PATH,
        deterministic=False,
        debug_mode=debug_mode,
    )
    
    USER_MESSAGE = (
        "Freedom is not worth having if it does not include "
        "the freedom to make mistakes. What does this quote mean?"
    )
    
    print("\n----------")
    print("USER:")
    print(USER_MESSAGE)
    print("----------")
    
    # Fewer tokens in debug mode due to verbose output
    max_tokens = 20 if debug_mode else 100
    response = chat.generate(USER_MESSAGE, max_tokens=max_tokens)
    
    print("\n----------")
    print("ASSISTANT:")
    print(response)
    print("----------")
    
    print(f"\nPrompt context norm: {chat._context_prompt.norm().item():.4f}")
    print(f"Gen context norm: {chat._context_gen.norm().item():.4f}")
    print(f"Genealogy: {len(chat.genealogy)} bytes")
    
    stats = chat._decode_stats
    total = sum(stats.values())
    if total > 0:
        print(f"\nDecode stats ({total} tokens):")
        print(f"  Exact 4-byte: {stats['exact']} ({100*stats['exact']/total:.1f}%)")
        print(f"  3-byte backoff: {stats['3byte']} ({100*stats['3byte']/total:.1f}%)")
        print(f"  2-byte backoff: {stats['2byte']} ({100*stats['2byte']/total:.1f}%)")
        print(f"  1-byte backoff: {stats['1byte']} ({100*stats['1byte']/total:.1f}%)")
        print(f"  Fallback: {stats['fallback']} ({100*stats['fallback']/total:.1f}%)")
    
    print("\n==========")
    print("COMPLETE")
    print("==========")


if __name__ == "__main__":
    run_test()