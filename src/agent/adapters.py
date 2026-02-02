"""
Adapters that bind external models/tokenizers to the fixed gyroscopic atlas.

This module is intentionally *not* part of the router atlas artifacts:
- atlas/* is model-agnostic kernel geometry.
- adapters are model-specific plumbing.

Provides:
- TokenBinding: external_token_id <-> internal_id (uint16, 0..65535)
- EmbeddingAdapter: projects arbitrary vectors to gyro dimension D = 256*K
- SemanticTokenCodec: LSH-based 4-byte semantic codes for large vocabularies
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from numpy.typing import NDArray


# =============================================================================
# Token binding
# =============================================================================

@dataclass(frozen=True)
class TokenBinding:
    """
    Binding between external token IDs and internal 16-bit IDs (0..65535).

    NPZ schema:
      - vocab_size: scalar int
      - external_to_internal: uint16[V] (required)
      - internal_to_external: int32[65536] (optional; -1 means unmapped)
    """
    vocab_size: int
    external_to_internal: NDArray[np.uint16]          # shape (V,)
    internal_to_external: NDArray[np.int32]           # shape (65536,)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        require_injective: bool = True,
    ) -> "TokenBinding":
        with np.load(path, allow_pickle=False) as z:
            if "vocab_size" not in z.files:
                raise ValueError(f"{path} missing 'vocab_size'")
            if "external_to_internal" not in z.files:
                raise ValueError(f"{path} missing 'external_to_internal'")

            vocab_size = int(np.asarray(z["vocab_size"]).item())
            ext2int = np.asarray(z["external_to_internal"], dtype=np.uint16)

            # Basic sanity: vocab_size must be within 16-bit internal ID space
            if not (1 <= vocab_size <= 65536):
                raise ValueError(f"{path}: vocab_size must be in [1,65536], got {vocab_size}")

            if ext2int.ndim != 1 or ext2int.shape[0] != vocab_size:
                raise ValueError(
                    f"{path}: external_to_internal must have shape (vocab_size,), "
                    f"got {ext2int.shape} vs vocab_size={vocab_size}"
                )

            # Optional internal_to_external; otherwise build it.
            if "internal_to_external" in z.files:
                int2ext = np.asarray(z["internal_to_external"], dtype=np.int32)
                if int2ext.shape != (65536,):
                    raise ValueError(f"{path}: internal_to_external must be shape (65536,), got {int2ext.shape}")
            else:
                int2ext = np.full((65536,), -1, dtype=np.int32)
                # Fill; if collisions exist, keep the first by default (but you can forbid via require_injective).
                for ext_id, internal_id in enumerate(ext2int.tolist()):
                    if int2ext[int(internal_id)] == -1:
                        int2ext[int(internal_id)] = int(ext_id)

            # Validate range
            # (uint16 already constrains 0..65535, but keep explicit)
            if int(ext2int.max()) > 65535:
                raise ValueError(f"{path}: internal id out of range (max={int(ext2int.max())})")

            # Optional injectivity check (recommended)
            if require_injective:
                uniq = np.unique(ext2int)
                if uniq.size != vocab_size:
                    raise ValueError(
                        f"{path}: external_to_internal is not injective "
                        f"(unique internal ids={uniq.size}, vocab_size={vocab_size})"
                    )

            # If mapping is required to be injective, enforce bidirectional consistency.
            if require_injective:
                for ext_id, internal_id in enumerate(ext2int.tolist()):
                    mapped = int(int2ext[int(internal_id)])
                    if mapped != int(ext_id):
                        raise ValueError(
                            f"{path}: inconsistent mapping: external_to_internal[{ext_id}]={internal_id}, "
                            f"but internal_to_external[{internal_id}]={mapped}"
                        )

            return cls(vocab_size=vocab_size, external_to_internal=ext2int, internal_to_external=int2ext)

    def to_internal(self, external_token_id: int) -> int:
        if not (0 <= external_token_id < self.vocab_size):
            raise ValueError(f"external_token_id {external_token_id} out of range [0, {self.vocab_size})")
        return int(self.external_to_internal[int(external_token_id)])

    def to_external(self, internal_id: int) -> int:
        if not (0 <= internal_id <= 0xFFFF):
            raise ValueError(f"internal_id {internal_id} out of range [0, 65535]")
        ext = int(self.internal_to_external[int(internal_id)])
        if ext < 0:
            raise ValueError(f"internal_id {internal_id} is not mapped to any external token")
        return ext


# =============================================================================
# Embedding adapter
# =============================================================================

EmbeddingMode = Literal["identity", "slice", "tile", "linear"]


@dataclass
class EmbeddingAdapter:
    """
    Projects an input vector to the gyro dimension D = 256*K.

    NPZ schema:
      - input_dim: scalar int
      - output_dim: scalar int
      - mode: scalar string in {"identity","slice","tile","linear"}
      - W: float32[input_dim, output_dim] (required iff mode=="linear")
    """
    input_dim: int
    output_dim: int
    mode: EmbeddingMode
    W: Optional[NDArray[np.float32]] = None

    @classmethod
    def load(cls, path: Path) -> "EmbeddingAdapter":
        with np.load(path, allow_pickle=False) as z:
            for k in ("input_dim", "output_dim", "mode"):
                if k not in z.files:
                    raise ValueError(f"{path} missing '{k}'")

            input_dim = int(np.asarray(z["input_dim"]).item())
            output_dim = int(np.asarray(z["output_dim"]).item())
            mode = str(np.asarray(z["mode"]).item())

            if mode not in ("identity", "slice", "tile", "linear"):
                raise ValueError(f"{path}: invalid mode={mode}")

            W = None
            if mode == "linear":
                if "W" not in z.files:
                    raise ValueError(f"{path}: mode='linear' requires matrix 'W'")
                W = np.asarray(z["W"], dtype=np.float32)
                if W.shape != (input_dim, output_dim):
                    raise ValueError(f"{path}: W must have shape {(input_dim, output_dim)}, got {W.shape}")

            return cls(input_dim=input_dim, output_dim=output_dim, mode=mode, W=W)

    def project(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.input_dim:
            raise ValueError(f"EmbeddingAdapter expected input_dim={self.input_dim}, got {x.shape[0]}")

        if self.mode == "identity":
            if self.input_dim != self.output_dim:
                raise ValueError("mode='identity' requires input_dim == output_dim")
            return x.copy()

        if self.mode == "slice":
            if self.input_dim < self.output_dim:
                raise ValueError("mode='slice' requires input_dim >= output_dim")
            return x[: self.output_dim].copy()

        if self.mode == "tile":
            reps = (self.output_dim + self.input_dim - 1) // self.input_dim
            return np.tile(x, reps)[: self.output_dim].astype(np.float32, copy=False)

        # linear
        assert self.W is not None
        return (x @ self.W).astype(np.float32, copy=False)


# =============================================================================
# Gating masks for non-contiguous token bindings
# =============================================================================

@dataclass(frozen=True)
class ByteGatingMasks:
    """
    Precomputed boolean masks for byte selection when using non-contiguous token bindings.
    
    - allowed_b1[b1] = True iff there exists at least one valid token with that b1
    - allowed_b2[b1, b2] = True iff (b1, b2) forms a valid internal_id
    """
    allowed_b1: NDArray[np.bool_]    # shape (256,)
    allowed_b2: NDArray[np.bool_]    # shape (256, 256)

    @classmethod
    def from_token_binding(cls, binding: TokenBinding) -> "ByteGatingMasks":
        """Build masks from a TokenBinding."""
        allowed_pair = np.zeros((256, 256), dtype=np.bool_)
        
        for ext_id in range(binding.vocab_size):
            internal_id = int(binding.external_to_internal[ext_id])
            b1 = (internal_id >> 8) & 0xFF
            b2 = internal_id & 0xFF
            allowed_pair[b1, b2] = True
        
        allowed_b1_arr = allowed_pair.any(axis=1)
        allowed_b1 = np.asarray(allowed_b1_arr, dtype=np.bool_)
        
        return cls(allowed_b1=allowed_b1, allowed_b2=allowed_pair)
    
    def get_b2_mask(self, b1: int) -> NDArray[np.bool_]:
        """Get the allowed b2 values for a given b1."""
        return self.allowed_b2[b1, :]


# =============================================================================
# Semantic Token Codec (LSH-based 4-byte codes for large vocabularies)
# =============================================================================

@dataclass
class SemanticTokenCodec:
    """
    Semantic codebook: 4-byte LSH codes for vocabulary tokens.
    
    Encoding: token_id -> 4 bytes (32 sign bits from embedding projections)
    Decoding: 4 bytes -> token_id (prefix backoff with dot-product scoring)
    
    This bridges byte selection to vocabulary space semantically:
    similar embeddings share prefixes, so sequential byte selection
    navigates semantic space rather than arbitrary integer space.
    
    NPZ schema:
      - vocab_size: scalar int
      - embed_dim: scalar int
      - code_bytes: uint8[vocab_size, 4]
      - proj_vectors: float32[32, embed_dim] (projection vectors for LSH)
    
    Semantic gating:
      - _allowed_b0: bool[256] - valid first bytes
      - _allowed_b1: dict[(b0,) -> bool[256]] - valid second bytes given first
      - _allowed_b2: dict[(b0,b1) -> bool[256]] - valid third bytes given first two
      - _allowed_b3: dict[(b0,b1,b2) -> bool[256]] - valid fourth bytes given first three
    """
    vocab_size: int
    embed_dim: int
    code_bytes: NDArray[np.uint8]        # [vocab_size, 4]
    proj_vectors: NDArray[np.float32]    # [32, embed_dim]
    _prefix_index: dict[int, dict[tuple[int, ...], list[int]]]
    # Semantic gating masks
    _allowed_b0: NDArray[np.bool_] = None  # type: ignore[assignment]
    _allowed_b1: dict[tuple[int], NDArray[np.bool_]] = None  # type: ignore[assignment]
    _allowed_b2: dict[tuple[int, int], NDArray[np.bool_]] = None  # type: ignore[assignment]
    _allowed_b3: dict[tuple[int, int, int], NDArray[np.bool_]] = None  # type: ignore[assignment]
    
    @classmethod
    def build(
        cls,
        embed_tokens: "torch.Tensor",
        vocab_size: int,
        proj_source: str = "first32",
    ) -> "SemanticTokenCodec":
        """
        Build codec from token embeddings.
        
        Args:
            embed_tokens: [vocab_size, embed_dim] token embeddings
            vocab_size: vocabulary size
            proj_source: how to get projection vectors
                - "first32": use first 32 token embeddings (default)
                - "random": use random orthogonal projections
        """
        embed_dim = embed_tokens.shape[1]
        
        # Get projection vectors (32 vectors for 32 sign bits -> 4 bytes)
        if proj_source == "first32":
            proj = embed_tokens[:32].float().clone()
        else:
            # Random orthogonal projections
            proj = torch.randn(32, embed_dim)
            proj, _ = torch.linalg.qr(proj.T)
            proj = proj.T[:32].float()
        
        # Normalize projections for stable sign computation
        norms = torch.norm(proj, dim=1, keepdim=True)
        proj = proj / (norms + 1e-8)
        proj_np = proj.numpy().astype(np.float32)
        
        # Build codes using VECTORIZED bit packing (fast)
        print("Building semantic codebook (vectorized)...")
        code_bytes = cls._build_codes_vectorized(embed_tokens, proj, vocab_size)
        
        # Build prefix index
        prefix_index = cls._build_prefix_index_static(code_bytes, vocab_size)
        
        # Build allowed masks for semantic gating
        allowed_b0, allowed_b1, allowed_b2, allowed_b3 = cls._build_allowed_masks(
            code_bytes, vocab_size
        )
        
        n_unique = len(prefix_index[4])
        n_3byte = len(prefix_index[3])
        n_2byte = len(prefix_index[2])
        n_1byte = len(prefix_index[1])
        print(f"  Codebook built: {vocab_size} tokens")
        print(f"  Unique 4-byte codes: {n_unique}")
        print(f"  3-byte buckets: {n_3byte}")
        print(f"  2-byte buckets: {n_2byte}")
        print(f"  1-byte buckets: {n_1byte}")
        
        return cls(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            code_bytes=code_bytes,
            proj_vectors=proj_np,
            _prefix_index=prefix_index,
            _allowed_b0=allowed_b0,
            _allowed_b1=allowed_b1,
            _allowed_b2=allowed_b2,
            _allowed_b3=allowed_b3,
        )
    
    @staticmethod
    def _build_codes_vectorized(
        embed_tokens: "torch.Tensor",
        proj: "torch.Tensor",
        vocab_size: int,
    ) -> NDArray[np.uint8]:
        """Build 4-byte codes using vectorized operations (fast)."""
        codes = np.zeros((vocab_size, 4), dtype=np.uint8)
        
        # Process in batches for memory efficiency
        batch_size = 4096
        for start in range(0, vocab_size, batch_size):
            end = min(start + batch_size, vocab_size)
            
            # Get embeddings for this batch
            embeds = embed_tokens[start:end].float()  # [batch, embed_dim]
            
            # Compute all 32 dot products at once: [batch, 32]
            dots = embeds @ proj.T
            
            # Sign bits: 1 where positive, 0 where negative
            sign_bits = (dots > 0).to(torch.uint8).numpy()  # [batch, 32]
            
            # Pack 32 bits into 4 bytes using numpy vectorized ops
            # byte0 = bits[0:8], byte1 = bits[8:16], etc.
            for byte_idx in range(4):
                bit_start = byte_idx * 8
                # Pack 8 bits into one byte: sum(bit[i] * 2^(7-i))
                weights = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
                byte_vals = sign_bits[:, bit_start:bit_start + 8] @ weights
                codes[start:end, byte_idx] = byte_vals
        
        return codes
    
    @staticmethod
    def _build_prefix_index_static(
        code_bytes: NDArray[np.uint8],
        vocab_size: int,
    ) -> dict[int, dict[tuple[int, ...], list[int]]]:
        """Build prefix index: prefix_length -> {prefix_tuple -> [token_ids]}."""
        index: dict[int, dict[tuple[int, ...], list[int]]] = {
            1: defaultdict(list),
            2: defaultdict(list),
            3: defaultdict(list),
            4: defaultdict(list),
        }
        
        for tid in range(vocab_size):
            code = code_bytes[tid]
            b0, b1, b2, b3 = int(code[0]), int(code[1]), int(code[2]), int(code[3])
            index[1][(b0,)].append(tid)
            index[2][(b0, b1)].append(tid)
            index[3][(b0, b1, b2)].append(tid)
            index[4][(b0, b1, b2, b3)].append(tid)
        
        return index
    
    @staticmethod
    def _build_allowed_masks(
        code_bytes: NDArray[np.uint8],
        vocab_size: int,
    ) -> tuple[
        NDArray[np.bool_],
        dict[tuple[int], NDArray[np.bool_]],
        dict[tuple[int, int], NDArray[np.bool_]],
        dict[tuple[int, int, int], NDArray[np.bool_]],
    ]:
        """
        Build allowed-next-byte masks for semantic gating.
        
        Returns:
            allowed_b0: bool[256] - which bytes can start a valid code
            allowed_b1: {(b0,) -> bool[256]} - valid second bytes
            allowed_b2: {(b0,b1) -> bool[256]} - valid third bytes
            allowed_b3: {(b0,b1,b2) -> bool[256]} - valid fourth bytes
        """
        # Track which byte values appear at each position given prefix
        allowed_b0 = np.zeros(256, dtype=np.bool_)
        allowed_b1: dict[tuple[int], NDArray[np.bool_]] = {}
        allowed_b2: dict[tuple[int, int], NDArray[np.bool_]] = {}
        allowed_b3: dict[tuple[int, int, int], NDArray[np.bool_]] = {}
        
        for tid in range(vocab_size):
            code = code_bytes[tid]
            b0, b1, b2, b3 = int(code[0]), int(code[1]), int(code[2]), int(code[3])
            
            # First byte
            allowed_b0[b0] = True
            
            # Second byte given first
            key1 = (b0,)
            if key1 not in allowed_b1:
                allowed_b1[key1] = np.zeros(256, dtype=np.bool_)
            allowed_b1[key1][b1] = True
            
            # Third byte given first two
            key2 = (b0, b1)
            if key2 not in allowed_b2:
                allowed_b2[key2] = np.zeros(256, dtype=np.bool_)
            allowed_b2[key2][b2] = True
            
            # Fourth byte given first three
            key3 = (b0, b1, b2)
            if key3 not in allowed_b3:
                allowed_b3[key3] = np.zeros(256, dtype=np.bool_)
            allowed_b3[key3][b3] = True
        
        return allowed_b0, allowed_b1, allowed_b2, allowed_b3
    
    def save(self, path: Path) -> None:
        """Save codec to NPZ file."""
        np.savez_compressed(
            path,
            vocab_size=np.array(self.vocab_size, dtype=np.int64),
            embed_dim=np.array(self.embed_dim, dtype=np.int64),
            code_bytes=self.code_bytes,
            proj_vectors=self.proj_vectors,
        )
        print(f"Saved codec to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "SemanticTokenCodec":
        """Load codec from NPZ file."""
        with np.load(path, allow_pickle=False) as z:
            vocab_size = int(z["vocab_size"])
            embed_dim = int(z["embed_dim"])
            code_bytes = z["code_bytes"].astype(np.uint8)
            proj_vectors = z["proj_vectors"].astype(np.float32)
        
        if code_bytes.shape != (vocab_size, 4):
            raise ValueError(f"code_bytes shape mismatch: {code_bytes.shape}")
        if proj_vectors.shape != (32, embed_dim):
            raise ValueError(f"proj_vectors shape mismatch: {proj_vectors.shape}")
        
        # Rebuild prefix index and allowed masks
        prefix_index = cls._build_prefix_index_static(code_bytes, vocab_size)
        allowed_b0, allowed_b1, allowed_b2, allowed_b3 = cls._build_allowed_masks(
            code_bytes, vocab_size
        )
        
        print(f"Loaded codec from {path}: {vocab_size} tokens, {embed_dim}D")
        
        return cls(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            code_bytes=code_bytes,
            proj_vectors=proj_vectors,
            _prefix_index=prefix_index,
            _allowed_b0=allowed_b0,
            _allowed_b1=allowed_b1,
            _allowed_b2=allowed_b2,
            _allowed_b3=allowed_b3,
        )
    
    def encode(self, token_id: int) -> list[int]:
        """Encode token_id -> [b1, b2, b3, b4]."""
        if 0 <= token_id < self.vocab_size:
            return self.code_bytes[token_id].tolist()
        return [0, 0, 0, 0]
    
    def allowed_mask_for_prefix(self, prefix: list[int]) -> NDArray[np.bool_]:
        """
        Get allowed-next-byte mask for semantic gating.
        
        Args:
            prefix: bytes selected so far (0 to 3 elements)
            
        Returns:
            bool[256] mask of valid next bytes
        """
        n = len(prefix)
        if n == 0:
            return self._allowed_b0
        elif n == 1:
            key = (prefix[0],)
            return self._allowed_b1.get(key, np.zeros(256, dtype=np.bool_))
        elif n == 2:
            key = (prefix[0], prefix[1])
            return self._allowed_b2.get(key, np.zeros(256, dtype=np.bool_))
        elif n == 3:
            key = (prefix[0], prefix[1], prefix[2])
            return self._allowed_b3.get(key, np.zeros(256, dtype=np.bool_))
        else:
            # Already have 4 bytes, no more allowed
            return np.zeros(256, dtype=np.bool_)
    
    def decode(
        self,
        planned_bytes: list[int],
        probe: "torch.Tensor",
        embed_tokens: "torch.Tensor",
        stats: dict[str, int] | None = None,
    ) -> int:
        """
        Decode [b1, b2, b3, b4] -> token_id with prefix backoff.
        
        Args:
            planned_bytes: 4-byte code from byte selection
            probe: context probe vector [embed_dim] for scoring candidates
            embed_tokens: full embedding table for candidate scoring
            stats: optional dict to track match levels (exact/3byte/2byte/1byte/fallback)
        
        Returns:
            token_id with best match
        
        This is semantic erasure recovery: when exact code unknown,
        find best token in the semantic neighborhood.
        """
        # Try exact 4-byte match first
        key4 = tuple(planned_bytes)
        if key4 in self._prefix_index[4]:
            candidates = self._prefix_index[4][key4]
            if stats is not None:
                stats["exact"] = stats.get("exact", 0) + 1
            if len(candidates) == 1:
                return candidates[0]
            return self._pick_best(candidates, probe, embed_tokens)
        
        # 3-byte backoff
        key3 = tuple(planned_bytes[:3])
        if key3 in self._prefix_index[3]:
            candidates = self._prefix_index[3][key3]
            if stats is not None:
                stats["3byte"] = stats.get("3byte", 0) + 1
            return self._pick_best(candidates, probe, embed_tokens)
        
        # 2-byte backoff
        key2 = tuple(planned_bytes[:2])
        if key2 in self._prefix_index[2]:
            candidates = self._prefix_index[2][key2]
            if stats is not None:
                stats["2byte"] = stats.get("2byte", 0) + 1
            return self._pick_best(candidates, probe, embed_tokens)
        
        # 1-byte backoff
        key1 = (planned_bytes[0],)
        if key1 in self._prefix_index[1]:
            candidates = self._prefix_index[1][key1]
            if stats is not None:
                stats["1byte"] = stats.get("1byte", 0) + 1
            return self._pick_best(candidates, probe, embed_tokens)
        
        # Fallback (should rarely happen)
        if stats is not None:
            stats["fallback"] = stats.get("fallback", 0) + 1
        return 0
    
    def _pick_best(
        self,
        candidates: list[int],
        probe: "torch.Tensor",
        embed_tokens: "torch.Tensor",
    ) -> int:
        """Pick token with highest cosine similarity to probe."""
        if len(candidates) == 1:
            return candidates[0]
        
        # Score all candidates by COSINE similarity (not raw dot product)
        # This prevents high-norm tokens (often proper nouns) from dominating
        cand_embeds = embed_tokens[candidates].float()
        cand_norm = torch.nn.functional.normalize(cand_embeds, dim=1)
        probe_norm = torch.nn.functional.normalize(probe.float().unsqueeze(0), dim=1)
        scores = (cand_norm @ probe_norm.T).squeeze(1)
        
        best_idx = int(torch.argmax(scores).item())
        return candidates[best_idx]
    
    def get_bucket_size(self, prefix_len: int, prefix: tuple[int, ...]) -> int:
        """Get number of tokens in a prefix bucket."""
        if prefix_len not in self._prefix_index:
            return 0
        if prefix not in self._prefix_index[prefix_len]:
            return 0
        return len(self._prefix_index[prefix_len][prefix])