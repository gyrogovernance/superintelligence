"""
Adapters that bind external models/tokenizers to the fixed gyroscopic atlas.

This module is intentionally *not* part of the router atlas artifacts:
- atlas/* is model-agnostic kernel geometry.
- adapters are model-specific plumbing.

Provides:
- TokenBinding: external_token_id <-> internal_id (uint16, 0..65535)
- EmbeddingAdapter: projects arbitrary vectors to gyro dimension D = 256*K
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
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