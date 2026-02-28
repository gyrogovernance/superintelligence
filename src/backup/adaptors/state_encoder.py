"""
Encode Router layer state into Router-native feature vectors.

Multi-Rate structure:
  - FastFeatureBuilder: intron + L2 + last-mask (39 dims), fires every byte
  - SlowFeatureBuilder: L3 + L4 holographic (51 dims), fires at patch boundaries
  - FullFeatureBuilder: both combined (90 dims), for single-operator fallback

RouterFeatureBuilder is kept for backward compatibility (same 90-dim output).
"""

from __future__ import annotations

from typing import Any

import torch

from src.router.constants import LAYER_MASK_12, mask12_for_byte, unpack_state

ARCHETYPE_STATE24 = 0xAAA555


def walsh_expand(feat: torch.Tensor, dim: int) -> torch.Tensor:
    """Expand features to dim via FWHT. Shared by all builders."""
    from .fwht import fwht_1d

    n = int(feat.shape[-1])
    if dim < n + 1:
        raise ValueError("dim too small (need space for DC skip)")
    if dim & (dim - 1) != 0:
        raise ValueError("dim must be power-of-two for Walsh expansion")
    u = torch.zeros(dim, dtype=torch.float32, device=feat.device)
    u[1 : 1 + n] = feat.to(torch.float32)
    return fwht_1d(u)


class FastFeatureBuilder:
    """
    Fast features from L2 + last byte. Fires every byte.

    Structure (all +-1 or normalized):
      1. Last-byte intron decomposition (10 dims)
      2. L2 state bits (16 dims)
      3. Last-byte mask12 bits + weight (13 dims)
    Total: 39 dimensions
    """

    NUM_FEATURES = 39

    @staticmethod
    def build(l2_state16: int, last_byte: int) -> torch.Tensor:
        features: list[float] = []

        intron_val = (int(last_byte) & 0xFF) ^ 0xAA
        micro = intron_val & 0x3F
        family = (intron_val >> 6) & 0x3
        for i in range(6):
            features.append(1.0 if ((micro >> i) & 1) else -1.0)
        features.append(1.0 if (family & 1) else -1.0)
        features.append(1.0 if ((family >> 1) & 1) else -1.0)
        features.append(2.0 * float((family ^ (micro & 0x3)) & 1) - 1.0)
        features.append((bin(intron_val).count("1") / 4.0) - 1.0)

        for i in range(16):
            features.append(1.0 if ((int(l2_state16) >> i) & 1) else -1.0)

        m12 = mask12_for_byte(int(last_byte) & 0xFF) & LAYER_MASK_12
        for i in range(12):
            features.append(1.0 if ((m12 >> i) & 1) else -1.0)
        features.append(bin(m12).count("1") / 12.0)

        return torch.tensor(features, dtype=torch.float32)


class SlowFeatureBuilder:
    """
    Slow features from L3/L4. Fires at patch boundaries.

    Structure (all +-1 or normalized):
      1. L3 24-bit state (24 dims)
      2. L3 observables: horizon_distance/12, archetype_distance/24 (2 dims)
      3. L4 O/E/parity (25 dims)
    Total: 51 dimensions
    """

    NUM_FEATURES = 51

    @staticmethod
    def build(
        l3_state24: int, l4_O: int, l4_E: int, l4_parity: int
    ) -> torch.Tensor:
        features: list[float] = []

        for i in range(24):
            features.append(1.0 if ((int(l3_state24) >> i) & 1) else -1.0)

        a12, b12 = unpack_state(int(l3_state24) & 0xFFFFFF)
        h_dist = bin(a12 ^ ((b12 ^ LAYER_MASK_12) & LAYER_MASK_12)).count("1")
        a_dist = bin((int(l3_state24) & 0xFFFFFF) ^ ARCHETYPE_STATE24).count("1")
        features.append(h_dist / 12.0)
        features.append(a_dist / 24.0)

        for i in range(12):
            features.append(1.0 if ((int(l4_O) >> i) & 1) else -1.0)
        for i in range(12):
            features.append(1.0 if ((int(l4_E) >> i) & 1) else -1.0)
        features.append(1.0 if (int(l4_parity) & 1) else -1.0)

        return torch.tensor(features, dtype=torch.float32)


class FullFeatureBuilder:
    """
    Combined fast + slow features. For single-operator fallback.
    Total: 90 dimensions (39 fast + 51 slow)
    """

    NUM_FEATURES = FastFeatureBuilder.NUM_FEATURES + SlowFeatureBuilder.NUM_FEATURES

    @staticmethod
    def build(
        l2_state16: int,
        l3_state24: int,
        l4_O: int,
        l4_E: int,
        l4_parity: int,
        last_byte: int,
    ) -> torch.Tensor:
        fast = FastFeatureBuilder.build(l2_state16, last_byte)
        slow = SlowFeatureBuilder.build(l3_state24, l4_O, l4_E, l4_parity)
        return torch.cat([fast, slow], dim=0)


class RouterFeatureBuilder:
    """
    Build raw Router feature vector phi in R^D for SPC.
    Backward-compatible wrapper: same 90-dim output as FullFeatureBuilder.
    """

    NUM_CORE_FEATURES = FullFeatureBuilder.NUM_FEATURES

    @staticmethod
    def build_raw(
        l1_state8: int,
        l2_state16: int,
        l3_state24: int,
        l4_O: int,
        l4_E: int,
        l4_parity: int,
        last_byte: int,
        *,
        last_k_bytes: list[int] | None = None,
    ) -> torch.Tensor:
        """Build raw feature vector [90] (l1_state8 unused, kept for API compat)."""
        feat = FullFeatureBuilder.build(
            l2_state16, l3_state24, l4_O, l4_E, l4_parity, last_byte
        )
        if last_k_bytes:
            extra = [
                (float(int(b) & 0xFF) / 127.5) - 1.0 for b in last_k_bytes
            ]
            return torch.cat([feat, torch.tensor(extra, dtype=torch.float32)], dim=0)
        return feat

    @staticmethod
    def walsh_expand(feat: torch.Tensor, dim: int) -> torch.Tensor:
        """Expand raw features to dim via FWHT."""
        return walsh_expand(feat, dim)


class FullStateEncoder:
    """
    Encode the complete 4-layer Router state into R^2048.
    Uses RouterFeatureBuilder for raw features, then Walsh expansion to dim.
    """

    NUM_FEATURES = RouterFeatureBuilder.NUM_CORE_FEATURES

    def __init__(self, dim: int = 2048):
        if dim < self.NUM_FEATURES:
            raise ValueError(f"dim must be >= {self.NUM_FEATURES}")
        self.dim = int(dim)

    def encode(
        self,
        l1_state8: int,
        l2_state16: int,
        l3_state24: int,
        l4_O: int,
        l4_E: int,
        l4_parity: int,
        last_byte: int,
        kernel: Any = None,
        *,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        feat = RouterFeatureBuilder.build_raw(
            l1_state8, l2_state16, l3_state24,
            l4_O, l4_E, l4_parity, last_byte,
        ).to(device)
        return RouterFeatureBuilder.walsh_expand(feat, self.dim)


class RouterStateEncoder2048:
    """
    Deterministic state encoder for (O, E, parity) -> R^2048.
    Intentionally simple and auditable for v0.
    """

    def __init__(self, dim: int = 2048, bit_indices: list[int] | None = None):
        if dim <= 25:
            raise ValueError("dim must be greater than 25")
        self.dim = int(dim)
        self.bit_indices = list(range(24)) if bit_indices is None else list(bit_indices)
        if len(self.bit_indices) < 24:
            raise ValueError("bit_indices must provide at least 24 entries")

    def encode(
        self, O: int, E: int, parity: int, *, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        x = torch.zeros(self.dim, dtype=torch.float32, device=device)
        bits: list[int] = []
        for i in range(12):
            bits.append((int(O) >> i) & 1)
        for i in range(12):
            bits.append((int(E) >> i) & 1)
        for j, bit in enumerate(bits):
            idx = self.bit_indices[j]
            if idx < 0 or idx >= self.dim:
                raise IndexError(f"bit index out of range: {idx}")
            x[idx] = 1.0 if bit else -1.0
        x[24] = 1.0 if (int(parity) & 1) else -1.0
        return x
