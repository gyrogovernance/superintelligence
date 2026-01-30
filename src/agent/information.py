"""
Agent constants and observables.

Defines:
- CGM invariants (A*, δ_BU, m_a)
- Phenomenology parameters (K, η)
- Horizon index extraction
- Spectral observables
- Deterministic byte feature vectors
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Import kernel constants
from src.router.constants import (
    ARCHETYPE_A12,
    ARCHETYPE_STATE24,
    LAYER_MASK_12,
    mask12_for_byte,
    popcount,
)


# =============================================================================
# CGM Invariants
# =============================================================================

# Target aperture from CGM (continuous limit)
A_STAR: float = 0.020699553813

# Kernel intrinsic aperture (discrete)
A_KERNEL: float = 5.0 / 256.0  # = 0.01953125

# Aperture gap (canonical learning rate)
DELTA_A: float = abs(A_STAR - A_KERNEL)  # ≈ 0.00116830

# Monodromy defect (from Physics Tests Report)
DELTA_BU: float = 0.1953

# Aperture scale
M_A: float = 0.1995


# =============================================================================
# Phenomenology Parameters
# =============================================================================

# Default learning rate (from aperture gap)
ETA_DEFAULT: float = DELTA_A

# Standard K values (channels per horizon)
K_VALUES: tuple[int, ...] = (3, 6, 12, 16)

# Minimal K
K_MIN: int = 3

# Corresponding embedding dimensions D = 256 * K
D_VALUES: tuple[int, ...] = tuple(256 * k for k in K_VALUES)  # (768, 1536, 3072, 4096)

# M field clipping bound for numerical stability
M_CLIP: float = 10.0


# =============================================================================
# Token/Embedding/Channel Dimensions
# =============================================================================

# Token pack types (for reference, not enforced)
TOKEN_PACKS = ("int8", "int16", "int32", "float16", "float32", "float64")

# Embedding dimensions (powers of 2)
EMBEDDING_DIMS = (128, 256, 512, 1024, 2048, 4096)

# Channel dimensions (2^n × 3^m pattern)
CHANNEL_DIMS = (
    16, 32, 48, 64, 96, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096
)


# =============================================================================
# Horizon Index Lookup
# =============================================================================

def _build_mask_to_byte() -> dict[int, int]:
    """Build inverse lookup: mask12 → byte."""
    return {mask12_for_byte(b): b for b in range(256)}


MASK_TO_BYTE: dict[int, int] = _build_mask_to_byte()


def horizon_index(state24: int) -> int:
    """
    Extract horizon index from kernel state.
    
    For state (A, B), the horizon anchor has A_h = A, B_h = A XOR 0xFFF.
    The horizon index is the byte whose mask equals (A XOR ARCHETYPE_A12).
    
    Returns: h ∈ {0..255}
    """
    a = (state24 >> 12) & LAYER_MASK_12
    target_mask = a ^ ARCHETYPE_A12
    try:
        return MASK_TO_BYTE[target_mask]
    except KeyError as e:
        raise ValueError(f"State not in ontology-mask space; target_mask={target_mask:03x}") from e


def horizon_state(h: int) -> int:
    """
    Construct the horizon state for index h.
    
    Returns: 24-bit state where A = B XOR 0xFFF
    """
    mask = mask12_for_byte(h)
    a = ARCHETYPE_A12 ^ mask
    b = a ^ LAYER_MASK_12
    return ((a & LAYER_MASK_12) << 12) | (b & LAYER_MASK_12)


# =============================================================================
# Vertex Charge (K₄ Structure)
# =============================================================================

# Parity check vectors from Holography Tests
Q0: int = 0x033
Q1: int = 0x0F0


def vertex_charge(m12: int) -> int:
    """
    Compute K₄ vertex charge from 12-bit mask.
    
    Uses parity check vectors q0 = 0x033, q1 = 0x0F0.
    Returns: v ∈ {0, 1, 2, 3}
    """
    b0 = popcount(m12 & Q0) & 1
    b1 = popcount(m12 & Q1) & 1
    return (b1 << 1) | b0


def vertex_charge_for_byte(byte: int) -> int:
    """Vertex charge for a byte's mask."""
    return vertex_charge(mask12_for_byte(byte))


def vertex_charge_for_state(state24: int) -> int:
    """Vertex charge for a kernel state (based on A component)."""
    a = (state24 >> 12) & LAYER_MASK_12
    mask = a ^ ARCHETYPE_A12
    return vertex_charge(mask)


# =============================================================================
# Spectral Observables
# =============================================================================

def mask_weight(byte: int) -> float:
    """Normalized mask weight for byte: popcount / 12."""
    return popcount(mask12_for_byte(byte)) / 12.0


def archetype_distance_normalized(state24: int) -> float:
    """Normalized Hamming distance to archetype: [0, 1]."""
    return popcount(state24 ^ ARCHETYPE_STATE24) / 24.0


# =============================================================================
# Direction Factor (for Hebbian update)
# =============================================================================

def direction_factor(h_prev: int, h_curr: int, delta_mask: int) -> float:
    """
    Order-sensitive scalar for Hebbian update.
    
    Combines:
    - Mask weight (how much structural change)
    - Vertex charge transition (K₄ movement)
    
    Non-symmetric in (h_prev, h_curr): order matters.
    """
    # Mask weight component
    w = popcount(delta_mask) / 12.0
    
    # Vertex charge component
    chi_prev = vertex_charge(mask12_for_byte(h_prev))
    chi_curr = vertex_charge(mask12_for_byte(h_curr))
    
    if chi_prev == chi_curr:
        # Same wedge: moderate positive
        sign = 0.5
    elif (chi_prev ^ chi_curr) == 3:
        # Opposite wedge: negative (maximum opposition)
        sign = -1.0
    else:
        # Adjacent wedge: positive
        sign = 1.0
    
    # Combine: sign × (weight + baseline)
    # Baseline 0.1 prevents zero updates
    return sign * (w + 0.1)


# =============================================================================
# Deterministic Byte Feature Vectors
# =============================================================================

# bit groups respecting the 2×3×2 anatomy
_ROW_GROUPS: tuple[tuple[int, ...], ...] = (
    (0, 1, 6, 7),      # row 0
    (2, 3, 8, 9),      # row 1
    (4, 5, 10, 11),    # row 2
)

_FRAME_ROW_GROUPS: tuple[tuple[int, ...], ...] = (
    (0, 1), (2, 3), (4, 5),       # frame 0 rows 0..2
    (6, 7), (8, 9), (10, 11),     # frame 1 rows 0..2
)


def _mask_bits_pm1(m12: int) -> NDArray[np.float32]:
    """12-bit mask -> length-12 vector in {-1,+1}."""
    v = np.empty(12, dtype=np.float32)
    for i in range(12):
        v[i] = 1.0 if ((m12 >> i) & 1) else -1.0
    return v


def byte_feature_vector(byte: int, K: int) -> NDArray[np.float32]:
    """
    Deterministic, non-learned feature vector f_b ∈ ℝ^K derived ONLY from m12 anatomy.
    Supports K in {3, 6, 12, 16}. (Raise otherwise.)
    """
    m12 = mask12_for_byte(byte)
    bits12 = _mask_bits_pm1(m12)

    if K == 12:
        return bits12

    if K == 3:
        # 3 rows (sum over the 4 bits in each row group)
        out = np.zeros(3, dtype=np.float32)
        for r, idxs in enumerate(_ROW_GROUPS):
            out[r] = sum(bits12[i] for i in idxs) / 4.0
        return out

    if K == 6:
        # 2 frames × 3 rows
        out = np.zeros(6, dtype=np.float32)
        for k, idxs in enumerate(_FRAME_ROW_GROUPS):
            out[k] = sum(bits12[i] for i in idxs) / 2.0
        return out

    if K == 16:
        # 12 mask bits + 4 parity-style scalars (still deterministic, still Router-native)
        out = np.zeros(16, dtype=np.float32)
        out[:12] = bits12
        # additional 4: (global parity, frame0 parity, frame1 parity, vertex parity proxy)
        out[12] = float(popcount(m12) & 1) * 2.0 - 1.0
        out[13] = float(popcount(m12 & 0x03F) & 1) * 2.0 - 1.0
        out[14] = float(popcount((m12 >> 6) & 0x03F) & 1) * 2.0 - 1.0
        out[15] = float(vertex_charge(m12) & 1) * 2.0 - 1.0
        return out

    raise ValueError(f"Unsupported K={K}. Supported: 3, 6, 12, 16.")


def byte_feature_matrix(K: int) -> NDArray[np.float32]:
    """Stack feature vectors for all bytes: F[b,:] = f_b. Shape (256, K)."""
    F = np.zeros((256, K), dtype=np.float32)
    for b in range(256):
        F[b, :] = byte_feature_vector(b, K)
    return F