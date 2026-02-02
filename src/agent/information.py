"""
Agent constants and observables.

Defines:
- CGM invariants (A*, δ_BU, m_a)
- Inference parameters (K, η)
- Horizon index extraction
- Spectral observables
- Deterministic byte feature vectors

Note: K₄ parity vectors Q0/Q1 and vertex_charge_from_mask are now
in router/constants.py to avoid circular imports during atlas building.
"""

from __future__ import annotations

# Import kernel constants
from src.router.constants import (
    ARCHETYPE_A12,
    ARCHETYPE_STATE24,
    LAYER_MASK_12,
    mask12_for_byte,
    popcount,
    vertex_charge_from_mask,
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
# Inference Parameters
# =============================================================================

# Default learning rate (from aperture gap)
ETA_DEFAULT: float = DELTA_A

# Standard K values (channels per horizon)
# K=43 is special: OLMo MLP intermediate = 11008 = 256 * 43
K_VALUES: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 43)

# Minimal K
K_MIN: int = 1

# Corresponding embedding dimensions D = 256 * K
D_VALUES: tuple[int, ...] = tuple(256 * k for k in K_VALUES)

# M field clipping bound for numerical stability
M_CLIP: float = 10.0


# =============================================================================
# Token/Embedding/Channel Dimensions
# =============================================================================

TOKEN_PACKS = ("int8", "int16", "int32", "float16", "float32", "float64")
EMBEDDING_DIMS = (128, 256, 512, 1024, 2048, 4096)
CHANNEL_DIMS = (16, 32, 48, 64, 96, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)


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
# Vertex Charge (K₄ Structure) - delegates to router/constants.py
# =============================================================================

def vertex_charge(m12: int) -> int:
    """
    Compute K₄ vertex charge from 12-bit mask.
    
    Uses parity check vectors Q0 = 0x033, Q1 = 0x0F0.
    Returns: v ∈ {0, 1, 2, 3}
    """
    return vertex_charge_from_mask(m12)


def vertex_charge_for_byte(byte: int) -> int:
    """Vertex charge for a byte's mask."""
    return vertex_charge_from_mask(mask12_for_byte(byte))


def vertex_charge_for_state(state24: int) -> int:
    """Vertex charge for a kernel state (based on A component)."""
    a = (state24 >> 12) & LAYER_MASK_12
    mask = a ^ ARCHETYPE_A12
    return vertex_charge_from_mask(mask)


# =============================================================================
# Spectral Observables
# =============================================================================

def mask_weight(byte: int) -> float:
    """Normalised mask weight for byte: popcount / 12."""
    return popcount(mask12_for_byte(byte)) / 12.0


def archetype_distance_normalised(state24: int) -> float:
    """Normalised Hamming distance to archetype: [0, 1]."""
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
    w = popcount(delta_mask) / 12.0
    
    chi_prev = vertex_charge_for_byte(h_prev)
    chi_curr = vertex_charge_for_byte(h_curr)
    
    if chi_prev == chi_curr:
        sign = 0.5
    elif (chi_prev ^ chi_curr) == 3:
        sign = -1.0
    else:
        sign = 1.0
    
    return sign * (w + 0.1)

