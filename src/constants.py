# src/constants.py
"""
Gyroscopic ASI aQPU Kernel physics: Theoretical constants and transition laws.

24-bit state kernel with CGM byte formalism decomposition,
dipole-pair mask expansion, algebraic primitives.

State:      state24 = (A12 << 12) | B12, each component 12 bits.
Archetype:  GENE_MIC_S = 0xAA. Transcription: intron = byte ^ 0xAA.
Intron:     family = L0 boundary bits (0,7); payload = bits 1-6.
Expansion:  payload bit i flips dipole pair i -> 64 distinct 12-bit masks.
Transition: mutate A by mask; gyration complement controlled by family bits 0,7 (spinorial).

Precomputed tables, integrity checks, and derived functions: see src.api.
"""

from __future__ import annotations

import math

# ================================================================
# Bit-width constants
# ================================================================

LAYER_MASK_12: int = 0xFFF
MASK_STATE24: int = 0xFFFFFF

# ================================================================
# CGM bit masks (palindromic intron positions)
# ================================================================

L0_MASK: int = 0b10000001  # bits 0, 7 — Left Identity (boundary)
LI_MASK: int = 0b01000010  # bits 1, 6 — Left Inverse
FG_MASK: int = 0b00100100  # bits 2, 5 — Forward Gyration
BG_MASK: int = 0b00011000  # bits 3, 4 — Backward Gyration

# ================================================================
# GENE_Mic (archetype) and GENE_Mac (tensor rest state)
# ================================================================

GENE_MIC_S: int = 0xAA

GENE_MAC_A12: int = 0xAAA
GENE_MAC_B12: int = 0x555
GENE_MAC_REST: int = (GENE_MAC_A12 << 12) | GENE_MAC_B12

# ================================================================
# Intrinsic gates (Appendix G: horizon-preserving operations)
# ================================================================

GATE_S_BYTES: tuple[int, int] = (0xAA, 0x54)   # Swap: (A,B) -> (B,A)
GATE_C_BYTES: tuple[int, int] = (0xD5, 0x2B)   # Complement-swap: (A,B) -> (B^F,A^F)
HORIZON_GATE_BYTES: tuple[int, ...] = GATE_S_BYTES + GATE_C_BYTES

# ----------------------------------------
# Native 6-qubit / topology constants
# ----------------------------------------

CHIRALITY_QUBITS_6: int = 6
CHIRALITY_MASK_6: int = 0x3F
EPSILON_6: int = 0x3F
OMEGA_SIZE: int = 4096
HORIZON_SIZE: int = 64
BOUNDARY_SIZE: int = 128
BULK_SIZE: int = OMEGA_SIZE - BOUNDARY_SIZE

PAIR_MASKS_12: tuple[int, ...] = tuple(0x3 << (2 * i) for i in range(6))
GATE_NAMES: tuple[str, ...] = ("id", "S", "C", "F")


def is_on_equality_horizon(state24: int) -> bool:
    """Whether state satisfies A12 = B12 (equality horizon, UNA degeneracy)."""
    a, b = unpack_state(state24)
    return a == b


# ================================================================
# K4 parity check vectors
# ================================================================
# Designed for the old mask structure. Under the new dipole-pair
# expansion their geometric meaning (which pairs they probe) has
# changed. They still work as GF(2) parity checks; re-verify if
# using for geometric/K4 interpretation.

Q0: int = 0x033
Q1: int = 0x0F0

# ================================================================
# Aperture quantization (CGM Byte Formalism §7)
# ================================================================

DELTA_BU: float = 0.195342176580
M_A: float = 1.0 / (2.0 * math.sqrt(2.0 * math.pi))
RHO: float = DELTA_BU / M_A
APERTURE_GAP: float = 1.0 - RHO  # ~0.020699553913
APERTURE_GAP_Q256: int = 5  # best 8-bit dyadic approximation: 5/256


# ================================================================
# State packing
# ================================================================


def pack_state(a12: int, b12: int) -> int:
    """Pack two 12-bit components into a 24-bit state."""
    return ((a12 & LAYER_MASK_12) << 12) | (b12 & LAYER_MASK_12)


def unpack_state(state24: int) -> tuple[int, int]:
    """Unpack a 24-bit state into (A12, B12)."""
    s = int(state24) & MASK_STATE24
    return (s >> 12) & LAYER_MASK_12, s & LAYER_MASK_12


# ================================================================
# Transcription and intron decomposition
# ================================================================


def byte_to_intron(byte: int) -> int:
    """Transcription: intron = byte XOR 0xAA."""
    return (int(byte) & 0xFF) ^ GENE_MIC_S


def intron_family(intron: int) -> int:
    """L0 boundary bits (0, 7) -> 2-bit family index."""
    x = int(intron) & 0xFF
    return ((x >> 7) & 1) << 1 | (x & 1)


def intron_micro_ref(intron: int) -> int:
    """Payload bits (1-6) -> 6-bit micro-reference."""
    return (int(intron) >> 1) & 0x3F


def byte_family(byte: int) -> int:
    """Family index for a byte."""
    return intron_family(byte_to_intron(byte))


def byte_micro_ref(byte: int) -> int:
    """Micro-reference for a byte."""
    return intron_micro_ref(byte_to_intron(byte))


# ================================================================
# CGM stage parities
# ================================================================


def l0_parity(intron: int) -> int:
    """L0 parity: XOR of bits 0 and 7 (CS stage)."""
    x = int(intron) & 0xFF
    return (x & 1) ^ ((x >> 7) & 1)


def li_parity(intron: int) -> int:
    """LI parity: XOR of bits 1 and 6 (UNA stage)."""
    x = int(intron) & 0xFF
    return ((x >> 1) & 1) ^ ((x >> 6) & 1)


def fg_parity(intron: int) -> int:
    """FG parity: XOR of bits 2 and 5 (ONA stage)."""
    x = int(intron) & 0xFF
    return ((x >> 2) & 1) ^ ((x >> 5) & 1)


def bg_parity(intron: int) -> int:
    """BG parity: XOR of bits 3 and 4 (BU stage)."""
    x = int(intron) & 0xFF
    return ((x >> 3) & 1) ^ ((x >> 4) & 1)


def intron_cgm_parities(intron: int) -> dict[str, int]:
    """All four CGM stage parities for an intron."""
    x = int(intron) & 0xFF
    return {
        "L0": (x & 1) ^ ((x >> 7) & 1),
        "LI": ((x >> 1) & 1) ^ ((x >> 6) & 1),
        "FG": ((x >> 2) & 1) ^ ((x >> 5) & 1),
        "BG": ((x >> 3) & 1) ^ ((x >> 4) & 1),
    }


def byte_cgm_parities(byte: int) -> dict[str, int]:
    """All four CGM stage parities for a byte."""
    return intron_cgm_parities(byte_to_intron(byte))


# ================================================================
# Mask expansion
# ================================================================


def micro_ref_to_mask12(micro_ref: int) -> int:
    """
    6-bit payload -> 12-bit mask.
    Payload bit i controls dipole pair i (mask bits 2i and 2i+1).
    """
    m = int(micro_ref) & 0x3F
    mask12 = 0
    for i in range(6):
        if (m >> i) & 1:
            mask12 |= 0x3 << (2 * i)
    return mask12 & LAYER_MASK_12


def expand_intron_to_mask12(intron: int) -> int:
    """Expand intron to 12-bit Type A mask via payload bits."""
    return micro_ref_to_mask12(intron_micro_ref(intron))


_MASK12_BY_INTRON: tuple[int, ...] = tuple(
    expand_intron_to_mask12(i) for i in range(256)
)


# ================================================================
# Transition law (spinorial: family controls gyration complement)
# ================================================================


def _transition_internals(
    state24: int, byte: int
) -> tuple[int, int, int, int, int]:
    """
    Returns (intron, a_mut, a_next, b_next, next_state24).
    Single canonical implementation for step and trace.
    """
    intron = byte_to_intron(byte)
    m12 = _MASK12_BY_INTRON[intron]
    a12 = (int(state24) >> 12) & LAYER_MASK_12
    b12 = int(state24) & LAYER_MASK_12
    a_mut = (a12 ^ m12) & LAYER_MASK_12
    invert_a = LAYER_MASK_12 if (intron & 0x01) else 0
    invert_b = LAYER_MASK_12 if (intron & 0x80) else 0
    a_next = (b12 ^ invert_a) & LAYER_MASK_12
    b_next = (a_mut ^ invert_b) & LAYER_MASK_12
    next_state24 = pack_state(a_next, b_next)
    return intron, a_mut, a_next, b_next, next_state24


def step_state_by_byte(state24: int, byte: int) -> int:
    """
    Single-step transition with spinorial gyration.

    1. Payload (bits 1-6) defines mask -> mutate A.
    2. Family bit 0 controls whether A_next is complemented.
    3. Family bit 7 controls whether B_next is complemented.

    This gives 64 masks x 4 gyration phases = 256 distinct transitions.
    """
    return _transition_internals(state24, byte)[4]


def inverse_step_by_byte(state24: int, byte: int) -> int:
    """
    Inverse of spinorial transition.

    Given (A_next, B_next) and byte:
      B_pred = A_next ^ invert_a
      A_pred = (B_next ^ invert_b) ^ mask
    """
    intron = byte_to_intron(byte)
    m12 = _MASK12_BY_INTRON[intron]
    a_next = (int(state24) >> 12) & LAYER_MASK_12
    b_next = int(state24) & LAYER_MASK_12
    invert_a = LAYER_MASK_12 if (intron & 0x01) else 0
    invert_b = LAYER_MASK_12 if (intron & 0x80) else 0
    b_pred = (a_next ^ invert_a) & LAYER_MASK_12
    a_pred = ((b_next ^ invert_b) ^ m12) & LAYER_MASK_12
    return pack_state(a_pred, b_pred)


def single_step_trace(state24: int, byte: int) -> dict[str, int]:
    """
    Trace the 4 internal CGM stages of a single byte transition.

    Returns: cs (intron), una (a_mut), ona (a_next), bu (b_next), state24.
    """
    intron, a_mut, a_next, b_next, next_state24 = _transition_internals(
        state24, byte
    )
    return {
        "cs": intron,
        "una": a_mut,
        "ona": a_next,
        "bu": b_next,
        "state24": next_state24,
    }


# ================================================================
# State observables
# ================================================================


def popcount(x: int) -> int:
    """Count set bits."""
    return int(x).bit_count()


def archetype_distance(state24: int) -> int:
    """Hamming distance to GENE_Mac rest state."""
    return popcount(int(state24) ^ GENE_MAC_REST)


def horizon_distance(a12: int, b12: int) -> int:
    """Horizon distance: popcount(A12 ^ (B12 ^ 0xFFF)).
    Zero on the S-sector where chirality is maximal."""
    return popcount(int(a12) ^ (int(b12) ^ LAYER_MASK_12))


def is_on_horizon(state24: int) -> bool:
    """Whether state satisfies A12 = B12 ^ 0xFFF
    (complement horizon, maximal chirality, S-sector)."""
    a, b = unpack_state(state24)
    return a == (b ^ LAYER_MASK_12)


def ab_distance(a12: int, b12: int) -> int:
    """A/B Hamming distance (equality-horizon distance). Zero on equality horizon."""
    return popcount(int(a12) ^ int(b12))


def complementarity_invariant(a12: int, b12: int) -> bool:
    """True iff horizon_distance + ab_distance == 12 (antipodal pole conservation)."""
    return horizon_distance(a12, b12) + ab_distance(a12, b12) == 12


# ----------------------------------------
# Intrinsic gate actions (K4 on 24-bit state)
# ----------------------------------------


def apply_gate_S(state24: int) -> int:
    """Gate S (swap): (A, B) -> (B, A)."""
    a, b = unpack_state(state24)
    return pack_state(b, a)


def apply_gate_C(state24: int) -> int:
    """Gate C (complement-swap): (A, B) -> (B^F, A^F), F=0xFFF."""
    a, b = unpack_state(state24)
    return pack_state(b ^ LAYER_MASK_12, a ^ LAYER_MASK_12)


def apply_gate_F(state24: int) -> int:
    """Gate F = S o C (global complement): (A, B) -> (A^F, B^F)."""
    a, b = unpack_state(state24)
    return pack_state(a ^ LAYER_MASK_12, b ^ LAYER_MASK_12)


def apply_gate(state24: int, name: str) -> int:
    """
    Apply intrinsic gate by name: "id", "S", "C", "F".
    K4 = (Z/2)^2; id and F not realizable by a single byte.
    """
    if name == "id":
        return int(state24) & MASK_STATE24
    if name == "S":
        return apply_gate_S(state24)
    if name == "C":
        return apply_gate_C(state24)
    if name == "F":
        return apply_gate_F(state24)
    raise ValueError(f"Unknown gate: {name!r}")


def component_density(component12: int) -> float:
    """Component density: popcount / 12."""
    return popcount(int(component12)) / 12.0


# ================================================================
# K4 vertex charge
# ================================================================


def vertex_charge_from_mask(m12: int) -> int:
    """K4 vertex charge from 12-bit mask. Returns v in {0, 1, 2, 3}."""
    b0 = popcount(int(m12) & Q0) & 1
    b1 = popcount(int(m12) & Q1) & 1
    return (b1 << 1) | b0


# ================================================================
# GF(2) inner product (used by api for dual code and syndromes)
# ================================================================


def dot12(a: int, b: int) -> int:
    """GF(2) inner product on 12-bit vectors."""
    return popcount((int(a) & LAYER_MASK_12) & (int(b) & LAYER_MASK_12)) & 1