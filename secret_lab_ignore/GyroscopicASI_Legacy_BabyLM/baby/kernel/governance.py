# baby/kernel/governance.py
"""
Governance operations for GyroASI - the core physics of recursive structural alignment.
Minimal, dependency-free: GENEs + ψ + transform + batch helpers.
"""

from functools import reduce
from typing import cast

import numpy as np
from numpy.typing import NDArray

# -------------------------------------------------------------------
# GENEs and constants (self-contained)
# -------------------------------------------------------------------

GENE_Mic_S = 0xAA  # 10101010

# Archetypal 48-bit tensor [4, 2, 3, 2] with alternating ±1
GENE_Mac_S = np.array(
    [
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],  # Layer 0
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],  # Layer 1
        [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],  # Layer 2
        [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],  # Layer 3
    ],
    dtype=np.int8,
)

# Exon families (bit classes)
EXON_LI_MASK = 0b01000010  # UNA   bits (Parity / Reflection)
EXON_FG_MASK = 0b00100100  # ONA   bits (Forward Gyration)
EXON_BG_MASK = 0b00011000  # BU-Eg bits (Backward Gyration)
EXON_DYNAMIC_MASK = EXON_LI_MASK | EXON_FG_MASK | EXON_BG_MASK

EXON_BROADCAST_MASKS = {
    "li": EXON_LI_MASK,
    "fg": EXON_FG_MASK,
    "bg": EXON_BG_MASK,
    "dynamic": EXON_DYNAMIC_MASK,
}

def _build_masks_and_constants() -> tuple[int, int, int, list[int]]:
    """
    Compute FG/BG masks (layer-select) and intron broadcast patterns (ψ-domain).
    Returns (FG_MASK, BG_MASK, FULL_MASK, INTRON_BROADCAST_MASKS_LIST).
    """
    FG, BG = 0, 0
    # Flatten order = C; bit index: ((layer*2 + frame)*3 + row)*2 + col
    for layer in range(4):
        for frame in range(2):
            for row in range(3):
                for col in range(2):
                    bit_index = ((layer * 2 + frame) * 3 + row) * 2 + col
                    if layer in (0, 2):
                        FG |= 1 << bit_index
                    if layer in (1, 3):
                        BG |= 1 << bit_index
    FULL_MASK = (1 << 48) - 1

    intron_broadcast_masks_list: list[int] = []
    for i in range(256):
        # broadcast byte across 6 bytes (to 48 bits)
        mask = 0
        for j in range(6):
            mask |= i << (8 * j)
        intron_broadcast_masks_list.append(mask)

    return FG, BG, FULL_MASK, intron_broadcast_masks_list

FG_MASK, BG_MASK, FULL_MASK, INTRON_BROADCAST_MASKS_LIST = _build_masks_and_constants()
INTRON_BROADCAST_MASKS: NDArray[np.uint64] = np.array(INTRON_BROADCAST_MASKS_LIST, dtype=np.uint64)

# Transform mask per intron (precompute once)
XFORM_MASK = np.empty(256, dtype=np.uint64)
for i in range(256):
    m = 0
    if i & EXON_LI_MASK:
        m ^= FULL_MASK
    if i & EXON_FG_MASK:
        m ^= FG_MASK
    if i & EXON_BG_MASK:
        m ^= BG_MASK
    XFORM_MASK[i] = m

# -------------------------------------------------------------------
# Boundary, ψ, fold
# -------------------------------------------------------------------

def tensor_to_int(tensor: NDArray[np.int8]) -> int:
    """+1→0, -1→1; pack to 48-bit big-endian integer."""
    if tensor.shape != (4, 2, 3, 2):
        raise ValueError(f"Expected tensor shape (4,2,3,2), got {tensor.shape}")
    bits = (tensor.flatten(order="C") == -1).astype(np.uint8)
    packed = np.packbits(bits, bitorder="big")
    return int.from_bytes(packed.tobytes(), "big")

def int_to_tensor(state: int) -> NDArray[np.int8]:
    """Convert 48-bit integer back to tensor [4,2,3,2]."""
    # Convert to 48-bit binary representation
    bits = [(state >> i) & 1 for i in range(47, -1, -1)]

    # Convert bits to tensor values: 0→+1, 1→-1
    tensor_flat = np.array([1 if bit == 0 else -1 for bit in bits], dtype=np.int8)

    # Reshape to [4,2,3,2] using C order
    return tensor_flat.reshape((4, 2, 3, 2), order="C")

def transcribe_byte(byte: int) -> int:
    """ψ: byte → intron via XOR with GENE_Mic_S."""
    return (byte & 0xFF) ^ GENE_Mic_S

MASK8 = 0xFF

def fold(a: int, b: int) -> int:
    """
    Monodromic fold on 8-bit masks.
    Canonical: a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b))
    """
    a &= MASK8
    b &= MASK8
    gyr_b = b ^ (a & (~b & MASK8))
    return (a ^ gyr_b) & MASK8

def fold_sequence(introns: list[int], start: int = 0) -> int:
    return reduce(fold, introns, start)

# -------------------------------------------------------------------
# Physics transform
# -------------------------------------------------------------------

def apply_gyration_and_transform(state_int: int, intron: int) -> int:
    """
    Single-step state transform in the 48-bit manifold.
    """
    state_int = int(state_int) & ((1 << 48) - 1)
    ii = int(intron) & 0xFF
    temp = state_int ^ int(XFORM_MASK[ii])
    pattern = int(INTRON_BROADCAST_MASKS[ii])
    final_state = temp ^ (temp & pattern)
    return final_state & ((1 << 48) - 1)

def apply_gyration_and_transform_batch(states: NDArray[np.uint64], intron: int) -> NDArray[np.uint64]:
    ii = int(intron) & 0xFF
    mask = XFORM_MASK[ii]
    pattern = INTRON_BROADCAST_MASKS[ii]
    temp = states ^ mask
    res = temp ^ (temp & pattern)
    return cast("NDArray[np.uint64]", res.astype(np.uint64))

def apply_gyration_and_transform_all_introns(states: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """
    Vectorized successors: shape (states.size, 256).
    """
    temp = states[:, np.newaxis] ^ XFORM_MASK[np.newaxis, :]
    res = temp ^ (temp & INTRON_BROADCAST_MASKS[np.newaxis, :])
    return res.astype(np.uint64)

# -------------------------------------------------------------------
# Integrity check
# -------------------------------------------------------------------

def _validate_gene_mac_s() -> None:
    exp = np.array(
        [
            [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
            [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
            [[[-1, 1], [-1, 1], [-1, 1]], [[1, -1], [1, -1], [1, -1]]],
            [[[1, -1], [1, -1], [1, -1]], [[-1, 1], [-1, 1], [-1, 1]]],
        ],
        dtype=np.int8,
    )
    if GENE_Mac_S.shape != (4, 2, 3, 2) or GENE_Mac_S.dtype != np.int8:
        raise RuntimeError("GENE_Mac_S structure invalid")
    if not np.array_equal(np.unique(GENE_Mac_S), np.array([-1, 1], dtype=np.int8)):
        raise RuntimeError("GENE_Mac_S values must be ±1")
    if not np.array_equal(GENE_Mac_S, exp):
        raise RuntimeError("GENE_Mac_S pattern mismatch")

def _roundtrip_sanity():
    """Verify int↔tensor bit order is consistent."""
    T0 = GENE_Mac_S.copy()
    s = tensor_to_int(T0)
    T1 = int_to_tensor(s)
    if not np.array_equal(T0, T1):
        raise RuntimeError("tensor_to_int/int_to_tensor round-trip failed")

_validate_gene_mac_s()
_roundtrip_sanity()
