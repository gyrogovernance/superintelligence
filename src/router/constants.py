"""
Router constants and core physics.

Defines:
- 24-bit state: (A12, B12)
- transcription: intron = byte XOR 0xAA
- pure XOR transformation
- FIFO gyration (A↔B swap with flip)
- K₄ vertex charge computation
"""

from __future__ import annotations

from typing import Iterable, Union

import numpy as np
from numpy.typing import NDArray

GENE_MIC_S: int = 0xAA

LAYER_MASK_12: int = 0xFFF

ARCHETYPE_A12: int = 0xAAA
ARCHETYPE_B12: int = 0x555


# =============================================================================
# K₄ Parity Check Vectors
# =============================================================================

Q0: int = 0x033
Q1: int = 0x0F0


def pack_state(a12: int, b12: int) -> int:
    """Pack two 12-bit components into a 24-bit state."""
    return ((a12 & LAYER_MASK_12) << 12) | (b12 & LAYER_MASK_12)


def unpack_state(state24: int) -> tuple[int, int]:
    """Unpack a 24-bit state into (A12, B12)."""
    a12 = (state24 >> 12) & LAYER_MASK_12
    b12 = state24 & LAYER_MASK_12
    return a12, b12


ARCHETYPE_STATE24: int = pack_state(ARCHETYPE_A12, ARCHETYPE_B12)


def byte_to_intron(byte: int) -> int:
    """Transcription: intron = byte XOR 0xAA."""
    return (int(byte) & 0xFF) ^ GENE_MIC_S


def expand_intron_to_mask24(intron: int) -> int:
    """
    Spec-aligned expansion: intron -> transformation that MUTATES TYPE A ONLY.
    
    Bottom 12 bits (Type B mask) must be 0 because B is not mutated pre-gyration.
    """
    x = int(intron) & 0xFF
    
    # Direct mapping: use all 8 bits to build 12-bit patterns
    # Frame 0 (bits 0-5): lower 6 bits of intron
    # Frame 1 (bits 6-11): upper 2 bits + lower 4 bits (simple mix)
    frame0_a = x & 0x3F
    frame1_a = ((x >> 6) | ((x & 0x0F) << 2)) & 0x3F
    mask_a = frame0_a | (frame1_a << 6)
    
    # Critical: B is not mutated directly in this tick
    mask_b = 0
    
    return ((mask_a & 0xFFF) << 12) | (mask_b & 0xFFF)


def build_xform_mask_by_byte() -> NDArray[np.uint32]:
    """Precompute 24-bit transformation mask for each byte."""
    masks = np.empty(256, dtype=np.uint32)
    for byte in range(256):
        intron = byte_to_intron(byte)
        masks[byte] = np.uint32(expand_intron_to_mask24(intron))
    return masks


XFORM_MASK_BY_BYTE: NDArray[np.uint32] = build_xform_mask_by_byte()


def step_state_by_byte(state24: int, byte: int) -> int:
    """
    Spec-aligned tick:
      1) intron = byte ^ 0xAA
      2) mutate TYPE A only: A' = A ^ mask_a
      3) gyrate FIFO with flip:
           new_A = old_B ^ 0xFFF
           new_B = A' ^ 0xFFF
    """
    mask24 = int(XFORM_MASK_BY_BYTE[int(byte) & 0xFF])
    mask_a = (mask24 >> 12) & LAYER_MASK_12
    
    a, b = unpack_state(state24)
    
    # Mutate TYPE A only
    a1 = (a ^ mask_a) & LAYER_MASK_12
    
    # FIFO gyration with flip
    new_a = b ^ LAYER_MASK_12
    new_b = a1 ^ LAYER_MASK_12
    
    return pack_state(new_a, new_b)


def popcount(x: int) -> int:
    """Count the number of set bits (population count)."""
    return int(x).bit_count()


def archetype_distance(state24: int) -> int:
    """
    Hamming distance to archetype (canonical observable from §2.2.4).
    
    Returns the number of bits that differ between state24 and ARCHETYPE_STATE24.
    """
    return popcount(state24 ^ ARCHETYPE_STATE24)


def horizon_distance(a12: int, b12: int) -> int:
    """
    Horizon distance (canonical observable from §2.2.4).
    
    Horizon set H = {(a,b): a = (b ^ 0xFFF)}.
    Returns popcount(A12 ^ (B12 ^ 0xFFF)).
    """
    return popcount(a12 ^ (b12 ^ LAYER_MASK_12))


def ab_distance(a12: int, b12: int) -> int:
    """
    A/B Hamming distance (canonical observable from §2.2.4).
    
    Returns popcount(A12 ^ B12).
    """
    return popcount(a12 ^ b12)


def component_density(component12: int) -> float:
    """
    Component density (canonical observable from §2.2.4).
    
    Returns popcount(component12) / 12.0.
    """
    return popcount(component12) / 12.0


def mask12_for_byte(byte: int) -> int:
    """
    Extract the 12-bit Type-A mask for a given byte from the precomputed table.
    
    Used for parity-law and code-based integrity checks.
    """
    mask24 = int(XFORM_MASK_BY_BYTE[int(byte) & 0xFF])
    return (mask24 >> 12) & LAYER_MASK_12


def vertex_charge_from_mask(m12: int) -> int:
    """
    Compute K₄ vertex charge from 12-bit mask.
    
    Uses parity check vectors Q0 = 0x033, Q1 = 0x0F0.
    Returns: v ∈ {0, 1, 2, 3}
    """
    b0 = popcount(m12 & Q0) & 1
    b1 = popcount(m12 & Q1) & 1
    return (b1 << 1) | b0


def dot12(a: int, b: int) -> int:
    """GF(2) inner product on 12-bit vectors."""
    return (popcount((a & LAYER_MASK_12) & (b & LAYER_MASK_12))) & 1


def _compute_c_perp() -> tuple[int, ...]:
    """
    Compute dual code C_perp: all 12-bit vectors orthogonal to all mask codewords.
    
    C_perp = {v: v · c = 0 for all c in C}
    where C is the set of mask12 values from XFORM_MASK_BY_BYTE.
    """
    masks = set(mask12_for_byte(b) for b in range(256))
    c_perp = tuple(s for s in range(1 << 12) if all(dot12(s, c) == 0 for c in masks))
    return c_perp


C_PERP_12: tuple[int, ...] = _compute_c_perp()
assert len(C_PERP_12) == 16, f"Expected |C_PERP_12|=16, got {len(C_PERP_12)}"


ByteItem = Union[int, bytes, bytearray, memoryview]


def trajectory_parity_commitment(items: Iterable[ByteItem]) -> tuple[int, int, int]:
    """
    Fast algebraic integrity check for trajectory histories.
    
    This detects most accidental corruptions (bit-rot, transmission errors).
    It is NOT cryptographically secure. For adversarial integrity, use
    SHA-256 hashes and signature verification.
    
    Args:
        items: iterable of bytes (as int 0-255) or byte-like sequences (bytes, bytearray, memoryview)
        
    Returns:
        (O, E, parity) where:
        - O = XOR of masks at even-indexed positions
        - E = XOR of masks at odd-indexed positions  
        - parity = len(trajectory) % 2
    """
    O = E = 0
    idx = 0
    
    for item in items:
        if isinstance(item, (bytes, bytearray, memoryview)):
            for b in item:
                m = mask12_for_byte(b)
                if idx % 2 == 0:
                    O ^= m
                else:
                    E ^= m
                idx += 1
        else:
            m = mask12_for_byte(int(item) & 0xFF)
            if idx % 2 == 0:
                O ^= m
            else:
                E ^= m
            idx += 1
    
    return (O, E, idx % 2)


def trajectory_commitment_bytes(O: int, E: int, parity: int) -> bytes:
    """
    Encode parity commitment (O, E, parity) as compact 5-byte representation.
    
    Format: O (2 bytes big-endian) || E (2 bytes big-endian) || parity (1 byte)
    """
    return (O & 0xFFF).to_bytes(2, "big") + (E & 0xFFF).to_bytes(2, "big") + bytes([parity & 1])


def syndrome_is_valid_mask(m12: int) -> bool:
    """
    Fast algebraic integrity check: check if m12 is a valid mask (zero syndrome with respect to C_perp).
    
    This detects most accidental corruptions (bit-rot, transmission errors).
    It is NOT cryptographically secure. For adversarial integrity, use
    SHA-256 hashes and signature verification.
    
    Returns True if all dot products with C_perp elements are zero.
    """
    return all(dot12(m12, v) == 0 for v in C_PERP_12)


def syndrome_detects_corruption(m12: int) -> bool:
    """
    Fast algebraic integrity check: check if m12 is corrupted (non-zero syndrome with respect to C_perp).
    
    This detects most accidental corruptions (bit-rot, transmission errors).
    It is NOT cryptographically secure. For adversarial integrity, use
    SHA-256 hashes and signature verification.
    
    Returns True if any dot product with C_perp elements is one.
    """
    return any(dot12(m12, v) == 1 for v in C_PERP_12)


def mask12_syndrome(m12: int) -> int:
    """
    Fast algebraic integrity check: compute 16-bit syndrome bitmap for m12 with respect to C_perp.
    
    This detects most accidental corruptions (bit-rot, transmission errors).
    It is NOT cryptographically secure. For adversarial integrity, use
    SHA-256 hashes and signature verification.
    
    Returns integer with bit i set if dot12(m12, C_PERP_12[i]) == 1.
    """
    syndrome = 0
    for i, v in enumerate(C_PERP_12):
        if dot12(m12, v) == 1:
            syndrome |= (1 << i)
    return syndrome