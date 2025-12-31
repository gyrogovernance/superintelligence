"""
Router constants and core physics.

Defines:
- 24-bit state: (A12, B12)
- transcription: intron = byte XOR 0xAA
- pure XOR transformation
- FIFO gyration (A↔B swap with flip)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

GENE_MIC_S: int = 0xAA

LAYER_MASK_12: int = 0xFFF

ARCHETYPE_A12: int = 0xAAA
ARCHETYPE_B12: int = 0x555


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
    return bin(x).count("1")


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