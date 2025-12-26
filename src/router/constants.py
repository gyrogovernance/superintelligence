"""
Router constants and core physics.

Defines:
- 24-bit state: (A12, B12)
- transcription: intron = byte XOR 0xAA
- pure XOR transformation
- FIFO gyration (A↔B swap with flip)
- K4/Hodge kernel for aperture
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class K4Kernel:
    """Fixed K4 edge list and cycle projector."""
    edges: NDArray[np.uint8]
    p_cycle: NDArray[np.float64]


def build_k4_kernel() -> K4Kernel:
    edges = np.array(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        dtype=np.uint8,
    )

    B = np.zeros((4, 6), dtype=np.float64)
    for e, (u, v) in enumerate(edges):
        B[int(u), e] = -1.0
        B[int(v), e] = 1.0

    BBt = B @ B.T
    P_grad = B.T @ np.linalg.pinv(BBt) @ B
    P_cycle = np.eye(6) - P_grad
    return K4Kernel(edges=edges, p_cycle=P_cycle)


K4 = build_k4_kernel()


def signed_edge_value(u12: int, v12: int) -> float:
    """Signed correlation on 12-bit space."""
    diff = (int(u12) ^ int(v12)) & LAYER_MASK_12
    d = diff.bit_count()
    return (12.0 - 2.0 * float(d)) / 12.0