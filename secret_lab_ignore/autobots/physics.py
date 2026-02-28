"""
Lossless physics computations for the HGT forward pass.

These functions compute exact FSM states using the function face
(bitwise operations). No tables needed at definition time.
No approximation. No src.* imports at runtime.

Total information content: 384 bytes (256 masks x 12 bits).
Everything else is derived from GENE_MIC_S = 0xAA.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

# Constants (duplicated from src.router.constants for deployment independence)
GENE_MIC_S: int = 0xAA
Q0: int = 0x033
Q1: int = 0x0F0
ARCHETYPE_A12: int = 0xAAA
ARCHETYPE_B12: int = 0x555
LAYER_MASK_12: int = 0xFFF
ARCHETYPE_STATE24: int = 0xAAA555


def intron(byte_val: int) -> int:
    """Transcription: intron = byte XOR 0xAA."""
    return (int(byte_val) & 0xFF) ^ GENE_MIC_S


def expand_intron_to_mask12(intron_val: int) -> int:
    """
    Expand 8-bit intron to 12-bit Type-A mask.
    Same mapping as src.router.constants expand_intron_to_mask24 (mask_a part).
    """
    x = int(intron_val) & 0xFF
    frame0_a = x & 0x3F
    frame1_a = ((x >> 6) | ((x & 0x0F) << 2)) & 0x3F
    mask_a = frame0_a | (frame1_a << 6)
    return mask_a & LAYER_MASK_12


def vertex_charge(mask12: int) -> int:
    """
    Compute K4 vertex charge from 12-bit mask.
    Uses parity check vectors Q0 = 0x033, Q1 = 0x0F0.
    Returns: v in {0, 1, 2, 3}
    """

    def popcount(x: int) -> int:
        return int(x).bit_count()

    b0 = popcount(mask12 & Q0) & 1
    b1 = popcount(mask12 & Q1) & 1
    return (b1 << 1) | b0


@torch.jit.script
def compute_l1_trajectory(introns: Tensor) -> Tensor:
    batch: int = introns.shape[0]
    seq: int = introns.shape[1]
    out = torch.zeros(batch, seq, dtype=torch.int32, device=introns.device)
    out[:, 0] = introns[:, 0].to(torch.int32)
    for i in range(1, seq):
        out[:, i] = (out[:, i - 1] ^ introns[:, i].to(torch.int32)) & 0xFF
    return out.to(torch.uint8)


@torch.jit.script
def compute_l2_trajectory(introns: Tensor) -> Tuple[Tensor, Tensor]:
    batch: int = introns.shape[0]
    seq: int = introns.shape[1]
    device = introns.device
    a8 = torch.zeros(batch, seq + 1, dtype=torch.int32, device=device)
    b8 = torch.zeros(batch, seq + 1, dtype=torch.int32, device=device)
    a8[:, 0] = 0xAA
    b8[:, 0] = 0x55
    x = introns.to(torch.int32)
    for i in range(seq):
        a_mut = (a8[:, i] ^ x[:, i]) & 0xFF
        a8[:, i + 1] = (b8[:, i] ^ 0xFF) & 0xFF
        b8[:, i + 1] = (a_mut ^ 0xFF) & 0xFF
    return a8[:, 1:].to(torch.uint8), b8[:, 1:].to(torch.uint8)


@torch.jit.script
def compute_l3_trajectory(introns: Tensor, mask12s: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Full 24-bit: (A12, B12) with mask expansion + gyration.
    introns: [batch, seq], mask12s: [batch, seq] (12-bit values)
    returns: (l3_a12, l3_b12) each [batch, seq] int32
    """
    layer_mask: int = 0xFFF
    arch_a: int = 0xAAA
    arch_b: int = 0x555
    batch, seq = introns.shape[0], introns.shape[1]
    device = introns.device
    a12 = torch.zeros(batch, seq + 1, dtype=torch.int32, device=device)
    b12 = torch.zeros(batch, seq + 1, dtype=torch.int32, device=device)
    a12[:, 0] = arch_a
    b12[:, 0] = arch_b
    m = mask12s.to(torch.int32) & layer_mask
    for i in range(seq):
        a_mut = (a12[:, i] ^ m[:, i]) & layer_mask
        a12[:, i + 1] = (b12[:, i] ^ layer_mask) & layer_mask
        b12[:, i + 1] = (a_mut ^ layer_mask) & layer_mask
    return a12[:, 1:], b12[:, 1:]


@torch.jit.script
def compute_l4_commitments(mask12s: Tensor) -> Tuple[Tensor, Tensor]:
    layer_mask: int = 0xFFF
    batch: int = mask12s.shape[0]
    seq: int = mask12s.shape[1]
    device = mask12s.device
    O = torch.zeros(batch, seq, dtype=torch.int32, device=device)
    E = torch.zeros(batch, seq, dtype=torch.int32, device=device)
    o_acc = torch.zeros(batch, dtype=torch.int32, device=device)
    e_acc = torch.zeros(batch, dtype=torch.int32, device=device)
    m = mask12s.to(torch.int32) & layer_mask
    for i in range(seq):
        if (i & 1) == 0:
            o_acc = (o_acc ^ m[:, i]) & layer_mask
        else:
            e_acc = (e_acc ^ m[:, i]) & layer_mask
        O[:, i] = o_acc
        E[:, i] = e_acc
    return O, E


def compute_mask12_table() -> Tensor:
    """
    Frozen buffer: 256 int32 values (12-bit masks).
    The ONLY precomputed artifact. Same as expand_intron_to_mask12(b ^ GENE_MIC_S)
    for each byte b in 0..255.
    """
    table = torch.zeros(256, dtype=torch.int32)
    for b in range(256):
        intr = intron(b)
        table[b] = expand_intron_to_mask12(intr)
    return table


def _popcount_parity(x: Tensor) -> Tensor:
    """Element-wise popcount mod 2 using parallel bit count. x: int64 tensor."""
    t = x - (torch.bitwise_right_shift(x, 1) & 0x5555555555555555)
    t = (t & 0x3333333333333333) + (torch.bitwise_right_shift(t, 2) & 0x3333333333333333)
    t = (t + torch.bitwise_right_shift(t, 4)) & 0x0F0F0F0F0F0F0F0F
    t = (t + torch.bitwise_right_shift(t, 8) + torch.bitwise_right_shift(t, 16) + torch.bitwise_right_shift(t, 24) + torch.bitwise_right_shift(t, 32)) & 0x7F
    return (t & 1).long()


def compute_vertex_batch(mask12s: Tensor, q0_val: int, q1_val: int) -> Tensor:
    """Batch vertex charge from mask12s [batch, seq]. Returns [batch, seq] long 0-3."""
    m = mask12s.to(torch.int64)
    b0 = _popcount_parity(m & q0_val)
    b1 = _popcount_parity(m & q1_val)
    return torch.bitwise_or(torch.bitwise_left_shift(b1, 1), b0)


def _popcount_tensor(x: Tensor) -> Tensor:
    """Element-wise popcount. x: int64 tensor."""
    t = x - (torch.bitwise_right_shift(x, 1) & 0x5555555555555555)
    t = (t & 0x3333333333333333) + (torch.bitwise_right_shift(t, 2) & 0x3333333333333333)
    t = (t + torch.bitwise_right_shift(t, 4)) & 0x0F0F0F0F0F0F0F0F
    return (t + torch.bitwise_right_shift(t, 8) + torch.bitwise_right_shift(t, 16)
            + torch.bitwise_right_shift(t, 24) + torch.bitwise_right_shift(t, 32)) & 0x7F


def compute_horizon_distance(a12: Tensor, b12: Tensor) -> Tensor:
    """popcount(A12 ^ (B12 ^ 0xFFF)) - distance to holographic boundary."""
    x = (a12 ^ (b12 ^ LAYER_MASK_12)).to(torch.int64)
    return _popcount_tensor(x & 0xFFF).float()


def compute_ab_distance(a12: Tensor, b12: Tensor) -> Tensor:
    """popcount(A12 ^ B12) - chiral imbalance."""
    x = (a12 ^ b12).to(torch.int64)
    return _popcount_tensor(x & 0xFFF).float()


def compute_archetype_distance(state24: Tensor) -> Tensor:
    """popcount(state24 ^ 0xAAA555) - distance to origin."""
    x = (state24 ^ ARCHETYPE_STATE24).to(torch.int64)
    return _popcount_tensor(x & 0xFFFFFF).float()


def step_state_l3_scalar(state24: int, byte_val: int) -> int:
    """Single L3 step: (A12,B12) with mask + gyration. Returns next 24-bit state."""
    a12 = (state24 >> 12) & LAYER_MASK_12
    b12 = state24 & LAYER_MASK_12
    m12 = expand_intron_to_mask12(intron(byte_val & 0xFF)) & LAYER_MASK_12
    a_mut = (a12 ^ m12) & LAYER_MASK_12
    a_next = (b12 ^ LAYER_MASK_12) & LAYER_MASK_12
    b_next = (a_mut ^ LAYER_MASK_12) & LAYER_MASK_12
    return (a_next << 12) | b_next


def compute_component_density(component12: Tensor) -> Tensor:
    """popcount(component12) / 12.0 - phase balance."""
    x = (component12 & LAYER_MASK_12).to(torch.int64)
    return _popcount_tensor(x).float() / 12.0
