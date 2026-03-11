"""
Physics tests 2: Code, dual code, and integrity primitives.

Uses api tables and functions only (no atlas):
- MASK12_BY_BYTE, C_PERP_12
- mask12_for_byte, mask12_syndrome, syndrome_*
- trajectory_parity_commitment, trajectory_commitment_bytes
- depth4 projections (48-bit, 32-bit)

Code structure under new kernel:
- |C| = 64 (unique mask set)
- |C_perp| = 64
- C == C_perp (self-dual)
- Walsh: W(s) in {0, 64}, support size 64, support equals C_perp
"""

from __future__ import annotations

import numpy as np
import pytest

from src.api import (
    C_PERP_12,
    MASK12_BY_BYTE,
    depth4_intron_sequence32,
    depth4_mask_projection48,
    mask12_for_byte,
    mask12_syndrome,
    syndrome_detects_corruption,
    syndrome_is_valid_mask,
    trajectory_commitment_bytes,
    trajectory_parity_commitment,
)
from src.constants import LAYER_MASK_12, byte_to_intron, dot12
from tests._physics_utils import coeffs_poly_1_plus_z2_pow_k


def walsh_at_s(s: int, mask_set: set[int]) -> int:
    """W(s) = sum_{c in C} (-1)^<s,c> over GF(2)."""
    total = 0
    for c in mask_set:
        total += 1 if dot12(s, c) == 0 else -1
    return total


class TestPairDiagonalCode:
    """Unique mask set = diagonal subspace {00,11}^6; weight enumerator (1+z^2)^6."""

    def test_every_unique_mask_has_pair_equal_bits(self):
        """For each unique mask, bits (0,1), (2,3), ... (10,11) are equal."""
        unique_masks = set(MASK12_BY_BYTE)
        for m in unique_masks:
            for i in range(6):
                lo = (m >> (2 * i)) & 1
                hi = (m >> (2 * i + 1)) & 1
                assert lo == hi, f"mask {m:03x} pair {i} differs"

    def test_unique_mask_weight_enumerator_is_1_plus_z2_pow_6(self):
        """Weight distribution of 64 unique masks is (1+z^2)^6."""
        unique_masks = set(MASK12_BY_BYTE)
        weights = [bin(m).count("1") for m in unique_masks]
        counts = [weights.count(w) for w in range(13)]
        expected = coeffs_poly_1_plus_z2_pow_k(6)
        for w in range(13):
            assert counts[w] == expected[w], f"weight {w}: got {counts[w]}, expected {expected[w]}"

    def test_byte_table_weight_enumerator_is_4_times_1_plus_z2_pow_6(self):
        """Byte-level table: 4 families per mask -> 4 * (1+z^2)^6."""
        weights = [bin(m).count("1") for m in MASK12_BY_BYTE]
        counts = [weights.count(w) for w in range(13)]
        base = coeffs_poly_1_plus_z2_pow_k(6)
        for w in range(13):
            assert counts[w] == 4 * base[w], f"weight {w}: got {counts[w]}, expected 4*{base[w]}"


class TestMaskCodeSize:
    """Unique mask set has size 64."""

    def test_unique_mask_set_size_is_64(self):
        masks = set(MASK12_BY_BYTE)
        assert len(masks) == 64

    def test_dual_code_size_is_64(self):
        assert len(C_PERP_12) == 64


class TestSelfDuality:
    """Mask code equals its dual (C == C_perp)."""

    def test_mask_set_equals_dual_code_set(self):
        mask_set = set(MASK12_BY_BYTE)
        dual_set = set(C_PERP_12)
        assert mask_set == dual_set


class TestWalshTheorem:
    """W(s) in {0, 64}; support size 64; support equals C_perp."""

    def test_walsh_values_in_0_or_64(self):
        mask_set = set(MASK12_BY_BYTE)
        for s in range(1 << 12):
            w = walsh_at_s(s, mask_set)
            assert w in (0, 64), f"W({s}) = {w}"

    def test_walsh_support_size_64(self):
        mask_set = set(MASK12_BY_BYTE)
        support = [s for s in range(1 << 12) if walsh_at_s(s, mask_set) != 0]
        assert len(support) == 64

    def test_walsh_support_equals_c_perp(self):
        mask_set = set(MASK12_BY_BYTE)
        support = set(s for s in range(1 << 12) if walsh_at_s(s, mask_set) != 0)
        assert support == set(C_PERP_12)


class TestSyndrome:
    """Syndrome and corruption detection."""

    def test_all_masks_have_zero_syndrome(self):
        for b in range(256):
            m12 = mask12_for_byte(b)
            assert syndrome_is_valid_mask(m12)

    def test_syndrome_detects_single_bit_flip(self):
        m12 = mask12_for_byte(0x42)
        for bit in range(12):
            corrupted = m12 ^ (1 << bit)
            assert syndrome_detects_corruption(corrupted)

    def test_syndrome_bitmap_consistency(self):
        for b in range(256):
            m12 = mask12_for_byte(b)
            assert mask12_syndrome(m12) == 0


class TestDepth4Projections:
    """48-bit mask projection has collisions; 32-bit intron sequence is bijective."""

    def test_48bit_mask_projection_has_collisions(self):
        """Different 4-byte inputs can yield same 48-bit projection (64 masks, 256 bytes)."""
        mask_to_bytes: dict[int, list[int]] = {}
        for b in range(256):
            m = mask12_for_byte(b)
            mask_to_bytes.setdefault(m, []).append(b)
        same_mask = [bs for bs in mask_to_bytes.values() if len(bs) >= 2]
        assert same_mask, "every mask has 4 bytes in 64-mask code"
        b1, b2 = same_mask[0][0], same_mask[0][1]
        p1 = depth4_mask_projection48(b1, 0, 0, 0)
        p2 = depth4_mask_projection48(b2, 0, 0, 0)
        assert p1 == p2 and (b1, 0, 0, 0) != (b2, 0, 0, 0)

    def test_32bit_intron_sequence_injective_sample(self):
        """32-bit intron sequence: 50k random 4-byte inputs yield distinct values."""
        np.random.seed(123)
        seen = set()
        for _ in range(50000):
            b0, b1, b2, b3 = [int(np.random.randint(0, 256)) for _ in range(4)]
            u32 = depth4_intron_sequence32(b0, b1, b2, b3)
            seen.add(u32)
        assert len(seen) == 50000


class TestTrajectoryCommitmentNotFullSignature:
    """trajectory_parity_commitment is integrity-only; same (O,E,parity) can yield different action."""

    def test_same_commitment_different_action(self):
        """Two sequences with same O, E, parity can act differently (family bits affect outcome)."""
        from src.constants import GENE_MAC_REST, step_state_by_byte

        seq1 = [0xAA, 0xAB]
        seq2 = [0x2A, 0x2B]
        c1 = trajectory_parity_commitment(seq1)
        c2 = trajectory_parity_commitment(seq2)
        assert c1 == c2, "same mask sequence -> same (O, E, parity)"

        s0 = GENE_MAC_REST
        for b in seq1:
            s0 = step_state_by_byte(s0, b)
        out1 = s0
        s0 = GENE_MAC_REST
        for b in seq2:
            s0 = step_state_by_byte(s0, b)
        out2 = s0
        assert out1 != out2, "commitment is not a full action signature"


class TestTrajectoryCommitment:
    """Commitment is consistent; serialization format; no overclaim on full invariance."""

    def test_commitment_consistent_with_naive(self):
        """trajectory_parity_commitment matches manual XOR of masks."""
        seq = [0x42, 0x43, 0x44]
        O, E, parity = trajectory_parity_commitment(seq)
        O_naive, E_naive = 0, 0
        for i, b in enumerate(seq):
            m = mask12_for_byte(b)
            if (i % 2) == 0:
                O_naive ^= m
            else:
                E_naive ^= m
        assert O == (O_naive & LAYER_MASK_12)
        assert E == (E_naive & LAYER_MASK_12)
        assert parity == len(seq) % 2

    def test_commitment_serialization_format(self):
        """trajectory_commitment_bytes returns 5 bytes."""
        O, E, parity = 0x123, 0x456, 1
        raw = trajectory_commitment_bytes(O, E, parity)
        assert len(raw) == 5
        assert raw[0:2] == (O & 0xFFF).to_bytes(2, "big")
        assert raw[2:4] == (E & 0xFFF).to_bytes(2, "big")
        assert raw[4] == (parity & 1)
