"""
Holography diagnostics 3: single-step trace and depth-4 observables.

Research-only tests for the current spinorial kernel.

These tests examine where the local hologram of one byte transition lives:
- the 4 internal trace stages of a single step
- the way family phase is discarded by 48-bit mask projection
- the way the 32-bit intron sequence preserves that discarded phase
- exact depth-4 preimage multiplicity: each 48-bit projection has 256 byte-level preimages.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from src.api import MASK12_BY_BYTE, depth4_intron_sequence32, depth4_mask_projection48
from src.constants import (
    GENE_MAC_REST,
    GENE_MIC_S,
    byte_cgm_parities,
    byte_family,
    byte_micro_ref,
    byte_to_intron,
    pack_state,
    single_step_trace,
    step_state_by_byte,
    unpack_state,
)

pytestmark = pytest.mark.research


def test_single_step_trace_matches_forward_step():
    rng = np.random.default_rng(20260302)
    for _ in range(256):
        a = int(rng.integers(0, 4096))
        b = int(rng.integers(0, 4096))
        byte = int(rng.integers(0, 256))
        state24 = pack_state(a, b)
        trace = single_step_trace(state24, byte)
        assert trace["state24"] == step_state_by_byte(state24, byte)


def test_single_step_trace_cs_recovers_byte_structure():
    for byte in range(256):
        trace = single_step_trace(GENE_MAC_REST, byte)
        intron = trace["cs"]
        assert intron == byte_to_intron(byte)
        assert byte_family(byte) == (((intron >> 7) & 1) << 1 | (intron & 1))
        assert byte_micro_ref(byte) == ((intron >> 1) & 0x3F)
        assert byte_cgm_parities(byte) == {
            "L0": ((intron >> 0) & 1) ^ ((intron >> 7) & 1),
            "LI": ((intron >> 1) & 1) ^ ((intron >> 6) & 1),
            "FG": ((intron >> 2) & 1) ^ ((intron >> 5) & 1),
            "BG": ((intron >> 3) & 1) ^ ((intron >> 4) & 1),
        }


def test_reference_trace_is_pure_swap_trace():
    state24 = pack_state(0x123, 0x456)
    trace = single_step_trace(state24, GENE_MIC_S)
    assert trace["cs"] == 0
    assert trace["una"] == 0x123
    assert trace["ona"] == 0x456
    assert trace["bu"] == 0x123
    a1, b1 = unpack_state(trace["state24"])
    assert a1 == 0x456 and b1 == 0x123


def test_depth4_mask_projection_discards_family_phase():
    """
    Fix one micro-reference and vary only family. The 48-bit mask projection
    stays unchanged while the 32-bit intron sequence changes.
    """
    payload = 0b101011
    introns = [
        ((0 << 7) | (payload << 1) | 0),
        ((0 << 7) | (payload << 1) | 1),
        ((1 << 7) | (payload << 1) | 0),
        ((1 << 7) | (payload << 1) | 1),
    ]
    bytes_ = [i ^ GENE_MIC_S for i in introns]

    p48 = {
        depth4_mask_projection48(b, GENE_MIC_S, GENE_MIC_S, GENE_MIC_S)
        for b in bytes_
    }
    s32 = {
        depth4_intron_sequence32(b, GENE_MIC_S, GENE_MIC_S, GENE_MIC_S)
        for b in bytes_
    }

    print("\nDepth-4 family-phase discard")
    print("---------------------------")
    print(f"  48-bit projections: {len(p48)}")
    print(f"  32-bit intron sequences: {len(s32)}")

    assert len(p48) == 1
    assert len(s32) == 4


def _mask_to_bytes() -> dict[int, list[int]]:
    """For each 12-bit mask, list of bytes b with mask12(b) = mask (exactly 4 per mask)."""
    out: dict[int, list[int]] = defaultdict(list)
    for b in range(256):
        m = int(MASK12_BY_BYTE[b]) & 0xFFF
        out[m].append(b)
    return dict(out)


class TestDepth4ExactPreimageMultiplicity:
    """
    For any fixed 4-mask tuple (48-bit projection), there are exactly 4^4 = 256
    byte-level preimages. The 32-bit intron sequence is the lift that resolves the fiber.
    """

    def test_every_48bit_projection_has_exactly_256_preimages(self):
        mask_to_bytes = _mask_to_bytes()
        assert all(len(bs) == 4 for bs in mask_to_bytes.values())
        rng = np.random.default_rng(20260315)
        for _ in range(32):
            b0, b1, b2, b3 = [int(rng.integers(0, 256)) for _ in range(4)]
            p48 = depth4_mask_projection48(b0, b1, b2, b3)
            m0 = (p48 >> 36) & 0xFFF
            m1 = (p48 >> 24) & 0xFFF
            m2 = (p48 >> 12) & 0xFFF
            m3 = p48 & 0xFFF
            preimages = [
                (x0, x1, x2, x3)
                for x0 in mask_to_bytes[m0]
                for x1 in mask_to_bytes[m1]
                for x2 in mask_to_bytes[m2]
                for x3 in mask_to_bytes[m3]
            ]
            assert len(preimages) == 256
            for x0, x1, x2, x3 in preimages:
                assert depth4_mask_projection48(x0, x1, x2, x3) == p48

    def test_48bit_fiber_32bit_lift_injective_on_fiber(self):
        """Within each 256-preimage fiber, the 32-bit intron sequence is injective."""
        mask_to_bytes = _mask_to_bytes()
        rng = np.random.default_rng(20260316)
        for _ in range(16):
            b0, b1, b2, b3 = [int(rng.integers(0, 256)) for _ in range(4)]
            p48 = depth4_mask_projection48(b0, b1, b2, b3)
            m0, m1, m2, m3 = (p48 >> 36) & 0xFFF, (p48 >> 24) & 0xFFF, (p48 >> 12) & 0xFFF, p48 & 0xFFF
            s32_vals = {
                depth4_intron_sequence32(x0, x1, x2, x3)
                for x0 in mask_to_bytes[m0]
                for x1 in mask_to_bytes[m1]
                for x2 in mask_to_bytes[m2]
                for x3 in mask_to_bytes[m3]
            }
            assert len(s32_vals) == 256
