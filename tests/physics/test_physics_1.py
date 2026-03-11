"""
Physics tests 1: Conformance and correctness of the kernel physics.

Strict baseline for:
- State representation (pack/unpack)
- Transcription (byte -> intron)
- Intron decomposition (family, micro_ref)
- 64-mask expansion + dipole flip
- Spinorial step and inverse
- Reference byte is pure swap
- Depth-4 alternation identity

No atlas, no XFORM_MASK_BY_BYTE, no ARCHETYPE_* names.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.api import mask12_for_byte, MASK12_BY_BYTE, so3_shadow_count
from src.constants import (
    GENE_MAC_A12,
    GENE_MAC_B12,
    GENE_MAC_REST,
    GENE_MIC_S,
    LAYER_MASK_12,
    byte_to_intron,
    expand_intron_to_mask12,
    intron_family,
    intron_micro_ref,
    inverse_step_by_byte,
    pack_state,
    step_state_by_byte,
    unpack_state,
)


class TestStateRepresentation:
    """Test 24-bit state packing and unpacking."""

    def test_pack_unpack_rest_state(self):
        """GENE_Mac rest state should round-trip correctly."""
        a, b = unpack_state(GENE_MAC_REST)
        assert a == GENE_MAC_A12
        assert b == GENE_MAC_B12
        assert pack_state(a, b) == GENE_MAC_REST

    def test_pack_unpack_invertible(self):
        """Pack/unpack must be invertible for all valid components."""
        for a_in, b_in in [(0x000, 0x000), (0xFFF, 0xFFF), (0xAAA, 0x555), (0x123, 0x456)]:
            packed = pack_state(a_in, b_in)
            a_out, b_out = unpack_state(packed)
            assert a_out == a_in and b_out == b_in

    def test_component_isolation(self):
        """A and B components should not interfere."""
        s1 = pack_state(0xFFF, 0x000)
        a1, b1 = unpack_state(s1)
        assert a1 == 0xFFF and b1 == 0x000
        s2 = pack_state(0x000, 0xFFF)
        a2, b2 = unpack_state(s2)
        assert a2 == 0x000 and b2 == 0xFFF


class TestTranscription:
    """Test byte -> intron transcription (XOR with 0xAA)."""

    def test_gene_mic_s_constant(self):
        assert GENE_MIC_S == 0xAA

    def test_transcription_involution(self):
        for byte in range(256):
            assert byte_to_intron(byte_to_intron(byte)) == byte

    def test_transcription_bijective(self):
        introns = set(byte_to_intron(b) for b in range(256))
        assert len(introns) == 256

    def test_specific_transcriptions(self):
        cases = [(0x00, 0xAA), (0xAA, 0x00), (0xFF, 0x55), (0x55, 0xFF)]
        for byte, expected_intron in cases:
            assert byte_to_intron(byte) == expected_intron


class TestIntronDecomposition:
    """Family = L0 bits (0,7); micro_ref = payload bits (1-6)."""

    def test_family_uses_L0_bits_0_and_7(self):
        """Family index must come from intron bits 0 and 7 (L0 boundary), not bit 6."""
        intron_bit7_zero = 0x00
        intron_bit7_one = 0x80
        f0 = intron_family(intron_bit7_zero)
        f1 = intron_family(intron_bit7_one)
        assert f0 != f1, "Changing bit 7 must change family (L0 = bits 0,7)"
        intron_bit6_zero = 0x00
        intron_bit6_one = 0x40
        f0b = intron_family(intron_bit6_zero)
        f1b = intron_family(intron_bit6_one)
        assert f0b == f1b, "Changing only bit 6 must not change family (payload bit)"

    def test_family_in_range(self):
        for byte in range(256):
            f = intron_family(byte_to_intron(byte))
            assert 0 <= f <= 3

    def test_four_families_of_64(self):
        families = [intron_family(byte_to_intron(b)) for b in range(256)]
        for f in range(4):
            assert families.count(f) == 64

    def test_micro_ref_in_range(self):
        for byte in range(256):
            m = intron_micro_ref(byte_to_intron(byte))
            assert 0 <= m <= 63


class TestFamilyActsOnlyThroughComplementPhase:
    """Family bits control only complement phase; same micro_ref -> same mask and a_mut."""

    def test_family_phase_only_differentiates_a_next_b_next(self):
        """Four bytes sharing same micro_ref: same mask12, same a_mut; only (a_next,b_next) differ by complements."""
        from src.constants import LAYER_MASK_12

        # Four introns with same payload (micro_ref 0): bits 1-6 = 0; bits 0,7 vary -> families 0,1,2,3
        introns_same_payload = [0x00, 0x01, 0x80, 0x81]
        bytes_same_micro = [0xAA ^ i for i in introns_same_payload]
        masks = [mask12_for_byte(b) for b in bytes_same_micro]
        assert len(set(masks)) == 1, "same micro_ref implies same mask12"
        m12 = masks[0]

        a12, b12 = 0x100, 0x200
        state = pack_state(a12, b12)
        a_mut = (a12 ^ m12) & LAYER_MASK_12

        seen_next = set()
        for b in bytes_same_micro:
            next_state = step_state_by_byte(state, b)
            an, bn = unpack_state(next_state)
            seen_next.add((an, bn))
        assert len(seen_next) == 4, "four families -> four distinct (a_next, b_next)"
        expected = {
            (b12, a_mut),
            (b12 ^ LAYER_MASK_12, a_mut),
            (b12, a_mut ^ LAYER_MASK_12),
            (b12 ^ LAYER_MASK_12, a_mut ^ LAYER_MASK_12),
        }
        assert seen_next == expected, "only complement phase differs"


class TestExpansion:
    """64 distinct masks; 256 distinct (family, mask12) pairs; dipole flip."""

    def test_exactly_64_distinct_masks(self):
        masks = set(mask12_for_byte(b) for b in range(256))
        assert len(masks) == 64

    def test_256_distinct_family_mask_pairs(self):
        pairs = set()
        for b in range(256):
            intron = byte_to_intron(b)
            f = intron_family(intron)
            m = mask12_for_byte(b)
            pairs.add((f, m))
        assert len(pairs) == 256

    def test_dipole_flip(self):
        """Toggling payload bit i changes exactly one 2-bit pair in the mask."""
        for base_payload in range(64):
            for bit in range(6):
                flipped = base_payload ^ (1 << bit)
                intron_a = (base_payload << 1) & 0x7E
                intron_b = (flipped << 1) & 0x7E
                mask_a = expand_intron_to_mask12(intron_a)
                mask_b = expand_intron_to_mask12(intron_b)
                diff = mask_a ^ mask_b
                expected_pair = 0x3 << (2 * bit)
                assert diff == expected_pair


class TestFIFOGyrationSpinorial:
    """Spinorial gyration: invert_a from intron bit 0, invert_b from intron bit 7."""

    def test_gyration_uses_spinorial_complement(self):
        state = GENE_MAC_REST
        byte = 0x42
        intron = byte_to_intron(byte)
        a, b = unpack_state(state)
        m12 = mask12_for_byte(byte)
        invert_a = LAYER_MASK_12 if (intron & 0x01) else 0
        invert_b = LAYER_MASK_12 if (intron & 0x80) else 0

        next_state = step_state_by_byte(state, byte)
        new_a, new_b = unpack_state(next_state)

        assert new_a == (b ^ invert_a) & LAYER_MASK_12
        assert new_b == ((a ^ m12) ^ invert_b) & LAYER_MASK_12


class TestReferenceByteIsPureSwap:
    """Byte 0xAA: intron 0 -> mask 0, invert_a=0, invert_b=0 -> (A,B) -> (B,A)."""

    def test_reference_byte_is_pure_swap(self):
        """0xAA swaps A and B with no complement."""
        state = pack_state(0x123, 0x456)
        next_state = step_state_by_byte(state, 0xAA)
        new_a, new_b = unpack_state(next_state)
        assert new_a == 0x456 and new_b == 0x123

    def test_reference_byte_fixed_points_are_a_eq_b(self):
        """Fixed points of 0xAA are exactly states where A == B."""
        for _ in range(500):
            v = np.random.randint(0, 4096)
            state = pack_state(v, v)
            next_state = step_state_by_byte(state, 0xAA)
            assert next_state == state

    def test_nonreference_bytes_are_not_all_pure_swap(self):
        """Some bytes have non-zero mask or invert -> not pure swap."""
        state = pack_state(0x123, 0x456)
        next_aa = step_state_by_byte(state, 0xAA)
        for byte in [0x00, 0x42, 0xFF]:
            if byte == 0xAA:
                continue
            next_b = step_state_by_byte(state, byte)
            assert next_b != next_aa or byte == 0xAA


class TestInverseStep:
    """Inverse of spinorial step."""

    def test_inverse_roundtrip(self):
        np.random.seed(0)
        for _ in range(2000):
            a = int(np.random.randint(0, 4096))
            b = int(np.random.randint(0, 4096))
            s = pack_state(a, b)
            byte = int(np.random.randint(0, 256))
            t = step_state_by_byte(s, byte)
            s_back = inverse_step_by_byte(t, byte)
            assert s_back == s


class TestShadowCount:
    """so3_shadow_count = 128: native holography anchor (SO(3)/SU(2) 2-to-1)."""

    def test_shadow_count_is_128_at_rest(self):
        assert so3_shadow_count(GENE_MAC_REST) == 128

    def test_shadow_count_is_128_on_sampled_states(self):
        np.random.seed(99)
        for _ in range(50):
            a = int(np.random.randint(0, 4096))
            b = int(np.random.randint(0, 4096))
            state = pack_state(a, b)
            assert so3_shadow_count(state) == 128

    def test_shadow_multitude_is_exactly_2_to_1(self):
        """Each of the 128 distinct next states from a given state has exactly 2 preimage bytes."""
        for state in [GENE_MAC_REST, pack_state(0x123, 0x456), pack_state(0xAAA, 0x555)]:
            next_from_b: dict[int, list[int]] = {}
            for b in range(256):
                t = step_state_by_byte(state, b)
                next_from_b.setdefault(t, []).append(b)
            assert len(next_from_b) == 128, f"Expected 128 distinct next states, got {len(next_from_b)}"
            for t, preimages in next_from_b.items():
                assert len(preimages) == 2, (
                    f"Expected exactly 2 preimages per next state, got {len(preimages)} for {t}"
                )


class TestInvariants:
    """Determinism and valid outputs."""

    def test_determinism(self):
        test_states = [GENE_MAC_REST, 0x123456, 0xABC555]
        test_bytes = [0x00, 0x42, 0xAA, 0xFF]
        for state in test_states:
            for byte in test_bytes:
                s1 = step_state_by_byte(state, byte)
                s2 = step_state_by_byte(state, byte)
                assert s1 == s2

    def test_all_bytes_produce_valid_state(self):
        state = GENE_MAC_REST
        for byte in range(256):
            next_state = step_state_by_byte(state, byte)
            a, b = unpack_state(next_state)
            assert 0 <= a <= LAYER_MASK_12 and 0 <= b <= LAYER_MASK_12


class TestDepth4AlternationIdentity:
    """Depth-4 alternation: xyxy = id (affine action with swap; translation cancels)."""

    def test_depth4_alternation_identity_all_pairs(self):
        """For GENE_Mac rest, XYXY = id for all byte pairs x,y."""
        s0 = GENE_MAC_REST
        for x in range(256):
            sx = step_state_by_byte(s0, x)
            for y in range(256):
                s_xy = step_state_by_byte(sx, y)
                s_xyx = step_state_by_byte(s_xy, x)
                s_xyxy = step_state_by_byte(s_xyx, y)
                assert s_xyxy == s0, f"XYXY != id for x={x}, y={y}"


def _bfs_omega() -> tuple[set[int], int, set[int]]:
    """BFS from GENE_MAC_REST. Returns (omega, radius, horizon_states)."""
    visited = {GENE_MAC_REST}
    frontier = {GENE_MAC_REST}
    depth = 0
    while frontier:
        next_frontier = set()
        for s in frontier:
            for b in range(256):
                s_next = step_state_by_byte(s, b)
                if s_next not in visited:
                    visited.add(s_next)
                    next_frontier.add(s_next)
        frontier = next_frontier
        depth += 1
    radius = depth - 1
    horizon = {
        s for s in visited
        if unpack_state(s)[0] == (unpack_state(s)[1] ^ 0xFFF)
    }
    return visited, radius, horizon


class TestExactOmegaTheorems:
    """
    New reachable universe (Omega) is 4096 states, radius 2, 64 horizon.
    Holographic ratio exact: 64^2 = 4096. One-step from horizon covers Omega 4-to-1.
    """

    def test_omega_size_radius_horizon_and_holographic_ratio(self):
        """|Omega| = 4096, radius = 2, horizon = 64, Area^2 = Volume."""
        omega, radius, horizon = _bfs_omega()
        assert len(omega) == 4096, f"|Omega| must be 4096, got {len(omega)}"
        assert radius == 2, f"Radius must be 2, got {radius}"
        assert len(horizon) == 64, f"Horizon in Omega must be 64, got {len(horizon)}"
        assert len(horizon) ** 2 == len(omega), "Holographic ratio: 64^2 = 4096"

    def test_holographic_dictionary_4_to_1(self):
        """From 64 horizon states and all 256 bytes, one-step map covers Omega exactly 4-to-1."""
        omega, _, horizon = _bfs_omega()
        next_states = [
            step_state_by_byte(h, b) for h in horizon for b in range(256)
        ]
        assert set(next_states) == omega, "One-step from horizon must cover Omega"
        for s in omega:
            assert next_states.count(s) == 4, (
                f"Each state in Omega must be hit exactly 4 times, got {next_states.count(s)} for {s}"
            )

    def test_omega_equals_U_cross_V(self):
        """BFS Omega equals the Cartesian product U x V (A_rest xor C64) x (B_rest xor C64)."""
        omega, _, _ = _bfs_omega()
        a_rest, b_rest = unpack_state(GENE_MAC_REST)
        c64 = set(int(m) & 0xFFF for m in MASK12_BY_BYTE)
        U = {a_rest ^ c for c in c64}
        V = {b_rest ^ c for c in c64}
        uv = {pack_state(u, v) for u in U for v in V}
        assert omega == uv, "Omega must equal U x V"
