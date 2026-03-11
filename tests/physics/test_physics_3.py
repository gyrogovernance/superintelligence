"""
Physics tests 3: Affine action algebra of the spinorial kernel.

Each byte defines an affine transform on (A,B) over GF(2)^12:
  A_next = B xor invert_a
  B_next = (A xor mask) xor invert_b

Linear part = swap; translation = (invert_a, mask xor invert_b).
Tests: composition, depth-4 alternation as affine cancellation,
commutator/monodromy on sampled states. No atlas, no Omega.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.constants import (
    GENE_MAC_REST,
    GENE_MIC_S,
    LAYER_MASK_12,
    byte_to_intron,
    pack_state,
    step_state_by_byte,
    unpack_state,
)
from src.api import mask12_for_byte


def _invert_a_b(byte: int) -> tuple[int, int]:
    intron = byte_to_intron(byte)
    inv_a = LAYER_MASK_12 if (intron & 0x01) else 0
    inv_b = LAYER_MASK_12 if (intron & 0x80) else 0
    return inv_a, inv_b


class TestAffineStructure:
    """Single step has swap linear part and family-dependent translation."""

    def test_step_has_swap_linear_part(self):
        """From (A,B), one step gives (B + t_a, A + t_b) for some translations."""
        state = pack_state(0x100, 0x200)
        byte = 0x42
        next_state = step_state_by_byte(state, byte)
        a, b = unpack_state(state)
        new_a, new_b = unpack_state(next_state)
        m12 = mask12_for_byte(byte)
        inv_a, inv_b = _invert_a_b(byte)
        expect_a = (b ^ inv_a) & LAYER_MASK_12
        expect_b = ((a ^ m12) ^ inv_b) & LAYER_MASK_12
        assert new_a == expect_a and new_b == expect_b


class TestDepth4AlternationAffine:
    """xyxy = id from affine cancellation (swap^4 = id, translations cancel)."""

    def test_depth4_alternation_on_sampled_states(self):
        np.random.seed(0)
        for _ in range(500):
            a = int(np.random.randint(0, 4096))
            b = int(np.random.randint(0, 4096))
            s0 = pack_state(a, b)
            x = int(np.random.randint(0, 256))
            y = int(np.random.randint(0, 256))
            s = s0
            for byte in [x, y, x, y]:
                s = step_state_by_byte(s, byte)
            assert s == s0


class TestInverseComposition:
    """Inverse step composes correctly (step then inverse_step returns identity)."""

    def test_step_inverse_roundtrip_sample(self):
        from src.constants import inverse_step_by_byte
        np.random.seed(1)
        for _ in range(300):
            a = int(np.random.randint(0, 4096))
            b = int(np.random.randint(0, 4096))
            s0 = pack_state(a, b)
            byte = int(np.random.randint(0, 256))
            s1 = step_state_by_byte(s0, byte)
            s_back = inverse_step_by_byte(s1, byte)
            assert s_back == s0


def _apply_word(state24: int, word: list[int]) -> int:
    s = state24
    for b in word:
        s = step_state_by_byte(s, b)
    return s


class TestAffineWordSignature:
    """
    Word action is determined by linear parity (id or swap) and 24-bit translation (tau_a, tau_b).
    So T_word(A, B) = either (A + tau_a, B + tau_b) or (B + tau_a, A + tau_b).
    """

    def test_word_action_has_affine_signature(self):
        np.random.seed(2)
        for _ in range(30):
            word_len = 2 + int(np.random.randint(0, 5))
            word = [int(np.random.randint(0, 256)) for _ in range(word_len)]

            s0 = pack_state(0, 0)
            tau_state = _apply_word(s0, word)
            tau_a, tau_b = unpack_state(tau_state)

            s1 = pack_state(1, 0)
            s2 = pack_state(0, 1)
            out1 = _apply_word(s1, word)
            out2 = _apply_word(s2, word)
            a1, b1 = unpack_state(out1)
            a2, b2 = unpack_state(out2)

            is_swap = (a1 == tau_a and (b1 ^ tau_b) == 1)
            is_id = ((a1 ^ tau_a) == 1 and b1 == tau_b)
            assert is_swap or is_id, "linear part must be identity or swap"

            for _ in range(10):
                a = int(np.random.randint(0, 4096))
                b = int(np.random.randint(0, 4096))
                s = pack_state(a, b)
                got = _apply_word(s, word)
                ga, gb = unpack_state(got)
                if is_swap:
                    expect_a = (b ^ tau_a) & LAYER_MASK_12
                    expect_b = (a ^ tau_b) & LAYER_MASK_12
                else:
                    expect_a = (a ^ tau_a) & LAYER_MASK_12
                    expect_b = (b ^ tau_b) & LAYER_MASK_12
                assert ga == expect_a and gb == expect_b
