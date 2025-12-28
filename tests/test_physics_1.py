"""
Physics tests: Core router constants and state transitions.

Tests the fundamental physics:
- State representation (pack/unpack)
- Transcription (byte → intron)
- Expansion (intron → mask)
- FIFO gyration
- Non-commutativity
- Invariant preservation
"""

import pytest
import numpy as np

from src.router.constants import (
    GENE_MIC_S,
    ARCHETYPE_A12,
    ARCHETYPE_B12,
    ARCHETYPE_STATE24,
    LAYER_MASK_12,
    pack_state,
    unpack_state,
    byte_to_intron,
    expand_intron_to_mask24,
    step_state_by_byte,
    XFORM_MASK_BY_BYTE,
)


class TestStateRepresentation:
    """Test 24-bit state packing and unpacking."""

    def test_pack_unpack_archetype(self):
        """Archetype should round-trip correctly."""
        a, b = unpack_state(ARCHETYPE_STATE24)
        assert a == ARCHETYPE_A12, f"A component: expected {hex(ARCHETYPE_A12)}, got {hex(a)}"
        assert b == ARCHETYPE_B12, f"B component: expected {hex(ARCHETYPE_B12)}, got {hex(b)}"
        
        reconstructed = pack_state(a, b)
        assert reconstructed == ARCHETYPE_STATE24

    def test_pack_unpack_invertible(self):
        """Pack/unpack must be invertible for all valid components."""
        test_cases = [
            (0x000, 0x000),
            (0xFFF, 0xFFF),
            (0xAAA, 0x555),
            (0x123, 0x456),
        ]
        
        for a_in, b_in in test_cases:
            packed = pack_state(a_in, b_in)
            a_out, b_out = unpack_state(packed)
            assert a_out == a_in, f"A: {hex(a_in)} → {hex(a_out)}"
            assert b_out == b_in, f"B: {hex(b_in)} → {hex(b_out)}"

    def test_component_isolation(self):
        """A and B components should not interfere."""
        # Set A = all ones, B = all zeros
        s1 = pack_state(0xFFF, 0x000)
        a1, b1 = unpack_state(s1)
        assert a1 == 0xFFF and b1 == 0x000
        
        # Set A = all zeros, B = all ones
        s2 = pack_state(0x000, 0xFFF)
        a2, b2 = unpack_state(s2)
        assert a2 == 0x000 and b2 == 0xFFF


class TestTranscription:
    """Test byte → intron transcription (XOR with 0xAA)."""

    def test_gene_mic_s_constant(self):
        """GENE_MIC_S must be 0xAA."""
        assert GENE_MIC_S == 0xAA

    def test_transcription_involution(self):
        """Applying transcription twice returns original byte."""
        for byte in range(256):
            intron = byte_to_intron(byte)
            recovered = byte_to_intron(intron)
            assert recovered == byte, f"Byte {hex(byte)} did not round-trip"

    def test_transcription_bijective(self):
        """All 256 bytes must produce 256 distinct introns."""
        introns = set(byte_to_intron(b) for b in range(256))
        assert len(introns) == 256

    def test_specific_transcriptions(self):
        """Verify known transcription cases."""
        cases = [
            (0x00, 0xAA),  # 0x00 XOR 0xAA = 0xAA
            (0xAA, 0x00),  # 0xAA XOR 0xAA = 0x00
            (0xFF, 0x55),  # 0xFF XOR 0xAA = 0x55
            (0x55, 0xFF),  # 0x55 XOR 0xAA = 0xFF
        ]
        for byte, expected_intron in cases:
            assert byte_to_intron(byte) == expected_intron


class TestExpansion:
    """Test intron → mask expansion (8 → 12 bits for Type A)."""

    def test_expansion_deterministic(self):
        """Same intron must always produce same mask."""
        for intron in range(256):
            mask1 = expand_intron_to_mask24(intron)
            mask2 = expand_intron_to_mask24(intron)
            assert mask1 == mask2

    def test_expansion_injective(self):
        """All 256 introns must produce distinct masks."""
        masks = set(expand_intron_to_mask24(i) for i in range(256))
        assert len(masks) == 256, f"Only {len(masks)} unique masks (expected 256)"

    def test_type_b_mask_always_zero(self):
        """Bottom 12 bits (Type B mask) must always be 0."""
        for intron in range(256):
            mask24 = expand_intron_to_mask24(intron)
            mask_b = mask24 & LAYER_MASK_12
            assert mask_b == 0, f"Intron {hex(intron)} has non-zero B mask: {hex(mask_b)}"

    def test_type_a_mask_nonzero(self):
        """At least some introns should produce non-zero A masks."""
        nonzero_count = 0
        for intron in range(256):
            mask24 = expand_intron_to_mask24(intron)
            mask_a = (mask24 >> 12) & LAYER_MASK_12
            if mask_a != 0:
                nonzero_count += 1
        
        # At least 200/256 should be non-zero (empirical threshold)
        assert nonzero_count > 200, f"Only {nonzero_count}/256 masks are non-zero"

    def test_precomputed_table_matches(self):
        """XFORM_MASK_BY_BYTE must match expand_intron_to_mask24."""
        for byte in range(256):
            intron = byte_to_intron(byte)
            expected = expand_intron_to_mask24(intron)
            actual = int(XFORM_MASK_BY_BYTE[byte])
            assert actual == expected, f"Byte {hex(byte)}: table={hex(actual)}, computed={hex(expected)}"


class TestFIFOGyration:
    """Test FIFO gyration properties."""

    def test_gyration_is_asymmetric(self):
        """new_A depends on old_B (not on mask), new_B depends on mutated_A."""
        byte = 0x42
        state = ARCHETYPE_STATE24
        _, b = unpack_state(state)
        
        next_state = step_state_by_byte(state, byte)
        new_a, _ = unpack_state(next_state)
        
        # new_A should equal ~old_B
        expected_new_a = b ^ LAYER_MASK_12
        assert new_a == expected_new_a, f"new_A = {hex(new_a)}, expected {hex(expected_new_a)}"

    def test_single_step_changes_state(self):
        """At least some bytes should change the state."""
        state = ARCHETYPE_STATE24
        changed = 0
        for byte in range(256):
            next_state = step_state_by_byte(state, byte)
            if next_state != state:
                changed += 1
        
        assert changed > 250, f"Only {changed}/256 bytes changed the state"

    def test_double_step_not_identity(self):
        """Applying same byte twice should not return to original (except special cases)."""
        state = ARCHETYPE_STATE24
        non_involutions = 0
        
        for byte in range(256):
            s1 = step_state_by_byte(state, byte)
            s2 = step_state_by_byte(s1, byte)
            if s2 != state:
                non_involutions += 1
        
        # Most bytes should not be involutions
        assert non_involutions > 200, f"Only {non_involutions}/256 bytes are non-involutions"


class TestNonCommutativity:
    """Test that byte order matters (non-commutativity from gyration)."""

    def test_order_matters(self):
        """Applying X then Y should differ from Y then X for most pairs."""
        state = ARCHETYPE_STATE24
        test_pairs = [
            (0x00, 0xFF),
            (0x42, 0x24),
            (0xAA, 0x55),
            (0x12, 0x34),
        ]
        
        noncommutative = 0
        for x, y in test_pairs:
            s_xy = step_state_by_byte(step_state_by_byte(state, x), y)
            s_yx = step_state_by_byte(step_state_by_byte(state, y), x)
            if s_xy != s_yx:
                noncommutative += 1
        
        assert noncommutative > 0, "No pairs showed non-commutativity"
        print(f"\n  Non-commutative pairs: {noncommutative}/{len(test_pairs)}")

    def test_commutativity_statistics(self, capsys):
        """
        Diagnostic: measure commutativity across random byte pairs.
        
        Note: This is a diagnostic test. The exact law (P6) states that
        T_y(T_x(s)) = T_x(T_y(s)) iff x=y, so the expected rate is 1/256 ≈ 0.39%.
        This test is kept as a diagnostic but the exact law test is authoritative.
        """
        state = ARCHETYPE_STATE24
        np.random.seed(42)
        
        sample_size = 100
        commutative = 0
        
        for _ in range(sample_size):
            x, y = np.random.randint(0, 256, size=2)
            s_xy = step_state_by_byte(step_state_by_byte(state, x), y)
            s_yx = step_state_by_byte(step_state_by_byte(state, y), x)
            if s_xy == s_yx:
                commutative += 1
        
        comm_rate = 100 * commutative / sample_size
        print(f"\n  Commutativity rate (diagnostic): {comm_rate:.1f}% ({commutative}/{sample_size} pairs, expected ~0.39%)")
        
        # Diagnostic only - no assertion (exact law is tested elsewhere)


class TestCGMChirality:
    """Test CS axiom: gyration provides fundamental chirality."""
    
    def test_gyration_not_pure_swap(self):
        """Gyration must not be a simple A↔B swap (needs the flip)."""
        state = pack_state(0x123, 0x456)
        # Apply identity mask (byte 0xAA → intron 0x00 → mask_a ≈ 0)
        next_state = step_state_by_byte(state, 0xAA)
        
        a, b = unpack_state(state)
        new_a, new_b = unpack_state(next_state)
        
        # If it were pure swap: new_a == b and new_b == a
        # But we have flip, so:
        assert new_a == (b ^ LAYER_MASK_12)
        assert new_b == (a ^ LAYER_MASK_12)
        
        # Not pure swap
        assert new_a != b or new_b != a, "Gyration is pure swap (missing flip)"
    
    def test_gyration_asymmetry(self):
        """new_A depends only on old_B; new_B depends on mutated_A."""
        # This is the CS chirality: right (B→A) preserves horizon structure,
        # left (A→B) incorporates mutation
        
        state = pack_state(0xAAA, 0x555)
        byte = 0x42
        
        mask24 = int(XFORM_MASK_BY_BYTE[byte])
        mask_a = (mask24 >> 12) & LAYER_MASK_12
        
        a, b = unpack_state(state)
        
        # Expected behavior
        expected_new_a = b ^ LAYER_MASK_12  # Right: B → A (preserving)
        expected_new_b = (a ^ mask_a) ^ LAYER_MASK_12  # Left: A mutation → B (altering)
        
        next_state = step_state_by_byte(state, byte)
        actual_new_a, actual_new_b = unpack_state(next_state)
        
        assert actual_new_a == expected_new_a, "Right transition broken"
        assert actual_new_b == expected_new_b, "Left transition broken"


class TestInvariants:
    """Test fundamental physics invariants."""

    def test_state_space_boundedness(self):
        """All states must fit in 24 bits."""
        for _ in range(100):
            a = np.random.randint(0, 4096)
            b = np.random.randint(0, 4096)
            state = pack_state(a, b)
            assert state < (1 << 24), f"State {hex(state)} exceeds 24 bits"

    def test_determinism(self):
        """Same (state, byte) must always produce same next state."""
        test_states = [ARCHETYPE_STATE24, 0x123456, 0xABC555]
        test_bytes = [0x00, 0x42, 0xAA, 0xFF]
        
        for state in test_states:
            for byte in test_bytes:
                s1 = step_state_by_byte(state, byte)
                s2 = step_state_by_byte(state, byte)
                assert s1 == s2

    def test_all_bytes_are_operations(self):
        """All 256 bytes must produce valid state transitions."""
        state = ARCHETYPE_STATE24
        for byte in range(256):
            try:
                next_state = step_state_by_byte(state, byte)
                a, b = unpack_state(next_state)
                assert 0 <= a <= LAYER_MASK_12
                assert 0 <= b <= LAYER_MASK_12
            except Exception as e:
                pytest.fail(f"Byte {hex(byte)} failed: {e}")


class TestClosedFormDepthLaws:
    """Exhaustive algebraic properties implied by the transition law."""

    def _mask_a(self, byte: int) -> int:
        mask24 = int(XFORM_MASK_BY_BYTE[int(byte) & 0xFF])
        return (mask24 >> 12) & LAYER_MASK_12

    def _inverse_step(self, next_state24: int, byte: int) -> int:
        """
        Explicit inverse of step_state_by_byte for a given byte.

        Forward:
          A1 = ~B0
          B1 = ~(A0 ^ m)

        Inverse:
          B0 = ~A1
          A0 = ~(B1 ^ m)
        """
        a1, b1 = unpack_state(next_state24)
        m = self._mask_a(byte)

        b0 = a1 ^ LAYER_MASK_12
        a0 = (b1 ^ m) ^ LAYER_MASK_12
        return pack_state(a0, b0)

    def test_step_is_bijective_with_explicit_inverse(self):
        """For each byte, step must be invertible with the explicit inverse."""
        np.random.seed(0)
        for _ in range(5000):
            a = int(np.random.randint(0, 4096))
            b = int(np.random.randint(0, 4096))
            s = pack_state(a, b)
            byte = int(np.random.randint(0, 256))

            t = step_state_by_byte(s, byte)
            s_back = self._inverse_step(t, byte)
            assert s_back == s

    def test_depth2_decoupling_closed_form(self):
        """
        For any state (A,B) and bytes x,y with masks mx,my:

          step(step((A,B), x), y) = (A ^ mx, B ^ my)

        This is the kernel's exact depth-2 law.
        """
        np.random.seed(1)
        for _ in range(2000):
            a = int(np.random.randint(0, 4096))
            b = int(np.random.randint(0, 4096))
            s = pack_state(a, b)

            x = int(np.random.randint(0, 256))
            y = int(np.random.randint(0, 256))

            mx = self._mask_a(x)
            my = self._mask_a(y)

            s2 = step_state_by_byte(step_state_by_byte(s, x), y)
            expected = pack_state(a ^ mx, b ^ my)

            assert s2 == expected

    def test_depth4_alternation_is_identity(self):
        """
        For any state and bytes x,y:

          x y x y returns to the same state (identity),
          and equals y x y x (BU-Egress discrete analogue).
        """
        np.random.seed(2)
        for _ in range(2000):
            a = int(np.random.randint(0, 4096))
            b = int(np.random.randint(0, 4096))
            s = pack_state(a, b)

            x = int(np.random.randint(0, 256))
            y = int(np.random.randint(0, 256))

            s_xyxy = step_state_by_byte(step_state_by_byte(step_state_by_byte(step_state_by_byte(s, x), y), x), y)
            s_yxyx = step_state_by_byte(step_state_by_byte(step_state_by_byte(step_state_by_byte(s, y), x), y), x)

            assert s_xyxy == s
            assert s_yxyx == s
            assert s_xyxy == s_yxyx

    def test_depth2_commutes_iff_same_byte(self):
        """
        For this kernel: step(step(s,x),y) == step(step(s,y),x) iff x==y.
        """
        s = ARCHETYPE_STATE24  # any state works; archetype is fine
        for x in range(256):
            for y in range(256):
                s_xy = step_state_by_byte(step_state_by_byte(s, x), y)
                s_yx = step_state_by_byte(step_state_by_byte(s, y), x)
                if x == y:
                    assert s_xy == s_yx
                else:
                    assert s_xy != s_yx

    def test_trajectory_closed_form_arbitrary_length(self):
        """
        Trajectory closed form for arbitrary byte sequences.

        For byte sequence b_1...b_n with masks m_i:
        - O = m_1 ⊕ m_3 ⊕ m_5 ⊕ ... (odd positions)
        - E = m_2 ⊕ m_4 ⊕ m_6 ⊕ ... (even positions)

        Then:
        - if n is even: (A_n, B_n) = (A_0 ⊕ O, B_0 ⊕ E)
        - if n is odd: (A_n, B_n) = (~B_0 ⊕ E, ~A_0 ⊕ O)
        """
        np.random.seed(3)
        for _ in range(1000):
            a0 = int(np.random.randint(0, 4096))
            b0 = int(np.random.randint(0, 4096))
            s0 = pack_state(a0, b0)

            # Generate random byte sequence of length 1-10
            n = int(np.random.randint(1, 11))
            bytes_seq = [int(np.random.randint(0, 256)) for _ in range(n)]
            masks = [self._mask_a(b) for b in bytes_seq]

            # Compute O and E
            O = 0
            E = 0
            for i in range(n):
                if (i + 1) % 2 == 1:  # odd position (1-indexed)
                    O ^= masks[i]
                else:  # even position
                    E ^= masks[i]

            # Apply steps manually
            s = s0
            for b in bytes_seq:
                s = step_state_by_byte(s, b)
            an_actual, bn_actual = unpack_state(s)

            # Compute expected using closed form
            if n % 2 == 0:  # even length
                an_expected = a0 ^ O
                bn_expected = b0 ^ E
            else:  # odd length
                an_expected = (b0 ^ LAYER_MASK_12) ^ E
                bn_expected = (a0 ^ LAYER_MASK_12) ^ O

            assert an_actual == an_expected, f"n={n}: A mismatch at step {n}"
            assert bn_actual == bn_expected, f"n={n}: B mismatch at step {n}"


# Statistics summary fixture
@pytest.fixture(scope="session", autouse=True)
def print_physics_summary(request):
    """Print summary statistics after all tests."""
    yield
    
    print("\n" + "="*70)
    print("PHYSICS TEST SUMMARY")
    print("="*10)
    
    # Count unique masks
    unique_masks = len(set(int(XFORM_MASK_BY_BYTE[b]) for b in range(256)))
    print(f"Unique masks: {unique_masks} / 256")
    
    # Count non-zero A masks
    nonzero_a = sum(1 for b in range(256) if (int(XFORM_MASK_BY_BYTE[b]) >> 12) != 0)
    print(f"Non-zero A masks: {nonzero_a} / 256")
    
    # Check B masks (should all be zero)
    nonzero_b = sum(1 for b in range(256) if (int(XFORM_MASK_BY_BYTE[b]) & LAYER_MASK_12) != 0)
    print(f"Non-zero B masks: {nonzero_b} / 256 (expected 0)")
    
    print("="*10)


class TestCSOperatorAndSeparatorLemmas:
    """
    Certify the CS-style separator behavior of byte 0xAA and the A/B chirality
    using exact algebra, without relying on atlas artifacts.
    """

    def _mask_a(self, byte: int) -> int:
        mask24 = int(XFORM_MASK_BY_BYTE[int(byte) & 0xFF])
        return (mask24 >> 12) & LAYER_MASK_12

    def test_R0xAA_is_involution_on_random_states(self):
        """
        R := T_0xAA should be an involution: R(R(s)) = s.
        This is a strong, kernel-native "reference move" invariant.
        """
        np.random.seed(123)
        for _ in range(5000):
            a = int(np.random.randint(0, 4096))
            b = int(np.random.randint(0, 4096))
            s = pack_state(a, b)

            s1 = step_state_by_byte(s, 0xAA)
            s2 = step_state_by_byte(s1, 0xAA)

            assert s2 == s, f"R∘R != id at state {hex(s)}"

    def test_separator_lemma_x_then_AA_updates_A_only(self):
        """
        Exact lemma:
          T_AA(T_x(A,B)) = (A XOR m_x, B)

        i.e. inserting the separator after x writes the mask effect into A only.
        """
        np.random.seed(124)
        for _ in range(2000):
            a = int(np.random.randint(0, 4096))
            b = int(np.random.randint(0, 4096))
            x = int(np.random.randint(0, 256))
            mx = self._mask_a(x)

            s0 = pack_state(a, b)
            s2 = step_state_by_byte(step_state_by_byte(s0, x), 0xAA)

            a2, b2 = unpack_state(s2)
            assert a2 == (a ^ mx), f"A mismatch: got {hex(a2)} expected {hex(a ^ mx)}"
            assert b2 == b, f"B mismatch: got {hex(b2)} expected {hex(b)}"

    def test_separator_lemma_AA_then_x_updates_B_only(self):
        """
        Exact lemma:
          T_x(T_AA(A,B)) = (A, B XOR m_x)

        i.e. inserting the separator before x writes the mask effect into B only.
        """
        np.random.seed(125)
        for _ in range(2000):
            a = int(np.random.randint(0, 4096))
            b = int(np.random.randint(0, 4096))
            x = int(np.random.randint(0, 256))
            mx = self._mask_a(x)

            s0 = pack_state(a, b)
            s2 = step_state_by_byte(step_state_by_byte(s0, 0xAA), x)

            a2, b2 = unpack_state(s2)
            assert a2 == a, f"A mismatch: got {hex(a2)} expected {hex(a)}"
            assert b2 == (b ^ mx), f"B mismatch: got {hex(b2)} expected {hex(b ^ mx)}"

    def test_depth4_alternation_identity_all_pairs_on_archetype(self):
        """
        Strengthen P7 beyond sampling: for the archetype, verify XYXY = id for all x,y.
        This is 256^2 pairs and is cheap.
        """
        s0 = ARCHETYPE_STATE24
        for x in range(256):
            sx = step_state_by_byte(s0, x)
            for y in range(256):
                s_xy = step_state_by_byte(sx, y)
                s_xyx = step_state_by_byte(s_xy, x)
                s_xyxy = step_state_by_byte(s_xyx, y)
                assert s_xyxy == s0, f"XYXY != id at archetype for x={x}, y={y}"


class TestInverseConjugationForm:
    """
    Certify the spec claim: T_x^{-1} = R ∘ T_x ∘ R, where R = T_0xAA.
    
    This shows that reversal can be expressed as a word over the same action alphabet,
    strengthening the "ledger as reversible coordination substrate" story.
    """

    def test_inverse_is_conjugation_by_R(self):
        """
        For any state s and byte x:
          T_x^{-1}(T_x(s)) = R ∘ T_x ∘ R(T_x(s)) = s
        
        where R = T_0xAA.
        """
        np.random.seed(314)
        for _ in range(5000):
            a = int(np.random.randint(0, 4096))
            b = int(np.random.randint(0, 4096))
            s = pack_state(a, b)

            x = int(np.random.randint(0, 256))

            # Forward: apply T_x
            t = step_state_by_byte(s, x)

            # Apply R ∘ T_x ∘ R to t (i.e., apply the inverse using only forward steps)
            t1 = step_state_by_byte(t, 0xAA)   # R
            t2 = step_state_by_byte(t1, x)     # T_x
            s_back = step_state_by_byte(t2, 0xAA)  # R

            assert s_back == s, f"Conjugation inverse failed for byte {x} at state {hex(s)}"