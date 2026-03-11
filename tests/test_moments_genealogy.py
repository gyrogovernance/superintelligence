# tests/test_moments_genealogy.py
"""
Moments Genealogy: depth-4 frame commitments and medium integrity.

Tests the genealogy layer of the Moments Economy Architecture:
- Depth-4 frame records as certification atoms
- Frame commitments: determinism, tamper detection, divergence localization
- Frame commitments vs state-only seals (strictly stronger)
- Parity commitments via kernel API
- Medium policy checks: duplicate identities, over-capacity
- Golden vectors for regression safety

Depends on src.constants, src.api, and src.kernel only.
No atlas files required.

Run:
    python -m pytest tests/test_moments_genealogy.py -v -s
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROGRAM_ROOT = Path(__file__).parent.parent
if str(PROGRAM_ROOT) not in sys.path:
    sys.path.insert(0, str(PROGRAM_ROOT))

from src.api import (
    depth4_intron_sequence32,
    depth4_mask_projection48,
    mask12_for_byte,
    trajectory_parity_commitment,
)
from src.constants import (
    GENE_MAC_REST,
    byte_to_intron,
    step_state_by_byte,
)
from src.kernel import Gyroscopic
from tests._moments_utils import Grant, Shell, identity_anchor, _make_shell


# ---------------------------------------------------------------------------
# Frame record computation (certification atom)
# ---------------------------------------------------------------------------

def frame_record(b0: int, b1: int, b2: int, b3: int) -> tuple[int, int, int]:
    """
    Depth-4 frame record: (mask48, phi_a, phi_b).

    mask48: 48-bit payload projection (4 x 12-bit masks packed).
    phi_a, phi_b: net family-phase invariants that survive depth-4 closure.

    This is the kernel-native certification atom. It is strictly more
    informative than the final 24-bit state alone.
    """
    mask48 = depth4_mask_projection48(b0, b1, b2, b3)

    fams = []
    for b in (b0, b1, b2, b3):
        intron = byte_to_intron(b)
        a_bit = intron & 1
        b_bit = (intron >> 7) & 1
        fams.append((a_bit, b_bit))

    phi_a = fams[0][1] ^ fams[1][0] ^ fams[2][1] ^ fams[3][0]
    phi_b = fams[0][0] ^ fams[1][1] ^ fams[2][0] ^ fams[3][1]

    return (mask48, phi_a, phi_b)


def frame_sequence(payload: bytes) -> list[tuple[int, int, int]]:
    """
    Segment a byte payload into depth-4 frames and compute frame records.

    If len(payload) is not a multiple of 4, the trailing bytes are ignored
    (they don't form a complete frame).
    """
    frames = []
    n = len(payload) - (len(payload) % 4)
    for i in range(0, n, 4):
        fr = frame_record(payload[i], payload[i + 1], payload[i + 2], payload[i + 3])
        frames.append(fr)
    return frames


def apply_word(state: int, word: bytes) -> int:
    """Apply a byte sequence to a state and return the final state."""
    s = state
    for b in word:
        s = step_state_by_byte(s, b)
    return s


# ===================================================================
# Part 1: Frame record determinism and basic properties
# ===================================================================

class TestFrameRecordBasics:
    """Depth-4 frame record: determinism, completeness, sensitivity."""

    def test_frame_record_is_deterministic(self):
        """Same 4 bytes always produce the same frame record."""
        rng = np.random.default_rng(2025)
        for _ in range(1000):
            word = [int(rng.integers(0, 256)) for _ in range(4)]
            r1 = frame_record(*word)
            r2 = frame_record(*word)
            assert r1 == r2

    def test_frame_record_components_have_correct_width(self):
        """mask48 is 48-bit, phi_a and phi_b are single bits."""
        rng = np.random.default_rng(42)
        for _ in range(500):
            word = [int(rng.integers(0, 256)) for _ in range(4)]
            mask48, phi_a, phi_b = frame_record(*word)
            assert 0 <= mask48 < (1 << 48)
            assert phi_a in (0, 1)
            assert phi_b in (0, 1)

    def test_frame_record_detects_every_single_byte_change(self):
        """Changing any single byte in a 4-byte frame always changes the record."""
        rng = np.random.default_rng(99)
        for _ in range(500):
            word = [int(rng.integers(0, 256)) for _ in range(4)]
            r_orig = frame_record(*word)

            for pos in range(4):
                alt = (word[pos] + 1) % 256
                word_alt = word.copy()
                word_alt[pos] = alt
                r_alt = frame_record(*word_alt)
                assert r_alt != r_orig, (
                    f"Frame record unchanged when byte {pos} changed: "
                    f"{word} -> {word_alt}"
                )

    def test_frame_sequence_determinism(self):
        """Same payload always produces the same frame sequence."""
        payload = b"deterministic frame sequence test!!"  # 34 bytes -> 8 frames
        fs1 = frame_sequence(payload)
        fs2 = frame_sequence(payload)
        assert fs1 == fs2
        assert len(fs1) == 8

    def test_frame_sequence_ignores_trailing_bytes(self):
        """Trailing bytes that don't form a complete frame are ignored."""
        payload = b"abcdefghij"  # 10 bytes -> 2 frames, 2 trailing
        frames = frame_sequence(payload)
        assert len(frames) == 2

        # Adding trailing bytes doesn't change existing frames
        payload_ext = b"abcdefghijXY"  # 12 bytes -> 3 frames
        frames_ext = frame_sequence(payload_ext)
        assert len(frames_ext) == 3
        assert frames_ext[:2] == frames


# ===================================================================
# Part 2: Frame commitment is stronger than final state
# ===================================================================

class TestFrameStrongerThanState:
    """
    Depth-4 frame records distinguish histories that collapse
    to the same final state. This is the key advantage over
    state-only seals.
    """

    def test_state_collision_with_different_frames_exists(self):
        """
        Find two 4-byte words that produce the same final state from rest
        but have different frame records.

        This proves that frame records carry strictly more information
        than the final state alone.
        """
        rng = np.random.default_rng(123)
        state_to_frames: dict[int, list[tuple[list[int], tuple[int, int, int]]]] = {}

        for _ in range(100_000):
            word = [int(rng.integers(0, 256)) for _ in range(4)]
            final = apply_word(GENE_MAC_REST, bytes(word))
            fr = frame_record(*word)
            state_to_frames.setdefault(final, []).append((word, fr))

        # Find states reached by multiple words with different frame records
        collisions = 0
        for final_state, entries in state_to_frames.items():
            frame_set = {fr for _, fr in entries}
            if len(frame_set) > 1:
                collisions += 1

        assert collisions > 0, (
            "Expected at least one state reachable by words with different "
            "frame records. This is guaranteed by the 128-way shadow projection."
        )

        print(f"\n  States with frame-distinguishable histories: {collisions}")

    def test_frame_commitment_distinguishes_colliding_histories(self):
        """
        Construct two specific 4-byte words that reach the same state
        but have different frame records, demonstrating that a genealogy
        using frame commitments detects history differences that
        state-only seals miss.
        """
        rng = np.random.default_rng(456)
        found = False

        for _ in range(200_000):
            w1 = [int(rng.integers(0, 256)) for _ in range(4)]
            w2 = [int(rng.integers(0, 256)) for _ in range(4)]

            s1 = apply_word(GENE_MAC_REST, bytes(w1))
            s2 = apply_word(GENE_MAC_REST, bytes(w2))

            if s1 == s2 and w1 != w2:
                fr1 = frame_record(*w1)
                fr2 = frame_record(*w2)
                if fr1 != fr2:
                    found = True
                    print(f"\n  Word 1: {w1} -> state {s1:#08x}, frame {fr1}")
                    print(f"  Word 2: {w2} -> state {s2:#08x}, frame {fr2}")
                    print("  Same state, different frames: frame commitment wins.")
                    break

        assert found, "Should find colliding states with distinct frame records"


# ===================================================================
# Part 3: Genealogy divergence localization
# ===================================================================

class TestDivergenceLocalization:
    """
    When two genealogies (byte logs) diverge at some point,
    frame-level comparison localizes the divergence to the
    specific 4-byte frame where the change occurred.
    """

    def test_divergence_localizes_to_affected_frame(self):
        """
        Two logs that diverge at byte k have identical frame sequences
        up to frame (k // 4 - 1) and differ at frame (k // 4).
        """
        rng = np.random.default_rng(77)

        for _ in range(200):
            log = list(rng.integers(0, 256, size=24).astype(int))

            flip_pos = int(rng.integers(0, 24))
            log_alt = log.copy()
            log_alt[flip_pos] = (log_alt[flip_pos] + 1) % 256

            frames_orig = frame_sequence(bytes(log))
            frames_alt = frame_sequence(bytes(log_alt))

            affected_frame = flip_pos // 4

            # All frames before the affected one must match
            for i in range(affected_frame):
                assert frames_orig[i] == frames_alt[i], (
                    f"Frame {i} should match (flip at byte {flip_pos})"
                )

            # The affected frame must differ
            assert frames_orig[affected_frame] != frames_alt[affected_frame], (
                f"Frame {affected_frame} should differ (flip at byte {flip_pos})"
            )

    def test_divergence_localization_vs_state_comparison(self):
        """
        Frame-level divergence localization is more precise than
        comparing final states.

        Final states can match even when histories diverge (false negative),
        or differ without indicating where (no localization).
        Frame sequences always localize to the exact affected frame.
        """
        rng = np.random.default_rng(88)

        localized_by_frame = 0
        missed_by_state = 0
        total = 0

        for _ in range(500):
            log = list(rng.integers(0, 256, size=20).astype(int))
            flip_pos = int(rng.integers(0, 20))
            log_alt = log.copy()
            log_alt[flip_pos] = (log_alt[flip_pos] + 1) % 256

            # Frame-level comparison
            frames_orig = frame_sequence(bytes(log))
            frames_alt = frame_sequence(bytes(log_alt))
            affected = flip_pos // 4

            if affected < len(frames_orig):
                total += 1
                if frames_orig[affected] != frames_alt[affected]:
                    localized_by_frame += 1

                # State-level comparison
                s_orig = apply_word(GENE_MAC_REST, bytes(log))
                s_alt = apply_word(GENE_MAC_REST, bytes(log_alt))
                if s_orig == s_alt:
                    missed_by_state += 1

        assert localized_by_frame == total, "Frame comparison must catch all divergences"

        print(f"\n  Total divergences tested: {total}")
        print(f"  Localized by frame: {localized_by_frame}/{total}")
        print(f"  Missed by final state: {missed_by_state}/{total}")
        if missed_by_state > 0:
            print("  Frame commitment is strictly stronger than state-only seal.")


# ===================================================================
# Part 4: Parity commitments via kernel API
# ===================================================================

class TestParityCommitments:
    """
    Trajectory parity commitment (O, E, parity) from src.api.
    Tests the API wrapper, not the algebraic proof (which is in physics tests).
    """

    def test_parity_commitment_determinism(self):
        """Same payload always produces the same commitment."""
        payload = b"parity commitment test payload"
        c1 = trajectory_parity_commitment(payload)
        c2 = trajectory_parity_commitment(payload)
        assert c1 == c2

    def test_parity_commitment_detects_single_byte_change(self):
        """
        When a byte change changes the 12-bit mask at that position, the
        commitment must change in the corresponding slot: O for even index,
        E for odd index. This tests the exact contract of trajectory_parity_commitment.
        """
        payload = list(range(40))
        orig_o, orig_e, orig_p = trajectory_parity_commitment(payload)

        for i in range(len(payload)):
            m_old = mask12_for_byte(payload[i])
            b_new = None
            for c in range(256):
                if mask12_for_byte(c) != m_old:
                    b_new = c
                    break
            assert b_new is not None, "every byte shares a mask (kernel invariant)"

            tampered = payload.copy()
            tampered[i] = b_new
            new_o, new_e, new_p = trajectory_parity_commitment(tampered)

            if (i & 1) == 0:
                assert new_o != orig_o, f"even index {i}: O must change when mask changes"
            else:
                assert new_e != orig_e, f"odd index {i}: E must change when mask changes"
            assert (new_o, new_e, new_p) != (orig_o, orig_e, orig_p)

    def test_parity_commitment_structure(self):
        """Commitment is (O, E, parity) with correct types and ranges."""
        payload = b"structure test"
        o, e, p = trajectory_parity_commitment(payload)
        assert isinstance(o, int)
        assert isinstance(e, int)
        assert p in (0, 1)
        assert 0 <= o < (1 << 12)
        assert 0 <= e < (1 << 12)
        assert p == len(payload) % 2


# ===================================================================
# Part 5: Medium policy checks
# ===================================================================

class TestMediumPolicyChecks:
    """
    Application-layer policy conditions that the medium should
    support detecting. These are not kernel constraints but
    economic integrity checks.
    """

    def test_duplicate_identity_detection_in_shell(self):
        """
        A Shell with two grants to the same identity should be detectable.
        The Shell still computes a seal (it's structurally valid),
        but the application layer should flag or reject duplicates.
        """
        ident, anchor = identity_anchor("alice")
        g1 = Grant("alice", ident, anchor, 87_600)
        g2 = Grant("alice", ident, anchor, 87_600)  # duplicate

        shell = Shell(header=b"test", grants=[g1, g2], total_capacity_mu=10**18)
        shell.compute_seal()

        # Shell computes a seal (structurally valid)
        assert len(shell.seal) == 6

        # But application layer can detect the duplicate
        ids = [g.identity_id for g in shell.grants]
        has_duplicate = len(ids) != len(set(ids))
        assert has_duplicate, "Duplicate identity should be detectable"

    def test_over_capacity_detection_in_shell(self):
        """
        A Shell where used > total capacity should be detectable.
        The seal still computes (structural validity), but the
        application layer should flag over-capacity.
        """
        ident, anchor = identity_anchor("alice")
        g = Grant("alice", ident, anchor, 1_000_000)

        shell = Shell(header=b"test", grants=[g], total_capacity_mu=500_000)
        shell.compute_seal()

        assert len(shell.seal) == 6
        assert shell.used_capacity_mu > shell.total_capacity_mu, (
            "Over-capacity should be detectable"
        )
        assert shell.free_capacity_mu < 0

    def test_empty_shell_has_deterministic_seal(self):
        """An empty Shell (no grants) still produces a deterministic seal."""
        s1 = Shell(header=b"empty:2026", grants=[], total_capacity_mu=10**18)
        s1.compute_seal()
        s2 = Shell(header=b"empty:2026", grants=[], total_capacity_mu=10**18)
        s2.compute_seal()

        assert s1.seal == s2.seal
        assert s1.used_capacity_mu == 0


# ===================================================================
# Part 6: Golden vectors (regression anchors)
# ===================================================================

class TestGoldenVectors:
    """
    Pinned outputs for known inputs. If any of these change,
    the kernel transition law or serialization has changed.
    """

    def test_identity_anchor_golden_vector(self):
        """Known identity -> known anchor (regression pins)."""
        GOLDEN_ALICE_ANCHOR = "aaa559"
        GOLDEN_BOB_ANCHOR = "6955a9"
        _, anchor_alice = identity_anchor("alice")
        _, anchor_bob = identity_anchor("bob")
        assert anchor_alice == GOLDEN_ALICE_ANCHOR
        assert anchor_bob == GOLDEN_BOB_ANCHOR
        assert anchor_alice != anchor_bob

    def test_shell_seal_golden_vector(self):
        """Known Shell -> known seal (regression pin)."""
        GOLDEN_SHELL_SEAL = "9966aa"
        shell = _make_shell(b"golden:2026", [
            ("alice", 87_600),
            ("bob", 175_200),
        ])
        assert shell.seal == GOLDEN_SHELL_SEAL
        assert len(shell.seal) == 6

    def test_meta_root_golden_vector(self):
        """Known bundle set -> known meta-root (regression pin)."""
        GOLDEN_META_ROOT = "555aa9"
        bundles = [b"program:Alpha", b"program:Beta", b"program:Gamma"]
        seals = []
        for p in bundles:
            r = Gyroscopic()
            sig = r.route_from_archetype(p)
            seals.append(bytes.fromhex(sig.state_hex))
        r = Gyroscopic()
        for s in seals:
            r.step_bytes(s)
        root = r.signature().state_hex
        assert root == GOLDEN_META_ROOT
        assert len(root) == 6

    def test_frame_record_golden_vector(self):
        """Known 4 bytes -> known frame record (regression pin)."""
        GOLDEN_MASK48 = 0x333F30000CCC
        GOLDEN_PHI_A = 0
        GOLDEN_PHI_B = 1
        fr = frame_record(0x00, 0x42, 0xAA, 0xFF)
        mask48, phi_a, phi_b = fr
        assert mask48 == GOLDEN_MASK48
        assert phi_a == GOLDEN_PHI_A
        assert phi_b == GOLDEN_PHI_B

    def test_parity_commitment_golden_vector(self):
        """Known payload -> known parity commitment (regression pin)."""
        GOLDEN_O = 0xC0C
        GOLDEN_E = 0xCC0
        GOLDEN_PARITY = 0
        payload = b"golden parity vector"
        o, e, p = trajectory_parity_commitment(payload)
        assert o == GOLDEN_O
        assert e == GOLDEN_E
        assert p == GOLDEN_PARITY
        assert p == len(payload) % 2

    def test_rest_state_golden_vector(self):
        """Rest state is the known archetype."""
        r = Gyroscopic()
        sig = r.signature()
        assert sig.state24 == 0xAAA555
        assert sig.state_hex == "aaa555"
        assert sig.a_hex == "aaa"
        assert sig.b_hex == "555"
        assert sig.step == 0


# ===================================================================
# Part 7: Genealogy-level integration
# ===================================================================

class TestGenealogyIntegration:
    """
    End-to-end genealogy tests combining byte log replay,
    frame commitments, and parity commitments.
    """

    def test_genealogy_replay_produces_identical_frames_and_state(self):
        """
        Two independent replays of the same byte log produce
        identical frame sequences and identical final states.
        """
        payload = b"genealogy replay consistency test payload!!"

        # Replay 1
        r1 = Gyroscopic()
        r1.step_bytes(payload)
        sig1 = r1.signature()
        frames1 = frame_sequence(payload)

        # Replay 2
        r2 = Gyroscopic()
        r2.step_bytes(payload)
        sig2 = r2.signature()
        frames2 = frame_sequence(payload)

        assert sig1.state24 == sig2.state24
        assert sig1.state_hex == sig2.state_hex
        assert frames1 == frames2

    def test_genealogy_has_three_certification_layers(self):
        """
        A genealogy provides three layers of certification:
        1. Final state (shared moment)
        2. Frame sequence (depth-4 certification)
        3. Parity commitment (algebraic integrity)

        All three are deterministic and independently computable.
        """
        payload = b"three layer certification"

        # Layer 1: state
        r = Gyroscopic()
        r.step_bytes(payload)
        state = r.signature().state_hex

        # Layer 2: frame sequence
        frames = frame_sequence(payload)

        # Layer 3: parity commitment
        o, e, p = trajectory_parity_commitment(payload)

        # All deterministic
        r2 = Gyroscopic()
        r2.step_bytes(payload)
        assert r2.signature().state_hex == state
        assert frame_sequence(payload) == frames
        assert trajectory_parity_commitment(payload) == (o, e, p)

        print(f"\n  Payload: {payload!r}")
        print(f"  State:   {state}")
        print(f"  Frames:  {len(frames)} depth-4 frames")
        print(f"  Parity:  O={o:#05x} E={e:#05x} p={p}")

    def test_forked_genealogies_detected_at_all_layers(self):
        """
        Two genealogies that share a prefix but diverge are
        detectable at state, frame, and parity layers.
        """
        prefix = b"shared prefix data"
        suffix_a = b"branch A continuation"
        suffix_b = b"branch B continuation"

        # Genealogy A
        ra = Gyroscopic()
        ra.step_bytes(prefix + suffix_a)
        state_a = ra.signature().state_hex
        frames_a = frame_sequence(prefix + suffix_a)
        parity_a = trajectory_parity_commitment(prefix + suffix_a)

        # Genealogy B
        rb = Gyroscopic()
        rb.step_bytes(prefix + suffix_b)
        state_b = rb.signature().state_hex
        frames_b = frame_sequence(prefix + suffix_b)
        parity_b = trajectory_parity_commitment(prefix + suffix_b)

        # Prefix frames should match
        prefix_frame_count = len(prefix) // 4
        assert frames_a[:prefix_frame_count] == frames_b[:prefix_frame_count]

        # At least one detection layer catches the fork
        detected = (
            state_a != state_b
            or frames_a != frames_b
            or parity_a != parity_b
        )
        assert detected, "Fork must be detectable at some layer"

        # Frame sequence always catches it (strongest guarantee)
        assert frames_a != frames_b, "Frame sequence must detect fork"

        print(f"\n  State differs:  {state_a != state_b}")
        print(f"  Frames differ:  {frames_a != frames_b}")
        print(f"  Parity differs: {parity_a != parity_b}")

    def test_genealogy_inverse_replay(self):
        """
        A genealogy can be replayed backwards using inverse stepping,
        recovering the rest state exactly.
        """
        payload = b"inverse replay genealogy test"

        r = Gyroscopic()
        r.step_bytes(payload)
        final_state = r.signature().state24
        assert final_state != GENE_MAC_REST

        r.step_bytes_inverse(payload)
        assert r.signature().state24 == GENE_MAC_REST

    def test_genealogy_continuation_preserves_history(self):
        """
        A genealogy can be continued from any point.
        The frame sequence of prefix + suffix equals
        frame_sequence(prefix) + frame_sequence(suffix)
        when both are frame-aligned.
        """
        part1 = bytes(range(0, 32))       # 8 frames
        part2 = bytes(range(32, 64))      # 8 frames
        combined = part1 + part2          # 16 frames

        frames_combined = frame_sequence(combined)
        frames_part1 = frame_sequence(part1)
        frames_part2 = frame_sequence(part2)

        assert len(frames_combined) == 16
        assert len(frames_part1) == 8
        assert len(frames_part2) == 8

        # First 8 frames of combined match part1's frames
        assert frames_combined[:8] == frames_part1

        # Frame records depend only on the 4 bytes in the frame,
        # not on prior state, so part2 frames match too
        assert frames_combined[8:] == frames_part2


if __name__ == "__main__":
    os.chdir(PROGRAM_ROOT)
    raise SystemExit(pytest.main(["-s", "-v", __file__]))