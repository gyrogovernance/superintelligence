"""
aQPU tests: Algebraic Quantum Processing Unit structure.

Tests the intrinsic quantum-algebraic properties documented in Appendix G:
- Dual horizons (complement and equality) and their structural relationship
- Four intrinsic gates {id, S, C, F} forming K4 = (Z/2)^2
- Gate action on both horizons (stabilizers and permutations)
- Bloch sphere latitude distribution from 6 independent chirality qubits
- Complementarity invariant: horizon_distance + ab_distance = 12
- Non-cloning structural properties
- Commutativity rate 1/64 from 6 DoF
- Gate F and id not achievable at depth 1

Does NOT retest: mask code structure (test_physics_2), shadow projection
(test_physics_1, test_holography), commutator defects (test_physics_5/6),
K4 fiber bundle (test_physics_6), depth-4 alternation (test_physics_1/3),
Omega topology (test_physics_1/4), or horizon-preserving byte count
(test_physics_6).
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from src.api import (
    MASK12_BY_BYTE,
    chirality_word6,
    mask12_for_byte,
    q_word6,
)
from src.constants import (
    GATE_C_BYTES,
    GATE_S_BYTES,
    GENE_MAC_REST,
    HORIZON_GATE_BYTES,
    LAYER_MASK_12,
    ab_distance,
    apply_gate,
    byte_family,
    byte_to_intron,
    horizon_distance,
    is_on_equality_horizon,
    is_on_horizon,
    pack_state,
    step_state_by_byte,
    unpack_state,
)

C64 = set(int(m) & 0xFFF for m in MASK12_BY_BYTE)


def _bfs_omega() -> set[int]:
    """BFS from GENE_MAC_REST. Returns reachable set."""
    visited = {GENE_MAC_REST}
    frontier = {GENE_MAC_REST}
    while frontier:
        nxt = set()
        for s in frontier:
            for b in range(256):
                t = step_state_by_byte(s, b)
                if t not in visited:
                    visited.add(t)
                    nxt.add(t)
        frontier = nxt
    return visited


def _cycle_census(states: set[int], gate: str) -> tuple[int, int]:
    """Returns (fixed_points, two_cycles) for involutory gate action on a state set."""
    fixed = 0
    two_cycles = 0
    visited: set[int] = set()
    for s in states:
        if s in visited:
            continue
        t = apply_gate(s, gate)
        if t == s:
            fixed += 1
            visited.add(s)
        else:
            assert t in states, f"Gate {gate} maps state outside set"
            assert apply_gate(t, gate) == s, f"Gate {gate} not involutory"
            two_cycles += 1
            visited.add(s)
            visited.add(t)
    return fixed, two_cycles


# ================================================================
# PART 1: DUAL HORIZONS
# ================================================================


class TestDualHorizons:
    """
    Two structurally necessary boundary sets in Omega:
    complement horizon (S-sector, A == B ^ 0xFFF, maximal chirality)
    and equality horizon (UNA degeneracy, A == B, zero chirality).
    """

    def test_rest_state_on_complement_not_equality(self):
        assert is_on_horizon(GENE_MAC_REST)
        assert not is_on_equality_horizon(GENE_MAC_REST)

    def test_both_horizons_have_64_states(self):
        omega = _bfs_omega()
        comp_h = {s for s in omega if is_on_horizon(s)}
        eq_h = {s for s in omega if is_on_equality_horizon(s)}
        assert len(comp_h) == 64
        assert len(eq_h) == 64

    def test_horizons_are_disjoint(self):
        omega = _bfs_omega()
        comp_h = {s for s in omega if is_on_horizon(s)}
        eq_h = {s for s in omega if is_on_equality_horizon(s)}
        assert comp_h.isdisjoint(eq_h)

    def test_boundary_128_bulk_3968(self):
        omega = _bfs_omega()
        boundary = {
            s for s in omega
            if is_on_horizon(s) or is_on_equality_horizon(s)
        }
        assert len(boundary) == 128
        assert len(omega - boundary) == 3968

    def test_complement_horizon_at_max_chirality(self):
        """All complement horizon states have ab_distance = 12."""
        omega = _bfs_omega()
        for s in omega:
            if is_on_horizon(s):
                a, b = unpack_state(s)
                assert ab_distance(a, b) == 12

    def test_equality_horizon_at_zero_chirality(self):
        """All equality horizon states have ab_distance = 0."""
        omega = _bfs_omega()
        for s in omega:
            if is_on_equality_horizon(s):
                a, b = unpack_state(s)
                assert ab_distance(a, b) == 0


# ================================================================
# PART 2: COMPLEMENTARITY INVARIANT
# ================================================================


class TestComplementarityInvariant:
    """
    horizon_distance + ab_distance = 12 for ALL states (not just Omega).
    This is the Bloch sphere pole-to-pole distance conservation.
    """

    def test_complementarity_exhaustive_in_omega(self):
        omega = _bfs_omega()
        for s in omega:
            a, b = unpack_state(s)
            assert horizon_distance(a, b) + ab_distance(a, b) == 12

    def test_complementarity_universal(self):
        """Holds for arbitrary 24-bit states, not just Omega."""
        rng = np.random.default_rng(42)
        for _ in range(50000):
            a = int(rng.integers(0, 4096))
            b = int(rng.integers(0, 4096))
            assert horizon_distance(a, b) + ab_distance(a, b) == 12


# ================================================================
# PART 3: GATE IDENTIFICATION AND K4 GROUP
# ================================================================


class TestGateDefinitions:
    """
    The 4 horizon-preserving bytes realize 2 distinct 24-bit operations:
    S (swap) and C (complement-swap). Together with id and F=S*C they
    form K4 = (Z/2)^2.
    """

    def test_gate_constants_match_horizon_condition(self):
        """The declared gate bytes are exactly the horizon-preserving set."""
        preserving = []
        for b in range(256):
            intron = byte_to_intron(b)
            inv_a = LAYER_MASK_12 if (intron & 0x01) else 0
            inv_b = LAYER_MASK_12 if (intron & 0x80) else 0
            m = mask12_for_byte(b)
            if m == (inv_a ^ inv_b):
                preserving.append(b)
        assert sorted(preserving) == sorted(HORIZON_GATE_BYTES)

    def test_gate_s_is_swap(self):
        rng = np.random.default_rng(100)
        for _ in range(2000):
            a = int(rng.integers(0, 4096))
            b = int(rng.integers(0, 4096))
            s = pack_state(a, b)
            for gb in GATE_S_BYTES:
                t = step_state_by_byte(s, gb)
                ta, tb = unpack_state(t)
                assert ta == b and tb == a

    def test_gate_c_is_complement_swap(self):
        F = LAYER_MASK_12
        rng = np.random.default_rng(101)
        for _ in range(2000):
            a = int(rng.integers(0, 4096))
            b = int(rng.integers(0, 4096))
            s = pack_state(a, b)
            for gb in GATE_C_BYTES:
                t = step_state_by_byte(s, gb)
                ta, tb = unpack_state(t)
                assert ta == (b ^ F) and tb == (a ^ F)

    def test_shadow_pairing(self):
        """Each gate pair differs by 0xFE (shadow fiber law)."""
        assert GATE_S_BYTES[0] ^ GATE_S_BYTES[1] == 0xFE
        assert GATE_C_BYTES[0] ^ GATE_C_BYTES[1] == 0xFE


class TestK4GateGroup:
    """Full Klein four-group verification."""

    def test_all_three_nontrivial_gates_are_involutions(self):
        rng = np.random.default_rng(200)
        for _ in range(1000):
            s = pack_state(int(rng.integers(0, 4096)), int(rng.integers(0, 4096)))
            for g in ("S", "C", "F"):
                assert apply_gate(apply_gate(s, g), g) == s, f"{g}^2 != id"

    def test_s_compose_c_equals_f(self):
        rng = np.random.default_rng(201)
        for _ in range(1000):
            s = pack_state(int(rng.integers(0, 4096)), int(rng.integers(0, 4096)))
            sc = apply_gate(apply_gate(s, "C"), "S")
            assert sc == apply_gate(s, "F")

    def test_gates_commute(self):
        """S*C = C*S = F (abelian group)."""
        rng = np.random.default_rng(202)
        for _ in range(1000):
            s = pack_state(int(rng.integers(0, 4096)), int(rng.integers(0, 4096)))
            sc = apply_gate(apply_gate(s, "C"), "S")
            cs = apply_gate(apply_gate(s, "S"), "C")
            assert sc == cs

    def test_full_cayley_table(self):
        """Verify complete K4 multiplication table on a fixed state."""
        s = pack_state(0x123, 0x456)
        table = {
            ("id", "id"): "id", ("id", "S"): "S", ("id", "C"): "C", ("id", "F"): "F",
            ("S", "id"): "S",   ("S", "S"): "id", ("S", "C"): "F", ("S", "F"): "C",
            ("C", "id"): "C",   ("C", "S"): "F",  ("C", "C"): "id", ("C", "F"): "S",
            ("F", "id"): "F",   ("F", "S"): "C",  ("F", "C"): "S", ("F", "F"): "id",
        }
        for (g1, g2), expected in table.items():
            result = apply_gate(apply_gate(s, g2), g1)
            assert result == apply_gate(s, expected), (
                f"{g1}*{g2} should be {expected}"
            )

    def test_gate_f_via_kernel_depth_2(self):
        """F = S*C requires two kernel steps (one S-byte, one C-byte)."""
        rng = np.random.default_rng(203)
        for _ in range(1000):
            s = pack_state(int(rng.integers(0, 4096)), int(rng.integers(0, 4096)))
            expected = apply_gate(s, "F")
            t1 = step_state_by_byte(step_state_by_byte(s, GATE_S_BYTES[0]), GATE_C_BYTES[0])
            t2 = step_state_by_byte(step_state_by_byte(s, GATE_C_BYTES[0]), GATE_S_BYTES[0])
            assert t1 == expected
            assert t2 == expected

    def test_no_single_byte_produces_f(self):
        """No single byte implements (A,B) → (A⊕F, B⊕F) for all states."""
        F = LAYER_MASK_12
        probes = [
            pack_state(0x000, 0x000),
            pack_state(0xAAA, 0x555),
            pack_state(0x123, 0x456),
            pack_state(0xFFF, 0x000),
        ]
        for b in range(256):
            all_match = True
            for s in probes:
                a, bcomp = unpack_state(s)
                t = step_state_by_byte(s, b)
                ta, tb = unpack_state(t)
                if ta != (a ^ F) or tb != (bcomp ^ F):
                    all_match = False
                    break
            assert not all_match, f"Byte {b:#x} implements F"

    def test_no_single_byte_is_identity(self):
        """No single byte implements (A,B) → (A,B) for all states."""
        probes = [
            pack_state(0x000, 0x000),
            pack_state(0xAAA, 0x555),
            pack_state(0x123, 0x456),
            pack_state(0xFFF, 0x000),
        ]
        for b in range(256):
            all_match = True
            for s in probes:
                if step_state_by_byte(s, b) != s:
                    all_match = False
                    break
            assert not all_match, f"Byte {b:#x} is identity"


# ================================================================
# PART 4: GATE ACTION ON HORIZONS
# ================================================================


class TestGateActionOnHorizons:
    """
    | Gate | Complement horizon     | Equality horizon       |
    |------|------------------------|------------------------|
    | id   | fixes all 64           | fixes all 64           |
    | S    | 32 two-cycles, 0 fixed | fixes all 64           |
    | C    | fixes all 64           | 32 two-cycles, 0 fixed |
    | F    | 32 two-cycles, 0 fixed | 32 two-cycles, 0 fixed |
    """

    def test_complement_horizon_census(self):
        omega = _bfs_omega()
        comp_h = {s for s in omega if is_on_horizon(s)}

        fixed, cycles = _cycle_census(comp_h, "id")
        assert (fixed, cycles) == (64, 0), "id must fix all"

        fixed, cycles = _cycle_census(comp_h, "S")
        assert (fixed, cycles) == (0, 32), "S must permute as 32 two-cycles"

        fixed, cycles = _cycle_census(comp_h, "C")
        assert (fixed, cycles) == (64, 0), "C must fix all"

        fixed, cycles = _cycle_census(comp_h, "F")
        assert (fixed, cycles) == (0, 32), "F must permute as 32 two-cycles"

    def test_equality_horizon_census(self):
        omega = _bfs_omega()
        eq_h = {s for s in omega if is_on_equality_horizon(s)}

        fixed, cycles = _cycle_census(eq_h, "id")
        assert (fixed, cycles) == (64, 0), "id must fix all"

        fixed, cycles = _cycle_census(eq_h, "S")
        assert (fixed, cycles) == (64, 0), "S must fix all"

        fixed, cycles = _cycle_census(eq_h, "C")
        assert (fixed, cycles) == (0, 32), "C must permute as 32 two-cycles"

        fixed, cycles = _cycle_census(eq_h, "F")
        assert (fixed, cycles) == (0, 32), "F must permute as 32 two-cycles"

    def test_gates_preserve_horizons_as_sets(self):
        """All four gates map each horizon to itself (as a set)."""
        omega = _bfs_omega()
        comp_h = {s for s in omega if is_on_horizon(s)}
        eq_h = {s for s in omega if is_on_equality_horizon(s)}
        for gate in ("id", "S", "C", "F"):
            for s in comp_h:
                assert apply_gate(s, gate) in comp_h, (
                    f"{gate} maps complement horizon state outside"
                )
            for s in eq_h:
                assert apply_gate(s, gate) in eq_h, (
                    f"{gate} maps equality horizon state outside"
                )


# ================================================================
# K4 orbit stratification on all of Omega
# ================================================================


class TestK4OrbitStratification:
    """
    Full K4 orbit structure on Omega:
      - 32 size-2 orbits on complement horizon
      - 32 size-2 orbits on equality horizon
      - 992 size-4 bulk orbits
    """

    def test_k4_orbit_stratification_of_omega(self):
        omega = _bfs_omega()
        seen: set[int] = set()
        orbit_sizes = Counter()
        comp_orbits = 0
        eq_orbits = 0
        bulk_orbits = 0

        for s in omega:
            if s in seen:
                continue
            orbit = {
                apply_gate(s, "id"),
                apply_gate(s, "S"),
                apply_gate(s, "C"),
                apply_gate(s, "F"),
            }
            for t in orbit:
                seen.add(t)

            orbit_sizes[len(orbit)] += 1

            on_comp = any(is_on_horizon(t) for t in orbit)
            on_eq = any(is_on_equality_horizon(t) for t in orbit)

            if on_comp:
                assert len(orbit) == 2
                assert all(is_on_horizon(t) for t in orbit)
                comp_orbits += 1
            elif on_eq:
                assert len(orbit) == 2
                assert all(is_on_equality_horizon(t) for t in orbit)
                eq_orbits += 1
            else:
                assert len(orbit) == 4
                assert all(
                    (not is_on_horizon(t)) and (not is_on_equality_horizon(t))
                    for t in orbit
                )
                bulk_orbits += 1

        assert seen == omega
        assert orbit_sizes == Counter({2: 64, 4: 992})
        assert comp_orbits == 32
        assert eq_orbits == 32
        assert bulk_orbits == 992


# ================================================================
# Pointwise stabilizer bytes for each horizon
# ================================================================


class TestHorizonPointwiseStabilizers:
    """
    Pointwise depth-1 stabilizers:
      - complement horizon fixed pointwise exactly by C-bytes
      - equality horizon fixed pointwise exactly by S-bytes
    """

    def test_complement_horizon_pointwise_stabilizer_bytes(self):
        omega = _bfs_omega()
        comp_h = [s for s in omega if is_on_horizon(s)]

        pointwise = []
        for b in range(256):
            if all(step_state_by_byte(s, b) == s for s in comp_h):
                pointwise.append(b)

        assert sorted(pointwise) == sorted(GATE_C_BYTES)

    def test_equality_horizon_pointwise_stabilizer_bytes(self):
        omega = _bfs_omega()
        eq_h = [s for s in omega if is_on_equality_horizon(s)]

        pointwise = []
        for b in range(256):
            if all(step_state_by_byte(s, b) == s for s in eq_h):
                pointwise.append(b)

        assert sorted(pointwise) == sorted(GATE_S_BYTES)


# ================================================================
# 6-bit chirality register transport law
# ================================================================


class TestChiralityRegisterTransport:
    """
    Exact 6-bit transport law on Omega:
      chi(T_b(s)) = chi(s) ^ q6(b)
    """

    def test_chirality_word_is_6bit_on_omega(self):
        omega = _bfs_omega()
        words = {chirality_word6(s) for s in omega}
        assert len(words) == 64
        assert words == set(range(64))

    def test_byte_transport_on_chirality_word(self):
        omega = _bfs_omega()
        for s in omega:
            chi = chirality_word6(s)
            for b in range(256):
                t = step_state_by_byte(s, b)
                chi_next = chirality_word6(t)
                assert chi_next == (chi ^ q_word6(b)), (
                    f"Transport law failed at state={s:#08x}, byte={b:#04x}"
                )


# ================================================================
# Same gate action, different spin phase (byte pairs)
# ================================================================


class TestGatePhaseSeparation:
    """
    Gate-byte pairs collapse to the same 24-bit operation
    but retain different intron/family phase.
    """

    def test_s_gate_bytes_same_action_different_phase(self):
        b0, b1 = GATE_S_BYTES
        assert byte_to_intron(b0) != byte_to_intron(b1)
        assert byte_family(b0) != byte_family(b1)

        rng = np.random.default_rng(400)
        for _ in range(1000):
            s = pack_state(
                int(rng.integers(0, 4096)),
                int(rng.integers(0, 4096)),
            )
            assert step_state_by_byte(s, b0) == step_state_by_byte(s, b1)

    def test_c_gate_bytes_same_action_different_phase(self):
        b0, b1 = GATE_C_BYTES
        assert byte_to_intron(b0) != byte_to_intron(b1)
        assert byte_family(b0) != byte_family(b1)

        rng = np.random.default_rng(401)
        for _ in range(1000):
            s = pack_state(
                int(rng.integers(0, 4096)),
                int(rng.integers(0, 4096)),
            )
            assert step_state_by_byte(s, b0) == step_state_by_byte(s, b1)


# ================================================================
# Bulk has trivial K4 stabilizer
# ================================================================


class TestBulkTrivialK4Stabilizer:
    """
    Bulk states are not fixed by any nontrivial K4 gate.
    """

    def test_bulk_has_no_nontrivial_k4_fixed_points(self):
        omega = _bfs_omega()
        bulk = {
            s for s in omega
            if (not is_on_horizon(s)) and (not is_on_equality_horizon(s))
        }

        for s in bulk:
            assert apply_gate(s, "S") != s
            assert apply_gate(s, "C") != s
            assert apply_gate(s, "F") != s


# ================================================================
# PART 5: BLOCH SPHERE LATITUDE STRUCTURE
# ================================================================


class TestBlochSphereLatitude:
    """
    Chirality spectrum in Omega: ab_distance takes values {0,2,4,6,8,10,12}.
    Count at each latitude follows C(6, (12-d)/2) * 64, giving a
    binomial distribution from 6 independent chirality qubits.
    Poles are the two horizons. Equator (d=6) has maximum population.
    """

    def test_chirality_spectrum_is_binomial(self):
        omega = _bfs_omega()
        counts: dict[int, int] = Counter()
        for s in omega:
            a, b = unpack_state(s)
            counts[ab_distance(a, b)] += 1

        for d in counts:
            assert d % 2 == 0, f"Odd ab_distance {d} in Omega"

        expected = {}
        for d in range(0, 13, 2):
            j = (12 - d) // 2
            expected[d] = math.comb(6, j) * 64

        assert counts == expected, (
            f"Spectrum mismatch:\n  got:      {dict(sorted(counts.items()))}"
            f"\n  expected: {dict(sorted(expected.items()))}"
        )

    def test_poles_are_horizons(self):
        omega = _bfs_omega()
        for s in omega:
            a, b = unpack_state(s)
            d = ab_distance(a, b)
            if d == 0:
                assert is_on_equality_horizon(s)
            if d == 12:
                assert is_on_horizon(s)

    def test_equator_is_maximum(self):
        omega = _bfs_omega()
        equator = [s for s in omega if ab_distance(*unpack_state(s)) == 6]
        assert len(equator) == math.comb(6, 3) * 64  # 1280

    def test_spectrum_is_symmetric(self):
        omega = _bfs_omega()
        counts: dict[int, int] = Counter()
        for s in omega:
            counts[ab_distance(*unpack_state(s))] += 1
        for k in range(7):
            assert counts[2 * k] == counts[12 - 2 * k]

    def test_pair_alignment_in_omega(self):
        """In Omega, A XOR B is always pair-diagonal: each pair is 00 or 11."""
        omega = _bfs_omega()
        for s in omega:
            a, b = unpack_state(s)
            diff = a ^ b
            for i in range(6):
                pair = (diff >> (2 * i)) & 0x3
                assert pair in (0x0, 0x3), (
                    f"Non-diagonal pair {i}: {pair:#x} at {s:#08x}"
                )

    def test_ab_distance_equals_twice_anti_aligned_pairs(self):
        """ab_distance = 2 * (number of anti-aligned pairs) in Omega."""
        omega = _bfs_omega()
        for s in omega:
            a, b = unpack_state(s)
            diff = a ^ b
            anti = sum(1 for i in range(6) if ((diff >> (2 * i)) & 0x3) == 0x3)
            assert ab_distance(a, b) == 2 * anti

    def test_all_gates_preserve_chirality(self):
        """ab_distance is invariant under all K4 gates."""
        rng = np.random.default_rng(300)
        for _ in range(2000):
            a = int(rng.integers(0, 4096))
            b = int(rng.integers(0, 4096))
            s = pack_state(a, b)
            d = ab_distance(a, b)
            for gate in ("S", "C", "F"):
                t = apply_gate(s, gate)
                assert ab_distance(*unpack_state(t)) == d, (
                    f"Gate {gate} changed chirality from {d}"
                )


# ================================================================
# PART 6: NON-CLONING PROPERTIES
# ================================================================


class TestNonCloning:
    """
    The archetype GENE_MIC_S = 0xAA is the unique common source.
    Non-cloning: no operation within the system can duplicate
    the reference frame.
    """

    def test_transcription_has_no_fixed_points(self):
        """No byte is its own intron: byte XOR 0xAA != byte for any byte."""
        for b in range(256):
            assert byte_to_intron(b) != b

    def test_archetype_is_unique_zero_intron_source(self):
        """Exactly one byte produces intron 0x00: the archetype 0xAA itself."""
        zero_sources = [b for b in range(256) if byte_to_intron(b) == 0x00]
        assert zero_sources == [0xAA]

    def test_equality_horizon_carries_redundant_information(self):
        """
        On the equality horizon (A==B), both components are identical.
        Information content = one C64 coset coordinate (6 bits).
        The 'clone' adds zero additional information.
        """
        omega = _bfs_omega()
        eq_h = {s for s in omega if is_on_equality_horizon(s)}
        a_values = {unpack_state(s)[0] for s in eq_h}
        b_values = {unpack_state(s)[1] for s in eq_h}
        assert a_values == b_values
        assert len(a_values) == 64

    def test_complement_horizon_carries_relational_information(self):
        """
        On the complement horizon (A==B^F), knowing A determines B uniquely.
        The information is in the RELATIONSHIP (chirality), not duplication.
        """
        omega = _bfs_omega()
        comp_h = {s for s in omega if is_on_horizon(s)}
        a_to_b: dict[int, set[int]] = {}
        for s in comp_h:
            a, b = unpack_state(s)
            a_to_b.setdefault(a, set()).add(b)
        for a, bs in a_to_b.items():
            assert len(bs) == 1, f"A={a:#x} maps to multiple B values"
            assert next(iter(bs)) == a ^ LAYER_MASK_12

    def test_horizons_stable_under_gates(self):
        """
        Gates permute within horizons, never between them.
        You cannot 'clone' (reach equality horizon from complement
        horizon) by applying gate operations alone.
        """
        omega = _bfs_omega()
        comp_h = {s for s in omega if is_on_horizon(s)}
        eq_h = {s for s in omega if is_on_equality_horizon(s)}
        for gate in ("S", "C", "F"):
            for s in comp_h:
                assert apply_gate(s, gate) in comp_h
            for s in eq_h:
                assert apply_gate(s, gate) in eq_h


# ================================================================
# PART 7: COMMUTATIVITY RATE
# ================================================================


class TestCommutativityRate:
    """
    Exactly 1/64 = 2^(-6) of byte pairs commute.
    The exponent 6 is the number of independent DoF.
    Each DoF contributes independently to the commutation class.
    """

    def test_commutativity_fraction_is_1_over_64(self):
        """1024 commuting pairs out of 65536 total = 1/64."""
        s = GENE_MAC_REST
        commuting = 0
        for x in range(256):
            sx = step_state_by_byte(s, x)
            for y in range(256):
                lhs = step_state_by_byte(sx, y)
                rhs = step_state_by_byte(step_state_by_byte(s, y), x)
                if lhs == rhs:
                    commuting += 1
        assert commuting == 1024
        assert commuting * 64 == 256 * 256

    def test_each_byte_commutes_with_exactly_4(self):
        """Uniform multiplicity: every byte has exactly 4 commuting partners."""
        s = GENE_MAC_REST
        for x in range(256):
            sx = step_state_by_byte(s, x)
            count = sum(
                1 for y in range(256)
                if step_state_by_byte(sx, y)
                == step_state_by_byte(step_state_by_byte(s, y), x)
            )
            assert count == 4, f"Byte {x:#x}: {count} partners, expected 4"