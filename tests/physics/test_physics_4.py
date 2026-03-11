"""
Physics tests 4: Research-only diagnostics.

No atlas, no hard CGM matching assertions.
Explores: new universe size (BFS), spinorial 4-cycle, commutator holonomy,
holographic ratio. Run with: pytest -m research tests/physics/test_physics_4.py -s
"""

from __future__ import annotations

import numpy as np
import pytest

from src.api import MASK12_BY_BYTE
from src.constants import (
    GENE_MAC_REST,
    GENE_MIC_S,
    inverse_step_by_byte,
    pack_state,
    step_state_by_byte,
    unpack_state,
)

pytestmark = pytest.mark.research


def _bfs_omega() -> tuple[set[int], list[int]]:
    """BFS from GENE_MAC_REST. Returns (omega_set, omega_list)."""
    visited = {GENE_MAC_REST}
    frontier = {GENE_MAC_REST}
    while frontier:
        next_frontier = set()
        for s in frontier:
            for b in range(256):
                s_next = step_state_by_byte(s, b)
                if s_next not in visited:
                    visited.add(s_next)
                    next_frontier.add(s_next)
        frontier = next_frontier
    return visited, sorted(visited)


def _cycle_lengths_of_permutation(perm: list[int]) -> list[int]:
    """Return list of cycle lengths for permutation perm (index -> image index)."""
    n = len(perm)
    seen = [False] * n
    lengths = []
    for start in range(n):
        if seen[start]:
            continue
        length = 0
        i = start
        while not seen[i]:
            seen[i] = True
            i = perm[i]
            length += 1
        lengths.append(length)
    return sorted(lengths, reverse=True)


def _pop12(x: int) -> int:
    return bin(int(x) & 0xFFF).count("1")


class TestMaskCodeDiagnostics:
    """Print-only diagnostics for the unique 64-mask code. No CGM asserts."""

    def test_mask_weight_distribution_diagnostic(self):
        """Weight enumerator of the unique mask set (64 codewords); print only."""
        unique_masks = set(MASK12_BY_BYTE)
        assert len(unique_masks) == 64
        weights = [_pop12(m) for m in unique_masks]
        unique_weights = set(weights)
        counts = [weights.count(w) for w in range(13)]
        print("\n  Unique mask set (64): weight distribution")
        for w in range(13):
            if counts[w]:
                print(f"    weight {w}: {counts[w]} masks")
        assert len(unique_weights) >= 1
        assert sum(counts) == 64


class TestTheNewUniverse:
    """Explore the size, radius, and boundaries of the new spinorial state space."""

    def test_bfs_discover_new_ontology_size(self):
        """
        Run a Breadth-First Search from GENE_MAC_REST.
        Finds exact size of the new reachable universe (Omega) and its radius.
        """
        visited = {GENE_MAC_REST}
        frontier = {GENE_MAC_REST}
        depth = 0

        print("\n\nMapping the New Spinorial Universe (BFS)")
        print("----------------------------------------")
        print(f"Depth {depth:2d}: {len(frontier):6d} states (Total: {len(visited):6d})")

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
            if frontier:
                print(f"Depth {depth:2d}: {len(frontier):6d} states (Total: {len(visited):6d})")

            if len(visited) > 100_000:
                print("Safety brake triggered: Universe > 100,000 states")
                break

        horizon_count = sum(
            1 for s in visited
            if unpack_state(s)[0] == (unpack_state(s)[1] ^ 0xFFF)
        )

        print("\nNew Universe Summary:")
        print(f"  Total Reachable States (Volume): {len(visited)}")
        print(f"  Maximum Depth (Radius):          {depth - 1}")
        print(f"  Horizon States in Omega (Area):  {horizon_count}")
        print(f"  Holographic Ratio (Area^2 / Vol): {(horizon_count**2) / len(visited):.4f}")


class TestUniversalSpinorialClosure:
    """Certify the 4-phase spinorial cycle (720 degrees)."""

    def test_every_byte_is_a_4_cycle(self):
        """
        Applying ANY byte 4 times returns exactly to the start state.
        Applying it 2 times results in a pure symmetric translation.
        """
        np.random.seed(42)
        s0 = pack_state(
            int(np.random.randint(0, 4096)), int(np.random.randint(0, 4096))
        )

        for b in range(256):
            s1 = step_state_by_byte(s0, b)
            s2 = step_state_by_byte(s1, b)
            s3 = step_state_by_byte(s2, b)
            s4 = step_state_by_byte(s3, b)

            assert s4 == s0, f"Byte {b} failed T^4 = id"

            a0, b0 = unpack_state(s0)
            a2, b2 = unpack_state(s2)
            delta_a = a0 ^ a2
            delta_b = b0 ^ b2
            assert delta_a == delta_b, "T^2 must be a symmetric translation"


class TestExactCommutatorHolonomy:
    """Analyze the exact defect generated by the physical commutator."""

    def test_commutator_defect_is_pure_translation(self):
        """
        K(x,y) = T_x o T_y o T_x^-1 o T_y^-1
        Verify this always results in a pure translation (A + d, B + d)
        and find what space 'd' lives in.
        """
        s0 = GENE_MAC_REST
        a0, b0 = unpack_state(s0)

        def apply_commutator(s: int, x: int, y: int) -> int:
            s = inverse_step_by_byte(s, y)
            s = inverse_step_by_byte(s, x)
            s = step_state_by_byte(s, y)
            s = step_state_by_byte(s, x)
            return s

        defects: set[int] = set()
        for x in [0x00, 0x42, 0xAA, 0xFF]:
            for y in range(256):
                s_out = apply_commutator(s0, x, y)
                a_out, b_out = unpack_state(s_out)

                delta_a = a0 ^ a_out
                delta_b = b0 ^ b_out

                assert delta_a == delta_b, "Commutator defect is not symmetric"
                defects.add(delta_a)

        print("\n\nCommutator Holonomy Defect")
        print("--------------------------")
        print(f"  Number of unique translation defects observed: {len(defects)}")

        mask_code = set(MASK12_BY_BYTE)
        all_in_code = all(d in mask_code for d in defects)
        print(f"  Do all defects live purely in the Mask Code C_64? {all_in_code}")


class TestCycleCensusOnOmega:
    """Exact cycle and spectral census on the 4096-state Omega."""

    def test_reference_byte_cycle_structure_on_omega(self):
        """
        On Omega, byte 0xAA (reference) is involution: 64 fixed points, 2016 2-cycles.
        """
        omega_set, omega_list = _bfs_omega()
        state_to_idx = {s: i for i, s in enumerate(omega_list)}
        n = len(omega_list)

        perm = [0] * n
        for i in range(n):
            s_next = step_state_by_byte(omega_list[i], GENE_MIC_S)
            perm[i] = state_to_idx[s_next]

        lengths = _cycle_lengths_of_permutation(perm)
        count_1 = lengths.count(1)
        count_2 = lengths.count(2)

        print("\n\nCycle census on Omega: reference byte 0xAA")
        print("------------------------------------------------")
        print(f"  Fixed points (cycle length 1): {count_1}")
        print(f"  2-cycles (cycle length 2):     {count_2}")
        print(f"  Total cycles:                  {len(lengths)}")

        assert count_1 == 64, f"Reference byte must have 64 fixed points on Omega, got {count_1}"
        assert count_2 == 2016, f"Reference byte must have 2016 2-cycles on Omega, got {count_2}"
        assert count_1 + 2 * count_2 == n, "64 + 2*2016 = 4096"

    def test_cycle_census_sample_bytes_on_omega(self):
        """Print cycle length multiset for a few non-reference bytes on Omega."""
        _, omega_list = _bfs_omega()
        state_to_idx = {s: i for i, s in enumerate(omega_list)}
        n = len(omega_list)

        for byte in [0x00, 0x42, 0xFF]:
            perm = [0] * n
            for i in range(n):
                s_next = step_state_by_byte(omega_list[i], byte)
                perm[i] = state_to_idx[s_next]
            lengths = _cycle_lengths_of_permutation(perm)
            from collections import Counter
            multiset = Counter(lengths)
            print(f"\n  Byte 0x{byte:02X}: cycle lengths {dict(sorted(multiset.items()))}")
