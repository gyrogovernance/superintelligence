"""
Holography diagnostics 1: shadow projection, byte fibers, Omega product, horizon.

Research-only tests for the current spinorial kernel.

- One-step shadow: 256 bytes -> 128 distinct next states; (state24, intron) injective.
- Byte-fiber pairing: T_b(s) = T_(b xor 0xFE)(s).
- Omega = U x V with |U|=|V|=64, U and V cosets of C64.
- Horizon = diagonal of Omega, boundary A-set = U.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.api import MASK12_BY_BYTE, so3_shadow_count
from src.constants import (
    GENE_MAC_A12,
    GENE_MAC_B12,
    GENE_MAC_REST,
    byte_to_intron,
    pack_state,
    step_state_by_byte,
    unpack_state,
)

pytestmark = pytest.mark.research

C64 = set(MASK12_BY_BYTE)


def _bfs_omega() -> tuple[set[int], set[int], set[int], set[int]]:
    """BFS from GENE_MAC_REST. Returns (omega, horizon_states, U_set, V_set)."""
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
    U_set = {(s >> 12) & 0xFFF for s in visited}
    V_set = {s & 0xFFF for s in visited}
    horizon_states = {
        s for s in visited
        if unpack_state(s)[0] == (unpack_state(s)[1] ^ 0xFFF)
    }
    return visited, horizon_states, U_set, V_set


def _shadow_fibers(state24: int) -> dict[int, list[int]]:
    fibers: dict[int, list[int]] = {}
    for b in range(256):
        s1 = step_state_by_byte(state24, b)
        fibers.setdefault(s1, []).append(b)
    return fibers


def test_shadow_projection_from_rest_has_128_images():
    assert so3_shadow_count(GENE_MAC_REST) == 128


def test_shadow_projection_is_128_on_sampled_states():
    rng = np.random.default_rng(20260301)
    for _ in range(64):
        state24 = int(rng.integers(0, 1 << 24))
        assert so3_shadow_count(state24) == 128


def test_shadow_fibers_from_rest_are_exact_byte_pairs():
    fibers = _shadow_fibers(GENE_MAC_REST)
    sizes = sorted(len(v) for v in fibers.values())

    print("\nShadow fibers from GENE_MAC_REST")
    print("--------------------------------")
    print(f"  distinct next states: {len(fibers)}")
    print(f"  fiber sizes: {sorted(set(sizes))}")

    assert len(fibers) == 128
    assert set(sizes) == {2}


def test_state_plus_intron_separates_shadow_collisions():
    seen: set[tuple[int, int]] = set()
    for b in range(256):
        pair = (step_state_by_byte(GENE_MAC_REST, b), byte_to_intron(b))
        seen.add(pair)
    assert len(seen) == 256


def test_byte_fiber_pairing_law():
    """
    Exact 2-to-1 collapse: T_b(s) = T_(b xor 0xFE)(s) for every state s and byte b.
    Pairs bytes b and b xor 0xFE into the same shadow fiber.
    """
    rng = np.random.default_rng(20260310)
    for _ in range(32):
        a = int(rng.integers(0, 4096))
        b = int(rng.integers(0, 4096))
        s = pack_state(a, b)
        for byte in range(256):
            next_same = step_state_by_byte(s, byte)
            next_pair = step_state_by_byte(s, byte ^ 0xFE)
            assert next_same == next_pair, (
                f"T_{byte}(s) != T_{byte ^ 0xFE}(s) at state {s}"
            )


class TestOmegaExactProductTheorem:
    """Omega = U x V with |U| = |V| = 64; U = GENE_MAC_A12 xor C64, V = GENE_MAC_B12 xor C64."""

    def test_omega_has_product_form_U_cross_V(self):
        omega, _, U_set, V_set = _bfs_omega()
        assert len(omega) == 4096
        assert len(U_set) == 64
        assert len(V_set) == 64
        for u in U_set:
            for v in V_set:
                assert pack_state(u, v) in omega

    def test_U_and_V_are_cosets_of_C64(self):
        _, _, U_set, V_set = _bfs_omega()
        U_expected = {GENE_MAC_A12 ^ c for c in C64}
        V_expected = {GENE_MAC_B12 ^ c for c in C64}
        assert U_set == U_expected
        assert V_set == V_expected


class TestHorizonExactDiagonalTheorem:
    """Horizon (S-sector) in Omega: A == B ^ 0xFFF; its A-set is U and B-set is V."""

    def test_horizon_is_diagonal_of_omega(self):
        omega, horizon_states, U_set, _ = _bfs_omega()
        assert len(horizon_states) == 64
        horizon_A = {unpack_state(s)[0] for s in horizon_states}
        assert horizon_A == U_set

    def test_every_u_in_U_gives_horizon_state_u_u_complement(self):
        omega, _, U_set, _ = _bfs_omega()
        for u in U_set:
            s = pack_state(u, u ^ 0xFFF)
            assert unpack_state(s)[0] == (unpack_state(s)[1] ^ 0xFFF)
            assert s in omega
