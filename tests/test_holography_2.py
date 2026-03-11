"""
Holography diagnostics 2: pair-diagonal mask code, frame factorization, horizon Walsh, Hilbert-lift.

Research-only tests for the current 64-mask code.

- Unique mask code: 6-pair diagonal {00,11}^6, weight enumerator (1+z^2)^6.
- Horizon Walsh: support on C64, magnitude 64, phase from GENE_MAC_A12.
- Hilbert-lift on C64: separable (one state) -> reduced entropy 0; graph subset -> entropy 6.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pytest

from src.api import MASK12_BY_BYTE
from src.constants import (
    GENE_MAC_A12,
    GENE_MAC_REST,
    dot12,
    step_state_by_byte,
    unpack_state,
)

pytestmark = pytest.mark.research

C64 = set(MASK12_BY_BYTE)


def _pop12(x: int) -> int:
    return bin(int(x) & 0xFFF).count("1")


def _unique_masks() -> list[int]:
    return sorted(set(int(m) for m in MASK12_BY_BYTE))


def test_unique_mask_code_is_pair_diagonal():
    masks = _unique_masks()
    assert len(masks) == 64

    for m in masks:
        for i in range(6):
            b0 = (m >> (2 * i)) & 1
            b1 = (m >> (2 * i + 1)) & 1
            assert b0 == b1, f"pair {i} is not diagonal in mask 0x{m:03x}"


def test_frame_factorization_is_8_by_8():
    masks = _unique_masks()

    frame0 = {(m & 0x3F) for m in masks}
    frame1 = {((m >> 6) & 0x3F) for m in masks}
    pairs = {((m & 0x3F), ((m >> 6) & 0x3F)) for m in masks}

    print("\nFrame factorization of unique masks")
    print("-----------------------------------")
    print(f"  |frame0 values| = {len(frame0)}")
    print(f"  |frame1 values| = {len(frame1)}")
    print(f"  |frame pairs|   = {len(pairs)}")

    assert len(frame0) == 8
    assert len(frame1) == 8
    assert len(pairs) == 64


def test_unique_mask_weight_enumerator_is_one_plus_z2_pow6():
    masks = _unique_masks()
    counts = Counter(_pop12(m) for m in masks)
    observed = [counts.get(w, 0) for w in range(13)]
    expected = [1, 0, 6, 0, 15, 0, 20, 0, 15, 0, 6, 0, 1]
    assert observed == expected


def test_byte_table_weight_enumerator_is_4_times_unique():
    counts = Counter(_pop12(int(m)) for m in MASK12_BY_BYTE)
    observed = [counts.get(w, 0) for w in range(13)]
    expected_unique = [1, 0, 6, 0, 15, 0, 20, 0, 15, 0, 6, 0, 1]
    expected_table = [4 * x for x in expected_unique]
    assert observed == expected_table


def _bfs_omega() -> set[int]:
    """BFS from GENE_MAC_REST; returns Omega."""
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
    return visited


class TestHorizonWalshTheorem:
    """
    Horizon A-set H = GENE_MAC_A12 xor C64 (coset of 64-mask code).
    Walsh W_H(s) = sum_{a in H} (-1)^dot(s,a): support exactly C64, magnitude 64, phase from rest A.
    """

    def test_walsh_support_is_exactly_C64(self):
        omega = _bfs_omega()
        H = {
            unpack_state(s)[0]
            for s in omega
            if unpack_state(s)[0] == unpack_state(s)[1]
        }
        assert len(H) == 64
        for s in range(1 << 12):
            w = sum(1 if dot12(s, a) == 0 else -1 for a in H)
            in_c64 = (s in C64)
            if in_c64:
                assert w != 0
            else:
                assert w == 0

    def test_walsh_magnitude_and_phase_on_C64(self):
        omega = _bfs_omega()
        H = {
            unpack_state(s)[0]
            for s in omega
            if unpack_state(s)[0] == unpack_state(s)[1]
        }
        for s in C64:
            w = sum(1 if dot12(s, a) == 0 else -1 for a in H)
            assert abs(w) == 64
            phase = 1 if dot12(s, GENE_MAC_A12) == 0 else -1
            assert w == 64 * phase


def _von_neumann_entropy_bits(rho: np.ndarray) -> float:
    """Von Neumann entropy -sum lambda log2(lambda) for eigenvalues lambda of rho."""
    eigs = np.linalg.eigvalsh(rho)
    eigs = eigs[eigs > 1e-12]
    return float(-np.sum(eigs * np.log2(eigs)))


def _reduced_density_A(c64_list: list[int], pairs: list[tuple[int, int]]) -> np.ndarray:
    """
    Pairs are (u, v) as 12-bit mask values in C64.
    |psi> = (1/sqrt(|S|)) sum_{(u,v) in S} |u>|v>. Compute rho_A = Tr_B(|psi><psi|).
    rho_A[i,j] = (1/|S|) sum_v 1 if (i,v) in S and (j,v) in S else 0.
    """
    n = len(c64_list)
    u_to_i = {u: i for i, u in enumerate(c64_list)}
    by_v: dict[int, list[int]] = defaultdict(list)
    for (u, v) in pairs:
        by_v[v].append(u_to_i[u])
    rho = np.zeros((n, n), dtype=float)
    s_size = len(pairs)
    for _v, u_indices in by_v.items():
        for i in u_indices:
            for j in u_indices:
                rho[i, j] += 1.0 / s_size
    return rho


class TestHilbertLiftOnC64:
    """
    Bipartite Hilbert space on C64 x C64. Separable (single state) -> reduced entropy 0;
    graph subset {(u, u xor t) : u in C64} -> reduced entropy log2(64) = 6.
    """

    def test_separable_single_state_has_reduced_entropy_zero(self):
        c64_list = sorted(C64)
        u0, v0 = c64_list[0], c64_list[1]
        pairs = [(u0, v0)]
        rho = _reduced_density_A(c64_list, pairs)
        ent = _von_neumann_entropy_bits(rho)
        assert ent == 0.0

    def test_graph_subset_has_reduced_entropy_six(self):
        c64_list = sorted(C64)
        t = c64_list[7]
        pairs = [(u, u ^ t) for u in C64]
        assert len(pairs) == 64
        rho = _reduced_density_A(c64_list, pairs)
        ent = _von_neumann_entropy_bits(rho)
        assert abs(ent - 6.0) < 1e-10
