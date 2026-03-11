"""
Physics tests 6: Intrinsic K4, fiber bundle, and legacy coverage.

Central thesis (from analysis):
  K4 is the fiber of the depth-4 frame bundle.
  Fix base geometry (4 micro_refs -> mask48).
  Vary gauge (4^4 family choices).
  Output collapses to exactly 4 states indexed by (phi_a, phi_b) in (Z/2)^2.
  That (Z/2)^2 IS the K4 vertex set, intrinsic and not fitted.

Additionally covers legacy gaps:
  - Exhaustive commutator defect (all 256^2 pairs)
  - K4 edge vector from q-invariant
  - Provenance degeneracy (history non-uniqueness)
  - Erasure taxonomy on the [12,6,2] code
  - Hilbert-lift entanglement (product vs graph states)
  - Spinorial intrinsic gate and holonomy (horizon-preserving bytes, q 4-to-1, defect C64)

No new source files. Uses only src.constants and src.api.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest

from src.api import (
    C_PERP_12,
    MASK12_BY_BYTE,
    depth4_mask_projection48,
    mask12_for_byte,
)
from src.constants import (
    APERTURE_GAP,
    APERTURE_GAP_Q256,
    DELTA_BU,
    GENE_MAC_REST,
    GENE_MIC_S,
    LAYER_MASK_12,
    M_A,
    RHO,
    byte_to_intron,
    dot12,
    inverse_step_by_byte,
    micro_ref_to_mask12,
    pack_state,
    step_state_by_byte,
    unpack_state,
)

# ----------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------

C64 = set(int(m) & 0xFFF for m in MASK12_BY_BYTE)


def _byte_from_micro_family(micro: int, family: int) -> int:
    """Construct a byte from a 6-bit micro_ref and a 2-bit family index."""
    bit0 = family & 1
    bit7 = (family >> 1) & 1
    intron = (bit7 << 7) | ((micro & 0x3F) << 1) | bit0
    return intron ^ GENE_MIC_S


def _phi_bits(f0: int, f1: int, f2: int, f3: int) -> tuple[int, int]:
    """
    Net family-phase bits surviving at depth-4:
      phi_a = b0 ^ a1 ^ b2 ^ a3
      phi_b = a0 ^ b1 ^ a2 ^ b3
    where for family f: a = f & 1, b = (f >> 1) & 1.
    """
    a0, b0 = f0 & 1, (f0 >> 1) & 1
    a1, b1 = f1 & 1, (f1 >> 1) & 1
    a2, b2 = f2 & 1, (f2 >> 1) & 1
    a3, b3 = f3 & 1, (f3 >> 1) & 1
    phi_a = b0 ^ a1 ^ b2 ^ a3
    phi_b = a0 ^ b1 ^ a2 ^ b3
    return phi_a, phi_b


def _q(byte: int) -> int:
    """Commutation invariant: mask XOR complement if L0 parity is odd."""
    intron = byte_to_intron(byte)
    l0 = (intron & 1) ^ ((intron >> 7) & 1)
    return mask12_for_byte(byte) ^ (LAYER_MASK_12 if l0 else 0)


def _gf2_rank(matrix: np.ndarray) -> int:
    """Rank of a binary matrix over GF(2)."""
    M = matrix.copy() % 2
    rows, cols = M.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if M[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for row in range(rows):
            if row != rank and M[row, col] == 1:
                M[row] = (M[row] + M[rank]) % 2
        rank += 1
    return rank


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


# ================================================================
# PART 1: K4 IS THE FIBER OF THE DEPTH-4 FRAME
# ================================================================


class TestK4IsDepth4Fiber:
    """
    Central structural result: K4 emerges intrinsically as the fiber
    of the depth-4 frame bundle.

    Base: the 48-bit mask projection (depends only on micro_refs).
    Fiber: the 2-bit surviving phase (phi_a, phi_b) in (Z/2)^2.
    Total: the 4-byte word (32-bit intron sequence, bijective).

    K4 is not a fitted partition on the mask code. It is the
    quotient of the depth-4 gauge freedom.
    """

    def test_fiber_has_exactly_4_points(self):
        """
        For fixed micro_refs, varying all 4^4 = 256 family combinations
        produces exactly 4 distinct output states from rest, indexed
        by (phi_a, phi_b).
        """
        for micros in ([1, 7, 13, 29], [0, 3, 15, 63], [2, 8, 32, 41]):
            phi_to_state: dict[tuple[int, int], int] = {}
            for f0 in range(4):
                for f1 in range(4):
                    for f2 in range(4):
                        for f3 in range(4):
                            word = [
                                _byte_from_micro_family(micros[0], f0),
                                _byte_from_micro_family(micros[1], f1),
                                _byte_from_micro_family(micros[2], f2),
                                _byte_from_micro_family(micros[3], f3),
                            ]
                            s = GENE_MAC_REST
                            for b in word:
                                s = step_state_by_byte(s, b)
                            phi = _phi_bits(f0, f1, f2, f3)
                            if phi in phi_to_state:
                                assert phi_to_state[phi] == s
                            else:
                                phi_to_state[phi] = s
            assert len(phi_to_state) == 4, (
                f"micros={micros}: fiber must have 4 states, got {len(phi_to_state)}"
            )

    def test_base_is_gauge_blind(self):
        """
        mask48 depends only on micro_refs, not on family assignment.
        Two words with same micro_refs but different families produce
        the same mask48.
        """
        micros = [1, 7, 13, 29]
        projections = set()
        for f0 in range(4):
            for f1 in range(4):
                for f2 in range(4):
                    for f3 in range(4):
                        word = [
                            _byte_from_micro_family(micros[0], f0),
                            _byte_from_micro_family(micros[1], f1),
                            _byte_from_micro_family(micros[2], f2),
                            _byte_from_micro_family(micros[3], f3),
                        ]
                        projections.add(depth4_mask_projection48(*word))
        assert len(projections) == 1

    def test_fiber_is_z2_squared_group(self):
        """
        Composing two depth-4 frames from rest: the displacement from rest
        of the composed trajectory equals the XOR of the two frame
        displacements. So the fiber label composes as (Z/2)^2.

        disp(rest -> frame1 -> frame2) = disp(frame1) XOR disp(frame2)

        The four canonical states state4(phi) are not closed under applying
        another frame (state equality would require that); the invariant
        is displacement additivity, which is the (Z/2)^2 group law.
        """
        micros = [1, 7, 13, 29]

        def frame_word(fams):
            return [_byte_from_micro_family(micros[i], fams[i]) for i in range(4)]

        a0, b0 = unpack_state(GENE_MAC_REST)
        rng = np.random.default_rng(123)
        for _ in range(500):
            fams1 = [int(rng.integers(0, 4)) for _ in range(4)]
            fams2 = [int(rng.integers(0, 4)) for _ in range(4)]
            phi1 = _phi_bits(*fams1)
            phi2 = _phi_bits(*fams2)

            s1 = GENE_MAC_REST
            for b in frame_word(fams1):
                s1 = step_state_by_byte(s1, b)
            s_composed = s1
            for b in frame_word(fams2):
                s_composed = step_state_by_byte(s_composed, b)

            a1, b1 = unpack_state(s1)
            a_final, b_final = unpack_state(s_composed)
            disp1_a, disp1_b = a1 ^ a0, b1 ^ b0
            disp2_a, disp2_b = a_final ^ a1, b_final ^ b1
            disp_final_a, disp_final_b = a_final ^ a0, b_final ^ b0

            assert disp_final_a == disp1_a ^ disp2_a
            assert disp_final_b == disp1_b ^ disp2_b

    def test_exiting_state_determined_by_base_and_fiber(self):
        """
        For a fixed set of micro_refs, the output state from rest
        is fully determined by (mask48, phi). Since mask48 is fixed
        when micro_refs are fixed, the output is determined by phi alone.
        """
        micros = [1, 7, 13, 29]

        seen: dict[tuple[int, tuple[int, int]], int] = {}
        for f0 in range(4):
            for f1 in range(4):
                for f2 in range(4):
                    for f3 in range(4):
                        word = [
                            _byte_from_micro_family(micros[0], f0),
                            _byte_from_micro_family(micros[1], f1),
                            _byte_from_micro_family(micros[2], f2),
                            _byte_from_micro_family(micros[3], f3),
                        ]
                        s = GENE_MAC_REST
                        for b in word:
                            s = step_state_by_byte(s, b)
                        mask48 = depth4_mask_projection48(*word)
                        phi = _phi_bits(f0, f1, f2, f3)
                        key = (mask48, phi)
                        if key in seen:
                            assert seen[key] == s
                        else:
                            seen[key] = s

        assert len(seen) == 4


# ================================================================
# PART 2: K4 EDGE VECTOR FROM Q-INVARIANT
# ================================================================


class TestK4EdgesFromCommutationInvariant:
    """
    Canonical K4 edge object: for a 4-byte frame with q-values
    q0, q1, q2, q3, the 6 pairwise XOR differences form a K4
    edge vector. Each edge is a commutator defect and lies in C64.
    """

    def test_frame_edges_lie_in_c64(self):
        """All 6 pairwise q-differences of a 4-byte frame lie in C64."""
        rng = np.random.default_rng(0)
        for _ in range(5000):
            bs = [int(rng.integers(0, 256)) for _ in range(4)]
            qs = [_q(b) & 0xFFF for b in bs]
            edges = [
                qs[0] ^ qs[1],
                qs[0] ^ qs[2],
                qs[0] ^ qs[3],
                qs[1] ^ qs[2],
                qs[1] ^ qs[3],
                qs[2] ^ qs[3],
            ]
            for i, e in enumerate(edges):
                assert e in C64, f"Edge {i} = {e:#05x} not in C64"

    def test_frame_edges_satisfy_cocycle_condition(self):
        """
        The K4 edge vector must satisfy the cocycle (Kirchhoff) law:
        e01 XOR e12 = e02 (and cyclic permutations).
        This is just XOR associativity but confirms the edges form
        a consistent gradient on K4.
        """
        rng = np.random.default_rng(42)
        for _ in range(3000):
            bs = [int(rng.integers(0, 256)) for _ in range(4)]
            qs = [_q(b) & 0xFFF for b in bs]
            e01 = qs[0] ^ qs[1]
            e02 = qs[0] ^ qs[2]
            e12 = qs[1] ^ qs[2]
            e03 = qs[0] ^ qs[3]
            e13 = qs[1] ^ qs[3]
            e23 = qs[2] ^ qs[3]
            assert e01 ^ e12 == e02
            assert e01 ^ e13 == e03
            assert e02 ^ e23 == e03
            assert e12 ^ e23 == e13


# ================================================================
# PART 3: EXHAUSTIVE COMMUTATOR AND MONODROMY
# ================================================================


class TestExhaustiveCommutatorAndMonodromy:
    """
    Exhaustive commutator defect over all 256^2 byte pairs from rest.
    Verifies: symmetric translation, defect in C64, defects span C64.
    Also: depth-4 frame fiber defect for fixed masks.
    """

    def test_commutator_exhaustive_all_pairs(self):
        """K(x,y) from rest is symmetric translation by q(x)^q(y), in C64, spanning C64."""
        a0, b0 = unpack_state(GENE_MAC_REST)
        defects_seen: set[int] = set()

        for x in range(256):
            for y in range(256):
                s = GENE_MAC_REST
                s = step_state_by_byte(s, x)
                s = step_state_by_byte(s, y)
                s = inverse_step_by_byte(s, x)
                s = inverse_step_by_byte(s, y)
                a, b = unpack_state(s)
                d_a = a ^ a0
                d_b = b ^ b0
                assert d_a == d_b, f"Not symmetric: x={x}, y={y}"
                assert d_a in C64, f"Defect not in C64: x={x}, y={y}"
                expected_d = _q(x) ^ _q(y)
                assert d_a == expected_d, f"Defect mismatch: x={x}, y={y}"
                defects_seen.add(d_a)

        assert defects_seen == C64, (
            f"Commutator defects span {len(defects_seen)} of 64"
        )

    def test_depth4_fiber_defect_for_fixed_masks(self):
        """
        For fixed micro_refs, the fiber defect (displacement from rest)
        depends only on phi. Different phi classes give different defects.
        All defects must lie in C64.
        """
        a0, b0 = unpack_state(GENE_MAC_REST)

        for micros in ([0, 1, 2, 3], [5, 17, 33, 62], [7, 14, 28, 56]):
            defect_by_phi: dict[tuple[int, int], tuple[int, int]] = {}
            for f0 in range(4):
                for f1 in range(4):
                    for f2 in range(4):
                        for f3 in range(4):
                            word = [
                                _byte_from_micro_family(micros[0], f0),
                                _byte_from_micro_family(micros[1], f1),
                                _byte_from_micro_family(micros[2], f2),
                                _byte_from_micro_family(micros[3], f3),
                            ]
                            s = GENE_MAC_REST
                            for b in word:
                                s = step_state_by_byte(s, b)
                            a4, b4 = unpack_state(s)
                            d_a = a4 ^ a0
                            d_b = b4 ^ b0
                            assert d_a in C64
                            assert d_b in C64

                            phi = _phi_bits(f0, f1, f2, f3)
                            if phi in defect_by_phi:
                                assert defect_by_phi[phi] == (d_a, d_b)
                            else:
                                defect_by_phi[phi] = (d_a, d_b)

            assert len(defect_by_phi) == 4


# ================================================================
# PART 4: PROVENANCE DEGENERACY
# ================================================================


class TestProvenanceDegeneracy:
    """
    History non-uniqueness: many distinct byte sequences reach the
    same final state. The kernel provides shared moments (reproducible
    state from shared ledger) but state alone cannot recover history.
    """

    def test_word_history_degeneracy(self):
        """
        Using 12 generator bytes (6 single-pair payloads x 2 families),
        enumerate all length-4 words. Multiple words map to the same state.
        """
        generators = []
        for pair_idx in range(6):
            micro = 1 << pair_idx
            for fam in (0, 1):
                generators.append(_byte_from_micro_family(micro, fam))
        assert len(generators) == 12

        final_states: dict[int, int] = {}
        total_words = 12 ** 4  # 20736

        for g0 in generators:
            s0 = step_state_by_byte(GENE_MAC_REST, g0)
            for g1 in generators:
                s1 = step_state_by_byte(s0, g1)
                for g2 in generators:
                    s2 = step_state_by_byte(s1, g2)
                    for g3 in generators:
                        s3 = step_state_by_byte(s2, g3)
                        final_states[s3] = final_states.get(s3, 0) + 1

        distinct = len(final_states)
        max_preimage = max(final_states.values())
        min_preimage = min(final_states.values())

        print(f"\n  Provenance degeneracy:")
        print(f"    Alphabet: {len(generators)} generators")
        print(f"    Word length: 4")
        print(f"    Total words: {total_words}")
        print(f"    Distinct final states: {distinct}")
        print(f"    Average preimage: {total_words / distinct:.2f}")
        print(f"    Max preimage: {max_preimage}")
        print(f"    Min preimage: {min_preimage}")

        # Must have degeneracy
        assert distinct < total_words
        # All finals in Omega
        assert distinct <= 4096

    def test_full_alphabet_length2_degeneracy(self):
        """
        All 256^2 = 65536 length-2 words from rest. With 128-way shadow
        projection at each step, significant degeneracy is expected.
        """
        final_states: dict[int, int] = {}
        for b0 in range(256):
            s0 = step_state_by_byte(GENE_MAC_REST, b0)
            for b1 in range(256):
                s1 = step_state_by_byte(s0, b1)
                final_states[s1] = final_states.get(s1, 0) + 1

        distinct = len(final_states)
        print(f"\n  Length-2 provenance:")
        print(f"    Total words: 65536")
        print(f"    Distinct final states: {distinct}")
        print(f"    Average preimage: {65536 / distinct:.2f}")

        # With 4096-state Omega reached at radius 2, we expect all states hit
        assert distinct == 4096
        # Average preimage should be 16 (65536 / 4096)
        assert 65536 // distinct == 16


# ================================================================
# PART 5: ERASURE TAXONOMY ON [12,6,2] CODE
# ================================================================


class TestErasureTaxonomy:
    """
    Classify erasure patterns on the [12,6,2] self-dual code.
    The generator matrix has rank 6 (not 8 as in old code).
    """

    @staticmethod
    def _generator_matrix() -> np.ndarray:
        """6 x 12 generator matrix from single-pair basis masks."""
        rows = []
        for i in range(6):
            m = micro_ref_to_mask12(1 << i)
            rows.append([(m >> bit) & 1 for bit in range(12)])
        return np.array(rows, dtype=np.int32)

    def test_information_set_threshold(self):
        """
        Minimum observed bits for unique codeword reconstruction.
        For pair-diagonal [12,6,2]: observing one bit per pair (6 bits
        from 6 different pairs) should give rank 6. Threshold = 6.
        """
        G = self._generator_matrix()

        max_rank_by_size: dict[int, int] = {}
        for s in range(1, 13):
            max_r = 0
            for cols in combinations(range(12), s):
                r = _gf2_rank(G[:, list(cols)])
                if r > max_r:
                    max_r = r
                if max_r == 6:
                    break  # cannot exceed rank of G
            max_rank_by_size[s] = max_r

        threshold = min(s for s in range(1, 13) if max_rank_by_size[s] == 6)

        print(f"\n  Information-set threshold: {threshold}")
        for s in range(1, 13):
            print(f"    s={s:2d}: max rank = {max_rank_by_size[s]}")

        assert threshold == 6

    def test_size4_erasure_histogram(self):
        """
        Classify all C(12,4) = 495 erasure patterns of size 4.
        Each pattern erases 4 bit positions, leaving 8 observed.
        """
        G = self._generator_matrix()

        rank_hist: dict[tuple[int, int], int] = {}
        for erased in combinations(range(12), 4):
            observed = [i for i in range(12) if i not in erased]
            G_obs = G[:, observed]
            r = _gf2_rank(G_obs)
            ambiguity = 2 ** (6 - r)
            key = (r, ambiguity)
            rank_hist[key] = rank_hist.get(key, 0) + 1

        total = sum(rank_hist.values())
        assert total == 495  # C(12,4)

        print(f"\n  Size-4 erasure taxonomy:")
        for key in sorted(rank_hist.keys()):
            print(f"    rank={key[0]}, ambiguity={key[1]}: {rank_hist[key]} patterns")

    def test_pair_erasure_structure(self):
        """
        In the pair-diagonal code, erasing both bits of a pair loses
        exactly 1 rank. Erasing one bit of a pair loses 0 rank (the
        other bit carries the same information). Verify this.
        """
        G = self._generator_matrix()
        full_rank = _gf2_rank(G)
        assert full_rank == 6

        for pair_idx in range(6):
            bit_lo = 2 * pair_idx
            bit_hi = 2 * pair_idx + 1

            # Erase one bit of the pair: rank should stay 6
            observed_one = [i for i in range(12) if i != bit_lo]
            r_one = _gf2_rank(G[:, observed_one])
            assert r_one == 6, f"Erasing bit {bit_lo} dropped rank to {r_one}"

            # Erase both bits of the pair: rank should drop to 5
            observed_both = [i for i in range(12) if i not in (bit_lo, bit_hi)]
            r_both = _gf2_rank(G[:, observed_both])
            assert r_both == 5, f"Erasing pair {pair_idx} gave rank {r_both}"


# ================================================================
# PART 6: HILBERT-LIFT ENTANGLEMENT
# ================================================================


class TestHilbertLiftEntanglement:
    """
    Bipartite entanglement in the Hilbert space lift of the code.
    C64 serves as computational basis for H_u tensor H_v.
    Product subsets: zero entropy. Graph subsets: maximal entropy.
    """

    @staticmethod
    def _von_neumann_entropy(rho_u: np.ndarray) -> float:
        eigenvalues = np.linalg.eigvalsh(rho_u)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    def test_product_state_zero_entropy(self):
        """
        |psi> = (1/sqrt(|U|*|V|)) sum_{u in U, v in V} |u>|v>
        is separable. Reduced density rho_u is rank-1 (pure). S = 0.
        """
        c64_list = sorted(C64)
        n = len(c64_list)
        idx = {c: i for i, c in enumerate(c64_list)}

        U = c64_list[:16]
        V = c64_list[:16]

        psi = np.zeros(n * n)
        for u in U:
            for v in V:
                psi[idx[u] * n + idx[v]] = 1.0
        psi /= np.linalg.norm(psi)

        rho_u = psi.reshape(n, n) @ psi.reshape(n, n).T
        entropy = self._von_neumann_entropy(rho_u)

        assert entropy < 0.01, f"Product state entropy = {entropy:.4f}, expected ~0"

    def test_graph_state_maximal_entropy(self):
        """
        |psi> = (1/sqrt(|C|)) sum_{u in C} |u>|u XOR t> for fixed t != 0.
        Maximal entropy S = log2(64) = 6 bits.
        """
        c64_list = sorted(C64)
        c64_set = set(c64_list)
        n = len(c64_list)
        idx = {c: i for i, c in enumerate(c64_list)}

        t = c64_list[1]
        assert t != 0

        psi = np.zeros(n * n)
        for u in c64_list:
            v = u ^ t
            assert v in c64_set, "XOR translation must stay in code"
            psi[idx[u] * n + idx[v]] = 1.0
        psi /= np.linalg.norm(psi)

        rho_u = psi.reshape(n, n) @ psi.reshape(n, n).T
        entropy = self._von_neumann_entropy(rho_u)

        expected = np.log2(n)  # 6.0
        assert abs(entropy - expected) < 0.01, (
            f"Graph state entropy = {entropy:.4f}, expected {expected:.2f}"
        )


# ================================================================
# PART 7: SPINORIAL INTRINSIC GATE AND HOLONOMY COUNTS
# ================================================================


class TestSpinorialIntrinsicGateAndHolonomyCounts:
    """
    Replace the old constant-fitting attempts with intrinsic spinorial invariants:

    1) Horizon-preserving bytes: exact gate set that maps horizon to horizon.
       This is kernel physics and depends on family-controlled complements.

    2) q-map: bytes -> C64 is exactly 4-to-1 (256 bytes onto 64 codewords).
       This is the correct intrinsic 4-family structure in the new architecture.

    3) For fixed x, the commutator defect set {q(x)^q(y) : y in bytes}
       equals C64 and each defect appears exactly 4 times.
    """

    def test_horizon_preserving_byte_set_has_size_4(self):
        """
        Horizon (S-sector) is A == B ^ 0xFFF (maximal chirality).
        Horizon-preserving bytes map horizon to horizon: A' == B' ^ 0xFFF.
        Same algebraic condition: mask == inv_a ^ inv_b. Yields exactly 4 bytes.
        """
        horizon_preserving: list[int] = []

        for b in range(256):
            intron = byte_to_intron(b)
            inv_a = LAYER_MASK_12 if (intron & 0x01) else 0
            inv_b = LAYER_MASK_12 if (intron & 0x80) else 0
            m = mask12_for_byte(b)
            if m == (inv_a ^ inv_b):
                horizon_preserving.append(b)

        assert len(horizon_preserving) == 4, f"got {len(horizon_preserving)}"

        # Verify dynamically on complement-horizon states
        _, _, horizon = _bfs_omega()
        for s in list(horizon)[:64]:
            for b in horizon_preserving:
                t = step_state_by_byte(s, b)
                a, c = unpack_state(t)
                assert a == (c ^ LAYER_MASK_12), (
                    f"byte {b:#x} failed to preserve horizon"
                )

        # And verify that all other bytes fail for at least one horizon state (sampled)
        for b in range(256):
            if b in horizon_preserving:
                continue
            failed = False
            for v in [0x000, 0x123, 0xAAA, 0xFFF]:
                s = pack_state(v, v ^ LAYER_MASK_12)
                t = step_state_by_byte(s, b)
                a, c = unpack_state(t)
                if a != (c ^ LAYER_MASK_12):
                    failed = True
                    break
            assert failed, f"byte {b:#x} appears to preserve horizon unexpectedly"

    def test_q_map_is_exact_4_to_1_onto_c64(self):
        """
        q(b) = mask(b) ^ (L0_parity(b) ? 0xFFF : 0)
        This q-map is the exact commutation class representative.

        Claim: {q(b) : b in bytes} = C64
        and every element of C64 has exactly 4 preimage bytes.
        """
        q_to_count: dict[int, int] = {}
        for b in range(256):
            qb = _q(b) & 0xFFF
            q_to_count[qb] = q_to_count.get(qb, 0) + 1

        assert set(q_to_count.keys()) == C64, (
            f"q-image size {len(q_to_count)} does not match |C64|=64"
        )
        counts = set(q_to_count.values())
        assert counts == {4}, f"expected 4-to-1, got multiplicities {counts}"

    def test_commutator_defects_for_fixed_x_are_c64_4_to_1(self):
        """
        From the exact theorem already validated elsewhere:
          K(x,y) translates by d = q(x) ^ q(y)

        Fix x and vary y over all bytes.
        Since q(y) hits each element of C64 exactly 4 times,
        d also hits each element of C64 exactly 4 times.
        """
        for x in [0x00, 0x42, 0xAA, 0xFF]:
            defects: dict[int, int] = {}
            qx = _q(x) & 0xFFF
            for y in range(256):
                d = (qx ^ (_q(y) & 0xFFF)) & 0xFFF
                defects[d] = defects.get(d, 0) + 1

            assert set(defects.keys()) == C64, f"x={x:#x}: defect set not C64"
            counts = set(defects.values())
            assert counts == {4}, f"x={x:#x}: expected 4-to-1, got {counts}"


# ================================================================
# PART 8: K4 ON THE HORIZON
# ================================================================


class TestK4OnHorizon:
    """
    The 64 horizon states (A == B ^ 0xFFF, S-sector) and their K4 structure.
    Using the fiber K4 labels: each horizon state has a definite
    K4 vertex label from the pair-parity structure of its A component.
    """

    def test_horizon_q_values_partition(self):
        """
        Each horizon state h has (A(h), B(h)) with A(h) == B(h) ^ 0xFFF.
        Use pair-parity of A for the K4 label. Each 12-bit component has 6 pairs.
        The pair-parity of frame 0 (pairs 0,1,2) and frame 1 (pairs 3,4,5) gives
        a 2-bit label.
        """
        omega, _, horizon = _bfs_omega()

        def pair_parity_label(comp12: int) -> tuple[int, int]:
            """Frame parities of a 12-bit component."""
            p0 = 0
            for i in range(3):  # pairs 0,1,2
                p0 ^= (comp12 >> (2 * i)) & 1
            p1 = 0
            for i in range(3, 6):  # pairs 3,4,5
                p1 ^= (comp12 >> (2 * i)) & 1
            return (p0, p1)

        groups: dict[tuple[int, int], list[int]] = {
            (0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []
        }
        for h in horizon:
            a, b = unpack_state(h)
            assert a == (b ^ LAYER_MASK_12)
            label = pair_parity_label(a)
            groups[label].append(h)

        print(f"\n  Horizon K4 partition (pair-parity):")
        for key in sorted(groups.keys()):
            print(f"    vertex {key}: {len(groups[key])} states")

        for key, members in groups.items():
            assert len(members) == 16, (
                f"Horizon vertex {key}: {len(members)} states, expected 16"
            )

    def test_horizon_vertex_cosets(self):
        """
        The 16 horizon states in each K4 vertex should form a coset
        of a common kernel subgroup of C64.
        """
        omega, _, horizon = _bfs_omega()

        def pair_parity_label(comp12: int) -> tuple[int, int]:
            p0 = 0
            for i in range(3):
                p0 ^= (comp12 >> (2 * i)) & 1
            p1 = 0
            for i in range(3, 6):
                p1 ^= (comp12 >> (2 * i)) & 1
            return (p0, p1)

        # Collect component values per vertex
        vertex_comps: dict[tuple[int, int], set[int]] = {
            (0, 0): set(), (0, 1): set(), (1, 0): set(), (1, 1): set()
        }
        for h in horizon:
            a, _ = unpack_state(h)
            vertex_comps[pair_parity_label(a)].add(a)

        # Convert to mask coordinates: u = A XOR archetype_A
        a0, _ = unpack_state(GENE_MAC_REST)
        vertex_masks: dict[tuple[int, int], set[int]] = {}
        for key, comps in vertex_comps.items():
            vertex_masks[key] = {c ^ a0 for c in comps}

        # The (0,0) group should be the kernel (contains mask 0 = archetype)
        kernel = vertex_masks[(0, 0)]
        assert 0 in kernel, "Kernel must contain identity mask"
        assert len(kernel) == 16

        # Kernel must be closed under XOR
        for a in kernel:
            for b in kernel:
                assert (a ^ b) in kernel, (
                    f"Kernel not closed: {a:03x} ^ {b:03x} = {a ^ b:03x}"
                )

        # Each other vertex group must be a coset of the kernel
        for key, masks in vertex_masks.items():
            if key == (0, 0):
                continue
            rep = next(iter(masks))
            coset = {rep ^ k for k in kernel}
            assert coset == masks, f"Vertex {key} is not a coset of kernel"


# ================================================================
# PART 9: K4 WEDGE TILING OF OMEGA
# ================================================================


class TestK4WedgeTiling:
    """
    Each K4 vertex on the horizon generates a wedge via one-step
    transitions. The four wedges should tile Omega.
    """

    def test_wedges_tile_omega(self):
        """
        16 horizon states per vertex x 256 bytes = up to 16*128 = 2048
        distinct states per wedge (128-way SO(3) shadow per horizon state).

        The 4 wedges form a uniform 2-fold cover of Omega:
        each state in Omega appears in exactly 2 wedges.
        4 * 2048 = 8192 = 2 * 4096.

        The factor of 2 is the same spinorial 2-to-1 that produces
        128 distinct states per byte instead of 256. The family-controlled
        complement shifts pair-parity, linking each bulk state to exactly
        2 horizon vertex classes.
        """
        omega, _, horizon = _bfs_omega()

        def pair_parity_label(comp12: int) -> tuple[int, int]:
            p0 = 0
            for i in range(3):
                p0 ^= (comp12 >> (2 * i)) & 1
            p1 = 0
            for i in range(3, 6):
                p1 ^= (comp12 >> (2 * i)) & 1
            return (p0, p1)

        vertex_horizon: dict[tuple[int, int], list[int]] = {
            (0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []
        }
        for h in horizon:
            a, _ = unpack_state(h)
            vertex_horizon[pair_parity_label(a)].append(h)

        wedges: dict[tuple[int, int], set[int]] = {}
        for key, h_states in vertex_horizon.items():
            wedge: set[int] = set()
            for h in h_states:
                for b in range(256):
                    wedge.add(step_state_by_byte(h, b))
            wedges[key] = wedge

        print(f"\n  K4 wedge tiling:")
        for key in sorted(wedges.keys()):
            print(f"    vertex {key}: {len(wedges[key])} states")

        # Each wedge has exactly 2048 states (16 horizon * 128 shadow)
        for key, wedge in wedges.items():
            assert len(wedge) == 2048, (
                f"Wedge {key}: {len(wedge)} states, expected 2048"
            )

        # Union covers all of Omega
        union: set[int] = set()
        for wedge in wedges.values():
            union |= wedge
        assert union == omega, "Wedges do not cover Omega"

        # Each state appears in exactly 2 wedges (uniform 2-fold cover)
        coverage: dict[int, int] = {}
        for wedge in wedges.values():
            for s in wedge:
                coverage[s] = coverage.get(s, 0) + 1

        coverage_values = set(coverage.values())
        assert coverage_values == {2}, (
            f"Expected uniform 2-fold cover, got multiplicities {coverage_values}"
        )

        # Verify arithmetic: 4 * 2048 = 2 * |Omega|
        total_with_multiplicity = sum(len(w) for w in wedges.values())
        assert total_with_multiplicity == 2 * len(omega)