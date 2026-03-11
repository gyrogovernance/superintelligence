"""
Moments physics 2: operator structure and capacity models.

Follow-up to test_moments_physics.py. Establishes:

1. Clifford: byte actions are exact Clifford unitaries (affine in 12-bit label space,
   conjugate Pauli X and Z to Pauli X and Z).
2. Stabilizer: the self-dual code defines a commuting Pauli stabilizer family.
3. Weyl: finite Weyl/charge-flux algebra on the 64-element code subspace.
4. Capacity models: spatial cell vs EM mode count (model-discrimination).
5. Generated operator family: byte alphabet generates exactly 8192 operators.
6. Central spinorial involution: 4-family cycle is global complement and central.
7. Frame operator quotient: depth-4 family freedom collapses to 4 operator classes.

Uses src.constants and src.api only.
"""

from __future__ import annotations

import math

import numpy as np

from src.api import MASK12_BY_BYTE
from src.constants import (
    GENE_MIC_S,
    LAYER_MASK_12,
    byte_to_intron,
    dot12,
    micro_ref_to_mask12,
    pack_state,
    step_state_by_byte,
    unpack_state,
)

ALL6 = 0x3F
F_CS = 9_192_631_770
OMEGA = 4096
HORIZON = 64

# Codeword index: micro_ref i -> mask micro_ref_to_mask12(i)
CODEWORD_BY_INDEX = tuple(micro_ref_to_mask12(r) for r in range(64))
INDEX_BY_CODEWORD = {c: r for r, c in enumerate(CODEWORD_BY_INDEX)}


def byte_affine_label_map(label12: int, byte: int) -> int:
    """
    Label space = 12 bits (u6 << 6) | v6, codeword indices.
    Byte action: u' = v xor alpha*111111, v' = u xor micro xor beta*111111.
    """
    u = (label12 >> 6) & 0x3F
    v = label12 & 0x3F
    intron = byte_to_intron(byte)
    alpha = intron & 1
    beta = (intron >> 7) & 1
    micro = (intron >> 1) & 0x3F
    u_next = v ^ (ALL6 if alpha else 0)
    v_next = u ^ micro ^ (ALL6 if beta else 0)
    return ((u_next & 0x3F) << 6) | (v_next & 0x3F)


def linear_part(label12: int) -> int:
    """Block swap on 6+6 bits."""
    u = (label12 >> 6) & 0x3F
    v = label12 & 0x3F
    return (v << 6) | u


def translation_part(byte: int) -> int:
    intron = byte_to_intron(byte)
    alpha = intron & 1
    beta = (intron >> 7) & 1
    micro = (intron >> 1) & 0x3F
    u_t = ALL6 if alpha else 0
    v_t = micro ^ (ALL6 if beta else 0)
    return (u_t << 6) | v_t


def label_to_state(label12: int) -> int:
    """Label (u6,v6) -> state24 = (codeword(u), codeword(v))."""
    u = (label12 >> 6) & 0x3F
    v = label12 & 0x3F
    a12 = CODEWORD_BY_INDEX[u]
    b12 = CODEWORD_BY_INDEX[v]
    return pack_state(a12, b12)


def state_to_label(state24: int) -> int:
    """State24 -> label (u6,v6) via codeword indices."""
    a12, b12 = unpack_state(state24)
    u = INDEX_BY_CODEWORD[a12]
    v = INDEX_BY_CODEWORD[b12]
    return (u << 6) | v


def apply_word_to_label(label12: int, word: list[int]) -> int:
    """Apply a word (list of bytes) to a 12-bit label."""
    x = label12
    for b in word:
        x = byte_affine_label_map(x, b)
    return x


def word_signature(word: list[int]) -> tuple[int, int]:
    """
    Return operator signature (parity, tau) where:
      parity = len(word) mod 2
      tau = action on zero label
    and the full action is x -> L^parity(x) xor tau.
    """
    parity = len(word) & 1
    tau = apply_word_to_label(0, word)
    return parity, tau


def compose_signatures(
    sig1: tuple[int, int], sig2: tuple[int, int]
) -> tuple[int, int]:
    """
    Composition: f_{p1,t1} o f_{p2,t2} = f_{p1 xor p2, L^{p1}(t2) xor t1}.
    """
    p1, t1 = sig1
    p2, t2 = sig2
    t2_push = linear_part(t2) if p1 else t2
    return (p1 ^ p2, t2_push ^ t1)


def byte_from_micro_family(micro: int, family: int) -> int:
    """Construct byte from 6-bit micro_ref and 2-bit family."""
    bit0 = family & 1
    bit7 = (family >> 1) & 1
    intron = (bit7 << 7) | ((micro & 0x3F) << 1) | bit0
    return intron ^ GENE_MIC_S


def phi_bits_from_families(
    f0: int, f1: int, f2: int, f3: int
) -> tuple[int, int]:
    """Net family-phase bits at depth-4: phi_a, phi_b."""
    a0, b0 = f0 & 1, (f0 >> 1) & 1
    a1, b1 = f1 & 1, (f1 >> 1) & 1
    a2, b2 = f2 & 1, (f2 >> 1) & 1
    a3, b3 = f3 & 1, (f3 >> 1) & 1
    phi_a = b0 ^ a1 ^ b2 ^ a3
    phi_b = a0 ^ b1 ^ a2 ^ b3
    return phi_a, phi_b


def gf2_rank(mat: np.ndarray) -> int:
    """Rank of matrix over GF(2)."""
    M = mat.copy() % 2
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
                M[row] = (M[row] ^ M[rank]) % 2
        rank += 1
    return rank


def symplectic_product(r1: np.ndarray, r2: np.ndarray) -> int:
    """
    r = [x | z] in GF(2)^24.
    omega(r1, r2) = x1.z2 + z1.x2 mod 2
    """
    x1, z1 = r1[:12], r1[12:]
    x2, z2 = r2[:12], r2[12:]
    return int(np.dot(x1, z2) + np.dot(z1, x2)) % 2


C64 = sorted(set(int(m) & 0xFFF for m in MASK12_BY_BYTE))


# -----
# 1. Clifford: byte actions are exact Clifford unitaries
# -----


class TestByteActionsAreClifford:
    """Byte actions as affine maps conjugate Pauli X and Z to Pauli X and Z."""

    def test_affine_decomposition(self):
        """f_b(x) = Lx xor t_b exactly."""
        for byte in range(256):
            t = translation_part(byte)
            for x in range(1 << 12):
                lhs = byte_affine_label_map(x, byte)
                rhs = linear_part(x) ^ t
                assert lhs == rhs

    def test_formula_matches_kernel(self):
        """Byte action in label space matches step_state_by_byte on Omega."""
        for byte in range(256):
            for u in range(64):
                for v in range(64):
                    label = (u << 6) | v
                    state = label_to_state(label)
                    next_state = step_state_by_byte(state, byte)
                    expected_label = state_to_label(next_state)
                    actual_label = byte_affine_label_map(label, byte)
                    assert actual_label == expected_label

    def test_conjugates_X_to_X(self):
        """
        U_b X(p) U_b^dag = X(L p)
        verified on all labels for random p and all bytes.
        """
        rng = np.random.default_rng(0)
        for byte in range(256):
            for _ in range(20):
                p = int(rng.integers(0, 1 << 12))
                lp = linear_part(p)
                for y in range(1 << 12):
                    t = translation_part(byte)
                    x = linear_part(y ^ t)
                    x_after = x ^ p
                    y_after = byte_affine_label_map(x_after, byte)
                    assert y_after == (y ^ lp)

    def test_conjugates_Z_to_Z_up_to_phase(self):
        """
        U_b Z(q) U_b^dag = (-1)^{q.(L t)} Z(L q)
        """
        rng = np.random.default_rng(1)
        for byte in range(256):
            t = translation_part(byte)
            lt = linear_part(t)
            for _ in range(20):
                q = int(rng.integers(0, 1 << 12))
                lq = linear_part(q)
                phase = dot12(q, lt)
                for y in range(1 << 12):
                    x = linear_part(y ^ t)
                    lhs = dot12(q, x)
                    rhs = phase ^ dot12(lq, y)
                    assert lhs == rhs


# -----
# 2. Stabilizer: self-dual code defines commuting Pauli stabilizers
# -----


class TestSelfDualCodeDefinesStabilizer:
    """The 64-element code supports a commuting Pauli stabilizer family."""

    def test_pair_basis_generators(self):
        """Six pair generators span the 64-element code."""
        G = []
        for i in range(6):
            m = micro_ref_to_mask12(1 << i)
            row = np.array([(m >> bit) & 1 for bit in range(12)], dtype=np.uint8)
            G.append(row)
        G = np.stack(G)
        assert gf2_rank(G) == 6

    def test_XZ_stabilizers_commute(self):
        """
        Build 12 Pauli stabilizer generators: X(g_i), Z(g_i) for six code generators.
        They must commute because the code is self-orthogonal.
        """
        rows = []
        for i in range(6):
            m = micro_ref_to_mask12(1 << i)
            g = np.array([(m >> bit) & 1 for bit in range(12)], dtype=np.uint8)
            row_x = np.concatenate([g, np.zeros(12, dtype=np.uint8)])
            row_z = np.concatenate([np.zeros(12, dtype=np.uint8), g])
            rows.append(row_x)
            rows.append(row_z)
        rows = np.stack(rows)
        for i in range(len(rows)):
            for j in range(len(rows)):
                assert symplectic_product(rows[i], rows[j]) == 0

    def test_stabilizer_rank_is_12(self):
        """
        12 independent commuting stabilizers on 12 qubits define a unique stabilizer state.
        """
        rows = []
        for i in range(6):
            m = micro_ref_to_mask12(1 << i)
            g = np.array([(m >> bit) & 1 for bit in range(12)], dtype=np.uint8)
            rows.append(np.concatenate([g, np.zeros(12, dtype=np.uint8)]))
            rows.append(np.concatenate([np.zeros(12, dtype=np.uint8), g]))
        S = np.stack(rows)
        assert gf2_rank(S) == 12


# -----
# 3. Finite Weyl algebra on the code subspace
# -----


class TestFiniteWeylAlgebra:
    """Code subspace carries exact finite Heisenberg-Weyl algebra."""

    def test_weyl_commutation_on_code(self):
        """
        On the code subspace:
          X_d |x> = |x xor d>, Z_s |x> = (-1)^{s.x} |x>
          Z_s X_d = (-1)^{s.d} X_d Z_s
        """
        code_set = set(C64)
        for s in C64[:16]:
            for d in C64[:16]:
                for x in C64[:16]:
                    assert (x ^ d) in code_set
                    lhs_phase = dot12(s, x ^ d)
                    rhs_phase = dot12(s, d) ^ dot12(s, x)
                    assert lhs_phase == rhs_phase

    def test_translation_group_is_code(self):
        """Translations on the code close exactly under xor."""
        code_set = set(C64)
        for a in C64:
            for b in C64:
                assert (a ^ b) in code_set


# -----
# 4. Physical capacity models: spatial cells vs EM modes
# -----


class TestPhysicalCapacityModels:
    """Compare spatial coarse-graining vs EM mode count for capacity."""

    def test_spatial_cell_vs_em_mode_count(self):
        """
        Compare:
          N_cells = V / lambda^3 = (4/3) pi f^3
          N_modes = (8 pi V / 3 c^3) f^3 = (32 pi^2 / 9) f^3
        """
        n_cells = (4 / 3) * math.pi * F_CS**3
        n_modes = (32 * math.pi**2 / 9) * F_CS**3
        ratio = n_modes / n_cells
        expected = 8 * math.pi / 3
        assert abs(ratio - expected) / expected < 1e-12
        csm_cells = n_cells / OMEGA
        csm_modes = n_modes / OMEGA
        print("\n  N_cells: %e" % n_cells)
        print("  N_modes: %e" % n_modes)
        print("  N_modes / N_cells: %.6f" % ratio)
        print("  CSM_cells: %e" % csm_cells)
        print("  CSM_modes: %e" % csm_modes)

    def test_boundary_normalized_alternatives(self):
        """Compare bulk and boundary normalizations under both physical models."""
        n_cells = (4 / 3) * math.pi * F_CS**3
        n_area = 4 * math.pi * F_CS**2
        n_modes = (32 * math.pi**2 / 9) * F_CS**3
        csm_bulk_cells = n_cells / OMEGA
        csm_bulk_modes = n_modes / OMEGA
        csm_boundary = n_area / HORIZON
        print("\n  bulk(cells): %e" % csm_bulk_cells)
        print("  bulk(modes): %e" % csm_bulk_modes)
        print("  boundary(area): %e" % csm_boundary)


# -----
# 5. Generated AQPU operator family
# -----


class TestGeneratedAQPUOperatorFamily:
    """Exact generated operator family of the byte alphabet."""

    def test_word_signature_recovers_full_action(self):
        """Any word acts as x -> L^p x xor tau."""
        rng = np.random.default_rng(2027)
        for _ in range(500):
            word_len = int(rng.integers(1, 8))
            word = [int(rng.integers(0, 256)) for _ in range(word_len)]
            p, tau = word_signature(word)
            for _ in range(50):
                x = int(rng.integers(0, 1 << 12))
                pred = (linear_part(x) if p else x) ^ tau
                got = apply_word_to_label(x, word)
                assert got == pred

    def test_signature_composition_law(self):
        """Operator signatures compose exactly by the semidirect-product law."""
        rng = np.random.default_rng(2028)
        for _ in range(300):
            w1 = [int(rng.integers(0, 256)) for _ in range(int(rng.integers(1, 6)))]
            w2 = [int(rng.integers(0, 256)) for _ in range(int(rng.integers(1, 6)))]
            sig1 = word_signature(w1)
            sig2 = word_signature(w2)
            sig12 = word_signature(w1 + w2)
            pred = compose_signatures(sig2, sig1)
            assert sig12 == pred

    def test_even_two_byte_words_realize_all_4096_translations(self):
        """
        Length-2 words realize all 4096 even operators (identity linear part).
        """
        even_ops: dict[int, tuple[int, int]] = {}
        for x in range(256):
            for y in range(256):
                sig = word_signature([x, y])
                p, tau = sig
                assert p == 0
                even_ops.setdefault(tau, (x, y))
        assert len(even_ops) == 4096

    def test_full_generated_family_has_size_8192(self):
        """
        The byte-generated operator family has exactly:
          4096 even operators + 4096 odd operators = 8192.
        """
        even_ops: dict[int, tuple[int, int]] = {}
        for x in range(256):
            for y in range(256):
                sig = word_signature([x, y])
                p, tau = sig
                assert p == 0
                even_ops.setdefault(tau, (x, y))
        assert len(even_ops) == 4096

        odd_ops = set()
        for tau, word2 in even_ops.items():
            sig = word_signature([0xAA, word2[0], word2[1]])
            p, tau_odd = sig
            assert p == 1
            odd_ops.add(tau_odd)
        assert len(odd_ops) == 4096

        total_family = {(0, t) for t in even_ops} | {(1, t) for t in odd_ops}
        assert len(total_family) == 8192


# -----
# 6. Central spinorial involution
# -----


class TestSpinorialCenter:
    """The 4-family cycle defines a central involution."""

    def test_family_cycle_word_has_same_signature_for_all_microrefs(self):
        """
        For every micro_ref, the word over families (0,1,2,3)
        produces the same operator signature: global complement.
        """
        signatures = set()
        for micro in range(64):
            word4 = [byte_from_micro_family(micro, f) for f in (0, 1, 2, 3)]
            signatures.add(word_signature(word4))
        assert len(signatures) == 1

        sig = next(iter(signatures))
        parity, tau = sig
        assert parity == 0
        expected_tau = (ALL6 << 6) | ALL6
        assert tau == expected_tau

    def test_family_cycle_is_global_complement_on_all_labels(self):
        """The 4-family cycle acts as x -> x xor ((111111 << 6) | 111111)."""
        expected_tau = (ALL6 << 6) | ALL6
        for micro in range(64):
            word4 = [byte_from_micro_family(micro, f) for f in (0, 1, 2, 3)]
            for x in range(1 << 12):
                got = apply_word_to_label(x, word4)
                assert got == (x ^ expected_tau)

    def test_global_complement_is_central(self):
        """The 4-family-cycle involution commutes with every byte action."""
        center_sig = word_signature(
            [byte_from_micro_family(0, f) for f in (0, 1, 2, 3)]
        )
        for byte in range(256):
            byte_sig = word_signature([byte])
            lhs = compose_signatures(center_sig, byte_sig)
            rhs = compose_signatures(byte_sig, center_sig)
            assert lhs == rhs


# -----
# 7. Depth-4 frame as exact operator quotient
# -----


class TestFrameOperatorQuotient:
    """Depth-4 family freedom collapses to a 4-class operator quotient."""

    def test_fixed_payload_frame_has_exactly_4_operator_classes(self):
        """
        For fixed micro-refs, 256 family assignments collapse to exactly
        4 operator classes, each with multiplicity 64; class = (phi_a, phi_b).
        """
        micros = [1, 7, 13, 29]
        phi_to_sig: dict[tuple[int, int], tuple[int, int]] = {}
        counts: dict[tuple[int, int], int] = {}

        for f0 in range(4):
            for f1 in range(4):
                for f2 in range(4):
                    for f3 in range(4):
                        word = [
                            byte_from_micro_family(micros[0], f0),
                            byte_from_micro_family(micros[1], f1),
                            byte_from_micro_family(micros[2], f2),
                            byte_from_micro_family(micros[3], f3),
                        ]
                        sig = word_signature(word)
                        phi = phi_bits_from_families(f0, f1, f2, f3)
                        counts[phi] = counts.get(phi, 0) + 1
                        if phi in phi_to_sig:
                            assert phi_to_sig[phi] == sig
                        else:
                            phi_to_sig[phi] = sig

        assert len(phi_to_sig) == 4
        assert set(counts.values()) == {64}
