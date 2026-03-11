"""
Physics tests 5: CGM constants bridge + frame-level emergence.

Bridge (constants.py <-> CGM theory):
  (1) continuous invariants, (2) aperture quantization chain,
  (3) DOF doubling law, (4) optical conjugacy on Omega.

Frame physics (depth-4 PP-PF, not covered as theorems elsewhere):
  Depth-4 closed form, net family-phase (phi_a, phi_b) survival,
  mask-projection gauge blindness, exact commutation law,
  exact commutator defect formula, optical conjugacy via Omega = U x V,
  universal family-cycle (4-step sign flip, 8-step identity).
"""

from __future__ import annotations

import math

import numpy as np

from src.api import MASK12_BY_BYTE, depth4_mask_projection48, mask12_for_byte
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
    component_density,
    inverse_step_by_byte,
    step_state_by_byte,
    unpack_state,
)

C64 = set(int(m) & 0xFFF for m in MASK12_BY_BYTE)


def _intron_bits(byte: int) -> tuple[int, int, int, int]:
    """
    Returns (intron, a, b, micro):
      a = intron bit0, b = intron bit7, micro = payload bits 1-6.
    """
    intron = byte_to_intron(byte)
    a = intron & 1
    b = (intron >> 7) & 1
    micro = (intron >> 1) & 0x3F
    return intron, a, b, micro


def _byte_from_micro_family(micro: int, family: int) -> int:
    """family in {0,1,2,3} -> (bit7, bit0) = (family>>1, family&1)."""
    bit0 = family & 1
    bit7 = (family >> 1) & 1
    intron = (bit7 << 7) | ((micro & 0x3F) << 1) | bit0
    return intron ^ GENE_MIC_S


def _phi_bits(f0: int, f1: int, f2: int, f3: int) -> tuple[int, int]:
    """
    Net family-phase bits surviving at depth-4 (closed form):
      phi_a = b0 ^ a1 ^ b2 ^ a3,  phi_b = a0 ^ b1 ^ a2 ^ b3
    where for family f: a = f&1, b = (f>>1)&1.
    """
    a0, b0 = f0 & 1, (f0 >> 1) & 1
    a1, b1 = f1 & 1, (f1 >> 1) & 1
    a2, b2 = f2 & 1, (f2 >> 1) & 1
    a3, b3 = f3 & 1, (f3 >> 1) & 1
    phi_a = b0 ^ a1 ^ b2 ^ a3
    phi_b = a0 ^ b1 ^ a2 ^ b3
    return phi_a, phi_b


class TestCGMContinuousInvariants:
    """
    The physical constants stored in constants.py must be
    mutually consistent with the CGM theoretical framework.
    These tests are the only ones that connect the kernel
    constants back to continuous CGM physics.
    """

    def test_fundamental_aperture_constraint(self):
        """Q_G * m_a^2 = 1/2."""
        Q_G = 4 * math.pi
        assert abs(Q_G * M_A ** 2 - 0.5) < 1e-12

    def test_rho_aperture_consistency(self):
        """RHO = DELTA_BU / M_A and APERTURE_GAP = 1 - RHO."""
        assert abs(RHO - DELTA_BU / M_A) < 1e-12
        assert abs(APERTURE_GAP - (1.0 - RHO)) < 1e-12

    def test_ma_definition(self):
        """M_A = 1 / (2 * sqrt(2*pi))."""
        assert abs(M_A - 1.0 / (2.0 * math.sqrt(2.0 * math.pi))) < 1e-15

    def test_fine_structure_constant_prediction(self):
        """alpha = delta_BU^4 / m_a as stated in CGM paper."""
        alpha_cgm = DELTA_BU ** 4 / M_A
        alpha_exp = 0.0072973525693
        assert abs(alpha_cgm - alpha_exp) / alpha_exp < 4e-4

    def test_delta_bu_near_dyadic(self):
        """delta_BU is within ~0.5% of pi/16 (near-dyadic structure)."""
        assert abs(DELTA_BU - math.pi / 16) / (math.pi / 16) < 0.006

    def test_kqg_identity(self):
        """K_QG = (pi/4)/m_a = pi^2/sqrt(2*pi): two derivations agree."""
        kqg_1 = (math.pi / 4) / M_A
        kqg_2 = math.pi ** 2 / math.sqrt(2 * math.pi)
        assert abs(kqg_1 - kqg_2) < 1e-12

    def test_stage_action_ratios(self):
        """E_ONA/E_CS = 1/2 and E_UNA/E_CS = 2/(pi*sqrt(2))."""
        s_cs = (math.pi / 2) / M_A
        s_una = (1 / math.sqrt(2)) / M_A
        s_ona = (math.pi / 4) / M_A
        assert abs(s_ona / s_cs - 0.5) < 1e-12
        assert abs(s_una / s_cs - 2 / (math.pi * math.sqrt(2))) < 1e-12

    def test_monodromy_hierarchy(self):
        """omega(ONA-BU) < delta_BU < SU(2) holonomy < 4-leg toroidal."""
        omega = DELTA_BU / 2
        assert omega < DELTA_BU
        assert DELTA_BU < 0.587901
        assert 0.587901 < 0.862833

    def test_single_transition_memory(self):
        """delta_BU = 2 * omega(ONA<->BU) where omega = 0.097671."""
        assert abs(DELTA_BU / 2 - 0.097671) < 1e-5


class TestApertureQuantizationChain:
    """
    CGM Byte Formalism Section 7: the continuous aperture gap
    maps to exact discrete approximants at the 8-bit and 48-bit scales.
    """

    def test_byte_horizon_5_over_256(self):
        """Best 8-bit dyadic approximation of Delta is 5/256."""
        assert APERTURE_GAP_Q256 == 5
        assert round(APERTURE_GAP * 256) == 5

    def test_depth4_horizon_1_over_48(self):
        """48 * Delta rounds to 1: depth-4 aperture horizon."""
        product = 48 * APERTURE_GAP
        assert round(product) == 1
        assert abs(product - 1.0) < 0.01

    def test_turn_space_1_over_32(self):
        """delta_BU / (2*pi) quantizes to 8/256 = 1/32 turn."""
        tau = DELTA_BU / (2 * math.pi)
        assert round(256 * tau) == 8

    def test_chirality_space_ratio_two_thirds(self):
        """(1/48) / (1/32) = 2/3 = chirality/space."""
        from fractions import Fraction
        assert Fraction(1, 48) / Fraction(1, 32) == Fraction(2, 3)


class TestDOFDoublingLaw:
    """
    CGM continuous theory gives 1, 3, 6 DOF at CS, UNA, ONA.
    The discrete kernel has reachable state counts 4, 64, 4096.
    The connection is: discrete states = 2^(2 * continuous_DOF)
    because each continuous DOF maps to one dipole pair (2 bits).
    This is the only test that verifies this specific bridging claim.
    """

    @staticmethod
    def _bytes_restricted_to(intron_bits: list[int]) -> list[int]:
        mask = sum(1 << b for b in intron_bits)
        return [i ^ GENE_MIC_S for i in range(256) if (i & ~mask) == 0]

    @staticmethod
    def _reachable(allowed: list[int]) -> int:
        visited = {GENE_MAC_REST}
        frontier = {GENE_MAC_REST}
        while frontier:
            nxt = set()
            for s in frontier:
                for b in allowed:
                    t = step_state_by_byte(s, b)
                    if t not in visited:
                        visited.add(t)
                        nxt.add(t)
            frontier = nxt
        return len(visited)

    def test_dof_doubling_at_cs_una_ona(self):
        cs_n = self._reachable(self._bytes_restricted_to([0, 7]))
        una_n = self._reachable(self._bytes_restricted_to([0, 1, 6, 7]))
        ona_n = self._reachable(list(range(256)))

        assert cs_n == 4     # 2^(2*1)
        assert una_n == 64   # 2^(2*3)
        assert ona_n == 4096 # 2^(2*6)

        assert math.log2(cs_n) / 2 == 1.0
        assert math.log2(una_n) / 2 == 3.0
        assert math.log2(ona_n) / 2 == 6.0


class TestOpticalConjugacyOnOmega:
    """
    Optical conjugacy (density law) on Omega using the proven product form:
      Omega = U x V where U = A_rest xor C64 and V = B_rest xor C64.
    No BFS; reachability is covered elsewhere.
    """

    def test_all_omega_components_have_popcount_6(self):
        a0, b0 = unpack_state(GENE_MAC_REST)
        U = {a0 ^ c for c in C64}
        V = {b0 ^ c for c in C64}
        assert len(U) == 64 and len(V) == 64
        for u in U:
            assert bin(u & 0xFFF).count("1") == 6
        for v in V:
            assert bin(v & 0xFFF).count("1") == 6

    def test_density_sum_and_product_constants(self):
        a0, b0 = unpack_state(GENE_MAC_REST)
        U = {a0 ^ c for c in C64}
        V = {b0 ^ c for c in C64}
        for u in U:
            for v in V:
                du = component_density(u)
                dv = component_density(v)
                assert du == 0.5 and dv == 0.5
                assert du + dv == 1.0
                assert du * dv == 0.25


# -----------------------------------------------------------------------
# Frame physics: depth-4 PP-PF and exact CGM algebra
# -----------------------------------------------------------------------

class TestDepth4FrameClosedForm:
    """
    Depth-4 frame theorem (Prefix, Present, Past, Future): closed-form
    output. Validates PP-PF frame memory, not random bytes.
    """

    def test_depth4_closed_form_matches_kernel(self):
        a0, b0 = unpack_state(GENE_MAC_REST)
        rng = np.random.default_rng(12345)
        for _ in range(2000):
            word = [int(rng.integers(0, 256)) for _ in range(4)]
            b0y, b1y, b2y, b3y = word

            m0 = mask12_for_byte(b0y)
            m1 = mask12_for_byte(b1y)
            m2 = mask12_for_byte(b2y)
            m3 = mask12_for_byte(b3y)

            _, a0bit, b0bit, _ = _intron_bits(b0y)
            _, a1bit, b1bit, _ = _intron_bits(b1y)
            _, a2bit, b2bit, _ = _intron_bits(b2y)
            _, a3bit, b3bit, _ = _intron_bits(b3y)

            u0 = LAYER_MASK_12 if a0bit else 0
            v0 = LAYER_MASK_12 if b0bit else 0
            u1 = LAYER_MASK_12 if a1bit else 0
            v1 = LAYER_MASK_12 if b1bit else 0
            u2 = LAYER_MASK_12 if a2bit else 0
            v2 = LAYER_MASK_12 if b2bit else 0
            u3 = LAYER_MASK_12 if a3bit else 0
            v3 = LAYER_MASK_12 if b3bit else 0

            pred_a = (a0 ^ m0 ^ m2 ^ v0 ^ u1 ^ v2 ^ u3) & 0xFFF
            pred_b = (b0 ^ m1 ^ m3 ^ u0 ^ v1 ^ u2 ^ v3) & 0xFFF

            s = GENE_MAC_REST
            for b in word:
                s = step_state_by_byte(s, b)
            a4, b4 = unpack_state(s)
            assert a4 == pred_a
            assert b4 == pred_b


class TestDepth4NetFamilyPhase:
    """
    For fixed masks (fixed micro_ref at each position), the depth-4 frame
    retains only two net family-phase bits (phi_a, phi_b). Exactly 4
    distinct outputs; each (phi_a, phi_b) maps to a unique state.
    """

    def test_fixed_mask_frame_surviving_invariants_are_phi_a_phi_b(self):
        micros = [1, 7, 13, 29]
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
                        phi_a, phi_b = _phi_bits(f0, f1, f2, f3)
                        key = (phi_a, phi_b)
                        if key in phi_to_state:
                            assert phi_to_state[key] == s
                        else:
                            phi_to_state[key] = s
        assert len(phi_to_state) == 4


class TestDepth4MaskProjectionGaugeBlind:
    """
    At PP-PF frame level: depth-4 mask projection depends only on
    micro_refs (payload), not on family (gauge). Same payloads -> same
    48-bit projection regardless of family assignment.
    """

    def test_depth4_mask_projection_depends_only_on_micro_refs(self):
        micros = [1, 7, 13, 29]
        w_a = [_byte_from_micro_family(micros[i], 0) for i in range(4)]
        w_b = [_byte_from_micro_family(micros[i], 3) for i in range(4)]
        p_a = depth4_mask_projection48(*w_a)
        p_b = depth4_mask_projection48(*w_b)
        assert p_a == p_b


class TestExactDepth2CommutationLaw:
    """
    Exact discrete UNA/ONA: T_x T_y = T_y T_x  <=>  q(x) = q(y),
    where q(b) = mask(b) xor (L0_parity(b) ? 0xFFF : 0).
    """

    @staticmethod
    def _q(byte: int) -> int:
        intron = byte_to_intron(byte)
        l0 = (intron & 1) ^ ((intron >> 7) & 1)
        return mask12_for_byte(byte) ^ (LAYER_MASK_12 if l0 else 0)

    def test_commutation_iff_q_equal(self):
        rng = np.random.default_rng(2026)
        for _ in range(5000):
            x = int(rng.integers(0, 256))
            y = int(rng.integers(0, 256))
            lhs = step_state_by_byte(step_state_by_byte(GENE_MAC_REST, x), y)
            rhs = step_state_by_byte(step_state_by_byte(GENE_MAC_REST, y), x)
            assert (lhs == rhs) == (self._q(x) == self._q(y))


class TestExactCommutatorDefectFormula:
    """
    K(x,y) = T_x T_y T_x^{-1} T_y^{-1} is translation by d = q(x) xor q(y);
    (A,B) -> (A^d, B^d). Defect d in C64.
    """

    @staticmethod
    def _q(byte: int) -> int:
        intron = byte_to_intron(byte)
        l0 = (intron & 1) ^ ((intron >> 7) & 1)
        return mask12_for_byte(byte) ^ (LAYER_MASK_12 if l0 else 0)

    def test_commutator_is_symmetric_translation_by_qxor(self):
        a0, b0 = unpack_state(GENE_MAC_REST)
        rng = np.random.default_rng(999)
        for _ in range(5000):
            x = int(rng.integers(0, 256))
            y = int(rng.integers(0, 256))
            d = self._q(x) ^ self._q(y)
            s = GENE_MAC_REST
            s = step_state_by_byte(s, x)
            s = step_state_by_byte(s, y)
            s = inverse_step_by_byte(s, x)
            s = inverse_step_by_byte(s, y)
            a, b = unpack_state(s)
            assert a == (a0 ^ d) & 0xFFF
            assert b == (b0 ^ d) & 0xFFF
            assert d in C64


class TestUniversalFamilyCycle:
    """
    For every micro_ref, the 4-family cycle (0->1->2->3) over 4 bytes
    is a global sign flip; two cycles return to rest.
    """

    def test_all_microrefs_have_4step_signflip_and_8step_identity(self):
        a0, b0 = unpack_state(GENE_MAC_REST)
        for micro in range(64):
            word4 = [_byte_from_micro_family(micro, f) for f in (0, 1, 2, 3)]
            s4 = GENE_MAC_REST
            for b in word4:
                s4 = step_state_by_byte(s4, b)
            a4, b4 = unpack_state(s4)
            assert a4 == (a0 ^ LAYER_MASK_12) & 0xFFF
            assert b4 == (b0 ^ LAYER_MASK_12) & 0xFFF
            s8 = s4
            for b in word4:
                s8 = step_state_by_byte(s8, b)
            assert s8 == GENE_MAC_REST
