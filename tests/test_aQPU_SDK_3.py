from __future__ import annotations

from collections import Counter
from fractions import Fraction
from math import comb

import pytest

import src.api as api
import src.sdk as sdk
from src.api import Q_WEIGHT_BY_BYTE
from src.constants import EPSILON_6, GENE_MAC_REST, step_state_by_byte

# GF(64) test-side helpers
GF64_POLY = 0x43   # x^6 + x + 1
GF64_MASK = 0x3F


def gf64_multiply(a: int, b: int) -> int:
    a &= GF64_MASK
    b &= GF64_MASK
    out = 0
    while b:
        if b & 1:
            out ^= a
        b >>= 1
        a <<= 1
        if a & 0x40:
            a ^= GF64_POLY
        a &= 0x7F
    return out & GF64_MASK


def gf64_pow(x: int, e: int) -> int:
    x &= GF64_MASK
    out = 1
    n = int(e)
    while n > 0:
        if n & 1:
            out = gf64_multiply(out, x)
        x = gf64_multiply(x, x)
        n >>= 1
    return out & GF64_MASK


def gf64_inverse(a: int) -> int:
    a &= GF64_MASK
    if a == 0:
        raise ZeroDivisionError("0 has no inverse in GF(64)")
    return gf64_pow(a, 62)


def gf64_frobenius(x: int, k: int = 1) -> int:
    out = x & GF64_MASK
    for _ in range(k % 6):
        out = gf64_multiply(out, out)
    return out & GF64_MASK


def gf64_trace(x: int) -> int:
    acc = 0
    cur = x & GF64_MASK
    for _ in range(6):
        acc ^= cur
        cur = gf64_multiply(cur, cur)
    return acc & 1


def gf64_norm(x: int) -> int:
    x &= GF64_MASK
    if x == 0:
        return 0
    return 1


def gf64_multiply_as_gf2_matrix(element: int) -> tuple[int, ...]:
    cols = [gf64_multiply(element, 1 << j) for j in range(6)]
    rows = []
    for r in range(6):
        row = 0
        for j, col in enumerate(cols):
            if (col >> r) & 1:
                row |= 1 << j
        rows.append(row)
    return tuple(rows)


def gf64_matvec(rows: tuple[int, ...], x: int) -> int:
    out = 0
    for r, row in enumerate(rows):
        bit = ((row & x).bit_count() & 1)
        out |= bit << r
    return out & GF64_MASK


def gf64_primitive_element() -> int:
    for g in range(2, 64):
        x = 1
        seen = set()
        for _ in range(63):
            x = gf64_multiply(x, g)
            seen.add(x)
        if len(seen) == 63:
            return g
    raise AssertionError("No primitive element found")


def _omega_states_from_rest() -> tuple[int, ...]:
    measure = sdk.MomentOps.future_cone(GENE_MAC_REST, 2)
    assert measure.distinct_states == 4096
    return tuple(state24 for state24, _ in measure.state_counts)


@pytest.fixture(scope="module")
def omega_states() -> tuple[int, ...]:
    return _omega_states_from_rest()


class TestGF4ModeLayer:
    def test_frobenius_pair_is_involution(self) -> None:
        for p in (0x0, 0x1, 0x2, 0x3):
            q = sdk.StateOps.frobenius_pair(p)
            assert sdk.StateOps.frobenius_pair(q) == p

        assert sdk.StateOps.frobenius_pair(0x0) == 0x0
        assert sdk.StateOps.frobenius_pair(0x3) == 0x3
        assert sdk.StateOps.frobenius_pair(0x2) == 0x1
        assert sdk.StateOps.frobenius_pair(0x1) == 0x2

    def test_trace_and_norm_on_all_pairs(self) -> None:
        assert sdk.StateOps.gf4_trace(0x0) == 0
        assert sdk.StateOps.gf4_trace(0x3) == 0
        assert sdk.StateOps.gf4_trace(0x2) == 1
        assert sdk.StateOps.gf4_trace(0x1) == 1

        assert sdk.StateOps.gf4_norm(0x0) == 0
        assert sdk.StateOps.gf4_norm(0x3) == 1
        assert sdk.StateOps.gf4_norm(0x2) == 1
        assert sdk.StateOps.gf4_norm(0x1) == 1

    def test_reachable_component_characterization(
        self, omega_states: tuple[int, ...]
    ) -> None:
        for s in omega_states:
            charts = sdk.StateOps.charts(s)
            assert sdk.StateOps.is_reachable_component(charts.a12)
            assert sdk.StateOps.is_reachable_component(charts.b12)

    def test_frobenius_component_matches_global_pair_flip_on_reachable_components(
        self,
        omega_states: tuple[int, ...],
    ) -> None:
        seen = set()
        for s in omega_states:
            charts = sdk.StateOps.charts(s)
            seen.add(charts.a12)
            seen.add(charts.b12)

        for c12 in seen:
            assert sdk.StateOps.is_reachable_component(c12)
            assert sdk.StateOps.frobenius_component12(c12) == (c12 ^ 0xFFF)


class TestShadowPartnersAndConjugation:
    def test_shadow_partner_formula(self) -> None:
        mapping = sdk.StateOps.shadow_partner_map()
        assert len(mapping) == 256

        for b in range(256):
            p = sdk.StateOps.shadow_partner_byte(b)
            assert p == (b ^ 0xFE)
            assert mapping[b] == p
            assert sdk.StateOps.shadow_partner_byte(p) == b
            assert mapping[p] == b

    def test_shadow_partner_same_omega_permutation(
        self, omega_states: tuple[int, ...]
    ) -> None:
        for b in range(256):
            p = sdk.StateOps.shadow_partner_byte(b)
            for s in omega_states:
                assert step_state_by_byte(s, b) == step_state_by_byte(s, p)

    def test_state_conjugate_f_matches_gate_F(
        self, omega_states: tuple[int, ...]
    ) -> None:
        for s in omega_states:
            assert sdk.StateOps.state_conjugate_f(s) == sdk.StateOps.gate(
                s, "F"
            )


class TestK4LociAndOrbits:
    def test_fixed_loci_match_horizons(self) -> None:
        equality = frozenset(sdk.MomentOps.states_on_locus(0))
        complement = frozenset(sdk.MomentOps.states_on_locus(6))

        assert sdk.StateOps.fixed_locus("id") == frozenset(
            s for w in range(7) for s in sdk.MomentOps.states_on_locus(w)
        )
        assert sdk.StateOps.fixed_locus("S") == equality
        assert sdk.StateOps.fixed_locus("C") == complement
        assert sdk.StateOps.fixed_locus("F") == frozenset()

    def test_k4_stabilizers(self, omega_states: tuple[int, ...]) -> None:
        for s in sdk.MomentOps.states_on_locus(0):
            assert sdk.StateOps.k4_stabilizer(s) == frozenset({"S"})
        for s in sdk.MomentOps.states_on_locus(6):
            assert sdk.StateOps.k4_stabilizer(s) == frozenset({"C"})

        bulk = [
            s
            for s in omega_states
            if sdk.MomentOps.stabilizer_type(s) == "bulk"
        ]
        for s in bulk[:256]:
            assert sdk.StateOps.k4_stabilizer(s) == frozenset()

    def test_k4_orbit_census(self, omega_states: tuple[int, ...]) -> None:
        seen = set()
        orbit_sizes = Counter()
        equality_orbits = 0
        complement_orbits = 0
        bulk_orbits = 0

        for s in omega_states:
            if s in seen:
                continue
            orb = sdk.StateOps.k4_orbit(s)
            seen.update(orb)
            orbit_sizes[len(orb)] += 1

            rep = next(iter(orb))
            stype = sdk.MomentOps.stabilizer_type(rep)
            if len(orb) == 2 and stype == "equality":
                equality_orbits += 1
            elif len(orb) == 2 and stype == "complement":
                complement_orbits += 1
            elif len(orb) == 4:
                bulk_orbits += 1

        assert orbit_sizes[2] == 64
        assert orbit_sizes[4] == 992
        assert equality_orbits == 32
        assert complement_orbits == 32
        assert bulk_orbits == 992
        assert len(seen) == 4096


class TestEvenSectorAndUniformization:
    def test_length2_even_operator_count(self) -> None:
        seen = set()
        for b1 in range(256):
            for b2 in range(256):
                sig = sdk.StateOps.omega_signature(bytes([b1, b2]))
                seen.add(sdk.StateOps.pack_omega_signature12(sig))
        assert len(seen) == 4096

    def test_two_step_uniformization_exact(
        self, omega_states: tuple[int, ...]
    ) -> None:
        sources = (
            GENE_MAC_REST,
            omega_states[0],
            omega_states[127],
            omega_states[2048],
            omega_states[4095],
        )

        for src in sources:
            counts = Counter()
            for b1 in range(256):
                s1 = step_state_by_byte(src, b1)
                for b2 in range(256):
                    s2 = step_state_by_byte(s1, b2)
                    counts[s2] += 1

            assert len(counts) == 4096
            assert set(counts.values()) == {16}


class TestOpticalVerificationAndShellMarkov:
    def test_verify_optical_conjugacy(
        self, omega_states: tuple[int, ...]
    ) -> None:
        def obs_plus(s: int) -> Fraction:
            charts = sdk.StateOps.charts(s)
            return Fraction(charts.constitutional.ab_distance, 12)

        def obs_minus(s: int) -> Fraction:
            charts = sdk.StateOps.charts(s)
            return Fraction(charts.constitutional.horizon_distance, 12)

        assert sdk.MomentOps.verify_optical_conjugacy(
            omega_states, obs_plus, obs_minus
        )

    def test_shell_markov_step_matches_kernel_rows(self) -> None:
        delta_shell_6 = tuple(Fraction(int(w == 6), 1) for w in range(7))
        for j in range(7):
            out = sdk.MomentOps.shell_markov_step(delta_shell_6, j)
            expected = tuple(
                sdk.MomentOps.shell_transition_probability(6, j, wp)
                for wp in range(7)
            )
            assert out == expected


class TestGF64ChiralityAlgebra:
    def test_basic_field_laws(self) -> None:
        for a in range(64):
            assert gf64_multiply(a, 1) == a

        for a in range(1, 64):
            assert gf64_multiply(a, gf64_inverse(a)) == 1

        for x in range(64):
            assert gf64_frobenius(x, 6) == x
            assert gf64_trace(x) in (0, 1)
            assert gf64_norm(x) in (0, 1)

    def test_trace_distribution(self) -> None:
        counts = Counter(gf64_trace(x) for x in range(64))
        assert counts[0] == 32
        assert counts[1] == 32

    def test_subfield_membership_sizes(self) -> None:
        gf2 = [x for x in range(64) if gf64_frobenius(x, 1) == x]
        gf4 = [x for x in range(64) if gf64_frobenius(x, 2) == x]
        gf8 = [x for x in range(64) if gf64_frobenius(x, 3) == x]

        assert len(gf2) == 2
        assert len(gf4) == 4
        assert len(gf8) == 8

    def test_matrix_representation_matches_multiplication(self) -> None:
        for a in range(64):
            M = gf64_multiply_as_gf2_matrix(a)
            for b in range(64):
                assert gf64_matvec(M, b) == gf64_multiply(a, b)

    def test_primitive_element(self) -> None:
        g = gf64_primitive_element()
        seen = set()
        x = 1
        for _ in range(63):
            x = gf64_multiply(x, g)
            seen.add(x)
        assert len(seen) == 63


class TestQFiberShadowGeometry:
    def test_each_q_fiber_splits_into_two_omega_maps(self) -> None:
        fiber_summary = []

        for q in range(64):
            fiber = [b for b in range(256) if sdk.SpectralOps.q_class(b) == q]
            assert len(fiber) == 4

            packed_sigs = [
                sdk.StateOps.pack_omega_signature12(
                    sdk.StateOps.omega_signature(bytes([b]))
                )
                for b in fiber
            ]
            counts = Counter(packed_sigs)

            fiber_summary.append((q, dict(counts)))

            assert len(counts) == 2
            assert sorted(counts.values()) == [2, 2]

        print("\nq-fiber -> Omega-map split summary (first 8 fibers):")
        for q, counts in fiber_summary[:8]:
            print(f"  q={q:02d} -> {counts}")

    def test_shadow_partner_stays_in_same_q_fiber_and_same_omega_map(
        self,
    ) -> None:
        for b in range(256):
            p = sdk.StateOps.shadow_partner_byte(b)

            assert sdk.SpectralOps.q_class(b) == sdk.SpectralOps.q_class(p)

            sig_b = sdk.StateOps.omega_signature(bytes([b]))
            sig_p = sdk.StateOps.omega_signature(bytes([p]))
            assert sig_b == sig_p


class TestByteCommutationGeometry:
    def test_bytes_commute_iff_q_classes_match(self) -> None:
        byte_sigs = [
            sdk.StateOps.omega_signature(bytes([b]))
            for b in range(256)
        ]
        q_words = [sdk.SpectralOps.q_class(b) for b in range(256)]

        commute_counts = []

        for x in range(256):
            cx = 0
            for y in range(256):
                xy = sdk.StateOps.compose_omega_signatures(
                    byte_sigs[y], byte_sigs[x]
                )
                yx = sdk.StateOps.compose_omega_signatures(
                    byte_sigs[x], byte_sigs[y]
                )
                commute = (xy == yx)

                assert commute == (q_words[x] == q_words[y])

                if commute:
                    cx += 1

            commute_counts.append(cx)

        print("\nper-byte commuting partner counts (distinct bytes, including self):")
        print(f"  unique counts = {sorted(set(commute_counts))}")

        assert set(commute_counts) == {4}


class TestDepth4ClosureAlgebra:
    def test_every_byte_has_order_4_on_omega(self) -> None:
        id_sig = sdk.OmegaSignature12(parity=0, tau_u6=0, tau_v6=0)

        for b in range(256):
            acc = sdk.OmegaSignature12(parity=0, tau_u6=0, tau_v6=0)
            sig_b = sdk.StateOps.omega_signature(bytes([b]))
            for _ in range(4):
                acc = sdk.StateOps.compose_omega_signatures(sig_b, acc)
            assert acc == id_sig

    def test_xyxy_is_identity_for_all_byte_pairs(self) -> None:
        id_sig = sdk.OmegaSignature12(parity=0, tau_u6=0, tau_v6=0)
        byte_sigs = [
            sdk.StateOps.omega_signature(bytes([b]))
            for b in range(256)
        ]

        for x in range(256):
            sx = byte_sigs[x]
            for y in range(256):
                sy = byte_sigs[y]
                acc = sdk.OmegaSignature12(parity=0, tau_u6=0, tau_v6=0)
                for sig in (sx, sy, sx, sy):
                    acc = sdk.StateOps.compose_omega_signatures(sig, acc)
                assert acc == id_sig


class TestSectorFactorization:
    def test_even_sector_factorizes_into_common_shift_and_chirality_translation(
        self,
    ) -> None:
        packed_even = set()

        for b1 in range(256):
            for b2 in range(256):
                sig = sdk.StateOps.omega_signature(bytes([b1, b2]))
                assert sig.parity == 0
                packed_even.add(sdk.StateOps.pack_omega_signature12(sig))

        assert len(packed_even) == 4096

        factor_pairs = set()
        shell_preserving = 0

        for packed in packed_even:
            sig = sdk.StateOps.unpack_omega_signature12(packed)
            common_shift = sig.tau_u6
            chirality_translation = sig.tau_u6 ^ sig.tau_v6
            factor_pairs.add((common_shift, chirality_translation))
            if chirality_translation == 0:
                shell_preserving += 1

        print("\neven-sector factorization:")
        print(f"  distinct even signatures             = {len(packed_even)}")
        print(f"  distinct (common_shift, q_shift)    = {len(factor_pairs)}")
        print(f"  shell-preserving even operators     = {shell_preserving}")

        assert len(factor_pairs) == 4096
        assert shell_preserving == 64

    def test_full_odd_sector_generated_from_even_sector_by_swap(self) -> None:
        packed_even = set()

        for b1 in range(256):
            for b2 in range(256):
                sig = sdk.StateOps.omega_signature(bytes([b1, b2]))
                packed_even.add(sdk.StateOps.pack_omega_signature12(sig))

        swap_sig = sdk.StateOps.omega_signature(bytes([0xAA]))
        packed_odd = set()

        for packed in packed_even:
            sig_even = sdk.StateOps.unpack_omega_signature12(packed)
            sig_odd = sdk.StateOps.compose_omega_signatures(swap_sig, sig_even)
            packed_odd.add(sdk.StateOps.pack_omega_signature12(sig_odd))

        print("\nodd-sector generation from even sector via swap:")
        print(f"  distinct odd signatures = {len(packed_odd)}")

        assert len(packed_odd) == 4096

        for packed in packed_odd:
            sig = sdk.StateOps.unpack_omega_signature12(packed)
            assert sig.parity == 1


class TestHorizonTransportByQWeight:
    def test_equality_horizon_maps_to_shell_equal_to_q_weight(self) -> None:
        equality_states = sdk.MomentOps.states_on_locus(0)

        for b in range(256):
            j = Q_WEIGHT_BY_BYTE[b]
            for s in equality_states:
                out = step_state_by_byte(s, b)
                w = sdk.MomentOps.locus_of_state(out)
                assert w == j

    def test_complement_horizon_maps_to_shell_six_minus_q_weight(self) -> None:
        complement_states = sdk.MomentOps.states_on_locus(6)

        for b in range(256):
            j = Q_WEIGHT_BY_BYTE[b]
            for s in complement_states:
                out = step_state_by_byte(s, b)
                w = sdk.MomentOps.locus_of_state(out)
                assert w == 6 - j


class TestShellSpectralEigenvalues:
    def test_exact_krawtchouk_eigenvalue_formula(self) -> None:
        """
        Exact shell spectral law:
            T_j K_k = (K_j(k) / C(6,j)) K_k
        where:
            T_j = shell transition matrix for q-weight j
            K_k(w) = api.KRAWTCHOUK_7[w][k]
        Note the indexing:
            api.KRAWTCHOUK_7[w][k] = K_k(w),
        so the eigenvalue is api.KRAWTCHOUK_7[k][j] / comb(6, j).
        """
        for j in range(7):
            T = sdk.MomentOps.shell_transition_matrix(j)

            for k in range(7):
                lam = Fraction(api.KRAWTCHOUK_7[k][j], comb(6, j))
                vec = [Fraction(api.KRAWTCHOUK_7[w][k], 1) for w in range(7)]

                for w in range(7):
                    lhs = sum(T[w][wp] * vec[wp] for wp in range(7))
                    rhs = lam * vec[w]
                    assert lhs == rhs


class TestEvenSectorMultiplicity:
    def test_length2_even_signature_map_is_exactly_16_to_1(self) -> None:
        """
        There are 256^2 = 65536 length-2 words and 4096 distinct even
        Omega-signatures. This test checks that the projection is exactly
        uniform: each even Omega-signature is realized by exactly 16
        two-byte words.
        """
        counts = Counter()

        for b1 in range(256):
            for b2 in range(256):
                sig = sdk.StateOps.omega_signature(bytes([b1, b2]))
                assert sig.parity == 0
                packed = sdk.StateOps.pack_omega_signature12(sig)
                counts[packed] += 1

        print("\nlength-2 even signature multiplicities:")
        print(f"  distinct signatures = {len(counts)}")
        print(f"  unique multiplicities = {sorted(set(counts.values()))}")

        assert len(counts) == 4096
        assert set(counts.values()) == {16}


class TestQFiberExactSignatureFormula:
    def test_q_fiber_exact_signature_formula(self) -> None:
        """
        For each q-class q, the 4-byte q-fiber collapses to exactly two
        odd Omega-signatures:
            (parity=1, tau_u=0,         tau_v=q)
            (parity=1, tau_u=EPSILON_6, tau_v=EPSILON_6 ^ q)
        each with multiplicity 2.
        """
        for q in range(64):
            fiber = [b for b in range(256) if sdk.SpectralOps.q_class(b) == q]
            assert len(fiber) == 4

            actual = Counter(
                sdk.StateOps.pack_omega_signature12(
                    sdk.StateOps.omega_signature(bytes([b]))
                )
                for b in fiber
            )

            sig0 = sdk.OmegaSignature12(parity=1, tau_u6=0, tau_v6=q)
            sig1 = sdk.OmegaSignature12(
                parity=1,
                tau_u6=EPSILON_6,
                tau_v6=EPSILON_6 ^ q,
            )

            expected = Counter({
                sdk.StateOps.pack_omega_signature12(sig0): 2,
                sdk.StateOps.pack_omega_signature12(sig1): 2,
            })

            assert actual == expected


class TestShellPreservingEvenSubgroup:
    def test_shell_preserving_even_signatures_are_exactly_diagonal_shifts(
        self,
    ) -> None:
        """
        An even Omega-signature preserves shell exactly iff tau_u6 == tau_v6.
        There are exactly 64 such signatures, one for each common shift c:
            (parity=0, tau_u=c, tau_v=c)
        """
        packed_even = set()
        shell_preserving = set()

        for b1 in range(256):
            for b2 in range(256):
                sig = sdk.StateOps.omega_signature(bytes([b1, b2]))
                assert sig.parity == 0
                packed = sdk.StateOps.pack_omega_signature12(sig)
                packed_even.add(packed)

                if sig.tau_u6 == sig.tau_v6:
                    shell_preserving.add(packed)

        expected = {
            sdk.StateOps.pack_omega_signature12(
                sdk.OmegaSignature12(parity=0, tau_u6=c, tau_v6=c)
            )
            for c in range(64)
        }

        print("\nshell-preserving even subgroup:")
        print(f"  observed size = {len(shell_preserving)}")
        print(f"  expected size = {len(expected)}")

        assert len(packed_even) == 4096
        assert shell_preserving == expected


class TestQFiberCommutationNeighborhood:
    def test_same_q_fiber_same_commutation_neighborhood(self) -> None:
        """
        All bytes in the same q-fiber have identical commutation neighborhoods.
        """
        byte_sigs = [
            sdk.StateOps.omega_signature(bytes([b]))
            for b in range(256)
        ]

        neighborhoods = {}

        for b in range(256):
            commute_set = frozenset(
                y
                for y in range(256)
                if sdk.StateOps.compose_omega_signatures(
                    byte_sigs[y], byte_sigs[b]
                )
                == sdk.StateOps.compose_omega_signatures(
                    byte_sigs[b], byte_sigs[y]
                )
            )
            neighborhoods[b] = commute_set

        for q in range(64):
            fiber = [b for b in range(256) if sdk.SpectralOps.q_class(b) == q]
            n0 = neighborhoods[fiber[0]]
            for b in fiber[1:]:
                assert neighborhoods[b] == n0
