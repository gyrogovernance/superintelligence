# tests/test_aQPU_SDK_2.py
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from math import comb
import random

import pytest

torch = pytest.importorskip("torch")
from torch import Tensor as TorchTensor

import src.sdk as sdk
from src.api import (
    EPS_A6_BY_BYTE,
    EPS_B6_BY_BYTE,
    FULL_BYTE_SHELL_DISTRIBUTION,
    KRAWTCHOUK_7,
    MICRO_REF_BY_BYTE,
    Q_WEIGHT_BY_BYTE,
    shell_krawtchouk_transform_float,
    shell_transition_probability,
)
from src.constants import GENE_MAC_REST, step_state_by_byte


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _shell_distribution_from_counts(counts: list[int] | tuple[int, ...], total: int) -> tuple[Fraction, ...]:
    return tuple(Fraction(int(c), int(total)) for c in counts)


def _print_shell_table(title: str, counts: list[int] | tuple[int, ...], total: int | None = None) -> None:
    print(f"\n{title}")
    print("  shell | count | probability")
    print("  ------+-------+------------")
    for w, c in enumerate(counts):
        if total is None or total == 0:
            p = "-"
        else:
            p = f"{float(Fraction(c, total)):.6f}"
        print(f"   {w:>2}   | {c:>5} | {p}")


def _omega_states_from_rest() -> tuple[int, ...]:
    measure = sdk.MomentOps.future_cone(GENE_MAC_REST, 2)
    assert measure.distinct_states == 4096
    return tuple(state24 for state24, _ in measure.state_counts)


def _sample_states(states: tuple[int, ...], n: int, seed: int = 12345) -> tuple[int, ...]:
    rng = random.Random(seed)
    if n >= len(states):
        return states
    idxs = sorted(rng.sample(range(len(states)), n))
    return tuple(states[i] for i in idxs)


def _mixed_state_batch(omega_states: tuple[int, ...]) -> TorchTensor:
    mixed = [
        omega_states[0],
        omega_states[1],
        omega_states[127],
        omega_states[2048],
        0x000000,
        0x123456,
        0xFFFFFF,
        0x024924,
    ]
    return torch.tensor(mixed, dtype=torch.int32, device="cpu")


def _bytes_tensor(seq: list[int] | bytes) -> TorchTensor:
    if isinstance(seq, bytes):
        return torch.tensor(list(seq), dtype=torch.uint8, device="cpu")
    return torch.tensor(seq, dtype=torch.uint8, device="cpu")


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def omega_states() -> tuple[int, ...]:
    states = _omega_states_from_rest()
    print(f"\n[fixture] Ω states recovered from future cone at n=2: {len(states)}")
    return states


@pytest.fixture(scope="module")
def witness_depth_histogram(omega_states: tuple[int, ...]) -> dict[int, int]:
    hist = Counter()
    for s in omega_states:
        w = sdk.StateOps.witness_from_rest(s)
        hist[w.depth] += 1
    print("\n[fixture] witness depth histogram from rest:")
    for depth in sorted(hist):
        print(f"  depth {depth}: {hist[depth]}")
    return dict(hist)


@pytest.fixture(scope="module")
def native_ready() -> bool:
    try:
        sdk.initialize_native()
    except Exception as e:
        print(f"\n[native] initialize_native() raised: {e!r}")
    available = bool(sdk.gyrolabe_available())
    print(f"\n[native] GyroLabe available: {available}")
    return available


# ---------------------------------------------------------------------
# Core Ω-chart tests
# ---------------------------------------------------------------------


class TestOmegaChartCore:
    def test_rest_state_coordinates(self) -> None:
        omega = sdk.StateOps.to_omega12(GENE_MAC_REST)
        print("\nrest state in Ω-chart:")
        print(f"  state24 = 0x{GENE_MAC_REST:06x}")
        print(f"  omega   = (u6=0x{omega.u6:02x}, v6=0x{omega.v6:02x})")
        print(f"  shell   = {omega.shell}")
        assert omega.u6 == 0x00
        assert omega.v6 == 0x3F
        assert omega.shell == 6
        assert omega.is_on_complement_horizon
        assert not omega.is_on_equality_horizon

    def test_roundtrip_on_all_omega_states(self, omega_states: tuple[int, ...]) -> None:
        for s in omega_states:
            omega = sdk.StateOps.to_omega12(s)
            rebuilt = sdk.StateOps.from_omega12(omega)
            assert rebuilt == s

        print(f"\nroundtrip verified on all Ω states: {len(omega_states)}")
        assert len(omega_states) == 4096

    def test_structural_membership_and_off_omega_behavior(self) -> None:
        off = 0x000000
        charts = sdk.StateOps.charts(off)
        print("\noff-Ω sample:")
        print(f"  state24 = 0x{off:06x}")
        print(f"  omega12 = {charts.omega12}")
        print(f"  optical_shell = {charts.optical_shell}")
        assert not sdk.StateOps.is_in_omega(off)
        assert charts.omega12 is None
        assert charts.chirality_weight6 is None
        assert charts.optical_shell is None
        assert charts.optical_eq is None
        assert charts.optical_comp is None
        assert charts.optical_mu is None

    def test_witness_depth_histogram(self, witness_depth_histogram: dict[int, int]) -> None:
        assert witness_depth_histogram == {0: 1, 1: 127, 2: 3968}


# ---------------------------------------------------------------------
# Ω-step / Ω-signature / Ω-gate consistency
# ---------------------------------------------------------------------


class TestOmegaStepAndSignatures:
    @pytest.mark.parametrize("word", [
        bytes([]),
        bytes([0xAA]),
        bytes([0x00]),
        bytes([0xD5]),
        bytes([0x2B, 0x54]),
        bytes([0x7E, 0x81, 0x00]),
        bytes([0x11, 0x22, 0x33, 0x44]),
    ])
    def test_omega_signature_matches_carrier_signature(self, word: bytes) -> None:
        wsig = sdk.StateOps.omega_signature_from_word_signature(sdk.word_signature(word))
        osig = sdk.StateOps.omega_signature(word)
        print(f"\nword={list(word)}")
        print(f"  omega_signature_from_word_signature = {wsig}")
        print(f"  omega_signature(word)               = {osig}")
        assert wsig == osig

    def test_omega_step_matches_carrier_step(self, omega_states: tuple[int, ...]) -> None:
        sample_states = _sample_states(omega_states, 64, seed=7)
        sample_bytes = list(range(0, 256, 17)) + [0xAA, 0xD5, 0x2B, 0x54]
        sample_bytes = sorted(set(b & 0xFF for b in sample_bytes))

        checked = 0
        for s in sample_states:
            omega = sdk.StateOps.to_omega12(s)
            for b in sample_bytes:
                carrier_next = step_state_by_byte(s, b)
                omega_next = sdk.StateOps.step_omega12(omega, b)
                rebuilt = sdk.StateOps.from_omega12(omega_next)
                assert rebuilt == carrier_next
                checked += 1

        print(f"\nΩ-step vs carrier-step matches on {checked} sampled (state, byte) pairs")

    def test_exhaustive_omega_step_equivalence(self, omega_states: tuple[int, ...]) -> None:
        for s in omega_states:
            omega = sdk.StateOps.to_omega12(s)
            for b in range(256):
                carrier_next = step_state_by_byte(s, b)
                omega_next = sdk.StateOps.step_omega12(omega, b)
                rebuilt = sdk.StateOps.from_omega12(omega_next)
                assert rebuilt == carrier_next

    def test_apply_omega_signature_matches_carrier_word_action(self, omega_states: tuple[int, ...]) -> None:
        words = [
            bytes([0xAA]),
            bytes([0x00, 0x7F]),
            bytes([0xD5, 0x2B, 0x54]),
            bytes([0x11, 0x22, 0x33, 0x44]),
        ]
        sample_states = _sample_states(omega_states, 32, seed=23)

        for word in words:
            osig = sdk.StateOps.omega_signature(word)
            wsig = sdk.word_signature(word)
            print(f"\nword={list(word)}")
            print(f"  carrier signature = {wsig}")
            print(f"  omega signature   = {osig}")
            for s in sample_states:
                omega = sdk.StateOps.to_omega12(s)
                carrier_out = sdk.apply_word_signature(s, wsig)
                omega_out = sdk.StateOps.from_omega12(
                    sdk.StateOps.apply_omega_signature(omega, osig)
                )
                assert omega_out == carrier_out

    def test_omega_signature_composition_matches_concatenation_and_is_associative(
        self,
    ) -> None:
        rng = random.Random(42)
        for _ in range(64):
            w1 = bytes(rng.choices(range(256), k=rng.randint(0, 8)))
            w2 = bytes(rng.choices(range(256), k=rng.randint(0, 8)))
            w3 = bytes(rng.choices(range(256), k=rng.randint(0, 8)))

            sig1 = sdk.StateOps.omega_signature(w1)
            sig2 = sdk.StateOps.omega_signature(w2)
            sig3 = sdk.StateOps.omega_signature(w3)

            sig12 = sdk.StateOps.compose_omega_signatures(sig2, sig1)
            sig123 = sdk.StateOps.compose_omega_signatures(sig3, sig12)
            sig23 = sdk.StateOps.compose_omega_signatures(sig3, sig2)
            sig123_alt = sdk.StateOps.compose_omega_signatures(sig23, sig1)

            assert sdk.StateOps.omega_signature(w1 + w2) == sig12
            assert sdk.StateOps.omega_signature(w1 + w2 + w3) == sig123
            assert sig123 == sig123_alt

    def test_apply_omega_signature_composed_equals_repeated_application(
        self, omega_states: tuple[int, ...]
    ) -> None:
        words = [
            bytes([0xAA, 0x7E]),
            bytes([0x11, 0x22, 0x33]),
            bytes([0xD5, 0x2B, 0x54, 0x81, 0x00]),
        ]
        sample = _sample_states(omega_states, 16, seed=7)
        for word in words:
            composed = sdk.StateOps.omega_signature(word)
            for s in sample:
                omega = sdk.StateOps.to_omega12(s)
                out_composed = sdk.StateOps.apply_omega_signature(omega, composed)
                out_repeated = omega
                for b in word:
                    out_repeated = sdk.StateOps.step_omega12(out_repeated, b)
                assert out_composed == out_repeated

    def test_k4_gate_actions_match_carrier_gates(self, omega_states: tuple[int, ...]) -> None:
        sample_states = _sample_states(omega_states, 64, seed=99)
        gate_names = ("id", "S", "C", "F")

        for gate in gate_names:
            print(f"\nchecking Ω-gate consistency for gate {gate}")
            for s in sample_states:
                omega = sdk.StateOps.to_omega12(s)
                carrier_out = sdk.StateOps.gate(s, gate)
                omega_out = sdk.StateOps.from_omega12(
                    sdk.StateOps.omega_gate(omega, gate)
                )
                assert omega_out == carrier_out

    def test_packed_omega_helpers(self, omega_states: tuple[int, ...]) -> None:
        s = omega_states[123]
        omega = sdk.StateOps.to_omega12(s)
        packed = sdk.StateOps.pack_omega12(omega)
        unpacked = sdk.StateOps.unpack_omega12(packed)

        word = bytes([0xD5, 0x2B, 0x54])
        sig = sdk.StateOps.omega_signature(word)
        packed_sig = sdk.StateOps.pack_omega_signature12(sig)
        unpacked_sig = sdk.StateOps.unpack_omega_signature12(packed_sig)

        print("\npacked Ω helpers:")
        print(f"  state24      = 0x{s:06x}")
        print(f"  omega        = {omega}")
        print(f"  packed omega = 0x{packed:03x}")
        print(f"  packed sig   = 0x{packed_sig:04x}")

        assert unpacked == omega
        assert unpacked_sig == sig


# ---------------------------------------------------------------------
# Shell algebra / optical layer
# ---------------------------------------------------------------------


class TestShellAndOpticalLayer:
    def test_shell_populations(self, omega_states: tuple[int, ...]) -> None:
        counts = [0] * 7
        for s in omega_states:
            w = sdk.MomentOps.locus_of_state(s)
            counts[w] += 1

        _print_shell_table("Ω shell populations", counts, total=len(omega_states))

        expected = [sdk.MomentOps.shell_population(w) for w in range(7)]
        assert counts == expected

    def test_states_on_locus_counts_and_horizon_identification(self) -> None:
        counts = []
        for w in range(7):
            states = sdk.MomentOps.states_on_locus(w)
            counts.append(len(states))
            print(f"\nlocus {w}: {len(states)} states")

        assert counts == [sdk.MomentOps.shell_population(w) for w in range(7)]
        assert counts[0] == 64
        assert counts[6] == 64

        eq_states = sdk.MomentOps.states_on_locus(0)
        comp_states = sdk.MomentOps.states_on_locus(6)

        assert all(sdk.StateOps.charts(s).constitutional.on_equality_horizon for s in eq_states)
        assert all(sdk.StateOps.charts(s).constitutional.on_complement_horizon for s in comp_states)

    def test_optical_coordinates_match_distance_observables(self, omega_states: tuple[int, ...]) -> None:
        sample_states = _sample_states(omega_states, 32, seed=314)
        for s in sample_states:
            charts = sdk.StateOps.charts(s)
            w = charts.optical_shell
            assert w is not None
            assert charts.chirality_weight6 == w

            eq, comp, mu = sdk.MomentOps.optical_coordinates(s)
            assert eq == Fraction(w, 6)
            assert comp == Fraction(6 - w, 6)
            assert mu == Fraction(2 * w - 6, 6)

            assert charts.constitutional.ab_distance == 2 * w
            assert charts.constitutional.horizon_distance == 2 * (6 - w)

        example = sample_states[0]
        charts = sdk.StateOps.charts(example)
        print("\nexample optical coordinates:")
        print(f"  state24 = 0x{example:06x}")
        print(f"  shell   = {charts.optical_shell}")
        print(f"  eq      = {charts.optical_eq}")
        print(f"  comp    = {charts.optical_comp}")
        print(f"  mu      = {charts.optical_mu}")

    def test_future_locus_measure(self, omega_states: tuple[int, ...]) -> None:
        rest_dist_0 = sdk.MomentOps.future_locus_measure(GENE_MAC_REST, 0)
        rest_dist_1 = sdk.MomentOps.future_locus_measure(GENE_MAC_REST, 1)
        rest_dist_2 = sdk.MomentOps.future_locus_measure(GENE_MAC_REST, 2)

        print("\nfuture locus measure at rest:")
        print(f"  n=0 -> {rest_dist_0}")
        print(f"  n=1 -> {rest_dist_1}")
        print(f"  n=2 -> {rest_dist_2}")

        assert rest_dist_0[6] == 1
        assert sum(rest_dist_0.values(), Fraction(0, 1)) == 1

        expected = {w: FULL_BYTE_SHELL_DISTRIBUTION[w] for w in range(7)}
        assert rest_dist_1 == expected
        assert rest_dist_2 == expected

        sample = omega_states[777]
        sample_dist = sdk.MomentOps.future_locus_measure(sample, 5)
        assert sample_dist == expected

    def test_shell_transition_kernels_are_stochastic(self) -> None:
        for j in range(7):
            mat = sdk.MomentOps.shell_transition_matrix(j)
            print(f"\nq-weight {j} shell transition row sums:")
            for w in range(7):
                row_sum = sum(mat[w], Fraction(0, 1))
                print(f"  source shell {w}: {row_sum}")
                assert row_sum == 1

    def test_full_byte_average_shell_law_is_binomial_and_source_independent(self) -> None:
        q_weight_distribution = [Fraction(sdk.MomentOps.shell_population(j), 4096) * 64 for j in range(7)]
        # simpler exact distribution of q-weights over q ∈ GF(2)^6:
        q_weight_distribution = [Fraction(int(__import__("math").comb(6, j)), 64) for j in range(7)]

        expected = tuple(FULL_BYTE_SHELL_DISTRIBUTION)

        for w in range(7):
            averaged = []
            for wp in range(7):
                p = sum(
                    q_weight_distribution[j] * sdk.MomentOps.shell_transition_probability(w, j, wp)
                    for j in range(7)
                )
                averaged.append(p)
            averaged = tuple(averaged)
            print(f"\nsource shell {w} averaged one-byte shell law:")
            print(f"  {averaged}")
            assert averaged == expected

    def test_exact_shell_transition_formula(self) -> None:
        for j in range(7):
            for w in range(7):
                for wp in range(7):
                    impl = sdk.MomentOps.shell_transition_probability(w, j, wp)
                    delta = w + j - wp
                    if delta < 0 or (delta & 1):
                        expected = Fraction(0, 1)
                    else:
                        t = delta // 2
                        if t < 0 or t > min(w, j) or (j - t) < 0 or (j - t) > (6 - w):
                            expected = Fraction(0, 1)
                        else:
                            expected = Fraction(
                                comb(w, t) * comb(6 - w, j - t), comb(6, j)
                            )
                    assert impl == expected

    def test_one_byte_shell_law_source_independent_direct(
        self, omega_states: tuple[int, ...]
    ) -> None:
        binomial = tuple(FULL_BYTE_SHELL_DISTRIBUTION)
        sources = [
            GENE_MAC_REST,
            omega_states[0],
            omega_states[127],
            omega_states[2048],
            omega_states[4095],
        ]
        for s in sources:
            dist = sdk.MomentOps.future_locus_measure(s, 1)
            assert dist == {w: binomial[w] for w in range(7)}

    def test_exact_horizon_characterization_in_omega_coordinates(
        self, omega_states: tuple[int, ...]
    ) -> None:
        for s in omega_states:
            omega = sdk.StateOps.to_omega12(s)
            u6, v6 = omega.u6, omega.v6
            chi = omega.chirality6
            assert (u6 == v6) == (chi == 0)
            assert omega.is_on_equality_horizon == (u6 == v6)
            assert omega.is_on_complement_horizon == (chi == 63)
            assert (u6 ^ v6 == 0x3F) == (chi == 63)

    def test_q_weight_shell_kernel_independence(self) -> None:
        for j in range(7):
            bytes_j = [b for b in range(256) if Q_WEIGHT_BY_BYTE[b] == j]
            if len(bytes_j) < 2:
                continue
            b1, b2 = bytes_j[0], bytes_j[1]
            mat = sdk.MomentOps.shell_transition_matrix(j)
            for w in range(7):
                for wp in range(7):
                    p1 = shell_transition_probability(
                        w, Q_WEIGHT_BY_BYTE[b1], wp
                    )
                    p2 = shell_transition_probability(
                        w, Q_WEIGHT_BY_BYTE[b2], wp
                    )
                    assert p1 == p2 == mat[w][wp]


# ---------------------------------------------------------------------
# Krawtchouk shell spectral layer
# ---------------------------------------------------------------------


class TestKrawtchoukShellSpectralLayer:
    def test_constant_function_transform(self) -> None:
        f = [1] * 7
        coeffs = sdk.SpectralOps.shell_krawtchouk_transform_exact(f)
        print("\nKrawtchouk transform of constant shell function:")
        print(f"  coeffs = {coeffs}")
        assert coeffs[0] == 1
        assert all(c == 0 for c in coeffs[1:])

    def test_inverse_roundtrip_on_basis_vectors(self) -> None:
        for w0 in range(7):
            basis = [0] * 7
            basis[w0] = 1
            coeffs = sdk.SpectralOps.shell_krawtchouk_transform_exact(basis)
            rebuilt = sdk.SpectralOps.shell_krawtchouk_inverse_exact(coeffs)
            assert tuple(rebuilt) == tuple(Fraction(x, 1) for x in basis)

    def test_inverse_roundtrip_on_nontrivial_shell_function(self) -> None:
        f = [3, -1, 2, 0, 5, 7, -2]
        coeffs = sdk.SpectralOps.shell_krawtchouk_transform_exact(f)
        rebuilt = sdk.SpectralOps.shell_krawtchouk_inverse_exact(coeffs)

        print("\nKrawtchouk roundtrip:")
        print(f"  input   = {tuple(Fraction(x, 1) for x in f)}")
        print(f"  coeffs  = {coeffs}")
        print(f"  rebuilt = {rebuilt}")

        assert tuple(rebuilt) == tuple(Fraction(x, 1) for x in f)

    def test_float_transform_matches_exact_transform(self) -> None:
        f = [0.25, -1.5, 2.0, 0.0, 3.25, 1.5, -0.75]
        f_frac = [Fraction(x) for x in f]
        coeffs_exact = sdk.SpectralOps.shell_krawtchouk_transform_exact(f_frac)
        coeffs_float = shell_krawtchouk_transform_float(f)

        for a, b in zip(coeffs_exact, coeffs_float):
            assert abs(float(a) - float(b)) < 1e-12

    def test_krawtchouk_diagonalizes_shell_transition_matrices(self) -> None:
        for j in range(7):
            T = sdk.MomentOps.shell_transition_matrix(j)
            K = [[KRAWTCHOUK_7[w][k] for k in range(7)] for w in range(7)]
            for k in range(7):
                col = [K[w][k] for w in range(7)]
                T_col = [
                    sum(float(T[w][wp]) * col[wp] for wp in range(7))
                    for w in range(7)
                ]
                if col[0] != 0:
                    lam = T_col[0] / col[0]
                else:
                    idx = next(i for i in range(7) if col[i] != 0)
                    lam = T_col[idx] / col[idx]
                for w in range(7):
                    assert abs(T_col[w] - lam * col[w]) < 1e-12

    def test_krawtchouk_parseval_orthogonality(self) -> None:
        for j in range(7):
            for k in range(7):
                inner = sum(
                    comb(6, w) * KRAWTCHOUK_7[w][j] * KRAWTCHOUK_7[w][k]
                    for w in range(7)
                )
                expected = 64 * comb(6, j) if j == k else 0
                assert inner == expected


# ---------------------------------------------------------------------
# Moment-level Ω / shell information
# ---------------------------------------------------------------------


class TestMomentsCarryOmegaData:
    def test_moment_contains_omega_signature(self) -> None:
        word = bytes([0xAA, 0x2B, 0xD5, 0x54])
        moment = sdk.MomentOps.make(word)

        print("\nmoment with omega signature:")
        print(f"  ledger          = {list(word)}")
        print(f"  state24         = 0x{moment.state24:06x}")
        print(f"  carrier sig     = {moment.signature}")
        print(f"  omega signature = {moment.omega_signature}")
        print(f"  q transport     = 0x{moment.q_transport6:02x}")

        assert moment.omega_signature == sdk.StateOps.omega_signature(word)
        assert moment.charts.omega12 is not None

    def test_compare_ledgers_prefix(self) -> None:
        left = bytes([0xAA, 0x11, 0x22, 0x33])
        right = bytes([0xAA, 0x11, 0x99, 0x88])

        cmp = sdk.MomentOps.compare(left, right)
        print("\nledger comparison:")
        print(f"  common_prefix_len    = {cmp.common_prefix_len}")
        print(f"  common_prefix_state  = 0x{cmp.common_prefix_state24:06x}")
        print(f"  first_divergence     = {cmp.first_divergence_index}")
        print(f"  left_next_byte       = {cmp.left_next_byte}")
        print(f"  right_next_byte      = {cmp.right_next_byte}")

        assert cmp.common_prefix_len == 2
        assert cmp.first_divergence_index == 2
        assert cmp.left_next_byte == 0x22
        assert cmp.right_next_byte == 0x99


# ---------------------------------------------------------------------
# Native runtime Ω / shell surfaces
# ---------------------------------------------------------------------


@pytest.mark.skipif(not torch, reason="torch not available")
class TestRuntimeNativeOmegaAndShell:
    def test_runtime_state24_to_omega12_and_back(self, omega_states: tuple[int, ...], native_ready: bool) -> None:
        states = _mixed_state_batch(omega_states)
        omega_packed, valid = sdk.RuntimeOps.omega12_from_states(states)
        rebuilt = sdk.RuntimeOps.states_from_omega12(omega_packed)

        print("\nruntime state24 -> omega12 -> state24:")
        for i in range(states.numel()):
            print(
                f"  state=0x{int(states[i]):06x} "
                f"valid={int(valid[i])} "
                f"omega=0x{int(omega_packed[i]):03x} "
                f"rebuilt=0x{int(rebuilt[i]):06x}"
            )

        for i in range(states.numel()):
            s = int(states[i])
            is_omega = sdk.StateOps.is_in_omega(s)
            assert int(valid[i]) == int(is_omega)
            if is_omega:
                assert int(rebuilt[i]) == s

    def test_runtime_step_omega12_batch_matches_python(self, omega_states: tuple[int, ...], native_ready: bool) -> None:
        sample = torch.tensor(
            [sdk.StateOps.pack_omega12(sdk.StateOps.to_omega12(s)) for s in omega_states[:16]],
            dtype=torch.int32,
            device="cpu",
        )
        byte = 0x7E

        native_out = sdk.RuntimeOps.step_omega12_batch(sample, byte)
        py_out = []
        for x in sample:
            om = sdk.StateOps.unpack_omega12(int(x))
            py_out.append(sdk.StateOps.pack_omega12(sdk.StateOps.step_omega12(om, byte)))
        py_out_t = torch.tensor(py_out, dtype=torch.int32, device="cpu")

        print(f"\nruntime Ω step batch byte=0x{byte:02x}")
        print(f"  native = {native_out.tolist()[:8]}")
        print(f"  python = {py_out_t.tolist()[:8]}")

        assert torch.equal(native_out, py_out_t)

    def test_runtime_apply_omega_signature_batch_matches_python(self, omega_states: tuple[int, ...], native_ready: bool) -> None:
        packed_states = torch.tensor(
            [sdk.StateOps.pack_omega12(sdk.StateOps.to_omega12(s)) for s in omega_states[:8]],
            dtype=torch.int32,
            device="cpu",
        )

        words = [
            bytes([0xAA]),
            bytes([0x11, 0x22]),
            bytes([0xD5, 0x2B, 0x54]),
            bytes([0x7E, 0x81, 0x00, 0x44]),
            bytes([0x54]),
            bytes([0xAA, 0xAA]),
            bytes([0x01, 0x02, 0x03]),
            bytes([0xFE, 0xEF]),
        ]
        packed_sigs = torch.tensor(
            [sdk.StateOps.pack_omega_signature12(sdk.StateOps.omega_signature(w)) for w in words],
            dtype=torch.int32,
            device="cpu",
        )

        native_out = sdk.RuntimeOps.apply_omega_signature_batch(packed_states, packed_sigs)

        py_out = []
        for x, sig in zip(packed_states.tolist(), packed_sigs.tolist()):
            om = sdk.StateOps.unpack_omega12(x)
            sg = sdk.StateOps.unpack_omega_signature12(sig)
            py_out.append(
                sdk.StateOps.pack_omega12(sdk.StateOps.apply_omega_signature(om, sg))
            )
        py_out_t = torch.tensor(py_out, dtype=torch.int32, device="cpu")

        print("\nruntime Ω signature application:")
        print(f"  native = {native_out.tolist()}")
        print(f"  python = {py_out_t.tolist()}")

        assert torch.equal(native_out, py_out_t)

    def test_runtime_shell_histograms(self, omega_states: tuple[int, ...], native_ready: bool) -> None:
        packed_states = torch.tensor(list(omega_states[:256]), dtype=torch.int32, device="cpu")
        packed_omega = torch.tensor(
            [sdk.StateOps.pack_omega12(sdk.StateOps.to_omega12(s)) for s in omega_states[:256]],
            dtype=torch.int32,
            device="cpu",
        )

        hist_state = sdk.RuntimeOps.shell_histogram_state24(packed_states)
        hist_omega = sdk.RuntimeOps.shell_histogram_omega12(packed_omega)
        hist_checked, invalid = sdk.RuntimeOps.shell_histogram_state24_checked(packed_states)

        print("\nruntime shell histograms:")
        print(f"  from state24 unchecked = {hist_state.tolist()}")
        print(f"  from omega12           = {hist_omega.tolist()}")
        print(f"  from state24 checked   = {hist_checked.tolist()}, invalid={invalid}")

        assert torch.equal(hist_omega, hist_checked)
        assert invalid == 0

    def test_runtime_omega_signature_scan(self, native_ready: bool) -> None:
        word = _bytes_tensor([0xAA, 0x7E, 0x81, 0x00, 0x54, 0x2B])
        packed_sigs = sdk.RuntimeOps.omega_signature_scan(word)

        py = []
        acc = sdk.OmegaSignature12(parity=0, tau_u6=0, tau_v6=0)
        for b in word.tolist():
            sig_b = sdk.OmegaSignature12(
                parity=1,
                tau_u6=EPS_A6_BY_BYTE[b],
                tau_v6=MICRO_REF_BY_BYTE[b] ^ EPS_B6_BY_BYTE[b],
            )
            acc = sdk.StateOps.compose_omega_signatures(sig_b, acc)
            py.append(sdk.StateOps.pack_omega_signature12(acc))

        py_t = torch.tensor(py, dtype=torch.int32, device="cpu")

        print("\nruntime omega signature scan:")
        print(f"  native = {packed_sigs.tolist()}")
        print(f"  python = {py_t.tolist()}")

        assert torch.equal(packed_sigs, py_t)

    def test_runtime_omega12_scan_from_omega12(self, native_ready: bool) -> None:
        payload = _bytes_tensor([0x11, 0x22, 0x33, 0x44, 0x55])
        start = sdk.StateOps.pack_omega12(sdk.StateOps.to_omega12(GENE_MAC_REST))
        native_scan = sdk.RuntimeOps.omega12_scan_from_omega12(payload, start)

        py = []
        om = sdk.StateOps.unpack_omega12(start)
        for b in payload.tolist():
            om = sdk.StateOps.step_omega12(om, b)
            py.append(sdk.StateOps.pack_omega12(om))
        py_t = torch.tensor(py, dtype=torch.int32, device="cpu")

        print("\nruntime omega12 continuation scan:")
        print(f"  native = {native_scan.tolist()}")
        print(f"  python = {py_t.tolist()}")

        assert torch.equal(native_scan, py_t)

    def test_runtime_apply_omega_gate_batch_numeric_and_named(self, omega_states: tuple[int, ...], native_ready: bool) -> None:
        packed_omega = torch.tensor(
            [sdk.StateOps.pack_omega12(sdk.StateOps.to_omega12(s)) for s in omega_states[:16]],
            dtype=torch.int32,
            device="cpu",
        )
        gate_pairs = [(0, "id"), (1, "S"), (2, "C"), (3, "F")]

        for code, name in gate_pairs:
            out_num = sdk.RuntimeOps.apply_omega_gate_batch(packed_omega, code)
            out_named = sdk.RuntimeOps.apply_omega_gate_batch_named(packed_omega, name)

            print(f"\nruntime Ω gate batch {name} / code {code}:")
            print(f"  numeric = {out_num.tolist()[:8]}")
            print(f"  named   = {out_named.tolist()[:8]}")

            assert torch.equal(out_num, out_named)