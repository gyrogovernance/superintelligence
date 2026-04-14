from __future__ import annotations

"""
High-level Quantum SDK surface for the Gyroscopic ASI aQPU.

This file keeps the public exposure compact and centered on:
- Moments
- Gyrostates and charts
- Future-cone structure
- Exact structural derivatives
- State synthesis from rest
- Native aQPU runtime entry points (RuntimeOps over tools.gyroscopic.ops)

It is intentionally thin over src.constants and src.api.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import log2
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from src.tools.gyroscopic.ops import GyroGraphWordSignature

from src.api import (
    ByteItem,
    FULL_BYTE_SHELL_DISTRIBUTION,
    OmegaSignature12,
    OmegaState12,
    WordSignature,
    apply_omega_gate,
    pack_omega12,
    pack_omega_signature12,
    unpack_omega12,
    unpack_omega_signature12,
    apply_omega_gate_C,
    apply_omega_gate_F,
    apply_omega_gate_S,
    apply_omega_signature,
    apply_word_signature,
    chirality_word6,
    compose_omega_signatures,
    depth4_intron_sequence32,
    depth4_mask_projection48,
    fixed_locus,
    fixed_states_of_gate,
    frobenius_component12,
    frobenius_pair,
    gf4_norm,
    gf4_trace,
    is_in_omega24,
    is_reachable_component,
    is_trace1_pair,
    k4_orbit,
    k4_stabilizer,
    omega12_to_state24,
    omega_signature_from_word_signature,
    omega_word_signature,
    optical_coordinates_from_state24,
    q_word6,
    q_word6_for_items,
    shadow_partner_byte,
    shadow_partner_map,
    shell_krawtchouk_inverse_exact,
    shell_krawtchouk_transform_exact,
    shell_markov_step,
    shell_population,
    shell_transition_matrix_for_q_weight,
    shell_transition_probability,
    stabilizer_type_from_state24,
    state24_to_omega12,
    state24_to_spin6_pair,
    state_conjugate_f,
    step_omega12_by_byte,
    trajectory_parity_commitment,
    try_state24_to_omega12,
    verify_optical_conjugacy,
    walsh_hadamard64,
    word_signature,
)
from src.constants import (
    GENE_MAC_REST,
    GENE_MIC_S,
    MASK_STATE24,
    ab_distance,
    archetype_distance,
    apply_gate,
    component_density,
    horizon_distance,
    is_on_equality_horizon,
    is_on_horizon,
    pack_state,
    step_state_by_byte,
    unpack_state,
)


def byte_transition(state24: int, byte: int) -> int:
    """SDK 5.1.1: single-byte transition on state24 (exact Python kernel)."""
    return step_state_by_byte(int(state24) & MASK_STATE24, int(byte) & 0xFF)


def compare_ledgers_pair(
    a: bytes,
    b: bytes,
) -> tuple[int, int]:
    """
    SDK 7.3 / native parity: return (cmp, common_prefix_len) matching
    gyrograph_compare_ledgers (0 equal, -1 a prefix, +1 diverge).
    """
    from src.tools.gyroscopic.ops import gyrograph_compare_ledgers_native

    return gyrograph_compare_ledgers_native(a, b)

ObservableInt: TypeAlias = Callable[[int], int]
ObservableNum: TypeAlias = Callable[[int], int | float]


def _flatten_items(items: Iterable[ByteItem]) -> bytes:
    out = bytearray()
    for item in items:
        if isinstance(item, (bytes, bytearray, memoryview)):
            out.extend(int(b) & 0xFF for b in item)
        else:
            out.append(int(item) & 0xFF)
    return bytes(out)


@dataclass(frozen=True)
class ConstitutionalChart:
    rest_distance: int
    horizon_distance: int
    ab_distance: int
    on_complement_horizon: bool
    on_equality_horizon: bool
    a_density: float
    b_density: float
    complementarity_sum: int


@dataclass(frozen=True)
class StateCharts:
    state24: int
    state_hex: str
    a12: int
    b12: int
    a_hex: str
    b_hex: str
    spin_a6: tuple[int, ...] | None
    spin_b6: tuple[int, ...] | None
    chirality6: int
    constitutional: ConstitutionalChart
    omega12: OmegaState12 | None
    chirality_weight6: int | None
    optical_shell: int | None
    optical_eq: Fraction | None
    optical_comp: Fraction | None
    optical_mu: Fraction | None


@dataclass(frozen=True)
class Moment:
    step: int
    state24: int
    last_byte: int
    ledger: bytes
    state_hex: str
    a_hex: str
    b_hex: str
    charts: StateCharts
    signature: WordSignature
    omega_signature: OmegaSignature12
    parity_commitment: tuple[int, int, int]
    q_transport6: int


@dataclass(frozen=True)
class MomentComparison:
    same_ledger: bool
    same_moment: bool
    common_prefix_len: int
    common_prefix_state24: int
    first_divergence_index: int | None
    left_next_byte: int | None
    right_next_byte: int | None
    left_final_state24: int
    right_final_state24: int


@dataclass(frozen=True)
class FutureConeMeasure:
    source_state24: int
    length: int
    total_words: int
    state_counts: tuple[tuple[int, int], ...]
    entropy_bits: float
    exact_uniform: bool

    @property
    def distinct_states(self) -> int:
        return len(self.state_counts)

    def as_dict(self) -> dict[int, int]:
        return dict(self.state_counts)

    def probability_of(self, state24: int) -> Fraction:
        count = dict(self.state_counts).get(int(state24) & MASK_STATE24, 0)
        return Fraction(count, self.total_words)


@dataclass(frozen=True)
class ReachabilityWitness:
    target_state24: int
    word: bytes
    depth: int
    signature: WordSignature
    omega_signature: OmegaSignature12


def state_charts(state24: int) -> StateCharts:
    s = int(state24) & MASK_STATE24
    a12, b12 = unpack_state(s)

    try:
        spin_a6, spin_b6 = state24_to_spin6_pair(s)
    except ValueError:
        spin_a6 = None
        spin_b6 = None

    rest_d = archetype_distance(s)
    horiz_d = horizon_distance(a12, b12)
    ab_d = ab_distance(a12, b12)

    constitutional = ConstitutionalChart(
        rest_distance=rest_d,
        horizon_distance=horiz_d,
        ab_distance=ab_d,
        on_complement_horizon=is_on_horizon(s),
        on_equality_horizon=is_on_equality_horizon(s),
        a_density=component_density(a12),
        b_density=component_density(b12),
        complementarity_sum=horiz_d + ab_d,
    )

    omega = try_state24_to_omega12(s)
    if omega is None:
        chirality_weight6 = None
        optical_shell = None
        optical_eq = None
        optical_comp = None
        optical_mu = None
    else:
        chirality_weight6 = omega.shell
        optical_shell = omega.shell
        optical_eq = omega.optical_eq
        optical_comp = omega.optical_comp
        optical_mu = omega.optical_mu

    return StateCharts(
        state24=s,
        state_hex=f"{s:06x}",
        a12=a12,
        b12=b12,
        a_hex=f"{a12:03x}",
        b_hex=f"{b12:03x}",
        spin_a6=spin_a6,
        spin_b6=spin_b6,
        chirality6=chirality_word6(s),
        constitutional=constitutional,
        omega12=omega,
        chirality_weight6=chirality_weight6,
        optical_shell=optical_shell,
        optical_eq=optical_eq,
        optical_comp=optical_comp,
        optical_mu=optical_mu,
    )


def moment_from_ledger(
    items: Iterable[ByteItem],
    start_state24: int = GENE_MAC_REST,
) -> Moment:
    ledger = _flatten_items(items)
    s = int(start_state24) & MASK_STATE24

    for b in ledger:
        s = step_state_by_byte(s, b)

    last_byte = GENE_MIC_S if len(ledger) == 0 else int(ledger[-1]) & 0xFF
    charts = state_charts(s)
    sig = word_signature(ledger)
    omega_sig = omega_signature_from_word_signature(sig)
    parity = trajectory_parity_commitment(ledger)
    q_transport = q_word6_for_items(ledger)

    return Moment(
        step=len(ledger),
        state24=s,
        last_byte=last_byte,
        ledger=ledger,
        state_hex=charts.state_hex,
        a_hex=charts.a_hex,
        b_hex=charts.b_hex,
        charts=charts,
        signature=sig,
        omega_signature=omega_sig,
        parity_commitment=parity,
        q_transport6=q_transport,
    )


def verify_moment(
    moment: Moment,
    items: Iterable[ByteItem] | None = None,
    start_state24: int = GENE_MAC_REST,
) -> bool:
    ledger = moment.ledger if items is None else _flatten_items(items)
    rebuilt = moment_from_ledger(ledger, start_state24=start_state24)
    return (
        rebuilt.step == moment.step
        and rebuilt.state24 == moment.state24
        and rebuilt.last_byte == moment.last_byte
        and rebuilt.signature == moment.signature
        and rebuilt.omega_signature == moment.omega_signature
        and rebuilt.parity_commitment == moment.parity_commitment
        and rebuilt.q_transport6 == moment.q_transport6
    )


def compare_ledgers(
    left: Iterable[ByteItem],
    right: Iterable[ByteItem],
    start_state24: int = GENE_MAC_REST,
) -> MomentComparison:
    lbytes = _flatten_items(left)
    rbytes = _flatten_items(right)

    common = 0
    upper = min(len(lbytes), len(rbytes))
    while common < upper and lbytes[common] == rbytes[common]:
        common += 1

    prefix_state = int(start_state24) & MASK_STATE24
    for b in lbytes[:common]:
        prefix_state = step_state_by_byte(prefix_state, b)

    left_m = moment_from_ledger(lbytes, start_state24=start_state24)
    right_m = moment_from_ledger(rbytes, start_state24=start_state24)

    return MomentComparison(
        same_ledger=lbytes == rbytes,
        same_moment=(left_m.state24 == right_m.state24 and left_m.step == right_m.step),
        common_prefix_len=common,
        common_prefix_state24=prefix_state,
        first_divergence_index=None if lbytes == rbytes else common,
        left_next_byte=None if common >= len(lbytes) else lbytes[common],
        right_next_byte=None if common >= len(rbytes) else rbytes[common],
        left_final_state24=left_m.state24,
        right_final_state24=right_m.state24,
    )


@lru_cache(maxsize=256)
def _future_cone_counts_cached(
    source_state24: int,
    length: int,
) -> tuple[tuple[int, int], ...]:
    s0 = int(source_state24) & MASK_STATE24
    n = int(length)

    if n < 0:
        raise ValueError(f"length must be non-negative, got {length}")

    if n == 0:
        return ((s0, 1),)

    if _is_in_omega(s0):
        if n == 1:
            counts: dict[int, int] = {}
            for b in range(256):
                out = step_state_by_byte(s0, b)
                counts[out] = counts.get(out, 0) + 1
            return tuple(sorted(counts.items()))

        per_state = (256**n) // 4096
        return tuple((state24, per_state) for state24 in _omega_states())

    current: dict[int, int] = {s0: 1}
    for _ in range(n):
        nxt: dict[int, int] = {}
        for state24, count in current.items():
            for b in range(256):
                out = step_state_by_byte(state24, b)
                nxt[out] = nxt.get(out, 0) + count
        current = nxt

    return tuple(sorted(current.items()))


def future_cone_measure(
    source_state24: int,
    length: int,
) -> FutureConeMeasure:
    items = _future_cone_counts_cached(int(source_state24) & MASK_STATE24, int(length))
    total_words = 256 ** int(length)

    if total_words == 0:
        entropy = 0.0
    else:
        entropy = 0.0
        for _, count in items:
            p = count / total_words
            entropy -= p * log2(p)

    exact_uniform = True
    if len(items) > 1:
        first = items[0][1]
        exact_uniform = all(count == first for _, count in items)

    return FutureConeMeasure(
        source_state24=int(source_state24) & MASK_STATE24,
        length=int(length),
        total_words=total_words,
        state_counts=items,
        entropy_bits=entropy,
        exact_uniform=exact_uniform,
    )


def future_entropy_bits(source_state24: int, length: int) -> float:
    return future_cone_measure(source_state24, length).entropy_bits


def future_expectation_exact(
    source_state24: int,
    length: int,
    observable: ObservableInt,
) -> Fraction:
    measure = future_cone_measure(source_state24, length)
    acc = 0
    for state24, count in measure.state_counts:
        acc += int(observable(state24)) * count
    return Fraction(acc, measure.total_words)


def future_expectation_float(
    source_state24: int,
    length: int,
    observable: ObservableNum,
) -> float:
    measure = future_cone_measure(source_state24, length)
    acc = 0.0
    for state24, count in measure.state_counts:
        acc += float(observable(state24)) * count
    return acc / float(measure.total_words)


def directional_derivative(
    source_state24: int,
    byte: int,
    observable: ObservableNum,
) -> int | float:
    s = int(source_state24) & MASK_STATE24
    out = step_state_by_byte(s, int(byte) & 0xFF)
    lhs = observable(out)
    rhs = observable(s)
    return lhs - rhs


def byte_derivative_table(
    source_state24: int,
    observable: ObservableNum,
) -> tuple[int | float, ...]:
    s = int(source_state24) & MASK_STATE24
    return tuple(directional_derivative(s, b, observable) for b in range(256))


def exact_transport_table(source_state24: int) -> tuple[int, ...]:
    s = int(source_state24) & MASK_STATE24
    chi_in = chirality_word6(s)
    out: list[int] = []
    for b in range(256):
        chi_out = chirality_word6(step_state_by_byte(s, b))
        out.append(chi_in ^ chi_out)
    return tuple(out)


@lru_cache(maxsize=1)
def _rest_witness_index() -> dict[int, bytes]:
    index: dict[int, bytes] = {GENE_MAC_REST: b""}

    for b in range(256):
        s1 = step_state_by_byte(GENE_MAC_REST, b)
        index.setdefault(s1, bytes([b]))

    for b1 in range(256):
        s1 = step_state_by_byte(GENE_MAC_REST, b1)
        for b2 in range(256):
            s2 = step_state_by_byte(s1, b2)
            index.setdefault(s2, bytes([b1, b2]))

    return index


@lru_cache(maxsize=1)
def _omega_states() -> tuple[int, ...]:
    index = _rest_witness_index()
    return tuple(sorted(index.keys()))


def _is_in_omega(state24: int) -> bool:
    return is_in_omega24(state24)


def locus_of_state(state24: int) -> int:
    omega = state24_to_omega12(state24)
    return omega.shell


@lru_cache(maxsize=7)
def _states_on_locus_cached(w: int) -> tuple[int, ...]:
    if not (0 <= w <= 6):
        raise ValueError(f"Shell index must be 0..6, got {w}")

    out: list[int] = []
    chis = [chi for chi in range(64) if chi.bit_count() == w]
    for chi in chis:
        for u6 in range(64):
            out.append(omega12_to_state24(OmegaState12(u6=u6, v6=u6 ^ chi)))
    return tuple(sorted(out))


def states_on_locus(w: int) -> tuple[int, ...]:
    return _states_on_locus_cached(int(w))


def optical_coordinates(state24: int) -> tuple[Fraction, Fraction, Fraction]:
    return optical_coordinates_from_state24(state24)


def stabilizer_type(state24: int) -> str:
    return stabilizer_type_from_state24(state24)


def shell_histogram(states: Iterable[int]) -> tuple[int, ...]:
    counts = [0] * 7
    for s in states:
        omega = state24_to_omega12(s)
        counts[omega.shell] += 1
    return tuple(counts)


def future_locus_measure(
    source_state24: int,
    length: int,
) -> dict[int, Fraction]:
    s = int(source_state24) & MASK_STATE24
    n = int(length)

    if n < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    if not _is_in_omega(s):
        raise ValueError(
            f"future_locus_measure is defined exactly on Omega only; "
            f"state {s:#08x} is not in Omega"
        )

    if n == 0:
        w0 = state24_to_omega12(s).shell
        return {w: Fraction(int(w == w0), 1) for w in range(7)}

    return {w: FULL_BYTE_SHELL_DISTRIBUTION[w] for w in range(7)}


def witness_from_rest(target_state24: int) -> ReachabilityWitness:
    target = int(target_state24) & MASK_STATE24
    index = _rest_witness_index()
    if target not in index:
        raise ValueError(
            f"Target {target:#08x} is not in Omega or has no depth<=2 witness from rest"
        )
    word = index[target]
    sig = word_signature(word)
    return ReachabilityWitness(
        target_state24=target,
        word=word,
        depth=len(word),
        signature=sig,
        omega_signature=omega_signature_from_word_signature(sig),
    )


def execute_witness_from_rest(target_state24: int) -> Moment:
    witness = witness_from_rest(target_state24)
    return moment_from_ledger(witness.word, start_state24=GENE_MAC_REST)


def depth4_frame(
    b0: int,
    b1: int,
    b2: int,
    b3: int,
) -> dict[str, int]:
    return {
        "mask48": depth4_mask_projection48(b0, b1, b2, b3),
        "introns32": depth4_intron_sequence32(b0, b1, b2, b3),
        "q_transport6": q_word6_for_items((b0, b1, b2, b3)),
    }


def state_scan_from_state(
    payload: bytes | Iterable[int],
    start_state24: int,
) -> tuple[int, ...]:
    s = int(start_state24) & MASK_STATE24
    out: list[int] = []
    if isinstance(payload, (bytes, bytearray)):
        seq = bytes(payload)
    else:
        seq = bytes(int(x) & 0xFF for x in payload)
    for b in seq:
        s = step_state_by_byte(s, b) & MASK_STATE24
        out.append(s)
    return tuple(out)


def pack_matrix64(W, n_bits: int):
    import numpy as np

    from src.tools.gyroscopic.pack_matrix import pack_matrix64 as _pack

    return _pack(np.asarray(W), int(n_bits))


def apply_packed64_gemv(packed, x):
    import numpy as np

    from src.tools.gyroscopic.pack_matrix import apply_packed64_gemv as _gemv

    return _gemv(packed, np.asarray(x))


def dyadic_wht64_normalized(values: list[int]):
    from src.tools.gyroscopic.dyadic_wht import wht64_dyadic_normalized

    return wht64_dyadic_normalized(list(values))


def shell_order_parameters_from_hist(shell_hist: list[int] | tuple[int, ...]):
    """QuBEC climate order parameters rho, eta, m, Var(N) from shell_hist7 (docs Sections 3-5)."""
    from src.tools.gyroscopic.climate import shell_order_parameters_from_hist as _f

    return _f(shell_hist)


def m2_empirical_from_chi_hist(chi_hist: list[int] | tuple[int, ...]) -> float:
    """M2 effective support on the 64 chirality modes from chi_hist64 (native empirical estimator)."""
    from src.tools.gyroscopic.climate import m2_empirical_from_chi_hist as _f

    return _f(chi_hist)


def m2_equilibrium_from_shell_hist(shell_hist: list[int] | tuple[int, ...]) -> float:
    """Equilibrium M2 on Omega from shell_hist7: 4096 / (1 + eta^2)^6 (docs Section 5)."""
    from src.tools.gyroscopic.climate import m2_equilibrium_from_shell_hist as _f

    return _f(shell_hist)


def cell_climate_from_histograms(
    chi_hist64: list[int] | tuple[int, ...],
    shell_hist7: list[int] | tuple[int, ...],
    family_hist4: list[int] | tuple[int, ...],
    *,
    byte_ensemble_256: list[int] | tuple[int, ...] | None = None,
):
    """Per-cell climate observables: rho, eta, m, M2, shell/gauge spectra, optional byte anisotropy (Section 16)."""
    from src.tools.gyroscopic.climate import cell_climate_from_histograms as _f

    return _f(
        chi_hist64,
        shell_hist7,
        family_hist4,
        byte_ensemble_256=byte_ensemble_256,
    )


class TensorOps:
    pack_matrix64 = staticmethod(pack_matrix64)
    apply_packed64_gemv = staticmethod(apply_packed64_gemv)
    dyadic_wht64_normalized = staticmethod(dyadic_wht64_normalized)


class StateOps:
    charts = staticmethod(state_charts)
    pack = staticmethod(pack_state)
    unpack = staticmethod(unpack_state)
    gate = staticmethod(apply_gate)
    witness_from_rest = staticmethod(witness_from_rest)
    execute_witness_from_rest = staticmethod(execute_witness_from_rest)
    is_in_omega = staticmethod(is_in_omega24)
    try_to_omega12 = staticmethod(try_state24_to_omega12)
    to_omega12 = staticmethod(state24_to_omega12)
    from_omega12 = staticmethod(omega12_to_state24)
    step_omega12 = staticmethod(step_omega12_by_byte)
    omega_signature = staticmethod(omega_word_signature)
    omega_signature_from_word_signature = staticmethod(
        omega_signature_from_word_signature
    )
    apply_omega_signature = staticmethod(apply_omega_signature)
    compose_omega_signatures = staticmethod(compose_omega_signatures)
    omega_gate = staticmethod(apply_omega_gate)
    omega_gate_S = staticmethod(apply_omega_gate_S)
    omega_gate_C = staticmethod(apply_omega_gate_C)
    omega_gate_F = staticmethod(apply_omega_gate_F)
    pack_omega12 = staticmethod(pack_omega12)
    unpack_omega12 = staticmethod(unpack_omega12)
    pack_omega_signature12 = staticmethod(pack_omega_signature12)
    unpack_omega_signature12 = staticmethod(unpack_omega_signature12)

    frobenius_pair = staticmethod(frobenius_pair)
    frobenius_component12 = staticmethod(frobenius_component12)
    is_trace1_pair = staticmethod(is_trace1_pair)
    is_reachable_component = staticmethod(is_reachable_component)
    gf4_trace = staticmethod(gf4_trace)
    gf4_norm = staticmethod(gf4_norm)

    shadow_partner_byte = staticmethod(shadow_partner_byte)
    shadow_partner_map = staticmethod(shadow_partner_map)
    state_conjugate_f = staticmethod(state_conjugate_f)
    state_scan_from_state = staticmethod(state_scan_from_state)

    k4_orbit = staticmethod(k4_orbit)
    k4_stabilizer = staticmethod(k4_stabilizer)
    fixed_locus = staticmethod(fixed_locus)
    fixed_states_of_gate = staticmethod(fixed_states_of_gate)


class MomentOps:
    make = staticmethod(moment_from_ledger)
    verify = staticmethod(verify_moment)
    compare = staticmethod(compare_ledgers)
    future_cone = staticmethod(future_cone_measure)
    future_entropy = staticmethod(future_entropy_bits)
    future_expectation_exact = staticmethod(future_expectation_exact)
    future_expectation_float = staticmethod(future_expectation_float)
    derivative = staticmethod(directional_derivative)
    byte_derivatives = staticmethod(byte_derivative_table)
    transport_table = staticmethod(exact_transport_table)
    depth4_frame = staticmethod(depth4_frame)
    locus_of_state = staticmethod(locus_of_state)
    states_on_locus = staticmethod(states_on_locus)
    future_locus_measure = staticmethod(future_locus_measure)
    shell_histogram = staticmethod(shell_histogram)
    shell_population = staticmethod(shell_population)
    shell_transition_probability = staticmethod(shell_transition_probability)
    shell_transition_matrix = staticmethod(shell_transition_matrix_for_q_weight)
    shell_markov_step = staticmethod(shell_markov_step)
    optical_coordinates = staticmethod(optical_coordinates)
    stabilizer_type = staticmethod(stabilizer_type)
    verify_optical_conjugacy = staticmethod(verify_optical_conjugacy)


class SpectralOps:
    walsh_matrix = staticmethod(walsh_hadamard64)
    q_class = staticmethod(q_word6)
    q_transport = staticmethod(q_word6_for_items)
    shell_krawtchouk_transform_exact = staticmethod(
        shell_krawtchouk_transform_exact
    )
    shell_krawtchouk_inverse_exact = staticmethod(
        shell_krawtchouk_inverse_exact
    )


class ClimateOps:
    """QuBEC climate naming aligned with docs/theory/QuBEC_Climate_Dynamics.md.

    rho = N_mean/6, eta = 1 - 2*rho, m = 2*rho - 1, Var(N) = 6*rho*(1-rho) (Sections 3-4).
    M2 equilibrium vs empirical chirality support (Section 5). Bundled cell view (Section 16).
    """

    shell_order_parameters_from_hist = staticmethod(shell_order_parameters_from_hist)
    m2_empirical_from_chi_hist = staticmethod(m2_empirical_from_chi_hist)
    m2_equilibrium_from_shell_hist = staticmethod(m2_equilibrium_from_shell_hist)
    cell_climate_from_histograms = staticmethod(cell_climate_from_histograms)


def _gyroscopic_ops():
    from src.tools.gyroscopic import ops as _ops

    return _ops


class RuntimeOps:
    """Native GyroGraph/GyroLabe bindings and batched word4 paths (Quantum SDK spec Section 11)."""

    @staticmethod
    def initialize_native() -> None:
        _gyroscopic_ops().gyromatmul_runtime_caps()

    @staticmethod
    def signature_from_bytes(data: bytes):
        return _gyroscopic_ops().gyrograph_word_signature_from_bytes(data)

    @staticmethod
    def compose_signatures(left, right):
        return _gyroscopic_ops().gyrograph_compose_signatures(left, right)

    @staticmethod
    def apply_signature(state24: int, sig: "GyroGraphWordSignature") -> int:
        return _gyroscopic_ops().gyrograph_apply_signature(state24, sig)

    @staticmethod
    def apply_signature_to_rest(sig: "GyroGraphWordSignature") -> int:
        return _gyroscopic_ops().gyrograph_apply_signature(GENE_MAC_REST, sig)

    @staticmethod
    def moment_from_ledger_native(ledger: bytes):
        return _gyroscopic_ops().gyrograph_moment_from_ledger_native(ledger)

    @staticmethod
    def verify_moment_native(moment, ledger: bytes) -> bool:
        return _gyroscopic_ops().gyrograph_verify_moment_native(moment, ledger)

    @staticmethod
    def compare_ledgers_native(a: bytes, b: bytes) -> tuple[int, int]:
        return _gyroscopic_ops().gyrograph_compare_ledgers_native(a, b)

    @staticmethod
    def trace_word4_batch_indexed(*args, **kwargs):
        return _gyroscopic_ops().gyrograph_trace_word4_batch_indexed(*args, **kwargs)

    @staticmethod
    def apply_trace_word4_batch_indexed(*args, **kwargs):
        return _gyroscopic_ops().gyrograph_apply_trace_word4_batch_indexed(*args, **kwargs)

    @staticmethod
    def ingest_word4_batch_indexed(*args, **kwargs):
        return _gyroscopic_ops().gyrograph_ingest_word4_batch_indexed(*args, **kwargs)

    @staticmethod
    def step_state24_by_byte(state24: int, byte: int) -> int:
        return _gyroscopic_ops().gyrograph_step_state24_by_byte(state24, byte)

    @staticmethod
    def inverse_step_state24_by_byte(state24: int, byte: int) -> int:
        return _gyroscopic_ops().gyrograph_inverse_step_state24_by_byte(state24, byte)

    @staticmethod
    def chirality_distance(state24_a: int, state24_b: int) -> int:
        return (chirality_word6(state24_a) ^ chirality_word6(state24_b)).bit_count()

    @staticmethod
    def chirality_distances_along_trajectory(
        payload: bytes | Iterable[int],
        start_state24: int = GENE_MAC_REST,
    ) -> list[int]:
        seq = state_scan_from_state(payload, start_state24)
        if len(seq) < 2:
            return []
        return [
            RuntimeOps.chirality_distance(seq[i], seq[i + 1])
            for i in range(len(seq) - 1)
        ]

    state_scan_from_state = staticmethod(state_scan_from_state)
    q_class = staticmethod(q_word6)
    q_map_for_items = staticmethod(q_word6_for_items)


__all__ = [
    "ConstitutionalChart",
    "StateCharts",
    "Moment",
    "MomentComparison",
    "byte_transition",
    "compare_ledgers_pair",
    "FutureConeMeasure",
    "ReachabilityWitness",
    "OmegaState12",
    "OmegaSignature12",
    "state_charts",
    "moment_from_ledger",
    "verify_moment",
    "compare_ledgers",
    "future_cone_measure",
    "future_entropy_bits",
    "future_expectation_exact",
    "future_expectation_float",
    "directional_derivative",
    "byte_derivative_table",
    "exact_transport_table",
    "locus_of_state",
    "states_on_locus",
    "future_locus_measure",
    "shell_histogram",
    "shell_population",
    "shell_transition_probability",
    "shell_transition_matrix_for_q_weight",
    "optical_coordinates",
    "stabilizer_type",
    "witness_from_rest",
    "execute_witness_from_rest",
    "depth4_frame",
    "state_scan_from_state",
    "pack_matrix64",
    "apply_packed64_gemv",
    "dyadic_wht64_normalized",
    "shell_order_parameters_from_hist",
    "m2_empirical_from_chi_hist",
    "m2_equilibrium_from_shell_hist",
    "cell_climate_from_histograms",
    "ClimateOps",
    "RuntimeOps",
    "TensorOps",
    "StateOps",
    "MomentOps",
    "SpectralOps",
]



