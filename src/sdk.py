from __future__ import annotations

"""
High-level Quantum SDK surface for the Gyroscopic ASI aQPU.

This file keeps the public exposure compact and centered on:
- Moments
- Gyrostates and charts
- Future-cone structure
- Exact structural derivatives
- State synthesis from rest
- Tensor and spectral access through GyroLabe when available

It is intentionally thin over src.constants, src.api, and src.tools.gyrolabe.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import log2
from typing import Any, TypeAlias

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

ObservableInt: TypeAlias = Callable[[int], int]
ObservableNum: TypeAlias = Callable[[int], int | float]


def _omega_gate_name_to_code(name: str) -> int:
    if name == "id":
        return 0
    if name == "S":
        return 1
    if name == "C":
        return 2
    if name == "F":
        return 3
    raise ValueError(f"Unknown Omega gate: {name!r}")


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


def gyrolabe_available() -> bool:
    try:
        from src.tools.gyrolabe import ops

        return bool(ops.native_available())
    except Exception:
        return False


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

    @staticmethod
    def wht64(x: Any) -> Any:
        import torch
        from src.tools.gyrolabe import ops

        x_t = x if isinstance(x, torch.Tensor) else torch.as_tensor(
            x, dtype=torch.float32, device="cpu"
        )
        return ops.wht64(x_t)


class QubecFixedVector64:
    """Fixed-point vector for QuBEC matmul pairing. Exact over the chosen fixed-point representation, not IEEE-754 exact."""

    def __init__(self, packed: Any) -> None:
        self._packed = packed


class QubecFixedMatrix64:
    """Fixed-point matrix for QuBEC matmul. Exact over the chosen fixed-point representation, not IEEE-754 exact."""

    def __init__(self, W: Any, n_bits: int = 16) -> None:
        self._W = W
        self._n_bits = int(n_bits)

    def gemv(self, x: Any) -> Any:
        import torch
        from src.tools.gyrolabe import ops

        x_t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32, device="cpu")
        return ops.gyromatmul_f32_gemv(self._W, x_t, n_bits=self._n_bits)

    def qubec_gemv_pair(self, x: QubecFixedVector64) -> Any:
        raise NotImplementedError("qubec_gemv_pair is deprecated")

    def gemm_packed_batch(self, X: Any) -> Any:
        """Batched GEMM: Y[b] = W @ X[b]. X: [batch, cols]. Returns [batch, rows]."""
        import torch
        from src.tools.gyrolabe import ops

        X_t = X if isinstance(X, torch.Tensor) else torch.as_tensor(
            X, dtype=torch.float32, device="cpu"
        )
        return ops.gyromatmul_f32_gemm(self._W, X_t, n_bits=self._n_bits)


class TensorOps:
    """
    Internal fixed-point tensor engine. Uses fixed-point quantization internally.
    Exact over the chosen fixed-point representation, not IEEE-754 exact.
    """

    @staticmethod
    def gemv64(W: Any, x: Any, n_bits: int = 16) -> Any:
        """Fixed-point GEMV. Exact over the chosen fixed-point representation, not IEEE-754 exact."""
        import torch
        from src.tools.gyrolabe import ops

        W_t = W if isinstance(W, torch.Tensor) else torch.as_tensor(
            W, dtype=torch.float32, device="cpu"
        )
        x_t = x if isinstance(x, torch.Tensor) else torch.as_tensor(
            x, dtype=torch.float32, device="cpu"
        )
        return ops.gyromatmul_f32_gemv(W_t, x_t, n_bits=n_bits)

    @staticmethod
    def qubec_matrix64(W: Any, n_bits: int = 16) -> QubecFixedMatrix64:
        import torch
        from src.tools.gyrolabe import ops

        W_t = W if isinstance(W, torch.Tensor) else torch.as_tensor(
            W, dtype=torch.float32, device="cpu"
        )
        return QubecFixedMatrix64(W_t, n_bits=n_bits)

    @staticmethod
    def qubec_vector64(x: Any, n_bits: int = 16) -> QubecFixedVector64:
        import torch
        from src.tools.gyrolabe import ops

        x_t = x if isinstance(x, torch.Tensor) else torch.as_tensor(
            x, dtype=torch.float32, device="cpu"
        )
        return QubecFixedVector64(ops.QubecPackedVector64(x_t, n_bits=n_bits))

    @staticmethod
    def qubec_matrix64_opencl(W: Any, n_bits: int = 8) -> "GPUBackend64":
        """Pack matrix for GPU batched GEMM. Uses 8-bit fixed-point by default for best throughput."""
        import torch
        from src.tools.gyrolabe import ops
        from src.tools.gyrolabe import opencl_backend

        W_t = W if isinstance(W, torch.Tensor) else torch.as_tensor(
            W, dtype=torch.float32, device="cpu"
        )
        scale_max = (1 << (n_bits - 1)) - 1
        scale = scale_max / max(float(W_t.abs().max().item()), 1e-12)
        packed_cpu = torch.round(W_t * scale).to(torch.int32)
        opencl_backend.initialize()
        return GPUBackend64(opencl_backend.OpenCLPackedMatrix64I32(packed_cpu, n_bits=n_bits))


class GPUBackend64:
    """GPU-resident QuBEC matrix for batched GEMM. Use TensorOps.qubec_matrix64_opencl to create."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def gemm_batch(self, X: Any) -> Any:
        """Batched GEMM: Y[b] = W @ X[b]. X: [batch, cols]. Returns [batch, rows] on CPU."""
        return self._backend.gemm_packed_batch(X)

    def close(self) -> None:
        """Release GPU resources."""
        self._backend.close()


class RuntimeOps:
    @staticmethod
    def signature_scan(payload: Any) -> Any:
        from src.tools.gyrolabe import ops

        return ops.signature_scan(payload)

    @staticmethod
    def extract_scan(payload: Any) -> Any:
        from src.tools.gyrolabe import ops

        return ops.extract_scan(payload)

    @staticmethod
    def states_from_signatures(signatures: Any) -> Any:
        from src.tools.gyrolabe import ops

        return ops.signatures_to_states(signatures)

    @staticmethod
    def chirality_states_from_bytes(payload: Any) -> Any:
        from src.tools.gyrolabe import ops

        return ops.chirality_states_from_bytes(payload)

    @staticmethod
    def apply_signature_to_state(state24: Any, signature: Any) -> Any:
        from src.tools.gyrolabe import ops

        return ops.apply_signature_to_state(state24, signature)

    @staticmethod
    def apply_signature_batch(states: Any, signatures: Any) -> Any:
        from src.tools.gyrolabe import ops

        return ops.apply_signature_batch(states, signatures)

    @staticmethod
    def step_byte_batch(states: Any, byte: int) -> Any:
        from src.tools.gyrolabe import ops

        return ops.step_byte_batch(states, byte)

    @staticmethod
    def state_scan_from_state(payload: Any, start_state24: int) -> Any:
        from src.tools.gyrolabe import ops

        return ops.state_scan_from_state(payload, start_state24)

    @staticmethod
    def chirality_distance(states_a: Any, states_b: Any) -> Any:
        from src.tools.gyrolabe import ops

        return ops.chirality_distance(states_a, states_b)

    @staticmethod
    def chirality_distance_adjacent(states: Any, lookahead: int = 1) -> Any:
        from src.tools.gyrolabe import ops

        return ops.chirality_distance_adjacent(states, lookahead=lookahead)

    @staticmethod
    def qmap_extract(payload: Any) -> Any:
        from src.tools.gyrolabe import ops

        return ops.qmap_extract(payload)

    @staticmethod
    def apply_signature_to_rest(signature: Any) -> Any:
        from src.tools.gyrolabe import ops

        return ops.signatures_to_states(signature)

    @staticmethod
    def omega12_from_states(states: Any) -> Any:
        from src.tools.gyrolabe import ops

        return ops.state24_to_omega12_batch(states)

    @staticmethod
    def omega12_and_valid_from_states(states: Any) -> Any:
        """Returns (omega12, valid) tuple from state24 batch."""
        from src.tools.gyrolabe import ops

        return ops.state24_to_omega12_batch(states)

    @staticmethod
    def states_from_omega12(omega12: Any) -> Any:
        from src.tools.gyrolabe import ops

        return ops.omega12_to_state24_batch(omega12)

    @staticmethod
    def step_omega12_batch(omega12: Any, byte: int) -> Any:
        from src.tools.gyrolabe import ops

        return ops.step_omega12_batch(omega12, byte)

    @staticmethod
    def apply_omega_signature_batch(omega12: Any, signatures: Any) -> Any:
        from src.tools.gyrolabe import ops

        return ops.apply_omega_signature_batch(omega12, signatures)

    @staticmethod
    def shell_histogram_state24(states: Any) -> Any:
        from src.tools.gyrolabe import ops

        return ops.shell_histogram_state24(states)

    @staticmethod
    def shell_histogram_omega12(omega12: Any) -> Any:
        from src.tools.gyrolabe import ops

        return ops.shell_histogram_omega12(omega12)

    @staticmethod
    def omega_signature_scan(payload: Any) -> Any:
        from src.tools.gyrolabe import ops

        return ops.omega_signature_scan(payload)

    @staticmethod
    def omega12_scan_from_omega12(payload: Any, start_omega12: int) -> Any:
        from src.tools.gyrolabe import ops

        return ops.omega12_scan_from_omega12(payload, start_omega12)

    @staticmethod
    def shell_histogram_state24_checked(states: Any) -> Any:
        from src.tools.gyrolabe import ops

        return ops.shell_histogram_state24_checked(states)

    @staticmethod
    def apply_omega_gate_batch(omega12: Any, gate_code: int) -> Any:
        from src.tools.gyrolabe import ops

        return ops.apply_omega_gate_batch(omega12, gate_code)

    @staticmethod
    def apply_omega_gate_batch_named(omega12: Any, gate: str) -> Any:
        from src.tools.gyrolabe import ops

        return ops.apply_omega_gate_batch(
            omega12, _omega_gate_name_to_code(gate)
        )


def initialize_native() -> None:
    """Initialize native GyroLabe tables and state once per process."""
    from src.tools.gyrolabe import ops

    ops.initialize_native()


__all__ = [
    "ConstitutionalChart",
    "StateCharts",
    "Moment",
    "MomentComparison",
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
    "gyrolabe_available",
    "initialize_native",
    "QubecFixedVector64",
    "QubecFixedMatrix64",
    "StateOps",
    "MomentOps",
    "SpectralOps",
    "TensorOps",
    "RuntimeOps",
]



