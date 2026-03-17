#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import time
from dataclasses import dataclass
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.tools.gyrograph import ops as gg_ops
from src.tools.gyrograph.core import GyroGraph


@dataclass
class TraceState:
    omega12: np.ndarray
    step: np.ndarray
    last_byte: np.ndarray
    has_closed_word: np.ndarray
    word4_store: np.ndarray
    chi_ring64: np.ndarray
    chi_ring_pos: np.ndarray
    chi_valid_len: np.ndarray
    chi_hist64: np.ndarray
    shell_hist7: np.ndarray
    family_ring64: np.ndarray
    family_hist4: np.ndarray
    omega_sig: np.ndarray
    parity_O12: np.ndarray
    parity_E12: np.ndarray
    parity_bit: np.ndarray
    resonance_key: np.ndarray


def _random_words(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, 256, size=(n, 4), dtype=np.uint8)


def _random_state(rng: np.random.Generator, n: int, *, omega12: np.ndarray | None = None) -> TraceState:
    if omega12 is None:
        omega12 = rng.integers(0, 4096, size=n, dtype=np.int32)
    else:
        omega12 = omega12.copy()
    return TraceState(
        omega12=omega12,
        step=np.zeros(n, dtype=np.uint64),
        last_byte=np.full(n, 0xAA, dtype=np.uint8),
        has_closed_word=np.zeros(n, dtype=np.uint8),
        word4_store=np.zeros((n, 4), dtype=np.uint8),
        chi_ring64=np.zeros((n, 64), dtype=np.uint8),
        chi_ring_pos=np.zeros(n, dtype=np.uint8),
        chi_valid_len=np.zeros(n, dtype=np.uint8),
        chi_hist64=np.zeros((n, 64), dtype=np.uint16),
        shell_hist7=np.zeros((n, 7), dtype=np.uint16),
        family_ring64=np.zeros((n, 64), dtype=np.uint8),
        family_hist4=np.zeros((n, 4), dtype=np.uint16),
        omega_sig=np.zeros(n, dtype=np.int32),
        parity_O12=np.zeros(n, dtype=np.uint16),
        parity_E12=np.zeros(n, dtype=np.uint16),
        parity_bit=np.zeros(n, dtype=np.uint8),
        resonance_key=np.zeros(n, dtype=np.uint32),
    )


def _copy_state(s: TraceState) -> TraceState:
    return TraceState(
        omega12=s.omega12.copy(),
        step=s.step.copy(),
        last_byte=s.last_byte.copy(),
        has_closed_word=s.has_closed_word.copy(),
        word4_store=s.word4_store.copy(),
        chi_ring64=s.chi_ring64.copy(),
        chi_ring_pos=s.chi_ring_pos.copy(),
        chi_valid_len=s.chi_valid_len.copy(),
        chi_hist64=s.chi_hist64.copy(),
        shell_hist7=s.shell_hist7.copy(),
        family_ring64=s.family_ring64.copy(),
        family_hist4=s.family_hist4.copy(),
        omega_sig=s.omega_sig.copy(),
        parity_O12=s.parity_O12.copy(),
        parity_E12=s.parity_E12.copy(),
        parity_bit=s.parity_bit.copy(),
        resonance_key=s.resonance_key.copy(),
    )


def _time_fn(repeats: int, fn) -> float:
    for _ in range(2):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return time.perf_counter() - start


def _print_rate(label: str, n: int, elapsed: float, repeats: int) -> None:
    sec = elapsed / repeats
    if sec <= 0.0:
        print(f"{label}: {n} cells, {repeats} runs, avg {sec:.6e} s (too fast to measure)")
        return
    cells_per_sec = n / sec
    words_per_sec = (n * 4) / sec
    print(f"{label}: {n} cells, {repeats} runs, avg {sec*1000:.3f} ms, {cells_per_sec:.0f} cells/s, {words_per_sec:.0f} words/s")


def _check_eq(name: str, a: np.ndarray, b: np.ndarray) -> None:
    if not np.array_equal(a, b):
        raise RuntimeError(f"parity check failed for {name}")


def _snapshot_graph_state(graph: GyroGraph):
    return (
        graph._allocated.copy(),
        graph._has_closed_word.copy(),
        graph._omega12.copy(),
        graph._step.copy(),
        graph._last_byte.copy(),
        graph._word4.copy(),
        graph._chi_ring64.copy(),
        graph._chi_ring_pos.copy(),
        graph._chi_valid_len.copy(),
        graph._chi_hist64.copy(),
        graph._shell_hist7.copy(),
        graph._family_ring64.copy(),
        graph._family_hist4.copy(),
        graph._omega_sig.copy(),
        graph._parity_O12.copy(),
        graph._parity_E12.copy(),
        graph._parity_bit.copy(),
        graph._resonance_key.copy(),
        graph._resonance_buckets.copy(),
    )


def _restore_graph_state(graph: GyroGraph, state) -> None:
    (
        allocated,
        has_closed_word,
        omega12,
        step,
        last_byte,
        word4,
        chi_ring64,
        chi_ring_pos,
        chi_valid_len,
        chi_hist64,
        shell_hist7,
        family_ring64,
        family_hist4,
        omega_sig,
        parity_O12,
        parity_E12,
        parity_bit,
        resonance_key,
        resonance_buckets,
    ) = state
    graph._allocated[:] = allocated
    graph._has_closed_word[:] = has_closed_word
    graph._omega12[:] = omega12
    graph._step[:] = step
    graph._last_byte[:] = last_byte
    graph._word4[:] = word4
    graph._chi_ring64[:] = chi_ring64
    graph._chi_ring_pos[:] = chi_ring_pos
    graph._chi_valid_len[:] = chi_valid_len
    graph._chi_hist64[:] = chi_hist64
    graph._shell_hist7[:] = shell_hist7
    graph._family_ring64[:] = family_ring64
    graph._family_hist4[:] = family_hist4
    graph._omega_sig[:] = omega_sig
    graph._parity_O12[:] = parity_O12
    graph._parity_E12[:] = parity_E12
    graph._parity_bit[:] = parity_bit
    graph._resonance_key[:] = resonance_key
    graph._resonance_buckets[:] = resonance_buckets


def benchmark_trace_word4_batch(ns: list[int], repeats: int) -> None:
    print("=== trace_word4_batch ===")
    for n in ns:
        rng = np.random.default_rng(104729 + n)
        omega12 = rng.integers(0, 4096, size=n, dtype=np.int32)
        cell_ids = np.arange(n, dtype=np.int64)
        words4 = _random_words(rng, n)

        omega_py, chi_py = gg_ops._py_trace_word4_batch(omega12.copy(), words4)
        omega_native, chi_native = gg_ops.trace_word4_batch_indexed(
            cell_ids,
            omega12.copy(),
            words4,
        )
        _check_eq("trace omega", omega_py, omega_native)
        _check_eq("trace chi", chi_py, chi_native)

        if gg_ops.native_available():
            print(f"trace_word4_batch parity OK for n={n}")
            elapsed = _time_fn(
                repeats,
                lambda: gg_ops.trace_word4_batch_indexed(cell_ids, omega12, words4),
            )
            _print_rate("trace_word4_batch native", n, elapsed, repeats)
        else:
            print("trace_word4_batch native not available, skipping speed line")

        if gg_ops.opencl_available():
            omega_opencl, chi_opencl = gg_ops.trace_word4_batch_indexed_opencl(
                cell_ids, omega12, words4
            )
            _check_eq("trace omega opencl", omega_py, omega_opencl)
            _check_eq("trace chi opencl", chi_py, chi_opencl)
            elapsed_opencl = _time_fn(
                repeats,
                lambda: gg_ops.trace_word4_batch_indexed_opencl(cell_ids, omega12, words4),
            )
            _print_rate("trace_word4_batch indexed opencl", n, elapsed_opencl, repeats)

        elapsed_py = _time_fn(
            repeats,
            lambda: gg_ops._py_trace_word4_batch(omega12, words4),
        )
        _print_rate("trace_word4_batch python", n, elapsed_py, repeats)


def benchmark_apply_trace_word4_batch(ns: list[int], repeats: int) -> None:
    print("=== apply_trace_word4_batch ===")
    for n in ns:
        rng = np.random.default_rng(271828 + n)
        base_omega = rng.integers(0, 4096, size=n, dtype=np.int32)
        cell_ids = np.arange(n, dtype=np.int64)
        words4 = _random_words(rng, n)
        omega_trace, chi_trace = gg_ops._py_trace_word4_batch(base_omega.copy(), words4)

        state = _random_state(rng, n, omega12=base_omega)
        profile = 1

        py_state = _copy_state(state)
        py_state.resonance_key[:] = 0
        gg_ops._py_apply_trace_word4_batch_indexed(
            cell_ids,
            py_state.omega12,
            py_state.step,
            py_state.last_byte,
            py_state.has_closed_word,
            py_state.word4_store,
            py_state.chi_ring64,
            py_state.chi_ring_pos,
            py_state.chi_valid_len,
            py_state.chi_hist64,
            py_state.shell_hist7,
            py_state.family_ring64,
            py_state.family_hist4,
            py_state.omega_sig,
            py_state.parity_O12,
            py_state.parity_E12,
            py_state.parity_bit,
            words4,
            omega_trace,
            chi_trace,
            py_state.resonance_key,
            profile,
        )

        native_state = _copy_state(state)
        native_state.resonance_key[:] = 0
        gg_ops.apply_trace_word4_batch_indexed(
            cell_ids,
            native_state.omega12,
            native_state.step,
            native_state.last_byte,
            native_state.has_closed_word,
            native_state.word4_store,
            native_state.chi_ring64,
            native_state.chi_ring_pos,
            native_state.chi_valid_len,
            native_state.chi_hist64,
            native_state.shell_hist7,
            native_state.family_ring64,
            native_state.family_hist4,
            native_state.omega_sig,
            native_state.parity_O12,
            native_state.parity_E12,
            native_state.parity_bit,
            words4,
            omega_trace,
            chi_trace,
            native_state.resonance_key,
            profile,
        )

        for attr in (
            "resonance_key",
            "omega12",
            "step",
            "last_byte",
            "has_closed_word",
            "word4_store",
            "chi_ring64",
            "chi_ring_pos",
            "chi_valid_len",
            "chi_hist64",
            "shell_hist7",
            "family_ring64",
            "family_hist4",
            "omega_sig",
            "parity_O12",
            "parity_E12",
            "parity_bit",
        ):
            _check_eq(f"apply {attr}", getattr(py_state, attr), getattr(native_state, attr))
        print(f"apply_trace_word4_batch parity OK for n={n}")

        elapsed = _time_fn(
            repeats,
            lambda: _apply_once(state, cell_ids, words4, omega_trace, chi_trace, profile),
        )
        _print_rate("apply_trace_word4_batch", n, elapsed, repeats)


def benchmark_ingest_word4_batch(ns: list[int], repeats: int) -> None:
    print("=== ingest_word4_batch ===")
    for n in ns:
        rng = np.random.default_rng(314159 + n)
        profile = 1
        cell_ids = np.arange(n, dtype=np.int64)
        state = _random_state(rng, n)
        words4 = _random_words(rng, n)

        py_state = _copy_state(state)
        omega_trace, chi_trace = gg_ops._py_trace_word4_batch(py_state.omega12, words4)
        py_state.resonance_key[:] = 0
        gg_ops._py_apply_trace_word4_batch_indexed(
            cell_ids,
            py_state.omega12,
            py_state.step,
            py_state.last_byte,
            py_state.has_closed_word,
            py_state.word4_store,
            py_state.chi_ring64,
            py_state.chi_ring_pos,
            py_state.chi_valid_len,
            py_state.chi_hist64,
            py_state.shell_hist7,
            py_state.family_ring64,
            py_state.family_hist4,
            py_state.omega_sig,
            py_state.parity_O12,
            py_state.parity_E12,
            py_state.parity_bit,
            words4,
            omega_trace,
            chi_trace,
            py_state.resonance_key,
            profile,
        )

        native_state = _copy_state(state)
        native_state.resonance_key[:] = 0
        gg_ops.ingest_word4_batch_indexed(
            cell_ids,
            native_state.omega12,
            native_state.step,
            native_state.last_byte,
            native_state.has_closed_word,
            native_state.word4_store,
            native_state.chi_ring64,
            native_state.chi_ring_pos,
            native_state.chi_valid_len,
            native_state.chi_hist64,
            native_state.shell_hist7,
            native_state.family_ring64,
            native_state.family_hist4,
            native_state.omega_sig,
            native_state.parity_O12,
            native_state.parity_E12,
            native_state.parity_bit,
            native_state.resonance_key,
            words4,
            profile,
        )

        for attr in (
            "resonance_key",
            "omega12",
            "step",
            "last_byte",
            "has_closed_word",
            "word4_store",
            "chi_ring64",
            "chi_ring_pos",
            "chi_valid_len",
            "chi_hist64",
            "shell_hist7",
            "family_ring64",
            "family_hist4",
            "omega_sig",
            "parity_O12",
            "parity_E12",
            "parity_bit",
        ):
            _check_eq(f"ingest {attr}", getattr(py_state, attr), getattr(native_state, attr))
        print(f"ingest_word4_batch parity OK for n={n}")

        elapsed = _time_fn(
            repeats,
            lambda: _ingest_once(state, cell_ids, words4, profile),
        )
        _print_rate("ingest_word4_batch", n, elapsed, repeats)


def benchmark_indexed_batch_noncontiguous(ns: list[int], repeats: int) -> None:
    print("=== indexed non-contiguous ids ===")
    for n in ns:
        rng = np.random.default_rng(271828 + n)
        capacity = n * 2
        cell_ids = np.arange(0, capacity, 2, dtype=np.int64)[:n]
        state = _random_state(rng, capacity)
        words4 = _random_words(rng, n)
        profile = 1

        py_state = _copy_state(state)
        omega_trace, chi_trace = gg_ops._py_trace_word4_batch(
            py_state.omega12[cell_ids], words4
        )
        py_state.resonance_key[:] = 0
        gg_ops._py_apply_trace_word4_batch_indexed(
            cell_ids,
            py_state.omega12,
            py_state.step,
            py_state.last_byte,
            py_state.has_closed_word,
            py_state.word4_store,
            py_state.chi_ring64,
            py_state.chi_ring_pos,
            py_state.chi_valid_len,
            py_state.chi_hist64,
            py_state.shell_hist7,
            py_state.family_ring64,
            py_state.family_hist4,
            py_state.omega_sig,
            py_state.parity_O12,
            py_state.parity_E12,
            py_state.parity_bit,
            words4,
            omega_trace,
            chi_trace,
            py_state.resonance_key,
            profile,
        )

        native_state = _copy_state(state)
        native_state.resonance_key[:] = 0
        gg_ops.ingest_word4_batch_indexed(
            cell_ids,
            native_state.omega12,
            native_state.step,
            native_state.last_byte,
            native_state.has_closed_word,
            native_state.word4_store,
            native_state.chi_ring64,
            native_state.chi_ring_pos,
            native_state.chi_valid_len,
            native_state.chi_hist64,
            native_state.shell_hist7,
            native_state.family_ring64,
            native_state.family_hist4,
            native_state.omega_sig,
            native_state.parity_O12,
            native_state.parity_E12,
            native_state.parity_bit,
            native_state.resonance_key,
            words4,
            profile,
        )
        _check_eq("noncontiguous omega12", py_state.omega12, native_state.omega12)
        _check_eq("noncontiguous resonance_key", py_state.resonance_key, native_state.resonance_key)

        elapsed = _time_fn(
            repeats,
            lambda: _ingest_once(native_state, cell_ids, words4, profile),
        )
        _print_rate("ingest_word4_batch_indexed_noncontiguous", n, elapsed, repeats)


def _apply_once(
    state: TraceState,
    cell_ids: np.ndarray,
    words4: np.ndarray,
    omega_trace: np.ndarray,
    chi_trace: np.ndarray,
    profile: int,
) -> None:
    current = _copy_state(state)
    current.resonance_key[:] = 0
    gg_ops.apply_trace_word4_batch_indexed(
        cell_ids,
        current.omega12,
        current.step,
        current.last_byte,
        current.has_closed_word,
        current.word4_store,
        current.chi_ring64,
        current.chi_ring_pos,
        current.chi_valid_len,
        current.chi_hist64,
        current.shell_hist7,
        current.family_ring64,
        current.family_hist4,
        current.omega_sig,
        current.parity_O12,
        current.parity_E12,
        current.parity_bit,
        words4,
        omega_trace,
        chi_trace,
        current.resonance_key,
        profile,
    )


def _ingest_once(
    state: TraceState,
    cell_ids: np.ndarray,
    words4: np.ndarray,
    profile: int,
) -> None:
    current = _copy_state(state)
    current.resonance_key[:] = 0
    gg_ops.ingest_word4_batch_indexed(
        cell_ids,
        current.omega12,
        current.step,
        current.last_byte,
        current.has_closed_word,
        current.word4_store,
        current.chi_ring64,
        current.chi_ring_pos,
        current.chi_valid_len,
        current.chi_hist64,
        current.shell_hist7,
        current.family_ring64,
        current.family_hist4,
        current.omega_sig,
        current.parity_O12,
        current.parity_E12,
        current.parity_bit,
        current.resonance_key,
        words4,
        profile,
    )


def benchmark_end_to_end(
    ns: list[int],
    repeats: int,
) -> None:
    print("=== GyroGraph.ingest end-to-end ===")
    for n in ns:
        rng = np.random.default_rng(1618 + n)
        words = _random_words(rng, n)
        cell_ids = np.arange(n, dtype=np.int64)
        packets = [(i, bytes(words[i])) for i in range(n)]

        if gg_ops.native_available():
            g_native = GyroGraph(
                n,
                use_native_hotpath=True,
                use_opencl_hotpath=False,
            )
            g_native.allocate_cells(n)
            g_py = GyroGraph(
                n,
                use_native_hotpath=False,
                use_opencl_hotpath=False,
            )
            g_py.allocate_cells(n)

            g_native.ingest(packets)
            g_py.ingest(packets)

            for name in (
                "_omega12",
                "_step",
                "_last_byte",
                "_word4",
                "_chi_ring64",
                "_family_ring64",
                "_chi_ring_pos",
                "_chi_valid_len",
                "_chi_hist64",
                "_family_hist4",
                "_shell_hist7",
                "_omega_sig",
                "_parity_O12",
                "_parity_E12",
                "_parity_bit",
                "_resonance_key",
                "_resonance_buckets",
            ):
                _check_eq(f"graph ingest {name}", getattr(g_native, name), getattr(g_py, name))
            print(f"GyroGraph.ingest parity OK for n={n}")
        else:
            print("GyroGraph.ingest native path unavailable; parity check skipped")

        g_native = None
        if gg_ops.native_available():
            g_native = GyroGraph(
                n,
                use_native_hotpath=True,
                use_opencl_hotpath=False,
            )
            g_native.allocate_cells(n)
            native_snapshot = _snapshot_graph_state(g_native)
        else:
            native_snapshot = None

        g_py = GyroGraph(
            n,
            use_native_hotpath=False,
            use_opencl_hotpath=False,
        )
        g_py.allocate_cells(n)
        py_snapshot = _snapshot_graph_state(g_py)

        for mode_label, reset_each_repeat in (
            ("(in-place)", False),
            ("(reset each run)", True),
        ):
            def _run_native_once() -> None:
                assert g_native is not None
                if reset_each_repeat and native_snapshot is not None:
                    _restore_graph_state(g_native, native_snapshot)
                g_native.ingest_flat(cell_ids, words)

            def _run_python_once() -> None:
                if reset_each_repeat:
                    _restore_graph_state(g_py, py_snapshot)
                g_py.ingest_flat(cell_ids, words)

            if gg_ops.native_available():
                elapsed = _time_fn(repeats, _run_native_once)
                _print_rate(
                    f"GyroGraph.ingest native {mode_label}",
                    n,
                    elapsed,
                    repeats,
                )

            elapsed_py = _time_fn(repeats, _run_python_once)
            _print_rate(
                f"GyroGraph.ingest python {mode_label}",
                n,
                elapsed_py,
                repeats,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GyroGraph hotpaths with parity checks."
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[256, 4096, 65536],
        help="Batch sizes to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Repetitions per benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ns = args.n
    repeats = max(1, args.repeats)
    print(f"native_available={gg_ops.native_available()}, opencl_available={gg_ops.opencl_available()}")
    print(f"bench scales: {ns}, repeats: {repeats}")

    benchmark_trace_word4_batch(ns, repeats)
    benchmark_apply_trace_word4_batch(ns, repeats)
    benchmark_ingest_word4_batch(ns, repeats)
    benchmark_indexed_batch_noncontiguous(ns, repeats)
    benchmark_end_to_end(ns, repeats)


if __name__ == "__main__":
    main()
