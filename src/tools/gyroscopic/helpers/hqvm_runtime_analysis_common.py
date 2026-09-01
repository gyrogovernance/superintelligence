#!/usr/bin/env python3
"""hqvm_runtime_analysis_common.py — Native-backed shared infrastructure.

Role: require C gyroscopic_native; report/timing helpers for the fat measurement script.
Inputs: src.tools.gyroscopic.ops (ctypes).
Outputs: thin wrappers used by hqvm_runtime_analysis_1.py.
Companion: hqvm_runtime_analysis_run.py.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_HELPERS = Path(__file__).resolve().parent
_REPO = _HELPERS.parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

RESULTS_PATH = _HELPERS / "hqvm_runtime_analysis_results.txt"
WORKNOTES_PATH = _HELPERS / "hqvm_runtime_analysis_temp_worknotes.txt"

VERBOSE = False

try:
    from src.tools.gyroscopic import ops as _ops

    _ops.build_native(force=False)
    _NATIVE_OK = True
    _NATIVE_ERR = ""
except Exception as exc:  # noqa: BLE001
    _ops = None  # type: ignore[assignment]
    _NATIVE_OK = False
    _NATIVE_ERR = str(exc)


def require_native() -> None:
    if not _NATIVE_OK or _ops is None:
        raise RuntimeError(
            "hqvm_runtime_analysis requires the C backend "
            f"(kernel.c). Build failed: {_NATIVE_ERR}"
        )


def ops():
    require_native()
    return _ops


def set_verbose(v: bool) -> None:
    global VERBOSE
    VERBOSE = bool(v)


def vprint(msg: str) -> None:
    if VERBOSE:
        print(msg)


def info(msg: str) -> None:
    print(f"  [INFO] {msg}")


@dataclass
class ReportState:
    gates: list[tuple[str, bool]] = field(default_factory=list)
    section_n: int = 0
    headlines: dict[str, str] = field(default_factory=dict)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def section(state: ReportState, title: str) -> None:
    state.section_n += 1
    print(f"\n{state.section_n}. {title}")
    print("=" * 5)


def check(
    state: ReportState,
    label: str,
    ok: bool,
    *,
    quantity: str,
    measured: str,
    threshold: str,
) -> None:
    state.gates.append((quantity, bool(ok)))
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {quantity}")
    print(f"         measured: {measured}")
    print(f"         threshold: {threshold}")
    if label and label != quantity:
        vprint(f"         note: {label}")


def bench_n(fn, n_ops: int, *, repeat: int = 3) -> tuple[float, float]:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, n_ops / best if best > 0 else 0.0
