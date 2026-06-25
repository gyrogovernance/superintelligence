#!/usr/bin/env python3
"""Parity: gyroscopic kernel gravity metadata vs aqpu_gravity_common."""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path


def _repo_root() -> Path:
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("repo root not found")


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.references.experiments import aqpu_gravity_common as gc


def _wht64_int32(data: list[int]) -> list[int]:
    arr = list(data)
    stride = 32
    while stride >= 1:
        for i in range(0, 64, 2 * stride):
            for j in range(stride):
                a = arr[i + j]
                b = arr[i + j + stride]
                arr[i + j] = a + b
                arr[i + j + stride] = a - b
        stride >>= 1
    return arr


def _chi_wht(signs: int) -> int:
    data = [1 if (signs >> i) & 1 else -1 for i in range(64)]
    w = _wht64_int32(data)
    best_k = max(range(64), key=lambda k: abs(w[k]))
    return best_k & 0x3F


def _py_analyze_q10(sign_bytes: bytes) -> tuple[int, int, int]:
    signs_a = struct.unpack_from("<Q", sign_bytes, 0)[0]
    signs_b = struct.unpack_from("<Q", sign_bytes, 8)[0]
    chi_a = _chi_wht(signs_a)
    chi_b = _chi_wht(signs_b)
    q = chi_a ^ chi_b
    shell = q.bit_count()
    k4 = (chi_a.bit_count() & 1) | ((chi_b.bit_count() & 1) << 1)
    return q, shell, k4


def _py_gravity_scale(layer: int, total: int, k4: int, shell: int) -> float:
    del k4, shell
    g1 = gc.dln_g_dpsi()
    return math.exp(g1 * layer / total)


def _try_native():
    try:
        from src.tools.gyroscopic import ops
        from src.tools.gyroscopic.helpers import diagnostics as diag
    except Exception:
        return None
    try:
        ops.build_native()
    except Exception:
        return None
    return ops, diag


def test_gravity_g1_parity() -> None:
    pair = _try_native()
    if pair is None:
        return
    native, _diag = pair
    g1_py = gc.dln_g_dpsi()
    g1_c = native.gravity_g1()
    assert abs(g1_c - g1_py) < 1e-4


def test_gravity_scale_grid() -> None:
    pair = _try_native()
    if pair is None:
        return
    native, _diag = pair

    for layer in (0, 10, 35):
        for k4 in range(4):
            for shell in range(7):
                scale_py = _py_gravity_scale(layer, 36, k4, shell)
                scale_c = native.gravity_scale(layer, 36, k4, shell)
                assert scale_py > 0.0
                assert abs(scale_c - scale_py) < 1e-4 * max(1.0, abs(scale_py))


def test_analyze_q10_group_parity() -> None:
    pair = _try_native()
    if pair is None:
        return
    native, diag = pair

    for seed in range(20):
        sign_bytes = bytes((seed * 17 + i * 31) & 0xFF for i in range(16))
        q_py, sh_py, k4_py = _py_analyze_q10(sign_bytes)
        q_c, sh_c, k4_c = diag.analyze_q1_group(sign_bytes)
        assert q_c == q_py
        assert sh_c == sh_py
        assert k4_c == k4_py
        scale_py = _py_gravity_scale(10, 36, k4_py, sh_py)
        scale_c = native.gravity_scale(10, 36, k4_c, sh_c)
        assert scale_py > 0.0
        assert abs(scale_c - scale_py) < 1e-4 * max(1.0, abs(scale_py))
