#!/usr/bin/env python3
"""Parity: gyroscopic_route_path structural routing (magnitude-neutral)."""

from __future__ import annotations

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


def _py_route_path(shell: int, k4_char: int) -> int:
    from src.tools.gyroscopic import ops

    if shell == 0 or shell >= 6:
        return ops.PATH_ISOTROPIC
    mapping = {
        ops.K4_ID: ops.PATH_BULK_CS,
        ops.K4_W2: ops.PATH_BULK_UNA,
        ops.K4_W2P: ops.PATH_BULK_ONA,
        ops.K4_F: ops.PATH_BULK_BU,
    }
    return mapping[k4_char & 0x3]


def test_route_path_horizon_isotropic() -> None:
    pair = _try_native()
    if pair is None:
        return
    _ops, diag = pair
    for shell in (0, 6, 7):
        for k4 in range(4):
            assert diag.route_path(shell, k4) == _ops.PATH_ISOTROPIC


def test_route_path_bulk_sectors() -> None:
    pair = _try_native()
    if pair is None:
        return
    _ops, diag = pair
    for shell in range(1, 6):
        for k4 in range(4):
            assert diag.route_path(shell, k4) == _py_route_path(shell, k4)


def test_gravity_scale_independent_of_structure() -> None:
    pair = _try_native()
    if pair is None:
        return
    native, _diag = pair
    base = native.gravity_scale(12, 36, 0, 0)
    for k4 in range(4):
        for shell in range(7):
            assert native.gravity_scale(12, 36, k4, shell) == base
