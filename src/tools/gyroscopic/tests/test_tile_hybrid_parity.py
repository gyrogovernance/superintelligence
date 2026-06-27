"""Parity tests: native tile hybrid GEMV vs tiles.py reference."""

from __future__ import annotations

import random

import numpy as np

from src.tools.gyroscopic import ops
from src.tools.gyroscopic.helpers.diagnostics import decompose_tile, project_chi


def _dense_matvec(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    return W @ x


def test_hybrid_matvec_matches_dense() -> None:
    rng = np.random.default_rng(42)
    for _ in range(32):
        W = rng.normal(size=(64, 64))
        x = rng.normal(size=(64,))
        y_ref = _dense_matvec(W, x)
        y_hyb = np.array(ops.tile_hybrid_matvec(W.ravel().tolist(), x.tolist()), dtype=np.float64)
        assert np.allclose(y_ref, y_hyb, rtol=1e-5, atol=1e-5)


def test_decompose_ratios_match_python() -> None:
    rng = np.random.default_rng(7)
    for _ in range(16):
        signs = rng.integers(0, 2, size=(64, 64), dtype=np.int8)
        W = np.where(signs, 1.0, -1.0)
        py = decompose_tile(W)
        nat = ops.tile_decompose_ratios(W.ravel().tolist())
        for key in ("r_shell", "r_chi", "r_chi_minus_shell", "r_defect"):
            assert abs(py[key] - nat[key]) < 1e-4, f"{key}: py={py[key]} nat={nat[key]}"


def test_chi_projection_idempotent_native() -> None:
    """project_chi(P_chi(W)) == P_chi(W): r_chi -> 1 on projected tile."""
    rng = random.Random(1)
    W = [rng.uniform(-1.0, 1.0) for _ in range(64 * 64)]
    Wnp = np.array(W, dtype=np.float64).reshape(64, 64)
    p = project_chi(Wnp)
    ratios = ops.tile_decompose_ratios(p.ravel().tolist())
    assert ratios["r_chi"] > 0.99
    assert ratios["r_defect"] < 0.01
