"""64x64 weight tile decomposition for static GGUF diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

TILE = 64


def _popcount_arr(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint8, copy=False)
    c = np.zeros(v.shape, dtype=np.int32)
    for _ in range(8):
        c += (v & 1).astype(np.int32)
        v >>= 1
    return c


def _xor_indices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(TILE, dtype=np.uint8)
    i = idx[:, None]
    j = idx[None, :]
    d = np.bitwise_xor(i, j).astype(np.uint8)
    shell = _popcount_arr(d)
    return i, j, shell


_IJ, _JI, _SHELL = _xor_indices()


def frobenius_norm(mat: np.ndarray) -> float:
    return float(np.linalg.norm(mat, ord="fro"))


def project_shell(W: np.ndarray) -> np.ndarray:
    """Shell-radial projection: average over pairs with same popcount(i XOR j)."""
    out = np.zeros((TILE, TILE), dtype=np.float64)
    for r in range(7):
        mask = _SHELL == r
        if not np.any(mask):
            continue
        out[mask] = float(W[mask].mean())
    return out


def project_chi(W: np.ndarray) -> np.ndarray:
    """Chirality translation-invariant (XOR-circulant) projection."""
    f = np.zeros(TILE, dtype=np.float64)
    idx = np.arange(TILE, dtype=np.uint8)
    for d in range(TILE):
        b = np.bitwise_xor(idx, d)
        f[d] = float(W[idx, b].mean())
    out = np.zeros((TILE, TILE), dtype=np.float64)
    for i in range(TILE):
        out[i, :] = f[np.bitwise_xor(i, idx)]
    return out


def split_tile_gyro_halves(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split 64x64 tile into row halves (gyrophase proxy along output axis)."""
    W = np.asarray(W, dtype=np.float64)
    if W.shape != (TILE, TILE):
        raise ValueError(f"expected {TILE}x{TILE} tile, got {W.shape}")
    mid = TILE // 2
    return W[:mid, :], W[mid:, :]


def decompose_tile_gyro_halves(W: np.ndarray) -> dict[str, float]:
    """Frobenius energy ratios for active vs passive column halves of a tile."""
    W = np.asarray(W, dtype=np.float64)
    norm_w = frobenius_norm(W)
    if norm_w <= 0.0:
        return {"r_active": 0.0, "r_passive": 0.0, "r_imbalance": 0.0, "norm": 0.0}
    wa, wb = split_tile_gyro_halves(W)
    ra = frobenius_norm(wa) / norm_w
    rb = frobenius_norm(wb) / norm_w
    return {
        "r_active": ra,
        "r_passive": rb,
        "r_imbalance": abs(ra - rb),
        "norm": norm_w,
    }


def decompose_tile(W: np.ndarray) -> dict[str, float]:
    """Return Frobenius energy ratios for shell, chi, chi\\shell, and defect."""
    W = np.asarray(W, dtype=np.float64)
    if W.shape != (TILE, TILE):
        raise ValueError(f"expected {TILE}x{TILE} tile, got {W.shape}")
    norm_w = frobenius_norm(W)
    if norm_w <= 0.0:
        return {
            "r_shell": 0.0,
            "r_chi": 0.0,
            "r_chi_minus_shell": 0.0,
            "r_defect": 0.0,
            "norm": 0.0,
        }
    p_shell = project_shell(W)
    p_chi = project_chi(W)
    d_chi = W - p_chi
    p_chi_only = p_chi - p_shell
    return {
        "r_shell": frobenius_norm(p_shell) / norm_w,
        "r_chi": frobenius_norm(p_chi) / norm_w,
        "r_chi_minus_shell": frobenius_norm(p_chi_only) / norm_w,
        "r_defect": frobenius_norm(d_chi) / norm_w,
        "norm": norm_w,
    }


def summarize_ratios(ratios: list[dict[str, float]]) -> dict[str, Any]:
    if not ratios:
        return {"count": 0}
    keys = ("r_shell", "r_chi", "r_chi_minus_shell", "r_defect")
    out: dict[str, Any] = {"count": len(ratios)}
    for key in keys:
        vals = sorted(r[key] for r in ratios)
        out[key] = {
            "mean": round(float(np.mean(vals)), 6),
            "percentiles": {
                "p10": round(vals[max(0, int(0.10 * (len(vals) - 1)))], 6),
                "p50": round(vals[len(vals) // 2], 6),
                "p90": round(vals[min(len(vals) - 1, int(0.90 * (len(vals) - 1)))], 6),
            },
        }
    return out


def summarize_gyro_halves(ratios: list[dict[str, float]]) -> dict[str, Any]:
    if not ratios:
        return {"count": 0}
    keys = ("r_active", "r_passive", "r_imbalance")
    out: dict[str, Any] = {"count": len(ratios)}
    for key in keys:
        vals = sorted(r[key] for r in ratios)
        out[key] = {
            "mean": round(float(np.mean(vals)), 6),
            "percentiles": {
                "p10": round(vals[max(0, int(0.10 * (len(vals) - 1)))], 6),
                "p50": round(vals[len(vals) // 2], 6),
                "p90": round(vals[min(len(vals) - 1, int(0.90 * (len(vals) - 1)))], 6),
            },
        }
    return out


def random_tile_reference(n: int = 2000, *, seed: int = 0) -> dict[str, Any]:
    """Binomial/null reference: random +/-1 tiles (no trained structure)."""
    rng = np.random.default_rng(seed)
    ratios: list[dict[str, float]] = []
    for _ in range(n):
        signs = rng.integers(0, 2, size=(TILE, TILE), dtype=np.int8)
        W = np.where(signs, 1.0, -1.0)
        ratios.append(decompose_tile(W))
    return summarize_ratios(ratios)
