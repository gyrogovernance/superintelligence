"""
Orthonormal Walsh-Hadamard transform on length-64 vectors (batch).

Uses the Sylvester Hadamard with 1/sqrt(N) scaling so forward equals inverse
and Parseval holds: sum(x**2) == sum(c**2).
"""

from __future__ import annotations

import numpy as np

WHT_SIZE = 64


def wht_64_batch(x: np.ndarray) -> np.ndarray:
    """
    Apply orthonormal WHT along the last axis (size must be 64).

    x: float array shape (..., 64)
    returns same shape, float64
    """
    if x.shape[-1] != WHT_SIZE:
        raise ValueError(f"last dim must be {WHT_SIZE}, got {x.shape[-1]}")
    h = np.asarray(x, dtype=np.float64).copy()
    n = WHT_SIZE
    stride = 1
    while stride < n:
        for i in range(0, n, 2 * stride):
            u = h[..., i : i + stride]
            v = h[..., i + stride : i + 2 * stride]
            a = u + v
            b = u - v
            h[..., i : i + stride] = a
            h[..., i + stride : i + 2 * stride] = b
        stride *= 2
    return h / np.sqrt(float(n))


def topk_energy_fractions(coeffs: np.ndarray, ks: tuple[int, ...]) -> dict[int, float]:
    """
    coeffs: (..., 64) WHT coefficients
    Returns mean fraction of total energy in top-k |coeff|^2 per row.
    """
    e = coeffs * coeffs
    # sort descending along last axis
    sorted_e = np.sort(e, axis=-1)[..., ::-1]
    total = np.sum(sorted_e, axis=-1, keepdims=True)
    total = np.maximum(total, 1e-30)
    out: dict[int, float] = {}
    for k in ks:
        kk = min(k, WHT_SIZE)
        frac = np.sum(sorted_e[..., :kk], axis=-1) / total[..., 0]
        out[k] = float(np.mean(frac))
    return out


def topk_reconstruction_rel_l2(
    x: np.ndarray, coeffs: np.ndarray, ks: tuple[int, ...]
) -> dict[int, float]:
    """
    x: (n, 64) original
    coeffs: (n, 64) WHT(x)
    For each k, keep k largest |coeff|^2 entries (per row), IWHT, report mean ||x-xhat||/||x||.
    """
    if x.shape != coeffs.shape or x.shape[-1] != WHT_SIZE:
        raise ValueError("x and coeffs must be (n, 64)")
    n = x.shape[0]
    x64 = np.asarray(x, dtype=np.float64)
    c64 = np.asarray(coeffs, dtype=np.float64)
    e = c64 * c64
    idx = np.argsort(-e, axis=-1)
    out: dict[int, float] = {}
    x_norm = np.linalg.norm(x64, axis=-1)
    mask_denom = np.maximum(x_norm, 1e-30)
    rows = np.arange(n, dtype=np.intp)[:, None]
    for k in ks:
        kk = min(k, WHT_SIZE)
        sparse = np.zeros_like(c64)
        cols = idx[:, :kk]
        vals = np.take_along_axis(c64, idx[:, :kk], axis=-1)
        sparse[rows, cols] = vals
        recon = wht_64_batch(sparse)
        err = np.linalg.norm(x64 - recon, axis=-1) / mask_denom
        out[k] = float(np.mean(err))
    return out
