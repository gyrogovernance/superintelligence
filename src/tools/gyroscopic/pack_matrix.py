from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PackedMatrix64:
    """SDK 11.10 reference ABI: sign mask plus magnitude bit planes for 64-wide rows."""

    w_sign: np.ndarray
    w_bp: np.ndarray
    scale_w: float
    n_bits: int
    n_rows: int


def pack_matrix64(W: np.ndarray, n_bits: int) -> PackedMatrix64:
    if n_bits < 1 or n_bits > 52:
        raise ValueError("n_bits must be in 1..52")
    w = np.asarray(W, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != 64:
        raise ValueError("W must have shape [rows, 64]")
    rows = int(w.shape[0])
    scale_w = float(np.max(np.abs(w))) if w.size else 1.0
    if scale_w <= 0.0:
        scale_w = 1.0
    wn = w / scale_w
    signs = np.zeros(rows, dtype=np.uint64)
    bp = np.zeros((rows, n_bits), dtype=np.uint64)
    max_mag = (1 << n_bits) - 1
    for r in range(rows):
        for j in range(64):
            v = float(wn[r, j])
            if v < 0.0:
                signs[r] |= np.uint64(1) << np.uint64(j)
            m = int(np.floor(abs(v) * float(max_mag) + 0.5))
            if m > max_mag:
                m = max_mag
            for k in range(n_bits):
                if (m >> k) & 1:
                    bp[r, k] |= np.uint64(1) << np.uint64(j)
    return PackedMatrix64(signs, bp, scale_w, n_bits, rows)


def apply_packed64_gemv(packed: PackedMatrix64, x: np.ndarray) -> np.ndarray:
    """Reference GEMV via float64 reconstruction of the packed lattice surface."""
    xv = np.asarray(x, dtype=np.float64).ravel()
    if xv.size != 64:
        raise ValueError("x must have length 64")
    max_mag = (1 << packed.n_bits) - 1
    y = np.zeros(packed.n_rows, dtype=np.float64)
    for r in range(packed.n_rows):
        row = np.zeros(64, dtype=np.float64)
        for j in range(64):
            sgn = -1.0 if (int(packed.w_sign[r]) >> j) & 1 else 1.0
            mag = 0
            for k in range(packed.n_bits):
                if (int(packed.w_bp[r, k]) >> j) & 1:
                    mag |= 1 << k
            row[j] = sgn * (float(mag) / float(max_mag))
        y[r] = float(np.dot(row, xv)) * packed.scale_w
    return y


__all__ = ["PackedMatrix64", "apply_packed64_gemv", "pack_matrix64"]
