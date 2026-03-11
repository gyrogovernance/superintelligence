"""
Shared utility functions for physics tests.

Popcount, weight enumerators, polynomial convolve, Krawtchouk, etc.
Legacy Omega/atlas helpers moved to tests._research_utils.
"""

from __future__ import annotations

from math import comb

import numpy as np
from numpy.typing import NDArray

# Popcount and parity LUTs for 12-bit values
_POP12 = np.array([bin(i).count("1") for i in range(4096)], dtype=np.uint8)
_PAR12 = (_POP12 & 1).astype(np.uint8)


def popcount(x: int) -> int:
    """Popcount for arbitrary integer (full width)."""
    return bin(int(x) & 0xFFFFFFFFFFFFFFFF).count("1")


def popcount12_arr(x: NDArray[np.uint16]) -> NDArray[np.uint8]:
    """Popcount for array of 12-bit values using LUT."""
    return _POP12[np.asarray(x, dtype=np.uint16)]


def parity12_arr(x: NDArray[np.uint16]) -> NDArray[np.uint8]:
    """Parity (mod 2) for array of 12-bit values using LUT."""
    return _PAR12[np.asarray(x, dtype=np.uint16)]


def hamming24(a: int, b: int) -> int:
    """24-bit Hamming distance."""
    return popcount(int(a) ^ int(b))


def weight_enumerator_counts(codewords12: NDArray[np.uint16]) -> NDArray[np.int64]:
    """Return weight enumerator counts for codewords (weights 0..12)."""
    w = popcount12_arr(codewords12).astype(np.int64)
    counts = np.bincount(w, minlength=13).astype(np.int64)
    assert int(counts.sum()) == int(codewords12.size)
    return counts


def coeffs_poly_1_plus_z_pow_k(k: int) -> list[int]:
    """Coefficients of (1 + z)^k."""
    return [comb(k, j) for j in range(k + 1)]


def coeffs_poly_1_plus_z2_pow_k(k: int) -> list[int]:
    """Coefficients of (1 + z^2)^k as a list up to degree 2k."""
    out = [0] * (2 * k + 1)
    for j in range(k + 1):
        out[2 * j] = comb(k, j)
    return out


def poly_convolve(a: list[int], b: list[int]) -> list[int]:
    """Discrete convolution of two polynomial coefficient lists."""
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def dual_code_from_parity_checks(parity_checks: list[int]) -> NDArray[np.uint16]:
    """Generate dual code from parity check vectors."""
    r = len(parity_checks)
    out = np.zeros(1 << r, dtype=np.uint16)
    for s in range(1 << r):
        w = 0
        for i in range(r):
            if (s >> i) & 1:
                w ^= int(parity_checks[i])
        out[s] = np.uint16(w & 0xFFF)
    return out


def krawtchouk(n: int, w: int, j: int) -> int:
    """
    Binary Krawtchouk polynomial K_w(j) for length n.

    K_w(j) = sum_{t=0..w} (-1)^t * C(j,t) * C(n-j, w-t)
    """
    s = 0
    for t in range(w + 1):
        if t <= j and (w - t) <= (n - j):
            s += ((-1) ** t) * comb(j, t) * comb(n - j, w - t)
    return s


def table(title: str, rows: list[tuple[str, str]], enable: bool = True) -> None:
    """Print a formatted table (gated by enable flag)."""
    if not enable:
        return
    print("\n" + title)
    print("-" * len(title))
    w = max(len(k) for k, _ in rows) if rows else 0
    for k, v in rows:
        print(f"  {k:<{w}} : {v}")


def fmt(x: float, digits: int = 12) -> str:
    """Format float to specified decimal places."""
    return f"{x:.{digits}f}"

