"""
Shared utility functions for physics tests.

This module centralizes common helpers used across test_physics_2.py and test_physics_3.py
to reduce duplication and improve maintainability.
"""

from __future__ import annotations

from math import comb
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from src.router.constants import ARCHETYPE_A12, ARCHETYPE_B12


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


def cycle_lengths_of_permutation(perm: NDArray[np.int64]) -> List[int]:
    """
    Return sorted list of cycle lengths for a permutation.

    Perm must be an array of shape (n,) of indices.
    """
    n = int(perm.size)
    visited = np.zeros(n, dtype=np.bool_)
    lengths: List[int] = []
    for start in range(n):
        if visited[start]:
            continue
        length = 0
        idx = start
        while not visited[idx]:
            visited[idx] = True
            idx = int(perm[idx])
            length += 1
        lengths.append(length)
    return sorted(lengths, reverse=True)


def uv_from_state24(state24: int) -> tuple[int, int]:
    """Extract (u,v) coordinates from a 24-bit state."""
    a = (int(state24) >> 12) & 0xFFF
    b = int(state24) & 0xFFF
    u = a ^ ARCHETYPE_A12
    v = b ^ ARCHETYPE_B12
    return u & 0xFFF, v & 0xFFF


def apply_word_to_indices(epi: NDArray[np.uint32], idxs: NDArray[np.int64], bytes_seq: list[int]) -> NDArray[np.int64]:
    """
    Apply a byte word to state indices using epistemology.

    Equivalent to compose_epi but with different naming convention.
    """
    cur = idxs
    for b in bytes_seq:
        cur = epi[cur, int(b) & 0xFF].astype(np.int64)
    return cur


def compose_epi(epi: NDArray[np.uint32], idxs: NDArray[np.int64], bytes_seq: List[int]) -> NDArray[np.int64]:
    """
    Compose atlas transitions using epistemology.

    Alias for apply_word_to_indices (kept for backward compatibility).
    """
    return apply_word_to_indices(epi, idxs, bytes_seq)


def word_odd_even_xors(bytes_seq: list[int], masks_a12: NDArray[np.uint16]) -> tuple[int, int, int]:
    """
    Return (parity, O, E) for a byte sequence:
      O = XOR of masks at odd positions (1-indexed): 1,3,5,...
      E = XOR of masks at even positions (1-indexed): 2,4,6,...
      parity = len(seq) mod 2
    """
    O = 0
    E = 0
    for i, b in enumerate(bytes_seq):
        m = int(masks_a12[int(b) & 0xFF])
        if (i % 2) == 0:
            O ^= m
        else:
            E ^= m
    return (len(bytes_seq) & 1), (O & 0xFFF), (E & 0xFFF)


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


def coeffs_mask_weight_enumerator_closed_form() -> list[int]:
    """
    Exact coefficients for the kernel mask code weight distribution:
      (1 + z^2)^4 (1 + z)^4
    """
    return poly_convolve(coeffs_poly_1_plus_z2_pow_k(4), coeffs_poly_1_plus_z_pow_k(4))


def coeffs_archetype_distance_enumerator_closed_form() -> list[int]:
    """
    Exact coefficients for the 24-bit archetype distance distribution over Ω:
      ( (1 + z^2)^4 (1 + z)^4 )^2 = (1 + z^2)^8 (1 + z)^8
    This matches the fact Ω = C × C.
    """
    return poly_convolve(coeffs_poly_1_plus_z2_pow_k(8), coeffs_poly_1_plus_z_pow_k(8))


def dual_code_from_parity_checks(parity_checks: List[int]) -> NDArray[np.uint16]:
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


def table(title: str, rows: List[Tuple[str, str]], enable: bool = True) -> None:
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

