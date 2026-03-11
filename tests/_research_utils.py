"""
Research-only utilities (legacy Omega/atlas-era helpers).

Not used by the conformance physics suite. Kept for future exploration.
"""

from __future__ import annotations

from math import comb

import numpy as np
from numpy.typing import NDArray

from src.constants import GENE_MAC_A12, GENE_MAC_B12


def cycle_lengths_of_permutation(perm: NDArray[np.int64]) -> list[int]:
    """Return sorted list of cycle lengths for a permutation (array of indices)."""
    n = int(perm.size)
    visited = np.zeros(n, dtype=np.bool_)
    lengths: list[int] = []
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
    """Extract (u,v) from 24-bit state relative to GENE_Mac rest."""
    a = (int(state24) >> 12) & 0xFFF
    b = int(state24) & 0xFFF
    u = a ^ GENE_MAC_A12
    v = b ^ GENE_MAC_B12
    return u & 0xFFF, v & 0xFFF


def word_odd_even_xors(
    bytes_seq: list[int], masks_a12: NDArray[np.uint16]
) -> tuple[int, int, int]:
    """Return (parity, O, E) for a byte sequence using mask table (1-indexed O/E)."""
    O = 0
    E = 0
    for i, b in enumerate(bytes_seq):
        m = int(masks_a12[int(b) & 0xFF])
        if (i % 2) == 0:
            O ^= m
        else:
            E ^= m
    return (len(bytes_seq) & 1), (O & 0xFFF), (E & 0xFFF)


def _coeffs_1_plus_z_pow_k(k: int) -> list[int]:
    return [comb(k, j) for j in range(k + 1)]


def _coeffs_1_plus_z2_pow_k(k: int) -> list[int]:
    out = [0] * (2 * k + 1)
    for j in range(k + 1):
        out[2 * j] = comb(k, j)
    return out


def _poly_convolve(a: list[int], b: list[int]) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def coeffs_mask_weight_enumerator_closed_form_legacy() -> list[int]:
    """Old kernel mask code (1+z^2)^4 (1+z)^4. Legacy only."""
    return _poly_convolve(_coeffs_1_plus_z2_pow_k(4), _coeffs_1_plus_z_pow_k(4))


def coeffs_archetype_distance_enumerator_closed_form() -> list[int]:
    """Legacy: 24-bit archetype distance over old Omega; (1+z^2)^8 (1+z)^8."""
    return _poly_convolve(_coeffs_1_plus_z2_pow_k(8), _coeffs_1_plus_z_pow_k(8))
