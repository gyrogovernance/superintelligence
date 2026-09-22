"""Weight tiles as Walsh functions on GF(2)^6, not as fake instruction bytes.

A 64-wide block is a function on the chirality domain. Its climate is the
Plancherel energy of the 64-WHT, not a packed instruction ledger. Peak
chirality is a derived label (ties broken with jitter). Adjacent tiles give
a transport shell by XOR. Adjacent chiralities are a point of Ω = U × V.
The 4096-d row is an Ω function for AffineSpectralCodec.

Instruction-byte packing is not used here.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.family import byte_from_family_micro
from src.tools.autoencoder.datasets import D
from src.tools.autoencoder.models.general import walsh_matrix_64

TILE = 64
HIDDEN = 4096
POPCOUNT6 = np.array([int(i).bit_count() for i in range(TILE)], dtype=np.int64)


@lru_cache(maxsize=1)
def _w64() -> np.ndarray:
    return walsh_matrix_64().astype(np.float64)


def rows_to_tiles(rows: np.ndarray) -> np.ndarray:
    """[N, 4096] -> [N, 64, 64] float tiles."""
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != HIDDEN:
        raise ValueError(f"expected [N, {HIDDEN}], got {rows.shape}")
    return rows.reshape(rows.shape[0], TILE, TILE)


def tile_walsh_coeff(tiles: np.ndarray) -> np.ndarray:
    """64-WHT of each tile's signs. [N, 64, 64] -> [N, 64, 64] coeffs."""
    w = _w64()
    signs = np.where(np.asarray(tiles) >= 0.0, 1.0, -1.0)
    return signs @ w.T


def tile_mode_energy(tiles: np.ndarray) -> np.ndarray:
    """Plancherel occupation of the 64 Walsh modes. [N, 64, 64] sums to 1 on last axis."""
    coeff = tile_walsh_coeff(tiles)
    energy = coeff * coeff
    return energy / (energy.sum(axis=-1, keepdims=True) + 1e-30)


def energy_shell_hist(energy: np.ndarray) -> np.ndarray:
    """Average Walsh energy by Hamming shell. [..., 64] -> [7]."""
    e = np.asarray(energy, dtype=np.float64).reshape(-1, TILE)
    hist = np.empty(7, dtype=np.float64)
    for s in range(7):
        hist[s] = float(e[:, POPCOUNT6 == s].sum())
    return hist / (hist.sum() + 1e-30)


def participation(energy: np.ndarray) -> np.ndarray:
    """Effective number of Walsh modes per tile. [..., 64] -> [...]."""
    e = np.asarray(energy, dtype=np.float64)
    return 1.0 / (np.square(e).sum(axis=-1) + 1e-30)


def chirality_from_tiles(
    tiles: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Peak Walsh mode of each 64-vector. [N, 64] -> [N, 64] chi in 0..63.

    Exact ±1 tiles make Walsh ties common. A tiny jitter (when rng is given)
    breaks those ties uniformly; without rng, argmax would prefer low indices
    and fake a low-shell climate.
    """
    abs_c = np.abs(tile_walsh_coeff(tiles))
    if rng is not None:
        abs_c = abs_c + rng.random(abs_c.shape) * 1e-6
    return np.argmax(abs_c, axis=-1).astype(np.int64)


def shells_from_chi(chi: np.ndarray) -> np.ndarray:
    x = np.asarray(chi, dtype=np.int64)
    wt = np.zeros(x.shape, dtype=np.int64)
    for b in range(6):
        wt += (x >> b) & 1
    return wt


def transport_shells(chi: np.ndarray) -> np.ndarray:
    """XOR-shell between adjacent tiles on a row. [N, 64] -> [N, 63]."""
    a = np.asarray(chi, dtype=np.int64)
    return shells_from_chi(a[:, 1:] ^ a[:, :-1])


def omega_pairs(chi: np.ndarray) -> np.ndarray:
    """Adjacent tile chiralities as points of Ω = U × V. [N, 64] -> [N, 63]."""
    a = np.asarray(chi, dtype=np.int64)
    return ((a[:, :-1] & 63) << 6) | (a[:, 1:] & 63)


def permute_tiles(chi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Keep each row's chi multiset, scramble tile order."""
    a = np.asarray(chi, dtype=np.int64)
    n, t = a.shape
    # Argsort of random keys: one shuffle per row, no Python loop.
    keys = rng.random((n, t))
    return a[np.arange(n)[:, None], np.argsort(keys, axis=1)]


def q1_pair_mask(n_tiles: int = TILE) -> tuple[np.ndarray, np.ndarray]:
    """Adjacent pairs that sit inside one Q1_0 block vs across a block cut.

    Q1_0 stores 128 consecutive hidden coords (two 64-tiles) under one scale.
    Even-index pairs (0,1), (2,3), ... are inside a block. Odd-index pairs
    (1,2), (3,4), ... cross a block boundary.
    """
    idx = np.arange(n_tiles - 1)
    return idx % 2 == 0, idx % 2 == 1


def micro_frames(chi: np.ndarray) -> np.ndarray:
    """Depth-4 frames of family-0 bytes whose micro is tile chirality.

    [N, 64] chi -> [N * 16, 4] uint8. Super reads these as words.
    """
    a = np.asarray(chi, dtype=np.int64)
    n, t = a.shape
    if t % 4:
        raise ValueError("tile count must be divisible by 4")
    frames = a.reshape(-1, 4)
    out = np.empty(frames.shape, dtype=np.uint8)
    for i in range(len(frames)):
        for j in range(4):
            out[i, j] = byte_from_family_micro(0, int(frames[i, j]), D)
    return out
