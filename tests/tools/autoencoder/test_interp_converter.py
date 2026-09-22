"""Converter primitives: 64-tile chirality and Ω pair embedding."""

from __future__ import annotations

import numpy as np

from src.tools.autoencoder.models.general import walsh_matrix_64
from src.tools.autoencoder.programs.interpretability.converter import (
    chirality_from_tiles,
    energy_shell_hist,
    omega_pairs,
    permute_tiles,
    q1_pair_mask,
    shells_from_chi,
    tile_mode_energy,
    transport_shells,
)


def test_chirality_recovers_walsh_character():
    w = walsh_matrix_64().astype(np.float64)
    tiles = w[None, :, :]
    chi = chirality_from_tiles(tiles)
    assert chi.shape == (1, 64)
    assert np.array_equal(chi[0], np.arange(64))


def test_omega_pair_is_uv_chart():
    chi = np.array([[3, 5, 8]], dtype=np.int64)
    omega = omega_pairs(chi)
    assert omega.shape == (1, 2)
    assert int(omega[0, 0]) == (3 << 6) | 5
    assert int(omega[0, 1]) == (5 << 6) | 8


def test_transport_shell_is_xor_weight():
    chi = np.array([[0b000111, 0b000001]], dtype=np.int64)
    tr = transport_shells(chi)
    assert int(tr[0, 0]) == 2
    assert int(shells_from_chi(chi)[0, 0]) == 3


def test_q1_pair_mask_splits_even_odd():
    within, across = q1_pair_mask(64)
    assert int(within.sum()) == 32
    assert int(across.sum()) == 31
    assert bool(within[0]) and not bool(across[0])
    assert bool(across[1]) and not bool(within[1])


def test_walsh_character_energy_is_a_spike():
    w = walsh_matrix_64().astype(np.float64)
    energy = tile_mode_energy(w[None, :, :])
    assert energy.shape == (1, 64, 64)
    assert np.allclose(energy.sum(axis=-1), 1.0)
    peaks = np.argmax(energy[0], axis=-1)
    assert np.array_equal(peaks, np.arange(64))
    hist = energy_shell_hist(energy)
    from math import comb

    census = np.array([comb(6, s) / 64.0 for s in range(7)])
    assert np.allclose(hist, census, atol=1e-12)


def test_random_sign_energy_matches_census():
    rng = np.random.default_rng(0)
    tiles = rng.choice([-1.0, 1.0], size=(64, 64, 64))
    hist = energy_shell_hist(tile_mode_energy(tiles))
    from math import comb

    census = np.array([comb(6, s) / 64.0 for s in range(7)])
    tv = 0.5 * float(np.abs(hist - census).sum())
    assert tv < 0.02


def test_permute_tiles_preserves_multiset():
    rng = np.random.default_rng(0)
    chi = np.arange(64, dtype=np.int64).reshape(1, 64)
    perm = permute_tiles(chi, rng)
    assert sorted(perm[0].tolist()) == list(range(64))
    assert not np.array_equal(perm[0], chi[0])
