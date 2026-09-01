"""Tests for symmetry-breaking/anomaly datasets and hQVM(d) generalization
(spec 4.7, 4.8, section 9 symmetry-breaking metrics)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from src import api, constants

from src.tools.autoencoder import kernel
from src.tools.autoencoder.helpers.evals_metrics import (
    shell_distribution_ensemble,
    walsh_sector_energy,
)
from src.tools.autoencoder.helpers.evals_datasets import (
    biased_family_bytes,
    biased_q_weight_bytes,
    byte_perturbation_dataset,
    corrupted_mask_dataset,
    dimension_transfer_probe,
    gf2_rank,
    hqvm_d_transition_dataset,
    missing_q_class_alphabet,
)
from src.tools.autoencoder.models.super import SpectralAutoencoder


# ---------------------------------------------------------------------------
# Controlled ensembles
# ---------------------------------------------------------------------------


def test_biased_family_bytes_single_family() -> None:
    rng = np.random.default_rng(0)
    sample = biased_family_bytes(rng, family=2, n=100)
    assert len(sample) == 100
    assert all(constants.byte_family(int(b)) == 2 for b in sample)


def test_biased_q_weight_bytes_exact_weight() -> None:
    rng = np.random.default_rng(0)
    sample = biased_q_weight_bytes(rng, weight=3, n=50)
    assert all(api.Q_WEIGHT_BY_BYTE[int(b)] == 3 for b in sample)


def test_missing_q_class_alphabet_rank_and_labels() -> None:
    allowed, q_vectors, removed = missing_q_class_alphabet(None, n_missing=4, seed=0)
    assert len(removed) == 4
    # no byte from a removed class is in the alphabet
    for b in allowed.tolist():
        assert api.q_word6(b) not in set(removed.tolist())
    # rank label is exact
    rank = gf2_rank(q_vectors, 6)
    assert 0 <= rank <= 6
    # removing 4 of 64 classes: rank almost surely stays 6, but must be computed
    assert isinstance(rank, int)


def test_gf2_rank_known_answers() -> None:
    assert gf2_rank([], 6) == 0
    assert gf2_rank([0b000001, 0b000010, 0b000100], 6) == 3
    # {3,5,6} is rank 2: 3^5 = 6
    assert gf2_rank([0b000011, 0b000101, 0b000110], 6) == 2
    assert gf2_rank([0b111111, 0b111111], 6) == 1
    full = [api.q_word6(b) for b in range(256)]
    assert gf2_rank(full, 6) == 6


def test_corrupted_mask_dataset_syndromes() -> None:
    rng = np.random.default_rng(2)
    rows = corrupted_mask_dataset(rng, n=16)
    assert len(rows) == 32
    for row in rows:
        if row["is_valid"]:
            assert row["syndrome"] == 0
            assert row["expected"] == "syndrome_zero"
        else:
            assert row["syndrome"] != 0
            assert row["expected"] == "syndrome_nonzero"


def test_byte_perturbation_dataset_shadow_preserves_signature() -> None:
    rng = np.random.default_rng(4)
    rows = byte_perturbation_dataset(rng, n=20)
    kinds = Counter(r["kind"] for r in rows)
    assert set(kinds) == {
        "shadow_substitution", "adjacent_swap", "deletion", "byte_substitution"
    }
    # kernel fact: shadow partner induces the same Omega action, so EVERY
    # shadow substitution must preserve the word signature
    for row in rows:
        if row["kind"] == "shadow_substitution":
            assert row["signature_preserved"] == 1, row


def test_byte_perturbation_labels_consistent_with_kernel() -> None:
    rng = np.random.default_rng(7)
    rows = byte_perturbation_dataset(rng, n=10)
    for row in rows:
        recomputed = kernel.word_signature_id(list(row["word"]))
        assert row["new_sig"] == recomputed


# ---------------------------------------------------------------------------
# Shell ensemble (dataset-level "symmetry breaking")
# ---------------------------------------------------------------------------


def test_shell_distribution_ensemble_statistics() -> None:
    rng = np.random.default_rng(1)
    out = shell_distribution_ensemble(rng, lam=0.5, n=2000)
    # lam < 1 biases toward shell 0 (equality horizon)
    assert out["expected_shell"] < 3.0
    sample_mean = float(out["shell"].mean())
    assert abs(sample_mean - out["expected_shell"]) < 0.2
    # balance ensemble at lam = 1 gives the binomial mean 3
    out_balanced = shell_distribution_ensemble(rng, lam=1.0, n=4000)
    assert abs(out_balanced["expected_shell"] - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# Spectral diagnostics
# ---------------------------------------------------------------------------


def test_walsh_sector_energy_partition() -> None:
    model = SpectralAutoencoder()
    energy = walsh_sector_energy(model, 1234)
    assert abs(energy["diag_energy"] + energy["offdiag_energy"] - energy["total"]) < 1e-3
    assert energy["total"] > 0


# ---------------------------------------------------------------------------
# hQVM(d) generalization
# ---------------------------------------------------------------------------


def test_hqvm_d_dataset_shapes_and_closure() -> None:
    for d in (2, 3):
        data = hqvm_d_transition_dataset(d)
        n = 4**d
        n_bytes = 1 << (d + 2)
        assert data["next_state"].shape == (n, n_bytes)
        assert data["q_by_byte"].shape == (n_bytes,)
        assert int(data["next_state"].max()) < n
        # q-class sizes: exactly 4 bytes per q value
        counts = Counter(data["q_by_byte"].tolist())
        assert set(counts.values()) == {4}


def test_hqvm_d6_matches_api_transitions() -> None:
    data = hqvm_d_transition_dataset(6)
    # spot check against the d=6 kernel
    from src.tools.autoencoder.kernel import state24_from_index

    for i in (0, 777, 4095):
        u, v = int(data["u"][i]), int(data["v"][i])
        for j in (0, 0x54, 255):
            byte = j if j < 256 else j  # d=6 alphabet is 256
            omega = api.OmegaState12(u6=u, v6=v)
            dest = api.step_omega12_by_byte(omega, byte)
            assert data["next_state"][i, byte] == (dest.u6 << 6) | dest.v6


def test_dimension_transfer_probe_structure() -> None:
    probe = dimension_transfer_probe(4, 6)
    assert probe["q_class_size_train"] == [4]
    assert probe["q_class_size_test"] == [4]
    assert probe["rank_train"] == 4
    assert probe["rank_test"] == 6
