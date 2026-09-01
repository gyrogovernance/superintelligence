"""Seed threading: the percolation eval path must be reproducible by seed."""

from __future__ import annotations

import torch

from src.tools.autoencoder.helpers.evals_run import evaluate_percolation_accuracy
from src.tools.autoencoder.models.narrow import PercolationLearner


def _model() -> PercolationLearner:
    torch.manual_seed(0)
    return PercolationLearner(hidden_dim=64)


def test_percolation_eval_seed_deterministic() -> None:
    model = _model()
    r1 = evaluate_percolation_accuracy(model, seed=11)
    r2 = evaluate_percolation_accuracy(model, seed=11)
    assert r1 == r2


def test_percolation_eval_seed_changes_holdout() -> None:
    model = _model()
    r1 = evaluate_percolation_accuracy(model, seed=11)
    r2 = evaluate_percolation_accuracy(model, seed=12)
    # Different seeds select different held-out rows; the report's n_samples
    # is the same size, but the underlying rows differ. The accuracy may or
    # may not be the same, but the held-out row indices must differ.
    from src.tools.autoencoder.helpers.evals_datasets import percolation_dataset

    a = percolation_dataset(seed=11)
    b = percolation_dataset(seed=12)
    import numpy as np

    assert not np.array_equal(a["transport_rank"], b["transport_rank"])
    assert r1["n_samples"] == r2["n_samples"]
