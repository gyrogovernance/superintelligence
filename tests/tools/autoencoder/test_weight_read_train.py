"""Train Walsh sector gains on activation corpus."""

from __future__ import annotations

import numpy as np
import torch

from src.tools.autoencoder.helpers.weight_read import HIDDEN, evaluate_on_corpus, read_weight
from src.tools.autoencoder.helpers.weight_read_train import train_weight_sectors
from src.tools.autoencoder.models.super import SpectralAutoencoder


def test_apply_pq_activation_matches_numpy() -> None:
    model = SpectralAutoencoder()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, HIDDEN)).astype(np.float32)
    with torch.no_grad():
        y = model.apply_pq_activation(torch.as_tensor(x)).numpy()
    from src.tools.autoencoder.helpers.weight_read import apply_pq

    g = model.bottleneck.block_gains().detach().cpu().numpy()
    for i in range(4):
        y0 = apply_pq(g, x[i])
        assert np.max(np.abs(y0 - y[i])) < 1e-4


def test_train_recovers_pure_operative_weight() -> None:
    """W in the PQ class should train to low activation error."""
    from src.tools.autoencoder.helpers.weight_read import (
        N_BLOCKS,
        block_id_flat,
        walsh_inverse_matrix_4096,
        walsh_matrix_4096,
    )

    rng = np.random.default_rng(2)
    gains_true = rng.standard_normal(N_BLOCKS) * 0.2
    bid = block_id_flat()
    G = walsh_inverse_matrix_4096()
    T = walsh_matrix_4096()
    Pc = np.zeros((HIDDEN, HIDDEN))
    for k in range(N_BLOCKS):
        idx = np.nonzero(bid == k)[0]
        Pc[np.ix_(idx, idx)] = gains_true[k] * np.eye(len(idx))
    W = G @ Pc @ T
    corpus = rng.standard_normal((64, HIDDEN))
    _, stats = train_weight_sectors(W, corpus, epochs=120, lr=0.05, seed=2)
    assert stats["final_pq_max_rel"] < 0.15
