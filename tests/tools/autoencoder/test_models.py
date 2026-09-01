"""Model unit tests (spec 6.2, 6.3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tools.autoencoder import kernel
from src.tools.autoencoder.models.narrow import (
    MLPAutoencoder,
    k4_generator_batch,
    soft_equivariance_loss,
    state_to_bits,
)


@pytest.fixture(scope="module")
def k4_perm() -> torch.Tensor:
    action, _ = kernel.k4_action_arrays()
    return torch.as_tensor(action.astype(np.int64))


def test_state_to_bits_roundtrip() -> None:
    idx = torch.tensor([0, 1, 64, 4095], dtype=torch.long)
    bits = state_to_bits(idx)
    assert bits.shape == (4, 12)
    # reconstruct index from bits
    u = bits[:, :6]
    v = bits[:, 6:]
    u_val = (u * torch.tensor([1, 2, 4, 8, 16, 32])).sum(dim=-1).long()
    v_val = (v * torch.tensor([1, 2, 4, 8, 16, 32])).sum(dim=-1).long()
    assert torch.equal(u_val * 64 + v_val, idx)


def test_mlp_forward_shapes() -> None:
    model = MLPAutoencoder(latent_dim=8, hidden_dim=32)
    idx = torch.arange(16, dtype=torch.long)
    z = model.encoder(idx)
    logits = model.decoder(z)
    assert z.shape == (16, 8)
    assert logits.shape == (16, 4096)


def test_k4_generator_batch_matches_kernel(k4_perm) -> None:
    idx = torch.tensor([0, 777, 2089, 4095], dtype=torch.long)
    transformed = k4_generator_batch(idx, k4_perm)
    for gate_i, gate in enumerate(("id", "S", "C", "F")):
        for b in range(len(idx)):
            expected = kernel.apply_k4_index(int(idx[b]), gate)
            assert int(transformed[gate_i][b]) == expected


def test_soft_equivariance_loss_scalar_latent_finite(k4_perm) -> None:
    model = MLPAutoencoder(latent_dim=1, hidden_dim=16)
    idx = torch.arange(0, 4096, 64, dtype=torch.long)
    # a scalar sign representation: trivial on gate 0, sign flips on the
    # nontrivial K4 elements (all involutions in the character table)
    def latent_action(gate_i, z):
        signs = torch.tensor([1.0, -1.0, -1.0, -1.0])
        return signs[gate_i] * z

    loss = soft_equivariance_loss(model.encoder, idx, k4_perm, latent_action)
    assert torch.isfinite(loss)
    assert float(loss.detach()) >= 0.0


def test_soft_equivariance_loss_requires_latent_action(k4_perm) -> None:
    model = MLPAutoencoder(latent_dim=4, hidden_dim=16)
    idx = torch.arange(0, 256, 16, dtype=torch.long)
    with pytest.raises(ValueError, match="latent_action"):
        soft_equivariance_loss(model.encoder, idx, k4_perm)
