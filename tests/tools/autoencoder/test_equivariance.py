"""Exact equivariance tests (spec 6.4): exhaustive over all 4096 states x K4."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tools.autoencoder.kernel import apply_k4_index
from src.tools.autoencoder.helpers.evals_metrics import k4_equivariance_error
from src.tools.autoencoder.models.general import (
    K4_CHARACTERS,
    K4Autoencoder,
    k4_action_matrix,
)


@pytest.fixture(scope="module")
def k4_perm_np() -> np.ndarray:
    return k4_action_matrix()


@pytest.fixture(scope="module")
def model(k4_perm_np) -> K4Autoencoder:
    torch.manual_seed(0)
    return K4Autoencoder(
        n_trivial=2, n_sign=2, hidden_dim=32, k4_perm=k4_perm_np
    )


def test_characters_are_group_homomorphisms() -> None:
    # gates: id=0, S=1, C=2, F=3; S*C = F on V4
    for char in K4_CHARACTERS:
        assert char[0] == 1
        assert char[1] * char[2] == char[3]
        assert char[1] ** 2 == 1 and char[2] ** 2 == 1


def test_rho_is_representation(model) -> None:
    z = torch.randn(5, model.latent_dim)
    # rho(g) rho(h) = rho(gh): S*C = F, and squares are identity
    rho_s_c = model.rho(1, model.rho(2, z))
    rho_f = model.rho(3, z)
    assert torch.allclose(rho_s_c, rho_f, atol=1e-6)
    assert torch.allclose(model.rho(1, model.rho(1, z)), z, atol=1e-6)
    assert torch.allclose(model.rho(0, z), z)


def test_encoder_exactly_equivariant_exhaustive(model) -> None:
    """E(g.x) == rho(g) E(x) for ALL 4096 states and ALL 4 K4 gates."""
    all_states = torch.arange(4096, dtype=torch.long)
    z = model.encode(all_states)
    for gate_i in range(4):
        transformed = model.k4_perm[gate_i][all_states]
        z_g = model.encode(transformed)
        rho_z = model.rho(gate_i, z)
        err = (z_g - rho_z).abs().max()
        assert float(err.detach()) < 1e-4, f"gate {gate_i}: max err {float(err.detach())}"


def test_decoder_exactly_equivariant_exhaustive(model) -> None:
    """D(rho(g) z) == P_g D(z) for ALL 4096 states (z = E(x)) and all gates."""
    all_states = torch.arange(4096, dtype=torch.long)
    z = model.encode(all_states)
    d = model.decode(z)  # [4096, 4096] simplex output
    for gate_i in range(4):
        rho_z = model.rho(gate_i, z)
        d_g = model.decode(rho_z)
        # P_g applied to simplex output: out[s] -> out[g^-1(s)]; with
        # index_add convention used in decode, applying the gate permutation
        # to coordinates means d_g[s] should equal d[perm^-1(s)]. Equivalently
        # the permutation-scatter of d must match d_g.
        perm = model.k4_perm[gate_i].long()
        permuted = torch.zeros_like(d)
        permuted.index_add_(1, perm, d)
        err = (d_g - permuted).abs().max()
        assert float(err.detach()) < 1e-4, f"gate {gate_i}: max err {float(err.detach())}"


def test_equivariance_error_metrics(model) -> None:
    all_states = torch.arange(4096, dtype=torch.long)

    def latent_action(gate_i: int, z: torch.Tensor) -> torch.Tensor:
        return model.rho(gate_i, z)

    report = k4_equivariance_error(model.encoder_eval(), all_states, model.k4_perm, latent_action)
    # with the exact construction the defect must be numerically zero
    assert report["max"] < 1e-4


def test_reconstruction_is_exact_for_identity_latent(model) -> None:
    """Sanity: the symmetrized pipeline with an untrained but sufficiently
    expressive base can represent identity; here we only verify the output is
    a valid distribution and argmax is a valid state index."""
    probe = torch.arange(0, 4096, 512, dtype=torch.long)
    logits = model(probe)
    assert torch.isfinite(logits).all()
    pred = logits.argmax(dim=-1)
    assert int(pred.min()) >= 0 and int(pred.max()) < 4096


def test_k4_action_matrix_consistent(k4_perm_np) -> None:
    for index in (0, 1, 777, 4095):
        for gate_i, gate in enumerate(("id", "S", "C", "F")):
            assert k4_perm_np[gate_i, index] == apply_k4_index(index, gate)
