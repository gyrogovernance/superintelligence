"""Tests for named latent components and transition-model equivariance
metric."""

from __future__ import annotations

import numpy as np
import torch

from src import api

from src.tools.autoencoder.helpers.evals_metrics import transition_k4_equivariance_error
from src.tools.autoencoder.models.general import K4Autoencoder
from src.tools.autoencoder.models.narrow import TransitionModel


def test_named_latent_components_cover_latent() -> None:
    model = K4Autoencoder(n_trivial=3, n_sign=2)
    slices = model.z_slices
    assert set(slices) == {"z_inv", "z_chi", "z_shell", "z_irrep"}
    # slices tile the latent exactly, in order
    assert slices["z_inv"] == slice(0, 3)
    assert slices["z_chi"] == slice(3, 5)
    assert slices["z_shell"] == slice(5, 7)
    assert slices["z_irrep"] == slice(7, 9)
    assert model.latent_dim == 9


def test_named_components_shapes_and_equivariance() -> None:
    torch.manual_seed(0)
    model = K4Autoencoder(n_trivial=2, n_sign=2)
    idx = torch.arange(0, 4096, 97)
    parts = model.named_components(idx)
    assert set(parts) == {"z_inv", "z_chi", "z_shell", "z_irrep"}
    for part in parts.values():
        assert part.shape == (len(idx), 2)
    # the invariant block must be exactly K4-invariant
    with torch.no_grad():
        for gate_i in range(4):
            moved = model.k4_perm[gate_i][idx]
            parts_moved = model.named_components(moved)
            err = (parts_moved["z_inv"] - parts["z_inv"]).abs().max()
            assert float(err) < 1e-5, gate_i


def _k4_state_perm() -> torch.Tensor:
    """[4, 4096] exact K4 permutation on state indices from the kernel."""
    perm = np.empty((4, 4096), dtype=np.int64)
    for i in range(4096):
        s24 = api.omega12_to_state24(api.OmegaState12(u6=(i >> 6) & 63, v6=i & 63))
        for gate_i, gate in enumerate(("id", "S", "C", "F")):
            d = api.state24_to_omega12(api.apply_gate(s24, gate))
            perm[gate_i, i] = (d.u6 << 6) | d.v6
    return torch.as_tensor(perm)


def _byte_shadow_perm() -> torch.Tensor:
    """[4, 256] byte-side action: shadow partner for F, identity otherwise."""
    return torch.stack(
        [
            torch.arange(256),
            torch.arange(256),
            torch.arange(256),
            torch.tensor([api.shadow_partner_byte(b) for b in range(256)]),
        ]
    )


def test_transition_equivariance_zero_for_exact_kernel() -> None:
    """The exact kernel transition must show zero defect under (F, shadow)."""
    next_table = np.empty((4096, 256), dtype=np.int64)
    for i in range(4096):
        s24 = api.omega12_to_state24(api.OmegaState12(u6=(i >> 6) & 63, v6=i & 63))
        for b in range(256):
            d = api.step_omega12_by_byte(api.state24_to_omega12(s24), b)
            next_table[i, b] = (d.u6 << 6) | d.v6

    class ExactKernel:
        def eval(self) -> None:
            pass

        def __call__(self, state_index: torch.Tensor, byte: torch.Tensor) -> torch.Tensor:
            targets = torch.as_tensor(next_table[state_index.numpy(), byte.numpy()])
            out = torch.zeros((targets.shape[0], 4096), dtype=torch.float32)
            out[torch.arange(targets.shape[0]), targets] = 1.0
            return out

    model = ExactKernel()
    k4_perm = _k4_state_perm()
    byte_perm = _byte_shadow_perm()
    idx = torch.arange(16) * 257
    byte = torch.arange(16) * 13
    out = transition_k4_equivariance_error(
        model, idx, byte, k4_perm, byte_perm, gate_pair=(3, 0)
    )
    assert out["max"] == 0.0 and out["mean"] == 0.0


def test_transition_equivariance_detects_untrained_model() -> None:
    """An untrained MLP must show a nonzero defect (the metric is sensitive)."""
    torch.manual_seed(1)
    k4_perm = _k4_state_perm()
    byte_perm = _byte_shadow_perm()
    model = TransitionModel(hidden_dim=32)
    idx = torch.arange(16) * 257
    byte = torch.arange(16) * 13
    out = transition_k4_equivariance_error(model, idx, byte, k4_perm, byte_perm)
    assert out["max"] > 0.0
