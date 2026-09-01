"""Checkpoint round-trip: ``model.get_config()`` must reconstruct an identical
architecture, including the K4 latent layout (n_trivial, n_sign)."""

from __future__ import annotations

import pytest
import torch

from src.tools.autoencoder.helpers.evals_run import load_any_checkpoint
from src.tools.autoencoder.models import build_model
from src.tools.autoencoder.models.narrow import (
    PercolationLearner,
    RawByteTransitionModel,
    TransitionModel,
    WordActionModel,
)


def _save_and_reload(model, kind: str, tmp_path) -> torch.nn.Module:
    """Save a checkpoint with the same Trainer payload shape used in production."""
    path = tmp_path / f"{kind}.pt"
    payload = {
        "config": {},
        "extra": {
            "model_kind": kind,
            "model_config": model.get_config(),
        },
        "model_state": model.state_dict(),
    }
    torch.save(payload, path)
    reloaded, _ = load_any_checkpoint(path, device="cpu")
    return reloaded


@pytest.mark.parametrize(
    "model",
    [
        TransitionModel(hidden_dim=96),
        RawByteTransitionModel(hidden_dim=96),
        WordActionModel(hidden_dim=96),
        PercolationLearner(hidden_dim=96),
    ],
    ids=["transition", "rawbyte", "word", "percolation"],
)
def test_task_model_get_config_roundtrip(model, tmp_path) -> None:
    reloaded = _save_and_reload(model, "transition", tmp_path) if isinstance(
        model, TransitionModel
    ) else _save_and_reload(
        model,
        "rawbyte" if isinstance(model, RawByteTransitionModel) else
        "word" if isinstance(model, WordActionModel) else "percolation",
        tmp_path,
    )
    assert reloaded.get_config() == model.get_config()
    assert reloaded.state_dict().keys() == model.state_dict().keys()


@pytest.mark.parametrize("n_trivial, n_sign", [(2, 2), (4, 3), (1, 1)])
def test_k4_n_trivial_n_sign_roundtrip(n_trivial, n_sign, tmp_path) -> None:
    from src.tools.autoencoder.models.general import K4Autoencoder

    model = K4Autoencoder(n_trivial=n_trivial, n_sign=n_sign, hidden_dim=32)
    cfg = model.get_config()
    assert cfg == {"n_trivial": n_trivial, "n_sign": n_sign, "hidden_dim": 32}

    path = tmp_path / "k4.pt"
    torch.save(
        {
            "config": {},
            "extra": {
                "model_kind": "k4",
                "model_config": cfg,
            },
            "model_state": model.state_dict(),
        },
        path,
    )
    reloaded, _ = load_any_checkpoint(path, device="cpu")
    assert isinstance(reloaded, K4Autoencoder)
    assert reloaded.n_trivial == n_trivial
    assert reloaded.n_sign == n_sign
    assert reloaded.latent_dim == n_trivial + 3 * n_sign


def test_build_model_k4_forwards_n_trivial_n_sign() -> None:
    model = build_model("k4", n_trivial=5, n_sign=2, hidden_dim=24)
    assert model.n_trivial == 5
    assert model.n_sign == 2
    assert model.latent_dim == 5 + 3 * 2
