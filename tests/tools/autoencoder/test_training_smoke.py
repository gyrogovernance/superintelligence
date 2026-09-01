"""Reproducible CPU smoke training (spec section 11, acceptance 6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.tools.autoencoder.helpers.training_losses import (
    LossWeights,
    weighted_total,
)
from src.tools.autoencoder.helpers.training_run import (
    EarlyStoppingCallback,
    HistoryCallback,
    TrainConfig,
    Trainer,
    ValidationCallback,
    iterate_batches,
    set_seed,
)
from src.tools.autoencoder.models.narrow import MLPAutoencoder


@pytest.fixture(scope="module")
def state_array() -> np.ndarray:
    return np.arange(4096, dtype=np.int64)


def _make_loss_fn(model: MLPAutoencoder, weights: LossWeights):
    def loss_fn(batch: dict) -> tuple[torch.Tensor, dict]:
        idx = batch["state_index"]
        logits = model(idx)
        ce = torch.nn.functional.cross_entropy(logits, idx)
        components = {"state_ce": ce}
        total, logs = weighted_total(components, weights)
        return total, logs

    return loss_fn


def test_smoke_training_reduces_loss(tmp_path, state_array) -> None:
    set_seed(0)
    config = TrainConfig(epochs=3, batch_size=256, device="cpu", seed=0,
                         checkpoint_dir=str(tmp_path))
    model = MLPAutoencoder(latent_dim=8, hidden_dim=64)
    trainer = Trainer(model, config)
    history = HistoryCallback()
    arrays = {"state_index": state_array}
    batches = lambda: iterate_batches(arrays, config.batch_size, config.seed)
    loss_fn = _make_loss_fn(model, LossWeights(state_ce=1.0))

    # initial loss for reference
    initial = trainer.evaluate(batches(), loss_fn)
    stats = trainer.fit(batches(), loss_fn, callbacks=[history])
    final = trainer.evaluate(batches(), loss_fn)

    assert stats["epochs_run"] == 3
    assert len(history.history["state_ce"]) == 3
    assert final < initial


def test_training_is_deterministic(tmp_path, state_array) -> None:
    def run() -> list[float]:
        set_seed(42)
        config = TrainConfig(epochs=2, batch_size=512, device="cpu", seed=42,
                             checkpoint_dir=str(tmp_path / "run"))
        model = MLPAutoencoder(latent_dim=4, hidden_dim=32)
        trainer = Trainer(model, config)
        history = HistoryCallback()
        arrays = {"state_index": state_array}
        trainer.fit(
            iterate_batches(arrays, config.batch_size, config.seed),
            _make_loss_fn(model, LossWeights(state_ce=1.0)),
            callbacks=[history],
        )
        return history.history["state_ce"]

    assert run() == run()


def test_checkpoint_roundtrip(tmp_path, state_array) -> None:
    set_seed(1)
    config = TrainConfig(epochs=1, batch_size=512, device="cpu", seed=1,
                         checkpoint_dir=str(tmp_path))
    model = MLPAutoencoder(latent_dim=4, hidden_dim=32)
    trainer = Trainer(model, config)
    arrays = {"state_index": state_array}
    trainer.fit(
        iterate_batches(arrays, config.batch_size, config.seed),
        _make_loss_fn(model, LossWeights(state_ce=1.0)),
    )
    path = trainer.save_checkpoint("smoke", extra={"note": "phase2"})
    fresh = Trainer(MLPAutoencoder(latent_dim=4, hidden_dim=32), config)
    payload = fresh.load_checkpoint(path)
    assert payload["extra"]["note"] == "phase2"

    # identical outputs after loading
    probe = {"state_index": state_array[:64]}
    torch.manual_seed(0)
    out_a = trainer.model(torch.as_tensor(probe["state_index"]))
    out_b = fresh.model(torch.as_tensor(probe["state_index"]))
    assert torch.allclose(out_a, out_b)


def test_checkpoint_roundtrip_unified_via_metadata(tmp_path, state_array) -> None:
    """Regression: unified-full checkpoints must reload with the exact
    constructor config recorded in ``extra["model_config"]``, not a hand-
    maintained parallel copy of defaults."""
    from src.tools.autoencoder.helpers.evals_run import load_any_checkpoint
    from src.tools.autoencoder.models import UnifiedAutoencoder

    set_seed(7)
    config = TrainConfig(epochs=1, batch_size=512, device="cpu", seed=7,
                         checkpoint_dir=str(tmp_path))
    model = UnifiedAutoencoder(symmetry="full", ladder="full")
    trainer = Trainer(model, config)
    arrays = {"state_index": state_array}

    def loss_fn(batch):
        idx = batch["state_index"]
        logits = model(idx)
        ce = torch.nn.functional.cross_entropy(logits, idx)
        total, logs = weighted_total(
            {"state_ce": ce}, LossWeights(state_ce=1.0)
        )
        return total, logs

    trainer.fit(
        iterate_batches(arrays, config.batch_size, config.seed),
        loss_fn,
    )
    extra = {
        "model_config": model.get_config(),
        "model_kind": "unified",
        "symmetry": "full",
    }
    path = trainer.save_checkpoint("unified_smoke", extra=extra)

    reloaded, meta = load_any_checkpoint(path, device="cpu")
    assert reloaded.get_config() == model.get_config()
    probe = torch.as_tensor(state_array[:64])
    torch.manual_seed(0)
    out_a = model(probe)
    out_b = reloaded(probe)
    assert torch.allclose(out_a, out_b)


def test_validation_and_early_stopping(tmp_path, state_array) -> None:
    set_seed(2)
    config = TrainConfig(epochs=50, batch_size=1024, device="cpu", seed=2,
                         checkpoint_dir=str(tmp_path))
    model = MLPAutoencoder(latent_dim=2, hidden_dim=16)
    trainer = Trainer(model, config)
    arrays = {"state_index": state_array}
    loss_fn = _make_loss_fn(model, LossWeights(state_ce=1.0))
    val = ValidationCallback(
        lambda loader: trainer.evaluate(loader, loss_fn),
        iterate_batches(arrays, 1024, 7, shuffle=False),
        epoch_interval=1,
    )
    early = EarlyStoppingCallback(monitor="val_loss", patience=3)
    stats = trainer.fit(
        iterate_batches(arrays, config.batch_size, config.seed),
        loss_fn,
        callbacks=[val, early],
    )
    assert stats["epochs_run"] < 50
    assert len(val.history["val_loss"]) == stats["epochs_run"]


def test_jsonl_log_written(tmp_path, state_array) -> None:
    config = TrainConfig(epochs=1, batch_size=512, device="cpu", seed=3,
                         checkpoint_dir=str(tmp_path))
    model = MLPAutoencoder(latent_dim=2, hidden_dim=16)
    trainer = Trainer(model, config)
    arrays = {"state_index": state_array}
    trainer.fit(
        iterate_batches(arrays, config.batch_size, config.seed),
        _make_loss_fn(model, LossWeights(state_ce=1.0)),
    )
    log = Path(tmp_path) / "train_log.jsonl"
    assert log.exists()
    import json

    lines = [json.loads(line) for line in log.read_text().strip().splitlines()]
    assert len(lines) == 1
    assert "state_ce" in lines[0]
