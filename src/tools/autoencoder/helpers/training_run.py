"""Training loop with a four-hook callback protocol, checkpointing, and
JSONL logging."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import torch

from .training_losses import LossWeights, weighted_total


# ---------------------------------------------------------------------------
# Callback protocol
# ---------------------------------------------------------------------------


class Callback:
    def start_of_training(self) -> None: ...
    def start_of_epoch(self, epoch: int) -> None: ...
    def end_of_epoch(self, epoch: int, logs: dict) -> None: ...
    def end_of_training(self) -> None: ...


class HistoryCallback(Callback):
    """Accumulates per-epoch metric history."""

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = {}

    def end_of_epoch(self, epoch: int, logs: dict) -> None:
        for key, value in logs.items():
            self.history.setdefault(key, []).append(float(value))


class ValidationCallback(Callback):
    """Evaluates validation loss every epoch_interval epochs."""

    def __init__(self, evaluate_fn, val_loader, epoch_interval: int = 10) -> None:
        self.evaluate_fn = evaluate_fn
        self.val_loader = val_loader
        self.epoch_interval = epoch_interval
        self.history: dict[str, list[float]] = {"val_epoch": [], "val_loss": []}

    def end_of_epoch(self, epoch: int, logs: dict) -> None:
        if (epoch + 1) % self.epoch_interval == 0:
            loss = self.evaluate_fn(self.val_loader)
            self.history["val_epoch"].append(epoch)
            self.history["val_loss"].append(float(loss))
            logs["val_loss"] = float(loss)


class EarlyStoppingCallback(Callback):
    def __init__(self, monitor: str = "val_loss", patience: int = 10, min_delta: float = 0.0) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.wait = 0
        self.stop = False

    def end_of_epoch(self, epoch: int, logs: dict) -> None:
        value = logs.get(self.monitor)
        if value is None:
            return
        if value < self.best - self.min_delta:
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop = True


class BestCheckpointCallback(Callback):
    """Saves the model weights whenever the monitored metric improves.

    The monitor value is read from the epoch logs (e.g. ``val_loss``), so the
    callback eagerly waits for the validation signal before writing the first
    checkpoint. The best weights are saved as ``{name}.best.pt`` alongside the
    regular ``{name}.pt`` output, with a small sidecar JSON.
    """

    def __init__(self, model: torch.nn.Module, path: Path, monitor: str = "val_loss") -> None:
        self.model = model
        self.path = Path(path)
        self.monitor = monitor
        self.best = float("inf")
        self.saved_at = -1

    def end_of_epoch(self, epoch: int, logs: dict) -> None:
        value = logs.get(self.monitor)
        if value is None:
            return
        if value < self.best:
            self.best = float(value)
            self.saved_at = epoch
            torch.save(self.model.state_dict(), self.path)
            sidecar = self.path.with_suffix(".json")
            sidecar.write_text(
                json.dumps(
                    {
                        "monitor": self.monitor,
                        "best": float(value),
                        "epoch": epoch,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )


# ---------------------------------------------------------------------------
# Config and trainer
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    seed: int = 0
    gradient_clip: float | None = None
    checkpoint_dir: str = "checkpoints"
    log_file: str | None = None
    loss_weights: LossWeights = field(default_factory=LossWeights)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def iterate_batches(
    arrays: dict[str, np.ndarray], batch_size: int, seed: int, shuffle: bool = True
) -> Iterator[dict[str, torch.Tensor]]:
    """Yields tensor batches from aligned numpy arrays of equal length."""
    n = len(next(iter(arrays.values())))
    order = np.arange(n)
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(order)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield {
            # cast integer arrays to long: uint16/uint8 numpy arrays would
            # otherwise produce uint tensors that cross_entropy and indexing
            # reject. Float arrays (one-hot or continuous targets) pass
            # through untouched so future denoising/multi-task targets are
            # not silently truncated.
            name: (
                torch.as_tensor(arr[idx])
                if np.issubdtype(arr.dtype, np.floating)
                else torch.as_tensor(arr[idx]).long()
            )
            for name, arr in arrays.items()
        }


class Trainer:
    """Generic torch trainer: optimizer, callbacks, checkpointing, JSONL logs."""

    def __init__(self, model: torch.nn.Module, config: TrainConfig) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.callbacks: list[Callback] = []

    def fit(
        self,
        batch_iter: "Callable[[], Iterable[Any]] | Iterable[Any]",
        loss_fn: "Callable[..., Any]",
        callbacks: list[Callback] | None = None,
    ) -> dict:
        """batch_iter is either a callable returning a fresh batch iterator per
        epoch, or a materialized sequence of batches (reused every epoch).
        loss_fn(batch) -> (total, logs)."""
        self.callbacks = list(callbacks or [])
        log_path = (
            Path(self.config.checkpoint_dir) / "train_log.jsonl"
            if self.config.log_file is None and self.config.checkpoint_dir
            else Path(self.config.log_file) if self.config.log_file else None
        )
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

        if callable(batch_iter):
            epoch_batches: Callable[[], Iterable[Any]] = lambda: batch_iter()
        else:
            materialized = list(batch_iter)
            epoch_batches = lambda: iter(materialized)

        for cb in self.callbacks:
            cb.start_of_training()
        epoch = 0
        for epoch in range(self.config.epochs):
            self.model.train()
            for cb in self.callbacks:
                cb.start_of_epoch(epoch)
            epoch_logs: dict[str, Any] = {}
            for batch in epoch_batches():
                batch = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                self.optimizer.zero_grad(set_to_none=True)
                total, logs = loss_fn(batch)
                total.backward()
                if self.config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                self.optimizer.step()
                for key, value in logs.items():
                    epoch_logs.setdefault(key, []).append(value)
            epoch_logs = {
                k: float(np.mean(v)) if isinstance(v, list) else float(v)
                for k, v in epoch_logs.items()
            }
            epoch_logs["epoch"] = epoch
            for cb in self.callbacks:
                cb.end_of_epoch(epoch, epoch_logs)
            if log_path is not None:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(epoch_logs) + "\n")
            if any(getattr(cb, "stop", False) for cb in self.callbacks):
                break
        for cb in self.callbacks:
            cb.end_of_training()
        return {"epochs_run": epoch + 1}

    @torch.inference_mode()
    def evaluate(
        self,
        batch_iter: "Callable[[], Iterable[Any]] | Iterable[Any]",
        loss_fn: "Callable[..., Any]",
    ) -> float:
        """Loss over a batch source."""
        self.model.eval()
        if callable(batch_iter):
            batches = batch_iter()
        elif hasattr(batch_iter, "__iter__"):
            batches = batch_iter
        else:
            raise TypeError(
                "evaluate() expects a callable returning an iterable, or an "
                f"iterable; got {type(batch_iter).__name__}"
            )
        values = []
        for batch in batches:
            batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            total, _ = loss_fn(batch)
            values.append(float(total))
        return float(np.mean(values)) if values else float("nan")

    def save_checkpoint(self, name: str, extra: dict | None = None) -> Path:
        out_dir = Path(self.config.checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "config": asdict(self.config),
            "extra": extra or {},
            "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        path = out_dir / f"{name}.pt"
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: Path) -> dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"])
        return payload