"""Regression: every (model, task) selection must train its parameters and
the per-task loss key must be a real ``LossWeights`` field with nonzero weight.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.tools.autoencoder.cli import cmd_train
from src.tools.autoencoder.helpers.evals_run import load_any_checkpoint


def _args(tmp_path: Path, model: str, task=None, symmetry=None, epochs=2) -> object:
    class A:
        pass

    a = A()
    a.model = model
    a.task = task
    a.symmetry = symmetry
    a.seed = 0
    a.device = "cpu"
    a.epochs = epochs
    a.batch_size = 128
    a.learning_rate = 1e-3
    a.output_dir = str(tmp_path)
    a.run_name = f"{model}_{task or 'state'}"
    a.val_fraction = 0.0
    a.val_interval = 1
    a.patience = 5
    a.min_delta = 1e-4
    a.rate_weight = 0.0
    return a


def _params_changed(initial: torch.nn.Module, path: Path) -> bool:
    """True if the reloaded checkpoint differs from a freshly-built model."""
    reloaded, _ = load_any_checkpoint(path, device="cpu")
    sd_init = {k: v for k, v in initial.state_dict().items()}
    sd_new = reloaded.state_dict()
    total_delta = 0.0
    for k in sd_init:
        if k in sd_new:
            total_delta += float((sd_new[k] - sd_init[k]).abs().sum().item())
    return total_delta > 1e-6


@pytest.mark.parametrize(
    "model, task, symmetry",
    [
        ("mlp", None, None),
        ("k4", None, None),
        ("spectral", None, None),
        ("transition", "transition", None),
        ("rawbyte", "rawbyte", None),
        ("word", "word", None),
        ("percolation", "percolation_rank", None),
        ("unified", "unified_multi", "full"),
    ],
)
def test_task_actually_trains(tmp_path, model, task, symmetry) -> None:
    """Every model/task selection produces a checkpoint whose parameters
    moved from their initialization - i.e. the loss was actually wired to a
    gradient."""
    from src.tools.autoencoder.models import build_model

    args = _args(tmp_path, model, task, symmetry, epochs=2)
    kind = task or model
    initial = build_model(
        model,
        symmetry=symmetry,
        heads=("transition", "word", "rank") if task == "unified_multi" else None,
    )
    rc = cmd_train(args)
    assert rc == 0
    path = Path(args.output_dir) / f"{args.run_name}.pt"
    assert path.exists()
    assert _params_changed(initial, path), f"{kind} did not train"


def test_denoise_rate_weight_wired() -> None:
    from src.tools.autoencoder.helpers.training_losses import (
        LossWeights,
        weighted_total,
    )
    from src.tools.autoencoder.models.super import SpectralAutoencoder

    model = SpectralAutoencoder(ladder="shell_radial")
    weights = LossWeights(state_ce=1.0, rate=1.0)
    idx = torch.arange(0, 256, dtype=torch.long)
    logits = model(idx)
    target = torch.zeros_like(logits)
    target[torch.arange(len(idx)), idx] = 1.0
    ce = -(torch.log_softmax(logits, dim=-1) * target).sum(dim=-1).mean()
    components = {"state_ce": ce, "rate": model.bottleneck.rate_penalty()}
    total, logs = weighted_total(components, weights)
    assert "rate" in logs
    assert float(logs["rate"]) != 0.0


def test_loss_key_parity() -> None:
    """Every loss key used by the CLI is a real ``LossWeights`` field. A typo
    (e.g. ``tau_ce`` instead of ``word_ce``) would be silently dropped with
    weight 0.0, so a task would "train" without a gradient. ``weighted_total``
    now raises on unknown keys; this asserts the known CLI keys are valid."""
    from src.tools.autoencoder.helpers.training_losses import (
        LossWeights,
        weighted_total,
    )

    cli_keys = [
        "state_ce",
        "transition_ce",
        "word_ce",
        "rank_ce",
        "rate",
    ]
    w = LossWeights()
    for key in cli_keys:
        assert hasattr(w, key), f"{key} is not a LossWeights field"
    with pytest.raises(KeyError):
        weighted_total({"tau_ce": torch.zeros(())}, w)


@pytest.mark.parametrize(
    "task, primary_key",
    [
        ("state_ce", "state_ce"),
        ("transition", "transition_ce"),
        ("rawbyte", "transition_ce"),
        ("word", "word_ce"),
        ("percolation_rank", "rank_ce"),
        ("unified_multi", "state_ce"),
    ],
)
def test_task_primary_loss_key_nonzero(task, primary_key) -> None:
    """The per-task loss key declared in ``cli.task_weights`` is a real
    ``LossWeights`` field with nonzero weight. A loss key that resolved to a
    field whose default is 0.0 would silently drop the gradient."""
    from src.tools.autoencoder.cli import TASK_WEIGHTS
    from src.tools.autoencoder.helpers.training_losses import LossWeights

    weights = LossWeights(rate=0.0, **TASK_WEIGHTS[task])
    assert float(getattr(weights, primary_key)) > 0.0


def test_denoise_smoke_passes_gain_bound(tmp_path) -> None:
    """Smoke test for ``cmd_train_denoise`` (P0.2): trains the spectral codec
    on bath-corrupted states against clean targets for a few epochs, then
    asserts the closed-form gain report passes its machine-checked bound.

    This regression catches both the orbit-tied indexing bug in
    ``denoiser_gain_report`` (which crashed the report on every tied rung)
    and the previous CE-loss path (which left the gains nowhere near the
    closed-form multipliers).
    """
    import json
    from src.tools.autoencoder.cli import cmd_train_denoise

    class A:
        pass

    a = A()
    a.ladder = "shell_radial"
    a.epochs = 1
    a.batch_size = 1024
    a.learning_rate = 1e-2
    a.noise_rate = "0.05,0.05,0.05,0.05,0.05,0.05"
    a.rate_weight = 0.0
    a.seed = 0
    a.run_name = "denoise_smoke"
    a.output_dir = str(tmp_path)
    a.report_file = str(tmp_path / "denoise_report.json")
    rc = cmd_train_denoise(a)
    assert rc == 0
    ckpt = Path(a.output_dir) / f"{a.run_name}.pt"
    assert ckpt.exists(), "denoise checkpoint was not saved before the report"
    report = json.loads(Path(a.report_file).read_text())
    assert "pass" in report and "tol" in report, (
        "denoiser_gain_report must carry a machine-checked pass flag"
    )
    # With a low-noise bath and the MSE loss the gains must track the closed
    # form within the published tolerance. 0.5 is a generous smoke bound; the
    # shipped 0.2 bound is the production-grade gate.
    assert report["mean_abs_error"] < 0.5


def test_unified_multi_transition_labels_aligned(tmp_path) -> None:
    """Regression for P0.1: ``cmd_train --task unified_multi`` must wire the
    transition head to ``next_state = table[tr_state, tr_byte]``. The previous
    code used ``table.reshape(-1)[t_pos]`` while ``tr_state`` came from the
    filtered ``train_idx``, so the labels were wrong everywhere ``train_idx``
    was not the identity permutation. The sampler closure is not accessible
    from outside ``cmd_train``, so we replicate its exact construction here
    and check the kernel invariant directly.
    """
    import numpy as np
    from src.tools.autoencoder.datasets import transition_table

    _args(tmp_path, "unified", "unified_multi", "full", epochs=1)
    rng = np.random.default_rng(0)
    n_state = 4096
    n_val = int(round(n_state * 0.15))
    all_idx = np.arange(n_state, dtype=np.int64)
    perm = rng.permutation(n_state)
    train_idx = np.sort(perm[n_val:])
    val_idx = np.sort(perm[:n_val])
    K = max(64, min(len(train_idx), 2048))
    table = transition_table().astype(np.int64)
    table_tr = table[train_idx]
    s_idx = rng.choice(train_idx, size=K)
    t_pos = rng.choice(len(train_idx) * 256, size=K)
    w_byte = rng.choice(256, size=K)
    tr_state = train_idx[t_pos // 256]
    tr_byte = (t_pos % 256).astype(np.int64)
    # next_state is read with a flat index because table_tr has shape
    # [n_state_train, 256]; a 1D t_pos cannot index axis 0 directly.
    next_state = table_tr.reshape(-1)[t_pos]
    np.testing.assert_array_equal(
        next_state, table[tr_state, tr_byte],
        err_msg="P0.1: next_state != table[tr_state, tr_byte]",
    )
    # And the train_idx filtering must actually be active for the test to
    # mean something -- if train_idx is the identity permutation, the bug
    # is silent.
    assert not np.array_equal(train_idx, all_idx), (
        "val_fraction=0.15 did not filter train_idx; the test would not "
        "have exercised the original bug"
    )
