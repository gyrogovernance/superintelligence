"""Command-line interface: train / evaluate / verify-equivariance / generate."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .helpers.evals_run import (
    json_safe,
    evaluate_checkpoint,
    save_report,
    verify_full_g_equivariance,
    verify_k4_equivariance,
)
from .helpers.training_losses import LossWeights, weighted_total
from . import paths
from .helpers.training_run import (
    BestCheckpointCallback,
    EarlyStoppingCallback,
    TrainConfig,
    Trainer,
    ValidationCallback,
    iterate_batches,
    set_seed,
)
from .models import (
    MODEL_KINDS,
    TIER_MEMBERS,
    build_model,
)
from .models.general import K4Autoencoder
from .models.super import (
    SpectralAutoencoder,
    SpectralBottleneck,
)


def _provenance() -> dict:
    """Reproducible identity for every checkpoint and report: the git commit,
    kernel fingerprint, library versions, and metric-schema version. Merged
    into the checkpoint ``extra`` so any artifact can be reproduced exactly."""
    import subprocess

    git_commit = "unknown"
    try:
        git_commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[3],
                capture_output=True,
                text=True,
                timeout=10,
            )
            .stdout.strip()
            or "unknown"
        )
    except Exception:
        git_commit = "unknown"
    from .datasets import DatasetManifest

    return {
        "git_commit": git_commit,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "kernel_fingerprint": DatasetManifest.kernel_fingerprint_of(
            Path(__file__).resolve().parents[2]
        ),
        "metric_schema_version": 1,
    }


def _model_metadata(model: nn.Module, kind: str, symmetry: str | None = None) -> dict:
    """Checkpoint record: the model kind plus the model's actual constructor
    configuration (captured from the live instance via ``model.get_config``),
    so ``load_any_checkpoint`` can rebuild the exact architecture without a
    hand-maintained parallel copy of constructor defaults. Provenance fields
    (git commit, kernel fingerprint, versions) are merged for reproducibility.
    """
    cfg = getattr(model, "get_config", None)
    config = cfg() if callable(cfg) else {}
    meta = {
        "model_kind": kind,
        "symmetry": symmetry,
        "model_config": config,
    }
    meta.update(_provenance())
    return meta


TASK_WEIGHTS: dict[str | None, dict[str, float]] = {
    None: dict(state_ce=1.0),
    "state_ce": dict(state_ce=1.0),
    "transition": dict(transition_ce=1.0),
    "rawbyte": dict(transition_ce=1.0),
    "word": dict(word_ce=1.0),
    "percolation_rank": dict(rank_ce=1.0),
    "unified_multi": dict(
        state_ce=1.0, transition_ce=1.0, word_ce=1.0, rank_ce=1.0
    ),
}


def cmd_train(args) -> int:
    set_seed(args.seed)
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    task = getattr(args, "task", None)
    # Upfront compatibility table: each model kind declares which tasks it
    # implements and which are forbidden. A bad (--model, --task) combination
    # surfaces here as a clear message rather than as a deep reshape/TypeError
    # inside the loss.
    model_kind = args.model
    if ":" in model_kind:
        model_kind = model_kind.split(":", 1)[0]
    _MODEL_TASK_ALLOWED: dict[str, tuple[str | None, ...]] = {
        "mlp": (None,),
        "k4": (None,),
        "spectral": (None,),
        "transition": ("transition",),
        "rawbyte": ("rawbyte",),
        "word": ("word",),
        "percolation": ("percolation_rank",),
        "unified": (None, "unified_multi"),
    }
    allowed = _MODEL_TASK_ALLOWED.get(model_kind)
    if allowed is None:
        print(
            f"unknown --model {args.model!r}; expected one of "
            f"{sorted(_MODEL_TASK_ALLOWED)} (with optional ladder suffix)",
            file=sys.stderr,
        )
        return 2
    if task not in allowed:
        print(
            f"--task {task!r} is not supported by --model {args.model!r}; "
            f"allowed tasks: {list(allowed)}",
            file=sys.stderr,
        )
        return 2
    task_weights = TASK_WEIGHTS[task]
    loss_weights = LossWeights(rate=args.rate_weight, **task_weights)
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        device=device,
        seed=args.seed,
        checkpoint_dir=args.output_dir,
        # Keep per-run logs: each named run writes <run_name>_train_log.jsonl so
        # runs in the same output directory do not clobber each other.
        log_file=str(Path(args.output_dir) / f"{args.run_name}_train_log.jsonl"),
        loss_weights=loss_weights,
    )
    heads = ("transition", "word", "rank") if task == "unified_multi" else None
    model: nn.Module = build_model(
        args.model,
        symmetry=getattr(args, "symmetry", None),
        heads=heads,
        hidden_dim=getattr(args, "hidden_dim", None),
    )
    # Echo the resolved model kind so a tier/spectral selection (e.g. --model
    # super -> spectral) is unambiguous in the run log.
    print(f"[train] requested={args.model} resolved={type(model).__name__} task={task}")
    trainer = Trainer(model, config)
    weights = config.loss_weights

    rng = np.random.default_rng(args.seed)
    all_idx = np.arange(4096, dtype=np.int64)
    val_frac = getattr(args, "val_fraction", 0.15)
    n_val = int(round(4096 * val_frac))
    if n_val >= 4096:
        n_val = 0
    if n_val > 0:
        val_idx = rng.choice(4096, size=n_val, replace=False)
        val_idx.sort()
        val_mask = np.zeros(4096, dtype=bool)
        val_mask[val_idx] = True
        train_idx = all_idx[~val_mask]
    else:
        val_idx = np.zeros(0, dtype=np.int64)
        val_mask = np.zeros(4096, dtype=bool)
        train_idx = all_idx

    if task in ("transition", "rawbyte"):
        # byte-conditioned next-state prediction over the full transition
        # table; the state/byte rows are training pairs, val rows are the
        # held-out states (all 256 bytes each).
        from .datasets import transition_table

        table = transition_table().astype(np.int64)
        tr_rows = train_idx
        va_rows = val_idx
        arrays = {
            "state_index": np.repeat(tr_rows, 256),
            "byte": np.tile(np.arange(256, dtype=np.int64), len(tr_rows)),
            "next_state": table[np.repeat(tr_rows, 256), np.tile(np.arange(256), len(tr_rows))],
        }
        val_arrays = (
            {
                "state_index": np.repeat(va_rows, 256),
                "byte": np.tile(np.arange(256, dtype=np.int64), len(va_rows)),
                "next_state": table[np.repeat(va_rows, 256), np.tile(np.arange(256), len(va_rows))],
            }
            if len(va_rows)
            else {"state_index": np.zeros(0, np.int64), "byte": np.zeros(0, np.int64), "next_state": np.zeros(0, np.int64)}
        )

        def loss_fn(batch):
            logits = model(batch["state_index"], batch["byte"])
            ce = torch.nn.functional.cross_entropy(logits, batch["next_state"])
            return weighted_total({"transition_ce": ce}, weights)[0], {
                "transition_ce": float(ce.detach())
            }

    elif task == "word":
        # per-byte (tau_u, tau_v) prediction from the kernel signature table;
        # composition is exact in the model, so per-byte accuracy is the
        # learned quantity.
        from .kernel import word_signature_id

        byte_sigs = np.array([word_signature_id([b]) for b in range(256)], dtype=np.int64)
        tau_u = (byte_sigs >> 6) & 63
        tau_v = byte_sigs & 63
        if val_frac > 0:
            perm = rng.permutation(256)
            n_hold = max(1, int(round(256 * val_frac)))
            val_bytes, train_bytes = perm[:n_hold], perm[n_hold:]
        else:
            val_bytes = np.zeros(0, np.int64)
            train_bytes = np.arange(256)
        arrays = {
            "byte": train_bytes.astype(np.int64),
            "tau_u": tau_u[train_bytes],
            "tau_v": tau_v[train_bytes],
        }
        val_arrays = {
            "byte": val_bytes.astype(np.int64),
            "tau_u": tau_u[val_bytes] if len(val_bytes) else np.zeros(0, np.int64),
            "tau_v": tau_v[val_bytes] if len(val_bytes) else np.zeros(0, np.int64),
        }

        def loss_fn(batch):
            logits = model.byte_logits(batch["byte"])
            ce_u = torch.nn.functional.cross_entropy(logits[:, :64], batch["tau_u"])
            ce_v = torch.nn.functional.cross_entropy(logits[:, 64:], batch["tau_v"])
            return weighted_total({"word_ce": ce_u + ce_v}, weights)[0], {
                "word_ce": float((ce_u + ce_v).detach())
            }

    elif task == "percolation_rank":
        # supervised rank-recovery: the learner reads the kernel's exact
        # percolation labels off the packed allowed mask (Dataset F). The
        # kernel remains the authority for the labels; the model only learns
        # to map the 256-bit alphabet mask to the kernel-computed rank.
        from .helpers.evals_datasets import percolation_dataset

        ds = percolation_dataset(
            n_singletons=64, n_rank_samples=5, n_random=120, seed=args.seed
        )
        rank = ds["transport_rank"].astype(np.int64)
        n = len(rank)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_hold = int(round(n * val_frac)) if val_frac > 0 else 0
        tr, va = idx[n_hold:], idx[:n_hold]
        arrays = {
            "allowed_mask": ds["allowed_mask"][tr],
            "rank": rank[tr],
        }
        val_arrays = (
            {
                "allowed_mask": ds["allowed_mask"][va],
                "rank": rank[va],
            }
            if len(va)
            else {"allowed_mask": ds["allowed_mask"][:0], "rank": np.zeros(0, np.int64)}
        )

        def loss_fn(batch):
            logits = model(batch["allowed_mask"])["rank_logits"]
            ce = torch.nn.functional.cross_entropy(logits, batch["rank"])
            return weighted_total({"rank_ce": ce}, weights)[0], {"rank_ce": float(ce.detach())}

    elif task == "unified_multi":
        # one shared latent (the unified-full gated Walsh spectrum) across all
        # four objectives: state reconstruction, byte-conditioned transition,
        # per-byte word signature, and percolation rank. The task heads read
        # per-block pooled features of the shared code, so the codec's exact
        # equivariance is never disturbed by the heads. Per epoch, each task
        # is sampled to one common length so the trainer's aligned batching
        # stays uniform.
        assert model.symmetry == "full" and model.task_heads is not None, (
            "unified_multi requires --model unified --symmetry full"
        )
        from .datasets import transition_table
        from .helpers.evals_datasets import percolation_dataset
        from .kernel import word_signature_id

        table = transition_table().astype(np.int64)
        table_tr = table[train_idx]  # [n_state_train, 256] — restrict rows to train split
        table_val = table[val_idx]    # [n_val_state, 256] — restrict rows to val split
        byte_sigs = np.array([word_signature_id([b]) for b in range(256)], dtype=np.int64)
        tau_u = (byte_sigs >> 6) & 63
        tau_v = byte_sigs & 63
        ds = percolation_dataset(
            n_singletons=64, n_rank_samples=5, n_random=120, seed=args.seed
        )
        rank = ds["transport_rank"].astype(np.int64)
        n_mask = len(rank)
        n_state = len(train_idx)
        n_val_state = max(1, int(round(n_state * val_frac))) if val_frac > 0 else 0

        # full transition row space = n_state_train x 256
        K = max(64, min(n_state, 2048))
        # No validation rows when val_fraction is 0: keep K_val at 0 so
        # sample_val returns empty arrays instead of crashing on an empty
        # val_idx (rv.choice rejects an empty population even for size 0).
        K_val = min(n_val_state, 128) if n_val_state > 0 else 0

        def sample_arrays(n: int, rng_state: np.random.Generator) -> dict[str, np.ndarray]:
            """Sample every task to exactly ``n`` rows, all from the held-out
            training states so the transition head sees real state indices."""
            s_idx = rng_state.choice(train_idx, size=n)
            t_pos = rng_state.choice(len(train_idx) * 256, size=n)
            w_byte = rng_state.choice(256, size=n)
            m_idx = rng_state.choice(n_mask, size=n)
            tr_state = train_idx[t_pos // 256]
            tr_byte = (t_pos % 256).astype(np.int64)
            return {
                "state_index": s_idx,
                "tr_state": tr_state,
                "tr_byte": tr_byte,
                "next_state": table_tr.reshape(-1)[t_pos],
                "word_state": s_idx,
                "word_byte": w_byte,
                "tau_u": tau_u[w_byte],
                "tau_v": tau_v[w_byte],
                "allowed_mask": ds["allowed_mask"][m_idx],
                "rank": rank[m_idx],
            }

        def sample_val(n: int) -> dict[str, np.ndarray]:
            rv = np.random.default_rng(args.seed + 999)
            s_idx = rv.choice(val_idx, size=n) if n else np.zeros(0, np.int64)
            t_pos = rv.choice(len(val_idx) * 256, size=n) if n else np.zeros(0, np.int64)
            w_byte = rv.choice(256, size=n) if n else np.zeros(0, np.int64)
            m_idx = rv.choice(n_mask, size=n) if n else np.zeros(0, np.int64)
            tr_state = val_idx[t_pos // 256] if n else np.zeros(0, np.int64)
            tr_byte = (t_pos % 256).astype(np.int64) if n else np.zeros(0, np.int64)
            return {
                "state_index": s_idx,
                "tr_state": tr_state,
                "tr_byte": tr_byte,
                "next_state": table_val.reshape(-1)[t_pos] if n else np.zeros(0, np.int64),
                "word_state": s_idx,
                "word_byte": w_byte,
                "tau_u": tau_u[w_byte],
                "tau_v": tau_v[w_byte],
                "allowed_mask": ds["allowed_mask"][m_idx],
                "rank": rank[m_idx],
            }

        def loss_fn(batch):
            heads = model.task_heads
            assert heads is not None
            components: dict[str, torch.Tensor] = {}
            # state reconstruction over the unified-full spectral codec
            if len(batch["state_index"]):
                ce = torch.nn.functional.cross_entropy(
                    model(batch["state_index"]), batch["state_index"]
                )
                components["state_ce"] = ce
            # transition head reads per-block features of the same code plus
            # the raw source state identity (needed for a state-conditioned map)
            if len(batch["tr_state"]):
                feats = model.per_block_features(batch["tr_state"])
                t_logits = heads.transition_logits(
                    feats, batch["tr_byte"], batch["tr_state"]
                )
                components["transition_ce"] = torch.nn.functional.cross_entropy(
                    t_logits, batch["next_state"]
                )
            # word head reads per-block features from (state, byte)
            if len(batch["word_state"]):
                w_feats = model.per_block_features(batch["word_state"])
                w_logits = heads.word_logits(
                    w_feats, batch["word_byte"], batch["word_state"]
                )
                components["word_ce"] = torch.nn.functional.cross_entropy(
                    w_logits[:, :64], batch["tau_u"]
                ) + torch.nn.functional.cross_entropy(w_logits[:, 64:], batch["tau_v"])
            # rank head reads the packed allowed mask directly
            if len(batch["allowed_mask"]):
                r_logits = heads.rank_logits(batch["allowed_mask"])
                components["rank_ce"] = torch.nn.functional.cross_entropy(
                    r_logits, batch["rank"]
                )
            weights = config.loss_weights
            total, logs = weighted_total(components, weights)
            return total, logs

        epoch_rng = {"g": np.random.default_rng(args.seed)}
        arrays_holder: dict[str, np.ndarray] = {}

        def make_batches():
            # refresh per-epoch samples so each pass sees different rows, and
            # advance the epoch counter so the resample is actually different.
            seed = args.seed + epoch["n"]
            epoch["n"] += 1
            rng_epoch = np.random.default_rng(seed)
            arrays_holder.clear()
            arrays_holder.update(sample_arrays(K, rng_epoch))
            return iterate_batches(arrays_holder, config.batch_size, seed)

        val_arrays = sample_val(K_val)

        def val_batches():
            return iterate_batches(val_arrays, 1024, config.seed, shuffle=True)

    else:
        arrays = {"state_index": train_idx}
        val_arrays = {"state_index": val_idx}

        def loss_fn(batch):
            idx = batch["state_index"]
            logits = model(idx)
            ce = torch.nn.functional.cross_entropy(logits, idx)
            components = {"state_ce": ce}
            # Resolve the spectral bottleneck whether it lives on the bare
            # model or nested under the unified model's spectral sub-module.
            bottleneck: "SpectralBottleneck | None" = getattr(model, "bottleneck", None)
            spectral_sub: "SpectralAutoencoder | None" = getattr(model, "spectral", None)
            if bottleneck is None and spectral_sub is not None:
                bottleneck = spectral_sub.bottleneck
            if bottleneck is not None and weights.rate > 0:
                components["rate"] = bottleneck.rate_penalty()
            total, logs = weighted_total(components, weights)
            return total, logs

    epoch = {"n": 0}

    custom_batches = task == "unified_multi"
    if not custom_batches:
        def make_batches():
            # Vary the shuffle seed per epoch so each pass sees a different order.
            it = iterate_batches(arrays, config.batch_size, config.seed + epoch["n"])
            epoch["n"] += 1
            return it

        def val_batches():
            return iterate_batches(val_arrays, 1024, config.seed, shuffle=True)

    callbacks: list = []
    if n_val > 0 and getattr(args, "patience", None):
        from .helpers.training_run import (
            BestCheckpointCallback,
            EarlyStoppingCallback,
            ValidationCallback,
        )

        val_interval = max(1, getattr(args, "val_interval", 1))
        from functools import partial

        callbacks.extend(
            [
                ValidationCallback(
                    partial(trainer.evaluate, loss_fn=loss_fn),
                    val_batches,
                    epoch_interval=val_interval,
                ),
                EarlyStoppingCallback(monitor="val_loss", patience=args.patience, min_delta=getattr(args, "min_delta", 0.0)),
                BestCheckpointCallback(
                    model,
                    Path(args.output_dir) / f"{args.run_name}.best.pt",
                    monitor="val_loss",
                ),
            ]
        )

    stats = trainer.fit(make_batches, loss_fn, callbacks=callbacks)
    extra = {
        **_model_metadata(model, args.model, getattr(args, "symmetry", None)),
        "fit": stats,
        "n_train": int(len(train_idx)),
        "n_val": int(n_val),
    }
    # If early stopping produced a best-weights file, promote those weights
    # into the primary checkpoint so the shipped artifact is the converged
    # optimum rather than the final-epoch weights.
    best_path = Path(args.output_dir) / f"{args.run_name}.best.pt"
    if best_path.exists():
        best_state = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(best_state)
        best_json = best_path.with_suffix(".json")
        if best_json.exists():
            extra["best"] = json.loads(best_json.read_text(encoding="utf-8"))
    path = trainer.save_checkpoint(args.run_name, extra=extra)
    if best_path.exists():
        best_path.unlink()
        best_path.with_suffix(".json").unlink(missing_ok=True)
    print(f"checkpoint: {path}")
    print(json.dumps(stats))
    return 0


def cmd_train_denoise(args) -> int:
    """Train a spectral codec on bath-corrupted states against clean targets.

    The byte bath flips chirality axes independently with probability eta_i,
    acting as (u, v) -> (u ^ delta, v ^ delta). The closed-form optimal
    per-block denoiser gain is prod_i (1 - 2 eta_i)^((a^b)_i) (the Walsh
    multiplier of the carrier frequency); the trained gains are compared
    against it in the emitted report.
    """
    from .helpers.evals_metrics import denoiser_gain_report

    set_seed(args.seed)
    device = "cpu"
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        device=device,
        seed=args.seed,
        checkpoint_dir=args.output_dir,
        log_file=str(Path(args.output_dir) / f"{args.run_name}_train_log.jsonl"),
        # recon_mse (not state_ce) is the loss whose minimizer equals the
        # closed-form posterior-mean denoiser gain. CE sharpens logits
        # unbounded under a low-noise bath and does not converge to the
        # per-block Walsh multiplier; MSE on the one-hot target does.
        loss_weights=LossWeights(recon_mse=1.0, rate=args.rate_weight),
    )
    model = SpectralAutoencoder(ladder=args.ladder)
    trainer = Trainer(model, config)
    weights = config.loss_weights

    rng = np.random.default_rng(args.seed)
    eta = np.asarray(
        [float(x) for x in args.noise_rate.split(",")], dtype=np.float64
    )
    assert eta.shape == (6,), "--noise-rate must be six comma-separated flip probabilities"
    clean = np.arange(4096, dtype=np.int64)

    def corrupt(states: np.ndarray, gen: np.random.Generator) -> np.ndarray:
        # flip chirality axis i by XORing BOTH registers: the state index
        # delta is 65 * d for a 6-bit flip vector d (bits i and i+6)
        d = (gen.random((len(states), 6)) < eta[None, :]).astype(np.int64) @ (
            1 << np.arange(6)
        )
        return states ^ (65 * d)  # (u, v) -> (u ^ d, v ^ d)

    epoch = {"n": 0}

    def make_batches():
        gen = np.random.default_rng(args.seed + 1000 + epoch["n"])
        epoch["n"] += 1
        noisy = corrupt(clean, gen)
        return iterate_batches(
            {"state_index": noisy, "target": clean},
            config.batch_size,
            config.seed + epoch["n"],
        )

    def loss_fn(batch):
        logits = model(batch["state_index"])
        target = torch.zeros_like(logits)
        target[torch.arange(len(batch["target"])), batch["target"]] = 1.0
        # MSE against the clean one-hot is the loss whose minimizer equals
        # the closed-form posterior-mean denoiser (per-block Walsh multiplier
        # of the carrier frequency a ^ b). CE does not converge to the closed
        # form under a low-noise bath.
        mse = ((logits - target) ** 2).mean()
        components = {"recon_mse": mse}
        if weights.rate > 0:
            components["rate"] = model.bottleneck.rate_penalty()
        return weighted_total(components, weights)[0], {
            "recon_mse": float(mse.detach()),
            "rate": float(components["rate"].detach()) if "rate" in components else 0.0,
        }

    stats = trainer.fit(make_batches, loss_fn)
    # Save the checkpoint first; the gain report is downstream and must not
    # block the trained artifact from being persisted.
    extra = {
        **_model_metadata(model, f"spectral:{args.ladder}" if args.ladder else "spectral"),
        "fit": stats,
        "task": "denoise",
        "eta": eta.tolist(),
    }
    path = trainer.save_checkpoint(args.run_name, extra=extra)
    report = denoiser_gain_report(model, eta.tolist())
    summary = {
        "checkpoint": path.as_posix(),
        "epochs_run": stats.get("epochs_run"),
        "eta": eta.tolist(),
        **report,
    }
    if args.report_file:
        save_report(summary, Path(args.report_file))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_evaluate(args) -> int:
    report = evaluate_checkpoint(
        Path(args.checkpoint),
        task=getattr(args, "task", None),
        seed=getattr(args, "seed", 7),
    )
    print(json.dumps(json_safe(report), indent=2))
    if args.report_file:
        save_report(report, Path(args.report_file))
    return 0


def cmd_verify_equivariance(args) -> int:
    from .helpers.evals_run import load_any_checkpoint

    model, _ = load_any_checkpoint(Path(args.checkpoint))
    if isinstance(model, K4Autoencoder):
        report = verify_k4_equivariance(model)
    else:
        report = verify_full_g_equivariance(model, seed=args.seed)
    print(json.dumps(json_safe(report), indent=2))
    if args.report_file:
        save_report(report, Path(args.report_file))
    return 0


def cmd_verify_groups(args) -> int:
    """Kernel-side group sanity: K4 + signature invariants (fast subset).

    Checks the structural facts the model relies on rather than only printing
    constants: K4 acts as a group of order 4 on every state, every signature
    has a proper inverse, and signature composition is a group of order 8192.
    """
    from src import api
    from .kernel import (
        apply_k4_index,
        apply_signature_index,
        signature_inverse_id,
        signature_from_id,
        signature_id,
    )

    # K4 closure: composing any two gates yields one of the four gates, and
    # each gate is an involution-or-identity on every state.
    gates = ("id", "S", "C", "F")
    k4_closure_ok = True
    k4_involution_ok = True
    for idx in range(0, 4096, 17):
        for g in gates:
            dest = apply_k4_index(idx, g)
            # g is a permutation: applying it to the destination returns here
            if apply_k4_index(dest, g) != idx:
                k4_involution_ok = False
        # S then C = F (the Klein four structure); verify on a sample.
        if apply_k4_index(apply_k4_index(idx, "S"), "C") != apply_k4_index(idx, "F"):
            k4_closure_ok = False

    # Signature inverse: g . g^{-1} == id for every signature.
    sig_inverse_ok = True
    for sig in range(0, 8192, 7):
        inv = signature_inverse_id(sig)
        base = apply_signature_index(1234, sig)
        back = apply_signature_index(base, inv)
        if back != 1234:
            sig_inverse_ok = False

    # Signature group order: composition identity is consistent with parts.
    comp_ok = True
    for sig in (1, 64, 4131, 8191):
        parity, tu, tv = signature_from_id(sig)
        reb = signature_id(parity, tu, tv)
        if reb != sig:
            comp_ok = False

    report = {
        "omega_states": len(api.OMEGA_STATES_4096),
        "signature_count": 8192,
        "k4_gates": gates,
        "checks": {
            "k4_closure": bool(k4_closure_ok),
            "k4_involution": bool(k4_involution_ok),
            "signature_inverse": bool(sig_inverse_ok),
            "signature_encoding_roundtrip": bool(comp_ok),
        },
    }
    report["passed"] = all(report["checks"].values())
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def cmd_verify_full_g_exhaustive(args) -> int:
    """Closed-form full-G equivariance certificate (sub-second).

    Reaches the full 4096 states x 8192 signatures by the gain-symmetry
    condition, and certifies the shipped artifact when a checkpoint is given
    (the gains are read from the trained model)."""
    from .helpers.evals_run import exhaustive_full_g_verify

    report = exhaustive_full_g_verify(
        max_err=getattr(args, "max_err", 1e-3), checkpoint=args.checkpoint
    )
    report["checkpoint"] = args.checkpoint
    print(json.dumps(json_safe(report), indent=2))
    if args.report_file:
        out = Path(args.report_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


def cmd_sample_ensemble(args) -> int:
    """Emit a lambda-ensemble state corpus with its exact climate labels."""
    from .helpers.evals_datasets import (
        corpus_shell_histogram,
        lambda_grid,
        sample_lambda_corpus,
    )
    from .helpers.evals_datasets import shell_ensemble_labels

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lams = lambda_grid(args.grid).tolist() if args.grid != "custom" else [args.lambda_value]
    rows = []
    for lam in lams:
        corpus = sample_lambda_corpus(lam, args.n, seed=args.seed)
        path = out_dir / f"ensemble_lambda_{lam:.4g}.npy"
        np.save(path, corpus.astype(np.int64))
        hist = corpus_shell_histogram(corpus)
        labels = shell_ensemble_labels([lam])
        row = {
            "lambda": lam,
            "n": int(args.n),
            "corpus_file": str(path),
            "seed": args.seed,
            "rho": float(labels["rho"][0]),
            "eta": float(labels["eta"][0]),
            "M2": float(labels["M2"][0]),
            "shell_hist": hist.tolist(),
        }
        rows.append(row)
        print(json.dumps(row))
    summary_path = out_dir / "ensemble_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")
    return 0


def cmd_sweep_lambda(args) -> int:
    """The converged symmetry-breaking experiment: train on each lambda
    ensemble, then read psi_hat for the broken generator W2 and for a
    never-broken stabilizer. Pre-registered prediction: psi_hat(W2) is
    negative for lambda != 1 and rises toward 0/+ as lambda -> 1 (the
    corpus loses its chirality polarization), while psi_hat on the
    diagonal stabilizer stays pinned. Writes JSONL, one row per lambda."""
    from .helpers.evals_datasets import (
        _closed_under_composition,
        sample_lambda_corpus,
        w2_word_signature_ids,
    )
    from .kernel import apply_signature_index

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "sweep_lambda.jsonl"

    # broken generator: the canonical W2(m=0) word
    w2_ids = w2_word_signature_ids()
    w2_id = w2_ids[0]
    # never-broken generator: the diagonal swap (1, 1, 1), in H for all lambda
    stabilizer_id = (1 << 12) | (1 << 6) | 1

    def perm_for(sig_id: int, corpus: np.ndarray) -> np.ndarray:
        return np.array(
            [apply_signature_index(int(i), sig_id) for i in corpus.tolist()],
            dtype=np.int64,
        )

    lams = np.geomspace(0.125, 8.0, 7)
    with report_path.open("w", encoding="utf-8") as fh:
        for lam in lams:
            corpus = sample_lambda_corpus(float(lam), args.n, seed=args.seed)
            model: nn.Module = build_model(args.model, symmetry=getattr(args, "symmetry", None))
            config = TrainConfig(
                epochs=args.epochs,
                batch_size=min(args.batch_size, len(corpus)),
                lr=args.learning_rate,
                device="cpu",
                seed=args.seed,
                checkpoint_dir=str(out_dir),
                loss_weights=LossWeights(state_ce=1.0),
            )
            trainer = Trainer(model, config)
            arrays = {"state_index": corpus}
            weights = config.loss_weights

            def loss_fn(batch):
                idx = batch["state_index"]
                logits = model(idx)
                ce = torch.nn.functional.cross_entropy(logits, idx)
                return weighted_total({"state_ce": ce}, weights)

            epoch = {"n": 0}

            def make_batches(a=arrays):
                it = iterate_batches(a, config.batch_size, config.seed + epoch["n"])
                epoch["n"] += 1
                return it

            trainer.fit(make_batches, loss_fn)
            # All model kinds expose ``encode`` as the latent map (spectral and
            # unified-full reconstruct in forward, so encode returns the Walsh
            # coefficient latent; K4 and MLP have explicit encoders). ``psi_hat``
            # calls ``.eval()`` on its encoder, so wrap in a Module.
            class _LatentEncoder(torch.nn.Module):
                def __init__(self, m: torch.nn.Module) -> None:
                    super().__init__()
                    self.m = m

                def forward(self, idx: torch.Tensor) -> torch.Tensor:
                    return self.m.encode(idx)

            enc = _LatentEncoder(model)
            from .helpers.evals_metrics import psi_hat

            gens = {
                w2_id: perm_for(w2_id, corpus),
                stabilizer_id: perm_for(stabilizer_id, corpus),
            }
            psi = psi_hat(enc, corpus, gens, device="cpu")
            row = {
                "lambda": float(lam),
                "model": args.model,
                "n": int(args.n),
                "epochs": args.epochs,
                "W2_sig_id": int(w2_id),
                "psi_W2": psi[w2_id],
                "H_sig_id": int(stabilizer_id),
                "psi_H": psi[stabilizer_id],
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(json.dumps(row))
    print(f"sweep report: {report_path}")
    return 0


def _load_spectral_model(checkpoint: str | None) -> "SpectralAutoencoder":
    """Return the spectral codec for a spectral checkpoint.

    A unified container's ``.spectral`` sub-module is extracted; a bare
    spectral checkpoint is returned as-is. With no checkpoint path, a fresh
    identity codec is returned.
    """
    from .models.super import SpectralAutoencoder
    from .helpers.evals_run import load_any_checkpoint

    if not checkpoint:
        return SpectralAutoencoder()
    model, _ = load_any_checkpoint(checkpoint, device="cpu")
    spectral = getattr(model, "spectral", None)
    if isinstance(spectral, SpectralAutoencoder):
        return spectral
    if isinstance(model, SpectralAutoencoder):
        return model
    raise TypeError(
        f"export-embeddings/audit-dictionary requires a spectral checkpoint; "
        f"got {type(model).__name__} at {checkpoint}"
    )


def cmd_export_embeddings(args) -> int:
    """Export the verified-dictionary embedding corpus (charter artifact)."""
    from .corpus import export_embeddings

    set_seed(args.seed)
    model = _load_spectral_model(args.checkpoint)
    out_dir = Path(args.output_dir)
    # The identity export (no checkpoint) writes the bare filenames; a trained
    # checkpoint writes "<name>_<suffix>.npy" so it never clobbers the identity.
    suffix = ""
    if args.checkpoint:
        # Use the checkpoint's stem (file name without suffix) and sanitize
        # any non-alphanumeric characters so Windows backslash paths and
        # multi-segment suffixes both reduce to one portable token.
        stem = Path(args.checkpoint).stem
        suffix = "".join(c if c.isalnum() else "_" for c in stem) or "trained"
    arrays = export_embeddings(
        model,
        out_dir,
        checkpoint_hash=str(args.checkpoint or "identity"),
        seed=args.seed,
        suffix=suffix,
    )
    print(
        f"exported {len(arrays)} arrays to {out_dir} "
        f"(states={arrays['state_embedding'].shape[0]}, "
        f"bytes={arrays['byte_mask12'].shape[0]}, "
        f"signatures={arrays['signature_embedding'].shape[0]})"
        + (f" with suffix _{suffix}" if suffix else " (identity)")
    )
    return 0


def cmd_audit_dictionary(args) -> int:
    """Run the one-pass verified-dictionary audit and write the JSON report."""
    from .helpers.evals_run import audit_dictionary, write_audit_report

    set_seed(args.seed)
    model = _load_spectral_model(args.checkpoint)
    report = audit_dictionary(
        model, checkpoint_hash=str(args.checkpoint or "identity"), seed=args.seed
    )
    write_audit_report(report, Path(args.report_file))
    print(json.dumps({"passed": report["passed"], "out": Path(args.report_file).as_posix()}, indent=2))
    return 0 if report["passed"] else 1


def cmd_generate(args) -> int:
    """Generate exact hQVM datasets with versioned manifests."""
    from .datasets import generate_all, generate_dataset

    data_dir = Path(args.data_dir)
    if args.dataset == "all":
        out_paths = generate_all(data_dir)
    elif args.dataset in ("bytes", "states", "transitions", "actions", "signatures"):
        out_paths = [generate_dataset(args.dataset, data_dir)]
    else:
        raise ValueError(f"unknown dataset {args.dataset!r}")
    for path in out_paths:
        print(f"generated {path}")
    return 0


def cmd_genomics(args) -> int:
    """Train-less 9-layer genomics compile of a sequence window.

    Reads a FASTA/plain sequence file (gzip-aware), lifts it through the
    carrier byte stream, and assembles the certified ``GenomicCompile``
    (byte_fold_w, fold_poles, family_sheet, omega_signature, depth4_parity,
    chi_shells, qubec_order, ab_horizon, boundary_keys) under one of the 24
    NCBI nucleotide encodings. Pure data transform - there is no model.
    """
    from .helpers.genomics import (
        all_nucleotide_encodings,
        compile_climate_summary,
        compile_interval,
        read_sequence_file,
    )

    enc = all_nucleotide_encodings()[(args.enc % 24)]
    seq = read_sequence_file(Path(args.input_file), max_bases=args.max_bases)
    gc_obj = compile_interval(seq, enc, label=Path(args.input_file).stem)
    layers = {lay.name: {k: float(v) for k, v in lay.values} for lay in gc_obj.layers}
    report = {
        "input_file": str(args.input_file),
        "sequence_length": gc_obj.seq_len,
        "n_bytes": gc_obj.n_bytes,
        "encoding_index": args.enc % 24,
        "layers": layers,
        "climate_summary": compile_climate_summary(gc_obj),
    }
    print(json.dumps(json_safe(report), indent=2))
    if args.report_file:
        save_report(report, Path(args.report_file))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.tools.autoencoder.cli",
        description="hQVM group-equivariant autoencoder CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_generate = sub.add_parser(
        "generate", help="generate exact hQVM datasets with versioned manifests"
    )
    p_generate.add_argument(
        "--dataset",
        default="all",
        help="dataset name or 'all' (bytes, states, transitions, actions, signatures)",
    )

    p_generate.add_argument(
        "--data-dir",
        default=str(paths.DATA_HOME),
        help="this package's data home; each dataset lands in <data-home>/dataset_<name>/",
    )
    p_generate.set_defaults(func=cmd_generate)

    p_train = sub.add_parser("train", help="train a model on the state census")
    p_train.add_argument(
        "--model",
        default="spectral",
        help="individual kind: mlp | k4 | spectral | transition | rawbyte | word "
        "| unified | percolation; or a spectral:<ladder> rung (full, diagonal, "
        "shell, offdiagonal, trivial, shell_radial, shell_gauge, chirality_gauge); "
        "or a tier selector narrow | general | super | all (trains the tier's "
        "first member). unified requires --symmetry. A tier/spectral selection "
        "is echoed in the run log so sweeps are unambiguous.",
    )
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument(
        "--learning-rate",
        "--lr",
        dest="learning_rate",
        type=float,
        default=1e-3,
        help="step size (alias --lr)",
    )
    p_train.add_argument(
        "--val-fraction", type=float, default=0.15,
        help="fraction of states held out for validation CE (0 disables validation)"
    )
    p_train.add_argument(
        "--val-interval", type=int, default=1,
        help="check validation loss every N epochs"
    )
    p_train.add_argument(
        "--patience", type=int, default=0,
        help="stop after this many epochs without validation improvement "
        "(0 disables early stopping; requires --val-fraction > 0)"
    )
    p_train.add_argument(
        "--min-delta", type=float, default=0.0,
        help="minimum absolute validation improvement counted as progress "
        "(guards against noise-driven false plateaus)"
    )
    p_train.add_argument(
        "--rate-weight",
        type=float,
        default=0.0,
        help="L1 rate penalty weight on spectral gains (learned bottleneck)",
    )
    p_train.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="hidden width for the model (default: per-kind default from build_model)",
    )
    p_train.add_argument(
        "--run-name",
        "--name",
        dest="run_name",
        default="model",
        help="run name; per-run logs write <run_name>_train_log.jsonl (alias --name)",
    )
    p_train.add_argument(
        "--output-dir",
        "--out",
        dest="output_dir",
        default=str(paths.checkpoints_dir()),
        help="output directory for checkpoints and logs (alias --out)",
    )
    p_train.add_argument(
        "--symmetry",
        default=None,
        choices=("free", "k4", "full"),
        help="Symmetry level for the unified autoencoder: with model=unified, "
        "selects the level directly (full = Walsh block gains, k4 = K4 Reynolds, "
        "free = unconstrained); with model=spectral:<ladder>, wraps that spectral "
        "model in the unified autoencoder and applies the given level.",
    )
    p_train.add_argument(
        "--task",
        default=None,
        choices=("state_ce", "transition", "rawbyte", "word", "percolation_rank", "unified_multi"),
        help="Learning task: state_ce (default, census reconstruction), "
        "transition/rawbyte (byte-conditioned next-state prediction), "
        "word (per-byte tau prediction), percolation_rank (rank recovery "
        "from the allowed byte mask), unified_multi (all four objectives "
        "over one shared spectral latent). Requires the matching --model.",
    )
    p_train.set_defaults(func=cmd_train)

    p_train_denoise = sub.add_parser(
        "train-denoise", help="train a spectral codec to denoise bath-corrupted states"
    )
    p_train_denoise.add_argument("--ladder", default="shell_radial", help="spectral rung")
    p_train_denoise.add_argument("--epochs", type=int, default=5)
    p_train_denoise.add_argument("--batch-size", type=int, default=256)
    p_train_denoise.add_argument(
        "--learning-rate",
        "--lr",
        dest="learning_rate",
        type=float,
        default=1e-3,
        help="step size (alias --lr)",
    )
    p_train_denoise.add_argument("--noise-rate", "--eta", dest="noise_rate",
                                default="0.03,0.03,0.03,0.03,0.03,0.03",
                                help="six comma-separated axis flip probabilities (alias --eta)")
    p_train_denoise.add_argument("--rate-weight", type=float, default=0.0)
    p_train_denoise.add_argument("--seed", type=int, default=0)
    p_train_denoise.add_argument(
        "--run-name",
        "--name",
        dest="run_name",
        default="denoise",
        help="run name (alias --name)",
    )
    p_train_denoise.add_argument(
        "--output-dir",
        "--out",
        dest="output_dir",
        default=str(paths.checkpoints_dir()),
        help="output directory (alias --out)",
    )
    p_train_denoise.add_argument("--report-file", "--report", dest="report_file", default=None,
                                 help="save JSON report here")
    p_train_denoise.set_defaults(func=cmd_train_denoise)

    p_eval = sub.add_parser("evaluate", help="evaluate a checkpoint")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument(
        "--task",
        default=None,
        choices=("state_ce", "transition", "rawbyte", "word", "percolation_rank", "unified_multi"),
        help="default reconstruction/equivariance, or the matching model "
        "task inferred from the checkpoint when omitted",
    )
    p_eval.add_argument("--seed", type=int, default=7,
                        help="seed for held-out split (percolation) and any sampling")
    p_eval.add_argument("--report-file", "--out", dest="report_file", default=None,
                    help="save JSON report here")
    p_eval.set_defaults(func=cmd_evaluate)

    p_verify = sub.add_parser("verify-equivariance", help="verify exact equivariance")
    p_verify.add_argument("--checkpoint", required=True)
    p_verify.add_argument("--seed", type=int, default=0)
    p_verify.add_argument("--report-file", "--out", dest="report_file", default=None,
                          help="save JSON report here")
    p_verify.set_defaults(func=cmd_verify_equivariance)

    p_groups = sub.add_parser("verify-groups", help="kernel group sanity summary")
    p_groups.set_defaults(func=cmd_verify_groups)

    p_verify_ex = sub.add_parser(
        "verify-full-g-exhaustive",
        help="closed-form full-G equivariance certificate (sub-second)",
    )
    p_verify_ex.add_argument(
        "--checkpoint", default=None, help="trained checkpoint to certify (optional)"
    )
    p_verify_ex.add_argument("--max-err", type=float, default=1e-3)
    p_verify_ex.add_argument(
        "--report-file", "--out", dest="report_file", default=None,
        help="save JSON report here"
    )
    p_verify_ex.set_defaults(func=cmd_verify_full_g_exhaustive)

    p_sample = sub.add_parser(
        "sample-ensemble", help="emit lambda-ensemble corpora with climate labels"
    )
    p_sample.add_argument("--n", type=int, default=100_000)
    p_sample.add_argument(
        "--grid",
        default="log",
        choices=("linear", "log", "custom"),
        help="lambda grid: log (default) | linear | custom (use --lambda-value)",
    )
    p_sample.add_argument("--lambda-value", type=float, default=1.0)
    p_sample.add_argument("--seed", type=int, default=0)
    p_sample.add_argument(
        "--output-dir",
        "--out",
        dest="output_dir",
        default=str(paths.dataset_dir("ensembles")),
        help="output directory (alias --out)",
    )
    p_sample.set_defaults(func=cmd_sample_ensemble)

    p_sweep = sub.add_parser(
        "sweep-lambda",
        help="train per-lambda ensembles and read the psi-hat order parameter",
    )
    p_sweep.add_argument("--model", default="mlp")
    p_sweep.add_argument("--epochs", type=int, default=3)
    p_sweep.add_argument("--n", type=int, default=8192)
    p_sweep.add_argument("--batch-size", type=int, default=512)
    p_sweep.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-3,
                         help="step size (alias --lr)")
    p_sweep.add_argument("--seed", type=int, default=0)
    p_sweep.add_argument(
        "--output-dir",
        "--out",
        dest="output_dir",
        default=str(paths.checkpoints_dir()),
        help="output directory (alias --out)",
    )
    p_sweep.set_defaults(func=cmd_sweep_lambda)

    p_export = sub.add_parser(
        "export-embeddings", help="export the verified-dictionary embedding corpus"
    )
    p_export.add_argument("--model", default="spectral", choices=("spectral",))
    p_export.add_argument("--checkpoint", default=None, help="optional trained checkpoint")
    p_export.add_argument(
        "--output-dir",
        "--out",
        dest="output_dir",
        default=str(paths.dataset_dir("embeddings")),
        help="output directory (alias --out)",
    )
    p_export.add_argument("--seed", type=int, default=0)
    p_export.set_defaults(func=cmd_export_embeddings)

    p_audit = sub.add_parser(
        "audit-dictionary", help="run the one-pass verified-dictionary audit"
    )
    p_audit.add_argument("--model", default="spectral", choices=("spectral",))
    p_audit.add_argument("--checkpoint", default=None, help="optional trained checkpoint")
    p_audit.add_argument(
        "--report-file",
        "--out",
        dest="report_file",
        default=str(paths.reports_dir() / "embedding_corpus_audit.json"),
        help="save JSON audit report here (alias --out)",
    )
    p_audit.add_argument("--seed", type=int, default=0)
    p_audit.set_defaults(func=cmd_audit_dictionary)

    p_genomics = sub.add_parser(
        "genomics",
        help="train-less 9-layer compile of a sequence window",
    )
    p_genomics.add_argument("--input-file", required=True, help="FASTA/plain sequence file (gzip-aware)")
    p_genomics.add_argument("--enc", type=int, default=0, help="nucleotide encoding index in 0..23")
    p_genomics.add_argument(
        "--max-bases",
        type=int,
        default=500000,
        help="read at most this many bases from the file",
    )
    p_genomics.add_argument(
        "--report-file",
        "--out",
        dest="report_file",
        default=None,
        help="save the compile JSON report here (alias --out)",
    )
    p_genomics.set_defaults(func=cmd_genomics)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
