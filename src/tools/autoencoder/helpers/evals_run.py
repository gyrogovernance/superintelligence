"""Checkpoint evaluation, benchmark suites, and exact verification.

Two parts live in this one module:

- Evaluation (section 1): checkpoint loading, eval runners, report saving, and
  the benchmark suites (percolation rank recovery, anomaly ROC/PR, climate
  sweep) collected from evaluate.py + benchmarks.py.
- Verification (section 2): exact equivariance checks (K4 and full-G), the
  closed-form exhaustive full-G certificate, and the one-pass
  ``audit_dictionary`` plus its report writer (moved from corpus.py).

All evaluation/verification work lives in the ``evals_`` domain. Helpers import
root models and other helpers acyclically.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.tools.autoencoder.models.general import K4Autoencoder
from src.tools.autoencoder.models.narrow import MLPAutoencoder, PercolationLearner
from src.tools.autoencoder.models import build_model
from src.tools.autoencoder.models.super import (
    SpectralAutoencoder,
    full_g_equivariance_error,
    irrep_block_index,
)

from .evals_datasets import percolation_dataset
from .evals_metrics import k4_equivariance_error, percolation_rank_accuracy


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_any_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[torch.nn.Module, dict]:
    """Reconstruct a model from a checkpoint saved via Trainer.

    Reads the saved ``model_kind`` / ``symmetry`` metadata so the right class is
    rebuilt (not just spectral/k4 defaults). Newer checkpoints carry the model's
    exact constructor configuration under ``extra["model_config"]`` (captured
    via ``model.get_config()``); that is used verbatim. Older checkpoints fall
    back to the recorded ``hidden_dim`` / ``latent_dim`` / ``n_trivial`` /
    ``n_sign`` / ``ladder`` fields. Accepts both the wrapped
    ``{"model_state": ...}`` payload and a bare state_dict. The path may be
    given as a ``str`` or a ``Path``.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload.get("config", {})
    extra = payload.get("extra", {})
    model_kind = extra.get("model_kind", "spectral")
    symmetry = extra.get("symmetry", None)
    model_config = extra.get("model_config", None)

    # Reconstruct through the single registry entry point using the exact
    # constructor configuration recorded at save time (``model_config`` from
    # ``model.get_config()``). This removes the parallel per-kind switch that
    # previously missed ``percolation`` and could disagree with the CLI's
    # ``build_model`` on hidden widths. Older/looser checkpoints fall back to
    # the recorded scalar fields.
    cfg = model_config if isinstance(model_config, dict) else {}
    hidden_dim = cfg.get("hidden_dim", extra.get("hidden_dim"))
    ladder = cfg.get("ladder", extra.get("ladder"))
    heads = cfg.get("heads")
    latent_dim = cfg.get("latent_dim", extra.get("latent_dim", 8))
    n_trivial = cfg.get("n_trivial", extra.get("n_trivial"))
    n_sign = cfg.get("n_sign", extra.get("n_sign"))
    sector_mask = cfg.get("sector_mask")
    orbit_index = cfg.get("orbit_index")

    if model_kind.startswith("spectral:"):
        ladder = model_kind.split(":", 1)[1]
        if symmetry is not None:
            from src.tools.autoencoder.models import UnifiedAutoencoder

            model = UnifiedAutoencoder(
                symmetry=symmetry,
                hidden_dim=hidden_dim or 128,
                latent_dim=latent_dim,
                ladder=ladder,
                sector_mask=sector_mask,
                orbit_index=orbit_index,
            )
        else:
            model = SpectralAutoencoder(
                ladder=ladder,
                init_gain=float(cfg.get("init_gain", 1.0)),
                sector_mask=sector_mask,
                orbit_index=orbit_index,
            )
    elif model_kind == "unified" and symmetry is None and not cfg:
        # very old unified checkpoint with no recorded config: spectral default
        model = SpectralAutoencoder()
    else:
        model = build_model(
            model_kind,
            symmetry=symmetry,
            ladder=ladder,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            heads=heads,
            n_trivial=n_trivial,
            n_sign=n_sign,
            sector_mask=sector_mask,
            orbit_index=orbit_index,
        )

    state = payload.get("model_state", payload)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, {"config": config, "extra": extra}


@torch.inference_mode()
def evaluate_reconstruction(model, state_indices: np.ndarray) -> dict[str, float]:
    """Exact-state accuracy of argmax decoding over the given states."""
    idx = torch.as_tensor(state_indices.astype(np.int64))
    out = model(idx)
    pred = out.argmax(dim=-1).numpy()
    return {
        "exact_accuracy": float((pred == state_indices).mean()),
        "n_states": int(len(state_indices)),
    }


def evaluate_transition_accuracy(model, n_states: int = 256, seed: int = 0) -> dict:
    """Argmax next-state accuracy of a byte-conditioned transition model."""
    from src.tools.autoencoder.datasets import transition_table

    table = transition_table().astype(np.int64)
    rng = np.random.default_rng(seed)
    states = rng.choice(4096, size=n_states, replace=False)
    idx = torch.as_tensor(np.repeat(states, 8).astype(np.int64))
    byt = torch.as_tensor(np.tile(np.arange(0, 256, 32), n_states).astype(np.int64))
    pred = model(idx, byt).argmax(dim=-1).numpy()
    truth = table[np.repeat(states, 8), np.tile(np.arange(0, 256, 32), n_states)]
    correct = int((pred == truth).sum())
    total = len(truth)
    return {"transition_accuracy": correct / total, "n_samples": total}


def evaluate_percolation_accuracy(model, seed: int = 7) -> dict:
    """Exact-rank recovery of a trained ``PercolationLearner`` on held-out
    kernel-exact labels (Dataset F). The labels are the kernel's
    ``restriction_labels``; the learner must predict the GF(2)^6 rank exactly."""
    assert isinstance(model, PercolationLearner)
    ds = percolation_dataset(
        n_singletons=64, n_rank_samples=5, n_random=120, seed=seed
    )
    rank = ds["transport_rank"].astype(np.int64)
    idx = np.arange(len(rank), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_hold = int(round(len(rank) * 0.15))
    va = idx[:n_hold]
    with torch.no_grad():
        logits = model(torch.as_tensor(ds["allowed_mask"][va], dtype=torch.float32))
    pred_rank = logits["rank_logits"].argmax(dim=-1).numpy()
    return {
        "rank_accuracy": percolation_rank_accuracy(pred_rank, rank[va]),
        "n_samples": int(len(va)),
    }


def evaluate_checkpoint(
    path: Path, mode: str = "auto", task: str | None = None, seed: int = 7
) -> dict:
    model, meta = load_any_checkpoint(path)
    out: dict = {"checkpoint": str(path), "meta": meta}
    # Infer the task from the checkpoint's recorded model kind when the caller
    # does not pass one. Without this, a task-model checkpoint (e.g. a
    # TransitionModel) falls through to evaluate_reconstruction, which calls
    # model(state_index) and crashes because the task model needs a byte too.
    if task is None:
        kind = (meta.get("extra") or {}).get("model_kind", "spectral")
        if kind in ("transition", "rawbyte", "word", "percolation", "percolation_rank"):
            # normalize the percolation kinds to the eval routing key
            task = "percolation_rank" if kind.startswith("percolation") else kind
    if task in ("transition", "rawbyte"):
        out["transition"] = evaluate_transition_accuracy(model)
        return out
    if task == "word":
        from src.tools.autoencoder.kernel import word_signature_id
        from src.tools.autoencoder.models.narrow import WordActionModel

        assert isinstance(model, WordActionModel)
        byte_sigs = np.array(
            [word_signature_id([b]) for b in range(256)], dtype=np.int64
        )
        with torch.no_grad():
            logits = model.byte_logits(torch.arange(256, dtype=torch.long))
        pred_u = logits[:, :64].argmax(dim=-1).numpy()
        pred_v = logits[:, 64:].argmax(dim=-1).numpy()
        tau_u, tau_v = (byte_sigs >> 6) & 63, byte_sigs & 63
        out["word"] = {
            "tau_u_accuracy": float((pred_u == tau_u).mean()),
            "tau_v_accuracy": float((pred_v == tau_v).mean()),
            "n_bytes": 256,
        }
        return out
    if task == "percolation_rank":
        out["percolation"] = evaluate_percolation_accuracy(model, seed=seed)
        return out
    if task == "unified_multi":
        out["reconstruction"] = evaluate_reconstruction(model, np.arange(4096))
        out["equivariance"] = verify_full_g_equivariance(model)
        return out
    out["reconstruction"] = evaluate_reconstruction(model, np.arange(4096))
    if isinstance(model, K4Autoencoder):
        out["equivariance"] = verify_k4_equivariance(model)
    else:
        out["equivariance"] = verify_full_g_equivariance(model)
    return out


def json_safe(obj):
    """Recursively coerce values into JSON-serializable Python types.

    Checkpoint metadata can carry numpy scalar/array fields (e.g. the spectral
    ``sector_mask`` inside ``model_config``); those are fine for ``torch.save``
    but crash ``json.dumps``. This walks the structure and converts numpy and
    non-primitive values so reports always serialize.
    """
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stamp reproducible identity on every written report (git commit, kernel
    # fingerprint, versions) unless the caller already supplied it.
    stamped = dict(report)
    if "_provenance" not in stamped:
        try:
            import subprocess
            import sys

            import torch

            commit = (
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
            stamped["_provenance"] = {
                "git_commit": commit,
                "torch_version": torch.__version__,
                "python_version": sys.version.split()[0],
                "metric_schema_version": 1,
            }
        except Exception:
            pass
    path.write_text(json.dumps(json_safe(stamped), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Benchmark suites (spec section 3.1-3.3)
# ---------------------------------------------------------------------------


def percolation_suite(seed: int = 7) -> dict[str, float]:
    """Rank recovery, mechanism-vs-correlate, threshold, anchor dependence."""
    from .evals_datasets import percolation_dataset

    ds = percolation_dataset(
        n_singletons=16, n_rank_samples=3, n_random=24, seed=seed
    )
    rank = ds["transport_rank"].astype(np.int64)
    cluster = ds["predicted_cluster"].astype(np.float64)
    reach = ds["reach_size"].astype(np.float64)
    rank_recov = float(np.corrcoef(rank, cluster)[0, 1])
    mechanism_gap = float(np.mean(np.abs(cluster - reach)))
    threshold_acc = float(np.mean((rank == 6) == (ds["full_reachability"] == 1)))
    singleton_rank = float(rank[0]) if len(rank) else float("nan")
    return {
        "rank_recovery_corr": rank_recov,
        "mechanism_vs_correlate_gap": mechanism_gap,
        "threshold_accuracy": threshold_acc,
        "singleton_rank": singleton_rank,
    }


def anomaly_benchmark(n: int = 128, seed: int = 3) -> dict[str, float]:
    """ROC / PR over the corrupted-mask and perturbation datasets with exact
    miss conditions. The classifier is the kernel syndrome itself."""
    from .evals_datasets import byte_perturbation_dataset, corrupted_mask_dataset

    rng = np.random.default_rng(seed)
    mask_rows = corrupted_mask_dataset(rng, n)
    y_true = np.array([r["is_valid"] for r in mask_rows])
    score = -np.array([abs(r["syndrome"]) for r in mask_rows], dtype=np.float64)
    auc = _roc_auc(score, y_true)
    ap = _average_precision(score, y_true)

    pert = byte_perturbation_dataset(rng, n // 2)
    shadow_rows = [r for r in pert if r["kind"] == "shadow_substitution"]
    shadow_acc = (
        float(np.mean([r["signature_preserved"] for r in shadow_rows]))
        if shadow_rows
        else 1.0
    )
    return {"mask_auc": auc, "mask_ap": ap, "shadow_signature_preserve_acc": shadow_acc}


def _roc_auc(scores: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = y[order]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / n_pos
    fpr = fp / n_neg
    area = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))
    return area


def _average_precision(scores: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = y[order]
    precisions = []
    tp = 0
    for i, label in enumerate(y, 1):
        if label == 1:
            tp += 1
            precisions.append(tp / i)
    return float(np.mean(precisions)) if precisions else 0.0


def climate_sweep(lambdas: list[float]) -> dict[str, np.ndarray]:
    """Per-lambda climate order parameters and a coarse regime label.

    The hQVM_QuBEC_Theory.md §5.3 regime taxonomy is keyed on |eta| and M2: condensed when
    |eta| > 2/3 and M2 < 64 + 4096/3 (close to a horizon), thermal when
    |eta| < 1/3 (close to the maximally spread climate), and intermediate
    otherwise. Earlier this labeled on rho, which inverted the semantics -
    high rho is complement-horizon condensation, not thermalization.
    """
    from .evals_datasets import shell_ensemble_labels

    labels = shell_ensemble_labels(lambdas)
    regimes = []
    for i, _lam in enumerate(lambdas):
        eta = float(labels["eta"][i])
        m2 = float(labels["M2"][i])
        if abs(eta) > 2.0 / 3.0 and m2 < 64.0 + 4096.0 / 3.0:
            regimes.append("condensed")
        elif abs(eta) < 1.0 / 3.0:
            regimes.append("thermal")
        else:
            regimes.append("intermediate")
    return {
        "lambda": np.array(lambdas),
        "rho": labels["rho"],
        "eta": labels["eta"],
        "M2": labels["M2"],
        "regime": np.array(regimes),
    }


class Benchmarks:
    """Runs the benchmark suites (percolation, anomaly, climate) at smoke budget."""

    def percolation(self, seed: int = 7) -> dict[str, float]:
        return percolation_suite(seed)

    def anomaly(self, n: int = 128, seed: int = 3) -> dict[str, float]:
        return anomaly_benchmark(n, seed)

    def climate(self, lambdas: list[float] | None = None) -> dict[str, np.ndarray]:
        return climate_sweep(lambdas or [0.25, 1.0, 4.0])


# ---------------------------------------------------------------------------
# Verification: exact equivariance checks and the closed-form full-G certificate
# ---------------------------------------------------------------------------


def verify_k4_equivariance(model, tol: float = 1e-4) -> dict:
    """Exact K4 equivariance over all 4096 states and the four K4 gates.

    Checks both halves: the encoder (``z`` rotates with the group) and the
    decoder (``D(rho(g) z) = P_g D(z)``), which together are the full
    group-equivariance of the autoencoder map."""
    from .evals_metrics import k4_decoder_equivariance_error

    all_states = torch.arange(4096, dtype=torch.long)
    report = k4_equivariance_error(
        model.encoder_eval(), all_states, model.k4_perm, model.rho
    )
    report["passed"] = bool(report["max"] < tol)
    if hasattr(model, "decode") and hasattr(model, "encode"):
        dec = k4_decoder_equivariance_error(model, all_states, model.k4_perm, model.rho)
        report["decoder_max"] = dec["max"]
        report["decoder_mean"] = dec["mean"]
        report["passed"] = report["passed"] and bool(dec["max"] < tol)
    return report


def verify_full_g_equivariance(
    model, state_indices=None, sig_ids=None, seed: int = 0
) -> dict:
    """Full-group equivariance over sampled states and signatures.

    Spectral models (and unified-full, which nests a spectral carrier) expose
    ``walsh_coefficients``, so they take the exact coefficient-level path. Any
    other model (mlp, k4, unified-free/k4) is measured by the generic
    output-permutation path: applying a signature to the input and then
    decoding must equal applying that signature to the decoded output. That
    generic path is the honest way to show a non-spectral model is *not*
    equivariant (the intended contrast artifact), so it reports a large error
    instead of crashing.
    """
    from .evals_metrics import generic_full_g_equivariance_error
    from ..kernel import apply_signature_index

    if state_indices is None:
        state_indices = torch.arange(0, 4096, 17, dtype=torch.long)
    if sig_ids is None:
        sig_ids = torch.tensor(
            [0, 1, 64, 4131, 8191, 4096, 2048], dtype=torch.long
        )
    if hasattr(model, "walsh_coefficients"):
        report = full_g_equivariance_error(model, state_indices, sig_ids)
        report["passed"] = bool(report["max"] < 1e-3 and report["forward_max"] < 1e-3)
        return report
    report = generic_full_g_equivariance_error(
        model, state_indices, sig_ids, apply_signature_index
    )
    report["passed"] = bool(report["max"] < 1e-3)
    return report


def exhaustive_full_g_verify(
    max_err: float = 1e-3, checkpoint: str | None = None
) -> dict[str, float]:
    """Offline full-scale full-G equivariance check (closed form).

    The spectral autoencoder is exactly equivariant by construction: the Walsh
    bottleneck applies a per-(a,b)-block scalar gain, and the affine group
    acts on coefficients as a sign (parity 0) or a swap of the pair
    (parity 1). Equivariance holds iff the gain is symmetric under the swap
    (a,b) <-> (b,a) - which is exactly how ``block_id`` groups coefficients
    into blocks. This is the necessary-and-sufficient condition, so the
    closed-form check over 4096 states x 8192 signatures reduces to a single
    O(blocks) table op over the gain. Runtime is sub-second on CPU.

    When ``checkpoint`` is given, the gains are loaded from the trained model
    in it, so the verifier certifies the artifact that is actually shipped
    (including a learned or frozen ladder mask). Without a checkpoint it
    certifies the architecture class with the default gains.

    A numerical sampled cross-check on a few signatures is layered on top to
    confirm the code path matches the algebra.
    """
    if checkpoint is not None:
        model, _info = load_any_checkpoint(checkpoint)
        spectral = getattr(model, "spectral", model)
        bottleneck = spectral.bottleneck
    else:
        model = SpectralAutoencoder()
        bottleneck = model.bottleneck

    bid, _ = irrep_block_index()  # [64, 64], block_id per (a,b)
    # Use block_gains() so tied rungs (shell_radial etc.) are expanded from
    # their per-orbit free vector to the full [2080] block vector before
    # swap-symmetry is checked; ``bottleneck.gain`` would be [n_orbits] for
    # those rungs and indexing it with block ids would crash.
    gain = np.asarray(bottleneck.block_gains().detach().cpu().numpy(), dtype=np.float64)

    # Max asymmetry of the gain under the swap (a,b) <-> (b,a).
    a_idx = np.arange(64)[:, None]
    b_idx = np.arange(64)[None, :]
    gain_ab = gain[bid[a_idx, b_idx]]
    gain_ba = gain[bid[b_idx, a_idx]]
    max_asym = float(np.max(np.abs(gain_ab - gain_ba)))

    # Numerical sampled cross-check on a spread of signatures.
    sample_sigs = [0, 1, 64, 4131, 8191, 4096, 2048]
    states = np.arange(0, 4096, 17, dtype=np.int64)
    num_err = 0.0
    for sig in sample_sigs:
        rep = full_g_equivariance_error(
            model,
            torch.as_tensor(states, dtype=torch.long),
            torch.tensor([sig], dtype=torch.long),
        )
        num_err = max(num_err, rep["max"])

    report = {
        "max_error": float(max(max_asym, num_err)),
        "algebraic_max_asymmetry": max_asym,
        "numerical_sampled_max": float(num_err),
        "passed": bool(max(max_asym, num_err) <= max_err),
    }
    if checkpoint is not None:
        report["checkpoint"] = Path(checkpoint).as_posix()
    return report


# ---------------------------------------------------------------------------
# One-pass dictionary audit (charter artifact)
# ---------------------------------------------------------------------------


def audit_dictionary(
    model,
    device: str = "cpu",
    checkpoint_hash: str = "",
    seed: int = 0,
) -> dict:
    """One-pass verified-dictionary audit.

    Composes the existing exact checks (reconstruction, full-G equivariance,
    closed-form probe recovery of the 2+6 split, H-invariance of the diagonal
    rung, shadow invariance, frame-parity-zero) and adds the two headline
    invariants: the psi_hat character-energy identity and the H-invariance of
    the diagonal rung. Determinism pinned via checkpoint hash + seed.

    The reconstruction gate targets the full-ladder codec (the exact identity
    reconstruction); for a lossy ladder checkpoint the reconstruction gate is
    reported as informational (``reconstruction_pass`` is only set for full).
    """
    from src.tools.autoencoder.corpus import N_STATES, _embed_bytes
    from src import api
    from src.tools.autoencoder.datasets import transition_table
    from src.tools.autoencoder.kernel import apply_signature_index, k4_action_arrays
    from .evals_metrics import (
        factorization_target_matrix,
        probe_from_latent,
        shadow_invariance_error,
        psi_hat,
    )

    ladder = getattr(model, "ladder", None)
    report: dict = {
        "checkpoint_hash": checkpoint_hash,
        "seed": seed,
        "ladder": ladder,
        "checks": {},
    }
    idx = torch.arange(0, N_STATES, 17, dtype=torch.long, device=device)

    # 1. reconstruction. The full-ladder spectral codec is the exact identity
    #    reconstruction (gains all 1.0); lossy ladders are not asserted at the
    #    1e-3 gate. A learned bottleneck whose gains are no longer identity is
    #    not claimed to be the identity codec, so its reconstruction quality is
    #    reported as a measured quantity but does not gate the dictionary pass
    #    (the dictionary contract is equivariance, exact kernel labels, and the
    #    named invariants).
    with torch.inference_mode():
        recon = model(idx)
    truth = torch.zeros_like(recon)
    truth[torch.arange(len(idx)), idx] = 1.0
    recon_err = float((recon - truth).abs().max())
    report["checks"]["reconstruction_max_err"] = recon_err
    gain = getattr(getattr(model, "bottleneck", model), "gain", None)
    gains_identity = (
        gain is None or float((gain.detach() - 1.0).abs().max()) < 1e-2
    )
    report["checks"]["gains_identity"] = bool(gains_identity)
    is_full = ladder is None or ladder == "full"
    # The exact-reconstruction contract applies only to the identity codec
    # (gains all 1.0). For a lossy ladder or a learned bottleneck the measured
    # error is reported but the gate is not applicable (None), so the audit
    # does not demand identity reconstruction of a non-identity codec.
    if is_full and gains_identity:
        report["checks"]["reconstruction_pass"] = bool(recon_err < 1e-3)
    else:
        report["checks"]["reconstruction_pass"] = None

    # 2. full-G equivariance (existing check function)
    eq = full_g_equivariance_error(
        model, idx, torch.tensor([0, 1, 64, 4131, 8191], dtype=torch.long)
    )
    report["checks"]["equivariance_max_err"] = eq["max"]
    report["checks"]["equivariance_pass"] = bool(eq["max"] < 1e-3)

    # 3. closed-form probe capability (informational, not gating pass). Two parts:
    #    (a) the probe math is exact on an oracle latent (the census target
    #        matrix itself), validating the linear-probe machinery;
    #    (b) probe accuracy on the audited model's actual byte embedding, which
    #        measures how much of the 2+6 split the current model has learned.
    targets = factorization_target_matrix()
    oracle = torch.as_tensor(targets, dtype=torch.float64)
    probe_self_test = bool(
        torch.equal(
            (probe_from_latent(oracle, torch.as_tensor(targets)) > 0.5).long(),
            torch.as_tensor(targets).long(),
        )
    )
    report["checks"]["factorization_probe_self_test"] = probe_self_test
    byte_emb = _embed_bytes(model, device)  # [256, 4096]
    latent = torch.as_tensor(byte_emb, dtype=torch.float64)
    pred = probe_from_latent(latent, torch.as_tensor(targets))
    probe_acc = float((pred.round() == torch.as_tensor(targets)).float().mean())
    report["checks"]["byte_embedding_probe_accuracy"] = probe_acc

    # 4. H-invariance of the diagonal rung (headline invariant). If the audited
    #    model itself is a diagonal-rung spectral model, audit it directly;
    #    otherwise run the library self-test on a fresh diagonal model.
    if isinstance(model, SpectralAutoencoder) and model.ladder == "diagonal":
        diag = model
    else:
        diag = SpectralAutoencoder(ladder="diagonal")
    x = torch.arange(0, N_STATES, 17, dtype=torch.long, device=device)
    base = diag(x)
    max_h = 0.0
    for parity in (0, 1):
        for t in range(64):
            sig = (parity << 12) | (t << 6) | t
            moved = torch.tensor(
                [apply_signature_index(int(i), sig) for i in x.tolist()],
                dtype=torch.long,
            )
            max_h = max(max_h, float((diag(moved) - base).detach().abs().max()))
    report["checks"]["h_invariance_max_err"] = max_h
    report["checks"]["h_invariance_pass"] = bool(max_h < 1e-3)

    # 5. shadow invariance on the exact kernel table (existing metric)
    table = transition_table()

    class _ExactTable(torch.nn.Module):
        def forward(self, s, b):
            o = torch.full((len(s), N_STATES), float("-inf"))
            for i, (sv, bv) in enumerate(zip(s.tolist(), b.tolist())):
                o[i, int(table[sv, bv])] = 0.0
            return o

    shadow_err = shadow_invariance_error(
        _ExactTable(), idx, torch.full((len(idx),), 0x51, dtype=torch.long)
    )
    report["checks"]["shadow_invariance_err"] = shadow_err
    report["checks"]["shadow_invariance_pass"] = bool(shadow_err == 0.0)

    # 6. frame-parity-zero (existing dataset check) plus the per-byte mask
    #    geometry check
    from .evals_datasets import (
        depth4_frame_dataset,
        frame_masks_pair_diagonal,
        frame_parity_zero,
    )

    frames = depth4_frame_dataset(64, seed=seed)
    report["checks"]["frame_parity_zero"] = bool(frame_parity_zero(frames))
    report["checks"]["frame_masks_pair_diagonal"] = bool(
        frame_masks_pair_diagonal(frames)
    )

    # 7. psi_hat character-energy identity (headline invariant). Reuse the
    #    diagonal model from check 4 so the audit uses one consistent object.
    action, _ = k4_action_arrays()
    sig_perm = {g: action[g][idx.numpy()] for g in range(4)}
    psi = psi_hat(diag, idx.numpy(), sig_perm, device=device)
    psi_ok = all(abs(abs(v) - 1.0) < 1e-6 for v in psi.values())
    report["checks"]["psi_hat_character_energy"] = {int(k): float(v) for k, v in psi.items()}
    report["checks"]["psi_hat_pass"] = bool(psi_ok)

    # 8. labels actually match the kernel (authoritative source), not merely
    #    asserted. Verify the byte census columns against src.api for a sample.
    from src.tools.autoencoder.datasets import byte_census_arrays

    census = byte_census_arrays()
    sample = np.arange(0, 256, 7, dtype=np.int64)
    kernel_mask12 = np.array([int(api.MASK12_BY_BYTE[int(b)]) for b in sample])
    kernel_intron = np.array([int(api.byte_to_intron(int(b))) for b in sample])
    label_ok = bool(
        np.array_equal(census["mask12"][sample].astype(np.int64), kernel_mask12)
        and np.array_equal(census["intron_u8"][sample].astype(np.int64), kernel_intron)
    )
    report["checks"]["labels_match_kernel_census"] = label_ok

    # `passed` gates on the invariants that must hold for any correct
    # dictionary: full-G equivariance, shadow invariance, the kernel-label
    # match (the dictionary's core contract), and frame-parity-zero. These are
    # model-independent and always applicable. The 2+6 factorization probe, the
    # diagonal-rung H-invariance, and the psi_hat character-energy check are
    # informational (the diagonal model audited there is a fresh library
    # self-test, not necessarily the audited artifact), so they are deliberately
    # EXCLUDED from the gate even though their keys happen to end in `_pass`.
    # Any non-boolean gate value fails loudly rather than silently passing.
    required_gates = (
        "equivariance_pass",
        "shadow_invariance_pass",
        "labels_match_kernel_census",
        "frame_parity_zero",
    )
    informational_pass_keys = ("h_invariance_pass", "psi_hat_pass")
    gate_results: list[bool] = []
    for key in required_gates:
        value = report["checks"].get(key)
        if not isinstance(value, bool):
            # A missing or non-boolean required gate is a hard fail, never a
            # silent pass.
            gate_results.append(False)
        else:
            gate_results.append(value)
    report["passed"] = all(gate_results) and bool(label_ok)
    report["gate_detail"] = {
        "required": {k: report["checks"].get(k) for k in required_gates},
        "excluded_informational": {
            k: report["checks"].get(k) for k in informational_pass_keys
        },
    }
    return report


def write_audit_report(report: dict, out_path: str | Path) -> None:
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
