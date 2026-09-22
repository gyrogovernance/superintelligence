"""Genomics synthesis census battery and suite entry.

Owns GateResult, host/catalog census gates, ledger merge, and ``run_suite``.
Margins and floors trace to Analysis anchors (see MARGINS and census details).
"""

from __future__ import annotations

import json
import math
import random as py_random
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src import api
from src.family import (
    build_hqvm_d,
    gf2_rank,
    intron_family_d,
    intron_from_byte,
    predicted_cluster_size,
)
from src.tools.autoencoder.datasets import transition_table
from src.tools.autoencoder.helpers.evals_run import verify_k4_equivariance
from src.tools.autoencoder.kernel import (
    step_index,
    word_signature_id,
)
from src.tools.autoencoder.models.super import Super
from src.tools.autoencoder.programs.genomics.genomics import (
    BASES,
    STANDARD_CODE,
    _INTRON_FAMILY_IDX,
    pack_codon_bits,
    pair_inversion_encoding,
)
from src.tools.autoencoder.programs.genomics.synthesis.climate import (
    climate_of_sequence,
)
from src.tools.autoencoder.programs.genomics.synthesis.constellation import (
    FrozenConstellation,
    load_frozen_constellation,
    random_constellation,
    state_latents,
    super_forward,
)
from src.tools.autoencoder.programs.genomics.synthesis.reads import (
    HostCache,
    byte_stream,
    extract_chr22_splice_flanks,
    extract_chr22_splice_windows,
    null_gene_streams,
    prepare_host,
    SPLICE_WINDOW,
)
from src.tools.autoencoder.programs.genomics.synthesis.registry import (
    assert_registry_complete,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "RESULTS.txt"

GATES_JSON = HERE / "gates.json"

# Pre-registered G4 margin (trained must clear random by at least this).
G4_MARGIN = 0.05


@dataclass(frozen=True)
class GateResult:
    name: str
    n: int
    trained: float
    random: float
    margin: float
    compile_floor: float | None
    passed: bool
    detail: str
    outcome: Literal["PASS", "FLOOR", "NEGATIVE", "DEFERRED", "REPORT"] | None = None
    valid: bool = True
    kernel_ref: float | None = None

    def __post_init__(self) -> None:
        if self.outcome is None:
            if self.passed:
                outcome: str = "PASS"
            else:
                outcome = "NEGATIVE" if self.valid else "DEFERRED"
            object.__setattr__(self, "outcome", outcome)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def collision_pairs(
    stream: Sequence[int], n_pairs: int, seed: int
) -> list[tuple[list[int], list[int]]]:
    """4-byte words that share a word signature (G4 collision construction)."""
    rng = np.random.default_rng(seed)
    sig_map: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    n = len(stream)
    if n < 4:
        return []
    starts = rng.choice(n - 4, size=min(50_000, n - 4), replace=False)
    for s in starts:
        word = (
            int(stream[s]),
            int(stream[s + 1]),
            int(stream[s + 2]),
            int(stream[s + 3]),
        )
        sig = word_signature_id(list(word))
        sig_map[sig].append(word)

    pairs: list[tuple[list[int], list[int]]] = []
    rng2 = np.random.default_rng(seed + 1)
    for words in sig_map.values():
        if len(words) < 2:
            continue
        idxs = rng2.choice(len(words), size=min(3, len(words) - 1), replace=False)
        for k in range(len(idxs) - 1):
            a = list(words[int(idxs[k])])
            b = list(words[int(idxs[k + 1])])
            if a != b:
                pairs.append((a, b))
        if len(pairs) >= n_pairs:
            break
    return pairs[:n_pairs]


def permutation_pairs(
    stream: Sequence[int], n_pairs: int, seed: int
) -> list[tuple[list[int], list[int]]]:
    """Same signature and same byte multiset; differ only in walk order.

    Swapping positions 0<->2 or 1<->3 preserves word_signature_id by the
    register crossing law and leaves the multiset unchanged.
    """
    rng = np.random.default_rng(seed)
    n = len(stream)
    if n < 4:
        return []
    starts = rng.choice(n - 4, size=min(80_000, n - 4), replace=False)
    pairs: list[tuple[list[int], list[int]]] = []
    for s in starts:
        w = [
            int(stream[s]),
            int(stream[s + 1]),
            int(stream[s + 2]),
            int(stream[s + 3]),
        ]
        if w[0] != w[2]:
            t = list(w)
            t[0], t[2] = t[2], t[0]
        elif w[1] != w[3]:
            t = list(w)
            t[1], t[3] = t[3], t[1]
        else:
            continue
        if w == t:
            continue
        assert word_signature_id(w) == word_signature_id(t)
        assert sorted(w) == sorted(t)
        pairs.append((w, t))
        if len(pairs) >= n_pairs:
            break
    return pairs


def _cosine_dist_pair(
    model: Super, word_a: list[int], word_b: list[int], device: str
) -> float:
    mask = torch.zeros(4, dtype=torch.bool, device=device)
    mask[3] = True
    a = torch.as_tensor([word_a], dtype=torch.long, device=device)
    b = torch.as_tensor([word_b], dtype=torch.long, device=device)
    m = mask.unsqueeze(0)
    with torch.inference_mode():
        za = model(a, mask=m, hide_masked=True)["context"]
        zb = model(b, mask=m, hide_masked=True)["context"]
    cos = float(F.cosine_similarity(za, zb, dim=-1).item())
    return 1.0 - cos


def _context_dist_pair(
    model: Super, word_a: list[int], word_b: list[int], device: str
) -> float:
    """Cosine distance with all bytes visible (no mask)."""
    a = torch.as_tensor([word_a], dtype=torch.long, device=device)
    b = torch.as_tensor([word_b], dtype=torch.long, device=device)
    with torch.inference_mode():
        za = model(a, mask=None, hide_masked=True)["context"]
        zb = model(b, mask=None, hide_masked=True)["context"]
    cos = float(F.cosine_similarity(za, zb, dim=-1).item())
    return 1.0 - cos


def g4_census(
    *,
    host: str = "ecoli",
    n_pairs: int = 400,
    seed: int = 42,
    device: str = "cpu",
    trained: FrozenConstellation | None = None,
    random: FrozenConstellation | None = None,
    stream: Sequence[int] | None = None,
) -> GateResult:
    """G4 provenance: order memory on permutation pairs (sig+multiset fixed)."""
    if stream is None:
        cache = prepare_host(host, max_genes=200, seed=seed, need_full_stream=True)
        stream = cache.full_stream or []
    pairs = permutation_pairs(stream, n_pairs, seed)
    if not pairs:
        return GateResult(
            name=f"g4_{host}",
            n=0,
            trained=0.0,
            random=0.0,
            margin=0.0,
            compile_floor=None,
            passed=False,
            detail="no permutation pairs",
            outcome="NEGATIVE",
            valid=False,
        )

    trained = trained or load_frozen_constellation(device=device)
    random = random or random_constellation(seed=seed, device=device)
    t_dists = [
        _cosine_dist_pair(trained.super_model, a, b, device) for a, b in pairs
    ]
    r_dists = [
        _cosine_dist_pair(random.super_model, a, b, device) for a, b in pairs
    ]
    t_mean = float(np.mean(t_dists))
    r_mean = float(np.mean(r_dists))
    margin = t_mean - r_mean
    ok = bool(margin > G4_MARGIN and t_mean > 0.1)
    return GateResult(
        name=f"g4_{host}",
        n=len(pairs),
        trained=t_mean,
        random=r_mean,
        margin=margin,
        compile_floor=None,
        passed=ok,
        detail=(
            "permutation pairs (sig+multiset fixed, order differs); "
            f"last-byte-masked; threshold margin>{G4_MARGIN} and trained>0.1"
        ),
        outcome="PASS" if ok else "NEGATIVE",
    )


def format_gate(g: GateResult) -> list[str]:
    tier = _gate_tier(g)
    lines = [
        f"census: {g.name}",
        f"  tier: {tier}",
        f"  n={g.n}",
        f"  trained={g.trained:.4f}",
        f"  random={g.random:.4f}",
        f"  tr_gap={g.margin:+.4f}",
    ]
    if g.compile_floor is not None:
        lines.append(f"  gate_floor={g.compile_floor:.4f}")
    if g.kernel_ref is not None:
        lines.append(f"  kernel_ref={g.kernel_ref:.4f}")
    lines.append(f"  detail: {g.detail}")
    lines.append(f"  outcome: {g.outcome}")
    return lines


# Margins from hqvm_cgm_genomics_results.txt anchors and suite calibration.
MARGINS: dict[str, float] = {
    "g4": 0.05,
    "prov": 0.05,
    "splice_acc": 0.75,
    "defect_band_lo": 2.5,
    "defect_band_hi": 3.5,
    "defect_abs_min": 0.05,
    "defect_null_gap": 0.05,
    "shuffle_auc_min": 0.55,
    "shuffle_null_gap": 0.05,
    "recode_auc_min": 0.55,
    "recode_null_gap": 0.05,
    "stop_macro_floor": 1.0 / 3.0,
    "stop_null_gap": 0.05,
    "climate_shell_spearman": 0.95,
    "climate_eta_min": 0.01,
    "climate_m2_max": 4096.0,
    "g4_shadow_max_dist": 0.05,
    "align_sig": 1.0,
    "align_parity": 1.0,
    "k4_err": 1e-4,
    "spec_tr_gap": 0.05,
}

SEED = 42
DEFECT_N_PERM = 24
DEFECT_N_FOLDS = 5
NARROW_WINDOWS = 4
NARROW_WALK_CAP = 512
LABEL_PERM_N = 24
SPLICE_MU_NULL_N = 24
SPLICE_MU_GAP = 0.05
TIER_ORDER: tuple[str, ...] = (
    "preflight",
    "alignment",
    "provenance",
    "specialization",
    "use-reads",
    "report",
)

# Host floors / filters applied in run_battery before calling a census.
HOST_ELIGIBILITY: dict[str, Any] = {
    "align": 20,
    "defect": 40,
    "shuffle": 40,
    "recode": 40,
    "stop": 30,
    "climate": frozenset({"ecoli", "yeast"}),
    "radial": frozenset({"ecoli", "yeast"}),
    "prov": 10,  # minimum independent within-signature groups
}

_REMOVED_LEDGER_PREFIXES: tuple[str, ...] = (
    "matrix_",
    "narrow_",
    "sheet_",
    "sheetreg_",
    "masked_",
    "annex_",
    "stop_",
    "preflight_",
    "occupy_",
    "percolation_",
    "align_sig_",
    "align_parity_",
    "defect_band_",
    "g4_shadow_",
    "rebase_",
    "defect_k4_",
    "ingress_k4_",
    "ingress_super_",
    "splice_mu_",
    "splice_chr22",
)

# When a section merge emits updates, drop prior gates under these prefixes
# that were not re-emitted (eligibility skips must not leave stale DEFERRED).
_SECTION_REPLACE_PREFIXES: tuple[str, ...] = (
    "align_sig_",
    "align_parity_",
    "occupy_",
    "stop_",
    "climate_",
    "prov_",
    "defect_band_",
    "defect_k4_",
    "defect_",
    "shuffle_",
    "recode_",
    "g4_shadow_",
    "g4_endpoint_",
    "g4_",
    "splice_",
    "rebase_",
    "ingress_",
    "radial_",
)


def _merge_gates(
    previous: list[GateResult], updates: list[GateResult]
) -> list[GateResult]:
    """Replace same-name gates; scrub removed/stale/DEFERRED; keep order stable."""
    by_name = {g.name: g for g in previous}
    order = [g.name for g in previous]

    def _drop(predicate) -> None:
        stale = [n for n in list(order) if predicate(n)]
        for n in stale:
            order.remove(n)
            by_name.pop(n, None)

    if any(g.name.startswith("percolation_") for g in updates):
        _drop(lambda n: n.startswith("percolation_"))
    _drop(lambda n: any(n.startswith(p) for p in _REMOVED_LEDGER_PREFIXES))
    # Drop any retained DEFERRED rows (end-state invariant).
    _drop(
        lambda n: by_name.get(n) is not None
        and getattr(by_name[n], "outcome", None) == "DEFERRED"
    )
    update_names = {g.name for g in updates}
    for prefix in _SECTION_REPLACE_PREFIXES:
        if any(n.startswith(prefix) for n in update_names):
            _drop(lambda n, p=prefix: n.startswith(p) and n not in update_names)
    for g in updates:
        if g.outcome == "DEFERRED":
            continue
        if g.name not in by_name:
            order.append(g.name)
        by_name[g.name] = g
    return [by_name[name] for name in order if name in by_name]


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    """Least-squares ridge; intercept is unpenalized (last coefficient)."""
    n, d = X.shape
    Xa = np.concatenate([X, np.ones((n, 1), dtype=np.float64)], axis=1)
    pen = lam * np.eye(d + 1)
    pen[-1, -1] = 0.0
    A = Xa.T @ Xa + pen
    b = Xa.T @ y
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xa = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    return Xa @ w


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom < 1e-12:
        return 0.0
    return float((ra * rb).sum() / denom)


def _stratified_split(
    labels: np.ndarray, seed: int, train_fraction: float = 0.7
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic class-preserving train/test split."""
    rng = np.random.default_rng(seed)
    train: list[int] = []
    test: list[int] = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        cut = max(1, min(len(idx) - 1, int(len(idx) * train_fraction)))
        train.extend(int(i) for i in idx[:cut])
        test.extend(int(i) for i in idx[cut:])
    rng.shuffle(train)
    rng.shuffle(test)
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def _auc_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC from scores directly (no sklearn). labels in {0,1}."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float(gt + 0.5 * eq) / float(len(pos) * len(neg))


def _gene_grouped_oof_auc(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = SEED,
) -> float:
    """5-fold gene-grouped CV with pooled OOF scores and one AUC.

    Real and null rows that share a group id stay in the same fold.
    """
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2 or len(y) < 4:
        return 0.5
    n_folds = int(min(n_folds, len(unique)))
    rng = np.random.default_rng(seed)
    order = unique.copy()
    rng.shuffle(order)
    folds = np.array_split(order, n_folds)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for fold_genes in folds:
        te = np.isin(groups, fold_genes)
        tr = ~te
        if not te.any() or not tr.any():
            continue
        if len(np.unique(y[tr])) < 2:
            continue
        w = _ridge_fit(X[tr], y[tr].astype(np.float64))
        oof[te] = _ridge_predict(X[te], w)
    valid = np.isfinite(oof)
    if int(valid.sum()) < 4 or len(np.unique(y[valid])) < 2:
        return 0.5
    return _auc_binary(oof[valid], y[valid])


def _window(stream: Sequence[int], max_len: int = 512) -> list[int]:
    return list(stream[:max_len])


def _ctx(
    model_holder: FrozenConstellation,
    stream: Sequence[int],
    *,
    full: bool = False,
    scan_extras: bool = True,
) -> np.ndarray:
    """Super context vector. ``full`` uses the whole gene stream (matrix)."""
    ledger_bytes = list(stream) if full else _window(stream)
    if scan_extras:
        return np.asarray(
            super_forward(model_holder, ledger_bytes)["context"], dtype=np.float64
        )
    if not ledger_bytes:
        return np.zeros(64, dtype=np.float64)
    ledger = torch.as_tensor(
        [ledger_bytes], dtype=torch.long, device=model_holder.device
    )
    with torch.inference_mode():
        out = model_holder.super_model(
            ledger, mask=None, hide_masked=True, scan_extras=False
        )
    return out["context"].detach().cpu().numpy().reshape(-1).astype(np.float64)


def _progress(census: str, host: str, t0: float) -> None:
    elapsed = time.perf_counter() - t0
    print(f"[suite] {census} {host} {elapsed:.1f}s", flush=True)


def _gate_tier(g: GateResult) -> str:
    name = g.name
    if name.startswith(("align_k4", "domain_gap_", "depth_loc_")):
        return "preflight"
    if name.startswith("splice_"):
        return "alignment"
    if name.startswith(("defect_", "climate_", "ingress_", "radial_")):
        return "specialization"
    if name.startswith("g4") or name.startswith("prov"):
        return "provenance"
    if name.startswith(("shuffle", "recode")):
        return "use-reads"
    return "report"


def _gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    s = seq.upper()
    return float(s.count("G") + s.count("C")) / float(len(s))


def _gc_residual(y: np.ndarray, gc: np.ndarray) -> np.ndarray:
    """Residual of y after ridge fit on 1-d GC fraction (intercept unpenalized)."""
    X = np.asarray(gc, dtype=np.float64).reshape(-1, 1)
    w = _ridge_fit(X, np.asarray(y, dtype=np.float64))
    return np.asarray(y, dtype=np.float64) - _ridge_predict(X, w)


def _label_permutation_auc(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_perm: int = LABEL_PERM_N,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Observed gene-grouped OOF AUC vs label-permutation null mean/max."""
    observed = _gene_grouped_oof_auc(X, y, groups, seed=seed)
    rng = np.random.default_rng(seed + 7919)
    perms: list[float] = []
    for _ in range(int(n_perm)):
        perms.append(
            _gene_grouped_oof_auc(X, rng.permutation(y), groups, seed=seed)
        )
    null_mean = float(np.mean(perms)) if perms else float("nan")
    null_max = float(np.max(perms)) if perms else float("nan")
    return float(observed), null_mean, null_max


def _gene_grouped_oof_spearman(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = SEED,
) -> float:
    """Gene-grouped CV with pooled OOF ridge scores and one Spearman."""
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 8 or len(y) < 8:
        return float("nan")
    n_folds = int(min(n_folds, len(unique)))
    rng = np.random.default_rng(seed)
    order = unique.copy()
    rng.shuffle(order)
    folds = np.array_split(order, n_folds)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for fold_genes in folds:
        te = np.isin(groups, fold_genes)
        tr = ~te
        if int(tr.sum()) < 8 or int(te.sum()) < 4:
            continue
        w = _ridge_fit(X[tr], y[tr])
        oof[te] = _ridge_predict(X[te], w)
    valid = np.isfinite(oof)
    if int(valid.sum()) < 8:
        return float("nan")
    return _spearman(y[valid], oof[valid])


def _label_permutation_grouped_spearman(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_perm: int = DEFECT_N_PERM,
    seed: int = SEED,
) -> tuple[float, float, float]:
    observed = abs(_gene_grouped_oof_spearman(X, y, groups, seed=seed))
    rng = np.random.default_rng(seed + 7919)
    perms: list[float] = []
    for _ in range(int(n_perm)):
        rho = _gene_grouped_oof_spearman(
            X, rng.permutation(y), groups, seed=seed
        )
        if np.isfinite(rho):
            perms.append(abs(rho))
    null_mean = float(np.mean(perms)) if perms else float("nan")
    null_max = float(np.max(perms)) if perms else float("nan")
    return float(observed), null_mean, null_max


def _specialization_binary_gate(
    *,
    name: str,
    n: int,
    Xt: np.ndarray,
    Xr: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    detail: str,
    kernel_ref: float | None = None,
) -> GateResult:
    """Trained vs random-init on the same read, plus label-permutation null."""
    auc_t, null_mean, null_max = _label_permutation_auc(Xt, y, groups, seed=seed)
    auc_r = _gene_grouped_oof_auc(Xr, y, groups, seed=seed)
    gap_null = float(auc_t - null_mean)
    gap_tr = float(auc_t - auc_r)
    ok = bool(
        auc_t >= MARGINS["shuffle_auc_min"]
        and gap_null >= MARGINS["shuffle_null_gap"]
        and gap_tr >= MARGINS["spec_tr_gap"]
    )
    gate_floor = float(MARGINS["shuffle_auc_min"])
    note = f"gate_floor={gate_floor:.4f} (preregistered AUC bar)"
    if kernel_ref is not None:
        note += f"; kernel_ref=mean_shell pooled AUC={float(kernel_ref):.4f}"
        if float(kernel_ref) > auc_t:
            note += (
                f"; mean_shell {float(kernel_ref):.4f} exceeds "
                f"trained climate_logits {auc_t:.4f} on this null"
            )
    return GateResult(
        name=name,
        n=n,
        trained=float(auc_t),
        random=float(auc_r),
        margin=gap_tr,
        compile_floor=gate_floor,
        passed=ok,
        detail=(
            f"{detail}; AUC_trained={auc_t:.4f} AUC_random={auc_r:.4f} "
            f"label_null_mean={null_mean:.4f} label_null_max={null_max:.4f}; "
            f"{note}; need AUC>={MARGINS['shuffle_auc_min']} "
            f"gap_null>={MARGINS['shuffle_null_gap']} "
            f"T-R>={MARGINS['spec_tr_gap']}"
        ),
        outcome="PASS" if ok else "NEGATIVE",
        kernel_ref=None if kernel_ref is None else float(kernel_ref),
    )


def _macro_recall(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    recalls: list[float] = []
    for k in range(n_classes):
        mask = y_true == k
        if not mask.any():
            continue
        recalls.append(float((y_pred[mask] == k).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def _family_l1_from_streams(streams: Sequence[Sequence[int]]) -> float:
    from src.tools.autoencoder.programs.genomics.genomics import _INTRON_FAMILY_IDX

    h = [0, 0, 0, 0]
    for stream in streams:
        for b in stream:
            fam = int(_INTRON_FAMILY_IDX[int(b) & 0xFF]) & 3
            h[fam] += 1
    n = sum(h) or 1
    mu = [c / n for c in h]
    return float(sum(abs(x - 0.25) for x in mu))


def _prov_group_count(cache: HostCache) -> int:
    groups: dict[int, list[list[int]]] = defaultdict(list)
    for g in cache.genes:
        window = _window(g.stream)
        if len(window) < 8:
            continue
        groups[word_signature_id(window)].append(window)
    return sum(1 for windows in groups.values() if len(windows) >= 2)


def _eligible(census: str, host: str, cache: HostCache) -> bool:
    rule = HOST_ELIGIBILITY.get(census)
    if rule is None:
        return True
    if census in ("climate", "radial"):
        return host in rule  # type: ignore[operator]
    if census == "prov":
        return _prov_group_count(cache) >= int(rule)
    if isinstance(rule, (int, float)):
        return len(cache.genes) >= int(rule)
    return True


def _reconstruct_intron_from_neighbors(prev_intron: int, next_intron: int) -> int:
    """Stride-1: masked intron bits 0-5 from prev[2:8], bits 2-8 from next[0:6]."""
    low = (int(prev_intron) >> 2) & 0x03
    high = (int(next_intron) & 0x3F) << 2
    return low | high


def occupy_oracle_shell_error(stream: Sequence[int] | None = None) -> float:
    """Oracle next-state shell error via stride-1 byte reconstruction (0 when exact)."""
    from src import api
    from src.tools.autoencoder.datasets import byte_census_arrays

    if stream is None:
        return 0.0
    if len(stream) < 3:
        return float("nan")
    table = transition_table()
    introns = byte_census_arrays()["intron_u8"]
    intron_to_byte = {int(api.INTRON_BY_BYTE[b]): b for b in range(256)}
    errs: list[float] = []
    cur = 0
    for i in range(len(stream)):
        b = int(stream[i])
        true_next = int(table[cur, b])
        true_shell = int(bin(((true_next >> 6) & 63) ^ (true_next & 63)).count("1"))
        if 0 < i < len(stream) - 1:
            prev_i = int(introns[int(stream[i - 1])])
            next_i = int(introns[int(stream[i + 1])])
            recon_i = _reconstruct_intron_from_neighbors(prev_i, next_i)
            recon_b = intron_to_byte.get(recon_i)
            if recon_b is None:
                errs.append(1.0)
            else:
                pred_next = int(table[cur, int(recon_b)])
                pred_shell = int(
                    bin(((pred_next >> 6) & 63) ^ (pred_next & 63)).count("1")
                )
                errs.append(float(abs(pred_shell - true_shell)))
        cur = true_next
    return float(np.mean(errs)) if errs else float("nan")


def masked_oracle_family_accuracy(stream: Sequence[int] | None = None) -> float:
    """Oracle family accuracy from neighbor intron bits (1.0 when exact)."""
    from src.tools.autoencoder.datasets import byte_census_arrays

    if stream is None:
        return 1.0
    if len(stream) < 3:
        return float("nan")
    census = byte_census_arrays()
    introns = census["intron_u8"]
    family = census["family_u2"]
    hits = 0
    n = 0
    for i in range(1, len(stream) - 1):
        prev_i = int(introns[int(stream[i - 1])])
        next_i = int(introns[int(stream[i + 1])])
        # family bit0 = prev intron bit2; family bit1 = next intron bit5.
        pred_fam = ((prev_i >> 2) & 1) | (((next_i >> 5) & 1) << 1)
        true_fam = int(family[int(stream[i])])
        hits += int(pred_fam == true_fam)
        n += 1
    return hits / n if n else float("nan")


def _mean_shell_auc_from_climates(
    real_shells: Sequence[float], null_shells: Sequence[float]
) -> float:
    """Pooled AUC of -mean_shell as a real-vs-null compile floor."""
    scores: list[float] = []
    labels: list[int] = []
    for ms in real_shells:
        if np.isfinite(ms):
            scores.append(-float(ms))
            labels.append(1)
    for ms in null_shells:
        if np.isfinite(ms):
            scores.append(-float(ms))
            labels.append(0)
    if len(scores) < 4 or len(set(labels)) < 2:
        return 0.5
    return _auc_binary(
        np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    )


# Back-compat alias for callers that still use the recode name.
_recode_mean_shell_auc_from_climates = _mean_shell_auc_from_climates


def _tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def domain_gap_census(caches: dict[str, HostCache]) -> list[GateResult]:
    """TV distance: uniform null-atlas prior vs empirical genomic marginals.

    Training was on kernel-null occupation (uniform / canonical / lambda). The
    genomic byte, family, and q6 histograms are the evaluation measure. Large
    TV means every genomic PASS below is measured across a domain gap.
    """
    out: list[GateResult] = []
    uniform256 = np.full(256, 1.0 / 256.0, dtype=np.float64)
    uniform4 = np.full(4, 0.25, dtype=np.float64)
    uniform64 = np.full(64, 1.0 / 64.0, dtype=np.float64)
    for host, cache in caches.items():
        streams = [g.stream for g in cache.genes if len(g.stream) >= 8]
        if cache.full_stream and len(cache.full_stream) >= 8:
            streams.append(list(cache.full_stream[:200_000]))
        if not streams:
            continue
        byte_hist = np.zeros(256, dtype=np.float64)
        fam_hist = np.zeros(4, dtype=np.float64)
        q_hist = np.zeros(64, dtype=np.float64)
        n_b = 0
        for stream in streams:
            for b in stream[:4096]:
                bi = int(b) & 0xFF
                byte_hist[bi] += 1.0
                fam_hist[int(_INTRON_FAMILY_IDX[bi]) & 3] += 1.0
                q_hist[(int(intron_from_byte(bi, 6)) >> 1) & 63] += 1.0
                n_b += 1
        if n_b < 100:
            continue
        byte_hist /= byte_hist.sum()
        fam_hist /= fam_hist.sum()
        q_hist /= q_hist.sum()
        tv_b = _tv_distance(byte_hist, uniform256)
        tv_f = _tv_distance(fam_hist, uniform4)
        tv_q = _tv_distance(q_hist, uniform64)
        mean_tv = float((tv_b + tv_f + tv_q) / 3.0)
        out.append(
            GateResult(
                name=f"domain_gap_{host}",
                n=n_b,
                trained=mean_tv,
                random=0.0,
                margin=mean_tv,
                compile_floor=0.0,
                passed=True,
                detail=(
                    f"preflight TV(uniform null atlas, genomic): "
                    f"byte={tv_b:.4f} family={tv_f:.4f} q6={tv_q:.4f}; "
                    "large TV confirms zero-shot transfer across a measure gap"
                ),
                outcome="FLOOR",
            )
        )
    return out


def depth_localization_census(cache: HostCache) -> list[GateResult]:
    """Even-depth commutators are identity; odd-depth carries order (Analysis §26)."""
    from src.api import word_signature
    from src.tools.autoencoder.programs.genomics.synthesis.field import (
        report_commutator,
    )

    even_ident = 0
    even_total = 0
    odd_nonident = 0
    odd_total = 0
    for g in cache.genes[:80]:
        s = list(g.stream)[:512]
        for depth in (2, 4):
            step = depth
            for i in range(0, len(s) - 2 * depth + 1, step):
                left = word_signature(s[i : i + depth])
                right = word_signature(s[i + depth : i + 2 * depth])
                even_ident += int(report_commutator(left, right).is_identity)
                even_total += 1
        for depth in (3, 5):
            step = depth
            for i in range(0, len(s) - 2 * depth + 1, step):
                left = word_signature(s[i : i + depth])
                right = word_signature(s[i + depth : i + 2 * depth])
                odd_nonident += int(not report_commutator(left, right).is_identity)
                odd_total += 1
    even_frac = even_ident / max(1, even_total)
    odd_frac = odd_nonident / max(1, odd_total)
    ok = bool(even_frac >= 0.999 and odd_frac > 0.0)
    return [
        GateResult(
            name=f"depth_loc_{cache.host}",
            n=even_total + odd_total,
            trained=even_frac,
            random=odd_frac,
            margin=even_frac - 1.0,
            compile_floor=0.999,
            passed=ok,
            detail=(
                f"preflight even-depth commutator identity frac={even_frac:.4f} "
                f"(theorem=1.0); odd-depth non-identity frac={odd_frac:.4f}; "
                "path memory confined to odd depth"
            ),
            outcome="FLOOR" if ok else "NEGATIVE",
        )
    ]


def align_k4_gate(trained: FrozenConstellation) -> GateResult:
    """K4 equivariance certificate (preflight; not a genomic discrimination)."""
    report = verify_k4_equivariance(trained.k4, tol=MARGINS["k4_err"])
    max_err = float(report.get("max", 1.0))
    ok = bool(report.get("passed", max_err < MARGINS["k4_err"]))
    return GateResult(
        name="align_k4",
        n=4096,
        trained=max_err,
        random=max_err,
        margin=max_err,
        compile_floor=MARGINS["k4_err"],
        passed=ok,
        detail=(
            f"preflight K4 equivariance max_error={max_err:.2e}; "
            f"gate_floor<{MARGINS['k4_err']:.0e}; certificate, not a genomics AUC"
        ),
        outcome="FLOOR" if ok else "NEGATIVE",
    )


def prov_census(
    *,
    cache: HostCache,
    trained: FrozenConstellation,
    random: FrozenConstellation,
) -> list[GateResult]:
    groups: dict[int, list[list[int]]] = defaultdict(list)
    for g in cache.genes:
        window = _window(g.stream)
        if len(window) < 8:
            continue
        sig = word_signature_id(window)
        groups[sig].append(window)
    t_w: list[float] = []
    r_w: list[float] = []
    for windows in groups.values():
        if len(windows) < 2:
            continue
        ca = _ctx(trained, windows[0])
        cb = _ctx(trained, windows[1])
        ra = _ctx(random, windows[0])
        rb = _ctx(random, windows[1])
        t_w.append(1.0 - float(np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12)))
        r_w.append(1.0 - float(np.dot(ra, rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12)))
    if len(t_w) < 10:
        return []
    t_mean = float(np.mean(t_w)) if t_w else 0.0
    r_mean = float(np.mean(r_w)) if r_w else 0.0
    margin = t_mean - r_mean
    return [
        GateResult(
            name=f"prov_{cache.host}",
            n=len(t_w),
            trained=t_mean,
            random=r_mean,
            margin=margin,
            compile_floor=None,
            passed=bool(margin > MARGINS["prov"]),
            detail=f"within-signature context distance T vs R; margin>{MARGINS['prov']}",
        )
    ]



def _binary_auc_ctx(
    real_streams: list[list[int]],
    null_streams: list[list[int]],
    holder: FrozenConstellation,
    seed: int,
    *,
    scan_extras: bool = True,
) -> float:
    """Gene-grouped 5-fold OOF AUC; real+null for the same gene stay together."""
    X: list[np.ndarray] = []
    y: list[int] = []
    groups: list[int] = []
    for i, (rs, ns) in enumerate(zip(real_streams, null_streams)):
        X.append(_ctx(holder, rs, scan_extras=scan_extras))
        y.append(1)
        groups.append(i)
        X.append(_ctx(holder, ns, scan_extras=scan_extras))
        y.append(0)
        groups.append(i)
    return _gene_grouped_oof_auc(
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.int64),
        np.asarray(groups),
        seed=seed,
    )


def _walk_states(stream: Sequence[int], max_len: int = 512) -> np.ndarray:
    """Carrier state index before each byte along a genomic walk."""
    states: list[int] = []
    cur = 0
    for b in list(stream)[:max_len]:
        states.append(cur)
        cur = int(step_index(cur, int(b)))
    if not states:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(states, dtype=np.int64)


def _narrow_window_feats(
    holder: FrozenConstellation, stream: Sequence[int], *, kind: str = "mlp"
) -> np.ndarray:
    """Contiguous-window Narrow latent means and spreads along the walk.

    The future-cone theorem uniformizes the carrier state marginal, so a
    whole-walk mean converges toward the state-space average and washes out
    gene identity. Windowed means keep along-gene position, windowed spreads
    keep local heterogeneity.
    """
    key = "k4" if kind == "k4" else "mlp"
    states = _walk_states(stream, max_len=NARROW_WALK_CAP)
    if len(states) == 0:
        dim = int(state_latents(holder, np.zeros(1, dtype=np.int64))[key].shape[1])
        return np.zeros(2 * NARROW_WINDOWS * dim, dtype=np.float64)
    z = state_latents(holder, states)[key].astype(np.float64)
    bounds = np.linspace(0, len(states), NARROW_WINDOWS + 1).astype(np.int64)
    parts: list[np.ndarray] = []
    for k in range(NARROW_WINDOWS):
        seg = z[bounds[k] : bounds[k + 1]]
        if len(seg) == 0:
            seg = z[:1]
        parts.append(seg.mean(axis=0))
        parts.append(seg.std(axis=0))
    return np.concatenate(parts)


def _plaquette_defect_series(stream: Sequence[int]) -> np.ndarray:
    """Stride-1 plaquette defect: popcount(q6_t XOR q6_{t+1}) along the byte walk."""
    if len(stream) < 2:
        return np.zeros(0, dtype=np.float64)
    q = np.asarray([int(api.q_word6(int(b))) for b in stream], dtype=np.int64)
    xor = np.bitwise_xor(q[:-1], q[1:])
    return np.asarray([int(x).bit_count() for x in xor], dtype=np.float64)


def plaquette_defect_mean(stream: Sequence[int]) -> float:
    series = _plaquette_defect_series(stream)
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


_SER_TCN = ("TCT", "TCC", "TCA", "TCG")
_SER_AGY = ("AGT", "AGC")
_CODONS = tuple("".join(p) for p in product(BASES, repeat=3))


def _one_base_neighbors(codon: str) -> list[str]:
    out: list[str] = []
    for i in range(3):
        for b in BASES:
            if b == codon[i]:
                continue
            out.append(codon[:i] + b + codon[i + 1 :])
    return out


def _sense_edge_diffs(enc, code: dict[str, str] | None = None) -> list[int]:
    code = dict(STANDARD_CODE if code is None else code)
    out: list[int] = []
    for c in _CODONS:
        if code[c] == "*":
            continue
        pc = pack_codon_bits(c, enc)
        for n in _one_base_neighbors(c):
            if code[n] == code[c]:
                out.append((pc ^ pack_codon_bits(n, enc)) & 0x3F)
    return out


def _stop_diff(enc) -> int:
    return (pack_codon_bits("TAA", enc) ^ pack_codon_bits("TGA", enc)) & 0x3F


def _serine_chords(enc) -> list[int]:
    return [
        (pack_codon_bits(a, enc) ^ pack_codon_bits(b, enc)) & 0x3F
        for a in _SER_TCN
        for b in _SER_AGY
    ]


def _span_elements(vecs: Sequence[int]) -> list[int]:
    """All linear combinations of a GF(2) generating set (rank <= 6)."""
    basis: dict[int, int] = {}
    for v in vecs:
        x = int(v) & 0x3F
        while x:
            p = x.bit_length() - 1
            if p in basis:
                x ^= basis[p]
            else:
                basis[p] = x
                break
    keys = sorted(basis.keys(), reverse=True)
    cols = [basis[k] for k in keys]
    out = [0]
    for col in cols:
        out.extend([u ^ col for u in out])
    return out


def _allowed_q_subspace(qs: set[int], *, family: int | None) -> list[int]:
    out: list[int] = []
    for q in qs:
        for b in api.BYTES_BY_Q6[q]:
            if family is None or intron_family_d(intron_from_byte(b, 6), 6) == family:
                out.append(int(b))
    return out


def _bfs_reach_from(eng, start_uv: tuple[int, int], allowed: Sequence[int]) -> tuple[int, int, int]:
    allowed_set = {int(b) for b in allowed}
    byte_idx = [i for i in range(256) if i in allowed_set]
    visited = bytearray(eng.n_omega)
    sidx = eng.uv_to_idx[start_uv]
    q: deque[int] = deque([sidx])
    visited[sidx] = 1
    while q:
        i = q.popleft()
        row = eng.transitions[i]
        for bi in byte_idx:
            j = int(row[bi])
            if not visited[j]:
                visited[j] = 1
                q.append(j)
    uv_of = {v: k for k, v in eng.uv_to_idx.items()}
    us: set[int] = set()
    vs: set[int] = set()
    n = 0
    for i in range(eng.n_omega):
        if visited[i]:
            n += 1
            u, v = uv_of[i]
            us.add(u)
            vs.add(v)
    return n, len(us), len(vs)


def kernel_percolation_ladder() -> list[tuple[str, int, int, int, int]]:
    """Analysis §6.1 / script-7 kernel percolation ladder rows.

    Each row is (name, rank, reach, predict, expect) for sense / +stop / +ser / full.
    """
    enc = pair_inversion_encoding()
    sense = _sense_edge_diffs(enc)
    stop = _stop_diff(enc)
    ser = _serine_chords(enc)
    eng = build_hqvm_d(6)
    specs = (
        ("sense", sense, 256),
        ("sense_stop", sense + [stop], 1024),
        ("sense_ser", sense + list(ser), 1024),
        ("full", sense + [stop] + list(ser), 4096),
    )
    rows: list[tuple[str, int, int, int, int]] = []
    for name, vecs, expect in specs:
        qs = set(_span_elements(vecs))
        r = int(gf2_rank(list(qs), 6))
        allow = _allowed_q_subspace(qs, family=0)
        n, _nu, _nv = _bfs_reach_from(eng, (0, 0), allow)
        pred = int(predicted_cluster_size(r))
        rows.append((name, r, int(n), pred, int(expect)))
    return rows


def _kfold_idx(n: int, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return list(np.array_split(idx, int(n_folds)))


def _oof_spearman_folds(
    X: np.ndarray, y: np.ndarray, folds: Sequence[np.ndarray]
) -> float:
    """One Spearman over pooled K-fold OOF ridge predictions (fixed folds)."""
    y = np.asarray(y, dtype=np.float64)
    all_idx = np.concatenate(folds)
    oof = np.full(len(y), np.nan)
    for te in folds:
        tr = np.setdiff1d(all_idx, te)
        if len(te) == 0 or len(tr) < 8:
            continue
        w = _ridge_fit(X[tr], y[tr])
        oof[te] = _ridge_predict(X[te], w)
    valid = np.isfinite(oof)
    if int(valid.sum()) < 8:
        return float("nan")
    return _spearman(y[valid], oof[valid])


def _oof_ridge_spearman(
    X: np.ndarray, y: np.ndarray, *, n_folds: int = DEFECT_N_FOLDS, seed: int = SEED
) -> float:
    return _oof_spearman_folds(X, y, _kfold_idx(len(y), n_folds, seed))


def _label_permutation_spearman(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_perm: int = DEFECT_N_PERM,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Observed OOF |Spearman| against a fixed-fold label-permutation null.

    Folds are drawn once. Each replicate permutes the target among genes and
    reruns the full OOF ridge, so the null varies only through the labels.
    Returns (observed_abs, null_mean_abs, null_max_abs).
    """
    folds = _kfold_idx(len(y), DEFECT_N_FOLDS, seed)
    observed = abs(_oof_spearman_folds(X, y, folds))
    rng = np.random.default_rng(seed + 7919)
    perms: list[float] = []
    for _ in range(int(n_perm)):
        rho = _oof_spearman_folds(X, rng.permutation(y), folds)
        if np.isfinite(rho):
            perms.append(abs(rho))
    null_mean = float(np.mean(perms)) if perms else float("nan")
    null_max = float(np.max(perms)) if perms else float("nan")
    return float(observed), null_mean, null_max


def shuffle_census(
    *,
    cache: HostCache,
    trained: FrozenConstellation,
    random: FrozenConstellation,
    seed: int = SEED,
) -> list[GateResult]:
    if len(cache.genes) < 40:
        return []
    nulls = null_gene_streams(cache.genes, kind="shuffle", seed=seed)
    reals = [g.stream for g in cache.genes]
    X: list[np.ndarray] = []
    y: list[int] = []
    groups: list[int] = []
    for i, (rs, ns) in enumerate(zip(reals, nulls)):
        X.append(_ctx(trained, rs))
        y.append(1)
        groups.append(i)
        X.append(_ctx(trained, ns))
        y.append(0)
        groups.append(i)
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.int64)
    ga = np.asarray(groups)
    auc_t, null_mean, null_max = _label_permutation_auc(Xa, ya, ga, seed=seed)
    auc_r = _binary_auc_ctx(reals, nulls, random, seed)
    gap = auc_t - null_mean
    ok = bool(
        auc_t >= MARGINS["shuffle_auc_min"]
        and gap >= MARGINS["shuffle_null_gap"]
    )
    return [
        GateResult(
            name=f"shuffle_{cache.host}",
            n=len(reals),
            trained=auc_t,
            random=auc_r,
            margin=gap,
            compile_floor=MARGINS["shuffle_auc_min"],
            passed=ok,
            detail=(
                "trained Super context AUC real vs gc_matched_shuffle; "
                f"AUC>={MARGINS['shuffle_auc_min']} and "
                f"AUC-label_perm_null>={MARGINS['shuffle_null_gap']} "
                f"(null_mean={null_mean:.4f}, null_max={null_max:.4f}); "
                f"random_init_AUC={auc_r:.4f} informational"
            ),
            outcome="PASS" if ok else "NEGATIVE",
        )
    ]


def recode_census(
    *,
    cache: HostCache,
    trained: FrozenConstellation,
    random: FrozenConstellation,
    seed: int = SEED,
) -> list[GateResult]:
    if len(cache.genes) < 40:
        return []
    from src.tools.autoencoder.programs.genomics.synthesis.climate import (
        protein_fixed_resample,
    )

    rng = np.random.default_rng(seed)
    null_seqs = [protein_fixed_resample(g.seq, rng=rng) for g in cache.genes]
    nulls = [byte_stream(s) for s in null_seqs]
    reals = [g.stream for g in cache.genes]
    X: list[np.ndarray] = []
    y: list[int] = []
    groups: list[int] = []
    for i, (rs, ns) in enumerate(zip(reals, nulls)):
        X.append(_ctx(trained, rs))
        y.append(1)
        groups.append(i)
        X.append(_ctx(trained, ns))
        y.append(0)
        groups.append(i)
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.int64)
    ga = np.asarray(groups)
    auc_t, null_mean, null_max = _label_permutation_auc(Xa, ya, ga, seed=seed)
    auc_r = _binary_auc_ctx(reals, nulls, random, seed)
    gap = auc_t - null_mean
    ok = bool(
        auc_t >= MARGINS["recode_auc_min"]
        and gap >= MARGINS["recode_null_gap"]
    )
    real_shells = [float(g.climate.mean_shell) for g in cache.genes]
    null_shells = [
        float(climate_of_sequence(s, certificates=False).mean_shell) for s in null_seqs
    ]
    floor = _mean_shell_auc_from_climates(real_shells, null_shells)
    tr_gap = float(auc_t - auc_r)
    return [
        GateResult(
            name=f"recode_{cache.host}",
            n=len(reals),
            trained=auc_t,
            random=auc_r,
            margin=tr_gap,
            compile_floor=MARGINS["recode_auc_min"],
            passed=ok,
            detail=(
                "use-reads Super context AUC real vs protein_fixed_resample; "
                f"preregistered on label-null bar (not spec_tr_gap); "
                f"AUC>={MARGINS['recode_auc_min']} and "
                f"AUC-label_perm_null>={MARGINS['recode_null_gap']} "
                f"(null_mean={null_mean:.4f}, null_max={null_max:.4f}, "
                f"gap_null={gap:.4f}); "
                f"kernel_ref=mean_shell pooled AUC={floor:.4f}; "
                f"random_init_AUC={auc_r:.4f} informational; "
                f"tr_gap={tr_gap:+.4f}"
            ),
            outcome="PASS" if ok else "NEGATIVE",
            kernel_ref=floor,
        )
    ]


def splice_census(
    *,
    trained: FrozenConstellation,
    random: FrozenConstellation,
    seed: int = SEED,
) -> list[GateResult]:
    """Junction motif floors: Narrow/K4 on §20 8-bp flanks; Super on 128-bp windows.

    Donor GT at flank positions 2-3 vs acceptor AG at 4-5 is linearly present
    in carrier-state latents for trained and random-init encoders. This is an
    alignment motif certificate, not a trained-weight claim.
    """
    out: list[GateResult] = []
    donors_f, acceptors_f = extract_chr22_splice_flanks(max_per_class=400)
    if len(donors_f) >= 20 and len(acceptors_f) >= 20:
        d_f = [byte_stream(w) for w in donors_f]
        a_f = [byte_stream(w) for w in acceptors_f]
        streams = d_f + a_f
        yf = np.asarray([1] * len(d_f) + [0] * len(a_f), dtype=np.int64)
        gf = np.asarray(
            list(range(len(d_f))) + list(range(20000, 20000 + len(a_f))),
            dtype=np.int64,
        )
        y_float = yf.astype(np.float64)
        for kind, name, label in (
            ("mlp", "splice_narrow_chr22", "Narrow MLP"),
            ("k4", "splice_k4_flanks", "General K4"),
        ):
            Xt = np.asarray(
                [_narrow_window_feats(trained, s, kind=kind) for s in streams],
                dtype=np.float64,
            )
            Xr = np.asarray(
                [_narrow_window_feats(random, s, kind=kind) for s in streams],
                dtype=np.float64,
            )
            auc_t, null_mean, null_max = _label_permutation_auc(
                Xt, yf, gf, seed=seed
            )
            auc_r = _gene_grouped_oof_auc(Xr, yf, gf, seed=seed)
            tr, te = _stratified_split(y_float, seed)
            w = _ridge_fit(Xt[tr], y_float[tr])
            pred = (_ridge_predict(Xt[te], w) >= 0.5).astype(np.int64)
            accuracy = float((pred == yf[te]).mean())
            ok = bool(
                accuracy >= MARGINS["splice_acc"]
                and auc_t >= MARGINS["shuffle_auc_min"]
            )
            out.append(
                GateResult(
                    name=name,
                    n=len(donors_f) + len(acceptors_f),
                    trained=float(auc_t),
                    random=float(auc_r),
                    margin=float(auc_t - auc_r),
                    compile_floor=MARGINS["splice_acc"],
                    passed=ok,
                    detail=(
                        f"alignment {label} 8-bp flanks donor GT vs acceptor AG "
                        f"(motif floor); acc={accuracy:.4f}; AUC_trained={auc_t:.4f} "
                        f"AUC_random={auc_r:.4f} label_null_mean={null_mean:.4f} "
                        f"label_null_max={null_max:.4f}; architecture floor"
                    ),
                    outcome="FLOOR" if ok else "NEGATIVE",
                )
            )

    donors_w, acceptors_w = extract_chr22_splice_windows(max_per_class=200)
    if len(donors_w) >= 20 and len(acceptors_w) >= 20:
        d_streams = [byte_stream(w) for w in donors_w]
        a_streams = [byte_stream(w) for w in acceptors_w]
        y = np.asarray(
            [1] * len(d_streams) + [0] * len(a_streams), dtype=np.float64
        )

        def flank_accuracy(holder: FrozenConstellation) -> float:
            X = np.asarray(
                [_ctx(holder, s) for s in d_streams + a_streams], dtype=np.float64
            )
            tr2, te2 = _stratified_split(y, seed)
            w = _ridge_fit(X[tr2], y[tr2])
            pred = (_ridge_predict(X[te2], w) >= 0.5).astype(np.int64)
            return float((pred == y[te2].astype(np.int64)).mean())

        t_acc = flank_accuracy(trained)
        r_acc = flank_accuracy(random)
        ok_s = bool(t_acc >= MARGINS["splice_acc"])
        out.append(
            GateResult(
                name="splice_super_chr22",
                n=len(donors_w) + len(acceptors_w),
                trained=t_acc,
                random=r_acc,
                margin=float(t_acc - r_acc),
                compile_floor=MARGINS["splice_acc"],
                passed=ok_s,
                detail=(
                    "alignment Super scan floor donor vs acceptor 128-bp windows; "
                    f"acc_trained={t_acc:.4f} acc_random={r_acc:.4f}; "
                    "ExactHQVMScan separates junctions; architecture floor"
                ),
                outcome="FLOOR" if ok_s else "NEGATIVE",
            )
        )
    return out


def climate_census(
    *,
    cache: HostCache,
    trained: FrozenConstellation,
    random: FrozenConstellation,
    seed: int = SEED,
) -> list[GateResult]:
    """Super climate_logits vs GC-shuffle; random-init is the analytic channel.

    climate_logits = q_weight table * log(lambda_hat) + climate_head(q_hist).
    Random-init Super keeps the frozen analytic term. Weight credit is trained
    minus random on the same GC-shuffle pairs. compile_floor is pooled
    mean_shell AUC on the same real-vs-GC-shuffle pairs (kernel climate read).
    """
    from src.tools.autoencoder.programs.genomics.synthesis.climate import (
        climate_of_sequence,
        gc_matched_shuffle,
    )

    genes = [g for g in cache.genes if len(g.stream) >= 8]
    if len(genes) < 40:
        return []
    rng = np.random.default_rng(seed)
    Xt: list[np.ndarray] = []
    Xr: list[np.ndarray] = []
    y: list[int] = []
    groups: list[int] = []
    real_shells: list[float] = []
    null_shells: list[float] = []
    for i, g in enumerate(genes):
        null_seq = gc_matched_shuffle(g.seq, rng=rng)
        null_stream = byte_stream(null_seq)[:512]
        real = g.stream[:512]
        Xt.append(
            np.asarray(super_forward(trained, real)["climate_logits"], dtype=np.float64).ravel()
        )
        Xt.append(
            np.asarray(
                super_forward(trained, null_stream)["climate_logits"], dtype=np.float64
            ).ravel()
        )
        Xr.append(
            np.asarray(super_forward(random, real)["climate_logits"], dtype=np.float64).ravel()
        )
        Xr.append(
            np.asarray(
                super_forward(random, null_stream)["climate_logits"], dtype=np.float64
            ).ravel()
        )
        y.extend([1, 0])
        groups.extend([i, i])
        real_shells.append(float(g.climate.mean_shell))
        null_shells.append(
            float(climate_of_sequence(null_seq, certificates=False).mean_shell)
        )
    floor = _mean_shell_auc_from_climates(real_shells, null_shells)
    return [
        _specialization_binary_gate(
            name=f"climate_super_{cache.host}",
            n=len(genes),
            Xt=np.asarray(Xt, dtype=np.float64),
            Xr=np.asarray(Xr, dtype=np.float64),
            y=np.asarray(y, dtype=np.int64),
            groups=np.asarray(groups),
            seed=seed,
            detail=(
                "specialization Super climate_logits real vs GC-shuffle "
                "(Analysis §22); random-init is analytic q-hist channel"
            ),
            kernel_ref=floor,
        )
    ]


def radial_census(
    *,
    cache: HostCache,
    trained: FrozenConstellation,
    random: FrozenConstellation,
    seed: int = SEED,
) -> list[GateResult]:
    """Super climate_logits vs synonymous recode; random-init is analytic channel."""
    from src.tools.autoencoder.programs.genomics.synthesis.climate import (
        protein_fixed_resample,
    )

    genes = [g for g in cache.genes if len(g.stream) >= 8]
    if len(genes) < 40:
        return []
    rng = np.random.default_rng(seed)
    Xt: list[np.ndarray] = []
    Xr: list[np.ndarray] = []
    y: list[int] = []
    groups: list[int] = []
    for i, g in enumerate(genes):
        null_stream = byte_stream(protein_fixed_resample(g.seq, rng=rng))[:512]
        real = g.stream[:512]
        Xt.append(
            np.asarray(super_forward(trained, real)["climate_logits"], dtype=np.float64).ravel()
        )
        Xt.append(
            np.asarray(
                super_forward(trained, null_stream)["climate_logits"], dtype=np.float64
            ).ravel()
        )
        Xr.append(
            np.asarray(super_forward(random, real)["climate_logits"], dtype=np.float64).ravel()
        )
        Xr.append(
            np.asarray(
                super_forward(random, null_stream)["climate_logits"], dtype=np.float64
            ).ravel()
        )
        y.extend([1, 0])
        groups.extend([i, i])
    return [
        _specialization_binary_gate(
            name=f"radial_super_{cache.host}",
            n=len(genes),
            Xt=np.asarray(Xt, dtype=np.float64),
            Xr=np.asarray(Xr, dtype=np.float64),
            y=np.asarray(y, dtype=np.int64),
            groups=np.asarray(groups),
            seed=seed,
            detail=(
                "specialization Super climate_logits real vs synonymous recode "
                "(Analysis §42); random-init is analytic q-hist channel"
            ),
        )
    ]


def g4_extended_census(
    *,
    host: str,
    n_pairs: int,
    seed: int,
    device: str,
    trained: FrozenConstellation,
    random: FrozenConstellation,
    cache: HostCache | None = None,
) -> list[GateResult]:
    """Same-sig and same-endpoint Super context distance (trained vs random)."""
    stream: Sequence[int] | None = None
    if cache is not None and cache.full_stream:
        stream = cache.full_stream
    gates = [
        g4_census(
            host=host,
            n_pairs=n_pairs,
            seed=seed,
            device=device,
            trained=trained,
            random=random,
            stream=stream,
        )
    ]
    if cache is None:
        cache = prepare_host(host, max_genes=200, seed=seed, need_full_stream=True)
    stream_list = cache.full_stream or (cache.genes[0].stream if cache.genes else [])
    if len(stream_list) < 1000:
        return gates
    end_map: dict[int, list[list[int]]] = defaultdict(list)
    table = transition_table()
    for i in range(0, min(len(stream_list) - 4, 40000), 7):
        word = [int(stream_list[i + k]) for k in range(4)]
        s = 0
        for b in word:
            s = int(table[s, b])
        end_map[s].append(word)
    ep_pairs = []
    for words in end_map.values():
        if len(words) < 2:
            continue
        if words[0] != words[1]:
            ep_pairs.append((words[0], words[1]))
        if len(ep_pairs) >= n_pairs:
            break
    # Calibration: mean distance over arbitrary non-equivalent pairs.
    rng = np.random.default_rng(seed + 17)
    cal_pairs: list[tuple[list[int], list[int]]] = []
    n_stream = len(stream_list)
    while len(cal_pairs) < min(200, n_pairs) and n_stream >= 8:
        i = int(rng.integers(0, n_stream - 4))
        j = int(rng.integers(0, n_stream - 4))
        a = [int(stream_list[i + k]) for k in range(4)]
        b = [int(stream_list[j + k]) for k in range(4)]
        if a != b:
            cal_pairs.append((a, b))
    cal_t = float(
        np.mean(
            [_cosine_dist_pair(trained.super_model, a, b, device) for a, b in cal_pairs]
        )
    ) if cal_pairs else float("nan")
    cal_r = float(
        np.mean(
            [_cosine_dist_pair(random.super_model, a, b, device) for a, b in cal_pairs]
        )
    ) if cal_pairs else float("nan")
    if ep_pairs:
        t_e = [
            _cosine_dist_pair(trained.super_model, a, b, device) for a, b in ep_pairs
        ]
        r_e = [
            _cosine_dist_pair(random.super_model, a, b, device) for a, b in ep_pairs
        ]
        t_mean = float(np.mean(t_e))
        r_mean = float(np.mean(r_e))
        margin = t_mean - r_mean
        ok = bool(margin > MARGINS["g4"] and t_mean > 0.1)
        gates.append(
            GateResult(
                name=f"g4_endpoint_{host}",
                n=len(t_e),
                trained=t_mean,
                random=r_mean,
                margin=margin,
                compile_floor=None,
                passed=ok,
                detail=(
                    "supporting same-endpoint collision pairs "
                    "(lead provenance is permutation-pair g4); "
                    f"gated margin>{G4_MARGIN}; "
                    f"calib_non_equiv trained={cal_t:.4f} random={cal_r:.4f} "
                    "(non-equiv calibration tracks context spread)"
                ),
                outcome="PASS" if ok else "NEGATIVE",
            )
        )
    return gates


def _gate_to_dict(g: GateResult) -> dict[str, Any]:
    return {
        "name": g.name,
        "n": g.n,
        "trained": g.trained,
        "random": g.random,
        "margin": g.margin,
        "compile_floor": g.compile_floor,
        "kernel_ref": g.kernel_ref,
        "tier": _gate_tier(g),
        "passed": g.passed,
        "outcome": g.outcome,
        "valid": g.valid,
        "detail": g.detail,
    }


def _dict_to_gate(d: dict[str, Any]) -> GateResult:
    return GateResult(
        name=str(d["name"]),
        n=int(d["n"]),
        trained=float(d["trained"]),
        random=float(d["random"]),
        margin=float(d["margin"]),
        compile_floor=(
            None if d.get("compile_floor") is None else float(d["compile_floor"])
        ),
        passed=bool(d["passed"]),
        detail=str(d.get("detail", "")),
        outcome=d.get("outcome"),  # type: ignore[arg-type]
        valid=bool(d.get("valid", True)),
        kernel_ref=(
            None if d.get("kernel_ref") is None else float(d["kernel_ref"])
        ),
    )


def _load_gates_ledger(path: Path) -> list[GateResult]:
    if not path.exists():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw = doc.get("gates") if isinstance(doc, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[GateResult] = []
    for item in raw:
        if isinstance(item, dict) and "name" in item:
            out.append(_dict_to_gate(item))
    return out


def _header_lines(
    *,
    seed: int,
    device: str,
    hosts: Sequence[str],
    max_genes: int,
    n_pairs: int,
) -> list[str]:
    lines = [
        "genomics synthesis",
        f"when_utc: {_now()}",
        "domain_training: none",
        "model_training_domain: hQVM kernel-null carrier and ledger structure",
        "genomic_finetuning: none",
        "checkpoint_policy: frozen",
        "evaluation_domain: community genomic catalogs (zero-shot)",
        f"seed: {seed}",
        f"device: {device}",
        f"hosts: {','.join(hosts)}",
        f"max_genes: {max_genes}",
        f"n_pairs: {n_pairs}",
    ]
    try:
        from src.tools.autoencoder.programs.genomics.synthesis.constellation import (
            production_checkpoint_paths,
        )
        import hashlib

        for name, path in sorted(production_checkpoint_paths().items()):
            if path.exists():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                lines.append(f"checkpoint_{name}_sha256_16: {digest}")
    except Exception:
        pass
    lines.append("")
    return lines


def _write_artifacts(
    *,
    gates: list[GateResult],
    seed: int,
    device: str,
    hosts: Sequence[str],
    max_genes: int,
    n_pairs: int,
    results_file: Path | None,
    gates_file: Path | None,
) -> dict[str, Any]:
    lines = _header_lines(
        seed=seed,
        device=device,
        hosts=hosts,
        max_genes=max_genes,
        n_pairs=n_pairs,
    )
    by_tier: dict[str, list[GateResult]] = {t: [] for t in TIER_ORDER}
    for g in gates:
        by_tier.setdefault(_gate_tier(g), []).append(g)
    for tier in TIER_ORDER:
        tier_gates = by_tier.get(tier) or []
        if not tier_gates:
            continue
        lines.append(tier)
        lines.append("")
        for g in tier_gates:
            lines.extend(format_gate(g))
            lines.append("")
    out_path = results_file or RESULTS
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "status": "ok",
        "margins": MARGINS,
        "gates": [_gate_to_dict(g) for g in gates],
        "meta": {
            "seed": seed,
            "device": device,
            "hosts": list(hosts),
            "max_genes": max_genes,
            "n_pairs": n_pairs,
            "when_utc": _now(),
        },
    }
    gpath = gates_file or GATES_JSON
    gpath.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "ok",
        "gates": gates,
        "report_lines": lines,
        "results_file": str(out_path),
        "gates_file": str(gpath),
    }




def run_battery(
    *,
    only: str | None = None,
    hosts: Sequence[str] = ("ecoli", "yeast", "sars"),
    n_pairs: int = 400,
    max_genes: int = 200,
    seed: int = SEED,
    device: str = "cpu",
    results_file: Path | None = None,
    gates_file: Path | None = None,
) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True, warn_only=True)
    assert_registry_complete()
    sections = {
        "align",
        "g4",
        "climate",
        "prov",
        "shuffle",
        "recode",
        "radial",
        "splice",
    }
    if only is not None:
        if only not in sections:
            raise ValueError(f"--only must be one of {sorted(sections)}")
        sections = {only}

    ledger_path = gates_file or GATES_JSON
    previous = _load_gates_ledger(ledger_path) if only is not None else []
    results_path = results_file or RESULTS
    gates_path = gates_file or GATES_JSON

    trained = load_frozen_constellation(device=device)
    random = random_constellation(seed=seed, device=device)
    updates: list[GateResult] = []

    host_caches: dict[str, HostCache] = {}
    for host in hosts:
        host_caches[host] = prepare_host(
            host,
            max_genes=max_genes,
            seed=seed,
            need_full_stream=("g4" in sections),
        )
        if len(host_caches[host].genes) < 1:
            return {
                "status": "blocked",
                "gates": updates,
                "results_file": str(results_path),
                "gates_file": str(gates_path),
            }

    if "align" in sections:
        t0 = time.perf_counter()
        updates.append(align_k4_gate(trained))
        updates.extend(domain_gap_census(host_caches))
        for host in hosts:
            if host in ("ecoli", "yeast") and host in host_caches:
                updates.extend(depth_localization_census(host_caches[host]))
        _progress("align", "preflight", t0)

    if "g4" in sections:
        # Zero-shot human chr22 chromosome stream (ACGT-only; N dropped).
        try:
            from src.tools.autoencoder.programs.genomics.genomics import GENOMICS_DIR
            from src.tools.autoencoder.programs.genomics.synthesis.reads import (
                read_fasta_seq,
            )

            chr22_path = GENOMICS_DIR / "chr22.fa.gz"
            if chr22_path.exists():
                t0 = time.perf_counter()
                seq = read_fasta_seq(chr22_path, max_bases=2_000_000, keep_n=False)
                stream22 = byte_stream(seq)
                updates.append(
                    g4_census(
                        host="chr22",
                        n_pairs=n_pairs,
                        seed=seed,
                        device=device,
                        trained=trained,
                        random=random,
                        stream=stream22,
                    )
                )
                _progress("g4", "chr22", t0)
        except Exception as exc:
            print(f"[suite] skip g4 chr22 ({exc})", flush=True)

    for host in hosts:
        cache = host_caches[host]
        if "g4" in sections:
            t0 = time.perf_counter()
            updates.extend(
                g4_extended_census(
                    host=host,
                    n_pairs=n_pairs,
                    seed=seed,
                    device=device,
                    trained=trained,
                    random=random,
                    cache=cache,
                )
            )
            _progress("g4", host, t0)
        if "climate" in sections:
            if not _eligible("climate", host, cache):
                print(f"[suite] skip climate {host}", flush=True)
            else:
                t0 = time.perf_counter()
                updates.extend(
                    climate_census(
                        cache=cache, trained=trained, random=random, seed=seed
                    )
                )
                _progress("climate", host, t0)
        if "prov" in sections:
            if not _eligible("prov", host, cache):
                print(f"[suite] skip prov {host}", flush=True)
            else:
                t0 = time.perf_counter()
                updates.extend(
                    prov_census(cache=cache, trained=trained, random=random)
                )
                _progress("prov", host, t0)
        if "shuffle" in sections:
            if not _eligible("shuffle", host, cache):
                print(
                    f"[suite] skip shuffle {host} (n={len(cache.genes)}<{HOST_ELIGIBILITY['shuffle']})",
                    flush=True,
                )
            else:
                t0 = time.perf_counter()
                updates.extend(
                    shuffle_census(
                        cache=cache, trained=trained, random=random, seed=seed
                    )
                )
                _progress("shuffle", host, t0)
        if "recode" in sections:
            if not _eligible("recode", host, cache):
                print(
                    f"[suite] skip recode {host} (n={len(cache.genes)}<{HOST_ELIGIBILITY['recode']})",
                    flush=True,
                )
            else:
                t0 = time.perf_counter()
                updates.extend(
                    recode_census(
                        cache=cache, trained=trained, random=random, seed=seed
                    )
                )
                _progress("recode", host, t0)
        if "radial" in sections:
            if not _eligible("radial", host, cache):
                print(f"[suite] skip radial {host}", flush=True)
            else:
                t0 = time.perf_counter()
                updates.extend(
                    radial_census(
                        cache=cache, trained=trained, random=random, seed=seed
                    )
                )
                _progress("radial", host, t0)

    if "splice" in sections:
        t0 = time.perf_counter()
        updates.extend(splice_census(trained=trained, random=random, seed=seed))
        _progress("splice", "chr22", t0)

    if only is None:
        gates = updates
    else:
        gates = _merge_gates(previous, updates)

    return _write_artifacts(
        gates=gates,
        seed=seed,
        device=device,
        hosts=hosts,
        max_genes=max_genes,
        n_pairs=n_pairs,
        results_file=results_file,
        gates_file=gates_file,
    )


def run_suite(
    *,
    only: str | None = None,
    hosts: Sequence[str] = ("ecoli", "yeast", "sars"),
    n_pairs: int = 400,
    max_genes: int = 200,
    seed: int = 42,
    device: str = "cpu",
    results_file: Path | None = None,
) -> dict[str, Any]:
    """Run the genomics synthesis census battery; write RESULTS.txt and gates.json."""
    from src.tools.autoencoder.programs.genomics.synthesis.constellation import (
        production_checkpoint_paths,
    )

    missing = [
        str(p) for p in production_checkpoint_paths().values() if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "missing production checkpoints: " + ", ".join(missing)
        )

    return run_battery(
        only=only,
        hosts=hosts,
        n_pairs=n_pairs,
        max_genes=max_genes,
        seed=seed,
        device=device,
        results_file=results_file or RESULTS,
    )


run_transfer = run_suite

