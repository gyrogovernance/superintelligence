#!/usr/bin/env python3
"""Fixed-peptide synonymous order correspondence on Gamble 2016 Table S1.

Within each encoded tripeptide, protein identity and length are fixed while
codon order varies. Frozen AE and kernel reads are scored before any
expression information is used. The primary statistic is the within-peptide
Spearman rank between each read and synGFPSEQ on peptide groups with at
least five variants, evaluated on the full eligible set and on peptide-held-out
folds.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[5]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common import RESULT_JSON, require, sha256_file, upsert_results, write_json
from src.tools.autoencoder import paths
from src.tools.autoencoder.programs.genomics.genomics import (
    clean_acgt,
    pair_inversion_encoding,
)
from src.tools.autoencoder.programs.genomics.synthesis.constellation import (
    load_frozen_constellation,
    production_checkpoint_paths,
    read_constellation,
)

_DEFAULT_TABLE = (
    paths.dataset_dir("topology") / "gamble_2016__NIHMS800838-supplement-6.xlsx"
)
_MIN_GROUP = 5
_HOLDOUT_FOLDS = 5
_SEED = 20260922
_ENC = pair_inversion_encoding()

_BASES = "TCAG"
_AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
_CODE = {
    b1 + b2 + b3: _AA[i]
    for i, (b1, b2, b3) in enumerate(product(_BASES, repeat=3))
}


def translate0(seq: str) -> str:
    return "".join(_CODE.get(seq[i : i + 3], "?") for i in range(0, len(seq) - len(seq) % 3, 3))


def gc_frac(seq: str) -> float:
    if not seq:
        return float("nan")
    return float(sum(1 for b in seq if b in "GC")) / float(len(seq))


def codon_count_vector(seq: str) -> np.ndarray:
    counts = np.zeros(64, dtype=np.float64)
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        if len(codon) != 3:
            continue
        try:
            idx = _BASES.index(codon[0]) * 16 + _BASES.index(codon[1]) * 4 + _BASES.index(codon[2])
        except ValueError:
            continue
        counts[idx] += 1.0
    return counts


def within_group_spearman(
    scores: np.ndarray,
    expression: np.ndarray,
    groups: np.ndarray,
    *,
    min_n: int,
) -> dict[str, Any]:
    by_group: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for score, expr, group in zip(scores, expression, groups, strict=True):
        if not np.isfinite(score) or not np.isfinite(expr):
            continue
        by_group[str(group)].append((float(score), float(expr)))

    rhos: list[float] = []
    sizes: list[int] = []
    for pairs in by_group.values():
        if len(pairs) < min_n:
            continue
        xs = np.asarray([p[0] for p in pairs], dtype=np.float64)
        ys = np.asarray([p[1] for p in pairs], dtype=np.float64)
        if np.unique(xs).size < 2 or np.unique(ys).size < 2:
            continue
        rho = float(spearmanr(xs, ys).correlation)
        if not np.isfinite(rho):
            continue
        rhos.append(rho)
        sizes.append(len(pairs))

    if not rhos:
        return {
            "n_groups": 0,
            "n_variants": 0,
            "mean_rho": float("nan"),
            "median_rho": float("nan"),
            "positive_fraction": float("nan"),
            "group_rhos": [],
        }
    arr = np.asarray(rhos, dtype=np.float64)
    return {
        "n_groups": int(len(arr)),
        "n_variants": int(sum(sizes)),
        "mean_rho": float(arr.mean()),
        "median_rho": float(np.median(arr)),
        "positive_fraction": float(np.mean(arr > 0.0)),
        "group_rhos": [float(x) for x in arr],
    }


def sign_flip_p(values: list[float], *, seed: int, permutations: int = 5000) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    observed = float(arr.mean())
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(permutations):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(arr))
        if float((arr * signs).mean()) >= observed:
            count += 1
    return float((count + 1) / (permutations + 1))


def peptide_holdout_mean(
    scores: np.ndarray,
    expression: np.ndarray,
    groups: np.ndarray,
    *,
    min_n: int,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    unique = np.asarray(sorted({str(g) for g in groups}), dtype=object)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    fold_ids = np.empty(len(unique), dtype=np.int64)
    for i, idx in enumerate(order):
        fold_ids[idx] = i % folds
    peptide_to_fold = {str(unique[i]): int(fold_ids[i]) for i in range(len(unique))}

    fold_means: list[float] = []
    for fold in range(folds):
        mask = np.asarray([peptide_to_fold[str(g)] == fold for g in groups], dtype=bool)
        summary = within_group_spearman(
            scores[mask], expression[mask], groups[mask], min_n=min_n
        )
        if summary["n_groups"] > 0 and np.isfinite(summary["mean_rho"]):
            fold_means.append(float(summary["mean_rho"]))
    if not fold_means:
        return {
            "n_folds": 0,
            "mean_rho": float("nan"),
            "fold_means": [],
            "positive_folds": 0,
        }
    arr = np.asarray(fold_means, dtype=np.float64)
    return {
        "n_folds": int(len(arr)),
        "mean_rho": float(arr.mean()),
        "fold_means": [float(x) for x in arr],
        "positive_folds": int(np.sum(arr > 0.0)),
    }


def score_variants(
    variants: list[str],
    *,
    device: str,
) -> dict[str, np.ndarray]:
    holder = load_frozen_constellation(device=device)
    n = len(variants)
    narrow = np.empty(n, dtype=np.float64)
    super_climate = np.empty(n, dtype=np.float64)
    super_context_norm = np.empty(n, dtype=np.float64)
    kernel_shell = np.empty(n, dtype=np.float64)
    gc = np.empty(n, dtype=np.float64)
    codon_count_energy = np.empty(n, dtype=np.float64)

    # Host-independent codon identity energy: negative entropy of the 3-codon multiset.
    for i, raw in enumerate(variants):
        seq = clean_acgt(str(raw).upper().replace("U", "T"))
        require(len(seq) == 9, f"expected length-9 variant, got {len(seq)}")
        read = read_constellation(holder, seq, enc=_ENC)
        narrow[i] = float(read.narrow_het)
        super_climate[i] = float(read.super_climate_mean)
        context = np.asarray(read.super_context, dtype=np.float64)
        super_context_norm[i] = float(np.linalg.norm(context))
        kernel_shell[i] = float(read.kernel_mean_shell)
        gc[i] = gc_frac(seq)
        counts = codon_count_vector(seq)
        total = float(counts.sum())
        if total <= 0:
            codon_count_energy[i] = float("nan")
        else:
            probs = counts[counts > 0] / total
            codon_count_energy[i] = float(-(probs * np.log(probs)).sum())
        if (i + 1) % 2000 == 0 or i + 1 == n:
            print(f"scored {i + 1}/{n}", flush=True)

    return {
        "narrow_het": narrow,
        "super_climate": super_climate,
        "super_context_norm": super_context_norm,
        "kernel_mean_shell": kernel_shell,
        "gc": gc,
        "codon_count_entropy": codon_count_energy,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=_DEFAULT_TABLE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-group", type=int, default=_MIN_GROUP)
    parser.add_argument("--holdout-folds", type=int, default=_HOLDOUT_FOLDS)
    parser.add_argument("--seed", type=int, default=_SEED)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)

    table_path = Path(args.table).resolve()
    require(table_path.is_file(), f"missing Gamble table: {table_path}")
    df = pd.read_excel(table_path)
    require({"Variant", "synGFPSEQ"}.issubset(df.columns), "Gamble table missing required columns")

    variants = [clean_acgt(str(v).upper().replace("U", "T")) for v in df["Variant"].tolist()]
    expression = np.asarray(df["synGFPSEQ"], dtype=np.float64)
    peptides = np.asarray([translate0(v) for v in variants], dtype=object)
    keep = np.asarray(
        [
            len(v) == 9 and "*" not in p and "?" not in p
            for v, p in zip(variants, peptides, strict=True)
        ],
        dtype=bool,
    )
    variants = [v for v, ok in zip(variants, keep, strict=True) if ok]
    expression = expression[keep]
    peptides = peptides[keep]

    counts = pd.Series(peptides).value_counts()
    eligible_peptides = set(counts[counts >= int(args.min_group)].index.astype(str))
    eligible = np.asarray([str(p) in eligible_peptides for p in peptides], dtype=bool)
    variants = [v for v, ok in zip(variants, eligible, strict=True) if ok]
    expression = expression[eligible]
    peptides = peptides[eligible]

    print(
        f"eligible variants={len(variants)} peptides={len(set(map(str, peptides)))} "
        f"min_group={args.min_group}",
        flush=True,
    )
    scored = score_variants(variants, device=str(args.device))

    reads = {
        "super_climate": scored["super_climate"],
        "super_context_norm": scored["super_context_norm"],
        "narrow_het": scored["narrow_het"],
        "kernel_mean_shell": scored["kernel_mean_shell"],
        "gc": scored["gc"],
        "codon_count_entropy": scored["codon_count_entropy"],
    }

    summaries: dict[str, Any] = {}
    holdouts: dict[str, Any] = {}
    for name, values in reads.items():
        summary = within_group_spearman(
            values, expression, peptides, min_n=int(args.min_group)
        )
        # Stable per-read seed: Python's hash() is randomized across processes.
        name_digest = int.from_bytes(
            hashlib.sha256(name.encode("utf-8")).digest()[:4], "little"
        )
        summary["sign_flip_p_one_sided_greater"] = sign_flip_p(
            summary["group_rhos"],
            seed=int(args.seed) + (name_digest % 10000),
            permutations=int(args.permutations),
        )
        # Drop raw group list from the printed payload size; keep count stats.
        group_rhos = list(summary.pop("group_rhos"))
        summary["group_rhos_retained"] = False
        summaries[name] = summary
        holdouts[name] = peptide_holdout_mean(
            values,
            expression,
            peptides,
            min_n=int(args.min_group),
            folds=int(args.holdout_folds),
            seed=int(args.seed),
        )
        # Restore for primary-gate selection only in memory.
        summary["_group_rhos"] = group_rhos

    # Primary AE order read: Super climate; Narrow and kernel published beside it.
    primary = "super_climate"
    primary_summary = summaries[primary]
    primary_holdout = holdouts[primary]
    passed = bool(
        primary_summary["n_groups"] > 0
        and primary_summary["mean_rho"] > 0.0
        and primary_summary["sign_flip_p_one_sided_greater"] < 0.05
        and primary_holdout["mean_rho"] > 0.0
        and primary_holdout["positive_folds"] >= max(1, int(args.holdout_folds) // 2)
    )

    for summary in summaries.values():
        summary.pop("_group_rhos", None)

    checkpoint_paths = {
        name: path.resolve() for name, path in production_checkpoint_paths().items()
    }
    output_json = Path(args.output_json).resolve() if args.output_json else RESULT_JSON["order"]
    payload = {
        "status": "ok",
        "claim": (
            "Within fixed tripeptides from Gamble 2016 Table S1, frozen Super climate "
            "ranks synonymous expression by codon order."
            if passed
            else None
        ),
        "passed": passed,
        "primary_read": primary,
        "gates": {
            "primary_mean_rho_positive": bool(primary_summary["mean_rho"] > 0.0),
            "primary_sign_flip_p_below_0_05": bool(
                primary_summary["sign_flip_p_one_sided_greater"] < 0.05
            ),
            "primary_holdout_mean_rho_positive": bool(primary_holdout["mean_rho"] > 0.0),
            "primary_holdout_positive_folds_at_least_half": bool(
                primary_holdout["positive_folds"] >= max(1, int(args.holdout_folds) // 2)
            ),
        },
        "data": {
            "table_path": str(table_path),
            "table_sha256": sha256_file(table_path),
            "n_variants_raw": int(len(df)),
            "n_variants_eligible": int(len(variants)),
            "n_peptides_eligible": int(len(set(map(str, peptides)))),
            "min_group": int(args.min_group),
            "expression_column": "synGFPSEQ",
            "holdout_folds": int(args.holdout_folds),
        },
        "within_peptide": summaries,
        "peptide_holdout": holdouts,
        "controls_note": (
            "GC and codon-count entropy are composition controls published on the "
            "same within-peptide estimand as the AE and kernel reads."
        ),
        "provenance": {
            "generated_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "production_checkpoints": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in sorted(checkpoint_paths.items())
            },
            "cli": {
                "device": str(args.device),
                "seed": int(args.seed),
                "min_group": int(args.min_group),
                "holdout_folds": int(args.holdout_folds),
                "permutations": int(args.permutations),
            },
        },
    }

    lines = [
        "stage=order",
        (
            f"status=ok n_variants={payload['data']['n_variants_eligible']} "
            f"n_peptides={payload['data']['n_peptides_eligible']} min_group={args.min_group}"
        ),
        f"primary_read={primary}",
        (
            f"primary_mean_rho={primary_summary['mean_rho']:+.4f} "
            f"positive_frac={primary_summary['positive_fraction']:.3f} "
            f"p_greater={primary_summary['sign_flip_p_one_sided_greater']:.4f} "
            f"n_groups={primary_summary['n_groups']}"
        ),
        (
            f"primary_holdout_mean_rho={primary_holdout['mean_rho']:+.4f} "
            f"positive_folds={primary_holdout['positive_folds']}/{primary_holdout['n_folds']}"
        ),
    ]
    for name in (
        "super_climate",
        "super_context_norm",
        "narrow_het",
        "kernel_mean_shell",
        "gc",
        "codon_count_entropy",
    ):
        s = summaries[name]
        lines.append(
            f"{name}_mean_rho={s['mean_rho']:+.4f} holdout={holdouts[name]['mean_rho']:+.4f}"
        )
    lines.append(f"passed={passed}")

    write_json(output_json, payload)
    upsert_results("order", lines)
    print("\n".join(lines), flush=True)
    print(f"wrote {output_json} and RESULTS.txt", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
