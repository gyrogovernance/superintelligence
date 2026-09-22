#!/usr/bin/env python3
"""Held-out incremental test for frozen AE reads on membrane topology.

The primary AE comparison is kernel plus biological controls versus the same
features plus the frozen Narrow read. Super is scored beside Narrow as a
companion design. Windows from the same accession stay in the same
cross-validation fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[5]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common import RESULT_JSON, upsert_results, write_json

_DEFAULT_ATLAS = RESULT_JSON["atlas"]
_DEFAULT_JSON = RESULT_JSON["holdout"]
_DEFAULT_SEED_START = 20260920
_DEFAULT_SEED_COUNT = 10
_DEFAULT_SPLITS = 10
_DEFAULT_PERMUTATIONS = 20000

FEATURE_NAMES = (
    "kernel_shell",
    "gc",
    "window_nt",
    "narrow_het",
    "super_climate",
    "ae_walk_bytes",
    "ae_truncated",
)
DESIGNS = {
    "kernel": (0,),
    "kernel_controls": (0, 1, 2, 6),
    "kernel_controls_narrow": (0, 1, 2, 6, 3, 5),
    "kernel_controls_super": (0, 1, 2, 6, 4, 5),
    "kernel_controls_ae": (0, 1, 2, 6, 3, 4, 5),
}
PRIMARY_HOLDOUT = "kernel_controls_narrow"
METRIC_NAMES = ("auc", "average_precision", "balanced_accuracy", "log_loss")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def load_rows(atlas_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    genes = atlas.get("genes")
    if not isinstance(genes, list) or not genes:
        raise ValueError("atlas is missing genes")

    values: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    n_truncated = 0
    for gene in genes:
        accession = str(gene.get("accession", ""))
        if not accession:
            raise ValueError("atlas gene is missing accession")
        for label, key in ((1, "tm_windows"), (0, "cyto_windows")):
            windows = gene.get(key)
            if not isinstance(windows, list) or not windows:
                raise ValueError(f"atlas gene {accession} is missing {key}")
            for window in windows:
                truncated = bool(window.get("ae_truncated", False))
                n_truncated += int(truncated)
                values.append(
                    [
                        float(window["mean_shell"]),
                        float(gene["tm_gc"] if label == 1 else gene["cyto_gc"]),
                        float(window["nt"]),
                        float(window["ae_narrow_het"]),
                        float(window["ae_super_climate"]),
                        float(window["ae_walk_bytes"]),
                        1.0 if truncated else 0.0,
                    ]
                )
                labels.append(label)
                groups.append(accession)

    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    group_array = np.asarray(groups, dtype=str)
    if x.shape != (len(y), len(FEATURE_NAMES)):
        raise ValueError("feature matrix shape is inconsistent")
    if len(np.unique(group_array)) != len({gene.get("accession") for gene in genes}):
        raise ValueError("row groups do not match atlas genes")

    metadata = {
        "n_rows": int(len(y)),
        "n_genes": int(len(np.unique(group_array))),
        "class_counts": {
            "tm": int(np.sum(y == 1)),
            "cytoplasmic": int(np.sum(y == 0)),
        },
        "label_encoding": {"tm": 1, "cytoplasmic": 0},
        "feature_names": list(FEATURE_NAMES),
        "n_ae_truncated_windows": int(n_truncated),
        "ae_read_policy": "per-window AE reads; windows whose AE walk exceeds the frozen read cap are flagged and reported separately",
        "atlas_path": str(atlas_path.resolve()),
        "atlas_sha256": sha256_file(atlas_path),
        "cohort_manifest_sha256": str(atlas.get("cohort_manifest", {}).get("sha256", "")),
    }
    return x, y, group_array, metadata


def make_model() -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
        ),
    )


def score_design(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    indices: tuple[int, ...],
    seeds: tuple[int, ...],
    n_splits: int,
) -> dict[str, Any]:
    matrix = x[:, indices]
    seed_results: dict[str, dict[str, Any]] = {}
    all_folds: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}

    for seed in seeds:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        fold_metrics: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
        for train, test in splitter.split(matrix, y, groups):
            model = make_model()
            model.fit(matrix[train], y[train])
            probabilities = model.predict_proba(matrix[test])[:, 1]
            fold_metrics["auc"].append(float(roc_auc_score(y[test], probabilities)))
            fold_metrics["average_precision"].append(float(average_precision_score(y[test], probabilities)))
            fold_metrics["balanced_accuracy"].append(float(balanced_accuracy_score(y[test], probabilities >= 0.5)))
            fold_metrics["log_loss"].append(float(log_loss(y[test], probabilities, labels=[0, 1])))
            for name in METRIC_NAMES:
                all_folds[name].append(fold_metrics[name][-1])

        seed_results[str(seed)] = {
            "n_folds": n_splits,
            **{name: float(np.mean(values)) for name, values in fold_metrics.items()},
        }

    return {
        "feature_indices": list(indices),
        "feature_names": [FEATURE_NAMES[index] for index in indices],
        "seed_results": seed_results,
        "mean": {name: float(np.mean(values)) for name, values in all_folds.items()},
        "sd": {name: float(np.std(values)) for name, values in all_folds.items()},
    }


def truncation_split_auc(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seeds: tuple[int, ...],
    n_splits: int,
) -> dict[str, Any]:
    """Score the AE-increment design on each AE-read truncation class.

    Windows whose AE walk exceeds the frozen read cap are scored separately from
    windows that fit inside it, so the incremental read is reported per stratum.
    """
    truncated_index = FEATURE_NAMES.index("ae_truncated")
    out: dict[str, Any] = {}
    for label, mask in (
        ("ae_read_untruncated", x[:, truncated_index] < 0.5),
        ("ae_read_truncated", x[:, truncated_index] >= 0.5),
    ):
        count = int(mask.sum())
        if count < 10 or len(np.unique(y[mask])) < 2:
            out[label] = {
                "n_rows": count,
                "status": "insufficient_rows",
            }
            continue
        baseline = score_design(
            x[mask],
            y[mask],
            groups[mask],
            DESIGNS["kernel_controls"],
            seeds,
            min(n_splits, count),
        )
        candidate = score_design(
            x[mask],
            y[mask],
            groups[mask],
            DESIGNS[PRIMARY_HOLDOUT],
            seeds,
            min(n_splits, count),
        )
        out[label] = {
            "n_rows": count,
            "n_genes": int(len(np.unique(groups[mask]))),
            "class_counts": {
                "tm": int(np.sum(y[mask] == 1)),
                "cytoplasmic": int(np.sum(y[mask] == 0)),
            },
            "designs": {
                "kernel_controls": baseline,
                PRIMARY_HOLDOUT: candidate,
            },
            "incremental_auc": summarize_increment(
                baseline,
                candidate,
                design_name=PRIMARY_HOLDOUT,
                n_permutations=1,
                seed=0,
            ),
        }
    return out


def paired_sign_flip_p(
    values: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
    alternative: str,
) -> float:
    if alternative not in {"less", "greater", "two-sided"}:
        raise ValueError("invalid sign-flip alternative")
    observed = float(values.mean())
    rng = np.random.Generator(np.random.PCG64(seed))
    hits = 0
    for _ in range(n_permutations):
        signs = rng.integers(0, 2, size=values.size, dtype=np.int8) * 2 - 1
        permuted = float((values * signs).mean())
        if alternative == "greater":
            hit = permuted >= observed
        elif alternative == "less":
            hit = permuted <= observed
        else:
            hit = abs(permuted) >= abs(observed)
        hits += int(hit)
    return (hits + 1) / (n_permutations + 1)


def summarize_increment(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    design_name: str,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    baseline_auc = np.asarray(
        [float(result["auc"]) for result in baseline["seed_results"].values()],
        dtype=np.float64,
    )
    holdout_auc = np.asarray(
        [float(result["auc"]) for result in candidate["seed_results"].values()],
        dtype=np.float64,
    )
    deltas = holdout_auc - baseline_auc
    p_two_sided = paired_sign_flip_p(
        deltas,
        n_permutations=n_permutations,
        seed=seed,
        alternative="two-sided",
    )
    p_one_sided = paired_sign_flip_p(
        deltas,
        n_permutations=n_permutations,
        seed=seed + 1000003,
        alternative="greater",
    )
    gates = {
        "mean_incremental_auc_positive": bool(float(deltas.mean()) > 0.0),
        "two_sided_p_below_0_05": bool(p_two_sided < 0.05),
        "positive_seed_fraction_at_least_0_6": bool(float(np.mean(deltas > 0.0)) >= 0.6),
    }
    return {
        "baseline": "kernel_controls",
        "design": design_name,
        "unit": "seed-level mean AUC across grouped windows",
        "n_seeds": int(deltas.size),
        "mean": float(deltas.mean()),
        "sd": float(deltas.std()),
        "min": float(deltas.min()),
        "max": float(deltas.max()),
        "positive_seeds": int(np.sum(deltas > 0.0)),
        "positive_seed_fraction": float(np.mean(deltas > 0.0)),
        "paired_sign_flip_p_two_sided": float(p_two_sided),
        "paired_sign_flip_p_one_sided_greater": float(p_one_sided),
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether frozen AE reads improve the Kernel topology classifier."
    )
    parser.add_argument("--atlas", type=Path, default=_DEFAULT_ATLAS)
    parser.add_argument("--output-json", type=Path, default=_DEFAULT_JSON)
    parser.add_argument("--seed-start", type=int, default=_DEFAULT_SEED_START)
    parser.add_argument("--seed-count", type=int, default=_DEFAULT_SEED_COUNT)
    parser.add_argument("--n-splits", type=int, default=_DEFAULT_SPLITS)
    parser.add_argument("--permutations", type=int, default=_DEFAULT_PERMUTATIONS)
    args = parser.parse_args()
    if args.seed_count < 2:
        parser.error("--seed-count must be at least 2")
    if args.n_splits < 3:
        parser.error("--n-splits must be at least 3")
    if args.permutations < 1000:
        parser.error("--permutations must be at least 1000")
    return args


def main() -> int:
    args = parse_args()
    atlas_path = args.atlas.resolve()
    output_json = args.output_json.resolve()
    if not atlas_path.is_file():
        print(f"missing atlas: {atlas_path}", file=sys.stderr)
        return 1

    x, y, groups, metadata = load_rows(atlas_path)
    seeds = tuple(range(args.seed_start, args.seed_start + args.seed_count))
    scored = {
        name: score_design(x, y, groups, indices, seeds, args.n_splits)
        for name, indices in DESIGNS.items()
    }
    increment = summarize_increment(
        scored["kernel_controls"],
        scored[PRIMARY_HOLDOUT],
        design_name=PRIMARY_HOLDOUT,
        n_permutations=args.permutations,
        seed=args.seed_start + 2000003,
    )
    companion_super = summarize_increment(
        scored["kernel_controls"],
        scored["kernel_controls_super"],
        design_name="kernel_controls_super",
        n_permutations=args.permutations,
        seed=args.seed_start + 3000003,
    )
    companion_full = summarize_increment(
        scored["kernel_controls"],
        scored["kernel_controls_ae"],
        design_name="kernel_controls_ae",
        n_permutations=args.permutations,
        seed=args.seed_start + 4000003,
    )
    truncation_split = truncation_split_auc(
        x, y, groups, seeds, args.n_splits
    )
    gates = dict(increment["gates"])
    passed = bool(increment["passed"])

    payload = {
        "status": "ok",
        "claim": (
            "Frozen Narrow AE reads raise held-out topology-window AUC above "
            "kernel plus biological controls."
            if passed
            else None
        ),
        "passed": passed,
        "gates": gates,
        "primary_ae_design": PRIMARY_HOLDOUT,
        "study_unit": "topology window",
        "grouping": "accession; windows from one accession remain in one fold",
        "label_encoding": metadata["label_encoding"],
        "data": metadata,
        "designs": scored,
        "incremental_auc": increment,
        "companion_increments": {
            "kernel_controls_super": companion_super,
            "kernel_controls_ae": companion_full,
        },
        "truncation_split": truncation_split,
        "model": {
            "name": "standardized logistic regression",
            "c": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 5000,
        },
        "statistical_test": {
            "name": "paired sign-flip permutation",
            "paired_values": "seed-level holdout AUC minus seed-level baseline AUC",
            "n_permutations": args.permutations,
            "seeds": seeds,
            "primary_alternative": "two-sided",
            "one_sided_alternative": "greater",
        },
        "provenance": {
            "generated_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "atlas_path": str(atlas_path),
            "atlas_sha256": metadata["atlas_sha256"],
        },
    }

    lines = [
        "stage=holdout",
        f"status=ok rows={metadata['n_rows']} genes={metadata['n_genes']} tm={metadata['class_counts']['tm']} cytoplasmic={metadata['class_counts']['cytoplasmic']}",
        f"primary_ae_design={PRIMARY_HOLDOUT}",
        f"baseline_auc={scored['kernel_controls']['mean']['auc']:.6f} holdout_auc={scored[PRIMARY_HOLDOUT]['mean']['auc']:.6f}",
        f"incremental_auc_mean={increment['mean']:+.6f} sd={increment['sd']:.6f} positive_seeds={increment['positive_seeds']}/{increment['n_seeds']}",
        (
            f"incremental_auc_p_two_sided={increment['paired_sign_flip_p_two_sided']:.6f} "
            f"p_one_sided_greater={increment['paired_sign_flip_p_one_sided_greater']:.6f}"
        ),
        (
            f"companion_super_incremental_auc_mean={companion_super['mean']:+.6f} "
            f"companion_full_ae_incremental_auc_mean={companion_full['mean']:+.6f}"
        ),
        f"passed={passed}",
        f"incremental_gate_passed={increment['passed']}",
    ]
    for name in ("ae_read_untruncated", "ae_read_truncated"):
        entry = truncation_split[name]
        if entry.get("status") == "insufficient_rows":
            lines.append(f"{name}_rows={entry['n_rows']} status=insufficient_rows")
            continue
        sub = entry["incremental_auc"]
        lines.append(
            f"{name}_rows={entry['n_rows']} genes={entry['n_genes']} "
            f"incremental_auc_mean={sub['mean']:+.6f}"
        )
    write_json(output_json, payload)
    upsert_results("holdout", lines)
    print("\n".join(lines), flush=True)
    print(f"wrote {output_json} and RESULTS.txt", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
