"""Exhaustive convertor census over every Q1_0 matrix row in Bonsai.

This is the serious weight question: for every stored Q1 row (split into 4096-d
Ω charts when wider), measure tile Walsh energy climate, peak-chi climate,
adjacent XOR transport vs within-row order shuffle, and Q1 within-block vs
across-block transport. Null is the closed-form census shell law, not a tiny
Monte Carlo.

Does not load Super/K4. Does not sample three layers. Writes one report with
per-tensor rows and per-family layer curves.

Usage::

  python -m src.tools.autoencoder.programs.interpretability.census
  python -m src.tools.autoencoder.programs.interpretability.census --families emb,attn_q --batch 512
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

from src.tools.autoencoder.helpers.evals_run import json_safe
from src.tools.autoencoder.paths import ensure, reports_dir
from src.tools.autoencoder.programs.interpretability.adapter import (
    DEFAULT_GGUF,
    iter_tensor_rows,
    list_q1_matrix_tensors,
    tensor_meta,
)
from src.tools.autoencoder.programs.interpretability.converter import (
    POPCOUNT6,
    chirality_from_tiles,
    participation,
    permute_tiles,
    q1_pair_mask,
    rows_to_tiles,
    shells_from_chi,
    tile_mode_energy,
    transport_shells,
)

CENSUS_SHELL = np.array([comb(6, s) / 64.0 for s in range(7)], dtype=np.float64)
FAMILIES = (
    "emb",
    "out",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
)


def _tv(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(a - b).sum())


def _hist7_add(counts: np.ndarray, shells: np.ndarray) -> None:
    counts += np.bincount(shells.reshape(-1).astype(np.int64), minlength=7)


def _norm7(counts: np.ndarray) -> np.ndarray:
    s = float(counts.sum())
    return counts.astype(np.float64) / (s + 1e-30)


def family_of(name: str) -> str | None:
    if name == "token_embd.weight":
        return "emb"
    if name == "output.weight":
        return "out"
    m = re.match(r"blk\.(\d+)\.(.+)\.weight$", name)
    if not m:
        return None
    kind = m.group(2)
    if kind in FAMILIES:
        return kind
    return None


def layer_of(name: str) -> int | None:
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def select_tensors(gguf: Path, families: set[str]) -> list[str]:
    names = list_q1_matrix_tensors(gguf)
    out = []
    for n in names:
        fam = family_of(n)
        if fam is not None and fam in families:
            out.append(n)
    return out


def accumulate_batch(
    rows: np.ndarray,
    *,
    rng: np.random.Generator,
    energy_mass: np.ndarray,
    chi_counts: np.ndarray,
    tr_counts: np.ndarray,
    tr_shuf_counts: np.ndarray,
    tr_within: np.ndarray,
    tr_across: np.ndarray,
    pr_sum: list[float],
    pr_n: list[int],
) -> None:
    tiles = rows_to_tiles(rows)
    energy = tile_mode_energy(tiles)
    # Accumulate Plancherel mass before normalizing (exact pooled climate).
    e = energy.reshape(-1, 64)
    for s in range(7):
        energy_mass[s] += float(e[:, POPCOUNT6 == s].sum())
    pr = participation(energy)
    pr_sum[0] += float(pr.sum())
    pr_n[0] += int(pr.size)

    chi = chirality_from_tiles(tiles, rng=rng)
    _hist7_add(chi_counts, shells_from_chi(chi))
    tr = transport_shells(chi)
    _hist7_add(tr_counts, tr)

    # Within-row order shuffle: same chi multiset, broken adjacency.
    chi_shuf = permute_tiles(chi, rng)
    _hist7_add(tr_shuf_counts, transport_shells(chi_shuf))

    within_m, across_m = q1_pair_mask(chi.shape[1])
    _hist7_add(tr_within, tr[:, within_m])
    _hist7_add(tr_across, tr[:, across_m])


def census_tensor(
    gguf: Path,
    tensor_name: str,
    *,
    batch: int,
    seed: int,
    max_stored_rows: int | None = None,
) -> dict[str, Any]:
    meta = tensor_meta(gguf, tensor_name)
    rng = np.random.default_rng(seed)
    energy_mass = np.zeros(7, dtype=np.float64)
    chi_counts = np.zeros(7, dtype=np.float64)
    tr_counts = np.zeros(7, dtype=np.float64)
    tr_shuf_counts = np.zeros(7, dtype=np.float64)
    tr_within = np.zeros(7, dtype=np.float64)
    tr_across = np.zeros(7, dtype=np.float64)
    pr_sum = [0.0]
    pr_n = [0]
    n_charts = 0
    t0 = time.perf_counter()
    for rows in iter_tensor_rows(
        gguf, tensor_name, batch=batch, max_rows=max_stored_rows
    ):
        accumulate_batch(
            rows,
            rng=rng,
            energy_mass=energy_mass,
            chi_counts=chi_counts,
            tr_counts=tr_counts,
            tr_shuf_counts=tr_shuf_counts,
            tr_within=tr_within,
            tr_across=tr_across,
            pr_sum=pr_sum,
            pr_n=pr_n,
        )
        n_charts += int(rows.shape[0])
    elapsed = time.perf_counter() - t0
    e_hist = energy_mass / (energy_mass.sum() + 1e-30)
    chi_hist = _norm7(chi_counts)
    tr_hist = _norm7(tr_counts)
    tr_shuf = _norm7(tr_shuf_counts)
    within_h = _norm7(tr_within)
    across_h = _norm7(tr_across)
    return {
        "tensor": tensor_name,
        "family": family_of(tensor_name),
        "layer": layer_of(tensor_name),
        "stored_rows": meta["n_rows"],
        "width": meta["width"],
        "segments": meta["segments"],
        "n_charts": n_charts,
        "n_tiles": n_charts * 64,
        "seconds": elapsed,
        "energy_shell": e_hist.tolist(),
        "chi_shell": chi_hist.tolist(),
        "transport_shell": tr_hist.tolist(),
        "transport_order_shuffle": tr_shuf.tolist(),
        "transport_q1_within": within_h.tolist(),
        "transport_q1_across": across_h.tolist(),
        "tile_pr_mean": pr_sum[0] / max(pr_n[0], 1),
        "tv_energy_vs_census": _tv(e_hist, CENSUS_SHELL),
        "tv_chi_vs_census": _tv(chi_hist, CENSUS_SHELL),
        "tv_transport_vs_order_shuffle": _tv(tr_hist, tr_shuf),
        "tv_q1_within_vs_across": _tv(within_h, across_h),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("error"):
            continue
        by_family[str(r["family"])].append(r)

    family_summary = {}
    for fam, items in sorted(by_family.items()):
        tvs_e = [float(x["tv_energy_vs_census"]) for x in items]
        tvs_tr = [float(x["tv_transport_vs_order_shuffle"]) for x in items]
        tvs_q1 = [float(x["tv_q1_within_vs_across"]) for x in items]
        prs = [float(x["tile_pr_mean"]) for x in items]
        layer_curve = None
        if all(x["layer"] is not None for x in items):
            layer_curve = [
                {
                    "layer": int(x["layer"]),
                    "tv_energy_vs_census": x["tv_energy_vs_census"],
                    "tv_transport_vs_order_shuffle": x["tv_transport_vs_order_shuffle"],
                    "tv_q1_within_vs_across": x["tv_q1_within_vs_across"],
                    "tile_pr_mean": x["tile_pr_mean"],
                    "n_charts": x["n_charts"],
                }
                for x in sorted(items, key=lambda z: int(z["layer"]))
            ]
        family_summary[fam] = {
            "n_tensors": len(items),
            "n_charts_total": int(sum(x["n_charts"] for x in items)),
            "tv_energy_vs_census_mean": float(np.mean(tvs_e)),
            "tv_energy_vs_census_max": float(np.max(tvs_e)),
            "tv_transport_vs_order_shuffle_mean": float(np.mean(tvs_tr)),
            "tv_transport_vs_order_shuffle_max": float(np.max(tvs_tr)),
            "tv_q1_within_vs_across_mean": float(np.mean(tvs_q1)),
            "tv_q1_within_vs_across_max": float(np.max(tvs_q1)),
            "tile_pr_mean": float(np.mean(prs)),
            "layer_curve": layer_curve,
        }
    return family_summary


def run_census(
    *,
    gguf: Path,
    families: set[str],
    batch: int,
    seed: int,
    max_stored_rows: int | None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    ensure()
    tensors = select_tensors(gguf, families)
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, name in enumerate(tensors):
        print(f"[{i+1}/{len(tensors)}] {name}", flush=True)
        try:
            row = census_tensor(
                gguf,
                name,
                batch=batch,
                seed=seed + i,
                max_stored_rows=max_stored_rows,
            )
            print(
                f"  charts={row['n_charts']}  "
                f"tvE={row['tv_energy_vs_census']:.5f}  "
                f"tvTR={row['tv_transport_vs_order_shuffle']:.5f}  "
                f"tvQ1={row['tv_q1_within_vs_across']:.5f}  "
                f"pr={row['tile_pr_mean']:.2f}  "
                f"{row['seconds']:.1f}s",
                flush=True,
            )
            rows.append(row)
        except Exception as exc:
            print(f"  ERROR {exc}", flush=True)
            rows.append({"tensor": name, "family": family_of(name), "error": str(exc)})
    report = {
        "gguf": str(gguf).replace("\\", "/"),
        "seed": seed,
        "batch": batch,
        "max_stored_rows": max_stored_rows,
        "null": "closed-form census shell C(6,s)/64; order-shuffle preserves chi multiset",
        "families_requested": sorted(families),
        "n_tensors": len(tensors),
        "seconds_total": time.perf_counter() - t0,
        "census_shell": CENSUS_SHELL.tolist(),
        "family_summary": summarize(rows),
        "tensors": rows,
    }
    out = Path(out_path) if out_path is not None else reports_dir() / "weight_census.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    report["_written"] = str(out).replace("\\", "/")
    return report


def combine_census_shards(
    shard_paths: list[Path] | None = None,
    *,
    out_path: Path | None = None,
) -> dict:
    """Combine separately-run family shards into one census summary."""
    paths = shard_paths or [
        reports_dir() / "weight_census_emb_out.json",
        reports_dir() / "weight_census_attn.json",
        reports_dir() / "weight_census_ffn.json",
    ]
    rows = []
    secs = 0.0
    for p in paths:
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.extend(d.get("tensors") or [])
        secs += float(d.get("seconds_total") or 0.0)
    report = {
        "null": "closed-form census shell C(6,s)/64; order-shuffle preserves chi multiset",
        "n_tensors": len(rows),
        "n_charts_total": int(sum(r.get("n_charts", 0) for r in rows if "n_charts" in r)),
        "seconds_sum_of_shards": secs,
        "census_shell": CENSUS_SHELL.tolist(),
        "family_summary": summarize(rows),
        "sources": [str(p).replace("\\", "/") for p in paths if p.exists()],
    }
    out = out_path or reports_dir() / "weight_census.json"
    out.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    report["_written"] = str(out).replace("\\", "/")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    ap.add_argument(
        "--families",
        default=",".join(FAMILIES),
        help="Comma list from: " + ",".join(FAMILIES),
    )
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--max-stored-rows",
        type=int,
        default=None,
        help="Cap stored GGUF rows per tensor (debug only). Default: all rows.",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--combine",
        action="store_true",
        help="Combine existing family shard JSONs into one weight_census.json",
    )
    args = ap.parse_args(argv)
    if args.combine:
        report = combine_census_shards(out_path=args.out)
        print(f"wrote {report['_written']}")
        print(f"tensors={report['n_tensors']} charts={report['n_charts_total']}")
        return 0
    if not args.gguf.is_file():
        print(f"GGUF not found: {args.gguf}")
        return 2
    families = {f.strip() for f in args.families.split(",") if f.strip()}
    unknown = families - set(FAMILIES)
    if unknown:
        print(f"unknown families: {sorted(unknown)}")
        return 2
    report = run_census(
        gguf=args.gguf,
        families=families,
        batch=args.batch,
        seed=args.seed,
        max_stored_rows=args.max_stored_rows,
        out_path=args.out,
    )
    print(f"wrote {report['_written']}")
    print(f"total {report['seconds_total']:.1f}s over {report['n_tensors']} tensors")
    for fam, s in report["family_summary"].items():
        print(
            f"  {fam}: tensors={s['n_tensors']} charts={s['n_charts_total']} "
            f"tvE_mean={s['tv_energy_vs_census_mean']:.5f} "
            f"tvE_max={s['tv_energy_vs_census_max']:.5f} "
            f"tvTR_max={s['tv_transport_vs_order_shuffle_max']:.5f} "
            f"tvQ1_max={s['tv_q1_within_vs_across_max']:.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
