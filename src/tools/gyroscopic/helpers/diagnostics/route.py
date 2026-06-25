"""Q1_0 GGUF structure scan vs binomial reference (bench --diag)."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.tools.gyroscopic import ops

from . import gguf, kernel, tiles

PATH_NAMES = ("iso", "cs", "una", "ona", "bu")

PROJ_BUCKETS = (
    (0.0, 0.1),
    (0.1, 0.2),
    (0.2, 0.3),
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.01),
)


def _wht64(values: list[int]) -> list[int]:
    arr = list(values)
    stride = 32
    while stride >= 1:
        for i in range(0, 64, 2 * stride):
            for j in range(stride):
                a = arr[i + j]
                b = arr[i + j + stride]
                arr[i + j] = a + b
                arr[i + j + stride] = a - b
        stride >>= 1
    return arr


def _signs64_to_wht(signs64: int) -> list[int]:
    data = [1 if (signs64 >> i) & 1 else -1 for i in range(64)]
    return _wht64(data)


def chirality_projection_ratio(signs64: int, chi: int) -> float:
    """Energy fraction in the assigned 6-bit WHT mode."""
    w = _signs64_to_wht(signs64)
    total = sum(x * x for x in w)
    if total <= 0:
        return 0.0
    idx = chi & 0x3F
    coeff = w[idx]
    return float(coeff * coeff) / float(total)


def group_projection_energy_ratio(sign_bytes: bytes) -> float:
    """Mean chirality-mode energy fraction across both 64-bit halves of a group."""
    signs_a = struct.unpack_from("<Q", sign_bytes, 0)[0]
    signs_b = struct.unpack_from("<Q", sign_bytes, 8)[0]
    chi_a = ops.chirality_from_signs64(signs_a)
    chi_b = ops.chirality_from_signs64(signs_b)
    ra = chirality_projection_ratio(signs_a, chi_a)
    rb = chirality_projection_ratio(signs_b, chi_b)
    return 0.5 * (ra + rb)


def expected_shell_probs() -> dict[int, float]:
    return {k: math.comb(6, k) / 64.0 for k in range(7)}


def expected_k4_probs() -> dict[int, float]:
    return {j: 0.25 for j in range(4)}


def expected_route_probs() -> dict[str, float]:
    shell_p = expected_shell_probs()
    k4_p = expected_k4_probs()
    out = {name: 0.0 for name in PATH_NAMES}
    for shell in range(7):
        for k4 in range(4):
            path_idx = kernel.route_path(shell, k4)
            out[PATH_NAMES[path_idx]] += shell_p[shell] * k4_p[k4]
    return out


@dataclass
class CountSnapshot:
    groups: int = 0
    route: dict[str, int] = field(default_factory=lambda: {n: 0 for n in PATH_NAMES})
    shell: dict[int, int] = field(default_factory=lambda: dict.fromkeys(range(7), 0))
    k4: dict[int, int] = field(default_factory=lambda: dict.fromkeys(range(4), 0))


def _fractions(counts: dict[Any, int], total: int) -> dict[Any, float]:
    if total <= 0:
        return {k: 0.0 for k in counts}
    return {k: counts[k] / total for k in counts}


def _compare(
    observed: dict[Any, float],
    expected: dict[Any, float],
    *,
    label: str,
    total: int = 0,
) -> dict[str, Any]:
    keys = sorted(set(observed) | set(expected), key=lambda x: (isinstance(x, str), x))
    rows: list[dict[str, Any]] = []
    max_dev = 0.0
    chi2 = 0.0
    for key in keys:
        obs = observed.get(key, 0.0)
        exp = expected.get(key, 0.0)
        dev = obs - exp
        max_dev = max(max_dev, abs(dev))
        if total > 0 and exp > 0:
            obs_count = obs * total
            exp_count = exp * total
            chi2 += (obs_count - exp_count) ** 2 / exp_count
        rows.append({"key": key, "observed": round(obs, 6), "expected": round(exp, 6), "delta": round(dev, 6)})
    return {
        "label": label,
        "max_abs_deviation": round(max_dev, 6),
        "chi2": round(chi2, 4),
        "rows": rows,
    }


def distribution_report(counts: CountSnapshot) -> dict[str, Any]:
    total = counts.groups
    shell_obs = _fractions(counts.shell, total)
    k4_obs = _fractions(counts.k4, total)
    route_obs = _fractions(counts.route, total)
    shell_cmp = _compare(shell_obs, expected_shell_probs(), label="shell", total=total)
    k4_cmp = _compare(k4_obs, expected_k4_probs(), label="k4", total=total)
    route_cmp = _compare(route_obs, expected_route_probs(), label="route_path", total=total)
    faithful = (
        shell_cmp["max_abs_deviation"] <= 0.05
        and k4_cmp["max_abs_deviation"] <= 0.05
        and route_cmp["max_abs_deviation"] <= 0.05
    )
    return {
        "groups": total,
        "shell": {"counts": dict(counts.shell), "fractions": shell_obs, "compare": shell_cmp},
        "k4": {"counts": dict(counts.k4), "fractions": k4_obs, "compare": k4_cmp},
        "route_path": {"counts": dict(counts.route), "fractions": route_obs, "compare": route_cmp},
        "binomial_faithful": faithful,
    }


def _projection_histogram(ratios: list[float]) -> dict[str, Any]:
    buckets = {f"{lo:.1f}-{hi:.1f}": 0 for lo, hi in PROJ_BUCKETS}
    for r in ratios:
        for lo, hi in PROJ_BUCKETS:
            if lo <= r < hi:
                buckets[f"{lo:.1f}-{hi:.1f}"] += 1
                break
    if not ratios:
        return {"count": 0, "buckets": buckets, "percentiles": {}}
    sorted_r = sorted(ratios)

    def pct(p: float) -> float:
        idx = min(len(sorted_r) - 1, max(0, int(p * (len(sorted_r) - 1))))
        return sorted_r[idx]

    return {
        "count": len(ratios),
        "mean": round(sum(ratios) / len(ratios), 6),
        "percentiles": {
            "p10": round(pct(0.10), 6),
            "p50": round(pct(0.50), 6),
            "p90": round(pct(0.90), 6),
        },
        "buckets": buckets,
    }


def analyze_sign_bytes(sign_iter, *, max_groups: int | None = None) -> dict[str, Any]:
    """Full static scan of Q1 sign payloads via WHT analyze_q1_group."""
    counts = CountSnapshot()
    proj_ratios: list[float] = []
    for i, sign_bytes in enumerate(sign_iter):
        if max_groups is not None and i >= max_groups:
            break
        if i > 0 and i % 2000 == 0:
            print(f"  [route-diag] WHT scan: {i} groups...", flush=True)
        _, shell, k4 = kernel.analyze_q1_group(sign_bytes)
        path_idx = kernel.route_path(shell, k4)
        counts.groups += 1
        counts.shell[shell] += 1
        counts.k4[k4] += 1
        counts.route[PATH_NAMES[path_idx]] += 1
        proj_ratios.append(group_projection_energy_ratio(sign_bytes))
    report = distribution_report(counts)
    report["projection_energy_ratio"] = _projection_histogram(proj_ratios)
    return report


def run_weight_scan(gguf_path, *, max_groups: int | None = None) -> dict[str, Any]:
    path = Path(gguf_path)
    if not path.is_file():
        return {"error": f"GGUF not found: {path}", "groups": 0}
    try:
        sign_iter = gguf.iter_q1_sign_bytes(path, max_groups=max_groups)
        sign_report = analyze_sign_bytes(sign_iter, max_groups=max_groups)
        max_tiles = max_groups if max_groups is not None else 512
        tile_ratios: list[dict[str, float]] = []
        gyro_ratios: list[dict[str, float]] = []
        for _, tile in gguf.iter_q1_tiles(path, max_tiles=max_tiles):
            tile_ratios.append(tiles.decompose_tile(tile))
            gyro_ratios.append(tiles.decompose_tile_gyro_halves(tile))
        sign_report["tile_projection"] = tiles.summarize_ratios(tile_ratios)
        sign_report["tile_projection"]["random_reference"] = tiles.random_tile_reference(512)
        sign_report["tile_projection"]["tiles_sampled"] = len(tile_ratios)
        sign_report["tile_gyro_halves"] = tiles.summarize_gyro_halves(gyro_ratios)
        sign_report["tile_gyro_halves"]["tiles_sampled"] = len(gyro_ratios)
        return sign_report
    except Exception as exc:
        return {"error": str(exc), "groups": 0}


def run_route_diagnostic(*, gguf_path=None, max_groups: int | None = None) -> dict[str, Any]:
    """Static GGUF structure scan vs binomial reference (no runtime trace)."""
    out: dict[str, Any] = {
        "expected": {
            "shell": expected_shell_probs(),
            "k4": expected_k4_probs(),
            "route_path": expected_route_probs(),
        },
    }
    if gguf_path is not None:
        out["weights"] = run_weight_scan(gguf_path, max_groups=max_groups)
    primary = out.get("weights")
    if isinstance(primary, dict) and primary.get("groups", 0) > 0:
        out["binomial_faithful"] = primary.get("binomial_faithful")
        tp = primary.get("tile_projection")
        if isinstance(tp, dict) and tp.get("count", 0) > 0:
            ref = tp.get("random_reference", {})
            trained_chi = tp.get("r_chi", {}).get("mean", 0.0)
            random_chi = ref.get("r_chi", {}).get("mean", 0.0)
            out["tile_chi_vs_random"] = {
                "trained_mean": trained_chi,
                "random_mean": random_chi,
            }
    elif isinstance(primary, dict) and primary.get("error"):
        out["error"] = primary["error"]
    return out
