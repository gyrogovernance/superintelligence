#!/usr/bin/env python3
"""
hQVM(d) finite-size scaling and thermodynamic percolation (analysis_5).

Uses gyroscopic.hQVM.family. At d=6, dynamics match gyroscopic.hQVM.api
and hqvm_percolation_analysis_1.

Usage:
  python hqvm_percolation_analysis_5.py           # full run + UTF-8 results file
  python hqvm_percolation_analysis_5.py --fast    # deterministic sections only (no MC)
  python hqvm_percolation_analysis_5.py --no-tee  # stdout only, no results file
"""
from __future__ import annotations

import argparse
import random
import sys
from math import comb
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from gyroscopic.hQVM.family import (
    HqvmD,
    bfs_reach,
    bisect_p_c_rank_micro_ref,
    build_hqvm_d,
    closed_form_p_c_rank_micro_ref,
    delta_depth4_horizon_d,
    delta_dyadic_byte_d,
    delta_spinorial_residual_d,
    depth4_projection_bits,
    enumerate_bytes,
    exact_micro_ref_p_rank_full,
    exact_micro_ref_p_rank_full_cond,
    exact_micro_ref_theta_cond,
    fold_disagreement_d,
    gf2_rank,
    holonomy_micro_cov,
    max_fold_disagreement_d,
    mean_byte_curvature_rate,
    mean_carrier_entanglement_d,
    mean_fold_disagreement_d,
    partition_Z1_coeff_d,
    p_from_z_root_micro_ref,
    predicted_cluster_size,
    rank_excess_z,
    rank_excess_limit_c_root_extrapolated,
    verify_carrier_entanglement_exact,
    verify_d6_against_api,
    verify_exact_micro_ref_rank_distribution_brute,
    verify_exact_micro_ref_rank_distribution_pair_brute,
    verify_exact_micro_ref_rank_distribution_algebraic,
    verify_exact_root_rank_lock,
    verify_f_squared_rest_d,
    verify_micro_ref_full_iff_rank_d,
    z_root_micro_ref,
)

SEED = 20260702
D_VALUES = (1, 2, 3, 4, 5, 6, 7, 8)
D_MC_HEAVY = (1, 2, 3, 4, 5, 6, 7)
D_ASYM = (6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32)
Z_TARGETS = (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
Z_CHI_GRID = (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

# _group_sweep row columns (micro-ref / q-class MC)
COL_P = 0
COL_P_FULL_U = 2
COL_P_RANK_U = 3
COL_P_RANK_C = 6
COL_THETA_C = 8
COL_Z = 9

# _byte_sweep row columns (byte-fraction MC, legacy 7-tuple)
BYTE_COL_P_FULL_U = 2
BYTE_COL_P_RANK_C = 6
ENGINES: Dict[int, HqvmD] = {}


def _configure_stdout_utf8() -> None:
    import codecs

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")


def _engine(d: int) -> HqvmD:
    if d not in ENGINES:
        print(
            f"  building hQVM({d}): |Omega|={1 << (2 * d)}, |A|={1 << (d + 2)}...",
            flush=True,
        )
        ENGINES[d] = build_hqvm_d(d)
    return ENGINES[d]


def _n_mc(d: int) -> int:
    if d <= 4:
        return 1000
    if d == 5:
        return 600
    if d == 6:
        return 400
    if d == 7:
        return 200
    return 100


def _p_grid(d: int) -> List[float]:
    if d <= 2:
        return [i / 20.0 for i in range(21)]
    if d <= 4:
        vals = [i / 100.0 for i in range(0, 51, 2)]
        vals += [i / 20.0 for i in range(3, 11)]
        return sorted(set(vals))
    vals = [i / 1000.0 for i in range(0, 101, 5)]
    vals += [i / 200.0 for i in range(11, 41, 2)]
    vals += [i / 20.0 for i in range(3, 11)]
    return sorted(set(vals))


def _find_crossing(rows: Sequence[Tuple[float, ...]], col: int) -> float | None:
    for i in range(len(rows) - 1):
        y1, y2 = rows[i][col], rows[i + 1][col]
        if y1 < 0.5 <= y2 and y2 > y1:
            frac = (0.5 - y1) / (y2 - y1)
            return rows[i][0] + frac * (rows[i + 1][0] - rows[i][0])
    return None


def _bisect_p_c_rank(d: int) -> float:
    return bisect_p_c_rank_micro_ref(d)


def _p_for_z(d: int, z: float) -> float:
    return p_from_z_root_micro_ref(z, d)


def _interp_at_p(
    rows: Sequence[Tuple[float, ...]],
    p_target: float,
    col: int,
) -> float | None:
    if not rows:
        return None
    if p_target <= rows[0][0]:
        return rows[0][col]
    if p_target >= rows[-1][0]:
        return rows[-1][col]
    for i in range(len(rows) - 1):
        p0, p1 = rows[i][0], rows[i + 1][0]
        if p0 <= p_target <= p1:
            if p1 == p0:
                return rows[i][col]
            t = (p_target - p0) / (p1 - p0)
            return rows[i][col] + t * (rows[i + 1][col] - rows[i][col])
    return None


def _interp_at_z(
    rows: Sequence[Tuple[float, ...]],
    z_target: float,
    col: int,
    z_col: int = COL_Z,
) -> float | None:
    if not rows:
        return None
    by_z = sorted(rows, key=lambda r: r[z_col])
    if z_target <= by_z[0][z_col]:
        return by_z[0][col]
    if z_target >= by_z[-1][z_col]:
        return by_z[-1][col]
    for i in range(len(by_z) - 1):
        z0, z1 = by_z[i][z_col], by_z[i + 1][z_col]
        if z0 <= z_target <= z1:
            if z1 == z0:
                return by_z[i][col]
            t = (z_target - z0) / (z1 - z0)
            return by_z[i][col] + t * (by_z[i + 1][col] - by_z[i][col])
    return None


def _micro_threshold_rows(
    micro_rows: Dict[int, list],
) -> List[Tuple[int, float, float, float, float, float]]:
    """Per d: d, p_c full, z_c full, p_c rank MC, gap at p_c, chi_z_max."""
    out: List[Tuple[int, float, float, float, float, float]] = []
    for d in sorted(micro_rows):
        rows = micro_rows[d]
        p_full = _find_crossing(rows, COL_P_FULL_U)
        if p_full is None:
            continue
        p_rank = _find_crossing(rows, COL_P_RANK_U)
        best = min(rows, key=lambda r: abs(r[0] - p_full))
        gap = best[COL_P_FULL_U] - best[COL_P_RANK_U]
        chi_max = 0.0
        for i in range(len(Z_CHI_GRID) - 1):
            z0, z1 = Z_CHI_GRID[i], Z_CHI_GRID[i + 1]
            th0 = _interp_at_z(rows, z0, COL_THETA_C)
            th1 = _interp_at_z(rows, z1, COL_THETA_C)
            if th0 is None or th1 is None:
                continue
            dz = z1 - z0
            if dz > 0:
                chi_max = max(chi_max, (th1 - th0) / dz)
        out.append(
            (
                d,
                p_full,
                z_root_micro_ref(p_full, d),
                p_rank or float("nan"),
                gap,
                chi_max,
            )
        )
    return out


def _alphabet_q_weight_exact(eng: HqvmD, w: int) -> List[int]:
    return [b for b in range(eng.n_bytes) if eng.q_weight[b] == w]


def _alphabet_q_weight_at_most(eng: HqvmD, w: int) -> List[int]:
    return [b for b in range(eng.n_bytes) if eng.q_weight[b] <= w]


# ---------------------------------------------------------------------------
# §0
# ---------------------------------------------------------------------------


def section_0_api_check() -> None:
    print("\n" + "=" * 5)
    print("0. d=6 API CROSS-CHECK")
    print("=" * 5)
    ok, msg = verify_d6_against_api()
    print(f"  status: {'PASS' if ok else 'FAIL'}")
    print(f"  detail: {msg}")


def section_0c_bridge_gate() -> None:
    print("\n" + "=" * 5)
    print("0c. BRIDGE GATE: micro-ref E_full <=> rank=d  [exhaustive d<=4]")
    print("=" * 5)
    for d in range(1, 5):
        ok, passed, total = verify_micro_ref_full_iff_rank_d(d)
        print(f"  d={d}: {passed}/{total}  {'PASS' if ok else 'FAIL'}")


def section_0d_rank_distribution_check() -> None:
    print("\n" + "=" * 5)
    print("0d. EXACT RANK DISTRIBUTION")
    print("=" * 5)
    print("  micro-ref brute d=1..4")
    for d in range(1, 5):
        ok, _, _ = verify_exact_micro_ref_rank_distribution_brute(d)
        print(f"    d={d}: {'PASS' if ok else 'FAIL'}")
    print("  pair-quotient brute d=5")
    ok5, _, _ = verify_exact_micro_ref_rank_distribution_pair_brute(5)
    print(f"    d=5: {'PASS' if ok5 else 'FAIL'}")
    print("  analytic (Mobius) d=6,8")
    for d in (6, 8):
        ok, detail = verify_exact_micro_ref_rank_distribution_algebraic(d)
        print(f"    d={d}: {'PASS' if ok else 'FAIL'}  ({detail})")


def section_0e_root_rank_lock() -> None:
    print("\n" + "=" * 5)
    print("0e. ROOT RANK LOCK  [P_root(n) vs P(rank=d)]")
    print("=" * 5)
    for d in (4, 6, 8, 12, 16):
        ok, root_full, exact_full = verify_exact_root_rank_lock(d=d)
        print(
            f"  d={d}: root={root_full:.10f} exact={exact_full:.10f} "
            f"{'PASS' if ok else 'FAIL'}"
        )


# ---------------------------------------------------------------------------
# §1
# ---------------------------------------------------------------------------


def section_1_square_root(d_values: Sequence[int]) -> Tuple[int, int]:
    print("\n" + "=" * 5)
    print("1. SQUARE-ROOT IDENTITY BY d")
    print("=" * 5)
    print("  Fiber-complete weight shells; |Reach| vs (2^r)^2.\n")
    print(f"  {'d':<4} {'w':<4} {'rank':<6} {'|Reach|':<10} {'pred':<10} {'PASS':<6}")
    print("  " + "-" * 5)
    fails = 0
    tests = 0
    for d in d_values:
        eng = _engine(d)
        cases = [(0, _alphabet_q_weight_exact(eng, 0))]
        for w in range(1, d + 1):
            cases.append((w, _alphabet_q_weight_at_most(eng, w)))
        cases.append((d + 1, list(enumerate_bytes(d))))
        for w, alphabet in cases:
            if not alphabet:
                continue
            tests += 1
            qs = [eng.q_by_byte[b] for b in alphabet]
            r = gf2_rank(qs, d)
            reach, _, _, _ = bfs_reach(eng, alphabet)
            pred = predicted_cluster_size(r)
            ok = reach == pred
            if not ok:
                fails += 1
            label = "full" if w > d else str(w)
            print(
                f"  {d:<4} {label:<4} {r:<6} {reach:<10} {pred:<10} {'PASS' if ok else 'FAIL':<6}"
            )
    print(f"\n  aggregate: {tests - fails}/{tests} PASS")
    return tests - fails, tests


# ---------------------------------------------------------------------------
# §2
# ---------------------------------------------------------------------------


def section_2_rank_analytics(d_values: Sequence[int]) -> None:
    print("\n" + "=" * 5)
    print("2. MICRO-REF RANK THRESHOLD (exact)")
    print("=" * 5)
    c_star = rank_excess_limit_c_root_extrapolated()
    print(f"  micro-ref root variable: z_root = 2^(d-1)*(1-(1-p)^2) - (d-1)")
    print(f"  c*_inf (1/d extrapolation, d=28,32) = {c_star:.6f}")
    print(f"  asymptotic: p_c(rank) ~ 1 - sqrt(1 - ( (d-1)+c*_root )/2^(d-1) )")

    print("\n  2a. P(rank=d) vs z_root (exact)")
    print(f"\n  {'z_root':<8}", end="")
    for d in d_values:
        print(f"  d={d:<2}", end="")
    print(f"  {'spread':<8}")
    print("  " + "-" * 5)
    for zt in Z_TARGETS:
        print(f"  {zt:<8.2f}", end="")
        vals: List[float] = []
        for d in d_values:
            p = _p_for_z(d, zt)
            if p <= 0.0 or p > 1.0:
                print(f"  {'--':<6}", end="")
                continue
            v = exact_micro_ref_p_rank_full(p, d)
            vals.append(v)
            print(f"  {v:<6.3f}", end="")
        spread = max(vals) - min(vals) if vals else float("nan")
        print(f"  {spread:<8.3f}" if vals else f"  {'--':<8}")

    print("\n  2b. p_c(rank), z_root,c, closed form")
    print(
        f"\n  {'d':<4} {'2^d':<8} {'p_c':<12} {'z_root,c':<10} {'p_form':<12} {'|err|':<10}"
    )
    print("  " + "-" * 5)
    for d in d_values:
        p_c = _bisect_p_c_rank(d)
        z_c = z_root_micro_ref(p_c, d)
        p_form = closed_form_p_c_rank_micro_ref(d, c_star)
        print(
            f"  {d:<4} {1 << d:<8} {p_c:<12.6f} {z_c:<10.4f} "
            f"{p_form:<12.6f} {abs(p_c - p_form):<10.6f}"
        )


def section_2c_asymptotic_convergence() -> None:
    print("\n" + "=" * 5)
    print("2c. ASYMPTOTIC CONVERGENCE (analytic-only)")
    print("=" * 5)
    print("  z_root,c(d) where exact P(rank=d)=1/2, reported in correct root variable")

    z_by_d: Dict[int, float] = {}
    print(f"\n  {'d':<4} {'p_c':<12} {'z_root,c':<12}")
    print("  " + "-" * 5)
    for d in D_ASYM:
        p_c = _bisect_p_c_rank(d)
        z_by_d[d] = z_root_micro_ref(p_c, d)
        print(f"  {d:<4} {p_c:<12.4e} {z_by_d[d]:<12.6f}")

    # First-order finite-size law z_root,c(d) = c_inf - a/d, fit on two largest d
    d1, d2 = D_ASYM[-2], D_ASYM[-1]
    z1, z2 = z_by_d[d1], z_by_d[d2]
    a_fit = (z2 - z1) / (1.0 / d1 - 1.0 / d2)
    c_inf = z2 + a_fit / d2
    print(f"\n  fit z_root,c(d) = c_inf - a/d on d={d1},{d2}:")
    print(f"  c_inf = {c_inf:.6f}   a = {a_fit:.6f}")
    print(f"\n  {'d':<4} {'z_root,c':<12} {'fit':<12} {'resid':<12}")
    print("  " + "-" * 5)
    for d in D_ASYM:
        fit = c_inf - a_fit / d
        print(f"  {d:<4} {z_by_d[d]:<12.6f} {fit:<12.6f} {z_by_d[d] - fit:<+12.6f}")


# ---------------------------------------------------------------------------
# §3
# ---------------------------------------------------------------------------


def section_3_thermodynamic_shell(d_values: Sequence[int]) -> None:
    print("\n" + "=" * 5)
    print("3. SHELL CENSUS AND Z1(lam)")
    print("=" * 5)
    print(f"\n  {'d':<4} {'|H|':<8} {'|Omega|':<10} {'Z1(1)':<12} {'horizon frac':<12}")
    print("  " + "-" * 5)
    for d in d_values:
        horizon = 1 << d
        omega = 1 << (2 * d)
        z1 = partition_Z1_coeff_d(d, 1.0)
        h_frac = 2.0 / horizon if horizon else 0.0
        print(f"  {d:<4} {horizon:<8} {omega:<10} {z1:<12.1f} {h_frac:<12.6f}")
    print("\n  equatorial shell fraction C(d,d/2)/2^d for even d:")
    for d in d_values:
        if d % 2 == 0:
            frac = comb(d, d // 2) / (1 << d)
            print(f"    d={d}: {frac:.6f}")


# ---------------------------------------------------------------------------
# §4
# ---------------------------------------------------------------------------


def section_4_fold_curvature(d_values: Sequence[int]) -> None:
    print("\n" + "=" * 5)
    print("4. FOLD DISAGREEMENT VS d")
    print("=" * 5)
    print(
        f"\n  {'d':<4} {'pairs':<6} {'|A|':<8} {'flat':<8} {'curved':<8} {'curve rate':<12}"
    )
    print("  " + "-" * 5)
    for d in d_values:
        eng = _engine(d)
        flat = sum(1 for fd in eng.fold_disagree if fd == 0)
        curved = eng.n_bytes - flat
        rate = curved / eng.n_bytes
        npairs = max_fold_disagreement_d(d)
        print(
            f"  {d:<4} {npairs:<6} {eng.n_bytes:<8} {flat:<8} {curved:<8} {rate:<12.6f}"
        )


# ---------------------------------------------------------------------------
# §5–7
# ---------------------------------------------------------------------------


def _group_sweep(
    eng: HqvmD,
    groups: Sequence[Sequence[int]],
    p_values: Sequence[float],
    n_samples: int,
    label: str,
    z_fn: Callable[[float, int], float],
) -> List[Tuple[float, float, float, float, float, float, float, float, float, float]]:
    """Row: p, P(span/full/rank)_uncond, P(span/full/rank)_cond, E[r]/d|cond, theta|cond, z."""
    n_omega = eng.n_omega
    d = eng.d
    rows: List[
        Tuple[float, float, float, float, float, float, float, float, float, float]
    ] = []

    for pi, p in enumerate(p_values):
        if pi % 10 == 0:
            print(
                f"    progress {label} d={d} {pi + 1}/{len(p_values)} p={p:.4f}",
                flush=True,
            )

        span_u = full_u = rank_u = 0
        span_c = full_c = rank_c = 0
        rank_sum = theta_sum = 0.0
        nz = 0

        for _ in range(n_samples):
            allowed: List[int] = []
            q_seen: List[int] = []
            for grp in groups:
                if random.random() < p:
                    allowed.extend(grp)
                    q_seen.extend(eng.q_by_byte[b] for b in grp)

            if not allowed:
                continue

            nz += 1
            reach, spans, _, full = bfs_reach(eng, allowed)
            r = gf2_rank(q_seen, d)
            rank_full = r == d

            if spans:
                span_u += 1
            if full:
                full_u += 1
            if rank_full:
                rank_u += 1

            if spans:
                span_c += 1
            if full:
                full_c += 1
            if rank_full:
                rank_c += 1

            rank_sum += r / d
            theta_sum += reach / n_omega

        z = z_fn(p, d)
        if nz > 0:
            rows.append(
                (
                    p,
                    span_u / n_samples,
                    full_u / n_samples,
                    rank_u / n_samples,
                    span_c / nz,
                    full_c / nz,
                    rank_c / nz,
                    rank_sum / nz,
                    theta_sum / nz,
                    z,
                )
            )
        else:
            rows.append((p, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, z))

    return rows


def section_5_micro_ref_mc(d_values: Sequence[int]) -> Dict[int, list]:
    print("\n" + "=" * 5)
    print("5. MICRO-REF MC")
    print("=" * 5)
    out: Dict[int, list] = {}
    for d in d_values:
        eng = _engine(d)
        n = _n_mc(d)
        p_vals = _p_grid(d)
        print(f"\n  --- d={d}, n={n} ---")
        rows = _group_sweep(
            eng, eng.micro_ref_groups, p_vals, n, "micro_ref", z_root_micro_ref
        )
        out[d] = rows
        p_c = _find_crossing(rows, COL_P_FULL_U)
        p_c_rank = _find_crossing(rows, COL_P_RANK_U)
        print(
            f"\n  {'p':<9} {'z_root':<8} {'P(full)':<10} {'P(rank=d)':<10} {'theta':<10}"
        )
        print("  " + "-" * 5)
        keep = {0, len(rows) - 1}
        if p_c is not None:
            for i, row in enumerate(rows):
                if abs(row[0] - p_c) <= max(0.015, row[0] * 0.12):
                    keep.add(i)
        for i in sorted(keep):
            p, _, pfull, prank, _, _, _, _, theta, z = rows[i]
            print(f"  {p:<9.4f} {z:<8.4f} {pfull:<10.4f} {prank:<10.4f} {theta:<10.4f}")
        if p_c is not None:
            print(f"  p_c(full) ~ {p_c:.5f}  z_root,c ~ {z_root_micro_ref(p_c, d):.4f}")
        if p_c_rank is not None:
            print(f"  p_c(rank=d) ~ {p_c_rank:.5f}")
        p_ref = p_c if p_c is not None else 0.1
        print(
            f"  exact P(rank=d) at p={p_ref:.4f}: "
            f"{exact_micro_ref_p_rank_full(p_ref, d):.4f}"
        )
    return out


def section_6_qclass_mc(d_values: Sequence[int]) -> None:
    print("\n" + "=" * 5)
    print("6. Q-CLASS MC")
    print("=" * 5)
    for d in d_values:
        eng = _engine(d)
        n = _n_mc(d)
        p_vals = _p_grid(d)
        print(f"\n  --- d={d}, n={n} ---")
        rows = _group_sweep(
            eng, eng.q_class_groups, p_vals, n, "q_class", rank_excess_z
        )
        p_c = _find_crossing(rows, COL_P_FULL_U)
        if p_c is not None:
            print(f"  p_c(full) ~ {p_c:.5f}  z_c ~ {rank_excess_z(p_c, d):.4f}")


def _byte_sweep(
    eng: HqvmD,
    p_values: Sequence[float],
    n_samples: int,
) -> List[Tuple[float, float, float, float, float, float, float]]:
    """Byte-fraction protocol. Row: p, P(span), P(full), E[r]/d, theta, z_byte, P(rank=d)."""
    n_omega = eng.n_omega
    d = eng.d
    n_bytes = eng.n_bytes
    rows: List[Tuple[float, float, float, float, float, float, float]] = []
    for pi, p in enumerate(p_values):
        if pi % 10 == 0:
            print(
                f"    progress byte d={d} {pi + 1}/{len(p_values)} p={p:.4f}",
                flush=True,
            )
        span_c = full_c = rank_full_c = 0
        rank_sum = theta_sum = 0.0
        nz = 0
        for _ in range(n_samples):
            allowed = [b for b in range(n_bytes) if random.random() < p]
            if not allowed:
                continue
            nz += 1
            q_seen = [eng.q_by_byte[b] for b in allowed]
            reach, spans, _, full = bfs_reach(eng, allowed)
            r = gf2_rank(q_seen, d)
            rank_sum += r / d
            theta_sum += reach / n_omega
            if spans:
                span_c += 1
            if full:
                full_c += 1
            if r == d:
                rank_full_c += 1
        z_byte = p * n_bytes - d
        rows.append(
            (
                p,
                span_c / n_samples,
                full_c / n_samples,
                rank_sum / max(nz, 1),
                theta_sum / max(nz, 1),
                z_byte,
                rank_full_c / max(nz, 1),
            )
        )
    return rows


def section_7_byte_mc(d_values: Sequence[int]) -> Dict[int, list]:
    print("\n" + "=" * 5)
    print("7. BYTE-FRACTION MC")
    print("=" * 5)
    out: Dict[int, list] = {}
    for d in d_values:
        if d not in D_MC_HEAVY and d == 8:
            continue
        eng = _engine(d)
        n = _n_mc(d)
        p_vals = _p_grid(d)
        print(f"\n  --- d={d}, n={n} ---")
        rows = _byte_sweep(eng, p_vals, n)
        out[d] = rows
        p_c = _find_crossing(rows, BYTE_COL_P_FULL_U)
        p_rank = _find_crossing(rows, BYTE_COL_P_RANK_C)
        if p_c is not None:
            p_c_val = p_c
            best = min(rows, key=lambda r: abs(r[0] - p_c_val))
            gap = best[BYTE_COL_P_FULL_U] - best[BYTE_COL_P_RANK_C]
            print(f"  p_c(full) ~ {p_c:.5f}  gap P(full)-P(rank=d) ~ {gap:.4f}")
        if p_rank is not None:
            print(f"  p_c(rank=d) ~ {p_rank:.5f}")
    return out


# ---------------------------------------------------------------------------
# §8
# ---------------------------------------------------------------------------


def section_8_scaling_collapse(micro_rows: Dict[int, list]) -> None:
    print("\n" + "=" * 5)
    print("8. SCALING vs z_root = 2^(d-1)*(1-(1-p)^2) - (d-1)")
    print("=" * 5)
    if not micro_rows:
        print("  SKIP: no MC rows (--fast mode)")
        return

    d_list = sorted(micro_rows)

    print("\n  8a. P(full) MC at fixed z_root")
    print(f"\n  {'z_root':<8}", end="")
    for d in d_list:
        print(f"  d={d:<2}", end="")
    print(f"  {'spread':<8}")
    print("  " + "-" * 5)
    for zt in Z_TARGETS:
        print(f"  {zt:<8.2f}", end="")
        mc_vals: List[float] = []
        for d in d_list:
            p = _p_for_z(d, zt)
            if p <= 0.0 or p > 1.0:
                print(f"  {'--':<6}", end="")
                continue
            val = _interp_at_p(micro_rows[d], p, COL_P_FULL_U)
            if val is None:
                print(f"  {'na':<6}", end="")
            else:
                mc_vals.append(val)
                print(f"  {val:<6.3f}", end="")
        spread = max(mc_vals) - min(mc_vals) if mc_vals else float("nan")
        print(f"  {spread:<8.3f}" if mc_vals else f"  {'--':<8}")

    print("\n  8b. P(full) - P(rank=d) MC (micro-ref, unconditional)")
    print(f"\n  {'z_root':<8}", end="")
    for d in d_list:
        print(f"  d={d:<2}", end="")
    print()
    print("  " + "-" * 5)
    for zt in Z_TARGETS:
        print(f"  {zt:<8.2f}", end="")
        for d in d_list:
            p = _p_for_z(d, zt)
            if p <= 0.0 or p > 1.0:
                print(f"  {'--':<6}", end="")
                continue
            pfull = _interp_at_p(micro_rows[d], p, COL_P_FULL_U)
            prank_u = _interp_at_p(micro_rows[d], p, COL_P_RANK_U)
            if pfull is None or prank_u is None:
                print(f"  {'na':<6}", end="")
            else:
                print(f"  {pfull - prank_u:<6.3f}", end="")
        print()

    print("\n  8c. MC P(rank=d) vs exact (conditional)")
    print(f"\n  {'z_root':<8}", end="")
    for d in d_list:
        print(f"  d={d:<2}", end="")
    print()
    print("  " + "-" * 5)
    for zt in Z_TARGETS:
        print(f"  {zt:<8.2f} MC", end="")
        for d in d_list:
            p = _p_for_z(d, zt)
            if p <= 0.0 or p > 1.0:
                print(f"  {'--':<6}", end="")
                continue
            mc_val = _interp_at_p(micro_rows[d], p, COL_P_RANK_C)
            if mc_val is not None:
                print(f"  {mc_val:<6.3f}", end="")
            else:
                print(f"  {'na':<6}", end="")
        print()
        print(f"  {'':8} EX", end="")
        for d in d_list:
            p = _p_for_z(d, zt)
            if p <= 0.0 or p > 1.0:
                print(f"  {'--':<6}", end="")
                continue
            print(f"  {exact_micro_ref_p_rank_full_cond(p, d):<6.3f}", end="")
        print()


# ---------------------------------------------------------------------------
# §9
# ---------------------------------------------------------------------------


def section_9_rank_full_gap(
    micro_rows: Dict[int, list],
    byte_rows: Dict[int, list],
) -> None:
    print("\n" + "=" * 5)
    print("9. RANK=d vs P(full) GAP AT p_c")
    print("=" * 5)
    if not micro_rows:
        print("  SKIP micro-ref: no MC rows (--fast mode)")
    else:
        print("\n  9a. Micro-ref at p_c(full)")
        print(
            f"\n  {'d':<4} {'p_c':<10} {'z_c':<10} {'P(rank)':<10} {'P(full)':<10} {'gap':<10}"
        )
        print("  " + "-" * 5)
        for d, p_full, z_c, _, gap, _ in _micro_threshold_rows(micro_rows):
            rows = micro_rows[d]
            best = min(rows, key=lambda r: abs(r[0] - p_full))
            print(
                f"  {d:<4} {p_full:<10.5f} {z_c:<10.4f} {best[COL_P_RANK_U]:<10.4f} "
                f"{best[COL_P_FULL_U]:<10.4f} {gap:<10.4f}"
            )

    if not byte_rows:
        print("\n  9b. Byte-fraction: SKIP (--fast mode)")
        return

    print("\n  9b. Byte-fraction at p_c(full)")
    print(f"\n  {'d':<4} {'p_c':<10} {'P(rank)':<10} {'P(full)':<10} {'gap':<10}")
    print("  " + "-" * 5)
    for d in sorted(byte_rows):
        rows = byte_rows[d]
        p_c = _find_crossing(rows, BYTE_COL_P_FULL_U)
        if p_c is None:
            continue
        p_c_val = p_c
        best = min(rows, key=lambda r: abs(r[0] - p_c_val))
        gap = best[BYTE_COL_P_FULL_U] - best[BYTE_COL_P_RANK_C]
        print(
            f"  {d:<4} {p_c:<10.5f} {best[BYTE_COL_P_RANK_C]:<10.4f} "
            f"{best[BYTE_COL_P_FULL_U]:<10.4f} {gap:<10.4f}"
        )


# ---------------------------------------------------------------------------
# §10
# ---------------------------------------------------------------------------


def section_10_parity(d_values: Sequence[int]) -> None:
    print("\n" + "=" * 5)
    print("10. PARITY OBSTRUCTION")
    print("=" * 5)
    for d in d_values:
        eng = _engine(d)
        even_b = [b for b in enumerate_bytes(d) if eng.q_weight[b] % 2 == 0]
        odd_b = [b for b in enumerate_bytes(d) if eng.q_weight[b] % 2 == 1]
        r_e = gf2_rank([eng.q_by_byte[b] for b in even_b], d)
        r_o = gf2_rank([eng.q_by_byte[b] for b in odd_b], d)
        reach_e, _, _, full_e = bfs_reach(eng, even_b)
        reach_o, _, _, full_o = bfs_reach(eng, odd_b)
        print(
            f"  d={d}: even rank={r_e} Reach={reach_e} full={full_e} | "
            f"odd rank={r_o} Reach={reach_o} full={full_o}"
        )


# ---------------------------------------------------------------------------
# §11
# ---------------------------------------------------------------------------


def section_11_holonomy(d_values: Sequence[int]) -> None:
    print("\n" + "=" * 5)
    print("11. HOLONOMY 1-(1-p^4)^(2^d)")
    print("=" * 5)
    print(f"\n  {'p':<8}", end="")
    for d in d_values:
        print(f"  d={d:<2}", end="")
    print()
    print("  " + "-" * 5)
    for p in (0.05, 0.1, 0.2, 0.3, 0.5):
        print(f"  {p:<8.2f}", end="")
        for d in d_values:
            print(f"  {holonomy_micro_cov(p, d):<6.4f}", end="")
        print()


# ---------------------------------------------------------------------------
# §12
# ---------------------------------------------------------------------------


def section_12_susceptibility(micro_rows: Dict[int, list]) -> None:
    print("\n" + "=" * 5)
    print("12. SUSCEPTIBILITY d(theta)/dz  [MC witness]")
    print("=" * 5)
    print(f"  common z grid: {list(Z_CHI_GRID)}")
    if not micro_rows:
        print("  SKIP: no MC rows (--fast mode)")
        return

    print(f"\n  {'d':<4} {'chi_z max':<12} {'z at chi max':<12}")
    print("  " + "-" * 5)
    for d in sorted(micro_rows):
        rows = micro_rows[d]
        chi_max = 0.0
        z_at = Z_CHI_GRID[0]
        for i in range(len(Z_CHI_GRID) - 1):
            z0, z1 = Z_CHI_GRID[i], Z_CHI_GRID[i + 1]
            th0 = _interp_at_z(rows, z0, COL_THETA_C)
            th1 = _interp_at_z(rows, z1, COL_THETA_C)
            if th0 is None or th1 is None:
                continue
            dz = z1 - z0
            if dz > 0:
                chi = (th1 - th0) / dz
                if chi > chi_max:
                    chi_max = chi
                    z_at = 0.5 * (z0 + z1)
        print(f"  {d:<4} {chi_max:<12.4f} {z_at:<12.4f}")


def section_12b_exact_susceptibility(
    d_values: Sequence[int],
    micro_rows: Dict[int, list],
) -> None:
    print("\n" + "=" * 5)
    print("12b. EXACT SUSCEPTIBILITY d(theta)/dz")
    print("=" * 5)
    print(f"  z grid: {list(Z_CHI_GRID)}")
    print("  theta from exact rank PMF (Mobius, all d in D_VALUES)")

    print(
        f"\n  {'d':<4} {'chi exact':<12} {'z at max':<12} {'chi MC':<12} {'|gap|':<10}"
    )
    print("  " + "-" * 5)
    for d in sorted(d_values):
        chi_max = 0.0
        z_at = Z_CHI_GRID[0]
        for i in range(len(Z_CHI_GRID) - 1):
            z0, z1 = Z_CHI_GRID[i], Z_CHI_GRID[i + 1]
            p0 = p_from_z_root_micro_ref(z0, d)
            p1 = p_from_z_root_micro_ref(z1, d)
            th0 = exact_micro_ref_theta_cond(p0, d)
            th1 = exact_micro_ref_theta_cond(p1, d)
            dz = z1 - z0
            if dz > 0:
                chi = (th1 - th0) / dz
                if chi > chi_max:
                    chi_max = chi
                    z_at = 0.5 * (z0 + z1)

        chi_mc = float("nan")
        if micro_rows and d in micro_rows:
            rows = micro_rows[d]
            chi_mc_val = 0.0
            for i in range(len(Z_CHI_GRID) - 1):
                z0, z1 = Z_CHI_GRID[i], Z_CHI_GRID[i + 1]
                th0 = _interp_at_z(rows, z0, COL_THETA_C)
                th1 = _interp_at_z(rows, z1, COL_THETA_C)
                if th0 is None or th1 is None:
                    continue
                dz = z1 - z0
                if dz > 0:
                    chi_mc_val = max(chi_mc_val, (th1 - th0) / dz)
            chi_mc = chi_mc_val

        gap = abs(chi_max - chi_mc) if micro_rows and d in micro_rows else float("nan")
        mc_str = f"{chi_mc:<12.4f}" if chi_mc == chi_mc else f"{'--':<12}"
        gap_str = f"{gap:<10.4f}" if gap == gap else f"{'--':<10}"
        print(f"  {d:<4} {chi_max:<12.4f} {z_at:<12.4f} {mc_str} {gap_str}")


# ---------------------------------------------------------------------------
# §13
# ---------------------------------------------------------------------------


def section_13_threshold_summary(micro_rows: Dict[int, list]) -> None:
    print("\n" + "=" * 5)
    print("13. THRESHOLD SUMMARY")
    print("=" * 5)
    c_star = rank_excess_limit_c_root_extrapolated()
    if not micro_rows:
        print(f"\n  analytic-only (no MC):")
        print(f"  {'d':<4} {'z_root,c ana':<12} {'p_c ana':<12} {'p_form':<12}")
        print("  " + "-" * 5)
        for d in D_VALUES:
            p_a = _bisect_p_c_rank(d)
            print(
                f"  {d:<4} {z_root_micro_ref(p_a, d):<12.4f} {p_a:<12.6f} "
                f"{closed_form_p_c_rank_micro_ref(d, c_star):<12.6f}"
            )
        return

    print(
        f"\n  {'d':<4} {'|Omega|':<10} {'p_c full':<12} {'z_root,c full':<14} "
        f"{'p_c rank MC':<12} {'p_c rank ana':<12} {'p_form':<12}"
    )
    print("  " + "-" * 5)
    for d in sorted(micro_rows):
        rows = micro_rows[d]
        p_full = _find_crossing(rows, COL_P_FULL_U)
        p_rank = _find_crossing(rows, COL_P_RANK_U)
        p_rank_ana = _bisect_p_c_rank(d)
        p_form = closed_form_p_c_rank_micro_ref(d, c_star)
        z_c = z_root_micro_ref(p_full, d) if p_full else float("nan")
        print(
            f"  {d:<4} {1 << (2 * d):<10} "
            f"{(p_full if p_full else float('nan')):<12.5f} {z_c:<10.4f} "
            f"{(p_rank if p_rank else float('nan')):<12.5f} {p_rank_ana:<12.5f} "
            f"{p_form:<12.5f}"
        )


# ---------------------------------------------------------------------------
# §14
# ---------------------------------------------------------------------------


def section_14_d6_pa1_crosscheck() -> None:
    print("\n" + "=" * 5)
    print("14. d=6 BFS vs analysis_1")
    print("=" * 5)
    import hqvm_percolation_analysis_1 as pa1

    eng = _engine(6)
    omega24 = pa1.enumerate_omega()
    eng24 = pa1.build_transition_engine(omega24)
    tests = [
        ("all 256", list(range(256))),
        ("q weight <= 3", [b for b in range(256) if eng.q_weight[b] <= 3]),
        ("q weight odd", [b for b in range(256) if eng.q_weight[b] % 2 == 1]),
    ]
    fails = 0
    print(f"\n  {'case':<20} {'family':<10} {'pa1':<10} {'PASS':<6}")
    print("  " + "-" * 5)
    for name, allowed in tests:
        if not allowed:
            continue
        r_f, _, _, _ = bfs_reach(eng, allowed)
        r_1 = pa1.compute_reachability(eng24, allowed, track_depth=False).reachable
        ok = r_f == r_1
        if not ok:
            fails += 1
        print(f"  {name:<20} {r_f:<10} {r_1:<10} {'PASS' if ok else 'FAIL':<6}")
    print(f"\n  aggregate: {'PASS' if fails == 0 else f'{fails} FAIL'}")


# ---------------------------------------------------------------------------
# §15
# ---------------------------------------------------------------------------


def section_15_aperture_delta(d_values: Sequence[int]) -> None:
    print("\n" + "=" * 5)
    print("15. DEPTH-4 APERTURE Delta(d)")
    print("=" * 5)
    print("  mean_fd = mean fold_disagree / n_pairs (byte aperture, 1/2 at all d)")
    print("  Delta_spin = mean_fd / (4d) = 1/(8d) when mean_fd = 1/2")
    print("  Delta_horizon = 1/(8d);  (8d)*Delta_horizon = 1")
    print("  E[S]/d = mean popcount(u xor v)/d on Omega")

    try:
        from gyroscopic.hQVM.constants import APERTURE_GAP
    except ImportError:
        APERTURE_GAP = None

    print(
        f"\n  {'d':<4} {'8d':<6} {'mean_fd':<10} {'E[S]/d':<10} {'S exact':<8} "
        f"{'Delta_sp':<10} {'1/(8d)':<10} {'8d*D':<10} {'F^2 rest':<10}"
    )
    print("  " + "-" * 5)
    for d in d_values:
        mean_fd = mean_fold_disagreement_d(d)
        ent = mean_carrier_entanglement_d(d)
        exact_ok, got_s, exp_s = verify_carrier_entanglement_exact(d)
        delta_sp = delta_spinorial_residual_d(d)
        horizon = delta_depth4_horizon_d(d)
        prod = depth4_projection_bits(d) * horizon
        f_ok, f_n = verify_f_squared_rest_d(d)
        print(
            f"  {d:<4} {depth4_projection_bits(d):<6} {mean_fd:<10.6f} {ent:<10.6f} "
            f"{'PASS' if exact_ok else 'FAIL':<8} {delta_sp:<10.6f} {horizon:<10.6f} "
            f"{prod:<10.6f} {f_ok}/{f_n:<6}"
        )
        if not exact_ok:
            print(f"       S sum: got={got_s} expect={exp_s}")

    print(f"\n  {'d':<4} {'5/2^n':<10} {'byte_curved':<12} {'compress':<10}")
    print("  " + "-" * 5)
    for d in d_values:
        _, _, curved_rate = mean_byte_curvature_rate(d)
        dyadic = delta_dyadic_byte_d(d)
        horizon = delta_depth4_horizon_d(d)
        compress = mean_fold_disagreement_d(d) / horizon if horizon > 0 else 0.0
        print(f"  {d:<4} {dyadic:<10.6f} {curved_rate:<12.6f} {compress:<10.3f}")

    if APERTURE_GAP is not None:
        h6 = delta_depth4_horizon_d(6)
        dsp6 = delta_spinorial_residual_d(6)
        print(f"\n  d=6 APERTURE_GAP     = {APERTURE_GAP:.12f}")
        print(f"  d=6 1/(8d) = 1/48   = {h6:.12f}")
        print(f"  d=6 Delta_spinorial  = {dsp6:.12f}")
        print(f"  |1/48 - APERTURE_GAP| = {abs(h6 - APERTURE_GAP):.12f}")
        print(f"  |Delta_spin - APERTURE_GAP| = {abs(dsp6 - APERTURE_GAP):.12f}")

    print("\n  d=6 fold_disagreement vs hqvm_wavefunction_kernel:")
    try:
        from hqvm_wavefunction_kernel import fold_disagreement as fold_disagreement_wf
    except ImportError:
        print("    SKIP: hqvm_wavefunction_kernel not importable")
        return
    mism = sum(
        1 for b in range(256) if fold_disagreement_d(b, 6) != fold_disagreement_wf(b)
    )
    print(f"    mismatches / 256 = {mism}  {'PASS' if mism == 0 else 'FAIL'}")


# ---------------------------------------------------------------------------
# §16
# ---------------------------------------------------------------------------


def section_16_headline_checklist(
    d_values: Sequence[int],
    micro_rows: Dict[int, list],
    square_root_pass: Tuple[int, int] | None = None,
) -> None:
    print("\n" + "=" * 5)
    print("16. HEADLINE CHECKLIST")
    print("=" * 5)
    c_star = rank_excess_limit_c_root_extrapolated()
    print(f"  c*_inf (1/d extrapolation) = {c_star:.6f}")

    print("\n  1. Square-root |Reach| = (2^r)^2 (fiber-complete)")
    if square_root_pass:
        ok, tot = square_root_pass
        print(f"     {ok}/{tot} PASS")
    else:
        print("     see section 1")

    print("\n  2. Delta(d) = 1/(8d), 8d*Delta = 1")
    delta_ok = all(
        abs(delta_spinorial_residual_d(d) - delta_depth4_horizon_d(d)) < 1e-9
        for d in d_values
    )
    ent_ok = all(verify_carrier_entanglement_exact(d)[0] for d in d_values)
    print(f"     Delta_spin = 1/(8d): {'PASS' if delta_ok else 'FAIL'}")
    print(f"     S = d/2 exact:       {'PASS' if ent_ok else 'FAIL'}")

    print("\n  3. Parity obstruction (even rank d-1, odd full)")
    print("     see section 10")

    print("\n  4. Rank threshold p_c(rank) [asymptotic d>=6, root variable]")
    d_gate = (14, 16, 18, 20)
    max_err = max(
        abs(_bisect_p_c_rank(d) - closed_form_p_c_rank_micro_ref(d, c_star))
        for d in d_gate
    )
    print(
        f"     max |p_c - p_form| (d=14..20) = {max_err:.2e}  {'PASS' if max_err < 1e-4 else 'check'}"
    )
    print(f"     finite-size 1/d correction at small d:")
    for d in (6, 7, 8):
        err = abs(_bisect_p_c_rank(d) - closed_form_p_c_rank_micro_ref(d, c_star))
        print(f"       d={d}: |p_c - p_form| = {err:.6f}")

    if micro_rows:
        print("\n  5. Micro-ref P(full) vs P(rank=d) at p_c  [unconditional]")
        print(f"     {'d':<4} {'gap':<10} {'status':<10}")
        print("     " + "-" * 5)
        for d, _, _, _, gap, _ in _micro_threshold_rows(micro_rows):
            status = "PASS" if abs(gap) < 0.05 else "check"
            print(f"     {d:<4} {gap:<10.4f} {status:<10}")


def _run_analysis(*, fast: bool) -> None:
    random.seed(SEED)
    print("hQVM(d) finite-size scaling (analysis_5)")
    print("=" * 5)
    print(f"  seed={SEED}")
    print(f"  kernel: gyroscopic.hQVM.family")
    print(f"  d values: {list(D_VALUES)}")
    print(f"  mode: {'fast (no MC)' if fast else 'full'}")

    section_0_api_check()
    section_0c_bridge_gate()
    section_0d_rank_distribution_check()
    section_0e_root_rank_lock()
    sr_pass = section_1_square_root(D_VALUES)
    section_2_rank_analytics(D_VALUES)
    section_2c_asymptotic_convergence()
    section_3_thermodynamic_shell(D_VALUES)
    section_4_fold_curvature(D_VALUES)

    micro: Dict[int, list] = {}
    byte: Dict[int, list] = {}
    if fast:
        print("\n" + "=" * 5)
        print("MC SECTIONS 5-7 SKIPPED (--fast)")
        print("=" * 5)
    else:
        micro = section_5_micro_ref_mc(D_MC_HEAVY)
        section_6_qclass_mc(D_MC_HEAVY)
        byte = section_7_byte_mc(D_MC_HEAVY)

    section_8_scaling_collapse(micro)
    section_9_rank_full_gap(micro, byte)
    section_10_parity(D_VALUES)
    section_11_holonomy(D_VALUES)
    section_12_susceptibility(micro)
    section_12b_exact_susceptibility(D_VALUES, micro)
    section_13_threshold_summary(micro)
    if not fast:
        section_14_d6_pa1_crosscheck()
    else:
        print("\n" + "=" * 5)
        print("14. d=6 BFS vs analysis_1  SKIPPED (--fast)")
        print("=" * 5)
    section_15_aperture_delta(D_VALUES)
    section_16_headline_checklist(D_VALUES, micro, square_root_pass=sr_pass)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="hQVM(d) finite-size scaling (analysis_5)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Deterministic sections only (skip MC and d=6 pa1 cross-check)",
    )
    parser.add_argument(
        "--no-tee",
        action="store_true",
        help="Do not write hqvm_percolation_analysis_5_results.txt",
    )
    args = parser.parse_args()
    _configure_stdout_utf8()

    results_path = _EXPERIMENTS_DIR / "hqvm_percolation_analysis_5_results.txt"
    if args.no_tee:
        _run_analysis(fast=args.fast)
        return

    import io

    buf = io.StringIO()

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data: str) -> int:
            for s in self._streams:
                s.write(data)
            return len(data)

        def flush(self) -> None:
            for s in self._streams:
                s.flush()

    orig = sys.stdout
    sys.stdout = _Tee(orig, buf)
    try:
        _run_analysis(fast=args.fast)
    finally:
        sys.stdout = orig

    results_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nWrote {results_path}", file=orig)


if __name__ == "__main__":
    main()
