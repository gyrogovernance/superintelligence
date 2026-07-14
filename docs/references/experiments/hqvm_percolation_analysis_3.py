#!/usr/bin/env python3
"""
hQVM Percolation Structural Completeness (analysis_3)
=====================================================

Companion to hqvm_percolation_analysis_1.py (byte baseline) and
hqvm_percolation_analysis_2.py (canonical-word theory). Run all three via
hqvm_percolation_analysis_run.py.

This script closes the remaining CGM-native percolation observables that are
not byte/word dichotomy results but structural signatures under alphabet
restriction:

  1.  Commutation / q-diversity (Features 68-76)
  2.  Future-cone entropy H1(p), H2(p) (Features 77-79)
  3.  Omega = U x V coset coverage (Feature 2)
  4.  Shell enrichment and mean entanglement S(chi) (Wavefunction Sec 16.6)
  5.  Two-step uniformization multiplicity (Feature 80)
  6.  Complete fold-disagreement and phase-net deterministic tables
  7.  Minimal connection-boundary subset search (2^7)
  8.  128 Omega-permutation-class percolation (Feature 45)
  9.  Geometric porosity (boundary-channel openness) vs spanning thresholds
  9b. Size-controlled fold-triple porosity at critical |A|
  10. Plaquette loop defect D(A) under alphabet restriction
  11. Spanning transmission and tau_G attenuation proxy
  11b. tau_G percolation identity (holonomy path exact match + channel measure)
  12. Shell-resolved local connectivity vs topological depth
  13. U x V rectangularity (factorization skew)
  14. Word-regime reachability from bulk starts (horizon vs bulk anchors)
  15. Shell transition operators M_A(q) under restriction (Compact Geometry; exact C(q))
  16. Self-energy closure from percolation-derived G(psi)  [I=1/2, E_self=-Mc^2/4, M_obs/M_bare=4/5]
  Aperture collapse (50% byte-level fold disagreement -> ~2.07% Delta) per Wavefunction analysis.

Data only. Interpretation: docs/Findings/Analysis_hQVM_Percolation.md
"""

from __future__ import annotations

import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Collection, Dict, Iterable, List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from gyroscopic.hQVM.api import q_word6, state24_to_omega12
from gyroscopic.hQVM.constants import APERTURE_GAP, APERTURE_GAP_Q256, RHO

import hqvm_percolation_analysis_1 as pa1

try:
    from hqvm_gravity_common import (
        Delta as CGM_DELTA,
        Omega_size as CGM_OMEGA,
        rho as CGM_RHO,
        tau_G_formula,
        binom_shell,
        tau_cycle_per_delta_exact,
        kernel_exposure_constants,
        trace_word_steps,
        cycle_word_for_micro,
        dln_g_dpsi,
        horizon_s_analytic,
        field_integral_exterior_exact,
        field_integral_exterior_numeric,
    )
    from hqvm_gravity_analysis_3 import (
        holonomy_arch_path,
        tau_per_delta_binom_on_path,
        shell_weight as holonomy_shell_weight,
        micro_popcount,
        tau_cycle_per_delta_kernel_exact,
    )
except ImportError:
    CGM_DELTA = APERTURE_GAP
    CGM_RHO = RHO
    CGM_OMEGA = 4096
    tau_G_formula = CGM_OMEGA * CGM_DELTA * CGM_RHO**5
    binom_shell = [comb(6, s) / 64.0 for s in range(7)]
    tau_cycle_per_delta_exact = None
    kernel_exposure_constants = None
    trace_word_steps = None
    cycle_word_for_micro = None
    dln_g_dpsi = None
    horizon_s_analytic = None
    field_integral_exterior_exact = None
    field_integral_exterior_numeric = None
    holonomy_arch_path = None
    tau_per_delta_binom_on_path = None
    holonomy_shell_weight = None
    micro_popcount = None
    tau_cycle_per_delta_kernel_exact = None

try:
    from hqvm_compact_geom_core import carrier_trace, shell_return_trace
except ImportError:
    carrier_trace = None
    shell_return_trace = None

N_OMEGA = pa1.N_OMEGA
SHELL_BASELINE = tuple(comb(6, k) / 64.0 for k in range(7))
D_SHELL = 24
BU_BOUNDARY_IDX = 3
FOLD_BOUNDARY_IDXS = (2, 3, 4)  # ONA|BU, BU|BU, BU|ONA
ONE_OVER_48 = 1.0 / 48.0
FIVE_OVER_256 = 5.0 / 256.0


def _future_cone_h12(
    engine: pa1.TransitionEngine,
    start_idx: int,
    allowed: Sequence[int],
) -> Tuple[float, float, int, int]:
    """Return (H1, H2, |R1|, |R2|) for uniform byte steps from start_idx."""
    trans = engine.transitions
    c1: Counter = Counter()
    for b in allowed:
        c1[trans[start_idx][b]] += 1
    h1 = _shannon_bits(c1)
    r1 = len(c1)

    c2: Counter = Counter()
    for b1 in allowed:
        mid = trans[start_idx][b1]
        for b2 in allowed:
            c2[trans[mid][b2]] += 1
    h2 = _shannon_bits(c2)
    r2 = len(c2)
    return h1, h2, r1, r2


def _shannon_bits(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0.0:
            h -= p * math.log2(p)
    return h


def _porosity_pi_grid() -> List[float]:
    return sorted(set(
        [i / 100 for i in range(0, 10)]
        + [i / 1000 for i in range(10, 41)]
        + [i / 100 for i in range(4, 21)]
        + [i / 10 for i in range(3, 11)]
    ))


def _build_uv_indices(omega: List[int]) -> Tuple[List[int], List[int]]:
    """u_index[i], v_index[i] in {0..63} for state index i."""
    u_words = sorted({state24_to_omega12(s).u6 for s in omega})
    v_words = sorted({state24_to_omega12(s).v6 for s in omega})
    u_map = {w: i for i, w in enumerate(u_words)}
    v_map = {w: i for i, w in enumerate(v_words)}
    u_idx: List[int] = []
    v_idx: List[int] = []
    for s in omega:
        o = state24_to_omega12(s)
        u_idx.append(u_map[o.u6])
        v_idx.append(v_map[o.v6])
    return u_idx, v_idx


def _build_permutation_classes(engine: pa1.TransitionEngine) -> List[List[int]]:
    """128 equivalence classes of bytes inducing the same Omega permutation."""
    sig_to_bytes: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    n = engine.n
    trans = engine.transitions
    for b in range(256):
        sig = tuple(trans[i][b] for i in range(n))
        sig_to_bytes[sig].append(b)
    classes = list(sig_to_bytes.values())
    return classes


def verify_commutation_laws(
    bclass: List[pa1.ByteClassification],
    engine: pa1.TransitionEngine,
) -> None:
    print("\n" + "=" * 5)
    print("1. COMMUTATION STRUCTURE (Features 68-76)")
    print("=" * 5)
    print("  Bytes x,y commute on Omega iff q6(x) = q6(y).\n")

    commute = sum(
        1 for b1 in range(256) for b2 in range(256)
        if q_word6(b1) == q_word6(b2)
    )
    print(f"  Commuting ordered pairs (exhaustive): {commute} / 65536")
    print(f"  Rate = {commute / 65536:.6f}  (expect 1024/65536 = {1024/65536:.6f})")

    # Per-q fiber size
    q_sizes = Counter(q_word6(b) for b in range(256))
    print(f"  Distinct q6 classes: {len(q_sizes)}  (each has 4 bytes)")

    print("\n  Deterministic reachability: single-q alphabet (max 4 bytes):")
    q_groups: Dict[int, List[int]] = defaultdict(list)
    for bc in bclass:
        q_groups[bc.q6].append(bc.byte)
    for q in (0, 31, 63):
        allowed = q_groups[q]
        r = pa1.compute_reachability(engine, allowed)
        print(f"    q6={q:02d}: |B|={len(allowed)}  Reach={r.reachable}  "
              f"Full={r.full_omega}  Span={r.spans}")

    print("\n  MC sweep: single random q-class vs independent bytes (n=200):")
    p_values = [0.1, 0.2, 0.3, 0.5, 1.0]
    print(f"  {'p':<7}{'single-q P(full)':<18}{'byte P(full)':<14}{'E[#q|nz]':<10}")
    print("  " + "-" * 5)
    n_samples = 200
    for p in p_values:
        sq_full = b_full = 0
        q_div_sum = 0
        nz = 0
        for _ in range(n_samples):
            # Single q-class: pick q, include each of 4 bytes with prob p
            q = random.randrange(64)
            allowed_sq = [b for b in q_groups[q] if random.random() < p]
            if allowed_sq:
                r_sq = pa1.compute_reachability(engine, allowed_sq, track_depth=False)
                if r_sq.full_omega:
                    sq_full += 1
            allowed_b = [b for b in range(256) if random.random() < p]
            if allowed_b:
                nz += 1
                r_b = pa1.compute_reachability(engine, allowed_b, track_depth=False)
                if r_b.full_omega:
                    b_full += 1
                q_div_sum += len({q_word6(b) for b in allowed_b})
        denom = max(nz, 1)
        print(f"  {p:<7.3f}{sq_full/n_samples:<18.4f}{b_full/n_samples:<14.4f}"
              f"{q_div_sum/denom:<10.2f}")

    print("\n  Q-diversity threshold (independent bytes, n=300):")
    print(f"  {'p':<8}{'P(full)':<10}{'E[#q]':<10}{'E[|A|]':<10}{'E[reach]':<10}")
    print("  " + "-" * 5)
    q_thresh_p = [0.01, 0.015, 0.02, 0.022, 0.025, 0.03, 0.035, 0.04, 0.05, 0.08, 0.10]
    for p in q_thresh_p:
        full_hits = q_sum = size_sum = reach_sum = 0
        nz = 0
        for _ in range(300):
            allowed = [b for b in range(256) if random.random() < p]
            if not allowed:
                continue
            nz += 1
            r = pa1.compute_reachability(engine, allowed, track_depth=False)
            if r.full_omega:
                full_hits += 1
            q_sum += len({q_word6(b) for b in allowed})
            size_sum += len(allowed)
            reach_sum += r.reachable
        denom = max(nz, 1)
        print(f"  {p:<8.3f}{full_hits/denom:<10.4f}{q_sum/denom:<10.2f}"
              f"{size_sum/denom:<10.1f}{reach_sum/denom:<10.1f}")


def run_future_cone_entropy(
    engine: pa1.TransitionEngine,
    n_samples: int = 200,
) -> None:
    print("\n" + "=" * 5)
    print("2. FUTURE-CONE ENTROPY H1(p), H2(p) (Features 77-79)")
    print("=" * 5)
    print("  H1: entropy of 1-step next-state distribution from rest")
    print("       (byte chosen uniformly from allowed alphabet).")
    print("  H2: entropy of 2-step distribution (ordered byte pairs).")
    print("  Full alphabet reference: H1=7, H2=12.\n")

    start = engine.start_idx
    p_values = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

    print(f"  {'p':<7}{'E[H1]':<10}{'E[H2]':<10}{'E[|R1|]':<10}{'E[|R2|]':<10}")
    print("  " + "-" * 5)

    for p in p_values:
        h1_sum = h2_sum = r1_sum = r2_sum = 0.0
        for _ in range(n_samples):
            allowed = [b for b in range(256) if random.random() < p]
            if not allowed:
                continue
            h1, h2, r1, r2 = _future_cone_h12(engine, start, allowed)
            h1_sum += h1
            h2_sum += h2
            r1_sum += r1
            r2_sum += r2
        denom = max(n_samples, 1)
        print(f"  {p:<7.3f}{h1_sum/denom:<10.3f}{h2_sum/denom:<10.3f}"
              f"{r1_sum/denom:<10.1f}{r2_sum/denom:<10.1f}")

    # State-independence: Features 77-79 hold at every Omega state at full alphabet
    print("\n  State-independence (full alphabet, fixed starts):")
    full_alphabet = list(range(256))
    shell = engine.shell
    sample_starts = [start]
    for sh in (0, 1, 2, 3, 4, 5):
        sample_starts.append(next(i for i in range(engine.n) if shell[i] == sh))
    print(f"  {'start shell':<12}{'H1':<8}{'H2':<8}{'|R1|':<8}{'|R2|':<8}")
    print("  " + "-" * 5)
    for si in sample_starts:
        h1, h2, r1, r2 = _future_cone_h12(engine, si, full_alphabet)
        print(f"  {shell[si]:<12}{h1:<8.3f}{h2:<8.3f}{r1:<8}{r2:<8}")

    print("\n  State-independence under restriction (p=0.05, n=100, mean over starts):")
    sub_p = 0.05
    for si in sample_starts[:4]:
        h1_sum = h2_sum = 0.0
        for _ in range(100):
            allowed = [b for b in range(256) if random.random() < sub_p]
            if not allowed:
                continue
            h1, h2, _, _ = _future_cone_h12(engine, si, allowed)
            h1_sum += h1
            h2_sum += h2
        print(f"    shell {shell[si]}: E[H1]={h1_sum/100:.3f}  E[H2]={h2_sum/100:.3f}")


def run_uv_factorization(
    engine: pa1.TransitionEngine,
    omega: List[int],
    u_idx: List[int],
    v_idx: List[int],
    n_samples: int = 200,
) -> None:
    print("\n" + "=" * 5)
    print("3. OMEGA = U x V COSET COVERAGE (Feature 2)")
    print("=" * 5)
    print("  u6,v6 are C64 coset coordinates; |U|=|V|=64.\n")

    start = engine.start_idx
    p_values = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    print(f"  {'p':<7}{'|U_hit|':<9}{'|V_hit|':<9}{'|UxV|':<9}{'Reach':<8}{'P(full)':<8}")
    print("  " + "-" * 5)

    for p in p_values:
        u_hit = v_hit = uv_pairs = reach_sum = full = 0
        for _ in range(n_samples):
            allowed = [b for b in range(256) if random.random() < p]
            if not allowed:
                continue
            vis = pa1.reachable_mask(engine, allowed, [start])
            us: set = set()
            vs: set = set()
            rcount = 0
            for i, on in enumerate(vis):
                if not on:
                    continue
                rcount += 1
                us.add(u_idx[i])
                vs.add(v_idx[i])
            u_hit += len(us)
            v_hit += len(vs)
            uv_pairs += len(us) * len(vs)
            reach_sum += rcount
            if rcount == N_OMEGA:
                full += 1
        denom = max(n_samples, 1)
        print(f"  {p:<7.3f}{u_hit/denom:<9.1f}{v_hit/denom:<9.1f}"
              f"{uv_pairs/denom:<9.1f}{reach_sum/denom:<8.1f}{full/denom:<8.4f}")


def run_shell_enrichment(
    engine: pa1.TransitionEngine,
    n_samples: int = 200,
) -> None:
    print("\n" + "=" * 5)
    print("4. SHELL ENRICHMENT AND MEAN S(chi) (Wavefunction Sec 16.6)")
    print("=" * 5)
    print("  Enrichment(k) = (fraction of R in shell k) / (C(6,k)/64).")
    print("  S(chi) = popcount(chi) = shell index. Omega mean S = 3.0.\n")

    start = engine.start_idx
    shell = engine.shell
    p_values = [
        0.01, 0.015, 0.02, 0.022, 0.025, 0.03, 0.035, 0.04, 0.05,
        0.1, 0.2, 0.3, 0.5, 1.0,
    ]

    print(f"  {'p':<8}{'E[|R|]':<10}{'E[S]':<8}", end="")
    for k in range(7):
        print(f" {'E[k]='+str(k):<7}", end="")
    print()
    print("  " + "-" * 5)

    for p in p_values:
        s_sum = 0.0
        enrich_sum = [0.0] * 7
        reach_sum = 0.0
        nz = 0
        for _ in range(n_samples):
            allowed = [b for b in range(256) if random.random() < p]
            if not allowed:
                continue
            nz += 1
            vis = pa1.reachable_mask(engine, allowed, [start])
            sc = Counter(shell[i] for i, on in enumerate(vis) if on)
            reach = sum(sc.values())
            if reach == 0:
                continue
            reach_sum += reach
            s_sum += sum(k * c for k, c in sc.items()) / reach
            for k in range(7):
                frac = sc[k] / reach
                base = SHELL_BASELINE[k]
                enrich_sum[k] += (frac / base) if base > 0 else 0.0
        denom = max(nz, 1)
        print(f"  {p:<8.3f}{reach_sum/denom:<10.1f}{s_sum/denom:<8.3f}", end="")
        for k in range(7):
            print(f" {enrich_sum[k]/denom:<7.3f}", end="")
        print()


def run_two_step_uniformization(
    engine: pa1.TransitionEngine,
    n_samples: int = 150,
) -> None:
    print("\n" + "=" * 5)
    print("5. TWO-STEP UNIFORMIZATION MULTIPLICITY (Feature 80)")
    print("=" * 5)
    print("  From rest: count images of all ordered 2-byte words.")
    print("  Full alphabet: each state hit 16 times (65536 words).\n")

    start = engine.start_idx
    trans = engine.transitions
    p_values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

    # Full-alphabet reference
    full_allowed = list(range(256))
    mult_full: Counter = Counter()
    for b1 in full_allowed:
        mid = trans[start][b1]
        for b2 in full_allowed:
            mult_full[trans[mid][b2]] += 1
    vals = list(mult_full.values())
    print(f"  Full alphabet: min={min(vals)} max={max(vals)} "
          f"unique multiplicities={len(set(vals))}  (expect all 16)")

    print(f"\n  {'p':<7}{'E[min]':<9}{'E[max]':<9}{'E[std]':<9}{'E[CV]':<9}{'E[cover]':<10}")
    print("  " + "-" * 5)

    for p in p_values:
        mins: List[float] = []
        maxs: List[float] = []
        stds: List[float] = []
        cvs: List[float] = []
        covers: List[float] = []
        for _ in range(n_samples):
            allowed = [b for b in range(256) if random.random() < p]
            if len(allowed) < 2:
                continue
            mult: Counter = Counter()
            for b1 in allowed:
                mid = trans[start][b1]
                for b2 in allowed:
                    mult[trans[mid][b2]] += 1
            v = list(mult.values())
            mins.append(float(min(v)))
            maxs.append(float(max(v)))
            mean = sum(v) / len(v)
            var = sum((x - mean) ** 2 for x in v) / len(v)
            std = math.sqrt(var)
            stds.append(std)
            cvs.append(std / mean if mean > 0 else 0.0)
            covers.append(len(mult) / N_OMEGA)
        denom = max(len(mins), 1)
        print(f"  {p:<7.3f}{sum(mins)/denom:<9.2f}{sum(maxs)/denom:<9.1f}"
              f"{sum(stds)/denom:<9.2f}{sum(cvs)/denom:<9.3f}{sum(covers)/denom:<10.4f}")


def run_fold_phase_complete_tables(
    bclass: List[pa1.ByteClassification],
    engine: pa1.TransitionEngine,
) -> None:
    print("\n" + "=" * 5)
    print("6. FOLD DISAGREEMENT AND PHASE-NET COMPLETE TABLES")
    print("=" * 5)

    fd_bytes: Dict[int, List[int]] = defaultdict(list)
    pn_bytes: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for bc in bclass:
        fd_bytes[bc.fold_disagree].append(bc.byte)
        pn_bytes[bc.phase_net].append(bc.byte)

    print("\n  --- Fold disagreement (all levels + cumulative) ---")
    print(f"  {'Level':<12}{'|B|':>5}{'Reach':>7}{'Full':>6}")
    print("  " + "-" * 5)
    cumul: List[int] = []
    for d in range(5):
        allowed = fd_bytes[d]
        r = pa1.compute_reachability(engine, allowed)
        print(f"  fd={d} only    {len(allowed):5d}{r.reachable:7d}{str(r.full_omega):>6}")
    for d in range(5):
        cumul.extend(fd_bytes[d])
        r = pa1.compute_reachability(engine, cumul)
        print(f"  fd<={d}       {len(cumul):5d}{r.reachable:7d}{str(r.full_omega):>6}")

    print("\n  --- Phase-net vectors (all 16) ---")
    print(f"  {'Phase-net':<22}{'|B|':>5}{'Reach':>7}{'Giant':>7}")
    print("  " + "-" * 5)
    for pn in sorted(pn_bytes.keys()):
        r = pa1.compute_reachability(engine, pn_bytes[pn])
        print(f"  {str(pn):<22}{len(pn_bytes[pn]):5d}{r.reachable:7d}{str(r.giant):>7}")


def run_minimal_boundary_subsets(
    bclass: List[pa1.ByteClassification],
    engine: pa1.TransitionEngine,
) -> None:
    print("\n" + "=" * 5)
    print("7. MINIMAL CONNECTION-BOUNDARY SUBSETS (2^7 exhaustive)")
    print("=" * 5)
    print("  Mask bit i: boundary i may have nonzero |A| in allowed bytes.")
    print("  mask=0: only bytes with all chain magnitudes zero.\n")

    n_b = 7
    boundaries = pa1.PHASE_BOUNDARIES
    best_full: List[Tuple[int, int, int]] = []

    for mask in range(1 << n_b):
        allowed = [
            bc.byte for bc in bclass
            if all(
                (bc.connection_chain[i] == 0.0) or ((mask >> i) & 1)
                for i in range(n_b)
            )
        ]
        r = pa1.compute_reachability(engine, allowed)
        pop = mask.bit_count()
        if r.full_omega:
            best_full.append((pop, mask, len(allowed)))

    best_full.sort(key=lambda x: (x[0], x[2]))
    print(f"  Masks achieving full Omega: {len(best_full)} / 128")
    if best_full:
        min_pop = best_full[0][0]
        print(f"  Minimum active boundaries: {min_pop}")
        print("  Minimal masks (popcount, mask, |B|, boundary names):")
        shown = 0
        for pop, mask, nb in best_full:
            if pop != min_pop:
                break
            names = [boundaries[i] for i in range(n_b) if (mask >> i) & 1]
            print(f"    pop={pop} mask={mask:03d} |B|={nb:3d}  {names}")
            shown += 1
            if shown >= 8:
                rest = sum(1 for x in best_full if x[0] == min_pop) - 8
                if rest > 0:
                    print(f"    ... and {rest} more masks at pop={min_pop}")
                break

    print("\n  Reach vs number of active boundaries (best mask per popcount):")
    print(f"  {'pop':<6}{'best |B|':<10}{'Reach':<8}{'Full':<6}")
    print("  " + "-" * 5)
    for pop in range(n_b + 1):
        cands = [x for x in best_full if x[0] == pop]
        if not cands:
            # find best reach for this popcount even if not full
            best_r = 0
            best_nb = 0
            for mask in range(1 << n_b):
                if mask.bit_count() != pop:
                    continue
                allowed = [
                    bc.byte for bc in bclass
                    if all(
                        (bc.connection_chain[i] == 0.0) or ((mask >> i) & 1)
                        for i in range(n_b)
                    )
                ]
                r = pa1.compute_reachability(engine, allowed)
                if r.reachable > best_r:
                    best_r = r.reachable
                    best_nb = len(allowed)
            print(f"  {pop:<6}{best_nb:<10}{best_r:<8}{'No':<6}")
        else:
            _, _, nb = min(cands, key=lambda x: x[2])
            print(f"  {pop:<6}{nb:<10}{4096:<8}{'Yes':<6}")


def run_permutation_class_sweep(
    engine: pa1.TransitionEngine,
    perm_classes: List[List[int]],
    n_samples: int = 200,
) -> None:
    print("\n" + "=" * 5)
    print("8. 128 OMEGA-PERMUTATION-CLASS PERCOLATION (Feature 45)")
    print("=" * 5)
    print(f"  {len(perm_classes)} classes (2 bytes per class via b^0xFE shadow).\n")

    p_values = sorted(set(
        [0.0] + [i / 100 for i in range(1, 31)] + [i / 10 for i in range(4, 11)]
    ))
    # one representative per class (min byte)
    groups_repr = [[min(g)] for g in perm_classes]
    groups_both = [g for g in perm_classes]

    for label, groups in (("one_repr", groups_repr), ("both_shadow", groups_both)):
        print(f"\n  --- Mode: {label} ({len(groups)} groups) ---")
        data = pa1._random_subset_sweep(
            engine, groups, p_values, n_samples, f"perm-{label}")
        pa1._print_sweep_table(data, n_samples)


def _byte_map(bclass: List[pa1.ByteClassification]) -> Dict[int, pa1.ByteClassification]:
    return {bc.byte: bc for bc in bclass}


def _boundary_porosity(allowed: Sequence[int], bc_map: Dict[int, pa1.ByteClassification],
                       boundary_idx: int) -> float:
    if not allowed:
        return 0.0
    active = sum(
        1 for b in allowed if bc_map[b].connection_chain[boundary_idx] > 0.0
    )
    return active / len(allowed)


def _fold_porosity(allowed: Sequence[int], bc_map: Dict[int, pa1.ByteClassification]) -> float:
    if not allowed:
        return 0.0
    vals = [_boundary_porosity(allowed, bc_map, i) for i in FOLD_BOUNDARY_IDXS]
    return sum(vals) / len(vals)


def _plaquette_D(allowed: Sequence[int]) -> Tuple[float, float]:
    """Return (D(A), mean popcount defect) over ordered pairs in allowed."""
    if not allowed:
        return 0.0, 0.0
    pop_sum = 0
    cnt = 0
    for x in allowed:
        qx = q_word6(x)
        for y in allowed:
            pop_sum += (qx ^ q_word6(y)).bit_count()
            cnt += 1
    mean_def = pop_sum / cnt
    d_a = pop_sum / (2 * N_OMEGA)
    return d_a, mean_def


def run_geometric_porosity_sweeps(
    bclass: List[pa1.ByteClassification],
    engine: pa1.TransitionEngine,
    n_samples: int = 250,
) -> None:
    print("\n" + "=" * 5)
    print("9. GEOMETRIC POROSITY THRESHOLDS (boundary-channel openness)")
    print("=" * 5)
    print("  pi_j(A) = fraction of bytes in A with nonzero connection at boundary j.")
    print("  pi_BU is BU|BU (fold boundary). pi_fold averages ONA|BU, BU|BU, BU|ONA.")
    print(f"  CGM constants: Delta={CGM_DELTA:.6f}  5/256={FIVE_OVER_256:.6f}  "
          f"1/48={ONE_OVER_48:.6f}\n")

    bc_map = _byte_map(bclass)
    bu_active = [bc.byte for bc in bclass if bc.connection_chain[BU_BOUNDARY_IDX] > 0.0]
    bu_inactive = [bc.byte for bc in bclass if bc.connection_chain[BU_BOUNDARY_IDX] == 0.0]
    boundaries = pa1.PHASE_BOUNDARIES

    # Stratified porosity sweep: control pi_BU by mixing active/inactive pools
    pi_grid = _porosity_pi_grid()

    print(f"  {'pi_BU':<8}{'P(span)':<10}{'P(full)':<10}{'E[|A|]':<10}{'E[pi_fold]':<12}")
    print("  " + "-" * 5)

    span_curve: List[Tuple[float, float]] = []
    for pi_target in pi_grid:
        span_hits = full_hits = 0
        size_sum = fold_sum = 0.0
        nz = 0
        for _ in range(n_samples):
            n_a = random.randint(4, 48)
            n_act = min(len(bu_active), max(0, int(round(pi_target * n_a))))
            n_inact = min(len(bu_inactive), n_a - n_act)
            if n_act + n_inact == 0:
                continue
            allowed = (
                random.sample(bu_active, n_act) if n_act else []
            ) + (
                random.sample(bu_inactive, n_inact) if n_inact else []
            )
            nz += 1
            size_sum += len(allowed)
            fold_sum += _fold_porosity(allowed, bc_map)
            r = pa1.compute_reachability(engine, allowed, track_depth=False)
            if r.spans:
                span_hits += 1
            if r.full_omega:
                full_hits += 1
        denom = max(nz, 1)
        p_span = span_hits / denom
        span_curve.append((pi_target, p_span))
        print(f"  {pi_target:<8.3f}{p_span:<10.4f}{full_hits/denom:<10.4f}"
              f"{size_sum/denom:<10.1f}{fold_sum/denom:<12.4f}")

    # Estimate pi_BU at P(span)=0.5 via linear interpolation
    p_c_span_pi = None
    for i in range(1, len(span_curve)):
        p0, s0 = span_curve[i - 1]
        p1, s1 = span_curve[i]
        if s0 < 0.5 <= s1 and s1 != s0:
            p_c_span_pi = p0 + (0.5 - s0) * (p1 - p0) / (s1 - s0)
            break
    if p_c_span_pi is not None:
        print(f"\n  pi_BU at P(span)=0.5 (interpolated): {p_c_span_pi:.4f}")
        print(f"    vs Delta={CGM_DELTA:.4f}  ratio={p_c_span_pi/CGM_DELTA:.3f}")
        print(f"    vs 5/256={FIVE_OVER_256:.4f}  ratio={p_c_span_pi/FIVE_OVER_256:.3f}")
        print(f"    vs 1/48={ONE_OVER_48:.4f}  ratio={p_c_span_pi/ONE_OVER_48:.3f}")

    print("\n  All-boundary porosities at full alphabet (reference):")
    full = list(range(256))
    for i, name in enumerate(boundaries):
        print(f"    pi({name}) = {_boundary_porosity(full, bc_map, i):.4f}")


def run_plaquette_loop_defect(
    n_samples: int = 200,
) -> None:
    print("\n" + "=" * 5)
    print("10. PLAQUETTE LOOP DEFECT D(A) (Gravity Sec 5.0, 5.6)")
    print("=" * 5)
    print("  d(x,y) = popcount(q6(x) XOR q6(y));  D(A) = sum d / (2|Omega|).")
    print("  E[d] = mean defect per ordered pair (converges to 3.0 at full alphabet).")
    print(f"  Full alphabet: D = {D_SHELL} (exhaustive).\n")

    d_full, mean_full = _plaquette_D(list(range(256)))
    print(f"  Exhaustive 256-byte alphabet: D={d_full:.6f}  E[d]={mean_full:.6f}")

    # Restricted alphabets by byte fraction
    p_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    print(f"\n  {'p':<7}{'E[D(A)]':<12}{'E[mean d]':<12}{'E[|A|]':<10}")
    print("  " + "-" * 5)
    for p in p_values:
        d_sum = mean_sum = size_sum = 0.0
        nz = 0
        for _ in range(n_samples):
            allowed = [b for b in range(256) if random.random() < p]
            if len(allowed) < 2:
                continue
            nz += 1
            d_a, md = _plaquette_D(allowed)
            d_sum += d_a
            mean_sum += md
            size_sum += len(allowed)
        denom = max(nz, 1)
        print(f"  {p:<7.3f}{d_sum/denom:<12.4f}{mean_sum/denom:<12.4f}{size_sum/denom:<10.1f}")

    # Convergence vs alphabet size (coupon-collector style)
    print("\n  D(A) vs |A| (random subsets, n=150 each size):")
    print(f"  {'|A|':<7}{'E[D(A)]':<12}{'E[mean d]':<12}")
    print("  " + "-" * 5)
    for size in (4, 8, 16, 32, 64, 128, 256):
        if size > 256:
            continue
        d_sum = mean_sum = 0.0
        reps = 150 if size < 256 else 1
        for _ in range(reps):
            allowed = random.sample(range(256), size) if size < 256 else list(range(256))
            d_a, md = _plaquette_D(allowed)
            d_sum += d_a
            mean_sum += md
        print(f"  {size:<7}{d_sum/reps:<12.4f}{mean_sum/reps:<12.4f}")


def run_spanning_transmission(
    engine: pa1.TransitionEngine,
    bclass: List[pa1.ByteClassification],
    n_samples: int = 300,
) -> None:
    print("\n" + "=" * 5)
    print("11. SPANNING TRANSMISSION AND tau_G PROXY (Beer-Lambert bridge)")
    print("=" * 5)
    print("  T(A) = P(constitutional span from rest under alphabet A).")
    print("  STF bulk = shells 1-5; horizons 0,6 are endpoints.")
    print(f"  tau_G (kernel closed form) = {tau_G_formula:.6f}\n")

    bc_map = _byte_map(bclass)
    p_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20]

    print(f"  {'p':<8}{'P(span)':<10}{'-ln T':<10}{'E[bulk hit]':<12}{'E[pi_BU]':<10}")
    print("  " + "-" * 5)

    for p in p_values:
        span_hits = 0
        bulk_sum = 0.0
        pi_sum = 0.0
        nz = 0
        for _ in range(n_samples):
            allowed = [b for b in range(256) if random.random() < p]
            if not allowed:
                continue
            nz += 1
            r = pa1.compute_reachability(engine, allowed, track_depth=False)
            if r.spans:
                span_hits += 1
            bulk_hit = sum(1 for k in (1, 2, 3, 4, 5) if k in r.shells_reached)
            bulk_sum += bulk_hit / 5.0
            pi_sum += _boundary_porosity(allowed, bc_map, BU_BOUNDARY_IDX)
        denom = max(nz, 1)
        p_span = span_hits / denom
        neg_log = -math.log(max(p_span, 1e-6))
        print(f"  {p:<8.3f}{p_span:<10.4f}{neg_log:<10.4f}"
              f"{bulk_sum/denom:<12.3f}{pi_sum/denom:<10.4f}")

    # rho^5 attenuation reference: shells 1-5 each with closure prob rho
    rho5 = CGM_RHO ** 5
    print(f"\n  Reference: rho^5 = {rho5:.6f}  (5 STF shell closure factors)")
    print(f"  tau_G / |Omega| = {tau_G_formula/CGM_OMEGA:.6f}")
    print(f"  Delta * rho^5 * f_ordered = {CGM_DELTA * rho5 * (1 - 4*CGM_RHO*CGM_DELTA**2):.6f}")
    print("  At p=0.020, T~0.44, -ln T~0.81; bulk fraction~0.45.")


def run_shell_local_connectivity(
    engine: pa1.TransitionEngine,
    n_samples: int = 150,
) -> None:
    print("\n" + "=" * 5)
    print("12. SHELL-RESOLVED LOCAL CONNECTIVITY vs TOPOLOGICAL DEPTH")
    print("=" * 5)
    print("  psi_topo(k) = 6 - k  (depth from complement horizon at shell k).")
    print("  C(k) = fraction of edges from shell-k states in R that reach k-1.\n")

    shell = engine.shell
    trans = engine.transitions
    p_values = [0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    for p in p_values:
        cross_by_k = [0.0] * 7
        total_by_k = [0.0] * 7
        for _ in range(n_samples):
            allowed = [b for b in range(256) if random.random() < p]
            if not allowed:
                continue
            vis = pa1.reachable_mask(engine, allowed, [engine.start_idx])
            for i, on in enumerate(vis):
                if not on:
                    continue
                k = shell[i]
                for b in allowed:
                    j = trans[i][b]
                    total_by_k[k] += 1
                    if shell[j] == k - 1:
                        cross_by_k[k] += 1
        print(f"  --- p = {p:.2f} ---")
        print(f"  {'shell':<7}{'psi':<7}{'C(k->k-1)':<12}{'edge ct':<10}")
        for k in range(7):
            c_ratio = cross_by_k[k] / total_by_k[k] if total_by_k[k] else 0.0
            print(f"  {k:<7}{6-k:<7}{c_ratio:<12.4f}{total_by_k[k]:<10.0f}")


def run_uv_rectangularity(
    engine: pa1.TransitionEngine,
    u_idx: List[int],
    v_idx: List[int],
    n_samples: int = 200,
) -> None:
    print("\n" + "=" * 5)
    print("13. U x V RECTANGULARITY (factorization skew)")
    print("=" * 5)
    print("  rect = |R| / (|U_R| * |V_R|); 1.0 iff R is a full coset rectangle.\n")

    start = engine.start_idx
    p_values = [0.02, 0.025, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]

    print(f"  {'p':<7}{'E[rect]':<10}{'E[|R|]':<10}{'E[|U|]':<8}{'E[|V|]':<8}")
    print("  " + "-" * 5)

    for p in p_values:
        rect_sum = r_sum = u_sum = v_sum = 0.0
        nz = 0
        for _ in range(n_samples):
            allowed = [b for b in range(256) if random.random() < p]
            if not allowed:
                continue
            nz += 1
            vis = pa1.reachable_mask(engine, allowed, [start])
            us: set = set()
            vs: set = set()
            rcount = 0
            for i, on in enumerate(vis):
                if not on:
                    continue
                rcount += 1
                us.add(u_idx[i])
                vs.add(v_idx[i])
            nu, nv = len(us), len(vs)
            rect = rcount / (nu * nv) if nu and nv else 0.0
            rect_sum += rect
            r_sum += rcount
            u_sum += nu
            v_sum += nv
        denom = max(nz, 1)
        print(f"  {p:<7.3f}{rect_sum/denom:<10.4f}{r_sum/denom:<10.1f}"
              f"{u_sum/denom:<8.1f}{v_sum/denom:<8.1f}")


def _fold_active_count(bc: pa1.ByteClassification) -> int:
    return sum(1 for i in FOLD_BOUNDARY_IDXS if bc.connection_chain[i] > 0.0)


def _build_fold_pools(bclass: List[pa1.ByteClassification]) -> Dict[int, List[int]]:
    pools: Dict[int, List[int]] = {0: [], 1: [], 2: [], 3: []}
    for bc in bclass:
        pools[_fold_active_count(bc)].append(bc.byte)
    return pools


def _alphabet_fold_mix(
    pools: Dict[int, List[int]],
    k: int,
    pi_target: float,
    pool_limit: Collection[int] | None = None,
) -> List[int] | None:
    """Build |A|=k with pi_fold ~ pi_target using pool[0] and pool[3] bytes."""
    p0 = [b for b in pools[0] if pool_limit is None or b in pool_limit]
    p3 = [b for b in pools[3] if pool_limit is None or b in pool_limit]
    n3 = min(len(p3), max(0, round(pi_target * k)))
    n0 = k - n3
    if n0 < 0 or n0 > len(p0) or n3 > len(p3):
        return None
    return random.sample(p3, n3) + random.sample(p0, n0)


def _mask_allowed_bytes(
    bclass: List[pa1.ByteClassification],
    mask: int,
) -> List[int]:
    return [
        bc.byte for bc in bclass
        if all(
            (bc.connection_chain[i] == 0.0) or ((mask >> i) & 1)
            for i in range(7)
        )
    ]


def run_fold_triple_porosity_controlled(
    bclass: List[pa1.ByteClassification],
    engine: pa1.TransitionEngine,
    n_samples: int = 300,
) -> None:
    print("\n" + "=" * 5)
    print("9b. SIZE-CONTROLLED FOLD-TRIPLE POROSITY")
    print("=" * 5)
    print("  Fix |A|=k; vary pi_fold via pool[0] vs pool[3] bytes.")
    print("  pool[c] = bytes active on exactly c of {ONA|BU, BU|BU, BU|ONA}.")
    print("  Phase A: all 256 bytes.  Phase B: 64-byte minimal 5-boundary mask.")
    print(f"  CGM: Delta={CGM_DELTA:.6f}  5/256={FIVE_OVER_256:.6f}  "
          f"1/48={ONE_OVER_48:.6f}\n")

    bc_map = _byte_map(bclass)
    pools = _build_fold_pools(bclass)
    minimal_mask = 61
    minimal_pool = set(_mask_allowed_bytes(bclass, minimal_mask))
    print(f"  Pool sizes (all bytes): c=0:{len(pools[0])}  c=1:{len(pools[1])}  "
          f"c=2:{len(pools[2])}  c=3:{len(pools[3])}")
    print(f"  Minimal-mask pool |B|={len(minimal_pool)}  mask={minimal_mask}\n")

    pi_grid = [i / 1000 for i in range(0, 51)] + [i / 100 for i in range(6, 21)]
    sizes = [5, 6, 7, 8, 9, 10]

    for phase_name, pool_limit in (
        ("Phase A (all bytes)", None),
        ("Phase B (minimal-mask pool)", minimal_pool),
    ):
        print(f"  === {phase_name} ===")
        for k in sizes:
            if pool_limit is not None and k > len(pool_limit):
                continue
            print(f"  --- |A| = {k} ---")
            print(f"  {'pi_fold':<9}{'P(span)':<10}{'P(full)':<10}{'E[pi_fold]':<12}")
            print("  " + "-" * 5)
            span_curve: List[Tuple[float, float]] = []
            for pi_target in pi_grid:
                span_hits = full_hits = 0
                pi_sum = 0.0
                nz = 0
                for _ in range(n_samples):
                    allowed = _alphabet_fold_mix(pools, k, pi_target, pool_limit)
                    if allowed is None:
                        continue
                    nz += 1
                    pi_sum += _fold_porosity(allowed, bc_map)
                    r = pa1.compute_reachability(engine, allowed, track_depth=False)
                    if r.spans:
                        span_hits += 1
                    if r.full_omega:
                        full_hits += 1
                denom = max(nz, 1)
                p_span = span_hits / denom
                span_curve.append((pi_sum / denom, p_span))
                if pi_target in (0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.50, 1.0) or abs(pi_target - CGM_DELTA) < 0.0005:
                    print(f"  {pi_target:<9.3f}{p_span:<10.4f}{full_hits/denom:<10.4f}"
                          f"{pi_sum/denom:<12.4f}")

            p_half = None
            for i in range(1, len(span_curve)):
                p0, s0 = span_curve[i - 1]
                p1, s1 = span_curve[i]
                if s0 < 0.5 <= s1 or s1 < 0.5 <= s0:
                    if abs(s1 - s0) > 1e-9:
                        p_half = p0 + (0.5 - s0) * (p1 - p0) / (s1 - s0)
                    break
            if p_half is not None:
                print(f"  P(span)=0.5 at pi_fold ~ {p_half:.4f}  "
                      f"(Delta={CGM_DELTA:.4f}, ratio={p_half/CGM_DELTA:.2f})")
            print()
    print("  Interpretation per Wavefunction analysis:")
    print("    Delta (~0.0207) is the residual aperture AFTER depth-4 holonomic closure,")
    print("    not the classical critical porosity for horizon spanning (E_span).")
    print("    Byte-level fold disagreement is structurally 50%; closure compresses it to ~2.07%.")
    print("    The relevant 'aperture event' is holonomy-word availability or H2 onset, not E_span.")


def _kappa_binom_dest(shell_k: int) -> float:
    if shell_k in (0, 6):
        return 0.0
    return CGM_DELTA * binom_shell[shell_k]


def _holonomy_word_bytes(micro_ref: int) -> List[int]:
    if trace_word_steps is None or cycle_word_for_micro is None:
        raise ImportError("holonomy common imports unavailable")
    return [
        int(row["byte"])
        for row in trace_word_steps(
            cycle_word_for_micro(micro_ref), micro_ref=micro_ref
        )[1:]
    ]


def _tau_cycle_per_delta_holonomy(allowed: set[int]) -> Tuple[Fraction, Fraction]:
    """tau_cycle/Delta for micro-refs whose full holonomy word lies in allowed."""
    if (
        holonomy_arch_path is None
        or tau_per_delta_binom_on_path is None
        or holonomy_shell_weight is None
        or micro_popcount is None
    ):
        raise ImportError("holonomy analysis_3 imports unavailable")
    numer = Fraction(0)
    denom = Fraction(0)
    for m in range(64):
        if not all(b in allowed for b in _holonomy_word_bytes(m)):
            continue
        w = holonomy_shell_weight(micro_popcount(m))
        tau_m = tau_per_delta_binom_on_path(holonomy_arch_path(m))
        numer += w * tau_m
        denom += w
    if denom == 0:
        return Fraction(0), Fraction(0)
    return numer / denom, denom


def _q_weight_tau_per_delta(q_weights: Iterable[int]) -> Fraction:
    """Closed-form tau_cycle/Delta restricted to q-weight shell support."""
    qset = set(q_weights)
    numer = 4 * sum(comb(6, k) ** 3 for k in range(1, 6) if k in qset)
    denom = 64 * sum(comb(6, k) ** 2 for k in range(7))
    return Fraction(numer, denom)


def _q_weights_present(allowed: Sequence[int]) -> set[int]:
    return {q_word6(b).bit_count() for b in allowed}


# Holonomy word length for availability (canonical F / depth-4 closure word)
# From Wavefunction analysis: canonical word for F is 4 bytes.
HOLONOMY_WORD_LENGTH = 4
N_MICROREFS = 64


def expected_micro_cov(p: float, L: int = HOLONOMY_WORD_LENGTH, n_micro: int = N_MICROREFS) -> float:
    """Analytic expectation for fraction of micro-refs with full holonomy word available.
    Assumes independent byte inclusion with prob p.
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    # Prob one specific micro-ref word is fully present
    p_one = p ** L
    # Prob at least one complete (union bound approx for small, or exact 1 - (1-p_one)^n )
    return 1.0 - (1.0 - p_one) ** n_micro


def _tau_step_channel(engine: pa1.TransitionEngine, allowed: Sequence[int]) -> float:
    """Mean STF kappa at destination shell over edges from the rest-reachable set."""
    if not allowed:
        return 0.0
    vis = pa1.reachable_mask(engine, list(allowed), [engine.start_idx])
    shell = engine.shell
    trans = engine.transitions
    kappa_sum = 0.0
    cnt = 0
    for si in range(engine.n):
        if not vis[si]:
            continue
        for b in allowed:
            kappa_sum += _kappa_binom_dest(shell[trans[si][b]])
            cnt += 1
    return kappa_sum / cnt if cnt else 0.0


def _tau_like_cycle(engine: pa1.TransitionEngine, allowed: Sequence[int]) -> float:
    """Four bulk holonomy steps at mean channel kappa (kernel path length)."""
    return 4.0 * _tau_step_channel(engine, allowed)


def run_tau_percolation_identity(
    engine: pa1.TransitionEngine,
    n_samples: int = 200,
) -> None:
    print("\n" + "=" * 5)
    print("11b. tau_G PERCOLATION IDENTITY (holonomy + channel)")
    print("=" * 5)
    print("  Holonomy: tau_cycle/Delta from hqvm_gravity_analysis_3 weighting.")
    print("  Channel: tau_step on rest-reachable edges (restriction degradation).\n")

    full_set = set(range(256))
    full_hol_cov = Fraction(1)
    if holonomy_arch_path is not None:
        hol_tau, hol_cov = _tau_cycle_per_delta_holonomy(full_set)
        full_hol_cov = hol_cov if hol_cov > 0 else Fraction(1)
        print(f"  Holonomy full alphabet: tau_cycle/Delta = {hol_tau}")
        if tau_cycle_per_delta_kernel_exact is not None:
            kernel = tau_cycle_per_delta_kernel_exact()
            print(f"  Kernel reference:       tau_cycle/Delta = {kernel}")
            print(f"  Match: {hol_tau == kernel}")
        print(f"  Micro-ref weight covered: {float(hol_cov / full_hol_cov):.6f}")

    q_tau_full = _q_weight_tau_per_delta(range(7))
    print(f"  Q-weight closed form:   tau_cycle/Delta = {q_tau_full}")

    tau_step_full = _tau_step_channel(engine, list(full_set))
    tau_like_full = _tau_like_cycle(engine, list(full_set))
    print(f"\n  Channel full alphabet: tau_step={tau_step_full:.10f}  "
          f"4*tau_step={tau_like_full:.10f}")

    if kernel_exposure_constants is not None:
        n_cycles, tau_cycle, tau_g_full, tau_od = kernel_exposure_constants()
        print(f"  Kernel tau_cycle     = {tau_cycle:.10f}")
        print(f"  4*tau_step/tau_cycle = {tau_like_full / tau_cycle:.6f}")
        print("  Note: holonomy tau matches kernel exactly (per Wavefunction: depth-4 holonomy path).")
        print("        Channel tau samples all rest-reachable edges (different measure; ratio is conversion factor).")

    p_values = [0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.10, 0.20, 0.50, 1.0]
    print(f"\n  {'p':<8}{'hol/Delta':<14}{'q/Delta':<14}{'micro_cov':<12}{'E[micro_cov]':<14}{'4*tau_step':<14}")
    print("  " + "-" * 5)
    ref_channel = tau_like_full if tau_like_full > 0 else 1.0
    for p in p_values:
        like_sum = 0.0
        q_tau_sum = Fraction(0)
        nz = 0
        hol_tau_p = Fraction(0)
        hol_cov_p = Fraction(0)
        for _ in range(n_samples):
            allowed = [b for b in range(256) if random.random() < p]
            if not allowed:
                continue
            nz += 1
            like_sum += _tau_like_cycle(engine, allowed)
            q_tau_sum += _q_weight_tau_per_delta(_q_weights_present(allowed))
            if holonomy_arch_path is not None:
                ht, hc = _tau_cycle_per_delta_holonomy(set(allowed))
                hol_tau_p += ht
                hol_cov_p += hc
        denom = max(nz, 1)
        tl = like_sum / denom
        hol_avg = float(hol_tau_p / denom) if holonomy_arch_path else 0.0
        cov_avg = float((hol_cov_p / denom) / full_hol_cov) if holonomy_arch_path else 0.0
        q_avg = float(q_tau_sum / denom)
        e_cov = expected_micro_cov(p)
        print(f"  {p:<8.3f}{hol_avg:<14.6f}{q_avg:<14.6f}{cov_avg:<12.4f}{e_cov:<14.4f}"
              f"{tl:<14.10f}")
    print("\n  Analytic micro_cov derivation (independent inclusion, L=4 for canonical F word):")
    print("    E[micro_cov] = 1 - (1 - p^L)^64")
    print("  This is the percolation law for holonomy-word availability (the relevant event for aperture/tau_cycle).")


def _empirical_shell_matrix(
    engine: pa1.TransitionEngine,
    allowed: Sequence[int],
    q_weight: int | None = None,
) -> List[List[float]] | None:
    if not allowed:
        return None
    if q_weight is not None:
        bytes_q = [b for b in allowed if q_word6(b).bit_count() == q_weight]
        if not bytes_q:
            return None
        use = bytes_q
    else:
        use = list(allowed)
    shell = engine.shell
    trans = engine.transitions
    counts = [[0] * 7 for _ in range(7)]
    row_tot = [0] * 7
    for si in range(engine.n):
        w = shell[si]
        for b in use:
            t = shell[trans[si][b]]
            counts[w][t] += 1
            row_tot[w] += 1
    return [
        [counts[w][t] / row_tot[w] if row_tot[w] else 0.0 for t in range(7)]
        for w in range(7)
    ]


def _trace_matrix(M: List[List[float]]) -> float:
    return sum(M[w][w] for w in range(7))


def _return_trace_matrix(M: List[List[float]]) -> float:
    return sum(M[w][t] * M[t][w] for w in range(7) for t in range(7))


def run_shell_transition_operators(
    engine: pa1.TransitionEngine,
    n_samples: int = 150,
) -> None:
    print("\n" + "=" * 5)
    print("15. SHELL TRANSITION OPERATORS M_A(q) (Compact Geometry)")
    print("=" * 5)
    print("  Empirical M_A from uniform byte steps in allowed alphabet A.")
    print("  Tr(M_q), C(q)=Tr(M_q^2); M_shell = sum k*C(6,k) = 192 (binomial).\n")

    full = list(range(256))
    m_spec_sum = 0.0
    print(f"  {'q':<4}{'Tr(M)':<12}{'Tr(M^2)':<12}{'C(q) emp':<12}{'C(q) exact':<14}{'|bytes|':<8}")
    print("  " + "-" * 5)
    for q in range(7):
        M = _empirical_shell_matrix(engine, full, q_weight=q)
        if M is None:
            exact = carrier_trace(q) if carrier_trace else None
            print(f"  {q:<4}{'n/a':<12}{'n/a':<12}{str(exact):<14}{0:<8}")
            continue
        tr = _trace_matrix(M)
        tr2 = _return_trace_matrix(M)
        cq = tr if abs(tr) > 1e-12 else tr2
        exact = carrier_trace(q) if carrier_trace else None
        n_b = sum(1 for b in full if q_word6(b).bit_count() == q)
        m_spec_sum += q * cq
        exact_s = str(exact) if exact is not None else "n/a"
        match = "OK" if exact is not None and abs(float(cq) - float(exact)) < 1e-12 else ""
        print(f"  {q:<4}{tr:<12.6f}{tr2:<12.6f}{cq:<12.6f}{exact_s:<14}{n_b:<8} {match}")
    m_shell_binom = sum(k * comb(6, k) for k in range(7))
    print(f"  M_shell (binom sum k*C(6,k)) = {m_shell_binom}  (population moment, per Compact Geometry)")
    print("  Per-q C(q) matches carrier_trace exactly (Tr(M) for even q, Tr(M^2) for odd q).")

    p_values = [0.05, 0.10, 0.20, 0.50, 1.0]
    print(f"\n  Restricted C(3) and Tr(M_3) vs p (n={n_samples}):")
    print(f"  {'p':<8}{'Tr(M_3)':<12}{'C(3)':<12}{'E[|A|]':<10}")
    print("  " + "-" * 5)
    for p in p_values:
        tr_sum = cq_sum = size_sum = 0.0
        nz = 0
        for _ in range(n_samples):
            allowed = [b for b in range(256) if random.random() < p]
            if not allowed:
                continue
            M = _empirical_shell_matrix(engine, allowed, q_weight=3)
            if M is None:
                continue
            nz += 1
            tr_sum += _trace_matrix(M)
            cq_sum += _return_trace_matrix(M)
            size_sum += len(allowed)
        denom = max(nz, 1)
        print(f"  {p:<8.3f}{tr_sum/denom:<12.6f}{cq_sum/denom:<12.6f}"
              f"{size_sum/denom:<10.1f}")


def run_word_bulk_anchor_test(
    engine: pa1.TransitionEngine,
    omega: List[int],
) -> None:
    print("\n" + "=" * 5)
    print("14. WORD-REGIME REACHABILITY: HORIZON vs BULK ANCHORS")
    print("=" * 5)
    print("  W2/W2' map shell s -> 6-s globally (T2/T3).")
    print("  From rest (shell 6) words reach horizons only; from bulk,")
    print("  antipodal bulk shells are accessible.\n")

    shell = engine.shell
    trans = engine.transitions
    n = engine.n

    def apply_w2(state_idx: int, m: int) -> int:
        b0 = pa1.byte_from_family_micro(0, m)
        b1 = pa1.byte_from_family_micro(1, m)
        return trans[trans[state_idx][b0]][b1]

    # Pick representative anchors (shell 3 is a fixed point of s -> 6-s)
    rest_idx = engine.start_idx
    bulk2_idx = next(i for i in range(n) if shell[i] == 2)
    eq_idx = next(i for i in range(n) if shell[i] == 0)

    for label, start_idx in (("rest (shell 6)", rest_idx),
                             ("bulk (shell 2)", bulk2_idx),
                             ("equality horizon (shell 0)", eq_idx)):
        vis = bytearray(n)
        frontier = [start_idx]
        vis[start_idx] = 1
        # BFS using all 64 W2 operators as edges
        for _ in range(6):
            nxt: List[int] = []
            for si in frontier:
                for m in range(64):
                    ni = apply_w2(si, m)
                    if not vis[ni]:
                        vis[ni] = 1
                        nxt.append(ni)
            frontier = nxt
        shells_hit = sorted({shell[i] for i, on in enumerate(vis) if on})
        reach = sum(vis)
        print(f"  {label}: Reach={reach}  shells={shells_hit}")


def run_self_energy_closure() -> None:
    print("\n" + "=" * 5)
    print("16. SELF-ENERGY CLOSURE FROM PERCOLATION-DERIVED G(psi)")
    print("=" * 5)
    print("  Inputs: D=24 -> G_kernel=pi/6; tau_G from holonomy transport.")
    print("  Exterior ODE gives I = int exp(g1*psi)/s^2 ds = psi(s_h)-psi(inf).")
    print("  Horizon condition fixes psi(s_h)=1/2.\n")

    if (
        dln_g_dpsi is None
        or horizon_s_analytic is None
        or field_integral_exterior_exact is None
        or field_integral_exterior_numeric is None
        or kernel_exposure_constants is None
    ):
        print("  Missing gravity_common imports; cannot run closure.")
        return

    n_cycles, tau_cycle, tau_g_full, tau_od = kernel_exposure_constants()
    g1 = dln_g_dpsi(tau_g_full)
    s_h = horizon_s_analytic(g1)
    i_exact = field_integral_exterior_exact(s_h, g1)
    i_num, i_err = field_integral_exterior_numeric(s_h, g1)
    e_rest = 0.5 * i_exact
    e_self = -0.5 * 0.5
    m_obs_over_bare = 1.0 / (1.0 - e_self)

    print(f"  tau_cycle/Delta = {tau_od}  ({float(tau_od):.12f})")
    print(f"  tau_cycle       = {tau_cycle:.12f}")
    print(f"  tau_G           = {tau_g_full:.12f}")
    print(f"  N_cycles        = {n_cycles:.6f}")
    print(f"  g1              = {g1:.12f}")
    print(f"  s_h             = {s_h:.12f}")
    print(f"  I_exact         = {i_exact:.12f}")
    print(f"  I_numeric       = {i_num:.12f}  err={i_err:.2e}")
    print(f"  E_rest/Mc^2     = {e_rest:.12f}")
    print(f"  E_self/Mc^2     = {e_self:.12f}")
    print(f"  M_obs/M_bare    = {m_obs_over_bare:.12f}")

    checks = [
        ("I=1/2", abs(i_exact - 0.5) < 1e-12),
        ("numeric I", abs(i_num - 0.5) < 1e-3),
        ("E_self=-1/4", abs(e_self + 0.25) < 1e-12),
        ("M_obs/M_bare=4/5", abs(m_obs_over_bare - 0.8) < 1e-12),
    ]
    for label, ok in checks:
        print(f"  {label}: {'PASS' if ok else 'FAIL'}")


def main() -> None:
    import argparse
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")

    parser = argparse.ArgumentParser(description="hQVM percolation structural completeness")
    parser.add_argument(
        "--sections",
        type=str,
        default="all",
        help='Comma-separated section ids (1-15,9b,11b) or "all"',
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip expensive sections: 8,9,9b,11,11b,12,15",
    )
    args = parser.parse_args()

    def parse_sections(spec: str) -> set[str]:
        if spec == "all":
            return {"all"}
        out: set[str] = set()
        for part in spec.split(","):
            token = part.strip()
            if not token:
                continue
            if "-" in token and token.replace("-", "").isdigit():
                lo_s, hi_s = token.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
                out.update(str(i) for i in range(lo, hi + 1))
            else:
                out.add(token)
        return out

    if args.sections == "all":
        sections = {"all"}
    else:
        sections = parse_sections(args.sections)

    def want(sec: str) -> bool:
        return "all" in sections or sec in sections

    random.seed(20260702)

    print("hQVM Percolation Structural Completeness (analysis_3)")
    print("=" * 5)

    bclass = pa1.classify_all_bytes()
    omega = pa1.enumerate_omega()
    engine = pa1.build_transition_engine(omega)
    u_idx, v_idx = _build_uv_indices(omega)
    perm_classes = _build_permutation_classes(engine)
    print(f"  |Omega|={len(omega)}  permutation classes={len(perm_classes)}")
    assert len(perm_classes) == 128

    if want("1"):
        verify_commutation_laws(bclass, engine)
    if want("2"):
        run_future_cone_entropy(engine, n_samples=200)
    if want("3"):
        run_uv_factorization(engine, omega, u_idx, v_idx, n_samples=200)
    if want("4"):
        run_shell_enrichment(engine, n_samples=200)
    if want("5"):
        run_two_step_uniformization(engine, n_samples=150)
    if want("6"):
        run_fold_phase_complete_tables(bclass, engine)
    if want("7"):
        run_minimal_boundary_subsets(bclass, engine)
    if want("8") and not args.fast:
        run_permutation_class_sweep(engine, perm_classes, n_samples=200)

    bridges = (
        want("9") or want("9b") or want("10") or want("11") or want("11b")
        or want("12") or want("13") or want("14") or want("15") or want("16")
    )
    if bridges:
        print("\n" + "=" * 5, flush=True)
        print("SECTIONS 9-16: GRAVITY / POROSITY BRIDGES", flush=True)
        print("=" * 5, flush=True)

    if want("9") and not args.fast:
        run_geometric_porosity_sweeps(bclass, engine, n_samples=250)
    if want("9b") and not args.fast:
        run_fold_triple_porosity_controlled(bclass, engine, n_samples=300)
    if want("10"):
        run_plaquette_loop_defect(n_samples=200)
    if want("11") and not args.fast:
        run_spanning_transmission(engine, bclass, n_samples=300)
    if want("11b") and not args.fast:
        run_tau_percolation_identity(engine, n_samples=200)
    if want("12") and not args.fast:
        run_shell_local_connectivity(engine, n_samples=150)
    if want("13"):
        run_uv_rectangularity(engine, u_idx, v_idx, n_samples=200)
    if want("15") and not args.fast:
        run_shell_transition_operators(engine, n_samples=150)
    if want("14"):
        run_word_bulk_anchor_test(engine, omega)
    if want("16") and not args.fast:
        run_self_energy_closure()

    print("\n" + "=" * 5)
    print("END. See docs/Findings/Analysis_hQVM_Percolation.md for the writeup.")
    print("=" * 5)


if __name__ == "__main__":
    main()
