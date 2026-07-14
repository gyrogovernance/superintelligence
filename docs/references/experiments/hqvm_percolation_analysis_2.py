#!/usr/bin/env python3
"""
hQVM Canonical-Word Percolation: the CGM-native percolation theory
==================================================================

Companion to hqvm_percolation_analysis_1.py (byte baseline) and
hqvm_percolation_analysis_3.py (structural completeness).
Run all three via hqvm_percolation_analysis_run.py (runner).

WHY A SECOND SCRIPT
-------------------
analysis_1 showed that *byte-level* reachability on Omega is almost trivially
percolative: any single family (64 bytes), any single K4 gate class (64
bytes), even the identity-gate class alone, reaches all 4096 states. That is
not a deep CGM statement -- it is a short-circuit artifact. Byte-level edges
do NOT enforce the CGM causal cycle (CS -> UNA -> ONA -> BU depth-4 closure).
Legal byte sequences can hop around Omega in ways that are meaningless as
constitutional closure paths, so the weak "span" event (hit any equality-
horizon state from rest) is triggered by a single q6=63 byte in one step.

The CGM-native percolation object is the CANONICAL WORD OPERATOR, not the
byte. Analysis_hQVM_Wavefunction.md Theorems T1-T10 establish:

    W2(m)  = [byte(fam00,m), byte(fam01,m)]     depth-4 BU-Egress
    W2'(m) = [byte(fam10,m), byte(fam11,m)]     depth-4 BU-Ingress
    F(m)   = W2(m) o W2'(m)                      full 4-byte canonical word

with:
    T2/T3  W2, W2' : shell s -> 6-s            (pole swap, chi XOR 63)
    T4     F       : shell s -> s              (Z2 carrier flip within pole)
    T6     W2^2 = W2'^2 = F^2 = id             (involutions)
    T8     Egress  = W2 involution (Box B spectral closure)
    T9     Ingress = W2 pole-pairing (shadow = memory, 64<->64 invertible)
    T10    q(W2) = q(W2') = 63 ; q(F) = 0 for every micro_ref m

Every edge in this script IS a full depth-4 closure act. The percolation
dial is the set M of allowed micro_refs (the CGM-native 6-bit payload,
Gyroscopic_ASI_Specs_Formalism.md Sec 2.2/4 -- "payload bits define
transformation content; family bits select gauge phase"). Each micro_ref m
brings its W2(m) AND W2'(m) together, so the K4 gauge sector is always
complete per selected payload and we isolate payload-content percolation
under the depth-4 causal constraint.

PERCOLATION EVENTS (all reported, never conflated)
--------------------------------------------------
  E1  hit_equality_word : from rest, a word-path reaches any equality-
                          horizon state. NON-trivial here (no q6=63 single
                          byte short-circuit; the shortest word is W2, a
                          full depth-4 egress).
  E2  full_pairing      : from ALL 64 complement-horizon states, the word-
                          graph reaches ALL 64 equality-horizon states
                          (T9 ingress memory reconstruction, set-to-set).
  E3  full_omega        : word-graph transitivity on Omega (strongest).
  E4  curvature_complete: the allowed byte set realizes all 7 transport-
                          defect weights k=0..6 (popcount(q6(bi) XOR q6(bj)));
                          the kernel/Regge curvature-spectrum completion event.

SECTIONS (analysis_2)
---------------------
  1-2   Word engine + K4 algebra verification
  3     Single-pole reachability (W2 / F / W2+W2')
  4     Canonical-word micro_ref percolation (W2+W2')
  5     Transport-curvature spectrum (byte pairwise q6 defects)
  6     Null model (shuffled byte dynamics)
  7     Sector/shell confinement profile (bulk never reached)
  8     Edge-mode percolation (F-only vs W2+W2' vs all)
  9     Probe-word reachability (same-fam, F^2, reverse, constitutional)
  10    Helix Z2 cycle (F^2 = id from rest)
  11    Flat vs curved payload pools (fiber bundle dial)
  12    Constitutional byte shell trace (depth-4 and depth-8)

Run the full study: python experiments/hqvm_percolation_analysis_run.py

OUTPUTS
-------
Threshold tables, reachability counts, K4 algebra verification, null-model
curves with binomial CI. Interpretation lives in
docs/Findings/Analysis_hQVM_Percolation.md.
"""

from __future__ import annotations

import math
import sys
import random
from collections import Counter
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from gyroscopic.hQVM.constants import (
    GENE_MAC_REST,
    LAYER_MASK_12,
    byte_to_intron,
    byte_micro_ref,
    intron_family,
    is_on_equality_horizon,
    is_on_horizon,
    step_state_by_byte,
)
from gyroscopic.hQVM.api import chirality_word6, q_word6

from hqvm_wavefunction_kernel import decompose_byte

N_OMEGA = 4096
GIANT_THRESHOLD = N_OMEGA // 2
FAMILY_RAY_REF = 1  # reference micro_ref (wavefunction helix evolution)

# 1. Build Omega, the byte transition table, and the canonical word
#    permutations W2[m], W2'[m], F[m] (m = 0..63).

def enumerate_omega() -> List[int]:
    code: set = set()
    for m in range(64):
        v = 0
        for j in range(6):
            if (m >> j) & 1:
                v |= 0b11 << (2 * j)
        code.add(v & LAYER_MASK_12)
    cs = sorted(code)
    out: List[int] = []
    for u in cs:
        a12 = 0xAAA ^ u
        for v in cs:
            b12 = 0x555 ^ v
            out.append(((a12 & LAYER_MASK_12) << 12) | (b12 & LAYER_MASK_12))
    return out

def byte_from_family_micro(family: int, micro_ref: int) -> int:
    """Unique byte carrying (family, micro_ref). QuBEC decomposition
    (Gyroscopic_ASI_Specs_Formalism.md Sec 4)."""
    bit7 = (family >> 1) & 1
    bit0 = family & 1
    intron = (bit7 << 7) | ((micro_ref & 0x3F) << 1) | bit0
    return intron ^ 0xAA

@dataclass(frozen=True)
class WordEngine:
    """Canonical word permutations on Omega (index space)."""
    n: int
    state_to_idx: Dict[int, int]
    shell: List[int]
    start_idx: int
    comp_idx: List[int]      # complement-horizon state indices (shell 6)
    eq_idx: List[int]        # equality-horizon state indices (shell 0)
    w2: List[List[int]]      # w2[m][i]  = W2(m)(state i)
    w2p: List[List[int]]     # w2p[m][i] = W2'(m)(state i)
    fp: List[List[int]]      # fp[m][i]  = F(m)(state i)
    q6_of_byte: List[int]    # q_word6(b) for b in 0..255
    byte_trans: List[List[int]]  # 4096x256 byte transition table (for the null model)

def build_word_engine(omega: List[int]) -> WordEngine:
    print("  Building byte transition table (4096 x 256)...", flush=True)
    n = len(omega)
    state_to_idx = {s: i for i, s in enumerate(omega)}
    trans = [[0] * 256 for _ in range(n)]
    shell = [0] * n
    for i, s in enumerate(omega):
        shell[i] = chirality_word6(s).bit_count()
        row = trans[i]
        for b in range(256):
            row[b] = state_to_idx[step_state_by_byte(s, b)]

    print("  Composing canonical word permutations W2, W2', F (64 x 4096)...", flush=True)
    w2: List[List[int]] = []
    w2p: List[List[int]] = []
    fp: List[List[int]] = []
    for m in range(64):
        b00 = byte_from_family_micro(0, m)
        b01 = byte_from_family_micro(1, m)
        b10 = byte_from_family_micro(2, m)
        b11 = byte_from_family_micro(3, m)
        w2m = [0] * n
        w2pm = [0] * n
        fm = [0] * n
        for i in range(n):
            j = trans[i][b00]
            w2m[i] = trans[j][b01]              # W2(m) = T_b01 o T_b00
            k = trans[i][b10]
            w2pm[i] = trans[k][b11]             # W2'(m) = T_b11 o T_b10
            fm[i] = trans[w2m[i]][b10]
            fm[i] = trans[fm[i]][b11]           # F(m) = T_b11 o T_b10 o W2(m)
        w2.append(w2m)
        w2p.append(w2pm)
        fp.append(fm)

    comp_idx = [i for i, s in enumerate(omega) if is_on_horizon(s) and not is_on_equality_horizon(s)]
    eq_idx = [i for i, s in enumerate(omega) if is_on_equality_horizon(s)]
    q6_of_byte = [q_word6(b) for b in range(256)]
    print(f"  |comp_horizon|={len(comp_idx)}  |eq_horizon|={len(eq_idx)}  "
          f"(expect 64/64, holographic 64^2=4096)", flush=True)
    return WordEngine(n, state_to_idx, shell, state_to_idx[GENE_MAC_REST],
                      comp_idx, eq_idx, w2, w2p, fp, q6_of_byte, trans)

# 2. Verify the K4 word algebra against THIS engine (ground the study).

def verify_word_algebra(eng: WordEngine) -> None:
    print("\n" + "=" * 5)
    print("2. K4 WORD-ALGEBRA CROSS-CHECK (all 64 micro_refs x 4096 states)")
    print("=" * 5)
    n = eng.n
    shell = eng.shell
    t2 = t3 = t4 = t6_w2 = t6_w2p = t6_f = t9 = t10_w2 = t10_w2p = t10_f = True
    q6_w2_set, q6_w2p_set, q6_f_set = set(), set(), set()
    for m in range(64):
        w2m, w2pm, fm = eng.w2[m], eng.w2p[m], eng.fp[m]
        for i in range(n):
            if shell[w2m[i]] != 6 - shell[i]:
                t2 = False
            if shell[w2pm[i]] != 6 - shell[i]:
                t3 = False
            if shell[fm[i]] != shell[i]:
                t4 = False
            if w2m[w2m[i]] != i:
                t6_w2 = False
            if w2pm[w2pm[i]] != i:
                t6_w2p = False
            if fm[fm[i]] != i:
                t6_f = False
        # T9: W2 pairs complement-horizon bijectively onto equality-horizon
        eq_targets = set(w2m[i] for i in eng.comp_idx)
        if eq_targets != set(eng.eq_idx):
            t9 = False
    print(f"  T2  W2  : shell s -> 6-s         : {'VERIFIED' if t2 else 'FAILED'}")
    print(f"  T3  W2' : shell s -> 6-s         : {'VERIFIED' if t3 else 'FAILED'}")
    print(f"  T4  F   : shell s -> s           : {'VERIFIED' if t4 else 'FAILED'}")
    print(f"  T6  W2^2 = W2'^2 = F^2 = id      : "
          f"{'VERIFIED' if (t6_w2 and t6_w2p and t6_f) else 'FAILED'}")
    print(f"  T9  W2 : comp-horizon <-> eq-horizon bijection : {'VERIFIED' if t9 else 'FAILED'}")
    print("  (T10 q-transport is byte-level; re-checked via q_word6 below.)")

# 3. Single-pole non-transitivity theorem (exact, deterministic).

def _bfs_reach(eng: WordEngine, perms_list: List[List[int]],
               starts: List[int], max_depth: int = 16) -> bytearray:
    n = eng.n
    visited = bytearray(n)
    frontier = []
    for s in starts:
        if not visited[s]:
            visited[s] = 1
            frontier.append(s)
    for _ in range(max_depth):
        if not frontier:
            break
        nxt = []
        for s in frontier:
            for perm in perms_list:
                t = perm[s]
                if not visited[t]:
                    visited[t] = 1
                    nxt.append(t)
        frontier = nxt
    return visited

def single_pole_theorem(eng: WordEngine) -> None:
    print("\n" + "=" * 5)
    print("3. SINGLE-POLE REACHABILITY (word graph)")
    print("=" * 5)
    n = eng.n

    # W2-only with ALL 64 micro_refs
    vis_w2 = _bfs_reach(eng, eng.w2, [eng.start_idx])
    # F-only with ALL 64 micro_refs
    vis_f = _bfs_reach(eng, eng.fp, [eng.start_idx])
    # W2 + W2' with ALL 64 micro_refs (both poles)
    both = eng.w2 + eng.w2p
    vis_both = _bfs_reach(eng, both, [eng.start_idx])
    # W2 + W2' + F
    all3 = eng.w2 + eng.w2p + eng.fp
    vis_all = _bfs_reach(eng, all3, [eng.start_idx])

    print(f"\n  With ALL 64 micro_refs available (full payload, gauge-complete):")
    print(f"    W2-only  reachable from rest : {sum(vis_w2):5d} / {n}")
    print(f"    F-only   reachable from rest : {sum(vis_f):5d} / {n}")
    print(f"    W2+W2'   reachable from rest : {sum(vis_both):5d} / {n}")
    print(f"    W2+W2'+F reachable from rest : {sum(vis_all):5d} / {n}")

    # One-hop images from rest; two-hop closure reaches 128 under W2-only
    w2_one_hop = {eng.w2[m][eng.start_idx] for m in range(64)}
    w2_two_hop = w2_one_hop | {eng.w2[m][j] for m in range(64) for j in w2_one_hop}
    f_one_hop = {eng.fp[m][eng.start_idx] for m in range(64)}
    print(f"\n  W2-only one-hop from rest  : |{{W2(m)(rest)}}| = {len(w2_one_hop)}")
    print(f"  W2-only two-hop closure    : {len(w2_two_hop)}  (BFS reach {sum(vis_w2)})")
    print(f"  F-only one-hop from rest   : |{{F(m)(rest)}}| = {len(f_one_hop)}  "
          f"(BFS reach {sum(vis_f)})")

# 4. Canonical-word percolation sweep (the CGM percolation threshold).

def _perms_for_mode(eng: WordEngine, M: List[int], mode: str) -> List[List[int]]:
    """Collect word permutations for micro_refs in M under edge mode."""
    perms: List[List[int]] = []
    for m in M:
        if mode in ("w2", "w2w2p", "all"):
            perms.append(eng.w2[m])
        if mode in ("w2p", "w2w2p", "all"):
            perms.append(eng.w2p[m])
        if mode in ("f", "all"):
            perms.append(eng.fp[m])
    return perms


def _word_reach_generic(
    eng: WordEngine,
    perms: List[List[int]],
    starts: List[int],
) -> Tuple[bytearray, int, bool, bool, bool]:
    """BFS on arbitrary word permutations. Returns (visited, reach, hit_eq, pairing, full)."""
    vis = _bfs_reach(eng, perms, starts)
    reach = sum(vis)
    hit_eq = any(vis[i] for i in eng.eq_idx)
    pairing = all(vis[i] for i in eng.eq_idx)
    return vis, reach, hit_eq, pairing, reach == eng.n


def _word_reach_counts(
    eng: WordEngine,
    M: List[int],
    starts: List[int],
    mode: str = "w2w2p",
) -> Tuple[int, bool, bool, bool]:
    """BFS on word graph. Returns (reach, hit_eq, full_pairing, full_omega)."""
    perms = _perms_for_mode(eng, M, mode)
    _, reach, hit_eq, pairing, full = _word_reach_generic(eng, perms, starts)
    return reach, hit_eq, pairing, full


def _word_sweep_crossing(rows: List[Tuple], col: int) -> Optional[float]:
    for i in range(len(rows) - 1):
        y1, y2 = rows[i][col], rows[i + 1][col]
        if y1 < 0.5 <= y2:
            frac = (0.5 - y1) / (y2 - y1) if y2 > y1 else 0.5
            return rows[i][0] + frac * (rows[i + 1][0] - rows[i][0])
    return None


def _print_word_sweep_table(
    rows: List[Tuple[float, float, float, float, float, float, float]],
) -> None:
    """Compact table for word percolation sweep rows."""
    print(f"  {'p':<7}{'E[#m]':<7}{'P(pair)':<9}{'P(full)':<9}{'CI95':<10}{'<Reach>':<9}")
    print("  " + "-" * 5)
    keep = {0, len(rows) - 1}
    for i, row in enumerate(rows):
        p = row[0]
        if p <= 0.25 and abs(round(p * 100) - p * 100) < 1e-6:
            keep.add(i)
        elif p > 0.25:
            keep.add(i)
    p_c = _word_sweep_crossing(rows, 4)
    if p_c is not None:
        for i, row in enumerate(rows):
            if abs(row[0] - p_c) <= 0.025:
                keep.add(i)
    prev = -2
    for i in sorted(keep):
        if i - prev > 1:
            print("  ...")
        p, e_m, _, pp, pf, ci, ar = rows[i]
        print(f"  {p:<7.3f}{e_m:<7.2f}{pp:<9.4f}{pf:<9.4f}{ci:<10.4f}{ar:<9.1f}")
        prev = i
    if p_c is not None:
        print(f"  p_c(P(full)) ~ {p_c:.4f}  E[#m] ~ {64 * p_c:.2f}")


def word_percolation_sweep(eng: WordEngine, n_samples: int = 200) -> None:
    print("\n" + "=" * 5)
    print("4. CANONICAL-WORD PERCOLATION SWEEP (W2+W2', micro_ref dial)")
    print("=" * 5)
    print("  Include each of the 64 micro_refs w.p. p; each selected m brings")
    print("  BOTH W2(m) (Egress) and W2'(m) (Ingress) -- gauge-complete per m.")
    print("  Events: E1 hit_eq_word, E2 full_pairing (T9), E3 full_omega.")
    print("  Conditional on |M|>0.\n")

    p_values = [0.0] + [i / 100 for i in range(1, 31)] + [0.4, 0.5, 0.7, 1.0]
    starts = [eng.start_idx]
    comp_starts = eng.comp_idx
    z = 1.96

    rows: List[Tuple[float, float, float, float, float, float, float]] = []
    for pi, p in enumerate(p_values):
        if pi % 10 == 0:
            print(f"  progress: word sweep {pi + 1}/{len(p_values)} (p={p:.3f})", flush=True)
        hit = pair = full = 0
        reach_sum = 0
        nonzero = 0
        for _ in range(n_samples):
            M = [m for m in range(64) if random.random() < p]
            if not M:
                continue
            nonzero += 1
            reach, h, pr, fo = _word_reach_counts(eng, M, starts)
            reach_sum += reach
            if h:
                hit += 1
            if pr:
                pair += 1
            if fo:
                full += 1
        denom = max(nonzero, 1)
        ph = hit / denom
        pp = pair / denom
        pf = full / denom
        ar = reach_sum / denom
        se = math.sqrt(pf * (1 - pf) / denom) if denom > 1 else 0.0
        e_m = 64 * p
        rows.append((p, e_m, ph, pp, pf, z * se, ar))

    _print_word_sweep_table(rows)

    print("\n  Thresholds (conditional on |M|>0):")
    for name, col in (("E1 hit_eq_word", 2), ("E2 full_pairing", 3), ("E3 full_omega", 4)):
        p_c = _word_sweep_crossing(rows, col)
        if p_c is not None:
            print(f"    {name:<18}: p_c = {p_c:.4f}  E[#m] = {64 * p_c:.2f} of 64")
        else:
            print(f"    {name:<18}: no crossing in range")

    # Minimum micro_refs for full_omega (exact search by random subsets)
    print("\n  Minimum payload for word-transitivity (random subset search):")
    found_k = None
    for k in range(1, 17):
        successes = 0
        trials = 60
        for _ in range(trials):
            M = random.sample(range(64), k)
            _, _, _, fo = _word_reach_counts(eng, M, comp_starts)
            if fo:
                successes += 1
        if successes > 0:
            found_k = k
            print(f"    k={k:2d} micro_refs : {successes}/{trials} subsets achieve full_omega")
            if successes >= trials * 0.5:
                break
    if found_k is not None:
        print(f"    first k with any success: {found_k}")
    else:
        print("    no subset of <=16 micro_refs achieves full_omega in search")

# 5. Transport-curvature spectrum completion percolation.

def curvature_spectrum_sweep(eng: WordEngine, n_samples: int = 200) -> None:
    print("\n" + "=" * 5)
    print("5. TRANSPORT-CURVATURE SPECTRUM COMPLETION")
    print("=" * 5)
    print("  Transport defect between bytes bi, bj : d = popcount(q6(bi) XOR q6(bj))")
    print("  in {0..6}. The full curvature spectrum (all 7 weights) is the kernel/")
    print("  Regge completeness condition. Percolation event E4 = all k=0..6 present")
    print("  among pairwise defects of the allowed byte set.\n")

    q6 = eng.q6_of_byte
    p_values = [0.0] + [i / 100 for i in range(1, 21)] + [0.3, 0.5, 1.0]
    print(f"  {'p':<7}{'E[#bytes]':<10}{'P(spectrum complete)':<22}{'<#weights>':<10}")
    print("  " + "-" * 5)
    rows = []
    for p in p_values:
        complete = 0
        wsum = 0
        nonzero = 0
        for _ in range(n_samples):
            B = [b for b in range(256) if random.random() < p]
            if len(B) < 2:
                wsum += (1 if B else 0)
                continue
            nonzero += 1
            weights = set()
            for i in range(len(B)):
                qi = q6[B[i]]
                for j in range(i + 1, len(B)):
                    weights.add((qi ^ q6[B[j]]).bit_count())
            wsum += len(weights)
            if len(weights) == 7:
                complete += 1
        denom = max(nonzero, 1)
        pc = complete / denom
        rows.append((p, 256 * p, pc))
        print(f"  {p:<7.3f}{256*p:<10.1f}{pc:<22.4f}{wsum/max(n_samples,1):<10.2f}")

    # threshold
    for i in range(len(rows) - 1):
        if rows[i][2] < 0.5 <= rows[i + 1][2]:
            frac = (0.5 - rows[i][2]) / (rows[i + 1][2] - rows[i][2]) if rows[i + 1][2] > rows[i][2] else 0.5
            p_c = rows[i][0] + frac * (rows[i + 1][0] - rows[i][0])
            print(f"\n  E4 curvature-complete p_c ~ {p_c:.4f}  ->  E[#bytes] at p_c = {256*p_c:.1f}")
            break
    else:
        print("\n  E4 no 0.5-crossing in range")

# 6. Curve-level null with CI (shuffle byte -> transition mapping).

def null_model_curve(eng: WordEngine, n_samples: int = 120, n_shuffles: int = 5) -> None:
    print("\n" + "=" * 5)
    print("6. NULL MODEL -- SHUFFLED BYTE->TRANSITION MAPPING (curve-level)")
    print("=" * 5)
    print("  Byte KEEPS its true micro_ref label but its dynamics is swapped for a")
    print("  random other byte's dynamics. Breaks classification<->dynamics exactly.")
    print("  Word percolation (W2+W2') real vs shuffled, with binomial 95% CI.\n")

    # Build a shuffled word engine by permuting which byte-index each family/micro
    # byte maps to. We re-compose W2/W2' using a permuted byte->transition column.
    n = eng.n
    trans = eng.byte_trans  # reuse the table built in build_word_engine

    def build_shuffled_word_perms(perm: List[int]):
        w2s, w2ps = [], []
        for m in range(64):
            b00 = byte_from_family_micro(0, m)
            b01 = byte_from_family_micro(1, m)
            b10 = byte_from_family_micro(2, m)
            b11 = byte_from_family_micro(3, m)
            # byte b's dynamics replaced by byte perm[b]'s dynamics
            b00p, b01p, b10p, b11p = perm[b00], perm[b01], perm[b10], perm[b11]
            w2m = [trans[trans[i][b00p]][b01p] for i in range(n)]
            w2pm = [trans[trans[i][b10p]][b11p] for i in range(n)]
            w2s.append(w2m)
            w2ps.append(w2pm)
        return w2s, w2ps

    p_values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    z = 1.96
    print(f"  {'p':<7}{'Real P(full)':<14}{'Shuffled P(full)':<18}{'gap':<8}{'Real CI95':<14}")
    print("  " + "-" * 5)
    for p in p_values:
        # real
        real_full = 0
        rnonzero = 0
        for _ in range(n_samples):
            M = [m for m in range(64) if random.random() < p]
            if not M:
                continue
            rnonzero += 1
            perms = [eng.w2[m] for m in M] + [eng.w2p[m] for m in M]
            vis = _bfs_reach(eng, perms, [eng.start_idx])
            if sum(vis) == n:
                real_full += 1
        pr = real_full / max(rnonzero, 1)
        se_r = math.sqrt(pr * (1 - pr) / max(rnonzero, 1))
        # shuffled
        shuf_vals = []
        per_shuffle = max(1, n_samples // n_shuffles)
        for _ in range(n_shuffles):
            perm = list(range(256))
            random.shuffle(perm)
            w2s, w2ps = build_shuffled_word_perms(perm)
            sf = 0
            snz = 0
            for _ in range(per_shuffle):
                M = [m for m in range(64) if random.random() < p]
                if not M:
                    continue
                snz += 1
                perms = [w2s[m] for m in M] + [w2ps[m] for m in M]
                vis = _bfs_reach(eng, perms, [eng.start_idx])
                if sum(vis) == n:
                    sf += 1
            shuf_vals.append(sf / max(snz, 1))
        ps = sum(shuf_vals) / len(shuf_vals)
        print(f"  {p:<7.3f}{pr:<14.4f}{ps:<18.4f}{pr-ps:<8.4f}{z*se_r:<14.4f}")


# 7. Sector- and shell-resolved confinement (full W2+W2' word graph).

def sector_confinement_profile(eng: WordEngine) -> None:
    print("\n" + "=" * 5)
    print("7. SECTOR/SHELL CONFINEMENT (all 64 micro_refs, W2+W2')")
    print("=" * 5)
    print("  Constitutional sectors: comp_horizon(64), bulk(3968), eq_horizon(64).")
    print("  Shell = popcount(chi) in {0..6}.\n")

    M = list(range(64))
    perms = _perms_for_mode(eng, M, "w2w2p")
    vis, reach, _, pairing, full = _word_reach_generic(eng, perms, [eng.start_idx])

    comp_set = set(eng.comp_idx)
    eq_set = set(eng.eq_idx)
    shell_counts = [0] * 7
    comp_hit = eq_hit = bulk_hit = 0
    for i in range(eng.n):
        if not vis[i]:
            continue
        sh = eng.shell[i]
        shell_counts[sh] += 1
        if i in comp_set:
            comp_hit += 1
        elif i in eq_set:
            eq_hit += 1
        else:
            bulk_hit += 1

    print(f"  Reach from rest: {reach} / {eng.n}   full_omega={full}   bulk_hit={bulk_hit}")
    print(f"  comp_horizon reached: {comp_hit} / {len(comp_set)}")
    print(f"  eq_horizon reached:   {eq_hit} / {len(eq_set)}")
    print(f"\n  {'Shell':<8}{'Count':<8}{'Of shell':<10}")
    print("  " + "-" * 5)
    shell_sizes = [64, 384, 960, 1280, 960, 384, 64]
    for s in range(7):
        print(f"  {s:<8}{shell_counts[s]:<8}{shell_sizes[s]:<10}")

    # Multi-source from full complement horizon
    vis_c, reach_c, _, pairing_c, full_c = _word_reach_generic(
        eng, perms, eng.comp_idx)
    print(f"\n  From ALL comp_horizon sources:")
    print(f"    reach={reach_c}  full_pairing(eq)={pairing_c}  full_omega={full_c}")


# 8. Edge-mode percolation (F-only vs W2+W2' vs all three).

def edge_mode_percolation(eng: WordEngine, n_samples: int = 150) -> None:
    print("\n" + "=" * 5)
    print("8. EDGE-MODE PERCOLATION (micro_ref dial, conditional on |M|>0)")
    print("=" * 5)
    print("  Modes: w2w2p (Egress+Ingress), f (F only), all (W2+W2'+F).\n")

    p_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    modes = ("w2w2p", "f", "all")
    print(f"  {'p':<7}", end="")
    for mode in modes:
        print(f" {mode+' P(full)':<14}", end="")
    print()
    print("  " + "-" * 5)

    for p in p_values:
        print(f"  {p:<7.3f}", end="")
        for mode in modes:
            full = nz = 0
            for _ in range(n_samples):
                M = [m for m in range(64) if random.random() < p]
                if not M:
                    continue
                nz += 1
                _, _, _, fo = _word_reach_counts(eng, M, [eng.start_idx], mode=mode)
                if fo:
                    full += 1
            pf = full / max(nz, 1) if p > 0 else 0.0
            print(f" {pf:<14.4f}", end="")
        print()


# 9. Probe words (wavefunction Sec 13 structured nulls).

def _compose_byte_word(trans: List[List[int]], n: int, byte_seq: List[int]) -> List[int]:
    perm = [0] * n
    for i in range(n):
        j = i
        for b in byte_seq:
            j = trans[j][b]
        perm[i] = j
    return perm


def probe_word_reachability(eng: WordEngine) -> None:
    print("\n" + "=" * 5)
    print("9. PROBE-WORD REACHABILITY (Analysis_hQVM_Wavefunction Sec 13)")
    print("=" * 5)
    print("  Single-step word graph: one edge type per row, all 64 micro_refs.\n")

    trans = eng.byte_trans
    n = eng.n
    probes: List[Tuple[str, List[List[int]]]] = []

    # Canonical F(m) -- already in engine
    probes.append(("canonical F", eng.fp))

    # Canonical 4-family W2(m) = [fam00,fam01,fam10,fam11] per micro_ref
    probes.append(("canonical 4-fam W2", eng.w2))

    # F^2 = id (depth-8 canonical x2)
    f2 = []
    for m in range(64):
        fm = eng.fp[m]
        f2m = [fm[fm[i]] for i in range(n)]
        f2.append(f2m)
    probes.append(("canonical F^2 (depth-8)", f2))

    # Same-family 4-byte words (fam 00 and 11)
    for fam, label in ((0, "same-fam 00"), (3, "same-fam 11")):
        perms = []
        for m in range(64):
            b = byte_from_family_micro(fam, m)
            perms.append(_compose_byte_word(trans, n, [b, b, b, b]))
        probes.append((label, perms))

    # Reverse canonical byte order (fam 11,10,01,00)
    rev = []
    for m in range(64):
        seq = [byte_from_family_micro(f, m) for f in (3, 2, 1, 0)]
        rev.append(_compose_byte_word(trans, n, seq))
    probes.append(("reverse fam order", rev))

    print(f"  {'Probe':<22}{'Reach':>7}{'full_O':>8}{'on shell 0':>10}")
    print("  " + "-" * 5)
    for label, perms in probes:
        vis, reach, _, _, full = _word_reach_generic(eng, perms, [eng.start_idx])
        on_s0 = sum(1 for i in range(n) if vis[i] and eng.shell[i] == 0)
        print(f"  {label:<22}{reach:>7}{str(full):>8}{on_s0:>10}")


def constitutional_shell_trace(eng: WordEngine, omega: List[int]) -> None:
    print("\n" + "=" * 5)
    print("12. CONSTITUTIONAL BYTE SHELL TRACE (Wavefunction Sec 5)")
    print("=" * 5)
    print("  Byte-by-byte shells from rest for canonical 4-family turn")
    print(f"  (micro_ref={FAMILY_RAY_REF}, families 00->01->10->11).\n")

    def sector_label(s: int) -> str:
        if is_on_equality_horizon(s):
            return "eq-horizon"
        if is_on_horizon(s):
            return "comp-horizon"
        return "bulk"

    for n_turns, label in ((1, "depth-4 (1 turn)"), (2, "depth-8 (2 turns)")):
        state = omega[eng.start_idx]
        shells = [eng.shell[eng.start_idx]]
        sectors = [sector_label(state)]
        seq: List[int] = []
        for _ in range(n_turns):
            seq.extend(byte_from_family_micro(f, FAMILY_RAY_REF) for f in range(4))
        for b in seq:
            state = step_state_by_byte(state, b)
            shells.append(eng.shell[eng.state_to_idx[state]])
            sectors.append(sector_label(state))
        print(f"  {label}:")
        print(f"    shells  = {shells}")
        print(f"    sectors = {sectors}")
        byte_hex = " ".join(f"{b:02X}" for b in seq)
        print(f"    bytes   = {byte_hex}")


# 10. Helix Z2 cycle and flat/curved payload dial.

def helix_z2_cycle(eng: WordEngine) -> None:
    print("\n" + "=" * 5)
    print("10. HELIX Z2 CYCLE (F involution, depth-8 from rest)")
    print("=" * 5)
    print("  rest --F(m)--> swapped --F(m)--> rest  (T6, all m).\n")

    rest = eng.start_idx
    n_ok = 0
    for m in range(64):
        fm = eng.fp[m]
        once = fm[rest]
        twice = fm[once]
        if twice == rest:
            n_ok += 1
    print(f"  F(m)^2(rest)=rest for {n_ok} / 64 micro_refs")

    # Shell after one F(m) from rest (expect shell 0, Z2 sheet flip)
    shells_after_f = Counter(eng.shell[eng.fp[m][rest]] for m in range(64))
    print(f"  Shell distribution after one F(m) from rest: {dict(sorted(shells_after_f.items()))}")


def flat_curved_payload(eng: WordEngine, n_samples: int = 150) -> None:
    print("\n" + "=" * 5)
    print("11. FLAT vs CURVED PAYLOAD (fiber bundle Sec 16.1)")
    print("=" * 5)
    print("  micro_ref m is flat iff all 4 family bytes have fwd==rev (P=I).")
    print("  Word percolation with flat-only vs curved-only micro_ref pools.\n")

    flat_m: List[int] = []
    curved_m: List[int] = []
    for m in range(64):
        bytes4 = [byte_from_family_micro(f, m) for f in range(4)]
        if all(decompose_byte(b).is_flat for b in bytes4):
            flat_m.append(m)
        else:
            curved_m.append(m)

    print(f"  Flat micro_refs: {len(flat_m)}   Curved micro_refs: {len(curved_m)}")

    p_values = [0.1, 0.2, 0.3, 0.5, 1.0]
    print(f"\n  {'p':<7}{'flat P(pair)':<14}{'curved P(pair)':<16}{'flat Reach':<12}{'curved Reach':<12}")
    print("  " + "-" * 5)
    for p in p_values:
        stats: Dict[str, Tuple[float, float]] = {}
        for pool_name, pool in (("flat", flat_m), ("curved", curved_m)):
            pair = reach_sum = nz = 0
            if not pool:
                stats[pool_name] = (0.0, 0.0)
                continue
            for _ in range(n_samples):
                M = [m for m in pool if random.random() < p]
                if not M:
                    continue
                nz += 1
                r, _, pr, _ = _word_reach_counts(eng, M, eng.comp_idx, mode="w2w2p")
                reach_sum += r
                if pr:
                    pair += 1
            stats[pool_name] = (pair / max(nz, 1), reach_sum / max(nz, 1))
        fp, fr = stats["flat"]
        cp, cr = stats["curved"]
        print(f"  {p:<7.3f}{fp:<14.4f}{cp:<16.4f}{fr:<12.1f}{cr:<12.1f}")


# Main

def main() -> None:
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    random.seed(20260702)

    print("hQVM Canonical-Word Percolation -- the CGM-native percolation theory")
    print("=" * 5)
    print("Grounded in Analysis_hQVM_Wavefunction.md Theorems T1-T10 and")
    print("Gyroscopic_ASI_Specs_Formalism.md Sec 2.2/4 (6-bit payload dial).")

    print("\n1. MANIFOLD AND WORD OPERATOR CONSTRUCTION")
    print("-" * 5)
    omega = enumerate_omega()
    eng = build_word_engine(omega)

    verify_word_algebra(eng)
    single_pole_theorem(eng)
    word_percolation_sweep(eng, n_samples=200)
    curvature_spectrum_sweep(eng, n_samples=200)
    null_model_curve(eng, n_samples=120, n_shuffles=5)
    sector_confinement_profile(eng)
    edge_mode_percolation(eng, n_samples=150)
    probe_word_reachability(eng)
    constitutional_shell_trace(eng, omega)
    helix_z2_cycle(eng)
    flat_curved_payload(eng, n_samples=150)

    print("\n" + "=" * 5)
    print("END. See docs/Findings/Analysis_hQVM_Percolation.md for the writeup.")
    print("=" * 5)

if __name__ == "__main__":
    main()
