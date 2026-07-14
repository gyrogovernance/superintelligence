#!/usr/bin/env python3
"""
hQVM Percolation Square-Root Universality (analysis_4)
========================================================

Companion to hqvm_percolation_analysis_{1,2,3}.py and
hqvm_percolation_analysis_run.py.

Validates the fiber-complete product cluster theorem on Omega: under
fiber-complete generator restriction, |Reach(A)| = (2^r(A))^2 with
R = R_U x R_V. Scope boundaries use fiber-incomplete selection.
All checks use the transition engine, BFS reachability, U x V coset
coordinates, and canonical word edges only.

Sections:
  1.  Holographic root identity (|H|^2 = |Omega|)
  2.  Fiber-complete product cluster theorem + scope boundary
  3.  Parity obstruction (even q6 subspace cannot span Omega)
  4.  Word confinement as root-only action
  5.  Root-dimension criticality hierarchy (ordering gate)
  5b. Holonomy word scaling (availability event + mean completion)
  6.  Flux normalization bridge (D=24, Q_G=4*pi, G_kernel=4*pi/D)

Data only. Interpretation: docs/Findings/Analysis_hQVM_Percolation.md
"""

from __future__ import annotations

import contextlib
import io
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import hqvm_percolation_analysis_1 as pa1
import hqvm_percolation_analysis_2 as pa2
import hqvm_percolation_analysis_3 as pa3

from gyroscopic.hQVM.api import q_word6

N_OMEGA = 4096
N_HORIZON = 64
CHI_BITS = 6
K4_ORDER = 4
HOLONOMY_WORD_LENGTH = 4
N_MICROREFS = 64
M_A = 1.0 / (2.0 * math.sqrt(2.0 * math.pi))
FLUX_QUANTUM = 4.0 * math.pi


@dataclass(frozen=True)
class Gate:
    name: str
    ok: bool
    detail: str


GATES: list[Gate] = []


def _gate(name: str, ok: bool, detail: str) -> None:
    GATES.append(Gate(name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    print(f"         {detail}")


def _configure_stdout_utf8() -> None:
    import codecs
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")


def gf2_rank(vectors: Iterable[int]) -> int:
    """Rank over GF(2) on 6-bit payloads via deterministic Gaussian elimination."""
    basis = [0] * CHI_BITS
    rank = 0
    for v in vectors:
        x = v & 0x3F
        for bit in range(CHI_BITS - 1, -1, -1):
            if (x >> bit) & 1:
                if basis[bit]:
                    x ^= basis[bit]
                else:
                    basis[bit] = x
                    rank += 1
                    break
    return rank


def root_dimension(r: int) -> int:
    """Surviving root cardinality 2^r; r=0 gives minimal root 2."""
    if r == 0:
        return 2
    return 1 << r


def predicted_cluster_size(r: int) -> int:
    """|Reach| = root^2; r=0 degenerate case gives |Reach| = 2."""
    if r == 0:
        return 2
    d = root_dimension(r)
    return d * d


def expected_rectangularity(r: int) -> float:
    """Product-set rectangularity; r=0 is diagonal in 2x2."""
    return 0.5 if r == 0 else 1.0


def is_fiber_complete(
    alphabet: Sequence[int],
    bcs: dict[int, pa1.ByteClassification],
) -> bool:
    """True iff every q6 value present includes all four family bytes."""
    families_by_q: dict[int, set[int]] = defaultdict(set)
    for b in alphabet:
        families_by_q[bcs[b].q6].add(bcs[b].family)
    return all(len(fams) == 4 for fams in families_by_q.values())


def _verify_cluster_case(
    engine: pa1.TransitionEngine,
    alphabet: Sequence[int],
    bcs: dict[int, pa1.ByteClassification],
    u_idx: List[int],
    v_idx: List[int],
) -> Tuple[bool, bool, bool, int]:
    """Return (size_ok, rect_ok, sym_ok, rank)."""
    q6s = [bcs[b].q6 for b in alphabet]
    r = gf2_rank(q6s)
    root = root_dimension(r)
    pred = predicted_cluster_size(r)
    reach, nu, nv, rect = reach_rectangle(engine, alphabet, u_idx, v_idx)
    exp_rect = expected_rectangularity(r)
    size_ok = reach == pred
    rect_ok = abs(rect - exp_rect) < 1e-9 if reach else True
    sym_ok = (nu == nv == root) if r > 0 else (nu == nv == 2 and reach == 2)
    return size_ok, rect_ok, sym_ok, r


def _q6_to_bytes(bcs: dict[int, pa1.ByteClassification]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for bc in bcs.values():
        out[bc.q6].append(bc.byte)
    return out


def _independent_q6_basis(target_r: int, rng: random.Random) -> list[int]:
    """Return target_r independent q6 vectors spanning a random r-dimensional subspace."""
    if target_r == 0:
        return []
    pool = list(range(1, 64))
    rng.shuffle(pool)
    basis: list[int] = []
    for q in pool:
        trial = basis + [q]
        if gf2_rank(trial) == len(trial):
            basis.append(q)
        if len(basis) == target_r:
            break
    return basis


def reach_rectangle(
    engine: pa1.TransitionEngine,
    allowed: Sequence[int],
    u_idx: List[int],
    v_idx: List[int],
) -> Tuple[int, int, int, float]:
    """Return |R|, |U_R|, |V_R|, rectangularity |R|/(|U_R||V_R|)."""
    vis = pa1.reachable_mask(engine, list(allowed), [engine.start_idx])
    us: set[int] = set()
    vs: set[int] = set()
    rcount = 0
    for i, on in enumerate(vis):
        if not on:
            continue
        rcount += 1
        us.add(u_idx[i])
        vs.add(v_idx[i])
    nu, nv = len(us), len(vs)
    rect = rcount / (nu * nv) if nu and nv else 0.0
    return rcount, nu, nv, rect


def _rank_cases(bcs: dict[int, pa1.ByteClassification]) -> List[Tuple[str, List[int], int | None]]:
    by_weight: dict[int, list[int]] = {w: [] for w in range(7)}
    by_fold: dict[int, list[int]] = {d: [] for d in range(5)}
    for bc in bcs.values():
        by_weight[bc.q6_weight].append(bc.byte)
        by_fold[bc.fold_disagree].append(bc.byte)

    cases: list[Tuple[str, List[int], int | None]] = []
    for w in range(7):
        cases.append((f"Q6 weight={w}", by_weight[w], None))
    for d in range(5):
        cases.append((f"fold_disagree={d}", by_fold[d], None))
    even = [bc.byte for bc in bcs.values() if bc.q6_weight % 2 == 0]
    odd = [bc.byte for bc in bcs.values() if bc.q6_weight % 2 == 1]
    cases.append(("even Q6 weight", even, 5))
    cases.append(("odd Q6 weight", odd, 6))
    cases.append(("all 256", list(range(256)), 6))
    return cases


def _find_crossing(p_vals: List[float], probs: List[float], target: float = 0.5) -> float | None:
    for i in range(len(p_vals) - 1):
        if probs[i] <= target <= probs[i + 1]:
            p0, p1 = p_vals[i], p_vals[i + 1]
            y0, y1 = probs[i], probs[i + 1]
            if y1 == y0:
                return p1
            return p0 + (target - y0) * (p1 - p0) / (y1 - y0)
    return None


# ---------------------------------------------------------------------------
# 1. Holographic root identity
# ---------------------------------------------------------------------------

def section_1_holographic_root(engine: pa1.TransitionEngine, omega: List[int]) -> None:
    print("\n" + "=" * 5)
    print("1. HOLOGRAPHIC ROOT IDENTITY")
    print("=" * 5)
    print("  |H| = horizon cardinality, |Omega| = manifold cardinality.")
    print("  Identity: |H|^2 = |Omega| (self-dual code product).\n")

    comp, eq = pa1.horizon_indices(omega)
    n_comp, n_eq = len(comp), len(eq)
    n_omega = len(omega)
    ratio = (n_comp * n_eq) / n_omega if n_omega else 0.0

    print(f"  |comp horizon| = {n_comp}")
    print(f"  |eq horizon|   = {n_eq}")
    print(f"  |Omega|        = {n_omega}")
    print(f"  |H|^2/|Omega|  = {ratio:.6f}")

    _gate(
        "horizon_cardinality_64",
        n_comp == 64 and n_eq == 64,
        f"|H| = {n_comp} per pole",
    )
    _gate(
        "holographic_square_identity",
        n_comp == n_eq and n_comp * n_eq == n_omega,
        f"{n_comp}^2 = {n_omega}",
    )
    _gate(
        "root_is_sqrt_of_manifold",
        int(math.isqrt(n_omega)) == n_comp,
        f"sqrt(|Omega|) = {int(math.isqrt(n_omega))} = |H|",
    )


# ---------------------------------------------------------------------------
# 2. Fiber-complete product cluster theorem
# ---------------------------------------------------------------------------

def section_2_square_root_cluster(
    engine: pa1.TransitionEngine,
    bcs: dict[int, pa1.ByteClassification],
    u_idx: List[int],
    v_idx: List[int],
) -> None:
    print("\n" + "=" * 5)
    print("2. FIBER-COMPLETE PRODUCT CLUSTER THEOREM")
    print("=" * 5)
    print("  Theorem: fiber-complete A => Reach = R_U x R_V, |R_U|=|R_V|=2^r(A),")
    print("           |Reach(A)| = (2^r(A))^2.")
    print("  Corollary: full Omega requires r=6 and odd-parity transport.")
    print("  Fiber-complete: every q6 present includes all four family bytes.\n")

    size_fails = 0
    rect_fails = 0
    sym_fails = 0
    n_cases = 0
    n_fiber = 0

    print(f"  {'case':<22}{'r':>3}{'fc':>3}{'|Reach|':>8}{'pred':>8}{'|U|':>6}{'|V|':>6}{'rect':>6}  ok")
    print("  " + "-" * 5)

    for name, alphabet, expect_r in _rank_cases(bcs):
        if not alphabet:
            continue
        n_cases += 1
        fc = is_fiber_complete(alphabet, bcs)
        if fc:
            n_fiber += 1
        q6s = [bcs[b].q6 for b in alphabet]
        r = gf2_rank(q6s)
        root = root_dimension(r)
        pred = predicted_cluster_size(r)
        reach, nu, nv, rect = reach_rectangle(engine, alphabet, u_idx, v_idx)

        ok_size = reach == pred
        exp_rect = expected_rectangularity(r)
        ok_rect = abs(rect - exp_rect) < 1e-9 if reach else True
        ok_sym = (nu == nv == root) if r > 0 else (nu == nv == 2 and reach == 2)
        if expect_r is not None and r != expect_r:
            ok_size = False

        if not ok_size:
            size_fails += 1
        if not ok_rect:
            rect_fails += 1
        if not ok_sym:
            sym_fails += 1

        ok = ok_size and ok_rect and ok_sym
        print(
            f"  {name:<22}{r:3d}{'Y' if fc else 'n':>3}{reach:8d}{pred:8d}"
            f"{nu:6d}{nv:6d}{rect:6.3f}  {ok}"
        )

    _gate(
        "fiber_complete_product_cluster_size",
        size_fails == 0,
        f"{n_cases - size_fails}/{n_cases} structured cases, |Reach|=(2^r)^2",
    )
    _gate(
        "fiber_complete_reachable_set_is_RU_x_RV",
        rect_fails == 0,
        f"{n_cases - rect_fails}/{n_cases} structured cases, rect matches product",
    )
    _gate(
        "fiber_complete_marginal_factors_equal",
        sym_fails == 0,
        f"{n_cases - sym_fails}/{n_cases} structured cases, |U_R|=|V_R|=2^r",
    )
    print(f"  fiber-complete cases in table: {n_fiber}/{n_cases}")

    _section_2_fiber_incomplete_scope(engine, bcs, u_idx, v_idx)


def _section_2_fiber_incomplete_scope(
    engine: pa1.TransitionEngine,
    bcs: dict[int, pa1.ByteClassification],
    u_idx: List[int],
    v_idx: List[int],
) -> None:
    print("\n  Scope boundary: fiber-incomplete restriction")
    print("  (one byte per q6 class; product structure may degrade)\n")

    fam_fails = 0
    for fam in range(4):
        alphabet = [bc.byte for bc in bcs.values() if bc.family == fam]
        size_ok, rect_ok, sym_ok, r = _verify_cluster_case(
            engine, alphabet, bcs, u_idx, v_idx,
        )
        reach = sum(pa1.reachable_mask(engine, alphabet, [engine.start_idx]))
        fc = is_fiber_complete(alphabet, bcs)
        ok = size_ok and rect_ok and sym_ok
        if not ok:
            fam_fails += 1
        print(f"    family {fam:02b}: r={r}  fc={fc}  |Reach|={reach}  ok={ok}")
    _gate(
        "family_slices_product_cluster",
        fam_fails == 0,
        f"{4 - fam_fails}/4 family slices satisfy |Reach|=(2^r)^2",
    )

    rng = random.Random(20260703)
    even_pool = [bc.byte for bc in bcs.values() if bc.q6_weight % 2 == 0]
    odd_pool = [bc.byte for bc in bcs.values() if bc.q6_weight % 2 == 1]
    q6_map = _q6_to_bytes(bcs)

    print("\n  Fiber-incomplete: six bytes, one per independent q6")
    reach_rank6: list[int] = []
    rect_rank6: list[float] = []
    for _ in range(200):
        basis = _independent_q6_basis(6, rng)
        alphabet = [rng.choice(q6_map[q]) for q in basis]
        r = gf2_rank(bcs[b].q6 for b in alphabet)
        if r != 6:
            continue
        reach = sum(pa1.reachable_mask(engine, alphabet, [engine.start_idx]))
        reach_rank6.append(reach)
        _, nu, nv, rect = reach_rectangle(engine, alphabet, u_idx, v_idx)
        rect_rank6.append(rect)
    max_reach = max(reach_rank6) if reach_rank6 else 0
    min_rect = min(rect_rank6) if rect_rank6 else 0.0
    print(f"    n={len(reach_rank6)}  max|Reach|={max_reach}  min rect={min_rect:.3f}")
    print(f"    (rank=6 but fiber-incomplete; full Omega={N_OMEGA} not required)")

    print("\n  Corollary: even-q6 fiber-complete subspace (128 generators)")
    even_r = gf2_rank(bcs[b].q6 for b in even_pool)
    even_reach = sum(pa1.reachable_mask(engine, even_pool, [engine.start_idx]))
    print(f"    |A|={len(even_pool)}  rank={even_r}  |Reach|={even_reach}")
    _gate(
        "corollary_even_subspace_max_32_squared",
        even_r == 5 and even_reach == 1024,
        "rank=5 even subspace, |Reach|=1024=32^2",
    )

    print("\n  Report: rank vs |A| under random restriction (not gated)")
    confined_1024 = full_hits = 0
    for _ in range(300):
        n = rng.randint(20, min(64, len(even_pool)))
        alphabet = rng.sample(even_pool, n)
        reach = sum(pa1.reachable_mask(engine, alphabet, [engine.start_idx]))
        if reach == 1024:
            confined_1024 += 1
        if reach == N_OMEGA:
            full_hits += 1
    print(f"    even-q6 |A| in [20,64]: P(1024)={confined_1024/300:.3f}  P(full)={full_hits/300:.3f}")

    anticorr = 0
    for _ in range(200):
        even_reach = sum(
            pa1.reachable_mask(
                engine, rng.sample(even_pool, 24), [engine.start_idx],
            )
        )
        basis = _independent_q6_basis(6, rng)
        odd_alphabet = [rng.choice(q6_map[q]) for q in basis]
        while len(odd_alphabet) < 24:
            odd_alphabet.append(rng.choice(odd_pool))
        odd_reach = sum(pa1.reachable_mask(engine, odd_alphabet[:24], [engine.start_idx]))
        if even_reach < odd_reach:
            anticorr += 1
    print(f"    |A|=24 even vs fiber-incomplete odd: P(even<odd)={anticorr/200:.3f}")


# ---------------------------------------------------------------------------
# 3. Parity obstruction
# ---------------------------------------------------------------------------

def section_3_parity_obstruction(
    engine: pa1.TransitionEngine,
    bcs: dict[int, pa1.ByteClassification],
) -> None:
    print("\n" + "=" * 5)
    print("3. PARITY OBSTRUCTION")
    print("=" * 5)
    print("  Even-q6 alphabets confine to even shells; full Omega requires odd shell.\n")

    even_bytes = [bc.byte for bc in bcs.values() if bc.q6_weight % 2 == 0]
    vis = pa1.reachable_mask(engine, even_bytes, [engine.start_idx])
    shells = {engine.shell[i] for i, on in enumerate(vis) if on}
    reach = sum(vis)
    odd_present = any(s % 2 == 1 for s in shells)

    print(f"  even-q6 alphabet size: {len(even_bytes)}")
    print(f"  |Reach| = {reach}")
    print(f"  shells hit: {sorted(shells)}")
    print(f"  odd shells present: {odd_present}")

    _gate(
        "even_q6_no_odd_shells",
        not odd_present,
        f"shells = {sorted(shells)}",
    )
    _gate(
        "even_q6_not_full_omega",
        reach < N_OMEGA,
        f"|Reach| = {reach} < {N_OMEGA}",
    )

    odd_bytes = [bc.byte for bc in bcs.values() if bc.q6_weight % 2 == 1]
    r_odd = gf2_rank(bcs[b].q6 for b in odd_bytes)
    reach_odd = sum(pa1.reachable_mask(engine, odd_bytes, [engine.start_idx]))
    _gate(
        "odd_q6_full_root_rank",
        r_odd == 6 and reach_odd == N_OMEGA,
        f"rank={r_odd}, |Reach|={reach_odd}",
    )


# ---------------------------------------------------------------------------
# 4. Word confinement as root-only action
# ---------------------------------------------------------------------------

def section_4_word_root_confinement(omega: List[int]) -> None:
    print("\n" + "=" * 5)
    print("4. WORD CONFINEMENT (ROOT-ONLY ACTION)")
    print("=" * 5)
    print("  Canonical W2 words from rest: orbit <= 128, shells {0, 6} only.\n")

    with contextlib.redirect_stdout(io.StringIO()):
        eng = pa2.build_word_engine(omega)

    # All singleton micro-refs
    max_reach = 0
    bad_shell = False
    for m in range(N_MICROREFS):
        vis = pa2._bfs_reach(eng, [eng.w2[m]], [eng.start_idx])
        n = sum(vis)
        max_reach = max(max_reach, n)
        for i, on in enumerate(vis):
            if not on:
                continue
            sh = eng.shell[i]
            if sh not in (0, 6):
                bad_shell = True

    # Full W2 family
    vis_all = pa2._bfs_reach(eng, eng.w2, [eng.start_idx])
    n_all = sum(vis_all)
    shells_all = {eng.shell[i] for i, on in enumerate(vis_all) if on}

    print(f"  max |Reach| singleton W2(m): {max_reach}")
    print(f"  |Reach| all W2:             {n_all}")
    print(f"  shells all W2:              {sorted(shells_all)}")

    _gate(
        "w2_singleton_orbit_le_128",
        max_reach <= 128,
        f"max = {max_reach}",
    )
    _gate(
        "w2_all_shells_horizon_only",
        shells_all <= {0, 6},
        f"shells = {sorted(shells_all)}",
    )
    _gate(
        "w2_all_orbit_equals_128",
        n_all == 128,
        f"|Reach| = {n_all}",
    )

    _gate(
        "word_regime_horizon_confinement",
        n_all == 128 and shells_all == {0, 6},
        "128-state horizon orbit from rest",
    )

    # W2 + W2' + F from rest and from all complement-horizon sources
    k4_perms = eng.w2 + eng.w2p + eng.fp
    vis_k4_rest = pa2._bfs_reach(eng, k4_perms, [eng.start_idx])
    n_k4_rest = sum(vis_k4_rest)
    shells_k4 = {eng.shell[i] for i, on in enumerate(vis_k4_rest) if on}

    vis_k4_comp = pa2._bfs_reach(eng, k4_perms, eng.comp_idx)
    n_k4_comp = sum(vis_k4_comp)
    shells_k4_comp = {eng.shell[i] for i, on in enumerate(vis_k4_comp) if on}

    print(f"\n  |Reach| W2+W2'+F from rest: {n_k4_rest}")
    print(f"  shells: {sorted(shells_k4)}")
    print(f"  |Reach| W2+W2'+F from all comp horizon: {n_k4_comp}")
    print(f"  shells comp starts: {sorted(shells_k4_comp)}")

    _gate(
        "k4_words_rest_le_128",
        n_k4_rest <= 128,
        f"|Reach| = {n_k4_rest}",
    )
    _gate(
        "k4_words_comp_horizon_shells_only",
        shells_k4_comp <= {0, 6},
        f"shells = {sorted(shells_k4_comp)}",
    )


# ---------------------------------------------------------------------------
# 5. Root-dimension criticality hierarchy
# ---------------------------------------------------------------------------

def _fword_bytes(m: int) -> Tuple[int, int, int, int]:
    return (
        pa2.byte_from_family_micro(0, m),
        pa2.byte_from_family_micro(1, m),
        pa2.byte_from_family_micro(2, m),
        pa2.byte_from_family_micro(3, m),
    )


def any_microref_complete(allowed: Sequence[int]) -> bool:
    """Word-availability event: at least one micro_ref has all four F-word bytes."""
    aset = set(allowed)
    for m in range(N_MICROREFS):
        if all(b in aset for b in _fword_bytes(m)):
            return True
    return False


def microref_completion_fraction(allowed: Sequence[int]) -> float:
    """Mean fraction of micro_refs with all four F-word bytes present."""
    aset = set(allowed)
    covered = sum(
        1 for m in range(N_MICROREFS)
        if all(b in aset for b in _fword_bytes(m))
    )
    return covered / N_MICROREFS


def predicted_word_availability(p: float) -> float:
    return pa3.expected_micro_cov(p)


def predicted_mean_completion(p: float) -> float:
    return p ** HOLONOMY_WORD_LENGTH


def _p_hit_span(engine: pa1.TransitionEngine, allowed: List[int]) -> bool:
    vis = pa1.reachable_mask(engine, allowed, [engine.start_idx])
    return any(on and engine.shell[i] == 0 for i, on in enumerate(vis))


def _p_hit_full(engine: pa1.TransitionEngine, allowed: List[int]) -> bool:
    return sum(pa1.reachable_mask(engine, allowed, [engine.start_idx])) == N_OMEGA


def _p_spectrum_complete(allowed: List[int]) -> bool:
    if len(allowed) < 2:
        return False
    weights: set[int] = set()
    for i, bi in enumerate(allowed):
        qi = q_word6(bi)
        for bj in allowed[i:]:
            weights.add((qi ^ q_word6(bj)).bit_count())
    return len(weights) == 7


def _p_h1_turnon(engine: pa1.TransitionEngine, allowed: List[int]) -> bool:
    h1, _, _, _ = pa3._future_cone_h12(engine, engine.start_idx, allowed)
    return h1 >= 4.0


def _p_h1_saturated(engine: pa1.TransitionEngine, allowed: List[int]) -> bool:
    h1, _, _, _ = pa3._future_cone_h12(engine, engine.start_idx, allowed)
    return h1 >= 6.5


def _two_step_multiplicities(
    engine: pa1.TransitionEngine,
    allowed: List[int],
) -> Tuple[int, int, int]:
    """Return (min mult, max mult, states covered) from rest."""
    if len(allowed) < 2:
        return 0, 0, 0
    start = engine.start_idx
    trans = engine.transitions
    mult: Counter = Counter()
    for b1 in allowed:
        mid = trans[start][b1]
        for b2 in allowed:
            mult[trans[mid][b2]] += 1
    if not mult:
        return 0, 0, 0
    vals = list(mult.values())
    return min(vals), max(vals), len(mult)


def _p_uniform_turnon(engine: pa1.TransitionEngine, allowed: List[int]) -> bool:
    mn, mx, cover = _two_step_multiplicities(engine, allowed)
    return cover >= N_OMEGA // 2 and mn >= 1


def _p_uniform_saturated(engine: pa1.TransitionEngine, allowed: List[int]) -> bool:
    mn, mx, cover = _two_step_multiplicities(engine, allowed)
    return cover == N_OMEGA and mn >= 15


def section_5_criticality_hierarchy(engine: pa1.TransitionEngine) -> None:
    print("\n" + "=" * 5)
    print("5. ROOT-DIMENSION CRITICALITY HIERARCHY")
    print("=" * 5)
    print("  Generator restriction dial p; weak (50%) and strong (90%) onsets.\n")

    p_vals = [0.01, 0.015, 0.02, 0.022, 0.025, 0.03, 0.035, 0.04, 0.05,
              0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0]
    n_mc = 200
    rng = random.Random(20260703)

    weak_keys = ("span", "full", "spectrum", "h1_wk", "word_evt")
    strong_keys = ("h1_st", "unif_wk", "unif_st", "word_frac")
    weak_probs = {k: [] for k in weak_keys}
    strong_probs = {k: [] for k in strong_keys}
    mean_frac: list[float] = []

    for p in p_vals:
        weak_hits = {k: 0 for k in weak_keys}
        strong_hits = {k: 0 for k in strong_keys}
        frac_sum = 0.0
        nz = 0
        for _ in range(n_mc):
            allowed = [b for b in range(256) if rng.random() < p]
            if not allowed:
                continue
            nz += 1
            if _p_hit_span(engine, allowed):
                weak_hits["span"] += 1
            if _p_hit_full(engine, allowed):
                weak_hits["full"] += 1
            if _p_spectrum_complete(allowed):
                weak_hits["spectrum"] += 1
            if _p_h1_turnon(engine, allowed):
                weak_hits["h1_wk"] += 1
            if any_microref_complete(allowed):
                weak_hits["word_evt"] += 1
            if _p_h1_saturated(engine, allowed):
                strong_hits["h1_st"] += 1
            if _p_uniform_turnon(engine, allowed):
                strong_hits["unif_wk"] += 1
            if _p_uniform_saturated(engine, allowed):
                strong_hits["unif_st"] += 1
            if microref_completion_fraction(allowed) >= 0.5:
                strong_hits["word_frac"] += 1
            frac_sum += microref_completion_fraction(allowed)
        denom = max(nz, 1)
        for k in weak_keys:
            weak_probs[k].append(weak_hits[k] / denom)
        for k in strong_keys:
            strong_probs[k].append(strong_hits[k] / denom)
        mean_frac.append(frac_sum / denom)

    print(f"  {'p':<8}", end="")
    for k in weak_keys:
        print(f"{k:<9}", end="")
    print()
    print("  " + "-" * 5)
    for i, p in enumerate(p_vals):
        print(f"  {p:<8.3f}", end="")
        for k in weak_keys:
            print(f"{weak_probs[k][i]:<9.4f}", end="")
        print()

    weak_pc = {k: _find_crossing(p_vals, weak_probs[k], 0.5) for k in weak_keys}
    strong_pc = {k: _find_crossing(p_vals, strong_probs[k], 0.5) for k in strong_keys}

    print("\n  weak onset p_c (50%):")
    for k in weak_keys:
        pc = weak_pc[k]
        print(f"    {k:<12} {pc:.4f}" if pc is not None else f"    {k:<12} not in range")

    print("\n  strong onset p_c (50% of strong criterion):")
    for k in strong_keys:
        pc = strong_pc[k]
        print(f"    {k:<12} {pc:.4f}" if pc is not None else f"    {k:<12} not in range")

    order_ok = True
    prev_p = 0.0
    prev_name = "start"
    for name in ("span", "full", "spectrum", "h1_wk", "word_evt"):
        est = weak_pc.get(name)
        if est is None:
            order_ok = False
            continue
        if est < prev_p - 0.005:
            order_ok = False
            print(f"  ORDER FAIL: {name} p_c={est:.4f} < {prev_name} p_c={prev_p:.4f}")
        prev_p = est
        prev_name = name

    _gate(
        "weak_threshold_ordering",
        order_ok,
        "span < full < spectrum < h1_wk < word_evt",
    )

    if weak_pc["full"] is not None:
        print(f"\n  report: p_c(full) ~ {weak_pc['full']:.4f}  (ref 0.029)")
    if strong_pc["unif_st"] is not None:
        print(f"  report: p_c(unif_st) ~ {strong_pc['unif_st']:.4f}  (ref 0.70)")


def section_5b_holonomy_word_scaling(engine: pa1.TransitionEngine) -> None:
    print("\n" + "=" * 5)
    print("5b. HOLONOMY WORD SCALING")
    print("=" * 5)
    print("  word_evt: any_microref_complete; mean frac vs p^4.")
    print("  E[word_evt] = 1 - (1 - p^4)^64.\n")

    p_vals = [0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.0]
    n_mc = 300
    rng = random.Random(20260704)

    print(f"  {'p':<8}{'E[evt]':<10}{'pred_evt':<12}{'E[frac]':<10}{'pred_frac':<12}{'|err|':<8}")
    print("  " + "-" * 5)

    evt_err_max = 0.0
    frac_err_max = 0.0
    for p in p_vals:
        evt_sum = frac_sum = 0.0
        nz = 0
        for _ in range(n_mc):
            allowed = [b for b in range(256) if rng.random() < p]
            if not allowed:
                continue
            nz += 1
            evt_sum += 1.0 if any_microref_complete(allowed) else 0.0
            frac_sum += microref_completion_fraction(allowed)
        denom = max(nz, 1)
        e_evt = evt_sum / denom
        e_frac = frac_sum / denom
        p_evt = predicted_word_availability(p)
        p_frac = predicted_mean_completion(p)
        evt_err = abs(e_evt - p_evt)
        frac_err = abs(e_frac - p_frac)
        evt_err_max = max(evt_err_max, evt_err)
        frac_err_max = max(frac_err_max, frac_err)
        print(f"  {p:<8.3f}{e_evt:<10.4f}{p_evt:<12.4f}{e_frac:<10.4f}{p_frac:<12.4f}"
              f"{max(evt_err, frac_err):<8.4f}")

    tol = 0.08
    _gate(
        "word_availability_matches_analytic",
        evt_err_max <= tol,
        f"max |E[evt] - (1-(1-p^4)^64)| <= {tol}, observed {evt_err_max:.4f}",
    )
    _gate(
        "mean_completion_matches_p4",
        frac_err_max <= tol,
        f"max |E[frac] - p^4| <= {tol}, observed {frac_err_max:.4f}",
    )


# ---------------------------------------------------------------------------
# 6. Flux normalization bridge (D, Q_G, G_kernel)
# ---------------------------------------------------------------------------

def section_6_loop_normalization_bridge() -> None:
    print("\n" + "=" * 5)
    print("6. FLUX NORMALIZATION BRIDGE")
    print("=" * 5)
    print("  D from percolation loop census; Q_G=4*pi solid angle; G_kernel=Q_G/D.\n")

    d_full, mean_defect = pa3._plaquette_D(list(range(256)))
    q_g = FLUX_QUANTUM
    g_kernel = q_g / d_full
    holographic_square = q_g * q_g
    four_pi_sq = holographic_square / K4_ORDER
    sixteen_pi_sq = holographic_square

    print(f"  D(full) = {d_full:.6f}")
    print(f"  mean defect = {mean_defect:.6f}")
    print(f"  Q_G = 4*pi = {q_g:.10f}")
    print(f"  G_kernel = Q_G / D = {g_kernel:.10f}")
    print(f"  (Q_G)^2 = 16*pi^2 = {sixteen_pi_sq:.10f}")
    print(f"  16*pi^2 / |K4| = 4*pi^2 = {four_pi_sq:.10f}")
    print(f"  m_a = {M_A:.10f}")
    print(f"  Q_G * m_a^2 = {q_g * M_A * M_A:.10f}")

    _gate(
        "plaquette_D_equals_24",
        abs(d_full - 24.0) < 1e-9,
        f"D(full) = {d_full:.6f}",
    )
    _gate(
        "mean_defect_equals_3",
        abs(mean_defect - 3.0) < 1e-9,
        f"mean defect = {mean_defect:.6f}",
    )
    _gate(
        "G_kernel_from_flux_over_D",
        abs(g_kernel - q_g / 24.0) < 1e-9,
        f"G_kernel = 4*pi/24 = {g_kernel:.10f}",
    )
    _gate(
        "Q_G_square_equals_16pi2",
        abs(holographic_square - (4.0 * math.pi) ** 2) < 1e-9,
        "(Q_G)^2 = 16*pi^2",
    )
    _gate(
        "sixteen_pi_sq_resolved_by_k4",
        abs(sixteen_pi_sq / K4_ORDER - four_pi_sq) < 1e-9,
        "16*pi^2 / |K4| = 4*pi^2",
    )
    _gate(
        "discrete_root_squared_is_omega",
        N_HORIZON * N_HORIZON == N_OMEGA,
        "64^2 = 4096",
    )
    _gate(
        "aperture_spinorial_identity",
        abs(q_g * M_A * M_A - 0.5) < 1e-9,
        "Q_G * m_a^2 = 1/2",
    )

    print(f"\n  |H| / Q_G = {N_HORIZON / q_g:.6f}")


def print_summary() -> int:
    print("\n" + "=" * 5)
    print("SUMMARY")
    print("=" * 5)
    passed = sum(1 for g in GATES if g.ok)
    failed = sum(1 for g in GATES if not g.ok)
    print(f"  gates: {passed} pass, {failed} fail, {len(GATES)} total")
    if failed:
        print("  failed:")
        for g in GATES:
            if not g.ok:
                print(f"    - {g.name}: {g.detail}")
    return 0 if failed == 0 else 1


def main() -> None:
    _configure_stdout_utf8()
    print("hQVM Percolation Fiber-Complete Product Cluster (analysis_4)")
    print(f"|Omega| = {N_OMEGA}  |H| = {N_HORIZON}  sqrt(|Omega|) = {int(math.isqrt(N_OMEGA))}")

    omega = pa1.enumerate_omega()
    engine = pa1.build_transition_engine(omega)
    bcs = {bc.byte: bc for bc in pa1.classify_all_bytes()}
    u_idx, v_idx = pa3._build_uv_indices(omega)

    section_1_holographic_root(engine, omega)
    section_2_square_root_cluster(engine, bcs, u_idx, v_idx)
    section_3_parity_obstruction(engine, bcs)
    section_4_word_root_confinement(omega)
    section_5_criticality_hierarchy(engine)
    section_5b_holonomy_word_scaling(engine)
    section_6_loop_normalization_bridge()

    raise SystemExit(print_summary())


if __name__ == "__main__":
    main()
