#!/usr/bin/env python3
"""
hQVM Byte-Alphabet Percolation: CGM-grounded baseline study
==========================================================

ROLE
----
This is the BYTE-ALPHABET percolation baseline. It studies how observational
coherence propagates through Omega as a function of the available byte
transition alphabet, with all classification axes kept CGM-native (family,
K4 gate, q6 weight, phase-net, fold disagreement, micro_ref).

It is the FIRST of three complementary scripts:
  * hqvm_percolation_analysis_1.py  (this file) -- byte-alphabet baseline.
  * hqvm_percolation_analysis_2.py  -- canonical-word percolation theory.
  * hqvm_percolation_analysis_3.py  -- structural completeness supplement.
  * hqvm_percolation_analysis_run.py  -- runner for the full study (1, 2, 3).

Percolation in CGM is NOT the classical imposition of random edge-deletion
onto a pre-existing graph.  It is the study of how preservation of ancestry
propagates through Omega as a function of the available transition alphabet.

The transition structure IS the observational process.  The question is:
under what conditions on the available byte alphabet does the system achieve
global preservation of ancestry (constitutional spanning)?

All classification axes are CGM-native:
  - Family (K4 gauge sector: id, S, C, F)
  - Q6 chirality transport class (64 classes, 7 weight layers)
  - Phase-net vector (which of 4 CGM phases carry nonzero XOR across fold)
  - Connection 1-form chain (7 phase-boundary magnitudes)
  - Curvature 2-form at the BU fold
  - Fold disagreement (0..4 disagreeing phase pairs)
  - Micro_ref (6-bit payload -- the canonical CGM-native dial)

Percolation events tracked (weak -> strong):
  - span    : reach shell 6 from shell 0 (weak; trivially short-circuited)
  - hit_eq  : reach any equality-horizon state (weak)
  - full_omega : reachable set == all 4096 states (strong, discriminating)
The threshold inventory below reports p_c for the strong full_omega event.

Sections:
  I.    Manifold Construction and Byte Classification
  II.   Exact Deterministic Reachability (all structured restrictions)
  III.  Probabilistic Percolation Sweeps (Monte Carlo)
  IV.   Shell-Resolved Percolation Thresholds
  V.    Depth-Resolved Percolation
  VI.   K4 Gauge Closure and Percolation
  VII.  Curvature Chain Analysis
  VIII. Aperture Threshold Hypothesis Testing
  IX.   Holographic Identity Under Partial Alphabet
  X.    Null Model Comparison
  XI.   Horizon-Stabilizer Byte Percolation
  XII.  LI/FG/BG Payload-Axis Sweeps
  XIII. Shadow-Pair (q-Fiber) Percolation
  XIV.  Summary and Threshold Inventory

Interpretation of these results lives in
docs/Findings/Analysis_hQVM_Percolation.md, not in this script.
"""

from __future__ import annotations

import sys
import random
import itertools
from pathlib import Path
from collections import deque, Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from gyroscopic.hQVM.constants import (
    GENE_MAC_REST,
    HORIZON_GATE_BYTES,
    LAYER_MASK_12,
    byte_to_intron,
    byte_micro_ref,
    intron_family,
    is_on_equality_horizon,
    is_on_horizon,
    step_state_by_byte,
)

from gyroscopic.hQVM.api import (
    chirality_word6,
    omega12_to_state24,
    q_word6,
    state24_to_omega12,
)

from hqvm_wavefunction_kernel import (
    decompose_byte,
    fold_disagreement,
    compute_connection_chain,
    curvature_2form,
    classify_galois4_gate,
    PHASE_BOUNDARIES,
)

# Constants
M_A = 1.0 / (2.0 * np.sqrt(2.0 * np.pi))
DELTA_BU = 0.195342176580
RHO = DELTA_BU / M_A
DELTA = 1.0 - RHO
INV48 = 1.0 / 48.0
DYADIC_5_256 = 5.0 / 256.0

N_OMEGA = 4096
GIANT_THRESHOLD = N_OMEGA // 2  # 2048
MAX_BFS_DEPTH = 12  # enough for depth-8 holonomy cycle + margin

@dataclass(frozen=True)
class TransitionEngine:
    """Precomputed 4096 x 256 byte transition table for fast BFS."""
    transitions: List[List[int]]
    shell: List[int]
    start_idx: int
    state_to_idx: Dict[int, int]
    n: int

def build_transition_engine(omega: List[int]) -> TransitionEngine:
    print("  Precomputing transition table (4096 states x 256 bytes)...", flush=True)
    n = len(omega)
    state_to_idx = {s: i for i, s in enumerate(omega)}
    transitions = [[0] * 256 for _ in range(n)]
    shell = [0] * n
    for i, s in enumerate(omega):
        shell[i] = get_shell(s)
        row = transitions[i]
        for b in range(256):
            row[b] = state_to_idx[step_state_by_byte(s, b)]
    start_idx = state_to_idx[GENE_MAC_REST]
    print("  Transition table ready.", flush=True)
    return TransitionEngine(transitions, shell, start_idx, state_to_idx, n)

def build_shuffled_engine(engine: TransitionEngine, perm: List[int]) -> TransitionEngine:
    """Return a TransitionEngine where byte b's dynamics is swapped for
    byte perm[b]'s dynamics, while shell/start/state_to_idx (labels) are
    unchanged. Used for the corrected null model (Section X, Model 3):
    decouples classification labels from actual transition dynamics."""
    shuffled = [[row[perm[b]] for b in range(256)] for row in engine.transitions]
    return TransitionEngine(shuffled, engine.shell, engine.start_idx, engine.state_to_idx, engine.n)

def build_byte_partitions(
    bclass: List[ByteClassification],
) -> Dict[str, Dict]:
    """Group the 256 bytes by each CGM-native axis.

    micro_ref is the primary CGM-native dial (Gyroscopic_ASI_Specs_Formalism
    Sec. 2.2/4): the 6 payload bits (1-6) that carry all transformation
    content. Each of the 64 micro_refs has exactly one representative byte
    per family (4 bytes/micro_ref), so a micro_ref sweep always samples the
    full K4 gauge sector uniformly while dialing the payload content.
    """
    family_bytes: Dict[int, List[int]] = defaultdict(list)
    gate_bytes: Dict[str, List[int]] = defaultdict(list)
    q6_classes: Dict[int, List[int]] = defaultdict(list)
    micro_ref_bytes: Dict[int, List[int]] = defaultdict(list)
    for bc in bclass:
        family_bytes[bc.family].append(bc.byte)
        gate_bytes[bc.k4_gate].append(bc.byte)
        q6_classes[bc.q6].append(bc.byte)
        micro_ref_bytes[bc.micro_ref].append(bc.byte)
    return {
        "family": family_bytes,
        "gate": gate_bytes,
        "q6": q6_classes,
        "micro_ref": micro_ref_bytes,
    }

@dataclass(frozen=True)
class ByteClassification:
    """Complete CGM-native classification of a single byte."""
    byte: int
    intron: int
    family: int                  # 0..3 (K4 sector)
    k4_gate: str                 # 'id', 'S', 'C', 'F'
    micro_ref: int                # 6-bit payload (bits 1-6); the CGM-native dial
    q6: int                      # 6-bit chirality transport class
    q6_weight: int               # popcount(q6), 0..6
    fold_disagree: int           # 0..4
    phase_net: Tuple[int, int, int, int]  # XOR per phase (CS, UNA, ONA, BU)
    curvature_2form: float       # |F|^2 at BU fold
    connection_chain: Tuple[float, ...]    # 7 boundary magnitudes
    is_flat: bool                # fwd == rev (fold disagreement = 0)

def classify_all_bytes() -> List[ByteClassification]:
    """Classify all 256 bytes by every CGM-native axis."""
    result = []
    for b in range(256):
        bf = decompose_byte(b)
        chain = compute_connection_chain(b)
        curv = curvature_2form(b)
        chain_t = tuple(c.A_magnitude for c in chain)
        result.append(ByteClassification(
            byte=b,
            intron=bf.intron,
            family=bf.family,
            k4_gate=classify_galois4_gate(b),
            micro_ref=byte_micro_ref(b),
            q6=bf.q6,
            q6_weight=bf.q6.bit_count(),
            fold_disagree=fold_disagreement(b),
            phase_net=bf.phase_net,
            curvature_2form=curv,
            connection_chain=chain_t,
            is_flat=bf.is_flat,
        ))
    return result

def enumerate_omega() -> List[int]:
    """Enumerate all 4096 Omega states."""
    code: Set[int] = set()
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

def get_shell(state24: int) -> int:
    return chirality_word6(state24).bit_count()

def byte_from_family_micro(family: int, micro_ref: int) -> int:
    """Inverse of (intron_family, intron_micro_ref): construct the unique
    byte carrying the given (family, micro_ref) pair. See
    Gyroscopic_ASI_Specs_Formalism.md Sec 4 for the QuBEC decomposition."""
    bit7 = (family >> 1) & 1
    bit0 = family & 1
    intron = (bit7 << 7) | ((micro_ref & 0x3F) << 1) | bit0
    return intron ^ 0xAA

# I.5 K4 Algebraic Cross-Check (exact, exhaustive; grounds the percolation
#     study in the theorems already proved in Analysis_hQVM_Wavefunction.md)

def run_k4_algebraic_verification(engine: TransitionEngine) -> None:
    """Exhaustively re-verify, against this script's transition table,
    the K4 theorems (T1-T4, T10) that the percolation study depends on."""
    print("\n" + "=" * 5)
    print("I.5 K4 ALGEBRAIC CROSS-CHECK (exact, all 64 micro_refs x 4096 states)")
    print("=" * 5)

    transitions = engine.transitions
    shell = engine.shell
    n = engine.n

    t2_ok = t3_ok = t4_ok = involution_ok = closure_ok = True
    q_w2_vals: Set[int] = set()
    q_w2p_vals: Set[int] = set()

    for m in range(64):
        b00 = byte_from_family_micro(0, m)
        b01 = byte_from_family_micro(1, m)
        b10 = byte_from_family_micro(2, m)
        b11 = byte_from_family_micro(3, m)

        for i in range(n):
            # W2 = [fam00, fam01]
            j = transitions[i][b00]
            w2_i = transitions[j][b01]
            if shell[w2_i] != 6 - shell[i]:
                t2_ok = False

            # W2' = [fam10, fam11]
            j2 = transitions[i][b10]
            w2p_i = transitions[j2][b11]
            if shell[w2p_i] != 6 - shell[i]:
                t3_ok = False

            # F = W2 then W2' (canonical 4-byte word)
            jf1 = transitions[w2_i][b10]
            f_i = transitions[jf1][b11]
            if shell[f_i] != shell[i]:
                t4_ok = False

            # Involution: W2(W2(s)) == s
            j3 = transitions[w2_i][b00]
            w2_w2_i = transitions[j3][b01]
            if w2_w2_i != i:
                involution_ok = False

        # K4 closure sample: id.W2 == W2, W2.W2 == id (checked above),
        # W2.W2' should equal F on every state (composition order check)
        for i in range(0, n, 17):  # sparse but broad sample for composition check
            j = transitions[i][b00]
            w2_i = transitions[j][b01]
            jf1 = transitions[w2_i][b10]
            f_i = transitions[jf1][b11]
            j2 = transitions[i][b10]
            w2p_i = transitions[j2][b11]
            jf2 = transitions[w2p_i][b00]
            f_alt = transitions[jf2][b01]
            if f_i != f_alt:
                closure_ok = False

    print(f"\n  T2 (W2: shell -> 6-shell, all m, all states):    {'VERIFIED' if t2_ok else 'FAILED'}")
    print(f"  T3 (W2': shell -> 6-shell, all m, all states):   {'VERIFIED' if t3_ok else 'FAILED'}")
    print(f"  T4 (F preserves shell, all m, all states):       {'VERIFIED' if t4_ok else 'FAILED'}")
    print(f"  W2 involution (W2 o W2 = id, all m, all states): {'VERIFIED' if involution_ok else 'FAILED'}")
    print(f"  K4 commutation (W2.W2' = W2'.W2 = F, sampled):   {'VERIFIED' if closure_ok else 'FAILED'}")

@dataclass(frozen=True)
class ReachabilityResult:
    """Result of a reachability computation."""
    reachable: int                          # |reachable set|
    shells_reached: Tuple[int, ...]         # which shells are present
    spans: bool                             # shell 0 -> shell 6 path exists
    giant: bool                             # |reachable| >= 2048
    first_span_depth: Optional[int]         # depth at which spanning first occurs
    shell_first_depth: Dict[int, int]       # shell -> first depth reached
    full_omega: bool                        # reachable == 4096

def compute_reachability(
    engine: TransitionEngine,
    allowed_bytes: List[int],
    max_depth: int = MAX_BFS_DEPTH,
    track_depth: bool = True,
    start_indices: Optional[List[int]] = None,
) -> ReachabilityResult:
    """
    BFS using allowed_bytes and the precomputed transition table.
    Defaults to GENE_MAC_REST; pass start_indices for multi-source runs.
    """
    n = engine.n
    visited = bytearray(n)
    if start_indices is None:
        start_list = [engine.start_idx]
    else:
        start_list = start_indices

    frontier: List[int] = []
    visited_count = 0
    for si in start_list:
        if not visited[si]:
            visited[si] = 1
            visited_count += 1
            frontier.append(si)

    shell_first_depth: Dict[int, int] = {}
    shells_seen = bytearray(7)
    for si in start_list:
        sh0 = engine.shell[si]
        shells_seen[sh0] = 1
        if track_depth and sh0 not in shell_first_depth:
            shell_first_depth[sh0] = 0

    first_span_depth: Optional[int] = None
    if shells_seen[0] and shells_seen[6]:
        first_span_depth = 0

    transitions = engine.transitions
    shell = engine.shell
    allowed_tuple = tuple(allowed_bytes)

    for depth in range(1, max_depth + 1):
        if not frontier:
            break
        next_frontier: List[int] = []
        for si in frontier:
            row = transitions[si]
            for b in allowed_tuple:
                ni = row[b]
                if not visited[ni]:
                    visited[ni] = 1
                    visited_count += 1
                    next_frontier.append(ni)
                    sh = shell[ni]
                    if not shells_seen[sh]:
                        shells_seen[sh] = 1
                    if track_depth and sh not in shell_first_depth:
                        shell_first_depth[sh] = depth
        frontier = next_frontier

        if visited_count == n:
            break
        if first_span_depth is None and shells_seen[0] and shells_seen[6]:
            first_span_depth = depth

    shells = tuple(i for i in range(7) if shells_seen[i])
    spans = bool(shells_seen[0] and shells_seen[6])

    return ReachabilityResult(
        reachable=visited_count,
        shells_reached=shells,
        spans=spans,
        giant=visited_count >= GIANT_THRESHOLD,
        first_span_depth=first_span_depth,
        shell_first_depth=shell_first_depth,
        full_omega=visited_count == N_OMEGA,
    )

def reachable_mask(
    engine: TransitionEngine,
    allowed_bytes: List[int],
    start_indices: List[int],
    max_depth: int = MAX_BFS_DEPTH,
) -> bytearray:
    """Return visited bitmask for multi-source reachability (no shell metadata)."""
    n = engine.n
    visited = bytearray(n)
    frontier: List[int] = []
    for si in start_indices:
        if not visited[si]:
            visited[si] = 1
            frontier.append(si)
    transitions = engine.transitions
    allowed_tuple = tuple(allowed_bytes)
    for _ in range(max_depth):
        if not frontier:
            break
        next_frontier: List[int] = []
        for si in frontier:
            row = transitions[si]
            for b in allowed_tuple:
                ni = row[b]
                if not visited[ni]:
                    visited[ni] = 1
                    next_frontier.append(ni)
        frontier = next_frontier
    return visited

def complement_horizon_states(omega: List[int]) -> Set[int]:
    return {s for s in omega if is_on_horizon(s) and not is_on_equality_horizon(s)}

def equality_horizon_states(omega: List[int]) -> Set[int]:
    return {s for s in omega if is_on_equality_horizon(s)}


def horizon_indices(omega: List[int]) -> Tuple[List[int], List[int]]:
    """Return (complement_horizon, equality_horizon) state index lists."""
    comp: List[int] = []
    eq: List[int] = []
    for i, s in enumerate(omega):
        if is_on_equality_horizon(s):
            eq.append(i)
        elif is_on_horizon(s):
            comp.append(i)
    return comp, eq


# LI/FG/BG dipole-pair axes (Formalism Sec 2.1): micro_ref bit masks.
AXIS_MICRO_MASKS: Dict[str, int] = {
    "LI": 0x21,  # pairs 0,5  (intron bits 1,6)
    "FG": 0x12,  # pairs 1,4  (intron bits 2,5)
    "BG": 0x0C,  # pairs 2,3  (intron bits 3,4)
}


def axis_micro_ref_groups(micro_ref_bytes: Dict[int, List[int]], axis: str) -> List[List[int]]:
    """micro_ref buckets whose payload touches the given se(3) axis."""
    mask = AXIS_MICRO_MASKS[axis]
    return [micro_ref_bytes[m] for m in range(64) if m & mask]


@dataclass(frozen=True)
class HorizonConfinementResult:
    reachable: int
    horizon_confined: bool
    full_omega: bool


def compute_horizon_confinement(
    engine: TransitionEngine,
    omega: List[int],
    allowed_bytes: List[int],
    start_indices: List[int],
    on_horizon_pred,
    max_depth: int = MAX_BFS_DEPTH,
) -> HorizonConfinementResult:
    """Multi-source BFS; confined iff every visited state satisfies on_horizon_pred."""
    n = engine.n
    visited = bytearray(n)
    frontier: List[int] = []
    visited_count = 0
    for si in start_indices:
        if not visited[si]:
            if not on_horizon_pred(omega[si]):
                return HorizonConfinementResult(0, False, False)
            visited[si] = 1
            visited_count += 1
            frontier.append(si)

    transitions = engine.transitions
    allowed_tuple = tuple(allowed_bytes)
    for _ in range(max_depth):
        if not frontier:
            break
        next_frontier: List[int] = []
        for si in frontier:
            row = transitions[si]
            for b in allowed_tuple:
                ni = row[b]
                if not on_horizon_pred(omega[ni]):
                    return HorizonConfinementResult(visited_count, False, visited_count == n)
                if not visited[ni]:
                    visited[ni] = 1
                    visited_count += 1
                    next_frontier.append(ni)
        frontier = next_frontier
        if visited_count == n:
            break

    return HorizonConfinementResult(
        visited_count,
        True,
        visited_count == N_OMEGA,
    )


def _horizon_stabilizer_census(
    engine: TransitionEngine,
    omega: List[int],
    start_indices: List[int],
    on_horizon_pred,
) -> Dict[int, int]:
    """Per-byte count of start states that remain on the horizon after one step."""
    counts: Dict[int, int] = {}
    for b in range(256):
        ok = 0
        for idx in start_indices:
            ni = engine.transitions[idx][b]
            if on_horizon_pred(omega[ni]):
                ok += 1
        counts[b] = ok
    return counts


def _random_q_fiber_sweep(
    engine: TransitionEngine,
    q6_groups: List[List[int]],
    p_values: List[float],
    n_samples: int,
    mode: str,
) -> List[Tuple[float, float, float, float, float, int]]:
    """q-class sweep with shadow-pair inclusion modes."""
    sweep_data: List[Tuple[float, float, float, float, float, int]] = []
    for pi, p in enumerate(p_values):
        if pi % 10 == 0:
            print(f"  progress: q-fiber/{mode} {pi + 1}/{len(p_values)} (p={p:.3f})", flush=True)
        span_count = 0
        giant_count = 0
        full_count = 0
        total_reach = 0
        nz = 0
        for _ in range(n_samples):
            allowed: List[int] = []
            for group in q6_groups:
                if random.random() >= p:
                    continue
                if mode == "full_fiber":
                    allowed.extend(group)
                elif mode == "one_repr":
                    allowed.append(min(group))
                elif mode == "shadow_one":
                    picked: Set[int] = set()
                    for b in group:
                        lo = min(b, b ^ 0xFE)
                        if lo in picked:
                            continue
                        picked.add(lo)
                        allowed.append(random.choice([lo, lo ^ 0xFE]))
                else:
                    raise ValueError(mode)
            if not allowed:
                continue
            nz += 1
            r = compute_reachability(engine, allowed, track_depth=False)
            total_reach += r.reachable
            if r.spans:
                span_count += 1
            if r.giant:
                giant_count += 1
            if r.full_omega:
                full_count += 1
        sweep_data.append((
            p,
            span_count / n_samples,
            giant_count / n_samples,
            total_reach / max(nz, 1),
            full_count / n_samples,
            nz,
        ))
    return sweep_data

def run_deterministic_analysis(
    bclass: List[ByteClassification],
    engine: TransitionEngine,
    omega: List[int],
) -> None:
    """Exact reachability for all structured alphabet restrictions."""
    print("\n" + "=" * 5)
    print("II. EXACT DETERMINISTIC REACHABILITY")
    print("=" * 5)

    # --- A. Family restrictions ---
    print("\n--- A. Family Restrictions ---")
    family_bytes: Dict[int, List[int]] = defaultdict(list)
    for bc in bclass:
        family_bytes[bc.family].append(bc.byte)

    print(f"\n{'Alphabet':<25} {'|B|':>5} {'Reach':>6} {'Shells':>12} "
          f"{'Span':>5} {'Giant':>6} {'Full':>5} {'Span@D':>7}")
    print("-" * 5)

    family_singles: List[ReachabilityResult] = []
    for fam in range(4):
        allowed = family_bytes[fam]
        r = compute_reachability(engine, allowed)
        family_singles.append(r)
        _print_reach_row(f"Family {fam:02b}", len(allowed), r)

    pair_alloweds = [family_bytes[f1] + family_bytes[f2]
                     for f1, f2 in itertools.combinations(range(4), 2)]
    triple_alloweds = [family_bytes[f1] + family_bytes[f2] + family_bytes[f3]
                       for f1, f2, f3 in itertools.combinations(range(4), 3)]
    ref_fam = family_singles[0]
    if _combo_results_uniform(engine, pair_alloweds + triple_alloweds, ref_fam):
        print(f"  {'all pairs/triples (10 combos)':<25} "
              f"{'':>5} {ref_fam.reachable:6d} "
              f"{str(ref_fam.shells_reached):>12} {str(ref_fam.spans):>5} "
              f"{str(ref_fam.giant):>6} {str(ref_fam.full_omega):>5} "
              f"{'(same)':>7}")
    else:
        for f1, f2 in itertools.combinations(range(4), 2):
            allowed = family_bytes[f1] + family_bytes[f2]
            _print_reach_row(f"Families {f1:02b}+{f2:02b}", len(allowed),
                             compute_reachability(engine, allowed))

    allowed_all = list(range(256))
    _print_reach_row("All families", len(allowed_all),
                     compute_reachability(engine, allowed_all))

    # --- B. K4 gate restrictions ---
    print("\n--- B. K4 Gate Restrictions ---")
    gate_bytes: Dict[str, List[int]] = defaultdict(list)
    for bc in bclass:
        gate_bytes[bc.k4_gate].append(bc.byte)

    print(f"\n{'Alphabet':<25} {'|B|':>5} {'Reach':>6} {'Shells':>12} "
          f"{'Span':>5} {'Giant':>6} {'Full':>5} {'Span@D':>7}")
    print("-" * 5)

    gate_singles: List[ReachabilityResult] = []
    for gate in ['id', 'S', 'C', 'F']:
        allowed = gate_bytes[gate]
        r = compute_reachability(engine, allowed)
        gate_singles.append(r)
        _print_reach_row(f"Gate {gate} only", len(allowed), r)

    pair_alloweds_g = [gate_bytes[g1] + gate_bytes[g2]
                       for g1, g2 in itertools.combinations(['id', 'S', 'C', 'F'], 2)]
    ref_gate = gate_singles[0]
    if _combo_results_uniform(engine, pair_alloweds_g, ref_gate):
        print(f"  {'all gate pairs (6 combos)':<25} "
              f"{'':>5} {ref_gate.reachable:6d} "
              f"{str(ref_gate.shells_reached):>12} {str(ref_gate.spans):>5} "
              f"{str(ref_gate.giant):>6} {str(ref_gate.full_omega):>5} "
              f"{'(same)':>7}")
    else:
        for g1, g2 in itertools.combinations(['id', 'S', 'C', 'F'], 2):
            allowed = gate_bytes[g1] + gate_bytes[g2]
            _print_reach_row(f"Gates {g1}+{g2}", len(allowed),
                             compute_reachability(engine, allowed))

    for label, gates in [("id+F (preserve)", ['id', 'F']), ("S+C (swap)", ['S', 'C'])]:
        allowed = gate_bytes[gates[0]] + gate_bytes[gates[1]]
        _print_reach_row(label, len(allowed), compute_reachability(engine, allowed))

    # --- C. Q6 weight restrictions ---
    print("\n--- C. Q6 Weight Layer Restrictions ---")
    qw_bytes: Dict[int, List[int]] = defaultdict(list)
    for bc in bclass:
        qw_bytes[bc.q6_weight].append(bc.byte)

    print(f"\n{'Alphabet':<25} {'|B|':>5} {'Reach':>6} {'Shells':>12} "
          f"{'Span':>5} {'Giant':>6} {'Full':>5} {'Span@D':>7}")
    print("-" * 5)

    # Single layers
    for w in range(7):
        allowed = qw_bytes[w]
        r = compute_reachability(engine, allowed)
        sd = str(r.first_span_depth) if r.first_span_depth is not None else "-"
        print(f"  Q6 weight={w} only      {len(allowed):5d} {r.reachable:6d} "
              f"{str(r.shells_reached):>12} {str(r.spans):>5} {str(r.giant):>6} "
              f"{str(r.full_omega):>5} {sd:>7}")

    # Cumulative
    print()
    cumul: List[int] = []
    for w in range(7):
        cumul.extend(qw_bytes[w])
        r = compute_reachability(engine, cumul)
        sd = str(r.first_span_depth) if r.first_span_depth is not None else "-"
        print(f"  Q6 weight <= {w}        {len(cumul):5d} {r.reachable:6d} "
              f"{str(r.shells_reached):>12} {str(r.spans):>5} {str(r.giant):>6} "
              f"{str(r.full_omega):>5} {sd:>7}")

    # --- D. Fold disagreement restrictions ---
    print("\n--- D. Fold Disagreement Restrictions ---")
    fd_bytes: Dict[int, List[int]] = defaultdict(list)
    for bc in bclass:
        fd_bytes[bc.fold_disagree].append(bc.byte)

    print(f"\n{'Alphabet':<25} {'|B|':>5} {'Reach':>6} {'Shells':>12} "
          f"{'Span':>5} {'Giant':>6} {'Full':>5} {'Span@D':>7}")
    print("-" * 5)

    for d in range(5):
        allowed = fd_bytes[d]
        r = compute_reachability(engine, allowed)
        sd = str(r.first_span_depth) if r.first_span_depth is not None else "-"
        print(f"  Fold disagree={d} only   {len(allowed):5d} {r.reachable:6d} "
              f"{str(r.shells_reached):>12} {str(r.spans):>5} {str(r.giant):>6} "
              f"{str(r.full_omega):>5} {sd:>7}")

    print()
    cumul_fd: List[int] = []
    for d in range(5):
        cumul_fd.extend(fd_bytes[d])
        r = compute_reachability(engine, cumul_fd)
        sd = str(r.first_span_depth) if r.first_span_depth is not None else "-"
        print(f"  Fold disagree <= {d}     {len(cumul_fd):5d} {r.reachable:6d} "
              f"{str(r.shells_reached):>12} {str(r.spans):>5} {str(r.giant):>6} "
              f"{str(r.full_omega):>5} {sd:>7}")

    # --- E. Curvature 2-form restricted ---
    print("\n--- E. Curvature 2-Form Magnitude Restrictions ---")
    curv_bytes: Dict[float, List[int]] = defaultdict(list)
    for bc in bclass:
        # Round to avoid float key issues
        key = round(bc.curvature_2form, 6)
        curv_bytes[key].append(bc.byte)

    print(f"\n{'|F|^2 at fold':<18} {'|B|':>5} {'Reach':>6} {'Shells':>12} "
          f"{'Span':>5} {'Giant':>6} {'Full':>5}")
    print("-" * 5)
    for cv in sorted(curv_bytes.keys()):
        allowed = curv_bytes[cv]
        r = compute_reachability(engine, allowed)
        print(f"  {cv:<16.6f} {len(allowed):5d} {r.reachable:6d} "
              f"{str(r.shells_reached):>12} {str(r.spans):>5} {str(r.giant):>6} "
              f"{str(r.full_omega):>5}")

    
    # before each cumulative reachability test (fixes premature-print bug
    # where only the first byte of a new tier was included).
    print("\n  Cumulative (ascending |F|^2):")
    curv_groups: Dict[float, List[int]] = defaultdict(list)
    for bc in bclass:
        curv_groups[round(bc.curvature_2form, 6)].append(bc.byte)
    cumul_curv: List[int] = []
    for cv in sorted(curv_groups.keys()):
        cumul_curv.extend(curv_groups[cv])
        r = compute_reachability(engine, cumul_curv)
        print(f"    |F|^2 <= {cv:<12.6f}  {len(cumul_curv):5d} {r.reachable:6d} "
              f"{str(r.shells_reached):>12} {str(r.spans):>5} {str(r.giant):>6} "
              f"{str(r.full_omega):>5}")

    # --- F. Phase-net restrictions ---
    print("\n--- F. Phase-Net Vector Restrictions ---")
    print("  Phase-net = (CS_xor, UNA_xor, ONA_xor, BU_xor) across fold")
    pn_bytes: Dict[Tuple[int, int, int, int], List[int]] = defaultdict(list)
    for bc in bclass:
        pn_bytes[bc.phase_net].append(bc.byte)

    print(f"\n{'Phase-net':<20} {'|B|':>5} {'Reach':>6} {'Shells':>12} "
          f"{'Span':>5} {'Giant':>6}")
    print("-" * 5)
    for pn in sorted(pn_bytes.keys()):
        allowed = pn_bytes[pn]
        r = compute_reachability(engine, allowed)
        print(f"  {str(pn):<20s} {len(allowed):5d} {r.reachable:6d} "
              f"{str(r.shells_reached):>12} {str(r.spans):>5} {str(r.giant):>6}")

    # Phase-net cumulative: require each phase to be "active" (net=1)
    print("\n  Cumulative by active phases (net=1 in that phase):")
    for n_active in range(5):
        allowed = [bc.byte for bc in bclass
                    if sum(bc.phase_net) >= n_active]
        r = compute_reachability(engine, allowed)
        print(f"    >= {n_active} active phases: {len(allowed):5d} bytes  "
              f"Reach={r.reachable:6d}  Span={r.spans}  Giant={r.giant}")

    # --- G. Connection 1-form boundary restrictions ---
    print("\n--- G. Connection 1-Form Boundary Restrictions ---")
    print("  Testing which phase boundaries are essential for spanning.")
    for bi, bname in enumerate(PHASE_BOUNDARIES):
        # Bytes with nonzero A at this boundary
        allowed_nz = [bc.byte for bc in bclass if bc.connection_chain[bi] > 0]
        allowed_z = [bc.byte for bc in bclass if bc.connection_chain[bi] == 0]
        r_nz = compute_reachability(engine, allowed_nz)
        r_z = compute_reachability(engine, allowed_z)
        print(f"  {bname:8s}:  |A|>0 -> {len(allowed_nz):4d}B  "
              f"Reach={r_nz.reachable:5d} Span={str(r_nz.spans):>5}  |  "
              f"|A|=0 -> {len(allowed_z):4d}B  Reach={r_z.reachable:5d} Span={str(r_z.spans):>5}")

    # --- H. Horizon-to-horizon coverage under byte edges (baseline) ---
    # Byte-graph diagnostic only. NOT the W2 word-level pole pairing (T9);
    # see hqvm_percolation_analysis_2.py for canonical-word pairing.
    print("\n--- H. Horizon-to-horizon coverage (byte edges, baseline) ---")
    print("  From ALL 64 complement-horizon states: |eq hit| and full_omega")
    print("  under byte transitions (expected trivial for most restrictions).")
    comp_hor = complement_horizon_states(omega)
    eq_hor = equality_horizon_states(omega)
    comp_idx = [engine.state_to_idx[s] for s in comp_hor]
    eq_idx_set = set(engine.state_to_idx[s] for s in eq_hor)

    h2h_restrictions: Dict[str, List[int]] = {
        "All bytes": list(range(256)),
        "Family 00 only": family_bytes[0],
        "Family 01 only": family_bytes[1],
        "Family 10 only": family_bytes[2],
        "Family 11 only": family_bytes[3],
        "Gate id+F (shell-preserving)": [bc.byte for bc in bclass if bc.k4_gate in ('id', 'F')],
        "Gate S+C (pole-swap)": [bc.byte for bc in bclass if bc.k4_gate in ('S', 'C')],
    }
    print(f"\n  {'Restriction':<32} {'|B|':>5} {'Reach':>6} {'|Eq hit|':>9} {'Eq cover':>9} {'full_O':>7}")
    print("  " + "-" * 5)
    for label, allowed in h2h_restrictions.items():
        mask = reachable_mask(engine, allowed, comp_idx)
        reach = sum(mask)
        eq_hit = sum(1 for i in eq_idx_set if mask[i])
        full_pairing = eq_hit == 64
        print(f"  {label:<32} {len(allowed):5d} {reach:6d} {eq_hit:9d} "
              f"{str(full_pairing):>13} {str(reach == N_OMEGA):>11}")

def _print_reach_row(
    label: str,
    nbytes: int,
    r: ReachabilityResult,
    width: int = 25,
) -> None:
    sd = str(r.first_span_depth) if r.first_span_depth is not None else "-"
    print(f"  {label:<{width}} {nbytes:5d} {r.reachable:6d} "
          f"{str(r.shells_reached):>12} {str(r.spans):>5} {str(r.giant):>6} "
          f"{str(r.full_omega):>5} {sd:>7}")


def _combo_results_uniform(
    engine: TransitionEngine,
    combo_alloweds: List[List[int]],
    ref: ReachabilityResult,
) -> bool:
    for allowed in combo_alloweds:
        r = compute_reachability(engine, allowed)
        if (r.reachable, r.full_omega, r.spans, r.giant) != (
            ref.reachable, ref.full_omega, ref.spans, ref.giant
        ):
            return False
    return True


def _random_subset_sweep(
    engine: TransitionEngine,
    groups: List[List[int]],
    p_values: List[float],
    n_samples: int,
    label: str,
) -> List[Tuple[float, float, float, float, float, int]]:
    """p-sweep with conditional stats. Tuple ends with nonzero trial count."""
    sweep_data: List[Tuple[float, float, float, float, float, int]] = []
    n_p = len(p_values)
    for pi, p in enumerate(p_values):
        if pi % 10 == 0:
            print(f"  progress: {label} {pi + 1}/{n_p} (p={p:.3f})", flush=True)
        span_count = 0
        giant_count = 0
        full_count = 0
        total_reach = 0
        nz = 0
        for _ in range(n_samples):
            allowed: List[int] = []
            for g in groups:
                if random.random() < p:
                    allowed.extend(g)
            if not allowed:
                continue
            nz += 1
            r = compute_reachability(engine, allowed, track_depth=False)
            total_reach += r.reachable
            if r.spans:
                span_count += 1
            if r.giant:
                giant_count += 1
            if r.full_omega:
                full_count += 1
        sweep_data.append((
            p,
            span_count / n_samples,
            giant_count / n_samples,
            total_reach / max(nz, 1),
            full_count / n_samples,
            nz,
        ))
    return sweep_data


def _find_crossing(data: List[Tuple[float, ...]], col: int) -> Optional[float]:
    """Linear-interpolated p where column `col` crosses 0.5."""
    for i in range(len(data) - 1):
        row1, row2 = data[i], data[i + 1]
        y1, y2 = row1[col], row2[col]
        if y1 < 0.5 <= y2:
            frac = (0.5 - y1) / (y2 - y1) if y2 > y1 else 0.5
            return row1[0] + frac * (row2[0] - row1[0])
    return None


def _compact_sweep_indices(
    data: List[Tuple[float, ...]],
    col_full: int = 4,
) -> List[int]:
    """Row indices for compact console tables: endpoints, transition band, plateau."""
    n = len(data)
    if n == 0:
        return []
    keep = {0, n - 1}
    for i, row in enumerate(data):
        p = row[0]
        if p <= 0.0:
            continue
        if p <= 0.25 and abs(round(p * 100) - p * 100) < 1e-6:
            keep.add(i)
        elif p > 0.25 and abs(round(p * 20) - p * 20) < 1e-6:
            keep.add(i)
    p_c = _find_crossing(data, col_full)
    if p_c is not None:
        for i, row in enumerate(data):
            if abs(row[0] - p_c) <= 0.025:
                keep.add(i)
    plateau_i = next(
        (i for i in range(1, n) if data[i][col_full] >= 0.999 and data[i - 1][col_full] < 0.999),
        None,
    )
    if plateau_i is not None:
        keep.add(plateau_i)
    return sorted(keep)


def _print_sweep_table(
    sweep_data: List[Tuple[float, float, float, float, float, int]],
    n_samples: int,
) -> None:
    """Compact sweep table: unconditional + conditional on nonempty alphabet."""
    print(f"\n  {'p':<7} {'P(full)':<9} {'P(f|nz)':<9} {'P(nz)':<8} {'<R|nz>':<8}")
    print("  " + "-" * 5)
    indices = _compact_sweep_indices(sweep_data)
    prev_i = -2
    for i in indices:
        if i - prev_i > 1:
            print("  ...")
        row = sweep_data[i]
        p, _, _, reach_c, pf, nz = row
        pf_nz = pf * n_samples / nz if nz else 0.0
        print(f"  {p:<7.3f} {pf:<9.4f} {pf_nz:<9.4f} {nz / n_samples:<8.3f} {reach_c:<8.1f}")
        prev_i = i
    p_c = _find_crossing(sweep_data, 4)
    if p_c is not None:
        print(f"  p_c(P(full)) ~ {p_c:.4f}")
    plateau_i = next(
        (i for i in range(1, len(sweep_data))
         if sweep_data[i][4] >= 0.999 and sweep_data[i - 1][4] < 0.999),
        None,
    )
    if plateau_i is not None and plateau_i not in indices:
        p0, _, _, _, pf0, _ = sweep_data[plateau_i]
        omitted = len(sweep_data) - plateau_i - 1
        if omitted > 0:
            print(f"  ... p >= {p0:.3f}: P(full) = {pf0:.4f} ({omitted} plateau rows omitted)")
    omitted_total = len(sweep_data) - len(indices)
    if omitted_total > 0:
        print(f"  ({omitted_total} of {len(sweep_data)} p-values omitted; see p_c above)")

def run_probabilistic_sweeps(
    bclass: List[ByteClassification],
    engine: TransitionEngine,
    partitions: Dict[str, Dict],
    n_samples: int = 500,
) -> Dict[str, List[Tuple[float, float, float, float, float, int]]]:
    """Monte Carlo percolation sweeps for each restriction axis.

    Two percolation events are tracked at every p:
      - hit_equality ("P(span)" in the tables): from rest (shell 6), did we
        reach ANY equality-horizon state (shell 0)? This is the WEAK event;
        since rest already IS shell 6, it reduces to "non-empty selection
        reached shell 0", which is easy and not a deep geometric threshold
        for gauge-only restrictions (family/gate).
      - full_omega ("P(full)"): did we reach ALL 4096 states? This is the
        STRONG, discriminating percolation event that actually probes
        CGM structure (curvature, q6 weight, payload content).
    Both are reported; do not conflate them when reading thresholds.
    """
    print("\n" + "=" * 5)
    print("III. PROBABILISTIC PERCOLATION SWEEPS")
    print("=" * 5)
    print("\n  P(full)=unconditional; P(f|nz)=conditional on nonempty alphabet.")

    results: Dict[str, List[Tuple[float, float, float, float, float, int]]] = {}
    family_bytes: Dict[int, List[int]] = partitions["family"]
    q6_classes: Dict[int, List[int]] = partitions["q6"]
    micro_ref_bytes: Dict[int, List[int]] = partitions["micro_ref"]
    p_values = sorted(set(
        [0.0] + [i / 100 for i in range(1, 31)] + [i / 10 for i in range(4, 11)]
    ))

    # --- A. Family fraction sweep (4 groups: pure gauge dial) ---
    print("\n--- A. Family Fraction Sweep ---")
    print("  p = probability of including each of the 4 families (gauge only)")
    sweep_data = _random_subset_sweep(
        engine, [family_bytes[f] for f in range(4)], p_values, n_samples, "family")
    results['family'] = sweep_data
    _print_sweep_table(sweep_data, n_samples)

    # --- B. Full Byte Fraction Sweep ---
    print("\n--- B. Full Byte Fraction Sweep ---")
    print("  p = probability of including each of the 256 bytes")
    sweep_data_b = _random_subset_sweep(
        engine, [[b] for b in range(256)], p_values, n_samples, "byte")
    results['byte'] = sweep_data_b
    _print_sweep_table(sweep_data_b, n_samples)

    # --- C. Q6 Class Fraction Sweep ---
    print("\n--- C. Q6 Class Fraction Sweep ---")
    print("  p = probability of including each of the 64 q6-transport classes")
    sweep_data_q = _random_subset_sweep(
        engine, list(q6_classes.values()), p_values, n_samples, "q6")
    results['q6_class'] = sweep_data_q
    _print_sweep_table(sweep_data_q, n_samples)

    # --- D. Micro-Ref (6-bit Payload) Fraction Sweep ---
    print("\n--- D. Micro-Ref (6-bit Payload) Fraction Sweep ---")
    print("  p = probability of including each of the 64 micro_refs")
    print("  (each micro_ref carries all 4 families/gauge phases together)")
    sweep_data_m = _random_subset_sweep(
        engine, list(micro_ref_bytes.values()), p_values, n_samples, "micro_ref")
    results['micro_ref'] = sweep_data_m
    _print_sweep_table(sweep_data_m, n_samples)

    return results

def run_shell_resolved(
    bclass: List[ByteClassification],
    engine: TransitionEngine,
    n_samples: int = 300,
) -> None:
    """Per-shell percolation: probability of reaching each shell vs p."""
    print("\n" + "=" * 5)
    print("IV. SHELL-RESOLVED PERCOLATION THRESHOLDS")
    print("=" * 5)

    all_bytes = list(range(256))
    p_values = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1,
                0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

    print(f"\n  {'p':<8}", end="")
    for s in range(7):
        print(f" {'Shell'+str(s):>8}", end="")
    print()
    print("  " + "-" * 5)

    for pi, p in enumerate(p_values):
        if pi % 3 == 0:
            print(f"  progress: shell sweep {pi + 1}/{len(p_values)} (p={p:.3f})", flush=True)
        shell_probs = [0.0] * 7
        for _ in range(n_samples):
            allowed = [b for b in all_bytes if random.random() < p]
            if not allowed:
                continue
            r = compute_reachability(engine, allowed, track_depth=False)
            for s in range(7):
                if s in r.shells_reached:
                    shell_probs[s] += 1
        print(f"  {p:<8.3f}", end="")
        for s in range(7):
            print(f" {shell_probs[s]/n_samples:>8.3f}", end="")
        print()

def run_depth_resolved(
    bclass: List[ByteClassification],
    engine: TransitionEngine,
) -> None:
    """At what depth does spanning first occur for each alphabet restriction?"""
    print("\n" + "=" * 5)
    print("V. DEPTH-RESOLVED PERCOLATION")
    print("=" * 5)

    # Test key alphabet restrictions at increasing depth
    restrictions = {
        "All bytes": list(range(256)),
        "Families 00+01": [bc.byte for bc in bclass if bc.family in (0, 1)],
        "Families 10+11": [bc.byte for bc in bclass if bc.family in (2, 3)],
        "Families 00+10": [bc.byte for bc in bclass if bc.family in (0, 2)],
        "Families 01+11": [bc.byte for bc in bclass if bc.family in (1, 3)],
        "Gate id+F": [bc.byte for bc in bclass if bc.k4_gate in ('id', 'F')],
        "Gate S+C": [bc.byte for bc in bclass if bc.k4_gate in ('S', 'C')],
        "Q6 weight >= 3": [bc.byte for bc in bclass if bc.q6_weight >= 3],
        "Fold disagree >= 2": [bc.byte for bc in bclass if bc.fold_disagree >= 2],
    }

    print(f"\n  {'Restriction':<25} {'|B|':>5}", end="")
    for d in range(1, 13):
        print(f" {'D='+str(d):>5}", end="")
    print()
    print("  " + "-" * 5)

    for label, allowed in restrictions.items():
        visited = bytearray(engine.n)
        visited[engine.start_idx] = 1
        frontier = [engine.start_idx]
        span_depth = None
        allowed_tuple = tuple(allowed)
        transitions = engine.transitions
        shell = engine.shell

        print(f"  {label:<25} {len(allowed):5d}", end="")

        for depth in range(1, 13):
            next_frontier: List[int] = []
            for si in frontier:
                row = transitions[si]
                for b in allowed_tuple:
                    ni = row[b]
                    if not visited[ni]:
                        visited[ni] = 1
                        next_frontier.append(ni)
            frontier = next_frontier

            shells_seen = bytearray(7)
            for si in range(engine.n):
                if visited[si]:
                    shells_seen[shell[si]] = 1
            has_span = shells_seen[0] and shells_seen[6]
            if has_span and span_depth is None:
                span_depth = depth

            marker = "S" if (has_span and span_depth == depth) else (
                "." if not has_span else "s")
            print(f" {marker:>5}", end="")

        print(f"  |  Span@D={span_depth}")

def run_k4_analysis(
    bclass: List[ByteClassification],
    engine: TransitionEngine,
) -> None:
    """Test whether K4 gauge closure is required for constitutional spanning."""
    print("\n" + "=" * 5)
    print("VI. K4 GAUGE CLOSURE AND PERCOLATION")
    print("=" * 5)

    gate_bytes: Dict[str, List[int]] = defaultdict(list)
    for bc in bclass:
        gate_bytes[bc.k4_gate].append(bc.byte)

    print("\n  Theorem T1: {id, W2, W2', F} is K4 for every micro-ref m.")
    print("  Theorem T2/T3: W2 and W2' swap constitutional poles (shell s -> 6-s).")
    print("  Theorem T4: F preserves shell (Z2 carrier flip within pole).")
    print()
    print("  Key question: Does spanning REQUIRE at least one pole-swap gate?")

    # Test: only shell-preserving gates (id + F)
    allowed_preserve = gate_bytes['id'] + gate_bytes['F']
    r_preserve = compute_reachability(engine, allowed_preserve)

    # Test: only pole-swap gates (S + C)
    allowed_swap = gate_bytes['S'] + gate_bytes['C']
    r_swap = compute_reachability(engine, allowed_swap)

    print(f"\n  Shell-preserving only (id+F): {len(allowed_preserve)} bytes")
    print(f"    Reachable: {r_preserve.reachable}  Shells: {r_preserve.shells_reached}")
    print(f"    Spans: {r_preserve.spans}  Giant: {r_preserve.giant}")

    print(f"\n  Pole-swap only (S+C): {len(allowed_swap)} bytes")
    print(f"    Reachable: {r_swap.reachable}  Shells: {r_swap.shells_reached}")
    print(f"    Spans: {r_swap.spans}  Giant: {r_swap.giant}")

    # Test minimum gate combination for spanning
    print("\n  Minimum gate combinations for spanning:")
    for n_gates in range(1, 5):
        for combo in itertools.combinations(['id', 'S', 'C', 'F'], n_gates):
            allowed = []
            for g in combo:
                allowed.extend(gate_bytes[g])
            r = compute_reachability(engine, allowed)
            if r.spans:
                print(f"    {combo}: SPANS (|B|={len(allowed)}, "
                      f"Reach={r.reachable}, Span@D={r.first_span_depth})")

    # Test: within each family pair, which produces W2?
    print("\n  W2 structure by family pair:")
    print("  W2 = [fam00, fam01], W2' = [fam10, fam11] (canonical ordering)")
    for f1, f2 in itertools.combinations(range(4), 2):
        allowed = [bc.byte for bc in bclass if bc.family in (f1, f2)]
        r = compute_reachability(engine, allowed)
        # Check if this pair acts as a pole-swap involution
        print(f"    Families {f1:02b}+{f2:02b}: Reach={r.reachable}  "
              f"Span={r.spans}  Giant={r.giant}  Span@D={r.first_span_depth}")

def run_curvature_analysis(
    bclass: List[ByteClassification],
    engine: TransitionEngine,
) -> None:
    """Analyze percolation restricted by connection 1-form structure."""
    print("\n" + "=" * 5)
    print("VII. CURVATURE CHAIN ANALYSIS")
    print("=" * 5)

    # Group bytes by their full connection chain signature
    chain_sigs: Dict[Tuple[float, ...], List[int]] = defaultdict(list)
    for bc in bclass:
        chain_sigs[bc.connection_chain].append(bc.byte)

    print(f"\n  Distinct connection chain signatures: {len(chain_sigs)}")

    # Test: which individual boundaries are necessary for spanning?
    print("\n  Boundary necessity test (exclude bytes with |A|>0 at each boundary):")
    for bi, bname in enumerate(PHASE_BOUNDARIES):
        # Exclude bytes with nonzero A at this boundary
        allowed_excl = [bc.byte for bc in bclass if bc.connection_chain[bi] == 0]
        r = compute_reachability(engine, allowed_excl)
        print(f"    Exclude {bname:8s}: {len(allowed_excl):4d}B remaining  "
              f"Reach={r.reachable:5d}  Span={r.spans}")

    # Test: bytes with curvature ONLY at the fold (BU|BU)
    print("\n  Bytes with curvature ONLY at the BU|BU fold:")
    fold_only = [bc.byte for bc in bclass
                  if bc.connection_chain[3] > 0 and
                  sum(1 for x in bc.connection_chain if x > 0) == 1]
    r_fo = compute_reachability(engine, fold_only)
    print(f"    Count: {len(fold_only)}  Reach={r_fo.reachable}  Span={r_fo.spans}")

    # Test: bytes with NO curvature at the fold
    print("\n  Bytes with zero curvature at the BU|BU fold:")
    no_fold = [bc.byte for bc in bclass if bc.connection_chain[3] == 0]
    r_nf = compute_reachability(engine, no_fold)
    print(f"    Count: {len(no_fold)}  Reach={r_nf.reachable}  Span={r_nf.spans}")

    # Test: curvature 2-form distribution vs spanning
    print("\n  Curvature 2-form magnitude vs spanning capability:")
    curv_vals = sorted(set(round(bc.curvature_2form, 6) for bc in bclass))
    for cv in curv_vals:
        with_cv = [bc.byte for bc in bclass if round(bc.curvature_2form, 6) >= cv]
        r = compute_reachability(engine, with_cv)
        print(f"    |F|^2 >= {cv:.6f}: {len(with_cv):4d}B  "
              f"Reach={r.reachable:5d}  Span={r.spans}")

N_GROUPS_PER_SWEEP = {"family": 4, "byte": 256, "q6_class": 64, "micro_ref": 64}


def run_aperture_hypothesis(
    sweep_results: Dict[str, List[Tuple[float, float, float, float, float, int]]],
) -> None:
    """Compare discovered percolation thresholds to CGM constants.

    Reports BOTH the weak event (hit_equality, column 1 = "P(span)") and the
    strong event (full_omega, column 4 = "P(full)"). The weak event is
    expected to be structurally trivial for gauge-only axes (family, gate):
    p_c there is essentially the solution of 1-(1-p)^k = 0.5 for k groups
    (a combinatorial "non-empty selection" effect, not deep CGM geometry).
    The strong event (full_omega) is the discriminating one.
    """
    print("\n" + "=" * 5)
    print("VIII. APERTURE THRESHOLD HYPOTHESIS TESTING")
    print("=" * 5)

    cgm_constants = {
        "Delta (aperture gap)": DELTA,
        "1/48 (geometric quantization)": INV48,
        "5/256 (dyadic approximant)": DYADIC_5_256,
        "6/256 (6-generator dyadic)": 6.0 / 256.0,
        "1 - rho": 1 - RHO,
        "m_a": M_A,
        "rho * Delta": RHO * DELTA,
        "Delta^2": DELTA ** 2,
    }

    print("\n  CGM constants (candidate threshold scales):")
    for name, val in cgm_constants.items():
        print(f"    {name:<35s} = {val:.10f}")

    print("\n  Non-empty-selection null (k = number of independent groups):")
    print("  p_null solves 1-(1-p)^k = 0.5  =>  p_null = 1 - 0.5^(1/k)")
    for sweep_name, k in N_GROUPS_PER_SWEEP.items():
        p_null = 1 - 0.5 ** (1.0 / k)
        print(f"    {sweep_name:<12s} (k={k:3d}): p_null = {p_null:.6f}")

    for event_name, col in (("hit_equality (weak, P(span))", 1), ("full_omega (strong, P(full))", 4)):
        print(f"\n  --- Thresholds for event: {event_name} ---")
        for sweep_name, data in sweep_results.items():
            p_c = _find_crossing(data, col)
            if p_c is not None:
                k = N_GROUPS_PER_SWEEP.get(sweep_name, 1)
                n_gen = k * p_c
                print(f"\n    {sweep_name} sweep:  p_c ~ {p_c:.6f}  "
                      f"(expected included groups at p_c: {n_gen:.2f} of {k})")
                for name, val in cgm_constants.items():
                    ratio = p_c / val if val > 0 else float('inf')
                    print(f"        p_c / {name:<30s} = {ratio:.6f}")
            else:
                print(f"\n    {sweep_name} sweep:  no threshold detected in range")

def run_holographic_test(
    bclass: List[ByteClassification],
    omega: List[int],
    engine: TransitionEngine,
    n_samples: int = 200,
) -> None:
    """Test whether |H_eff|^2 ~ |Omega_reachable| under partial alphabet."""
    print("\n" + "=" * 5)
    print("IX. HOLOGRAPHIC IDENTITY UNDER PARTIAL ALPHABET")
    print("=" * 5)

    all_bytes = list(range(256))
    comp_hor = complement_horizon_states(omega)
    comp_hor_idx = [engine.state_to_idx[s] for s in comp_hor]
    eq_hor = equality_horizon_states(omega)

    print(f"\n  Full Omega: |Omega| = {len(omega)}")
    print(f"  |H_comp| = {len(comp_hor)}, |H_eq| = {len(eq_hor)}")
    print(f"  |H|^2 = {len(comp_hor)**2} (should equal {len(omega)})")
    print()

    p_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    print(f"  {'p':<8} {'<|Omega_r|>':<12} {'<|H_comp_r|>':<14} "
          f"{'<|H_comp_r|^2>':<16} {'Ratio':<10}")
    print("  " + "-" * 5)

    for p in p_values:
        total_omega_r = 0
        total_h_comp_r = 0
        total_h_sq = 0

        for _ in range(n_samples):
            allowed = [b for b in all_bytes if random.random() < p]
            if not allowed:
                continue

            visited = reachable_mask(engine, allowed, comp_hor_idx)
            omega_r = sum(visited)
            h_comp_r = sum(1 for si in comp_hor_idx if visited[si])
            total_omega_r += omega_r
            total_h_comp_r += h_comp_r
            total_h_sq += h_comp_r ** 2

        avg_omega_r = total_omega_r / n_samples
        avg_h_comp_r = total_h_comp_r / n_samples
        avg_h_sq = total_h_sq / n_samples
        ratio = avg_h_sq / avg_omega_r if avg_omega_r > 0 else 0

        print(f"  {p:<8.3f} {avg_omega_r:<12.1f} {avg_h_comp_r:<14.1f} "
              f"{avg_h_sq:<16.1f} {ratio:<10.4f}")

# X. Null Model Comparison

def run_null_model(
    bclass: List[ByteClassification],
    engine: TransitionEngine,
    partitions: Dict[str, Dict],
    n_samples: int = 200,
) -> None:
    """Compare percolation thresholds with shuffled label null models."""
    print("\n" + "=" * 5)
    print("X. NULL MODEL COMPARISON")
    print("=" * 5)

    real_family_bytes: Dict[int, List[int]] = partitions["family"]

    # Null model 1: Random family partition
    # Keep the same 256 bytes and transitions, but randomly reassign families
    print("\n--- Null Model 1: Shuffled Family Labels ---")
    print("  Same 256 permutations, randomly reassigned to 4 'families' of 64.")

    p_values = [0.2, 0.3, 0.5, 0.7, 1.0]

    # Real family sweep
    print("\n  Real CGM families:")
    for p in p_values:
        span_count = 0
        for _ in range(n_samples):
            allowed = []
            for fam in range(4):
                if random.random() < p:
                    allowed.extend(real_family_bytes[fam])
            if not allowed:
                continue
            r = compute_reachability(engine, allowed, track_depth=False)
            if r.spans:
                span_count += 1
        print(f"    p={p:.2f}: P(span) = {span_count/n_samples:.4f}")

    # Shuffled family sweep (average over several shuffles)
    print("\n  Shuffled family labels (avg over 5 shuffles):")
    all_byte_list = list(range(256))
    for p in p_values:
        span_counts = []
        for shuffle_idx in range(5):
            random.shuffle(all_byte_list)
            sham_families: Dict[int, List[int]] = {
                0: all_byte_list[0:64],
                1: all_byte_list[64:128],
                2: all_byte_list[128:192],
                3: all_byte_list[192:256],
            }
            sc = 0
            for _ in range(n_samples // 5):
                allowed = []
                for fam in range(4):
                    if random.random() < p:
                        allowed.extend(sham_families[fam])
                if not allowed:
                    continue
                r = compute_reachability(engine, allowed, track_depth=False)
                if r.spans:
                    sc += 1
            span_counts.append(sc / (n_samples // 5))
        avg_ps = sum(span_counts) / len(span_counts)
        print(f"    p={p:.2f}: P(span) = {avg_ps:.4f}  (real vs shuffled diff reveals CGM structure)")

    # Null model 2: Random byte subset (no family structure)
    print("\n--- Null Model 2: Random Byte Subset (No Family Structure) ---")
    print("  Randomly select k bytes from 256, measure spanning.")
    print("  Compare to selecting k bytes respecting family structure.")

    k_values = [4, 8, 16, 32, 64, 128]

    print(f"\n  {'k':>5} {'Random P(span)':<18} {'Family-struct P(span)':<22}")
    print("  " + "-" * 5)

    for k in k_values:
        # Random subset
        span_rand = 0
        for _ in range(n_samples):
            allowed = random.sample(range(256), min(k, 256))
            r = compute_reachability(engine, allowed, track_depth=False)
            if r.spans:
                span_rand += 1

        # Family-structured subset (equal from each family)
        span_struct = 0
        for _ in range(n_samples):
            allowed = []
            per_fam = max(1, k // 4)
            remaining = k - per_fam * 4
            for fam in range(4):
                fam_bytes = real_family_bytes[fam]
                n_take = per_fam + (1 if fam < remaining else 0)
                allowed.extend(random.sample(fam_bytes, min(n_take, len(fam_bytes))))
            r = compute_reachability(engine, allowed, track_depth=False)
            if r.spans:
                span_struct += 1

        print(f"  {k:>5d} {span_rand/n_samples:<18.4f} {span_struct/n_samples:<22.4f}")

    # --- Null Model 3: Shuffled byte -> transition mapping (corrected null) ---
    # Null Models 1-2 only permute LABELS (which byte belongs to which
    # family, or select generic subsets); they never touch the actual
    # dynamics, so they cannot distinguish "CGM structure" from "label
    # cardinality". The correct null decouples classification from
    # dynamics: byte b keeps its true family/q6/curvature LABEL, but its
    # TRANSITION FUNCTION is swapped for a randomly chosen other byte's
    # transition function. This breaks any true relation between the
    # classification axes and the actual reachability dynamics while
    # preserving the multiset of transition functions in play.
    print("\n--- Null Model 3: Shuffled Byte -> Transition Mapping ---")
    print("  Byte KEEPS its true family/q6 label but its dynamics is swapped")
    print("  for a random other byte's dynamics. This is the correct null:")
    print("  it breaks classification<->dynamics correspondence exactly.")

    n_shuffles = 5
    print(f"\n  Real dynamics vs shuffled dynamics (avg over {n_shuffles} shuffles),")
    print("  Family Fraction Sweep, event = full_omega (the discriminating one):")
    print(f"\n  {'p':<8} {'Real P(full)':<14} {'Shuffled P(full)':<18} {'Real P(span)':<14} {'Shuffled P(span)':<18}")
    print("  " + "-" * 5)

    fam_groups = [real_family_bytes[f] for f in range(4)]
    for p in p_values:
        real_full = real_span = 0
        for _ in range(n_samples):
            allowed = []
            for g in fam_groups:
                if random.random() < p:
                    allowed.extend(g)
            if not allowed:
                continue
            r = compute_reachability(engine, allowed, track_depth=False)
            if r.full_omega:
                real_full += 1
            if r.spans:
                real_span += 1

        shuf_full_list, shuf_span_list = [], []
        for _ in range(n_shuffles):
            perm = list(range(256))
            random.shuffle(perm)
            shuf_engine = build_shuffled_engine(engine, perm)
            sf = ss = 0
            for _ in range(max(1, n_samples // n_shuffles)):
                allowed = []
                for g in fam_groups:
                    if random.random() < p:
                        allowed.extend(g)
                if not allowed:
                    continue
                r = compute_reachability(shuf_engine, allowed, track_depth=False)
                if r.full_omega:
                    sf += 1
                if r.spans:
                    ss += 1
            denom = max(1, n_samples // n_shuffles)
            shuf_full_list.append(sf / denom)
            shuf_span_list.append(ss / denom)

        print(f"  {p:<8.2f} {real_full/n_samples:<14.4f} "
              f"{sum(shuf_full_list)/len(shuf_full_list):<18.4f} "
              f"{real_span/n_samples:<14.4f} "
              f"{sum(shuf_span_list)/len(shuf_span_list):<18.4f}")


def run_horizon_stabilizer_percolation(
    engine: TransitionEngine,
    omega: List[int],
    n_samples: int = 200,
) -> None:
    """Horizon-preserving bytes {0xAA,0x54,0xD5,0x2B} and confinement sweeps."""
    print("\n" + "=" * 5)
    print("XI. HORIZON-STABILIZER BYTE PERCOLATION")
    print("=" * 5)
    print("  Holonomic gate bytes preserve complement horizon pointwise")
    print("  (Features 19, 121; Formalism Sec 7.0).\n")

    comp_idx, eq_idx = horizon_indices(omega)
    n_comp, n_eq = len(comp_idx), len(eq_idx)
    print(f"  |comp_horizon|={n_comp}  |eq_horizon|={n_eq}")

    def on_comp(s: int) -> bool:
        return is_on_horizon(s) and not is_on_equality_horizon(s)

    def on_eq(s: int) -> bool:
        return is_on_equality_horizon(s)

    comp_census = _horizon_stabilizer_census(engine, omega, comp_idx, on_comp)
    eq_census = _horizon_stabilizer_census(engine, omega, eq_idx, on_eq)

    stabilizers_comp = [b for b, c in comp_census.items() if c == n_comp]
    stabilizers_eq = [b for b, c in eq_census.items() if c == n_eq]
    print(f"\n  Bytes preserving ALL comp_horizon states (1-step): "
          f"{len(stabilizers_comp)}")
    for b in sorted(stabilizers_comp):
        print(f"    0x{b:02X}")
    print(f"  Declared HORIZON_GATE_BYTES: "
          f"{', '.join(f'0x{x:02X}' for x in HORIZON_GATE_BYTES)}")
    print(f"  Bytes preserving ALL eq_horizon states (1-step): {len(stabilizers_eq)}")
    for b in sorted(stabilizers_eq):
        print(f"    0x{b:02X}")

    r_gates = compute_horizon_confinement(
        engine, omega, list(HORIZON_GATE_BYTES), comp_idx, on_comp)
    print(f"\n  BFS from comp_horizon, alphabet=4 gate bytes only:")
    print(f"    Reach={r_gates.reachable}  confined={r_gates.horizon_confined}  "
          f"full_omega={r_gates.full_omega}")

    p_values = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    print(f"\n  MC horizon confinement (conditional on nonempty alphabet):")
    print(f"  {'p':<7}{'P(conf|comp)':<14}{'P(conf|eq)':<14}{'P(full)':<10}")
    print("  " + "-" * 5)
    for p in p_values:
        conf_c = conf_e = full = nz = 0
        for _ in range(n_samples):
            allowed = [b for b in range(256) if random.random() < p]
            if not allowed:
                continue
            nz += 1
            rc = compute_horizon_confinement(engine, omega, allowed, comp_idx, on_comp)
            re = compute_horizon_confinement(engine, omega, allowed, eq_idx, on_eq)
            if rc.horizon_confined:
                conf_c += 1
            if re.horizon_confined:
                conf_e += 1
            r_rest = compute_reachability(engine, allowed, track_depth=False)
            if r_rest.full_omega:
                full += 1
        denom = max(nz, 1)
        pf = full / denom if p > 0 else 0.0
        print(f"  {p:<7.3f}{conf_c / denom:<14.4f}{conf_e / denom:<14.4f}{pf:<10.4f}")


def run_axis_sweeps(
    engine: TransitionEngine,
    partitions: Dict[str, Dict],
    n_samples: int = 200,
) -> Dict[str, List[Tuple[float, float, float, float, float, int]]]:
    """LI/FG/BG payload-axis micro_ref percolation sweeps."""
    print("\n" + "=" * 5)
    print("XII. LI/FG/BG PAYLOAD-AXIS SWEEPS")
    print("=" * 5)
    print("  Groups = micro_ref buckets active on each se(3) axis")
    print("  (Formalism Sec 2.1: LI pairs 0,5; FG pairs 1,4; BG pairs 2,3).\n")

    micro_ref_bytes: Dict[int, List[int]] = partitions["micro_ref"]
    p_values = sorted(set(
        [0.0] + [i / 100 for i in range(1, 31)] + [i / 10 for i in range(4, 11)]
    ))
    results: Dict[str, List[Tuple[float, float, float, float, float, int]]] = {}

    for axis in ("LI", "FG", "BG"):
        groups = axis_micro_ref_groups(micro_ref_bytes, axis)
        n_bytes = sum(len(g) for g in groups)
        print(f"\n--- {axis} axis: {len(groups)} micro_ref groups, {n_bytes} bytes ---")
        sweep_data = _random_subset_sweep(
            engine, groups, p_values, n_samples, f"axis-{axis}")
        results[axis] = sweep_data
        _print_sweep_table(sweep_data, n_samples)

    return results


def run_shadow_pair_sweeps(
    engine: TransitionEngine,
    partitions: Dict[str, Dict],
    n_samples: int = 200,
) -> Dict[str, List[Tuple[float, float, float, float, float, int]]]:
    """q-fiber percolation: full 4-byte fiber vs one repr vs one per shadow pair."""
    print("\n" + "=" * 5)
    print("XIII. SHADOW-PAIR (q-FIBER) PERCOLATION")
    print("=" * 5)
    print("  64 q6 classes x 4 bytes; b and b^0xFE share the same Omega map")
    print("  (Feature 55). Modes: full_fiber | one_repr | shadow_one.\n")

    q6_groups = [partitions["q6"][q] for q in sorted(partitions["q6"])]
    p_values = sorted(set(
        [0.0] + [i / 100 for i in range(1, 31)] + [i / 10 for i in range(4, 11)]
    ))
    results: Dict[str, List[Tuple[float, float, float, float, float, int]]] = {}

    for mode in ("full_fiber", "one_repr", "shadow_one"):
        print(f"\n--- q-fiber mode: {mode} ---")
        sweep_data = _random_q_fiber_sweep(
            engine, q6_groups, p_values, n_samples, mode)
        results[mode] = sweep_data
        _print_sweep_table(sweep_data, n_samples)

    return results


def run_summary(
    bclass: List[ByteClassification],
    engine: TransitionEngine,
    sweep_results: Dict[str, List[Tuple[float, float, float, float, float, int]]],
) -> None:
    """Threshold inventory and structural reachability checks (byte-level)."""
    print("\n" + "=" * 5)
    print("XIV. SUMMARY AND THRESHOLD INVENTORY")
    print("=" * 5)

    # Collect minimum spanning alphabets
    print("\n  MINIMUM SPANNING ALPHABETS (smallest byte subsets that span):")
    print("  Testing all single bytes and all byte pairs...")

    # Single bytes
    single_spanners = []
    for b in range(256):
        r = compute_reachability(engine, [b], max_depth=12)
        if r.spans:
            single_spanners.append(b)

    if single_spanners:
        print(f"    Single bytes that span: {len(single_spanners)}")
        for b in single_spanners[:10]:
            bc = bclass[b]
            print(f"      0x{b:02X}: fam={bc.family:02b} gate={bc.k4_gate} "
                  f"q6_w={bc.q6_weight} fd={bc.fold_disagree} "
                  f"|F|^2={bc.curvature_2form:.4f} phase_net={bc.phase_net}")
    else:
        print("    No single byte achieves spanning.")

    # Byte pairs (sample if too many)
    pair_spanners = []
    all_bytes = list(range(256))
    if len(all_bytes) ** 2 > 10000:
        # Sample pairs
        tested = 0
        for _ in range(5000):
            b1, b2 = random.sample(all_bytes, 2)
            r = compute_reachability(engine, [b1, b2], max_depth=12)
            if r.spans:
                pair_spanners.append((b1, b2))
            tested += 1
        print(f"    Byte pairs that span (sampled {tested}): {len(pair_spanners)}")
    else:
        for b1 in all_bytes:
            for b2 in all_bytes:
                if b1 >= b2:
                    continue
                r = compute_reachability(engine, [b1, b2], max_depth=12)
                if r.spans:
                    pair_spanners.append((b1, b2))
        print(f"    Byte pairs that span (exhaustive): {len(pair_spanners)}")

    # Analyze properties of spanning pairs
    if pair_spanners:
        print("\n    Properties of spanning pairs (first 20):")
        for b1, b2 in pair_spanners[:20]:
            bc1, bc2 = bclass[b1], bclass[b2]
            same_family = bc1.family == bc2.family
            same_gate = bc1.k4_gate == bc2.k4_gate
            print(f"      0x{b1:02X}+0x{b2:02X}: fams={bc1.family:02b}+{bc2.family:02b} "
                  f"gates={bc1.k4_gate}+{bc2.k4_gate} "
                  f"same_fam={same_family} same_gate={same_gate}")

    # CGM constant ratios (full_omega event)
    print("\n  THRESHOLD RATIOS vs CGM CONSTANTS (full_omega event):")
    cgm_refs = {
        "Delta": DELTA,
        "1/48": INV48,
        "5/256": DYADIC_5_256,
        "6/256": 6.0 / 256.0,
        "rho": RHO,
        "m_a": M_A,
    }

    for sweep_name, data in sweep_results.items():
        p_c = _find_crossing(data, 4)  # column 4 = P(full_omega)
        if p_c is not None:
            print(f"\n    {sweep_name}: p_c(full_omega) ~ {p_c:.6f}")
            for name, val in cgm_refs.items():
                print(f"      p_c / {name:<8s} = {p_c/val:.6f}")
        else:
            print(f"\n    {sweep_name}: no full_omega threshold detected in range")

    # Structural reachability (byte-level)
    print("\n  STRUCTURAL REACHABILITY (byte-level):")

    # Test: is pole-swap gate necessary?
    gate_bytes_d: Dict[str, List[int]] = defaultdict(list)
    for bc in bclass:
        gate_bytes_d[bc.k4_gate].append(bc.byte)

    r_preserve = compute_reachability(engine, gate_bytes_d['id'] + gate_bytes_d['F'])
    r_swap = compute_reachability(engine, gate_bytes_d['S'] + gate_bytes_d['C'])
    print(f"    Shell-preserving only (id+F): Span={r_preserve.spans}")
    print(f"    Pole-swap only (S+C):        Span={r_swap.spans}")

    # Test: can a single family span?
    for fam in range(4):
        allowed = [bc.byte for bc in bclass if bc.family == fam]
        r = compute_reachability(engine, allowed)
        print(f"    Family {fam:02b} alone: Span={r.spans}  Reach={r.reachable}")

    # Test: minimum family pairs for spanning
    print("    Family pairs that span:")
    for f1, f2 in itertools.combinations(range(4), 2):
        allowed = [bc.byte for bc in bclass if bc.family in (f1, f2)]
        r = compute_reachability(engine, allowed)
        if r.spans:
            print(f"      {f1:02b}+{f2:02b}: Span@D={r.first_span_depth}")

def main() -> None:
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    random.seed(20260702)

    print("hQVM Byte-Alphabet Percolation: CGM-grounded baseline (analysis_1)")
    print("=" * 5)

    # I. Classification
    print("\nI. MANIFOLD CONSTRUCTION AND BYTE CLASSIFICATION")
    print("-" * 5)

    bclass = classify_all_bytes()
    omega = enumerate_omega()

    print(f"  |Omega| = {len(omega)}")
    print(f"  |Byte alphabet| = 256")

    # Classification summary
    family_counts = Counter(bc.family for bc in bclass)
    gate_counts = Counter(bc.k4_gate for bc in bclass)
    qw_counts = Counter(bc.q6_weight for bc in bclass)
    fd_counts = Counter(bc.fold_disagree for bc in bclass)
    pn_counts = Counter(bc.phase_net for bc in bclass)
    curv_counts = Counter(round(bc.curvature_2form, 6) for bc in bclass)

    print(f"\n  Family distribution: {dict(sorted(family_counts.items()))}")
    print(f"  K4 gate distribution: {dict(sorted(gate_counts.items(), key=lambda x: x[0]))}")
    print(f"  Q6 weight distribution: {dict(sorted(qw_counts.items()))}")
    print(f"  Fold disagreement distribution: {dict(sorted(fd_counts.items()))}")
    print(f"  Phase-net vectors: {len(pn_counts)} distinct")
    print(f"  Curvature 2-form values: {dict(sorted(curv_counts.items()))}")

    # Connection chain diversity
    chain_sigs = set(bc.connection_chain for bc in bclass)
    print(f"  Distinct connection chain signatures: {len(chain_sigs)}")

    engine = build_transition_engine(omega)
    partitions = build_byte_partitions(bclass)

    # I.5 K4 algebraic cross-check
    run_k4_algebraic_verification(engine)

    # II. Deterministic
    run_deterministic_analysis(bclass, engine, omega)

    # III. Probabilistic
    sweep_results = run_probabilistic_sweeps(bclass, engine, partitions, n_samples=300)

    # IV. Shell-resolved
    run_shell_resolved(bclass, engine, n_samples=200)

    # V. Depth-resolved
    run_depth_resolved(bclass, engine)

    # VI. K4 gauge
    run_k4_analysis(bclass, engine)

    # VII. Curvature chain
    run_curvature_analysis(bclass, engine)

    # VIII. Aperture hypothesis
    run_aperture_hypothesis(sweep_results)

    # IX. Holographic identity
    run_holographic_test(bclass, omega, engine, n_samples=100)

    # X. Null model
    run_null_model(bclass, engine, partitions, n_samples=100)

    # XI-XIII. Extended formalism sweeps
    run_horizon_stabilizer_percolation(engine, omega, n_samples=150)
    axis_results = run_axis_sweeps(engine, partitions, n_samples=150)
    shadow_results = run_shadow_pair_sweeps(engine, partitions, n_samples=150)

    # XIV. Summary
    sweep_results.update({f"axis_{k}": v for k, v in axis_results.items()})
    sweep_results.update({f"qfiber_{k}": v for k, v in shadow_results.items()})
    run_summary(bclass, engine, sweep_results)

    print("\n" + "=" * 5)
    print("END. See docs/Findings/Analysis_hQVM_Percolation.md for the writeup.")
    print("=" * 5)

if __name__ == "__main__":
    main()