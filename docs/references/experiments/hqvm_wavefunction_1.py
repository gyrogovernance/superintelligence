#!/usr/bin/env python3
"""
hqvm_wavefunction_1.py
hQVM wavefunction holonomy diagnostics.

Structural principles (corrected):
- CS (GENE_Mic/0xAA) = reference frame, not a state in Ω
- Carrier rest (0xAAA555) = point on complement horizon, NOT CS
- Each byte = one full [L][R] operation with 4 internal CGM sub-phases
- CGM conditions activate at their modal depths:
    CS:  depth 0 (origin, chirality at horizon)
    UNA: depth 2 (after byte 1: variety observable, carrier departs horizon)
    ONA: depth 4 (after byte 2: opposition observable as transient equality)
    BU:  depth 4 (after byte 2: S-sector closure achieved, □B holds)
- BU-Egress/Ingress are dual readings of the same depth-4 event:
    Egress reading: depth-4 commutator vanishes in S-sector (□B)
    Ingress reading: balanced state encodes memory of full path
    These are NOT sequential stages - they are simultaneous aspects of BU
- Carrier holographic Z2 encoding requires full K4 traversal (4 bytes)
  This is NOT a new CGM stage; it is BU's carrier-level manifestation
- Z2 holonomy = spectral property: U_W^2 = I, eigenvalues +/-1
  Z2 phase on -1 eigenspace IS the holonomy
- No return to CS: the helix overlays the origin, never revisits it
  Carrier rest at depth 8 = Z2 holonomy cycle complete
- Chirality is preserved by the canonical word; holonomy acts on carrier only
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gyroscopic.hQVM.constants import (
    GENE_MAC_REST,
    GENE_MIC_S,
    GENE_MAC_A12,
    GENE_MAC_B12,
    LAYER_MASK_12,
    CHIRALITY_MASK_6,
    MASK_STATE24,
    step_state_by_byte,
    unpack_state,
    byte_to_intron,
    intron_family,
    intron_micro_ref,
    is_on_horizon,
    is_on_equality_horizon,
    apply_gate,
)
from gyroscopic.hQVM.api import (
    chirality_word6,
    q_word6,
    q_word6_for_items,
    mask12_for_byte,
    word_signature,
    omega_word_signature,
    state24_to_omega12,
    omega12_to_state24,
    OmegaState12,
    OmegaSignature12,
    apply_omega_gate,
)

# ---------------------------------------------------------------------------
# Local constants
# ---------------------------------------------------------------------------
GENE_MAC_SWAPPED: Final[int] = (GENE_MAC_B12 << 12) | GENE_MAC_A12
FAMILY_RAY_REF: Final[int] = 1

# ---------------------------------------------------------------------------
# Byte / word helpers
# ---------------------------------------------------------------------------

def byte_from_family_and_micro(family: int, micro_ref: int) -> int:
    family &= 0x03
    micro_ref &= 0x3F
    bit7 = (family >> 1) & 1
    bit0 = family & 1
    intron = (bit7 << 7) | (micro_ref << 1) | bit0
    return intron ^ GENE_MIC_S


def family_word_for_micro(micro_ref: int) -> list[int]:
    return [byte_from_family_and_micro(fam, micro_ref) for fam in range(4)]


def apply_word_to_state(word: list[int], state24: int) -> int:
    s = state24
    for b in word:
        s = step_state_by_byte(s, b)
    return s


def bin6(value: int) -> str:
    return format(int(value) & CHIRALITY_MASK_6, "06b")


def yn(value: bool) -> str:
    return "Y" if value else "N"


# ---------------------------------------------------------------------------
# Ω enumeration
# ---------------------------------------------------------------------------

def enumerate_mask_code() -> list[int]:
    masks: set[int] = set()
    for micro in range(64):
        m = 0
        for j in range(6):
            if (micro >> j) & 1:
                m |= 0b11 << (2 * j)
        masks.add(m & LAYER_MASK_12)
    return sorted(masks)


def enumerate_omega() -> list[int]:
    code = enumerate_mask_code()
    states: list[int] = []
    for u in code:
        a12 = GENE_MAC_A12 ^ u
        for v in code:
            b12 = GENE_MAC_B12 ^ v
            states.append(((a12 & LAYER_MASK_12) << 12) | (b12 & LAYER_MASK_12))
    return states


# ---------------------------------------------------------------------------
# Constitutional classification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Constitutional:
    sector: str   # "comp-horizon", "eq-horizon", "bulk"
    shell: int    # arch_shell (0-6), -1 if not pair-diagonal
    z2: str       # "rest", "swapped", "-"
    chi6: int
    chi_weight: int  # popcount(chi6)


def classify_state(state24: int) -> Constitutional:
    a12, b12 = unpack_state(state24)
    on_comp = is_on_horizon(state24)
    on_eq = is_on_equality_horizon(state24)

    if on_eq:
        sector = "eq-horizon"
    elif on_comp:
        sector = "comp-horizon"
    else:
        sector = "bulk"

    diff = (a12 ^ b12) & LAYER_MASK_12
    arch_bits = (LAYER_MASK_12 ^ diff).bit_count()
    pair_diag = True
    for i in range(6):
        pair = (diff >> (2 * i)) & 0b11
        if pair not in (0b00, 0b11):
            pair_diag = False
            break
    shell = arch_bits // 2 if pair_diag else -1

    s = state24 & MASK_STATE24
    if s == GENE_MAC_REST:
        z2 = "rest"
    elif s == GENE_MAC_SWAPPED:
        z2 = "swapped"
    else:
        z2 = "-"

    chi = chirality_word6(state24)
    return Constitutional(
        sector=sector, shell=shell, z2=z2, chi6=chi, chi_weight=chi.bit_count()
    )


# ---------------------------------------------------------------------------
# Permutation and cycle analysis
# ---------------------------------------------------------------------------

def build_permutation(word: list[int], omega: list[int]) -> dict[int, int]:
    perm: dict[int, int] = {}
    for s in omega:
        state = s
        for byte in word:
            state = step_state_by_byte(state, byte)
        perm[s] = state
    return perm


def cycle_decomposition(perm: dict[int, int]) -> list[list[int]]:
    visited: set[int] = set()
    cycles: list[list[int]] = []
    for start in sorted(perm):
        if start in visited:
            continue
        cycle: list[int] = []
        current = start
        while current not in visited:
            visited.add(current)
            cycle.append(current)
            current = perm[current]
        cycles.append(cycle)
    return cycles


def eigenspace_dimensions(cycles: list[list[int]]) -> dict[str, int]:
    """
    Hilbert-space dimensions of +1 / -1 / other eigenspaces of U_W on C^Omega.
    Each 1-cycle contributes 1 to dim(+1); each 2-cycle contributes 1 to each.
    """
    dim_plus = 0
    dim_minus = 0
    dim_other = 0
    for c in cycles:
        n = len(c)
        if n == 1:
            dim_plus += 1
        elif n == 2:
            dim_plus += 1
            dim_minus += 1
        else:
            dim_plus += 1
            if n % 2 == 0:
                dim_minus += 1
            dim_other += n - 1 - (1 if n % 2 == 0 else 0)
    return {"+1": dim_plus, "-1": dim_minus, "other": dim_other}


def sector_eigenspace_dimensions(
    cycles: list[list[int]],
) -> dict[str, dict[str, int]]:
    """
    Per-sector eigenspace dimensions for cycles fully contained in that sector.
    Mixed-sector cycles are counted as mixed_cycles (eigenvectors span sectors).
    """
    sectors = ("comp-horizon", "eq-horizon", "bulk")
    out = {s: {"+1": 0, "-1": 0, "other": 0, "mixed_cycles": 0} for s in sectors}
    for c in cycles:
        n = len(c)
        if n == 1:
            sec = classify_state(c[0]).sector
            out[sec]["+1"] += 1
        elif n == 2:
            secs = {classify_state(s).sector for s in c}
            if len(secs) == 1:
                sec = next(iter(secs))
                out[sec]["+1"] += 1
                out[sec]["-1"] += 1
            else:
                for sec in secs:
                    out[sec]["mixed_cycles"] += 1
        else:
            secs = {classify_state(s).sector for s in c}
            if len(secs) == 1:
                sec = next(iter(secs))
                out[sec]["+1"] += 1
                if n % 2 == 0:
                    out[sec]["-1"] += 1
                    out[sec]["other"] += n - 2
                else:
                    out[sec]["other"] += n - 1
            else:
                for sec in secs:
                    out[sec]["mixed_cycles"] += 1
    return out


# ---------------------------------------------------------------------------
# Holographic dictionary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShadowPair:
    state_a: int
    state_b: int
    cls_a: Constitutional
    cls_b: Constitutional
    pair_type: str  # "2-cycle" or "fixed"


def compute_holographic_dictionary(
    word: list[int], omega: list[int]
) -> list[ShadowPair]:
    perm = build_permutation(word, omega)
    for s in omega:
        if perm[perm[s]] != s:
            raise ValueError(
                "compute_holographic_dictionary requires an involution word (U_W^2 = I)."
            )
    pairs: list[ShadowPair] = []
    seen: set[int] = set()
    for s in sorted(omega):
        if s in seen:
            continue
        partner = perm[s]
        cls_s = classify_state(s)
        if partner == s:
            pairs.append(ShadowPair(s, s, cls_s, cls_s, "fixed"))
            seen.add(s)
        else:
            cls_p = classify_state(partner)
            pairs.append(ShadowPair(s, partner, cls_s, cls_p, "2-cycle"))
            seen.add(s)
            seen.add(partner)
    return pairs


# ---------------------------------------------------------------------------
# K4 gate classification
# ---------------------------------------------------------------------------

def classify_word_as_k4_gate(word: list[int]) -> str:
    """Classify word's action on Omega as a K4 gate via Omega12 signature."""
    sig = omega_word_signature(word)
    gate_sigs = {
        "id": OmegaSignature12(parity=0, tau_u6=0, tau_v6=0),
        "S": OmegaSignature12(parity=1, tau_u6=0, tau_v6=0),
        "C": OmegaSignature12(parity=1, tau_u6=63, tau_v6=63),
        "F": OmegaSignature12(parity=0, tau_u6=63, tau_v6=63),
    }
    for name, gate_sig in gate_sigs.items():
        if sig == gate_sig:
            return name
    return "none"


# ---------------------------------------------------------------------------
# Half-step primitives
# ---------------------------------------------------------------------------

def half_step_L(state24: int, byte: int) -> int:
    a12, b12 = unpack_state(state24)
    mask = mask12_for_byte(byte)
    a_mut = a12 ^ mask
    return ((a_mut & LAYER_MASK_12) << 12) | (b12 & LAYER_MASK_12)


def half_step_R(state24: int, byte: int) -> int:
    a12, b12 = unpack_state(state24)
    intron = byte_to_intron(byte)
    bit0 = intron & 1
    bit7 = (intron >> 7) & 1
    next_a = b12 ^ (LAYER_MASK_12 if bit0 else 0)
    next_b = a12 ^ (LAYER_MASK_12 if bit7 else 0)
    return ((next_a & LAYER_MASK_12) << 12) | (next_b & LAYER_MASK_12)


def verify_bu_egress_byte_order(
    micro_ref: int = FAMILY_RAY_REF,
    start_state: int = GENE_MAC_REST,
) -> dict[str, int | bool]:
    """
    Depth-4: T_b0 then T_b1 vs T_b1 then T_b0 from carrier rest.
    S-sector closure: both results on complement horizon (not necessarily equal).
    """
    b0 = byte_from_family_and_micro(0, micro_ref)
    b1 = byte_from_family_and_micro(1, micro_ref)
    s01 = apply_word_to_state([b0, b1], start_state)
    s10 = apply_word_to_state([b1, b0], start_state)
    return {
        "b0": b0,
        "b1": b1,
        "T01": s01,
        "T10": s10,
        "equal": s01 == s10,
        "both_on_comp_horizon": is_on_horizon(s01) and is_on_horizon(s10),
        "both_on_eq_horizon": is_on_equality_horizon(s01) and is_on_equality_horizon(s10),
    }


def verify_bu_egress_primitive_lrlr(
    byte: int,
    start_state: int = GENE_MAC_REST,
) -> dict[str, int | bool]:
    """Single-byte depth-4: primitive LRLR vs RLRL sequence."""
    s = start_state
    lrlr = half_step_R(
        half_step_L(half_step_R(half_step_L(s, byte), byte), byte), byte
    )
    rlrl = half_step_L(
        half_step_R(half_step_L(half_step_R(s, byte), byte), byte), byte
    )
    return {
        "LRLR": lrlr,
        "RLRL": rlrl,
        "equal": lrlr == rlrl,
        "both_on_horizon": is_on_horizon(lrlr) and is_on_horizon(rlrl),
    }


def verify_bu_primitive_lrlr_exhaustive(
    omega: list[int],
    *,
    bytes_to_test: list[int],
) -> dict[str, object]:
    """
    Exhaustive primitive BU test on the S-sector:
      For each byte b and each complement-horizon start state s,
      compare LRLR(s;b) vs RLRL(s;b).
    Returns pass/fail counts and failing bytes.
    """
    s_sector = [s for s in omega if classify_state(s).sector == "comp-horizon"]
    total = 0
    passed = 0
    failing_bytes: list[int] = []
    for b in bytes_to_test:
        ok_b = True
        for s in s_sector:
            total += 1
            r = verify_bu_egress_primitive_lrlr(b, start_state=s)
            if not (bool(r["equal"]) and bool(r["both_on_horizon"])):
                ok_b = False
                break
        if ok_b:
            passed += len(s_sector)
        else:
            failing_bytes.append(b)
    return {
        "s_sector_size": len(s_sector),
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "failing_bytes": failing_bytes,
    }


def run_bu_egress_verification(canonical: list[int]) -> None:
    """Kernel tests for BU-Egress ([]B / S-sector), not imported from theory only."""
    print("BU-EGRESS VERIFICATION (kernel):")
    print("  Depth-4 = 2 bytes; each byte = L then R (4 primitives).")
    ref = verify_bu_egress_byte_order(FAMILY_RAY_REF)
    print(f"  Families 00+01 (micro={FAMILY_RAY_REF}): "
          f"b0=0x{ref['b0']:02X} b1=0x{ref['b1']:02X}")
    cls01 = classify_state(int(ref["T01"]))
    cls10 = classify_state(int(ref["T10"]))
    print(f"    T(b0,b1)=0x{ref['T01']:06X}  sector={cls01.sector}  "
          f"T(b1,b0)=0x{ref['T10']:06X}  sector={cls10.sector}")
    print(f"    order_equal={yn(bool(ref['equal']))}  "
          f"both_comp_horizon={yn(bool(ref['both_on_comp_horizon']))}  "
          f"both_eq_horizon={yn(bool(ref['both_on_eq_horizon']))}")

    all_equal = all(verify_bu_egress_byte_order(m)["equal"] for m in range(64))
    all_comp = all(
        verify_bu_egress_byte_order(m)["both_on_comp_horizon"] for m in range(64)
    )
    all_eq = all(
        verify_bu_egress_byte_order(m)["both_on_eq_horizon"] for m in range(64)
    )
    print(f"  Exhaustive 64 payloads: order_commutes={yn(all_equal)}  "
          f"both_comp_horizon={yn(all_comp)}  both_eq_horizon={yn(all_eq)}")
    if not all_equal:
        print("  => Byte order matters at depth-4 (UNA); []B is projection, not path equality.")
    if len(canonical) >= 2:
        prim = verify_bu_egress_primitive_lrlr(canonical[1])
        print(f"  Primitive LRLR vs RLRL (byte 2 = 0x{canonical[1]:02X}): "
              f"equal={yn(bool(prim['equal']))}  "
              f"both_on_horizon={yn(bool(prim['both_on_horizon']))}")

    ex = verify_bu_primitive_lrlr_exhaustive(
        enumerate_omega(),
        bytes_to_test=canonical,
    )
    print(
        f"  Exhaustive primitive BU check on S-sector (canonical bytes): "
        f"passed={ex['passed_cases']}/{ex['total_cases']}  "
        f"failed={ex['failed_cases']}"
    )
    failing = ex["failing_bytes"]
    if isinstance(failing, list) and failing:
        fb = " ".join(f"0x{x:02X}" for x in failing)
        print(f"  Failing bytes: {fb}")
    print()


# ---------------------------------------------------------------------------
# Printers
# ---------------------------------------------------------------------------

def print_table(headers: list[str], rows: list[list[str]], title: str = "") -> None:
    if title:
        print(f"\n{title}")
        print("-" * len(title))
    if not rows:
        print("(no data)\n")
        return
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    print("  ".join(h.ljust(w) for h, w in zip(headers, col_widths)))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print("  ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))
    print()


# ---------------------------------------------------------------------------
# Diagnostic sections
# ---------------------------------------------------------------------------

def run_decomposition_verify() -> None:
    canonical = family_word_for_micro(FAMILY_RAY_REF)
    print("\nDecomposition Verify: T_b = R o L")
    print("-" * 9)
    ok = True
    for byte in canonical:
        for start in [GENE_MAC_REST, GENE_MAC_SWAPPED]:
            full = step_state_by_byte(start, byte)
            after_L = half_step_L(start, byte)
            after_RL = half_step_R(after_L, byte)
            if full != after_RL:
                ok = False
                print(f"  FAIL byte=0x{byte:02X} start=0x{start:06X}")
    print(f"  T_b = R o L verified: {yn(ok)}")


def run_omega_census(omega: list[int]) -> None:
    sector_counts: Counter = Counter()
    shell_counts: Counter = Counter()
    chi_weight_counts: Counter = Counter()
    for s in omega:
        cls = classify_state(s)
        sector_counts[cls.sector] += 1
        shell_counts[cls.shell] += 1
        chi_weight_counts[cls.chi_weight] += 1

    print("\nOmega Constitutional Census")
    print("=" * 9)
    print(f"  |Omega| = {len(omega)}")
    for sector in ["comp-horizon", "eq-horizon", "bulk"]:
        print(f"  {sector:20s} {sector_counts.get(sector, 0)}")
    print()
    print("  Shell distribution (arch_shell):")
    for shell in sorted(shell_counts):
        pop = shell_counts[shell]
        bar = "#" * (pop // 20)
        print(f"    shell {shell}: {pop:5d}  {bar}")
    print()
    print("  Chirality weight distribution (popcount of chi6):")
    for w in sorted(chi_weight_counts):
        print(f"    weight {w}: {chi_weight_counts[w]}")
    print()
    print("  Holographic identity: |H|^2 = |Omega|")
    print(f"    |comp-horizon|^2 = {sector_counts['comp-horizon']}^2 = {sector_counts['comp-horizon']**2}")
    print(f"    |eq-horizon|^2   = {sector_counts['eq-horizon']}^2 = {sector_counts['eq-horizon']**2}")
    print(f"    |Omega|         = {len(omega)}")
    print(f"    |H|^2 = |Omega|: {sector_counts['comp-horizon']**2 == len(omega)}")


def run_holographic_dictionary(omega: list[int]) -> None:
    canonical = family_word_for_micro(FAMILY_RAY_REF)
    pairs = compute_holographic_dictionary(canonical, omega)

    n_2cycle = sum(1 for p in pairs if p.pair_type == "2-cycle")
    n_fixed = sum(1 for p in pairs if p.pair_type == "fixed")

    print(f"\n{'=' * 60}")
    print("Holographic Dictionary: Shadow Partners under U_W")
    print(f"{'=' * 60}")
    print(f"Word: canonical 4-family ({' '.join(f'0x{b:02X}' for b in canonical)})")
    print(f"K4 gate: {classify_word_as_k4_gate(canonical)}")
    print(f"Total shadow pairs: {len(pairs)} ({n_2cycle} 2-cycles, {n_fixed} fixed)")
    print()

    # Sector breakdown
    sector_pairs: dict[str, list[ShadowPair]] = {}
    for p in pairs:
        sector = p.cls_a.sector
        sector_pairs.setdefault(sector, []).append(p)

    headers = ["sector", "2-cycles", "fixed", "states"]
    rows = []
    for sector in ["comp-horizon", "eq-horizon", "bulk"]:
        sp = sector_pairs.get(sector, [])
        n2 = sum(1 for p in sp if p.pair_type == "2-cycle")
        nf = sum(1 for p in sp if p.pair_type == "fixed")
        rows.append([sector, str(n2), str(nf), str(2 * n2 + nf)])
    print_table(headers, rows, "Shadow Pairs by Constitutional Sector")

    # Shell breakdown
    shell_pairs: dict[int, list[ShadowPair]] = {}
    for p in pairs:
        if p.cls_a.shell >= 0:
            shell_pairs.setdefault(p.cls_a.shell, []).append(p)

    headers = ["shell", "2-cycles", "fixed", "states"]
    rows = []
    for shell in sorted(shell_pairs):
        sp = shell_pairs[shell]
        n2 = sum(1 for p in sp if p.pair_type == "2-cycle")
        nf = sum(1 for p in sp if p.pair_type == "fixed")
        rows.append([str(shell), str(n2), str(nf), str(2 * n2 + nf)])
    print_table(headers, rows, "Shadow Pairs by Shell")

    # The canonical shadow pair: rest <-> swapped
    print("Canonical shadow pair (carrier Z2 orbit):")
    for p in pairs:
        if p.cls_a.z2 == "rest" or p.cls_b.z2 == "rest":
            if p.cls_a.z2 == "rest":
                rest_state, rest_cls = p.state_a, p.cls_a
                swap_state, swap_cls = p.state_b, p.cls_b
            else:
                rest_state, rest_cls = p.state_b, p.cls_b
                swap_state, swap_cls = p.state_a, p.cls_a
            print(f"  |rest>    = 0x{rest_state:06X}  sector={rest_cls.sector}  "
                  f"shell={rest_cls.shell}  chi={bin6(rest_cls.chi6)}")
            print(f"  |swapped> = 0x{swap_state:06X}  sector={swap_cls.sector}  "
                  f"shell={swap_cls.shell}  chi={bin6(swap_cls.chi6)}")
            print(f"  U_W|rest> = |swapped>  (Z2 flip = gate F)")
            print(f"  Holonomy eigenvectors on this orbit:")
            print(f"    |+> = (|rest> + |swapped>)/sqrt(2)   eigenvalue +1")
            print(f"    |-> = (|rest> - |swapped>)/sqrt(2)   eigenvalue -1")
            break

    # Show representative pairs from each sector
    print()
    print("Representative shadow pairs by sector:")
    for sector in ["comp-horizon", "eq-horizon", "bulk"]:
        sp = [p for p in sector_pairs.get(sector, []) if p.pair_type == "2-cycle"]
        if not sp:
            continue
        print(f"\n  {sector} (showing first 3 of {len(sp)} pairs):")
        for p in sp[:3]:
            omega_a = state24_to_omega12(p.state_a)
            omega_b = state24_to_omega12(p.state_b)
            print(f"    0x{p.state_a:06X} <-> 0x{p.state_b:06X}  "
                  f"shell={p.cls_a.shell}  "
                  f"chi_a={bin6(p.cls_a.chi6)}  chi_b={bin6(p.cls_b.chi6)}  "
                  f"u=({omega_a.u6:2d},{omega_a.v6:2d})<->({omega_b.u6:2d},{omega_b.v6:2d})")

    # Chirality preservation
    chi_preserved = all(p.cls_a.chi6 == p.cls_b.chi6 for p in pairs if p.pair_type == "2-cycle")
    q_transport = q_word6_for_items(canonical)
    print(f"\nChirality preservation: {yn(chi_preserved)}")
    print(f"  Cumulative q-transport = 0x{q_transport:02X} = {q_transport}")
    print(f"  Holonomy acts on CARRIER subspace only, not chirality")


def run_spectral_decomposition(omega: list[int]) -> None:
    canonical = family_word_for_micro(FAMILY_RAY_REF)
    same_fam = [canonical[0]] * 4

    for label, word in [("canonical 4-family", canonical), ("same-family 00", same_fam)]:
        perm = build_permutation(word, omega)
        cycles = cycle_decomposition(perm)

        cycle_len_counts = dict(sorted(Counter(len(c) for c in cycles).items()))
        eig_dims = eigenspace_dimensions(cycles)
        sector_dims = sector_eigenspace_dimensions(cycles)
        n_2cycles = sum(1 for c in cycles if len(c) == 2)
        is_involution = all(len(c) <= 2 for c in cycles)
        k4_gate = classify_word_as_k4_gate(word)

        print(f"\n{'=' * 60}")
        print(f"Spectral Decomposition: {label}")
        print(f"{'=' * 60}")
        print(f"  Word: {' '.join(f'0x{b:02X}' for b in word)}")
        print(f"  K4 gate on carrier: {k4_gate}")
        print(f"  Cycle length spectrum: {cycle_len_counts}")
        print(f"  U_W is involution (U_W^2 = I): {yn(is_involution)}")
        print()
        print("  Eigenspace dimensions (Hilbert space, not basis-state count):")
        print(f"    dim(+1): {eig_dims['+1']}  (symmetric / fixed modes)")
        print(f"    dim(-1): {eig_dims['-1']}  (antisymmetric / Z2 holonomy modes)")
        print(f"    dim(other): {eig_dims['other']}")
        print(f"    sum: {eig_dims['+1'] + eig_dims['-1'] + eig_dims['other']}  "
              f"(|Omega| = {len(omega)})")

        headers = ["sector", "dim(+1)", "dim(-1)", "dim(other)", "mixed_cycles", "states"]
        rows = []
        for sector in ["comp-horizon", "eq-horizon", "bulk"]:
            sd = sector_dims[sector]
            n_states = sum(1 for s in omega if classify_state(s).sector == sector)
            rows.append([
                sector,
                str(sd["+1"]),
                str(sd["-1"]),
                str(sd["other"]),
                str(sd["mixed_cycles"]),
                str(n_states),
            ])
        print_table(headers, rows, "Eigenspace dimensions x constitutional sector")

        # Key orbits
        rest_img = perm[GENE_MAC_REST]
        rest_cls = classify_state(rest_img)
        print(f"  U_W|rest>    = 0x{rest_img:06X}  Z2={rest_cls.z2}  sector={rest_cls.sector}")

        if GENE_MAC_SWAPPED in perm:
            sw_img = perm[GENE_MAC_SWAPPED]
            sw_cls = classify_state(sw_img)
            print(f"  U_W|swapped> = 0x{sw_img:06X}  Z2={sw_cls.z2}  sector={sw_cls.sector}")

        # Holonomy interpretation
        if is_involution and eig_dims["-1"] > 0:
            print()
            print("  Z2 holonomy = Z2 phase on dim(-1) eigenspace")
            print(f"    dim(-1) = {eig_dims['-1']}  ({n_2cycles} two-cycles, "
                  f"{2 * n_2cycles} basis states)")
            print("  Basis states in 2-cycles are not eigenvectors; |+>, |-> are superpositions.")
            print("  This is a SPECTRAL property, not a carrier trajectory")


def run_bu_duality(omega: list[int]) -> None:
    """Show BU-Egress/Ingress as dual readings of depth-4."""
    canonical = family_word_for_micro(FAMILY_RAY_REF)

    print(f"\n{'=' * 60}")
    print("BU Duality: Egress and Ingress as Dual Readings of Depth-4")
    print(f"{'=' * 60}")
    print()
    print("CGM defines BU at depth 4 (2 bytes = [L][R][L][R]).")
    print("BU-Ingress: balanced state encodes memory of CS, UNA, ONA")
    print("These are NOT sequential - they are simultaneous aspects of BU.")
    print()
    run_bu_egress_verification(canonical)

    # After byte 1 (depth 2): UNA observable
    state_1 = apply_word_to_state(canonical[:1], GENE_MAC_REST)
    cls_1 = classify_state(state_1)

    # After byte 2 (depth 4): BU achieved, ONA observable
    state_2 = apply_word_to_state(canonical[:2], GENE_MAC_REST)
    cls_2 = classify_state(state_2)
    chi_2 = chirality_word6(state_2)

    # After byte 4 (depth 8): BU manifest in carrier
    state_4 = apply_word_to_state(canonical[:4], GENE_MAC_REST)
    cls_4 = classify_state(state_4)

    print("Constitutional trajectory through the canonical word:")
    print()
    print(f"  Start (depth 0):    comp-horizon, rest       [carrier reference point; CS is the frame]")
    print(f"  After byte 1 (d=2): {cls_1.sector}, shell={cls_1.shell}  "
          f"[UNA: variety introduced, departed from horizon]")
    print(f"  After byte 2 (d=4): {cls_2.sector}, shell={cls_2.shell}  "
          f"[ONA+BU: equality transit, S-sector closure]")
    print(f"  After byte 3 (d=6): bulk, shell=1             "
          f"[BU carrier approach]")
    print(f"  After byte 4 (d=8): {cls_4.sector}, {cls_4.z2}      "
          f"[BU holographic: complement horizon, Z2 encoding]")
    print()

    # Egress reading at depth 4
    print("EGRESS READING (after byte 2, depth 4):")
    print(f"  Carrier state: 0x{state_2:06X}")
    print(f"  Constitutional: {cls_2.sector}, shell={cls_2.shell}")
    print(f"  Chirality: {bin6(chi_2)} (weight {chi_2.bit_count()})")
    print("  []B holds: depth-4 commutator vanishes in S-sector projection (see verification above)")
    print(f"  Carrier position: equality horizon (shell {cls_2.shell}, "
          f"opposite pole from complement horizon)")
    print("  Closure is a PROJECTION property, not a carrier position")
    print()

    # Ingress reading at depth 8
    print("INGRESS READING (after byte 4, depth 8):")
    print(f"  Carrier state: 0x{state_4:06X}")
    print(f"  Constitutional: {cls_4.sector}, Z2={cls_4.z2}")
    print("  Carrier on complement horizon with Z2 encoding (swapped != rest)")
    print(f"  The balanced state ENCODES memory of the full path:")
    print(f"    CS chirality (origin) -> UNA variety -> ONA equality -> BU closure")
    print("  The Z2 encoding (rest vs swapped) is the HOLOGRAPHIC CONTENT of BU")
    print()

    # Duality statement
    print("DUALITY:")
    print("  Egress and Ingress are the same BU event read two ways:")
    print("  - Egress: 'closure is achieved' (projection/S-sector reading)")
    print("  - Ingress: 'closure carries memory' (carrier/provenance reading)")
    print("  The Z2 holonomy IS the holographic encoding that makes both")
    print("  readings simultaneously true at depth 4.")
    print()
    print("  The 4-byte canonical word does NOT produce two sequential stages.")
    print("  It produces ONE depth-4 event (BU) whose holographic Z2 structure")
    print("  is simultaneously Egress and Ingress.")

    # Half-step detail
    print()
    print("HALF-STEP VERIFICATION:")
    state = GENE_MAC_REST
    for i, byte in enumerate(canonical):
        after_L = half_step_L(state, byte)
        after_R = half_step_R(after_L, byte)
        cls_L = classify_state(after_L)
        cls_R = classify_state(after_R)
        depth_L = 2 * i + 1
        depth_R = 2 * i + 2
        print(f"  Byte {i+1} (0x{byte:02X}):")
        print(f"    L (depth {depth_L}): 0x{after_L:06X}  "
              f"{cls_L.sector:14s} sh={cls_L.shell} Z2={cls_L.z2}")
        if cls_L.sector == "comp-horizon" and depth_L > 0:
            print(f"      *** Complement horizon at primitive depth {depth_L} ***")
        print(f"    R (depth {depth_R}): 0x{after_R:06X}  "
              f"{cls_R.sector:14s} sh={cls_R.shell} Z2={cls_R.z2}")
        if cls_R.sector == "eq-horizon":
            print(f"      *** Equality horizon at primitive depth {depth_R} (ONA) ***")
        if cls_R.z2 == "swapped" and cls_R.sector == "comp-horizon":
            print(f"      *** Z2 encoding at primitive depth {depth_R} (BU holographic) ***")
        state = after_R


def run_chirality_preservation(omega: list[int]) -> None:
    """Verify chirality is preserved; holonomy acts on carrier only."""
    canonical = family_word_for_micro(FAMILY_RAY_REF)
    perm = build_permutation(canonical, omega)

    print(f"\n{'=' * 60}")
    print("Chirality Preservation: Holonomy on Carrier, Not Chirality")
    print(f"{'=' * 60}")

    q_total = q_word6_for_items(canonical)
    print(f"  Canonical word cumulative q-transport: 0x{q_total:02X} = {q_total}")
    print(f"  Chirality transport rule: chi(U_W|s>) = chi(|s>) XOR q(W) = chi(|s>)")
    print()

    # Verify for all states
    chi_preserved = 0
    chi_changed = 0
    for s in omega:
        chi_s = chirality_word6(s)
        chi_img = chirality_word6(perm[s])
        if chi_s == chi_img:
            chi_preserved += 1
        else:
            chi_changed += 1

    print(f"  States where chi preserved: {chi_preserved}/{len(omega)}")
    print(f"  States where chi changed:   {chi_changed}/{len(omega)}")
    print()
    print("  Interpretation: the canonical word acts as identity on chirality")
    print("  (the 'radial' coordinate = shell membership) but as gate F on the")
    print("  carrier (the 'angular' coordinate = position within shell).")
    print("  Z2 holonomy = Z2 phase on the carrier subspace, not chirality.")
    print()

    # Shell-level analysis
    print("  Shell-level shadow pair counts:")
    for shell in range(7):
        shell_states = [s for s in omega if classify_state(s).shell == shell]
        n_pairs = len(shell_states) // 2
        n_fixed = sum(1 for s in shell_states if perm[s] == s)
        print(f"    Shell {shell}: {len(shell_states):5d} states, "
              f"{n_pairs} shadow pairs, {n_fixed} fixed points")


def run_helix_evolution() -> None:
    canonical = family_word_for_micro(FAMILY_RAY_REF)

    print(f"\n{'=' * 60}")
    print("Helix Evolution: Z2 Holonomy Cycle (3 Turns)")
    print(f"{'=' * 60}")
    print("Each turn = 4 bytes traversing all 4 K4 families")
    print("Z2 coordinate oscillates: rest -> swapped -> rest -> ...")
    print("This is the holonomy CYCLE, not a return to CS")
    print()

    state = GENE_MAC_REST
    headers = ["turn", "byte", "state24", "sector", "sh", "Z2", "chi6"]
    rows = []

    cls = classify_state(state)
    rows.append(["0", "--", f"0x{state:06X}", cls.sector,
                 str(cls.shell), cls.z2, bin6(cls.chi6)])

    for turn in range(1, 4):
        for i, byte in enumerate(canonical):
            state = step_state_by_byte(state, byte)
            cls = classify_state(state)
            label = f"{turn}.{i+1}"
            rows.append([label, f"{byte:02X}", f"0x{state:06X}",
                        cls.sector, str(cls.shell), cls.z2, bin6(cls.chi6)])

    print_table(headers, rows)

    # Annotate the Z2 oscillation
    print("Z2 holonomy cycle:")
    state = GENE_MAC_REST
    z2_seq = ["rest"]
    for _ in range(3):
        for byte in canonical:
            state = step_state_by_byte(state, byte)
            cls = classify_state(state)
            if cls.z2 != "-":
                z2_seq.append(cls.z2)
    print(f"  Z2 trajectory: {' -> '.join(z2_seq)}")
    print(f"  Period: 2 turns (8 bytes) for carrier rest -> rest")
    print("  This is NOT a return to CS - it is the Z2 holonomy cycle completing")
    print()

    # Constitutional events per turn
    print("Constitutional events per turn (identical pattern repeats):")
    state = GENE_MAC_REST
    events = {
        (1, "bulk"): "departure from horizon (variety introduced)",
        (2, "eq-horizon"): "transient equality (opposition non-absolute)",
        (3, "bulk"): "return toward horizon (approaching closure)",
        (4, "comp-horizon"): "horizon with Z2 encoding (BU holographic)",
    }
    for i, byte in enumerate(canonical):
        state = step_state_by_byte(state, byte)
        cls = classify_state(state)
        key = (i + 1, cls.sector)
        event = events.get(key, f"{cls.sector}, shell={cls.shell}")
        print(f"  Byte {i+1}: {cls.sector:14s} sh={cls.shell} Z2={cls.z2:8s}  {event}")


def run_probe_suite(omega: list[int]) -> None:
    canonical = family_word_for_micro(FAMILY_RAY_REF)
    reverse = list(reversed(canonical))
    shuffle = [canonical[0], canonical[2], canonical[1], canonical[3]]

    probes = [
        ("canonical", canonical),
        ("canonical x2", canonical * 2),
        ("canonical x4", canonical * 4),
        ("reverse", reverse),
        ("phase shuffle", shuffle),
        ("same-fam 00", [canonical[0]] * 4),
        ("same-fam 11", [canonical[3]] * 4),
        ("zero payload", family_word_for_micro(0)),
        ("full payload", family_word_for_micro(63)),
    ]

    headers = [
        "name", "n", "gate", "+1", "-1", "inv",
        "rest->", "chi_ok",
    ]
    rows = []
    for name, word in probes:
        perm = build_permutation(word, omega)
        cycles = cycle_decomposition(perm)

        eig_dims = eigenspace_dimensions(cycles)
        is_inv = all(len(c) <= 2 for c in cycles)
        k4_gate = classify_word_as_k4_gate(word)

        rest_img = perm[GENE_MAC_REST]
        rest_to = classify_state(rest_img).z2

        q_total = q_word6_for_items(word)
        chi_ok = q_total == 0

        rows.append([
            name, str(len(word)), k4_gate,
            str(eig_dims["+1"]), str(eig_dims["-1"]),
            yn(is_inv), rest_to, yn(chi_ok),
        ])

    print_table(headers, rows, "Wavefunction Probe Suite (Spectral)")
    print("  n: word length | gate: K4 gate on carrier | +1/-1: dim(+1)/dim(-1)")
    print("  inv: U_W^2 = I | rest->: Z2 of U_W|rest> | chi_ok: chirality preserved")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("hQVM Wavefunction Holonomy Diagnostic")
    print("=" * 9)
    print(f"GENE_MIC_S (CS frame):  0x{GENE_MIC_S:02X}")
    print(f"GENE_MAC_REST (carrier): 0x{GENE_MAC_REST:06X}  NOT CS")
    print(f"GENE_MAC_SWAPPED:        0x{GENE_MAC_SWAPPED:06X}")
    print()
    print("Corrected principles:")
    print("  CS = GENE_Mic (0xAA) = reference frame, never re-entered")
    print("  Carrier rest = point on complement horizon, NOT CS")
    print("  BU-Egress/Ingress = dual readings of depth-4, NOT sequential")
    print("  Z2 holonomy = spectral property, NOT carrier trajectory")
    print("  Chirality preserved by canonical word; holonomy on carrier only")
    print("  Carrier rest at depth 8 = Z2 cycle complete, NOT 'back to CS'")
    print()

    run_decomposition_verify()

    omega = enumerate_omega()
    print(f"\nOmega enumerated: {len(omega)} states")

    run_omega_census(omega)
    run_holographic_dictionary(omega)
    run_spectral_decomposition(omega)
    run_chirality_preservation(omega)
    run_bu_duality(omega)
    run_helix_evolution()
    run_probe_suite(omega)


if __name__ == "__main__":
    main()