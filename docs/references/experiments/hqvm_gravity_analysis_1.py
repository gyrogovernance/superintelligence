#!/usr/bin/env python3
"""
hqvm_gravity_analysis_1.py

CGM gravity coupling analysis:

  Part A  Residual closure: c4 = -7/4 and ppm validation
  Part B  rho^5 / 1+5 STF decomposition
  Part C  Kernel transport (tau_cycle, per-family, joint table)
  Part D  Delta expansion (series match closed form)
  Part E  Coupling summary

Ownership:
  analysis_3: exact kernel theorems (tau_cycle, carrier traces, c4, alpha*zeta)
  analysis_2: theorem registry / audit (Z2, Gauss, cross-checks)
  analysis_1: G prediction, wavefunction checks, transport diagnostics (this file)
  analysis_4/5: nonlinear G(psi), field derivations, shadow
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from math import comb, exp
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gyroscopic.hQVM.constants import (
    GENE_MAC_REST,
    GENE_MAC_A12,
    GENE_MAC_B12,
)

from hqvm_gravity_common import (
    AF,
    CHI6_FULL,
    Delta,
    F_CYCLE_PATH_TRAVERSE,
    FA_STF,
    G_kernel,
    G_meas,
    H_size,
    Omega_size,
    Q_G,
    TR_SIGMA_SHELL,
    W2_SHELL_DISPLACEMENT,
    W2_word,
    W2p_word,
    Z2_HOLONOMY_PATH_TRAVERSE,
    alpha_G_meas,
    apply_word_to_state,
    binom_shell,
    build_joint_table,
    byte_from_family_and_micro,
    chi6_step_bit_stats,
    configure_stdout_utf8,
    cycle_word_for_micro,
    d_BU,
    f_ordered,
    F_cycle_word,
    g_pred_from_tau,
    kappa_binom_step,
    kappa_pi_step,
    m_a,
    pi_norm_shell,
    poly_mul,
    print_joint_table_condensed,
    q_word6_for_items,
    rho,
    tau_conjugacy_depth,
    D_traverse,
    kernel_exposure_constants,
    tau_cycle_per_delta_exact,
    tau_cycle_weighted,
    tau_g_with_c4,
    tau_G_formula,
    tau_path_binom,
    tau_path_kappa,
    tau_required,
    trace_word_steps,
    v_EW,
    weights_pop,
)

configure_stdout_utf8()


SWAPPED24 = (GENE_MAC_B12 << 12) | GENE_MAC_A12


# ============================================================
# Helpers
# ============================================================


def translational_bits(micro: int) -> tuple[int, int, int]:
    """Translational sector bits (5,4,3) of the 6-bit payload."""
    return (
        (micro >> 5) & 1,
        (micro >> 4) & 1,
        (micro >> 3) & 1,
    )


def isotropic_translational_trace_exact() -> Fraction:
    """Exact covariance trace over all 64 micro-references; equals 3/4."""
    trace = Fraction(0, 1)
    for axis in range(3):
        vals = [translational_bits(m)[axis] for m in range(64)]
        mean = Fraction(sum(vals), 64)
        mean_sq = Fraction(sum(v * v for v in vals), 64)
        trace += mean_sq - mean * mean
    return trace


def verify_wavefunction_structure() -> dict[str, bool]:
    """Verify the wavefunction facts needed by the residual calculation."""
    ok_q_w2 = True
    ok_q_w2p = True
    ok_q_f = True
    ok_w2_midpoint = True
    ok_f_sheet = True
    ok_shell_path = True

    for m in range(64):
        w2 = W2_word(m)
        w2p = W2p_word(m)
        wf = F_cycle_word(m)

        if q_word6_for_items(w2) != CHI6_FULL:
            ok_q_w2 = False
        if q_word6_for_items(w2p) != CHI6_FULL:
            ok_q_w2p = False
        if q_word6_for_items(wf) != 0:
            ok_q_f = False

        rows = trace_word_steps(wf, micro_ref=m)
        shells = [r["arch_shell"] for r in rows]

        pop = bin(m).count("1")
        expected = [0, pop, 6, pop, 0]

        if shells != expected:
            ok_shell_path = False
        if rows[2]["arch_shell"] != 6:
            ok_w2_midpoint = False
        if apply_word_to_state(wf, GENE_MAC_REST) != SWAPPED24:
            ok_f_sheet = False

    return {
        "q_W2_is_63": ok_q_w2,
        "q_W2p_is_63": ok_q_w2p,
        "q_F_is_0": ok_q_f,
        "W2_midpoint_is_equality": ok_w2_midpoint,
        "F_maps_rest_to_swapped": ok_f_sheet,
        "shell_path_template": ok_shell_path,
    }


def tau_with_c4(c4: float) -> float:
    return Omega_size * Delta * rho**5 * (f_ordered + c4 * Delta**4)


def g_from_tau(tau: float) -> float:
    return G_kernel * exp(-tau) / (v_EW**2)


def ppm_error(g_value: float) -> float:
    return (g_value / G_meas - 1.0) * 1.0e6


# ============================================================
# Part A: Residual closure (c4 = -7/4)
# ============================================================


def part_a_residual_closure() -> tuple[float, float]:
    print()
    print("=" * 10)
    print("Part A: Residual closure (c4 = -7/4)")
    print("=" * 10)

    print()
    print("1. Wavefunction checks")
    print("-" * 9)
    wf = verify_wavefunction_structure()
    for key, value in wf.items():
        print(f"{key:32s} = {value}")
    wf_ok = all(wf.values())
    print(f"{'wavefunction_structure_ok':32s} = {wf_ok}")

    print()
    print("2. Tensor trace invariant")
    print("-" * 9)
    tr_iso_frac = isotropic_translational_trace_exact()
    tr_iso = float(tr_iso_frac)
    print(f"Tr_sigma_iso exact              = {tr_iso_frac}")
    print(f"Tr_sigma_iso decimal            = {tr_iso:.12f}")

    print()
    print("3. Fourth-order coefficient")
    print("-" * 9)
    c4_kernel_frac = -(Fraction(1, 1) + tr_iso_frac)
    c4_kernel = float(c4_kernel_frac)
    c4_anchor = (
        tau_required / (Omega_size * Delta * rho**5) - f_ordered
    ) / (Delta**4)
    print(f"c4_kernel exact                 = {c4_kernel_frac}")
    print(f"c4_kernel decimal               = {c4_kernel:.12f}")
    print()
    print("Anchor check (inverse from G_meas, validation only):")
    print(f"c4_anchor                       = {c4_anchor:.12f}")
    print(f"c4_anchor - c4_kernel           = {c4_anchor - c4_kernel:.12e}")

    print()
    print("4. Refractive Depth")
    print("-" * 9)
    tau_full = tau_with_c4(c4_kernel)
    print(f"tau_G (leading order)           = {tau_G_formula:.12f}")
    print(f"tau_required                    = {tau_required:.12f}")
    print(f"tau_G (full prediction)         = {tau_full:.12f}")
    print(f"leading_residual                = {tau_G_formula - tau_required:.12e}")
    print(f"full_residual                   = {tau_full - tau_required:.12e}")

    print()
    print("5. Gravitational coupling")
    print("-" * 9)
    g_leading = g_from_tau(tau_G_formula)
    g_full = g_from_tau(tau_full)
    print(f"G_pred (leading order)          = {g_leading:.12e}")
    print(f"G_pred (full prediction)        = {g_full:.12e}")
    print(f"G_measured                      = {G_meas:.12e}")
    print(f"ppm leading                     = {ppm_error(g_leading):+.6f}")
    print(f"ppm full                        = {ppm_error(g_full):+.6f}")

    print()
    print("c4 = -(1 + Tr_sigma_iso)")
    print(f"   = -(1 + {tr_iso_frac})")
    print(f"   = {c4_kernel_frac}")

    return c4_kernel, tau_full


# ============================================================
# Part B: rho^5 / 1+5 STF decomposition
# ============================================================


def part_b_rho5_structure() -> tuple[list[int], list[int], int]:
    print()
    print("=" * 10)
    print("Part B: rho^5 from 1+5 STF decomposition")
    print("=" * 10)

    l_quad = 2
    n_STF = 2 * l_quad + 1
    print(f"Quadrupole (l={l_quad}): {n_STF} STF components")
    print(f"Monopole (l=0): 1 trace component")
    print(f"Total: {n_STF + 1} = 6 (matches SE(3) DoF)")
    print()
    print("Trace (1 DoF): symmetric tensor monopole l=0")
    print("STF (5 DoF): symmetric tensor quadrupole l=2")
    print()

    E_trace = np.eye(3) / np.sqrt(3)
    E_stf = np.zeros((5, 3, 3))
    E_stf[0] = np.diag([1, -1, 0]) / np.sqrt(2)
    E_stf[1] = np.diag([1, 1, -2]) / np.sqrt(6)
    E_stf[2] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    E_stf[3] = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    E_stf[4] = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])

    print("STF basis verification:")
    all_ok = True
    for i in range(5):
        tr = np.trace(E_stf[i])
        norm = np.sum(E_stf[i] ** 2)
        ortho_trace = abs(np.sum(E_stf[i] * E_trace))
        if abs(tr) > 1e-10 or ortho_trace > 1e-10:
            all_ok = False
        print(f"  E_stf[{i}]: trace={tr:+.1e}, norm_sq={norm:.4f}, "
              f"ortho_trace={ortho_trace:.1e}")
    print(f"  All trace-free and orthogonal to trace: {all_ok}")

    gram = np.zeros((6, 6))
    all_basis = np.concatenate([[E_trace], E_stf])
    for i in range(6):
        for j in range(6):
            gram[i, j] = np.sum(all_basis[i] * all_basis[j])
    print(f"  Gram diagonal: {np.diag(gram)}")
    print(f"  Off-diagonal max: {np.max(np.abs(gram - np.diag(np.diag(gram)))):.1e}")

    print()
    print("STF decomposition by payload population:")
    print(f"{'pop':>4} {'Tr(sigma)':>10} {'p':>8} {'||pi||':>8} {'STF amp':>10} {'STF>0':>8}")
    print("-" * 9)
    for pop in range(7):
        tr_sigma = TR_SIGMA_SHELL[pop]
        p = tr_sigma / 3.0
        if 1 <= pop <= 5:
            pi_norm = FA_STF * tr_sigma
            stf_amp = pi_norm / np.sqrt(5)
            stf_flag = "YES"
        else:
            pi_norm = 0.0
            stf_amp = 0.0
            stf_flag = "NO"
        print(f"{pop:>4} {tr_sigma:>10.4f} {p:>8.4f} {pi_norm:>8.4f} "
              f"{stf_amp:>10.4f} {stf_flag:>8}")

    print()
    print("STF attenuation rho^5 and Refractive Depth")
    print("-" * 9)
    tau_per_channel = -np.log(rho)
    print(f"  dim(STF(3)) = {n_STF}; rho^5 = {rho**5:.10f}")
    print(f"  per-channel tau_1 = -ln(rho) = {tau_per_channel:.6f}")

    tau_naive = Omega_size * Delta
    tau_with_stf = Omega_size * Delta * rho**5
    tau_with_correction = Omega_size * Delta * rho**5 * (1 - 4 * rho * Delta**2)
    print()
    print("  Refractive Depth comparison:")
    print(f"    Naive (|Omega|*Delta):              {tau_naive:.6f}")
    print(f"    With rho^5:                         {tau_with_stf:.6f}")
    print(f"    With K4 (1-4*rho*Delta^2):          {tau_with_correction:.6f}")
    print(f"    Required (-ln alpha_G/G_kernel):    {tau_required:.6f}")
    print(f"    Conjugacy depth (2 ln E_CS/v):      {tau_conjugacy_depth:.6f}")
    print()
    print("  K4 correction factor (1 - 4 rho Delta^2):")
    print(f"    tau without K4: {tau_with_stf:.6f}")
    print(f"    tau with K4:    {tau_with_correction:.6f}")
    print(f"    4*rho*D^2:      {4 * rho * Delta**2:.6e}")

    print()
    print("Kernel shells: 5 bulk + 2 horizons")
    print("-" * 9)
    shell_pops = [64 * comb(6, k) for k in range(7)]
    bulk_shells: list[int] = []
    horizon_shells: list[int] = []
    print(f"{'arch_shell':>10} {'Population':>10} {'Fraction':>10} "
          f"{'Tr(sigma)':>10} {'||pi||':>8} {'Type':>15}")
    print("-" * 9)
    for k in range(7):
        frac = shell_pops[k] / Omega_size
        tr_sigma = TR_SIGMA_SHELL[k]
        pi_norm = FA_STF * tr_sigma if 1 <= k <= 5 else 0.0
        if k == 0:
            stype = "HORIZON (comp)"
            horizon_shells.append(k)
        elif k == 6:
            stype = "HORIZON (eq)"
            horizon_shells.append(k)
        else:
            stype = "BULK"
            bulk_shells.append(k)
        print(f"{k:>10} {shell_pops[k]:>10} {frac:>10.4f} {tr_sigma:>10.4f} "
              f"{pi_norm:>8.4f} {stype:>15}")
    print()
    print(f"Horizon arch_shells (||pi||=0): {horizon_shells}")
    print(f"Bulk shells (||pi||>0): {bulk_shells}")
    print(f"Bulk shell count = STF count = {len(bulk_shells)} = {n_STF}: "
          f"{len(bulk_shells) == n_STF}")

    return bulk_shells, horizon_shells, n_STF


# ============================================================
# Part C: Kernel transport
# ============================================================


def part_c_kernel_transport() -> None:
    print()
    print("=" * 10)
    print("Part C: Kernel transport")
    print("=" * 10)

    joint_table = build_joint_table()

    print()
    print("C.1  chi6 bit stats (per step)")
    chi6_step_bit_stats(joint_table)

    print()
    print("C.2  tau_cycle")
    tau_frac = tau_cycle_per_delta_exact()
    tau_cycle_exact = float(tau_frac) * Delta
    print("  Definitive derivation: hqvm_gravity_analysis_3.py section C")
    print(f"  tau_cycle/Delta = {tau_frac} = {float(tau_frac):.12f}")

    print()
    print("C.3  Shell displacement")
    print(f"  D_traverse = {D_traverse} per Z2 holonomy cycle")
    print(f"  W2 maps shell s -> 6-s, traverse from pole = {W2_SHELL_DISPLACEMENT}")
    print(f"  F = W2 o W2' preserves shell, path traverse = {F_CYCLE_PATH_TRAVERSE}")
    print(f"  Z2 holonomy (F o F = id): path traverse = {Z2_HOLONOMY_PATH_TRAVERSE}")
    print(f"  tau_cycle(holonomy) = 2*D_traverse*Delta^2/rho^3 = 48*Delta^2/rho^3")

    print()
    print("C.4  Tr / ||pi|| along Z2 holonomy cycle (64 micro avg)")
    for step_idx in range(1, 9):
        tr_vals: list[float] = []
        pi_vals: list[float] = []
        ratio_vals: list[float] = []
        for m in range(64):
            rows = trace_word_steps(cycle_word_for_micro(m), micro_ref=m)[1:]
            if step_idx > len(rows):
                continue
            arch = rows[step_idx - 1]["arch_shell"]
            tr = TR_SIGMA_SHELL[arch] if 0 <= arch <= 6 else 0.0
            pn = pi_norm_shell(arch)
            tr_vals.append(tr)
            pi_vals.append(pn)
            if tr > 1e-12:
                ratio_vals.append(pn / tr)
        tr_m = sum(tr_vals) / len(tr_vals) if tr_vals else 0.0
        pi_m = sum(pi_vals) / len(pi_vals) if pi_vals else 0.0
        r_m = sum(ratio_vals) / len(ratio_vals) if ratio_vals else 0.0
        print(
            f"  step {step_idx}: mean Tr={tr_m:.4f}  ||pi||={pi_m:.4f}  "
            f"||pi||/Tr={r_m:.4f}"
        )
    print("  (bulk steps: 1,3,5,7; horizon steps: 2,4,6,8)")

    print()
    print("C.5  Per-family tau (depth-4 word)")
    tau_fam: list[float] = []
    for fam in range(4):
        tau_sum = 0.0
        w_sum_f = 0.0
        for m in range(64):
            w = weights_pop[m]
            byte = byte_from_family_and_micro(fam, m)
            for row in trace_word_steps([byte] * 4, micro_ref=m)[1:]:
                tau_sum += w * kappa_pi_step(row["arch_shell"], row["chi6"])
                w_sum_f += w
        tau_word = tau_sum / w_sum_f if w_sum_f > 0 else 0.0
        tau_fam.append(tau_word)
        print(f"  family {fam}: tau_word={tau_word:.9f}  "
              f"tau_step={tau_word / 4:.9f}")
    mean_tw = sum(tau_fam) / 4.0
    var_tw = sum((t - mean_tw) ** 2 for t in tau_fam)
    print(f"  mean tau_word: {mean_tw:.9f}  var: {var_tw:.6e}")

    print()
    print("C.6  N_cycles (from analysis_3 section D)")
    print("  Definitive derivation: hqvm_gravity_analysis_3.py section D")
    n_cycles, tau_cycle_exp, tau_g_full, tau_od = kernel_exposure_constants()
    print(f"  tau_cycle/Delta     = {tau_od} (exact rational)")
    print(f"  N_cycles            = {n_cycles:.4f}  (exposure count)")
    print(f"  N * tau_cycle       = {n_cycles * tau_cycle_exp:.6f}")
    print(f"  tau_G (full)        = {tau_g_full:.6f}")
    rel = abs(n_cycles * tau_cycle_exp - tau_g_full) / tau_g_full
    print(f"  Match N*tau = tau_G: rel err {rel:.2e}")

    print()
    print("C.7  tau_cycle (binom shell weights)")
    tau_binom = tau_cycle_weighted(weights_pop, binom_shell)
    print(f"  tau_binom shell-weighted        = {tau_binom:.9f}")
    print(f"  tau_binom / tau_cycle (exact)   = {tau_binom / tau_cycle_exact:.6f}")
    print("  Horizon steps (arch_shell 0,6): ||pi||=0, zero contribution.")
    per_pop: dict[int, list[float]] = {}
    for micro in range(64):
        p = bin(micro).count("1")
        per_pop.setdefault(p, []).append(tau_path_kappa(micro, binom_shell))
    pop_parts = [
        f"p{p}={sum(v)/len(v):.6f}(n={len(v)})"
        for p, v in sorted(per_pop.items())
    ]
    print(f"  per-pop tau: {' '.join(pop_parts)}")

    n_cycles_meas = tau_G_formula / tau_cycle_exact if tau_cycle_exact > 0 else 0.0

    print()
    print("C.8  Effective rho exponent (data-driven)")
    denom_a = Omega_size * Delta * f_ordered
    a_fit = np.log(tau_required / denom_a) / np.log(rho)
    print(f"  a_fit        = {a_fit:.6f}")
    print(f"  rho^5        = {rho**5:.12f}")
    print(f"  rho^(8-3)    = {rho**8 / rho**3:.12f}")

    print()
    print("C.9  Binomial generating functions")
    binom_rows: list[str] = []
    for k in range(7):
        b5 = comb(5, k) if k <= 5 else 0
        b6 = comb(6, k)
        b5km1 = comb(5, k - 1) if k >= 1 else 0
        binom_rows.append(f"k{k}:{b5}/{b6}/{b6-b5}/{b5km1}")
    print(f"  b5/b6/diff/b5(k-1): {' | '.join(binom_rows)}")
    rho5_coeffs = [(-1) ** k * comb(5, k) for k in range(6)]
    gen6_coeffs = [comb(6, k) for k in range(7)]
    print(f"  (1-Delta)^5 coeffs: {rho5_coeffs}")
    print(f"  (1+Delta)^6 coeffs: {gen6_coeffs}")
    rho5_binom = sum((-1) ** k * comb(5, k) * Delta**k for k in range(6))
    print(f"  rho^5 numeric: {rho**5:.12f}")
    print(f"  (1-Delta)^5 binom: {rho5_binom:.12f}")

    print()
    print("C.10 Residual scaled by |Omega|*Delta^n")
    residual_tau = tau_G_formula - tau_required
    base_od = Omega_size * Delta
    scale_parts = [
        f"Delta^{n}={residual_tau / (base_od * Delta ** (n - 1)):.6f}"
        for n in (3, 4, 5)
    ]
    print(f"  {' '.join(scale_parts)}")
    ratio_d4 = residual_tau / (tau_G_formula * Delta**4)
    print(f"  residual / (tau_G * Delta^4): {ratio_d4:.6f}")
    print(f"  7/4 reference: {7.0 / 4.0:.6f}")

    print()
    print("C.11 F-cycle counting")
    print(f"  N cycles = tau_G / tau_cycle: {n_cycles_meas:.1f}")
    print(f"  tau_G / tau_required: {tau_G_formula / tau_required:.6f}")
    print(f"  K4 factor: {f_ordered:.12f}")

    print()
    print("C.12 Full-payload micro 63 (complement-horizon anchor)")
    micro_63 = 0x3F
    print(f"  pop={bin(micro_63).count('1')}  arch_shell=0 at rest")
    w2_q_ok = all(q_word6_for_items(W2_word(m)) == CHI6_FULL for m in range(64))
    w2p_q_ok = all(q_word6_for_items(W2p_word(m)) == CHI6_FULL for m in range(64))
    print(f"  q(W2)=63 all m  = {w2_q_ok}")
    print(f"  q(W2')=63 all m = {w2p_q_ok}")
    rows_63 = trace_word_steps(cycle_word_for_micro(63), micro_ref=63)
    print(f"  qxor path: {[r['qxor'] for r in rows_63[1:]]}")

    print()
    print("C.13 Joint table (64 micros x 8 steps)")
    print_joint_table_condensed(joint_table)


# ============================================================
# Part D: Delta expansion
# ============================================================


def part_d_delta_expansion() -> None:
    print()
    print("=" * 10)
    print("Part D: Delta expansion (series vs closed form)")
    print("=" * 10)

    p_rho5 = [comb(5, k) * ((-1) ** k) for k in range(6)]
    p_corr = [1.0, 0.0, -4.0, 4.0]
    p_prod = poly_mul(p_rho5, p_corr)
    tau_coeffs = [0.0] + [Omega_size * c for c in p_prod]
    tau_exact = Omega_size * Delta * rho**5 * f_ordered

    cn_ref = np.array([float(c) for c in p_prod], dtype=float)
    cn_meas = np.array(
        [tau_coeffs[n + 1] / Omega_size for n in range(len(p_prod))],
        dtype=float,
    )
    cn_diff = float(np.max(np.abs(cn_ref - cn_meas))) if len(cn_ref) else 0.0
    tau_poly = sum(tau_coeffs[n] * Delta**n for n in range(len(tau_coeffs)))

    print(f"Series = Delta*(1-Delta)^5*(1-4(1-Delta)*Delta^2)")
    print(f"  max |coeff_ref - coeff_meas|   = {cn_diff:.6e}")
    print(f"  max degree                     = {len(tau_coeffs) - 1}")
    print(f"  poly - closed form             = {tau_poly - tau_exact:.6e}")
    if cn_diff < 1e-12 and abs(tau_poly - tau_exact) < 1e-6:
        print("  THEOREM: series matches closed tau_G.")


# ============================================================
# Part E: Coupling summary
# ============================================================


def part_e_summary(
    bulk_shells: list[int],
    horizon_shells: list[int],
    n_STF: int,
) -> None:
    print()
    print("=" * 10)
    print("Part E: Coupling summary")
    print("=" * 10)

    tau_leading = tau_G_formula
    tau_full = tau_g_with_c4(-7.0 / 4.0)
    G_leading = g_pred_from_tau(tau_leading)
    G_pred = g_pred_from_tau(tau_full)
    alpha_G_pred = G_kernel * math.exp(-tau_full)

    print("CGM invariants:")
    print(f"  Q_G    = {Q_G:.10f}")
    print(f"  m_a    = {m_a:.12f}")
    print(f"  d_BU   = {d_BU:.12f}")
    print(f"  rho    = {rho:.12f}")
    print(f"  Delta  = {Delta:.12f}")
    print()
    print("Kernel invariants:")
    print(f"  |Omega|  = {Omega_size}")
    print(f"  |H|      = {H_size}")
    print(f"  D        = {Z2_HOLONOMY_PATH_TRAVERSE}")
    print(f"  G_kernel = pi/6 = {G_kernel:.12f}")
    print()
    print("Refractive Depth:")
    print(f"  leading order tau_G = {tau_leading:.12f}")
    print(f"  full prediction     = {tau_full:.12f}")
    print()
    print("Gravitational coupling:")
    print(f"  G_pred (full)        = {G_pred:.6e} GeV^-2")
    print(f"  G_meas               = {G_meas:.6e} GeV^-2")
    print(f"  ppm (full)           = {(G_pred/G_meas - 1)*1e6:+.3f}")
    print(f"  ppm (leading)        = {(G_leading/G_meas - 1)*1e6:+.3f}")
    print()
    print("Three routes to exponent 5:")
    print(f"  A (STF):    dim(STF(3)) = {n_STF}")
    print(f"  B (shells): bulk shells {bulk_shells} (count {len(bulk_shells)}); "
          f"horizons {horizon_shells}")
    print(f"  C (cycle):  rho exponent 8 - 3 = 5")

    print()
    print("=" * 10)
    print("SUMMARY")
    print("=" * 10)
    print(f"  rho = {rho:.12f}")
    print(f"  Delta = {Delta:.12f}")
    print(f"  STF dim = {n_STF}, bulk shells = {len(bulk_shells)}, |K4| = 4")
    print(f"  tau_G (full) = {tau_full:.6f}")
    print(f"  G_pred/G_meas - 1 (full) = {(G_pred/G_meas - 1) * 1e6:.3f} ppm")


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("=" * 10)
    print("CGM gravity analysis 1: c4 closure, rho^5 structure, G prediction")
    print("=" * 10)

    part_a_residual_closure()
    bulk_shells, horizon_shells, n_STF = part_b_rho5_structure()
    part_c_kernel_transport()
    part_d_delta_expansion()
    part_e_summary(bulk_shells, horizon_shells, n_STF)


if __name__ == "__main__":
    main()
