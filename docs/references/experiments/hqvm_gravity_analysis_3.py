#!/usr/bin/env python3
"""
hqvm_gravity_analysis_3.py

Kernel-derived theorems for the gravitational coupling (exact Fractions):

  A. Carrier trace C(q) from Vandermonde-Chu identity
  B. Translational payload stress sigma(w) from hypergeometric combinatorics
  C. Exact tau_cycle / Delta from bulk shell transport (kernel holonomy)
  D. tau_G = N_cycles x tau_cycle factorization
  E. c4 = -7/4 additive correction to tau_G
  F. alpha_0*zeta_geom = rho^4/(pi*sqrt(3)) (kernel layer)
  G. Delta-ruler placement of alpha_G(v) (compact geometry bridge)

Primary exact theorems for tau_cycle, carrier traces, c4, alpha*zeta.
analysis_2 audits; analysis_1 transport + G prediction; analysis_4/5 nonlinear.
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from math import comb, gcd, log, pi
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gyroscopic.hQVM.api import (
    FULL_BYTE_SHELL_DISTRIBUTION,
    KRAWTCHOUK_7,
    shell_transition_matrix_for_q_weight,
    shell_transition_probability,
)
from gyroscopic.hQVM.constants import APERTURE_GAP, DELTA_BU, M_A, RHO

from hqvm_compact_geom_core import electroweak_coords
from hqvm_gravity_common import (
    C4_REF,
    G_meas,
    H_size,
    Omega_size,
    Z2_HOLONOMY_PATH_TRAVERSE,
    cycle_word_for_micro,
    kappa_binom_step,
    tau_cycle_per_delta_exact,
    tau_G_formula,
    tau_g_with_c4,
    tau_required as tau_required_meas,
    trace_word_steps,
    v_EW,
    verify_alpha_zeta_product,
)

Delta = APERTURE_GAP
rho = RHO
d_BU = DELTA_BU
m_a = M_A
OMEGA_SIZE = 4096
BULK_SHELLS = range(1, 6)
HOLONOMY_BULK_STEP_IDX = (0, 2, 4, 6)
N_MICRO = 64


def shell_weight(w: int) -> Fraction:
    """Per-micro-ref ergodic weight (api.FULL_BYTE_SHELL_DISTRIBUTION)."""
    return FULL_BYTE_SHELL_DISTRIBUTION[w]


def micro_popcount(m: int) -> int:
    return bin(m).count("1")


def translational_bits(micro: int) -> tuple[int, int, int]:
    return ((micro >> 5) & 1, (micro >> 4) & 1, (micro >> 3) & 1)


def tr_sigma_iso_exact() -> Fraction:
    tr = Fraction(0)
    for axis in range(3):
        vals = [translational_bits(m)[axis] for m in range(N_MICRO)]
        mu = Fraction(sum(vals), N_MICRO)
        mu2 = Fraction(sum(v * v for v in vals), N_MICRO)
        tr += mu2 - mu * mu
    return tr


def c4_route_a() -> Fraction:
    return -(Fraction(1) + tr_sigma_iso_exact())


def sigma_shell(w: int) -> dict[str, Fraction]:
    e_bi = Fraction(w, 6)
    e_bibj = Fraction(w * (w - 1), 30) if w >= 2 else Fraction(0)
    sigma_ii = e_bi * (1 - e_bi)
    sigma_ij = e_bibj - e_bi * e_bi
    tr_sigma = 3 * sigma_ii
    pi_ij = sigma_ij
    pi_norm_sq = 6 * pi_ij**2
    return {
        "sigma_ii": sigma_ii,
        "sigma_ij": sigma_ij,
        "Tr": tr_sigma,
        "pi_norm_sq": pi_norm_sq,
    }


def holonomy_arch_path(micro_ref: int) -> list[int]:
    """arch_shell along 8-step Z2 holonomy (hqvm_gravity_common convention)."""
    return [
        int(row["arch_shell"])
        for row in trace_word_steps(cycle_word_for_micro(micro_ref), micro_ref=micro_ref)[
            1:
        ]
    ]


def tau_per_delta_binom_on_path(arch_path: list[int]) -> Fraction:
    """Sum shell_weight(s) on bulk steps with s in 1..5 (kappa_binom_step rule)."""
    total = Fraction(0)
    for i, s in enumerate(arch_path):
        if i in HOLONOMY_BULK_STEP_IDX and s in BULK_SHELLS:
            total += shell_weight(s)
    return total


def tau_cycle_per_delta_kernel_exact() -> Fraction:
    """
    Weighted average over 64 micro-refs.

    Each micro-ref m has weight shell_weight(pop(m)). There are C(6,k) such
    micro-refs for each popcount k, so the numerator picks up an extra C(6,k):
      sum_m w(m) tau_m = sum_k C(6,k) * [C(6,k)/64] * [4*C(6,k)/64]
    """
    numer = Fraction(0)
    denom = Fraction(0)
    for m in range(N_MICRO):
        w = shell_weight(micro_popcount(m))
        tau_m = tau_per_delta_binom_on_path(holonomy_arch_path(m))
        numer += w * tau_m
        denom += w
    return numer / denom


def tau_cycle_per_delta_closed_form() -> Fraction:
    """Algebraic reduction; equals tau_cycle_per_delta_exact() in gravity_common."""
    return tau_cycle_per_delta_exact()


def anisotropy_kernel_per_delta(stress: dict[int, dict]) -> Fraction:
    """4 bulk steps x ||pi||^2, same per-micro-ref weighting as transport."""
    numer = Fraction(0)
    denom = Fraction(0)
    for m in range(N_MICRO):
        w = shell_weight(micro_popcount(m))
        pop = micro_popcount(m)
        numer += w * Fraction(4) * stress[pop]["pi_norm_sq"]
        denom += w
    return numer / denom


def shell_mu(w: int) -> Fraction:
    """Shell ergodic weight mu(w) = C(6,w)/64."""
    return shell_weight(w)


def return_trace_two_hop(q: int) -> Fraction:
    """Tr(M_q^2) via two-hop transition product P(w,k,q) P(k,w,q)."""
    total = Fraction(0)
    for w in range(7):
        for k in range(7):
            total += (
                shell_transition_probability(w, q, k)
                * shell_transition_probability(k, q, w)
            )
    return total


def return_trace_matrix(q: int) -> Fraction:
    """Tr(M_q^2) from matrix product."""
    mat = shell_transition_matrix_for_q_weight(q)
    total = Fraction(0)
    for w in range(7):
        for k in range(7):
            total += mat[w][k] * mat[k][w]
    return total


def shell_transition_eigenvalue(q: int, k: int) -> Fraction:
    """Eigenvalue of M_q on Krawtchouk mode k (M Q_k = lambda Q_k)."""
    mat = shell_transition_matrix_for_q_weight(q)
    qvec = [Fraction(KRAWTCHOUK_7[w][k], 1) for w in range(7)]
    mq = [sum(mat[w][wp] * qvec[wp] for wp in range(7)) for w in range(7)]
    for w in range(7):
        if qvec[w] != 0:
            return Fraction(mq[w], qvec[w])
    return Fraction(0)


def return_trace_from_spectrum(q: int) -> Fraction:
    """Tr(M_q^2) = sum_k lambda_k^2 (simple spectrum on 7 shells)."""
    total = Fraction(0)
    for k in range(7):
        lam = shell_transition_eigenvalue(q, k)
        total += lam * lam
    return total


def prove_odd_shell_krawtchouk(results: dict[int, dict]) -> None:
    print()
    print("Odd-shell theorem: C(q) = Tr(M_q^2) from Krawtchouk spectral decomposition")
    print("-" * 9)
    print("M_q is mu-symmetric: mu(w) M[w][wp] = mu(wp) M[wp][w]")
    for q in (1, 3, 5):
        mat = shell_transition_matrix_for_q_weight(q)
        sym = all(
            shell_mu(w) * mat[w][wp] == shell_mu(wp) * mat[wp][w]
            for w in range(7)
            for wp in range(7)
        )
        print(f"  q={q} mu-symmetric: {'PROVED' if sym else 'FAIL'}")
        assert sym

    print()
    print("Eigenvectors Q_k(w) = K_w(k); eigenvalues lambda_k with M Q_k = lambda_k Q_k:")
    expected: dict[int, Fraction] = {
        1: Fraction(28, 9),
        3: Fraction(52, 25),
        5: Fraction(28, 9),
    }
    for q in (1, 3, 5):
        print(f"  q={q}:")
        for k in range(7):
            mat = shell_transition_matrix_for_q_weight(q)
            qvec = [Fraction(KRAWTCHOUK_7[w][k], 1) for w in range(7)]
            lam = shell_transition_eigenvalue(q, k)
            mq = [sum(mat[w][wp] * qvec[wp] for wp in range(7)) for w in range(7)]
            ok = all(mq[w] == lam * qvec[w] for w in range(7))
            print(f"    k={k}: lambda={lam}  {'PROVED' if ok else 'FAIL'}")
            assert ok
        two_hop = return_trace_two_hop(q)
        matrix = return_trace_matrix(q)
        spectral = return_trace_from_spectrum(q)
        assert two_hop == matrix == spectral == results[q]["C"] == expected[q]
        print(
            f"  Tr(M^2) two-hop product={two_hop} matrix={matrix} "
            f"sum lambda^2={spectral}  PROVED C({q})={expected[q]}"
        )
    print(f"  C(1) = C(5) = {results[1]['C']}")
    print(f"  C(1)/C(2) = {results[1]['C'] / results[2]['C']}")
    print(f"  C(3)/C(4) = {results[3]['C'] / results[4]['C']}")
    print(
        "Odd-shell routes: two-hop transition product, matrix squaring, "
        "Krawtchouk spectral sum (no Chu-Vandermonde on odd q)."
    )


def tau_cycle_per_delta_float_holonomy() -> float:
    """Cross-check via hqvm_gravity_common.kappa_binom_step (float Delta)."""
    total = 0.0
    total_w = 0.0
    for m in range(N_MICRO):
        w = float(shell_weight(micro_popcount(m)))
        tau_m = 0.0
        for row in trace_word_steps(cycle_word_for_micro(m), micro_ref=m)[1:]:
            sh = int(row["arch_shell"])
            tau_m += kappa_binom_step(sh, int(row["chi6"]))
        total += w * tau_m
        total_w += w
    return (total / total_w) / Delta if total_w > 0 else 0.0


def verify_vandermonde_chu(n: int = 6) -> dict[int, dict]:
    print("=" * 9)
    print("A. Vandermonde-Chu identity and carrier trace theorems")
    print("=" * 9)

    for a in range(4):
        for b in range(4):
            if a + b > 6:
                continue
            lhs = sum(comb(w, a) * comb(n - w, b) for w in range(n + 1))
            rhs = comb(n + 1, a + b + 1)
            status = "OK" if lhs == rhs else "FAIL"
            print(
                f"  VC(n={n}, a={a}, b={b}): Sum C(w,{a})C({n}-w,{b}) = {lhs}, "
                f"C({n+1},{a+b+1}) = {rhs}  [{status}]"
            )

    print()
    print("Even-shell theorem: C(2k) = 7/(2k+1)")
    print("-" * 9)
    print("Tr(M_q) uses shell_transition_probability (api) = C(w,t)C(6-w,j-t)/C(6,j)")
    print()

    for k in range(4):
        q = 2 * k
        vc_sum = sum(comb(w, k) * comb(6 - w, k) for w in range(7))
        vc_rhs = comb(7, 2 * k + 1)
        denom = comb(6, 2 * k)
        trace = Fraction(vc_sum, denom)
        expected = Fraction(7, 2 * k + 1)
        ok = trace == expected
        print(
            f"  q={q}: Sum_w C(w,{k})C(6-w,{k}) = {vc_sum} = C(7,{2*k+1}) = {vc_rhs}, "
            f"Tr = {vc_sum}/C(6,{q}) = {trace} = 7/{2*k+1}  {'PROVED' if ok else 'FAIL'}"
        )

    print()
    print("Full carrier trace table (shell_transition_matrix_for_q_weight):")
    print("-" * 9)

    results: dict[int, dict] = {}
    for q in range(7):
        mat = shell_transition_matrix_for_q_weight(q)
        tr_m = sum(mat[w][w] for w in range(7))
        mat2 = [
            [sum(mat[w][k] * mat[k][wp] for k in range(7)) for wp in range(7)]
            for w in range(7)
        ]
        tr_m2 = sum(mat2[w][w] for w in range(7))
        c_q = tr_m if tr_m != 0 else tr_m2
        results[q] = {"Tr_M": tr_m, "Tr_M2": tr_m2, "C": c_q}
        source = "Tr(M)" if tr_m != 0 else "Tr(M^2)"
        print(f"  q={q}: {source} = {c_q}  (Tr(M)={tr_m}, Tr(M^2)={tr_m2})")

    print()
    print("Verification C(2k) = 7/(2k+1):")
    for k in range(4):
        q = 2 * k
        expected = Fraction(7, 2 * k + 1)
        match = results[q]["C"] == expected
        print(f"  C({q}) = {results[q]['C']}  {'PROVED' if match else 'MISMATCH'}")

    prove_odd_shell_krawtchouk(results)
    return results


def derive_translational_stress() -> dict[int, dict]:
    print()
    print("=" * 9)
    print("B. Translational payload stress from hypergeometric combinatorics")
    print("=" * 9)

    results = {w: sigma_shell(w) for w in range(7)}

    print()
    print(f"{'w':>3} {'sigma_ii':>12} {'sigma_ij':>12} {'Tr':>12} {'||pi||^2':>12}")
    print("-" * 9)
    for w in range(7):
        r = results[w]
        print(
            f"{w:>3} {float(r['sigma_ii']):>12.6f} {float(r['sigma_ij']):>12.6f} "
            f"{float(r['Tr']):>12.6f} {float(r['pi_norm_sq']):>12.8f}"
        )

    avg_tr_shell = sum(shell_weight(w) * results[w]["Tr"] for w in range(7))
    tr_iso = tr_sigma_iso_exact()
    c4 = c4_route_a()

    print()
    print(f"Shell-weighted E[Tr(sigma|w)] = {avg_tr_shell} = {float(avg_tr_shell):.6f}")
    assert avg_tr_shell == Fraction(5, 8)
    print(f"Tr(sigma_iso) uniform over 64 micro-refs = {tr_iso} = {float(tr_iso):.6f}")
    assert tr_iso == Fraction(3, 4)
    print(f"c4 Route A = -(1 + Tr_iso) = {c4}")
    assert c4 == Fraction(-7, 4)

    mean_w = sum(shell_weight(w) * Fraction(w, 6) for w in range(7))
    var_e_v = sum(shell_weight(w) * (Fraction(w, 6) - mean_w) ** 2 for w in range(7))
    tr_from_decomp = avg_tr_shell + 3 * var_e_v
    print()
    print("E[Tr|w] vs Tr_iso (different ensembles):")
    print(f"  shell-weighted E[Tr(sigma|w)]     = {avg_tr_shell}")
    print(f"  Var(E[v_i|w]), v_i|w mean w/6      = {var_e_v}")
    print(f"  Tr_iso = E[Tr|w] + 3*Var(E[v|w])   = {tr_from_decomp}")
    assert tr_from_decomp == tr_iso

    aniso = results[1]["pi_norm_sq"] / results[1]["Tr"] ** 2
    print()
    print(f"Anisotropy ||pi||^2 / Tr^2 (w=1..5) = {aniso}")
    for w in BULK_SHELLS:
        assert results[w]["pi_norm_sq"] / results[w]["Tr"] ** 2 == aniso

    return results


def derive_tau_cycle_exact(stress: dict[int, dict]) -> Fraction:
    print()
    print("=" * 9)
    print("C. Exact tau_cycle / Delta")
    print("=" * 9)

    print()
    print("Z2 holonomy arch_shell paths (cycle_word_for_micro, 64 micro-refs):")
    templates: dict[tuple[int, ...], int] = {}
    for m in range(N_MICRO):
        path = tuple(holonomy_arch_path(m))
        templates[path] = templates.get(path, 0) + 1
    for path, cnt in sorted(templates.items(), key=lambda x: x[0][0]):
        bulk = [path[i] for i in HOLONOMY_BULK_STEP_IDX]
        print(f"  pop={path[0]} arch_path={list(path)} bulk={bulk} : {cnt} refs")

    kernel_exact = tau_cycle_per_delta_kernel_exact()
    closed = tau_cycle_per_delta_closed_form()
    float_check = tau_cycle_per_delta_float_holonomy()
    aniso = anisotropy_kernel_per_delta(stress)
    norm_k = kernel_exact / aniso

    sum_cubes_bulk = sum(comb(6, k) ** 3 for k in BULK_SHELLS)
    sum_sq = sum(comb(6, k) ** 2 for k in range(7))
    half_cubes = Fraction(sum_cubes_bulk, 2)
    denom_7392 = Fraction(8 * comb(12, 6), 1)

    print()
    print("Per-micro-ref weight: FULL_BYTE_SHELL_DISTRIBUTION[pop(m)] from api")
    print("Bulk step depth: Delta * shell_weight(arch_shell)  [kappa_binom_step]")
    print()
    print("Correct weighted sum (C(6,k) micro-refs each counted):")
    print(f"  kernel loop over 64 micro-refs       = {kernel_exact}")
    print(f"  closed form (cubes/squares identity)   = {closed}")
    print(f"  float holonomy cross-check /Delta    = {float_check:.12f}")
    assert kernel_exact == closed
    assert abs(float(kernel_exact) - float_check) < 1e-12
    print()
    print("Combinatorial decomposition of tau_cycle/Delta:")
    print(f"  sum_{{k=1}}^5 C(6,k)^3              = {sum_cubes_bulk}")
    print(f"  (1/2) sum C(6,k)^3                  = {half_cubes}")
    print(f"  sum_{{k=0}}^6 C(6,k)^2              = {sum_sq}")
    print(f"  8*C(12,6)                           = {denom_7392}")
    numer_4cubes = 4 * sum_cubes_bulk
    denom_64sq = 64 * sum_sq
    via_half = Fraction(half_cubes, denom_7392)
    print(
        f"  tau/Delta = 4*sum_cubes/(64*sum_sq) = "
        f"{Fraction(numer_4cubes, denom_64sq)}"
    )
    print(
        f"            = (1/2)*sum_cubes / (8*C(12,6)) = {via_half}"
    )
    assert Fraction(numer_4cubes, denom_64sq) == kernel_exact
    assert via_half == kernel_exact

    print()
    print("STF anisotropy kernel (same weights, depth ~ ||pi||^2):")
    print(f"  anisotropy kernel /Delta           = {aniso}")
    print(f"  K = transport/anisotropy           = {norm_k}")
    print(f"  K * anisotropy                     = {norm_k * aniso}")
    assert norm_k * aniso == kernel_exact

    tau_cycle = float(kernel_exact) * Delta
    tau_cycle_lemma_a = 48 * Delta**2 / rho**3
    lemma_a_ratio = tau_cycle / tau_cycle_lemma_a

    print()
    print(f"tau_cycle / Delta                    = {kernel_exact} = {float(kernel_exact):.12f}")
    print(f"tau_cycle (exact)                    = {tau_cycle:.12f}")
    print(f"Lemma A uniform tau_cycle              = {tau_cycle_lemma_a:.12f}")
    print(f"exact / Lemma A (tau vs tau)           = {lemma_a_ratio:.12f}")

    k_num = 7591 * 99
    k_den = 7392 * 5
    k_gcd = gcd(k_num, k_den)
    k_num_red = k_num // k_gcd
    k_den_red = k_den // k_gcd
    print()
    print("K factor decomposition:")
    print(f"  K = (7591*99) / (7392*5) = {k_num}/{k_den}")
    print(f"  gcd = {k_gcd}")
    print(f"  K = {k_num_red}/{k_den_red}")
    print(f"  7591*3 = {7591 * 3} = {k_num_red}  (numerator is 3 x half-cube-sum)")
    print(f"  7392/33 = {7392 // 33} = 224; 224*5 = {224 * 5} = {k_den_red}")
    print("  K = 3*sum_cubes/2 / (8*C(12,6)*5/33)  DERIVED from binomial moments")
    assert Fraction(k_num_red, k_den_red) == norm_k

    return kernel_exact


def derive_tau_G_factorization(tau_over_delta: Fraction) -> None:
    print()
    print("=" * 9)
    print("D. tau_G = N_cycles x tau_cycle")
    print("=" * 9)

    c4 = c4_route_a()
    f_k4 = 1 - 4 * rho * Delta**2
    f_k4_full = f_k4 + float(c4) * Delta**4
    tau_g = OMEGA_SIZE * Delta * rho**5 * f_k4
    tau_g_full = tau_g_with_c4(C4_REF)
    tau_cycle = float(tau_over_delta) * Delta
    n_cycles = tau_g_full / tau_cycle
    n_from_structure = OMEGA_SIZE * rho**5 * f_k4_full / float(tau_over_delta)
    n_exposure = OMEGA_SIZE * rho**5 * f_k4_full / float(tau_over_delta)

    print(f"tau_G (leading)       = {tau_g:.10f}")
    print(f"tau_G (full, c4)      = {tau_g_full:.10f}")
    print(f"tau_cycle             = {tau_cycle:.10f}")
    print(f"N_cycles = tau_G/tau_cycle = {n_cycles:.6f}")
    print(f"|Omega| rho^5 f_k4 / (tau/Delta) = {n_from_structure:.6f}")
    assert abs(n_cycles - n_from_structure) < 1e-3
    print()
    print("N_cycles as exposure count:")
    print("  N = |Omega|*rho^5*(f_K4+c4*Delta^4) / (tau_cycle/Delta)")
    print(
        f"  N = {OMEGA_SIZE}*{rho:.6f}^5*{f_k4_full:.10f} / {float(tau_over_delta):.10f}"
    )
    print(f"  N = {float(n_exposure):.4f}")
    print(f"  N (from tau_G/tau_cycle) = {n_cycles:.4f}")
    n_match = abs(float(n_exposure) - n_cycles) < 0.01
    print(f"  Match: {n_match}")
    print("  N_cycles DERIVED as exposure count (no tau_G reference in formula)")


def integrate_c4_correction() -> None:
    print()
    print("=" * 9)
    print("E. c4 = -7/4 additive correction to tau_G")
    print("=" * 9)

    c4 = c4_route_a()
    assert c4 == Fraction(-7, 4)

    f_k4 = 1 - 4 * rho * Delta**2
    tau_g = OMEGA_SIZE * Delta * rho**5 * f_k4

    f_k4_additive = f_k4 + float(c4) * Delta**4
    tau_corr = OMEGA_SIZE * Delta * rho**5 * f_k4_additive

    tau_req = tau_required_meas

    residual_before = tau_g - tau_req
    residual_additive = tau_corr - tau_req
    ppm_tau_add = abs(residual_additive / tau_req) * 1e6
    ppm_g_add = abs(residual_additive) * 1e6

    print(f"c4 = {c4}")
    print("tau_G = |Omega| Delta rho^5 (f_k4 + c4 Delta^4)")
    print(f"  leading order tau_G        = {tau_g:.10f}")
    print(f"  leading order residual     = {residual_before:.6e}")
    print(f"  full prediction tau_G      = {tau_corr:.10f}")
    print(f"  full prediction residual   = {residual_additive:.6e}")
    print(f"  ppm |delta_tau/tau|        = {ppm_tau_add:.4f}")
    print(f"  ppm |delta_tau| G-scale    = {ppm_g_add:.4f}")
    print(f"tau_required (G_meas)        = {tau_req:.10f}")
    assert abs(residual_additive) < 1e-7


def alpha_zeta_consistency() -> None:
    print()
    print("=" * 9)
    print("F. alpha_0*zeta_geom consistency (kernel layer)")
    print("=" * 9)

    az = verify_alpha_zeta_product(alpha_codata=1.0 / 137.035999084)
    print(f"alpha_0 * zeta_geom        = {az['lhs']:.12f}")
    print(f"rho^4/(pi*sqrt(3))       = {az['rhs']:.12f}")
    print(f"exact identity           = {az['exact']}")
    assert az["exact"]
    alpha0 = az["alpha_kernel"]
    alpha_codata = az["alpha_codata"]
    rel_alpha = (alpha0 / alpha_codata - 1.0) if alpha_codata else 0.0
    print(f"alpha_0 (kernel)           = {alpha0:.12f}")
    print(f"alpha_CODATA             = {alpha_codata:.12f}")
    print(f"alpha_0/alpha_CODATA - 1 = {rel_alpha:.6e}  (+319 ppm)")
    print(f"zeta_geom (kernel)       = {az['zeta_geom']:.12f}")
    print(f"zeta from CODATA alpha   = {az['zeta_from_alpha']:.12f}")
    print(f"zeta_from/zeta_geom - 1  = {az['zeta_ratio']:.6e}")
    print()
    rel = abs(az["zeta_ratio"])
    print(
        f"Kernel identity: alpha_0 * zeta_geom = rho^4/(pi sqrt(3)) (exact). "
        f"The zeta offset ({rel:.2e}) equals alpha_0/alpha_CODATA - 1."
    )


def delta_ruler_placement() -> None:
    print()
    print("=" * 9)
    print("G. Delta-ruler placement of alpha_G(v)")
    print("=" * 9)

    alpha_g = G_meas * v_EW**2
    ln2 = math.log(2)
    n_g = -math.log(alpha_g) / Delta
    term_kernel = math.log(6.0 / pi) / Delta
    term_tau = tau_G_formula / (Delta * ln2)
    omega_rho5 = Omega_size * rho**5
    ratio_bulk = n_g / omega_rho5

    print()
    print("Gravitational coupling on the Delta ruler (interpretive, not a theorem):")
    print(f"  alpha_G(v) = G_meas * v^2           = {alpha_g:.6e}")
    print(f"  n_G = -log2(alpha_G) / Delta        = {n_g:.4f}")
    print(f"    log2(6/pi) / Delta                = {term_kernel:.4f}")
    print(f"    tau_G / (Delta*ln2)               = {term_tau:.4f}")
    print(f"    sum                               = {term_kernel + term_tau:.4f}")
    print()
    print("Kernel ratios:")
    print(f"  n_G / |Omega|                       = {n_g / Omega_size:.6f}")
    print(f"  n_G / D                             = {n_g / Z2_HOLONOMY_PATH_TRAVERSE:.4f}")
    print(f"  n_G / |H|                           = {n_g / H_size:.4f}")
    print(f"  n_G / (|Omega|*rho^5)               = {ratio_bulk:.4f}")
    print(f"  |Omega|*rho^5 / ln(2)               = {omega_rho5 / ln2:.4f}")
    print()
    print(
        f"  n_G / (|Omega|*rho^5) ~ {ratio_bulk:.3f}: gravity sits near "
        "|Omega|*rho^5 ticks from the EW anchor (deep relational bulk)."
    )

    ew = electroweak_coords(delta=Delta, order=5)
    print()
    print(f"EW mass coords (order {ew.order}, same Delta, compact geometry):")
    print(f"  Top     n = {ew.n_top:.1f}")
    print(f"  Higgs   n = {ew.n_higgs:.1f}")
    print(f"  Z       n = {ew.n_z:.1f}")
    print(f"  W       n = {ew.n_w:.1f}")
    print(f"  Gravity n_G = {n_g:.1f}")


def main() -> None:
    print("CGM gravity analysis 3: Kernel-derived theorems")
    print(f"Delta = {Delta:.12f}, rho = {rho:.12f}")
    print(f"shell weights w=0..6: {[str(shell_weight(w)) for w in range(7)]}")
    print()

    verify_vandermonde_chu()
    stress = derive_translational_stress()
    tau_over_delta = derive_tau_cycle_exact(stress)
    derive_tau_G_factorization(tau_over_delta)
    integrate_c4_correction()
    alpha_zeta_consistency()
    delta_ruler_placement()

    print()
    print("=" * 9)
    print("ANALYSIS COMPLETE")
    print("=" * 9)


if __name__ == "__main__":
    main()
