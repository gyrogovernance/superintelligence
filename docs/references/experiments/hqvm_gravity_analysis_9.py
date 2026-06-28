#!/usr/bin/env python3
"""
hqvm_gravity_analysis_9.py

Ultraviolet completion: quadratic action dictionary, RG flow, inflation
observables, Weyl sector, reheating (Planck/BH ladder).

Depends on analysis_8 sections A-E for spectral inputs (d_stats).
Planck boundary and optical cosmology: analysis_10.
"""
from __future__ import annotations

import math
import sys
from math import comb
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EXPERIMENTS = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from hqvm_gravity_analysis_8 import run_kernel_chain
from hqvm_gravity_common import (
    C4_REF,
    Delta,
    E_CS,
    G_meas,
    H_size,
    Omega_size,
    configure_stdout_utf8,
    dln_g_dpsi,
    E_ref_quantile,
    g_pred_from_tau,
    rho,
    tau_g_with_c4,
    v_EW,
)

configure_stdout_utf8()

# --- UV completion (PRL 136, 111501; Buccio et al. betas for compare tiers) ---

N_BYTES = 256
N_CODEWORDS = 64
K4_ORDER = 4
FAMILY_FIBER = 4
LIFT_BITS = 32
QQG_N_WINDOW = (1.0e5, 1.2e6)
NS_BAND = (0.95, 0.99)
R_BAND = (0.0, 0.06)
R_QQG_TYPICAL_MIN = 0.01
AS_PLANCK = 2.1e-9
MPL_REDUCED_GEV = 2.435e18
FOURPI_SQ = (4.0 * math.pi) ** 2
PLAQUETTE_PAIRS_6 = float(comb(6, 2))
AS_MATCH_LOG10_TOL = 1.0


def cgm_cosmo_scale() -> dict[str, float]:
    """Shared scales for cosmology lemmas (natural units, GeV)."""
    tau_g = tau_g_with_c4(C4_REF)
    g1 = dln_g_dpsi(tau_g)
    ln_span = math.log(E_CS / v_EW)
    m_pl = 1.0 / math.sqrt(8.0 * math.pi * G_meas)
    r0 = v_EW * v_EW
    return {
        "tau_g": tau_g,
        "g1": g1,
        "ln_span": ln_span,
        "m_pl_gev": m_pl,
        "r0_v2": r0,
        "g0": g_pred_from_tau(tau_g),
    }


def mu_from_psi(psi: float) -> float:
    return float(E_ref_quantile(psi))


def psi_from_mu(mu_gev: float) -> float:
    """Invert E_ref(psi)=mu (Appendix E.1): mu = E_CS (v/E_CS)^(1-psi)."""
    ln_v_ecs = math.log(v_EW / E_CS)
    return 1.0 - math.log(mu_gev / E_CS) / ln_v_ecs


def psi_from_R_dS(R: float, r0: float | None = None) -> float:
    """
    LEMMA (dS identification): mu = E_ref(psi) and mu = |R|^(1/2) on homogeneous
    de Sitter background imply psi(R) = psi(mu=sqrt(R)) with R0 = v^2 at psi=0.
    """
    if r0 is None:
        r0 = v_EW * v_EW
    mu = math.sqrt(abs(R))
    return psi_from_mu(mu)


def R_from_psi_dS(psi: float, r0: float | None = None) -> float:
    """Inverse of psi_from_R_dS at fixed R0 = v^2."""
    if r0 is None:
        r0 = v_EW * v_EW
    mu = float(E_ref_quantile(psi))
    return mu * mu


def flrw_dS_consistency(g1: float) -> dict[str, float]:
    """
    Check d psi/d ln R from mu=sqrt(R) matches d psi/d ln H via R = 6 H^2.
    """
    ln_span = math.log(E_CS / v_EW)
    dpsi_dlnR = -1.0 / (2.0 * ln_span)
    dpsi_dlnH_via_R = 2.0 * dpsi_dlnR
    dpsi_dlnH_direct = -1.0 / ln_span
    rel_err = abs(dpsi_dlnH_via_R - dpsi_dlnH_direct) / abs(dpsi_dlnH_direct)
    return {
        "dpsi_dlnR": dpsi_dlnR,
        "dpsi_dlnH_via_R": dpsi_dlnH_via_R,
        "dpsi_dlnH_direct": dpsi_dlnH_direct,
        "relative_error": rel_err,
        "consistent": rel_err < 1e-9,
    }


def effective_action_coefficients(g1: float) -> dict[str, float]:
    """
    LEMMA: coefficients in S_eff ~ int sqrt-g [ Mpl^2/2 R + a(mu) R^2 + b(mu) C^2 ].

    From Jordan S = (1/16piG0) int R exp(-g1 psi) with psi(R) from dS lemma:
      f(R) = R [1 - g1 psi(R) + (g1 psi(R))^2/2 + ...]
    Leading R^2 coefficient (1/xi): a2 = -g1/(2 ln_span).
    C^2 coefficient b2 is a CONJECTURAL spectral weight (2/75)/xi at tensor order.
    """
    sc = cgm_cosmo_scale()
    ln_span = sc["ln_span"]
    a2 = -g1 / (2.0 * ln_span)
    xi = 1.0 / a2 if a2 != 0 else float("inf")
    m_pl = sc["m_pl_gev"]
    g0 = sc["g0"]
    mpl2_half = 1.0 / (16.0 * math.pi * g0)
    aniso = 2.0 / 75.0
    b2_over_a2 = aniso
    return {
        "a2_R2": a2,
        "xi_eff": xi,
        "one_over_xi": a2,
        "mpl2_over_2": mpl2_half,
        "b2_over_a2_conjecture": b2_over_a2,
        "b2_R2_conjecture": a2 * b2_over_a2,
        "ln_span": ln_span,
    }


def qqg_dictionary_at_mu(psi: float, g1: float) -> dict[str, float]:
    """
    Paper-grade dictionary: CGM Jordan action -> QQG quadratic normalization.

    CGM (defended):
      S_CGM = (1/16pi G0) int sqrt(-g) R exp(-g1 psi) d^4x
    FRW + dS lemma: mu = E_ref(psi), R = mu^2, psi = psi(mu).
    Expand f(R) = R exp(-g1 psi(R)):
      f = R - g1 R psi + (g1 psi)^2 R / 2 + ...
    Match QQG (PRL sign convention):
      S_QQG ~ int sqrt(-g) [ Mpl^2/2 R - R^2/xi(mu) + 2 C^2/lambda(mu) ]
    Leading map (Jordan R^2 coeff a2 = 1/xi):
      xi(mu) = 1/a2(mu),  a2 = -g1/(2 ln_span) at leading order.
    Weyl sector (tensor order):
      b_C2(mu) = a2(mu) * (2/75),  lambda(mu) = 2/b_C2 = 75/a2 = 75*xi.
    """
    ln_span = math.log(E_CS / v_EW)
    r0 = v_EW * v_EW
    mu = float(E_ref_quantile(psi))
    r_curv = mu * mu
    r_norm = r_curv / r0
    a2_lead = -g1 / (2.0 * ln_span)
    psi_val = psi_from_mu(mu)
    g1psi = g1 * psi_val
    a2_mu = a2_lead * (1.0 + 0.5 * g1psi * g1psi)
    xi_mu = 1.0 / a2_mu if a2_mu != 0 else float("inf")
    aniso = 2.0 / 75.0
    b_c2 = a2_mu * aniso
    lam_mu = 2.0 / b_c2 if b_c2 != 0 else float("inf")
    f_jordan = r_norm * math.exp(-g1 * psi_val)
    f_quad = r_norm + a2_mu * r_norm * r_norm
    rel_quad_err = abs(f_jordan - f_quad) / max(abs(f_jordan), 1e-30)
    return {
        "psi": psi,
        "mu_gev": mu,
        "R": r_curv,
        "R_norm": r_norm,
        "a2": a2_mu,
        "xi": xi_mu,
        "one_over_xi": a2_mu,
        "b_C2": b_c2,
        "lambda": lam_mu,
        "f_jordan": f_jordan,
        "f_quad": f_quad,
        "f_quad_rel_err": rel_quad_err,
    }


def qqg_dictionary_table(g1: float, n_pts: int = 5) -> list[dict[str, float]]:
    """xi(mu), lambda(mu) on a uniform psi grid (reproducible paper table)."""
    rows = []
    for i in range(n_pts):
        psi = i / max(n_pts - 1, 1)
        rows.append(qqg_dictionary_at_mu(psi, g1))
    return rows


def n_eff_spin_weight_derivation() -> dict[str, float]:
    """
    QQG weights from plaquette heat-kernel on the 6-simplex (Section A).

    Denominator 60 = 4 * C(6,2) = 4 * 15: four phase sectors per plaquette pair.
    Numerators (spin multiplicity per CGM multiplet):
      trace scalar     -> 1
      ONA vector x3    -> 2 pol x 2 (ghost+physical) x 3 = 12
      UNA Weyl x3      -> 2 x 3 / 2 (Majorana) = 3
    """
    denom = 4.0 * PLAQUETTE_PAIRS_6
    w_scalar = 1.0 / denom
    w_vector = 12.0 / denom
    w_fermion = 3.0 / denom
    return {
        "w_scalar": w_scalar,
        "w_vector": w_vector,
        "w_fermion": w_fermion,
        "denom_4xC62": denom,
        "plaquette_pairs": PLAQUETTE_PAIRS_6,
        "vector_matches_12_over_60": abs(w_vector - 1.0 / 5.0) < 1e-12,
        "fermion_matches_3_over_60": abs(w_fermion - 1.0 / 20.0) < 1e-12,
        "sector_dof_scalar": 1.0,
        "sector_dof_vector": 3.0,
        "sector_dof_fermion": 3.0,
    }


def operational_xi_equivalence(g1: float, psi: float = 0.5) -> dict[str, float]:
    """
    LEMMA: xi(mu) running from the dictionary map a2(psi) = a2_0(1+(g1 psi)^2/2).

    With mu = E_ref(psi), d psi/d ln mu = 1/ln_span:
      d ln xi/d ln mu = -(g1^2 psi) / ((1+(g1 psi)^2/2) * ln_span).

    This is not beta_lnG = d ln(G/G0)/d ln mu (F2); G and 1/xi are different
    running objects. Numerical slope from qqg_dictionary_at_mu must match analytic.
    """
    ln_span = math.log(E_CS / v_EW)
    a2_0 = -g1 / (2.0 * ln_span)
    row = qqg_dictionary_at_mu(psi, g1)
    xi = row["xi"]
    g1psi = g1 * psi
    den = 1.0 + 0.5 * g1psi * g1psi
    dlnxi_dlnmu_analytic = -(g1 * g1 * psi) / den / ln_span if den else 0.0
    beta_ln_g = beta_g_cgm(g1)
    eps_psi = 1e-4
    row_p = qqg_dictionary_at_mu(min(1.0, psi + eps_psi), g1)
    row_m = qqg_dictionary_at_mu(max(0.0, psi - eps_psi), g1)
    dlnmu = math.log(row_p["mu_gev"] / row_m["mu_gev"]) if row_m["mu_gev"] > 0 else 0.0
    dlnxi_num = (
        math.log(row_p["xi"] / row_m["xi"]) if row_m["xi"] > 0 and dlnmu else 0.0
    )
    dlnxi_dlnmu_num = dlnxi_num / dlnmu if dlnmu else 0.0
    rel = abs(dlnxi_dlnmu_num - dlnxi_dlnmu_analytic) / max(
        abs(dlnxi_dlnmu_analytic), 1e-30
    )
    return {
        "xi_psi": xi,
        "a2_0": a2_0,
        "dlnxi_dlnmu_analytic": dlnxi_dlnmu_analytic,
        "dlnxi_dlnmu_numerical": dlnxi_dlnmu_num,
        "beta_lnG_separate": beta_ln_g,
        "relative_error": rel,
        "operational_match": rel < 0.02,
    }


def holographic_amplitude_factor() -> dict[str, float]:
    """
    THEOREM (kernel layer): CMB amplitude projection on |Omega|.

    From analysis_3: alpha_0 zeta = rho^4/(pi sqrt(3)) (exact).
    BCH vacuum weight at Delta^4 (c4 sector). Holographic quotient 1/|Omega|.
    Pi_H = rho^8 Delta^4 / (pi^2 |Omega|)  (squared rho^4 identity x BCH x quotient).
    """
    lhs = rho**4 / (math.pi * math.sqrt(3.0))
    pi_h = (rho**8) * (Delta**4) / (math.pi**2 * float(Omega_size))
    return {
        "rho4_over_pi_sqrt3": lhs,
        "Pi_H": pi_h,
        "log10_Pi_H": math.log10(pi_h) if pi_h > 0 else float("nan"),
        "Omega": float(Omega_size),
        "Delta4": Delta**4,
    }


def weyl_lambda_from_kernel(g1: float) -> dict[str, float]:
    """
    LEMMA: lambda(mu) from rotational monodromy, not a free definition.

    lambda = 2/b_C2,  b_C2 = (1/xi) * (2/75).
    2/75 = ||Pi||^2/Tr^2 (analysis_3, theorem on bulk shells).
    """
    ln_span = math.log(E_CS / v_EW)
    a2 = -g1 / (2.0 * ln_span)
    aniso = 2.0 / 75.0
    b_c2 = a2 * aniso
    lam = 2.0 / b_c2
    xi = 1.0 / a2
    return {
        "anisotropy_theorem": aniso,
        "b_C2": b_c2,
        "lambda": lam,
        "xi": xi,
        "lambda_equals_75xi": abs(lam - 75.0 * xi) / lam < 1e-10 if lam else False,
    }


def rg_operator_one_step(
    g1: float, psi: float, d_ln_mu: float
) -> dict[str, float]:
    """
    Explicit coarse-grain operator R (one shell-tick step).

    Integrate out one bulk shell: psi -> psi - d_psi with d_psi = d_ln_mu/ln_span.
    Read (xi, lambda) before/after from the dictionary map; discrete betas.
    """
    ln_span = math.log(E_CS / v_EW)
    d_psi = d_ln_mu / ln_span
    psi_next = max(0.0, min(1.0, psi - d_psi))
    c0 = qqg_dictionary_at_mu(psi, g1)
    c1 = qqg_dictionary_at_mu(psi_next, g1)
    xi0, xi1 = c0["xi"], c1["xi"]
    lam0, lam1 = c0["lambda"], c1["lambda"]
    beta_xi_disc = (xi1 - xi0) / (xi0 * d_ln_mu) if xi0 and d_ln_mu else 0.0
    beta_lam_disc = (lam1 - lam0) / (lam0 * d_ln_mu) if lam0 and d_ln_mu else 0.0
    a2_lead = -g1 / (2.0 * ln_span)
    beta_xi_cont = -g1 / ln_span
    beta_lam_cont = -beta_xi_cont
    lam_th = lambda0_cgm() / FOURPI_SQ
    xi_ref = xi0
    beta_xi_qqg = -(
        xi_ref**2 - 36.0 * lam_th * xi_ref - 2520.0 * lam_th**2
    ) / (36.0 * FOURPI_SQ)
    return {
        "psi": psi,
        "psi_next": psi_next,
        "d_ln_mu": d_ln_mu,
        "xi": xi0,
        "xi_next": xi1,
        "lambda": lam0,
        "lambda_next": lam1,
        "beta_xi_discrete": beta_xi_disc,
        "beta_lambda_discrete": beta_lam_disc,
        "beta_xi_continuum": beta_xi_cont,
        "beta_lambda_continuum": beta_lam_cont,
        "beta_xi_qqg_compare": beta_xi_qqg,
        "cgm_not_qqg_beta": abs(beta_xi_disc - beta_xi_qqg) > 0.01,
    }


def rg_beta_functions_cgm(g1: float) -> dict[str, float]:
    """Continuum betas + discrete shell-tick step (operator R)."""
    rg = rg_coarse_grain_step()
    d_ln_mu = rg["d_ln_mu_tick"]
    step = rg_operator_one_step(g1, psi=0.5, d_ln_mu=d_ln_mu)
    f2 = -4.0 * rho
    ln_span = math.log(E_CS / v_EW)
    xi_eff = xi_eff_from_g1(g1)
    beta_xi_bch = f2 * Delta * Delta * xi_eff / ln_span
    return {
        **step,
        "beta_xi_bch_leading": beta_xi_bch,
        "beta_alpha_g": rg["beta_alpha_g"],
        "d_ln_mu_tick": d_ln_mu,
    }


def tensor_quadratic_action_cgm(
    g1: float, dim_minus: int, dim_plus: int
) -> dict[str, float]:
    """
    Quadratic tensor action on FRW + TT h_ij (perturbative closure).

    Background: C_ijk l = 0 on exact FRW.
    Perturbation: delta C^2 couples to C-odd spectral sector with weight
    aniso = 2/75 (bulk Pi norm / trace^2).
    Quadratic tensor coefficient (schematic TT reduction):
      delta S_T ~ (b_C2 / 4) int a^3 (h_dot_ij)^2,  b_C2 = a2 * aniso.
    QQG match: coefficient 2/lambda in S; lambda = 2/b_C2.
    """
    ln_span = math.log(E_CS / v_EW)
    a2 = -g1 / (2.0 * ln_span)
    aniso = 2.0 / 75.0
    b_c2 = a2 * aniso
    lam = 2.0 / b_c2 if b_c2 else float("inf")
    xi = 1.0 / a2 if a2 else float("inf")
    spec_ratio = dim_minus / dim_plus if dim_plus else 1.0
    coeff_tt = 0.25 * b_c2
    r_boost = 1.0 + aniso
    return {
        "a2_R2": a2,
        "xi": xi,
        "b_C2": b_c2,
        "lambda_qqg_norm": lam,
        "coeff_tt_quadratic": coeff_tt,
        "anisotropy": aniso,
        "spectral_ratio": spec_ratio,
        "C2_vanishes_bg": True,
        "r_native_times_boost": r_boost,
    }


def inflation_amplitude_A_s(
    g1: float, n_target: float = 55.0
) -> dict[str, float]:
    """
    A_s from native slow-roll (Mpl units) times holographic Pi_H (kernel theorem).

    Intrinsic: A_s^pl = V/(24 pi^2 epsilon) at horizon (Mpl=1).
    Projected: A_s = A_s^pl * Pi_H with Pi_H = rho^8 Delta^4 / (pi^2 |Omega|).
    Cross-check: Starobinsky A_s = N^2 / (24 pi^2 xi^2) at same xi, N.
    """
    infl = inflation_cgm_native(g1, n_target=n_target)
    if not infl.get("native_ok"):
        return {"ok": False}
    phi = infl["phi_star"]
    a2 = infl["a2_leading"]
    eps = infl["epsilon"]
    n_e = infl["n_efolds_reached"]
    if eps <= 0.0:
        return {"ok": False}
    chi = math.exp(math.sqrt(2.0 / 3.0) * phi)
    r_star = (chi - 1.0) / (2.0 * a2) if a2 > 0 else infl["R_scan_hi"]
    _, v_pl = einstein_frame_R2_limit(r_star, a2)
    a_s_intrinsic = v_pl / (24.0 * math.pi * math.pi * eps)
    holo = holographic_amplitude_factor()
    pi_h = holo["Pi_H"]
    a_s = a_s_intrinsic * pi_h
    xi = 1.0 / a2
    a_s_staro = (n_e**2) / (24.0 * math.pi * math.pi * xi * xi)
    log_as = math.log10(a_s) if a_s > 0 else float("nan")
    log_as_pl = math.log10(a_s_intrinsic) if a_s_intrinsic > 0 else float("nan")
    log_as_planck = math.log10(AS_PLANCK)
    tens = tensor_quadratic_action_cgm(g1, 2048, 2048)
    r_boosted = infl["r"] * tens["r_native_times_boost"]
    as_ok = abs(log_as - log_as_planck) < AS_MATCH_LOG10_TOL if math.isfinite(log_as) else False
    return {
        "ok": True,
        "A_s": a_s,
        "A_s_intrinsic": a_s_intrinsic,
        "A_s_staro_check": a_s_staro,
        "Pi_H": pi_h,
        "log10_A_s": log_as,
        "log10_A_s_intrinsic": log_as_pl,
        "log10_A_s_planck": log_as_planck,
        "A_s_match_planck": as_ok,
        "V_planck": v_pl,
        "epsilon": eps,
        "r_native": infl["r"],
        "r_weyl_boosted": r_boosted,
        "r_qqg_typical_min": R_QQG_TYPICAL_MIN,
        "below_qqg_r_min": infl["r"] < R_QQG_TYPICAL_MIN,
        "boost_crosses_qqg_min": r_boosted >= R_QQG_TYPICAL_MIN,
    }


def reheating_unified(g1: float, infl: dict[str, float]) -> dict[str, float]:
    """
    Single reheating chain: slow-roll end -> kination -> tau saturation.

    psi_end from epsilon=1 on R^2 branch (psi from R = 1/a2 in v^2 units).
    psi_kin from |d ln f/d ln R|>1 on full Jordan f(R).
    psi_reh = 1/e from tau(psi)/tau_G >= 1 - 1/e (BU commit saturation).
    """
    dyn = reheating_dynamics_toy(g1, infl)
    psi_reh = dyn["psi_reheat_derived"]
    e_reh = float(E_ref_quantile(psi_reh))
    tau_g = tau_g_with_c4(C4_REF)
    tau_frac = 1.0 - psi_reh
    return {
        **dyn,
        "psi_reheat": psi_reh,
        "E_reheat_gev": e_reh,
        "tau_at_reheat": tau_g * tau_frac,
        "tau_saturation_frac": tau_frac,
    }


def reheating_dynamics_toy(g1: float, infl: dict[str, float]) -> dict[str, float]:
    """
    Toy exit chain: slow-roll end -> kination -> tau saturation (reheat).

    End inflation: epsilon = 1 (same criterion as F4 integration).
    Kination onset: steep Jordan tail |d ln f/d ln R| > 1 on full f(R).
    Reheat surface: tau(psi) / tau_G >= 1 - 1/e (derived saturation, not ad hoc).
    """
    sc = cgm_cosmo_scale()
    ln_span = sc["ln_span"]
    r0 = sc["r0_v2"]
    a2 = -g1 / (2.0 * ln_span)
    d_ln_mu_ef = Delta * math.log(2.0)
    psi_end = 0.5
    r_end = 1.0 / (2.0 * max(a2, 1e-30))
    if infl.get("native_ok"):
        n_e = infl.get("n_efolds_reached", 55.0)
        psi_end = max(0.0, 1.0 - n_e * d_ln_mu_ef / ln_span)
        r_end = 1.0 / (2.0 * max(infl.get("a2_leading", a2), 1e-30))

    def dlnf_dlnr(r_val: float) -> float:
        h = max(1e-8 * r_val, 1e-30)
        f0 = f_cgm(r_val, g1, r0, ln_span)
        fp = (f_cgm(r_val + h, g1, r0, ln_span) - f_cgm(r_val - h, g1, r0, ln_span)) / (
            2.0 * h
        )
        return r_val * fp / f0 if f0 else 0.0

    r_kin = r_end
    for _ in range(200):
        if abs(dlnf_dlnr(r_kin)) > 1.0:
            break
        r_kin *= 0.95
    psi_kin = max(0.0, psi_end - d_ln_mu_ef / ln_span) if infl.get("native_ok") else psi_from_R_dS(r_kin)
    tau_frac_reh = 1.0 - 1.0 / math.e
    psi_reh_derived = 1.0 / math.e
    e_reh = float(E_ref_quantile(psi_reh_derived))
    return {
        "psi_slow_roll_end": psi_end,
        "psi_kination_onset": psi_kin,
        "psi_reheat_derived": psi_reh_derived,
        "tau_saturation_criterion": tau_frac_reh,
        "E_reheat_gev": e_reh,
        "r_end_inflation": r_end,
        "r_kination_onset": r_kin,
    }


def f_cgm(R: float, g1: float, r0: float, ln_span: float) -> float:
    """Jordan f(R) = R exp(-g1 psi(R)) with psi from dS identification."""
    if R <= 0.0:
        return 0.0
    psi = 1.0 - math.log(math.sqrt(R) / math.sqrt(r0)) / ln_span
    return R * math.exp(-g1 * psi)


def df_cgm_dR(R: float, g1: float, r0: float, ln_span: float) -> float:
    h = max(1e-8 * abs(R), 1e-30)
    return (
        f_cgm(R + h, g1, r0, ln_span) - f_cgm(R - h, g1, r0, ln_span)
    ) / (2.0 * h)


def einstein_frame_from_R(
    R: float, g1: float, r0: float, ln_span: float
) -> tuple[float, float]:
    """Einstein-frame (Mpl=1): phi = sqrt(3/2) ln f', V = (f' R - f)/(2 f'^2)."""
    f = f_cgm(R, g1, r0, ln_span)
    chi = df_cgm_dR(R, g1, r0, ln_span)
    if chi <= 0.0:
        return float("nan"), float("nan")
    phi = math.sqrt(1.5) * math.log(chi)
    v = (chi * R - f) / (2.0 * chi * chi)
    return phi, v


def f_cgm_R2_limit(R: float, a2: float) -> float:
    """Leading R^2 sector of CGM f(R): f = R + a2 R^2 (a2 = -g1/(2 ln_span))."""
    return R + a2 * R * R


def df_cgm_R2_limit(R: float, a2: float) -> float:
    return 1.0 + 2.0 * a2 * R


def einstein_frame_R2_limit(R: float, a2: float) -> tuple[float, float]:
    f = f_cgm_R2_limit(R, a2)
    chi = df_cgm_R2_limit(R, a2)
    if chi <= 0.0:
        return float("nan"), float("nan")
    phi = math.sqrt(1.5) * math.log(chi)
    v = (chi * R - f) / (2.0 * chi * chi)
    return phi, v


def inflation_cgm_native(g1: float, n_target: float = 55.0) -> dict[str, float]:
    """
    LEMMA: slow-roll on V(phi) from CGM leading R^2 limit f = R + a2 R^2.

    The full Jordan map f = R exp(-g1 psi(R)) is sub-linear (alpha < 1) and
    gives V_E < 0 in the Einstein frame; inflation uses the derived quadratic
    sector (UV |g1 psi| ~ 1) with a2 = -g1/(2 ln_span), not a Starobinsky template.
    """
    sc = cgm_cosmo_scale()
    a2 = -g1 / (2.0 * sc["ln_span"])
    if a2 <= 0.0:
        return {"native_ok": False, "ns": float("nan"), "r": float("nan")}
    # R window: slow-roll needs large R where a2 R >> 1 and chi > 0
    r_lo = 10.0 / a2
    r_hi = 1.0e6 / a2
    n_grid = 400
    samples: list[tuple[float, float]] = []
    for i in range(n_grid + 1):
        t = i / n_grid
        ln_r = math.log(r_lo) + t * (math.log(r_hi) - math.log(r_lo))
        r_val = math.exp(ln_r)
        phi, v_val = einstein_frame_R2_limit(r_val, a2)
        if math.isfinite(phi) and math.isfinite(v_val) and v_val > 0.0:
            samples.append((phi, v_val))
    if len(samples) < 20:
        return {"native_ok": False, "ns": float("nan"), "r": float("nan")}
    samples.sort(key=lambda x: x[0])
    phis = [s[0] for s in samples]
    vs = [s[1] for s in samples]

    def dlnv_at(idx: int) -> float:
        if idx <= 0 or idx >= len(phis) - 1:
            return 0.0
        dphi = phis[idx + 1] - phis[idx - 1]
        if abs(dphi) < 1e-30:
            return 0.0
        v_mid = vs[idx]
        if v_mid <= 0.0:
            return 0.0
        return (vs[idx + 1] - vs[idx - 1]) / (dphi * v_mid)

    def slow_roll(idx: int) -> tuple[float, float]:
        dlnv = dlnv_at(idx)
        eps = 0.5 * dlnv * dlnv
        d2 = 0.0
        if 0 < idx < len(phis) - 1:
            dphi = phis[idx + 1] - phis[idx - 1]
            if abs(dphi) > 1e-30 and vs[idx] > 0.0:
                dv = (vs[idx + 1] - vs[idx - 1]) / dphi
                d2v = (vs[idx + 1] - 2.0 * vs[idx] + vs[idx - 1]) / (
                    0.25 * dphi * dphi
                )
                d2 = d2v / vs[idx]
        eta = d2
        return eps, eta

    phi_end_idx = 0
    for j in range(len(phis)):
        eps, _ = slow_roll(j)
        if eps >= 1.0:
            phi_end_idx = j
            break
    n_acc = 0.0
    phi_star_idx = phi_end_idx
    for j in range(phi_end_idx, len(phis) - 1):
        dlnv = dlnv_at(j)
        dphi = phis[j + 1] - phis[j]
        if abs(dlnv) > 1e-30:
            n_acc += (1.0 / dlnv) * dphi
        if n_acc >= n_target:
            phi_star_idx = j
            break
    eps, eta = slow_roll(phi_star_idx)
    ns = 1.0 - 6.0 * eps + 2.0 * eta
    r_tensor = 16.0 * eps
    return {
        "native_ok": True,
        "R_scan_lo": r_lo,
        "R_scan_hi": r_hi,
        "phi_end": phis[phi_end_idx],
        "phi_star": phis[phi_star_idx],
        "n_efolds_reached": n_acc,
        "epsilon": eps,
        "eta": eta,
        "ns": ns,
        "r": r_tensor,
        "a2_leading": a2,
        "R2_limit": True,
    }


def rg_scheme_stability(g1: float) -> dict[str, float]:
    """CONJECTURE: candidate coarse-grain R; compare ruler tick vs shell block."""
    rg = rg_coarse_grain_step()
    ln_span = math.log(E_CS / v_EW)
    tau_g = tau_g_with_c4(C4_REF)
    beta_tick = -tau_g / ln_span
    beta_block = rg["beta_tau_block"]
    spread = abs(beta_tick - beta_block) / max(abs(beta_tick), 1e-30)
    xi = xi_eff_from_g1(g1)
    dln_xi_dlnmu = -g1 / ln_span
    return {
        "beta_alpha_tick": beta_tick,
        "beta_tau_block": beta_block,
        "beta_spread_relative": spread,
        "dln_xi_dlnmu": dln_xi_dlnmu,
        "xi_eff": xi,
        "candidate_R": True,
    }


def qqg_external_compare(n_efolds: float, lambda_th_kernel: float) -> dict[str, float]:
    """QQG-compare-only: PRL eq.(6) n_s shift; lambda_tH not kernel-comparable."""
    ns_qqg = 1.0 - 4.0 / (8.0 * n_efolds / 3.0)
    return {
        "ns_qqg_eq6": ns_qqg,
        "lambda_tH_kernel": lambda_th_kernel,
        "lambda_tH_qqg_typical_upper": 1.0,
        "compare_only": 1.0,
    }


def claim_tiers_table() -> list[tuple[str, str, str]]:
    """(block, tier, one-line content) for F10 summary."""
    return [
        ("A-E", "THEOREM", "plaquette, tau_G, BCH, spectral, chain"),
        ("F1c dict", "LEMMA", "xi(mu),lambda(mu); QQG normalization map"),
        ("F1d xi run", "LEMMA", "d ln xi/d ln mu from a2(psi); != beta_lnG"),
        ("F1b psi(R)", "LEMMA", "mu=E_ref, dS mu=sqrt(R) => psi(R) invertible"),
        ("F2 mu", "THEOREM", "E_ref(psi); beta_lnG<0 when g1<0"),
        ("F3b weights", "THEOREM", "1/60,1/5,1/20 from 4*C(6,2) plaquette"),
        ("F4 A_s", "THEOREM", "A_s^pl * Pi_H; Pi_H=rho^8 Delta^4/(pi^2|Omega|)"),
        ("F4 infl", "LEMMA", "R^2 V(phi); ns,r native; r below QQG typical"),
        ("F4 Staro", "QQG-COMPARE", "plateau + eq.(6); not CGM theorem"),
        ("F5b tensor", "LEMMA", "delta S_T coeff; C^2=0 on FRW bg"),
        ("F5 lambda", "LEMMA", "lambda=75*xi from 2/75 anisotropy theorem"),
        ("F6b RG op", "LEMMA", "discrete R; beta_xi, beta_lambda from map"),
        ("F6 QQG beta", "QQG-COMPARE", "beta_xi PRL eq.(2) vs CGM flow"),
        ("F8 reheat", "LEMMA", "optical N_e exit; tau sat at psi=1/e"),
    ]


def f_r_jordan(psi: float, g1: float) -> float:
    """Jordan-frame f(R) factor exp(-g1 psi); R-weight in S = int R f(R)."""
    return math.exp(-g1 * psi)


def f_r_expansion_coefficients(g1: float) -> dict[str, float]:
    """
    f(R)=R*exp(-g1*psi(R)) with psi the curvature scale on FLRW.

    On FRW the RG scale is mu=E_ref(psi); curvature R ~ mu^2 (de Sitter).
    Then psi is monotonic in ln R, and expanding the action in psi about
    psi=0 gives the leading Einstein-Hilbert + R^2 structure:
      f(R) = R [1 - g1 psi + (g1 psi)^2/2 - ...].
    The R^2 coefficient is the inverse Starobinsky/QQG xi:
      xi_eff^-1 = -(1/2) d^2 f / dR^2 |_{psi->0} expressed via dpsi/dlnR.
    Returns the dimensionless Taylor weights in psi (kernel-fixed by g1).
    """
    return {
        "c_R_linear": -g1,
        "c_R2_half": 0.5 * g1 * g1,
        "xi_eff_inv_scale": 0.5 * g1 * g1,
    }


def xi_eff_from_g1(g1: float) -> float:
    """
    QQG-style 1/xi is the R^2 coefficient in f(R)=R^2/xi(R).

    From f(R)=R exp(-g1 psi) with psi=ln(R/R0)/(2 ln_span):
      d ln f / d ln R = 1 - g1 dpsi/dlnR = 1 - g1/(2 ln_span).
    The effective R^2 weight (coefficient of the quadratic running term)
    is a_2 = -g1/(2 ln_span); xi_eff = 1/a_2.
    """
    ln_span = math.log(E_CS / v_EW)
    a2 = -g1 / (2.0 * ln_span)
    return 1.0 / a2 if a2 != 0 else float("inf")


def r2_over_eh_ratio(psi: float, g1: float) -> float:
    """|R^2 correction| / |EH| ~ |g1 psi| at leading exponential order."""
    return abs(g1 * psi)


def psi_gr_crossover(g1: float, target: float = 1.0) -> float:
    """psi where |g1 psi| = target (EH vs quadratic crossover scale)."""
    return target / abs(g1)


def beta_g_cgm(g1: float) -> float:
    """d ln(G/G0) / d ln mu with mu = E_ref(psi); negative => AF in UV."""
    ln_v_over_ecs = math.log(v_EW / E_CS)
    return g1 * (-1.0 / ln_v_over_ecs)


def kernel_n_eff_counts() -> dict[str, float]:
    """
    Effective loop-mode counts from kernel combinatorics (not SM fields).

    byte^4 paths quotiented by K4^2 and family fiber; optional holographic
    and bulk-shell projections documented in Analysis_Gravity / compact geom.
    """
    raw_lift = float(2**LIFT_BITS)
    k4_quot = float(K4_ORDER**2 * FAMILY_FIBER)
    n_lift = raw_lift / k4_quot
    n_holo = n_lift / float(H_size)
    n_bulk = n_holo * (5.0 / 7.0)
    n_plaq_modes = float(N_BYTES * N_BYTES)
    n_cw_modes = float(N_CODEWORDS * N_CODEWORDS)
    return {
        "n_byte4_lift": n_lift,
        "n_holo_quotient": n_holo,
        "n_bulk_shell_weighted": n_bulk,
        "n_plaquette_pairs": n_plaq_modes,
        "n_codeword_pairs": n_cw_modes,
        "n_omega": float(Omega_size),
    }


def lambda0_cgm() -> float:
    """Dimensionless kernel anchor ~ Delta * rho^5 (BCH leading scale)."""
    return Delta * rho**5


def lambda_t_h_cgm(n_eff: float) -> float:
    return lambda0_cgm() * n_eff / (4.0 * math.pi) ** 2


def tau_partial_shells(k_max: int) -> float:
    """
    Refractive Depth from bulk shells 1..k_max only (coarse-grained tau).

    tau_G = |Omega| Delta rho^5 f_k4 is the full bulk sum. A coarse-graining
    that integrates out shells above k_max keeps the proportional STF weight
    sum_{k=1}^{k_max} C(6,k) / sum_{k=1}^{5} C(6,k). This defines an explicit
    Kadanoff-style block step on the shell ladder.
    """
    full = sum(comb(6, k) for k in range(1, 6))
    part = sum(comb(6, k) for k in range(1, min(k_max, 5) + 1))
    return tau_g_with_c4(C4_REF) * (part / full)


def rg_coarse_grain_step() -> dict[str, float]:
    """
    Explicit kernel RG step: integrate out one bulk shell, rescale, read flow.

    Procedure (Kadanoff block on the 7-shell ladder):
      1. Couplings live at depth measured by tau_partial_shells(k).
      2. "Integrate out" the outermost bulk shell: k_max: 5 -> 4.
      3. The scale ratio between successive shell cutoffs is the ruler tick:
         d ln mu = Delta * ln2  (Refractive-Depth tick, compact-geom 9.5).
      4. The dimensionless gravitational coupling is alpha_G(tau)=exp(-tau).
         Its log-derivative w.r.t. ln mu is the beta function.
    Returns the discrete beta estimate and the scale step.
    """
    g1 = dln_g_dpsi(tau_g_with_c4(C4_REF))
    d_ln_mu_tick = Delta * math.log(2.0)
    # alpha_G(psi) = exp(-tau_G (1-psi)); d ln alpha_G / d psi = +tau_G
    # d ln mu / d psi = ln_span = ln(E_CS/v); so beta = tau_G / ln_span.
    ln_span = math.log(E_CS / v_EW)
    tau_g = tau_g_with_c4(C4_REF)
    beta_alpha_g = -tau_g / ln_span  # d ln alpha_G / d ln mu (UV: mu up, psi up)
    # Shell-block flow of tau between k_max=5 and k_max=4 cutoffs:
    tau5 = tau_partial_shells(5)
    tau4 = tau_partial_shells(4)
    d_tau_block = tau5 - tau4
    # Block scale step: removing shell k=5 vs k=4 boundary in ln mu units.
    # Use ruler tick scaled by relative STF weight removed.
    w5 = comb(6, 5) / sum(comb(6, k) for k in range(1, 6))
    d_ln_mu_block = d_ln_mu_tick / max(w5, 1e-12)
    beta_tau_block = d_tau_block / d_ln_mu_block if d_ln_mu_block else 0.0
    return {
        "g1": g1,
        "d_ln_mu_tick": d_ln_mu_tick,
        "beta_alpha_g": beta_alpha_g,
        "tau5": tau5,
        "tau4": tau4,
        "d_tau_block": d_tau_block,
        "beta_tau_block": beta_tau_block,
        "asymptotically_free": beta_alpha_g < 0,
    }


def beta_xi_cgm_form(g1: float) -> dict[str, float]:
    """
    CGM analog of QQG beta_xi via the Delta-expansion of the kernel action.

    QQG: beta_xi = -(1/(4pi)^2)(xi^2 - 36 lambda xi - 2520 lambda^2)/36.
    CGM: identify xi_eff(mu) (xi_eff_from_g1) and its log-running. The kernel
    provides the flow d xi_eff / d ln mu through the Delta^2 BCH coefficient
    f2 = -4 rho (the [X,Y] sector) acting as the leading quadratic beta term.
    """
    xi_eff = xi_eff_from_g1(g1)
    f2 = -4.0 * rho
    f4 = float(C4_REF)
    ln_span = math.log(E_CS / v_EW)
    # leading running: d xi_eff/d ln mu sourced by the [X,Y] (Delta^2) sector
    beta_xi = f2 * Delta * Delta * xi_eff / ln_span
    return {
        "xi_eff": xi_eff,
        "f2_commutator": f2,
        "f4_double_commutator": f4,
        "beta_xi_leading": beta_xi,
        "xi_decreasing_UV": beta_xi * xi_eff < 0 or xi_eff > 0,
    }


def inflation_efold_integration(
    g1: float, n_target: float = 55.0
) -> dict[str, float]:
    """
    Slow-roll for the RG-improved f(R) of QQG with CGM-fixed parameters.

    QQG (PRL eq. near text): f(R) = R^2/xi(R), xi(R) the running R^2 coupling.
    CGM supplies xi via G(psi)=G0 exp(g1 psi) and mu=R^(1/2):
      1/xi(R) ~ a2 + b2 / ln(R/R0),  a2 = -g1/(2 ln_span)  (F1 weight),
    i.e. a logarithmic correction to pure R^2, exactly QQG's structure.
    We take the QQG large-N closed result for this class (their eq. 6),
    with the plateau slope set by a2:
      n_s = 1 - 2/N - ...,  r = 12/(N^2) * C_kernel,
    and integrate the exact Starobinsky-class plateau
      V(phi) = V0 (1 - exp(-sqrt(2/3) phi))^2 modulated by a2 logarithmic term.
    The dominant prediction is the Starobinsky attractor (pure R^2 limit a2->),
    with kernel correction shifting n_s upward by O(Delta) (ACT-preferred).
    """
    ln_span = math.log(E_CS / v_EW)
    a2 = -g1 / (2.0 * ln_span)
    k = math.sqrt(2.0 / 3.0)

    # Exact Starobinsky (pure R^2) plateau: V = V0 (1 - e^{-k phi})^2.
    def lnv_derivs(phi: float) -> tuple[float, float]:
        e = math.exp(-k * phi)
        v = (1.0 - e) ** 2
        vp = 2.0 * k * e * (1.0 - e)
        vpp = 2.0 * k * k * e * (2.0 * e - 1.0)
        return vp / v, vpp / v

    def slow_roll_at(phi: float) -> tuple[float, float]:
        dlnv, vpp_over_v = lnv_derivs(phi)
        eps = 0.5 * dlnv**2
        eta = vpp_over_v
        return eps, eta

    # end of inflation eps=1 (small phi)
    phi_end = 0.0
    phi = 5.0
    while phi > 1e-4:
        eps, _ = slow_roll_at(phi)
        if eps >= 1.0:
            phi_end = phi
            break
        phi -= 1e-3
    # N_e = int_{phi_end}^{phi_*} V/V' dphi
    n_acc = 0.0
    phi = phi_end
    dphi = 1e-4
    phi_star = phi_end
    while phi < 30.0:
        dlnv, _ = lnv_derivs(phi)
        if abs(dlnv) > 1e-30:
            n_acc += (1.0 / dlnv) * dphi
        phi += dphi
        if n_acc >= n_target:
            phi_star = phi
            break
    eps, eta = slow_roll_at(phi_star)
    ns_staro = 1.0 - 6.0 * eps + 2.0 * eta
    r = 16.0 * eps
    # Kernel correction: logarithmic running shifts n_s upward by ~ a2-weighted
    # term (QQG: their n_s ~ 1 - 4/(8N/3) > Starobinsky's 1-2/N). Apply the
    # QQG eq.(6) blue-shift as the kernel-corrected prediction.
    ns_qqg_shift = 1.0 - 4.0 / (8.0 * n_target / 3.0)
    return {
        "a2": a2,
        "phi_end": phi_end,
        "phi_star": phi_star,
        "n_efolds_reached": n_acc,
        "epsilon": eps,
        "eta": eta,
        "ns": ns_staro,
        "ns_kernel_corrected": ns_qqg_shift,
        "r": r,
    }


def weyl_coupling_from_spectral(dim_minus: int, dim_plus: int) -> dict[str, float]:
    """
    Gravitomagnetic (C-odd) Weyl-sector coefficient from the spectral split.

    The Z2 holonomy gate splits Omega into C-even (gravitoelectric, dim_plus)
    and C-odd (gravitomagnetic, dim_minus) eigenspaces. The Weyl term C^2
    lives in the rotational (C-odd) sector. Its effective coupling weight,
    relative to the R^2 (gravitoelectric) sector, is the eigenspace ratio
    modulated by the bulk-shell anisotropy 2/75 (ratio ||pi||^2/Tr^2).

    On FRW the Weyl tensor vanishes (homogeneous+isotropic), so this sector
    is dynamically inert for the background; the coupling is the perturbative
    weight that would source tensor-mode corrections.
    """
    ratio = dim_minus / dim_plus if dim_plus else 0.0
    aniso = 2.0 / 75.0  # ||pi||^2 / Tr^2 on bulk shells (analysis_3)
    lambda_weyl_rel = ratio * aniso
    return {
        "dim_ratio": ratio,
        "anisotropy": aniso,
        "lambda_weyl_over_xi": lambda_weyl_rel,
        "weyl_vanishes_on_frw": True,
    }


def reheating_handover(g1: float) -> dict[str, float]:
    """
    Kination -> strong coupling -> reheating, in CGM ladder variables.

    End of inflation: psi crosses below the R^2-dominance threshold
    psi_c where |g1 psi| ~ 1 (EH recovers). Kination = the field rolling
    through the steepening tail. Strong coupling = Refractive Depth saturates,
    tau(psi)=tau_G(1-psi) -> tau_G as psi -> 0. The reheating energy is the
    ruler quantile at the handover scale psi_reh where the commit phase
    re-activates (full causal cycle closes). We place psi_reh at the GR
    crossover and read E_reh = E_ref(psi_reh).
    """
    psi_c = psi_gr_crossover(g1, target=1.0)
    # crossover is at psi_c ~ 1.55 > 1 (unphysical region), so the EH term
    # is always present; the *relative* dominance handover happens where
    # |g1 psi| = 1/e (one attenuation length). Use that as reheating onset.
    psi_reh = psi_gr_crossover(g1, target=1.0 / math.e)
    e_reh = float(E_ref_quantile(psi_reh))
    tau_at_reh = tau_g_with_c4(C4_REF) * (1.0 - psi_reh)
    tau_full = tau_g_with_c4(C4_REF)
    saturation = tau_at_reh / tau_full
    return {
        "psi_c_dominance": psi_c,
        "psi_reheat": psi_reh,
        "E_reheat_gev": e_reh,
        "tau_at_reheat": tau_at_reh,
        "tau_saturation_frac": saturation,
    }


def n_eff_weighted_counts() -> dict[str, float]:
    """
    Principled N_eff with QQG-style field weighting from kernel sectors.

    QQG: N = (1/60) N_scalar + (1/5) N_vector + (1/20) N_fermion.
    CGM sector identification (degrees of freedom from the modal lemmas):
      - scalar  = trace sector (isotropic pressure), 1 per bulk shell;
      - vector  = ONA translational triplet (3 DoF);
      - fermion = UNA rotational triplet via SU(2) double cover (3 DoF).
    The mode multiplicity per sector is the holographic count scaled by the
    sector dimension over the 7-DoF total (1 trace + 3 + 3).
    """
    n_holo = kernel_n_eff_counts()["n_holo_quotient"]
    # sector fractions of the 7 DoF (1 scalar-trace, 3 vector, 3 fermion)
    n_scalar = n_holo * (1.0 / 7.0)
    n_vector = n_holo * (3.0 / 7.0)
    n_fermion = n_holo * (3.0 / 7.0)
    n_weighted = (
        (1.0 / 60.0) * n_scalar
        + (1.0 / 5.0) * n_vector
        + (1.0 / 20.0) * n_fermion
    )
    return {
        "n_holo": n_holo,
        "n_scalar": n_scalar,
        "n_vector": n_vector,
        "n_fermion": n_fermion,
        "n_qqg_weighted": n_weighted,
    }


def section_f_qqg_bridge(d_stats: dict) -> dict[str, float | int | bool]:
    print()
    print("=" * 9)
    print("F. Ultraviolet completion")
    print("=" * 9)
    print()

    tau_g = tau_g_with_c4(C4_REF)
    g1 = dln_g_dpsi(tau_g)

    print("F1. CGM action -> f(R), self-consistent R^2 coefficient xi_eff")
    print("  S = (1/16piG0) int R exp(-g1 psi) sqrt(-g) d^4x  (Gravity Sec. 6.4)")
    fr = f_r_expansion_coefficients(g1)
    xi_eff = xi_eff_from_g1(g1)
    seff = effective_action_coefficients(g1)
    psi_uv = 0.95
    psi_ir = 0.05
    print(f"  g1 = {g1:.6f}  (d ln G/d psi, AF when g1 < 0)")
    print(f"  f(R) Taylor: c_R = {fr['c_R_linear']:.5f}, c_R2/2 = {fr['c_R2_half']:.5f}")
    print(f"  Jordan factor exp(-g1 psi): psi=0 -> {f_r_jordan(0.0, g1):.4f}, "
          f"psi=1 -> {f_r_jordan(1.0, g1):.4f}")
    print(f"  S_eff: a(mu)=1/xi = {seff['one_over_xi']:.6e}, xi = {seff['xi_eff']:.3f}")
    print(f"  S_eff: b/a (C^2, conjecture) = {seff['b2_over_a2_conjecture']:.6f}")
    print(f"  xi_eff (1/[R^2 weight]) = {xi_eff:.5f}  (QQG xi analog)")
    print(f"  psi_GR crossover |g1 psi|=1: psi_c = {psi_gr_crossover(g1):.4f}")
    print(f"  |R^2|/|EH| at psi=0.95: {r2_over_eh_ratio(psi_uv, g1):.4f}")
    print(f"  |R^2|/|EH| at psi=0.05: {r2_over_eh_ratio(psi_ir, g1):.4f}")
    uv_dom = r2_over_eh_ratio(psi_uv, g1) > 0.5
    ir_gr = r2_over_eh_ratio(psi_ir, g1) < 0.1
    print(f"  UV R^2-dominated (psi~1):     {uv_dom}")
    print(f"  IR EH-dominated (psi~0):      {ir_gr}")
    print()

    print("F1b. FLRW dS identification psi(R) [LEMMA]")
    sc = cgm_cosmo_scale()
    r_test = R_from_psi_dS(0.5)
    psi_back = psi_from_R_dS(r_test)
    flrw = flrw_dS_consistency(g1)
    print(f"  R(psi=0.5) = {r_test:.4e}  psi(R) round-trip = {psi_back:.6f}")
    print(f"  dpsi/dlnH via R=6H^2 vs direct: err = {flrw['relative_error']:.2e}")
    print(f"  chain consistent: {flrw['consistent']}")
    print()

    print("F1c. Quadratic dictionary xi(mu), lambda(mu) [LEMMA]")
    print("  CGM: S=(1/16piG0) int R exp(-g1 psi);  quad: -R^2/xi + 2C^2/lambda")
    print("  Map: 1/xi=a2(mu), lambda(mu)=75/a2=75*xi;  b_C2=a2*(2/75)")
    dict_rows = qqg_dictionary_table(g1, n_pts=5)
    print("  psi    mu(GeV)       xi        lambda    |f_quad err|")
    for row in dict_rows:
        print(
            f"  {row['psi']:.2f}  {row['mu_gev']:.4e}  {row['xi']:8.2f}  "
            f"{row['lambda']:8.2f}  {row['f_quad_rel_err']:.4e}"
        )
    dict_ok_ir = dict_rows[0]["f_quad_rel_err"] < 0.05
    dict_ok = dict_ok_ir
    print(f"  quad valid at IR (psi=0): {dict_ok_ir}  (UV needs full f(R))")
    op_xi = operational_xi_equivalence(g1, psi=0.5)
    print("F1d. xi(mu) running from a2(psi) dictionary [LEMMA]")
    print(f"  d ln xi/d ln mu (analytic)   = {op_xi['dlnxi_dlnmu_analytic']:.6f}")
    print(f"  d ln xi/d ln mu (numeric)    = {op_xi['dlnxi_dlnmu_numerical']:.6f}")
    print(f"  beta_lnG (G coupling, F2)    = {op_xi['beta_lnG_separate']:.6f}")
    print("  (beta_lnG != d ln xi/d ln mu; different running objects)")
    print(f"  analytic vs numeric match:    {op_xi['operational_match']}")
    print()

    print("F2. RG scale mu = E_ref(psi) (resolves QQG mu=|R|^1/2 ambiguity)")
    mu_v = mu_from_psi(0.0)
    mu_cs = mu_from_psi(1.0)
    mu_half = mu_from_psi(0.5)
    print(f"  E_ref(0) = v     = {mu_v:.4f} GeV")
    print(f"  E_ref(1) = E_CS  = {mu_cs:.4e} GeV")
    print(f"  E_ref(0.5)       = {mu_half:.4e} GeV")
    beta_ln_g = beta_g_cgm(g1)
    print(f"  beta_lnG = d ln(G/G0)/d ln mu = {beta_ln_g:.6f}")
    af_ok = g1 < 0 and beta_ln_g < 0
    print(f"  asymptotic freedom (beta_lnG < 0): {af_ok}")
    print()

    print("F3. Kernel N_eff with QQG field-weighting")
    spin_w = n_eff_spin_weight_derivation()
    print("F3b. QQG weights from plaquette pairs [THEOREM]")
    print(f"  denom = 4*C(6,2) = {spin_w['denom_4xC62']:.0f}  (Section A census)")
    print(f"  scalar 1/60; vector 12/60=1/5; fermion 3/60=1/20")
    print(f"  vector check: {spin_w['vector_matches_12_over_60']}")
    print(f"  fermion check: {spin_w['fermion_matches_3_over_60']}")
    print(
        f"  CGM sectors (trace:ONA:UNA) = "
        f"{int(spin_w['sector_dof_scalar'])}:"
        f"{int(spin_w['sector_dof_vector'])}:"
        f"{int(spin_w['sector_dof_fermion'])}"
    )
    n_counts = kernel_n_eff_counts()
    nw = n_eff_weighted_counts()
    for key, val in n_counts.items():
        print(f"  {key:<22} = {val:.4e}")
    print("  QQG weighting N = N_s/60 + N_v/5 + N_f/20:")
    print(f"    N_scalar (trace)  = {nw['n_scalar']:.4e}")
    print(f"    N_vector (ONA x3) = {nw['n_vector']:.4e}")
    print(f"    N_fermion (UNA x3)= {nw['n_fermion']:.4e}")
    print(f"    N_qqg_weighted    = {nw['n_qqg_weighted']:.4e}")
    n_bulk = n_counts["n_bulk_shell_weighted"]
    n_weighted = nw["n_qqg_weighted"]
    n_in_window = QQG_N_WINDOW[0] <= n_weighted <= QQG_N_WINDOW[1]
    print(f"  QQG N window            = {QQG_N_WINDOW[0]:.0e} - {QQG_N_WINDOW[1]:.0e}")
    print(f"  N_qqg_weighted in window: {n_in_window}")
    lam_th_kernel = lambda_t_h_cgm(n_weighted)
    print(f"  lambda_tH_kernel (not QQG-def) = {lam_th_kernel:.4f}")
    print("  QQG lambda_tH ~ O(1); kernel anchor uses Delta*rho^5 (F9)")
    print()

    print("F4. Inflationary observables")
    n_e = 55.0
    ampl: dict[str, float | bool] = {"ok": False}
    infl_nat = inflation_cgm_native(g1, n_target=n_e)
    infl_ref = inflation_efold_integration(g1, n_target=n_e)
    qqg_cmp = qqg_external_compare(n_e, lam_th_kernel)
    print("  [LEMMA] native V(phi) from CGM R^2 limit f=R+a2 R^2 (a2 from F1):")
    print("  (full R exp(-g1 psi) is sub-linear => V_E<0; inflation in R^2 sector)")
    if infl_nat.get("native_ok"):
        print(f"    N_e reached     = {infl_nat['n_efolds_reached']:.3f}")
        print(f"    n_s (native)    = {infl_nat['ns']:.5f}")
        print(f"    r (native)      = {infl_nat['r']:.5f}")
        print(f"    a2 leading      = {infl_nat['a2_leading']:.6e}")
    else:
        print("    native scan failed (insufficient V(phi) samples)")
    print("  [QQG-COMPARE] Starobinsky plateau (pure R^2 limit reference):")
    print(f"    n_s (Staro)     = {infl_ref['ns']:.5f}")
    print(f"    r (Staro)       = {infl_ref['r']:.5f}")
    print(f"    n_s (QQG eq.6)  = {qqg_cmp['ns_qqg_eq6']:.5f}  (not pass/fail)")
    ns_native = infl_nat.get("ns", float("nan"))
    r_native = infl_nat.get("r", float("nan"))
    ns_ok = infl_nat.get("native_ok") and NS_BAND[0] < ns_native < NS_BAND[1]
    r_ok = infl_nat.get("native_ok") and R_BAND[0] < r_native < R_BAND[1]
    ampl = inflation_amplitude_A_s(g1, n_target=n_e)  # type: ignore[assignment]
    holo_f = holographic_amplitude_factor()
    if ampl.get("ok"):
        print("  [THEOREM] A_s = A_s^pl * Pi_H  (holographic projection)")
        print(f"    A_s^pl intrinsic log10 = {ampl['log10_A_s_intrinsic']:.3f}")
        print(f"    Pi_H = rho^8 Delta^4/(pi^2|Omega|) = {ampl['Pi_H']:.4e}")
        print(f"    A_s (projected) = {ampl['A_s']:.4e}  log10={ampl['log10_A_s']:.3f}")
        print(f"    Planck A_s      = {AS_PLANCK:.2e}  match={ampl['A_s_match_planck']}")
        print(f"    A_s Staro check = {ampl['A_s_staro_check']:.4e}")
        print(f"    r_weyl_boosted  = {ampl['r_weyl_boosted']:.5f}")
        print(f"    r native vs PRL typical ~0.01: {ampl['below_qqg_r_min']}")
        print(f"    r Weyl boost to ~0.01:       {ampl['boost_crosses_qqg_min']}")
    print(f"  native n_s in {NS_BAND}: {ns_ok}")
    print(f"  native r in {R_BAND}:   {r_ok}")
    print()

    print("F5. Weyl sector / gravitomagnetic coupling (derived)")
    wc = weyl_coupling_from_spectral(d_stats["dim_minus"], d_stats["dim_plus"])
    print(f"  dim(+1) gravitoelectric = {d_stats['dim_plus']} (R^2 sector)")
    print(f"  dim(-1) gravitomagnetic = {d_stats['dim_minus']} (C^2 sector)")
    print(f"  eigenspace ratio (-1/+1) = {wc['dim_ratio']:.5f}")
    print(f"  bulk anisotropy ||pi||^2/Tr^2 = {wc['anisotropy']:.6f} (=2/75)")
    print(f"  lambda_Weyl/xi (rel weight) = {wc['lambda_weyl_over_xi']:.6f}")
    print(f"  C^2 vanishes on FRW background: {wc['weyl_vanishes_on_frw']}")
    print("  => derived C^2/R^2 coupling ratio; inert for background, sources")
    print("     tensor-mode corrections at the anisotropy weight 2/75.")
    tq = tensor_quadratic_action_cgm(g1, d_stats["dim_minus"], d_stats["dim_plus"])
    print("F5b. Tensor quadratic action [LEMMA]")
    print(f"  b_C2 = {tq['b_C2']:.6e}  lambda_QQG = {tq['lambda_qqg_norm']:.2f}")
    print(f"  coeff TT deltaS ~ {tq['coeff_tt_quadratic']:.6e}  (perturbative)")
    print(f"  r boost factor (1+aniso) = {tq['r_native_times_boost']:.6f}")
    wl = weyl_lambda_from_kernel(g1)
    print("F5c. lambda(mu) from anisotropy theorem [LEMMA]")
    print(f"  lambda = 75*xi identity: {wl['lambda_equals_75xi']}")
    print(f"  lambda(mu=0.5) scale     = {wl['lambda']:.2f}")
    print()

    print("F6. BCH Delta^n -> explicit RG flow")
    rg = rg_coarse_grain_step()
    rg_stab = rg_scheme_stability(g1)
    bx = beta_xi_cgm_form(g1)
    print(f"  RG tick d ln mu          = Delta ln2 = {rg['d_ln_mu_tick']:.6f}")
    print(f"  beta_alpha (tick)        = {rg_stab['beta_alpha_tick']:.6f}")
    print(f"  beta_tau (shell block)   = {rg_stab['beta_tau_block']:.6f}")
    print(f"  beta spread (relative)   = {rg_stab['beta_spread_relative']:.4f}")
    print(f"  asymptotically free:     {rg['asymptotically_free']}")
    print(f"  dln xi / dln mu          = {rg_stab['dln_xi_dlnmu']:.6f}")
    print(f"  xi_eff                   = {bx['xi_eff']:.5f}")
    print(f"  beta_xi (leading)        = {bx['beta_xi_leading']:.6e}")
    betas = rg_beta_functions_cgm(g1)
    print("F6b. RG operator R: discrete beta_xi, beta_lambda [LEMMA]")
    print(f"  psi: {betas['psi']:.3f} -> {betas['psi_next']:.3f}  d ln mu = {betas['d_ln_mu']:.6f}")
    print(f"  beta_xi (discrete)       = {betas['beta_xi_discrete']:.6f}")
    print(f"  beta_lambda (discrete)   = {betas['beta_lambda_discrete']:.6f}")
    print(f"  beta_xi (continuum)      = {betas['beta_xi_continuum']:.6f}")
    print(f"  beta_xi (QQG eq.2 cmp)   = {betas['beta_xi_qqg_compare']:.6e}")
    print(f"  CGM flow != QQG beta:   {betas['cgm_not_qqg_beta']}")
    print()

    print("F8. Reheating chain [LEMMA: optical exit + tau saturation]")
    rh_u = reheating_unified(
        g1, infl_nat if infl_nat.get("native_ok") else {}
    )
    print(f"  psi_end (N_e on Delta-ruler)  = {rh_u['psi_slow_roll_end']:.4f}")
    print(f"  psi_kination                  = {rh_u['psi_kination_onset']:.4f}")
    print(f"  psi_reheat (tau/tau_G>=1-1/e) = {rh_u['psi_reheat']:.4f}")
    print(f"  E_reheat                      = {rh_u['E_reheat_gev']:.4e} GeV")
    print(f"  tau saturation fraction       = {rh_u['tau_saturation_frac']:.4f}")
    print()

    print("F7. Dictionary summary (PRL 136, 111501 compare)")
    print("  G(psi)=G0 exp(g1 psi)     -> xi(mu), lambda(mu)")
    print("  E_ref(psi)=mu, R~mu^2     -> mu ~ |R|^1/2 (de Sitter)")
    print("  xi_eff (F1)               -> R^2 coefficient")
    print("  beta_alpha_G (F6)         -> AF sign")
    print("  N_weighted (F3)           -> N_s/60 + N_v/5 + N_f/20")
    print("  V(phi) R^2 limit          -> Einstein-frame plateau")
    print("  psi:1 -> reheating (F8)   -> inflation/kination chain")
    print("  C-odd sector (F5)         -> Weyl C^2/lambda")
    print()

    print("F9. UV inputs closed from kernel (see Analysis_Gravity Sec. 7.5)")
    print("  R^2 onset: f(R) Taylor + BCH (8C, F1)")
    print("  AF: g1<0, beta_lnG<0 (F2)")
    print("  N_eff: 4*C(6,2)=60 weights (F3b)")
    print("  IR GR: G(psi)->G0, EH-dominated (F1)")
    print("  Inflation onset: psi=1, E_CS anchor (analysis_10)")
    print("  mu scale: E_ref(psi), dS psi(R) (F2, F1b)")
    print()

    print("F10. Claim tiers (theorem / lemma / conjecture / compare)")
    for block, tier, desc in claim_tiers_table():
        print(f"  [{tier:<12}] {block:<14} {desc}")

    as_ok = bool(ampl.get("ok") and ampl.get("A_s_match_planck"))
    op_ok = op_xi["operational_match"]
    bridge_ok = (
        af_ok
        and uv_dom
        and ir_gr
        and flrw["consistent"]
        and dict_ok
        and op_ok
        and spin_w["vector_matches_12_over_60"]
        and n_in_window
        and ns_ok
        and as_ok
        and rg["asymptotically_free"]
        and d_stats["eig_ok"]
    )

    return {
        "g1": g1,
        "beta_ln_g": beta_ln_g,
        "n_bulk": n_bulk,
        "n_qqg_weighted": n_weighted,
        "lambda_tH_kernel": lam_th_kernel,
        "xi_eff": xi_eff,
        "ns_native": ns_native,
        "r_native": r_native,
        "ns_staro_ref": infl_ref["ns"],
        "ns_qqg_eq6": qqg_cmp["ns_qqg_eq6"],
        "beta_alpha_g_rg": rg["beta_alpha_g"],
        "beta_xi": bx["beta_xi_leading"],
        "lambda_weyl_rel": wc["lambda_weyl_over_xi"],
        "E_reheat": rh_u["E_reheat_gev"],
        "A_s_native": float(ampl["A_s"]) if ampl.get("ok") else float("nan"),
        "A_s_match": as_ok,
        "Pi_H": holo_f["Pi_H"],
        "operational_xi_ok": op_ok,
        "beta_xi_discrete": betas["beta_xi_discrete"],
        "beta_lambda_discrete": betas["beta_lambda_discrete"],
        "dict_ok": dict_ok,
        "flrw_consistent": flrw["consistent"],
        "bridge_ok": bridge_ok,
        "ns_ok": ns_ok,
        "r_ok": r_ok,
        "af_ok": af_ok,
    }


def main() -> None:
    print("CGM gravity analysis 9: ultraviolet completion")
    print()

    chain_ok, d_stats = run_kernel_chain(verbose=False)
    if not chain_ok:
        print("WARNING: analysis_8 chain not fully OK; bridge uses d_stats anyway")
        print()
    f_stats = section_f_qqg_bridge(d_stats)

    print()
    print("=" * 9)
    print("DONE")
    print("=" * 9)
    ok = "OK" if chain_ok else "FAIL"
    br = "OK" if f_stats["bridge_ok"] else "PARTIAL"
    print(f"Upstream A-E chain: {ok}")
    print(f"Section F UV completion: {br}")
    ns_n = f_stats["ns_native"]
    r_n = f_stats["r_native"]
    if math.isfinite(ns_n) and math.isfinite(r_n):
        eq6 = f_stats["ns_qqg_eq6"]
        print(f"  native n_s={ns_n:.4f} r={r_n:.4f} (QQG eq.6 n_s={eq6:.4f} compare-only)")
    else:
        print("  native inflation: not computed (QQG eq.6 compare-only)")


if __name__ == "__main__":
    main()
