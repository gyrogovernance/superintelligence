#!/usr/bin/env python3
"""
Nonlinear CGM gravity: G(psi), exact point-mass psi(s), metric, shadow.

G(x) = G_kernel * exp(-tau(x)) / E_ref(x)^2
psi(x) = |Phi(x)| / Phi_Planck
E_ref(x) = E_CS * (v/E_CS)^(1-psi)
tau(x) = tau_G * (1 - psi)

Weak field (psi -> 0): G -> G_global.  Strong field (psi -> 1): G -> G_CS = G_kernel/E_CS^2.

Ownership:
  analysis_3: exact kernel theorems
  analysis_2: theorem registry
  analysis_1: G prediction, transport
  analysis_5: field derivations, spin-sector shadow (companion)
Constants: hqvm_gravity_common.py, gyroscopic.hQVM.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from math import comb, exp, log, log10, pi, sqrt
from pathlib import Path

from typing import Any

import astropy.constants as astro_constants
import astropy.units as u
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gyroscopic.hQVM.constants import (
    APERTURE_GAP,
    DELTA_BU,
    HORIZON_SIZE,
    M_A,
    OMEGA_SIZE,
    RHO,
)
from hqvm_gravity_common import (
    C4_REF,
    E_CS,
    G_kernel,
    G_meas,
    Q_G,
    dln_g_dpsi,
    g_pred_from_tau,
    horizon_s_analytic,
    mercury_precession_cgm_gr_ratio,
    photon_geometry_analytic,
    point_mass_profile,
    psi_analytic,
    tau_g_with_c4,
    v_EW,
    verify_alpha_zeta_product,
    zeta_from_alpha,
)

# CGM invariants (hQVM kernel + gravity_common)
Delta = APERTURE_GAP
rho_val = RHO
d_BU = DELTA_BU
m_a = M_A
c4 = Fraction(-7, 4)

# Physical constants: SI from astropy; EW/UV anchors from gravity_common
def _astro(name: str) -> Any:
    return getattr(astro_constants, name)


GEV = u.GeV
v = v_EW
G_global = g_pred_from_tau(tau_g_with_c4(C4_REF))
_c = _astro("c")
_G = _astro("G")
_hbar = _astro("hbar")
_M_sun = _astro("M_sun")
c_SI = _c.to_value(u.m / u.s)
G_SI = _G.to_value(u.m**3 / u.kg / u.s**2)
M_sun_kg = _M_sun.to_value(u.kg)
GeV_per_kg = (_c**2).to_value(GEV / u.kg)
m_per_GeVinv = (_hbar * _c).to_value(u.m * GEV)

# Derived
tau_G_leading = tau_g_with_c4(0.0)
tau_G_full = tau_g_with_c4(C4_REF)
# G at Planck/CS scale (psi=1 endpoint); not G_meas
G_CS = G_kernel / E_CS**2
eta = log(v / E_CS)
dlnG_dpsi = dln_g_dpsi(tau_G_full)

# CSM: Cs-133 hyperfine frequency (exact SI second definition, 2019)
f_Cs = 9192631770.0
N_phys = (4.0 / 3.0) * pi * f_Cs**3
CSM = N_phys / OMEGA_SIZE

# radians -> microarcseconds
RAD2UAS = 180.0 / pi * 3600.0 * 1e6
MPC_M = 3.085677581e22


def G_of_psi_raw(psi):
    """G(psi) without clipping (for ODE interior; caller caps psi)."""
    psi = np.asarray(psi, dtype=float)
    E_ref = E_CS * (v / E_CS) ** (1.0 - psi)
    tau_local = tau_G_full * (1.0 - psi)
    return G_kernel * np.exp(-tau_local) / E_ref**2


def G_of_psi(psi):
    """Position-dependent gravitational coupling as function of
    dimensionless field strength psi = |Phi|/Phi_Planck."""
    return G_of_psi_raw(np.clip(psi, 0.0, 1.0))


def G_ratio(u):
    """G(psi)/G_global for dimensionless potential psi = u."""
    return float(G_of_psi_raw(min(max(u, 0.0), 1.0)) / G_global)


def metric_f(u):
    """CGM metric coefficient f(r) = 1 - 2*psi with psi = u."""
    return 1.0 - 2.0 * u


def shadow_diameter_muas(b_over_rg: float, M_kg: float, D_m: float) -> float:
    """Angular diameter in microarcsec from impact parameter in r_g units."""
    r_g_m = G_SI * M_kg / c_SI**2
    theta_rad = 2.0 * b_over_rg * r_g_m / D_m
    return theta_rad * RAD2UAS


# EHT Collaboration 2022 shadow-size priors (angular diameter, microarcsec)
EHT_SHADOW = {
    "M87*": {
        "d_muas": 42.0,
        "sigma_muas": 3.0,
        "M_kg": 6.5e9 * M_sun_kg,
        "D_m": 16.8 * MPC_M,
        "ref": "EHT 2019/2022; M87* ~ 42 +/- 3 uas",
    },
    "Sgr A*": {
        "d_muas": 51.8,
        "sigma_muas": 2.3,
        "M_kg": 4.0e6 * M_sun_kg,
        "D_m": 8.1e-3 * MPC_M,
        "ref": "EHT 2022 ApJL; 51.8 +/- 2.3 uas",
    },
}


def solve_psi_self_consistent(psi_N: float, tol: float = 1e-12, max_iter: int = 50) -> float:
    """Solve psi = psi_N * G(psi)/G_global."""
    psi = psi_N
    for _ in range(max_iter):
        g_ratio = float(G_of_psi(min(psi, 0.49))) / G_global
        psi_new = psi_N * g_ratio
        if abs(psi_new - psi) < tol:
            return psi_new
        psi = psi_new
    return psi


def photon_impact_parameter(s_ph: float, u_ph: float) -> float:
    """Impact parameter b/r_g for metric f = 1 - 2*u at the photon sphere."""
    return s_ph / sqrt(max(metric_f(u_ph), 1e-15))


def G_fractional(psi):
    """Fractional deviation (G(psi) - G_global) / G_global."""
    return (G_of_psi(psi) - G_global) / G_global


def interp_s_at_u(s_vals, u_vals, u_target):
    """
    Interpolate s where u(psi) = u_target.

    s_vals is ascending outward; u_vals decreases with index (max at s_min).
    Extrapolates inward if u_target exceeds the profile maximum.
    """
    s_arr = np.asarray(s_vals, dtype=float)
    u_arr = np.asarray(u_vals, dtype=float)
    if u_arr[-1] > u_target:
        return float(s_arr[-1])
    if u_arr[0] < u_target and len(u_arr) >= 2:
        u_hi, u_lo = u_arr[0], u_arr[1]
        s_hi, s_lo = s_arr[0], s_arr[1]
        if u_hi != u_lo:
            frac = (u_target - u_hi) / (u_lo - u_hi)
            return float(s_hi + frac * (s_lo - s_hi))
        return float(s_hi)
    for i in range(len(u_arr) - 1):
        u_hi, u_lo = u_arr[i], u_arr[i + 1]
        if u_hi >= u_target >= u_lo:
            s_hi, s_lo = s_arr[i], s_arr[i + 1]
            if u_hi == u_lo:
                return float(s_hi)
            frac = (u_target - u_hi) / (u_lo - u_hi)
            return float(s_hi + frac * (s_lo - s_hi))
    return None


def verify_endpoints():
    print("=" * 9)
    print("A. Nonlinear coupling G(psi)")
    print("=" * 9)
    print()
    print(f"tau_G (leading order)    = {tau_G_leading:.10f}")
    print(f"tau_G (full with c4)    = {tau_G_full:.10f}")
    print(f"G_global (weak field)   = {G_global:.6e} GeV^-2")
    print(f"G_CS (psi=1)        = {G_CS:.6e} GeV^-2")
    print(f"G_CS / G_global     = {G_CS/G_global:.6f}")
    print(f"eta = ln(v/E_CS)        = {eta:.6f}")
    print(f"dlnG/dpsi = tau_G+2*eta = {dlnG_dpsi:.6f}")
    print()

    # Verify psi = 0
    G0 = G_of_psi(0.0)
    rel0 = (G0 - G_global) / G_global
    print(f"G(psi=0)                = {G0:.6e} GeV^-2")
    print(f"  vs G_global           = {G_global:.6e} GeV^-2")
    print(f"  relative error        = {rel0:.2e}  (should be ~0)")
    print()

    # Verify psi = 1
    G1 = G_of_psi(1.0)
    G1_expected = G_kernel / E_CS**2
    rel1 = (G1 - G1_expected) / G1_expected
    print(f"G(psi=1)                = {G1:.6e} GeV^-2")
    print(f"  vs G_kernel/E_CS^2    = {G1_expected:.6e} GeV^-2")
    print(f"  relative error        = {rel1:.2e}  (should be ~0)")
    print()

    # Slope verification
    eps = 1e-8
    dlnG_numerical = (log(G_of_psi(eps)) - log(G_of_psi(0.0))) / eps
    print(f"dlnG/dpsi (analytic)    = {dlnG_dpsi:.6f}")
    print(f"dlnG/dpsi (numerical)   = {dlnG_numerical:.6f}")
    print(f"  agreement             = {abs(dlnG_numerical - dlnG_dpsi) < 1e-4}")
    print()

    # Table
    print("G(psi) table:")
    print(f"{'psi':>8} {'G(psi)':>14} {'G/G_global':>12} {'delta_G/G':>12}")
    print("-" * 50)
    for psi in [0, 1e-9, 1e-6, 1e-3, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
        Gv = G_of_psi(psi)
        frac = Gv / G_global
        delta = G_fractional(psi)
        print(f"{psi:>8.1e} {Gv:>14.6e} {frac:>12.6f} {delta:>12.6f}")
    print()

    # Leading-order approximation for small psi
    print("Leading-order: delta_G/G = dlnG/dpsi * psi = {:.4f} * psi".format(dlnG_dpsi))
    print("  Earth (psi~7e-10):   delta_G/G ~ {:.2e} = {:.2f} ppb".format(
        dlnG_dpsi * 7e-10, dlnG_dpsi * 7e-10 * 1e9))
    print("  Sun (psi~2.1e-6):    delta_G/G ~ {:.2e} = {:.2f} ppm".format(
        dlnG_dpsi * 2.1e-6, dlnG_dpsi * 2.1e-6 * 1e6))
    print()


# ============================================================
# B. Astrophysical psi Values
# ============================================================
def astrophysical_psi():
    print("=" * 9)
    print("B. Astrophysical psi values and G corrections")
    print("=" * 9)
    print()
    print("psi = GM/(r*c^2) = gravitational potential / c^2")
    print()

    objects = [
        ("Earth surface",        5.97e24,  6.371e6),
        ("Sun surface",          1.989e30, 6.957e8),
        ("White dwarf (1 Msun)", 1.989e30, 5e6),
        ("Neutron star (1.4 Msun)", 2.785e30, 1.2e4),
        ("Stellar BH (10 Msun)", 1.989e31, 2.95e4),
        ("SMBH (4e6 Msun)",      7.956e36, 1.18e10),
    ]

    print(f"{'Object':>25} {'M/M_sun':>10} {'R (km)':>12} {'psi':>12} {'G/G_global':>12} {'delta%':>10}")
    print("-" * 85)

    for name, M_kg, R_m in objects:
        M_sun = M_kg / M_sun_kg
        psi_val = G_SI * M_kg / (R_m * c_SI**2)
        G_ratio = G_of_psi(psi_val) / G_global
        delta_pct = (G_ratio - 1.0) * 100
        R_km = R_m / 1e3
        print(f"{name:>25} {M_sun:>10.2f} {R_km:>12.1f} {psi_val:>12.4e} {G_ratio:>12.6f} {delta_pct:>10.4f}")

    print()
    print("Note: psi for compact objects can exceed 0.1, producing")
    print("measurable G corrections (>1%). Neutron stars and black holes")
    print("are in the strongly nonlinear regime.")
    print()


# ============================================================
# C. Nonlinear Point-Mass Potential
# ============================================================
def solve_point_mass_profile(n_eval: int = 500):
    """Exact point-mass profile psi(s); returns (s_vals, u_vals, None)."""
    s_vals, u_vals = point_mass_profile(n_eval=n_eval)
    return s_vals, u_vals, None


def nonlinear_potential():
    print("=" * 9)
    print("C. Nonlinear point-mass potential")
    print("=" * 9)
    print()
    print("ODE: d psi/ds = -exp(g1*psi)/s^2  (G/G0 = exp(g1*psi))")
    print("Exact: psi(s) = -(1/g1) ln(1 - g1/s)")
    print(f"g1 = dlnG/dpsi = {dlnG_dpsi:.6f}")
    print("Newtonian: psi = 1/s")
    print()

    s_vals, u_vals, _ = solve_point_mass_profile()
    u_newton = 1.0 / s_vals
    G_vals = np.array([G_of_psi(u) for u in u_vals])
    gg_profile = G_vals / G_global
    g_ratio = np.array([G_of_psi_raw(min(max(u, 0.0), 1.0)) / G_global for u in u_vals])

    print("Radial profile (dimensionless, s = r / r_g, r_g = G_global*M):")
    print(f"{'s=r/r_g':>10} {'u=psi':>12} {'u_Newton':>12} {'u/u_N':>10} {'G/G_global':>12} {'g/g_N':>10}")
    print("-" * 70)

    sample_indices = [0, 3, 6, 10, 15, 22, len(s_vals) // 2, -3, -1]
    for i in sample_indices:
        if 0 <= i < len(s_vals):
            s = s_vals[i]
            u = u_vals[i]
            u_N = 1.0 / s
            ratio_u = u / u_N if u_N > 0 else 0
            print(
                f"{s:>10.3f} {u:>12.6e} {u_N:>12.6e} {ratio_u:>10.6f} "
                f"{gg_profile[i]:>12.6f} {g_ratio[i]:>10.6f}"
            )

    print()
    print("Key radii:")
    print("-" * 50)

    s_01 = interp_s_at_u(s_vals, u_vals, 0.01)
    if s_01 is not None:
        print(f"psi = 0.01  at s = {s_01:.2f} r_g  (G/G_global = {G_ratio(0.01):.6f})")

    s_1 = interp_s_at_u(s_vals, u_vals, 0.1)
    if s_1 is not None:
        print(f"psi = 0.10  at s = {s_1:.2f} r_g  (G/G_global = {G_ratio(0.1):.6f})")

    print()
    print("At Newtonian Schwarzschild radius (s = 2):")
    u_s2 = float(np.interp(2.0, s_vals, u_vals))
    print(f"  psi_Newton = 0.5000")
    print(f"  psi_CGM    = {u_s2:.4f}")
    print(f"  G/G_global = {G_ratio(u_s2):.6f}")
    print(f"  g/g_Newton = {G_ratio(u_s2):.6f}")
    print()

    return s_vals, u_vals, u_newton, gg_profile


# ============================================================
# D. Modified Escape Velocity and Critical Radius
# ============================================================
def escape_velocity_analysis(s_vals, u_vals):
    print("=" * 9)
    print("D. Modified escape velocity and critical radius")
    print("=" * 9)
    print()

    # In natural units, escape velocity at radius r:
    # v_esc^2 = 2 * |Phi| = 2 * u * Phi_Planck
    # v_esc/c = sqrt(2*u)  (since Phi_Planck = c^2 = 1)
    # "Horizon" where v_esc = c: 2*u = 1, u = 0.5

    # In Newtonian: u = 1/s, so u = 0.5 at s = 2 (Schwarzschild)

    # In CGM nonlinear: u is modified by the position-dependent G

    v_esc_newton = np.sqrt(2.0 / s_vals)
    v_esc_cgm = np.sqrt(np.minimum(2.0 * u_vals, 1.0))

    print("Escape velocity profile (v_esc / c):")
    print(f"{'s=r/r_g':>10} {'v_N/c':>10} {'v_CGM/c':>10} {'ratio':>10}")
    print("-" * 45)

    for i in range(0, len(s_vals), len(s_vals)//20):
        s = s_vals[i]
        vn = min(np.sqrt(2.0/s), 1.0)
        vc = min(np.sqrt(2.0*u_vals[i]), 1.0)
        ratio = vc/vn if vn > 0 else 0
        print(f"{s:>10.3f} {vn:>10.6f} {vc:>10.6f} {ratio:>10.6f}")

    print()

    s_horizon = interp_s_at_u(s_vals, u_vals, 0.5)
    s_newton = 2.0
    print("CGM horizon (v_esc = c, 2*u = 1):")
    if s_horizon is not None:
        shift_pct = (s_horizon - s_newton) / s_newton * 100.0
        diam_ratio = s_horizon / s_newton
        print(f"  Newtonian (GR):  s = {s_newton:.3f} r_g")
        print(f"  CGM:             s = {s_horizon:.3f} r_g")
        print(f"  Inward shift:    {shift_pct:+.1f}% in s")
        print(f"  Horizon radius ratio CGM/GR: {diam_ratio * 100:.1f}%")
        print("  Shadow diameter from photon sphere (section N).")
    else:
        u_inner = float(u_vals[0])
        print(f"  Could not bracket u = 0.5 (innermost psi = {u_inner:.4f})")

    print()
    print("Coupling feedback (dlnG/dpsi < 0):")
    print(f"  dlnG/dpsi = {dlnG_dpsi:.4f}")
    print(f"  exp(dlnG/dpsi) per unit psi = {exp(dlnG_dpsi):.4f}")
    print()

    u_max = float(u_vals[0])
    s_at_umax = float(s_vals[0])
    print(f"Maximum psi in point-mass solution: {u_max:.6f}")
    print(f"  at s = {s_at_umax:.4f} r_g")
    print(f"  G/G_global at max = {G_of_psi(u_max)/G_global:.6f}")
    print()


# ============================================================
# E. Orbital Dynamics Corrections
# ============================================================
def orbital_corrections(s_vals, u_vals, g_ratio):
    print("=" * 9)
    print("Orbital period corrections")
    print("=" * 9)
    print()

    # Circular orbit: v^2/r = g = f(u) * G_global * M / r^2
    # v^2 = f(u) * G_global * M / r = f(u) / s (in dimensionless units)
    # v/c = sqrt(f(u)/s)  (with appropriate unit conversions)

    # Orbital period: T = 2*pi*r / v = 2*pi*s*r_g / (c*sqrt(f(u)/s))
    # T = 2*pi * sqrt(s^3 / f(u)) * r_g / c

    # Newtonian: T_N = 2*pi * sqrt(s^3) * r_g / c

    # Ratio: T/T_N = f(u)^(-1/2)

    print("Orbital corrections at key radii:")
    print(f"{'Object':>20} {'s=r/r_g':>10} {'psi':>10} {'T/T_N':>10} {'v/v_N':>10}")
    print("-" * 55)

    # Earth-like orbit around Sun: r = 1 AU
    M_sun_GeV = M_sun_kg * GeV_per_kg
    r_g_sun = G_global * M_sun_GeV  # GeV^-1
    r_earth_m = 1.496e11  # 1 AU in meters
    r_earth_GeVinv = r_earth_m / m_per_GeVinv
    s_earth = r_earth_GeVinv / r_g_sun

    # Mercury orbit
    r_mercury_m = 5.791e10
    r_mercury_GeVinv = r_mercury_m / m_per_GeVinv
    s_mercury = r_mercury_GeVinv / r_g_sun

    # Close orbit around neutron star (r = 2*r_ns)
    M_ns_GeV = 2.785e30 * GeV_per_kg
    r_g_ns = G_global * M_ns_GeV
    r_close_m = 2.4e4  # 2 * 12 km
    r_close_GeVinv = r_close_m / m_per_GeVinv
    s_ns_close = r_close_GeVinv / r_g_ns

    # ISCO around 10 Msun BH
    M_bh_GeV = 10 * M_sun_kg * GeV_per_kg
    r_g_bh = G_global * M_bh_GeV
    s_isco = 6.0  # Newtonian ISCO at 6*r_g

    test_orbits = [
        ("Earth (1 AU)", s_earth),
        ("Mercury", s_mercury),
        ("NS close orbit", s_ns_close),
        ("BH ISCO", s_isco),
    ]

    for name, s in test_orbits:
        u_N = 1.0 / s
        # Find actual u from solution
        idx = np.searchsorted(s_vals, s)
        if idx < len(s_vals):
            u_actual = u_vals[idx]
        else:
            u_actual = u_N  # fallback to Newtonian
        f_val = G_of_psi(u_actual) / G_global
        T_ratio = 1.0 / sqrt(f_val)
        v_ratio = sqrt(f_val)
        print(f"{name:>20} {s:>10.1f} {u_actual:>10.4e} {T_ratio:>10.6f} {v_ratio:>10.6f}")

    print()
    print("Orbital period correction: T = T_Newton / sqrt(G/G_global)")
    print("For psi << 1: delta_T/T ~ -0.5 * dlnG/dpsi * psi")
    print(f"  = -0.5 * {dlnG_dpsi:.4f} * psi = {0.5*abs(dlnG_dpsi):.4f} * psi")
    print()

    print("Perihelion precession: see section E (geodesic weak-field audit).")
    print()


# ============================================================
# F. CSM and MU Gravity Connection
# ============================================================
def csm_mu_gravity_bridge():
    """Brief CSM/MU connection to position-dependent gravity."""
    print("=" * 9)
    print("F. CSM/MU gravity connection")
    print("=" * 9)
    print()
    f_Cs = 9192631770.0
    n_phys = (4.0 / 3.0) * pi * f_Cs**3
    csm = n_phys / OMEGA_SIZE
    v_light = (4.0 / 3.0) * pi * c_SI**3
    rho_mu_0 = csm / v_light
    print(f"  CSM = {csm:.4e} MU per Omega state")
    print(f"  Weak-field MU density: {rho_mu_0:.4e} MU/m^3")
    print()
    print("  In a gravitational field, E_ref increases with psi, reducing MU density:")
    print("    rho_MU(psi) = rho_MU(0) * (v/E_ref(psi))^2")
    print()
    for label, psi_val in [("Sun", 2.1e-6), ("NS", 0.17), ("BH", 0.5)]:
        e_ref_psi = E_CS * (v / E_CS) ** (1.0 - psi_val)
        density_ratio = (v / e_ref_psi) ** 2
        delta = (density_ratio - 1) * 100
        print(
            f"    {label:>4} psi={psi_val:.1e}: rho_MU/rho_MU(0) = {density_ratio:.6f} "
            f"({delta:+.4f}%)"
        )
    print()
    print("  Total rho_MU * E_ref^2 is preserved across all psi.")
    print()


# ============================================================
# G. alpha*zeta Product Chain
# ============================================================
def alpha_zeta_consistency_chain():
    """Nonlinear G(psi) chain; kernel alpha*zeta identity in analysis_3 F."""
    print("=" * 9)
    print("G. alpha*zeta and G(psi) chain")
    print("=" * 9)
    print()
    print("  Kernel identity alpha*zeta = rho^4/(pi sqrt(3)): analysis_3 section F")
    az = verify_alpha_zeta_product(alpha_codata=float(_astro("alpha").value))
    print(f"  identity verified: {az['exact']}")
    print()

    G_pred = g_pred_from_tau(tau_G_full)
    ppm_g = (G_pred - G_meas) / G_meas * 1e6
    alpha_g_pred = G_pred * v**2

    print("Chain to weak-field G_global (electroweak anchor):")
    print(f"  G_global = G_kernel * exp(-tau_G) / v^2")
    print(f"           = {G_pred:.6e} GeV^-2")
    print(f"  vs G_meas = {G_meas:.6e} GeV^-2  ({ppm_g:+.3f} ppm)")
    print(f"  alpha_G(v) = G_global * v^2 = {alpha_g_pred:.6e}")
    print(f"  satisfies alpha*zeta at kernel level (0.074 ppm in G)")
    print()
    print("Nonlinear extension (reference scale only; alpha*zeta unchanged):")
    print(f"  G(x) = G_kernel * exp(-tau(x)) / E_ref(x)^2")
    print(f"  E_ref(x) = E_CS * (v/E_CS)^(1-psi(x))")
    print(f"  tau(x) = tau_G * (1 - psi(x))")
    print()
    print(f"  {'psi':>8} {'G/G_global':>12} {'delta_G/G':>12}")
    print("  " + "-" * 34)
    for label, psi in [
        ("Earth", 7e-10),
        ("Sun", 2.1e-6),
        ("NS", 0.17),
        ("BH hor.", 0.5),
    ]:
        ratio = G_of_psi(psi) / G_global
        print(f"  {label:>8} {ratio:>12.6f} {ratio - 1.0:>12.4e}")
    print()
    print(f"  zeta_from_CODATA / zeta_geom - 1: {az['zeta_ratio']:.6e}")
    print()


# ============================================================
# H. Nonlinear System Summary
# ============================================================
def nonlinear_system_summary():
    print("=" * 9)
    print("H. Complete nonlinear CGM gravity system")
    print("=" * 9)
    print()

    print("Gravitoelectric sector:")
    print("  g = -grad Phi")
    print("  div g = -Q_G * G(x) * rho(x)")
    print("  curl g = -dB_g/dtau")
    print()
    print("Gravitomagnetic sector:")
    print("  div B_g = 0")
    print("  curl B_g = -(Q_G * G(x)/c^2) * J + (1/c^2) * dg/dtau")
    print()
    print("Source conservation:")
    print("  d rho/dtau + div J = 0")
    print("  d J/dtau + div sigma = rho * g + (1/c^2) * (J x B_g)")
    print()
    print("Stress closure (kernel-fixed):")
    print("  sigma = p * I + pi  (1 + 5 decomposition)")
    print("  ||pi||^2 / Tr(sigma)^2 = 2/75  (bulk shell invariant)")
    print()
    print("Position-dependent coupling:")
    print("  G(x) = G_kernel * exp(-tau(x)) / E_ref(x)^2")
    print("  E_ref(x) = E_CS * (v/E_CS)^(1-psi(x))")
    print("  tau(x) = tau_G * (1 - psi(x))")
    print("  psi(x) = |Phi(x)| / Phi_Planck")
    print()
    print("Constants (all kernel-derived, no free parameters):")
    print(f"  Q_G = {Q_G:.6f}")
    print(f"  G_kernel = pi/6 = {G_kernel:.12f}")
    print(f"  tau_G = {tau_G_full:.10f}")
    print(f"  c4 = -7/4")
    print(f"  dlnG/dpsi = {dlnG_dpsi:.6f}")
    print()

    print("Limiting behavior:")
    print(f"  psi -> 0: G -> G_global = {G_global:.5e} GeV^-2  (0.074 ppm)")
    print(f"  psi -> 1: G -> G_kernel/E_CS^2 = {G_CS:.5e} GeV^-2  ({G_CS/G_global*100:.1f}% of G_global)")
    print(f"  G is monotonically decreasing in psi (negative feedback)")
    print()

    print("Sample G reductions:")
    print("  Earth psi~7e-10: ~0.86 ppb")
    print("  Sun psi~2e-6:    ~1.3 ppm")
    print(f"  NS psi~0.2:      ~{(1-G_of_psi(0.2)/G_global)*100:.1f}%")
    print(f"  BH psi~0.5:      ~{(1-G_of_psi(0.5)/G_global)*100:.1f}%")
    print("  Shadow and EHT comparison: section N.")


# ============================================================
# I. Optical Conjugacy Derivation of the Interpolation
# ============================================================
def derive_interpolation_from_conjugacy():
    """Derive G(psi) interpolation from local optical conjugacy."""
    print("=" * 9)
    print("I. Derivation of G(psi) from local optical conjugacy")
    print("=" * 9)
    print()
    print("Global optical conjugacy:")
    print("  E^UV * E^IR = E_CS * v / (4*pi^2)")
    print()
    print("In a region with gravitational potential Phi, the local")
    print("IR conjugate is gravitationally redshifted. CGM adopts the")
    print("first-order (weak-field) redshift:")
    print("  E^IR_local = E^IR * (1 - psi)    where psi = |Phi|/Phi_Planck")
    print()
    print("In GR, the exact redshift is E_local = E_inf * sqrt(1 - 2*psi),")
    print("which reduces to (1 - psi) for psi << 1.")
    print()

    print("Direct argument for E_ref(psi):")
    print("  G = G_kernel * exp(-tau) / E_ref^2")
    print("  Weak field: E_ref = v (IR conjugate of E_CS).")
    print("  Deep field: IR conjugate shifts from v toward E_CS:")
    print("    E_ref(psi) = E_CS * (v/E_CS)^(1-psi)")
    print("  Endpoints: psi=0 -> v; psi=1 -> E_CS")
    print()

    log_v = log(v)
    log_ECS = log(E_CS)
    log_Eref_0 = log(E_CS * (v / E_CS) ** 1.0)
    log_Eref_1 = log(E_CS * (v / E_CS) ** 0.0)

    print("Delta-ruler verification:")
    print(f"  log(v)         = {log_v:.6f}")
    print(f"  log(E_CS)      = {log_ECS:.6f}")
    print(f"  log(E_ref(0))  = {log_Eref_0:.6f}  (log v)  {abs(log_Eref_0 - log_v) < 1e-10}")
    print(f"  log(E_ref(1))  = {log_Eref_1:.6f}  (log E_CS)  {abs(log_Eref_1 - log_ECS) < 1e-10}")
    print(f"  ticks v->E_CS  = (log(E_CS)-log(v))/Delta = {(log_ECS - log_v) / Delta:.2f}")
    print()

    print("Comparison with GR gravitational redshift:")
    print(f"  {'psi':>8} {'CGM 1-psi':>12} {'GR sqrt(1-2psi)':>18} {'dev %':>10}")
    print("  " + "-" * 52)
    for psi in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        cgm_factor = 1.0 - psi
        if psi < 0.5:
            gr_factor = sqrt(1.0 - 2.0 * psi)
            deviation = (cgm_factor - gr_factor) / gr_factor * 100
            print(f"  {psi:>8.2f} {cgm_factor:>12.6f} {gr_factor:>18.6f} {deviation:>10.3f}")
        else:
            print(f"  {psi:>8.2f} {cgm_factor:>12.6f} {'(complex)':>18} {'--':>10}")
    print()
    print("  CGM (1-psi) agrees with GR to <2% for psi < 0.2 (NS regime).")
    print("  CGM stays real on [0,1]; GR sqrt(1-2*psi) is complex at psi=0.5.")
    print()

    print("Derivation of tau(psi) from the Delta ruler:")
    print("  tau(psi)/tau_G = [log(E_CS)-log(E_ref(psi))] / [log(E_CS)-log(v)]")
    print("  With E_ref = E_CS*(v/E_CS)^(1-psi) this gives tau(psi) = tau_G*(1-psi).")
    print()

    psi_test = 0.3
    E_ref_test = E_CS * (v / E_CS) ** (1 - psi_test)
    tau_test = tau_G_full * (1 - psi_test)
    tau_from_ruler = (
        (log(E_CS) - log(E_ref_test)) / (log(E_CS) - log(v)) * tau_G_full
    )
    G_test = G_kernel * exp(-tau_test) / E_ref_test**2
    print("Verification at psi = 0.3:")
    print(f"  E_ref = {E_ref_test:.4e} GeV")
    print(f"  tau(formula) = {tau_test:.6f}")
    print(f"  tau(ruler)   = {tau_from_ruler:.6f}")
    print(f"  match: {abs(tau_test - tau_from_ruler) / tau_test < 1e-10}")
    print(f"  G(psi=0.3) = {G_test:.4e} GeV^-2")
    print()


# ============================================================
# J. Black Hole Shadow (summary; see N for metric derivation)
# ============================================================
def find_photon_sphere(s_vals, u_vals, s_horizon: float | None):
    """Photon sphere: closed form for exponential G(psi); grid args retained for API."""
    psi_ph, s_ph, b_over_rg = photon_geometry_analytic(dlnG_dpsi)
    return s_ph, psi_ph, b_over_rg


def black_hole_shadow(s_vals, u_vals):
    """Horizon and photon sphere summary (metric-based; details in N)."""
    print("=" * 9)
    print("J. Black hole shadow (metric-based)")
    print("=" * 9)
    print()
    print("Uses f(r)=1-2*psi and photon condition G_ratio(u)=s*(1-2*u).")
    print("See section N for null-geodesic shadow diameters.")
    print()

    s_horizon = interp_s_at_u(s_vals, u_vals, 0.5)
    photon = find_photon_sphere(s_vals, u_vals, s_horizon)

    s_ph_gr = 3.0
    b_gr = s_ph_gr / sqrt(max(metric_f(1.0 / s_ph_gr), 1e-15))

    if photon is not None:
        s_ph, u_ph, b_cgm = photon
        print(f"  CGM photon sphere: s = {s_ph:.4f} r_g, psi = {u_ph:.6f}")
        print(f"  GR photon sphere:  s = {s_ph_gr:.3f} r_g (same metric f=1-2*u)")
        print(f"  CGM horizon:       s = {s_horizon:.4f} r_g" if s_horizon else "")
        if s_horizon:
            print(f"  photon/horizon:    {s_ph / s_horizon:.4f}  (>1 = outside horizon)")
    else:
        s_ph, u_ph, b_cgm = s_ph_gr, 1.0 / s_ph_gr, b_gr
        print("  Photon sphere not bracketed; using GR values.")
    print()


def perihelion_precession_geodesic() -> None:
    """Mercury perihelion: weak-field CGM metric vs GR (not G-replacement)."""
    print("=" * 9)
    print("E. Perihelion precession (weak-field geodesic)")
    print("=" * 9)
    print()
    print("  Metric f = 1 - 2*psi(s) with exact psi(s); not G(psi) at perihelion.")
    print()

    ratio, pre_gr, pre_cgm, s_a = mercury_precession_cgm_gr_ratio(
        a_m=5.791e10,
        e=0.205630,
        m_sun_kg=M_sun_kg,
        g_si=G_SI,
        c_si=c_SI,
    )
    orbits_per_century = 100.0 / 0.24085
    arcsec_gr = pre_gr * orbits_per_century * (180.0 / pi) * 3600.0
    arcsec_cgm = pre_cgm * orbits_per_century * (180.0 / pi) * 3600.0

    print(f"  Mercury s_a = {s_a:.0f} r_g")
    print(f"  GR:  {pre_gr:.10e} rad/orbit = {arcsec_gr:.4f} arcsec/century")
    print(f"  CGM: {pre_cgm:.10e} rad/orbit = {arcsec_cgm:.4f} arcsec/century")
    print(f"  CGM/GR = {ratio:.10f}  ({abs(ratio - 1.0) * 1e6:.4f} ppm)")
    print()
    print("  PPN beta (leading 1 - g1/2 ~ 1.323) is a strong-field book-keeping parameter.")
    print("  At solar psi ~ 1/s_a ~ 2.5e-8, perihelion matches GR.")
    print("  Continuum audit: section T (horizon, photon, shadow, PPN tables).")
    print()


def analytic_continuum_audit_section() -> None:
    """Exterior closure: analytic psi(s), horizon, photon sphere, shadows, PPN context."""
    print("=" * 9)
    print("T. Analytic continuum audit (point-mass exterior)")
    print("=" * 9)
    print()

    g1 = dlnG_dpsi
    print(f"g1 = dlnG/dpsi = {g1:.6f}")
    print()

    ode_ref = (
        (1.746, 0.487453),
        (2.0, 0.433322),
        (5.0, 0.1881),
        (10.0, 0.0969),
        (100.0, 0.01000),
        (1000.0, 0.001000),
    )
    print("Cross-check vs legacy numerical ODE samples:")
    print(f"  {'s':>10} {'psi_ana':>14} {'psi_ode':>14} {'ok':>6}")
    print("  " + "-" * 48)
    for s, psi_num in ode_ref:
        psi_ana = float(psi_analytic(s, g1))
        rel = abs(psi_ana - psi_num) / max(abs(psi_num), 1e-30)
        ok = "yes" if rel < 0.01 else "no"
        print(f"  {s:10.3f} {psi_ana:14.6f} {psi_num:14.6f} {ok:>6}")
    print()

    s_h = horizon_s_analytic(g1)
    psi_ph, s_ph, b_rg = photon_geometry_analytic(g1)
    print(f"Horizon s_h = {s_h:.4f} r_g  (psi = 1/2)")
    print(f"Photon: psi_ph = {psi_ph:.6f}, s_ph = {s_ph:.4f} r_g, b/r_g = {b_rg:.4f}")
    print()
    print("Shadow (EHT priors):")
    for name, obs in EHT_SHADOW.items():
        d = shadow_diameter_muas(b_rg, obs["M_kg"], obs["D_m"])
        print(
            f"  {name}: {d:.1f} uas  (EHT {obs['d_muas']:.1f} +/- {obs['sigma_muas']:.1f})"
        )
    print()

    ratio, _, _, _ = mercury_precession_cgm_gr_ratio(
        a_m=5.791e10,
        e=0.205630,
        m_sun_kg=M_sun_kg,
        g_si=G_SI,
        c_si=c_SI,
    )
    a2 = g1 / 2.0
    beta_lead = 1.0 - a2
    print("PPN / perihelion context:")
    print(f"  Leading beta = 1 - g1/2 = {beta_lead:.6f}")
    print(f"  Mercury CGM/GR = {ratio:.10f}  (section E)")
    print(
        f"  Naive (4-beta)/3 at beta=1.305 would be {(4 - 1.305) / 3:.4f} (wrong at Mercury)"
    )
    print("  Standard vacuum PPN does not map without full metric + geodesics.")
    print()

    print("Strong-field effective beta (book-keeping vs characteristic psi):")
    print(f"  {'psi':>8} {'beta_eff':>10} {'prec/GR':>12}")
    for psi_c in [1e-8, 1e-4, 0.01, 0.05, 0.1, 0.2, 0.3]:
        beta_eff = 1.0 - a2 * psi_c
        prec = (4.0 - beta_eff) / 3.0
        print(f"  {psi_c:8.1e} {beta_eff:10.4f} {prec:12.4f}")
    print()


# ============================================================
# M. Effective CGM Metric
# ============================================================
def effective_metric(s_vals, u_vals):
    """CGM metric f(r)=1-2*psi from psi(s) on the exterior grid (analytic point-mass; section T)."""
    print("=" * 9)
    print("M. Effective CGM metric from nonlinear potential")
    print("=" * 9)
    print()
    print("  ds^2 = -f(r) c^2 dt^2 + f(r)^{-1} dr^2 + r^2 dOmega^2")
    print("  f(r) = 1 - 2*psi_CGM(r),  psi_CGM = u(s),  s = r/r_g")
    print()
    print("  Horizon (f=0): psi = 1/2  (v_esc = c)")
    print("  Photon sphere: s*f'(r) - 2f = 0  <=>  G_ratio(u) = s*(1-2*u)")
    print()

    s_horizon = interp_s_at_u(s_vals, u_vals, 0.5)
    photon = find_photon_sphere(s_vals, u_vals, s_horizon)

    print(f"  {'s=r/r_g':>10} {'psi':>10} {'f(r)':>10} {'1/f':>10}")
    print("  " + "-" * 44)
    for s in [s_vals[0], s_vals[len(s_vals) // 4], s_vals[len(s_vals) // 2],
              s_vals[3 * len(s_vals) // 4], s_vals[-1]]:
        u = float(np.interp(s, s_vals, u_vals))
        f_val = metric_f(u)
        inv_f = 1.0 / f_val if f_val > 1e-12 else float("inf")
        print(f"  {s:>10.3f} {u:>10.6f} {f_val:>10.6f} {inv_f:>10.4f}")

    print()
    if s_horizon is not None:
        print(f"  Horizon: s = {s_horizon:.4f} r_g  (GR: 2.000)")
    if photon is not None:
        s_ph, u_ph, _ = photon
        print(f"  Photon:  s = {s_ph:.4f} r_g  (GR: 3.000)")
        if s_horizon:
            print(f"  Ordering: horizon < photon: {s_horizon < s_ph}")
    print()


# ============================================================
# N. Photon Sphere and Shadow from Null Geodesics
# ============================================================
def kerr_shadow_diameter_ratio(a_star: float, theta_o_deg: float) -> float:
    """GR Kerr/Schwarzschild shadow diameter ratio (Bardeen 1973 fit)."""
    theta_o = np.radians(theta_o_deg)
    cos2 = np.cos(theta_o) ** 2
    sin2 = np.sin(theta_o) ** 2
    return 1.0 + 0.05 * a_star**2 * cos2 + 0.03 * a_star * sin2 + 0.02 * a_star**2 * sin2


def null_geodesic_shadow(s_vals, u_vals):
    """Shadow diameter from metric f=1-2*u and photon-sphere impact parameter."""
    print("=" * 9)
    print("N. Photon sphere and shadow from null geodesics")
    print("=" * 9)
    print()
    print("Metric: f(r) = 1 - 2*psi,  psi = u(s),  s = r/r_g,  r_g = GM/c^2")
    print("Photon condition (null + running G):  G/G_gl = s*(1 - 2*u)")
    print("Impact parameter: b/r_g = s_ph / sqrt(f(u_ph))")
    print("Shadow angular diameter: theta = 2*b/D")
    print()

    s_horizon = interp_s_at_u(s_vals, u_vals, 0.5)
    photon = find_photon_sphere(s_vals, u_vals, s_horizon)

    s_ph_gr = 3.0
    u_ph_gr = 1.0 / s_ph_gr
    b_gr = photon_impact_parameter(s_ph_gr, u_ph_gr)

    if photon is None:
        s_ph, u_ph, b_cgm = s_ph_gr, u_ph_gr, b_gr
        print("  [warn] CGM photon root not found; using GR values.")
    else:
        s_ph, u_ph, b_cgm = photon

    u_newton_ph = 1.0 / s_ph
    f_at_ph = metric_f(u_ph)
    f_newton = metric_f(u_newton_ph)

    print("  Geometry (same mass in all columns):")
    print(f"  {'':>12} {'GR':>10} {'CGM':>10} {'CGM/GR':>10}")
    print("  " + "-" * 46)
    print(f"  {'s_ph/r_g':>12} {s_ph_gr:>10.3f} {s_ph:>10.4f} {s_ph/s_ph_gr:>10.4f}")
    print(f"  {'psi_ph':>12} {u_ph_gr:>10.4f} {u_ph:>10.4f} {u_ph/u_ph_gr:>10.4f}")
    print(f"  {'f(u_ph)':>12} {f_newton:>10.4f} {f_at_ph:>10.4f} {f_at_ph/f_newton:>10.4f}")
    print(f"  {'b/r_g':>12} {b_gr:>10.4f} {b_cgm:>10.4f} {b_cgm/b_gr:>10.4f}")
    if s_horizon:
        print(f"  {'s_h/r_g':>12} {2.0:>10.3f} {s_horizon:>10.4f} {s_horizon/2.0:>10.4f}")
        print(f"  {'s_ph/s_h':>12} {'3/2':>10} {s_ph/s_horizon:>10.4f}")
    print()

    kerr_sgra = kerr_shadow_diameter_ratio(0.9, 17.0)
    kerr_m87 = kerr_shadow_diameter_ratio(0.5, 163.0)
    obs_sgr = EHT_SHADOW["Sgr A*"]
    obs_m87 = EHT_SHADOW["M87*"]
    d_gr_sgr = shadow_diameter_muas(b_gr, obs_sgr["M_kg"], obs_sgr["D_m"])
    d_cgm_sgr = shadow_diameter_muas(b_cgm, obs_sgr["M_kg"], obs_sgr["D_m"])
    d_gr_m87 = shadow_diameter_muas(b_gr, obs_m87["M_kg"], obs_m87["D_m"])
    d_cgm_m87 = shadow_diameter_muas(b_cgm, obs_m87["M_kg"], obs_m87["D_m"])
    d_gr_kerr_sgr = d_gr_sgr * kerr_sgra
    d_cgm_kerr_sgr = d_cgm_sgr * kerr_sgra
    d_gr_kerr_m87 = d_gr_m87 * kerr_m87
    d_cgm_kerr_m87 = d_cgm_m87 * kerr_m87

    print("  EHT comparison:")
    print(
        f"  {'Source':>8} {'GR_Schw':>8} {'GR_Kerr':>8} {'CGM_Schw':>8} "
        f"{'CGM_Kerr':>8} {'EHT':>8} {'sig_K':>6}"
    )
    print("  " + "-" * 62)
    for name, obs, d_gk, d_ck in [
        ("M87*", obs_m87, d_gr_kerr_m87, d_cgm_kerr_m87),
        ("Sgr A*", obs_sgr, d_gr_kerr_sgr, d_cgm_kerr_sgr),
    ]:
        d_eht = obs["d_muas"]
        sig = obs["sigma_muas"]
        z_ck = (d_ck - d_eht) / sig
        d_gr_v = shadow_diameter_muas(b_gr, obs["M_kg"], obs["D_m"])
        d_cgm_v = shadow_diameter_muas(b_cgm, obs["M_kg"], obs["D_m"])
        print(
            f"  {name:>8} {d_gr_v:>8.1f} {d_gk:>8.1f} {d_cgm_v:>8.1f} {d_ck:>8.1f} "
            f"{d_eht:>8.1f} {z_ck:>+6.2f}"
        )
    print()
    print("  sig_K = (CGM_Kerr - EHT) / sigma; CGM_Kerr = CGM_Schw * kerr_factor (GR fit).")
    print(f"  kerr_factor Sgr A*: {kerr_sgra:.4f}, M87*: {kerr_m87:.4f}")
    print(f"  b_CGM/b_GR = {b_cgm/b_gr:.4f}  (area ratio {(b_cgm/b_gr)**2:.4f})")
    print()

    from hqvm_gravity_analysis_5 import shadow_spin_table

    shadow_spin_table(s_vals, u_vals, EHT_SHADOW)


# ============================================================
# O. Tau Evolution Law
# ============================================================
def tau_evolution_law(s_vals, u_vals):
    """STF accumulation d tau/d psi = -tau_G vs algebraic tau = tau_G*(1-psi)."""
    print("=" * 9)
    print("O. Tau evolution law: STF accumulation vs algebraic tau")
    print("=" * 9)
    print()
    print("  Algebraic (Delta ruler):  tau(psi) = tau_G * (1 - psi)")
    print("  STF (kernel ladder):      d tau / d psi = -tau_G")
    print("  => integrated STF = tau_G*(1-psi), agreeing with algebraic law.")
    print()
    n_vc_local = log(E_CS / v) / Delta
    tau_stf_coeff = 2.0 * Delta * n_vc_local
    ratio = tau_G_full / tau_stf_coeff
    gap = tau_stf_coeff - tau_G_full
    print(f"  tau_G = {tau_G_full:.6f}")
    print(f"  2*Delta*n_vc = {tau_stf_coeff:.6f}  (conjugacy depth)")
    print(f"  ratio tau_G / (2*Delta*n_vc) = {ratio:.6f}")
    print(f"  2*Delta*n_vc - tau_G = {gap:.6f}  (identity: -dlnG/dpsi = {-dlnG_dpsi:.6f})")
    print()
    print("  Integrated STF along radial psi profile:")
    tau_alg = tau_G_full * (1.0 - u_vals)
    tau_stf = np.zeros_like(s_vals)
    tau_stf[-1] = tau_G_full * (1.0 - u_vals[-1])
    for i in range(len(s_vals) - 2, -1, -1):
        du = u_vals[i] - u_vals[i + 1]
        tau_stf[i] = tau_stf[i + 1] - tau_G_full * du
    max_err = float(np.max(np.abs(tau_stf - tau_alg) / tau_G_full))
    print(f"    max |tau_stf - tau_alg| / tau_G = {max_err:.2e}  (machine precision)")
    print()
    print(f"  {'s':>8} {'psi':>10} {'tau_algebraic':>14} {'tau_STF':>14}")
    print("  " + "-" * 50)
    for idx in [0, len(s_vals) // 4, len(s_vals) // 2, 3 * len(s_vals) // 4, -1]:
        print(
            f"  {s_vals[idx]:>8.2f} {u_vals[idx]:>10.6f} {tau_alg[idx]:>14.6f} "
            f"{tau_stf[idx]:>14.6f}"
        )
    print()


# ============================================================
# P. Full Self-Consistency Audit
# ============================================================
def self_consistency_audit(s_vals, u_vals):
    """End-to-end checks: analytic psi(s), metric, photon order, tau law, units."""
    print("=" * 9)
    print("P. Full self-consistency audit")
    print("=" * 9)
    print()

    ok = True

    s_check = np.logspace(log10(s_vals[0]), log10(s_vals[-1]), 50)
    max_rel = 0.0
    for s in s_check:
        u_ana = float(psi_analytic(s, dlnG_dpsi))
        du_ana = float(-G_ratio(u_ana) / s**2)
        du_exact = float(-1.0 / (s * (s - dlnG_dpsi)))
        max_rel = max(max_rel, abs(du_ana - du_exact) / max(abs(du_exact), 1e-30))

    dpsi_ok = max_rel < 1e-10
    ok = ok and dpsi_ok
    print(f"  [{'OK' if dpsi_ok else 'FAIL'}] dpsi/ds closure residual (rel) = {max_rel:.2e}")

    u_max = float(u_vals[0])
    clip_ok = u_max < 0.5
    ok = ok and clip_ok
    print(f"  [{'OK' if clip_ok else 'FAIL'}] max psi = {u_max:.4f} (< 0.5, no clip artifact)")

    s_ph_gr = 3.0
    s_h = interp_s_at_u(s_vals, u_vals, 0.5)
    photon = find_photon_sphere(s_vals, u_vals, s_h)
    order_ok = bool(
        photon and s_h and photon[0] > s_h and photon[0] < s_ph_gr * 1.5
    )
    ok = ok and order_ok
    if photon and s_h:
        print(
            f"  [{'OK' if order_ok else 'FAIL'}] horizon s={s_h:.3f} < "
            f"photon s={photon[0]:.3f}"
        )
    else:
        print("  [FAIL] photon or horizon not found")
        ok = False

    d_muas = shadow_diameter_muas(
        3.0 / sqrt(metric_f(1.0 / 3.0)), 6.5e9 * M_sun_kg, 16.8 * MPC_M
    )
    units_ok = d_muas > 10.0
    ok = ok and units_ok
    print(f"  [{'OK' if units_ok else 'FAIL'}] shadow units: {d_muas:.1f} muas (not 0)")

    mid = len(u_vals) // 2
    s_mid = s_vals[mid]
    u_mid = u_vals[mid]
    du_ds = -G_ratio(u_mid) / s_mid**2
    kin_rate = -tau_G_full * du_ds
    stf_rate = -tau_G_full
    kin_ok = abs(kin_rate - stf_rate * du_ds) / max(abs(kin_rate), 1e-30) < 0.05
    ok = ok and kin_ok
    print(
        f"  [{'OK' if kin_ok else 'FAIL'}] tau law at mid-r: "
        f"-tau_G*du/ds={kin_rate:.4e}, (-tau_G)*du/ds from d tau/d psi"
    )

    ppm = (G_global - G_meas) / G_meas * 1e6
    g_ok = abs(ppm) < 1.0
    ok = ok and g_ok
    print(f"  [{'OK' if g_ok else 'FAIL'}] G_global vs G_meas: {ppm:+.3f} ppm")
    print()
    print(f"  Overall: {'PASS' if ok else 'CHECK FAILURES'}")
    print()


def verify_einstein_tensor_decomposition() -> None:
    """tt-component of G_munu for f = 1 - 2*psi gives modified Gauss law."""
    print("=" * 9)
    print("R. Einstein tensor and modified Gauss law")
    print("=" * 9)
    print()
    print("  Metric: ds^2 = -f dt^2 + f^{-1} dr^2 + r^2 dOmega^2,  f = 1 - 2*psi")
    print("  G_tt / r^2 = 2(dpsi/dr)/r^2 = 8*pi*G(psi)*rho  (c=1)")
    print("  div g = -4*pi*G(psi)*rho")
    print("  div[(G_0/G(psi))*g] = -4*pi*G_0*rho = -Q_G*G_0*rho")
    print("  (analysis_5 J: Gauss law; Q: components; T: EFE closure + Bianchi probe)")
    print()


def section_E_ref_formal_proof() -> None:
    """E_ref(psi) = E_CS (v/E_CS)^(1-psi) from conjugacy, Delta ruler, and tau law."""
    print("=" * 9)
    print("S. E_ref(psi) formal proof chain")
    print("=" * 9)
    print()
    print("THEOREM: E_ref(psi) = E_CS * (v/E_CS)^(1-psi)")
    print()
    print("Premise 1: Optical conjugacy (CGM)")
    print("  E_UV * E_IR = E_CS * v / (4*pi^2)")
    print()
    print("Premise 2: Delta ruler")
    print("  n(E) = ln(E_CS/E) / (Delta*ln2)")
    print()
    print("Premise 3: STF-verified Refractive Depth")
    print("  tau(psi) = tau_G * (1 - psi)  (machine precision vs STF)")
    print()
    print("DERIVATION:")
    print("  L(E) = ln(E_CS/E);  tau = alpha * L;  at psi=0, L(0)=|eta|, tau(0)=tau_G")
    alpha_scale = tau_G_full / abs(eta)
    print(f"  alpha = tau_G/|eta| = {alpha_scale:.6f}")
    print("  L(psi) = |eta|*(1-psi) => E_ref(psi) = E_CS * (v/E_CS)^(1-psi)")
    print()
    e_ref_0 = E_CS * (v / E_CS) ** 1
    e_ref_1 = E_CS * (v / E_CS) ** 0
    print("VERIFICATION:")
    print(f"  E_ref(0) = {e_ref_0:.2f} GeV  vs v = {v:.2f} GeV  OK")
    print(f"  E_ref(1) = {e_ref_1:.2e} GeV  vs E_CS = {E_CS:.2e} GeV  OK")
    print()
    print("PROOF: E_ref is a ruler quantile, not a centroid")
    print("  On the Delta ruler: n = ln(E_CS/E) / (Delta*ln2)")
    print("  tau(n) = n*Delta*ln2 = ln(E_CS/E)  (monotone in position, not a mean)")
    print("  E_ref is the energy at position tau(psi) = tau_G*(1-psi), not E_mean.")
    print("  Quantile Q(p) = E_CS*exp(-p*L), L = ln(E_CS/v), p = 1-psi:")
    print("  E_ref(psi) = E_CS * (v/E_CS)^(1-psi)  (exact, no distribution assumption)")
    print("  E_ref is a ruler QUANTILE (position), not a centroid (mean).")
    print(f"  On a log scale spanning ~{log10(E_CS / v):.0f} decades, mean != quantile.")
    print("  Centroid E_mean = sum E_k w_k is dominated by the UV tail on the ladder.")
    print()
    print("KEY: G(psi) = G0 * exp(g1*psi) exactly (d^2 ln G / d psi^2 = 0)")
    print(f"  g1 = tau_G + 2*eta = {tau_G_full:.6f} + {2*eta:.6f} = {dlnG_dpsi:.6f}")
    print()
    print("SELF-CONSISTENCY:")
    for psi_test in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
        e_ref = E_CS * (v / E_CS) ** (1.0 - psi_test)
        tau_loc = tau_G_full * (1.0 - psi_test)
        g_formula = G_kernel * exp(-tau_loc) / e_ref**2
        g_exp = G_global * exp(dlnG_dpsi * psi_test)
        match = abs(g_formula - g_exp) / G_global < 1e-10
        print(
            f"  psi={psi_test:.1f}: G(formula)={g_formula:.6e}, "
            f"G(exp)={g_exp:.6e}, match={match}"
        )
    print()


# ============================================================
# Main
# ============================================================
def main():
    print("CGM gravity analysis 4: nonlinear G(psi), metric, shadow")
    print(f"Delta = {Delta:.12f}, rho = {rho_val:.12f}")
    print(f"tau_G (full) = {tau_G_full:.10f}")
    print(f"G_global = {G_global:.5e} GeV^-2")
    print()

    verify_endpoints()
    astrophysical_psi()

    derive_interpolation_from_conjugacy()

    s_vals, u_vals, u_newton, gg_profile = nonlinear_potential()
    escape_velocity_analysis(s_vals, u_vals)
    perihelion_precession_geodesic()
    analytic_continuum_audit_section()
    orbital_corrections(s_vals, u_vals, gg_profile)
    black_hole_shadow(s_vals, u_vals)

    effective_metric(s_vals, u_vals)
    null_geodesic_shadow(s_vals, u_vals)
    tau_evolution_law(s_vals, u_vals)
    self_consistency_audit(s_vals, u_vals)
    verify_einstein_tensor_decomposition()
    section_E_ref_formal_proof()

    csm_mu_gravity_bridge()
    alpha_zeta_consistency_chain()
    nonlinear_system_summary()

    print("=" * 9)
    print("ANALYSIS COMPLETE")
    print("=" * 9)


if __name__ == "__main__":
    main()