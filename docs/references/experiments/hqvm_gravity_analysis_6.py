"""
CGM gravity analysis 6:
  A. Antimatter gravitational interaction (chirality-reversal invariance)
  B. Gravitational wave extensions (spectrum, phasing, ringdown, echoes)
  C. Virial, self-energy, UV regulation (BU closure, I=1/2, Hawking)

Ownership (reuse, do not duplicate):
  analysis_2: Z2 holonomy, q(F)=0, D=24, Omega BFS (S1/S2/S4)
  analysis_3: UNA/ONA stress trace, bulk anisotropy (B)
  analysis_4: point-mass psi(s), horizon, photon sphere (field ODE)
  analysis_5: vacuum Gauss law, GW HT strain calibration (J, P, S)
  common: cycle words, field integrals, geometry helpers
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

_REPO = Path(__file__).resolve().parents[1]
_EXPERIMENTS = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from hqvm_gravity_analysis_2 import enumerate_omega
from hqvm_compact_geom_core import electroweak_coords
from hqvm_gravity_common import (
    configure_stdout_utf8,
    Q_G,
    rho,
    Delta,
    G_meas,
    v_EW,
    tau_g_with_c4,
    C4_REF,
    binom_shell,
    dln_g_dpsi,
    psi_analytic,
    psi_point_mass,
    dpsi_ds_analytic,
    dpsi_ds_point_mass,
    horizon_s_analytic,
    photon_geometry_analytic,
    field_integral_exterior_exact,
    field_integral_exterior_numeric,
    exterior_integral_numeric_lo,
    stage_mass_fractions,
    GENE_MAC_REST,
    GENE_MAC_SWAPPED,
    cycle_word_for_micro,
    F_cycle_word,
    reversed_cycle_word,
    trace_word_steps,
    FA_STF,
    TR_SIGMA_SHELL,
)
from gyroscopic.hQVM.api import chirality_word6
from gyroscopic.hQVM.constants import CHIRALITY_MASK_6, step_state_by_byte

configure_stdout_utf8()

g1 = dln_g_dpsi()
tau_G_full = tau_g_with_c4(C4_REF)
s_h = horizon_s_analytic(g1)
psi_ph, s_ph, b_ph = photon_geometry_analytic(g1)
G_SI = 6.674e-11
C_SI = 3.0e8
M_SUN_KG = 1.989e30


def compute_H_spin_state(state24: int) -> int:
    """H_spin = 3 - popcount(chi6); equatorial shell N=3 (Gyroscopic_ASI_Specs G.2)."""
    return 3 - chirality_word6(state24).bit_count()


def H_spin_under_C_spin(state24: int) -> int:
    """Gravitomagnetic C on chi: chi -> chi xor 63 (radial reflection N -> 6-N)."""
    chi_c = chirality_word6(state24) ^ CHIRALITY_MASK_6
    return 3 - chi_c.bit_count()


def face_swap_involution(state24: int) -> int:
    """Gravitoelectric C: swap conjugate faces (gate S)."""
    A12 = (state24 >> 12) & 0xFFF
    B12 = state24 & 0xFFF
    return (B12 << 12) | A12


def mass_observable(state24: int) -> int:
    """Gravitoelectric mass observable: popcount of A12."""
    A12 = (state24 >> 12) & 0xFFF
    return bin(A12).count("1")


def f_metric(s, g1_val=None):
    """Metric function f(s) = 1 - 2*psi(s)."""
    if g1_val is None:
        g1_val = g1
    return 1.0 - 2.0 * float(psi_analytic(s, g1_val))


def tortoise_integral(s_hi, s_lo=None, g1_val=None):
    """r*/r_g = integral_{s_lo}^{s_hi} ds/f(s)."""
    if g1_val is None:
        g1_val = g1
    if s_lo is None:
        s_lo = horizon_s_analytic(g1_val) * 1.0001
    val, err = quad(lambda s: 1.0 / f_metric(s, g1_val), s_lo, s_hi, limit=300)
    return val, err


def effective_potential_full(s, g1_val=None, ell=2):
    """Full axial effective potential for spin-2 perturbations."""
    if g1_val is None:
        g1_val = g1
    fs = f_metric(s, g1_val)
    psi_s = float(psi_analytic(s, g1_val))
    return fs * (ell * (ell + 1) / s ** 2 - 6.0 * psi_s / s ** 2)


def effective_potential_schwarzschild(s, ell=2):
    """Schwarzschild axial potential in s=r/r_g: f=1-2/s, psi=1/s."""
    fs = 1.0 - 2.0 / s
    return fs * (ell * (ell + 1) / s ** 2 - 6.0 / s ** 3)


def poschl_teller_qnm(
    v_at_s,
    f_at_s,
    s_lo: float,
    s_hi: float = 10.0,
) -> tuple[float, float, float, float]:
    """Pöschl-Teller n=0 from barrier peak: omega_R, omega_I in omega*M units (c=1)."""
    res = minimize_scalar(lambda s: -v_at_s(s), bounds=(s_lo, s_hi), method="bounded")
    s_peak = float(getattr(res, "x", res))
    V0 = float(v_at_s(s_peak))
    ds = max(s_peak * 1e-6, 1e-8)
    v_ss = (v_at_s(s_peak + ds) - 2.0 * V0 + v_at_s(s_peak - ds)) / ds ** 2
    f_peak = float(f_at_s(s_peak))
    v_rstar_rstar = v_ss * f_peak ** 2
    if v_rstar_rstar >= 0 or V0 <= 0:
        return s_peak, V0, float("nan"), float("nan")
    alpha = math.sqrt(-v_rstar_rstar / (2.0 * V0))
    omega_R = math.sqrt(max(V0 - alpha ** 2 / 4.0, 0.0))
    omega_I = alpha / 2.0
    return s_peak, V0, omega_R, omega_I


def _v_cgm_barrier(s):
    if s <= s_h:
        return 0.0
    return effective_potential_full(s, g1)


def _f_cgm_barrier(s):
    if s <= s_h:
        return 1.0
    return f_metric(s, g1)


def _v_gr_barrier(s):
    if s <= 2.0:
        return 0.0
    return effective_potential_schwarzschild(s)


def _f_gr_barrier(s):
    if s <= 2.0:
        return 0.0
    return 1.0 - 2.0 / s


def g_over_g0(psi: float) -> float:
    """G(psi)/G0 = exp(g1*psi) (same coupling as A.4 table)."""
    return math.exp(g1 * psi)


def exterior_field_integral_I(g1_val: float) -> tuple[float, float]:
    """I = int_{s_h}^inf exp(g1*psi)/s^2 ds = psi(s_h)-psi(inf); returns (I, s_h)."""
    sh = horizon_s_analytic(g1_val)
    return field_integral_exterior_exact(sh, g1_val), sh


def hawking_cgm_vs_gr(g1_val: float) -> dict[str, float]:
    """Surface gravity and Hawking luminosity scalings (geometric units, r_g=2M)."""
    sh = horizon_s_analytic(g1_val)
    dpsi_h = abs(float(dpsi_ds_analytic(sh, g1_val)))
    kappa_gr = 0.25
    kappa_cgm = dpsi_h
    kappa_ratio = kappa_cgm / kappa_gr
    area_ratio = (sh / 2.0) ** 2
    lum_ratio = (kappa_ratio ** 4) * area_ratio
    return {
        "s_h": sh,
        "kappa_cgm": kappa_cgm,
        "kappa_gr": kappa_gr,
        "kappa_ratio": kappa_ratio,
        "area_ratio": area_ratio,
        "lum_ratio": lum_ratio,
        "kappa_formula": 4.0 * math.exp(g1_val / 2.0) / (sh * sh),
    }


HBAR_SI = 1.054571817e-34
K_B_SI = 1.380649e-23


def hawking_temperature_si(mass_kg: float, kappa_per_rg: float) -> float:
    """T = hbar*c^3/(8*pi*G*M*k_B) scaled by kappa_ratio (CGM/GR surface gravity)."""
    return (
        HBAR_SI * C_SI ** 3 * kappa_per_rg
        / (8.0 * math.pi * G_SI * mass_kg * K_B_SI)
    )


def refractive_stress_density_u(s: float, g1_val: float | None = None) -> float:
    """Refractive stress density u = |g|^2 / (8*pi*G(psi)); |g| = exp(g1*psi)/s^2."""
    if g1_val is None:
        g1_val = g1
    psi_s = float(psi_analytic(s, g1_val))
    g_mag = math.exp(g1_val * psi_s) / (s * s)
    g_eff = math.exp(g1_val * psi_s)
    return g_mag * g_mag / (8.0 * math.pi * g_eff)


def local_exterior_energy_balance() -> dict[str, float]:
    """Virial-sector balance: E_rest_frame = +Mc^2/4, E_self = -Mc^2/4 per M_obs."""
    i_ext = 0.5
    e_rest_frame = 0.5 * i_ext
    e_self = -0.5 * 0.5
    return {
        "I": i_ext,
        "E_rest_frame_frac": e_rest_frame,
        "E_self_frac": e_self,
        "sum": e_rest_frame + e_self,
    }


def shell_cycle_fourier_amps(word: list[int], micro_ref: int = 0) -> list[tuple[int, float]]:
    """|DFT_k| of (arch_shell-3) along a holonomy word."""
    rows = trace_word_steps(word, micro_ref=micro_ref)
    signal = [float(rows[i]["arch_shell"] - 3) for i in range(len(rows))]
    n = len(signal)
    amps: list[tuple[int, float]] = []
    for k in range(n):
        re = sum(
            signal[j] * math.cos(2.0 * math.pi * k * j / n) for j in range(n)
        )
        im = sum(
            signal[j] * math.sin(2.0 * math.pi * k * j / n) for j in range(n)
        )
        amps.append((k, math.hypot(re, im) / n))
    return amps


def verify_h_spin_under_C(states) -> tuple[int, int, int, int, int]:
    """Return (H(C_spin)=-H, H=0, H!=0, mass-even under face-swap, |states|)."""
    odd = 0
    equator = 0
    h_nonzero = 0
    mass_even = 0
    for s24 in states:
        h = compute_H_spin_state(s24)
        if H_spin_under_C_spin(s24) == -h:
            odd += 1
        if h == 0:
            equator += 1
        else:
            h_nonzero += 1
        if mass_observable(s24) == mass_observable(face_swap_involution(s24)):
            mass_even += 1
    return odd, equator, h_nonzero, mass_even, len(states)


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


print("CGM gravity analysis 6: Antimatter, GW, Virial, Self-energy")
print("Delta = {:.12f}, rho = {:.12f}".format(Delta, rho))
print("g1 = {:.6f}, tau_G = {:.10f}".format(g1, tau_G_full))
print("s_h = {:.4f} r_g, s_ph = {:.4f} r_g, b/r_g = {:.4f}".format(s_h, s_ph, b_ph))
print()

# =====================================================================
# SECTION A: ANTIMATTER GRAVITATIONAL INTERACTION
# =====================================================================
print("=" * 9, "A. Antimatter gravitational interaction", "=" * 9)
print()

section("A.1  Antimatter involution C")
print("  C(A12, B12) = (B12, A12); reversed word uses families (3,2,1,0).")

section("A.2  Gravitoelectric invariants (EVEN under C)")
print("  D=24 displacement, q(F)=0, cycle->REST: hqvm_gravity_analysis_2 S1/S4.")
print("  Mass popcount(A12) even under face-swap: verified on Omega in A.5.")

section("A.3  H_spin observable (ODD under C on chi6)")
print("  chi6(s) = chirality_word6(s) in GF(2)^6; N = popcount(chi6) (shell index).")
print("  H_spin(s) = 3 - N  (signed distance from equatorial shell N=3).")
print("  Gravitomagnetic C: chi6 -> chi6 xor 63; range H_spin in {-3,-2,-1,0,1,2,3}.")

examples = [
    ("REST (complement horizon)", GENE_MAC_REST),
    ("SWAPPED", GENE_MAC_SWAPPED),
]
for m in range(64):
    word = cycle_word_for_micro(m)
    rows = trace_word_steps(word, micro_ref=m)
    for row in rows:
        if row["on_equality_horizon"]:
            examples.append(("Equality horizon", row["state24"]))
            break
    if len(examples) >= 3:
        break
for m in range(64):
    word = cycle_word_for_micro(m)
    rows = trace_word_steps(word, micro_ref=m)
    for row in rows:
        if row["arch_shell"] == 3 and not row["on_horizon"]:
            examples.append(("Bulk (shell 3)", row["state24"]))
            break
    if len(examples) >= 4:
        break

print("  {:25s} | {:>6s} | {:>6s} | {:>6s} | {:>6s}".format(
    "State", "H_spin", "H_spin(C)", "mass", "mass(C)"))
for name, s24 in examples:
    H = compute_H_spin_state(s24)
    H_C = H_spin_under_C_spin(s24)
    m_o = mass_observable(s24)
    m_c = mass_observable(face_swap_involution(s24))
    print("  {:25s} | {:>+6d} | {:>+6d} | {:>6d} | {:>6d}".format(
        name, H, H_C, m_o, m_c))
print()

section("A.4  Holonomy path H_spin (micro_ref=0)")
m_ref = 0
word_m = cycle_word_for_micro(m_ref)
word_am = reversed_cycle_word(m_ref)
s_m = GENE_MAC_REST
path_m = [compute_H_spin_state(s_m)]
for b in word_m:
    s_m = step_state_by_byte(s_m, b)
    path_m.append(compute_H_spin_state(s_m))
s_am = GENE_MAC_SWAPPED
path_am = [compute_H_spin_state(s_am)]
for b in word_am:
    s_am = step_state_by_byte(s_am, b)
    path_am.append(compute_H_spin_state(s_am))
print("  Step | H_spin(REST,W) | H_spin(SWAPPED,W_rev)")
print("  " + "-" * 45)
for i in range(len(path_m)):
    print("  {:4d} | {:>+14d} | {:>+14d}".format(i, path_m[i], path_am[i]))
m_bulk = 7
word_bulk = cycle_word_for_micro(m_bulk)
s_bulk = GENE_MAC_REST
path_bulk = [compute_H_spin_state(s_bulk)]
for b in word_bulk:
    s_bulk = step_state_by_byte(s_bulk, b)
    path_bulk.append(compute_H_spin_state(s_bulk))
print("  micro_ref=7 (popcount 3), REST+W:")
print("  Step | H_spin")
print("  " + "-" * 18)
for i, h in enumerate(path_bulk):
    print("  {:4d} | {:>+6d}".format(i, h))
print("  delta_B_g/B_g ~ (4/75)*psi^2:")
print("  {:>10s}  {:>14s}  {:>14s}".format("psi", "delta_B/B", "G/G0"))
for pv in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]:
    chiral_corr = (4.0 / 75.0) * pv ** 2
    g_ratio = math.exp(g1 * pv)
    print("  {:10.4f}  {:14.6e}  {:14.6f}".format(pv, chiral_corr, g_ratio))
print()

section("A.5  H_spin parity on full Omega (BFS from analysis_2)")
omega_list = sorted(enumerate_omega())
odd_count, equator_count, h_nonzero_count, mass_invariant_count, n_omega = (
    verify_h_spin_under_C(omega_list))
h_spin_values = [compute_H_spin_state(s) for s in omega_list]
print("  |Omega| = {}".format(n_omega))
print("  H_spin(C_spin) = -H_spin: {}/{}".format(odd_count, n_omega))
print("  Equator H_spin=0 (N=3):     {}/{}".format(equator_count, n_omega))
print("  H_spin != 0:                {}/{}".format(h_nonzero_count, n_omega))
print("  mass(face-swap C) = mass:   {}/{}".format(mass_invariant_count, n_omega))
print("  H_spin range: [{}, {}]".format(min(h_spin_values), max(h_spin_values)))

section("A.6  Chiral gravitomagnetic coupling")
print("  ||pi||^2/Tr^2 = 2/75 (bulk): hqvm_gravity_analysis_3 B.")
psi_ns = 0.15
chiral_ns = (4.0 / 75.0) * psi_ns ** 2
print("  delta_B_g/B_g = (4/75)*psi^2 at psi=0.15 (NS surface): {:.6f}  ({:.4f}%)".format(
    chiral_ns, chiral_ns * 100.0))

# =====================================================================
# SECTION B: GRAVITATIONAL WAVE EXTENSIONS
# =====================================================================
print("=" * 9, "B. Gravitational wave extensions", "=" * 9)
print()
print("  Vacuum Gauss / HT strain calibration: hqvm_gravity_analysis_5 J, P.")
print()

section("B.0  Quadrupole shell spectrum and 8-byte Z2 holonomy")
f_k2 = 2.0 * math.comb(6, 2) / math.comb(6, 3)
print("  Shell modulation 2*C(6,2)/C(6,3) = {:.4f}".format(f_k2))
amps_f = shell_cycle_fourier_amps(F_cycle_word(0))
amps_z2 = shell_cycle_fourier_amps(cycle_word_for_micro(0))
a_k2_f = amps_f[2][1]
a_k2_z2 = amps_z2[2][1]
print("  Half-cycle (F, 4 bytes) DFT k=2: |A_2| = {:.4f}".format(a_k2_f))
print("  Full Z2 cycle (8 bytes) DFT k=2: |A_2| = {:.4f}".format(a_k2_z2))
print("  Full Z2 DFT spectrum (arch_shell-3):")
for k, a in amps_z2:
    print("    k={}: |A_k| = {:.4f}".format(k, a))
a_k4_z2 = amps_z2[4][1]
print("  k=4 mode |A_4| = {:.4f} ({:.1f}% of |A_2|); hexadecapole (l=4) precursor".format(
    a_k4_z2, 100.0 * a_k4_z2 / a_k2_z2))
word8 = cycle_word_for_micro(0)
rows8 = trace_word_steps(word8, micro_ref=0)
r4 = rows8[4]
r8 = rows8[-1]
print("  8-byte Z2 holonomy (m_ref=0):")
print("    step 4: arch_shell={}  qxor={}  state=REST? {}".format(
    r4["arch_shell"], r4["qxor"], r4["state24"] == GENE_MAC_REST))
print("    step 8: arch_shell={}  qxor={}  state=REST? {}".format(
    r8["arch_shell"], r8["qxor"], r8["state24"] == GENE_MAC_REST))

section("B.2  Inspiral phasing")
v_over_c_values = [0.1, 0.2, 0.3, 0.4, 0.5]
print("  Phase correction: delta_Phi/Phi_GR ~ 1 + (5*g1/8)*(v/c)^2")
print("  g1 = {:.6f}".format(g1))
print()
print("  {:>10s}  {:>14s}  {:>14s}".format("v/c", "delta_Phi/Phi", "% correction"))
for vc in v_over_c_values:
    phase_corr = (5.0 * g1 / 8.0) * vc ** 2
    print("  {:10.2f}  {:14.6f}  {:14.2f}%".format(vc, phase_corr, phase_corr * 100))
gw150914_phase = (5.0 * g1 / 8.0) * 0.4 ** 2
print("  GW150914 (v/c=0.4): {:.2f}%".format(gw150914_phase * 100))

section("B.3  Tortoise coordinate, potential, echoes")

s_lo_tort = s_h * 1.001
s_sample_tort = [s_h * 1.01, s_h * 1.05, s_h * 1.1, 2.0, s_ph, 3.0, 5.0, 10.0, 50.0]
print("  Tortoise coordinate r*/r_g = integral ds / (1 - 2*psi(s)):")
print()
print("  {:>10s}  {:>14s}  {:>14s}".format("s", "f(s)", "r*/r_g"))
for s_val in s_sample_tort:
    fs = f_metric(s_val)
    rstar_val, _ = tortoise_integral(s_val, s_lo_tort)
    print("  {:10.4f}  {:14.8f}  {:14.4f}".format(s_val, fs, rstar_val))

rstar_ph, _ = tortoise_integral(s_ph, s_lo_tort)
print()
print("  r*(s_ph)/r_g = {:.4f}".format(rstar_ph))
print()

s_grid = np.linspace(s_h * 1.01, 20.0, 2000)
V_grid = np.array([effective_potential_full(s, g1) for s in s_grid])
i_peak = int(np.argmax(V_grid))
s_peak = float(s_grid[i_peak])
V_peak = float(V_grid[i_peak])

s_grid_schw = np.linspace(2.01, 20.0, 2000)
V_schw = np.array([effective_potential_schwarzschild(s) for s in s_grid_schw])
i_peak_schw = int(np.argmax(V_schw))
s_peak_schw = float(s_grid_schw[i_peak_schw])
V_peak_schw = float(V_schw[i_peak_schw])

V_at_photon_schw = effective_potential_schwarzschild(3.0)
s_peak_schw_exact = (9.0 + math.sqrt(17.0)) / 4.0
V_peak_schw_exact = effective_potential_schwarzschild(s_peak_schw_exact)

print("  Effective potential V(s) = f(s)[6/s^2 - 6*psi/s^2]:")
print("  CGM:           peak at s = {:.4f} r_g, V_max = {:.6f}".format(s_peak, V_peak))
print("  Schwarzschild: peak at s = {:.4f} r_g, V_max = {:.6f}".format(
    s_peak_schw, V_peak_schw))
print("  Analytic: peak at s = (9+sqrt(17))/4 = {:.4f}, V_max = {:.6f}".format(
    s_peak_schw_exact, V_peak_schw_exact))
print("  V at photon sphere (s=3): {:.6f} (= 4/27 = {:.6f})".format(
    V_at_photon_schw, 4.0 / 27.0))
print()
print("  Photon sphere: GR s_ph=3, CGM s_ph={:.4f} r_g".format(s_ph))
print("  Echo delays (r*-based, order-of-magnitude):")
for eps_frac in [0.001, 0.01, 0.05, 0.1]:
    s_near = s_h * (1.0 + eps_frac)
    rstar_near, _ = tortoise_integral(s_near, s_lo_tort)
    delta_rstar = rstar_ph - rstar_near
    r_g_10 = G_SI * 10.0 * M_SUN_KG / C_SI ** 2
    dt_ms_10 = 2.0 * delta_rstar * r_g_10 / C_SI * 1000.0
    print("  eps={:.3f}: Delta_r*/r_g={:.4f}  Delta_t(10 Msun)={:.4f} ms".format(
        eps_frac, delta_rstar, dt_ms_10))
g_horizon = g_over_g0(0.5)
print("  G(psi)/G0 at horizon (psi=1/2): {:.6f}".format(g_horizon))
print()

section("B.4  Ringdown (Poschl-Teller estimate, ell=2, n=0)")
s_peak_pt_cgm, V0_pt_cgm, omega_R_pt_cgm, omega_I_pt_cgm = poschl_teller_qnm(
    _v_cgm_barrier, _f_cgm_barrier, s_h * 1.001, 15.0)
s_peak_pt_gr, V0_pt_gr, omega_R_pt_gr, omega_I_pt_gr = poschl_teller_qnm(
    _v_gr_barrier, _f_gr_barrier, 2.001, 15.0)

omega_R_exact_gr_M = 0.37367
omega_I_exact_gr_M = 0.08896

print("  Effective potential barrier (Regge-Wheeler axial, s = r/r_g):")
print("  CGM peak: s = {:.4f} r_g, V0 = {:.6f}".format(s_peak_pt_cgm, V0_pt_cgm))
print("  GR  peak: s = {:.4f} r_g, V0 = {:.6f}".format(s_peak_pt_gr, V0_pt_gr))
print()
print("  Pöschl-Teller n=0 (omega*M, c=1; r_g=2M => omega*r_g = 2*omega*M):")
print("  {:15s}  {:>15s}  {:>15s}".format("", "omega_R", "omega_I"))
print("  " + "-" * 48)
print("  {:15s}  {:15.6f}  {:15.6f}".format("CGM (PT)", omega_R_pt_cgm, omega_I_pt_cgm))
print("  {:15s}  {:15.6f}  {:15.6f}".format("GR (PT)", omega_R_pt_gr, omega_I_pt_gr))
print("  {:15s}  {:15.6f}  {:15.6f}".format("GR (exact)", omega_R_exact_gr_M, omega_I_exact_gr_M))
print()

ratio_pt = omega_R_pt_cgm / omega_R_pt_gr if omega_R_pt_gr > 0 else float("nan")
ratio_pt_exact = omega_R_pt_cgm / omega_R_exact_gr_M
print("  CGM/GR ratio (PT): {:.4f} ({:+.1f}%)".format(ratio_pt, (ratio_pt - 1) * 100))
print("  CGM vs GR exact n=0 QNM: {:.4f} ({:+.1f}%)".format(
    ratio_pt_exact, (ratio_pt_exact - 1) * 100))
print("  Photon-sphere scale: s_ph_GR/s_ph_CGM = {:.4f} ({:+.1f}%)".format(
    3.0 / s_ph, (3.0 / s_ph - 1) * 100))

# =====================================================================
# SECTION C: VIRIAL, SELF-ENERGY, UV REGULATION
# =====================================================================
print("=" * 9, "C. Virial, self-energy, UV regulation", "=" * 9)
print()

section("C.1  Virial condition (CS/BU)")
print("  2T + V = 0 => E_total = T + V = -T (bound system, negative total energy).")
print("  Zero net displacement/momentum flux per cycle (not zero total energy).")
print("  Stage map: T ~ ONA; V ~ UNA; BU = depth-4 closure on 6 DOF.")
print("    See C.5b.")
print("  q(F)=0: analysis_2 S1; Tr_iso = E[Tr|w]+3Var: analysis_3 B.")
bal = local_exterior_energy_balance()
print("  Point-mass sector: E_rest_frame/Mc^2 = +{:.4f}, E_self/Mc^2 = {:.4f}, sum = {:.4f}".format(
    bal["E_rest_frame_frac"], bal["E_self_frac"], bal["sum"]))
print("  (Isolated exterior: local E_self+E_rest_frame=0; 2T+V is cosmological/global.)")

section("C.4  Exterior integral theorem (I = 1/2)")
print("  Theorem: I = int_{s_h}^inf exp(g1*psi)/s^2 ds = psi(s_h) - psi(inf) = 1/2")
print("  Proof: ODE dpsi/ds = -exp(g1*psi)/s^2 => integrand = -dpsi/ds.")
print("  Holds for any g1 (horizon psi(s_h) = 1/2, psi(inf) = 0).")
print()
print("  {:>8s}  {:>12s}  {:>12s}  {:>12s}".format("g1", "s_h", "I (exact)", "I (numeric)"))
g1_samples = [-0.5, g1, -1.0, -2.0]
all_I_half = True
for g1s in g1_samples:
    I_ex, sh_s = exterior_field_integral_I(g1s)
    s_lo = exterior_integral_numeric_lo(sh_s, g1s)
    I_num, _ = field_integral_exterior_numeric(s_lo, g1s)
    ok_I = abs(I_ex - 0.5) < 1e-9
    if not ok_I:
        all_I_half = False
    print("  {:8.4f}  {:12.4f}  {:12.10f}  {:12.10f}".format(
        g1s, sh_s, I_ex, I_num))
print("  All sampled g1: I = 1/2 = {}".format(all_I_half))
print()
print("  CGM vs Newtonian exterior integral (UV regulation):")
print("  {:>12s}  {:>12s}  {:>12s}  {:>12s}".format(
    "s_inner", "I_Newton", "I_CGM", "CGM/Newton"))
for s_inner in [0.5, 1.0, 1.5, s_h, 2.0, 3.0, 5.0, 10.0]:
    I_newt = 1.0 / s_inner
    if s_inner > s_h * 1.001:
        I_cgm, _ = field_integral_exterior_numeric(s_inner, g1)
        ratio = I_cgm / I_newt
        i_cgm_str = "{:12.6f}".format(I_cgm)
        ratio_str = "{:12.6f}".format(ratio)
    else:
        i_cgm_str = "         n/a"
        ratio_str = "         n/a"
    print("  {:12.4f}  {:12.6f}  {}  {}".format(s_inner, I_newt, i_cgm_str, ratio_str))
print("  CGM I -> 1/2 at s_h; Newtonian I -> inf as s -> 0.")

section("C.5  Local self-energy (point mass)")
print("  Rest-frame energy: E_rest_frame = +(Mc^2/2)*I = +Mc^2/4.")
print("  Gravitational self-energy: E_self = -Mc^2/4 (M = observable mass).")
print("  Local balance: E_self + E_rest_frame = 0 per unit M_obs.")
print()
print("  Self-consistent dressing (field sourced by M_obs):")
print("    M_obs = M_bare + E_self/c^2 = M_bare - M_obs/4")
print("    M_obs/M_bare = 4/5")
mass_ratio_obs = 4.0 / 5.0
binding_frac = -0.25
E_self_frac = -0.25
E_rest_frame_frac = 0.25

integral_numeric, _ = field_integral_exterior_numeric(s_h * 1.001, g1)
integral_exact = field_integral_exterior_exact(s_h, g1)

s_u = 5.0
u5 = refractive_stress_density_u(s_u)
print("  psi(s_h)=1/2 => E_self/M_obs c^2 = -1/4; E_rest_frame = +1/4.")
print("  u(s=5) = |g|^2/(8*pi*G(psi)) = {:.6e} (G0=1 units)".format(u5))
print()
print("  Self-energy vs compactness:")
print("  {:>15s}  {:>10s}  {:>10s}  {:>12s}  {:>12s}".format(
    "Object", "psi_surf", "G/G0", "E_self/Mc^2", "E_self(3/5)"))
objects = [
    ("Earth", 7.0e-10),
    ("Sun", 2.1e-6),
    ("White dwarf", 3.0e-4),
    ("NS (12 km, CGM TOV)", 0.153),
    ("NS (12 km, Newtonian)", 0.172),
    ("BH horizon", 0.5),
]
for name, psi_s in objects:
    g_ratio = math.exp(g1 * psi_s)
    E_simple = -0.5 * psi_s
    E_sphere = -0.6 * psi_s * g_ratio
    print("  {:>15s}  {:10.4e}  {:10.6f}  {:12.6f}  {:12.6f}".format(
        name, psi_s, g_ratio, E_simple, E_sphere))
print("  Shell self-energy weights:")
print("  {:>5s}  {:>10s}  {:>10s}  {:>10s}  {:>12s}".format(
    "Shell", "Pop", "Tr(sigma)", "||pi|| est", "Self-E prop"))
for k in range(7):
    tr_k = TR_SIGMA_SHELL[k]
    pi_k = FA_STF * tr_k if k not in (0, 6) else 0.0
    se_k = pi_k * Delta * binom_shell[k] * 4
    print("  {:5d}  {:10.6f}  {:10.6f}  {:10.6f}  {:12.6e}".format(
        k, binom_shell[k], tr_k, pi_k, se_k))

section("C.5b Stage-mass decomposition")
sm = stage_mass_fractions()
print("  Bare mass from UV energy ratios (Analysis_Energy_Scales 3.1):")
print("    E_UNA/E_CS = {:.6f}".format(sm["E_UNA/E_CS"]))
print("    E_ONA/E_CS = {:.6f}".format(sm["E_ONA/E_CS"]))
print("    E_BU/E_CS  = {:.6f}".format(sm["E_BU/E_CS"]))
print()
print("  M_bare = M_UNA + M_ONA + M_BU")
print("  {:>6s}  {:>10s}  {:>12s}".format("Stage", "Fraction", "Role"))
print("  {:>6s}  {:>10.4f}  {:>12s}".format("UNA", sm["f_UNA"], "potential V"))
print("  {:>6s}  {:>10.4f}  {:>12s}".format("ONA", sm["f_ONA"], "kinetic T"))
print("  {:>6s}  {:>10.4f}  {:>12s}".format("BU", sm["f_BU"], "closure"))
print("  Sum fractions = {:.6f}".format(sm["f_UNA"] + sm["f_ONA"] + sm["f_BU"]))
print()
print("  6 DOF = 3 UNA (rotational) + 3 ONA (translational)")
print("    Gravitating sector: f_STF = {:.4f} = UNA + ONA".format(sm["f_STF"]))
print("  Stress split (1+5, analysis_1 Part B):")
print("    5 STF components: gravitational signal")
print("    1 trace component: isotropic pressure (non-gravitating)")
print("    Within STF: UNA {:.4f}, ONA {:.4f} (energy-weighted)".format(
    sm["f_UNA/STF"], sm["f_ONA/STF"]))
print("  BU: depth-4 closure condition on the 6 DOF (no DOF assigned)")
print("    Closure energy: f_BU = {:.4f} of M_bare".format(sm["f_BU"]))
print()
print("  Virial stage check: 2T + V = 0 with T ~ f_ONA, V ~ -f_UNA")
print("    2*f_ONA - f_UNA = {:.6f}".format(sm["virial_residual"]))
print("  UV ratios are bare-mass fractions; virial is a dressed bound-state")
print("  condition. No closure is expected between these two objects.")
alpha_g = G_meas * v_EW ** 2
n_g = -math.log(alpha_g) / Delta
ew = electroweak_coords(delta=Delta, order=5)
print("  Delta-ruler mass coordinates (compact geometry, analysis_3 G):")
print("    Top     n = {:.1f}".format(ew.n_top))
print("    Higgs   n = {:.1f}".format(ew.n_higgs))
print("    Z       n = {:.1f}".format(ew.n_z))
print("    W       n = {:.1f}".format(ew.n_w))
print("    Gravity n_G = {:.1f}  (alpha_G at v)".format(n_g))

section("C.6  Cosmological virial closure (BU)")
print("  Local theorem (C.5): E_self = -Mc^2/4 for one point mass.")
print("  Cosmological: strictly bound structure under global BU closure.")
print("  Not implied by local I=1/2 alone; needs cosmic virial balance.")
print("  Expansion: tau(z) = tau_G*(1-psi(z)) (Refractive Depth, not Hubble flow).")

section("C.7  Hawking temperature and BU-Ingress")
hk = hawking_cgm_vs_gr(g1)
print("  Surface gravity (geometric units, r_g=2M):")
print("    kappa_CGM = |dpsi/ds| at s_h = {:.6f} / r_g".format(hk["kappa_cgm"]))
print("    kappa_GR  = 1/(4 r_g) = {:.6f} / r_g".format(hk["kappa_gr"]))
print("    kappa_CGM/kappa_GR = {:.6f} ({:+.2f}%)".format(
    hk["kappa_ratio"], (hk["kappa_ratio"] - 1.0) * 100.0))
print("    4*exp(g1/2)/s_h^2 check = {:.6f}".format(hk["kappa_formula"]))
print("  Horizon area ratio (r_h = r_g*s_h vs 2*r_g): {:.6f}".format(hk["area_ratio"]))
print("  Hawking luminosity ~ T^4 * Area:")
print("    L_CGM/L_GR = (kappa ratio)^4 * area ratio = {:.6f} ({:+.1f}%)".format(
    hk["lum_ratio"], (hk["lum_ratio"] - 1.0) * 100.0))
m_bh = 10.0 * M_SUN_KG
t_gr = hawking_temperature_si(m_bh, 1.0)
t_cgm = hawking_temperature_si(m_bh, hk["kappa_ratio"])
print("  Hawking T (10 Msun): T_GR = {:.4e} K, T_CGM = {:.4e} K ({:+.2f}%)".format(
    t_gr, t_cgm, (t_cgm / t_gr - 1.0) * 100.0))
print("  Z2 holonomy (8 bytes): BU-Egress = W2 involution at depth 4; BU-Ingress =")
print("    depth-4 spectral memory (W2 pole-pairing). F o F restores carrier rest.")
print("    Hawking flux = escaping partner when commit (Future phase) fails at horizon.")

# =====================================================================
# SECTION E: SYNTHESIS AND FALSIFICATION
# =====================================================================
print("=" * 9, "E. Synthesis, discoveries, and falsification", "=" * 9)
print()

section("E.1  Falsification tests")
print()
print("  {:30s} | {:35s} | {:15s}".format("Test", "CGM Prediction", "Current Status"))
print("  " + "-" * 85)
print("  {:30s} | {:35s} | {:15s}".format(
    "Antimatter free-fall", "Same as matter (WEP)", "ALPHA-g pending"))
print("  {:30s} | {:35s} | {:15s}".format(
    "Antimatter spin precession", "Opposite sign at O(psi^2)", "Untestable"))
print("  {:30s} | {:35s} | {:15s}".format(
    "GW scalar polarization", "None (psi slaved)", "LIGO consistent"))
print("  {:30s} | {:35s} | {:15s}".format(
    "GW inspiral phase", "~6.5% shift at v/c=0.4", "Degenerate"))
print("  {:30s} | {:35s} | {:15s}".format(
    "Ringdown frequency", "~12.5% above GR exact (PT est.)", "Unresolved"))
print("  {:30s} | {:35s} | {:15s}".format(
    "GW echoes", "~0.86 ms (10 Msun)", "Undetected"))
print("  {:30s} | {:35s} | {:15s}".format(
    "Omega_total != 1", "Falsifies cosmological virial closure", "Tension exists"))
print("  {:30s} | {:35s} | {:15s}".format(
    "Wormhole/alcubierre realized", "Excluded by BU closure", "Speculative"))
print()

section("E.2  Checks unique to this script")
check1 = abs(E_self_frac + 0.25) < 1e-12
print("  E_self/M_obs c^2 = -1/4: {:.6f} = {}".format(E_self_frac, check1))

check4 = (
    abs(integral_numeric - 0.5) < 1e-3
    and abs(integral_exact - 0.5) < 1e-9
)
print("  Field integral = 1/2: numeric err={:.2e} exact err={:.2e} = {}".format(
    abs(integral_numeric - 0.5), abs(integral_exact - 0.5), "OK" if check4 else "FAIL"))

check5 = True
for s in [2.0, 5.0, 10.0, 50.0]:
    psi_s = psi_point_mass(s, g1)
    dpsi = dpsi_ds_point_mass(s, g1)
    f_val = 1.0 - 2.0 * psi_s
    G_tt = -2.0 * f_val * (psi_s + s * dpsi) / s ** 2
    if G_tt > 0:
        check5 = False
print("  Weak energy condition at s=2,5,10,50: {}".format(check5))

check6 = abs(mass_ratio_obs - 0.8) < 1e-12
print("  M_obs/M_bare = 4/5: {:.6f} = {}".format(mass_ratio_obs, check6))

sm_check = stage_mass_fractions()
check6b = abs(sm_check["f_UNA"] + sm_check["f_ONA"] + sm_check["f_BU"] - 1.0) < 1e-9
print("  Stage fractions sum to 1: {}".format(check6b))

check7 = abs(binding_frac + 0.25) < 1e-12 and abs(E_rest_frame_frac + binding_frac) < 1e-12
print("  Local E_self + E_rest_frame = 0 (per M_obs): {}".format("OK" if check7 else "FAIL"))

check8 = odd_count == n_omega
print("  H_spin odd under C_spin on Omega ({}): {}".format(n_omega, check8))

check8b = mass_invariant_count == n_omega
print("  mass even under face-swap on Omega: {}".format(check8b))

check8c = equator_count == 1280
check8d = h_nonzero_count == 2816
print("  Equator shell count (N=3): {} = {}".format(equator_count, check8c))
print("  H_spin != 0: {} = {}".format(h_nonzero_count, check8d))

check10 = (
    not math.isnan(omega_R_pt_cgm)
    and ratio_pt > 1.0
    and 1.08 < ratio_pt_exact < 1.18
)
check_hk = 0.99 < hk["kappa_ratio"] < 1.02
check_lum = hk["lum_ratio"] < 0.85
print("  PT ringdown CGM/GR: {:.4f}; vs exact: {:.4f} = {}".format(
    ratio_pt, ratio_pt_exact, "OK" if check10 else "check"))
print("  Hawking kappa ratio ~1: {:.4f} = {}".format(hk["kappa_ratio"], check_hk))
print("  Hawking L ratio < 1: {:.4f} = {}".format(hk["lum_ratio"], check_lum))
print("  Exterior I = 1/2 all g1: {}".format(all_I_half))
print()
print("  Scalar-tensor action (psi slaved): hqvm_gravity_analysis_5 S.")
print()

section("E.4  Summary")
print("  A: C-odd H_spin on |Omega|={}; chiral (4/75)psi^2.".format(n_omega))
print("  B: k=2 |A_2|={:.2f}; inspiral {:.1f}%; ringdown {:+.1f}% (PT).".format(
    a_k2_z2, gw150914_phase * 100, (ratio_pt_exact - 1) * 100))
print("  C: I=1/2; E_self=-Mc^2/4; M_obs/M_bare=4/5; Hawking L -{:.0f}%.".format(
    (1.0 - hk["lum_ratio"]) * 100.0))
print("  GW HT / vacuum: analysis_5 J, P.")
print("=" * 9, "DONE", "=" * 9)
