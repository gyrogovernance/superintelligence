"""
Shared CGM gravity kernel helpers and invariants.

Used by hqvm_gravity_analysis_{1,2,3,4,5,ext}.py.

Convention: arch_shell = 6 - chi_shell, measuring distance from the complement
horizon (0 = complement horizon, 6 = equality horizon).

E_CS is the Planck-scale UV anchor (E_CS = E_Planck = 1.22e19 GeV; see
Analysis_Energy_Scales). Planck units presuppose G, so G = G_kernel / E_CS^2
must not be used to derive G. E_CS may appear only in consistency checks
(e.g. conjugacy depth 2 ln(E_CS/v)). The forward prediction is
G_pred = G_kernel * exp(-tau_G) / v^2 with kernel tau_G; tau_required from
G_meas is validation only.
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from math import comb, exp, log, sqrt
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import lambertw

from gyroscopic.hQVM.constants import (
    GENE_MAC_A12,
    GENE_MAC_B12,
    GENE_MAC_REST,
    GENE_MIC_S,
    CHIRALITY_MASK_6,
    step_state_by_byte,
    byte_to_intron,
    intron_family,
    intron_micro_ref,
    is_on_horizon,
    is_on_equality_horizon,
)
from gyroscopic.hQVM.api import (
    chirality_word6,
    q_word6,
    q_word6_for_items,
    state24_to_omega12,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

TR_SIGMA_SHELL = [0, 0.416667, 0.666667, 0.75, 0.666667, 0.416667, 0]
FA_STF = math.sqrt(6) / 9
CHI6_FULL = CHIRALITY_MASK_6
FAMILY_RAY_REF = 1
GENE_MAC_SWAPPED = (GENE_MAC_B12 << 12) | GENE_MAC_A12

Q_G = 4 * np.pi
m_a = 1 / (2 * np.sqrt(2 * np.pi))
d_BU = 0.195342176580
rho = d_BU / m_a
Delta = 1 - rho
G_kernel = math.pi / 6
Omega_size = 4096
H_size = 64
W2_SHELL_DISPLACEMENT = 6  # per W2 depth-4 half-word (2 bytes); wavefunction_2 T2,T5
F_CYCLE_PATH_TRAVERSE = 12  # F-cycle path length (4 bytes); F preserves shell per T4
Z2_HOLONOMY_PATH_TRAVERSE = 24  # Z2 holonomy path length (8 bytes, 2 F-cycles); net disp 0
D_traverse = Z2_HOLONOMY_PATH_TRAVERSE  # shell traverse invariant (not aperture Delta)
AF = 2 * Z2_HOLONOMY_PATH_TRAVERSE  # Z2 double-cover: holonomy cycle path x 2 for Refractive Depth round-trip (T6: F o F = id)
v_EW = 246.22
G_meas = 6.708810e-39

# E_CS = E_Planck (Analysis_Energy_Scales 3.2). Not for deriving G (circular).
E_CS = 1.22e19

# Stage UV energy ratios (Analysis_Energy_Scales 3.1; dimensionless).
E_UNA_OVER_CS = 2.0 / (math.pi * math.sqrt(2))
E_ONA_OVER_CS = 0.5
E_BU_OVER_CS = (2.0 * m_a * m_a) / math.pi


def stage_mass_fractions() -> dict[str, float]:
    """Bare-mass stage fractions from UV energy ratios (UNA + ONA + BU)."""
    e_sum = E_UNA_OVER_CS + E_ONA_OVER_CS + E_BU_OVER_CS
    f_una = E_UNA_OVER_CS / e_sum
    f_ona = E_ONA_OVER_CS / e_sum
    f_bu = E_BU_OVER_CS / e_sum
    stf = f_una + f_ona
    return {
        "f_UNA": f_una,
        "f_ONA": f_ona,
        "f_BU": f_bu,
        "E_UNA/E_CS": E_UNA_OVER_CS,
        "E_ONA/E_CS": E_ONA_OVER_CS,
        "E_BU/E_CS": E_BU_OVER_CS,
        "f_STF": stf,
        "f_UNA/STF": f_una / stf,
        "f_ONA/STF": f_ona / stf,
        "virial_residual": 2.0 * f_ona - f_una,
    }


alpha_G_meas = G_meas * v_EW**2
f_ordered = 1.0 - 4.0 * rho * Delta**2

# Validation only: inverts measured G through alpha_G(v) = G_kernel * exp(-tau).
tau_required = -math.log(alpha_G_meas / G_kernel)
tau_req_meas = tau_required

# Consistency check: Z2 Refractive Depth vs UV-IR conjugacy ladder (not a G derivation).
tau_conjugacy_depth = 2.0 * math.log(E_CS / v_EW)
tau_G_formula = Omega_size * Delta * rho**5 * f_ordered
binom_shell = [comb(6, s) / 64.0 for s in range(7)]
weights_pop = {m: binom_shell[bin(m).count("1")] for m in range(64)}
pi_eq = FA_STF * TR_SIGMA_SHELL[3]
V_EW_PDG = (246.22, 0.01)
C4_REF = -1.75

K4_CHANNEL_FLAGS = [
    ("top", 0, 0, 0),
    ("higgs", 1, 0, 0),
    ("z", 1, 1, 0),
    ("w", 1, 1, 1),
]


def configure_stdout_utf8():
    fn = getattr(sys.stdout, "reconfigure", None)
    if callable(fn):
        try:
            fn(encoding="utf-8", errors="replace")
        except Exception:
            pass


def byte_from_family_and_micro(family: int, micro_ref: int) -> int:
    family &= 0x03
    micro_ref &= 0x3F
    bit7 = (family >> 1) & 1
    bit0 = family & 1
    intron = (bit7 << 7) | (micro_ref << 1) | bit0
    return intron ^ GENE_MIC_S


def family_word_for_micro(micro_ref: int) -> list[int]:
    return [byte_from_family_and_micro(fam, micro_ref) for fam in range(4)]


def W2_word(micro_ref: int) -> list[int]:
    """Depth-4 half-word (families 00,01). W2 is an involution mapping shell s -> 6-s (T2)."""
    return [
        byte_from_family_and_micro(0, micro_ref),
        byte_from_family_and_micro(1, micro_ref),
    ]


def W2p_word(micro_ref: int) -> list[int]:
    """Depth-4 half-word (families 10,11). W2' is an involution mapping shell s -> 6-s (T3)."""
    return [
        byte_from_family_and_micro(2, micro_ref),
        byte_from_family_and_micro(3, micro_ref),
    ]


def F_cycle_word(micro_ref: int) -> list[int]:
    """One F-cycle: W2 o W2' (T6). 4 bytes; gate F on Omega. Z2 carrier flip; net shell displacement = 0 (T4)."""
    return family_word_for_micro(micro_ref)


def cycle_word_for_micro(micro_ref: int) -> list[int]:
    """Z2 holonomy cycle word: F o F = id (carrier Z2 round-trip). 8 bytes; two F-cycles completing Z2 holonomy. Carrier returns to rest, NOT to CS."""
    return family_word_for_micro(micro_ref) * 2


def reversed_cycle_word(micro_ref: int) -> list[int]:
    """Chirality-reversed Z2 holonomy: families 3,2,1,0 per F-cycle, twice."""
    word: list[int] = []
    for _ in range(2):
        for fam in (3, 2, 1, 0):
            word.append(byte_from_family_and_micro(fam, micro_ref))
    return word


def W2p_then_W2_cycle_word(micro_ref: int) -> list[int]:
    """Two-pass holonomy probe: W2' then W2, repeated (GW echo / shell order)."""
    half = W2p_word(micro_ref) + W2_word(micro_ref)
    return half + half


def arch_path_displacement(word: list[int], micro_ref: int = 0) -> int:
    """Sum of |Delta arch_shell| along a byte word (Z2 path length = 24)."""
    rows = trace_word_steps(word, micro_ref=micro_ref)
    return sum(
        abs(rows[i]["arch_shell"] - rows[i - 1]["arch_shell"])
        for i in range(1, len(rows))
    )


def holonomy_qxor_final(word: list[int], micro_ref: int = 0) -> int:
    """Accumulated chi6 XOR along word; zero <=> net chirality closure."""
    return trace_word_steps(word, micro_ref=micro_ref)[-1]["qxor"]


def apply_word_to_state(word: list[int], state24: int = GENE_MAC_REST) -> int:
    s = state24
    for b in word:
        s = step_state_by_byte(s, b)
    return s


def _shell_fields(state24: int) -> tuple[int, int]:
    chi_shell = state24_to_omega12(state24).shell
    return chi_shell, 6 - chi_shell


def trace_word_steps(
    word: list[int],
    start_state24: int = GENE_MAC_REST,
    micro_ref: int = 0,
) -> list[dict]:
    s = int(start_state24) & 0xFFFFFF
    q_acc = 0
    chi0, arch0 = _shell_fields(s)
    rows = [{
        "step": 0,
        "byte": None,
        "state24": s,
        "shell": chi0,
        "arch_shell": arch0,
        "chi6": chirality_word6(s),
        "qxor": 0,
        "family": 0,
        "micro": micro_ref & 0x3F,
        "intron": None,
        "on_horizon": is_on_horizon(s),
        "on_equality_horizon": is_on_equality_horizon(s),
    }]
    for step, byte in enumerate(word, start=1):
        s = step_state_by_byte(s, byte)
        q_acc = (q_acc ^ q_word6(byte)) & CHI6_FULL
        intron = byte_to_intron(byte)
        chi_sh, arch_sh = _shell_fields(s)
        rows.append({
            "step": step,
            "byte": byte,
            "state24": s,
            "shell": chi_sh,
            "arch_shell": arch_sh,
            "chi6": chirality_word6(s),
            "qxor": q_acc,
            "family": intron_family(intron),
            "micro": intron_micro_ref(intron),
            "intron": intron,
            "on_horizon": is_on_horizon(s),
            "on_equality_horizon": is_on_equality_horizon(s),
        })
    return rows


def poly_mul(a, b):
    out = np.zeros(len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, aj in enumerate(b):
            out[i + j] += ai * aj
    return out


def pi_norm_shell(shell_k):
    if shell_k in (0, 6) or shell_k < 0 or shell_k > 6:
        return 0.0
    return FA_STF * TR_SIGMA_SHELL[shell_k]


def build_joint_table() -> list[dict]:
    """Joint step table for one Z2 holonomy cycle (8 steps per micro_ref)."""
    table = []
    for m_ref in range(64):
        pop_m = bin(m_ref).count("1")
        weight = comb(6, pop_m) / 64.0
        word = cycle_word_for_micro(m_ref)
        for row in trace_word_steps(word, micro_ref=m_ref)[1:]:
            table.append({
                "m_ref": m_ref,
                "pop": pop_m,
                "weight": weight,
                "step": row["step"],
                "byte": row["byte"],
                "state24": row["state24"],
                "arch_shell": int(row["arch_shell"]),
                "intron": row["intron"],
                "family": row["family"],
                "micro": row["micro"],
                "qxor": row["qxor"],
                "chi6": row["chi6"],
            })
    return table


def kappa_pi_step(shell_k, chi6, chi6_full=CHI6_FULL):
    if shell_k in (0, 6) or shell_k < 0 or shell_k > 6:
        return 0.0
    pi_k = pi_norm_shell(shell_k)
    return Delta * (pi_k / pi_eq) if pi_eq > 0 else 0.0


def kappa_binom_step(shell_k, chi6, chi6_full=CHI6_FULL):
    if shell_k < 0 or shell_k > 6:
        return 0.0
    if shell_k in (0, 6):
        return 0.0
    return Delta * binom_shell[shell_k]


def tau_path_kappa(micro, shell_w=None):
    if shell_w is None:
        shell_w = binom_shell
    tau_path = 0.0
    for row in trace_word_steps(cycle_word_for_micro(micro))[1:]:
        s = row["arch_shell"]
        if s < 0 or s > 6:
            continue
        if s in (0, 6):
            continue
        tau_path += Delta * shell_w[s]
    return tau_path


def tau_path_binom(micro, chi6_full=CHI6_FULL):
    tau = 0.0
    for row in trace_word_steps(cycle_word_for_micro(micro))[1:]:
        tau += kappa_binom_step(row["arch_shell"], row["chi6"], chi6_full)
    return tau


def tau_cycle_weighted(micro_weights, shell_w=None):
    if shell_w is None:
        shell_w = binom_shell
    tau_sum = 0.0
    w_sum = 0.0
    for micro in range(64):
        w = micro_weights.get(micro, 0.0)
        if w <= 0:
            continue
        tau_sum += tau_path_kappa(micro, shell_w) * w
        w_sum += w
    return tau_sum / w_sum if w_sum > 0 else 0.0


def tau_cycle_per_delta_exact() -> Fraction:
    """tau_cycle/Delta from bulk shell transport (primary: analysis_3 section C)."""
    sum_cubes = sum(comb(6, k) ** 3 for k in range(1, 6))
    sum_sq = sum(comb(6, k) ** 2 for k in range(7))
    return Fraction(4 * sum_cubes, 64 * sum_sq)


def tau_g_with_c4(c4_val):
    f_ext = 1.0 - 4.0 * rho * Delta**2 + c4_val * Delta**4
    return Omega_size * Delta * rho**5 * f_ext


def kernel_exposure_constants() -> tuple[float, float, float, Fraction]:
    """
    N_cycles, tau_cycle, tau_G (full), tau_cycle/Delta from analysis_3 section D.

    Single source for exposure-count factorization (no two-lemma 3481 formula).
    """
    tau_over_delta = tau_cycle_per_delta_exact()
    tau_cycle = float(tau_over_delta) * Delta
    f_k4_full = f_ordered + float(C4_REF) * Delta**4
    n_cycles = Omega_size * rho**5 * f_k4_full / float(tau_over_delta)
    tau_g_full = tau_g_with_c4(C4_REF)
    return n_cycles, tau_cycle, tau_g_full, tau_over_delta


def dln_g_dpsi(tau_g_full: float | None = None) -> float:
    """Slope d ln(G/G0) / d psi for G(psi) = G0 exp(g1 psi)."""
    tau = tau_g_full if tau_g_full is not None else tau_g_with_c4(C4_REF)
    eta = math.log(v_EW / E_CS)
    return tau + 2.0 * eta


def psi_analytic(s: float | np.ndarray, g1: float | None = None) -> float | np.ndarray:
    """
    Exact point-mass profile: psi(s) = -(1/g1) ln(1 - g1/s).

    From d psi/ds = -exp(g1 psi)/s^2 with G/G0 = exp(g1 psi).
    Uses log1p for numerical stability near weak field.
    """
    if g1 is None:
        g1 = dln_g_dpsi()
    s_arr = np.asarray(s, dtype=float)
    arg = -g1 / s_arr
    if np.any(1.0 + arg <= 0.0):
        raise ValueError("s must exceed g1 for real psi (g1 < 0)")
    return -np.log1p(arg) / g1


def dpsi_ds_analytic(s: float | np.ndarray, g1: float | None = None) -> float | np.ndarray:
    """Exact d psi/ds = -1/(s(s - g1))."""
    if g1 is None:
        g1 = dln_g_dpsi()
    s_arr = np.asarray(s, dtype=float)
    return -1.0 / (s_arr * (s_arr - g1))


def horizon_s_analytic(g1: float | None = None) -> float:
    """Horizon s_h from psi = 1/2: s_h = g1/(1 - exp(-g1/2))."""
    if g1 is None:
        g1 = dln_g_dpsi()
    return g1 / (1.0 - exp(-g1 / 2.0))


def s_from_psi(psi: float, g1: float | None = None) -> float:
    """Invert psi(s): s = -g1 / expm1(-g1*psi) (stable near psi=0)."""
    if g1 is None:
        g1 = dln_g_dpsi()
    return -g1 / math.expm1(-g1 * psi)


def psi_point_mass(s: float, g1: float | None = None) -> float:
    """Scalar exact psi(s); same as psi_analytic, uses math.log1p."""
    if g1 is None:
        g1 = dln_g_dpsi()
    return -(1.0 / g1) * math.log1p(-g1 / s)


def dpsi_ds_point_mass(s: float, g1: float | None = None) -> float:
    """Scalar d psi/ds = -1/(s*(s-g1))."""
    if g1 is None:
        g1 = dln_g_dpsi()
    return -1.0 / (s * (s - g1))


def field_integral_exterior_exact(
    s_lo: float, g1: float | None = None, psi_inf: float = 0.0
) -> float:
    """int_{s_lo}^inf exp(g1*psi)/s^2 ds = psi(s_lo) - psi(inf); exterior ODE."""
    return psi_point_mass(s_lo, g1) - psi_inf


def exterior_integral_numeric_lo(s_h: float, g1: float) -> float:
    """Numeric lower limit: above horizon and above the s=g1 integrand pole."""
    eps = max(1e-8, 1e-5 * abs(s_h))
    return max(s_h * (1.0 + 1e-4), g1 + eps)


def field_integral_exterior_numeric(
    s_lo: float, g1: float | None = None, s_hi: float = 1e5
) -> tuple[float, float]:
    """Numeric exterior field integral; stable integrand 1/(s*(s-g1)) + analytic tail."""
    if g1 is None:
        g1 = dln_g_dpsi()
    s_lo = max(s_lo, exterior_integral_numeric_lo(s_lo, g1))
    val, err = quad(lambda s: 1.0 / (s * (s - g1)), s_lo, s_hi, limit=200)
    tail = -(1.0 / g1) * math.log1p(g1 / (s_hi - g1))
    return val + tail, err


def exp_g_path_ratio(psi: float, g1: float | None = None) -> float:
    """G_path(psi)/G0 = exp(g1*psi) along the point-mass exterior ODE."""
    if g1 is None:
        g1 = dln_g_dpsi()
    return math.exp(g1 * psi)


def E_ref_quantile(psi: float) -> float:
    """E_ref(psi) = E_CS * (v/E_CS)^(1-psi) (Delta ruler quantile)."""
    return E_CS * (v_EW / E_CS) ** (1.0 - psi)


def tau_of_psi(psi: float, tau_g_val: float | None = None) -> float:
    """Refractive depth gradient tau(psi) = tau_G * (1 - psi)."""
    tg = tau_g_val if tau_g_val is not None else tau_g_with_c4(C4_REF)
    return tg * (1.0 - psi)


def photon_sphere_closed(g1: float | None = None) -> tuple[float, float, float]:
    """
    Photon sphere (Lambert W): y = exp(g1*psi) solves y + 2*ln(y) = 1 + g1.
    Returns (psi_ph, s_ph, b/r_g).
    """
    if g1 is None:
        g1 = dln_g_dpsi()
    z = 0.5 * math.exp(0.5 * (1.0 + g1))
    u_w = float(lambertw(z).real)
    y = 2.0 * u_w
    psi_ph = math.log(y) / g1
    s_ph = g1 * y / (y - 1.0)
    f_ph = 1.0 - 2.0 * psi_ph
    b_over_rg = s_ph / math.sqrt(max(f_ph, 1e-15))
    return psi_ph, s_ph, b_over_rg


def psi_photon_analytic(g1: float | None = None) -> float:
    """Photon-sphere psi (same as photon_sphere_closed)."""
    psi_ph, _, _ = photon_sphere_closed(g1)
    return psi_ph


def photon_geometry_analytic(
    g1: float | None = None,
) -> tuple[float, float, float]:
    """Return (psi_ph, s_ph, b_over_rg) via Lambert-W closed form."""
    return photon_sphere_closed(g1)


photon_s_b = photon_geometry_analytic
horizon_s = horizon_s_analytic


def point_mass_profile(
    n_eval: int = 500,
    s_min: float | None = None,
    s_max: float = 1e6,
    g1: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Log-spaced (s, psi) on the exact point-mass exterior (no ODE solver)."""
    if g1 is None:
        g1 = dln_g_dpsi()
    if s_min is None:
        s_min = horizon_s_analytic(g1) * 1.001
    s_vals = np.logspace(math.log10(s_min), math.log10(s_max), n_eval)
    u_vals = np.asarray(psi_analytic(s_vals, g1), dtype=float)
    return s_vals, u_vals


def precession_schwarzschild_per_orbit(s_a: float, e: float) -> float:
    """GR perihelion advance per orbit (radians), s_a = a/r_g."""
    return 6.0 * math.pi / (s_a * (1.0 - e**2))


def mercury_precession_cgm_gr_ratio(
    a_m: float = 5.791e10,
    e: float = 0.205630,
    m_sun_kg: float = 1.989e30,
    g_si: float = 6.674e-11,
    c_si: float = 3.0e8,
    g1: float | None = None,
) -> tuple[float, float, float, float]:
    """
    Weak-field CGM/GR perihelion ratio from effective PPN factor.

    Returns (ratio, pre_gr, pre_cgm, s_a) with s_a in r_g units.
    """
    if g1 is None:
        g1 = dln_g_dpsi()
    r_g = g_si * m_sun_kg / c_si**2
    s_a = a_m / r_g
    pre_gr = precession_schwarzschild_per_orbit(s_a, e)
    a2 = g1 / 2.0
    beta_eff = 1.0 - a2 / s_a
    pre_cgm = pre_gr * (4.0 - beta_eff) / 3.0
    return pre_cgm / pre_gr, pre_gr, pre_cgm, s_a


# Spin sector (J != 0): wavefunction holonomy -> metric deficit on f = 1 - 2*psi
KAPPA_KERNEL = 3.0 / 4.0
KAPPA_METRIC = W2_SHELL_DISPLACEMENT * KAPPA_KERNEL
N_FAMILY = 4.0


def helix_z2_activation(n_turns: int = 2) -> float:
    """Fraction of helix steps with Z2 = swapped (canonical 4-family word)."""
    word = family_word_for_micro(FAMILY_RAY_REF)
    state = GENE_MAC_REST
    swapped = 0
    n = 0
    for _ in range(n_turns):
        for byte in word:
            state = step_state_by_byte(state, byte)
            if state == GENE_MAC_SWAPPED:
                swapped += 1
            if state in (GENE_MAC_REST, GENE_MAC_SWAPPED):
                n += 1
    return swapped / max(n, 1)


def zeta_polar(theta_deg: float) -> float:
    """Polar modulation: Z2 active at equator (sin^2 theta)."""
    return float(np.sin(np.radians(theta_deg)) ** 2)


def metric_spin_deficit(
    s: float,
    u: float,
    a_star: float,
    theta_o_deg: float,
    z2_amp: float | None = None,
) -> float:
    """Gravitomagnetic deficit h on f = 1 - 2*u."""
    if z2_amp is None:
        z2_amp = helix_z2_activation()
    zeta = zeta_polar(theta_o_deg)
    geom = u / max(1.0 - 2.0 * u, 1e-6)
    return (
        KAPPA_METRIC
        * N_FAMILY
        * z2_amp
        * zeta
        * (a_star / max(s, 1e-6)) ** 2
        * geom
        * s
        / float(W2_SHELL_DISPLACEMENT)
    )


def _spin_u_and_deriv(s: float, g1: float | None = None) -> tuple[float, float]:
    """psi(s) and d psi/ds on the exact point-mass exterior."""
    if g1 is None:
        g1 = dln_g_dpsi()
    u = float(psi_analytic(s, g1))
    u_prime = float(dpsi_ds_point_mass(s, g1))
    return u, u_prime


def _h_derivs(
    s: float,
    u: float,
    u_prime: float,
    a_star: float,
    theta_o_deg: float,
    z2_amp: float,
    g1: float | None = None,
) -> tuple[float, float, float]:
    """h and total dh/ds along the analytic exterior profile."""
    if g1 is None:
        g1 = dln_g_dpsi()
    h = metric_spin_deficit(s, u, a_star, theta_o_deg, z2_amp)
    ds = max(s * 1e-7, 1e-9)
    du = 1e-8
    if g1 is None:
        g1 = dln_g_dpsi()
    s_lo = horizon_s_analytic(g1) * 1.001
    u_sp, _ = _spin_u_and_deriv(s + ds, g1)
    u_sm, _ = _spin_u_and_deriv(max(s - ds, s_lo), g1)
    h_sp = metric_spin_deficit(s + ds, u_sp, a_star, theta_o_deg, z2_amp)
    h_sm = metric_spin_deficit(max(s - ds, s_lo), u_sm, a_star, theta_o_deg, z2_amp)
    dh_ds = (h_sp - h_sm) / (2.0 * ds)
    h_up = metric_spin_deficit(s, u + du, a_star, theta_o_deg, z2_amp)
    dh_du = (h_up - h) / du
    return h, dh_ds + dh_du * u_prime, h


def photon_residual_spin(
    s: float,
    a_star: float,
    theta_o_deg: float,
    z2_amp: float | None = None,
    g1: float | None = None,
) -> float:
    """Photon condition s*f_eff' - 2*f_eff = 0 with f_eff = 1 - 2*u - h."""
    u, u_prime = _spin_u_and_deriv(s, g1)
    if u >= 0.499:
        return 1e6
    if z2_amp is None:
        z2_amp = helix_z2_activation()
    h, dh_ds_total, _ = _h_derivs(
        s, u, u_prime, a_star, theta_o_deg, z2_amp, g1
    )
    f_eff = 1.0 - 2.0 * u - h
    f_eff_prime = -2.0 * u_prime - dh_ds_total
    return s * f_eff_prime - 2.0 * f_eff


def _spin_photon_bracket(
    s_schw: float,
    a_star: float,
    theta_o_deg: float,
    z2_amp: float,
    g1: float | None = None,
    s_max: float = 1e6,
) -> tuple[float, float] | None:
    """Bracket s where photon_residual_spin changes sign."""
    if g1 is None:
        g1 = dln_g_dpsi()
    s_a = max(s_schw * 0.98, horizon_s_analytic(g1) * 1.001)
    s_b = min(max(s_schw * 1.35, 4.0), s_max)
    prev_s, prev_r = s_a, photon_residual_spin(
        s_a, a_star, theta_o_deg, z2_amp, g1
    )
    for i in range(1, 49):
        s = s_a + (s_b - s_a) * i / 48
        r = photon_residual_spin(s, a_star, theta_o_deg, z2_amp, g1)
        if prev_r * r <= 0.0 and abs(r) < 1e3:
            return prev_s, s
        prev_s, prev_r = s, r
    return None


def photon_impact_spin(
    s_ph: float,
    u_ph: float,
    a_star: float,
    theta_o_deg: float,
    z2_amp: float | None = None,
) -> float:
    """Impact parameter b/r_g at fixed s_ph with spin metric deficit."""
    h = metric_spin_deficit(s_ph, u_ph, a_star, theta_o_deg, z2_amp)
    f_eff = max(1.0 - 2.0 * u_ph - h, 1e-9)
    return s_ph / float(np.sqrt(f_eff))


def find_photon_sphere_spin(
    a_star: float,
    theta_o_deg: float,
    g1: float | None = None,
    s_max: float = 1e6,
):
    """Photon sphere and b/r_g from spin-coupled null geodesic condition."""
    if g1 is None:
        g1 = dln_g_dpsi()
    _, s_schw, _ = photon_geometry_analytic(g1)
    u_schw = float(psi_analytic(s_schw, g1))
    photon = (s_schw, u_schw, s_schw / math.sqrt(max(1.0 - 2.0 * u_schw, 1e-9)))
    z2_amp = helix_z2_activation()
    if a_star <= 0.0:
        return photon
    bracket = _spin_photon_bracket(
        s_schw, a_star, theta_o_deg, z2_amp, g1, s_max
    )
    if bracket is None:
        s_ph, u_ph, _ = photon
        b = photon_impact_spin(s_ph, u_ph, a_star, theta_o_deg, z2_amp)
        return s_ph, u_ph, b
    s_lo, s_hi = bracket
    s_ph = float(
        brentq(
            lambda s: photon_residual_spin(
                s, a_star, theta_o_deg, z2_amp, g1
            ),
            s_lo,
            s_hi,
        )  # type: ignore[arg-type]
    )
    u_ph, _ = _spin_u_and_deriv(s_ph, g1)
    h = metric_spin_deficit(s_ph, u_ph, a_star, theta_o_deg, z2_amp)
    f_eff = max(1.0 - 2.0 * u_ph - h, 1e-9)
    b = s_ph / float(np.sqrt(f_eff))
    return s_ph, u_ph, b


def g_pred_from_tau(tau_val):
    return G_kernel * math.exp(-tau_val) / v_EW**2


def shell_bounds_embed(k: int) -> tuple[float, float]:
    """Shell k at r_k = k/6, width 1/6, clipped to [0, 1]."""
    dr = 1.0 / 6.0
    r_c = k / 6.0
    return max(0.0, r_c - dr / 2.0), min(1.0, r_c + dr / 2.0)


def enclosed_mass_binomial(r: float) -> float:
    """M(r) = integral_0^r 4 pi r'^2 rho(r') dr' from binomial shell fractions."""
    if r <= 0.0:
        return 0.0
    total = 0.0
    for k in range(7):
        r_in, r_out = shell_bounds_embed(k)
        if r <= r_in:
            break
        w_k = binom_shell[k]
        if r >= r_out:
            total += w_k
        else:
            vol_full = r_out**3 - r_in**3
            vol_part = r**3 - r_in**3
            if vol_full > 0:
                total += w_k * (vol_part / vol_full)
    return total


def kernel_field_g(r: float, m_enc: float | None = None) -> float:
    """g(r) = -G_kernel M(r) / r^2 from kernel Gauss law."""
    if r <= 0.0:
        return float("nan")
    m = enclosed_mass_binomial(r) if m_enc is None else m_enc
    return -G_kernel * m / (r * r)


def verify_gauss_law_bridge(*, n_ext: int = 40) -> dict:
    """Discrete flux closure and exterior inverse-square checks."""
    m_total = enclosed_mass_binomial(1.0)
    flux_bnd = 4.0 * math.pi * kernel_field_g(1.0)
    flux_pred = -Q_G * G_kernel * m_total
    rs = np.linspace(1.05, 3.0, n_ext)
    gr2 = [abs(kernel_field_g(r, m_total)) * r**2 for r in rs]
    mean_gr2 = float(np.mean(gr2))
    rel_std = float(np.std(gr2)) / mean_gr2 if mean_gr2 else float("nan")
    log_r = np.log(rs)
    log_g = np.log([abs(kernel_field_g(r, m_total)) for r in rs])
    slope = float(np.polyfit(log_r, log_g, 1)[0])
    return {
        "m_total": m_total,
        "flux_boundary": flux_bnd,
        "flux_expected": flux_pred,
        "flux_ratio": flux_bnd / flux_pred if flux_pred else float("nan"),
        "exterior_gr2_mean": mean_gr2,
        "exterior_gr2_rel_std": rel_std,
        "log_slope": slope,
        "ok_flux": abs(flux_bnd / flux_pred - 1.0) < 1e-12 if flux_pred else False,
        "ok_is": rel_std < 1e-14,
        "ok_slope": abs(slope + 2.0) < 1e-10,
    }


def alpha_lab_with_transport_corrections() -> float:
    """
    alpha after AB, HC, IDE transport corrections (hqvm_corrections_analysis_1).
    """
    d = d_BU
    mp_ = m_a
    r_curv = 0.993434896272
    h_hol = 4.417034
    rho_inv = 1.021137
    diff = 0.001874
    d_ap = 1.0 - d / mp_
    d2 = d_ap * d_ap
    d4 = d2 * d2
    phi = 3.0 * d + diff
    c_ab = 1.0 - (3.0 / 4.0) * r_curv * d2
    c_hc = 1.0 - (5.0 / 6.0) * (
        (phi / (3.0 * d) - 1.0)
        * (1.0 - d2 * h_hol)
        * d2
        / (4.0 * math.pi * math.sqrt(3.0))
    )
    c_ide = 1.0 + rho_inv * diff * d4
    alpha0 = d**4 / mp_
    return alpha0 * c_ab * c_hc * c_ide


def zeta_from_alpha(alpha: float) -> float:
    """zeta = rho^4 / (pi sqrt(3) alpha) from alpha*zeta identity."""
    return rho**4 / (math.pi * math.sqrt(3.0) * alpha)


def verify_alpha_zeta_product(*, alpha_codata: float | None = None) -> dict:
    """alpha_kernel * zeta = rho^4 / (pi sqrt(3))."""
    zeta_geom = 16.0 * math.sqrt(2.0 * math.pi / 3.0)
    alpha_kernel = d_BU**4 / m_a
    rhs = rho**4 / (math.pi * math.sqrt(3.0))
    lhs = alpha_kernel * zeta_geom
    out = {
        "zeta_geom": zeta_geom,
        "alpha_kernel": alpha_kernel,
        "lhs": lhs,
        "rhs": rhs,
        "exact": lhs == rhs,
    }
    if alpha_codata is not None and alpha_codata > 0:
        out["alpha_codata"] = alpha_codata
        out["zeta_from_alpha"] = rhs / alpha_codata
        out["zeta_ratio"] = out["zeta_from_alpha"] / zeta_geom - 1.0
    alpha_lab = alpha_lab_with_transport_corrections()
    out["alpha_lab"] = alpha_lab
    out["zeta_predicted_lab"] = zeta_from_alpha(alpha_lab)
    out["zeta_ratio_lab"] = out["zeta_predicted_lab"] / zeta_geom - 1.0
    return out


def c4_from_anchors(g_gev2, v_ew_gev):
    """Validation only: c4 implied by measured G. Do not use G = G_kernel/E_CS^2."""
    tau_req_loc = -math.log(g_gev2 * v_ew_gev**2 / G_kernel)
    k_need = tau_req_loc / (Omega_size * Delta * rho**5)
    x_ex = k_need - f_ordered
    return x_ex / Delta**4 if Delta > 0 else float("nan")


def k4_pq_charges():
    """EW trace-free charges (p, q) per K4 channel from gyrotriangle closure.

    Channel flags on the K4 edge walk (see hqvm_compact_geom_core.CHANNELS):
      b (base): breaks CS reference frame (Higgs path)
      r (rot):  ONA reversal increment on the edge
      bal:    BU balance increment on the edge

    Formulas match _pq() in hqvm_compact_geom_core: p = 1 + (-C1/2)*b + (C1/4)*r + 2*bal,
    q = 5/4 - 2*r - bal with C1=6 (CODE_C1). Returns (p, q) per channel name.
    """
    p0, q0 = 1.0, 5.0 / 4.0
    rows = []
    for name, b, r, bal in K4_CHANNEL_FLAGS:
        p = p0 + (0.0 if not b else -6.0 / 2.0)
        q = q0
        if r:
            p += 6.0 / 4.0
            q += -4.0 * 0.5
        if bal:
            p += 4.0 * 0.5
            q += -2.0 * 0.5
        rows.append((name, p, q))
    q_sum = sum(r[2] for r in rows)
    return rows, q_sum


def chi6_step_bit_stats(joint_table):
    by_step = {}
    for rec in joint_table:
        by_step.setdefault(rec["step"], []).append(int(rec["chi6"]) & 0x3F)
    for s in sorted(by_step):
        xs = by_step[s]
        n = len(xs)
        probs = [sum((x >> b) & 1 for x in xs) / n if n > 0 else 0.0 for b in range(6)]
        ps = " ".join(f"{p:.3f}" for p in probs)
        print(f"  step {s}: n={n}  P(bit=1) b0..b5 = {ps}")


def print_joint_table_condensed(table):
    n = len(table)
    print(f"  rows={n} (expected 512)")
    by_m = {}
    for rec in table:
        by_m.setdefault(rec["m_ref"], []).append(rec)

    fa_ref = [0, 1, 2, 3, 0, 1, 2, 3]
    mismatches = []
    by_pop = {}
    for m in range(64):
        rows = sorted(by_m.get(m, []), key=lambda r: r["step"])
        if not rows:
            continue
        pop = rows[0]["pop"]
        w = rows[0]["weight"]
        sh = [r["arch_shell"] for r in rows]
        mi = [r["micro"] for r in rows]
        ins = [r["intron"] for r in rows]
        fa = [r["family"] for r in rows]
        qx = [r["qxor"] for r in rows]
        exp_sh = [pop, 6, pop, 0, pop, 6, pop, 0]
        exp_mi = [m] * 8
        exp_in = [2 * m, 2 * m + 1, 128 + 2 * m, 129 + 2 * m] * 2
        exp_qx = [m, 63, m, 0] * 2
        if (sh, mi, ins, fa, qx) != (exp_sh, exp_mi, exp_in, fa_ref, exp_qx):
            mismatches.append(m)
        by_pop.setdefault(pop, []).append((m, w))

    print("  step 1..8 templates (verified on all 64 m_ref):")
    print("    fa: 0>1>2>3>0>1>2>3")
    print("    arch_sh(pop): pop>6>pop>0>pop>6>pop>0")
    print("    mi(m): m x8")
    print("    in(m): 2m>2m+1>128+2m>129+2m (half-cycle x2)")
    print("    qx(m): m>63>m>0 (half-cycle x2)")
    print("    w(pop)=binom(6,pop)/64")
    for pop in sorted(by_pop):
        ms = sorted(x[0] for x in by_pop[pop])
        w = by_pop[pop][0][1]
        m_str = ",".join(f"{x:02d}" for x in ms)
        print(f"    pop={pop} w={w:.6f} n={len(ms)} m=[{m_str}]")
    if mismatches:
        print(f"  template mismatches at m={mismatches}")
