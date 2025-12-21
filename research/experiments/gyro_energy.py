#!/usr/bin/env python3
# gyro_energy.py
# CGM energies in canonical order:
#   (1) dimensionless (pure SU(2)/Pauli structure)
#   (2) CS anchor (Planck units from {ħ, c, G})
#   (3) sector anchors: GUT (UNA+ONA), EW (BU)
#   (4) free-anchor diagnostics (predict ζ or G if you FORCE a stage to a scale)
#   (5) maps-dependent (atlas, if available)
#   (6) BU-centered duality (BU↔EW only)

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

# -----------------------------
# CODATA constants
# -----------------------------
try:
    import scipy.constants as sc
    HBAR = sc.hbar          # J·s
    C = sc.c                # m/s
    G = sc.G                # m^3/(kg s^2)
    E_CHARGE = sc.e         # C
except Exception:
    # CODATA fallbacks
    HBAR = 1.054571817e-34
    C = 299792458.0
    G = 6.67430e-11
    E_CHARGE = 1.602176634e-19

GEV_TO_J = 1e9 * E_CHARGE
J_TO_GEV = 1.0 / GEV_TO_J

# -----------------------------
# GyroSI core geometric invariants
# -----------------------------
PI = math.pi
FOUR_PI = 4.0 * PI

# Aperture parameter
m_p = 1.0 / (2.0 * math.sqrt(2.0 * PI))

# Actions for the three stages
S_UNA = (PI / 2.0) * m_p                 # minimal/forward
S_ONA = math.sqrt(3.0) * S_UNA           # geometric mean mode
S_BU  = 3.0 * S_UNA                       # reciprocal

# Geometric "units"
E_geom_unit = PI / 2.0                   # dimensionless energy unit
Q_G = FOUR_PI                            # survey/solid-angle invariant
S_geo = m_p * PI * math.sqrt(3.0) / 2.0  # geometric mean action
zeta = Q_G / S_geo                       # gravitational prefactor from geometry

# -----------------------------
# CS anchor (Planck units only)
# -----------------------------
def planck_anchor_cs() -> Dict[str, float]:
    # Standard Planck units from {ħ,c,G}
    T_pl = math.sqrt(HBAR * G / (C**5))
    L_pl = math.sqrt(HBAR * G / (C**3))
    E_pl = math.sqrt(HBAR * (C**5) / G)
    M_pl = math.sqrt(HBAR * C / G)
    return {
        "T_Pl": T_pl,
        "L_Pl": L_pl,
        "E_Pl_J": E_pl,
        "E_Pl_GeV": E_pl * J_TO_GEV,
        "M_Pl": M_pl,
    }

# -----------------------------
# Sector anchors (no E0 baseline)
# -----------------------------
def sector_anchors(E_GUT_GeV: float = 1.0e16,
                   E_EW_GeV: float = 246.0,
                   sector_lock: bool = False) -> Dict[str, float]:
    # UNA+ONA share the GUT sector if sector_lock=True
    E_UNA_GeV = E_GUT_GeV
    E_ONA_GeV = E_GUT_GeV if sector_lock else (E_GUT_GeV / math.sqrt(3.0))
    E_BU_GeV  = E_EW_GeV
    return {
        "E_GUT_GeV": E_GUT_GeV,
        "E_EW_GeV": E_EW_GeV,
        "E_UNA_GeV": E_UNA_GeV,
        "E_ONA_GeV": E_ONA_GeV,
        "E_BU_GeV":  E_BU_GeV,
        "sector_lock": float(sector_lock),
    }

# -----------------------------
# Free anchor diagnostics (not a calibration)
# -----------------------------
def free_anchor_calibration(anchor_stage: str,
                            anchor_energy_GeV: float,
                            predict: str = "zeta") -> Dict[str, float | str]:
    """
    Diagnostic: if you force a stage to a target energy, what ζ or G would be implied?
    """
    E_target_J = anchor_energy_GeV * GEV_TO_J

    if anchor_stage == "UNA":
        S_anchor = S_UNA
    elif anchor_stage == "ONA":
        S_anchor = S_ONA
    elif anchor_stage == "BU":
        S_anchor = S_BU
    else:
        raise ValueError(f"Invalid anchor stage: {anchor_stage}")

    A = G * (FOUR_PI ** 3) / (HBAR * (C ** 5))

    if predict == "zeta":
        zeta_pred = A * S_anchor * (E_target_J ** 2)
        return {
            "anchor_stage": anchor_stage,
            "anchor_energy_GeV": anchor_energy_GeV,
            "zeta_pred": zeta_pred,
            "zeta_ratio": zeta_pred / zeta,
        }
    elif predict == "G":
        G_pred = zeta * (C ** 5) * HBAR / ((FOUR_PI ** 3) * S_anchor * (E_target_J ** 2))
        return {
            "anchor_stage": anchor_stage,
            "anchor_energy_GeV": anchor_energy_GeV,
            "G_pred": G_pred,
            "G_ratio": G_pred / G,
        }
    else:
        raise ValueError(f"Invalid predict mode: {predict}")

# -----------------------------
# Maps-dependent (if available)
# -----------------------------
DEFAULT_META_DIR = Path("memories/public/meta")
PATH_EPI = DEFAULT_META_DIR / "epistemology.npy"

# Stage intron masks
EXON_LI_MASK = 0b01000010  # UNA bits
EXON_FG_MASK = 0b00100100  # ONA bits
EXON_BG_MASK = 0b00011000  # BU bits

def intron_subset(mask: int) -> NDArray[np.int32]:
    arr = np.arange(256, dtype=np.int32)
    keep = (arr & mask) != 0
    return arr[keep]

def matvec_P_row_stochastic(
    x: NDArray[np.float64],
    epi_memmap: NDArray[np.int32],
    allowed_introns: NDArray[np.int32],
    batch: int = 20000,
    lazy_eps: float = 0.05,
) -> NDArray[np.float64]:
    N = epi_memmap.shape[0]
    y = np.empty_like(x)
    k = float(allowed_introns.size)
    if k == 0:
        raise ValueError("No introns selected for this stage")

    for start in range(0, N, batch):
        end = min(start + batch, N)
        epi_batch = epi_memmap[start:end, :]
        succ = epi_batch[:, allowed_introns]
        mean_vals = np.mean(x[succ], axis=1)
        y[start:end] = mean_vals

    if lazy_eps > 0.0:
        y = (1.0 - lazy_eps) * y + lazy_eps * x
    return y

def estimate_lambda2_and_tau(
    epi_path: Path,
    allowed_introns: NDArray[np.int32],
    iters: int = 30,
    tol: float = 1e-8,
    batch: int = 20000,
    lazy_eps: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    epi = np.load(epi_path, mmap_mode="r")
    N = epi.shape[0]

    rng = np.random.default_rng(seed)
    v = rng.standard_normal(N).astype(np.float64)
    v -= np.mean(v)
    nrm = np.linalg.norm(v)
    if nrm == 0:
        v[0] = 1.0
        nrm = 1.0
    v /= nrm

    lambda_old = 0.0
    for _ in range(iters):
        Pv = matvec_P_row_stochastic(v, epi, allowed_introns, batch=batch, lazy_eps=lazy_eps)
        lam = float(np.dot(v, Pv) / np.dot(v, v))
        Pv -= np.mean(Pv)
        nrm = np.linalg.norm(Pv)
        if nrm == 0:
            break
        v = Pv / nrm
        if abs(lam - lambda_old) < tol:
            lambda_old = lam
            break
        lambda_old = lam

    lam2 = float(lambda_old)
    gap = max(1e-12, 1.0 - abs(lam2))
    tau = 1.0 / gap
    return lam2, tau

def compute_atlas_corrections() -> Dict[str, float | str]:
    epi_path = PATH_EPI
    if not epi_path.exists():
        return {"status": "skipped", "reason": "atlas data not found"}

    try:
        li_introns = intron_subset(EXON_LI_MASK)  # UNA
        fg_introns = intron_subset(EXON_FG_MASK)  # ONA
        bg_introns = intron_subset(EXON_BG_MASK)  # BU

        lam_UNA, tau_UNA = estimate_lambda2_and_tau(epi_path, li_introns, iters=10, batch=1000)
        lam_ONA, tau_ONA = estimate_lambda2_and_tau(epi_path, fg_introns, iters=10, batch=1000, seed=43)
        lam_BU,  tau_BU  = estimate_lambda2_and_tau(epi_path, bg_introns, iters=10, batch=1000, seed=44)

        return {
            "status": "computed",
            "lambda2_UNA": lam_UNA,
            "lambda2_ONA": lam_ONA,
            "lambda2_BU": lam_BU,
            "tau_UNA": tau_UNA,
            "tau_ONA": tau_ONA,
            "tau_BU": tau_BU,
        }
    except Exception as e:
        return {"status": "error", "reason": str(e)}

# -----------------------------
# BU-centered duality
# -----------------------------
def bu_duality(E_probe_GeV: float, E_BU_ref_GeV: float = 246.0) -> float:
    return (E_BU_ref_GeV ** 2) / E_probe_GeV

# -----------------------------
# Pretty printing
# -----------------------------
def fmt_si(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}e}"

def print_section(title: str) -> None:
    line = "-" * len(title)
    print(f"\n{title}\n{line}")

# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Dimensionless invariants
    print_section("DIMENSIONLESS (pure SU(2)/Pauli structure)")
    print(f"m_p                     = {m_p:.12f}")
    print(f"S_UNA                  = {S_UNA:.12f}")
    print(f"S_ONA                  = {S_ONA:.12f}")
    print(f"S_BU                   = {S_BU:.12f}")
    print(f"Q_G                    = {Q_G:.12f}")
    print(f"zeta                   = {zeta:.8f}")
    print(f"E_geom_unit            = {E_geom_unit:.12f}")
    print(f"Energy ratios (UNA, ONA, BU) = 1.000000, {1/math.sqrt(3):.6f}, {1/3:.6f}")

    # 2) CS anchor (Planck)
    planck = planck_anchor_cs()
    print_section("CS ANCHOR (Planck units from {ħ, c, G})")
    print(f"T_Pl   [s]  = {fmt_si(planck['T_Pl'])}")
    print(f"L_Pl   [m]  = {fmt_si(planck['L_Pl'])}")
    print(f"M_Pl   [kg] = {fmt_si(planck['M_Pl'])}")
    print(f"E_Pl   [J]  = {fmt_si(planck['E_Pl_J'])}   ({fmt_si(planck['E_Pl_GeV'])} GeV)")

    # 3) Sector anchors (UNA+ONA = GUT, BU = EW)
    anchors = sector_anchors(E_GUT_GeV=1.0e16, E_EW_GeV=246.0, sector_lock=True)
    print_section("SECTOR ANCHORS (UNA+ONA = GUT, BU = EW)")
    print(f"E_GUT  (UNA+ONA) = {fmt_si(anchors['E_GUT_GeV'])} GeV")
    print(f"E_EW   (BU)      = {fmt_si(anchors['E_EW_GeV'])} GeV")

    # 4) Free-anchor diagnostics (no CS violation)
    print_section("FREE-ANCHOR DIAGNOSTICS (what ζ or G would be implied)")
    gut_UNA = free_anchor_calibration("UNA", anchors["E_GUT_GeV"], "zeta")
    gut_ONA = free_anchor_calibration("ONA", anchors["E_GUT_GeV"], "zeta")
    ew_BU   = free_anchor_calibration("BU",  anchors["E_EW_GeV"],  "zeta")
    print(f"GUT at UNA: ζ_pred = {gut_UNA['zeta_pred']:.3e} (ratio: {gut_UNA['zeta_ratio']:.3e})")
    print(f"GUT at ONA: ζ_pred = {gut_ONA['zeta_pred']:.3e} (ratio: {gut_ONA['zeta_ratio']:.3e})")
    print(f"EW  at BU : ζ_pred = {ew_BU['zeta_pred']:.3e} (ratio: {ew_BU['zeta_ratio']:.3e})")

    # 5) MAPS-DEPENDENT (atlas)
    print_section("MAPS-DEPENDENT (atlas)")
    atlas = compute_atlas_corrections()
    if atlas["status"] == "computed":
        print(f"λ₂_UNA = {atlas['lambda2_UNA']:.8f}")
        print(f"λ₂_ONA = {atlas['lambda2_ONA']:.8f}")
        print(f"λ₂_BU  = {atlas['lambda2_BU']:.8f}")
        print(f"τ_UNA = {atlas['tau_UNA']:.3f}, τ_ONA = {atlas['tau_ONA']:.3f}, τ_BU = {atlas['tau_BU']:.3f}")

        # ζ shifts (simple linear probe around geometric ζ)
        lambda2_star = 0.5
        a = -0.1
        z_UNA = zeta * (1.0 + a * (float(atlas['lambda2_UNA']) - lambda2_star))
        z_ONA = zeta * (1.0 + a * (float(atlas['lambda2_ONA']) - lambda2_star))
        z_BU  = zeta * (1.0 + a * (float(atlas['lambda2_BU'])  - lambda2_star))
        print(f"ζ_UNA = {z_UNA:.6f} (ratio: {z_UNA/zeta:.6f})")
        print(f"ζ_ONA = {z_ONA:.6f} (ratio: {z_ONA/zeta:.6f})")
        print(f"ζ_BU  = {z_BU:.6f} (ratio: {z_BU/zeta:.6f})")

        # Optional τ-scaling test for BU=EW
        tau_UNA = float(atlas['tau_UNA'])
        tau_BU  = float(atlas['tau_BU'])
        tau_ratio_BU = tau_UNA / tau_BU

        E_BU_atlas = anchors['E_EW_GeV'] * tau_ratio_BU * (S_UNA / S_BU)
        print(f"BU atlas test: E_BU_atlas = {fmt_si(E_BU_atlas)} GeV ; E_BU_atlas / 246 GeV = {E_BU_atlas/246.0:.3e}")
        print(f"BU=EW support: {'YES' if abs(E_BU_atlas/246.0 - 1.0) < 0.1 else 'NO'}")
    else:
        print(f"Atlas: {atlas['reason']}")

    # 6) BU duality
    print_section("BU DUALITY (E_BU_ref = 246 GeV)")
    for E_test in [246.0, 1e16, planck['E_Pl_GeV']]:
        E_dual = bu_duality(E_test, 246.0)
        print(f"E = {fmt_si(E_test)} GeV → E_dual = {fmt_si(E_dual)} GeV (ratio: {fmt_si(E_dual/E_test)})")

    # 7) Summary
    print_section("SUMMARY (CS → GUT → EW)")
    print(f"CS / Planck: {fmt_si(planck['E_Pl_GeV'])} GeV")
    print(f"GUT (UNA+ONA): {fmt_si(anchors['E_GUT_GeV'])} GeV")
    print(f"EW  (BU):      {fmt_si(anchors['E_EW_GeV'])} GeV")
    print(f"Dimensionless energy ratios (UNA, ONA, BU) = 1.000000, {1/math.sqrt(3):.6f}, {1/3:.6f}")
    print("Free-anchor lines are diagnostics only; CS anchoring is unique and not modified.")

if __name__ == "__main__":
    main()
