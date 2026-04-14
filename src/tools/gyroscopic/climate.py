"""
QuBEC climate helpers aligned with docs/theory/QuBEC_Climate_Dynamics.md (Sections 2-5, 16).

Pure naming and packaging on top of existing C-backed ops; no new physics.
"""

from __future__ import annotations

import ctypes as ct
from typing import Any

from .ops import (
    gyrograph_compute_m2_empirical,
    gyrograph_compute_m2_equilibrium,
    gyrolabe_anisotropy_extract,
    gyrolabe_k4char4_float,
    gyrolabe_krawtchouk7_float,
)


def shell_order_parameters_from_hist(
    shell_hist: list[int] | tuple[int, ...],
) -> dict[str, float]:
    """
    Mean shell N, rho = N_mean/6, eta = 1 - 2*rho, m = 2*rho - 1, Var(N) = 6*rho*(1-rho).
    """
    if len(shell_hist) != 7:
        raise ValueError("shell_hist must have length 7")
    total = sum(int(shell_hist[w]) for w in range(7))
    if total == 0:
        return {
            "N_mean": 0.0,
            "rho": 0.0,
            "eta": 1.0,
            "m": -1.0,
            "var_N": 0.0,
        }
    n_mean = (
        sum(float(w) * float(shell_hist[w]) for w in range(7)) / float(total)
    )
    rho = n_mean / 6.0
    eta = 1.0 - 2.0 * rho
    m = 2.0 * rho - 1.0
    var_n = 6.0 * rho * (1.0 - rho)
    return {
        "N_mean": float(n_mean),
        "rho": float(rho),
        "eta": float(eta),
        "m": float(m),
        "var_N": float(var_n),
    }


def m2_empirical_from_chi_hist(chi_hist: list[int] | tuple[int, ...]) -> float:
    """Climate M2: 64 * native Rényi-2 on the 64-bin register when total > 0 (matches M2_equilibrium scale)."""
    if len(chi_hist) != 64:
        raise ValueError("chi_hist must have length 64")
    total = sum(int(chi_hist[i]) for i in range(64))
    arr = (ct.c_uint16 * 64)(*[int(chi_hist[i]) & 0xFFFF for i in range(64)])
    raw = float(gyrograph_compute_m2_empirical(arr, int(total)))
    if int(total) == 0:
        return raw
    return raw * 64.0


def m2_equilibrium_from_shell_hist(
    shell_hist: list[int] | tuple[int, ...],
) -> float:
    """Analytic Omega M2 = 4096 / (1 + eta^2)^6 from shell histogram (native)."""
    if len(shell_hist) != 7:
        raise ValueError("shell_hist must have length 7")
    total = sum(int(shell_hist[w]) for w in range(7))
    arr = (ct.c_uint16 * 7)(*[int(shell_hist[w]) & 0xFFFF for w in range(7)])
    return gyrograph_compute_m2_equilibrium(arr, int(total))


def cell_climate_from_histograms(
    chi_hist64: list[int] | tuple[int, ...],
    shell_hist7: list[int] | tuple[int, ...],
    family_hist4: list[int] | tuple[int, ...],
    *,
    byte_ensemble_256: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """
    One-cell climate observables (doc Section 16): order parameters, M2 pair,
    shell Krawtchouk spectrum, K4 gauge spectrum, optional byte anisotropy (Sec 9).
    """
    if len(family_hist4) != 4:
        raise ValueError("family_hist4 must have length 4")

    shell_stats = shell_order_parameters_from_hist(shell_hist7)
    m2_eq = m2_equilibrium_from_shell_hist(shell_hist7)
    m2_emp = m2_empirical_from_chi_hist(chi_hist64)

    sh = [float(shell_hist7[w]) for w in range(7)]
    shell_spectral = gyrolabe_krawtchouk7_float(sh)

    fam_f = [float(family_hist4[i]) for i in range(4)]
    gauge_spectral = gyrolabe_k4char4_float(fam_f)

    byte_anisotropy: list[float] | None = None
    if byte_ensemble_256 is not None:
        if len(byte_ensemble_256) != 256:
            raise ValueError("byte_ensemble_256 must have length 256")
        byte_anisotropy = gyrolabe_anisotropy_extract(byte_ensemble_256)

    return {
        "N_mean": shell_stats["N_mean"],
        "rho": shell_stats["rho"],
        "eta": shell_stats["eta"],
        "m": shell_stats["m"],
        "var_N": shell_stats["var_N"],
        "M2_empirical": m2_emp,
        "M2_equilibrium": m2_eq,
        "shell_spectral": shell_spectral,
        "gauge_spectral": gauge_spectral,
        "byte_anisotropy": byte_anisotropy,
    }


__all__ = [
    "cell_climate_from_histograms",
    "m2_empirical_from_chi_hist",
    "m2_equilibrium_from_shell_hist",
    "shell_order_parameters_from_hist",
]
