from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.api import shell_krawtchouk_transform_exact


_BITMASK_64x6 = (
    (np.arange(64, dtype=np.uint8)[:, None] >> np.arange(6, dtype=np.uint8)[None, :]) & 1
).astype(np.float64)
# Precomputed bit occupancy table avoids per-call Python loops in eta-vector parity stats.


@dataclass(frozen=True)
class QuBECClimate:
    rho: float
    m: float
    eta: float
    lam: float
    M2: float
    M2_eq: float
    shell_spectrum: tuple[float, ...]
    eta_vec: tuple[float, ...]
    gauge_spectrum: tuple[float, ...] | None
    samples: int


def compute_qubec_climate(
    chi_hist64: np.ndarray,
    shell_hist7: np.ndarray,
    samples: int,
) -> QuBECClimate:
    """
    Exact QuBEC climate from rolling chirality histogram.

    chi_hist64: length-64 histogram over GF(2)^6 chirality values
    shell_hist7: length-7 histogram over shells 0..6
    samples: total valid observations
    """
    if samples == 0:
        return QuBECClimate(
            rho=0.5,
            m=0.0,
            eta=0.0,
            lam=1.0,
            M2=4096.0,
            M2_eq=4096.0,
            shell_spectrum=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            eta_vec=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            gauge_spectrum=None,
            samples=0,
        )

    total = float(samples)
    shell = shell_hist7.astype(np.float64) / total

    mean_N = sum(w * shell[w] for w in range(7))
    rho = mean_N / 6.0
    m = 2.0 * rho - 1.0
    eta = -m

    if rho < 1.0 - 1e-12:
        lam = rho / (1.0 - rho)
    else:
        lam = 1e12

    eta_sq = eta * eta
    shell_values = list(int(x) for x in shell_hist7)
    shell_spectrum = tuple(
        float(x)
        for x in shell_krawtchouk_transform_exact(shell_values)
    )

    chi = chi_hist64.astype(np.float64)
    from src.tools.gyrograph import ops as gyrograph_ops

    M2 = gyrograph_ops.compute_m2_empirical_from_chi_hist(chi_hist64, samples)
    M2_eq = gyrograph_ops.compute_m2_equilibrium_from_shell_hist(shell_hist7, samples)

    bit_probs = (chi @ _BITMASK_64x6) / total
    eta_vec = tuple((1.0 - 2.0 * bit_probs).tolist())

    return QuBECClimate(
        rho=rho,
        m=m,
        eta=eta,
        lam=lam,
        M2=M2,
        M2_eq=M2_eq,
        shell_spectrum=shell_spectrum,
        eta_vec=eta_vec,
        gauge_spectrum=None,
        samples=samples,
    )


def gauge_spectrum_from_family_hist(
    family_hist: np.ndarray,
) -> tuple[float, ...]:
    """
    Compute K4 gauge damping from family distribution.

    Family encoding: family = ((bit7)<<1) | bit0
      family 0 = (0,0): neither bit set
      family 1 = (0,1): bit0 set
      family 2 = (1,0): bit7 set
      family 3 = (1,1): both set

    p_A = Pr(family & 1) = (count[1] + count[3]) / total
    p_B = Pr(family & 2) = (count[2] + count[3]) / total
    """
    total = float(family_hist.sum())
    if total == 0:
        return (1.0, 1.0, 1.0, 1.0)

    p_A = float(family_hist[1] + family_hist[3]) / total
    p_B = float(family_hist[2] + family_hist[3]) / total
    xi_A = 1.0 - 2.0 * p_A
    xi_B = 1.0 - 2.0 * p_B

    return (
        1.0,
        xi_A,
        xi_B,
        xi_A * xi_B,
    )
