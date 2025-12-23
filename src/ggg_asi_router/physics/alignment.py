"""
Alignment Metrics

Defines the canonical deviation and Superintelligence Index (SI) derived from
the aperture observable A in the Common Governance Model.

Definitions:
    D(A)  = max(A/A*, A*/A)
    SI(A) = 100 / D(A)

where A* is the CGM canonical aperture.
"""

from __future__ import annotations

from .cgm_constants import A_STAR


def compute_deviation(A: float, A_star: float = A_STAR) -> float:
    """
    Compute deviation D = max(A/A*, A*/A).

    Args:
        A: Aperture in [0, 1].
        A_star: Canonical aperture A* > 0.

    Returns:
        Deviation factor D >= 1, or +inf if ratios are undefined.
    """
    if A <= 0.0:
        return float("inf") if A_star > 0.0 else 1.0
    if A_star <= 0.0:
        return float("inf") if A > 0.0 else 1.0
    r1 = A / A_star
    r2 = A_star / A
    return float(r1 if r1 >= r2 else r2)


def compute_si(A: float, A_star: float = A_STAR) -> float:
    """
    Compute Superintelligence Index SI = 100 / max(A/A*, A*/A).

    Args:
        A: Aperture in [0, 1].
        A_star: Canonical aperture.

    Returns:
        SI in [0, 100].
    """
    D = compute_deviation(A, A_star=A_star)
    if D == float("inf") or D <= 0.0:
        return 0.0
    si = 100.0 / D
    if si < 0.0:
        return 0.0
    if si > 100.0:
        return 100.0
    return float(si)

