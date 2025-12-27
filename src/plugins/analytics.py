"""
Analytics helpers.

This is where you compute:
- GGG apertures (ledger-based)
- (optional) ecology-style derived summaries later

Uses the GGG geometry functions from src.app.ledger for consistency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.app.ledger import (
    compute_aperture,
    get_projections,
    hodge_decomposition,
)


EdgeVec = NDArray[np.float64]


@dataclass(frozen=True)
class HodgeDecomposition:
    y_grad: EdgeVec
    y_cycle: EdgeVec
    aperture: float


def hodge_decompose(y: EdgeVec) -> HodgeDecomposition:
    """
    Compute (y_grad, y_cycle, aperture) for a 6-vector y on K4.
    
    Uses exact GGG K4 projection matrices (audit-grade invariants).
    Follows GGG simulator export exactly with W=I (unweighted).
    """
    y = np.asarray(y, dtype=np.float64).reshape(6)
    P_grad, P_cycle = get_projections()
    
    y_grad, y_cycle = hodge_decomposition(y, P_grad, P_cycle)
    aperture = compute_aperture(y, y_cycle)
    
    return HodgeDecomposition(y_grad=y_grad, y_cycle=y_cycle, aperture=aperture)

