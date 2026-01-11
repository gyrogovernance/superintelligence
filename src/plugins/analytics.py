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

from src.app.ledger import get_projections


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
    
    This function works with float64 arrays directly (convenience wrapper).
    For integer micro-unit ledgers, use DomainLedgers directly.
    """
    y_float = np.asarray(y, dtype=np.float64).reshape(6)
    P_grad, P_cycle = get_projections()
    
    # Work directly with float64 (hodge_decomposition converts to float64 internally anyway)
    # This avoids quantization errors from int64 conversion
    y_grad = P_grad @ y_float
    y_cycle = P_cycle @ y_float
    
    # Compute aperture using the float64 arrays directly
    y_norm_sq = float(y_float @ y_float)
    if y_norm_sq == 0.0:
        aperture = 0.0
    else:
        cycle_norm_sq = float(y_cycle @ y_cycle)
        aperture = cycle_norm_sq / y_norm_sq
    
    return HodgeDecomposition(y_grad=y_grad, y_cycle=y_cycle, aperture=aperture)

