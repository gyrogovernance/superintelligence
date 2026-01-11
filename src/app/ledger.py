"""
Domain ledgers for GGG aperture.

Stores per-domain edge vectors y_D ∈ R^6.

Important:
- GGG aperture is defined on the evolving edge ledger y (not on kernel bit artifacts).
- Kernel provides the shared moment/order; App applies GovernanceEvents to ledgers.

This module implements the GGG Hodge decomposition exactly as exported from the simulator
with W = I (unweighted). Confidence is encoded in GovernanceEvent.signed_value().

Geometry:
- Incidence matrix B (4x6) for K4 graph
- General form: P_grad = B^T pinv(B B^T) B, P_cycle = I - P_grad
- For K4 with W=I: P_grad = (1/4) × (B^T B) (exact closed form, no pseudoinverse)
- Aperture: A = ||y_cycle||^2 / ||y||^2

This implementation uses the exact closed form for K4, ensuring audit-grade invariants
identical across all platforms and numpy/BLAS builds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .events import Domain, GovernanceEvent


EdgeVec = NDArray[np.int64]  # Ledger vectors use integer micro-units
EdgeVecFloat = NDArray[np.float64]  # For intermediate float computations
Mat46 = NDArray[np.float64]
Mat44 = NDArray[np.float64]
Mat66 = NDArray[np.float64]


# GGG Geometry Functions (exact K4 matrices, audit-grade invariants)

# Exact K4 incidence matrix B (4x6)
# Vertex order: (Gov, Info, Infer, Intel) == (0,1,2,3)
# Edge order: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
_B_K4 = np.array(
    [
        [-1, -1, -1,  0,  0,  0],  # Vertex 0: Governance
        [ 1,  0,  0, -1, -1,  0],  # Vertex 1: Information
        [ 0,  1,  0,  1,  0, -1],  # Vertex 2: Inference
        [ 0,  0,  1,  0,  1,  1],  # Vertex 3: Intelligence
    ],
    dtype=np.float64,
)

# Exact K4 projection matrices (computed from B^T B / 4)
# For K4: L = B B^T = 4I - J, L^+ = (1/4)I - (1/16)J
# Since JB = 0, we have P_grad = (1/4) B^T B
# This is exact (no pinv/SVD tolerance drift)
_P_GRAD_K4 = 0.25 * np.array(
    [
        [ 2,  1,  1, -1, -1,  0],
        [ 1,  2,  1,  1,  0, -1],
        [ 1,  1,  2,  0,  1,  1],
        [-1,  1,  0,  2,  1, -1],
        [-1,  0,  1,  1,  2,  1],
        [ 0, -1,  1, -1,  1,  2],
    ],
    dtype=np.float64,
)

_P_CYCLE_K4 = np.eye(6, dtype=np.float64) - _P_GRAD_K4

# Exact K4 cycle basis (integer basis for ker(B), dimension 3)
# Three triangle cycles: 0-1-2-0, 0-1-3-0, 0-2-3-0
_CYCLE_BASIS_K4 = np.array(
    [
        [ 1,  1,  0],  # Edge 0: Gov-Info
        [-1,  0,  1],  # Edge 1: Gov-Infer
        [ 0, -1, -1],  # Edge 2: Gov-Intel
        [ 1,  0,  0],  # Edge 3: Info-Infer
        [ 0,  1,  0],  # Edge 4: Info-Intel
        [ 0,  0,  1],  # Edge 5: Infer-Intel
    ],
    dtype=np.float64,
)

# Normalize cycle basis columns to unit norm (exact: each has 3 non-zero ±1 entries)
# All three columns have norm exactly sqrt(3), so multiply by 1/sqrt(3)
_INV_SQRT3 = 1.0 / np.sqrt(3.0)
_CYCLE_BASIS_K4 *= _INV_SQRT3


# Public API exports
__all__ = [
    "compute_aperture",
    "DomainLedgers",
    "get_cycle_basis",
    "get_incidence_matrix",
    "get_projections",
    "hodge_decomposition",
]


def get_incidence_matrix() -> Mat46:
    """
    GGG exact K4 incidence matrix B (4x6).
    
    Returns the canonical incidence matrix (constant, no computation).
    """
    return _B_K4.copy()


def get_projections() -> Tuple[Mat66, Mat66]:
    """
    GGG exact K4 projection matrices.
    
    Returns (P_grad, P_cycle) as exact constants (no pinv/SVD computation).
    These are audit-grade invariants: identical across all implementations.
    """
    return _P_GRAD_K4.copy(), _P_CYCLE_K4.copy()


def get_cycle_basis() -> NDArray[np.float64]:
    """
    GGG exact K4 cycle basis for ker(B).
    
    Returns shape (6,3) with columns normalized to unit norm.
    This is an exact integer basis (no SVD computation).
    """
    return _CYCLE_BASIS_K4.copy()


def hodge_decomposition(y: EdgeVec, P_grad: Mat66, P_cycle: Mat66) -> Tuple[EdgeVecFloat, EdgeVecFloat]:
    """
    GGG simulator-export exact Hodge decomposition:
      y = y_grad + y_cycle
    
    where y_grad ∈ Im(B^T) and y_cycle ∈ ker(B), orthogonal w.r.t. standard inner product.
    
    Input y is integer micro-units, output is float (projection matrices are float constants).
    """
    y_float = np.asarray(y, dtype=np.float64).reshape(6)
    y_grad = P_grad @ y_float
    y_cycle = P_cycle @ y_float
    return y_grad, y_cycle


def compute_aperture(y: EdgeVec, y_cycle: EdgeVecFloat) -> float:
    """
    GGG simulator-export exact unweighted aperture:
      A = ||y_cycle||^2 / ||y||^2
        = (y_cycle^T y_cycle) / (y^T y)
    
    Input y is integer micro-units, y_cycle is float from Hodge decomposition.
    Final ratio is float (exact representation not needed for final output).
    """
    y_float = np.asarray(y, dtype=np.float64).reshape(6)
    y_cycle = np.asarray(y_cycle, dtype=np.float64).reshape(6)
    y_norm_sq = float(y_float @ y_float)
    if y_norm_sq == 0.0:
        return 0.0
    y_cycle_norm_sq = float(y_cycle @ y_cycle)
    return y_cycle_norm_sq / y_norm_sq


def construct_edge_vector_with_aperture(
    x: NDArray[np.float64],
    target_aperture: Optional[float] = None,
    cycle_basis_vector: Optional[EdgeVecFloat] = None,
) -> EdgeVecFloat:
    """
    GGG exact construction with W=I:
      y_grad0 = B^T x
      G = ||y_grad0||^2
      C = A*G/(1-A)
      y = y_grad0 + k*u  (u is unit-norm cycle basis direction)
    
    Includes the G≈0 safeguard from the export.
    Uses exact K4 incidence matrix and cycle basis (no B parameter needed).
    """
    x = np.asarray(x, dtype=np.float64).reshape(4)
    y_grad0 = _B_K4.T @ x
    G = float(y_grad0 @ y_grad0)

    if target_aperture is None or target_aperture == 0:
        return y_grad0.astype(np.float64)

    A = float(target_aperture)
    if not (0.0 < A < 1.0):
        raise ValueError("target_aperture must be in (0,1)")

    # Choose cycle direction u (unit norm)
    if cycle_basis_vector is not None:
        u = np.asarray(cycle_basis_vector, dtype=np.float64).reshape(6)
        norm_sq = float(u @ u)
        if norm_sq <= 0:
            raise ValueError("cycle_basis_vector has zero norm")
        u = u / np.sqrt(norm_sq)
    else:
        # Use first column of exact cycle basis (deterministic)
        u = _CYCLE_BASIS_K4[:, 0]

    # G≈0 safeguard (export behavior)
    if G < 1e-10:
        x_scale = float(np.linalg.norm(x))
        if x_scale < 1e-10:
            x_scale = 1.0
        G = (x_scale ** 2) * 0.01

        grad_dir = _B_K4.T[:, 0]
        grad_norm_sq = float(grad_dir @ grad_dir)
        if grad_norm_sq > 0:
            y_grad0 = grad_dir * np.sqrt(G / grad_norm_sq)
        else:
            y_grad0 = np.ones(6, dtype=np.float64) * np.sqrt(G / 6.0)

    k_sq = (A / (1.0 - A)) * G
    k = np.sqrt(k_sq)
    c = k * u
    return y_grad0 + c


@dataclass
class LedgerSnapshot:
    y_econ: EdgeVec
    y_emp: EdgeVec
    y_edu: EdgeVec


class DomainLedgers:
    """
    Holds three ledgers:
      y_Econ, y_Emp, y_Edu  (each length-6 integer vector in micro-units)

    Computes unweighted Hodge projections (W=I) according to GGG simulator export.
    Confidence is encoded in GovernanceEvent.signed_value_micro().
    """

    def __init__(self) -> None:
        self._y: Dict[Domain, EdgeVec] = {
            Domain.ECONOMY: np.zeros(6, dtype=np.int64),
            Domain.EMPLOYMENT: np.zeros(6, dtype=np.int64),
            Domain.EDUCATION: np.zeros(6, dtype=np.int64),
        }

        # GGG exact projection matrices (constants, no computation)
        self._P_grad: Mat66
        self._P_cycle: Mat66
        self._P_grad, self._P_cycle = get_projections()

        # Event counter for deterministic replay / debugging
        self.event_count: int = 0

    def get(self, domain: Domain) -> EdgeVec:
        return self._y[domain].copy()

    def apply_event(self, ev: GovernanceEvent) -> None:
        """
        Apply a GovernanceEvent as a sparse update to y_D.
        """
        d = ev.domain
        e = int(ev.edge_id)
        self._y[d][e] += ev.signed_value_micro()
        self.event_count += 1

    def aperture(self, domain: Domain) -> float:
        """
        GGG aperture for a domain ledger y_D:
          A = ||y_cycle||^2 / ||y||^2
        
        Uses unweighted inner product (W=I).
        Follows GGG simulator export exactly.
        """
        y = self._y[domain]
        _, y_cycle = hodge_decomposition(y, self._P_grad, self._P_cycle)
        return compute_aperture(y, y_cycle)

    def decompose(self, domain: Domain) -> Tuple[EdgeVecFloat, EdgeVecFloat]:
        """
        Return (y_grad, y_cycle) using unweighted Hodge decomposition.
        Follows GGG simulator export exactly.
        Returns float arrays (projection matrices are float constants).
        """
        y = self._y[domain]
        return hodge_decomposition(y, self._P_grad, self._P_cycle)

    def snapshot(self) -> LedgerSnapshot:
        return LedgerSnapshot(
            y_econ=self.get(Domain.ECONOMY),
            y_emp=self.get(Domain.EMPLOYMENT),
            y_edu=self.get(Domain.EDUCATION),
        )
