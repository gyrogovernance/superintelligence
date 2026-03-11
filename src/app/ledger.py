"""
Application-layer domain ledgers for GGG aperture.

Stores per-domain edge vectors y_D in R^6.

The K4 Hodge decomposition, aperture definition, and target aperture A* = 0.0207
are application-layer constructs that remain valid independently of the kernel
transition law (Gyroscopic_ASI_Specs, Appendix C).

Geometry:
- Incidence matrix B (4x6) for K4 graph
- For K4 with W=I: P_grad = (1/4) * (B^T B) (exact closed form)
- Aperture: A = ||y_cycle||^2 / ||y||^2

Confidence is encoded in GovernanceEvent.signed_value_micro().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .events import Domain, GovernanceEvent

EdgeVec = NDArray[np.int64]  # Ledger vectors use integer micro-units
EdgeVecFloat = NDArray[np.float64]  # For intermediate float computations
Mat46 = NDArray[np.float64]
Mat44 = NDArray[np.float64]
Mat66 = NDArray[np.float64]


# Exact K4 incidence matrix B (4x6)
# Vertex order: (Gov, Info, Infer, Intel) == (0,1,2,3)
# Edge order: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
_B_K4 = np.array(
    [
        [-1, -1, -1,  0,  0,  0],
        [ 1,  0,  0, -1, -1,  0],
        [ 0,  1,  0,  1,  0, -1],
        [ 0,  0,  1,  0,  1,  1],
    ],
    dtype=np.float64,
)

# Exact K4 projection matrices (computed from B^T B / 4)
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
_CYCLE_BASIS_K4 = np.array(
    [
        [ 1,  1,  0],
        [-1,  0,  1],
        [ 0, -1, -1],
        [ 1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0,  1],
    ],
    dtype=np.float64,
)

_INV_SQRT3 = 1.0 / np.sqrt(3.0)
_CYCLE_BASIS_K4 *= _INV_SQRT3


__all__ = [
    "compute_aperture",
    "DomainLedgers",
    "get_cycle_basis",
    "get_incidence_matrix",
    "get_projections",
    "hodge_decomposition",
]


def get_incidence_matrix() -> Mat46:
    """Exact K4 incidence matrix B (4x6)."""
    return _B_K4.copy()


def get_projections() -> tuple[Mat66, Mat66]:
    """Exact K4 projection matrices (P_grad, P_cycle)."""
    return _P_GRAD_K4.copy(), _P_CYCLE_K4.copy()


def get_cycle_basis() -> NDArray[np.float64]:
    """Exact K4 cycle basis for ker(B), shape (6,3), unit-norm columns."""
    return _CYCLE_BASIS_K4.copy()


def hodge_decomposition(y: EdgeVec, P_grad: Mat66, P_cycle: Mat66) -> tuple[EdgeVecFloat, EdgeVecFloat]:
    """
    Hodge decomposition: y = y_grad + y_cycle.

    Input y is integer micro-units, output is float.
    """
    y_float = np.asarray(y, dtype=np.float64).reshape(6)
    y_grad = P_grad @ y_float
    y_cycle = P_cycle @ y_float
    return y_grad, y_cycle


def compute_aperture(y: EdgeVec, y_cycle: EdgeVecFloat) -> float:
    """
    Aperture: A = ||y_cycle||^2 / ||y||^2.

    Input y is integer micro-units, y_cycle is float from Hodge decomposition.
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
    target_aperture: float | None = None,
    cycle_basis_vector: EdgeVecFloat | None = None,
) -> EdgeVecFloat:
    """
    Construct edge vector y with target aperture from vertex potentials x.

    y = y_grad0 + k*u where u is a unit-norm cycle direction.
    """
    x = np.asarray(x, dtype=np.float64).reshape(4)
    y_grad0 = _B_K4.T @ x
    G = float(y_grad0 @ y_grad0)

    if target_aperture is None or target_aperture == 0:
        return y_grad0.astype(np.float64)

    A = float(target_aperture)
    if not (0.0 < A < 1.0):
        raise ValueError("target_aperture must be in (0,1)")

    if cycle_basis_vector is not None:
        u = np.asarray(cycle_basis_vector, dtype=np.float64).reshape(6)
        norm_sq = float(u @ u)
        if norm_sq <= 0:
            raise ValueError("cycle_basis_vector has zero norm")
        u = u / np.sqrt(norm_sq)
    else:
        u = _CYCLE_BASIS_K4[:, 0]

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
    Three domain ledgers: y_Econ, y_Emp, y_Edu (each length-6 integer micro-units).

    Computes unweighted Hodge projections (W=I).
    """

    def __init__(self) -> None:
        self._y: dict[Domain, EdgeVec] = {
            Domain.ECONOMY: np.zeros(6, dtype=np.int64),
            Domain.EMPLOYMENT: np.zeros(6, dtype=np.int64),
            Domain.EDUCATION: np.zeros(6, dtype=np.int64),
        }
        self._P_grad: Mat66
        self._P_cycle: Mat66
        self._P_grad, self._P_cycle = get_projections()
        self.event_count: int = 0

    def get(self, domain: Domain) -> EdgeVec:
        return self._y[domain].copy()

    def apply_event(self, ev: GovernanceEvent) -> None:
        """Apply a GovernanceEvent as a sparse update to y_D."""
        d = ev.domain
        e = int(ev.edge_id)
        self._y[d][e] += ev.signed_value_micro()
        self.event_count += 1

    def aperture(self, domain: Domain) -> float:
        """Aperture A = ||y_cycle||^2 / ||y||^2 for a domain ledger."""
        y = self._y[domain]
        _, y_cycle = hodge_decomposition(y, self._P_grad, self._P_cycle)
        return compute_aperture(y, y_cycle)

    def decompose(self, domain: Domain) -> tuple[EdgeVecFloat, EdgeVecFloat]:
        """Return (y_grad, y_cycle) for a domain ledger."""
        y = self._y[domain]
        return hodge_decomposition(y, self._P_grad, self._P_cycle)

    def snapshot(self) -> LedgerSnapshot:
        return LedgerSnapshot(
            y_econ=self.get(Domain.ECONOMY),
            y_emp=self.get(Domain.EMPLOYMENT),
            y_edu=self.get(Domain.EDUCATION),
        )