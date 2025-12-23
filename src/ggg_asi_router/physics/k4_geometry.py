"""
K4 Hodge Geometry for Router Metrics

Implements the K4 incidence structure, edge weight matrix, fixed cycle basis,
projection operators, and Hodge decomposition used to define aperture on the
Router edge-functional y.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def get_incidence_matrix_k4() -> NDArray[np.float64]:
    """
    Return the 4×6 signed incidence matrix B for the K4 graph.

    Vertex order follows the CGM stage ordering:
    0: CS, 1: UNA, 2: ONA, 3: BU.

    Edge order is:
        0: (CS, UNA)
        1: (CS, ONA)
        2: (CS, BU)
        3: (UNA, ONA)
        4: (UNA, BU)
        5: (ONA, BU)
    """
    return np.array(
        [
            [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, -1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        ],
        dtype=float,
    )


def get_weight_matrix_k4(weights: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:
    """
    Return the 6×6 edge weight matrix W for K4.

    Args:
        weights: Optional length-6 array of positive weights. If omitted, the
            identity matrix is used.
    """
    if weights is None:
        return np.eye(6, dtype=float)
    if weights.shape != (6,):
        raise ValueError("weights must have shape (6,)")
    if np.any(weights <= 0):
        raise ValueError("all weights must be positive")
    return np.diag(weights.astype(float))


def get_face_cycle_matrix_k4(W: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Return the face-cycle matrix F (6×3) for K4 triangular faces.

    Each column corresponds to one of the three triangular faces:
    - Column 0: CS-UNA-ONA cycle
    - Column 1: CS-UNA-BU cycle
    - Column 2: CS-ONA-BU cycle

    Edge order: 0=(CS,UNA), 1=(CS,ONA), 2=(CS,BU), 3=(UNA,ONA), 4=(UNA,BU), 5=(ONA,BU)

    Each face vector is normalized to unit W-norm but NOT orthogonalized, preserving
    the direct correspondence between defect channels and face cycles.

    Args:
        W: Edge weight matrix (6×6).

    Returns:
        F: Face-cycle matrix (6×3) with W-normalized columns.
    """
    # Face cycles as signed edge vectors (before normalization)
    # CS-UNA-ONA: edge 0 (CS→UNA) + edge 3 (UNA→ONA) - edge 1 (CS→ONA, reversed)
    # CS-UNA-BU:  edge 0 (CS→UNA) + edge 4 (UNA→BU) - edge 2 (CS→BU, reversed)
    # CS-ONA-BU:  edge 1 (CS→ONA) + edge 5 (ONA→BU) - edge 2 (CS→BU, reversed)
    F_raw = np.array(
        [
            [1.0, 1.0, 0.0],   # edge 0: (CS, UNA)
            [-1.0, 0.0, 1.0],  # edge 1: (CS, ONA)
            [0.0, -1.0, -1.0], # edge 2: (CS, BU)
            [1.0, 0.0, 0.0],   # edge 3: (UNA, ONA)
            [0.0, 1.0, 0.0],   # edge 4: (UNA, BU)
            [0.0, 0.0, 1.0],   # edge 5: (ONA, BU)
        ],
        dtype=float,
    )

    # Normalize each column to unit W-norm (but do NOT orthogonalize)
    F = np.zeros_like(F_raw)
    for j in range(F_raw.shape[1]):
        v = F_raw[:, j]
        norm_sq = float(v.T @ W @ v)
        norm = float(np.sqrt(max(norm_sq, 0.0)))
        if norm == 0:
            raise ValueError(f"Face cycle {j} has zero W-norm")
        F[:, j] = v / norm

    return F


def compute_projections_k4(
    B: NDArray[np.float64],
    W: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute gradient and cycle projection matrices for K4 using face-cycle matrix F.

    Uses the basis-invariant projection formula:
    P_cycle = F (F^T W F)^{-1} F^T W
    P_grad = I - P_cycle

    Args:
        B: Incidence matrix (unused, kept for API compatibility).
        W: Edge weight matrix.

    Returns:
        P_grad: 6×6 projection onto the gradient subspace.
        P_cycle: 6×6 projection onto the cycle subspace.
    """
    F = get_face_cycle_matrix_k4(W)
    
    # Compute P_cycle = F (F^T W F)^{-1} F^T W
    # Use solve() instead of inv() for numerical stability
    F_T_W = F.T @ W  # (3×6)
    F_T_W_F = F_T_W @ F  # (3×3)
    P_cycle = F @ np.linalg.solve(F_T_W_F, F_T_W)  # (6×6)
    
    P_grad = np.eye(6, dtype=float) - P_cycle
    return P_grad, P_cycle


def hodge_decomposition(
    y: NDArray[np.float64],
    P_grad: NDArray[np.float64],
    P_cycle: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Perform Hodge decomposition of a K4 edge vector y.

    Returns:
        y_grad: gradient component P_grad y.
        y_cycle: cycle component P_cycle y.
    """
    y_grad = P_grad @ y
    y_cycle = P_cycle @ y
    return y_grad, y_cycle


def aperture(
    y: NDArray[np.float64],
    y_cycle: NDArray[np.float64],
    W: NDArray[np.float64],
) -> float:
    """
    Compute aperture A = ||y_cycle||²_W / ||y||²_W in [0, 1].

    The W-norm is defined by ||v||²_W = vᵀ W v on the six K4 edges.
    """
    y_norm_sq = float(y.T @ W @ y)
    if y_norm_sq == 0.0:
        return 0.0
    y_cycle_norm_sq = float(y_cycle.T @ W @ y_cycle)
    return float(y_cycle_norm_sq / y_norm_sq)


