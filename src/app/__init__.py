"""App-layer coordination and domain ledgers."""

# Explicit exports for type checkers
from .ledger import (
    compute_aperture,
    get_cycle_basis,
    get_incidence_matrix,
    get_projections,
    hodge_decomposition,
)

__all__ = [
    "compute_aperture",
    "get_cycle_basis",
    "get_incidence_matrix",
    "get_projections",
    "hodge_decomposition",
]
