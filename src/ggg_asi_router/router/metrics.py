"""
Router Metrics

Re-exports alignment metrics from the physics layer for router convenience.
"""

from __future__ import annotations

from ..physics.alignment import compute_deviation, compute_si

__all__ = ["compute_deviation", "compute_si"]


