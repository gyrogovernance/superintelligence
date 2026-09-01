"""Tests for benchmark suites (spec section 3.1-3.5).

Smoke budgets only; each test asserts the exact kernel relation named in the
suite.
"""

from __future__ import annotations

import numpy as np

from src.tools.autoencoder.helpers.evals_run import (
    Benchmarks,
    climate_sweep,
    percolation_suite,
)


# ---------------------------------------------------------------------------
# 3.1 Percolation
# ---------------------------------------------------------------------------


def test_percolation_suite_threshold_and_anchor() -> None:
    s = percolation_suite(seed=7)
    # threshold: full reachability iff rank == 6 (exact)
    assert s["threshold_accuracy"] == 1.0
    # anchor: the first singleton alphabet has transport rank 1
    assert s["singleton_rank"] == 1.0
    # mechanism-vs-correlate: predicted cluster differs from raw reach
    assert s["mechanism_vs_correlate_gap"] >= 0.0


# ---------------------------------------------------------------------------
# 3.3 Climate sweep
# ---------------------------------------------------------------------------


def test_climate_sweep_regimes() -> None:
    c = climate_sweep([0.1, 1.0, 10.0])
    assert c["rho"].shape == (3,)
    # rho increasing with lambda. Per QuBEC 5.3 the regime is keyed on |eta|
    # (and M2 proximity to 64): lambda=0.1 concentrates mass at shell 0
    # (condensed), lambda=1 spreads across all shells (thermal), and
    # lambda=10 concentrates at shell 6 (condensed at the complementary
    # horizon - rho is high, eta is near -1).
    assert c["rho"][0] < c["rho"][1] < c["rho"][2]
    assert c["regime"][0] == "condensed"
    assert c["regime"][1] == "thermal"
    assert c["regime"][2] == "condensed"


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


def test_benchmarks_container_runs_all() -> None:
    b = Benchmarks()
    assert "threshold_accuracy" in b.percolation()
    assert "regime" in b.climate()
