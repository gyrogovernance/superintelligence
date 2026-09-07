"""Tests for scale items (spec section 4.1-4.5).

Smoke budgets; the exhaustive verifier (4.4) is marked slow and skipped by
default. Exact kernel relations are asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.tools.autoencoder.helpers.evals_metrics import (
    ScaleSuite,
    genomics_compile,
    operator_structure,
)


def test_operator_structure_commutant() -> None:
    o = operator_structure()
    # 64 diagonal + 2016 off-diagonal = 2080 commutant dimensions
    assert int(o["commutant_dim"][0]) == 2080
    assert o["block_id"].shape == (64, 64)


def test_genomics_compile_public_columns() -> None:
    g = genomics_compile([0x12, 0x34, 0x56])
    assert g["family"].shape == (3,)
    assert g["mask12"].shape == (3,)
    # values are the public census columns, not re-implemented masks
    from src.tools.autoencoder.datasets import byte_census_arrays

    census = byte_census_arrays()
    assert np.array_equal(g["mask12"], census["mask12"][np.array([0x12, 0x34, 0x56])])


def test_scalesuite_runs() -> None:
    s = ScaleSuite()
    assert "commutant_dim" in s.operator()
    g = s.genomics([1, 2, 3])
    assert g["q6"].shape == (3,)


@pytest.mark.slow
def test_exhaustive_full_g_verify() -> None:
    # 4096 x 8192 equivariance at full scale; slow, run explicitly.
    from src.tools.autoencoder.helpers.evals_run import exhaustive_full_g_verify

    r = exhaustive_full_g_verify()
    assert r["passed"]


def test_verify_full_g_exhaustive_cli_handler() -> None:
    """The closed-form verifier is sub-second, so the handler is a fast test
    (no checkpoint certifies the architecture class)."""
    from src.tools.autoencoder.cli import cmd_verify_full_g_exhaustive

    class _Args:
        checkpoint = None
        max_err = 1e-3
        report_file = None

    rc = cmd_verify_full_g_exhaustive(_Args())
    assert rc == 0
