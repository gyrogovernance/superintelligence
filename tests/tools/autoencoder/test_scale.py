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
from src.tools.autoencoder.models.super import MultiCellSpectral


def test_operator_structure_commutant() -> None:
    o = operator_structure()
    # 64 diagonal + 2016 off-diagonal = 2080 commutant dimensions
    assert int(o["commutant_dim"][0]) == 2080
    assert o["block_id"].shape == (64, 64)


def test_multicell_joint_walsh_deterministic() -> None:
    rng = np.random.default_rng(0)
    err = MultiCellSpectral(2).equivariance_check(rng)
    assert err < 1e-9


def test_multicell_product_register_equivariance() -> None:
    # B-cell product register: equivariance under a cell swap is exact (Kronecker
    # construction), and the joint spectrum is a genuine tensor product whose
    # concentration returns the trivial/low-band fractions.
    import torch

    m = MultiCellSpectral(2)
    rng = np.random.default_rng(1)
    c0 = torch.as_tensor(rng.integers(0, 4096, size=8), dtype=torch.long)
    c1 = torch.as_tensor(rng.integers(0, 4096, size=8), dtype=torch.long)
    rep = m.product_equivariance_check([c0, c1])
    assert rep < 1e-9
    conc = m.concentration(m.joint_spectrum([c0, c1]))
    assert 0.0 <= conc["trivial_fraction"] <= 1.0
    assert 0.0 <= conc["low_band_any_cell_fraction"] <= 1.0


def test_genomics_compile_public_columns() -> None:
    g = genomics_compile([0x12, 0x34, 0x56])
    assert g["family"].shape == (3,)
    assert g["mask12"].shape == (3,)
    # values are the public census columns, not re-implemented masks
    from src.tools.autoencoder.datasets import byte_census_arrays

    census = byte_census_arrays()
    assert np.array_equal(g["mask12"], census["mask12"][np.array([0x12, 0x34, 0x56])])


def test_scalesuite_runs() -> None:
    import torch

    s = ScaleSuite()
    assert "commutant_dim" in s.operator()
    g = s.genomics([1, 2, 3])
    assert g["q6"].shape == (3,)
    rng = np.random.default_rng(2)
    cells = [
        torch.as_tensor(rng.integers(0, 4096, size=4), dtype=torch.long) for _ in range(2)
    ]
    prod = s.multicell_product(cells)
    assert prod["equivariance_max_err"] < 1e-9


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
