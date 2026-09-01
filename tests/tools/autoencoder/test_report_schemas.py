"""Pin the production report schemas so any drift between reports and code
fails loudly under ``pytest`` rather than silently shipping.

These tests do not retrain anything. They read whatever reports are
present under ``src/tools/autoencoder/data/reports/`` and assert each one
has the keys the CLI currently writes, with the right types. The point is
that any change to ``cmd_evaluate`` / ``cmd_verify_*`` that drops or
renames a field will break the suite and force a coordinated update of
the production driver and the published reports.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPORTS = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "tools"
    / "autoencoder"
    / "data"
    / "reports"
)


def _load(name: str) -> dict | None:
    p = REPORTS / name
    if not p.exists():
        return None
    import json
    return json.loads(p.read_text())


# Map: report filename -> set of required top-level keys. We intentionally
# only pin the keys the publishable consumer needs (audit fields, pass flag,
# checkpoint path). Internal fields (model config, training log) are
# permitted but not asserted.
REPORT_SCHEMAS = {
    "spectral_full_eval.json": {"reconstruction", "equivariance", "meta"},
    "spectral_bottleneck_eval.json": {"reconstruction", "equivariance", "meta"},
    "spectral_denoise_eval.json": {"reconstruction", "equivariance", "meta"},
    "k4_full_eval.json": {"reconstruction", "equivariance", "meta"},
    "mlp_full_eval.json": {"reconstruction", "equivariance", "meta"},
    "spectral_denoise_train.json": {
        "max_abs_error", "mean_abs_error", "pass", "tol", "n_gains",
    },
    "spectral_full_fullg.json": {
        "max_error", "algebraic_max_asymmetry", "numerical_sampled_max", "passed",
    },
    "spectral_bottleneck_fullg.json": {
        "max_error", "algebraic_max_asymmetry", "numerical_sampled_max", "passed",
    },
    "spectral_denoise_fullg.json": {
        "max_error", "algebraic_max_asymmetry", "numerical_sampled_max", "passed",
    },
    "production_summary.json": {
        "spectral_full", "spectral_bottleneck", "spectral_denoise", "k4_full", "mlp_full",
    },
}


@pytest.mark.parametrize("name,required", sorted(REPORT_SCHEMAS.items()))
def test_report_schema(name, required) -> None:
    """Every published report carries the keys the current code is contracted
    to produce. Missing-key drift is the loudest possible failure mode for
    a reproducibility gap; this test turns it red on the next ``pytest`` run.
    """
    data = _load(name)
    if data is None:
        pytest.skip(f"{name} not generated yet; run scripts/make_production")
    missing = required - set(data.keys())
    assert not missing, f"{name} missing keys: {sorted(missing)}"


def test_production_summary_paths_are_posix() -> None:
    """Every path in ``production_summary.json`` uses forward slashes so the
    published JSON is portable across OSes."""
    data = _load("production_summary.json")
    if data is None:
        pytest.skip("production_summary.json not generated yet")
    for name, entry in data.items():
        for key, val in entry.items():
            if key.endswith("checkpoint") or key.endswith("eval") or key.endswith("equivariance") or key.endswith("full_g_closed_form") or key == "report":
                assert "\\" not in val, (
                    f"{name}.{key} contains a backslash: {val!r}"
                )


def test_denoise_train_passes_gain_bound() -> None:
    """The retrained denoiser's gain report must pass its own machine-checked
    bound. Pinned here so any future code change that breaks the closed-form
    tracking fails on the next ``pytest`` run.
    """
    data = _load("spectral_denoise_train.json")
    if data is None:
        pytest.skip("spectral_denoise_train.json not generated yet")
    assert data.get("pass") is True, (
        f"spectral_denoise mean_abs_error={data.get('mean_abs_error')!r} "
        f"max_abs_error={data.get('max_abs_error')!r} tol={data.get('tol')!r}"
    )
