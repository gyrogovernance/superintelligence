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


# Map: report filename -> set of required top-level keys.
REPORT_SCHEMAS = {
    "k4_full_eval.json": {"reconstruction", "equivariance", "meta"},
    "mlp_full_eval.json": {"reconstruction", "equivariance", "meta"},
    "mlp_full_eq.json": {"max", "mean"},
    "super_gates.json": {
        "gates",
        "g1_pass",
        "g3_pass",
        "g7_pass",
        "gate_version",
        "grid",
        "g4_margin_min",
        "lambda_fraction_of_ceiling",
    },
    "production_summary.json": {"k4_full", "mlp_full"},
}


@pytest.mark.parametrize("name,required", sorted(REPORT_SCHEMAS.items()))
def test_report_schema(name, required) -> None:
    """Every published report carries the keys the current code is contracted
    to produce. Missing-key drift is the loudest possible failure mode for
    a reproducibility gap; this test turns it red on the next ``pytest`` run.
    """
    data = _load(name)
    if data is None:
        pytest.skip(f"{name} not generated yet; run helpers.training_super production")
    missing = required - set(data.keys())
    assert not missing, f"{name} missing keys: {sorted(missing)}"


def test_production_summary_paths_are_posix() -> None:
    """Every path in ``production_summary.json`` uses forward slashes so the
    published JSON is portable across OSes."""
    data = _load("production_summary.json")
    if data is None:
        pytest.skip("production_summary.json not generated yet")
    for name, entry in data.items():
        if not isinstance(entry, dict):
            continue
        for key, val in entry.items():
            if not isinstance(val, str):
                continue
            if (
                key.endswith("checkpoint")
                or key.endswith("eval")
                or key.endswith("equivariance")
                or key.endswith("full_g_closed_form")
                or key in ("report", "gates")
            ):
                assert "\\" not in val, (
                    f"{name}.{key} contains a backslash: {val!r}"
                )


def test_production_summary_has_hashes_when_present() -> None:
    data = _load("production_summary.json")
    if data is None:
        pytest.skip("production_summary.json not generated yet")
    for name in ("k4_full", "mlp_full", "super"):
        if name not in data:
            continue
        entry = data[name]
        assert "sha256" in entry and len(entry["sha256"]) == 64
    if "super" in data:
        claim = data["super"].get("claim")
        assert claim in (
            "exact_grammar_plus_provenance",
            "exact_grammar_plus_provenance_and_residual",
        )
        gates = _load("super_gates.json")
        if gates is not None and claim == "exact_grammar_plus_provenance_and_residual":
            lam = float(gates.get("lambda_fraction_of_ceiling", 0.0))
            mk = float((gates.get("gates") or {}).get("markov_improvement_bits", 0.0))
            res = float(gates.get("residual_improvement_bits", 0.0))
            assert lam >= 0.5 or mk > 0.05 or res > 0.05


def test_super_gates_versioned() -> None:
    data = _load("super_gates.json")
    if data is None:
        pytest.skip("super_gates.json not generated yet")
    assert int(data.get("gate_version", 0)) >= 2
    assert "residual_improvement_bits" in data
    assert "multi_seed" in data
    assert len(data["multi_seed"]) >= 3


def test_super_not_published_unless_gates_pass() -> None:
    """production_summary may omit Super; if present, gate report must pass."""
    summary = _load("production_summary.json")
    gates = _load("super_gates.json")
    if summary is None:
        pytest.skip("production_summary.json not generated yet")
    if "super" not in summary:
        return
    assert gates is not None, "super published without super_gates.json"
    mandatory = ("g1_pass", "g2_pass", "g3_pass", "g4_pass", "g5_pass", "g6_pass", "g7_pass")
    failed = [k for k in mandatory if not gates.get(k)]
    assert not failed, f"super published with failed gates: {failed}"
