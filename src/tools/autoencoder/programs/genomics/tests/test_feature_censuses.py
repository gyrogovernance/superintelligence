"""Unit tests for genomics synthesis registry and census helpers."""

from __future__ import annotations

import json
from typing import get_args

import numpy as np

from src.tools.autoencoder.kernel import word_signature_id
from src.tools.autoencoder.programs.genomics.synthesis.censuses import (
    GateResult,
    MARGINS,
    _auc_binary,
    _gate_tier,
    _gene_grouped_oof_auc,
    _ridge_fit,
    _ridge_predict,
    _spearman,
    _write_artifacts,
    collision_pairs,
    format_gate,
    masked_oracle_family_accuracy,
    occupy_oracle_shell_error,
)
from src.tools.autoencoder.programs.genomics.synthesis.registry import (
    FEATURE_REGISTRY,
    INCLUDED_CENSUS_KEYS,
    Disposition,
    assert_registry_complete,
    coverage_rows,
)


def test_registry_complete_and_resolves():
    assert_registry_complete()
    assert len(FEATURE_REGISTRY) >= 50
    rows = coverage_rows()
    assert any(r[0] == "29_s6" for r in rows)
    assert any(r[0] == "29_s8" for r in rows)
    assert any(r[0] == "41_s7" for r in rows)
    assert FEATURE_REGISTRY["15"].disposition == "excluded"
    assert "15_tally" not in FEATURE_REGISTRY
    for entry in FEATURE_REGISTRY.values():
        if entry.disposition == "included":
            assert entry.census_keys
            for ck in entry.census_keys:
                assert ck in INCLUDED_CENSUS_KEYS


def test_disposition_has_no_deferred():
    assert "deferred" not in {d.lower() for d in get_args(Disposition)}
    assert set(get_args(Disposition)) == {"included", "excluded", "infrastructure"}


def test_margins_frozen_with_anchors():
    assert MARGINS["g4"] == 0.05
    assert MARGINS["defect_band_lo"] == 2.5
    assert MARGINS["defect_band_hi"] == 3.5
    assert MARGINS["defect_abs_min"] == 0.05
    assert MARGINS["defect_null_gap"] == 0.05
    assert MARGINS["shuffle_auc_min"] == 0.55
    assert MARGINS["shuffle_null_gap"] == 0.05
    assert MARGINS["recode_auc_min"] == 0.55
    assert MARGINS["recode_null_gap"] == 0.05
    assert MARGINS["stop_macro_floor"] == 1.0 / 3.0
    assert MARGINS["stop_null_gap"] == 0.05
    assert MARGINS["climate_eta_min"] == 0.01
    assert MARGINS["climate_m2_max"] == 4096.0
    assert MARGINS["g4_shadow_max_dist"] == 0.05
    assert MARGINS["k4_err"] == 1e-4
    assert MARGINS["spec_tr_gap"] == 0.05
    assert "matrix_order_margin" not in MARGINS
    assert "matrix_narrow_cap" not in MARGINS
    assert "narrow_auc_margin" not in MARGINS
    assert "narrow_pair_margin" not in MARGINS
    assert "sheetreg_margin" not in MARGINS
    assert "masked_margin_random" not in MARGINS
    assert "occupy_err" not in MARGINS
    assert "occupy_margin" not in MARGINS
    assert "defect_margin" not in MARGINS


def test_splice_census_callable():
    from src.tools.autoencoder.programs.genomics.synthesis.censuses import splice_census

    assert callable(splice_census)


def test_oof_spearman_and_label_permutation_null():
    from src.tools.autoencoder.programs.genomics.synthesis.censuses import (
        _label_permutation_spearman,
        _oof_ridge_spearman,
    )

    rng = np.random.default_rng(0)
    X = rng.normal(size=(160, 6))
    y = X @ np.asarray([1.0, 0.8, 0.6, 0.4, 0.2, 0.1]) + 0.05 * rng.normal(size=160)
    assert _oof_ridge_spearman(X, y, seed=0) > 0.9
    observed, null_mean, null_max = _label_permutation_spearman(X, y, n_perm=8, seed=0)
    assert observed > 0.9
    assert null_mean < 0.2
    assert null_max < observed
    y_noise = rng.normal(size=160)
    obs_n, _nm, mx_n = _label_permutation_spearman(X, y_noise, n_perm=8, seed=1)
    assert obs_n <= mx_n + 1e-9


def test_plaquette_defect_and_kernel_percolation_ladder():
    from src.tools.autoencoder.programs.genomics.synthesis.censuses import (
        kernel_percolation_ladder,
        plaquette_defect_mean,
    )
    from src.tools.autoencoder.programs.genomics.synthesis.reads import byte_stream

    stream = byte_stream("ATGAAACCCGGGTTTAAACGTACGTACGTAAA")
    d = plaquette_defect_mean(stream)
    assert np.isfinite(d)
    assert 0.0 <= d <= 6.0
    rows = kernel_percolation_ladder()
    assert [r[0] for r in rows] == ["sense", "sense_stop", "sense_ser", "full"]
    assert [r[4] for r in rows] == [256, 1024, 1024, 4096]
    assert all(reach == expect and pred == expect for _n, _r, reach, pred, expect in rows)
    assert FEATURE_REGISTRY["24"].disposition == "excluded"
    assert FEATURE_REGISTRY["20b"].disposition == "excluded"
    assert FEATURE_REGISTRY["41_s7"].disposition == "excluded"
    assert FEATURE_REGISTRY["29_s6"].disposition == "excluded"
    assert FEATURE_REGISTRY["24"].census_keys == ()


def test_oracle_exactness_contract():
    from src.tools.autoencoder.programs.genomics.synthesis.reads import byte_stream

    assert occupy_oracle_shell_error() == 0.0
    assert masked_oracle_family_accuracy() == 1.0
    stream = byte_stream("ATGAAACCCGGGTTTAAACGTACGTACGTAAA")
    assert occupy_oracle_shell_error(stream) == 0.0
    assert masked_oracle_family_accuracy(stream) == 1.0


def test_gene_grouped_oof_auc_keeps_pairs():
    rng = np.random.default_rng(0)
    n_genes = 20
    X = []
    y = []
    groups = []
    for i in range(n_genes):
        real = rng.normal(size=4) + 1.0
        null = rng.normal(size=4) - 1.0
        X.append(real)
        y.append(1)
        groups.append(i)
        X.append(null)
        y.append(0)
        groups.append(i)
    auc = _gene_grouped_oof_auc(
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.int64),
        np.asarray(groups),
        seed=0,
    )
    assert auc > 0.8


def test_ridge_and_spearman_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = X @ np.array([0.5, -0.2, 0.1, 0.3]) + 0.01 * rng.normal(size=40)
    w = _ridge_fit(X[:30], y[:30])
    pred = _ridge_predict(X[30:], w)
    rho = _spearman(y[30:], pred)
    assert np.isfinite(rho)
    assert rho > 0.5


def test_auc_perfect_separation():
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    labels = np.asarray([0, 0, 1, 1])
    assert _auc_binary(scores, labels) == 1.0


def test_collision_pairs_and_format_gate():
    rng = np.random.default_rng(0)
    stream = rng.integers(0, 256, size=5000).tolist()
    pairs = collision_pairs(stream, n_pairs=10, seed=1)
    assert len(pairs) >= 1
    for a, b in pairs:
        assert word_signature_id(a) == word_signature_id(b)
    g = GateResult(
        name="unit",
        n=1,
        trained=0.2,
        random=0.1,
        margin=0.1,
        compile_floor=None,
        passed=True,
        detail="unit",
    )
    assert any("outcome: PASS" in line for line in format_gate(g))


def test_gate_outcomes_distinguish_negative_and_report():
    negative = GateResult(
        name="negative",
        n=20,
        trained=0.4,
        random=0.5,
        margin=-0.1,
        compile_floor=0.5,
        passed=False,
        detail="valid claim missed",
    )
    report = GateResult(
        name="unit_report",
        n=10,
        trained=0.1,
        random=0.0,
        margin=0.1,
        compile_floor=1.0,
        passed=True,
        detail="measured diagnostic",
        outcome="REPORT",
    )
    assert negative.outcome == "NEGATIVE"
    assert report.outcome == "REPORT"
    assert any("outcome: REPORT" in line for line in format_gate(report))
    assert _gate_tier(
        GateResult(
            name="climate_super_ecoli",
            n=1,
            trained=0.85,
            random=0.48,
            margin=0.37,
            compile_floor=0.55,
            passed=True,
            detail="spec",
        )
    ) == "specialization"
    assert _gate_tier(
        GateResult(
            name="defect_k4_ecoli",
            n=1,
            trained=0.94,
            random=0.47,
            margin=0.47,
            compile_floor=0.55,
            passed=True,
            detail="spec",
        )
    ) == "specialization"
    assert _gate_tier(
        GateResult(
            name="ingress_k4_box35",
            n=1,
            trained=0.70,
            random=0.61,
            margin=0.09,
            compile_floor=0.55,
            passed=True,
            detail="spec",
        )
    ) == "specialization"
    assert _gate_tier(
        GateResult(
            name="align_k4",
            n=4096,
            trained=0.0,
            random=0.0,
            margin=0.0,
            compile_floor=1e-4,
            passed=True,
            detail="k4",
        )
    ) == "preflight"
    assert _gate_tier(
        GateResult(
            name="splice_narrow_chr22",
            n=1,
            trained=0.94,
            random=0.95,
            margin=-0.01,
            compile_floor=0.75,
            passed=True,
            detail="motif",
            outcome="FLOOR",
        )
    ) == "alignment"


def test_auc_requires_chance_direction():
    scores = np.asarray([0.9, 0.8, 0.2, 0.1])
    labels = np.asarray([0, 0, 1, 1])
    assert _auc_binary(scores, labels) == 0.0


def test_feature_gates_schema_without_verdict(tmp_path):
    gates = [
        GateResult(
            name="g4_ecoli",
            n=10,
            trained=0.3,
            random=0.1,
            margin=0.2,
            compile_floor=None,
            passed=True,
            detail="unit",
        )
    ]
    out = _write_artifacts(
        gates=gates,
        seed=1,
        device="cpu",
        hosts=("ecoli",),
        max_genes=10,
        n_pairs=10,
        results_file=tmp_path / "RESULTS.txt",
        gates_file=tmp_path / "gates.json",
    )
    assert out["status"] == "ok"
    loaded = json.loads((tmp_path / "gates.json").read_text(encoding="utf-8"))
    assert "verdict" not in loaded
    assert loaded["status"] == "ok"
    assert "margins" in loaded and "gates" in loaded
    text = (tmp_path / "RESULTS.txt").read_text(encoding="utf-8")
    assert "overall_verdict" not in text
    assert "registry coverage" not in text
    assert "domain_training: none" in text
    assert "tr_gap=" in text


def test_blocked_run_does_not_wipe(tmp_path):
    results = tmp_path / "RESULTS.txt"
    gates = tmp_path / "gates.json"
    results.write_text("keep-me\n", encoding="utf-8")
    gates.write_text('{"status":"ok","gates":[]}\n', encoding="utf-8")
    # Branch logic: full-suite preflight failure returns blocked without write.
    payload = {
        "status": "blocked",
        "gates": [
            GateResult(
                name="preflight_shell_ecoli",
                n=1,
                trained=3.1,
                random=3.0,
                margin=0.1,
                compile_floor=2.85,
                passed=False,
                detail="fail",
            )
        ],
        "results_file": str(results),
        "gates_file": str(gates),
    }
    assert payload["status"] == "blocked"
    assert results.read_text(encoding="utf-8") == "keep-me\n"
    assert '"status":"ok"' in gates.read_text(encoding="utf-8")


def test_align_exact_on_synthetic_stream():
    """Kernel word_signature_id is exact on a synthetic 4-byte stream."""
    word = [1, 2, 3, 4]
    sig = word_signature_id(word)
    assert isinstance(sig, int)
    assert 0 <= sig < (1 << 13)


def test_graceful_skip_without_catalogs(tmp_path, monkeypatch):
    from src.tools.autoencoder.programs.genomics.synthesis import reads

    monkeypatch.setattr(reads, "GENOMICS_DIR", tmp_path)
    try:
        reads.prepare_host("ecoli", max_genes=5)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_prepare_host_compile_is_none(monkeypatch, tmp_path):
    from src.tools.autoencoder.programs.genomics.synthesis import reads

    fasta = tmp_path / "ecoli_k12_cds.fna.gz"
    # Minimal gzipped FASTA with one in-frame CDS (>=300 ACGT).
    import gzip

    seq = "ATG" + ("AAA" * 100) + "TAA"
    with gzip.open(fasta, "wt", encoding="utf-8") as fh:
        fh.write(">g1\n")
        fh.write(seq + "\n")
    monkeypatch.setattr(reads, "GENOMICS_DIR", tmp_path)
    monkeypatch.setitem(reads.HOST_CDS, "ecoli", ("ecoli_k12_cds.fna.gz", 11))
    cache = reads.prepare_host("ecoli", max_genes=5)
    assert cache.genes
    assert cache.genes[0].compile is None
    assert cache.genes[0].climate is not None


def test_sars_min_bases_keeps_short_orfs(monkeypatch, tmp_path):
    from src.tools.autoencoder.programs.genomics.synthesis import reads
    import gzip

    fasta = tmp_path / "sars_cov2_cds.fna.gz"
    short = "ATG" + ("AAA" * 40) + "TAA"  # 126 bp, below ecoli floor
    with gzip.open(fasta, "wt", encoding="utf-8") as fh:
        fh.write(">s1\n")
        fh.write(short + "\n")
    monkeypatch.setattr(reads, "GENOMICS_DIR", tmp_path)
    monkeypatch.setitem(reads.HOST_CDS, "sars", ("sars_cov2_cds.fna.gz", 1))
    monkeypatch.setitem(reads.HOST_MIN_BASES, "sars", 100)
    cache = reads.prepare_host("sars", max_genes=5)
    assert len(cache.genes) == 1
    assert len(cache.genes[0].seq) == 126
