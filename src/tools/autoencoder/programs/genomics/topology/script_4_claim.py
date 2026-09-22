#!/usr/bin/env python3
"""Verify and record the kernel mean-shell membrane-topology result.

Primary result: kernel mean shell. Frozen Narrow scores are verified on the
same cohort and published as the AE feature source for the held-out
topology read. The claim stage refuses to write a package unless the atlas,
amino-acid null, and Narrow artifacts identify the same accession population
and all recorded source, script, cohort, and production-checkpoint hashes verify.
"""
from __future__ import annotations

from typing import Any, cast

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_HERE = _SCRIPT.parent
_REPO = _SCRIPT.parents[6]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_STAGES = _SCRIPT.parent

_TOP = _SCRIPT.parent
if str(_TOP) not in sys.path:
    sys.path.insert(0, str(_TOP))
if str(_STAGES) not in sys.path:
    sys.path.insert(0, str(_STAGES))

from common import (  # noqa: E402
    RESULT_JSON,
    upsert_results,
    write_json,
    as_dict,
    as_list,
    require,
)

import numpy as np

from src.tools.autoencoder import paths
from src.tools.autoencoder.programs.genomics.synthesis.constellation import (  # noqa: E402
    production_checkpoint_paths,
)

TOPOLOGY_DIR = paths.dataset_dir("topology")

_PASS_MEAN = -0.12
_PASS_FRAC = 0.70
_PASS_P = 0.01
_PASS_RESID = -0.03
_PASS_RESID_P = 0.05
_KERNEL_CLAIM = (
    "Within the same E. coli integral membrane protein, transmembrane-helix "
    "coding sequence has a lower kernel adjacent-codon-pair mean shell than "
    "the cytoplasmic domains of that gene."
)

_GATE_BARS = {
    "kernel_mean_shell_delta_at_most_minus_0_12": "mean gene contrast <= -0.12",
    "kernel_mean_shell_negative_fraction_at_least_0_70": "fraction of genes with a negative contrast >= 0.70",
    "kernel_mean_shell_perm_p_one_sided_less_below_0_01": "one-sided permutation p < 0.01",
    "kernel_mean_shell_AA_residual_at_most_minus_0_03": "mean residual beyond the amino-acid null <= -0.03",
    "kernel_mean_shell_AA_residual_p_one_sided_less_below_0_05": "residual permutation p < 0.05",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




def canonical_cohort_hash(
    accessions: list[str],
    cds_sha256: str,
    uniprot_sha256: str,
    min_aa: int,
) -> str:
    material = {
        "accessions": sorted(accessions),
        "cds_sha256": cds_sha256,
        "min_aa": int(min_aa),
        "uniprot_sha256": uniprot_sha256,
    }
    payload = json.dumps(material, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def atomic_write_pair(json_path: Path, txt_path: Path, payload: dict[str, Any], text: str) -> None:
    json_tmp = json_path.with_name(f".{json_path.name}.{os.getpid()}.tmp")
    txt_tmp = txt_path.with_name(f".{txt_path.name}.{os.getpid()}.tmp")
    try:
        with json_tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        with txt_tmp.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(json_tmp, json_path)
        os.replace(txt_tmp, txt_path)
    finally:
        for temp_path in (json_tmp, txt_tmp):
            if temp_path.exists():
                temp_path.unlink()


def validate_identity(
    identity: dict[str, Any],
    *,
    path: Path | None = None,
    sha256: str | None = None,
    label: str,
) -> tuple[Path, str]:
    require(identity.get("path"), f"{label} provenance is missing path")
    require(identity.get("sha256"), f"{label} provenance is missing sha256")
    resolved = Path(str(identity["path"]))
    require(resolved.is_file(), f"missing {label}: {resolved}")
    actual_hash = sha256_file(resolved)
    require(
        str(identity["sha256"]) == actual_hash,
        f"{label} hash does not match the file on disk",
    )
    if path is not None:
        require(resolved.resolve() == path.resolve(), f"{label} provenance path does not match the dependency")
    if sha256 is not None:
        require(str(identity["sha256"]) == sha256, f"{label} hash does not match the atlas record")
    return resolved, actual_hash


def verify_script(
    script_info: dict[str, Any],
    expected_path: Path,
    label: str,
) -> tuple[Path, str]:
    require(script_info.get("path") and script_info.get("sha256"), f"{label} provenance is missing script identity")
    script_path = Path(str(script_info["path"]))
    require(script_path.is_file(), f"missing {label}: {script_path}")
    require(script_path.resolve() == expected_path.resolve(), f"{label} script identity is not the corrected script")
    script_hash = sha256_file(script_path)
    require(script_hash == str(script_info["sha256"]), f"{label} script hash does not match the file on disk")
    return script_path, script_hash


def verify_checkpoints(
    checkpoint_provenance: dict[str, Any],
    expected_checkpoints: dict[str, Path],
    label: str,
) -> dict[str, str]:
    require(set(checkpoint_provenance) == set(expected_checkpoints), f"{label} production checkpoint set is incomplete")
    hashes: dict[str, str] = {}
    for name, expected_path in expected_checkpoints.items():
        recorded = as_dict(
            checkpoint_provenance[name],
            f"{label} checkpoint {name} is missing identity",
        )
        require(recorded.get("path") and recorded.get("sha256"), f"{label} checkpoint {name} is missing identity")
        recorded_path = Path(str(recorded["path"]))
        require(recorded_path.is_file(), f"missing {label} production checkpoint: {recorded_path}")
        require(recorded_path.resolve() == expected_path.resolve(), f"{label} checkpoint {name} path does not match production")
        recorded_hash = sha256_file(recorded_path)
        require(recorded_hash == str(recorded["sha256"]), f"{label} checkpoint {name} hash does not match the checkpoint file")
        hashes[name] = recorded_hash
    return hashes


def validate_atlas(
    atlas: dict[str, Any],
    atlas_path: Path,
    cds_path: Path,
    uniprot_path: Path,
) -> dict[str, Any]:
    require(atlas.get("status") == "ok", "atlas status is not ok")
    require(atlas.get("cohort_manifest", {}).get("sha256"), "atlas is missing cohort_manifest.sha256")
    atlas_provenance = as_dict(atlas.get("provenance", {}), "atlas is missing provenance")
    atlas_cli = as_dict(atlas_provenance.get("cli", {}), "atlas provenance is missing cli")
    require(atlas_cli.get("min_aa") is not None, "atlas provenance is missing cli.min_aa")
    min_aa = int(atlas_cli["min_aa"])
    require(min_aa >= 1, "atlas min_aa is invalid")
    selected_accessions = as_list(
        atlas.get("selected_accessions"), "atlas is missing selected_accessions"
    )
    genes = as_list(atlas.get("genes"), "atlas is missing genes")
    gene_accessions = [str(gene.get("accession", "")) for gene in genes]
    require(all(gene_accessions), "atlas gene is missing an accession")
    require(len(gene_accessions) == len(set(gene_accessions)), "atlas genes contain duplicate accessions")
    require(set(map(str, selected_accessions)) == set(gene_accessions), "atlas selected accessions and genes disagree")
    require(int(atlas.get("n_genes", -1)) == len(genes), "atlas n_genes does not match genes")
    require(bool(atlas.get("windowing")), "atlas is missing windowing description")
    atlas_estimator = as_dict(
        atlas.get("estimator"), "atlas is missing estimator description"
    )
    require(
        atlas_estimator.get("clipping_enabled") is False,
        "atlas coordinate policy enables span clipping",
    )
    require(
        atlas_estimator.get("coordinate_policy")
        == "UniProt 1-based inclusive spans must fall wholly within the translated CDS; invalid or out-of-range spans are excluded and never clipped",
        "atlas coordinate policy is not exclusion-not-clipping",
    )
    atlas_counts = atlas.get("counts", {})
    require(int(atlas_counts.get("clipped_topology_spans", -1)) == 0, "atlas reports clipped topology spans")
    require(int(atlas_counts.get("genes_with_clipped_topology_spans", -1)) == 0, "atlas reports genes with clipped topology spans")
    require("topology_span_clipping" not in atlas, "atlas contains a topology_span_clipping payload")
    atlas_exclusions = atlas.get("exclusions", {})
    require("clipping_details" not in atlas_exclusions, "atlas contains a clipping payload")
    permutation = atlas.get("permutation", {})
    require(permutation.get("shell_alternative") == "less", "atlas shell permutation alternative is not fixed to less")
    require(permutation.get("super_alternative") == "two-sided nonzero", "atlas Super permutation alternative is not labeled two-sided")
    require(bool(atlas.get("aa_null_reference")), "atlas is missing AA-null reference")

    atlas_path, atlas_sha256 = validate_identity(
        {"path": str(atlas_path), "sha256": sha256_file(atlas_path)},
        label="atlas",
    )
    atlas_inputs = as_dict(atlas_provenance.get("inputs", {}), "atlas provenance is missing inputs")
    cds_path, cds_sha256 = validate_identity(
        as_dict(atlas_inputs.get("cds", {}), "atlas provenance is missing CDS identity"),
        path=cds_path,
        label="atlas CDS",
    )
    uniprot_path, uniprot_sha256 = validate_identity(
        as_dict(atlas_inputs.get("uniprot", {}), "atlas provenance is missing UniProt identity"),
        path=uniprot_path,
        label="atlas UniProt",
    )
    atlas_script_path, atlas_script_sha256 = verify_script(
        as_dict(atlas_provenance.get("script", {}), "atlas provenance is missing script identity"),
        _HERE / "script_1_atlas.py",
        "atlas",
    )
    expected_checkpoints = production_checkpoint_paths()
    checkpoint_hashes = verify_checkpoints(
        as_dict(
            atlas_provenance.get("production_checkpoints", {}),
            "atlas provenance is missing production checkpoints",
        ),
        expected_checkpoints,
        "atlas",
    )
    cohort_hash = canonical_cohort_hash(
        [str(accession) for accession in selected_accessions],
        cds_sha256,
        uniprot_sha256,
        min_aa,
    )
    require(
        str(atlas["cohort_manifest"]["sha256"]) == cohort_hash,
        "atlas cohort hash does not match accessions and source digests",
    )
    return {
        "atlas_path": atlas_path,
        "atlas_sha256": atlas_sha256,
        "cds_path": cds_path,
        "cds_sha256": cds_sha256,
        "uniprot_path": uniprot_path,
        "uniprot_sha256": uniprot_sha256,
        "atlas_script_path": atlas_script_path,
        "atlas_script_sha256": atlas_script_sha256,
        "checkpoint_hashes": checkpoint_hashes,
        "cohort_hash": cohort_hash,
        "min_aa": min_aa,
        "selected_accessions": [str(accession) for accession in selected_accessions],
        "genes": genes,
        "expected_checkpoints": expected_checkpoints,
    }


def gate_report(name: str, *, bar: str, observed: float, passed: bool) -> dict[str, Any]:
    """State the bar, the observed value, and the open residual for every gate."""
    return {
        "gate": name,
        "bar": bar,
        "observed": float(observed),
        "passed": bool(passed),
        "status": "met" if passed else "missed",
        "open_residual": (
            ""
            if passed
            else f"{name} misses its bar; the retained record states the bar and the observed value, and the miss is an open residual on this estimand."
        ),
    }


def validate_aa_null(
    aa_null: dict[str, Any],
    aa_null_path: Path,
    atlas: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    require(aa_null.get("status") == "ok", "AA-null status is not ok")
    atlas_n = int(audit["selected_accessions"] and len(audit["selected_accessions"]))
    aa_n = int(aa_null.get("n", -1))
    recorded_atlas_n = int(aa_null.get("atlas_n", -2))
    require(aa_n == atlas_n, "AA-null n does not match atlas n")
    require(recorded_atlas_n == atlas_n, "AA-null atlas_n does not match atlas n")
    require(int(aa_null.get("n", -1)) == len(aa_null.get("genes", [])), "AA-null n does not match its per-gene population")
    require(aa_null.get("excluded") == {}, "AA-null excludes atlas population genes")
    require(aa_null.get("cohort_manifest_sha256") == str(audit["cohort_hash"]), "AA-null cohort hash does not match atlas cohort")
    require(aa_null.get("estimator", {}).get("alternative") == "less", "AA-null permutation alternative is not fixed to less")
    aa_genes = as_list(aa_null.get("genes", []), "AA-null is missing genes")
    aa_accessions = [str(gene.get("accession", "")) for gene in aa_genes]
    require(len(aa_accessions) == len(set(aa_accessions)), "AA-null genes contain duplicate accessions")
    require(set(aa_accessions) == set(audit["selected_accessions"]), "AA-null population differs from atlas population")

    aa_provenance = as_dict(aa_null.get("provenance", {}), "AA-null is missing provenance")
    aa_script_path, aa_script_sha256 = verify_script(
        as_dict(aa_provenance.get("script", {}), "AA-null provenance is missing script identity"),
        _HERE / "script_2_null.py",
        "AA-null",
    )
    aa_sources = as_dict(aa_null.get("sources", {}), "AA-null is missing sources")
    aa_atlas_path, aa_atlas_hash = validate_identity(
        as_dict(aa_sources.get("atlas", {}), "AA-null sources are missing atlas identity"),
        path=Path(str(audit["atlas_path"])),
        sha256=str(audit["atlas_sha256"]),
        label="AA-null atlas",
    )
    aa_cds_path, aa_cds_hash = validate_identity(
        as_dict(aa_sources.get("cds", {}), "AA-null sources are missing CDS identity"),
        path=Path(str(audit["cds_path"])),
        sha256=str(audit["cds_sha256"]),
        label="AA-null CDS",
    )
    aa_uniprot_path, aa_uniprot_hash = validate_identity(
        as_dict(aa_sources.get("uniprot", {}), "AA-null sources are missing UniProt identity"),
        path=Path(str(audit["uniprot_path"])),
        sha256=str(audit["uniprot_sha256"]),
        label="AA-null UniProt",
    )
    source_digests = as_dict(
        aa_provenance.get("source_digests", {}), "AA-null provenance is missing source digests"
    )
    require(str(source_digests.get("atlas")) == aa_atlas_hash, "AA-null atlas source digest does not match the atlas")
    require(str(source_digests.get("cds")) == aa_cds_hash, "AA-null CDS source digest does not match the CDS")
    require(str(source_digests.get("uniprot")) == aa_uniprot_hash, "AA-null UniProt source digest does not match UniProt")
    require(str(source_digests.get("atlas_script")) == str(audit["atlas_script_sha256"]), "AA-null atlas-script digest does not match the atlas script")
    aa_checkpoint_hashes = verify_checkpoints(
        as_dict(
            aa_provenance.get("atlas_production_checkpoints", {}),
            "AA-null provenance is missing atlas production checkpoints",
        ),
        cast(dict[str, Path], audit["expected_checkpoints"]),
        "AA-null",
    )
    require(aa_checkpoint_hashes == audit["checkpoint_hashes"], "AA-null checkpoint hashes differ from atlas")
    aa_null_sha256 = sha256_file(aa_null_path)
    return {
        "aa_null_path": aa_null_path,
        "aa_null_sha256": aa_null_sha256,
        "aa_null_script_path": aa_script_path,
        "aa_null_script_sha256": aa_script_sha256,
    }


def validate_narrow(
    narrow: dict[str, Any],
    narrow_path: Path,
    atlas: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    require(narrow.get("status") == "ok", "Narrow status is not ok")
    require(narrow.get("uses_frozen_constellation") is True, "Narrow does not declare frozen constellation use")
    require(int(narrow.get("n_genes", -1)) == len(audit["selected_accessions"]), "Narrow n_genes does not match atlas")
    require(int(narrow.get("atlas_n", -1)) == len(audit["selected_accessions"]), "Narrow atlas_n does not match atlas")
    require(int(narrow.get("n_AA_null_genes", -1)) == len(audit["selected_accessions"]), "Narrow AA-null population does not match atlas")
    require(narrow.get("exclusions", {}).get("counts") == {}, "Narrow excludes atlas population genes")
    require(narrow.get("cohort_manifest_sha256") == str(audit["cohort_hash"]), "Narrow cohort hash does not match atlas cohort")
    narrow_population = as_list(
        narrow.get("population_accessions"), "Narrow is missing population_accessions"
    )
    require(set(map(str, narrow_population)) == set(audit["selected_accessions"]), "Narrow population differs from atlas population")
    require(narrow.get("permutation", {}).get("alternative") == "less", "Narrow permutation alternative is not fixed to less")
    require(narrow.get("estimator", {}).get("windowing") == atlas.get("windowing"), "Narrow windowing does not match atlas")
    narrow_stats = as_dict(narrow.get("stats", {}), "Narrow is missing stats")
    atlas_stats = as_dict(atlas.get("stats", {}), "atlas is missing stats")
    require(abs(float(narrow_stats.get("mean_delta_narrow_tm_minus_cyto", float("nan"))) - float(atlas_stats.get("mean_delta_narrow_tm_minus_cyto", float("nan")))) <= 1e-12, "Narrow mean does not match atlas")
    require(abs(float(narrow_stats.get("mean_delta_gc_tm_minus_cyto", float("nan"))) - float(atlas_stats.get("mean_delta_gc_tm_minus_cyto", float("nan")))) <= 1e-12, "Narrow GC mean does not match atlas")
    require(abs(float(narrow_stats.get("kernel_mean_delta_shell_tm_minus_cyto", float("nan"))) - float(atlas_stats.get("mean_delta_shell_tm_minus_cyto", float("nan")))) <= 1e-12, "Narrow kernel shell mean does not match atlas")

    narrow_provenance = as_dict(narrow.get("provenance", {}), "Narrow is missing provenance")
    narrow_script_path, narrow_script_sha256 = verify_script(
        as_dict(narrow_provenance.get("script", {}), "Narrow provenance is missing script identity"),
        _HERE / "script_3_narrow.py",
        "Narrow",
    )
    narrow_atlas = as_dict(narrow_provenance.get("atlas", {}), "Narrow provenance is missing atlas identity")
    require(narrow_atlas.get("path") and narrow_atlas.get("sha256"), "Narrow provenance is missing atlas identity")
    narrow_atlas_path = Path(str(narrow_atlas["path"]))
    require(narrow_atlas_path.is_file(), f"missing Narrow atlas dependency: {narrow_atlas_path}")
    require(narrow_atlas_path.resolve() == Path(str(audit["atlas_path"])).resolve(), "Narrow atlas provenance path does not match claim atlas")
    require(str(narrow_atlas["sha256"]) == str(audit["atlas_sha256"]), "Narrow atlas hash does not match claim atlas")
    narrow_inputs = as_dict(narrow_provenance.get("inputs", {}), "Narrow provenance is missing inputs")
    narrow_cds = validate_identity(
        as_dict(narrow_inputs.get("cds", {}), "Narrow provenance is missing CDS identity"),
        path=Path(str(audit["cds_path"])),
        label="Narrow CDS",
    )
    narrow_uniprot = validate_identity(
        as_dict(narrow_inputs.get("uniprot", {}), "Narrow provenance is missing UniProt identity"),
        path=Path(str(audit["uniprot_path"])),
        label="Narrow UniProt",
    )
    require(str(narrow_cds[1]) == str(audit["cds_sha256"]), "Narrow CDS hash does not match atlas")
    require(str(narrow_uniprot[1]) == str(audit["uniprot_sha256"]), "Narrow UniProt hash does not match atlas")
    narrow_checkpoint_hashes = verify_checkpoints(
        as_dict(
            narrow_provenance.get("production_checkpoints", {}),
            "Narrow provenance is missing production checkpoints",
        ),
        cast(dict[str, Path], audit["expected_checkpoints"]),
        "Narrow",
    )
    require(narrow_checkpoint_hashes == audit["checkpoint_hashes"], "Narrow checkpoint hashes differ from atlas")
    narrow_sha256 = sha256_file(narrow_path)
    return {
        "narrow_path": narrow_path,
        "narrow_sha256": narrow_sha256,
        "narrow_script_path": narrow_script_path,
        "narrow_script_sha256": narrow_script_sha256,
    }


def validate_sources(
    atlas: dict[str, Any],
    aa_null: dict[str, Any],
    narrow: dict[str, Any],
    atlas_path: Path,
    aa_null_path: Path,
    narrow_path: Path,
    cds_path: Path,
    uniprot_path: Path,
) -> dict[str, Any]:
    atlas_audit = validate_atlas(atlas, atlas_path, cds_path, uniprot_path)
    aa_audit = validate_aa_null(aa_null, aa_null_path, atlas, atlas_audit)
    narrow_audit = validate_narrow(narrow, narrow_path, atlas, atlas_audit)
    return {**atlas_audit, **aa_audit, **narrow_audit}


def main() -> int:
    atlas_path = RESULT_JSON["atlas"].resolve()
    aa_null_path = RESULT_JSON["null"].resolve()
    narrow_path = RESULT_JSON["narrow"].resolve()
    cds_path = (paths.dataset_dir("genomics") / "ecoli_k12_cds.fna.gz").resolve()
    uniprot_path = (TOPOLOGY_DIR / "uniprot_ecoli_k12.tsv").resolve()
    for path in (atlas_path, aa_null_path, narrow_path, cds_path, uniprot_path):
        if not path.is_file():
            print(f"missing claim dependency: {path}", file=sys.stderr)
            return 1

    try:
        atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
        aa_null = json.loads(aa_null_path.read_text(encoding="utf-8"))
        narrow = json.loads(narrow_path.read_text(encoding="utf-8"))
        audit = validate_sources(
            atlas,
            aa_null,
            narrow,
            atlas_path,
            aa_null_path,
            narrow_path,
            cds_path,
            uniprot_path,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"claim audit failed: {error}", file=sys.stderr)
        return 1

    genes = atlas["genes"]
    d_shell = [float(gene["delta_shell_tm_minus_cyto"]) for gene in genes]
    d_climate = [float(gene["delta_climate_tm_minus_cyto"]) for gene in genes]
    d_narrow = [float(gene["delta_narrow_tm_minus_cyto"]) for gene in genes]
    d_gc = [float(gene["tm_gc"]) - float(gene["cyto_gc"]) for gene in genes]

    mean_shell = float(np.mean(d_shell))
    mean_climate = float(np.mean(d_climate))
    mean_narrow = float(np.mean(d_narrow))
    mean_gc = float(np.mean(d_gc))
    frac_neg = float(np.mean(np.asarray(d_shell) < 0))
    shell_p = float(atlas["stats"]["shell_perm_p_one_sided_less"])
    climate_p_two_sided = float(atlas["stats"]["climate_perm_p_two_sided_nonzero"])
    narrow_stats = narrow["stats"]
    narrow_p = float(narrow_stats["narrow_perm_p_one_sided_less"])
    mean_resid = float(aa_null["mean_resid_atlas_minus_null"])
    resid_p = float(aa_null["resid_perm_p_one_sided_less"])
    narrow_mean_resid = float(narrow_stats["mean_narrow_resid_beyond_AA"])
    narrow_resid_p = float(narrow_stats["narrow_resid_perm_p_one_sided_less"])
    narrow_usage_mean_resid = float(narrow_stats["mean_narrow_resid_beyond_usage_matched_AA"])
    narrow_usage_resid_p = float(
        narrow_stats["narrow_usage_matched_resid_perm_p_one_sided_less"]
    )

    gates = {
        "kernel_mean_shell_delta_at_most_minus_0_12": bool(mean_shell <= _PASS_MEAN),
        "kernel_mean_shell_negative_fraction_at_least_0_70": bool(frac_neg >= _PASS_FRAC),
        "kernel_mean_shell_perm_p_one_sided_less_below_0_01": bool(shell_p < _PASS_P),
        "kernel_mean_shell_AA_residual_at_most_minus_0_03": bool(mean_resid <= _PASS_RESID),
        "kernel_mean_shell_AA_residual_p_one_sided_less_below_0_05": bool(resid_p < _PASS_RESID_P),
    }
    passed = all(gates.values())

    gate_reports: dict[str, Any] = {}
    observed_by_gate: dict[str, float] = {
        "kernel_mean_shell_delta_at_most_minus_0_12": mean_shell,
        "kernel_mean_shell_negative_fraction_at_least_0_70": frac_neg,
        "kernel_mean_shell_perm_p_one_sided_less_below_0_01": shell_p,
        "kernel_mean_shell_AA_residual_at_most_minus_0_03": mean_resid,
        "kernel_mean_shell_AA_residual_p_one_sided_less_below_0_05": resid_p,
    }
    for name, value in gates.items():
        gate_reports[name] = gate_report(
            name,
            bar=_GATE_BARS[name],
            observed=observed_by_gate[name],
            passed=bool(value),
        )

    claim = _KERNEL_CLAIM if passed else None
    summary = {
        "status": "ok",
        "claim": claim,
        "primary_claim": "kernel_mean_shell",
        "ae_claim_stage": "holdout",
        "ae_claim": (
            "Held-out incremental AUC of frozen Narrow plus Super reads over "
            "kernel plus biological controls on per-window topology labels."
        ),
        "passed": passed,
        "gates": gates,
        "gate_reports": gate_reports,
        "gate_reporting_rule": (
            "Primary gates are the kernel mean-shell bars. The AE topology "
            "claim is the held-out incremental read in script_5_holdout.py."
        ),
        "pass_bars_fixed_before_eval": {
            "kernel_mean_shell": _PASS_MEAN,
            "kernel_negative_fraction": _PASS_FRAC,
            "kernel_perm_p": _PASS_P,
            "kernel_AA_residual": _PASS_RESID,
            "kernel_AA_residual_p": _PASS_RESID_P,
        },
        "population": {
            "n_genes": len(genes),
            "n_AA_null_genes": int(aa_null["n"]),
            "n_Narrow_AA_null_genes": int(narrow["n_AA_null_genes"]),
            "accessions": audit["selected_accessions"],
            "cohort_manifest_sha256": audit["cohort_hash"],
        },
        "stats": {
            "mean_delta_shell_tm_minus_cyto": mean_shell,
            "shell_negative_fraction": frac_neg,
            "n_shell_negative": int(sum(value < 0 for value in d_shell)),
            "shell_perm_p_one_sided_less": shell_p,
            "mean_delta_gc_tm_minus_cyto": mean_gc,
            "mean_obs_shell": float(aa_null["mean_obs"]),
            "mean_AA_null_shell": float(aa_null["mean_AA_null"]),
            "mean_shell_resid_atlas_minus_AA_null": mean_resid,
            "shell_resid_perm_p_one_sided_less": resid_p,
            "mean_delta_climate_tm_minus_cyto": mean_climate,
            "climate_perm_p_two_sided_nonzero": climate_p_two_sided,
            "mean_delta_narrow_tm_minus_cyto": mean_narrow,
            "narrow_perm_p_one_sided_less": narrow_p,
            "narrow_mean_AA_residual_atlas_minus_null": narrow_mean_resid,
            "narrow_resid_perm_p_one_sided_less": narrow_resid_p,
            "narrow_mean_usage_matched_AA_residual_atlas_minus_null": narrow_usage_mean_resid,
            "narrow_usage_matched_resid_perm_p_one_sided_less": narrow_usage_resid_p,
        },
        "ae_reads": {
            "role": "feature_source_for_holdout_topology_read",
            "claim_stage": "holdout",
            "super_climate": {
                "mean_delta_tm_minus_cyto": mean_climate,
                "perm_p_two_sided_nonzero": climate_p_two_sided,
            },
            "narrow_het": {
                "mean_delta_tm_minus_cyto": mean_narrow,
                "perm_p_one_sided_less": narrow_p,
                "mean_AA_residual_atlas_minus_null": narrow_mean_resid,
                "resid_perm_p_one_sided_less": narrow_resid_p,
                "mean_usage_matched_AA_residual_atlas_minus_null": narrow_usage_mean_resid,
                "usage_matched_resid_perm_p_one_sided_less": narrow_usage_resid_p,
            },
        },
        "datasets": {
            "cds": {"id": "dataset_genomics/ecoli_k12_cds", "filename": cds_path.name, "path": str(cds_path), "sha256": audit["cds_sha256"]},
            "uniprot": {"id": "uniprot/ecoli_k12/2026_03", "filename": uniprot_path.name, "path": str(uniprot_path), "sha256": audit["uniprot_sha256"]},
            "atlas": {"id": "membrane-topology-climate-atlas/v1", "filename": atlas_path.name, "path": str(atlas_path), "sha256": audit["atlas_sha256"], "cohort_manifest_sha256": audit["cohort_hash"]},
            "aa_null": {"id": "membrane-topology-climate-aa-null/v1", "filename": aa_null_path.name, "path": str(aa_null_path), "sha256": audit["aa_null_sha256"], "cohort_manifest_sha256": audit["cohort_hash"]},
            "narrow": {"id": "membrane-topology-narrow-ae/v1", "filename": narrow_path.name, "path": str(narrow_path), "sha256": audit["narrow_sha256"], "cohort_manifest_sha256": audit["cohort_hash"]},
        },
        "production_checkpoints": {
            name: {"id": path.name, "path": str(path), "sha256": audit["checkpoint_hashes"][name]}
            for name, path in sorted(production_checkpoint_paths().items())
        },
        "provenance": {
            "generated_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "script": {"path": str(_SCRIPT), "sha256": sha256_file(_SCRIPT)},
            "claim_script": {"path": str(_SCRIPT), "sha256": sha256_file(_SCRIPT)},
            "atlas_script": {"path": str(audit["atlas_script_path"]), "sha256": audit["atlas_script_sha256"]},
            "aa_null_script": {"path": str(audit["aa_null_script_path"]), "sha256": audit["aa_null_script_sha256"]},
            "narrow_script": {"path": str(audit["narrow_script_path"]), "sha256": audit["narrow_script_sha256"]},
            "cli": {
                "atlas": str(atlas_path),
                "aa_null": str(aa_null_path),
                "narrow": str(narrow_path),
                "cds": str(cds_path),
                "uniprot": str(uniprot_path),
            },
            "validation": {
                "atlas_accession_set_matches_genes": True,
                "atlas_source_hashes_verified": True,
                "atlas_script_hash_verified": True,
                "atlas_cohort_hash_verified": True,
                "aa_null_accession_set_matches_atlas": True,
                "aa_null_source_hashes_verified": True,
                "aa_null_script_hash_verified": True,
                "narrow_accession_set_matches_atlas": True,
                "narrow_source_hashes_verified": True,
                "narrow_script_hash_verified": True,
                "production_checkpoint_hashes_verified": True,
                "dependencies_verified_before_statistics": True,
            },
            "output_artifacts": {
                "json": str(RESULT_JSON["claim"]),
                "results": str((_HERE / "RESULTS.txt").resolve()),
            },
        },
        "estimator": {
            "primary": "per-topology-window kernel mean shell; gene contrast is mean(TM) minus mean(cytoplasmic); population statistic is mean gene contrast",
            "windowing": atlas["windowing"],
            "AA_null": aa_null["estimator"],
            "Narrow": narrow["estimator"],
        },
    }

    out_json = RESULT_JSON["claim"]
    lines = [
        "stage=claim",
        f"status=ok n_genes={len(genes)} aa_null_genes={aa_null['n']} narrow_aa_null_genes={narrow['n_AA_null_genes']}",
        f"cohort_manifest_sha256={audit['cohort_hash']}",
        f"kernel_mean_shell_delta={mean_shell:+.4f} frac_neg={frac_neg:.3f} p_less={shell_p:.4f}",
        f"kernel_AA_residual={mean_resid:+.4f} p_less={resid_p:.4f}",
        f"gc_mean_delta={mean_gc:+.4f}",
        f"super_climate_delta={mean_climate:+.4f} p_two_sided={climate_p_two_sided:.4f}",
        f"narrow_het_delta={mean_narrow:+.4f} p_less={narrow_p:.4f}",
        f"narrow_AA_residual={narrow_mean_resid:+.4f} p_less={narrow_resid_p:.4f}",
        f"narrow_usage_matched_AA_residual={narrow_usage_mean_resid:+.4f} p_less={narrow_usage_resid_p:.4f}",
        "ae_claim_stage=holdout",
        f"passed={passed}",
        "primary_gates:",
    ]
    for key, value in gates.items():
        lines.append(f"  {key}: {value}")
    lines.append("missed_gate_reports:")
    missed = [report for report in gate_reports.values() if not report["passed"]]
    if not missed:
        lines.append("  every primary gate is met")
    for report in missed:
        lines.append(
            f"  {report['gate']}: bar={report['bar']} observed={report['observed']:+.6f} "
            f"status={report['status']} open_residual={report['open_residual']}"
        )
    write_json(out_json, summary)
    upsert_results("claim", lines)
    print("\n".join(lines), flush=True)
    print(f"wrote {out_json} and RESULTS.txt")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
