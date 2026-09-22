#!/usr/bin/env python3
"""Amino-acid-preserving null for the exact topology climate atlas cohort."""
from __future__ import annotations

from typing import Any

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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
    as_int,
    as_list,
    codon_usage_counts,
    dinuc_controlled_contrast,
    junction_matched_choice_counts,
    read_fasta_accession_records,
    require,
    shell_value,
    synonymize,
    synonymize_junction_matched,
    usage_weighted_synonymize,
)

from src.tools.autoencoder import paths

TOPOLOGY_DIR = paths.dataset_dir("topology")
from src.tools.autoencoder.programs.genomics.genomics import (  # noqa: E402
    GENOMICS_DIR,
    clean_acgt,
)
from src.tools.autoencoder.programs.genomics.synthesis.climate import (  # noqa: E402
    codons_of,
    mean_adjacent_shell,
)
from src.tools.autoencoder.programs.genomics.synthesis.constellation import (  # noqa: E402
    production_checkpoint_paths,
)


def load_uniprot_accessions(path: Path) -> dict[str, dict[str, str]]:
    """Index UniProt TSV rows by Entry accession."""
    by_accession: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            accession = str(row.get("Entry", "")).strip()
            if not accession:
                continue
            if accession in by_accession:
                raise ValueError(f"duplicate UniProt accession {accession}")
            by_accession[accession] = dict(row)
    return by_accession


def build_host_usage(sequences: Iterable[str]) -> dict[str, float]:
    """Aggregate codon counts across a catalog of coding sequences."""
    total: dict[str, int] = {}
    for sequence in sequences:
        for codon, count in codon_usage_counts(sequence).items():
            total[codon] = total.get(codon, 0) + count
    require(bool(total), "host codon usage is empty")
    return {codon: float(count) for codon, count in total.items()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def negative_tail_p(
    residuals: np.ndarray,
    n_perm: int,
    seed: int,
) -> float:
    """Test the fixed alternative that the atlas-minus-null residual is less than zero."""
    observed = float(residuals.mean())
    rng = np.random.Generator(np.random.PCG64(seed))
    hits = 0
    for _ in range(n_perm):
        signs = rng.integers(0, 2, size=residuals.size, dtype=np.int8) * 2 - 1
        if float((residuals * signs).mean()) <= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def canonical_cohort_hash(
    selected_accessions: list[str],
    cds_sha256: str,
    uniprot_sha256: str,
    min_aa: int,
) -> str:
    material = {
        "accessions": sorted(selected_accessions),
        "cds_sha256": cds_sha256,
        "min_aa": int(min_aa),
        "uniprot_sha256": uniprot_sha256,
    }
    payload = json.dumps(material, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()




def validate_atlas(
    atlas: dict[str, Any],
    atlas_path: Path,
    cds_path: Path,
    uniprot_path: Path,
    min_aa: int,
) -> dict[str, Any]:
    selected_accessions = as_list(
        atlas.get("selected_accessions"), "atlas is missing selected_accessions"
    )
    genes = as_list(atlas.get("genes"), "atlas is missing genes")
    selected_set = {str(accession) for accession in selected_accessions}
    require(
        len(selected_set) == len(selected_accessions),
        "atlas selected_accessions contains duplicates",
    )
    gene_accessions = [str(gene.get("accession", "")) for gene in genes]
    require(all(gene_accessions), "atlas gene is missing an accession")
    require(
        len(set(gene_accessions)) == len(gene_accessions),
        "atlas genes contain duplicate accessions",
    )
    require(selected_set == set(gene_accessions), "atlas selected_accessions and genes disagree")
    require(
        int(atlas.get("n_genes", -1)) == len(selected_accessions),
        "atlas n_genes does not match selected_accessions",
    )
    atlas_counts = as_dict(atlas.get("counts", {}), "atlas is missing counts")
    require(
        int(atlas_counts.get("selected", -1)) == len(selected_accessions),
        "atlas counts.selected does not match selected_accessions",
    )

    atlas_provenance = as_dict(atlas.get("provenance", {}), "atlas is missing provenance")
    atlas_cli = as_dict(atlas_provenance.get("cli", {}), "atlas provenance is missing cli")
    atlas_min_aa = atlas_cli.get("min_aa")
    require(atlas_min_aa is not None, "atlas provenance is missing cli.min_aa")
    require(as_int(atlas_min_aa, "atlas provenance cli.min_aa is invalid") == int(min_aa), "AA-null --min-aa does not match atlas provenance")

    atlas_inputs = as_dict(atlas_provenance.get("inputs", {}), "atlas provenance is missing inputs")
    atlas_cds = as_dict(atlas_inputs.get("cds", {}), "atlas provenance is missing CDS identity")
    atlas_uniprot = as_dict(
        atlas_inputs.get("uniprot", {}), "atlas provenance is missing UniProt identity"
    )
    require(atlas_cds.get("path") and atlas_cds.get("sha256"), "atlas provenance is missing CDS identity")
    require(atlas_uniprot.get("path") and atlas_uniprot.get("sha256"), "atlas provenance is missing UniProt identity")
    atlas_cds_path = Path(str(atlas_cds["path"]))
    atlas_uniprot_path = Path(str(atlas_uniprot["path"]))
    require(atlas_cds_path.is_absolute(), "atlas CDS provenance path is not absolute")
    require(atlas_uniprot_path.is_absolute(), "atlas UniProt provenance path is not absolute")
    require(
        atlas_cds_path.resolve() == cds_path.resolve(),
        "atlas CDS provenance path does not match the dependency path",
    )
    require(
        atlas_uniprot_path.resolve() == uniprot_path.resolve(),
        "atlas UniProt provenance path does not match the dependency path",
    )

    cds_digest = sha256_file(cds_path)
    uniprot_digest = sha256_file(uniprot_path)
    require(str(atlas_cds["sha256"]) == cds_digest, "atlas CDS digest does not match the CDS input")
    require(str(atlas_uniprot["sha256"]) == uniprot_digest, "atlas UniProt digest does not match the UniProt input")

    expected_hash = canonical_cohort_hash(
        [str(accession) for accession in selected_accessions],
        cds_digest,
        uniprot_digest,
        min_aa,
    )
    atlas_manifest = as_dict(atlas.get("cohort_manifest", {}), "atlas is missing cohort_manifest")
    atlas_hash = atlas_manifest.get("sha256")
    require(bool(atlas_hash), "atlas is missing cohort_manifest.sha256")
    require(
        str(atlas_hash) == expected_hash,
        "atlas cohort hash does not match selected accessions and source digests",
    )

    atlas_script = as_dict(atlas_provenance.get("script", {}), "atlas provenance is missing script")
    atlas_script_hash = atlas_script.get("sha256")
    require(bool(atlas_script_hash), "atlas provenance is missing script.sha256")
    atlas_script_path = Path(str(atlas_script.get("path", "")))
    require(atlas_script_path.is_file(), f"missing atlas script: {atlas_script_path}")
    require(
        atlas_script_path.resolve()
        == (_HERE / "script_1_atlas.py").resolve(),
        "atlas script identity does not match the corrected atlas script",
    )
    require(
        sha256_file(atlas_script_path) == str(atlas_script_hash),
        "atlas script hash does not match the atlas script on disk",
    )

    atlas_checkpoints = as_dict(
        atlas_provenance.get("production_checkpoints", {}),
        "atlas provenance is missing production checkpoints",
    )
    expected_checkpoints = production_checkpoint_paths()
    require(
        set(atlas_checkpoints) == set(expected_checkpoints),
        "atlas production checkpoint set is incomplete",
    )
    checkpoint_hashes: dict[str, str] = {}
    for name, expected_path in expected_checkpoints.items():
        recorded = as_dict(atlas_checkpoints[name], f"checkpoint {name} is missing identity")
        require(recorded.get("path") and recorded.get("sha256"), f"checkpoint {name} is missing identity")
        recorded_path = Path(str(recorded["path"]))
        require(recorded_path.is_file(), f"missing production checkpoint: {recorded_path}")
        require(recorded_path.resolve() == expected_path.resolve(), f"checkpoint {name} path does not match production")
        recorded_hash = sha256_file(recorded_path)
        require(recorded_hash == str(recorded["sha256"]), f"checkpoint {name} hash does not match the checkpoint file")
        checkpoint_hashes[name] = recorded_hash

    permutation = as_dict(atlas.get("permutation", {}), "atlas is missing permutation")
    require(permutation.get("shell_alternative") == "less", "atlas shell permutation alternative is not fixed to less")
    require(
        permutation.get("super_alternative") == "two-sided nonzero",
        "atlas Super permutation alternative is not labeled two-sided",
    )
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
    atlas_counts = as_dict(atlas.get("counts", {}), "atlas is missing counts")
    require(int(atlas_counts.get("clipped_topology_spans", -1)) == 0, "atlas reports clipped topology spans")
    require(int(atlas_counts.get("genes_with_clipped_topology_spans", -1)) == 0, "atlas reports genes with clipped topology spans")
    require("topology_span_clipping" not in atlas, "atlas contains a topology_span_clipping payload")
    atlas_exclusions = as_dict(atlas.get("exclusions", {}), "atlas is missing exclusions")
    require("clipping_details" not in atlas_exclusions, "atlas contains a clipping payload")
    require(
        isinstance(atlas_exclusions.get("span_details"), list),
        "atlas is missing span exclusion details",
    )

    return {
        "selected_accessions": [str(accession) for accession in selected_accessions],
        "cohort_hash": expected_hash,
        "atlas_sha256": sha256_file(atlas_path),
        "cds_sha256": cds_digest,
        "uniprot_sha256": uniprot_digest,
        "min_aa": int(min_aa),
        "checkpoint_hashes": checkpoint_hashes,
    }


def window_signature(windows: list[dict[str, Any]]) -> list[tuple[int, int, int, str]]:
    signature: list[tuple[int, int, int, str]] = []
    for window in windows:
        require(isinstance(window.get("start_aa"), int), "atlas window is missing start_aa")
        require(isinstance(window.get("end_aa"), int), "atlas window is missing end_aa")
        require(isinstance(window.get("nt"), int), "atlas window is missing nt")
        require(bool(window.get("sequence_sha256")), "atlas window is missing sequence_sha256")
        signature.append(
            (
                int(window["start_aa"]),
                int(window["end_aa"]),
                int(window["nt"]),
                str(window["sequence_sha256"]),
            )
        )
    return signature


def extract_window(
    sequence: str,
    window: dict[str, Any],
    accession: str,
) -> tuple[str, float, tuple[int, int, int, str]]:
    start = int(window["start_aa"])
    end = int(window["end_aa"])
    expected_nt = int(window["nt"])
    fragment = clean_acgt(sequence[(start - 1) * 3 : end * 3])
    if start < 1 or end < start or len(fragment) != expected_nt or len(fragment) < 36 or len(fragment) % 3:
        raise ValueError(f"invalid exact atlas window for {accession}: {start}..{end}")
    digest = hashlib.sha256(fragment.encode("utf-8")).hexdigest()
    require(
        digest == str(window["sequence_sha256"]),
        f"atlas window sequence hash differs for {accession}: {start}..{end}",
    )
    shell = float(mean_adjacent_shell(codons_of(fragment)))
    require(
        abs(shell - float(window["mean_shell"])) <= 1e-12,
        f"atlas window mean shell differs for {accession}: {start}..{end}",
    )
    return fragment, shell, (start, end, expected_nt, digest)



def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the amino-acid-preserving null on the exact atlas cohort."
    )
    parser.add_argument("--atlas", type=Path, default=RESULT_JSON["atlas"])
    parser.add_argument("--uniprot", type=Path, default=TOPOLOGY_DIR / "uniprot_ecoli_k12.tsv")
    parser.add_argument("--cds", type=Path, default=GENOMICS_DIR / "ecoli_k12_cds.fna.gz")
    parser.add_argument("--null-draws", type=int, default=80)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--min-aa", type=int, default=12)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=RESULT_JSON["null"],
        help="RESULTS_script_2_null.json",
    )
    args = parser.parse_args()

    if args.null_draws < 80:
        parser.error("--null-draws must be at least 80")
    if args.permutations < 5000:
        parser.error("--permutations must be at least 5000")
    if args.min_aa < 1:
        parser.error("--min-aa must be at least 1")

    atlas_path = args.atlas.resolve()
    uniprot_path = args.uniprot.resolve()
    cds_path = args.cds.resolve()
    out_json = args.output_json.resolve()
    for path in (atlas_path, uniprot_path, cds_path):
        if not path.is_file():
            print(f"missing input: {path}", file=sys.stderr)
            return 1

    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    try:
        audit = validate_atlas(atlas, atlas_path, cds_path, uniprot_path, args.min_aa)
    except (KeyError, TypeError, ValueError) as error:
        print(f"atlas validation failed: {error}", file=sys.stderr)
        return 1

    selected_accessions = audit["selected_accessions"]
    by_acc = read_fasta_accession_records(cds_path)
    uniprot = load_uniprot_accessions(uniprot_path)
    host_usage = build_host_usage(by_acc.values())
    atlas_genes = {
        str(gene["accession"]): gene
        for gene in atlas["genes"]
    }
    null_rng = np.random.Generator(np.random.PCG64(args.seed))
    per_gene: list[dict[str, Any]] = []

    for accession in selected_accessions:
        atlas_gene = atlas_genes[accession]
        row = uniprot.get(accession)
        sequence = by_acc.get(accession)
        if row is None:
            raise ValueError(f"missing UniProt row for exact-cohort accession {accession}")
        if sequence is None:
            raise ValueError(f"missing CDS for exact-cohort accession {accession}")

        tm_windows = atlas_gene.get("tm_windows")
        cyto_windows = atlas_gene.get("cyto_windows")
        require(isinstance(tm_windows, list) and tm_windows, f"atlas TM windows are missing for {accession}")
        require(isinstance(cyto_windows, list) and cyto_windows, f"atlas cytoplasmic windows are missing for {accession}")

        tm_extracted = [extract_window(sequence, window, accession) for window in tm_windows]
        cyto_extracted = [extract_window(sequence, window, accession) for window in cyto_windows]
        tm_fragments = [item[0] for item in tm_extracted]
        cyto_fragments = [item[0] for item in cyto_extracted]
        tm_signatures = [item[2] for item in tm_extracted]
        cyto_signatures = [item[2] for item in cyto_extracted]
        require(tm_signatures == window_signature(tm_windows), f"atlas TM window signature differs for {accession}")
        require(cyto_signatures == window_signature(cyto_windows), f"atlas cytoplasmic window signature differs for {accession}")

        tm_shell = [float(item[1]) for item in tm_extracted]
        cyto_shell = [float(item[1]) for item in cyto_extracted]
        recomputed_delta = float(np.mean(tm_shell) - np.mean(cyto_shell))
        atlas_delta = float(atlas_gene["delta_shell_tm_minus_cyto"])
        require(
            abs(recomputed_delta - atlas_delta) <= 1e-12,
            f"atlas shell estimand differs for {accession}",
        )
        require(
            int(atlas_gene.get("n_tm_windows", -1)) == len(tm_windows)
            and int(atlas_gene.get("n_cyto_windows", -1)) == len(cyto_windows),
            f"atlas window counts differ for {accession}",
        )

        draw_values: list[float] = []
        usage_draw_values: list[float] = []
        tm_null_by_draw: list[list[str]] = []
        cyto_null_by_draw: list[list[str]] = []
        for _ in range(args.null_draws):
            tm_null = [synonymize(fragment, null_rng) for fragment in tm_fragments]
            cyto_null = [synonymize(fragment, null_rng) for fragment in cyto_fragments]
            tm_usage_null = [
                usage_weighted_synonymize(fragment, null_rng, host_usage)
                for fragment in tm_fragments
            ]
            cyto_usage_null = [
                usage_weighted_synonymize(fragment, null_rng, host_usage)
                for fragment in cyto_fragments
            ]
            tm_null_by_draw.append(tm_null)
            cyto_null_by_draw.append(cyto_null)
            draw_values.append(
                float(
                    np.mean([shell_value(s) for s in tm_null])
                    - np.mean([shell_value(s) for s in cyto_null])
                )
            )
            usage_draw_values.append(
                float(
                    np.mean([shell_value(s) for s in tm_usage_null])
                    - np.mean([shell_value(s) for s in cyto_usage_null])
                )
            )

        null_mean = float(np.mean(draw_values))
        null_sd = float(np.std(draw_values, ddof=1)) if len(draw_values) > 1 else 0.0
        usage_null_mean = float(np.mean(usage_draw_values))
        usage_null_sd = (
            float(np.std(usage_draw_values, ddof=1)) if len(usage_draw_values) > 1 else 0.0
        )
        native_contrast = float(np.mean(tm_shell) - np.mean(cyto_shell))
        native_residual = float(native_contrast - null_mean)
        usage_residual = float(native_contrast - usage_null_mean)

        # Exact junction constraint: count synonymous choices kept per codon.
        junction_counts = [
            count
            for fragment in tm_fragments + cyto_fragments
            for count in junction_matched_choice_counts(fragment)
        ]
        resamplable_codons = int(sum(1 for count in junction_counts if count > 1))

        # Dinucleotide-controlled contrast: residualize the native contrast and
        # the per-draw null contrasts through ONE junction-dinuc map fit on all
        # draws. The null side is NOT centered to zero first; if the map has no
        # explanatory power the control collapses back onto the plain residual.
        dinuc_result = dinuc_controlled_contrast(
            tm_fragments,
            cyto_fragments,
            tm_null_by_draw,
            cyto_null_by_draw,
            ridge=1.0,
        )

        # Exact junction-dinuc-matched AA-null (strong form): redraw synonyms that
        # keep every codon-boundary dinucleotide identical to native.
        matched_draw_values: list[float] = []
        for _ in range(args.null_draws):
            tm_matched = [
                synonymize_junction_matched(fragment, null_rng) for fragment in tm_fragments
            ]
            cyto_matched = [
                synonymize_junction_matched(fragment, null_rng) for fragment in cyto_fragments
            ]
            matched_draw_values.append(
                float(
                    np.mean([shell_value(s) for s in tm_matched])
                    - np.mean([shell_value(s) for s in cyto_matched])
                )
            )
        matched_null_mean = float(np.mean(matched_draw_values))
        matched_null_sd = (
            float(np.std(matched_draw_values, ddof=1)) if len(matched_draw_values) > 1 else 0.0
        )
        matched_residual = float(native_contrast - matched_null_mean)

        # Native's empirical quantile within its own synonymous neighborhood.
        n_deeper = int(sum(1 for value in draw_values if value < native_contrast))
        native_null_quantile = float((n_deeper + 1) / (len(draw_values) + 1))
        native_below_10th = bool(native_null_quantile <= 0.10)

        per_gene.append(
            {
                "gene": str(atlas_gene.get("gene", "")),
                "accession": accession,
                "n_tm_windows": len(tm_windows),
                "n_cyto_windows": len(cyto_windows),
                "tm_nt": int(sum(int(window["nt"]) for window in tm_windows)),
                "cyto_nt": int(sum(int(window["nt"]) for window in cyto_windows)),
                "tm_window_signatures": [list(signature) for signature in tm_signatures],
                "cyto_window_signatures": [list(signature) for signature in cyto_signatures],
                "atlas_tm_mean_shell": float(np.mean(tm_shell)),
                "atlas_cyto_mean_shell": float(np.mean(cyto_shell)),
                "atlas_delta_shell_tm_minus_cyto": atlas_delta,
                "native_contrast": native_contrast,
                "null_mean_delta_shell_tm_minus_cyto": null_mean,
                "null_sd_delta_shell_tm_minus_cyto": null_sd,
                "usage_matched_null_mean_delta_shell_tm_minus_cyto": usage_null_mean,
                "usage_matched_null_sd_delta_shell_tm_minus_cyto": usage_null_sd,
                "usage_matched_residual": usage_residual,
                "atlas_minus_null_residual": native_residual,
                "usage_matched_atlas_minus_null_residual": usage_residual,
                "null_minus_atlas_residual": -native_residual,
                "exact_junction_resamplable_codons": resamplable_codons,
                "exact_junction_codon_count": len(junction_counts),
                "native_null_quantile": native_null_quantile,
                "native_below_10th_percentile": native_below_10th,
                "n_null_draws": int(args.null_draws),
                "dinuc_native_contrast": float(dinuc_result.native_contrast),
                "dinuc_null_mean": float(dinuc_result.null_mean),
                "dinuc_null_sd": float(dinuc_result.null_sd),
                "dinuc_adjusted_residual": float(dinuc_result.adjusted_residual),
                "dinuc_null_contrasts": [float(v) for v in dinuc_result.null_contrasts],
                "dinuc_raw_native_contrast": float(dinuc_result.raw_native_contrast),
                "dinuc_raw_null_mean": float(np.mean(np.asarray(dinuc_result.raw_null_contrasts))),
                "exact_dinuc_null_mean": matched_null_mean,
                "exact_dinuc_null_sd": matched_null_sd,
                "exact_dinuc_residual": matched_residual,
            }
        )

    if len(per_gene) != len(selected_accessions):
        raise RuntimeError("AA-null per-gene output does not match the atlas cohort")

    observed = np.asarray(
        [float(gene["atlas_delta_shell_tm_minus_cyto"]) for gene in per_gene],
        dtype=np.float64,
    )
    null_means = np.asarray(
        [float(gene["null_mean_delta_shell_tm_minus_cyto"]) for gene in per_gene],
        dtype=np.float64,
    )
    residuals = observed - null_means
    native_quantiles = np.asarray(
        [float(gene["native_null_quantile"]) for gene in per_gene],
        dtype=np.float64,
    )
    dinuc_native_contrasts = np.asarray(
        [float(gene["dinuc_native_contrast"]) for gene in per_gene],
        dtype=np.float64,
    )
    dinuc_null_means = np.asarray(
        [float(gene["dinuc_null_mean"]) for gene in per_gene],
        dtype=np.float64,
    )
    dinuc_null_sds = np.asarray(
        [float(gene["dinuc_null_sd"]) for gene in per_gene],
        dtype=np.float64,
    )
    dinuc_adjusted = np.asarray(
        [float(gene["dinuc_adjusted_residual"]) for gene in per_gene],
        dtype=np.float64,
    )
    exact_matched_residuals = np.asarray(
        [float(gene["exact_dinuc_residual"]) for gene in per_gene],
        dtype=np.float64,
    )
    usage_residuals = np.asarray(
        [float(gene["usage_matched_residual"]) for gene in per_gene],
        dtype=np.float64,
    )
    usage_null_means = np.asarray(
        [float(gene["usage_matched_null_mean_delta_shell_tm_minus_cyto"]) for gene in per_gene],
        dtype=np.float64,
    )
    permutation_seed = int(args.seed + 1000003)
    usage_p = negative_tail_p(usage_residuals, args.permutations, permutation_seed + 3)
    n_usage_negative = int((usage_residuals < 0).sum())
    n_usage_positive = int((usage_residuals > 0).sum())
    resid_p = negative_tail_p(residuals, args.permutations, permutation_seed)
    dinuc_p = negative_tail_p(dinuc_adjusted, args.permutations, permutation_seed + 1)
    exact_dinuc_p = negative_tail_p(
        exact_matched_residuals, args.permutations, permutation_seed + 2
    )
    n_resid_negative = int((residuals < 0).sum())
    n_resid_positive = int((residuals > 0).sum())
    n_below_10th = int((native_quantiles <= 0.10).sum())
    n_exact_neg = int((exact_matched_residuals < 0).sum())

    summary = {
        "status": "ok",
        "n": int(len(observed)),
        "atlas_n": int(len(selected_accessions)),
        "population_accessions": list(selected_accessions),
        "cohort_manifest_sha256": str(audit["cohort_hash"]),
        "atlas_sha256": str(audit["atlas_sha256"]),
        "excluded": {},
        "excluded_accessions": {},
        "mean_obs": float(observed.mean()),
        "mean_AA_null": float(null_means.mean()),
        "mean_resid_null_minus_atlas": float(-residuals.mean()),
        "mean_resid_atlas_minus_null": float(residuals.mean()),
        "mean_resid": float(residuals.mean()),
        "mean_usage_matched_null": float(usage_null_means.mean()),
        "mean_usage_matched_residual": float(usage_residuals.mean()),
        "usage_matched_resid_p_one_sided_less": float(usage_p),
        "n_usage_resid_neg": n_usage_negative,
        "n_usage_resid_pos": n_usage_positive,
        "beyond_AA_usage_matched": bool(usage_residuals.mean() <= -0.03 and usage_p < 0.05),
        "native_null_quantile_mean": float(native_quantiles.mean()),
        "native_below_10th_percentile_fraction": float(n_below_10th / len(observed)),
        "dinuc_native_mean": float(dinuc_native_contrasts.mean()),
        "dinuc_null_mean": float(dinuc_null_means.mean()),
        "dinuc_null_sd_mean": float(dinuc_null_sds.mean()),
        "dinuc_adjusted_mean": float(dinuc_adjusted.mean()),
        "dinuc_p_more_neg": float(dinuc_p),
        "dinuc_perm_p_one_sided_less": float(dinuc_p),
        "exact_dinuc_residual_mean": float(exact_matched_residuals.mean()),
        "exact_dinuc_p_more_neg": float(exact_dinuc_p),
        "exact_dinuc_n_resid_neg": n_exact_neg,
        "n_resid_neg": n_resid_negative,
        "n_resid_pos": n_resid_positive,
        "resid_p_more_neg": float(resid_p),
        "resid_perm_p_one_sided_less": float(resid_p),
        "beyond_AA": bool(residuals.mean() <= -0.03 and resid_p < 0.05),
        "beyond_AA_exact_dinuc": bool(
            exact_matched_residuals.mean() <= -0.03 and exact_dinuc_p < 0.05
        ),
        "statistic": "difference of per-window mean shell values; windows are never concatenated",
        "estimator": {
            "unit": "gene",
            "window_statistic": "arithmetic mean adjacent-codon-pair shell per exact atlas topology window",
            "gene_contrast": "arithmetic mean(TM window shells) minus arithmetic mean(cytoplasmic window shells)",
            "population_statistic": "arithmetic mean of gene-level contrasts",
            "null_contrast": "atlas TM-minus-cytoplasmic mean minus null TM-minus-cytoplasmic mean",
            "permutation_test": "paired sign-flip permutation",
            "alternative": "less",
            "usage_matched_null": "synonymous draw weighted by host E. coli codon usage (common.usage_weighted_synonymize)",
            "dinucleotide_control": "linear junction-dinuc residualization (common.dinuc_controlled_contrast)",
            "dinucleotide_exact_matched_null": "synonymize_junction_matched preserves every codon-boundary dinucleotide exactly",
        },
        "null": {
            "type": "amino-acid-preserving synonymous resampling",
            "draws_per_gene": int(args.null_draws),
            "permutations": int(args.permutations),
            "seed": int(args.seed),
            "permutation_seed": permutation_seed,
            "dinucleotide_controlled": False,
            "dinucleotide_linear_residualized": True,
            "dinucleotide_exact_matched": True,
            "usage_matched": True,
        },
        "sources": {
            "atlas": {"path": str(atlas_path), "sha256": str(audit["atlas_sha256"])},
            "uniprot": {"path": str(uniprot_path), "sha256": str(audit["uniprot_sha256"])},
            "cds": {"path": str(cds_path), "sha256": str(audit["cds_sha256"])},
        },
        "provenance": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "script": {"path": str(_SCRIPT), "sha256": sha256_file(_SCRIPT)},
            "atlas": {
                "path": str(atlas_path),
                "sha256": str(audit["atlas_sha256"]),
                "cohort_manifest_sha256": str(audit["cohort_hash"]),
            },
            "cli": {
                "atlas": str(atlas_path),
                "uniprot": str(uniprot_path),
                "cds": str(cds_path),
                "output_json": str(out_json),
                "null_draws": int(args.null_draws),
                "permutations": int(args.permutations),
                "seed": int(args.seed),
                "min_aa": int(args.min_aa),
            },
            "seeds": {
                "null_draws": int(args.seed),
                "permutations": permutation_seed,
            },
            "source_digests": {
                "atlas": str(audit["atlas_sha256"]),
                "uniprot": str(audit["uniprot_sha256"]),
                "cds": str(audit["cds_sha256"]),
                "atlas_script": sha256_file(_HERE / "script_1_atlas.py"),
            },
            "atlas_validation": {
                "selected_accessions_match_genes": True,
                "atlas_hash_verified": True,
                "atlas_source_hashes_verified": True,
                "atlas_script_hash_verified": True,
                "production_checkpoint_hashes_verified": True,
                "min_aa_matches_atlas": True,
                "window_signatures_verified": True,
            },
            "atlas_production_checkpoints": {
                name: {"path": str(path), "sha256": str(audit["checkpoint_hashes"][name])}
                for name, path in sorted(production_checkpoint_paths().items())
            },
            "output_artifacts": {
                "json": str(out_json),
            },
        },
        "genes": per_gene,
    }
    lines = [
        "stage=null",
        f"status=ok atlas_n={summary['atlas_n']} n={summary['n']}",
        f"atlas_sha256={summary['atlas_sha256']}",
        f"cohort_manifest_sha256={summary['cohort_manifest_sha256']}",
        f"mean_obs={summary['mean_obs']:+.6f}",
        f"mean_AA_null={summary['mean_AA_null']:+.6f}",
        f"mean_resid_atlas_minus_null={summary['mean_resid_atlas_minus_null']:+.6f}",
        f"usage_matched_null_mean={summary['mean_usage_matched_null']:+.6f}",
        f"usage_matched_residual={summary['mean_usage_matched_residual']:+.6f} p_less={usage_p:.6f} neg={n_usage_negative}/{len(observed)} thresholds_met={summary['beyond_AA_usage_matched']}",
        f"n_resid_neg={n_resid_negative} n_resid_pos={n_resid_positive}",
        f"resid_p_one_sided_less={resid_p:.6f}",
        f"resid_thresholds_met={summary['beyond_AA']}",
        f"null_draws={args.null_draws} permutations={args.permutations} seed={args.seed} permutation_seed={permutation_seed}",
        f"dinuc_native_mean={dinuc_native_contrasts.mean():+.6f} dinuc_null_mean={dinuc_null_means.mean():+.6f} dinuc_null_sd_mean={dinuc_null_sds.mean():+.6f} dinuc_adjusted_mean={dinuc_adjusted.mean():+.6f} dinuc_p={dinuc_p:.6f}",
        f"exact_dinuc_residual_mean={exact_matched_residuals.mean():+.6f} exact_dinuc_p={exact_dinuc_p:.6f} n_exact_neg={n_exact_neg} exact_dinuc_resid_thresholds_met={summary['beyond_AA_exact_dinuc']}",
        f"native_null_quantile_mean={native_quantiles.mean():+.3f} native_below_10th_fraction={n_below_10th/len(observed):.3f}",
        "estimator=per-window mean shell; gene contrast=mean(TM) minus mean(cytoplasmic)",
        "dinuc control=linear junction residualization AND exact junction-matched synonymous null",
        "usage-matched null=host E. coli codon usage from the CDS catalog",
        "per-gene details are in the JSON artifact",
    ]
    write_json(out_json, summary)
    upsert_results("null", lines)
    print("\n".join(lines), flush=True)
    print(f"wrote {out_json.name} and RESULTS.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
