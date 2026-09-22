#!/usr/bin/env python3
"""Frozen Narrow AE scores on the exact membrane-topology atlas cohort.

Computes the frozen Narrow local-heterogeneity read on the exact atlas
windows, reports amino-acid-preserving and usage-matched residuals, and
publishes the per-window Narrow values that the held-out AE stage
consumes. The AE topology claim for this package is the holdout incremental
read in script_5_holdout.py.
"""
from __future__ import annotations

from typing import Any, cast

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

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
    codon_usage_counts,
    junction_matched_choice_counts,
    read_fasta_accession_records,
    require,
    synonymize,
    usage_weighted_synonymize,
)

from src.tools.autoencoder.helpers.evals_run import load_any_checkpoint  # noqa: E402
from src.tools.autoencoder.models.narrow import MLPAutoencoder  # noqa: E402
from src.tools.autoencoder import paths

TOPOLOGY_DIR = paths.dataset_dir("topology")
from src.tools.autoencoder.programs.genomics.genomics import (  # noqa: E402
    GENOMICS_DIR,
    clean_acgt,
)
from src.tools.autoencoder.programs.genomics.synthesis.constellation import (  # noqa: E402
    _ENC,
    ledger_bytes as _ledger,
    production_checkpoint_paths,
    walk_states as _walk_states,
    window_heterogeneity as _window_het,
)

_AE_READ_CAP_BYTES = 512


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


def negative_tail_p(values: list[float], *, n_perm: int, seed: int) -> float:
    """Test the fixed alternative that a TM-minus-cytoplasmic mean is below zero."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    observed = float(arr.mean())
    rng = np.random.Generator(np.random.PCG64(seed))
    hits = 0
    for _ in range(n_perm):
        signs = rng.integers(0, 2, size=arr.size, dtype=np.int8) * 2 - 1
        if float((arr * signs).mean()) <= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def gc_frac(seq: str) -> float:
    sequence = clean_acgt(seq)
    return float(sum(base in "GC" for base in sequence) / max(len(sequence), 1))


def load_frozen_narrow_model(path: Path, device: str) -> MLPAutoencoder:
    model, _ = load_any_checkpoint(path, device=device)
    require(isinstance(model, MLPAutoencoder), "Narrow checkpoint did not load an MLPAutoencoder")
    model = cast(MLPAutoencoder, model)
    model.eval()
    return model


def build_narrow_latent_table(model: MLPAutoencoder, device: str) -> np.ndarray:
    """Encode every possible Omega state once for fast window scoring."""
    states = torch.arange(4096, dtype=torch.long, device=device)
    with torch.inference_mode():
        return model.encode(states).detach().cpu().numpy().astype(
            np.float64, copy=False
        )


def narrow_het_one(
    latent_table: np.ndarray,
    fragment: str,
) -> float:
    states = _walk_states(_ledger(fragment, _ENC))
    require(len(states) > 0, "Narrow window produced no AE states")
    return _window_het(latent_table[states])


def narrow_het_batch(
    latent_table: np.ndarray,
    fragments: list[str],
) -> list[float]:
    output: list[float] = []
    for fragment in fragments:
        states = _walk_states(_ledger(fragment, _ENC))
        require(len(states) > 0, "Narrow window produced no AE states")
        output.append(_window_het(latent_table[states]))
    return output


def read_fasta_accessions(path: Path) -> dict[str, str]:
    """Index CDS records by their complete UniProt identifier."""
    return read_fasta_accession_records(path)


def read_uniprot_by_accession(path: Path) -> dict[str, dict[str, str]]:
    """Index UniProt rows by accession."""
    by_accession: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"Entry", "Length"}
        require(expected.issubset(reader.fieldnames or []), "UniProt table is missing required columns")
        for row in reader:
            accession = str(row.get("Entry", ""))
            require(bool(accession), "UniProt row is missing Entry")
            require(accession not in by_accession, f"duplicate UniProt accession {accession}")
            by_accession[accession] = dict(row)
    return by_accession


def validate_atlas_dependencies(
    atlas: dict[str, Any],
    atlas_path: Path,
    cds_path: Path,
    uniprot_path: Path,
) -> dict[str, Any]:
    selected_accessions = as_list(
        atlas.get("selected_accessions"), "atlas is missing selected_accessions"
    )
    genes = as_list(atlas.get("genes"), "atlas is missing genes")
    require(
        len(selected_accessions) == len(set(selected_accessions)),
        "atlas selected accessions contain duplicates",
    )
    gene_accessions = [str(gene.get("accession", "")) for gene in genes]
    require(all(gene_accessions), "atlas gene is missing an accession")
    require(
        len(gene_accessions) == len(set(gene_accessions)),
        "atlas genes contain duplicate accessions",
    )
    require(
        set(map(str, selected_accessions)) == set(gene_accessions),
        "atlas selected accessions and genes disagree",
    )
    require(int(atlas.get("n_genes", -1)) == len(genes), "atlas n_genes does not match genes")

    provenance = as_dict(atlas.get("provenance", {}), "atlas is missing provenance")
    cli = as_dict(provenance.get("cli", {}), "atlas provenance is missing cli")
    inputs = as_dict(provenance.get("inputs", {}), "atlas provenance is missing inputs")
    checkpoints = as_dict(
        provenance.get("production_checkpoints", {}),
        "atlas provenance is missing production checkpoints",
    )
    script_info = as_dict(provenance.get("script", {}), "atlas provenance is missing script")
    cohort_manifest = as_dict(atlas.get("cohort_manifest", {}), "atlas is missing cohort_manifest")
    permutation = as_dict(atlas.get("permutation", {}), "atlas is missing permutation")
    require(cli.get("min_aa") is not None, "atlas provenance is missing cli.min_aa")
    min_aa = int(cli["min_aa"])
    require(min_aa >= 1, "atlas min_aa is invalid")
    require(cohort_manifest.get("sha256"), "atlas is missing cohort_manifest.sha256")
    require(script_info.get("path") and script_info.get("sha256"), "atlas provenance is missing script identity")
    cds_input = as_dict(inputs.get("cds", {}), "atlas provenance is missing CDS identity")
    uniprot_input = as_dict(inputs.get("uniprot", {}), "atlas provenance is missing UniProt identity")
    require(cds_input.get("path") and cds_input.get("sha256"), "atlas provenance is missing CDS identity")
    require(uniprot_input.get("path") and uniprot_input.get("sha256"), "atlas provenance is missing UniProt identity")
    require(permutation.get("shell_alternative") == "less", "atlas shell permutation alternative is not fixed to less")
    require(permutation.get("super_alternative") == "two-sided nonzero", "atlas Super permutation alternative is not labeled two-sided")
    require(permutation.get("narrow_alternative") == "less", "atlas Narrow permutation alternative is not fixed to less")

    counts = as_dict(atlas.get("counts"), "atlas is missing truncation counts")
    ae_read = as_dict(atlas.get("ae_read"), "atlas is missing AE-read disclosure")
    require(int(ae_read.get("walk_cap_bytes", -1)) == _AE_READ_CAP_BYTES, "atlas AE read cap does not match the Narrow implementation")
    require(ae_read.get("truncation_disclosed") is True, "atlas does not disclose AE sequence truncation")
    require(ae_read.get("kernel_shell_uses_full_window") is True, "atlas does not state that kernel shell uses full windows")

    atlas_sha256 = sha256_file(atlas_path)
    truncated_window_count = 0
    genes_with_truncation = 0
    truncated_accessions: list[str] = []
    for gene in genes:
        accession = str(gene.get("accession", ""))
        windows = list(gene.get("tm_windows", [])) + list(gene.get("cyto_windows", []))
        require(bool(windows), f"atlas has no topology windows for {accession}")
        gene_truncated = False
        for window in windows:
            window = as_dict(window, f"atlas topology window is invalid for {accession}")
            require(isinstance(window.get("ae_truncated"), bool), f"atlas AE truncation flag is invalid for {accession}")
            require(isinstance(window.get("ae_walk_bytes"), int), f"atlas AE walk length is invalid for {accession}")
            walk_bytes = int(window["ae_walk_bytes"])
            expected_truncated = walk_bytes > _AE_READ_CAP_BYTES
            require(window["ae_truncated"] is expected_truncated, f"atlas AE truncation flag disagrees with walk length for {accession}")
            gene_truncated = gene_truncated or bool(window["ae_truncated"])
            truncated_window_count += int(bool(window["ae_truncated"]))
        genes_with_truncation += int(gene_truncated)
        if gene_truncated:
            truncated_accessions.append(accession)
    require(int(counts.get("ae_truncated_windows", -1)) == truncated_window_count, "atlas truncated-window count is inconsistent")
    require(int(counts.get("genes_with_ae_truncation", -1)) == genes_with_truncation, "atlas truncated-gene count is inconsistent")
    recorded_atlas_sha256 = atlas.get("atlas_sha256")
    if recorded_atlas_sha256 is not None:
        require(str(recorded_atlas_sha256) == atlas_sha256, "atlas self hash does not match the atlas file")
    require(
        str(atlas.get("atlas_cohort_manifest_sha256", cohort_manifest["sha256"])) == str(cohort_manifest["sha256"]),
        "atlas cohort hash aliases disagree",
    )

    for path in (atlas_path, cds_path, uniprot_path):
        require(path.is_file(), f"missing dependency: {path}")
    cds_sha256 = sha256_file(cds_path)
    uniprot_sha256 = sha256_file(uniprot_path)
    atlas_sha256 = sha256_file(atlas_path)
    atlas_script_path = Path(str(script_info["path"]))
    require(atlas_script_path.is_file(), f"missing atlas script: {atlas_script_path}")
    require(
        atlas_script_path.resolve()
        == (_HERE / "script_1_atlas.py").resolve(),
        "atlas script identity does not match the corrected atlas script",
    )
    require(
        sha256_file(atlas_script_path) == str(script_info["sha256"]),
        "atlas script hash does not match the script on disk",
    )
    require(
        Path(str(cds_input["path"])).resolve() == cds_path.resolve(),
        "atlas CDS provenance path does not match the dependency path",
    )
    require(
        Path(str(uniprot_input["path"])).resolve() == uniprot_path.resolve(),
        "atlas UniProt provenance path does not match the dependency path",
    )
    require(str(cds_input["sha256"]) == cds_sha256, "atlas CDS hash does not match the CDS file")
    require(str(uniprot_input["sha256"]) == uniprot_sha256, "atlas UniProt hash does not match the UniProt file")

    expected_checkpoints = production_checkpoint_paths()
    require(set(checkpoints) == set(expected_checkpoints), "atlas production checkpoint set is incomplete")
    checkpoint_hashes: dict[str, str] = {}
    for name, expected_path in expected_checkpoints.items():
        recorded = as_dict(checkpoints[name], f"checkpoint {name} is missing identity")
        require(recorded.get("path") and recorded.get("sha256"), f"checkpoint {name} is missing identity")
        recorded_path = Path(str(recorded["path"]))
        require(recorded_path.is_file(), f"missing production checkpoint: {recorded_path}")
        require(recorded_path.resolve() == expected_path.resolve(), f"checkpoint {name} path does not match production")
        recorded_hash = sha256_file(recorded_path)
        require(recorded_hash == str(recorded["sha256"]), f"checkpoint {name} hash does not match the checkpoint file")
        checkpoint_hashes[name] = recorded_hash

    cohort_hash = canonical_cohort_hash(
        [str(accession) for accession in selected_accessions],
        cds_sha256,
        uniprot_sha256,
        min_aa,
    )
    require(
        str(cohort_manifest["sha256"]) == cohort_hash,
        "atlas cohort hash does not match selected accessions and source digests",
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
    require(bool(atlas.get("aa_null_reference")), "atlas is missing AA-null reference")
    require(bool(atlas_exclusions), "atlas is missing explicit exclusion records")
    atlas_exclusions = as_dict(atlas["exclusions"], "atlas is missing exclusions")
    exclusion_counts = as_dict(
        atlas_exclusions.get("counts"), "atlas exclusions are missing counts"
    )
    accessions_by_reason = as_dict(
        atlas_exclusions.get("accessions_by_reason"),
        "atlas exclusions are missing accessions_by_reason",
    )
    require(
        set(map(str, exclusion_counts)) == set(map(str, accessions_by_reason)),
        "atlas exclusion count and accession reason sets disagree",
    )
    atlas_excluded_accessions: set[str] = set()
    for reason, count in exclusion_counts.items():
        require(isinstance(reason, str) and bool(reason), "atlas exclusion reason is invalid")
        require(isinstance(count, int) and count >= 0, f"atlas exclusion count is invalid for {reason}")
        reason_accessions = as_list(
            accessions_by_reason.get(reason),
            f"atlas exclusion accessions are invalid for {reason}",
        )
        require(all(isinstance(accession, str) and bool(accession) for accession in reason_accessions), f"atlas exclusion accession is invalid for {reason}")
        require(len(reason_accessions) == len(set(reason_accessions)), f"atlas exclusion accessions contain duplicates for {reason}")
        require(count == len(reason_accessions), f"atlas exclusion count disagrees for {reason}")
        atlas_excluded_accessions.update(str(accession) for accession in reason_accessions)
    require(
        atlas_excluded_accessions.isdisjoint(selected_accessions),
        "atlas exclusion population overlaps selected accessions",
    )
    atlas_script_sha256 = sha256_file(atlas_script_path)
    return {
        "atlas_sha256": atlas_sha256,
        "cds_sha256": cds_sha256,
        "uniprot_sha256": uniprot_sha256,
        "cohort_hash": cohort_hash,
        "min_aa": min_aa,
        "selected_accessions": [str(accession) for accession in selected_accessions],
        "checkpoint_hashes": checkpoint_hashes,
        "atlas_exclusions": atlas_exclusions,
        "atlas_script_path": atlas_script_path,
        "atlas_script_sha256": atlas_script_sha256,
        "truncated_window_count": truncated_window_count,
        "genes_with_truncation": genes_with_truncation,
        "truncated_accessions": truncated_accessions,
        "ae_read": ae_read,
    }


def window_signature(windows: list[dict[str, Any]], label: str) -> list[tuple[int, int, int, str]]:
    signature: list[tuple[int, int, int, str]] = []
    for window in windows:
        require(isinstance(window.get("start_aa"), int), f"atlas {label} window is missing start_aa")
        require(isinstance(window.get("end_aa"), int), f"atlas {label} window is missing end_aa")
        require(isinstance(window.get("nt"), int), f"atlas {label} window is missing nt")
        require(bool(window.get("sequence_sha256")), f"atlas {label} window is missing sequence_sha256")
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
    label: str,
) -> tuple[str, tuple[int, int, int, str]]:
    start = int(window["start_aa"])
    end = int(window["end_aa"])
    expected_nt = int(window["nt"])
    fragment = clean_acgt(sequence[(start - 1) * 3 : end * 3])
    require(start >= 1 and end >= start, f"invalid {label} window coordinates for {accession}")
    require(len(fragment) == expected_nt, f"{label} window length differs for {accession}")
    require(len(fragment) >= 36 and len(fragment) % 3 == 0, f"{label} window is not a valid codon sequence for {accession}")
    digest = hashlib.sha256(fragment.encode("utf-8")).hexdigest()
    require(digest == str(window["sequence_sha256"]), f"{label} window sequence hash differs for {accession}")
    return fragment, (start, end, expected_nt, digest)



def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score the frozen Narrow AE on the exact topology atlas cohort."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--null-draws", type=int, default=80)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    if args.null_draws < 80:
        parser.error("--null-draws must be at least 80")
    if args.permutations < 5000:
        parser.error("--permutations must be at least 5000")

    atlas_path = RESULT_JSON["atlas"].resolve()
    cds_path = (GENOMICS_DIR / "ecoli_k12_cds.fna.gz").resolve()
    uniprot_path = (TOPOLOGY_DIR / "uniprot_ecoli_k12.tsv").resolve()
    if not atlas_path.is_file():
        print(f"missing atlas: {atlas_path}", file=sys.stderr)
        return 1

    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    dependency_audit = validate_atlas_dependencies(
        atlas, atlas_path, cds_path, uniprot_path
    )
    genes = atlas["genes"]
    selected_accessions = dependency_audit["selected_accessions"]
    min_aa = int(dependency_audit["min_aa"])
    by_acc = read_fasta_accessions(cds_path)
    uniprot_by_accession = read_uniprot_by_accession(uniprot_path)
    host_usage_totals: dict[str, int] = {}
    for sequence in by_acc.values():
        for codon, count in codon_usage_counts(sequence).items():
            host_usage_totals[codon] = host_usage_totals.get(codon, 0) + count
    require(bool(host_usage_totals), "host codon usage is empty")
    host_usage = {codon: float(count) for codon, count in host_usage_totals.items()}
    missing_uniprot_accessions = set(selected_accessions) - set(uniprot_by_accession)
    require(
        not missing_uniprot_accessions,
        f"missing UniProt rows for exact-cohort accessions: {sorted(missing_uniprot_accessions)}",
    )
    gene_by_accession = {str(gene["accession"]): gene for gene in genes}
    require(set(gene_by_accession) == set(selected_accessions), "atlas gene map does not match selected accessions")

    null_rng = np.random.Generator(np.random.PCG64(args.seed))
    per_gene: list[dict[str, Any]] = []
    print("loading frozen Narrow checkpoint for the exact-cohort AA-preserving null", flush=True)
    narrow_model = load_frozen_narrow_model(
        production_checkpoint_paths()["mlp"], str(args.device)
    )
    narrow_latents = build_narrow_latent_table(
        narrow_model, str(args.device)
    )

    for accession in selected_accessions:
        gene = gene_by_accession[accession]
        sequence = by_acc.get(accession)
        if sequence is None:
            raise ValueError(f"missing CDS for exact-cohort accession {accession}")
        tm_windows = gene.get("tm_windows")
        cyto_windows = gene.get("cyto_windows")
        require(isinstance(tm_windows, list) and tm_windows, f"atlas TM windows are missing for {accession}")
        require(isinstance(cyto_windows, list) and cyto_windows, f"atlas cytoplasmic windows are missing for {accession}")
        tm_extracted = [extract_window(sequence, window, accession, "TM") for window in tm_windows]
        cyto_extracted = [extract_window(sequence, window, accession, "cytoplasmic") for window in cyto_windows]
        tm_fragments = [item[0] for item in tm_extracted]
        cyto_fragments = [item[0] for item in cyto_extracted]
        require(
            [item[1] for item in tm_extracted] == window_signature(tm_windows, "TM"),
            f"atlas TM window signature differs for {accession}",
        )
        require(
            [item[1] for item in cyto_extracted] == window_signature(cyto_windows, "cytoplasmic"),
            f"atlas cytoplasmic window signature differs for {accession}",
        )
        require(
            all(len(fragment) >= min_aa * 3 for fragment in tm_fragments + cyto_fragments),
            f"atlas window is below min_aa for {accession}",
        )

        tm_native = [
            narrow_het_one(narrow_latents, fragment)
            for fragment in tm_fragments
        ]
        cyto_native = [
            narrow_het_one(narrow_latents, fragment)
            for fragment in cyto_fragments
        ]
        observed = float(np.mean(tm_native) - np.mean(cyto_native))
        atlas_observed = float(gene["delta_narrow_tm_minus_cyto"])
        require(
            abs(observed - atlas_observed) <= 1e-6,
            f"atlas Narrow value differs for {accession}",
        )

        tm_null_values = np.empty(
            (args.null_draws, len(tm_fragments)), dtype=np.float64
        )
        cyto_null_values = np.empty(
            (args.null_draws, len(cyto_fragments)), dtype=np.float64
        )
        tm_usage_null_values = np.empty(
            (args.null_draws, len(tm_fragments)), dtype=np.float64
        )
        cyto_usage_null_values = np.empty(
            (args.null_draws, len(cyto_fragments)), dtype=np.float64
        )
        for draw_index in range(args.null_draws):
            tm_null_fragments = [
                synonymize(fragment, null_rng) for fragment in tm_fragments
            ]
            cyto_null_fragments = [
                synonymize(fragment, null_rng) for fragment in cyto_fragments
            ]
            tm_usage_null_fragments = [
                usage_weighted_synonymize(fragment, null_rng, host_usage)
                for fragment in tm_fragments
            ]
            cyto_usage_null_fragments = [
                usage_weighted_synonymize(fragment, null_rng, host_usage)
                for fragment in cyto_fragments
            ]
            tm_null_values[draw_index] = narrow_het_batch(
                narrow_latents, tm_null_fragments
            )
            cyto_null_values[draw_index] = narrow_het_batch(
                narrow_latents, cyto_null_fragments
            )
            tm_usage_null_values[draw_index] = narrow_het_batch(
                narrow_latents, tm_usage_null_fragments
            )
            cyto_usage_null_values[draw_index] = narrow_het_batch(
                narrow_latents, cyto_usage_null_fragments
            )
        draw_means = np.mean(tm_null_values, axis=1) - np.mean(
            cyto_null_values, axis=1
        )
        usage_draw_means = np.mean(tm_usage_null_values, axis=1) - np.mean(
            cyto_usage_null_values, axis=1
        )
        null_mean = float(np.mean(draw_means))
        usage_null_mean = float(np.mean(usage_draw_means))
        junction_counts = [
            count
            for fragment in tm_fragments + cyto_fragments
            for count in junction_matched_choice_counts(fragment)
        ]
        resamplable_codons = int(sum(1 for count in junction_counts if count > 1))
        per_gene.append(
            {
                "gene": str(gene.get("gene", "")),
                "accession": accession,
                "n_tm_windows": len(tm_windows),
                "n_cyto_windows": len(cyto_windows),
                "tm_nt": sum(int(window["nt"]) for window in tm_windows),
                "cyto_nt": sum(int(window["nt"]) for window in cyto_windows),
                "tm_window_signatures": [list(item[1]) for item in tm_extracted],
                "cyto_window_signatures": [list(item[1]) for item in cyto_extracted],
                "atlas_narrow_delta": atlas_observed,
                "recomputed_narrow_delta": observed,
                "mean_AA_null_narrow_delta": null_mean,
                "narrow_residual_atlas_minus_AA_null": observed - null_mean,
                "mean_usage_matched_AA_null_narrow_delta": usage_null_mean,
                "narrow_residual_atlas_minus_usage_matched_AA_null": observed - usage_null_mean,
                "exact_junction_resamplable_codons": resamplable_codons,
                "exact_junction_codon_count": len(junction_counts),
                "n_null_draws": args.null_draws,
                "ae_read_cap_bytes": int(atlas["ae_read"]["walk_cap_bytes"]),
                "ae_read_truncation_policy": "first 512 bytes of the encoded sequence are retained; kernel shell uses the full exact topology window",
                "n_tm_windows_truncated": sum(
                    int(window["ae_truncated"]) for window in tm_windows
                ),
                "n_cyto_windows_truncated": sum(
                    int(window["ae_truncated"]) for window in cyto_windows
                ),
            }
        )

    if len(per_gene) != len(genes):
        raise RuntimeError(f"Narrow AA-null gene count mismatch: atlas={len(genes)} null={len(per_gene)}")

    d_narrow = [float(gene["delta_narrow_tm_minus_cyto"]) for gene in genes]
    d_shell = [float(gene["delta_shell_tm_minus_cyto"]) for gene in genes]
    d_gc = [float(gene["tm_gc"]) - float(gene["cyto_gc"]) for gene in genes]
    mean_narrow = float(np.mean(d_narrow))
    mean_shell = float(np.mean(d_shell))
    mean_gc = float(np.mean(d_gc))
    frac_neg = float(np.mean(np.asarray(d_narrow) < 0))
    narrow_seed = args.seed + 2000003
    resid_seed = args.seed + 1000003
    narrow_p = negative_tail_p(d_narrow, n_perm=args.permutations, seed=narrow_seed)
    gc_ratio = abs(mean_narrow) / (abs(mean_gc) + 1e-12)
    residuals = [float(gene["narrow_residual_atlas_minus_AA_null"]) for gene in per_gene]
    mean_resid = float(np.mean(residuals))
    resid_p = negative_tail_p(residuals, n_perm=args.permutations, seed=resid_seed)
    require(len(residuals) == len(genes), "n_AA_null_genes does not equal n_genes")
    usage_residuals = [
        float(gene["narrow_residual_atlas_minus_usage_matched_AA_null"]) for gene in per_gene
    ]
    usage_resid_seed = args.seed + 3000003
    mean_usage_resid = float(np.mean(usage_residuals))
    usage_resid_p = negative_tail_p(
        usage_residuals, n_perm=args.permutations, seed=usage_resid_seed
    )

    gates = {
        "narrow_scores_complete": True,
        "n_AA_null_genes_equals_n_genes": bool(len(residuals) == len(genes)),
        "cohort_matches_atlas": bool(len(per_gene) == len(genes)),
    }
    passed = all(gates.values())
    claim = None
    summary = {
        "status": "ok",
        "claim": claim,
        "passed": passed,
        "gates": gates,
        "role": "ae_feature_source_for_holdout_topology_read",
        "ae_claim_stage": "holdout",
        "primary_ae_object": "frozen_narrow_local_heterogeneity",
        "n_genes": len(genes),
        "n_AA_null_genes": len(per_gene),
        "atlas_n": len(genes),
        "population_accessions": selected_accessions,
        "cohort_manifest_sha256": dependency_audit["cohort_hash"],
        "stats": {
            "mean_delta_narrow_tm_minus_cyto": mean_narrow,
            "narrow_negative_fraction": frac_neg,
            "n_negative_narrow_genes": int(sum(value < 0 for value in d_narrow)),
            "narrow_perm_p_one_sided_less": narrow_p,
            "mean_delta_gc_tm_minus_cyto": mean_gc,
            "narrow_vs_gc_ratio": gc_ratio,
            "mean_AA_null_narrow": float(np.mean([float(gene["mean_AA_null_narrow_delta"]) for gene in per_gene])),
            "mean_narrow_resid_beyond_AA": mean_resid,
            "narrow_resid_perm_p_one_sided_less": resid_p,
            "mean_usage_matched_AA_null_narrow": float(np.mean([float(gene["mean_usage_matched_AA_null_narrow_delta"]) for gene in per_gene])),
            "mean_narrow_resid_beyond_usage_matched_AA": mean_usage_resid,
            "narrow_usage_matched_resid_perm_p_one_sided_less": usage_resid_p,
            "kernel_mean_delta_shell_tm_minus_cyto": mean_shell,
        },
        "permutation": {
            "test": "paired sign-flip permutation",
            "alternative": "less",
            "contrast_direction": "TM-minus-cytoplasmic",
            "permutations": args.permutations,
            "seeds": {
                "narrow_het": narrow_seed,
                "AA_residual": resid_seed,
            },
        },
        "null": {
            "type": "amino-acid-preserving synonymous resampling",
            "draws_per_gene": args.null_draws,
            "seed": args.seed,
            "permutation_seed": resid_seed,
            "usage_matched": True,
            "usage_matched_seed": usage_resid_seed,
        },
        "estimator": {
            "unit": "gene",
            "pairing": "TM and cytoplasmic topology windows from the same accession",
            "windowing": atlas.get("windowing"),
            "narrow_estimator": "arithmetic mean frozen Narrow latent heterogeneity over TM windows minus arithmetic mean over cytoplasmic windows",
            "null_estimator": "atlas Narrow gene contrast minus the mean of amino-acid-preserving synonymous Narrow gene contrasts",
            "usage_matched_null_estimator": "atlas Narrow gene contrast minus the mean of usage-matched amino-acid-preserving synonymous Narrow gene contrasts",
            "permutation_test": "paired sign-flip permutation with fixed less alternative",
        },
        "provenance": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "script": {"path": str(_SCRIPT), "sha256": sha256_file(_SCRIPT)},
            "atlas": {
                "path": str(atlas_path),
                "sha256": dependency_audit["atlas_sha256"],
                "cohort_manifest_sha256": dependency_audit["cohort_hash"],
            },
            "inputs": {
                "cds": {"id": cds_path.name, "path": str(cds_path), "sha256": dependency_audit["cds_sha256"]},
                "uniprot": {"id": uniprot_path.name, "path": str(uniprot_path), "sha256": dependency_audit["uniprot_sha256"]},
            },
            "production_checkpoints": {
                name: {"id": path.name, "path": str(path), "sha256": dependency_audit["checkpoint_hashes"][name]}
                for name, path in sorted(production_checkpoint_paths().items())
            },
            "cli": {
                "device": args.device,
                "null_draws": args.null_draws,
                "permutations": args.permutations,
                "seed": args.seed,
            },
            "seeds": {
                "null_draws": args.seed,
                "narrow_permutation": narrow_seed,
                "AA_residual_permutation": resid_seed,
                "usage_matched_AA_residual_permutation": usage_resid_seed,
            },
            "validation": {
                "atlas_accession_set_matches_genes": True,
                "atlas_source_hashes_verified": True,
                "atlas_script_hash_verified": True,
                "production_checkpoint_hashes_verified": True,
                "cohort_hash_verified": True,
                "dependencies_verified_before_statistics": True,
                "window_signatures_verified": True,
                "n_AA_null_genes_equals_n_genes": True,
            },
            "output_artifacts": {
                "json": str(RESULT_JSON["narrow"]),
            },
        },
        "exclusions": {
            "counts": {},
            "accessions_by_reason": {},
            "note": f"No selected atlas accession was excluded; all {len(selected_accessions)} accessions are scored.",
        },
        "atlas_exclusions": dependency_audit["atlas_exclusions"],
        "truncation": {
            "walk_cap_bytes": int(dependency_audit["ae_read"]["walk_cap_bytes"]),
            "truncated_windows": dependency_audit["truncated_window_count"],
            "genes_with_truncation": dependency_audit["genes_with_truncation"],
            "accessions": dependency_audit["truncated_accessions"],
            "policy": "Narrow retains the first 512 bytes of the encoded sequence; kernel shell uses the full exact topology window.",
        },
        "per_gene": per_gene,
        "uses_frozen_constellation": True,
    }

    out_json = RESULT_JSON["narrow"]
    lines = [
        "stage=narrow",
        f"status=ok n_genes={len(genes)} aa_null_genes={len(per_gene)} atlas_n={len(genes)}",
        f"cohort_manifest_sha256={dependency_audit['cohort_hash']}",
        f"role=ae_feature_source_for_holdout_topology_read ae_claim_stage=holdout",
        f"narrow_mean_delta={mean_narrow:+.4f} frac_neg={frac_neg:.3f} p_less={narrow_p:.4f}",
        f"gc_mean_delta={mean_gc:+.4f} narrow_vs_gc_ratio={gc_ratio:.2f}",
        f"narrow_AA_residual={mean_resid:+.4f} p_less={resid_p:.4f}",
        f"narrow_usage_matched_AA_residual={mean_usage_resid:+.4f} p_less={usage_resid_p:.4f}",
        f"kernel_shell_mean_delta={mean_shell:+.4f}",
        f"null_draws={args.null_draws} permutations={args.permutations} seed={args.seed} narrow_permutation_seed={narrow_seed} residual_permutation_seed={resid_seed} usage_matched_residual_permutation_seed={usage_resid_seed}",
        f"passed={passed}",
        "gates:",
    ]
    for key, value in gates.items():
        lines.append(f"  {key}: {value}")
    lines.append("per-gene details are in the JSON artifact")
    write_json(out_json, summary)
    upsert_results("narrow", lines)
    print("\n".join(lines), flush=True)
    print(f"wrote {out_json.name} and RESULTS.txt")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
