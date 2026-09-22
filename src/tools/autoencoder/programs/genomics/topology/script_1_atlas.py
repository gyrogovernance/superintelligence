#!/usr/bin/env python3
"""Within-gene membrane topology atlas.

Primary readout: kernel adjacent-codon-pair mean shell. Frozen Narrow and
Super reads are scored on the same windows. The contrast is the within-gene
mean for transmembrane windows minus the mean for cytoplasmic windows.
"""
from __future__ import annotations

from typing import Any

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
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
    read_fasta_records,
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
    load_frozen_constellation,
    production_checkpoint_paths,
    read_constellation,
)

_TM_RE = re.compile(r"TRANSMEM\s+(\d+)(?:\.\.(\d+))?", re.I)
_TOPO_RE = re.compile(
    r"TOPO_DOM\s+(\d+)(?:\.\.(\d+))?.*?/note=\"([^\"]+)\"", re.I
)
_AE_READ_CAP_BYTES = 512
_UNIPROT_RELEASE = "2026_03"
_UNIPROT_QUERY = (
    "proteome:UP000000625&fields=accession,id,gene_primary,gene_oln,length,"
    "cc_subcellular_location,ft_transmem,ft_topo_dom"
)


@dataclass(frozen=True)
class CdsRecord:
    gene_name: str
    locus_tag: str
    accession: str
    sequence: str


_COORDINATE_POLICY = (
    "UniProt 1-based inclusive spans must fall wholly within the translated CDS; "
    "invalid or out-of-range spans are excluded and never clipped"
)
_TERMINAL_SPAN_RULE = (
    "exclude invalid, out-of-range, singleton, or below-min-aa spans; "
    "score only exact retained spans"
)


@dataclass(frozen=True)
class Span:
    start_aa: int
    end_aa: int
    label: str
    singleton: bool = False


def translated_cds_length_aa(sequence: str) -> int:
    """Return the translated residue count used for UniProt coordinates."""
    cleaned = clean_acgt(sequence)
    if not cleaned or len(cleaned) % 3:
        raise ValueError("CDS sequence is not a complete coding sequence")
    codon_count = len(cleaned) // 3
    if codon_count and cleaned[-3:] in {"TAA", "TAG", "TGA"}:
        return codon_count - 1
    return codon_count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def parse_spans(transmembrane: str, topology: str) -> list[Span]:
    spans: list[Span] = []
    for match in _TM_RE.finditer(transmembrane or ""):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        spans.append(Span(start, end, "tm", end == start))
    for match in _TOPO_RE.finditer(topology or ""):
        note = match.group(3).lower()
        if "cytoplasmic" in note:
            label = "cytoplasmic"
        elif "periplasmic" in note or "extracellular" in note:
            label = "periplasmic"
        else:
            label = "other"
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        spans.append(Span(start, end, label, end == start))
    return spans


def read_cds_by_accession(path: Path) -> dict[str, CdsRecord]:
    """Index CDS records by their complete UniProt identifier."""
    return {
        accession: CdsRecord(
            record.gene_name or record.locus_tag or record.accession,
            record.locus_tag,
            record.accession,
            record.sequence,
        )
        for accession, record in read_fasta_records(path).items()
    }


def load_uniprot(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def paired_sign_flip_p(
    values: np.ndarray, *, n_perm: int, seed: int, alternative: str = "less"
) -> float:
    """Paired sign-flip test with a named alternative.

    alternative = "less"  -> fixed alternative that the mean is below zero
    alternative = "greater" -> fixed alternative that the mean is above zero
    alternative = "two-sided nonzero" -> two-sided test against zero
    """
    if values.size == 0:
        return float("nan")
    observed = float(values.mean())
    if alternative == "two-sided nonzero":
        target = abs(observed)
        rng = np.random.Generator(np.random.PCG64(seed))
        hits = 0
        for _ in range(n_perm):
            signs = rng.integers(0, 2, size=values.size, dtype=np.int8) * 2 - 1
            permuted = float((values * signs).mean())
            if abs(permuted) >= target:
                hits += 1
        return (hits + 1) / (n_perm + 1)
    elif alternative == "greater":
        rng = np.random.Generator(np.random.PCG64(seed))
        hits = 0
        for _ in range(n_perm):
            signs = rng.integers(0, 2, size=values.size, dtype=np.int8) * 2 - 1
            permuted = float((values * signs).mean())
            if permuted >= observed:
                hits += 1
        return (hits + 1) / (n_perm + 1)
    else:
        rng = np.random.Generator(np.random.PCG64(seed))
        hits = 0
        for _ in range(n_perm):
            signs = rng.integers(0, 2, size=values.size, dtype=np.int8) * 2 - 1
            permuted = float((values * signs).mean())
            if permuted <= observed:
                hits += 1
        return (hits + 1) / (n_perm + 1)


def gc_fraction(sequence: str) -> float:
    sequence = clean_acgt(sequence)
    if not sequence:
        return float("nan")
    return (sequence.count("G") + sequence.count("C")) / len(sequence)


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


def record_exclusion(
    counts: dict[str, int],
    accessions_by_reason: dict[str, list[str]],
    accession: str,
    reason: str,
) -> None:
    counts[reason] = counts.get(reason, 0) + 1
    accessions_by_reason.setdefault(reason, []).append(accession)


def atomic_write_pair(
    out_json: Path, out_txt: Path, json_payload: str, text_payload: str
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    json_tmp = out_json.with_name(f".{out_json.name}.{os.getpid()}.tmp")
    txt_tmp = out_txt.with_name(f".{out_txt.name}.{os.getpid()}.tmp")
    try:
        with json_tmp.open("w", encoding="utf-8") as handle:
            handle.write(json_payload)
            handle.flush()
            os.fsync(handle.fileno())
        with txt_tmp.open("w", encoding="utf-8") as handle:
            handle.write(text_payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(txt_tmp, out_txt)
        os.replace(json_tmp, out_json)
    finally:
        for temp_path in (json_tmp, txt_tmp):
            if temp_path.exists():
                temp_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the E. coli membrane-topology climate atlas."
    )
    parser.add_argument(
        "--min-aa",
        type=int,
        default=12,
        help="Minimum exact topology-window length in amino acids (default: 12).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional seeded sample size without replacement. Omit it to use every eligible gene.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260919,
        help="Seed for sampling and permutation streams (default: 20260919).",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=5000,
        help="Sign-flip permutations per reported contrast (default: 5000).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=RESULT_JSON["atlas"],
        help="RESULTS_script_1_atlas.json",
    )
    args = parser.parse_args()

    if args.min_aa < 1:
        parser.error("--min-aa must be at least 1")
    if args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample-size must be at least 1")
    if args.permutations < 1:
        parser.error("--permutations must be at least 1")

    uniprot_path = (TOPOLOGY_DIR / "uniprot_ecoli_k12.tsv").resolve()
    cds_path = (GENOMICS_DIR / "ecoli_k12_cds.fna.gz").resolve()
    out_json = args.output_json.resolve()
    for path in (uniprot_path, cds_path):
        if not path.is_file():
            print(f"missing input: {path}", file=sys.stderr)
            return 1

    cds_by_accession = read_cds_by_accession(cds_path)
    rows_uniprot = load_uniprot(uniprot_path)
    population: list[tuple[CdsRecord, list[Span]]] = []
    exclusion_counts: dict[str, int] = {}
    excluded_accessions: dict[str, list[str]] = {}
    span_exclusion_counts: dict[str, int] = {}
    span_excluded_accessions: dict[str, list[str]] = {}
    span_exclusion_details: list[dict[str, Any]] = []

    def record_span_exclusion(
        accession: str,
        reason: str,
        span: Span,
        uniprot_length: int,
        translated_length: int,
    ) -> None:
        span_exclusion_counts[reason] = span_exclusion_counts.get(reason, 0) + 1
        span_excluded_accessions.setdefault(reason, []).append(accession)
        span_exclusion_details.append(
            {
                "accession": accession,
                "label": span.label,
                "start_aa": int(span.start_aa),
                "end_aa": int(span.end_aa),
                "length_aa": int(span.end_aa - span.start_aa + 1),
                "singleton": bool(span.singleton),
                "uniprot_length_aa": int(uniprot_length),
                "translated_cds_length_aa": int(translated_length),
                "reason": reason,
            }
        )

    for row in rows_uniprot:
        entry = (row.get("Entry") or "").strip()
        primary_gene = (row.get("Gene Names (primary)") or "").strip()
        accession = entry or primary_gene
        transmembrane = row.get("Transmembrane") or ""
        topology = row.get("Topological domain") or ""
        if not transmembrane.strip():
            record_exclusion(
                exclusion_counts,
                excluded_accessions,
                accession,
                "no_transmembrane_annotation",
            )
            continue

        parsed_spans = parse_spans(transmembrane, topology)
        tm_spans = [span for span in parsed_spans if span.label == "tm"]
        periplasmic_spans = [
            span for span in parsed_spans if span.label == "periplasmic"
        ]
        cytoplasmic_spans = [
            span for span in parsed_spans if span.label == "cytoplasmic"
        ]
        record = cds_by_accession.get(entry) if entry else None
        if record is None:
            isoform_available = any(
                candidate.startswith(f"{entry}-") for candidate in cds_by_accession
            )
            reason = "cds_isoform_mismatch" if isoform_available else "missing_cds_accession"
            record_exclusion(
                exclusion_counts, excluded_accessions, accession, reason
            )
            continue

        try:
            uniprot_protein_length = int((row.get("Length") or "0").strip())
        except ValueError:
            uniprot_protein_length = 0
        translated_aa_length = translated_cds_length_aa(record.sequence)
        if uniprot_protein_length < 1:
            record_exclusion(
                exclusion_counts,
                excluded_accessions,
                accession,
                "invalid_uniprot_length",
            )
            continue
        if translated_aa_length < 1:
            record_exclusion(
                exclusion_counts,
                excluded_accessions,
                accession,
                "translated_cds_has_no_residues",
            )
            continue
        if uniprot_protein_length > translated_aa_length:
            record_exclusion(
                exclusion_counts,
                excluded_accessions,
                accession,
                "uniprot_length_exceeds_translated_cds",
            )
            continue

        exact_tm: list[Span] = []
        exact_cytoplasmic: list[Span] = []
        exact_periplasmic: list[Span] = []
        for span in tm_spans + cytoplasmic_spans + periplasmic_spans:
            if span.start_aa < 1 or span.end_aa < span.start_aa:
                record_span_exclusion(
                    accession,
                    "invalid_span_coordinates",
                    span,
                    uniprot_protein_length,
                    translated_aa_length,
                )
                continue
            if span.end_aa > translated_aa_length:
                record_span_exclusion(
                    accession,
                    "span_beyond_translated_cds",
                    span,
                    uniprot_protein_length,
                    translated_aa_length,
                )
                continue
            if span.end_aa > uniprot_protein_length:
                record_span_exclusion(
                    accession,
                    "span_beyond_uniprot_length",
                    span,
                    uniprot_protein_length,
                    translated_aa_length,
                )
                continue
            if span.singleton:
                record_span_exclusion(
                    accession,
                    "singleton_topology_span",
                    span,
                    uniprot_protein_length,
                    translated_aa_length,
                )
                continue
            if span.end_aa - span.start_aa + 1 < args.min_aa:
                record_span_exclusion(
                    accession,
                    "span_below_min_aa",
                    span,
                    uniprot_protein_length,
                    translated_aa_length,
                )
                continue
            if span.label == "tm":
                exact_tm.append(span)
            elif span.label == "cytoplasmic":
                exact_cytoplasmic.append(span)
            elif span.label == "periplasmic":
                exact_periplasmic.append(span)

        if not exact_tm:
            record_exclusion(
                exclusion_counts,
                excluded_accessions,
                accession,
                "no_tm_span_meeting_min_aa",
            )
            continue
        if not exact_cytoplasmic:
            record_exclusion(
                exclusion_counts,
                excluded_accessions,
                accession,
                "no_cytoplasmic_span_meeting_min_aa",
            )
            continue
        population.append(
            (record, exact_tm + exact_cytoplasmic + exact_periplasmic)
        )

    population.sort(key=lambda item: item[0].accession)
    population_accessions = [record.accession for record, _ in population]
    if args.sample_size is not None and args.sample_size > len(population):
        print(
            f"--sample-size {args.sample_size} exceeds the eligible population of {len(population)}",
            file=sys.stderr,
        )
        return 1

    sampling_rng = np.random.Generator(np.random.PCG64(args.seed))
    if args.sample_size is None:
        selected_accessions = population_accessions
        omitted_accessions: list[str] = []
        sampling = {
            "mode": "all_eligible",
            "eligible_population_n": len(population),
            "sample_size": len(population),
            "seed": int(args.seed),
        }
    else:
        chosen_positions = sampling_rng.choice(
            len(population), size=args.sample_size, replace=False
        )
        chosen = {population_accessions[position] for position in chosen_positions}
        selected_accessions = sorted(chosen)
        omitted_accessions = sorted(set(population_accessions) - chosen)
        sampling = {
            "mode": "seeded_without_replacement",
            "eligible_population_n": len(population),
            "sample_size": int(args.sample_size),
            "seed": int(args.seed),
            "omitted_accessions": omitted_accessions,
        }

    selected_accession_set = set(selected_accessions)
    selected = {
        record.accession: spans
        for record, spans in population
        if record.accession in selected_accession_set
    }
    if len(selected) != len(selected_accessions):
        print("selected accession set could not be reconstructed", file=sys.stderr)
        return 1
    if len(selected_accessions) < 10:
        print(f"too few eligible genes ({len(selected_accessions)})", file=sys.stderr)
        return 1

    print("load frozen constellation ...", flush=True)
    holder = load_frozen_constellation(device=str(args.device))
    checkpoint_paths = {
        name: path.resolve() for name, path in production_checkpoint_paths().items()
    }
    checkpoint_hashes = {
        name: sha256_file(path) for name, path in sorted(checkpoint_paths.items())
    }

    shell_seed = args.seed + 1
    climate_seed = args.seed + 2
    narrow_seed = args.seed + 3
    gene_rows: list[dict[str, Any]] = []
    truncated_window_count = 0
    genes_with_truncation = 0

    def make_window(record: CdsRecord, span: Span) -> dict[str, Any]:
        fragment = clean_acgt(
            record.sequence[(span.start_aa - 1) * 3 : span.end_aa * 3]
        )
        if len(fragment) < 36 or len(fragment) % 3:
            raise ValueError(
                f"invalid exact window for {record.accession}: "
                f"{span.start_aa}..{span.end_aa}"
            )
        walk_bytes = max(0, len(fragment) - 3)
        truncated = walk_bytes > _AE_READ_CAP_BYTES
        return {
            "source_start_aa": int(span.start_aa),
            "source_end_aa": int(span.end_aa),
            "start_aa": int(span.start_aa),
            "end_aa": int(span.end_aa),
            "nt": len(fragment),
            "sequence_sha256": hashlib.sha256(fragment.encode("utf-8")).hexdigest(),
            "mean_shell": float(mean_adjacent_shell(codons_of(fragment))),
            "ae_walk_bytes": int(walk_bytes),
            "ae_truncated": bool(truncated),
        }

    for record, spans in (
        (cds_by_accession[accession], selected[accession])
        for accession in selected_accessions
    ):
        tm_spans = [span for span in spans if span.label == "tm"]
        cytoplasmic_spans = [span for span in spans if span.label == "cytoplasmic"]
        periplasmic_spans = [span for span in spans if span.label == "periplasmic"]
        tm_windows = [make_window(record, span) for span in tm_spans]
        cytoplasmic_windows = [
            make_window(record, span) for span in cytoplasmic_spans
        ]
        periplasmic_windows = [
            make_window(record, span) for span in periplasmic_spans
        ]
        if not tm_windows or not cytoplasmic_windows:
            print(
                f"selected gene {record.accession} has no scorable window",
                file=sys.stderr,
            )
            return 1

        tm_sequences = [
            clean_acgt(
                record.sequence[
                    (int(window["start_aa"]) - 1) * 3 : int(window["end_aa"]) * 3
                ]
            )
            for window in tm_windows
        ]
        cytoplasmic_sequences = [
            clean_acgt(
                record.sequence[
                    (int(window["start_aa"]) - 1) * 3 : int(window["end_aa"]) * 3
                ]
            )
            for window in cytoplasmic_windows
        ]
        tm_gc = [gc_fraction(sequence) for sequence in tm_sequences]
        cytoplasmic_gc = [gc_fraction(sequence) for sequence in cytoplasmic_sequences]

        tm_reads = [read_constellation(holder, sequence) for sequence in tm_sequences]
        cytoplasmic_reads = [
            read_constellation(holder, sequence) for sequence in cytoplasmic_sequences
        ]
        for window, read in zip(tm_windows, tm_reads):
            window["ae_narrow_het"] = float(read.narrow_het)
            window["ae_super_climate"] = float(read.super_climate_mean)
        for window, read in zip(cytoplasmic_windows, cytoplasmic_reads):
            window["ae_narrow_het"] = float(read.narrow_het)
            window["ae_super_climate"] = float(read.super_climate_mean)

        shell_tm = float(np.mean([float(w["mean_shell"]) for w in tm_windows]))
        shell_cytoplasmic = float(
            np.mean([float(w["mean_shell"]) for w in cytoplasmic_windows])
        )
        climate_tm = float(np.mean([float(w["ae_super_climate"]) for w in tm_windows]))
        climate_cytoplasmic = float(
            np.mean([float(w["ae_super_climate"]) for w in cytoplasmic_windows])
        )
        narrow_tm = float(np.mean([float(w["ae_narrow_het"]) for w in tm_windows]))
        narrow_cytoplasmic = float(
            np.mean([float(w["ae_narrow_het"]) for w in cytoplasmic_windows])
        )
        gene_truncated = any(
            bool(window["ae_truncated"])
            for window in tm_windows + cytoplasmic_windows
        )
        truncated_window_count += sum(
            int(bool(window["ae_truncated"]))
            for window in tm_windows + cytoplasmic_windows
        )
        genes_with_truncation += int(gene_truncated)
        gene_rows.append(
            {
                "gene": record.gene_name,
                "accession": record.accession,
                "n_tm_windows": len(tm_windows),
                "n_cyto_windows": len(cytoplasmic_windows),
                "tm_nt": int(sum(int(window["nt"]) for window in tm_windows)),
                "cyto_nt": int(
                    sum(int(window["nt"]) for window in cytoplasmic_windows)
                ),
                "tm_gc": float(np.mean(tm_gc)),
                "cyto_gc": float(np.mean(cytoplasmic_gc)),
                "tm_mean_shell": shell_tm,
                "cyto_mean_shell": shell_cytoplasmic,
                "delta_shell_tm_minus_cyto": shell_tm - shell_cytoplasmic,
                "tm_super_climate": climate_tm,
                "cyto_super_climate": climate_cytoplasmic,
                "delta_climate_tm_minus_cyto": climate_tm - climate_cytoplasmic,
                "tm_narrow_het": narrow_tm,
                "cyto_narrow_het": narrow_cytoplasmic,
                "delta_narrow_tm_minus_cyto": narrow_tm - narrow_cytoplasmic,
                "tm_windows": tm_windows,
                "cyto_windows": cytoplasmic_windows,
                "periplasmic_windows": periplasmic_windows,
            }
        )

    n_genes = len(gene_rows)
    delta_shell = np.asarray(
        [float(row["delta_shell_tm_minus_cyto"]) for row in gene_rows],
        dtype=np.float64,
    )
    delta_climate = np.asarray(
        [float(row["delta_climate_tm_minus_cyto"]) for row in gene_rows],
        dtype=np.float64,
    )
    delta_narrow = np.asarray(
        [float(row["delta_narrow_tm_minus_cyto"]) for row in gene_rows],
        dtype=np.float64,
    )
    delta_gc = np.asarray(
        [float(row["tm_gc"]) - float(row["cyto_gc"]) for row in gene_rows],
        dtype=np.float64,
    )

    mean_shell = float(delta_shell.mean())
    mean_climate = float(delta_climate.mean())
    mean_narrow = float(delta_narrow.mean())
    mean_gc = float(delta_gc.mean())
    n_shell_positive = int((delta_shell > 0).sum())
    n_shell_negative = int((delta_shell < 0).sum())
    shell_p = paired_sign_flip_p(
        delta_shell,
        n_perm=args.permutations,
        seed=shell_seed,
        alternative="less",
    )
    climate_p_two_sided = paired_sign_flip_p(
        delta_climate,
        n_perm=args.permutations,
        seed=climate_seed,
        alternative="two-sided nonzero",
    )
    narrow_p = paired_sign_flip_p(
        delta_narrow,
        n_perm=args.permutations,
        seed=narrow_seed,
        alternative="less",
    )

    claim_kernel = bool(
        mean_shell <= -0.12
        and n_shell_negative >= int(0.7 * n_genes)
        and shell_p < 0.01
    )
    claim_ae = bool(
        mean_climate <= 0.02
        and climate_p_two_sided < 0.05
        and not claim_kernel
    )
    claim = bool(claim_kernel)
    cds_sha256 = sha256_file(cds_path)
    uniprot_sha256 = sha256_file(uniprot_path)
    cohort_hash = canonical_cohort_hash(
        selected_accessions, cds_sha256, uniprot_sha256, args.min_aa
    )
    timestamp = datetime.now(timezone.utc).isoformat()

    summary = {
        "status": "ok",
        "claim": claim,
        "claim_kernel_shell": claim_kernel,
        "claim_ae_super_climate": claim_ae,
        "claim_sentence": None,
        "n_genes": n_genes,
        "selected_accessions": selected_accessions,
        "all_eligible_accessions": population_accessions,
        "uses_frozen_constellation": True,
        "windowing": "per-topology-window mean (not concatenated)",
        "contrast": "within-gene TM windows minus cytoplasmic windows",
        "cohort_manifest": {
            "algorithm": "SHA-256 over sorted selected accessions, input SHA-256 values, and min_aa",
            "sha256": cohort_hash,
        },
        "atlas_cohort_manifest_sha256": cohort_hash,
        "counts": {
            "eligible_population": len(population),
            "selected": len(selected_accessions),
            "excluded_from_eligibility": int(sum(exclusion_counts.values())),
            "excluded_topology_spans": int(sum(span_exclusion_counts.values())),
            "clipped_topology_spans": 0,
            "genes_with_clipped_topology_spans": 0,
            "ae_truncated_windows": truncated_window_count,
            "genes_with_ae_truncation": genes_with_truncation,
            "null_draws": 0,
            "per_gene_null_draws": 0,
            "permutations": int(args.permutations),
            "sign_flip_permutations": int(args.permutations),
        },
        "exclusions": {
            "counts": exclusion_counts,
            "reasons": sorted(exclusion_counts),
            "accessions_by_reason": {
                reason: sorted(accessions)
                for reason, accessions in sorted(excluded_accessions.items())
            },
            "span_counts": span_exclusion_counts,
            "span_accessions_by_reason": {
                reason: sorted(accessions)
                for reason, accessions in sorted(span_excluded_accessions.items())
            },
            "span_details": span_exclusion_details,
        },
        "sampling": sampling,
        "permutation": {
            "test": "paired sign-flip permutation",
            "shell_test": "one-sided paired sign-flip permutation",
            "shell_alternative": "less",
            "shell_contrast": "TM-minus-cytoplasmic",
            "narrow_test": "one-sided paired sign-flip permutation",
            "narrow_alternative": "less",
            "super_test": "two-sided exploratory paired sign-flip permutation",
            "super_alternative": "two-sided nonzero",
            "contrast_direction": "TM-minus-cytoplasmic",
            "permutations": int(args.permutations),
            "n_permutations": int(args.permutations),
            "seeds": {
                "mean_shell": shell_seed,
                "super_climate": climate_seed,
                "narrow_het": narrow_seed,
            },
        },
        "estimator": {
            "unit": "gene",
            "pairing": "TM and cytoplasmic topology windows from the same accession",
            "window_statistic": "arithmetic mean adjacent-codon-pair shell per exact topology span",
            "gene_contrast": "arithmetic mean(TM window shells) minus arithmetic mean(cytoplasmic window shells)",
            "population_statistic": "arithmetic mean of gene-level contrasts",
            "coordinate_policy": _COORDINATE_POLICY,
            "terminal_span_rule": _TERMINAL_SPAN_RULE,
            "translated_stop_codon_rule": "a terminal stop codon is excluded from the translated residue count",
            "clipping_enabled": False,
        },
        "ae_read": {
            "models": ["narrow", "super"],
            "walk_cap_bytes": _AE_READ_CAP_BYTES,
            "truncation_disclosed": True,
            "kernel_shell_uses_full_window": True,
        },
        "stats": {
            "mean_delta_shell_tm_minus_cyto": mean_shell,
            "mean_delta_climate_tm_minus_cyto": mean_climate,
            "mean_delta_narrow_tm_minus_cyto": mean_narrow,
            "mean_delta_gc_tm_minus_cyto": mean_gc,
            "n_shell_positive": n_shell_positive,
            "n_shell_negative": n_shell_negative,
            "shell_perm_p_one_sided_less": shell_p,
            "climate_perm_p_two_sided_nonzero": climate_p_two_sided,
            "narrow_perm_p_one_sided_less": narrow_p,
        },
        "aa_null_reference": {
            "path": "RESULTS_script_2_null.json",
            "note": "computed by script_2_null.py on the exact selected atlas cohort",
        },
        "provenance": {
            "timestamp_utc": timestamp,
            "script": {"path": str(_SCRIPT), "sha256": sha256_file(_SCRIPT)},
            "software": {
                "python_executable": sys.executable,
                "python_version": sys.version,
                "platform": sys.platform,
            },
            "git": {"repository": str(_REPO), "commit": git_commit()},
            "dataset_ids": {
                "uniprot_proteome": "UP000000625",
                "uniprot_release": _UNIPROT_RELEASE,
                "uniprot_query": _UNIPROT_QUERY,
                "ncbi_refseq": "NC_000913.3",
            },
            "inputs": {
                "cds": {
                    "dataset_id": "NC_000913.3",
                    "path": str(cds_path),
                    "sha256": cds_sha256,
                },
                "uniprot": {
                    "dataset_id": "UP000000625",
                    "release": _UNIPROT_RELEASE,
                    "query": _UNIPROT_QUERY,
                    "path": str(uniprot_path),
                    "sha256": uniprot_sha256,
                },
            },
            "production_checkpoints": {
                name: {"path": str(checkpoint_paths[name]), "sha256": digest}
                for name, digest in checkpoint_hashes.items()
            },
            "cli": {
                "min_aa": int(args.min_aa),
                "sample_size": args.sample_size,
                "seed": int(args.seed),
                "permutations": int(args.permutations),
                "device": str(args.device),
                "output_json": str(out_json),
            },
            "seeds": {
                "sampling": int(args.seed),
                "mean_shell_permutation": shell_seed,
                "super_climate_permutation": climate_seed,
                "narrow_het_permutation": narrow_seed,
            },
            "artifact": {
                "json": str(out_json),
                "cohort_manifest_sha256": cohort_hash,
            },
        },
        "genes": gene_rows,
    }

    lines = [
        "stage=atlas",
        f"kernel_shell_thresholds_met={claim_kernel} ae_super_thresholds_met={claim_ae}",
        f"n={n_genes} selected genes; eligible_population={len(population)}",
        f"mean delta shell (TM-cyto)={mean_shell:+.4f} p={shell_p:.4f} one-sided less ({n_shell_positive}+/{n_shell_negative}-)",
        f"mean delta Super climate={mean_climate:+.4f} p={climate_p_two_sided:.4f} two-sided nonzero",
        f"mean delta Narrow het={mean_narrow:+.4f} p={narrow_p:.4f} one-sided less",
        f"mean delta GC={mean_gc:+.4f}",
        f"ae_truncated_windows={truncated_window_count}; kernel_uses_full_windows=true",
        "clipped_topology_spans=0; genes_with_clipped_topology_spans=0; coordinate_policy=exclusion_not_clipping",
        f"cohort_manifest_sha256={cohort_hash}",
        "AA-null reference: RESULTS_script_2_null.json (exact selected atlas cohort; see JSON)",
    ]
    text_payload = "\n".join(lines) + "\n"
    write_json(out_json, summary)
    upsert_results("atlas", [ln for ln in text_payload.splitlines() if ln.strip()])
    print(text_payload, flush=True)
    print(f"wrote {out_json.name} and RESULTS.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
