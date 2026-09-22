"""Shared helpers and naming rules for the genomics topology program.

Naming convention (all artifacts live in this package folder; no temp/scratch
output paths):

  common.py
      Shared helpers and RESULT_JSON / RESULTS path constants.
  run.py
      Pipeline entry: atlas -> null -> narrow -> claim -> holdout -> order.
  script_<N>_<word>.py
      One script per stage or probe. N is the order index. <word> is one
      English token (atlas, null, narrow, claim, holdout, order).
  RESULTS.txt
      Single human-readable summary. Sections named after the script word
      (atlas, null, narrow, claim, holdout, order).
  RESULTS_script_<N>_<word>.json
      Machine-readable output for that script. Each JSON is also the rebuild
      cache: run.py skips a stage when this file exists and its provenance
      script sha256 matches the script file on disk.

Do not invent parallel names (membrane_topology_*, *_climate_*, paired .txt
side files). New outputs must follow RESULTS_script_<N>_<word>.json and update
RESULT_JSON in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar, cast

import gzip
import hashlib
import itertools
import re

import numpy as np

from src.tools.autoencoder.programs.genomics.genomics import (
    BASES,
    clean_acgt,
    translation_table,
)
from src.tools.autoencoder.programs.genomics.synthesis.climate import (
    codons_of,
    mean_adjacent_shell,
)
from src.tools.autoencoder.programs.genomics.synthesis.constellation import (
    production_checkpoint_paths,
)

T = TypeVar("T")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def as_dict(value: object, message: str) -> dict[str, Any]:
    require(isinstance(value, dict), message)
    return cast(dict[str, Any], value)


def as_list(value: object, message: str) -> list[Any]:
    require(isinstance(value, list), message)
    return cast(list[Any], value)


def as_str(value: object, message: str) -> str:
    require(isinstance(value, str) and bool(value), message)
    return str(value)


def as_int(value: object, message: str) -> int:
    require(isinstance(value, (int, float, str)) and not isinstance(value, bool), message)
    return int(value)  # type: ignore[arg-type]


def as_float(value: object, message: str) -> float:
    require(isinstance(value, (int, float, str)) and not isinstance(value, bool), message)
    return float(value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FastaRecord:
    """A complete coding-sequence record with optional UniProt annotations."""

    accession: str
    sequence: str
    gene_name: str = ""
    locus_tag: str = ""


def _fasta_records(path: Path) -> Iterable[FastaRecord]:
    """Yield complete UniProt-linked CDS records in one file pass."""
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    header = ""
    sequence: list[str] = []

    def store() -> FastaRecord | None:
        if not header or not sequence:
            return None
        accession_match = re.search(
            r"UniProtKB/(?:Swiss-Prot|TrEMBL):([A-Z0-9]+(?:-\d+)?)",
            header,
        )
        if accession_match is None:
            return None
        cleaned = clean_acgt("".join(sequence).upper().replace("U", "T"))
        if len(cleaned) < 90 or len(cleaned) % 3:
            return None
        gene_match = re.search(r"\[gene=([^\]]+)\]", header)
        locus_match = re.search(r"\[locus_tag=([^\]]+)\]", header)
        return FastaRecord(
            accession=accession_match.group(1),
            sequence=cleaned,
            gene_name=gene_match.group(1) if gene_match else "",
            locus_tag=locus_match.group(1) if locus_match else "",
        )

    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                record = store()
                if record is not None:
                    yield record
                header = line.strip()
                sequence = []
            else:
                sequence.append(line.strip())
        record = store()
        if record is not None:
            yield record


def read_fasta_records(path: Path) -> dict[str, FastaRecord]:
    """Read complete UniProt CDS records with gene and locus annotations."""
    records: dict[str, FastaRecord] = {}
    for record in _fasta_records(path):
        if record.accession in records:
            raise ValueError(f"duplicate CDS accession {record.accession}")
        records[record.accession] = record
    return records


def read_fasta_accession_records(path: Path) -> dict[str, str]:
    """Read complete UniProt CDS records keyed by accession."""
    return {
        accession: record.sequence
        for accession, record in read_fasta_records(path).items()
    }


def read_fasta_gene_records(path: Path) -> dict[str, str]:
    """Index complete CDS records by gene name, falling back to accession."""
    records: dict[str, str] = {}
    for record in read_fasta_records(path).values():
        key = record.gene_name or record.locus_tag or record.accession
        if key in records:
            raise ValueError(f"duplicate CDS gene key {key}")
        records[key] = record.sequence
    return records


def protein_of(seq: str, *, table_id: int = 11) -> str:
    """Translate a complete canonical CDS with the requested genetic code."""
    sequence = clean_acgt(seq)
    require(
        bool(sequence) and len(sequence) % 3 == 0,
        "protein_of needs complete codons",
    )
    code = translation_table(table_id)
    return "".join(
        code.get(sequence[index : index + 3], "X")
        for index in range(0, len(sequence), 3)
    )


def verify_production_checkpoints(atlas: dict[str, Any]) -> dict[str, str]:
    """Recompute production checkpoint digests and match them to the atlas.

    Downstream stages re-read the frozen checkpoints from disk and require the
    live digests to equal the digests recorded in the atlas provenance. A
    checkpoint that changed under a cached atlas stops the stage.
    """
    atlas_provenance = as_dict(atlas.get("provenance"), "atlas is missing provenance")
    atlas_checkpoints = as_dict(
        atlas_provenance.get("production_checkpoints"),
        "atlas is missing production_checkpoints",
    )
    require(bool(atlas_checkpoints), "atlas production_checkpoints is empty")
    live = production_checkpoint_paths()
    require(
        set(atlas_checkpoints) == set(live),
        "atlas checkpoint set differs from the production checkpoints",
    )
    digests: dict[str, str] = {}
    for name in sorted(atlas_checkpoints):
        recorded = atlas_checkpoints[name]
        require(isinstance(recorded, dict), f"atlas checkpoint {name} is malformed")
        live_digest = sha256_file(live[name])
        require(
            str(recorded.get("sha256", "")) == live_digest,
            f"atlas checkpoint digest differs from the live {name} checkpoint",
        )
        digests[name] = live_digest
    return digests


def ensure(value: T | None, message: str) -> T:
    require(value is not None, message)
    return cast(T, value)


# ---- null machinery ----

_CODE = translation_table(11)
_CODONS = tuple("".join(pair) for pair in itertools.product(BASES, repeat=3))
_FIBERS: dict[str, tuple[str, ...]] = {
    amino_acid: tuple(sorted(fiber))
    for amino_acid, fiber in (
        (
            amino_acid,
            [codon for codon in _CODONS if _CODE.get(codon) == amino_acid],
        )
        for amino_acid in {_CODE[codon] for codon in _CODONS}
        if amino_acid != "*"
    )
}
_DINUCS = tuple(
    "".join(pair) for pair in itertools.product(BASES, repeat=2)
)
_DINUC_INDEX = {dinuc: index for index, dinuc in enumerate(_DINUCS)}


def codon_usage_counts(sequence: str) -> dict[str, int]:
    """Count complete in-frame codons after retaining canonical bases."""
    sequence = clean_acgt(sequence)
    counts: dict[str, int] = {}
    for codon in codons_of(sequence):
        counts[codon] = counts.get(codon, 0) + 1
    return counts


def usage_weighted_synonymize(
    seq: str,
    rng: np.random.Generator,
    usage: dict[str, float],
) -> str:
    """Draw an amino-acid-preserving spelling from host codon frequencies."""
    require(bool(usage), "codon-usage weights are empty")
    output: list[str] = []
    for codon in codons_of(clean_acgt(seq)):
        amino_acid = _CODE.get(codon, "X")
        fiber = _FIBERS.get(amino_acid)
        if fiber is None:
            output.append(codon)
            continue
        codons = list(fiber)
        weights = np.asarray([max(float(usage.get(codon, 0.0)), 0.0) for codon in codons])
        total = float(weights.sum())
        require(total > 0.0, f"no host usage weight for amino acid {amino_acid}")
        output.append(codons[int(rng.choice(len(codons), p=weights / total))])
    return "".join(output)


def usage_weighted_synonymize_many(
    seq: str,
    rng: np.random.Generator,
    usage: dict[str, float],
    n: int,
) -> list[str]:
    """Draw n usage-matched amino-acid-preserving spellings."""
    require(n >= 0, "number of draws must be non-negative")
    require(bool(usage), "codon-usage weights are empty")
    codons = list(codons_of(clean_acgt(seq)))
    columns: list[list[str]] = []
    for codon in codons:
        amino_acid = _CODE.get(codon, "X")
        fiber = _FIBERS.get(amino_acid)
        if fiber is None:
            columns.append([codon] * n)
            continue
        choices = list(fiber)
        weights = np.asarray(
            [max(float(usage.get(choice, 0.0)), 0.0) for choice in choices],
            dtype=np.float64,
        )
        total = float(weights.sum())
        require(total > 0.0, f"no host usage weight for amino acid {amino_acid}")
        picks = rng.choice(len(choices), size=n, p=weights / total)
        columns.append([choices[int(index)] for index in picks])
    return ["".join(row) for row in zip(*columns)]


def junction_matched_choice_counts(seq: str) -> list[int]:
    """Count synonymous choices available at each exact-junction constraint."""
    sequence = clean_acgt(seq)
    require(len(sequence) >= 6 and len(sequence) % 3 == 0, "need at least two codons")
    codons = list(codons_of(sequence))
    junctions = [sequence[i : i + 2] for i in range(2, len(sequence) - 1, 3)]
    choices: list[int] = []
    for index, codon in enumerate(codons):
        amino_acid = _CODE.get(codon, "X")
        fiber = list(_FIBERS.get(amino_acid, (codon,)))
        if index == 0:
            allowed = [c for c in fiber if c[2] == junctions[0][0]] if junctions else fiber
        elif index == len(codons) - 1:
            allowed = [c for c in fiber if c[0] == junctions[index - 1][1]]
        else:
            allowed = [
                c
                for c in fiber
                if c[0] == junctions[index - 1][1] and c[2] == junctions[index][0]
            ]
        choices.append(len(allowed))
    return choices


def synonymize(seq: str, rng: np.random.Generator) -> str:
    """Draw one amino-acid-preserving spelling with the bacterial code."""
    output: list[str] = []
    for codon in codons_of(clean_acgt(seq)):
        amino_acid = _CODE.get(codon, "X")
        fiber = _FIBERS.get(amino_acid)
        output.append(codon if fiber is None else str(rng.choice(fiber)))
    return "".join(output)


def synonymize_many(seq: str, rng: np.random.Generator, n: int) -> list[str]:
    """Draw n amino-acid-preserving spellings; fiber choices drawn in batches."""
    if n <= 0:
        return []
    codons = list(codons_of(clean_acgt(seq)))
    columns: list[list[str]] = []
    for codon in codons:
        amino_acid = _CODE.get(codon, "X")
        fiber = _FIBERS.get(amino_acid)
        if fiber is None:
            columns.append([codon] * n)
            continue
        picks = rng.integers(0, len(fiber), size=n)
        columns.append([fiber[int(i)] for i in picks])
    return ["".join(row) for row in zip(*columns)]


def synonymize_junction_matched(seq: str, rng: np.random.Generator) -> str:
    """Synonymous redraw that exactly preserves codon-junction dinucleotides.

    Junctions are the dinucleotides straddling codon boundaries (bases at
    positions 3k+2 and 3k+3). This is the strong dinuc-matched null: order
    inside codons may change, junction letters may not.
    """
    sequence = clean_acgt(seq)
    require(len(sequence) >= 6 and len(sequence) % 3 == 0, "need at least two codons")
    codons = list(codons_of(sequence))
    amino_acids = [_CODE.get(codon, "X") for codon in codons]
    junctions = [sequence[i : i + 2] for i in range(2, len(sequence) - 1, 3)]
    require(len(junctions) == len(codons) - 1, "junction count mismatch")

    out: list[str] = []
    for index, amino_acid in enumerate(amino_acids):
        fiber = list(_FIBERS.get(amino_acid, (codons[index],)))
        if index == 0:
            allowed = [c for c in fiber if c[2] == junctions[0][0]] if junctions else fiber
        elif index == len(amino_acids) - 1:
            allowed = [c for c in fiber if c[0] == junctions[index - 1][1]]
        else:
            allowed = [
                c
                for c in fiber
                if c[0] == junctions[index - 1][1] and c[2] == junctions[index][0]
            ]
        if not allowed:
            # Exact match impossible at this site under the code; keep native codon.
            out.append(codons[index])
        else:
            out.append(str(rng.choice(allowed)))
    result = "".join(out)
    # Exact junction identity (strong control invariant).
    for i, junction in enumerate(junctions):
        require(
            result[3 * i + 2 : 3 * i + 4] == junction,
            "junction-matched synonymize broke a junction",
        )
    require(
        "".join(_CODE.get(c, "X") for c in codons_of(result))
        == "".join(amino_acids),
        "junction-matched synonymize changed the protein",
    )
    return result


def dinucleotide_profile(
    seq: str, *, junction_only: bool = False
) -> np.ndarray:
    """Return normalized adjacent or codon-junction dinucleotide frequencies."""
    sequence = clean_acgt(seq)
    if junction_only:
        positions = range(2, len(sequence) - 1, 3)
    else:
        positions = range(len(sequence) - 1)
    counts = np.zeros(len(_DINUCS), dtype=np.float64)
    for position in positions:
        dinuc = sequence[position : position + 2]
        index = _DINUC_INDEX.get(dinuc)
        if index is not None:
            counts[index] += 1.0
    total = float(counts.sum())
    return counts / total if total > 0 else counts


def shell_value(seq: str) -> float:
    return float(mean_adjacent_shell(codons_of(clean_acgt(seq))))


def dinuc_l1(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left - right).sum())


@dataclass(frozen=True)
class DinucResidualizer:
    """Linear map used to remove junction-dinucleotide variation."""

    intercept: float
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray

    def predict(self, shells: np.ndarray, profiles: np.ndarray) -> np.ndarray:
        standardized = (profiles - self.feature_mean) / self.feature_scale
        return self.intercept + standardized @ self.coefficients

    def residual(self, shells: np.ndarray, profiles: np.ndarray) -> np.ndarray:
        return np.asarray(shells, dtype=np.float64) - self.predict(
            shells, profiles
        )


def fit_dinuc_residualizer(
    profiles: np.ndarray,
    shells: np.ndarray,
    *,
    ridge: float,
) -> DinucResidualizer:
    """Fit shell ~ junction dinucleotides with a stable ridge penalty."""
    x = np.asarray(profiles, dtype=np.float64)
    y = np.asarray(shells, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("dinucleotide profiles and shell values have incompatible shapes")
    if x.shape[0] < 2 or x.shape[1] == 0:
        raise ValueError("at least two null draws and one dinucleotide feature are required")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    feature_scale = np.where(feature_scale > 1e-8, feature_scale, 1.0)
    standardized = (x - feature_mean) / feature_scale
    centered_y = y - float(y.mean())
    gram = standardized.T @ standardized
    gram += float(ridge) * np.eye(standardized.shape[1], dtype=np.float64)
    rhs = standardized.T @ centered_y
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return DinucResidualizer(
        intercept=float(y.mean()),
        coefficients=coefficients,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )


@dataclass(frozen=True)
class DinucControlledContrast:
    """One gene's native contrast after a dinucleotide-controlled null."""

    native_contrast: float
    null_contrasts: np.ndarray
    adjusted_residual: float
    raw_native_contrast: float
    raw_null_contrasts: np.ndarray
    native_tm_profile: np.ndarray
    native_cyto_profile: np.ndarray
    null_tm_profile_l1: np.ndarray
    null_cyto_profile_l1: np.ndarray
    residualizer: DinucResidualizer
    ridge: float

    @property
    def null_mean(self) -> float:
        return float(self.null_contrasts.mean()) if self.null_contrasts.size > 0 else float("nan")

    @property
    def null_sd(self) -> float:
        return float(self.null_contrasts.std(ddof=1)) if self.null_contrasts.size > 1 else float("nan")


def dinuc_controlled_contrast(
    tm_native: Iterable[str],
    cyto_native: Iterable[str],
    tm_null_by_draw: Iterable[Iterable[str]],
    cyto_null_by_draw: Iterable[Iterable[str]],
    *,
    ridge: float,
) -> DinucControlledContrast:
    """Residualize shell on junction dinucleotides within one gene.

    ``tm_null_by_draw[d]`` / ``cyto_null_by_draw[d]`` are the synonymous
    window sequences for draw ``d``. One junction-dinuc map is fit on all
    null windows across draws. Native and every null draw are residualized
    through that map, then each draw forms its own TM-minus-cytoplasmic
    contrast. The null side is NOT centered to zero.
    """
    tm_native_sequences = [clean_acgt(seq) for seq in tm_native]
    cyto_native_sequences = [clean_acgt(seq) for seq in cyto_native]
    tm_draws = [[clean_acgt(seq) for seq in draw] for draw in tm_null_by_draw]
    cyto_draws = [[clean_acgt(seq) for seq in draw] for draw in cyto_null_by_draw]

    if not tm_native_sequences or not cyto_native_sequences:
        raise ValueError("native TM and cytoplasmic windows are required")
    if not tm_draws or not cyto_draws:
        raise ValueError("at least one TM and cytoplasmic null draw is required")
    if len(tm_draws) != len(cyto_draws):
        raise ValueError("TM and cytoplasmic null draw counts disagree")

    tm_native_profiles = np.asarray(
        [dinucleotide_profile(seq, junction_only=True) for seq in tm_native_sequences],
        dtype=np.float64,
    )
    cyto_native_profiles = np.asarray(
        [dinucleotide_profile(seq, junction_only=True) for seq in cyto_native_sequences],
        dtype=np.float64,
    )
    tm_native_shells = np.asarray(
        [shell_value(seq) for seq in tm_native_sequences], dtype=np.float64
    )
    cyto_native_shells = np.asarray(
        [shell_value(seq) for seq in cyto_native_sequences], dtype=np.float64
    )
    raw_native_contrast = float(tm_native_shells.mean() - cyto_native_shells.mean())

    # Flatten all null windows for the residualizer fit.
    all_null_seqs: list[str] = []
    for draw in tm_draws:
        all_null_seqs.extend(draw)
    for draw in cyto_draws:
        all_null_seqs.extend(draw)
    all_null_profiles = np.asarray(
        [dinucleotide_profile(seq, junction_only=True) for seq in all_null_seqs],
        dtype=np.float64,
    )
    all_null_shells = np.asarray(
        [shell_value(seq) for seq in all_null_seqs], dtype=np.float64
    )
    residualizer = fit_dinuc_residualizer(
        all_null_profiles, all_null_shells, ridge=ridge
    )

    native_tm_residual = residualizer.residual(tm_native_shells, tm_native_profiles)
    native_cyto_residual = residualizer.residual(cyto_native_shells, cyto_native_profiles)
    native_contrast = float(native_tm_residual.mean() - native_cyto_residual.mean())

    raw_null_contrasts: list[float] = []
    per_draw_null_contrasts: list[float] = []
    null_tm_l1_vals: list[float] = []
    null_cyto_l1_vals: list[float] = []
    native_tm_profile = tm_native_profiles.mean(axis=0)
    native_cyto_profile = cyto_native_profiles.mean(axis=0)

    for tm_draw, cyto_draw in zip(tm_draws, cyto_draws):
        tm_shells = np.asarray([shell_value(seq) for seq in tm_draw], dtype=np.float64)
        cyto_shells = np.asarray([shell_value(seq) for seq in cyto_draw], dtype=np.float64)
        tm_profiles = np.asarray(
            [dinucleotide_profile(seq, junction_only=True) for seq in tm_draw],
            dtype=np.float64,
        )
        cyto_profiles = np.asarray(
            [dinucleotide_profile(seq, junction_only=True) for seq in cyto_draw],
            dtype=np.float64,
        )
        raw_null_contrasts.append(float(tm_shells.mean() - cyto_shells.mean()))
        tm_resid = residualizer.residual(tm_shells, tm_profiles)
        cyto_resid = residualizer.residual(cyto_shells, cyto_profiles)
        per_draw_null_contrasts.append(float(tm_resid.mean() - cyto_resid.mean()))
        for profile in tm_profiles:
            null_tm_l1_vals.append(dinuc_l1(profile, native_tm_profile))
        for profile in cyto_profiles:
            null_cyto_l1_vals.append(dinuc_l1(profile, native_cyto_profile))

    null_contrasts = np.asarray(per_draw_null_contrasts, dtype=np.float64)
    raw_null = np.asarray(raw_null_contrasts, dtype=np.float64)
    if null_contrasts.size == 0:
        adjusted_residual = float("nan")
    else:
        adjusted_residual = native_contrast - float(null_contrasts.mean())

    return DinucControlledContrast(
        native_contrast=native_contrast,
        null_contrasts=null_contrasts,
        adjusted_residual=adjusted_residual,
        raw_native_contrast=raw_native_contrast,
        raw_null_contrasts=raw_null,
        native_tm_profile=native_tm_profile,
        native_cyto_profile=native_cyto_profile,
        null_tm_profile_l1=np.asarray(null_tm_l1_vals, dtype=np.float64),
        null_cyto_profile_l1=np.asarray(null_cyto_l1_vals, dtype=np.float64),
        residualizer=residualizer,
        ridge=float(ridge),
    )


# ---- result artifacts ----
# Paths follow the naming rules in this module's docstring.

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "RESULTS.txt"

# One human RESULTS.txt; one JSON per script, all inside topology/.
RESULT_JSON = {
    "atlas": HERE / "RESULTS_script_1_atlas.json",
    "null": HERE / "RESULTS_script_2_null.json",
    "narrow": HERE / "RESULTS_script_3_narrow.json",
    "claim": HERE / "RESULTS_script_4_claim.json",
    "holdout": HERE / "RESULTS_script_5_holdout.json",
    "order": HERE / "RESULTS_script_6_order.json",
}
_SECTION_ORDER = (
    "atlas",
    "null",
    "narrow",
    "claim",
    "holdout",
    "order",
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one JSON artifact."""
    import json
    import os

    def _default(obj: object) -> object:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False, default=_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def upsert_results(section: str, lines: list[str]) -> None:
    """Update one named block in RESULTS.txt; keep other blocks."""
    section = section.strip().lower()
    if section not in _SECTION_ORDER:
        raise ValueError(f"unknown RESULTS section: {section}")

    existing: dict[str, list[str]] = {name: [] for name in _SECTION_ORDER}
    if RESULTS.is_file():
        current: str | None = None
        for raw in RESULTS.read_text(encoding="utf-8").splitlines():
            key = raw.strip().lower()
            if key in _SECTION_ORDER:
                current = key
                existing[current] = []
                continue
            if current is not None and raw.strip() and raw.strip() != "-----":
                if raw.strip() == "genomics topology":
                    continue
                existing[current].append(raw)

    existing[section] = list(lines)
    out: list[str] = ["genomics topology", ""]
    for name in _SECTION_ORDER:
        body = existing[name]
        if not body:
            continue
        out.append("-----")
        out.append(name)
        out.extend(body)
        out.append("")
    RESULTS.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")

