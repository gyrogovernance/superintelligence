"""Per-host preparation cache and catalog extraction for genomics synthesis censuses.

All catalog paths resolve through GENOMICS_DIR (ingest_genomics). Never read
the analysis tree's data/catalogs/genomics.
"""

from __future__ import annotations

import gzip
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.tools.autoencoder.programs.genomics.genomics import (
    GENOMICS_DIR,
    clean_acgt,
    genomic_byte_stream,
    pair_inversion_encoding,
)
from src.tools.autoencoder.programs.genomics.synthesis.climate import (
    climate_of_sequence,
    gc_matched_shuffle,
    protein_fixed_resample,
)

HOST_CDS: dict[str, tuple[str, int]] = {
    "ecoli": ("ecoli_k12_cds.fna.gz", 11),
    "yeast": ("yeast_s288c_cds.fna.gz", 1),
    "sars": ("sars_cov2_cds.fna.gz", 1),
}

# SARS has only ~12 annotated CDS; keep short ORFs so the host is usable.
HOST_MIN_BASES: dict[str, int] = {
    "ecoli": 300,
    "yeast": 300,
    "sars": 100,
}

HOST_FULL: dict[str, tuple[str, int | None]] = {
    "ecoli": ("ecoli_k12_full.fna.gz", None),
    "yeast": ("yeast_s288c_cds.fna.gz", 2_000_000),
    "sars": ("sars_cov2.fna.gz", None),
}

SPLICE_WINDOW = 128
MAX_SPLICE = 400


@dataclass
class GeneRecord:
    gene_id: str
    seq: str
    stream: list[int]
    compile: Any
    climate: Any


@dataclass
class HostCache:
    host: str
    genes: list[GeneRecord]
    full_stream: list[int] | None = None


def reverse_complement_seq(seq: str) -> str:
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(comp.get(c, "N") for c in reversed(seq.upper()))


def read_fasta_seq(
    path: Path,
    *,
    max_bases: int | None = None,
    keep_n: bool = False,
) -> str:
    """Read a FASTA sequence.

    When ``keep_n`` is True (chr22 genomic coordinates), uppercase and map U->T
    but keep N so GTF coordinates stay aligned. When False, return ACGT-only
    sequence via ``clean_acgt`` (catalog hosts without coordinate maps).
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    chunks: list[str] = []
    total = 0
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            raw = line.strip().upper().replace("U", "T")
            if keep_n:
                s = "".join(c for c in raw if c in "ACGTN")
            else:
                s = "".join(c for c in raw if c in "ACGT")
            if not s:
                continue
            chunks.append(s)
            total += len(s)
            if max_bases is not None and total >= max_bases:
                break
    seq = "".join(chunks)
    if max_bases is not None:
        seq = seq[:max_bases]
    if keep_n:
        return seq
    return clean_acgt(seq)


def read_cds_list(
    path: Path, *, min_bases: int = 300, max_records: int = 200
) -> list[tuple[str, str]]:
    """Read CDS records; drop any record that contains non-ACGT bases.

    Stripping ambiguous bases would shift the reading frame, so those records
    are excluded rather than silently cleaned.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    out: list[tuple[str, str]] = []
    header = ""
    buf: list[str] = []

    def flush() -> None:
        nonlocal header, buf
        if not header or not buf:
            header, buf = "", []
            return
        raw = "".join(buf).upper().replace("U", "T")
        if any(c not in "ACGT" for c in raw):
            header, buf = "", []
            return
        seq = raw
        if len(seq) >= min_bases and len(seq) % 3 == 0:
            out.append((header.split()[0].lstrip(">"), seq))
        header, buf = "", []

    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                flush()
                if len(out) >= max_records:
                    break
                header = line.strip()
                continue
            if len(out) >= max_records:
                break
            buf.append(line.strip())
        if len(out) < max_records:
            flush()
    return out


def byte_stream(seq: str) -> list[int]:
    enc = pair_inversion_encoding()
    return [int(b) for b in genomic_byte_stream(clean_acgt(seq), enc)]


def prepare_host(
    host: str,
    *,
    max_genes: int = 200,
    seed: int = 42,
    need_full_stream: bool = False,
) -> HostCache:
    if host not in HOST_CDS:
        raise ValueError(f"unknown host {host!r}")
    cds_name, _ncbi = HOST_CDS[host]
    path = GENOMICS_DIR / cds_name
    if not path.exists():
        raise FileNotFoundError(f"missing catalog {path}")
    min_bases = int(HOST_MIN_BASES.get(host, 300))
    genes_raw = read_cds_list(path, min_bases=min_bases, max_records=max_genes)
    genes: list[GeneRecord] = []
    for gid, seq in genes_raw:
        stream = byte_stream(seq)
        if len(stream) < 8:
            continue
        clim = climate_of_sequence(seq, label=gid, certificates=True)
        genes.append(
            GeneRecord(
                gene_id=gid,
                seq=seq,
                stream=stream,
                compile=None,
                climate=clim,
            )
        )
    full_stream: list[int] | None = None
    if need_full_stream and host in HOST_FULL:
        fname, cap = HOST_FULL[host]
        fpath = GENOMICS_DIR / fname
        if fpath.exists():
            full_seq = read_fasta_seq(fpath, max_bases=cap, keep_n=False)
            full_stream = byte_stream(full_seq)
    return HostCache(host=host, genes=genes, full_stream=full_stream)


def load_chr22_sequence() -> str:
    """Full chr22 sequence with N kept so GTF exon coordinates stay valid."""
    path = GENOMICS_DIR / "chr22.fa.gz"
    if not path.exists():
        return ""
    return read_fasta_seq(path, keep_n=True)


def _parse_gtf_exons(path: Path) -> dict[str, list[tuple[int, int, str]]]:
    by_tx: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9 or parts[2] != "exon" or parts[0] not in {"chr22", "22"}:
                continue
            tx = "tx"
            for tok in parts[8].split(";"):
                if "transcript_id" in tok:
                    tx = tok.split('"')[1] if '"' in tok else tx
            by_tx[tx].append((int(parts[3]) - 1, int(parts[4]), parts[6]))
    return by_tx


def extract_chr22_splice_flanks(
    *,
    max_per_class: int = 1500,
) -> tuple[list[str], list[str]]:
    """Analysis §20 8-bp donor/acceptor flanks (gates 144/145).

    Plus: donor ``[exon_end-2 : exon_end+6]``, acceptor
    ``[next_exon_start-6 : next_exon_start+2]``. Minus: RC of the symmetric
    genomic spans. Caps at ``max_per_class`` sequences per class.
    """
    seq = load_chr22_sequence()
    gtf = GENOMICS_DIR / "gencode.v47.annotation.gtf.gz"
    if not seq or not gtf.exists():
        return [], []
    by_tx = _parse_gtf_exons(gtf)
    nseq = len(seq)
    donors: list[str] = []
    acceptors: list[str] = []
    for _tx, spans in by_tx.items():
        if len(donors) >= max_per_class:
            break
        spans = sorted(spans, key=lambda t: t[0])
        if not spans:
            continue
        strand = spans[0][2]
        for i, (a, b, _s) in enumerate(spans):
            if strand == "+":
                if b >= 2 and b + 6 <= nseq:
                    donors.append(seq[b - 2 : b + 6].upper())
                if i + 1 < len(spans):
                    na = spans[i + 1][0]
                    if na >= 6:
                        acceptors.append(seq[na - 6 : na + 2].upper())
            else:
                if a >= 6:
                    donors.append(
                        reverse_complement_seq(seq[max(0, a - 6) : a + 2])
                    )
                if i + 1 < len(spans):
                    nb = spans[i + 1][1]
                    if nb + 6 <= nseq:
                        acceptors.append(
                            reverse_complement_seq(seq[nb - 2 : nb + 6])
                        )
    donors = [w for w in donors if len(w) == 8 and all(c in "ACGT" for c in w)]
    acceptors = [
        w for w in acceptors if len(w) == 8 and all(c in "ACGT" for c in w)
    ]
    return donors[:max_per_class], acceptors[:max_per_class]


def extract_chr22_splice_windows(
    *,
    window: int = SPLICE_WINDOW,
    max_per_class: int = MAX_SPLICE,
) -> tuple[list[str], list[str]]:
    """Strand-aware donor/acceptor windows from chr22 + GENCODE under GENOMICS_DIR.

    Transcript-oriented windows of length ``window`` (default 128). Requires
    GT at the donor intron start and AG at the acceptor intron end on the
    transcript strand (after RC for minus). Used by the AE context probe
    ``splice_chr22``; family-sheet mu uses ``extract_chr22_splice_flanks``.
    """
    seq = load_chr22_sequence()
    gtf = GENOMICS_DIR / "gencode.v47.annotation.gtf.gz"
    if not seq or not gtf.exists():
        return [], []
    by_tx = _parse_gtf_exons(gtf)
    nseq = len(seq)
    half = window // 2
    donors: list[str] = []
    acceptors: list[str] = []
    for _tx, spans in list(by_tx.items())[:30000]:
        if len(donors) >= max_per_class and len(acceptors) >= max_per_class:
            break
        spans = sorted(spans)
        if not spans:
            continue
        strand = spans[0][2]
        for i, (a, b, _s) in enumerate(spans):
            if i + 1 >= len(spans):
                continue
            na, nb, _ = spans[i + 1]
            if strand != "-":
                # Donor: exon end at b; intron starts at b with GT.
                if b + 2 <= nseq and seq[b : b + 2].upper() == "GT":
                    lo = max(0, b - half)
                    hi = min(nseq, lo + window)
                    if hi - lo == window:
                        w = seq[lo:hi].upper()
                        if all(c in "ACGT" for c in w):
                            donors.append(w)
                # Acceptor: exon start at na; intron ends with AG before na.
                if na >= 2 and seq[na - 2 : na].upper() == "AG":
                    hi = min(nseq, na + half)
                    lo = max(0, hi - window)
                    if hi - lo == window:
                        w = seq[lo:hi].upper()
                        if all(c in "ACGT" for c in w):
                            acceptors.append(w)
            else:
                # Minus: donor at exon start a (RC), acceptor at exon end nb (RC).
                if a >= 2 and reverse_complement_seq(seq[a - 2 : a])[:2] == "GT":
                    # Use genomic coords then RC the window.
                    lo = max(0, a - half)
                    hi = min(nseq, lo + window)
                    if hi - lo == window:
                        w = reverse_complement_seq(seq[lo:hi])
                        if all(c in "ACGT" for c in w) and "GT" in w[half - 2 : half + 2]:
                            donors.append(w)
                if nb + 2 <= nseq:
                    frag = reverse_complement_seq(seq[nb : nb + 2])
                    if frag[:2] == "AG":
                        hi = min(nseq, nb + half)
                        lo = max(0, hi - window)
                        if hi - lo == window:
                            w = reverse_complement_seq(seq[lo:hi])
                            if all(c in "ACGT" for c in w):
                                acceptors.append(w)
            if len(donors) >= max_per_class and len(acceptors) >= max_per_class:
                break
    return donors[:max_per_class], acceptors[:max_per_class]


_TSS_CONF_KEEP = frozenset({"C", "S"})


def _parse_rdb_span(text: str) -> tuple[int, int] | None:
    text = text.strip()
    if "-" not in text:
        return None
    a_s, b_s = text.split("-", 1)
    if not (a_s.isdigit() and b_s.isdigit()):
        return None
    a0, b_incl = int(a_s) - 1, int(b_s)
    if a0 < 0 or b_incl <= a0:
        return None
    return a0, b_incl


def _oriented_genomic_span(seq: str, strand: str, span: tuple[int, int]) -> str:
    a, b = span
    if a < 0 or b > len(seq) or a >= b:
        return ""
    frag = seq[a:b]
    if strand == "reverse":
        return reverse_complement_seq(frag)
    return frag


def extract_tss_box35(*, max_windows: int = 800) -> list[str]:
    """Analysis §20b RegulonDB C+S -35 boxes on the E. coli replicon."""
    seq_path = GENOMICS_DIR / "ecoli_k12_full.fna.gz"
    promo_path = GENOMICS_DIR / "regulondb_promoter_set.txt"
    if not seq_path.exists() or not promo_path.exists():
        return []
    seq = read_fasta_seq(seq_path, keep_n=False)
    if len(seq) < 200:
        return []
    out: list[str] = []
    for line in promo_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#") or line.startswith("1)id"):
            continue
        parts = line.split("\t")
        if len(parts) < 15:
            continue
        strand = parts[2].strip().lower()
        pos = parts[3].strip()
        conf = parts[14].strip().upper()
        if conf not in _TSS_CONF_KEEP:
            continue
        if not pos.isdigit() or strand not in ("forward", "reverse"):
            continue
        tss0 = int(pos) - 1
        if tss0 < 80 or tss0 >= len(seq) - 80:
            continue
        span = _parse_rdb_span(parts[10])
        if span is None:
            continue
        w = _oriented_genomic_span(seq, strand, span).upper()
        if len(w) < 4 or not all(c in "ACGT" for c in w):
            continue
        out.append(w)
        if len(out) >= max_windows:
            break
    return out



def null_gene_streams(
    genes: Sequence[GeneRecord],
    *,
    kind: str,
    seed: int = 42,
) -> list[list[int]]:
    """Build GC-shuffle or protein-fixed null byte streams for the same genes."""
    rng = np.random.default_rng(seed)
    out: list[list[int]] = []
    for g in genes:
        if kind == "shuffle":
            null_seq = gc_matched_shuffle(g.seq, rng=rng)
        elif kind == "recode":
            null_seq = protein_fixed_resample(g.seq, rng=rng)
        else:
            raise ValueError(kind)
        out.append(byte_stream(null_seq))
    return out
