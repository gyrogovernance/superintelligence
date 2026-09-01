#!/usr/bin/env python3
"""Ingest the frozen genomics catalog into the autoencoder data tree.

The read path in ``genomics.py`` points at ``data/dataset_genomics/`` (the
``dataset_<word>`` convention). This script populates that folder by copying
the frozen catalog from the science repo
(``AE_SCIENCE_CATALOG`` env var or ``--science-catalog`` flag, defaulting to
``F:\\Development\\science\\data\\catalogs\\genomics``) and records
``SOURCE.txt`` + ``MANIFEST.sha256`` alongside the files. ``--skip-network``
only hashes whatever is already on disk (no download and no source-copy
needed for a reproducibility pass).

Run from the package root::

    python -m src.tools.autoencoder.helpers.ingest_genomics
    python -m src.tools.autoencoder.helpers.ingest_genomics --skip-network
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.tools.autoencoder import paths
from src.tools.autoencoder.helpers.genomics import (
    _CODONS,
    CODE_OVERRIDES,
    GENOMICS_DIR,
    STANDARD_CODE,
)

# The science repo is the only upstream; the AE tree holds a local copy.
# The default is a Windows-specific path; pass ``--science-catalog`` (or set
# ``AE_SCIENCE_CATALOG``) to point at a different checkout. ``None`` is also
# accepted: the script will then skip the science-copy branch and download
# directly from the public URLs.
DEFAULT_SCIENCE_CATALOG = Path(r"F:\Development\science\data\catalogs\genomics")

# Files the genomics read path actually uses, in copy order. UniProt and
# RegulonDB are handled by their own dedicated steps below, so they are NOT
# in this set (avoids a redundant double-copy/duplicate manifest row).
CERTIFIED_FILES: Tuple[str, ...] = tuple(
    sorted(
        {
            "ecoli_k12_cds.fna.gz",
            "ecoli_k12_full.fna.gz",
            "yeast_s288c_cds.fna.gz",
            "sars_cov2.fna.gz",
            "sars_cov2_cds.fna.gz",
            "chr22.fa.gz",
            "gencode.v47.annotation.gtf.gz",
            "rebase_withrefm.txt",
        }
    )
)

# Download URLs, used only when the science catalog is absent.
URLS: Dict[str, str] = {
    "ecoli_k12_cds.fna.gz": (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/"
        "GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_cds_from_genomic.fna.gz"
    ),
    "ecoli_k12_full.fna.gz": (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/"
        "GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz"
    ),
    "yeast_s288c_cds.fna.gz": (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/146/045/"
        "GCF_000146045.2_R64/GCF_000146045.2_R64_cds_from_genomic.fna.gz"
    ),
    "sars_cov2.fna.gz": (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/009/858/895/"
        "GCF_009858895.2_ASM985889v3/GCF_009858895.2_ASM985889v3_genomic.fna.gz"
    ),
    "sars_cov2_cds.fna.gz": (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/009/858/895/"
        "GCF_009858895.2_ASM985889v3/GCF_009858895.2_ASM985889v3_cds_from_genomic.fna.gz"
    ),
    "chr22.fa.gz": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz",
    "gencode.v47.annotation.gtf.gz": (
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/"
        "gencode.v47.annotation.gtf.gz"
    ),
    "rebase_withrefm.txt": "https://ftp.neb.com/pub/rebase/withrefm.txt",
}

UNIPROT_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?query=(organism_id:83333)+AND+reviewed:true&format=txt"
)
UNIPROT_NAME = "ecoli_k12_uniprot.txt"

REGULONDB_URL = "https://regulondb.ccg.unam.mx/graphql"
REGULONDB_NAME = "regulondb_promoter_set.txt"

USER_AGENT = "Mozilla/5.0 (CGM-hQVM-ae-genomics-ingest)"
TIMEOUT_S = 300
REGULONDB_TIMEOUT_S = 600


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_ncbi_tables() -> Path:
    """Freeze the genetic-code translation tables into the AE catalog.

    A schema-versioned copy keys on the NCBI transl_table id; ``genomics.py``
    loads it when present and otherwise falls back to the built-in table, so
    the file is optional but always written by ingest for certification."""
    tables: Dict[str, dict] = {}
    for tid in sorted(CODE_OVERRIDES):
        code = dict(STANDARD_CODE)
        code.update(CODE_OVERRIDES[tid])
        tables[str(tid)] = {
            "id": tid,
            "name": str(tid),
            "aa": "".join(code[c] for c in _CODONS),
        }
    payload = {
        "source": "NCBI transl_table overrides frozen in helpers/genomics.py",
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "codon_order": "itertools.product(ACGT, repeat=3)",
        "tables": tables,
    }
    dest = GENOMICS_DIR / "ncbi_genetic_codes.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def _download(url: str, dest: Path, force: bool = False) -> Tuple[str, Optional[str]]:
    import ssl
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    if dest.exists() and dest.stat().st_size > 0 and not force:
        return "skip", sha256_file(dest)
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=TIMEOUT_S, context=ssl.create_default_context()) as resp:
            data = resp.read()
    except (HTTPError, URLError, OSError, TimeoutError) as exc:
        return f"FAIL {exc}", None
    if not data:
        return "FAIL empty body", None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return "wrote", _sha256_bytes(data)


def _download_uniprot(force: bool = False) -> Tuple[str, Optional[str]]:
    return _download(UNIPROT_URL, GENOMICS_DIR / UNIPROT_NAME, force=force)


def _download_regulondb(force: bool = False) -> Tuple[str, Optional[str]]:
    dest = GENOMICS_DIR / REGULONDB_NAME
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return "skip", sha256_file(dest)
    body = json.dumps(
        {"query": '{ getDataOfFile(fileName: "PromoterSet") { content } }'}
    ).encode("utf-8")
    import ssl
    from urllib.request import Request, urlopen

    req = Request(
        REGULONDB_URL,
        data=body,
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=REGULONDB_TIMEOUT_S, context=ssl.create_default_context()) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
        content = (payload.get("data") or {}).get("getDataOfFile") or {}
        text = content.get("content")
        if not text:
            return "FAIL empty PromoterSet", None
        dest.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        return "wrote", sha256_file(dest)
    except Exception as exc:  # network/parse: fall through to manifest SKIP
        return f"FAIL {exc}", None


def _copy_from_science(name: str, force: bool, catalog: Path) -> Tuple[str, Optional[str]]:
    src = catalog / name
    dest = GENOMICS_DIR / name
    if not src.exists() or src.stat().st_size == 0:
        return "no-source", None
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return "skip", sha256_file(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return "copied", sha256_file(dest)


def _record_present(name: str, entries: List[Tuple[str, str, Optional[str]]], notes: List[str]) -> None:
    dest = GENOMICS_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        digest = sha256_file(dest)
        entries.append((name, "present", digest))
        notes.append(f"{name} present sha256={digest}")
        print(f"  present {name}")
    else:
        entries.append((name, "SKIP", None))
        notes.append(f"{name} SKIP")
        print(f"  SKIP {name}")


def write_source_txt(rows: List[str]) -> Path:
    dest = GENOMICS_DIR / "SOURCE.txt"
    dest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return dest


def write_manifest(entries: List[Tuple[str, str, Optional[str]]]) -> Path:
    dest = GENOMICS_DIR / "MANIFEST.sha256"
    lines = ["# file\tstatus\tsha256"]
    for name, status, digest in entries:
        lines.append(f"{name}\t{status}\t{digest or '-'}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Populate data/dataset_genomics/ with the frozen catalog."
    )
    parser.add_argument("--force", action="store_true", help="re-copy / re-download if present")
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="do not download; hash whatever is already on disk",
    )
    parser.add_argument(
        "--science-catalog",
        default=os.environ.get("AE_SCIENCE_CATALOG", str(DEFAULT_SCIENCE_CATALOG)),
        help="local path to the science repo's data/catalogs/genomics "
        "(env: AE_SCIENCE_CATALOG). Pass an empty string to skip the "
        "science-copy branch and download from public URLs instead.",
    )
    args = parser.parse_args()

    catalog: Path | None = (
        Path(args.science_catalog) if args.science_catalog else None
    )

    GENOMICS_DIR.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    source_rows = [
        f"ingest_utc={date}",
        "scope=only catalogs loaded by helpers/genomics.py (P4 read path)",
        (
            "upstream="
            f"science repo {catalog.as_posix() if catalog else 'disabled'} "
            "(or direct download when absent)"
        ),
    ]
    entries: List[Tuple[str, str, Optional[str]]] = []

    ncbi = write_ncbi_tables()
    entries.append((ncbi.name, "wrote", sha256_file(ncbi)))
    source_rows.append(f"{ncbi.name} wrote sha256={sha256_file(ncbi)} consumer=genomics.translation_table")
    print(f"  wrote {ncbi.name}")

    has_science = (
        catalog is not None
        and catalog.exists()
        and not args.skip_network
    )
    if args.skip_network:
        print("  skip-network: hashing present files only")
        for name in CERTIFIED_FILES:
            _record_present(name, entries, source_rows)
        _record_present(UNIPROT_NAME, entries, source_rows)
        _record_present(REGULONDB_NAME, entries, source_rows)
    elif has_science:
        assert catalog is not None  # for type-checkers
        catalog_name = catalog.name
        print(f"  source={catalog}")
        for name in CERTIFIED_FILES:
            status, digest = _copy_from_science(name, force=args.force, catalog=catalog)
            print(f"  {status} {name}")
            entries.append((name, status, digest))
            source_rows.append(
                f"{name} {status} sha256={digest} from={catalog_name}"
            )
        status, digest = _copy_from_science(UNIPROT_NAME, force=args.force, catalog=catalog)
        print(f"  {status} {UNIPROT_NAME}")
        entries.append((UNIPROT_NAME, status, digest))
        source_rows.append(
            f"{UNIPROT_NAME} {status} sha256={digest} from={catalog_name}"
        )
        status, digest = _copy_from_science(REGULONDB_NAME, force=args.force, catalog=catalog)
        print(f"  {status} {REGULONDB_NAME}")
        entries.append((REGULONDB_NAME, status, digest))
        source_rows.append(
            f"{REGULONDB_NAME} {status} sha256={digest} from={catalog_name}"
        )
    else:
        print("  no science catalog; downloading from source URLs")
        for name, url in URLS.items():
            dest = GENOMICS_DIR / name
            print(f"  GET {name}")
            status, digest = _download(url, dest, force=args.force)
            print(f"  {status} {name}")
            entries.append((name, status, digest))
            source_rows.append(
                f"{name} {status} sha256={digest} url={url}"
            )
        print(f"  GET {UNIPROT_NAME}")
        status, digest = _download_uniprot(force=args.force)
        entries.append((UNIPROT_NAME, status, digest))
        source_rows.append(
            f"{UNIPROT_NAME} {status} sha256={digest} url={UNIPROT_URL}"
        )
        print(f"  GET {REGULONDB_NAME}")
        status, digest = _download_regulondb(force=args.force)
        entries.append((REGULONDB_NAME, status, digest))
        source_rows.append(
            f"{REGULONDB_NAME} {status} sha256={digest} url={REGULONDB_URL}"
        )

    write_source_txt(source_rows)
    man = write_manifest(entries)
    print(f"  wrote {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())