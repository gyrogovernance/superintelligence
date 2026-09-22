#!/usr/bin/env python3
"""Download CGM Genomics Analysis catalogs into dataset_genomics/.

Catalogs: E. coli CDS and full chromosome, yeast CDS, SARS genome and CDS,
human chr22 and GENCODE, REBASE, RegulonDB promoters, NCBI translation tables.

    python -m src.tools.autoencoder.programs.genomics.ingest_genomics
    python -m src.tools.autoencoder.programs.genomics.ingest_genomics --skip-network
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.tools.autoencoder.programs.genomics.genomics import (
    _CODONS,
    CODE_OVERRIDES,
    GENOMICS_DIR,
    STANDARD_CODE,
)

CERTIFIED_FILES: Tuple[str, ...] = (
    "ecoli_k12_cds.fna.gz",
    "ecoli_k12_full.fna.gz",
    "yeast_s288c_cds.fna.gz",
    "sars_cov2.fna.gz",
    "sars_cov2_cds.fna.gz",
    "chr22.fa.gz",
    "gencode.v47.annotation.gtf.gz",
    "rebase_withrefm.txt",
    "regulondb_promoter_set.txt",
    "ncbi_genetic_codes.json",
)

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

REGULONDB_URL = "https://regulondb.ccg.unam.mx/graphql"
REGULONDB_NAME = "regulondb_promoter_set.txt"

USER_AGENT = "Mozilla/5.0 (hQVM-ae-genomics-ingest)"
TIMEOUT_S = 300
REGULONDB_TIMEOUT_S = 600

REMOVED_FROM_ANALYSIS_INGEST: Tuple[str, ...] = (
    "archiveii.512.parquet",
    "ecoli_k12_uniprot.txt",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_ncbi_tables() -> Path:
    """Freeze the genetic-code translation tables into the local catalog."""
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
        "source": "NCBI transl_table overrides frozen in programs/genomics/genomics.py",
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
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=TIMEOUT_S, context=ssl.create_default_context()) as resp, dest.open(
            "wb"
        ) as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    except (HTTPError, URLError, OSError, TimeoutError) as exc:
        if dest.exists():
            dest.unlink(missing_ok=True)
        return f"FAIL {exc}", None
    if not dest.exists() or dest.stat().st_size == 0:
        return "FAIL empty body", None
    return "wrote", sha256_file(dest)


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
    except Exception as exc:
        return f"FAIL {exc}", None


def _record_present(
    name: str, entries: List[Tuple[str, str, Optional[str]]], notes: List[str]
) -> None:
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


def purge_non_analysis_artifacts() -> List[str]:
    """Delete files outside CERTIFIED_FILES / REGULONDB / NCBI tables."""
    removed: List[str] = []
    for name in REMOVED_FROM_ANALYSIS_INGEST:
        path = GENOMICS_DIR / name
        if path.exists():
            path.unlink()
            removed.append(name)
            print(f"  deleted {name}")
    from src.tools.autoencoder import paths

    riboseq_dir = paths.dataset_dir("riboseq")
    if riboseq_dir.exists():
        import shutil

        shutil.rmtree(riboseq_dir)
        removed.append(str(riboseq_dir))
        print(f"  deleted {riboseq_dir}")
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download CGM Genomics Analysis catalogs into "
            "src/tools/autoencoder/data/dataset_genomics/."
        )
    )
    parser.add_argument("--force", action="store_true", help="re-download even if the file is already present")
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="do not download; hash whatever is already on disk",
    )
    parser.add_argument(
        "--keep-orphans",
        action="store_true",
        help="do not delete ArchiveII / UniProt / dataset_riboseq leftovers",
    )
    args = parser.parse_args()

    GENOMICS_DIR.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    source_rows = [
        f"ingest_utc={date}",
        "scope=CGM Genomics Analysis local catalogs",
        f"dest={GENOMICS_DIR.as_posix()}",
        "upstream=public URLs (NCBI, UCSC, GENCODE, NEB REBASE, RegulonDB)",
        "excluded_from_ingest=ArchiveII UniProt Ribo-Seq",
    ]
    entries: List[Tuple[str, str, Optional[str]]] = []

    if not args.keep_orphans:
        print("  purge non-Analysis leftovers")
        purge_non_analysis_artifacts()

    ncbi = write_ncbi_tables()
    entries.append((ncbi.name, "wrote", sha256_file(ncbi)))
    source_rows.append(f"{ncbi.name} wrote sha256={sha256_file(ncbi)}")
    print(f"  wrote {ncbi.name}")

    if args.skip_network:
        print("  skip-network: hashing present Analysis catalogs only")
        for name in CERTIFIED_FILES:
            if name == "ncbi_genetic_codes.json":
                continue
            _record_present(name, entries, source_rows)
    else:
        print(f"  dest={GENOMICS_DIR}")
        for name, url in URLS.items():
            dest = GENOMICS_DIR / name
            print(f"  GET {name}")
            status, digest = _download(url, dest, force=args.force)
            print(f"  {status} {name}")
            entries.append((name, status, digest))
            source_rows.append(f"{name} {status} sha256={digest} url={url}")
        print(f"  GET {REGULONDB_NAME}")
        status, digest = _download_regulondb(force=args.force)
        entries.append((REGULONDB_NAME, status, digest))
        source_rows.append(f"{REGULONDB_NAME} {status} sha256={digest} url={REGULONDB_URL}")

    write_source_txt(source_rows)
    man = write_manifest(entries)
    print(f"  wrote {man}")

    failed = [name for name, status, _ in entries if str(status).startswith("FAIL")]
    if failed:
        print(f"  failed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
