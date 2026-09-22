#!/usr/bin/env python3
"""Prepare topology inputs under dataset_topology/.

Fetches the E. coli K-12 UniProt proteome TSV used by the topology atlas,
and verifies the Gamble 2016 Table S1 workbook that must be placed by hand
(journal supplement; not a public FTP pull).

    python -m src.tools.autoencoder.programs.genomics.ingest_topology
    python -m src.tools.autoencoder.programs.genomics.ingest_topology --skip-network
    python -m src.tools.autoencoder.programs.genomics.ingest_topology --check
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlencode

from src.tools.autoencoder import paths

TOPOLOGY_DIR = paths.dataset_dir("topology")

UNIPROT_NAME = "uniprot_ecoli_k12.tsv"
GAMBLE_NAME = "gamble_2016__NIHMS800838-supplement-6.xlsx"

UNIPROT_PROTEOME = "UP000000625"
UNIPROT_RELEASE_PIN = "2026_03"
UNIPROT_FIELDS = (
    "accession,id,gene_primary,gene_oln,length,"
    "cc_subcellular_location,ft_transmem,ft_topo_dom"
)
UNIPROT_URL = (
    "https://rest.uniprot.org/uniprotkb/stream?"
    + urlencode(
        {
            "format": "tsv",
            "query": f"(proteome:{UNIPROT_PROTEOME})",
            "fields": UNIPROT_FIELDS,
        }
    )
)

# Digests for the locked topology cohort. A forced UniProt refresh can change
# the TSV hash and requires rebuilding atlas → order.
EXPECTED_SHA256 = {
    UNIPROT_NAME: "c72205bac6e39188dad8f7e531961ec47ea663b5e40532565009c74c0a5c5dff",
    GAMBLE_NAME: "0afe42a917a6e6fb8d32d726fb5ba6282b7fcac1987af14bfc7a0a94a096ff2a",
}

GAMBLE_SOURCE = (
    "Gamble et al. 2016 Cell, NIHMS800838 Table S1 "
    "(PMC4967012 supplement NIHMS800838-supplement-6.xlsx). "
    "Place the workbook at dataset_topology/" + GAMBLE_NAME
)

USER_AGENT = "Mozilla/5.0 (hQVM-ae-topology-ingest)"
TIMEOUT_S = 600


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path, *, force: bool) -> Tuple[str, Optional[str]]:
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


def _record_file(name: str) -> Tuple[str, str, Optional[str]]:
    path = TOPOLOGY_DIR / name
    if path.is_file() and path.stat().st_size > 0:
        return name, "present", sha256_file(path)
    return name, "MISSING", None


def write_manifest(entries: list[Tuple[str, str, Optional[str]]]) -> Path:
    dest = TOPOLOGY_DIR / "MANIFEST.sha256"
    lines = [
        f"{digest}  {name}"
        for name, status, digest in entries
        if status in {"present", "skip", "wrote"} and digest
    ]
    dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return dest


def write_source(entries: list[Tuple[str, str, Optional[str]]]) -> Path:
    dest = TOPOLOGY_DIR / "SOURCE.txt"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [
        "dataset_topology",
        f"ingest_utc={stamp}",
        f"dest={TOPOLOGY_DIR.as_posix()}",
        "",
        f"{UNIPROT_NAME}",
        f"  proteome={UNIPROT_PROTEOME}",
        f"  release_pin={UNIPROT_RELEASE_PIN}",
        f"  fields={UNIPROT_FIELDS}",
        f"  url={UNIPROT_URL}",
        "",
        f"{GAMBLE_NAME}",
        f"  note={GAMBLE_SOURCE}",
        "",
        "retained digests:",
    ]
    for name, status, digest in entries:
        if digest:
            rows.append(f"{name}  status={status} sha256={digest}")
        else:
            rows.append(f"{name}  status={status}")
    dest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return dest


def check_expected(entries: list[Tuple[str, str, Optional[str]]]) -> list[str]:
    problems: list[str] = []
    by_name = {name: (status, digest) for name, status, digest in entries}
    for name, expected in EXPECTED_SHA256.items():
        status, digest = by_name.get(name, ("MISSING", None))
        if digest is None:
            problems.append(f"{name}: missing")
        elif digest != expected:
            problems.append(
                f"{name}: sha256 mismatch (have {digest}, expected {expected})"
            )
        elif status == "MISSING":
            problems.append(f"{name}: missing")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download UniProt even if present")
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="do not download; only hash files already on disk",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="require the locked cohort digests in EXPECTED_SHA256",
    )
    args = parser.parse_args(argv)

    TOPOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[Tuple[str, str, Optional[str]]] = []

    if args.skip_network:
        print("  skip-network: hashing present topology inputs only")
        name, status, digest = _record_file(UNIPROT_NAME)
        print(f"  {status} {name}")
        entries.append((name, status, digest))
    else:
        dest = TOPOLOGY_DIR / UNIPROT_NAME
        print(f"  GET {UNIPROT_NAME}")
        status, digest = _download(UNIPROT_URL, dest, force=bool(args.force))
        print(f"  {status} {UNIPROT_NAME}")
        entries.append((UNIPROT_NAME, status, digest))

    name, status, digest = _record_file(GAMBLE_NAME)
    print(f"  {status} {name}")
    if status == "MISSING":
        print(f"  place {GAMBLE_NAME} under {TOPOLOGY_DIR}", file=sys.stderr)
        print(f"  {GAMBLE_SOURCE}", file=sys.stderr)
    entries.append((name, status, digest))

    man = write_manifest(entries)
    src = write_source(entries)
    print(f"  wrote {man.name}")
    print(f"  wrote {src.name}")

    missing_or_failed = [
        f"{name}: {status}"
        for name, status, _ in entries
        if status == "MISSING" or str(status).startswith("FAIL")
    ]
    if missing_or_failed:
        for problem in missing_or_failed:
            print(f"  fail: {problem}", file=sys.stderr)
        return 1
    if args.check:
        problems = check_expected(entries)
        if problems:
            for problem in problems:
                print(f"  check fail: {problem}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
