"""CLI for genomics synthesis: climate census and census suite."""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from src.tools.autoencoder import paths
from src.tools.autoencoder.programs.genomics.genomics import GENOMICS_DIR
from src.tools.autoencoder.programs.genomics.synthesis.climate import (
    measure_reference_climate,
)


def _read_cds_named(
    path: Path, *, min_bases: int = 300, max_records: int = 220
) -> list[tuple[str, str]]:
    """Read gzipped FASTA CDS catalog into (id, sequence) pairs."""
    out: list[tuple[str, str]] = []
    limit = int(max_records)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        header = ""
        buf: list[str] = []

        def flush() -> None:
            nonlocal header, buf
            if not header or not buf:
                header, buf = "", []
                return
            seq = "".join(buf).upper().replace("U", "T")
            seq = "".join(c for c in seq if c in "ACGT")
            if len(seq) >= min_bases:
                gene_id = header.split()[0].lstrip(">")
                out.append((gene_id, seq))
            header, buf = "", []

        for line in fh:
            if line.startswith(">"):
                flush()
                if len(out) >= limit:
                    break
                header = line.strip()
                continue
            if len(out) >= limit:
                break
            buf.append(line.strip())
        if buf and len(out) < limit:
            flush()
    return out


def cmd_climate_census(args: argparse.Namespace) -> int:
    """Measure a host's reference climate over its catalog coding sequences."""
    filename, table = {
        "ecoli": ("ecoli_k12_cds.fna.gz", 11),
        "yeast": ("yeast_s288c_cds.fna.gz", 1),
    }[args.host]
    path = GENOMICS_DIR / filename
    if not path.exists():
        print(f"missing catalog {path}")
        return 1
    genes = _read_cds_named(path, min_bases=300, max_records=int(args.records))
    genes = [(gid, seq) for gid, seq in genes if len(seq) % 3 == 0]
    if not genes:
        print("no usable coding sequence found")
        return 1
    reference = measure_reference_climate(
        (seq for _gid, seq in genes),
        label=args.host,
        ncbi_id=table,
        provenance=f"mean over {len(genes)} catalog CDS records from {filename}",
    )
    rows = [
        {
            "observable": name,
            "mean": reference.mean[name],
            "stderr": reference.stderr.get(name, float("nan")),
        }
        for name in sorted(reference.mean)
    ]
    out = Path(args.output_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(
        f"{args.host}: {reference.n_sequences} sequences, "
        f"mean_shell={reference.mean['mean_shell']:.4f} "
        f"+- {reference.stderr['mean_shell']:.4f}"
    )
    print(f"wrote {out}")
    return 0


def cmd_suite(args: argparse.Namespace) -> int:
    from src.tools.autoencoder.programs.genomics.synthesis.censuses import run_suite

    hosts = tuple(h.strip() for h in str(args.hosts).split(",") if h.strip())
    try:
        payload = run_suite(
            only=args.only,
            hosts=hosts,
            n_pairs=int(args.n_pairs),
            max_genes=int(args.max_genes),
            seed=int(args.seed),
            device=args.device,
            results_file=Path(args.results_file) if args.results_file else None,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if payload.get("error"):
        return 1
    if payload.get("status") == "blocked":
        print("suite blocked on preflight; artifacts unchanged", file=sys.stderr)
        return 1
    return 0


_ONLY = (
    "align",
    "g4",
    "climate",
    "prov",
    "shuffle",
    "recode",
    "radial",
    "splice",
)


def main(argv: Optional[list[str]] = None) -> int:
    paths.ensure()
    p = argparse.ArgumentParser(
        prog="python -m src.tools.autoencoder.programs.genomics.synthesis.cli"
    )
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("climate-census")
    q.add_argument("--host", choices=("ecoli", "yeast"), required=True)
    q.add_argument("--records", type=int, default=220)
    q.add_argument("--output-tsv", required=True)

    q = sub.add_parser(
        "suite",
        help="genomics synthesis census suite",
    )
    q.add_argument("--only", choices=_ONLY, default=None)
    q.add_argument(
        "--hosts",
        default="ecoli,yeast,sars",
        help="comma-separated catalog hosts: ecoli,yeast,sars (chr22 splice is separate)",
    )
    q.add_argument("--n-pairs", type=int, default=400)
    q.add_argument("--max-genes", type=int, default=200)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--device", default="cpu")
    q.add_argument(
        "--results-file",
        default=None,
        help="RESULTS path (default: synthesis/RESULTS.txt)",
    )

    args = p.parse_args(argv)
    if args.command == "climate-census":
        return cmd_climate_census(args)
    if args.command == "suite":
        return cmd_suite(args)
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
