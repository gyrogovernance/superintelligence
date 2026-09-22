"""Run the genomics synthesis census suite.

    python -m src.tools.autoencoder.programs.genomics.synthesis.run
    python -m src.tools.autoencoder.programs.genomics.synthesis.run --only g4
    python -m src.tools.autoencoder.programs.genomics.synthesis.run --hosts ecoli,yeast

Requires production AE checkpoints and the genomics catalog:

  python -m src.tools.autoencoder.programs.genomics.ingest_genomics
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_repo_root = _THIS_DIR
for _ in range(6):
    _repo_root = _repo_root.parent
    if (_repo_root / "src").is_dir():
        break
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from src.tools.autoencoder import paths

_RESULTS = _THIS_DIR / "RESULTS.txt"

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




def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genomics synthesis census suite")
    parser.add_argument("--only", choices=_ONLY, default=None)
    parser.add_argument(
        "--hosts",
        default="ecoli,yeast,sars",
        help="comma-separated catalog hosts: ecoli, yeast, sars (chr22 splice runs with suite)",
    )
    parser.add_argument("--n-pairs", type=int, default=400)
    parser.add_argument("--max-genes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--results-file", default=None)
    args = parser.parse_args(argv)
    paths.ensure()

    from src.tools.autoencoder.programs.genomics.synthesis.censuses import run_suite

    hosts = tuple(h.strip() for h in str(args.hosts).split(",") if h.strip())
    results_path = Path(args.results_file) if args.results_file else None
    try:
        payload = run_suite(
            only=args.only,
            hosts=hosts,
            n_pairs=int(args.n_pairs),
            max_genes=int(args.max_genes),
            seed=int(args.seed),
            device=args.device,
            results_file=results_path,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if payload.get("error"):
        return 1
    if payload.get("status") == "blocked":
        print("suite blocked on preflight; artifacts unchanged", file=sys.stderr)
        return 1
    out = results_path or _RESULTS
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
