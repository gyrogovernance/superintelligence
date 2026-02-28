"""
Inspect router_operator npz artifacts: shapes, stats, roles.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _stats(arr: np.ndarray, name: str) -> None:
    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.dtype == np.uint8 and arr.size < 32:
        try:
            s = arr.tobytes().decode("utf-8")
            print(f"    -> {repr(s)}")
        except Exception:
            print(f"    min={float(arr.min())} max={float(arr.max())}")
    else:
        print(f"    min={float(arr.min()):.4f} max={float(arr.max()):.4f} mean={float(arr.mean()):.4f} std={float(arr.std()):.4f}")


def inspect_npz(path: Path) -> None:
    data = np.load(path, allow_pickle=False)
    print(f"\n--- {path.name} ---")
    for key in sorted(data.files):
        arr = data[key]
        if arr.ndim == 0:
            print(f"  {key}: scalar = {arr.item()}")
        else:
            _stats(arr, key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect router_operator npz files")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("data/router_operator"),
        help="Directory containing npz files",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_absolute():
        repo = Path(__file__).resolve().parents[1]
        root = repo / root

    files = [
        "fast_operator.npz",
        "slow_operator.npz",
        "boundary_head.npz",
        "candidate_scorer.npz",
    ]

    found = 0
    for f in files:
        p = root / f
        if p.exists():
            inspect_npz(p)
            found += 1

    if found == 0:
        print(f"No npz files found in {root}")
        sys.exit(1)

    print("\n--- Summary ---")
    for f in files:
        p = root / f
        if p.exists():
            sz = p.stat().st_size
            if sz >= 1024 * 1024:
                print(f"  {f}: {sz / (1024*1024):.2f} MB")
            else:
                print(f"  {f}: {sz / 1024:.2f} KB")


if __name__ == "__main__":
    main()
