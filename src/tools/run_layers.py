"""
CLI runner for src.tools.layers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.tools.layers import create_default_four_layers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and verify 4-layer FSM stack")
    parser.add_argument(
        "--l3-path",
        type=Path,
        default=Path("data/layers/l3_packed_u24.bin"),
        help="Path to packed L3 table file",
    )
    parser.add_argument(
        "--build-l3-if-missing",
        action="store_true",
        help="Build packed L3 table if file is missing",
    )
    parser.add_argument(
        "--verify-l3",
        action="store_true",
        help="Run L3 sampled verification checks",
    )
    parser.add_argument(
        "--verify-samples",
        type=int,
        default=20000,
        help="Sample count for optional L3 verification",
    )
    parser.add_argument(
        "--demo-bytes",
        type=str,
        default="hello",
        help="ASCII payload for demo ingestion",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    four = create_default_four_layers(
        l3_path=args.l3_path,
        build_l3_if_missing=bool(args.build_l3_if_missing),
    )

    payload = args.demo_bytes.encode("ascii", errors="ignore")
    for b in payload:
        four.ingest_byte(b)

    if args.verify_l3:
        four.l3.verify_table_matches_function_samples(samples=int(args.verify_samples))
        four.l3.verify_inverse_samples(samples=int(args.verify_samples))

    regs = four.regs
    print(f"L3 path: {args.l3_path}")
    print(f"bytes_ingested: {len(payload)}")
    print(f"l1_state8: {regs.l1_state8}")
    print(f"l2_state16: {regs.l2_state16}")
    print(f"l3_state24: {regs.l3_state24}")
    print(f"l4_O: {regs.l4.O}")
    print(f"l4_E: {regs.l4.E}")
    print(f"l4_parity: {regs.l4.parity}")
    print(f"l4_length: {regs.l4.length}")
    print("ok")


if __name__ == "__main__":
    main()
