"""
CLI runner for Router inference using converted resonator.

Requires:
  - L3 table (data/layers/l3_packed_u24.bin or --l3-path)
  - Resonator: --profile min for --lm-only, --profile full for full chain
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tools.layers import create_default_four_layers

from .router_inference import RouterInference, load_router_inference


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Router inference with WHT-converted Bolmo weights"
    )
    parser.add_argument(
        "--l3-path",
        type=Path,
        default=Path("data/layers/l3_packed_u24.bin"),
        help="Path to L3 packed table",
    )
    parser.add_argument(
        "--resonator-dir",
        type=Path,
        default=Path("data/resonators/Bolmo-1B-wht"),
        help="Path to converted resonator",
    )
    parser.add_argument(
        "--lm-only",
        action="store_true",
        help="Use only lm_head (works with profile min)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/models/Bolmo-1B/config.json"),
        help="Bolmo config path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello",
        help="ASCII prompt for generation",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=20,
        help="Max bytes to generate",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        default=True,
        help="Greedy decode (default)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample instead of greedy",
    )
    parser.add_argument(
        "--state-scale",
        type=float,
        default=1.0,
        help="Scale for Router-state encoder vector",
    )
    parser.add_argument(
        "--embed-scale",
        type=float,
        default=1.0,
        help="Scale for base-byte embedding residual",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    four = create_default_four_layers(l3_path=args.l3_path)

    inference = load_router_inference(
        resonator_dir=args.resonator_dir,
        four_layers=four,
        config_path=args.config,
        lm_only=args.lm_only,
        state_scale=args.state_scale,
        embed_scale=args.embed_scale,
    )

    prompt_bytes = args.prompt.encode("ascii", errors="replace")
    output = inference.generate(
        prompt_bytes,
        max_bytes=args.max_bytes,
        greedy=not args.sample,
    )

    print(f"Prompt: {args.prompt!r}")
    print(f"Output: {output!r}")
    try:
        print(f"Decoded: {output.decode('ascii', errors='replace')}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
