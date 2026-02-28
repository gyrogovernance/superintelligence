"""
CLI job: Convert Bolmo safetensors into resonator format.

EXPERIMENTAL: Not part of the main inference stack. Uses adaptors.backup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tools.adaptors.backup.convert import convert_bolmo_safetensors

_PROFILE_MIN = frozenset({
    "lm_head.weight",
    "model.local_encoder.byte_embedding.weight",
    "model.local_encoder.boundary_predictor_module.q_proj_layer.weight",
    "model.local_encoder.boundary_predictor_module.k_proj_layer.weight",
})

_PROFILE_MID = _PROFILE_MIN | frozenset({
    f"model.layers.{i}.self_attn.{p}.weight"
    for i in range(16)
    for p in ("q_proj", "k_proj", "v_proj", "o_proj")
})


def _resolve_profile(profile: str | None) -> tuple[frozenset[str] | None, tuple[str, ...] | None]:
    if profile == "min":
        return _PROFILE_MIN, None
    if profile == "mid":
        return _PROFILE_MID, None
    if profile == "full":
        return None, ("model.", "lm_head.")
    return None, None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Bolmo safetensors into resonator format")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/models/Bolmo-1B"),
        help="Path to source Bolmo model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/resonators/Bolmo-1B-wht"),
        help="Path to output resonator directory",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Output tensor dtype",
    )
    parser.add_argument(
        "--verify-roundtrip",
        action="store_true",
        help="Enable strict roundtrip verification per transformed tensor",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        dest="prefixes",
        default=["model.", "lm_head."],
        help="Tensor key prefix to include; can be repeated",
    )
    parser.add_argument(
        "--name",
        action="append",
        dest="names",
        default=[],
        help="Exact tensor key to include; can be repeated. Overrides prefixes when provided",
    )
    parser.add_argument(
        "--convert-1d",
        action="store_true",
        help="Enable WHT on power-of-two 1D tensors",
    )
    parser.add_argument(
        "--max-tensors-per-shard",
        type=int,
        default=None,
        help="Optional limit for dry runs",
    )
    parser.add_argument(
        "--profile",
        choices=["min", "mid", "full"],
        default=None,
        help="Shortcut: min=4 tensors, mid=min+attention, full=all",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.profile:
        prof_names, prof_prefixes = _resolve_profile(args.profile)
        include_names = set(prof_names) if prof_names else None
        include_prefixes = prof_prefixes
    elif args.names:
        include_names, include_prefixes = set(args.names), None
    else:
        include_names, include_prefixes = None, tuple(args.prefixes) if args.prefixes else None

    out_index, out_manifest = convert_bolmo_safetensors(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        include_prefixes=include_prefixes,
        include_names=include_names,
        output_dtype=args.dtype,
        verify_roundtrip=bool(args.verify_roundtrip),
        convert_1d=bool(args.convert_1d),
        max_tensors_per_shard=args.max_tensors_per_shard,
    )

    print(f"Source: {args.model_dir.resolve()}")
    print(f"Output: {args.output_dir.resolve()}")
    print(f"Index:  {out_index.resolve()}")
    print(f"Manifest: {out_manifest.resolve()}")


if __name__ == "__main__":
    main()
