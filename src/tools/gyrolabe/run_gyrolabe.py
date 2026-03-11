# src/tools/gyrolabe/run_gyrolabe.py
"""
Minimal runner for GyroLabe Bolmo bridge.

Uses local data/models/Bolmo-1B (offline, no HF download).
Run from repo root: python -m src.tools.gyrolabe.run_gyrolabe
Or by path: python src/tools/gyrolabe/run_gyrolabe.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.tools.gyrolabe import ops as gyro_ops
from src.tools.gyrolabe.bolmo_bridge import (
    DEFAULT_BOLMO_MODEL_PATH,
    GyrolabeBolmoBridge,
    GyrolabeSettings,
    load_gyrolabe_bolmo,
)


def _get_tokenizer(bridge: GyrolabeBolmoBridge) -> Any | None:
    try:
        tc = getattr(bridge.base_model.model, "tokenizer_config", None)
        if tc is not None and hasattr(tc, "build"):
            return tc.build()
    except Exception:
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GyroLabe Bolmo bridge (local data/models/Bolmo-1B)."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_BOLMO_MODEL_PATH,
        help=f"Path to Bolmo model dir (default: {DEFAULT_BOLMO_MODEL_PATH})",
    )
    parser.add_argument(
        "--no-embedding-bias",
        action="store_true",
        help="Disable embedding bias.",
    )
    parser.add_argument(
        "--no-boundary-bias",
        action="store_true",
        help="Disable boundary bias.",
    )
    parser.add_argument(
        "--no-strict-cpu",
        action="store_true",
        help="Allow non-CPU input (e.g. CUDA).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load and run one forward pass; no generate.",
    )
    parser.add_argument(
        "--bfloat16",
        action="store_true",
        help="Load model in bfloat16 (2x less memory). May fail if mLSTM lacks bfloat16 support.",
    )
    args = parser.parse_args()

    hf_kwargs = {}
    if args.bfloat16:
        hf_kwargs["torch_dtype"] = torch.bfloat16

    settings = GyrolabeSettings(
        enable_embedding_bias=not args.no_embedding_bias,
        enable_boundary_bias=not args.no_boundary_bias,
        enable_qclass_sparsity=False,
        strict_cpu=not args.no_strict_cpu,
    )

    print("Loading GyroLabe Bolmo bridge (local, offline)...")
    model = load_gyrolabe_bolmo(
        args.model_path,
        settings=settings,
        **hf_kwargs,
    )
    model.eval()
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Loaded. Parameters: {nparams:,}")
    c_lib = gyro_ops._get_lib()
    print(f"GyroLabe C library: {'loaded' if c_lib else 'fallback (Python)'}")

    # Sanity forward pass (small dummy sequence)
    with torch.no_grad():
        dummy_ids = torch.zeros(1, 8, dtype=torch.long)
        for i in range(8):
            dummy_ids[0, i] = 4 + (i % 256)
        t0 = time.perf_counter()
        out = model(dummy_ids)
        t1 = time.perf_counter()
        logits = out.logits if hasattr(out, "logits") else out[0]
        print(f"Forward OK. logits shape: {logits.shape} ({1000*(t1-t0):.1f} ms)")

    if args.dry_run:
        print("Dry run done.")
        return

    tokenizer = _get_tokenizer(model)
    if tokenizer is None:
        print("No tokenizer available (model.tokenizer_config.build() failed). Skip generate.")
        print("Run without --dry-run and with dolma2 tokenizer for generate.")
        return

    with torch.no_grad():
        prompt = "The"
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        if model.settings.strict_cpu:
            input_ids = input_ids.cpu()
        max_new = 12
        t0 = time.perf_counter()
        gen = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "pad_token_id", 0),
        )
        t1 = time.perf_counter()
    new_tokens = gen.shape[1] - input_ids.shape[1]
    tks = new_tokens / (t1 - t0) if (t1 - t0) > 0 else 0
    text = tokenizer.decode(gen[0].tolist(), skip_special_tokens=True)
    print(f"Generate OK. Sample: {text[:120]}...")
    print(f"Stats: {new_tokens} new tokens in {1000*(t1-t0):.1f} ms ({tks:.1f} tok/s)")
    print("Done.")


if __name__ == "__main__":
    main()
