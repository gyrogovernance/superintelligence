# src/tools/gyrolabe/chat_gyrolabe.py
"""
Interactive chat with GyroLabe Bolmo bridge. All acceleration enabled.

Uses: C library (signature_scan, chirality, qmap, etc.), decode expand cache.
Model: local data/models/Bolmo-1B (offline).

Run from repo root: python -m src.tools.gyrolabe.chat_gyrolabe
Or: python src/tools/gyrolabe/chat_gyrolabe.py
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
    GyrolabeSettings,
    load_gyrolabe_bolmo,
)


def _get_tokenizer(model: Any) -> Any | None:
    try:
        tc = getattr(model.model, "tokenizer_config", None)
        if tc is not None and hasattr(tc, "build"):
            return tc.build()
    except Exception:
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive chat with GyroLabe Bolmo (all acceleration)."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_BOLMO_MODEL_PATH,
        help=f"Path to Bolmo model (default: {DEFAULT_BOLMO_MODEL_PATH})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens per turn (default: 128)",
    )
    parser.add_argument(
        "--bfloat16",
        action="store_true",
        help="(Unsupported) Bolmo mLSTM has dtype mismatch with bfloat16. Use float32.",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Low CPU memory loading (for 7B+). Uses low_cpu_mem_usage, device_map=cpu.",
    )
    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Model not found: {args.model_path}")
        print("Place Bolmo-1B at data/models/Bolmo-1B or pass --model-path")
        sys.exit(1)

    hf_kwargs: dict[str, Any] = {}
    if args.bfloat16:
        print("Warning: bfloat16 causes dtype errors with Bolmo mLSTM. Ignoring --bfloat16.")
    if args.low_memory:
        hf_kwargs["low_cpu_mem_usage"] = True
        hf_kwargs["device_map"] = "cpu"

    settings = GyrolabeSettings(
        enable_embedding_bias=True,
        enable_boundary_bias=True,
        enable_decode_expand_cache=True,
        enable_qclass_sparsity=False,
        strict_cpu=True,
    )

    print("Loading GyroLabe Bolmo (all acceleration)...")
    model = load_gyrolabe_bolmo(args.model_path, settings=settings, **hf_kwargs)
    model.eval()

    c_lib = gyro_ops._get_lib()
    opencl_ok = False
    try:
        from src.tools.gyrolabe import opencl_backend
        opencl_ok = opencl_backend.available()
    except (ImportError, OSError):
        pass

    nparams = sum(p.numel() for p in model.parameters())
    print(f"Loaded. Parameters: {nparams:,}")
    print(f"Backends: C library={'loaded' if c_lib else 'fallback'}, OpenCL={'yes' if opencl_ok else 'no'}")
    print(f"Decode expand cache: on")
    print("-" * 40)

    tokenizer = _get_tokenizer(model)
    if tokenizer is None:
        print("No tokenizer. Model needs tokenizer_config.build().")
        sys.exit(1)

    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    print("Chat (type 'quit' or 'exit' to stop).")
    print()

    with torch.no_grad():
        while True:
            try:
                prompt = input("You: ").strip()
            except EOFError:
                break
            if not prompt:
                continue
            if prompt.lower() in ("quit", "exit", "q"):
                break

            enc = tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].cpu()

            t0 = time.perf_counter()
            gen = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
            t1 = time.perf_counter()

            reply = tokenizer.decode(gen[0].tolist(), skip_special_tokens=True)
            if reply.startswith(prompt):
                reply = reply[len(prompt):].strip()
            new_tokens = gen.shape[1] - input_ids.shape[1]
            tok_s = new_tokens / (t1 - t0) if (t1 - t0) > 0 else 0

            print(f"Reply: {reply}")
            print(f"[{new_tokens} tokens in {1000*(t1-t0):.0f} ms, {tok_s:.1f} tok/s]")
            print()

    print("Bye.")


if __name__ == "__main__":
    main()
