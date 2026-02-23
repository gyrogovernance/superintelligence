"""
Bolmo Kernel Port — Modular Laboratory (single entry point)

Module 0: Baseline — pure Bolmo, no kernel (sanity check)
Module 1: Observer — kernel runs alongside, records observables
Module 2: Boundary extraction + kernel-Bolmo correlation (memory-heavy; small chunk by default)

Default run: modules 0 and 1 only (normal behavior). Add --module 2 for boundary suite.
Paths resolved from repo root via common.PROJECT_ROOT. Run from repo root or blomo_port/.

python secret_lab_ignore/blomo_port/lab.py

"""

from __future__ import annotations

import argparse

import common
from common import PROJECT_ROOT, load_bolmo
from module_0_baseline import baseline_generate
from module_1_observer import (
    ByteObservation,
    KernelObserver,
    ObservationLog,
    print_observation_summary,
)
from bolmo_adaptor import build_and_save_default_adaptor


def _run_module_0(model, tokenizer, prompts, max_new_tokens: int = 200) -> None:
    print("\n--- Module 0: Baseline Generation ---")
    for prompt in prompts:
        text, gen_ids, elapsed = baseline_generate(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens
        )
        n_bytes = len(gen_ids)
        print(f"Prompt: {prompt!r}")
        print(f"  {n_bytes} bytes in {elapsed:.2f}s ({n_bytes/max(elapsed,1e-9):.1f} tok/s)")
        display = text[:300].replace("\n", "\\n")
        print(f"  Output: {display}")
        print()


def _run_module_1(model, tokenizer, atlas_dir, prompts, max_new_tokens: int = 200) -> None:
    print("\n--- Module 1: Kernel Observer ---")
    observer = KernelObserver(atlas_dir=atlas_dir)
    offset = int(getattr(tokenizer, "offset", 4))

    for prompt in prompts:
        observer.reset()
        text, gen_ids, elapsed = baseline_generate(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens
        )
        observer.observe_token_sequence(gen_ids, offset)
        print(f"Prompt: {prompt!r}")
        print_observation_summary(observer.log, label="  ")
        print()


def _run_module_2(model, tokenizer, device, atlas_dir, model_dir, chunk_size: int) -> None:
    print("\n--- Module 2: Build bolmo_adaptor.npz ---")
    build_and_save_default_adaptor(model, tokenizer, device, model_dir, chunk_size=chunk_size)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bolmo Kernel Port lab. Default: modules 0 and 1 (baseline + observer)."
    )
    parser.add_argument(
        "--module", "-m", type=int, action="append", default=[],
        choices=[0, 1, 2],
        help="Module to run (repeat for multiple). Default: 0 1.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=200,
        help="Max new tokens for generation (modules 0, 1).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=16,
        help="Chunk size for module 2 boundary extraction (small = less memory).",
    )
    args = parser.parse_args()

    modules = args.module if args.module else [0, 1]
    atlas_dir = PROJECT_ROOT / "data" / "atlas"
    model_dir = PROJECT_ROOT / "data" / "models" / "Bolmo-1B"

    if not atlas_dir.exists():
        raise RuntimeError(f"Atlas not found: {atlas_dir}")
    if not model_dir.exists():
        raise RuntimeError(f"Model not found: {model_dir}")

    from src.tools.gyrolabe import detect_device
    device = detect_device()
    model, tokenizer = load_bolmo(model_dir, device)

    prompts = [
        "Language modeling is ",
        "The quick brown fox ",
        "def fibonacci(n):\n",
    ]

    if 0 in modules:
        _run_module_0(model, tokenizer, prompts, max_new_tokens=args.max_tokens)
    if 1 in modules:
        _run_module_1(
            model, tokenizer, atlas_dir, prompts,
            max_new_tokens=args.max_tokens,
        )
    if 2 in modules:
        _run_module_2(
            model, tokenizer, device, atlas_dir, model_dir,
            chunk_size=args.chunk_size,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
