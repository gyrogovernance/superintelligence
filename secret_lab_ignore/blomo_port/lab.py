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
from common import PROJECT_ROOT, load_bolmo, maybe_patch_expand_byte_ids
from module_0_baseline import baseline_generate
from module_1_observer import (
    ByteObservation,
    KernelObserver,
    ObservationLog,
    print_observation_summary,
)
from adaptors.boundary_adaptor import build_and_save_default_adaptor
from module_3_prefill_eval import evaluate_prompt_prefill
from module_4_patch_stats import patch_stats_for_prompt
from module_5_prefill_replace import run_module_5_porting_suite
from module_6_prefill_fast_port import run_module_6_prefill_fast_port
from module_7 import run_module_7

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
    print("\n--- Module 2: Build boundary_adaptor.npz ---")
    build_and_save_default_adaptor(model, tokenizer, device, model_dir, chunk_size=chunk_size)

def _run_module_3(model, tokenizer, model_dir):
    print("\n--- Module 3: Prefill boundary eval (adaptor vs Bolmo) ---")
    adaptor_path = PROJECT_ROOT / "data" / "cache" / "blomo_port" / "analysis" / "boundary_adaptor.npz"
    prompts = [
        "Language modeling is ",
        "The quick brown fox ",
        "def fibonacci(n):\n",
        "Hello world! This is a test of boundaries.\n",
    ]
    for K in (2048, 8192, 16384, 32768):
        print(f"\nK={K}:")
        for p in prompts:
            r = evaluate_prompt_prefill(model, tokenizer, adaptor_path, p, K=K)
            print(f"  prompt={p!r}")
            print(f"    compared={r.n_compared}/{r.n_positions}  R2={r.r2:.4f}  pearson={r.pearson:.4f}  MAE={r.mean_abs_err:.4f}  agree@0.5={r.agree_at_0p5:.3f}")

def _run_module_4(model, tokenizer, model_dir) -> None:
    print("\n--- Module 4: Patch stats (Bolmo vs Adaptor) ---")

    adaptor_path = PROJECT_ROOT / "data" / "cache" / "blomo_port" / "analysis" / "boundary_adaptor.npz"
    if not adaptor_path.exists():
        raise RuntimeError(
            f"Adaptor not found: {adaptor_path}\n"
            "Run module 2 first:\n"
            "  python secret_lab_ignore/blomo_port/lab.py --module 2"
        )

    prompts = [
        "Language modeling is ",
        "The quick brown fox ",
        "def fibonacci(n):\n",
    ]

    K = 16384
    thr = 0.5
    print(f"Using K={K}, threshold={thr}")

    for p in prompts:
        out = patch_stats_for_prompt(model, tokenizer, adaptor_path, p, K=K, threshold=thr)
        print(f"\nprompt={p!r}")
        print(f"  bolmo:   {out['bolmo']}")
        print(f"  adaptor: {out['adaptor']}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bolmo Kernel Port lab. Default: modules 0 and 1 (baseline + observer)."
    )
    parser.add_argument(
        "--module", "-m", type=int, action="append", default=[],
        choices=[0, 1, 2, 3, 4, 5, 6, 7],
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
    maybe_patch_expand_byte_ids(tokenizer)

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
    if 3 in modules:
        _run_module_3(model, tokenizer, model_dir)
        
    if 4 in modules:
        _run_module_4(model, tokenizer, model_dir)

    if 5 in modules:
        print("\n--- Module 5: Porting suite (analysis + 512 token A/B + deterministic equivalence) ---")
        run_module_5_porting_suite(
            model, tokenizer,
            prompts=[
                "Language modeling is ",
                "The quick brown fox ",
                "def fibonacci(n):\n",
            ],
            K=16384,
            max_new_tokens=max(512, args.max_tokens),
            preview_chars=1200,
        )
    if 6 in modules:
        print("\n--- Module 6: Prefill fast port (compute removal) ---")
        run_module_6_prefill_fast_port(
            model, tokenizer,
            prompt="Language modeling is ",
            K=16384,
            max_new_tokens=max(512, args.max_tokens),
        )

    if 7 in modules:
        print("\n--- Module 7: Suffix Residual Router / Embedding Analysis ---")
        run_module_7(model, tokenizer)

    print("\nDone.")

if __name__ == "__main__":
    main()
