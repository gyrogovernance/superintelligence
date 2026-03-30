#!/usr/bin/env python3
"""Benchmark real decode paths for Bolmo exact/stochastic wiring."""

from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path
from typing import TypedDict
import argparse

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.tools.gyrolabe.bridges.bolmo_config import (
    BolmoEncodeBridgeConfig,
    GyroLabeBolmoEncodeBridge,
    load_base_bolmo,
)
from src.tools.gyrograph.bridges.bolmo_config import (
    BolmoDecodeBridgeConfig,
    GyroGraphBolmoDecodeBridge,
)

PROMPTS = (
    "In 2026, exact byte-level decoding should still produce coherent language about",
    "The fundamental limits of computation are no longer just parameter counts because",
    "Gyroscopic climate control during decoding should preserve readable English while",
)

DEFAULT_REPEATS = 1
DEFAULT_MAX_NEW_TOKENS = 8
EXTRA_MODE_ENV = "QUBEC_EXTRA_WIRING"


class BenchmarkResult(TypedDict):
    text: str
    secs: float
    tokens: int
    tokens_per_sec: float
    quality: dict[str, float | int]
    patch_geometry: dict[str, int | float]
    hook_counts: dict[str, int]
    selection_counts: dict[str, int]


def _quality_metrics(text: str) -> dict[str, float | int]:
    s = text.strip()
    ascii_printable = sum(
        1 for ch in s
        if ord(ch) in (9, 10, 13) or (32 <= ord(ch) <= 126)
    )
    ascii_ratio = ascii_printable / max(1, len(s))

    max_run = 0
    cur = 0
    prev = None
    for ch in s:
        if ch == prev:
            cur += 1
        else:
            cur = 1
            prev = ch
        if cur > max_run:
            max_run = cur

    unique_ratio = len(set(s)) / max(1, len(s))
    spaces = s.count(" ")
    wordish_ratio = spaces / max(1, len(s))

    return {
        "length": len(s),
        "ascii_ratio": ascii_ratio,
        "max_run": max_run,
        "unique_ratio": unique_ratio,
        "wordish_ratio": wordish_ratio,
    }


def _ids(tok, text: str) -> torch.Tensor:
    return torch.tensor([tok.encode(text)], dtype=torch.long, device="cpu")


def _load_bolmo_model(*, show_load_output: bool) -> object:
    from transformers.utils import logging as hf_logging

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    if show_load_output:
        hf_logging.set_verbosity_info()
        return load_base_bolmo(torch_dtype=torch.float32, low_cpu_mem_usage=True)

    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            return load_base_bolmo(torch_dtype=torch.float32, low_cpu_mem_usage=True)


def _run_benchmark_generation(
    raw_bolmo,
    tokenizer,
    prompt: str,
    *,
    encode_cfg: BolmoEncodeBridgeConfig,
    decode_cfg: BolmoDecodeBridgeConfig,
    stream_id: str,
    max_new_tokens: int,
    do_sample: bool,
    max_time: float | None = None,
) -> BenchmarkResult:
    input_ids = _ids(tokenizer, prompt)
    encode_bridge = GyroLabeBolmoEncodeBridge(raw_bolmo, config=encode_cfg)
    encode_bridge.eval()
    decode_bridge = GyroGraphBolmoDecodeBridge(
        config=decode_cfg,
        cell_capacity=256,
        use_opencl_hotpath=False,
    )

    try:
        print(f"[{stream_id}] attach")
        t0 = time.perf_counter()
        with torch.no_grad():
            with decode_bridge.session(
                encode_bridge,
                batch_size=1,
                stream_ids=[stream_id],
            ):
                print(f"[{stream_id}] generate start")
                out = encode_bridge.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    use_cache=True,
                    **({"max_time": max_time} if max_time is not None else {}),
                )
                print(f"[{stream_id}] generate done")
                report = decode_bridge.emit_stream_report(stream_id)
                hook_counts = decode_bridge.last_hook_counts()
                selection_counts = decode_bridge.last_selection_counts()
    finally:
        encode_bridge.uninstall()
        decode_bridge.detach()

    dt = time.perf_counter() - t0
    generated = out[0, input_ids.shape[1] :].to(torch.int64).cpu()
    new_tokens = int(generated.numel())
    text = tokenizer.decode(generated.tolist())

    return {
        "text": text,
        "secs": dt,
        "tokens": new_tokens,
        "tokens_per_sec": new_tokens / max(dt, 1e-12),
        "quality": _quality_metrics(text),
        "patch_geometry": report.patch_geometry,
        "hook_counts": hook_counts,
        "selection_counts": selection_counts,
    }


def run_full_exact(
    raw_bolmo,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    stream_id: str,
    max_time: float | None = None,
) -> BenchmarkResult:
    encode_cfg = BolmoEncodeBridgeConfig.wiring_quality_approved()
    decode_cfg = BolmoDecodeBridgeConfig.wiring_quality_approved(selection_mode="flat")
    return _run_benchmark_generation(
        raw_bolmo=raw_bolmo,
        tokenizer=tokenizer,
        prompt=prompt,
        encode_cfg=encode_cfg,
        decode_cfg=decode_cfg,
        stream_id=stream_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        max_time=max_time,
    )


def run_wiring_quality(
    raw_bolmo,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    stream_id: str,
    max_time: float | None = None,
) -> BenchmarkResult:
    return run_full_exact(
        raw_bolmo=raw_bolmo,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        stream_id=stream_id,
        max_time=max_time,
    )


def run_decode_sampling(
    raw_bolmo,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    stream_id: str,
    max_time: float | None = None,
) -> BenchmarkResult:
    encode_cfg = BolmoEncodeBridgeConfig.wiring_quality_approved()
    decode_cfg = BolmoDecodeBridgeConfig.wiring_sampling()
    return _run_benchmark_generation(
        raw_bolmo=raw_bolmo,
        tokenizer=tokenizer,
        prompt=prompt,
        encode_cfg=encode_cfg,
        decode_cfg=decode_cfg,
        stream_id=stream_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        max_time=max_time,
    )


def run_wiring_sampling(
    raw_bolmo,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    stream_id: str,
    max_time: float | None = None,
) -> BenchmarkResult:
    return run_decode_sampling(
        raw_bolmo=raw_bolmo,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        stream_id=stream_id,
        max_time=max_time,
    )


def run_experimental_full(
    raw_bolmo,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    stream_id: str,
    max_time: float | None = None,
) -> BenchmarkResult:
    print("[experimental_full] semantic drift expected")
    encode_cfg = BolmoEncodeBridgeConfig.exact_prefill()
    decode_cfg = BolmoDecodeBridgeConfig.global_semantic_experimental_stable(
        selection_mode="paired", proof_mode=True
    )
    return _run_benchmark_generation(
        raw_bolmo=raw_bolmo,
        tokenizer=tokenizer,
        prompt=prompt,
        encode_cfg=encode_cfg,
        decode_cfg=decode_cfg,
        stream_id=stream_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        max_time=max_time,
    )


def run_global_semantic_experimental(
    raw_bolmo,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    stream_id: str,
    max_time: float | None = None,
) -> BenchmarkResult:
    print("[global_semantic_experimental] semantic rewrite path")
    encode_cfg = BolmoEncodeBridgeConfig.exact_prefill()
    decode_cfg = BolmoDecodeBridgeConfig.global_semantic_experimental_stable(
        selection_mode="flat", proof_mode=False
    )
    return _run_benchmark_generation(
        raw_bolmo=raw_bolmo,
        tokenizer=tokenizer,
        prompt=prompt,
        encode_cfg=encode_cfg,
        decode_cfg=decode_cfg,
        stream_id=stream_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        max_time=max_time,
    )


def run_global_matmul_only(
    raw_bolmo,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    stream_id: str,
    max_time: float | None = None,
) -> BenchmarkResult:
    print("[global_matmul_only] matmul surface replacement only")
    encode_cfg = BolmoEncodeBridgeConfig.gyromatmul_prefill_only()
    decode_cfg = BolmoDecodeBridgeConfig.global_matmul_only()
    return _run_benchmark_generation(
        raw_bolmo=raw_bolmo,
        tokenizer=tokenizer,
        prompt=prompt,
        encode_cfg=encode_cfg,
        decode_cfg=decode_cfg,
        stream_id=stream_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        max_time=max_time,
    )


def run_full_wiring(
    raw_bolmo,
    tokenizer,
    prompt,
    max_new_tokens: int,
    stream_id: str,
    max_time: float | None = None,
) -> BenchmarkResult:
    # legacy alias for compatibility with existing docs/scripts
    return run_global_semantic_experimental(
        raw_bolmo=raw_bolmo,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        stream_id=stream_id,
        max_time=max_time,
    )


def _print_result(
    mode: str,
    prompt_index: int,
    prompt_total: int,
    prompt: str,
    result: BenchmarkResult,
) -> None:
    print(f"\n[prompt {prompt_index}/{prompt_total}]")
    print(f"  prompt={prompt!r}")
    print(f"  text[:220]={result['text'][:220]!r}")
    print(f"  secs={result['secs']:.3f}")
    print(f"  tokens={result['tokens']}")
    print(f"  tokens_per_sec={result['tokens_per_sec']:.2f}")
    print(f"  quality={result['quality']}")

    if mode in {
        "full_exact",
        "wiring_quality",
        "decode_sampling",
        "wiring_sampling",
        "experimental_full",
        "global_semantic_experimental",
        "global_matmul_only",
        "full_wiring",
    }:
        if "patch_geometry" in result:
            print(f"  patch_geometry={result['patch_geometry']}")
        if "hook_counts" in result:
            print(f"  hook_counts={result['hook_counts']}")
        if "selection_counts" in result:
            print(f"  selection_counts={result['selection_counts']}")


def _print_run_checks(mode: str, result: BenchmarkResult) -> None:
    q = result["quality"]
    if not isinstance(q, dict):
        return

    if mode in {"full_exact", "wiring_quality"}:
        if q["ascii_ratio"] < 0.98 or q["max_run"] >= 25 or q["unique_ratio"] <= 0.05:
            print("  FAIL: non-garbage heuristic")
        selection_counts = result["selection_counts"]
        if selection_counts.get("exact_qsector", 0) <= 0 and selection_counts.get(
            "flat", 0
        ) <= 0:
            print("  FAIL: no selector path recorded for full_exact")
        if result["patch_geometry"].get("patch_count", 0) < 1:
            print("  FAIL: no patch activity recorded")

    if mode in {"decode_sampling", "wiring_sampling"}:
        if result["selection_counts"].get("sample", 0) <= 0:
            print("  FAIL: sample selection did not run")
        if result["patch_geometry"].get("patch_count", 0) < 1:
            print("  FAIL: no patch activity recorded")

    if mode in {"full_wiring", "global_semantic_experimental", "global_matmul_only"}:
        selection_counts = result["selection_counts"]
        if selection_counts.get("exact_qsector", 0) <= 0 and selection_counts.get(
            "flat", 0
        ) <= 0:
            print("  FAIL: no selector path recorded")
        if result["patch_geometry"].get("patch_count", 0) < 1:
            print("  FAIL: no patch activity recorded")


def _run_one_mode(
    mode: str,
    runner,
    raw_bolmo,
    tokenizer,
    max_new_tokens: int,
    max_time: float | None,
    repeats: int,
    prompts: tuple[str, ...],
) -> None:
    for rep in range(max(1, repeats)):
        if repeats > 1:
            print(f"\n[run {rep + 1}/{repeats}]")
        for i, prompt in enumerate(prompts, start=1):
            result = runner(
                raw_bolmo,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                max_time=max_time,
                stream_id=f"{mode}:{rep}:{i}",
            )
            _print_result(mode, i, len(prompts), prompt, result)
            _print_run_checks(mode, result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run focused wiring benchmarks on Bolmo decode paths."
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        choices=[
            "full_exact",
            "wiring_quality",
            "decode_sampling",
            "wiring_sampling",
            "global_matmul_only",
            "global_semantic_experimental",
            "full_wiring",
            "experimental_full",
        ],
        default=None,
        help="Benchmark mode to run (default: full_exact plus optional extras via QUBEC_EXTRA_WIRING).",
    )
    parser.add_argument(
        "--max-time",
        dest="max_time",
        type=float,
        default=None,
        help="Optional generation timeout in seconds. Omit to disable.",
    )
    parser.add_argument(
        "--prompt-index",
        dest="prompt_index",
        type=int,
        default=None,
        help="Optional 1-based prompt index from the built-in prompt list.",
    )
    parser.add_argument(
        "--repeats",
        dest="repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="How many repeats for each selected prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum tokens to generate per prompt.",
    )

    args = parser.parse_args()

    repeats = max(1, args.repeats)
    max_new_tokens = max(1, args.max_new_tokens)
    max_time = args.max_time
    if max_time is not None and max_time <= 0:
        max_time = None
    include_extra_modes = os.environ.get(EXTRA_MODE_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}

    raw_bolmo = _load_bolmo_model(show_load_output=False)
    raw_bolmo.eval()
    tokenizer = raw_bolmo.model.tokenizer
    if args.prompt_index is not None:
        if not (1 <= args.prompt_index <= len(PROMPTS)):
            raise ValueError(
                f"prompt_index out of range: {args.prompt_index}. "
                f"Expected 1 to {len(PROMPTS)}."
            )
    selected_prompts = (PROMPTS[args.prompt_index - 1],) if args.prompt_index else (PROMPTS[0],)

    print(f"[mode {'full_exact' if not args.mode else args.mode}]")
    print(
        f"  repeats={repeats}, max_new_tokens={max_new_tokens}, "
        f"max_time={max_time if max_time is not None else 'off'}"
    )
    mode_to_runner = {
        "full_exact": run_full_exact,
        "wiring_quality": run_wiring_quality,
        "decode_sampling": run_decode_sampling,
        "wiring_sampling": run_wiring_sampling,
        "global_matmul_only": run_global_matmul_only,
        "global_semantic_experimental": run_global_semantic_experimental,
        "experimental_full": run_experimental_full,
        "full_wiring": run_full_wiring,
    }

    if args.mode is not None:
        _run_one_mode(
            args.mode,
            mode_to_runner[args.mode],
            raw_bolmo,
            tokenizer,
            max_new_tokens=max_new_tokens,
            max_time=max_time,
            repeats=repeats,
            prompts=selected_prompts,
        )
        return

    _run_one_mode(
        "full_exact",
        run_full_exact,
        raw_bolmo,
        tokenizer,
        max_new_tokens=max_new_tokens,
        max_time=max_time,
        repeats=repeats,
        prompts=selected_prompts,
    )

    if include_extra_modes:
        _run_one_mode(
            "decode_sampling",
            run_decode_sampling,
            raw_bolmo,
            tokenizer,
            max_new_tokens=max_new_tokens,
            max_time=max_time,
            repeats=repeats,
            prompts=selected_prompts,
        )
        _run_one_mode(
            "experimental_full",
            run_experimental_full,
            raw_bolmo,
            tokenizer,
            max_new_tokens=max_new_tokens,
            max_time=max_time,
            repeats=repeats,
            prompts=selected_prompts,
        )
        _run_one_mode(
            "global_matmul_only",
            run_global_matmul_only,
            raw_bolmo,
            tokenizer,
            max_new_tokens=max_new_tokens,
            max_time=max_time,
            repeats=repeats,
            prompts=selected_prompts,
        )
        _run_one_mode(
            "global_semantic_experimental",
            run_global_semantic_experimental,
            raw_bolmo,
            tokenizer,
            max_new_tokens=max_new_tokens,
            max_time=max_time,
            repeats=repeats,
            prompts=selected_prompts,
        )
        _run_one_mode(
            "full_wiring",
            run_full_wiring,
            raw_bolmo,
            tokenizer,
            max_new_tokens=max_new_tokens,
            max_time=max_time,
            repeats=repeats,
            prompts=selected_prompts,
        )


if __name__ == "__main__":
    main()
