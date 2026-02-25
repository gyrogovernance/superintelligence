# secret_lab_ignore/blomo_port/module_7.py
"""
Module 7 (v4): Generate-level streaming suffix via hook (lab-only patch)

Goal
----
Eliminate O(L) expand_byte_ids(..., n_last=1) in the decode loop by using a
decode-expand hook. All logic lives in the lab; modeling_bolmo.py only has
minimal hook call sites (init / get_expanded / append).

How it works
------------
- tokenizer._decode_expand_hook is set by the lab for the duration of generation.
- generate() calls the hook at init (after rollback), get_expanded (each step),
  and append (after appending next_tokens). The hook maintains state and
  returns expanded ids instead of calling expand_byte_ids.

Outputs
-------
- Base vs fused micro-test (expand_byte_ids equality)
- Baseline vs streaming generation time and token-id equality
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from common import bolmo_reset_local_caches
from module_0_baseline import baseline_generate

from adaptors.decode_streaming_adaptor import (
    DecodeStreamingHook,
    install_decode_streaming_hook,
    uninstall_decode_streaming_hook,
)


def _first_diverge(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


# ----- Hook patch context manager -----


class DecodeExpandHookPatch:
    def __init__(self, tokenizer: Any, hook: Any):
        self.tokenizer = tokenizer
        self.hook = hook
        self._had = False
        self._old: Optional[Any] = None

    def __enter__(self) -> "DecodeExpandHookPatch":
        self._had = hasattr(self.tokenizer, "_decode_expand_hook")
        self._old = getattr(self.tokenizer, "_decode_expand_hook", None)
        self.tokenizer._decode_expand_hook = self.hook  # type: ignore[attr-defined]
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._had:
            self.tokenizer._decode_expand_hook = self._old  # type: ignore[attr-defined]
        else:
            delattr(self.tokenizer, "_decode_expand_hook")


# ----- Base vs fused micro-test -----


def _run_base_vs_fused_test(tokenizer: Any) -> None:
    """Test whether expand_byte_ids(base) == expand_byte_ids(fused) for same byte."""
    off = tokenizer.offset
    b = ord("a")
    base = [tokenizer.bos_token_id, off + b]
    fused = [tokenizer.bos_token_id, off + 256 + b]
    e_base = tokenizer.expand_byte_ids(base)
    e_fused = tokenizer.expand_byte_ids(fused)
    print("[Base vs fused] base:", e_base, "fused:", e_fused, "equal?", e_base == e_fused)


def run_module_7(
    model: Any,
    tokenizer: Any,
    prompt: str = "Language modeling is ",
    max_new_tokens: int = 300,
) -> None:
    print("\n--- Module 7 (v4): Streaming suffix via decode-expand hook ---")
    print("Goal: hook provides expanded ids from streaming state; no expand_byte_ids in loop.")

    _run_base_vs_fused_test(tokenizer)

    # Clean up any stale hook
    uninstall_decode_streaming_hook(tokenizer)

    # Baseline: no hook
    bolmo_reset_local_caches(model)
    torch.manual_seed(42)
    base_text, base_ids, base_dt = baseline_generate(
        model, tokenizer, prompt, max_new_tokens=max_new_tokens,
        temperature=1.0, top_k=0,
    )
    print("\n[Baseline]")
    print(f"  tokens={len(base_ids)} time={base_dt:.2f}s")
    print("  preview:", base_text[:220].replace("\n", "\\n"))

    # Streaming: install hook and run
    hook = install_decode_streaming_hook(tokenizer, None)

    bolmo_reset_local_caches(model)
    torch.manual_seed(42)
    pat_text, pat_ids, pat_dt = baseline_generate(
        model, tokenizer, prompt, max_new_tokens=max_new_tokens,
        temperature=1.0, top_k=0,
    )

    uninstall_decode_streaming_hook(tokenizer)

    print("\n[Streaming (hook)]")
    print(f"  tokens={len(pat_ids)} time={pat_dt:.2f}s")
    print("  preview:", pat_text[:220].replace("\n", "\\n"))

    n = min(len(base_ids), len(pat_ids))
    eq = sum(1 for i in range(n) if base_ids[i] == pat_ids[i])
    div = _first_diverge(base_ids, pat_ids)
    print("\n[Equivalence]")
    print(f"  token-id match: {eq}/{n} = {eq/max(1,n):.3f}")
    print(f"  first divergence step: {div}")
    if div == n and len(base_ids) == len(pat_ids):
        print("  RESULT: exact match (streaming hook preserves behavior).")
    else:
        print("  RESULT: mismatch.")

    if pat_dt > 1e-9:
        print("\n[Speed]")
        print(f"  baseline: {base_dt:.2f}s")
        print(f"  streaming: {pat_dt:.2f}s")
        print(f"  ratio: {base_dt/pat_dt:.2f}x")
    print("\nModule 7 done.")
