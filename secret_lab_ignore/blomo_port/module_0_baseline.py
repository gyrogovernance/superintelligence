"""
Module 0: Baseline — pure Bolmo generation, no kernel (sanity check).

Uses model.generate() so behavior matches Bolmo's native boundary predictor
and decode logic; no custom decode loop.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from common import bolmo_reset_local_caches


@torch.inference_mode()
def baseline_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 40,
) -> tuple[str, list[int], float]:
    """
    Pure Bolmo generation using model.generate (no kernel, no GyroLabe).
    Returns:
        text     - full decoded text (prompt + completion)
        new_ids  - list of *new* token ids (excluding the prompt prefix)
        elapsed  - wallclock seconds
    """
    bolmo_reset_local_caches(model)
    torch.manual_seed(42)

    device = next(model.parameters()).device

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    input_len = input_ids.shape[1]

    t0 = time.perf_counter()
    output_ids = model.generate(
        input_ids,
        max_length=input_len + max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
    )
    elapsed = time.perf_counter() - t0

    full_ids = output_ids[0].tolist()
    new_ids = full_ids[input_len:]

    text = tokenizer.decode(full_ids, skip_special_tokens=True)

    bolmo_reset_local_caches(model)
    return text, new_ids, elapsed
