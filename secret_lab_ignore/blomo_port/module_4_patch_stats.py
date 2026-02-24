"""
Module 4: Patch statistics (Bolmo vs Adaptor)

Given a prompt, compute:
- Bolmo prefill boundary probs -> boundaries -> patch lengths
- Adaptor boundary probs (from boundary_adaptor.npz) -> boundaries -> patch lengths
Compare rates and distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from module_3_prefill_eval import (
    adaptor_boundary_prob_for_adjacent_pairs,
    bolmo_prefill_boundary_probs,
    load_boundary_adaptor,
)
from common import token_to_byte_and_fused


@dataclass
class PatchStats:
    n_bytes: int
    n_pairs: int
    boundary_rate: float
    mean_bytes_per_patch: float
    median_bytes_per_patch: float
    max_patch: int


def _patch_lengths_from_boundary_probs(p: np.ndarray, threshold: float = 0.5) -> list[int]:
    """
    p length Npairs. Boundary after byte t iff p[t] > threshold.
    Convert into patch lengths over bytes.
    """
    if p.size == 0:
        return []
    boundaries = (p > threshold)
    lengths: list[int] = []
    cur = 1
    for is_bnd in boundaries:
        if is_bnd:
            lengths.append(cur)
            cur = 1
        else:
            cur += 1
    lengths.append(cur)
    return lengths


def _stats(lengths: list[int], n_bytes: int, n_pairs: int) -> PatchStats:
    if not lengths:
        return PatchStats(n_bytes=n_bytes, n_pairs=n_pairs, boundary_rate=0.0,
                         mean_bytes_per_patch=0.0, median_bytes_per_patch=0.0, max_patch=0)
    arr = np.array(lengths, dtype=np.int32)
    boundary_rate = float((len(lengths) - 1) / max(1, n_pairs))
    return PatchStats(
        n_bytes=n_bytes,
        n_pairs=n_pairs,
        boundary_rate=boundary_rate,
        mean_bytes_per_patch=float(np.mean(arr)),
        median_bytes_per_patch=float(np.median(arr)),
        max_patch=int(np.max(arr)),
    )


@torch.inference_mode()
def patch_stats_for_prompt(
    model: Any,
    tokenizer: Any,
    adaptor_path: Path,
    prompt: str,
    K: int = 16384,
    threshold: float = 0.5,
) -> dict[str, PatchStats]:

    adaptor = load_boundary_adaptor(adaptor_path)

    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)  # [1, L]
    ids = input_ids[0].detach().cpu().numpy().astype(np.int64)

    # Bolmo prefill boundary probs (length ~L-1)
    bolmo_p = bolmo_prefill_boundary_probs(model, input_ids)

    # Align to byte tokens only
    offset = int(getattr(tokenizer, "offset", 4))
    byte_at_pos = np.full(ids.shape[0], -1, dtype=np.int32)
    for t, tid in enumerate(ids.tolist()):
        b, _ = token_to_byte_and_fused(tid, offset)
        if b is not None:
            byte_at_pos[t] = int(b)

    cur_bytes: list[int] = []
    next_bytes: list[int] = []
    bolmo_vals: list[float] = []

    max_t = min(len(bolmo_p), len(byte_at_pos) - 1)
    for t in range(max_t):
        b0 = int(byte_at_pos[t])
        b1 = int(byte_at_pos[t + 1])
        if b0 >= 0 and b1 >= 0:
            cur_bytes.append(b0)
            next_bytes.append(b1)
            bolmo_vals.append(float(bolmo_p[t]))

    cur = np.array(cur_bytes, dtype=np.uint8)
    nxt = np.array(next_bytes, dtype=np.uint8)
    bolmo_arr = np.array(bolmo_vals, dtype=np.float32)

    adaptor_arr = adaptor_boundary_prob_for_adjacent_pairs(adaptor, cur, nxt, K=K)

    bolmo_lengths = _patch_lengths_from_boundary_probs(bolmo_arr, threshold=threshold)
    adaptor_lengths = _patch_lengths_from_boundary_probs(adaptor_arr, threshold=threshold)

    n_pairs = int(cur.size)
    n_bytes = int(cur.size + 1) if cur.size > 0 else 0

    return {
        "bolmo": _stats(bolmo_lengths, n_bytes, n_pairs),
        "adaptor": _stats(adaptor_lengths, n_bytes, n_pairs),
    }