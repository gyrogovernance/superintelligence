"""
Shared oracle-bound utilities for Phase-1 curriculum evaluation.

KeyB: (state24, O, E, t_parity, last_byte)
KeyC: (state24, t_parity, last_byte) - coarser, more collisions
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Callable

from . import physics


def make_key_b(state24: int, o: int, e: int, t_parity: int, last_byte: int) -> tuple:
    return (state24, o, e, t_parity, last_byte)


def make_key_c(state24: int, o: int, e: int, t_parity: int, last_byte: int) -> tuple:
    return (state24, t_parity, last_byte)


def build_hist(
    sequences: list[list[int]],
    key_fn: Callable[[int, int, int, int, int], tuple],
) -> dict[tuple, Counter]:
    hist: dict[tuple, Counter] = defaultdict(Counter)
    mtab = physics.compute_mask12_table()
    for seq in sequences:
        if len(seq) < 2:
            continue
        state24 = physics.ARCHETYPE_STATE24
        o_acc, e_acc = 0, 0
        mask12s = [int(mtab[b].item()) & 0xFFF for b in seq]
        for t in range(len(seq) - 1):
            m = mask12s[t]
            if (t & 1) == 0:
                o_acc = (o_acc ^ m) & 0xFFF
            else:
                e_acc = (e_acc ^ m) & 0xFFF
            state24 = physics.step_state_l3_scalar(state24, seq[t])
            k = key_fn(state24, o_acc, e_acc, t % 2, seq[t])
            hist[k][seq[t + 1]] += 1
    return hist


def oracle_metrics_with_min_count(
    hist: dict[tuple, Counter], min_count: int = 1
) -> dict:
    total_pairs = sum(sum(c.values()) for c in hist.values())
    kept_pairs = 0
    keys_kept = 0
    singleton_keys = 0
    per: list[tuple[int, float, float]] = []

    for cnt in hist.values():
        n = sum(cnt.values())
        if n == 1:
            singleton_keys += 1
        if n < min_count:
            continue
        keys_kept += 1
        kept_pairs += n
        probs = [v / n for v in cnt.values()]
        ent_k = -sum(p * math.log(p + 1e-12) for p in probs)
        bayes_k = max(cnt.values()) / n
        per.append((n, ent_k, bayes_k))

    ent_total = 0.0
    bayes_total = 0.0
    for n, ent_k, bayes_k in per:
        w = n / max(total_pairs, 1)
        ent_total += w * ent_k
        bayes_total += w * bayes_k

    ent_kept = 0.0
    bayes_kept = 0.0
    for n, ent_k, bayes_k in per:
        w = n / max(kept_pairs, 1)
        ent_kept += w * ent_k
        bayes_kept += w * bayes_k

    return {
        "entropy_total_mass": ent_total,
        "bayes_total_mass": bayes_total,
        "entropy_cond_kept": ent_kept,
        "bayes_cond_kept": bayes_kept,
        "num_keys_kept": keys_kept,
        "total_keys": len(hist),
        "singleton_keys": singleton_keys,
        "total_pairs": total_pairs,
        "kept_pairs": kept_pairs,
        "kept_pair_frac": kept_pairs / max(total_pairs, 1),
    }
