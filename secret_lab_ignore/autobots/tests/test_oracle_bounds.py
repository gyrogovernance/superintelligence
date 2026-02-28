"""
Oracle bounds for Phase-1 curriculum.

Computes the best achievable loss (empirical conditional entropy) and
Bayes accuracy if we condition only on physics state. This defines
"perfect" for Phase-1: the model should approach these bounds.

Key: (state24, O, E, t_parity, last_byte) - aligns with model context.

Usage:
    pytest secret_lab_ignore/autobots/tests/test_oracle_bounds.py -v -s
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
import pytest
import torch

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from secret_lab_ignore.autobots import physics
from secret_lab_ignore.autobots.curriculum import CURRICULUM_TYPE_NAMES, CurriculumDataset
from secret_lab_ignore.autobots.oracle_utils import (
    build_hist,
    make_key_b,
    make_key_c,
    oracle_metrics_with_min_count,
)

CURRICULUM_DIR = _root / "data" / "autobots" / "curriculum"


def _compute_oracle_stats(sequences: list[list[int]], types: list[int] | None) -> dict:
    """KeyB stats with min_count=1."""
    hist = build_hist(sequences, make_key_b)
    stats = oracle_metrics_with_min_count(hist, min_count=1)
    return {
        "entropy": stats["entropy_cond_kept"],
        "bayes_acc": stats["bayes_cond_kept"],
        "num_keys": stats["total_keys"],
        "total_pairs": stats["total_pairs"],
    }


def _compute_per_type_bounds(
    sequences: list[list[int]], types: list[int]
) -> dict[int, dict]:
    """Compute oracle bounds per curriculum type."""
    by_type: dict[int, list] = defaultdict(list)
    for seq, t in zip(sequences, types):
        by_type[t].append(seq)
    out = {}
    for t, seqs in by_type.items():
        out[t] = _compute_oracle_stats(seqs, None)
    return out


class TestOracleBounds:
    """Empirical conditional entropy and Bayes accuracy from curriculum."""

    def test_oracle_bounds_print(self):
        """Print dataset lower bounds (best possible loss / accuracy)."""
        if not CURRICULUM_DIR.exists():
            pytest.skip(f"Curriculum not found at {CURRICULUM_DIR}")
        dataset = CurriculumDataset(CURRICULUM_DIR)
        if len(dataset.sequences) == 0:
            pytest.skip("Curriculum is empty")
        print("\n" + "=" * 70)
        print("ORACLE BOUNDS (physics-conditioned)")
        print("=" * 70)
        print("Key: (state24, O, E, t_parity, last_byte)")
        stats = _compute_oracle_stats(
            dataset.sequences,
            dataset.sequence_types if dataset.sequence_types else None,
        )
        print(f"\n  Overall:")
        print(f"    Conditional entropy (nats): {stats['entropy']:.4f}")
        print(f"    Bayes accuracy:             {stats['bayes_acc']:.4f}")
        print(f"    Unique keys:                {stats['num_keys']}")
        print(f"    Total (context, next) pairs:{stats['total_pairs']}")

        if dataset.sequence_types:
            print("\n  Per curriculum type:")
            per_type = _compute_per_type_bounds(dataset.sequences, dataset.sequence_types)
            print(f"    {'Type':<16} {'Entropy':>10} {'Bayes acc':>10} {'Keys':>8} {'Pairs':>10}")
            print("    " + "-" * 56)
            for t in sorted(per_type.keys()):
                s = per_type[t]
                name = CURRICULUM_TYPE_NAMES[t] if t < len(CURRICULUM_TYPE_NAMES) else f"type{t}"
                print(f"    {name:<16} {s['entropy']:>10.4f} {s['bayes_acc']:>10.4f} "
                      f"{s['num_keys']:>8} {s['total_pairs']:>10}")

        print("\n  Collision stats (KeyB): min_count=1 is memorization bound; min_count>=5 is repeatable-context bound")
        hist_b = build_hist(dataset.sequences, make_key_b)
        for mc in [1, 2, 5]:
            s = oracle_metrics_with_min_count(hist_b, min_count=mc)
            print(f"    min_count={mc}: kept_frac={s['kept_pair_frac']:.4f}, "
                  f"singletons={s['singleton_keys']}, keys_kept={s['num_keys_kept']}")

        print("\n  KeyC (coarser, drop O/E): (state24, t_parity, last_byte)")
        hist_c = build_hist(dataset.sequences, make_key_c)
        for mc in [1, 2, 5]:
            s = oracle_metrics_with_min_count(hist_c, min_count=mc)
            print(f"    min_count={mc}: H_kept={s['entropy_cond_kept']:.4f} bayes_kept={s['bayes_cond_kept']:.4f} "
                  f"kept_frac={s['kept_pair_frac']:.4f} keys_kept={s['num_keys_kept']}")

        print("\n  Theoretical limits:")
        print(f"    ln(256) uniform:  {math.log(256):.4f} nats")
        print(f"    ln(64) fam-locked:{math.log(64):.4f} nats")
        print(f"    ln(4) micro-lock: {math.log(4):.4f} nats")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
