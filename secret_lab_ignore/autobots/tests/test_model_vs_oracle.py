"""
Compare trained model to oracle bounds on the Phase-1 curriculum.

- Oracle: empirical conditional entropy + Bayes acc
  KeyB: (state24, O, E, t_parity, last_byte)
  KeyC: (state24, t_parity, last_byte) - coarser, collision-aware

- Model: NLL + top-1 acc on the same token pairs.

Usage:
  pytest secret_lab_ignore/autobots/tests/test_model_vs_oracle.py -s -v
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from secret_lab_ignore.autobots.curriculum import (
    CURRICULUM_TYPE_NAMES,
    CurriculumDataset,
)
from secret_lab_ignore.autobots.model import HGTForCausalLM
from secret_lab_ignore.autobots.oracle_utils import (
    build_hist,
    make_key_b,
    make_key_c,
    oracle_metrics_with_min_count,
)

MODEL_DIR = _root / "data" / "autobots" / "model"
CURRICULUM_DIR = _root / "data" / "autobots" / "curriculum"


@torch.no_grad()
def _model_metrics_on_sequences(
    model: HGTForCausalLM,
    sequences: list[list[int]],
    device: torch.device,
    max_tokens: int = 300_000,
) -> tuple[float, float, int]:
    """
    Compute average NLL and top-1 acc on given sequences, up to max_tokens pairs.
    """
    nll_sum = 0.0
    correct = 0
    count = 0

    for seq in sequences:
        if len(seq) < 2:
            continue
        ids = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        out = model(input_ids=ids)
        logits = out.logits[0]

        for t in range(len(seq) - 1):
            tgt = seq[t + 1]
            logp = F.log_softmax(logits[t], dim=-1)[tgt].item()
            nll_sum += -logp
            pred = int(logits[t].argmax().item())
            correct += int(pred == tgt)
            count += 1
            if count >= max_tokens:
                return nll_sum / count, correct / count, count

    return nll_sum / max(count, 1), correct / max(count, 1), count


class TestModelVsOracle:
    def test_print_model_vs_oracle(self):
        if not MODEL_DIR.exists():
            pytest.skip(f"Model not found at {MODEL_DIR}")
        if not CURRICULUM_DIR.exists():
            pytest.skip(f"Curriculum not found at {CURRICULUM_DIR}")

        ds = CurriculumDataset(CURRICULUM_DIR)
        if len(ds.sequences) == 0:
            pytest.skip("Curriculum empty")

        model = HGTForCausalLM.from_pretrained(MODEL_DIR)
        model.eval()
        device = torch.device("cpu")
        model.to(device)

        print("\n" + "=" * 70)
        print("MODEL vs ORACLE (Phase-1 curriculum)")
        print("=" * 70)

        hist_b = build_hist(ds.sequences, make_key_b)
        hist_c = build_hist(ds.sequences, make_key_c)
        m_nll, m_acc, m_pairs = _model_metrics_on_sequences(
            model, ds.sequences, device=device, max_tokens=400_000
        )

        print("\nKeyB (state24, O, E, t_parity, last_byte):")
        for mc in [1, 2, 5]:
            s = oracle_metrics_with_min_count(hist_b, min_count=mc)
            print(f"  min_count={mc}: H_kept={s['entropy_cond_kept']:.4f} bayes_kept={s['bayes_cond_kept']:.4f} "
                  f"kept_frac={s['kept_pair_frac']:.4f} singletons={s['singleton_keys']}")
        s1 = oracle_metrics_with_min_count(hist_b, min_count=1)
        print(f"  model_NLL={m_nll:.4f} model_acc={m_acc:.4f} (gap to KeyB H_kept: {m_nll - s1['entropy_cond_kept']:+.4f})")

        print("\nKeyC (state24, t_parity, last_byte) - coarser:")
        for mc in [1, 2, 5]:
            s = oracle_metrics_with_min_count(hist_c, min_count=mc)
            print(f"  min_count={mc}: H_kept={s['entropy_cond_kept']:.4f} bayes_kept={s['bayes_cond_kept']:.4f} "
                  f"kept_frac={s['kept_pair_frac']:.4f} keys_kept={s['num_keys_kept']}")

        LN256 = math.log(256)
        LN64 = math.log(64)
        LN4 = math.log(4)
        POLICY_BASELINE = {
            "family_locked": LN64,
            "micro_locked": LN4,
            "separator": 0.5 * LN256,
            "random": LN256,
        }

        if ds.sequence_types:
            print("\nPer curriculum type (oracle vs model):")
            print(f"{'type':<16} {'oracleH':>9} {'modelNLL':>9} {'gap':>9} {'policy':>9} {'gap_pol':>9}")
            by_type: dict[int, list[list[int]]] = defaultdict(list)
            for seq, t in zip(ds.sequences, ds.sequence_types):
                by_type[int(t)].append(seq)

            for t in sorted(by_type.keys()):
                seqs = by_type[t]
                hist_t = build_hist(seqs, make_key_b)
                s_t = oracle_metrics_with_min_count(hist_t, min_count=1)
                o_ent_t = s_t["entropy_cond_kept"]
                m_nll_t, m_acc_t, _ = _model_metrics_on_sequences(
                    model, seqs, device=device, max_tokens=100_000
                )
                name = CURRICULUM_TYPE_NAMES[t] if t < len(CURRICULUM_TYPE_NAMES) else f"type{t}"
                policy_h = POLICY_BASELINE.get(name)
                if policy_h is not None:
                    gap_pol = m_nll_t - policy_h
                    print(f"{name:<16} {o_ent_t:>9.3f} {m_nll_t:>9.3f} {m_nll_t - o_ent_t:>+9.3f} "
                          f"{policy_h:>9.3f} {gap_pol:>+9.3f}")
                else:
                    print(f"{name:<16} {o_ent_t:>9.3f} {m_nll_t:>9.3f} {m_nll_t - o_ent_t:>+9.3f} "
                          f"{'':>9} {'':>9}")

        print("\nReference (policy baselines for random-by-design types):")
        print(f"  ln(256)={LN256:.4f}  ln(64)={LN64:.4f}  ln(4)={LN4:.4f}  0.5*ln256={0.5*LN256:.4f}")
