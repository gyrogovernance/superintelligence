"""
Module 6: Prefill Fast Port (compute removal)

What it does:
- Monkeypatch model.model.prefill_boundary_prediction_forward to compute boundary_mask
  from boundary_adaptor.npz directly (no local_encoder pass for boundaries).
- Counts calls to boundary_predictor_module.forward to prove compute reduction.
- Runs 512-token deterministic equivalence test:
    baseline deterministic (Bolmo boundary mask = p>0.5)
    patched deterministic (Adaptor boundary mask = p>0.5, computed in prefill fast path)
  and reports token-id match.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from common import PROJECT_ROOT, bolmo_reset_local_caches
from module_0_baseline import baseline_generate
from module_5_prefill_replace import (
    load_boundary_adaptor,
    adaptor_boundary_probs_for_adjacent_pairs,
    BoundaryPredictorDeterminizeBolmoPatch,  # from pumped module 5
)


@dataclass
class CallCounter:
    n_calls: int = 0


class BoundaryPredictorCallCountPatch:
    """Wrap boundary_predictor_module.forward to count invocations."""
    def __init__(self, model: Any, counter: CallCounter):
        self.model = model
        self.counter = counter
        self._orig = None

    def __enter__(self):
        bp = self.model.model.local_encoder.boundary_predictor_module
        self._orig = bp.forward

        def wrapped(*args, **kwargs):
            self.counter.n_calls += 1
            assert self._orig is not None
            return self._orig(*args, **kwargs)

        bp.forward = wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        bp = self.model.model.local_encoder.boundary_predictor_module
        if self._orig is not None:
            bp.forward = self._orig


class PrefillBoundaryFastAdaptorPatch:
    """
    Replace model.model.prefill_boundary_prediction_forward so boundary masks are computed
    from boundary_adaptor.npz instead of local_encoder boundary head.
    """
    def __init__(self, model: Any, adaptor: dict[str, Any], tokenizer: Any, K: int = 16384, threshold: float = 0.5):
        self.model = model
        self.adaptor = adaptor
        self.tokenizer = tokenizer
        self.K = int(K)
        self.threshold = float(threshold)
        self._orig = None

    def __enter__(self):
        m = self.model.model
        self._orig = m.prefill_boundary_prediction_forward

        offset = int(getattr(self.tokenizer, "offset", 4))
        eps = float(self.adaptor.get("eps", 1e-6))

        def patched_prefill_boundary_prediction_forward(
            byte_input_ids: torch.Tensor,
            expanded_input_ids: Optional[torch.Tensor] = None,
            sequence_start_indices: Optional[torch.Tensor] = None,
            **kwargs,
        ):
            # byte_input_ids: [B, L]
            device = byte_input_ids.device
            ids = byte_input_ids.detach().cpu().numpy().astype(np.int64)
            B, L = ids.shape

            # boundary_mask must be shape [B, L] in Bolmo
            if L == 0:
                return torch.empty((B, 0), device=device, dtype=torch.bool)

            if L == 1:
                return torch.zeros((B, 1), device=device, dtype=torch.bool)

            # gaps are [B, L-1]
            cur = ids[:, :-1]
            nxt = ids[:, 1:]

            # valid raw bytes are in [offset, offset+255]
            valid = (
                (cur >= offset) & (cur < offset + 256) &
                (nxt >= offset) & (nxt < offset + 256)
            )

            # default force boundary for invalid gaps
            p = np.ones((B, L - 1), dtype=np.float32)

            if np.any(valid):
                b_cur = (cur[valid] - offset).astype(np.uint8)
                b_nxt = (nxt[valid] - offset).astype(np.uint8)
                p_valid = adaptor_boundary_probs_for_adjacent_pairs(self.adaptor, b_cur, b_nxt, K=self.K)
                p[valid] = p_valid

            # deterministic boundary mask (this module is about deterministic port)
            mask_gap = (p > self.threshold)

            # pad to length L (lookahead=1) by adding one False column
            mask_full = np.zeros((B, L), dtype=np.bool_)
            mask_full[:, : L - 1] = mask_gap

            return torch.from_numpy(mask_full).to(device=device)

        m.prefill_boundary_prediction_forward = patched_prefill_boundary_prediction_forward
        return self

    def __exit__(self, exc_type, exc, tb):
        m = self.model.model
        if self._orig is not None:
            m.prefill_boundary_prediction_forward = self._orig


def _first_diverge(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def run_module_6_prefill_fast_port(
    model: Any,
    tokenizer: Any,
    prompt: str = "Language modeling is ",
    K: int = 16384,
    max_new_tokens: int = 512,
) -> None:
    adaptor_path = PROJECT_ROOT / "data" / "cache" / "blomo_port" / "analysis" / "boundary_adaptor.npz"
    if not adaptor_path.exists():
        raise RuntimeError(
            f"boundary_adaptor.npz not found: {adaptor_path}\n"
            "Run module 2 first."
        )

    adaptor = load_boundary_adaptor(adaptor_path)

    print("\n[Module 6] Prefill fast port + compute removal proof")
    print(f"  prompt={prompt!r}")
    print(f"  K={K}")
    print(f"  max_new_tokens={max_new_tokens}")

    # Baseline deterministic boundaries (force boundary_mask = p>0.5)
    # This creates a stable reference.
    bolmo_reset_local_caches(model)
    base_ctr = CallCounter()
    with BoundaryPredictorCallCountPatch(model, base_ctr):
        with BoundaryPredictorDeterminizeBolmoPatch(model, threshold=0.5):
            base_text, base_ids, base_t = baseline_generate(
                model, tokenizer, prompt, max_new_tokens=max_new_tokens
            )

    print(f"\nBaseline deterministic:")
    print(f"  tokens={len(base_ids)} time={base_t:.2f}s  boundary_predictor_calls={base_ctr.n_calls}")
    print("  preview:")
    print(base_text[:900].replace("\n", "\\n"))

    # Patched: prefill boundary computed from adaptor, and still determinize any remaining prefill boundary sampling
    bolmo_reset_local_caches(model)
    pat_ctr = CallCounter()
    with BoundaryPredictorCallCountPatch(model, pat_ctr):
        with PrefillBoundaryFastAdaptorPatch(model, adaptor, tokenizer, K=K, threshold=0.5):
            # keep determinize patch too (harmless; should not matter if prefill head not called)
            with BoundaryPredictorDeterminizeBolmoPatch(model, threshold=0.5):
                pat_text, pat_ids, pat_t = baseline_generate(
                    model, tokenizer, prompt, max_new_tokens=max_new_tokens
                )

    print(f"\nPatched deterministic (prefill fast):")
    print(f"  tokens={len(pat_ids)} time={pat_t:.2f}s  boundary_predictor_calls={pat_ctr.n_calls}")
    print("  preview:")
    print(pat_text[:900].replace("\n", "\\n"))

    # Compare token outputs
    n = min(len(base_ids), len(pat_ids))
    eq = sum(1 for i in range(n) if base_ids[i] == pat_ids[i])
    div = _first_diverge(base_ids, pat_ids)
    print(f"\nToken-id match: {eq}/{n} = {eq/n:.3f}")
    print(f"First divergence at step: {div}")

    if pat_ctr.n_calls < base_ctr.n_calls:
        print("\nCompute removal: boundary_predictor_module.forward calls reduced.")
    else:
        print("\nNote: boundary_predictor_module.forward call count did not reduce (may be used elsewhere).")