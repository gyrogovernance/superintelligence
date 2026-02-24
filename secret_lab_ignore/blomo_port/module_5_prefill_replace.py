"""
Module 5 (Porting Suite): Prefill boundary replacement + equivalence checks + 512-token quality

This module:
- Loads boundary_adaptor.npz
- Compares Bolmo prefill boundary probabilities vs adaptor probabilities on prompts
- Compares patch stats under:
    - deterministic threshold (p > 0.5)
    - sampled boundaries using shared random U (so mismatch probability is meaningful)
- Runs 512+ token generation in 4 modes:
    A) Baseline (Bolmo prefill boundary sampling)
    B) Patched  (Adaptor prefill boundary sampling)
    C) Baseline deterministic (Bolmo probs, mask = p>0.5)
    D) Patched deterministic  (Adaptor probs, mask = p>0.5)
- Prints previews + token-id match rates + first divergence positions.

Goal: “solid run” proving replacement is valid and showing what causes divergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from common import PROJECT_ROOT, bolmo_reset_local_caches, token_to_byte_and_fused
from module_0_baseline import baseline_generate


# -----------------------------------------------------------------------------
# Adaptor loading
# -----------------------------------------------------------------------------

def load_boundary_adaptor(path: Path | str) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)

    def _decode_bytes(x: np.ndarray) -> str:
        if x.ndim == 0:
            b = x.item()
            if isinstance(b, bytes):
                return b.rstrip(b"\x00").decode("utf-8", errors="ignore")
        return ""

    return {
        "grand_mean": float(data["grand_mean"]),
        "row_effects": data["row_effects"].astype(np.float32, copy=False),
        "col_effects": data["col_effects"].astype(np.float32, copy=False),
        "u": data["u"].astype(np.uint8, copy=False),         # ranked
        "v": data["v"].astype(np.uint8, copy=False),
        "coeffs": data["coeffs"].astype(np.float32, copy=False),
        "eps": float(data["eps"]) if "eps" in data.files else 1e-6,
        "model_key": _decode_bytes(data["model_key"]) if "model_key" in data.files else "",
    }


# -----------------------------------------------------------------------------
# Metrics helpers
# -----------------------------------------------------------------------------

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64, copy=False)
    y_pred = y_pred.astype(np.float64, copy=False)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-12
    return float(np.sum(x * y) / denom)


def _first_diverge(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


@dataclass
class PatchStats:
    n_bytes: int
    n_pairs: int
    boundary_rate: float
    mean_bytes_per_patch: float
    median_bytes_per_patch: float
    max_patch: int


def _patch_lengths_from_boundary_mask(mask: np.ndarray) -> list[int]:
    """
    mask length = n_pairs. boundary after byte t if mask[t] is True.
    """
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size == 0:
        return []
    lengths: list[int] = []
    cur = 1
    for is_bnd in mask:
        if is_bnd:
            lengths.append(cur)
            cur = 1
        else:
            cur += 1
    lengths.append(cur)
    return lengths


def _patch_stats(mask: np.ndarray) -> PatchStats:
    mask = np.asarray(mask, dtype=bool).ravel()
    n_pairs = int(mask.size)
    n_bytes = int(n_pairs + 1) if n_pairs > 0 else 0
    lengths = _patch_lengths_from_boundary_mask(mask)
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


# -----------------------------------------------------------------------------
# Bolmo prefill probability extractor (true Bolmo)
# -----------------------------------------------------------------------------

@torch.inference_mode()
def bolmo_prefill_boundary_probs(model: Any, input_ids: torch.Tensor) -> np.ndarray:
    """
    Returns boundary probabilities for a single sequence [1, L].
    Uses local_encoder boundary_logprobs => exp to get probs.
    Output length is L-1 (gaps).
    """
    device = input_ids.device
    B, L = input_ids.shape
    assert B == 1

    expand = getattr(model.model.local_encoder, "add_expanded_embeddings", False)
    if expand:
        expanded_list = [model.model.tokenizer.expand_byte_ids(input_ids[0].tolist())]
        expanded = torch.tensor(expanded_list, device=device, dtype=torch.long)
    else:
        expanded = None

    seq_start = torch.zeros(1, dtype=torch.long, device=device)

    model.model.local_encoder.free_inference_cache()
    _, _, boundary_logprobs, _ = model.model.local_encoder(
        input_ids,
        expanded_input_ids=expanded,
        sequence_start_indices=seq_start,
        boundary_state=None,
        pad_state=None,
    )

    blp = boundary_logprobs[0].detach().cpu().float().numpy()  # [L-1]
    p = np.exp(blp).astype(np.float32)
    return p


# -----------------------------------------------------------------------------
# Adaptor probability for adjacent byte pairs
# -----------------------------------------------------------------------------

_POP8 = np.array([int(i).bit_count() for i in range(256)], dtype=np.int32)

def _predict_residual_walsh_bytes(
    b_cur: np.ndarray,
    b_next: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    b_cur = np.asarray(b_cur, dtype=np.uint8).ravel()
    b_next = np.asarray(b_next, dtype=np.uint8).ravel()
    x1 = (b_cur ^ 0xAA).astype(np.uint8)
    x2 = (b_next ^ 0xAA).astype(np.uint8)

    out = np.zeros(b_cur.size, dtype=np.float64)
    K = int(coeffs.size)
    for k in range(K):
        uk = int(u[k])
        vk = int(v[k])
        c = float(coeffs[k])
        parity = _POP8[x1 & uk] + _POP8[x2 & vk]
        sign = 1.0 - 2.0 * ((parity & 1).astype(np.float64))
        out += c * sign

    out /= (256.0 * 256.0)
    return out.astype(np.float32)


def adaptor_boundary_probs_for_adjacent_pairs(
    adaptor: dict[str, Any],
    b_cur: np.ndarray,
    b_next: np.ndarray,
    K: int,
) -> np.ndarray:
    u = adaptor["u"][:K]
    v = adaptor["v"][:K]
    coeffs = adaptor["coeffs"][:K]

    add = adaptor["grand_mean"] + adaptor["row_effects"][b_cur] + adaptor["col_effects"][b_next]
    res = _predict_residual_walsh_bytes(b_cur, b_next, u, v, coeffs)
    logit = add + res

    logit64 = logit.astype(np.float64)
    logit64 = np.clip(logit64, -50.0, 50.0)
    p = np.exp(logit64) / (1.0 + np.exp(logit64))
    return p.astype(np.float32)


# -----------------------------------------------------------------------------
# Prefill alignment helper: extract byte-byte gaps from a prompt
# -----------------------------------------------------------------------------

def _byte_byte_gaps_from_prompt(tokenizer: Any, input_ids_1d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      cur_bytes [N], next_bytes [N], indices_t [N] where t refers to the gap after token t
    Only includes positions where token[t] and token[t+1] are both bytes.
    """
    ids = input_ids_1d.astype(np.int64)
    offset = int(getattr(tokenizer, "offset", 4))

    byte_at_pos = np.full(ids.shape[0], -1, dtype=np.int32)
    for t, tid in enumerate(ids.tolist()):
        b, _ = token_to_byte_and_fused(tid, offset)
        if b is not None:
            byte_at_pos[t] = int(b)

    cur_bytes: list[int] = []
    next_bytes: list[int] = []
    idxs: list[int] = []

    for t in range(ids.shape[0] - 1):
        b0 = int(byte_at_pos[t])
        b1 = int(byte_at_pos[t + 1])
        if b0 >= 0 and b1 >= 0:
            cur_bytes.append(b0)
            next_bytes.append(b1)
            idxs.append(t)

    return (
        np.array(cur_bytes, dtype=np.uint8),
        np.array(next_bytes, dtype=np.uint8),
        np.array(idxs, dtype=np.int32),
    )


# -----------------------------------------------------------------------------
# Monkeypatches
# -----------------------------------------------------------------------------

@dataclass
class BoundaryPatchMode:
    """How to turn probabilities into a boundary_mask."""
    mode: str  # "sample" or "threshold"
    threshold: float = 0.5


class BoundaryPredictorAdaptorPatch:
    """
    Patches boundary_predictor_module.forward so local_encoder uses adaptor probabilities.
    Returns (boundary_logprobs, boundary_mask) padded to length L (lookahead).
    """

    def __init__(self, model: Any, tokenizer: Any, adaptor: dict[str, Any], K: int, mode: BoundaryPatchMode):
        self.model = model
        self.tokenizer = tokenizer
        self.adaptor = adaptor
        self.K = K
        self.mode = mode
        self._orig_le_forward = None
        self._orig_bp_forward = None

    def __enter__(self):
        le = self.model.model.local_encoder
        bp = le.boundary_predictor_module

        self._orig_le_forward = le.forward
        self._orig_bp_forward = bp.forward

        offset = int(getattr(self.tokenizer, "offset", 4))
        eps = float(self.adaptor.get("eps", 1e-6))

        def le_forward_wrapped(input_ids, *args, **kwargs):
            le._last_input_ids_for_patch = input_ids
            le._last_offset_for_patch = offset
            assert self._orig_le_forward is not None
            return self._orig_le_forward(input_ids, *args, **kwargs)

        def bp_forward_wrapped(hidden_states, *args, **kwargs):
            input_ids = getattr(le, "_last_input_ids_for_patch", None)
            if input_ids is None:
                assert self._orig_bp_forward is not None
                return self._orig_bp_forward(hidden_states, *args, **kwargs)

            device = hidden_states.device
            ids = input_ids.detach().cpu().numpy().astype(np.int64)
            B, L = ids.shape

            lookahead = int(getattr(bp, "boundary_predictor_lookahead", 1))
            if L <= 0:
                empty_lp = torch.empty((B, 0), device=device, dtype=torch.float32)
                empty_m = torch.empty((B, 0), device=device, dtype=torch.bool)
                return empty_lp, empty_m

            if L < 2:
                # Must return shape (B, L) for pooling compatibility
                lp = torch.full((B, L), -100_000.0, device=device, dtype=torch.float32)
                m = torch.zeros((B, L), device=device, dtype=torch.bool)
                return lp, m

            cur = ids[:, :-1]
            nxt = ids[:, 1:]
            valid = (
                (cur >= offset) & (cur < offset + 256) &
                (nxt >= offset) & (nxt < offset + 256)
            )

            # default: force boundary prob 1.0 for invalid gaps
            p = np.ones((B, L - 1), dtype=np.float32)

            if np.any(valid):
                b_cur = (cur[valid] - offset).astype(np.uint8)
                b_nxt = (nxt[valid] - offset).astype(np.uint8)
                p_valid = adaptor_boundary_probs_for_adjacent_pairs(self.adaptor, b_cur, b_nxt, K=self.K)
                p[valid] = p_valid

            # boundary_logprobs and mask are over gaps (L-1), then right-pad to L
            p_t = torch.from_numpy(p).to(device=device, dtype=torch.float32)

            if self.mode.mode == "sample":
                boundary_mask = (torch.rand_like(p_t) < p_t)
            else:
                boundary_mask = (p_t > float(self.mode.threshold))

            p_t = torch.clamp(p_t, min=eps, max=1.0 - eps)
            boundary_logprobs = torch.log(p_t)

            # right-pad by lookahead to length L
            if lookahead > 0:
                pad_lp = torch.full((B, lookahead), -100_000.0, device=device, dtype=torch.float32)
                pad_m = torch.zeros((B, lookahead), device=device, dtype=torch.bool)
                boundary_logprobs = torch.cat([boundary_logprobs, pad_lp], dim=1)
                boundary_mask = torch.cat([boundary_mask, pad_m], dim=1)

            return boundary_logprobs, boundary_mask

        le.forward = le_forward_wrapped
        bp.forward = bp_forward_wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        le = self.model.model.local_encoder
        bp = le.boundary_predictor_module
        if self._orig_le_forward is not None:
            le.forward = self._orig_le_forward
        if self._orig_bp_forward is not None:
            bp.forward = self._orig_bp_forward
        for attr in ("_last_input_ids_for_patch", "_last_offset_for_patch"):
            if hasattr(le, attr):
                delattr(le, attr)


class BoundaryPredictorDeterminizeBolmoPatch:
    """
    Baseline helper: keep Bolmo boundary_logprobs but force deterministic boundary_mask=(p>threshold).
    This removes boundary sampling noise in baseline.
    """

    def __init__(self, model: Any, threshold: float = 0.5):
        self.model = model
        self.threshold = float(threshold)
        self._orig_forward = None

    def __enter__(self):
        le = self.model.model.local_encoder
        bp = le.boundary_predictor_module

        self._orig_forward = bp.forward

        def forward_wrapped(hidden_states, *args, **kwargs):
            assert self._orig_forward is not None
            boundary_logprobs, boundary_mask = self._orig_forward(hidden_states, *args, **kwargs)
            # boundary_logprobs expected shape (B, L) already
            p = torch.exp(boundary_logprobs)
            boundary_mask2 = (p > self.threshold)
            return boundary_logprobs, boundary_mask2

        bp.forward = forward_wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        bp = self.model.model.local_encoder.boundary_predictor_module
        if self._orig_forward is not None:
            bp.forward = self._orig_forward


# -----------------------------------------------------------------------------
# Module 5 runner
# -----------------------------------------------------------------------------

def run_module_5_porting_suite(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    K: int = 16384,
    max_new_tokens: int = 512,
    preview_chars: int = 1200,
) -> None:
    adaptor_path = PROJECT_ROOT / "data" / "cache" / "blomo_port" / "analysis" / "boundary_adaptor.npz"
    if not adaptor_path.exists():
        raise RuntimeError(
            f"boundary_adaptor.npz not found at {adaptor_path}\n"
            "Run module 2 first:\n"
            "  python secret_lab_ignore/blomo_port/lab.py --module 2"
        )
    adaptor = load_boundary_adaptor(adaptor_path)

    device = next(model.parameters()).device
    print(f"\n[Module 5] Using K={K}, max_new_tokens={max_new_tokens}")
    print(f"Adaptor key: {adaptor.get('model_key','')!r}")

    # We run one “full 512 token generation” prompt (first prompt),
    # and do analysis for all prompts.
    gen_prompt = prompts[0] if prompts else "Language modeling is "

    for prompt in prompts:
        print(f"\n=== Prefill analysis: {prompt!r} ===")
        bolmo_reset_local_caches(model)

        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        ids_1d = input_ids[0].detach().cpu().numpy()

        # Bolmo probabilities for all gaps
        p_b = bolmo_prefill_boundary_probs(model, input_ids)  # length ~L-1

        # Extract byte-byte gaps
        cur, nxt, idxs = _byte_byte_gaps_from_prompt(tokenizer, ids_1d)
        if cur.size == 0:
            print("  No byte-byte gaps found; skipping.")
            continue

        # Align bolmo gaps to byte-byte subset
        p_b_sub = np.array([p_b[t] for t in idxs.tolist()], dtype=np.float32)
        p_a_sub = adaptor_boundary_probs_for_adjacent_pairs(adaptor, cur, nxt, K=K)

        r2 = _r2(p_b_sub, p_a_sub)
        pr = _pearson(p_b_sub, p_a_sub)
        mae = float(np.mean(np.abs(p_b_sub - p_a_sub)))
        agree = float(np.mean((p_b_sub > 0.5) == (p_a_sub > 0.5)))

        # Expected mismatch if sampled with SAME uniform u: P(mismatch)=|p_b - p_a|
        exp_mis = float(np.sum(np.abs(p_b_sub - p_a_sub)))
        exp_rate = float(np.mean(np.abs(p_b_sub - p_a_sub)))

        print(
            f"  compared={cur.size}/{len(p_b)}  R2={r2:.4f}  pearson={pr:.4f}  "
            f"MAE={mae:.4f}  agree@0.5={agree:.3f}"
        )
        print(
            f"  expected sample mismatches ≈ {exp_mis:.2f} "
            f"({100*exp_rate:.2f}% per position)"
        )

        # Patch stats: deterministic threshold mode
        mask_b_det = (p_b_sub > 0.5)
        mask_a_det = (p_a_sub > 0.5)
        print("  Patch stats (deterministic threshold=0.5):")
        print(f"    bolmo:   {_patch_stats(mask_b_det)}")
        print(f"    adaptor: {_patch_stats(mask_a_det)}")

        # Patch stats: sampled with shared RNG U (fixed seed)
        rng = np.random.default_rng(123)
        u = rng.random(p_b_sub.size).astype(np.float32)
        mask_b_s = (u < p_b_sub)
        mask_a_s = (u < p_a_sub)
        mismatch = float(np.mean(mask_b_s != mask_a_s))
        print("  Patch stats (sampled with shared U seed=123):")
        print(f"    bolmo:   {_patch_stats(mask_b_s)}")
        print(f"    adaptor: {_patch_stats(mask_a_s)}")
        print(f"    mismatch rate: {mismatch:.3f}")

    # -------------------------------------------------------------------------
    # Generation suite on gen_prompt
    # -------------------------------------------------------------------------
    print(f"\n=== 512-token quality suite on prompt: {gen_prompt!r} ===")

    # A) Baseline sampled boundaries
    print("\n[A] Baseline (Bolmo, sampled boundaries)")
    base_text, base_ids, base_t = baseline_generate(model, tokenizer, gen_prompt, max_new_tokens=max_new_tokens)
    print(f"  tokens={len(base_ids)} time={base_t:.2f}s")
    print("  preview:")
    print(base_text[:preview_chars].replace("\n", "\\n"))

    # B) Patched sampled boundaries (adaptor)
    print(f"\n[B] Patched (Adaptor, sampled boundaries) K={K}")
    with BoundaryPredictorAdaptorPatch(model, tokenizer, adaptor, K=K, mode=BoundaryPatchMode("sample")):
        pat_text, pat_ids, pat_t = baseline_generate(model, tokenizer, gen_prompt, max_new_tokens=max_new_tokens)
    print(f"  tokens={len(pat_ids)} time={pat_t:.2f}s")
    print("  preview:")
    print(pat_text[:preview_chars].replace("\n", "\\n"))

    # Compare A vs B
    n = min(len(base_ids), len(pat_ids))
    if n > 0:
        eq = sum(1 for i in range(n) if base_ids[i] == pat_ids[i])
        div = _first_diverge(base_ids, pat_ids)
        print(f"\n  A vs B token-id match: {eq}/{n} = {eq/n:.3f}")
        print(f"  first divergence step: {div}")

    # C) Baseline deterministic boundaries (Bolmo probs, mask=p>0.5)
    print("\n[C] Baseline deterministic boundaries (Bolmo mask=p>0.5)")
    with BoundaryPredictorDeterminizeBolmoPatch(model, threshold=0.5):
        base_det_text, base_det_ids, base_det_t = baseline_generate(
            model, tokenizer, gen_prompt, max_new_tokens=max_new_tokens
        )
    print(f"  tokens={len(base_det_ids)} time={base_det_t:.2f}s")
    print("  preview:")
    print(base_det_text[:preview_chars].replace("\n", "\\n"))

    # D) Patched deterministic boundaries (Adaptor probs, mask=p>0.5)
    print(f"\n[D] Patched deterministic boundaries (Adaptor mask=p>0.5) K={K}")
    with BoundaryPredictorAdaptorPatch(model, tokenizer, adaptor, K=K, mode=BoundaryPatchMode("threshold", 0.5)):
        pat_det_text, pat_det_ids, pat_det_t = baseline_generate(
            model, tokenizer, gen_prompt, max_new_tokens=max_new_tokens
        )
    print(f"  tokens={len(pat_det_ids)} time={pat_det_t:.2f}s")
    print("  preview:")
    print(pat_det_text[:preview_chars].replace("\n", "\\n"))

    # Compare C vs D (this is the “substantial finish”)
    n2 = min(len(base_det_ids), len(pat_det_ids))
    if n2 > 0:
        eq2 = sum(1 for i in range(n2) if base_det_ids[i] == pat_det_ids[i])
        div2 = _first_diverge(base_det_ids, pat_det_ids)
        print(f"\n  C vs D token-id match: {eq2}/{n2} = {eq2/n2:.3f}")
        print(f"  first divergence step: {div2}")

        # Also show C vs A drift (effect of determinizing boundaries)
        n3 = min(len(base_ids), len(base_det_ids))
        eq3 = sum(1 for i in range(n3) if base_ids[i] == base_det_ids[i])
        div3 = _first_diverge(base_ids, base_det_ids)
        print(f"\n  A vs C token-id match (sampling vs deterministic boundaries): {eq3}/{n3} = {eq3/n3:.3f}")
        print(f"  first divergence step: {div3}")