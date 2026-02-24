"""
Module 3 (Lab): Prefill Boundary Evaluation
Compare Bolmo's *actual* prefill boundary probabilities on real prompts
against the byte-pair Walsh adaptor (boundary_adaptor.npz).

This answers: does the adaptor learned from the 256×256 surface
(BOS+b1+b2) generalize to longer sequences?

If yes -> next step is boundary replacement.
If no  -> extend adaptor to be kernel-state-conditioned (still finite).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import PROJECT_ROOT, bolmo_reset_local_caches, token_to_byte_and_fused


@dataclass
class PrefillEvalResult:
    n_positions: int
    n_compared: int
    r2: float
    pearson: float
    mean_abs_err: float
    agree_at_0p5: float


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


def load_boundary_adaptor(path: Path | str) -> dict[str, Any]:
    data = np.load(path)
    def _decode_bytes(arr: np.ndarray) -> str:
        if arr.ndim == 0:
            b = arr.item()
        else:
            b = arr.tobytes()
        if isinstance(b, bytes):
            return b.rstrip(b"\x00").decode("utf-8", errors="ignore")
        return str(b)

    return {
        "grand_mean": float(data["grand_mean"]),
        "row_effects": data["row_effects"].astype(np.float32, copy=False),
        "col_effects": data["col_effects"].astype(np.float32, copy=False),
        "u": data["u"].astype(np.uint8, copy=False),
        "v": data["v"].astype(np.uint8, copy=False),
        "coeffs": data["coeffs"].astype(np.float32, copy=False),
        "model_key": _decode_bytes(data["model_key"]) if "model_key" in data.files else "",
    }


_POP8 = np.array([int(i).bit_count() for i in range(256)], dtype=np.int32)


def predict_residual_walsh_pairwise(
    b_cur: np.ndarray,
    b_next: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """
    Predict residual logit for each adjacent byte pair (b_t, b_{t+1}).
    Introns: x = byte ^ 0xAA
    residual = (1/256^2) * sum_k coeff_k * (-1)^(<u_k,x_t> + <v_k,x_{t+1}>)
    """
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


def adaptor_boundary_prob_for_adjacent_pairs(
    adaptor: dict[str, Any],
    b_cur: np.ndarray,
    b_next: np.ndarray,
    K: int,
) -> np.ndarray:
    """
    Predict boundary prob after each byte in b_cur, using b_next as lookahead.
    logit = grand_mean + row[b_cur] + col[b_next] + residual_walsh(b_cur,b_next)
    prob  = sigmoid(logit)
    """
    u = adaptor["u"][:K]
    v = adaptor["v"][:K]
    coeffs = adaptor["coeffs"][:K]

    add = adaptor["grand_mean"] + adaptor["row_effects"][b_cur] + adaptor["col_effects"][b_next]
    res = predict_residual_walsh_pairwise(b_cur, b_next, u, v, coeffs)
    logit = add + res

    logit64 = logit.astype(np.float64)
    logit64 = np.clip(logit64, -50.0, 50.0)
    p = np.exp(logit64) / (1.0 + np.exp(logit64))
    return p.astype(np.float32)


@torch.inference_mode()
def bolmo_prefill_boundary_probs(
    model: Any,
    input_ids: torch.Tensor,
) -> np.ndarray:
    """
    Returns Bolmo boundary probabilities for a single sequence (batch=1).
    boundary_logprobs is log(prob_boundary) per position (length L-lookahead).
    We output probs as float32.
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

    # boundary_logprobs: shape [1, L-lookahead] or [1, L-1]
    blp = boundary_logprobs[0].detach().cpu().float().numpy()
    probs = np.exp(blp).astype(np.float32)
    return probs


def evaluate_prompt_prefill(
    model: Any,
    tokenizer: Any,
    adaptor_path: Path,
    prompt: str,
    K: int = 16384,
) -> PrefillEvalResult:
    adaptor = load_boundary_adaptor(adaptor_path)

    bolmo_reset_local_caches(model)
    device = next(model.parameters()).device

    # Use the same tokenization style as generation prefill
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)  # [1, L]
    ids = input_ids[0].detach().cpu().numpy().astype(np.int64)

    # Bolmo probabilities per boundary position t (after token t)
    bolmo_p = bolmo_prefill_boundary_probs(model, input_ids)  # length ~L-1

    # Build byte-aligned arrays for positions where token is a byte
    offset = int(getattr(tokenizer, "offset", 4))

    # For each token position, determine if it is a byte token and its byte value
    byte_at_pos = np.full(ids.shape[0], -1, dtype=np.int32)
    for t, tid in enumerate(ids.tolist()):
        b, fused = token_to_byte_and_fused(tid, offset)
        if b is not None:
            byte_at_pos[t] = int(b)

    # Compare only positions t where both token t and t+1 are bytes
    # and where bolmo_p has that t index
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

    if len(cur_bytes) < 4:
        return PrefillEvalResult(
            n_positions=int(len(bolmo_p)),
            n_compared=int(len(cur_bytes)),
            r2=0.0,
            pearson=0.0,
            mean_abs_err=0.0,
            agree_at_0p5=0.0,
        )

    cur = np.array(cur_bytes, dtype=np.uint8)
    nxt = np.array(next_bytes, dtype=np.uint8)
    bolmo_arr = np.array(bolmo_vals, dtype=np.float32)

    adaptor_arr = adaptor_boundary_prob_for_adjacent_pairs(adaptor, cur, nxt, K=K)

    r2 = _r2(bolmo_arr, adaptor_arr)
    pearson = _pearson(bolmo_arr, adaptor_arr)
    mae = float(np.mean(np.abs(bolmo_arr - adaptor_arr)))
    agree = float(np.mean((bolmo_arr > 0.5) == (adaptor_arr > 0.5)))

    return PrefillEvalResult(
        n_positions=int(len(bolmo_p)),
        n_compared=int(len(cur)),
        r2=float(r2),
        pearson=float(pearson),
        mean_abs_err=float(mae),
        agree_at_0p5=float(agree),
    )