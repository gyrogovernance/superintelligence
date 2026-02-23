"""
Module 2 (final): Bolmo boundary -> Walsh adaptor builder.

Purpose:
- Obtain Bolmo boundary probabilities on the full 256×256 byte-pair grid (BOS+b1+b2)
- Convert to logit surface L[b1,b2]
- Decompose L into:
    L = grand_mean + row_effects[b1] + col_effects[b2] + residual[b1,b2]
- Express residual in intron-indexed 2D Walsh basis and export a ranked spectrum.

Output:
- data/cache/blomo_port/analysis/bolmo_adaptor.npz
  Contains:
    grand_mean, row_effects, col_effects,
    u[65536], v[65536], coeffs[65536]  (ranked by |coeff|)

Runtime usage:
- slice K coefficients to approximate Bolmo boundary logic at chosen fidelity.

Notes:
- No hidden-state caching; we only need boundary probabilities.
- Cache stores only scores.npy (+meta.json), keyed by model path/config.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from common import PROJECT_ROOT, bolmo_reset_local_caches


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "blomo_port"
ANALYSIS_DIR = CACHE_DIR / "analysis"


@dataclass(frozen=True)
class CacheMeta:
    model_key: str
    model_dir: str
    tokenizer_id: str
    boundary_predictor_lookahead: int
    add_expanded_embeddings: bool
    d_model: int


def _cache_key(model: Any, model_dir: Path) -> str:
    lookahead = getattr(
        model.model.local_encoder.boundary_predictor_module,
        "boundary_predictor_lookahead",
        1,
    )
    expand = getattr(model.model.local_encoder, "add_expanded_embeddings", False)
    tokenizer_id = getattr(
        getattr(model.model, "tokenizer_config", None),
        "original_identifier",
        "",
    )
    key_str = f"{model_dir.resolve()}_{lookahead}_{expand}_{tokenizer_id}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _cache_subdir(model: Any, model_dir: Path) -> Path:
    return CACHE_DIR / _cache_key(model, model_dir)


def _load_scores_cache(cache_subdir: Path) -> Optional[np.ndarray]:
    p = cache_subdir / "scores.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


def _save_scores_cache(cache_subdir: Path, scores: np.ndarray, meta: CacheMeta) -> None:
    cache_subdir.mkdir(parents=True, exist_ok=True)
    np.save(cache_subdir / "scores.npy", scores.astype(np.float32))
    with open(cache_subdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta.__dict__, f, indent=2)


# ---------------------------------------------------------------------
# Extraction (scores only)
# ---------------------------------------------------------------------

@torch.inference_mode()
def extract_bolmo_boundary_scores_256x256(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    chunk_size: int = 16,
) -> np.ndarray:
    """
    Extract boundary probabilities scores[b1,b2] for all b1,b2 in 0..255.
    Uses the local_encoder boundary_logprobs (Bolmo prefill boundary predictor).
    """
    offset = int(getattr(tokenizer, "offset", 4))
    bos_id = int(tokenizer.bos_token_id)
    expand = getattr(model.model.local_encoder, "add_expanded_embeddings", False)

    scores = np.zeros((256, 256), dtype=np.float32)
    t0 = time.perf_counter()

    for b1_start in range(0, 256, chunk_size):
        b1_end = min(b1_start + chunk_size, 256)
        for b2_start in range(0, 256, chunk_size):
            b2_end = min(b2_start + chunk_size, 256)

            triples: list[list[int]] = []
            for b1 in range(b1_start, b1_end):
                for b2 in range(b2_start, b2_end):
                    triples.append([bos_id, b1 + offset, b2 + offset])

            batch = torch.tensor(triples, device=device, dtype=torch.long)
            B = batch.shape[0]

            if expand:
                expanded_list = [
                    model.model.tokenizer.expand_byte_ids(seq.tolist())
                    for seq in batch
                ]
                expanded = torch.tensor(expanded_list, device=device, dtype=torch.long)
            else:
                expanded = None

            seq_start = torch.zeros(B, dtype=torch.long, device=device)

            model.model.local_encoder.free_inference_cache()

            # We only need bnd_logprobs; ignore h_byte output
            _, _, bnd_logprobs, _ = model.model.local_encoder(
                batch,
                expanded_input_ids=expanded,
                sequence_start_indices=seq_start,
                boundary_state=None,
                pad_state=None,
            )

            bnd_probs = torch.exp(bnd_logprobs[:, 1]).detach().cpu().numpy()

            idx = 0
            for b1 in range(b1_start, b1_end):
                for b2 in range(b2_start, b2_end):
                    scores[b1, b2] = float(bnd_probs[idx])
                    idx += 1

        elapsed = time.perf_counter() - t0
        print(f"  b1=[{b1_start}..{b1_end}] done ({elapsed:.1f}s)")

    elapsed_total = time.perf_counter() - t0
    print(f"Extraction complete in {elapsed_total:.1f}s")
    return scores


def get_scores_cached_or_extract(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    model_dir: Path,
    chunk_size: int = 16,
) -> tuple[np.ndarray, CacheMeta]:
    subdir = _cache_subdir(model, model_dir)
    cached = _load_scores_cache(subdir)
    if cached is not None:
        meta_path = subdir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cm = CacheMeta(**meta)
        else:
            cm = CacheMeta(
                model_key=subdir.name,
                model_dir=str(model_dir.resolve()),
                tokenizer_id="",
                boundary_predictor_lookahead=1,
                add_expanded_embeddings=bool(
                    getattr(model.model.local_encoder, "add_expanded_embeddings", False)
                ),
                d_model=int(model.config.hidden_size),
            )
        print(f"Loaded boundary scores from cache ({subdir})")
        return cached, cm

    print("Extracting boundary scores (no hidden states cached)...")
    bolmo_reset_local_caches(model)
    scores = extract_bolmo_boundary_scores_256x256(
        model, tokenizer, device, chunk_size=chunk_size
    )

    tokenizer_id = getattr(
        getattr(model.model, "tokenizer_config", None),
        "original_identifier",
        "",
    )
    lookahead = getattr(
        model.model.local_encoder.boundary_predictor_module,
        "boundary_predictor_lookahead",
        1,
    )
    expand = getattr(model.model.local_encoder, "add_expanded_embeddings", False)

    cm = CacheMeta(
        model_key=subdir.name,
        model_dir=str(model_dir.resolve()),
        tokenizer_id=str(tokenizer_id),
        boundary_predictor_lookahead=int(lookahead),
        add_expanded_embeddings=bool(expand),
        d_model=int(model.config.hidden_size),
    )
    _save_scores_cache(subdir, scores, cm)
    print(f"Cached to {subdir}")
    return scores, cm


# ---------------------------------------------------------------------
# Walsh tools
# ---------------------------------------------------------------------

def _r2_scalar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64, copy=False)
    y_pred = y_pred.astype(np.float64, copy=False)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def _wht_axis(a: np.ndarray, axis: int) -> np.ndarray:
    """Unnormalized Walsh–Hadamard transform along axis; requires length 256."""
    x = np.swapaxes(a, axis, -1).copy()
    n = x.shape[-1]
    assert n == 256
    x = x.reshape(-1, n)

    h = 1
    while h < n:
        x = x.reshape(-1, n // (2 * h), 2 * h)
        u = x[..., :h].copy()
        v = x[..., h:2*h].copy()
        x[..., :h] = u + v
        x[..., h:2*h] = u - v
        x = x.reshape(-1, n)
        h *= 2

    x = x.reshape(*np.swapaxes(a, axis, -1).shape)
    x = np.swapaxes(x, -1, axis)
    return x


def wht2_256(mat: np.ndarray) -> np.ndarray:
    """Unnormalized 2D WHT on 256×256."""
    out = _wht_axis(mat, axis=0)
    out = _wht_axis(out, axis=1)
    return out


# ---------------------------------------------------------------------
# Adaptor builder + runtime evaluator
# ---------------------------------------------------------------------

_POP8 = np.array([int(i).bit_count() for i in range(256)], dtype=np.int32)


def predict_residual_walsh_bytes(
    b1: np.ndarray,
    b2: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """
    residual(x1,x2) = (1/256^2) * sum_k coeffs[k] * (-1)^(<u_k,x1> + <v_k,x2>)
    where x = byte ^ 0xAA and parity is popcount(bitwise_and).
    """
    b1 = np.asarray(b1, dtype=np.uint8).ravel()
    b2 = np.asarray(b2, dtype=np.uint8).ravel()
    x1 = (b1 ^ 0xAA).astype(np.uint8)
    x2 = (b2 ^ 0xAA).astype(np.uint8)

    out = np.zeros(b1.size, dtype=np.float64)
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


def build_bolmo_adaptor(
    scores: np.ndarray,
    meta: CacheMeta,
    out_path: Path,
    atlas_version: str = "2.2",
    eval_Ks: tuple[int, ...] = (2048, 4096, 8192, 16384, 32768),
) -> dict[str, Any]:
    """
    Build bolmo_adaptor.npz:
      - additive (grand_mean, row_effects, col_effects)
      - ranked residual Walsh spectrum (u,v,coeffs)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eps = 1e-6
    scores_clipped = np.clip(scores, eps, 1.0 - eps)
    logits = np.log(scores_clipped / (1.0 - scores_clipped)).astype(np.float32)

    grand_mean = float(np.mean(logits))
    row_effects = logits.mean(axis=1) - grand_mean
    col_effects = logits.mean(axis=0) - grand_mean
    additive = grand_mean + row_effects[:, None] + col_effects[None, :]
    residual = logits - additive

    # intron reindex: x = byte ^ 0xAA
    perm = np.arange(256, dtype=np.int64) ^ 0xAA
    Rx = residual[perm][:, perm].astype(np.float32, copy=True)

    # Walsh spectrum on residual in intron coords
    F_res = wht2_256(Rx)

    # Rank by magnitude
    absF = np.abs(F_res).reshape(-1)
    order = np.argsort(-absF)

    u_all = (order // 256).astype(np.uint8)
    v_all = (order % 256).astype(np.uint8)
    c_all = F_res.reshape(-1)[order].astype(np.float32)

    np.savez(
        out_path,
        adaptor_version=np.string_("1.0"),
        atlas_version=np.string_(atlas_version),
        model_key=np.string_(meta.model_key),
        model_dir=np.string_(meta.model_dir),
        tokenizer_id=np.string_(meta.tokenizer_id),
        boundary_predictor_lookahead=np.int32(meta.boundary_predictor_lookahead),
        add_expanded_embeddings=np.uint8(1 if meta.add_expanded_embeddings else 0),
        d_model=np.int32(meta.d_model),
        eps=np.float32(eps),
        grand_mean=np.float32(grand_mean),
        row_effects=row_effects.astype(np.float32),
        col_effects=col_effects.astype(np.float32),
        u=u_all,
        v=v_all,
        coeffs=c_all,
    )

    # Evaluate slice fidelities
    flat_logits = logits.flatten().astype(np.float64)
    flat_resid = residual.flatten().astype(np.float64)
    frac_residual = float(np.var(residual) / (np.var(logits) + 1e-12))

    b1_all = np.repeat(np.arange(256, dtype=np.uint8), 256)
    b2_all = np.tile(np.arange(256, dtype=np.uint8), 256)

    report: dict[str, Any] = {
        "out_path": str(out_path),
        "frac_residual": frac_residual,
        "K_report": {},
    }

    for K in eval_Ks:
        uK = u_all[:K]
        vK = v_all[:K]
        cK = c_all[:K]
        pred_res = predict_residual_walsh_bytes(b1_all, b2_all, uK, vK, cK).astype(np.float64)
        r2_res = _r2_scalar(flat_resid, pred_res)
        r2_full = _r2_scalar(flat_logits, (additive.flatten().astype(np.float64) + pred_res))
        report["K_report"][int(K)] = {
            "residual_r2": float(r2_res),
            "full_logit_r2": float(r2_full),
        }

    return report


def analyze_scores_basic(scores: np.ndarray) -> dict[str, float]:
    flat = scores.reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "median": float(np.median(flat)),
        "q25": float(np.quantile(flat, 0.25)),
        "q75": float(np.quantile(flat, 0.75)),
        "rate_gt_0p5": float(np.mean(flat > 0.5)),
    }