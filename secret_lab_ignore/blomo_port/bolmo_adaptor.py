# secret_lab_ignore/blomo_port/bolmo_adaptor.py
"""
Bolmo Adaptor Builder (Lab Utility)

Purpose
-------
Extract Bolmo's 256×256 boundary probability surface (BOS + b1 + b2),
convert it to a logit surface L[b1,b2], decompose:

    L = grand_mean + row_effects[b1] + col_effects[b2] + residual[b1,b2]

Then represent residual in intron-indexed 2D Walsh basis and export a single,
sliceable adaptor artifact:

    data/cache/blomo_port/analysis/bolmo_adaptor.npz

This file contains:
  - grand_mean (float32)
  - row_effects[256] (float32)
  - col_effects[256] (float32)
  - u[65536], v[65536] (uint8): ranked Walsh frequency indices (intron coords)
  - coeffs[65536] (float32): ranked residual Walsh coefficients

You choose K at analysis time by slicing u/v/coeffs[:K].

Notes
-----
- This script caches only scores.npy (+ meta.json). No hidden-state caching.
- This is a lab-only adaptor; keep it separate from the normative kernel atlas.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from common import PROJECT_ROOT, bolmo_reset_local_caches


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "blomo_port"
ANALYSIS_DIR = CACHE_DIR / "analysis"


# -----------------------------------------------------------------------------
# Cache metadata
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoresCacheMeta:
    model_key: str
    model_dir: str
    tokenizer_id: str
    boundary_predictor_lookahead: int
    add_expanded_embeddings: bool
    d_model: int
    offset: int
    bos_id: int


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


def _load_meta(cache_subdir: Path) -> Optional[ScoresCacheMeta]:
    p = cache_subdir / "meta.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return ScoresCacheMeta(**data)


def _save_scores_cache(cache_subdir: Path, scores: np.ndarray, meta: ScoresCacheMeta) -> None:
    cache_subdir.mkdir(parents=True, exist_ok=True)
    np.save(cache_subdir / "scores.npy", scores.astype(np.float32))
    (cache_subdir / "meta.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Extraction: scores[b1,b2] only
# -----------------------------------------------------------------------------

@torch.inference_mode()
def extract_scores_256x256(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    chunk_size: int = 16,
) -> np.ndarray:
    """
    Extract boundary probabilities scores[b1,b2] for all bytes b1,b2 in 0..255.

    We send [BOS, b1+offset, b2+offset] into local_encoder and read boundary_logprobs.
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
            B = int(batch.shape[0])

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

            # We only need bnd_logprobs
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

    print(f"Extraction complete in {time.perf_counter() - t0:.1f}s")
    return scores


def get_scores_cached_or_extract(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    model_dir: Path,
    chunk_size: int = 16,
) -> tuple[np.ndarray, ScoresCacheMeta]:
    """
    Loads cache if available; otherwise extracts and writes cache.
    """
    subdir = _cache_subdir(model, model_dir)
    cached = _load_scores_cache(subdir)
    meta = _load_meta(subdir)

    if cached is not None and meta is not None:
        print(f"Loaded boundary scores from cache ({subdir})")
        return cached, meta

    print("No score cache found; extracting...")
    bolmo_reset_local_caches(model)

    scores = extract_scores_256x256(model, tokenizer, device, chunk_size=chunk_size)

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

    offset = int(getattr(tokenizer, "offset", 4))
    bos_id = int(tokenizer.bos_token_id)

    meta = ScoresCacheMeta(
        model_key=subdir.name,
        model_dir=str(model_dir.resolve()),
        tokenizer_id=str(tokenizer_id),
        boundary_predictor_lookahead=int(lookahead),
        add_expanded_embeddings=bool(expand),
        d_model=int(model.config.hidden_size),
        offset=offset,
        bos_id=bos_id,
    )

    _save_scores_cache(subdir, scores, meta)
    print(f"Cached scores to {subdir}")
    return scores, meta


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


# -----------------------------------------------------------------------------
# Walsh utilities (256×256)
# -----------------------------------------------------------------------------

def _r2_scalar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64, copy=False)
    y_pred = y_pred.astype(np.float64, copy=False)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def _wht_axis(a: np.ndarray, axis: int) -> np.ndarray:
    """Unnormalized Walsh–Hadamard transform along `axis` (length must be 256)."""
    x = np.swapaxes(a, axis, -1).copy()
    n = x.shape[-1]
    assert n == 256
    x = x.reshape(-1, n)

    h = 1
    while h < n:
        x = x.reshape(-1, n // (2 * h), 2 * h)
        u = x[..., :h].copy()
        v = x[..., h : 2 * h].copy()
        x[..., :h] = u + v
        x[..., h : 2 * h] = u - v
        x = x.reshape(-1, n)
        h *= 2

    x = x.reshape(*np.swapaxes(a, axis, -1).shape)
    x = np.swapaxes(x, -1, axis)
    return x


def wht2_256(mat: np.ndarray) -> np.ndarray:
    """Unnormalized 2D WHT: apply along rows then cols."""
    out = _wht_axis(mat, axis=0)
    out = _wht_axis(out, axis=1)
    return out


# -----------------------------------------------------------------------------
# Adaptor build + (analysis-only) evaluator
# -----------------------------------------------------------------------------

_POP8 = np.array([int(i).bit_count() for i in range(256)], dtype=np.int32)


def predict_residual_walsh(
    b1: np.ndarray,
    b2: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """
    residual(x1,x2) = (1/256^2) * sum_k coeffs[k] * (-1)^(<u_k,x1> + <v_k,x2>)
    where x = byte ^ 0xAA and <u,x> is parity(popcount(u & x)).
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
    meta: ScoresCacheMeta,
    out_path: Path,
    atlas_version: str = "2.2",
    eval_Ks: tuple[int, ...] = (2048, 4096, 8192, 16384, 32768),
) -> dict[str, Any]:
    """
    Export bolmo_adaptor.npz with:
      - additive: grand_mean, row_effects, col_effects
      - residual Walsh spectrum ranked by |coeff|: u,v,coeffs
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

    # intron indexing
    perm = np.arange(256, dtype=np.int64) ^ 0xAA
    Rx = residual[perm][:, perm].astype(np.float32, copy=True)

    # Walsh spectrum
    F_res = wht2_256(Rx)

    # Rank all coefficients by magnitude
    absF = np.abs(F_res).reshape(-1)
    order = np.argsort(-absF)

    u_all = (order // 256).astype(np.uint8)
    v_all = (order % 256).astype(np.uint8)
    c_all = F_res.reshape(-1)[order].astype(np.float32)

    np.savez(
        out_path,
        adaptor_version=np.bytes_(b"1.0"),
        atlas_version=np.bytes_(atlas_version.encode("utf-8")),
        # provenance
        model_key=np.bytes_(meta.model_key.encode("utf-8")),
        model_dir=np.bytes_(meta.model_dir.encode("utf-8")),
        tokenizer_id=np.bytes_(meta.tokenizer_id.encode("utf-8")),
        boundary_predictor_lookahead=np.int32(meta.boundary_predictor_lookahead),
        add_expanded_embeddings=np.uint8(1 if meta.add_expanded_embeddings else 0),
        d_model=np.int32(meta.d_model),
        offset=np.int32(meta.offset),
        bos_id=np.int32(meta.bos_id),
        eps=np.float32(eps),
        # additive
        grand_mean=np.float32(grand_mean),
        row_effects=row_effects.astype(np.float32),
        col_effects=col_effects.astype(np.float32),
        # ranked residual spectrum
        u=u_all,
        v=v_all,
        coeffs=c_all,
    )

    # --- analysis report ---
    flat_logits = logits.flatten().astype(np.float64)
    flat_resid = residual.flatten().astype(np.float64)

    b1_all = np.repeat(np.arange(256, dtype=np.uint8), 256)
    b2_all = np.tile(np.arange(256, dtype=np.uint8), 256)

    rep: dict[str, Any] = {
        "out_path": str(out_path),
        "frac_residual": float(np.var(residual) / (np.var(logits) + 1e-12)),
        "K_report": {},
    }

    for K in eval_Ks:
        uK = u_all[:K]
        vK = v_all[:K]
        cK = c_all[:K]
        pred_res = predict_residual_walsh(b1_all, b2_all, uK, vK, cK).astype(np.float64)
        r2_res = _r2_scalar(flat_resid, pred_res)
        r2_full = _r2_scalar(flat_logits, additive.flatten().astype(np.float64) + pred_res)
        rep["K_report"][int(K)] = {"residual_r2": float(r2_res), "full_logit_r2": float(r2_full)}

    return rep


def build_and_save_default_adaptor(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    model_dir: Path,
    chunk_size: int = 16,
) -> None:
    """
    Convenience function for lab.py: load cached scores or extract, then export adaptor.
    """
    scores, meta = get_scores_cached_or_extract(
        model, tokenizer, device, model_dir=model_dir, chunk_size=chunk_size
    )

    stats = analyze_scores_basic(scores)
    print("Bolmo boundary score statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / "bolmo_adaptor.npz"

    rep = build_bolmo_adaptor(scores, meta, out_path)
    print(f"\nExported bolmo_adaptor.npz -> {rep['out_path']}")
    print(f"frac_residual: {rep['frac_residual']:.6f}")
    for K, r in rep["K_report"].items():
        print(f"  K={K}: residual R²={r['residual_r2']:.6f}  full-logit R²={r['full_logit_r2']:.6f}")