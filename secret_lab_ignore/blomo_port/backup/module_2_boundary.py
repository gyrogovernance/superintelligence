"""
Module 2: Boundary calibration — extract Bolmo's boundary behavior over 256x256.

Extracts:
1. Continuous boundary probabilities (from boundary_logprobs)
2. Encoder hidden states ê at positions 1 and 2 (what the boundary predictor actually sees)

When model_dir is provided, loads/saves a disk cache (scores, h1_by_b1, h2_by_pair)
keyed by model path, boundary_predictor_lookahead, add_expanded_embeddings, tokenizer id.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from common import PROJECT_ROOT, bolmo_reset_local_caches

CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "blomo_port"


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


def _load_cache(cache_subdir: Path) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    scores_path = cache_subdir / "scores.npy"
    h1_path = cache_subdir / "h1_by_b1.npy"
    h2_path = cache_subdir / "h2_by_pair.npy"
    if not (scores_path.exists() and h1_path.exists() and h2_path.exists()):
        return None
    scores = np.load(scores_path).astype(np.float32)
    h1_by_b1 = np.load(h1_path).astype(np.float32)
    h2_by_pair = np.load(h2_path).astype(np.float32)
    d = h1_by_b1.shape[1]
    h_pos1 = np.broadcast_to(h1_by_b1[:, None, :], (256, 256, d)).copy()
    return scores, h_pos1, h2_by_pair


def _save_cache(
    cache_subdir: Path,
    scores: np.ndarray,
    h_pos1: np.ndarray,
    h_pos2: np.ndarray,
    meta: dict[str, Any],
) -> None:
    cache_subdir.mkdir(parents=True, exist_ok=True)
    np.save(cache_subdir / "scores.npy", scores)
    h1_by_b1 = h_pos1.mean(axis=1).astype(np.float16)
    np.save(cache_subdir / "h1_by_b1.npy", h1_by_b1)
    np.save(cache_subdir / "h2_by_pair.npy", h_pos2.astype(np.float16))
    with open(cache_subdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


@torch.inference_mode()
def extract_bolmo_boundary_scores(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    chunk_size: int = 16,
    model_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract Bolmo's CONTINUOUS boundary probability AND encoder hidden states
    for all 256x256 byte pairs.

    Returns:
        scores:  [256, 256] float array of boundary probabilities
        h_pos1:  [256, 256, d_model] encoder hidden states at position 1 (after b1)
        h_pos2:  [256, 256, d_model] encoder hidden states at position 2 (after b2)
    """
    if model_dir is not None:
        key = _cache_key(model, model_dir)
        cache_subdir = CACHE_DIR / key
        loaded = _load_cache(cache_subdir)
        if loaded is not None:
            print(f"Loaded boundary extraction from cache ({cache_subdir})")
            return loaded

    print("Extracting Bolmo continuous boundary scores + encoder states for all 256x256 byte pairs...")

    offset = int(getattr(tokenizer, "offset", 4))
    bos_id = int(tokenizer.bos_token_id)
    expand = getattr(model.model.local_encoder, "add_expanded_embeddings", False)
    d_model = model.config.hidden_size

    scores = np.zeros((256, 256), dtype=np.float32)
    h_pos1 = np.zeros((256, 256, d_model), dtype=np.float32)
    h_pos2 = np.zeros((256, 256, d_model), dtype=np.float32)
    t0 = time.perf_counter()

    for b1_start in range(0, 256, chunk_size):
        b1_end = min(b1_start + chunk_size, 256)
        for b2_start in range(0, 256, chunk_size):
            b2_end = min(b2_start + chunk_size, 256)

            triples = []
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

            # h_byte is the encoder output BEFORE pooling — this is ê
            h_byte, _, bnd_logprobs, _ = model.model.local_encoder(
                batch,
                expanded_input_ids=expanded,
                sequence_start_indices=seq_start,
                boundary_state=None,
                pad_state=None,
            )

            bnd_probs = torch.exp(bnd_logprobs[:, 1]).cpu().numpy()
            h_np = h_byte.cpu().numpy()

            idx = 0
            for b1 in range(b1_start, b1_end):
                for b2 in range(b2_start, b2_end):
                    scores[b1, b2] = bnd_probs[idx]
                    h_pos1[b1, b2, :] = h_np[idx, 1, :]
                    h_pos2[b1, b2, :] = h_np[idx, 2, :]
                    idx += 1

        elapsed = time.perf_counter() - t0
        print(f"  b1=[{b1_start}..{b1_end}] done ({elapsed:.1f}s)")

    elapsed_total = time.perf_counter() - t0
    print(f"Extraction complete in {elapsed_total:.1f}s")

    if model_dir is not None:
        key = _cache_key(model, model_dir)
        cache_subdir = CACHE_DIR / key
        d_model = h_pos1.shape[2]
        lookahead = getattr(
            model.model.local_encoder.boundary_predictor_module,
            "boundary_predictor_lookahead",
            1,
        )
        expand = getattr(model.model.local_encoder, "add_expanded_embeddings", False)
        meta = {
            "model_key": key,
            "d_model": d_model,
            "boundary_predictor_lookahead": lookahead,
            "add_expanded_embeddings": expand,
        }
        _save_cache(cache_subdir, scores, h_pos1, h_pos2, meta)
        print(f"Cached to {cache_subdir}")

    return scores, h_pos1, h_pos2


def analyze_bolmo_boundaries(scores: np.ndarray) -> dict[str, Any]:
    flat = scores.flatten()
    stats: dict[str, Any] = {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "median": float(np.median(flat)),
        "q25": float(np.quantile(flat, 0.25)),
        "q75": float(np.quantile(flat, 0.75)),
        "rate_at_0.5": float(np.mean(flat > 0.5)),
    }

    row_means = np.mean(scores, axis=1)
    col_means = np.mean(scores, axis=0)
    stats["highest_boundary_after"] = int(np.argmax(row_means))
    stats["lowest_boundary_after"] = int(np.argmin(row_means))
    stats["highest_boundary_before"] = int(np.argmax(col_means))
    stats["lowest_boundary_before"] = int(np.argmin(col_means))

    return stats