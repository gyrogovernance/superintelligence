from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from common import PROJECT_ROOT
from module_2b_correlation import compute_kernel_boundary_features
from src.router.constants import mask12_for_byte, popcount


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def linear_probe_r2(x: np.ndarray, y: np.ndarray) -> float:
    x_bias = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    w, _, _, _ = np.linalg.lstsq(x_bias, y, rcond=None)
    y_pred = x_bias @ w
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def cosine_boundary(h_q: np.ndarray, h_k: np.ndarray, wq: np.ndarray, wk: np.ndarray) -> np.ndarray:
    n = h_q.shape[0] * h_q.shape[1]
    q = h_q.reshape(n, h_q.shape[-1]) @ wq.T
    k = h_k.reshape(n, h_k.shape[-1]) @ wk.T
    q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    k /= (np.linalg.norm(k, axis=1, keepdims=True) + 1e-12)
    cos_sim = np.clip(np.sum(q * k, axis=1), -1.0, 1.0)
    return (0.5 * (1.0 - cos_sim)).reshape(256, 256).astype(np.float32)


def effective_rank_from_singulars(s: np.ndarray) -> dict[str, float]:
    e = s ** 2
    p = e / (np.sum(e) + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    energy_cum = np.cumsum(e) / (np.sum(e) + 1e-12)
    k95 = int(np.searchsorted(energy_cum, 0.95) + 1)
    k99 = int(np.searchsorted(energy_cum, 0.99) + 1)
    stable_rank = float(np.sum(e) / (np.max(e) + 1e-12))
    return {
        "effective_rank_entropy": float(np.exp(entropy)),
        "stable_rank": stable_rank,
        "k95": float(k95),
        "k99": float(k99),
    }


def xor_feature_regression(scores: np.ndarray) -> dict[str, float]:
    y = scores.reshape(-1)
    dvals = np.zeros((256, 256), dtype=np.int32)
    masks = np.array([mask12_for_byte(b) for b in range(256)], dtype=np.int32)
    for b1 in range(256):
        for b2 in range(256):
            dvals[b1, b2] = popcount(int(masks[b1] ^ masks[b2]))

    x = np.zeros((256 * 256, 13), dtype=np.float32)
    flat_d = dvals.reshape(-1)
    x[np.arange(x.shape[0]), flat_d] = 1.0
    return {"r2_score_from_xor_popcount_bins": float(linear_probe_r2(x, y[:, None]))}


def load_boundary_weights(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    idx_path = model_dir / "model.safetensors.index.json"
    index = json.loads(idx_path.read_text(encoding="utf-8"))
    q_key = "model.local_encoder.boundary_predictor_module.q_proj_layer.weight"
    k_key = "model.local_encoder.boundary_predictor_module.k_proj_layer.weight"
    q_file = model_dir / index["weight_map"][q_key]
    k_file = model_dir / index["weight_map"][k_key]

    with safe_open(str(q_file), framework="np") as f:
        wq = f.get_tensor(q_key).astype(np.float32)
    with safe_open(str(k_file), framework="np") as f:
        wk = f.get_tensor(k_key).astype(np.float32)
    return wq, wk


def main() -> None:
    project_root = PROJECT_ROOT
    cache_subdir = project_root / "data" / "cache" / "blomo_port" / "69eb4607b69f0909"
    model_dir = project_root / "data" / "models" / "Bolmo-1B"
    atlas_dir = project_root / "data" / "atlas"

    scores = np.load(cache_subdir / "scores.npy").astype(np.float32)
    h1_by_b1 = np.load(cache_subdir / "h1_by_b1.npy").astype(np.float32)
    h2_by_pair = np.load(cache_subdir / "h2_by_pair.npy").astype(np.float32)
    h1_by_pair = np.broadcast_to(h1_by_b1[:, None, :], (256, 256, h1_by_b1.shape[1])).copy()

    wq, wk = load_boundary_weights(model_dir)

    recon_impl = cosine_boundary(h1_by_pair, h2_by_pair, wq, wk)
    recon_alt = cosine_boundary(h2_by_pair, h1_by_pair, wq, wk)
    r2_impl = r2(scores, recon_impl)
    r2_alt = r2(scores, recon_alt)

    h1_total_var = np.sum(np.var(h1_by_pair.reshape(65536, -1), axis=0))
    h1_within_var = sum(np.sum(np.var(h1_by_pair[b1], axis=0)) for b1 in range(256)) / 256.0
    h1_cross = float(h1_within_var / (h1_total_var + 1e-12))
    h2_cross = float(
        (sum(np.sum(np.var(h2_by_pair[:, b2, :], axis=0)) for b2 in range(256)) / 256.0)
        / (np.sum(np.var(h2_by_pair.reshape(65536, -1), axis=0)) + 1e-12)
    )

    h2_by_b2 = h2_by_pair.mean(axis=0)
    q_by_b1 = h1_by_b1 @ wq.T
    k_by_b2 = h2_by_b2 @ wk.T
    q_by_b1 /= (np.linalg.norm(q_by_b1, axis=1, keepdims=True) + 1e-12)
    k_by_b2 /= (np.linalg.norm(k_by_b2, axis=1, keepdims=True) + 1e-12)
    g = q_by_b1 @ k_by_b2.T
    s = np.linalg.svd(g, compute_uv=False)
    g_rank = effective_rank_from_singulars(s)

    g_centered = g - g.mean(axis=0, keepdims=True) - g.mean(axis=1, keepdims=True) + g.mean()
    s_centered = np.linalg.svd(g_centered, compute_uv=False)
    g_rank_centered = effective_rank_from_singulars(s_centered)

    bases = compute_kernel_boundary_features(atlas_dir)
    basis_fit: dict[str, Any] = {}
    for name, basis in bases.items():
        if name == "byte_onehot":
            continue
        basis_fit[f"r2_q_from_{name}"] = float(linear_probe_r2(basis, q_by_b1))
        basis_fit[f"r2_k_from_{name}"] = float(linear_probe_r2(basis, k_by_b2))

    y = scores.reshape(-1)
    eps = 1e-6
    y_logit = np.log(np.clip(y, eps, 1 - eps) / (1 - np.clip(y, eps, 1 - eps)))
    bilinear_fit: dict[str, float] = {}
    for name, basis in bases.items():
        if name == "byte_onehot":
            continue
        d = basis.shape[1]
        n = 256 * 256
        x = np.zeros((n, d + d + d * d), dtype=np.float32)
        idx = 0
        for b1 in range(256):
            v1 = basis[b1]
            for b2 in range(256):
                v2 = basis[b2]
                x[idx] = np.concatenate([v1, v2, np.outer(v1, v2).reshape(-1)])
                idx += 1
        x_bias = np.concatenate([x, np.ones((n, 1), dtype=np.float32)], axis=1)
        w, _, _, _ = np.linalg.lstsq(x_bias, y_logit, rcond=None)
        y_pred = x_bias @ w
        ss_res = float(np.sum((y_logit - y_pred) ** 2))
        ss_tot = float(np.sum((y_logit - np.mean(y_logit)) ** 2)) + 1e-12
        bilinear_fit[f"r2_logit_bilinear_{name}"] = float(1.0 - ss_res / ss_tot)

    xor_fit = xor_feature_regression(scores)

    out = {
        "cache_dir": str(cache_subdir),
        "shapes": {
            "scores": list(scores.shape),
            "h1_by_b1": list(h1_by_b1.shape),
            "h2_by_pair": list(h2_by_pair.shape),
            "Wq": list(wq.shape),
            "Wk": list(wk.shape),
        },
        "reconstruction_r2": {
            "impl_orientation_h1_wq_h2_wk": float(r2_impl),
            "alt_orientation_h2_wq_h1_wk": float(r2_alt),
        },
        "cross_dependence_ratio": {
            "h1_var_when_b2_changes": h1_cross,
            "h2_var_when_b1_changes": h2_cross,
        },
        "g_rank_raw": g_rank,
        "g_rank_centered": g_rank_centered,
        "qk_basis_fit": basis_fit,
        "boundary_logit_bilinear_fit": bilinear_fit,
    }
    out.update(xor_fit)

    out_dir = project_root / "data" / "cache" / "blomo_port" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "module2c_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
