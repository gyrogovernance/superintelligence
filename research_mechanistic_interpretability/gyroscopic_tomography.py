#!/usr/bin/env python3
"""
Gyroscopic Tomography (Iteration 5)

Purpose:
- Decide if Router↔Transformer coupling is real by comparing kernel bindings:
  A) low8: step kernel with (token_id & 0xFF)
  B) bytes4: step kernel with token_id.to_bytes(4,'big') (agent-consistent)

- Decide if "3D commutator closure" is signal or artifact by a null suite:
  For each layer and binding, compare 16D extractors:
    H: Router horizon row
    N: neighbor row (H+1)
    R: deterministic pseudo-random row
    P: random 16D projection from full 4096D

Outputs:
- Binding A/B MRI effect size per layer (variance ratio vs permuted baseline)
- Curvature concentration per layer/binding (horizon16 holonomy vs proj256 holonomy)
- Null suite per layer/binding/extractor: dim3, dim5, median ||G||_F

Assumes:
- OLMo-3-7B-Instruct in data/models/Olmo-3-7B-Instruct
- Manifold atlas in data/atlas/olmo_3_7b_manifolds (pool, mean, std)
- Router atlas in data/atlas
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

torch.set_grad_enabled(False)
np.set_printoptions(precision=6, suppress=True, linewidth=120)

MODEL_DIR = Path("data/models/Olmo-3-7B-Instruct")
MANIFOLD_DIR = Path("data/atlas/olmo_3_7b_manifolds")
ROUTER_ATLAS_DIR = Path("data/atlas")

DEVICE = "cpu"
DTYPE = torch.bfloat16

SEED = 83
SEQ_LEN = 32
N_POOL = 2048

LAYERS_MRI = [0, 7, 15, 31]
LAYERS_CORE = [7, 15, 31]

N_SEQ_MRI = 32
MRI_BATCH = 8

PROJ_DIM = 256
RADIUS = 0.20
N_LOOP_POINTS = 6
N_BASE = 1

EPS = 0.01
DIR_SAMPLES = 6
H_K = 48
H_Q = 8


@dataclass
class NullStats:
    dim3: int
    dim5: int
    med_g_norm: float


def _set_threads():
    try:
        n = min(12, os.cpu_count() or 12)
        torch.set_num_threads(n)
    except Exception:
        pass


def _sep(title: str):
    print("=====")
    print(title)
    print("=====")


def load_router():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.router.kernel import RouterKernel
    return RouterKernel(ROUTER_ATLAS_DIR)


def load_olmo():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    dt = time.time() - t0
    print(f"Loaded OLMo in {dt:.1f}s  hidden={model.config.hidden_size}  layers={model.config.num_hidden_layers}  heads={model.config.num_attention_heads}")
    return model, tokenizer


def load_manifold(layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = MANIFOLD_DIR / f"layer_{layer_idx}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing manifold for layer {layer_idx}: {p}")
    d = np.load(p)
    pool = torch.from_numpy(d["pool"]).float()
    mean = torch.from_numpy(d["mean"]).float()
    std = torch.from_numpy(d["std"]).float()
    return pool, mean, std


def make_orthonormal_projection(in_dim: int, out_dim: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    A = torch.randn(in_dim, out_dim, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q.contiguous()


def generate_random_sequences(tokenizer, n_seq: int, seq_len: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    lo = 1000
    hi = tokenizer.vocab_size - 1000
    return torch.randint(lo, hi, (n_seq, seq_len), dtype=torch.long)


def get_random_plane(hidden_dim: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    u = torch.randn(hidden_dim, dtype=torch.float32)
    u = u / (u.norm() + 1e-12)
    v = torch.randn(hidden_dim, dtype=torch.float32)
    v = v - (v @ u) * u
    v = v / (v.norm() + 1e-12)
    return u, v


def build_embedding_loop(base_embeds: torch.Tensor, u: torch.Tensor, v: torch.Tensor, radius: float, n_points: int) -> torch.Tensor:
    thetas = torch.linspace(0.0, 2.0 * np.pi, n_points + 1, dtype=torch.float32)[:-1]
    cu = torch.cos(thetas).view(-1, 1, 1)
    sv = torch.sin(thetas).view(-1, 1, 1)
    delta = radius * (cu * u.view(1, 1, -1) + sv * v.view(1, 1, -1))
    return base_embeds.unsqueeze(0) + delta


def forward_collect_with_embed_override(model, embeds_batch: torch.Tensor, layer_idx: int) -> torch.Tensor:
    acts: List[torch.Tensor] = []

    def _hook(_module, _inp, output):
        t = output[0] if isinstance(output, tuple) else output
        acts.append(t.detach().to("cpu", torch.float32))

    handle = model.model.layers[layer_idx].post_feedforward_layernorm.register_forward_hook(_hook)

    dummy_ids = torch.zeros(embeds_batch.shape[0], embeds_batch.shape[1], dtype=torch.long, device=DEVICE)
    orig_forward = model.model.embed_tokens.forward

    def _embed_override(_x):
        return embeds_batch.to(device=DEVICE, dtype=DTYPE)

    model.model.embed_tokens.forward = _embed_override
    with torch.no_grad():
        model.forward(dummy_ids)
    model.model.embed_tokens.forward = orig_forward
    handle.remove()

    if not acts:
        raise RuntimeError("No activations captured")
    return acts[0]


def collect_layers_last_token(model, input_ids: torch.Tensor, layer_indices: List[int]) -> Dict[int, torch.Tensor]:
    n_layers = model.config.num_hidden_layers
    layer_set = set(layer_indices)
    acts_tmp: Dict[int, List[torch.Tensor]] = {idx: [] for idx in layer_indices}

    def hook_factory(idx: int):
        def _hook(_module, _inp, output):
            if idx not in layer_set:
                return
            t = output[0] if isinstance(output, tuple) else output
            acts_tmp[idx].append(t.detach().to("cpu", torch.float32)[:, -1, :])
        return _hook

    handles = []
    for i in range(n_layers):
        if i in layer_set:
            handles.append(model.model.layers[i].post_feedforward_layernorm.register_forward_hook(hook_factory(i)))

    with torch.no_grad():
        model(input_ids.to(DEVICE))

    for h in handles:
        h.remove()

    out: Dict[int, torch.Tensor] = {}
    for idx in layer_indices:
        out[idx] = torch.cat(acts_tmp[idx], dim=0)
    return out


def whiten_full(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std.clamp(min=1e-8)


def route_low8(kernel, token_ids_1d: np.ndarray) -> int:
    kernel.reset()
    for tid in token_ids_1d.astype(np.int64):
        kernel.step_byte(int(tid) & 0xFF)
    return int(kernel.state_horizon[kernel.state_index])


def route_bytes4(kernel, token_ids_1d: np.ndarray) -> int:
    kernel.reset()
    for tid in token_ids_1d.astype(np.uint32):
        bs = int(tid).to_bytes(4, "big", signed=False)
        for b in bs:
            kernel.step_byte(int(b))
    return int(kernel.state_horizon[kernel.state_index])


def _knn_indices(x: torch.Tensor, pool: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, pool.shape[0])
    d = torch.norm(pool - x.unsqueeze(0), dim=1)
    _, idx = torch.topk(d, k, largest=False, sorted=True)
    return idx


def _subset_knn_indices(x: torch.Tensor, pool: torch.Tensor, cand_idx: torch.Tensor, k: int) -> torch.Tensor:
    cand = pool[cand_idx]
    k = min(k, cand.shape[0])
    d = torch.norm(cand - x.unsqueeze(0), dim=1)
    _, rel = torch.topk(d, k, largest=False, sorted=True)
    return cand_idx[rel]


def procrustes_so(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    M = Y.T @ X
    U, _, Vt = torch.linalg.svd(M)
    R = U @ Vt
    if torch.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def edge_transport(z_i: torch.Tensor, z_j: torch.Tensor, pool: torch.Tensor, k: int, q: int) -> torch.Tensor:
    p = z_i.shape[0]
    I = torch.eye(p, dtype=torch.float32)

    k0 = min(4 * k, pool.shape[0])
    mid = 0.5 * (z_i + z_j)
    C = _knn_indices(mid, pool, k0)
    Is = _subset_knn_indices(z_i, pool, C, k)
    It = _subset_knn_indices(z_j, pool, C, k)

    X0 = pool[Is] - z_i.unsqueeze(0)
    Y0 = pool[It] - z_j.unsqueeze(0)

    wX = torch.exp(-torch.norm(pool[Is] - mid.unsqueeze(0), dim=1))
    wY = torch.exp(-torch.norm(pool[It] - mid.unsqueeze(0), dim=1))
    wX = wX / (wX.sum() + 1e-12)
    wY = wY / (wY.sum() + 1e-12)

    muX = (wX.unsqueeze(1) * X0).sum(dim=0)
    muY = (wY.unsqueeze(1) * Y0).sum(dim=0)
    X = X0 - muX
    Y = Y0 - muY

    stacked = torch.cat([X, Y], dim=0)
    _, _, Vt = torch.linalg.svd(stacked, full_matrices=False)
    B = Vt[:q, :].T.contiguous()

    Xq = X @ B
    Yq = Y @ B

    Xw = Xq * (wX.sqrt().unsqueeze(1))
    Yw = Yq * (wY.sqrt().unsqueeze(1))

    Rq = procrustes_so(Xw, Yw)
    Rp = B @ Rq @ B.T + (I - B @ B.T)
    return Rp


def loop_hnorm(loop_vecs: torch.Tensor, pool: torch.Tensor, k: int, q: int) -> float:
    n = loop_vecs.shape[0]
    p = loop_vecs.shape[1]
    H = torch.eye(p, dtype=torch.float32)
    for i in range(n):
        j = (i + 1) % n
        H = edge_transport(loop_vecs[i], loop_vecs[j], pool, k=k, q=q) @ H
    diff = H - torch.eye(p, dtype=torch.float32)
    return float(torch.norm(diff, p="fro").item() / (2.0 * np.sqrt(p)))


def skew_np(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M - M.T)


def effective_dim_99(mats: List[np.ndarray]) -> int:
    X = np.stack([m.reshape(-1) for m in mats], axis=0)
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    if np.sum(S**2) <= 0:
        return 0
    cum = np.cumsum(S**2) / np.sum(S**2)
    return int(np.searchsorted(cum, 0.99) + 1)


def compute_null_stats_from_outputs(
    last_w: torch.Tensor,
    pool16_w: torch.Tensor,
    extractor: str,
    row_idx: int,
    P16: torch.Tensor,
) -> NullStats:
    z0 = None
    z_list: List[torch.Tensor] = []

    if extractor == "row":
        grid = last_w.view(last_w.shape[0], 256, 16)
        vecs = grid[:, row_idx, :]
        z0 = vecs[0]
        z_list = [vecs[i] for i in range(1, vecs.shape[0])]
    else:
        vecs = last_w @ P16
        z0 = vecs[0]
        z_list = [vecs[i] for i in range(1, vecs.shape[0])]

    Gs: List[np.ndarray] = []
    norms: List[float] = []

    for z1 in z_list:
        R = edge_transport(z0, z1, pool16_w, k=H_K, q=H_Q).double().numpy()
        G = skew_np((R - np.eye(16)) / EPS)
        Gs.append(G)
        norms.append(float(np.linalg.norm(G, ord="fro")))

    if not Gs:
        return NullStats(dim3=0, dim5=0, med_g_norm=0.0)

    Xmat = np.stack([G.reshape(-1) for G in Gs], axis=0)
    _, _, Vt = np.linalg.svd(Xmat, full_matrices=False)
    X = skew_np(Vt[0].reshape(16, 16))
    Y = skew_np(Vt[1].reshape(16, 16))
    W = skew_np(X @ Y - Y @ X)
    A = skew_np(X @ W - W @ X)
    B = skew_np(Y @ W - W @ Y)

    dim3 = effective_dim_99([X, Y, W])
    dim5 = effective_dim_99([X, Y, W, A, B])
    medn = float(np.median(np.array(norms, dtype=np.float64)))
    return NullStats(dim3=dim3, dim5=dim5, med_g_norm=medn)


def main():
    _set_threads()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    t0 = time.time()
    model, tokenizer = load_olmo()
    kernel = load_router()

    hidden = model.config.hidden_size
    if hidden != 4096:
        raise RuntimeError(f"Expected hidden 4096, got {hidden}")

    P256 = make_orthonormal_projection(hidden, PROJ_DIM, seed=SEED + 1)
    P16 = make_orthonormal_projection(hidden, 16, seed=SEED + 2)

    toks = generate_random_sequences(tokenizer, N_SEQ_MRI, SEQ_LEN, seed=SEED + 3)
    acts_mri = collect_layers_last_token(model, toks, LAYERS_MRI)

    perm = np.random.default_rng(SEED + 4).permutation(hidden)

    _sep("BINDING MRI (LOW8 VS BYTES4)")
    for layer_idx in LAYERS_MRI:
        a = acts_mri[layer_idx].numpy()
        ratios_a = []
        ratios_b = []
        ratios_pa = []
        ratios_pb = []

        for j in range(a.shape[0]):
            ids = toks[j].numpy()
            h_a = route_low8(kernel, ids)
            h_b = route_bytes4(kernel, ids)

            v = a[j].reshape(256, 16)
            rv = v.var(axis=1)
            med = float(np.median(rv) + 1e-30)
            ratios_a.append(float(rv[h_a] / med))
            ratios_b.append(float(rv[h_b] / med))

            vp = a[j][perm].reshape(256, 16)
            rvp = vp.var(axis=1)
            medp = float(np.median(rvp) + 1e-30)
            ratios_pa.append(float(rvp[h_a] / medp))
            ratios_pb.append(float(rvp[h_b] / medp))

        def eff_d(x, y):
            x = np.array(x, dtype=np.float64)
            y = np.array(y, dtype=np.float64)
            return float((x.mean() - y.mean()) / (np.sqrt(0.5 * (x.var() + y.var())) + 1e-30))

        d_low8 = eff_d(ratios_a, ratios_pa)
        d_b4 = eff_d(ratios_b, ratios_pb)
        print(f"L{layer_idx:02d}  d_low8 {d_low8:+.2f}  d_bytes4 {d_b4:+.2f}")

    _sep("CURVATURE CONCENTRATION (R=0.20)")
    base = generate_random_sequences(tokenizer, N_BASE, SEQ_LEN, seed=SEED + 10)[0:1]
    base_ids_1d = base[0].numpy()

    u, v = get_random_plane(hidden, seed=SEED + 11)
    base_emb = model.model.embed_tokens.forward(base.to(DEVICE)).float().cpu()[0]
    loop_embeds = build_embedding_loop(base_emb, u, v, radius=RADIUS, n_points=N_LOOP_POINTS)

    for layer_idx in LAYERS_CORE:
        pool_full, mean_full, std_full = load_manifold(layer_idx)
        pool_w = whiten_full(pool_full, mean_full, std_full)
        pool_grid = pool_w.view(N_POOL, 256, 16)
        pool_proj = pool_w @ P256

        acts = forward_collect_with_embed_override(model, loop_embeds, layer_idx)
        last = acts[:, -1, :]
        last_w = whiten_full(last, mean_full, std_full)

        h_low8 = route_low8(kernel, base_ids_1d)
        h_b4 = route_bytes4(kernel, base_ids_1d)

        for tag, h_idx in [("low8", h_low8), ("b4", h_b4)]:
            vec_h = last_w.view(N_LOOP_POINTS, 256, 16)[:, h_idx, :]
            pool_h = pool_grid[:, h_idx, :]
            mu = pool_h.mean(dim=0)
            sd = (pool_h - mu).std(dim=0).clamp(min=1e-8)
            pool_h_w = (pool_h - mu) / sd
            vec_h_w = (vec_h - mu) / sd

            h16 = loop_hnorm(vec_h_w, pool_h_w, k=H_K, q=H_Q)
            h256 = loop_hnorm(last_w @ P256, pool_proj, k=64, q=32)
            print(f"L{layer_idx:02d} {tag}  h16 {h16:.3f}  h256 {h256:.3f}  ratio {h16/(h256+1e-30):.3f}")

    _sep("NULL SUITE (DIM3/DIM5/MED||G||)")
    dirs = []
    torch.manual_seed(SEED + 20)
    for _ in range(DIR_SAMPLES):
        d = torch.randn(hidden, dtype=torch.float32)
        d = d / (d.norm() + 1e-12)
        dirs.append(d)

    for layer_idx in LAYERS_CORE:
        pool_full, mean_full, std_full = load_manifold(layer_idx)
        pool_w = whiten_full(pool_full, mean_full, std_full)
        pool_grid = pool_w.view(N_POOL, 256, 16)
        pool_p16 = pool_w @ P16

        h_low8 = route_low8(kernel, base_ids_1d)
        h_b4 = route_bytes4(kernel, base_ids_1d)

        rng = np.random.default_rng(SEED + 100 + layer_idx)
        r_row = int(rng.integers(0, 256))

        embeds = [base_emb] + [base_emb + EPS * d for d in dirs]
        batch = torch.stack(embeds, dim=0)
        acts = forward_collect_with_embed_override(model, batch, layer_idx)
        last = acts[:, -1, :]
        last_w = whiten_full(last, mean_full, std_full)

        def run_set(binding: str, h_idx: int):
            n_idx = (h_idx + 1) % 256
            items = [("H", "row", h_idx), ("N", "row", n_idx), ("R", "row", r_row), ("P", "proj", -1)]
            for name, kind, idx in items:
                if kind == "row":
                    pool_h = pool_grid[:, idx, :]
                    mu = pool_h.mean(dim=0)
                    sd = (pool_h - mu).std(dim=0).clamp(min=1e-8)
                    pool_h_w = (pool_h - mu) / sd
                    stats = compute_null_stats_from_outputs(last_w, pool_h_w, "row", idx, P16)
                else:
                    mu = pool_p16.mean(dim=0)
                    sd = (pool_p16 - mu).std(dim=0).clamp(min=1e-8)
                    pool_p16_w = (pool_p16 - mu) / sd
                    stats = compute_null_stats_from_outputs(last_w, pool_p16_w, "proj", 0, P16)
                print(f"L{layer_idx:02d} {binding} {name}  d3 {stats.dim3}  d5 {stats.dim5}  medG {stats.med_g_norm:.2f}")

        run_set("low8", h_low8)
        run_set("b4", h_b4)

    dt = time.time() - t0
    print(f"Total time: {dt/60:.1f} min")


if __name__ == "__main__":
    main()