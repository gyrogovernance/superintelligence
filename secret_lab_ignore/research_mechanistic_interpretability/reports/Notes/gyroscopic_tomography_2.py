#!/usr/bin/env python3
"""
gyroscopic_tomography_2.py

Core question: Does the Router horizon row consistently exhibit different
holonomy characteristics than random rows, across many sequences and layers?

Method:
- Generate many random token sequences
- Route each through the kernel (BYTES4 binding) to get horizon index
- Collect last-token activations at layers 7, 15, 31
- For each sequence, measure holonomy in:
  (a) the horizon row (router-selected)
  (b) a random comparison row
  (c) neighbor row (horizon+1)
- Use S-P style transport: whitening, shared-midpoint k-NN, SO Procrustes
- Report statistics: do horizon rows differ systematically from random?

This tests whether the Router-Transformer coupling is real and consistent,
not just an artifact of a single sequence.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

torch.set_grad_enabled(False)
np.set_printoptions(precision=4, suppress=True, linewidth=140)

MODEL_DIR = Path("data/models/Olmo-3-7B-Instruct")
ROUTER_ATLAS_DIR = Path("data/atlas")

DEVICE = "cpu"
DTYPE = torch.bfloat16

SEED = 201
SEQ_LEN = 32
N_SEQ = 48
BATCH_SIZE = 8
LAYERS = [7, 15, 31]

EPS = 0.02
KNN_K = 32
SUBSPACE_Q = 8


def load_router():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.router.kernel import RouterKernel
    return RouterKernel(ROUTER_ATLAS_DIR)


def load_olmo():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=DTYPE, low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    print(f"OLMo loaded in {time.time()-t0:.1f}s  h={model.config.hidden_size} L={model.config.num_hidden_layers}")
    return model, tok


def route_bytes4(kernel, tids: np.ndarray) -> int:
    kernel.reset()
    for tid in tids.astype(np.uint32):
        for b in int(tid).to_bytes(4, "big"):
            kernel.step_byte(int(b))
    return int(kernel.state_horizon[kernel.state_index])


def collect_batch(model, input_ids: torch.Tensor, layers: List[int]) -> Dict[int, torch.Tensor]:
    layer_set = set(layers)
    acts: Dict[int, List[torch.Tensor]] = {L: [] for L in layers}

    def hook_factory(idx):
        def _hook(_m, _i, out):
            if idx in layer_set:
                t = out[0] if isinstance(out, tuple) else out
                acts[idx].append(t[:, -1, :].detach().cpu().float())
        return _hook

    handles = []
    for i in range(model.config.num_hidden_layers):
        if i in layer_set:
            h = model.model.layers[i].post_feedforward_layernorm.register_forward_hook(hook_factory(i))
            handles.append(h)

    with torch.no_grad():
        model(input_ids.to(DEVICE))

    for h in handles:
        h.remove()

    return {L: torch.cat(acts[L], dim=0) for L in layers}


def procrustes_so(M: torch.Tensor) -> torch.Tensor:
    U, _, Vt = torch.linalg.svd(M)
    R = U @ Vt
    if torch.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def edge_transport(z_i: torch.Tensor, z_j: torch.Tensor, pool: torch.Tensor, k: int, q: int) -> torch.Tensor:
    p = z_i.shape[0]
    I = torch.eye(p, dtype=torch.float32)
    
    mid = 0.5 * (z_i + z_j)
    d_mid = torch.norm(pool - mid.unsqueeze(0), dim=1)
    _, cand_idx = torch.topk(d_mid, min(4*k, pool.shape[0]), largest=False)
    cand = pool[cand_idx]
    
    d_i = torch.norm(cand - z_i.unsqueeze(0), dim=1)
    d_j = torch.norm(cand - z_j.unsqueeze(0), dim=1)
    _, ii = torch.topk(d_i, k, largest=False)
    _, ij = torch.topk(d_j, k, largest=False)
    
    union_idx = torch.unique(torch.cat([ii, ij]))
    mu = cand[union_idx].mean(dim=0)
    
    X = cand[ii] - mu
    Y = cand[ij] - mu
    
    stacked = torch.cat([X, Y], dim=0)
    _, _, Vt = torch.linalg.svd(stacked, full_matrices=False)
    B = Vt[:q, :].T
    
    Xq, Yq = X @ B, Y @ B
    Rq = procrustes_so(Yq.T @ Xq)
    return B @ Rq @ B.T + (I - B @ B.T)


def rect_holonomy(z0: torch.Tensor, u: torch.Tensor, v: torch.Tensor, eps: float, pool: torch.Tensor, k: int, q: int) -> float:
    pts = [z0, z0 + eps*u, z0 + eps*(u+v), z0 + eps*v]
    p = z0.shape[0]
    H = torch.eye(p, dtype=torch.float32)
    for i in range(4):
        H = edge_transport(pts[i], pts[(i+1)%4], pool, k, q) @ H
    diff = H - torch.eye(p, dtype=torch.float32)
    return float(torch.norm(diff, p="fro") / (2.0 * np.sqrt(p)))


def whiten_row_pool(pool_4096: torch.Tensor, row: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grid = pool_4096.view(pool_4096.shape[0], 256, 16)
    row_pool = grid[:, row, :]
    mu = row_pool.mean(dim=0)
    std = row_pool.std(dim=0).clamp(min=1e-8)
    return (row_pool - mu) / std, mu, std


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    try:
        torch.set_num_threads(min(12, os.cpu_count() or 4))
    except:
        pass

    t0 = time.time()
    print("=" * 50)
    print("GYROSCOPIC TOMOGRAPHY v2")
    print("=" * 50)

    model, tok = load_olmo()
    kernel = load_router()

    lo, hi = 1000, tok.vocab_size - 1000
    input_ids = torch.randint(lo, hi, (N_SEQ, SEQ_LEN), dtype=torch.long)

    horizons = np.array([route_bytes4(kernel, input_ids[i].numpy()) for i in range(N_SEQ)], dtype=np.int32)
    print(f"Horizons: unique={len(np.unique(horizons))} range=[{horizons.min()},{horizons.max()}]")

    all_acts: Dict[int, List[torch.Tensor]] = {L: [] for L in LAYERS}
    print(f"Collecting activations in batches of {BATCH_SIZE}...")
    
    for start in range(0, N_SEQ, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N_SEQ)
        batch_ids = input_ids[start:end]
        batch_acts = collect_batch(model, batch_ids, LAYERS)
        for L in LAYERS:
            all_acts[L].append(batch_acts[L])
        print(f"  batch {start//BATCH_SIZE + 1}/{(N_SEQ + BATCH_SIZE - 1)//BATCH_SIZE}")

    layer_acts = {L: torch.cat(all_acts[L], dim=0) for L in LAYERS}
    print(f"Activations collected: {layer_acts[LAYERS[0]].shape}")

    print("=" * 50)
    print("HOLONOMY ANALYSIS")
    print("=" * 50)

    rng = np.random.default_rng(SEED + 100)

    for L in LAYERS:
        acts = layer_acts[L]
        
        mu_global = acts.mean(dim=0)
        std_global = acts.std(dim=0).clamp(min=1e-8)
        acts_w = (acts - mu_global) / std_global

        h_holonomies = []
        n_holonomies = []
        r_holonomies = []

        for i in range(N_SEQ):
            h_row = int(horizons[i])
            n_row = (h_row + 1) % 256
            r_row = int(rng.integers(0, 256))

            torch.manual_seed(SEED + 2000 + L*100 + i)
            u = torch.randn(16, dtype=torch.float32)
            u = u / (u.norm() + 1e-12)
            v = torch.randn(16, dtype=torch.float32)
            v = v - (v @ u) * u
            v = v / (v.norm() + 1e-12)

            z_4096 = acts_w[i]
            grid = z_4096.view(256, 16)

            for row, results_list in [(h_row, h_holonomies), (n_row, n_holonomies), (r_row, r_holonomies)]:
                pool_w, mu_r, std_r = whiten_row_pool(acts_w, row)
                z0 = (grid[row] - mu_r) / std_r
                hol = rect_holonomy(z0, u, v, EPS, pool_w, KNN_K, SUBSPACE_Q)
                results_list.append(hol)

        h_arr = np.array(h_holonomies)
        n_arr = np.array(n_holonomies)
        r_arr = np.array(r_holonomies)

        def stats(arr):
            return f"mean={arr.mean():.4f} std={arr.std():.4f} med={np.median(arr):.4f}"

        print(f"\nLayer {L}:")
        print(f"  Horizon (H):  {stats(h_arr)}")
        print(f"  Neighbor (N): {stats(n_arr)}")
        print(f"  Random (R):   {stats(r_arr)}")
        
        ratio_h_r = h_arr.mean() / (r_arr.mean() + 1e-8)
        ratio_h_n = h_arr.mean() / (n_arr.mean() + 1e-8)
        
        from scipy import stats as sp_stats
        t_hr, p_hr = sp_stats.ttest_ind(h_arr, r_arr)
        t_hn, p_hn = sp_stats.ttest_ind(h_arr, n_arr)
        
        print(f"  H/R ratio: {ratio_h_r:.3f}  t={t_hr:.2f} p={p_hr:.4f}")
        print(f"  H/N ratio: {ratio_h_n:.3f}  t={t_hn:.2f} p={p_hn:.4f}")

        h_rank = np.mean([np.sum(r_arr < h_arr[i]) / len(r_arr) for i in range(len(h_arr))])
        print(f"  H rank in R distribution: {h_rank:.3f} (0.5=no difference)")

    print("=" * 50)
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()