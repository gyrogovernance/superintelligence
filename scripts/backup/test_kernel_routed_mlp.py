# scripts/experiments_kernel_routing.py
"""
Kernel-routed transformer MLP experiments.
Respects 4096 = 256 × 16 boundary × fiber factorization.
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.getcwd())
from src.router.kernel import RouterKernel

MODEL_DIR = Path("data/models/Olmo-3-7B-Instruct")
ATLAS_DIR = Path("data/atlas")
RESULTS_DIR = Path("data/experiments/kernel_routing")

D_MODEL = 4096
N_BOUNDARY = 256
N_FIBER = 16
N_LAYERS = 32


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# =============================================================================
# Mask Strategies
# =============================================================================

class MaskStrategy(Enum):
    HARD_WINDOW = "hard_window"
    SOFT_GAUSSIAN = "soft_gaussian"
    VERTEX_MODULATED = "vertex_modulated"
    PHASE_MODULATED = "phase_modulated"
    WEDGE_ALIGNED = "wedge_aligned"


def hard_window_mask(h: int, device, dtype, window_size: int = 64, **kw) -> Tensor:
    mask = torch.zeros(N_BOUNDARY, device=device, dtype=dtype)
    for i in range(window_size):
        idx = (h + i - window_size // 2) % N_BOUNDARY
        mask[idx] = 1.0
    return mask


def soft_gaussian_mask(h: int, device, dtype, sigma: float = 32.0, **kw) -> Tensor:
    idx = torch.arange(N_BOUNDARY, device=device, dtype=torch.float32)
    dist = torch.minimum((idx - h) % N_BOUNDARY, (h - idx) % N_BOUNDARY).float()
    mask = torch.exp(-0.5 * (dist / max(sigma, 1e-6)) ** 2)
    mask = mask / (mask.mean() + 1e-8)
    return mask.to(dtype)


def vertex_modulated_mask(h: int, device, dtype, sigma: float = 32.0, kernel=None, **kw) -> Tensor:
    chi = kernel.current_vertex if kernel else 0
    s = sigma * {0: 0.5, 1: 0.75, 2: 1.0, 3: 1.5}.get(chi, 1.0)
    return soft_gaussian_mask(h, device, dtype, sigma=s)


def phase_modulated_mask(h: int, device, dtype, sigma: float = 32.0, kernel=None, **kw) -> Tensor:
    p = kernel.current_phase if kernel else 0
    return soft_gaussian_mask((h + p * 16) % N_BOUNDARY, device, dtype, sigma=sigma)


def wedge_aligned_mask(h: int, device, dtype, sigma: float = 32.0, kernel=None, **kw) -> Tensor:
    base = soft_gaussian_mask(h, device, dtype, sigma=sigma)
    cw = h // 64
    ww = torch.ones(N_BOUNDARY, device=device, dtype=dtype)
    for i in range(N_BOUNDARY):
        wd = min(abs(i // 64 - cw), 4 - abs(i // 64 - cw))
        ww[i] = {0: 1.0, 1: 0.7}.get(wd, 0.3)
    return base * ww


MASK_FNS = {
    MaskStrategy.HARD_WINDOW: hard_window_mask,
    MaskStrategy.SOFT_GAUSSIAN: soft_gaussian_mask,
    MaskStrategy.VERTEX_MODULATED: vertex_modulated_mask,
    MaskStrategy.PHASE_MODULATED: phase_modulated_mask,
    MaskStrategy.WEDGE_ALIGNED: wedge_aligned_mask,
}


# =============================================================================
# Semantic Byte Methods
# =============================================================================

class SemanticByteMethod(Enum):
    SIGN_PROBE = "sign_probe"
    HORIZON_TOPK = "horizon_topk"
    HORIZON_ENERGY = "horizon_energy"
    FIBER_PATTERN = "fiber_pattern"


def byte_sign_probe(x: Tensor) -> int:
    xf = x.float().reshape(-1)
    n = xf.numel()
    if n == 0:
        return 0
    stride = max(1, n // 8)
    bits = sum((1 << i) for i in range(8) if xf[min(i * stride, n - 1)].item() > 0)
    return bits & 0xFF


def byte_horizon_topk(x: Tensor) -> int:
    xf = x.float().reshape(-1, N_BOUNDARY, N_FIBER)
    hm = xf.mean(dim=(0, 2))
    tk = torch.topk(hm.abs(), k=8).indices.sort().values
    bits = sum((1 << i) for i, idx in enumerate(tk.cpu().numpy()) if hm[idx].item() > 0)
    return bits & 0xFF


def byte_horizon_energy(x: Tensor) -> int:
    xf = x.float().reshape(-1, N_BOUNDARY, N_FIBER)
    he = (xf ** 2).sum(dim=(0, 2))
    he = he / (he.max() + 1e-8)
    bins = (he * 7.99).long().clamp(0, 7)
    bits = sum((bins[i * 32].item() & 1) << i for i in range(8))
    return bits & 0xFF


def byte_fiber_pattern(x: Tensor) -> int:
    xf = x.float().reshape(-1, N_BOUNDARY, N_FIBER)
    he = (xf ** 2).sum(dim=(0, 2))
    mh = he.argmax().item()
    fv = xf[:, mh, :].mean(dim=0)
    bits = sum((1 << i) for i in range(8) if fv[min(i * 2, N_FIBER - 1)].item() > 0)
    return bits & 0xFF


BYTE_FNS = {
    SemanticByteMethod.SIGN_PROBE: byte_sign_probe,
    SemanticByteMethod.HORIZON_TOPK: byte_horizon_topk,
    SemanticByteMethod.HORIZON_ENERGY: byte_horizon_energy,
    SemanticByteMethod.FIBER_PATTERN: byte_fiber_pattern,
}


# =============================================================================
# Layer Strategies
# =============================================================================

class LayerStrategy(Enum):
    EVERY_NTH = "every_nth"
    EARLY_ONLY = "early_only"
    DEPTH_ADAPTIVE = "depth_adaptive"
    ALL_LAYERS = "all_layers"


def get_routed_layers(strat: LayerStrategy, **kw) -> list[int]:
    if strat == LayerStrategy.EVERY_NTH:
        n = kw.get("n", 4)
        return [i for i in range(N_LAYERS) if i % n == 0]
    if strat == LayerStrategy.EARLY_ONLY:
        return list(range(N_LAYERS // 3))
    if strat == LayerStrategy.DEPTH_ADAPTIVE:
        layers = []
        for i in range(N_LAYERS):
            if i < 8 and i % 2 == 0:
                layers.append(i)
            elif 8 <= i < 20 and i % 4 == 0:
                layers.append(i)
            elif i >= 20 and i % 8 == 0:
                layers.append(i)
        return layers
    if strat == LayerStrategy.ALL_LAYERS:
        return list(range(N_LAYERS))
    return []


def window_for_layer(i: int, base: int, strat: str) -> int:
    if strat == "constant":
        return base
    if strat == "depth_decay":
        if i < 8:
            return base
        if i < 16:
            return min(int(base * 1.5), N_BOUNDARY)
        if i < 24:
            return min(base * 2, N_BOUNDARY)
        return min(base * 3, N_BOUNDARY)
    if strat == "depth_grow":
        if i < 8:
            return min(base * 2, N_BOUNDARY)
        if i < 16:
            return min(int(base * 1.5), N_BOUNDARY)
        return base
    return base


# =============================================================================
# Routed MLP
# =============================================================================

class BoundaryRoutedMLP(nn.Module):
    def __init__(self, mlp, kernel, mask_strat, byte_method, window, sigma):
        super().__init__()
        self.mlp = mlp
        self.kernel = kernel
        self.mask_fn = MASK_FNS[mask_strat]
        self.byte_fn = BYTE_FNS[byte_method]
        self.window = window
        self.sigma = sigma

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        dtype = x.dtype
        xf = x.reshape(-1, D_MODEL).view(-1, N_BOUNDARY, N_FIBER)
        h = self.kernel.current_horizon

        mask = self.mask_fn(
            h=h, device=x.device, dtype=dtype,
            window_size=self.window, sigma=self.sigma, kernel=self.kernel
        ).view(1, N_BOUNDARY, 1)

        xm = (xf * mask).view(-1, D_MODEL)
        gate = F.silu(self.mlp.gate_proj(xm))
        up = self.mlp.up_proj(xm)
        out = self.mlp.down_proj(gate * up)

        self.kernel.step_byte(self.byte_fn(out))
        return out.view(*shape[:-1], D_MODEL)


# =============================================================================
# Experiment Config
# =============================================================================

@dataclass
class Config:
    name: str
    mask: MaskStrategy = MaskStrategy.SOFT_GAUSSIAN
    byte_method: SemanticByteMethod = SemanticByteMethod.HORIZON_TOPK
    layers: LayerStrategy = LayerStrategy.EVERY_NTH
    layer_n: int = 4
    window_strat: str = "constant"
    base_window: int = 64
    base_sigma: float = 32.0
    seed: int = 42
    gen_tokens: int = 30


# =============================================================================
# Runner
# =============================================================================

class Runner:
    def __init__(self, model, tokenizer, prompts: list[str]):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.device = next(model.parameters()).device

    def _install(self, cfg: Config, kernel: RouterKernel) -> dict[int, nn.Module]:
        routed = set(get_routed_layers(cfg.layers, n=cfg.layer_n))
        originals = {}
        for i, layer in enumerate(self.model.model.layers):
            if i in routed:
                w = window_for_layer(i, cfg.base_window, cfg.window_strat)
                s = cfg.base_sigma * (w / max(cfg.base_window, 1))
                originals[i] = layer.mlp
                layer.mlp = BoundaryRoutedMLP(
                    layer.mlp, kernel, cfg.mask, cfg.byte_method, w, s
                )
        return originals

    def _restore(self, originals: dict[int, nn.Module]):
        for i, mlp in originals.items():
            self.model.model.layers[i].mlp = mlp

    def run(self, cfg: Config) -> dict[str, Any]:
        set_seed(cfg.seed)
        routed_set = set(get_routed_layers(cfg.layers, n=cfg.layer_n))

        # --- Similarity ---
        prompt = self.prompts[0]
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            full_hs = self.model(ids, output_hidden_states=True).hidden_states

        kernel = RouterKernel(atlas_dir=ATLAS_DIR)
        orig = self._install(cfg, kernel)
        try:
            with torch.no_grad():
                routed_hs = self.model(ids, output_hidden_states=True).hidden_states
        finally:
            self._restore(orig)

        cos_all, cos_routed, cos_unrouted = [], [], []
        for i in range(N_LAYERS):
            c = F.cosine_similarity(
                full_hs[i + 1].float().reshape(-1),
                routed_hs[i + 1].float().reshape(-1),
                dim=0
            ).item()
            cos_all.append(c)
            (cos_routed if i in routed_set else cos_unrouted).append(c)

        avg_all = float(np.mean(cos_all))
        avg_routed = float(np.mean(cos_routed)) if cos_routed else 1.0
        avg_unrouted = float(np.mean(cos_unrouted)) if cos_unrouted else 1.0
        cos_final = cos_all[-1]

        # --- Generation ---
        gens = []
        t_full_total, t_routed_total = 0.0, 0.0

        for prompt in self.prompts:
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

            set_seed(cfg.seed)
            t0 = time.perf_counter()
            with torch.no_grad():
                out_f = self.model.generate(
                    ids, max_new_tokens=cfg.gen_tokens,
                    do_sample=True, temperature=0.7, top_k=40,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            t_full = time.perf_counter() - t0
            t_full_total += t_full
            txt_f = self.tokenizer.decode(out_f[0], skip_special_tokens=True)

            kernel = RouterKernel(atlas_dir=ATLAS_DIR)
            orig = self._install(cfg, kernel)
            try:
                set_seed(cfg.seed)
                t0 = time.perf_counter()
                with torch.no_grad():
                    out_r = self.model.generate(
                        ids, max_new_tokens=cfg.gen_tokens,
                        do_sample=True, temperature=0.7, top_k=40,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                t_routed = time.perf_counter() - t0
                t_routed_total += t_routed
                txt_r = self.tokenizer.decode(out_r[0], skip_special_tokens=True)
            finally:
                self._restore(orig)

            gens.append({"prompt": prompt, "full": txt_f, "routed": txt_r})

        speedup = t_full_total / max(t_routed_total, 1e-9)

        return {
            "name": cfg.name,
            "routed_layers": len(routed_set),
            "avg_cos_all": avg_all,
            "avg_cos_routed": avg_routed,
            "avg_cos_unrouted": avg_unrouted,
            "cos_final": cos_final,
            "t_full": t_full_total,
            "t_routed": t_routed_total,
            "speedup": speedup,
            "generations": gens,
        }


# =============================================================================
# Experiments
# =============================================================================

def get_experiments() -> list[Config]:
    exps = []

    # Baselines
    for m in [MaskStrategy.HARD_WINDOW, MaskStrategy.SOFT_GAUSSIAN]:
        exps.append(Config(f"baseline_{m.value}", mask=m))

    # Sigma sweep
    for s in [16, 24, 32, 48, 64]:
        exps.append(Config(f"sigma_{s}", base_sigma=float(s)))

    # Layer strategies
    for ls in [LayerStrategy.EARLY_ONLY, LayerStrategy.DEPTH_ADAPTIVE, LayerStrategy.ALL_LAYERS]:
        exps.append(Config(f"layers_{ls.value}", layers=ls))
    for n in [2, 4, 8]:
        exps.append(Config(f"every_{n}", layers=LayerStrategy.EVERY_NTH, layer_n=n))

    # Kernel observables
    for m in [MaskStrategy.VERTEX_MODULATED, MaskStrategy.PHASE_MODULATED, MaskStrategy.WEDGE_ALIGNED]:
        exps.append(Config(f"obs_{m.value}", mask=m))

    # Semantic byte
    for b in SemanticByteMethod:
        exps.append(Config(f"byte_{b.value}", byte_method=b))

    # Window strategies
    for ws in ["constant", "depth_decay", "depth_grow"]:
        exps.append(Config(f"window_{ws}", window_strat=ws, layers=LayerStrategy.ALL_LAYERS))

    return exps


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("KERNEL ROUTING EXPERIMENTS")
    print("=" * 80)

    if not MODEL_DIR.exists() or not ATLAS_DIR.exists():
        print("Error: model or atlas not found")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_DIR.name}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, dtype=torch.bfloat16, device_map="cpu"
    )
    model.eval()

    prompts = [
        "The purpose of good governance is",
        "Mathematics reveals that",
        "In three dimensions, the structure",
    ]

    runner = Runner(model, tokenizer, prompts)
    experiments = get_experiments()

    print(f"\nRunning {len(experiments)} experiments...\n")

    results = []
    for i, cfg in enumerate(experiments):
        print(f"[{i+1}/{len(experiments)}] {cfg.name}")
        try:
            r = runner.run(cfg)
            results.append(r)

            print(f"  Layers routed: {r['routed_layers']}")
            print(f"  Cos(routed): {r['avg_cos_routed']:.4f}  Cos(all): {r['avg_cos_all']:.4f}  Cos(final): {r['cos_final']:.4f}")
            print(f"  Time: full={r['t_full']:.1f}s routed={r['t_routed']:.1f}s speedup={r['speedup']:.2f}x")
            print(f"  Sample: {r['generations'][0]['routed'][:70]}...")
            print()
        except Exception as e:
            print(f"  FAILED: {e}\n")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY (ranked by avg_cos_routed)")
    print("=" * 80)

    hdr = f"{'Experiment':<30} {'#Layers':>7} {'CosRtd':>8} {'CosAll':>8} {'CosFin':>8} {'Speed':>7}"
    print(hdr)
    print("-" * 80)

    for r in sorted(results, key=lambda x: -x["avg_cos_routed"]):
        print(
            f"{r['name']:<30} {r['routed_layers']:>7} "
            f"{r['avg_cos_routed']:>8.4f} {r['avg_cos_all']:>8.4f} "
            f"{r['cos_final']:>8.4f} {r['speedup']:>6.2f}x"
        )

    # --- Save text summary ---
    summary_path = RESULTS_DIR / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("KERNEL ROUTING EXPERIMENTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(hdr + "\n")
        f.write("-" * 80 + "\n")
        for r in sorted(results, key=lambda x: -x["avg_cos_routed"]):
            f.write(
                f"{r['name']:<30} {r['routed_layers']:>7} "
                f"{r['avg_cos_routed']:>8.4f} {r['avg_cos_all']:>8.4f} "
                f"{r['cos_final']:>8.4f} {r['speedup']:>6.2f}x\n"
            )
        f.write("\n\nBEST GENERATIONS:\n")
        f.write("-" * 80 + "\n")
        if results:
            best = max(results, key=lambda x: x["avg_cos_routed"])
            f.write(f"Config: {best['name']}\n\n")
            for g in best["generations"]:
                f.write(f"Prompt: {g['prompt']}\n")
                f.write(f"Full:   {g['full']}\n")
                f.write(f"Routed: {g['routed']}\n\n")

    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()