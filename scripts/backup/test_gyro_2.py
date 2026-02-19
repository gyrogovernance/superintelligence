from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

sys.path.insert(0, os.getcwd())
from src.router.kernel import RouterKernel

MODEL_DIR = Path("data/models/Olmo-3-7B-Instruct")
ATLAS_DIR = Path("data/atlas")

# 2^n × 3^m frame
D_MODEL = 4096          # 2^12
N_BOUNDARY = 256        # 2^8
N_FIBER = 16            # 2^4
SIGMA_CHOICES = (32.0, 48.0, 64.0)  # 2^5, 2^4*3, 2^6


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def circular_dist(idx: Tensor, center: int, n: int) -> Tensor:
    return torch.minimum((idx - center) % n, (center - idx) % n).float()


def byte_from_fiber_pattern(out_flat: Tensor) -> int:
    """
    Stable byte extractor (your best-performing family):
    - reshape to [N,256,16]
    - pick horizon with max energy
    - take sign bits of fiber indices 0,2,4,...,14
    """
    with torch.no_grad():
        xf = out_flat.float().view(-1, N_BOUNDARY, N_FIBER)
        h_energy = (xf * xf).sum(dim=(0, 2))
        h = int(torch.argmax(h_energy).item())
        v = xf[:, h, :].mean(dim=0)  # [16]

        bits = 0
        for i in range(8):
            if v[i * 2].item() > 0:
                bits |= (1 << i)
        return bits & 0xFF


class TokenCoupler:
    """
    Aggregate per-layer bytes into one per-token byte.
    XOR + layer salt to avoid per-layer cancellation.
    """
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.acc: int = 0
        self.n: int = 0

    def add(self, layer_idx: int, b: int) -> None:
        salt = (layer_idx * 29 + 17) & 0xFF
        self.acc ^= ((int(b) & 0xFF) ^ salt)
        self.n += 1

    def commit(self, kernel: RouterKernel) -> int:
        b = self.acc & 0xFF
        kernel.step_byte(b)
        return b


@dataclass(frozen=True)
class DynMaskParams:
    phase_shift: int = 8  # phase->center shift: h + p*8


def choose_sigma(chi: int, p: int) -> float:
    """
    Discrete sigma choice in the 2^n×3^m family.
    Uses kernel observables only.
    """
    # Simple, deterministic: index by (chi,p) in Z4×Z4 → {0,1,2}
    idx = (int(chi) + 2 * int(p)) % len(SIGMA_CHOICES)
    return float(SIGMA_CHOICES[idx])


class DynamicRoutedMLP(nn.Module):
    """
    Wraps layer.mlp. Applies a kernel-driven boundary mask.
    Reports bytes to coupler; does not step kernel directly.
    """
    def __init__(
        self,
        mlp: nn.Module,
        *,
        kernel: RouterKernel,
        coupler: TokenCoupler,
        layer_idx: int,
        params: DynMaskParams,
    ):
        super().__init__()
        self.mlp = mlp
        self.kernel = kernel
        self.coupler = coupler
        self.layer_idx = int(layer_idx)
        self.params = params

        # diagnostics
        self.last_sigma: float = float("nan")

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        dtype = x.dtype
        device = x.device

        h = int(self.kernel.current_horizon)
        chi = int(self.kernel.current_vertex)
        p = int(self.kernel.current_phase)

        sigma = choose_sigma(chi, p)
        self.last_sigma = sigma

        h_center = (h + p * int(self.params.phase_shift)) % N_BOUNDARY

        idx = torch.arange(N_BOUNDARY, device=device, dtype=torch.float32)
        dist = circular_dist(idx, h_center, N_BOUNDARY)
        mask = torch.exp(-0.5 * (dist / max(sigma, 1e-6)) ** 2)
        mask = mask / (mask.mean() + 1e-8)
        mask = mask.to(dtype).view(1, N_BOUNDARY, 1)

        x_bf = x.reshape(-1, D_MODEL).view(-1, N_BOUNDARY, N_FIBER)
        x_masked = (x_bf * mask).view(-1, D_MODEL)

        gate = F.silu(self.mlp.gate_proj(x_masked))
        up = self.mlp.up_proj(x_masked)
        out_flat = self.mlp.down_proj(gate * up)

        b = byte_from_fiber_pattern(out_flat)
        self.coupler.add(self.layer_idx, b)

        return out_flat.view(*shape[:-1], D_MODEL)


def full_attention_layers_from_config(model: Any) -> list[int]:
    """
    OLMo3 config.json contains layer_types with 'full_attention' markers.
    Use that as the routing lattice (principled).
    """
    lt = getattr(model.config, "layer_types", None)
    if not isinstance(lt, list):
        # fallback to known pattern (still principled): every 4th starting at 3
        return [3, 7, 11, 15, 19, 23, 27, 31]
    out = [i for i, t in enumerate(lt) if str(t) == "full_attention"]
    return out if out else [3, 7, 11, 15, 19, 23, 27, 31]


class DynamicRouter:
    def __init__(self, model: AutoModelForCausalLM, kernel: RouterKernel, params: DynMaskParams):
        self.model = model
        self.kernel = kernel
        self.params = params
        self.coupler = TokenCoupler()
        self.original_mlps: dict[int, nn.Module] = {}
        self.routed_mlps: dict[int, DynamicRoutedMLP] = {}
        self.routed_layers: list[int] = []

    def install(self) -> None:
        layers = full_attention_layers_from_config(self.model)
        self.routed_layers = layers

        for i, layer in enumerate(self.model.model.layers):
            if i in layers:
                self.original_mlps[i] = layer.mlp
                wrapped = DynamicRoutedMLP(
                    layer.mlp,
                    kernel=self.kernel,
                    coupler=self.coupler,
                    layer_idx=i,
                    params=self.params,
                )
                self.routed_mlps[i] = wrapped
                layer.mlp = wrapped

        print(f"Installed dynamic routing on {len(layers)} layers: {layers}")

    def restore(self) -> None:
        for i, mlp in self.original_mlps.items():
            self.model.model.layers[i].mlp = mlp
        self.original_mlps.clear()
        self.routed_mlps.clear()
        self.routed_layers.clear()

    def begin_token(self) -> None:
        self.coupler.reset()

    def end_token(self) -> int:
        return self.coupler.commit(self.kernel)

    def sigma_stats(self) -> tuple[float, dict[float, int]]:
        sigmas = [m.last_sigma for m in self.routed_mlps.values() if np.isfinite(m.last_sigma)]
        if not sigmas:
            return float("nan"), {}
        counts: dict[float, int] = {}
        for s in sigmas:
            counts[float(s)] = counts.get(float(s), 0) + 1
        return float(np.mean(sigmas)), counts


def entropy(counts: np.ndarray) -> float:
    p = counts.astype(np.float64)
    p = p / max(p.sum(), 1.0)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


@torch.inference_mode()
def generate_stepwise_cached(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    router: DynamicRouter | None,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_k: int = 40,
) -> tuple[str, dict[str, list[int]], float]:
    """
    Token-by-token generation using past_key_values if supported.

    If router is provided, it commits one kernel byte per generated token.
    """
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Detect cache support (no silent fallback: if absent, we raise with message)
    past = None
    use_cache = True

    traj = {"h": [], "chi": [], "p": [], "b": [], "tok": []}

    t0 = time.perf_counter()
    for t in range(max_new_tokens):
        if router is not None:
            router.begin_token()

        out = model(input_ids, use_cache=use_cache, past_key_values=past)
        logits = out.logits
        past = getattr(out, "past_key_values", None)
        if past is None:
            raise RuntimeError("Model did not return past_key_values; cache not supported in this mode.")

        if router is not None:
            b = router.end_token()
        else:
            b = -1

        next_logits = logits[0, -1] / float(temperature)
        if top_k > 0:
            topv, topi = torch.topk(next_logits, k=top_k)
            probs = torch.softmax(topv, dim=-1)
            next_tok = int(topi[torch.multinomial(probs, 1)].item())
        else:
            probs = torch.softmax(next_logits, dim=-1)
            next_tok = int(torch.multinomial(probs, 1).item())

        if router is not None:
            traj["h"].append(router.kernel.current_horizon)
            traj["chi"].append(router.kernel.current_vertex)
            traj["p"].append(router.kernel.current_phase)
            traj["b"].append(int(b))
        traj["tok"].append(next_tok)

        # feed only the new token next step
        input_ids = torch.tensor([[next_tok]], device=device, dtype=input_ids.dtype)

        if tokenizer.eos_token_id is not None and next_tok == int(tokenizer.eos_token_id):
            break

        if (t + 1) % 10 == 0 and router is not None:
            hh = np.array(traj["h"][-10:], dtype=np.int64)
            cc = np.array(traj["chi"][-10:], dtype=np.int64)
            bb = np.array(traj["b"][-10:], dtype=np.int64)
            print(
                f"[{t+1:3}] h_mean={hh.mean():6.1f} h_std={hh.std():5.1f} "
                f"chi_dist={np.bincount(cc, minlength=4).tolist()} "
                f"b_unique={len(set(bb.tolist()))}"
            )

    dt = time.perf_counter() - t0
    # reconstruct full text by concatenating prompt + generated tokens
    # (we kept only last token in input_ids, so use traj tokens)
    full_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"] + traj["tok"]
    text = tokenizer.decode(full_ids, skip_special_tokens=True)
    return text, traj, dt


def main() -> None:
    if not MODEL_DIR.exists() or not ATLAS_DIR.exists():
        print("Model or atlas not found.")
        return

    print("Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()

    prompts = [
        "The purpose of good governance is",
        "Mathematics reveals that",
        "In three dimensions, the structure",
    ]

    for prompt in prompts:
        print("\n" + "=" * 90)
        print("PROMPT:", prompt)

        set_seed(0)
        base_text, _, base_dt = generate_stepwise_cached(
            model, tok, router=None, prompt=prompt, max_new_tokens=30
        )
        print("\nBaseline:")
        print(base_text)
        print(f"(baseline) time={base_dt:.2f}s  tok/s={30/max(base_dt,1e-9):.2f}")

        kernel = RouterKernel(atlas_dir=ATLAS_DIR)
        router = DynamicRouter(model, kernel, DynMaskParams(phase_shift=8))
        router.install()

        try:
            set_seed(0)
            routed_text, traj, routed_dt = generate_stepwise_cached(
                model, tok, router=router, prompt=prompt, max_new_tokens=30
            )
        finally:
            router.restore()

        print("\nRouted:")
        print(routed_text)

        h = np.array(traj["h"], dtype=np.int64)
        chi = np.array(traj["chi"], dtype=np.int64)
        b = np.array(traj["b"], dtype=np.int64)

        h_counts = np.bincount(h, minlength=256) if len(h) else np.zeros(256, dtype=np.int64)
        chi_counts = np.bincount(chi, minlength=4) if len(chi) else np.zeros(4, dtype=np.int64)

        print("\nKernel stats:")
        print(f"  steps={len(h)} unique_h={len(np.unique(h))}/256  H(h)={entropy(h_counts):.2f} bits")
        print(f"  chi_dist={chi_counts.tolist()}  unique_b={len(np.unique(b))}/256")
        print(f"(routed) time={routed_dt:.2f}s  tok/s={30/max(routed_dt,1e-9):.2f}")

        mean_sigma, sigma_counts = router.sigma_stats()
        if sigma_counts:
            print(f"  sigma_mean={mean_sigma:.1f} sigma_counts={sigma_counts}")


if __name__ == "__main__":
    main()
