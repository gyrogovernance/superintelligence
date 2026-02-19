from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

sys.path.insert(0, os.getcwd())
from src.router.kernel import RouterKernel

MODEL_DIR = Path("data/models/Olmo-3-7B-Instruct")
ATLAS_DIR = Path("data/atlas")

# 2^n × 3^m structure
D_MODEL = 4096
N_BOUNDARY = 256
N_FIBER = 16
N_LAYERS = 32


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def circular_dist(idx: Tensor, center: int, n: int) -> Tensor:
    return torch.minimum((idx - center) % n, (center - idx) % n).float()


@dataclass
class DynMaskParams:
    base_sigma: float = 48.0          # your experiments: 48/64 were best
    phase_shift: int = 8              # shift center by p*8
    vertex_scale: tuple[float, float, float, float] = (0.75, 0.9, 1.0, 1.25)


class TokenCoupler:
    """
    Aggregates per-layer observations into one token byte.
    Prevents per-layer kernel stepping cancellation.
    """
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.byte_acc: int = 0
        self.count: int = 0

    def add(self, *, layer_idx: int, b: int) -> None:
        # Layer-salt ensures that if b repeats across layers,
        # the aggregated byte still changes.
        salt = (layer_idx * 29 + 17) & 0xFF
        self.byte_acc ^= (int(b) ^ salt) & 0xFF
        self.count += 1

    def commit(self, kernel: RouterKernel) -> int:
        b = self.byte_acc & 0xFF
        kernel.step_byte(b)
        return b


def byte_from_fiber_pattern(out_flat: Tensor) -> int:
    """
    Stable semantic byte (similar to your best experiment 'byte_fiber_pattern'):
    - reshape to [N,256,16]
    - take horizon with max energy
    - take signs of 8 fiber coordinates (0,2,4,...,14)
    """
    with torch.no_grad():
        xf = out_flat.float().view(-1, N_BOUNDARY, N_FIBER)
        h_energy = (xf * xf).sum(dim=(0, 2))              # [256]
        h = int(torch.argmax(h_energy).item())
        v = xf[:, h, :].mean(dim=0)                       # [16]
        bits = 0
        for i in range(8):
            if v[i * 2].item() > 0:
                bits |= (1 << i)
        return bits & 0xFF


class DynamicRoutedMLP(nn.Module):
    """
    Wraps layer.mlp. Uses kernel state to mask boundary axis dynamically.
    Does NOT step kernel itself; it reports a byte to TokenCoupler.
    """

    def __init__(
        self,
        mlp: nn.Module,
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

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        dtype = x.dtype
        device = x.device

        # Read kernel observables
        h = int(self.kernel.current_horizon)
        chi = int(self.kernel.current_vertex)
        p = int(self.kernel.current_phase)

        sigma = float(self.params.base_sigma) * float(self.params.vertex_scale[chi & 3])
        h_center = (h + (p * int(self.params.phase_shift))) % N_BOUNDARY

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
        self.coupler.add(layer_idx=self.layer_idx, b=b)

        return out_flat.view(*shape[:-1], D_MODEL)


class DynamicRouter:
    """
    Installs dynamic MLP wrappers on a chosen layer schedule.
    """

    def __init__(self, model: AutoModelForCausalLM, kernel: RouterKernel, params: DynMaskParams):
        self.model = model
        self.kernel = kernel
        self.params = params
        self.coupler = TokenCoupler()
        self.original_mlps: dict[int, nn.Module] = {}
        self.routed_layers: list[int] = []

    def schedule_layers(self) -> list[int]:
        # Same schedule you used, but explicit
        out: list[int] = []
        for i in range(N_LAYERS):
            if i < 8 and i % 2 == 0:
                out.append(i)
            elif 8 <= i < 20 and i % 4 == 0:
                out.append(i)
            elif i >= 20 and i % 8 == 0:
                out.append(i)
        return out

    def install(self) -> None:
        layers = self.schedule_layers()
        self.routed_layers = layers
        for i, layer in enumerate(self.model.model.layers):
            if i in layers:
                self.original_mlps[i] = layer.mlp
                layer.mlp = DynamicRoutedMLP(
                    mlp=layer.mlp,
                    kernel=self.kernel,
                    coupler=self.coupler,
                    layer_idx=i,
                    params=self.params,
                )
        print(f"Installed dynamic routing on {len(layers)} layers: {layers}")

    def restore(self) -> None:
        for i, mlp in self.original_mlps.items():
            self.model.model.layers[i].mlp = mlp
        self.original_mlps.clear()
        self.routed_layers.clear()

    def begin_token(self) -> None:
        self.coupler.reset()

    def end_token(self) -> int:
        return self.coupler.commit(self.kernel)


def generate_dynamic(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    router: DynamicRouter,
    prompt: str,
    max_new_tokens: int,
    seed: int,
) -> tuple[str, dict[str, list[int]]]:
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    traj = {"h": [], "chi": [], "p": [], "b_token": [], "tok": []}

    for t in range(max_new_tokens):
        router.begin_token()

        with torch.inference_mode():
            out = model(input_ids)
            logits = out.logits  # [1, seq, vocab]

        # Commit one kernel step per generated token
        b_token = router.end_token()

        next_logits = logits[0, -1] / 0.7
        probs = F.softmax(next_logits, dim=-1)
        next_tok = int(torch.multinomial(probs, 1).item())

        traj["h"].append(router.kernel.current_horizon)
        traj["chi"].append(router.kernel.current_vertex)
        traj["p"].append(router.kernel.current_phase)
        traj["b_token"].append(b_token)
        traj["tok"].append(next_tok)

        next_tensor = torch.tensor([[next_tok]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, next_tensor], dim=-1)

        if (t + 1) % 10 == 0:
            hh = np.array(traj["h"][-10:], dtype=np.int64)
            cc = np.array(traj["chi"][-10:], dtype=np.int64)
            bb = np.array(traj["b_token"][-10:], dtype=np.int64)
            print(
                f"[{t+1:3}] h_mean={hh.mean():6.1f} h_std={hh.std():5.1f} "
                f"chi_dist={np.bincount(cc, minlength=4).tolist()} "
                f"b_unique={len(set(bb.tolist()))}"
            )

        # EOS handling (tokenizer eos is fine for this test)
        if tokenizer.eos_token_id is not None and next_tok == int(tokenizer.eos_token_id):
            break

    text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
    return text, traj


def analyze_traj(traj: dict[str, list[int]]) -> None:
    h = np.array(traj["h"], dtype=np.int64)
    chi = np.array(traj["chi"], dtype=np.int64)
    p = np.array(traj["p"], dtype=np.int64)
    b = np.array(traj["b_token"], dtype=np.int64)

    print("\nKERNEL TRAJ SUMMARY")
    print("-------------------")
    print(f"steps={len(h)} unique_h={len(np.unique(h))}/256 unique_b={len(np.unique(b))}/256")
    print(f"chi_dist={np.bincount(chi, minlength=4).tolist()}  p_dist={np.bincount(p, minlength=4).tolist()}")
    if len(h) > 1:
        dh = np.diff(h)
        jump = np.minimum(np.abs(dh), 256 - np.abs(dh))
        print(f"mean_jump={jump.mean():.2f} max_jump={jump.max()} static={(jump==0).mean():.2%}")


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
        print("\n" + "=" * 80)
        print("PROMPT:", prompt)

        set_seed(0)
        with torch.inference_mode():
            base_ids = tok(prompt, return_tensors="pt").input_ids
            base_out = model.generate(
                base_ids,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                top_k=40,
                pad_token_id=tok.pad_token_id,
            )
        baseline = tok.decode(base_out[0].tolist(), skip_special_tokens=True)
        print("\nBaseline:\n", baseline)

        # Dynamic
        kernel = RouterKernel(atlas_dir=ATLAS_DIR)
        params = DynMaskParams(base_sigma=48.0)  # your sweep: 48/64 were strongest
        router = DynamicRouter(model, kernel, params)
        router.install()

        try:
            set_seed(0)
            routed, traj = generate_dynamic(model, tok, router, prompt, max_new_tokens=30, seed=0)
        finally:
            router.restore()

        print("\nRouted:\n", routed)
        analyze_traj(traj)


if __name__ == "__main__":
    main()
