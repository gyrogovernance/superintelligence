#!/usr/bin/env python3
"""
Manifold Atlas Generator

Generates and saves activation pools and whitening stats for all OLMo layers.
The atlas stores raw activation pools plus mean and std so that downstream
scripts can whiten and project as needed.
"""

import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_DIR = Path("data/models/Olmo-3-7B-Instruct")
OUTPUT_DIR = Path("data/atlas/olmo_3_7b_manifolds")

DEVICE = "cpu"
DTYPE = torch.bfloat16

N_POOL = 2048
SEQ_LEN = 32
BATCH_SIZE = 2

SEED = 23


def extract_all_layers(model, input_ids: torch.Tensor) -> dict[int, torch.Tensor]:
    """
    Extract post-MLP activations for all transformer layers in one forward pass.

    Returns a dict: layer_idx -> (batch*seq_len, hidden_dim) tensor on CPU.
    """
    n_layers = model.config.num_hidden_layers
    layer_acts: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}

    def hook_factory(idx: int):
        def _hook(_module, _inp, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            act = out.detach().to("cpu", torch.float32).reshape(-1, out.shape[-1])
            layer_acts[idx].append(act)
        return _hook

    handles = []
    for i in range(n_layers):
        h = model.model.layers[i].post_feedforward_layernorm.register_forward_hook(
            hook_factory(i)
        )
        handles.append(h)

    with torch.no_grad():
        model(input_ids.to(DEVICE))

    for h in handles:
        h.remove()

    return {i: torch.cat(acts, dim=0) for i, acts in layer_acts.items()}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading OLMo from {MODEL_DIR}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    n_layers = model.config.num_hidden_layers
    print(f"Model hidden: {model.config.hidden_size}  layers: {n_layers}")

    n_seq = (N_POOL // SEQ_LEN) + 1
    lo = 1000
    hi = tokenizer.vocab_size - 1000
    input_ids = torch.randint(lo, hi, (n_seq, SEQ_LEN), dtype=torch.long)

    all_layer_data: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}

    print("Collecting activation pools...")
    for i in tqdm(range(0, n_seq, BATCH_SIZE)):
        batch_ids = input_ids[i : i + BATCH_SIZE]
        batch_acts = extract_all_layers(model, batch_ids)
        for idx, act in batch_acts.items():
            all_layer_data[idx].append(act)

    print("Finalizing and saving manifolds...")
    for idx in range(n_layers):
        pool = torch.cat(all_layer_data[idx], dim=0)[:N_POOL].to(torch.float32)
        mean = pool.mean(dim=0)
        centered = pool - mean
        std = centered.std(dim=0).clamp(min=1e-8)

        np.savez_compressed(
            OUTPUT_DIR / f"layer_{idx}.npz",
            pool=pool.numpy(),
            mean=mean.numpy(),
            std=std.numpy(),
        )
        print(f"  Saved layer {idx}  pool shape {tuple(pool.shape)}")

    print(f"Atlas saved under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()