# === secret_lab_ignore/agent/adaptor_build.py ===
"""
Build a GyroAdaptor from OLMo static weights.

v4.1: intron/byte-xor sensitivity (correct for [12,8] mask code).

We measure 8 independent directions corresponding to delta bytes:
  δ_i = 1 << i  for i=0..7
Pairs are always valid: h' = h XOR δ_i.

For each operator and layer:
  - build per-input-boundary "output magnitude profile"
  - measure mean ||profile[h] - profile[h XOR δ]|| for each δ
  - record coherence = mean/std
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoModelForCausalLM

# Robust paths (independent of current working directory)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR_DEFAULT = DATA_DIR / "models" / "Olmo-3-7B-Instruct"
OUT_DEFAULT = DATA_DIR / "adaptor.npz"

sys.path.insert(0, str(PROJECT_ROOT))

from src.router.constants import mask12_for_byte, vertex_charge_from_mask

OP_NAMES = ("up_proj", "gate_proj", "down_proj")


def _profiles_from_W(
    W: NDArray[np.float32],
    nb_out: int, nf_out: int,
    nb_in: int, nf_in: int,
) -> NDArray[np.float32]:
    """
    Build profiles[h_in, :] from W by using magnitude (no sign cancellation).

    W reshaped to T[h_out, c_out, h_in, c_in].
    For each (h_out,c_out,h_in), collapse c_in by L2:
        P[h_out,c_out,h_in] = sqrt(sum_{c_in} T^2)
    Then profiles[h_in] = flatten over (h_out,c_out).
    """
    T = W.reshape(nb_out, nf_out, nb_in, nf_in).astype(np.float32, copy=False)
    P = np.sqrt(np.sum(T * T, axis=3, dtype=np.float32))  # [nb_out,nf_out,nb_in]
    profiles = np.transpose(P, (2, 0, 1)).reshape(nb_in, nb_out * nf_out)  # [nb_in, nb_out*nf_out]
    return profiles.astype(np.float32, copy=False)


def _delta_sensitivity(
    profiles: NDArray[np.float32],
    delta: int,
) -> tuple[float, float, int]:
    """
    Mean ||profiles[h]-profiles[h^delta]|| over unique pairs (h < h^delta).
    Coherence = mean/std.
    """
    idx = np.arange(256, dtype=np.int32)
    j = idx ^ int(delta)
    keep = idx < j
    a = profiles[idx[keep]]
    b = profiles[j[keep]]
    diff = a - b
    norms = np.linalg.norm(diff, axis=1)
    mean = float(norms.mean()) if norms.size else 0.0
    std = float(norms.std()) if norms.size else 0.0
    coherence = mean / (std + 1e-8) if norms.size else 0.0
    return mean, coherence, int(keep.sum())


def _vertex_flow_matrix(
    W: NDArray[np.float32],
    nb_out: int, nf_out: int,
    nb_in: int, nf_in: int,
) -> NDArray[np.float32]:
    """
    K4 vertex-to-vertex coarse flow using Frobenius magnitude on boundary blocks.
    """
    # Build mask_table once via byte labels (boundary indices are bytes)
    mask_table = np.array([mask12_for_byte(b) for b in range(256)], dtype=np.uint16)
    v_out = np.array([vertex_charge_from_mask(int(mask_table[h])) for h in range(256)], dtype=np.int32)
    v_in = np.array([vertex_charge_from_mask(int(mask_table[h])) for h in range(256)], dtype=np.int32)

    T = W.reshape(nb_out, nf_out, nb_in, nf_in).astype(np.float32, copy=False)
    E = np.sqrt(np.sum(T * T, axis=(1, 3), dtype=np.float32))  # [nb_out, nb_in]

    V = np.zeros((4, 4), dtype=np.float32)
    for a in range(4):
        out_mask = (v_out == a)
        for b in range(4):
            in_mask = (v_in == b)
            V[a, b] = float(E[np.ix_(out_mask, in_mask)].sum())
    return V


def build_adaptor(model_dir: Path, output_path: Path, layers: list[int] | None = None) -> None:
    print("=" * 60)
    print("  GyroAdaptor Builder v4.1 (intron/byte-xor sensitivity)")
    print("=" * 60)

    t0 = time.time()

    print(f"\nLoading model from {model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,  # avoids the torch_dtype deprecation warning
    )

    n_layers = len(model.model.layers)
    d_model = int(model.config.hidden_size)
    mlp_dim = int(model.model.layers[0].mlp.up_proj.weight.shape[0])

    nb = 256
    nf_hidden = d_model // nb
    nf_mlp = mlp_dim // nb
    K = nf_hidden

    print(f"  d_model={d_model}, mlp_dim={mlp_dim}")
    print(f"  nb={nb}, nf_hidden={nf_hidden}, nf_mlp={nf_mlp}, K={K}")
    print(f"  n_layers={n_layers}")

    if layers is None:
        layers = list(range(n_layers))
    print(f"  Processing {len(layers)} layers")

    # 8 independent delta directions in byte space
    deltas = np.array([1 << i for i in range(8)], dtype=np.uint8)

    # Operator shapes
    OP_SHAPES = {
        "up_proj":   (nb, nf_mlp, nb, nf_hidden),   # [11008,4096]
        "gate_proj": (nb, nf_mlp, nb, nf_hidden),
        "down_proj": (nb, nf_hidden, nb, nf_mlp),   # [4096,11008]
    }

    n_proc = len(layers)
    sens = np.zeros((n_proc, len(OP_NAMES), 8), dtype=np.float32)
    coher = np.zeros((n_proc, len(OP_NAMES), 8), dtype=np.float32)
    pair_counts = np.zeros((n_proc, len(OP_NAMES), 8), dtype=np.int32)
    vertex_flow = np.zeros((n_proc, len(OP_NAMES), 4, 4), dtype=np.float32)

    for li, layer_idx in enumerate(layers):
        layer = model.model.layers[layer_idx]
        print(f"\n  Layer {layer_idx}:")

        for oi, op in enumerate(OP_NAMES):
            W = getattr(layer.mlp, op).weight.detach().float().numpy()
            nb_out, nf_out, nb_in, nf_in = OP_SHAPES[op]

            profiles = _profiles_from_W(W, nb_out, nf_out, nb_in, nf_in)

            for di, d in enumerate(deltas):
                m, c, n_pairs = _delta_sensitivity(profiles, int(d))
                sens[li, oi, di] = np.float32(m)
                coher[li, oi, di] = np.float32(c)
                pair_counts[li, oi, di] = int(n_pairs)

            vertex_flow[li, oi] = _vertex_flow_matrix(W, nb_out, nf_out, nb_in, nf_in)

            # Quick summary per operator
            mean_s = float(np.mean(sens[li, oi]))
            mean_c = float(np.mean(coher[li, oi]))
            print(f"    {op:10s}: mean_sens={mean_s:.4f}  mean_coh={mean_c:.4f}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving adaptor to {output_path}...")
    np.savez(
        output_path,
        adaptor_version="4.1",
        model_name=str(model_dir.name),
        nb=np.int32(nb),
        nf_hidden=np.int32(nf_hidden),
        nf_mlp=np.int32(nf_mlp),
        K=np.int32(K),
        n_layers=np.int32(n_layers),
        processed_layers=np.array(layers, dtype=np.int32),
        op_names=np.array(OP_NAMES),
        deltas=deltas,                      # [8]
        intron_sensitivity=sens,            # [n_layers, 3, 8]
        intron_coherence=coher,             # [n_layers, 3, 8]
        intron_pair_counts=pair_counts,     # [n_layers, 3, 8]
        vertex_flow=vertex_flow,            # [n_layers, 3, 4, 4]
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Adaptor size: {output_path.stat().st_size:,} bytes ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # Summary
    mean_dir = np.mean(sens, axis=(0, 1))  # [8]
    print("\nMean intron-direction sensitivities (delta bytes):")
    for i, d in enumerate(deltas.tolist()):
        print(f"  delta=0x{d:02x} (bit {i}): {float(mean_dir[i]):.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build GyroAdaptor v4.1 (intron/byte-xor sensitivity)")
    p.add_argument("--model", type=Path, default=MODEL_DIR_DEFAULT)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--layers", type=str, default=None, help="Comma-separated layer indices, or 'all'")
    args = p.parse_args()

    layers = None
    if args.layers and args.layers != "all":
        layers = [int(x.strip()) for x in args.layers.split(",")]

    build_adaptor(args.model, args.out, layers)


if __name__ == "__main__":
    main()
