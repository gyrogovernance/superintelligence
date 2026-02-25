"""
Embeddings Adaptor (lossless Walsh port of byte embeddings).

Takes the 256 base byte embedding rows, reindexes them into intron
coordinates (x = b ^ 0xAA), applies a 256-point Walsh-Hadamard transform
along the byte axis, and saves the Walsh coefficients as a reusable
artifact:

    data/cache/blomo_port/analysis/embeddings_adaptor.npz

This is mathematically exact (invertible) and does not change model
behavior by itself. It just gives a Router-native coordinate system
for the byte embedding table and a future compression knob.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Allow running as script from repo root: ensure blomo_port (parent) is on sys.path
_this_dir = Path(__file__).resolve().parent
_blomo_port = _this_dir.parent
if str(_blomo_port) not in sys.path:
    sys.path.insert(0, str(_blomo_port))

from common import PROJECT_ROOT, load_bolmo


ANALYSIS_DIR = PROJECT_ROOT / "data" / "cache" / "blomo_port" / "analysis"


def _wht_axis(a: np.ndarray, axis: int) -> np.ndarray:
    """Unnormalized Walsh-Hadamard transform along axis (length must be 256)."""
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


@dataclass
class EmbeddingsAdaptor:
    """Container for Walsh coefficients of the 256 base byte embeddings."""
    offset: int
    gene_mic_s: int  # intron mask (XOR with 0xAA)
    d_model: int
    walsh_coeffs: np.ndarray  # [256, d_model], float32

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            offset=np.int32(self.offset),
            gene_mic_s=np.int32(self.gene_mic_s),
            d_model=np.int32(self.d_model),
            walsh_coeffs=self.walsh_coeffs.astype(np.float32),
        )


def build_embeddings_adaptor(model: Any, tokenizer: Any) -> EmbeddingsAdaptor:
    """
    Extract the 256 base byte embedding rows and compute their Walsh spectrum
    in intron coordinates (x = b ^ 0xAA).
    """
    le = model.model.local_encoder
    weight = le.byte_embedding.weight.detach().cpu().numpy()  # [512, d_model]
    offset = int(getattr(tokenizer, "offset", 4))

    base = weight[offset : offset + 256]  # [256, d_model], b in 0..255
    d_model = int(base.shape[1])

    # Reindex into intron coordinates: x = b ^ 0xAA
    perm = np.arange(256, dtype=np.int64) ^ 0xAA
    base_intron = base[perm]  # [256, d_model], row index is x

    # Walsh-Hadamard along the byte axis (axis 0)
    walsh = _wht_axis(base_intron, axis=0).astype(np.float32)

    return EmbeddingsAdaptor(
        offset=offset,
        gene_mic_s=0xAA,
        d_model=d_model,
        walsh_coeffs=walsh,
    )


def main() -> EmbeddingsAdaptor | None:
    print("=" * 10)
    print("EMBEDDINGS ADAPTOR")
    print("=" * 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = PROJECT_ROOT / "data" / "models" / "Bolmo-1B"

    print("\nLoading model and tokenizer...")
    model, tokenizer = load_bolmo(model_dir, device)

    adaptor = build_embeddings_adaptor(model, tokenizer)

    out_path = ANALYSIS_DIR / "embeddings_adaptor.npz"
    adaptor.save(out_path)
    print(f"\nSaved embeddings adaptor to: {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1024:.1f} KB")
    print(f"  d_model: {adaptor.d_model}")

    # Optional sanity check: reconstruct and measure max abs error
    print("\nReconstruction sanity check...")
    # Reconstruct base_intron from walsh (inverse WHT)
    walsh = adaptor.walsh_coeffs.astype(np.float64)
    recon_intron = _wht_axis(walsh, axis=0) / (256.0)  # inverse up to scale

    le = model.model.local_encoder
    weight = le.byte_embedding.weight.detach().cpu().numpy()
    offset = adaptor.offset
    base = weight[offset : offset + 256].astype(np.float64)
    perm = np.arange(256, dtype=np.int64) ^ adaptor.gene_mic_s
    base_intron = base[perm]

    err = np.max(np.abs(recon_intron - base_intron))
    print(f"  max |recon - original_intron| = {err:.6e}")

    print("\n" + "=" * 10)
    print("DONE")
    print("=" * 10)
    return adaptor


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result is not None else 1)

