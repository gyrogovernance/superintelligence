#!/usr/bin/env python3
"""Train Super Walsh sector gains to read a Bonsai weight on an activation corpus.

Distills W·x into P_Q(gains)·x (no defect) on real embedding activations.

  python -m src.tools.autoencoder.helpers.weight_read_train
  python -m src.tools.autoencoder.helpers.weight_read_train --epochs 400 --ladder diagonal
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.tools.autoencoder.helpers.weight_read import (  # noqa: E402
    embedding_corpus,
    evaluate_on_corpus,
    export_operative_bundle,
    read_weight,
)
from src.tools.autoencoder.models.super import SpectralAutoencoder, codec_ladder_mask  # noqa: E402
from src.tools.gyroscopic.config import get_gyroscopic_llm_config, resolve_gguf_path  # noqa: E402


@dataclass
class TrainReport:
    tensor: str
    epochs: int
    ladder: str | None
    init_pq_max_rel: float
    final_pq_max_rel: float
    init_proj_energy: float
    final_proj_energy: float
    checkpoint: str


def _load_weight(
    gguf: Path,
    tensor: str,
    weight_cache: Path | None,
) -> np.ndarray:
    from src.tools.autoencoder.helpers.weight_read import bonsai_dequant_tensor

    return bonsai_dequant_tensor(gguf, tensor, cache_npz=weight_cache)


def train_weight_sectors(
    W: np.ndarray,
    corpus: np.ndarray,
    *,
    epochs: int = 300,
    lr: float = 0.05,
    batch_size: int = 32,
    ladder: str | None = None,
    rate: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[SpectralAutoencoder, dict[str, float]]:
    """Adam on bottleneck gains; MSE(W·x, P_Q·x) on activation corpus."""
    torch.manual_seed(seed)
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(corpus, dtype=np.float64)
    if W.shape[1] != X.shape[1]:
        raise ValueError(f"W cols {W.shape[1]} != corpus dim {X.shape[1]}")

    sector_mask = codec_ladder_mask(ladder) if ladder else None
    rep0 = read_weight(W, sector_mask=sector_mask)
    init_stats = evaluate_on_corpus(rep0, W, X, pq_only=True)

    model = SpectralAutoencoder(
        init_gain=1.0,
        sector_mask=sector_mask,
    ).to(device)
    with torch.no_grad():
        model.bottleneck.gain.copy_(
            torch.as_tensor(rep0.gains.astype(np.float32), device=device)
        )

    Wt = torch.as_tensor(W.T, dtype=torch.float32, device=device)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    opt = torch.optim.Adam([model.bottleneck.gain], lr=lr)

    n = Xt.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = Xt[idx]
            y_tgt = xb @ Wt
            y_pred = model.apply_pq_activation(xb)
            loss = F.mse_loss(y_pred, y_tgt)
            if rate > 0:
                loss = loss + rate * model.bottleneck.rate_penalty()
            opt.zero_grad()
            loss.backward()
            opt.step()

    gains = model.bottleneck.block_gains().detach().cpu().numpy().astype(np.float64)
    rep1 = read_weight(W, sector_mask=sector_mask, gains=gains)
    final_stats = evaluate_on_corpus(rep1, W, X, pq_only=True)
    return model, {
        "init_pq_max_rel": init_stats["max_rel_err"],
        "final_pq_max_rel": final_stats["max_rel_err"],
        "init_proj_energy": rep0.proj_energy_ratio,
        "final_proj_energy": rep1.proj_energy_ratio,
        "trained_rep": rep1,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Walsh sector gains for weight read")
    ap.add_argument("--gguf", type=Path, default=None)
    ap.add_argument("--tensor", default="blk.0.attn_q.weight")
    ap.add_argument("--weight-cache", type=Path, default=None)
    ap.add_argument("--corpus-cache", type=Path, default=None)
    ap.add_argument("--corpus-rows", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ladder", default=None, help="codec ladder mask (e.g. diagonal, full)")
    ap.add_argument("--rate", type=float, default=0.0, help="L1 rate on gains")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    gguf = args.gguf or resolve_gguf_path(get_gyroscopic_llm_config())
    if not gguf.is_file():
        print(f"GGUF missing: {gguf}", file=sys.stderr)
        return 2

    weight_cache = args.weight_cache or (
        _REPO / "data" / "checkpoints" / "blk0_attn_q.npz"
    )
    corpus_cache = args.corpus_cache or (
        _REPO / "data" / "checkpoints" / f"emb_corpus_{args.corpus_rows}.npz"
    )

    print(f"Train weight read: {args.tensor}")
    W = _load_weight(gguf, args.tensor, weight_cache)
    print(f"  W {W.shape}")

    rng = np.random.default_rng(args.seed)
    row_idx = rng.integers(0, 151669, size=args.corpus_rows)
    corpus = embedding_corpus(gguf, row_idx, cache_npz=corpus_cache)
    print(f"  corpus {corpus.shape}")

    _, stats = train_weight_sectors(
        W,
        corpus,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        ladder=args.ladder,
        rate=args.rate,
        seed=args.seed,
    )
    rep: object = stats.pop("trained_rep")

    print("-" * 5)
    print(f"  pq_max_rel  {stats['init_pq_max_rel']:.4f} -> {stats['final_pq_max_rel']:.4f}")
    print(f"  proj_energy {stats['init_proj_energy']:.4f} -> {stats['final_proj_energy']:.4f}")

    out = args.out or (
        _REPO / "data" / "checkpoints" / "blk0_attn_q_operative_trained.npz"
    )
    export_operative_bundle(out, rep, tensor_name=args.tensor)
    ckpt = out.with_suffix(".pt")
    torch.save(
        {
            "tensor": args.tensor,
            "gains": rep.gains.astype(np.float32),
            "ladder": args.ladder,
        },
        ckpt,
    )
    report_path = out.with_suffix(".json")
    report = TrainReport(
        tensor=args.tensor,
        epochs=args.epochs,
        ladder=args.ladder,
        init_pq_max_rel=float(stats["init_pq_max_rel"]),
        final_pq_max_rel=float(stats["final_pq_max_rel"]),
        init_proj_energy=float(stats["init_proj_energy"]),
        final_proj_energy=float(stats["final_proj_energy"]),
        checkpoint=str(out.relative_to(_REPO)),
    )
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"  saved {out.relative_to(_REPO)}")
    print(f"  saved {report_path.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
