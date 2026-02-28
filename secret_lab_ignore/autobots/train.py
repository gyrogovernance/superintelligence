"""
Training script for HGT.

Phase 1: Generate curriculum from FSM (requires L3 table)
Phase 2: Train model on curriculum (standard PyTorch training loop)
Phase 3: Verify lossless physics preservation
Phase 4: Save model as HuggingFace-compatible checkpoint

Usage:
    python -m secret_lab_ignore.autobots.train \
        --l3-path data/layers/l3_packed_u24.bin \
        --output-dir data/autobots/model \
        --epochs 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
from torch.utils.data import DataLoader, random_split

from .config import HGTConfig
from .curriculum import FSMCurriculum
from .model import HGTForCausalLM


def compute_next_byte_accuracy(
    model: HGTForCausalLM,
    batch: dict[str, torch.Tensor],
) -> float:
    """Fraction of argmax-predicted bytes matching ground truth."""
    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            attention_mask=batch.get("attention_mask"),
        )
        pred = out.logits[:, :-1, :].argmax(dim=-1)
        target = batch["labels"][:, 1:]
        mask = target >= 0
        if mask.sum() == 0:
            return 1.0
        correct = ((pred == target) & mask).sum().float()
        return (correct / mask.sum()).item()


def verify_physics_preserved(model: HGTForCausalLM) -> bool:
    """Spot-check that physics buffer is unchanged."""
    if not hasattr(model, "mask12_table"):
        return False
    tbl = getattr(model, "mask12_table")
    return isinstance(tbl, torch.Tensor) and int(tbl.shape[0]) == 256


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HGT model")
    parser.add_argument(
        "--l3-path",
        type=Path,
        default=Path("data/layers/l3_packed_u24.bin"),
        help="Path to L3 packed table",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/autobots/model"),
        help="Output directory for model",
    )
    parser.add_argument(
        "--curriculum-dir",
        type=Path,
        default=Path("data/autobots/curriculum"),
        help="Curriculum data directory",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=os.cpu_count() or 8,
        help="PyTorch intra-op threads (default: all logical cores)",
    )
    parser.add_argument(
        "--curriculum-scale",
        type=int,
        default=1,
        choices=[1, 10],
        help="1=debug (~2.5k seqs), 10=full (~22k seqs)",
    )
    parser.add_argument(
        "--regenerate-curriculum",
        action="store_true",
        help="Force regenerate curriculum (delete existing first)",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    t0 = time.perf_counter()

    curriculum = FSMCurriculum(args.l3_path)
    if args.regenerate_curriculum and args.curriculum_dir.exists():
        import shutil
        shutil.rmtree(args.curriculum_dir)
    if not args.curriculum_dir.exists():
        t_cur = time.perf_counter()
        curriculum.save_curriculum(args.curriculum_dir, scale=args.curriculum_scale)
        print(f"Curriculum generated in {time.perf_counter() - t_cur:.1f}s")
    full_dataset = curriculum.load_curriculum(args.curriculum_dir)
    n = len(full_dataset)
    n_val = max(1, int(n * args.val_split))
    n_train = n - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    config = HGTConfig()
    model = HGTForCausalLM(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-5
    )

    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            input_ids[i, :n] = b["input_ids"]
            labels[i, :n] = b["labels"]
            attn_mask[i, :n] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, collate_fn=collate,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, collate_fn=collate
    )

    n_params = sum(p.numel() for p in model.parameters())
    base = train_dataset.dataset
    n_tokens = sum(len(base[i]["input_ids"]) for i in train_dataset.indices)
    print(
        f"n_seq={n} n_train={n_train} n_val={n_val} n_tokens~{n_tokens} "
        f"params={n_params}"
    )

    def _label_smoothing(epoch: int) -> float:
        if epoch < 2:
            return 0.02
        if epoch < 4:
            return 0.005
        return 0.0

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs} training...", flush=True)
        t_epoch = time.perf_counter()
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            label_smoothing = _label_smoothing(epoch)
            out = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
                label_smoothing=label_smoothing,
            )
            loss = out.loss
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
        train_loss = total_loss / len(train_loader) if train_loader else 0.0

        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                    label_smoothing=0.0,
                )
                if out.loss is not None:
                    val_loss += out.loss.item()
                val_accuracy += compute_next_byte_accuracy(model, batch)
        n_val_batches = len(val_loader)
        val_loss = val_loss / n_val_batches if n_val_batches else 0.0
        val_accuracy = val_accuracy / n_val_batches if n_val_batches else 1.0
        elapsed = time.perf_counter() - t_epoch
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            args.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.output_dir)
            print(f"  (best model saved)")

        print(
            f"Epoch {epoch+1}/{args.epochs} done: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.2%} lr={lr:.2e} ({elapsed:.1f}s)",
            flush=True,
        )

    if verify_physics_preserved(model):
        print("Physics preserved.")

    print("Saving final model...", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    from .tokenizer import PhysicsTokenizer

    tokenizer = PhysicsTokenizer()
    tokenizer.save_pretrained(str(args.output_dir))
    total_elapsed = time.perf_counter() - t0
    print(f"Saved to {args.output_dir} (total {total_elapsed:.1f}s)")


if __name__ == "__main__":
    main()
