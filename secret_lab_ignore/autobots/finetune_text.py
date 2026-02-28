"""
Text fine-tuning pipeline for HGT.

Loads FSM-pretrained weights and fine-tunes on UTF-8 text.
Freezes physics buffer (mask12_table), trains only neural weights.
Uses standard CLM loss.

Usage:
    python -m secret_lab_ignore.autobots.finetune_text \
        --model-dir data/autobots/model \
        --data-path data/text_corpus.txt \
        --output-dir data/autobots/finetuned \
        --epochs 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
from torch.utils.data import Dataset

from .config import HGTConfig
from .model import HGTForCausalLM
from .tokenizer import PhysicsTokenizer


class UTF8ByteDataset(Dataset):
    """Dataset of UTF-8 byte sequences from text via PhysicsTokenizer."""

    def __init__(self, text: str, seq_len: int = 128):
        self.tokenizer = PhysicsTokenizer()
        encoded = self.tokenizer.encode_with_physics(text)
        self.bytes_list = encoded["input_ids"]
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.bytes_list) - self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx
        end = min(idx + self.seq_len + 1, len(self.bytes_list))
        chunk = self.bytes_list[start:end]
        real_len = min(self.seq_len, max(0, end - start - 1))
        pad_len = self.seq_len + 1 - len(chunk)
        if pad_len > 0:
            chunk = chunk + [0] * pad_len
        ids = torch.tensor(chunk[: self.seq_len + 1], dtype=torch.long)
        attn_mask = torch.zeros(self.seq_len, dtype=torch.long)
        attn_mask[:real_len] = 1
        input_ids = ids[: self.seq_len]
        labels = ids[1 : self.seq_len + 1].clone()
        labels[real_len:] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn_mask,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune HGT on UTF-8 text")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=os.cpu_count() or 8,
        help="PyTorch intra-op threads (default: all logical cores)",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    model = HGTForCausalLM.from_pretrained(args.model_dir)

    if args.data_path and args.data_path.exists():
        text = args.data_path.read_text(encoding="utf-8", errors="replace")
    else:
        text = "Hello world. " * 500

    dataset = UTF8ByteDataset(text, seq_len=args.seq_len)
    def collate(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
            )
            if out.loss is not None:
                out.loss.backward()
                optimizer.step()
                total_loss += out.loss.item()
        n = len(loader)
        print(f"Epoch {epoch + 1} loss: {total_loss / n:.4f}" if n else "No batches")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer = PhysicsTokenizer()
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
