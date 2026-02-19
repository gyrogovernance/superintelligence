# === secret_lab_ignore/agent/distill_policy.py ===
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from src.router.kernel import RouterKernel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "models" / "Olmo-3-7B-Instruct"
ATLAS_DIR = DATA_DIR / "atlas"

# Output artifacts
POLICY_COUNTS_PATH = DATA_DIR / "policy_counts.npy"      # [65536,256] uint32
TOKEN_UNIGRAM_PATH = DATA_DIR / "token_unigram.npy"      # [vocab] uint32


def iter_token_ids_from_text_file(tokenizer, path: Path, max_lines: int | None):
    """
    Yields token_id sequences per line/document.
    Keeps memory bounded.
    """
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line)
            if len(ids) >= 2:
                yield ids
            n += 1
            if max_lines is not None and n >= max_lines:
                break


def build_policy(
    text_path: Path,
    *,
    max_lines: int | None = None,
    smoothing: float = 0.5,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    vocab = int(tokenizer.vocab_size)

    kernel = RouterKernel(atlas_dir=ATLAS_DIR)

    n_states = int(kernel.ontology.shape[0])  # 65536
    counts = np.zeros((n_states, 256), dtype=np.uint32)
    unigram = np.zeros((vocab,), dtype=np.uint32)

    docs = 0
    toks = 0

    for ids in iter_token_ids_from_text_file(tokenizer, text_path, max_lines):
        docs += 1

        # Reset per document so policy is about “local language continuation”
        kernel.reset()

        # Teacher-forcing on the corpus: learn next-byte distribution
        # We use driving byte = token_id & 0xFF (same “common language” as GyroLabe).
        for t in range(len(ids) - 1):
            tid = int(ids[t])
            tid_next = int(ids[t + 1])

            unigram[tid] += 1
            toks += 1

            state_idx = int(kernel.state_index[0])

            b_next = tid_next & 0xFF
            counts[state_idx, b_next] += 1

            # advance kernel with current token's driving byte
            kernel.step_byte(tid & 0xFF)

        # last token unigram
        unigram[int(ids[-1])] += 1
        toks += 1

        if docs % 1000 == 0:
            print(f"  processed docs={docs:,} tokens≈{toks:,}")

    print(f"\nFinished: docs={docs:,} tokens≈{toks:,}")
    POLICY_COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(POLICY_COUNTS_PATH, counts)
    np.save(TOKEN_UNIGRAM_PATH, unigram)
    print(f"Saved:\n  {POLICY_COUNTS_PATH}\n  {TOKEN_UNIGRAM_PATH}")

    # Optional: also save a normalized float16 policy for fast runtime sampling
    probs = counts.astype(np.float64) + float(smoothing)
    probs /= probs.sum(axis=1, keepdims=True) + 1e-18
    probs_f16 = probs.astype(np.float16)
    out_probs = DATA_DIR / "policy_probs_f16.npy"
    np.save(out_probs, probs_f16)
    print(f"Saved:\n  {out_probs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=Path, required=True, help="Path to a newline-delimited text file")
    ap.add_argument("--max_lines", type=int, default=None)
    ap.add_argument("--smoothing", type=float, default=0.5)
    args = ap.parse_args()

    build_policy(args.text, max_lines=args.max_lines, smoothing=args.smoothing)


if __name__ == "__main__":
    main()
