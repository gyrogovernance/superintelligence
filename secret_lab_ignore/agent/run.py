# === secret_lab_ignore/agent/run.py ===
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer

from secret_lab_ignore.agent.runtime_policy import PolicyConfig, PolicyRunner

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "models" / "Olmo-3-7B-Instruct"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

    cfg = PolicyConfig(
        atlas_dir=str(DATA_DIR / "atlas"),
        model_dir=str(MODEL_DIR),
        policy_probs_path=str(DATA_DIR / "policy_probs_f16.npy"),
        token_unigram_path=str(DATA_DIR / "token_unigram.npy"),
        vocab_size=int(tokenizer.vocab_size),
        seed=42,
        top_tokens_per_byte=64,
    )

    runner = PolicyRunner(cfg)

    prompt = "The purpose of good governance is"
    prompt_ids = tokenizer.encode(prompt)

    # Prime kernel with prompt driving bytes (same as distillation)
    for tid in prompt_ids:
        runner.kernel.step_byte(int(tid) & 0xFF)

    out = []
    for _ in range(80):
        b = runner.step_byte()
        tid = runner.byte_to_token(b)
        out.append(tid)
        if tokenizer.eos_token_id is not None and tid == tokenizer.eos_token_id:
            break

    print(tokenizer.decode(prompt_ids + out, skip_special_tokens=True))


if __name__ == "__main__":
    main()
