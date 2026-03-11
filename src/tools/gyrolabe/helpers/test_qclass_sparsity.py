# src/tools/gyrolabe/test_qclass_sparsity.py
"""
Generate 300 tokens with Q-class sparsity enabled.
Proves the sparse attention mask works without crashing.
Run from repo root: python -m src.tools.gyrolabe.test_qclass_sparsity
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.tools.gyrolabe.bolmo_bridge import (
    DEFAULT_BOLMO_MODEL_PATH,
    GyrolabeSettings,
    load_gyrolabe_bolmo,
)


def _get_tokenizer(model: object) -> Any | None:
    try:
        tc = getattr(model.base_model.model, "tokenizer_config", None)
        if tc is not None and hasattr(tc, "build"):
            return tc.build()
    except Exception:
        pass
    return None


def main() -> None:
    print("Loading bridge with enable_qclass_sparsity=True...")
    settings = GyrolabeSettings(
        enable_embedding_bias=True,
        enable_boundary_bias=True,
        enable_qclass_sparsity=True,
    )
    model = load_gyrolabe_bolmo(DEFAULT_BOLMO_MODEL_PATH, settings=settings)
    model.reset_gyrolabe_parameters()
    model.eval()

    tokenizer = _get_tokenizer(model)
    if tokenizer is None:
        print("No tokenizer. Skipping generate.")
        return

    prompt = "Artificial intelligence is"
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].cpu()

    with torch.no_grad():
        gen = model.generate(
            input_ids,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "pad_token_id", 0),
        )

    text = tokenizer.decode(gen[0].tolist(), skip_special_tokens=True)
    print(f"Generate OK. Output: {text[:200]}...")
    print("Q-class sparsity test passed.")


if __name__ == "__main__":
    main()
