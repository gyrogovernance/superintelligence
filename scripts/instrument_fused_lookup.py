"""
Instrumentation script to count fused embedding lookups.

Patches byte_embedding.forward to count how often token IDs fall in the
fused range (offset+256 .. offset+511) vs base (offset .. offset+255).

What we found (Step 1):
- Generation code asserts input_ids < boundary_offset (260), so fused IDs
  are never fed to the embedding layer during generate().
- Running this script reports 0 fused lookups (0.0000%) on instrumented
  generation. So byte_embedding rows 256..511 are DEAD WEIGHT at runtime.
- Implication: we can stop researching fused-delta compression; the model
  can conceptually use a 256-row byte embedding for generation.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow running as script: repo root must be on path for secret_lab_ignore
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch
import numpy as np

from secret_lab_ignore.blomo_port.common import PROJECT_ROOT, bolmo_reset_local_caches, load_bolmo, token_to_byte_and_fused


@dataclass
class EmbeddingLookupStats:
    """Statistics for byte_embedding lookups."""
    base_lookups: int = 0      # token_id in [offset, offset+255]
    fused_lookups: int = 0     # token_id in [offset+256, offset+511]
    special_lookups: int = 0  # token_id < offset
    out_of_range: int = 0       # token_id >= offset+512
    total_tokens: int = 0

    def report(self, offset: int) -> None:
        print("\n[Embedding Lookup Statistics]")
        print(f"  Offset (special tokens): {offset}")
        print(f"  Base range: [{offset}, {offset+255}] -> {self.base_lookups:,} lookups")
        print(f"  Fused range: [{offset+256}, {offset+511}] -> {self.fused_lookups:,} lookups")
        print(f"  Special range: [0, {offset-1}] -> {self.special_lookups:,} lookups")
        print(f"  Out of range: -> {self.out_of_range:,} lookups")
        print(f"  Total: {self.total_tokens:,}")
        if self.total_tokens > 0:
            print(f"\n  FUSED LOOKUP PERCENTAGE: {100*self.fused_lookups/self.total_tokens:.4f}%")
            if self.fused_lookups == 0:
                print("  *** NO FUSED LOOKUPS DETECTED ***")
                print("  This confirms: byte_embedding rows 256..511 are DEAD WEIGHT during generation.")


class FusedLookupInstrumenter:
    """Patches byte_embedding to count fused vs base lookups."""

    def __init__(self, model: Any, offset: int):
        self.model = model
        self.offset = offset
        self.stats = EmbeddingLookupStats()
        self._original_forward = None
        self._hook_handle = None

    def __enter__(self):
        le = self.model.model.local_encoder
        byte_emb = le.byte_embedding
        self._original_forward = byte_emb.forward
        stats = self.stats
        offset = self.offset

        def instrumented_forward(tokens: torch.Tensor) -> torch.Tensor:
            # Count lookups before calling original
            flat_tokens = tokens.view(-1)
            for tid in flat_tokens.cpu().numpy():
                stats.total_tokens += 1
                if tid < offset:
                    stats.special_lookups += 1
                elif tid < offset + 256:
                    stats.base_lookups += 1
                elif tid < offset + 512:
                    stats.fused_lookups += 1
                else:
                    stats.out_of_range += 1

            assert self._original_forward is not None
            return self._original_forward(tokens)

        byte_emb.forward = instrumented_forward
        return self

    def __exit__(self, exc_type, exc, tb):
        le = self.model.model.local_encoder
        if self._original_forward is not None:
            le.byte_embedding.forward = self._original_forward


def run_instrumented_generation(
    model: Any,
    tokenizer: Any,
    prompt: str = "Language modeling is ",
    max_new_tokens: int = 200,
) -> tuple[EmbeddingLookupStats, torch.Tensor]:
    """Run generation with instrumentation; returns (stats, output_ids)."""
    bolmo_reset_local_caches(model)
    torch.manual_seed(42)
    offset = int(getattr(tokenizer, "offset", 4))
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    with FusedLookupInstrumenter(model, offset) as inst:
        out = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_k=0,
        )
    return (inst.stats, out)


def main():
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = PROJECT_ROOT / "data" / "models" / "Bolmo-1B"

    print("=" * 10)
    print("FUSED EMBEDDING LOOKUP INSTRUMENTATION")
    print("Goal: Determine if byte_embedding rows 256..511 are ever used")
    print("=" * 10)

    model, tokenizer = load_bolmo(model_dir, device)
    offset = int(getattr(tokenizer, "offset", 4))

    print(f"\nTokenizer offset: {offset}")
    print(f"  Base token range: [{offset}, {offset+255}]")
    print(f"  Fused token range: [{offset+256}, {offset+511}]")
    print(f"  boundary_offset (offset+256): {offset+256}")

    # Run instrumented generation
    prompt = "The theory of relativity states that "
    max_new = 300

    print(f"\nRunning instrumented generation...")
    print(f"  Prompt: '{prompt}'")
    print(f"  Max new tokens: {max_new}")

    t0 = time.perf_counter()
    stats, out = run_instrumented_generation(model, tokenizer, prompt, max_new)
    dt = time.perf_counter() - t0

    # Report results
    print(f"\nGeneration completed in {dt:.2f}s")
    stats.report(offset)

    # Decode output
    generated_text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"\n[Generated text sample]")
    print(f"  {generated_text[:200]}...")

    # Also check: what tokens were actually generated?
    print("\n[Token ID range check]")
    all_ids = out[0].tolist()
    min_id = min(all_ids)
    max_id = max(all_ids)
    fused_count = sum(1 for tid in all_ids if offset + 256 <= tid < offset + 512)
    print(f"  Min token ID: {min_id}")
    print(f"  Max token ID: {max_id}")
    print(f"  Generated fused tokens: {fused_count}")

    return stats


if __name__ == "__main__":
    import sys
    stats = main()
    sys.exit(0 if stats.fused_lookups == 0 else 1)
