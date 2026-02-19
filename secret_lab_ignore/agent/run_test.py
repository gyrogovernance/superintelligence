# === secret_lab_ignore/agent/run_test.py ===
"""
Test: Kernel-navigated token selection via boundary field scoring.

Instead of distilling the forward pass, we use:
1. Kernel navigation (which boundary position matters)  
2. Accumulated M field (semantic memory at kernel coordinates)
3. Unembedding table viewed as [256, 16] fields
4. Fiber-level dot products at the kernel's horizon position

This replaces 32 layers of attention+MLP with:
  score(token) = M[h, p, :] · unembedding[token][h, :]

Per-byte token groups (~391 tokens/byte) keep search tractable.
Total computation per token: 256 bytes × ~391 candidates × 16 dims ≈ 1.6M multiplies.
Compare full forward pass: ~billions.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.router.constants import trajectory_parity_commitment
from src.router.kernel import RouterKernel

MODEL_DIR = PROJECT_ROOT / "data" / "models" / "Olmo-3-7B-Instruct"
ATLAS_DIR = PROJECT_ROOT / "data" / "atlas"
EMBEDDING_PATH = PROJECT_ROOT / "data" / "embeddings.npy"
UNEMBED_PATH = PROJECT_ROOT / "data" / "unembeddings.npy"


def extract_unembeddings_if_needed() -> None:
    """Extract lm_head weights (unembedding matrix) from OLMo."""
    if UNEMBED_PATH.exists():
        return

    import torch
    from transformers import AutoModelForCausalLM

    print(f"Extracting unembeddings from {MODEL_DIR}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16,
    )

    # In OLMo, lm_head may be tied to embed_tokens
    if hasattr(model, "lm_head") and model.lm_head is not None:
        U = model.lm_head.weight.detach().float().numpy()
        print(f"  lm_head shape: {U.shape}")
    else:
        # Weight-tied: use embed_tokens
        U = model.model.embed_tokens.weight.detach().float().numpy()
        print(f"  Weight-tied, using embed_tokens: {U.shape}")

    np.save(UNEMBED_PATH, U)
    print(f"  Saved to {UNEMBED_PATH}")
    del model


def extract_embeddings_if_needed() -> None:
    if EMBEDDING_PATH.exists():
        return

    import torch
    from transformers import AutoModelForCausalLM

    print(f"Extracting embeddings from {MODEL_DIR}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16,
    )
    E = model.model.embed_tokens.weight.detach().float().numpy()
    np.save(EMBEDDING_PATH, E)
    print(f"  Saved: {E.shape}")
    del model


def build_byte_to_tokens(vocab_size: int, bytes_per_token: int = 4) -> dict[int, list[int]]:
    """
    Map each byte value (0..255) to the list of token IDs
    whose first byte (big-endian) equals that value.

    For L=4: token_id = b0 << 24 | b1 << 16 | b2 << 8 | b3
    The LAST byte (the one that drives the kernel for the final step) is token_id & 0xFF.
    """
    byte_to_tokens: dict[int, list[int]] = {b: [] for b in range(256)}
    for tid in range(vocab_size):
        last_byte = tid & 0xFF
        byte_to_tokens[last_byte].append(tid)
    return byte_to_tokens


class FieldScorer:
    """
    Score candidate tokens using boundary-field dot products.

    Each token's unembedding vector is viewed as a [256, 16] field.
    Scoring at horizon h uses only the 16-dim fiber at position h:

        score(token, h) = query[16] · unembed_field[token, h, 16]
    """

    def __init__(
        self,
        unembed: np.ndarray,    # [vocab, 4096]
        byte_to_tokens: dict[int, list[int]],
        nb: int = 256,
        nf: int = 16,
    ):
        self.nb = nb
        self.nf = nf
        self.vocab_size = unembed.shape[0]
        self.byte_to_tokens = byte_to_tokens

        # Reshape unembedding table to [vocab, 256, 16]
        assert unembed.shape[1] == nb * nf
        self.unembed_fields = unembed.reshape(self.vocab_size, nb, nf)

        # Precompute per-byte fiber arrays for fast scoring
        # For each byte b and horizon h, we need unembed_fields[tokens_of_b, h, :]
        # We store per-byte: [n_tokens_for_b, 256, 16] — but only index into h at runtime

        # Actually, just store the token lists. Indexing is fast enough.
        self._max_per_byte = max(len(v) for v in byte_to_tokens.values())
        print(f"  Max tokens per byte: {self._max_per_byte}")
        print(f"  Min tokens per byte: {min(len(v) for v in byte_to_tokens.values())}")

    def score_byte(
        self,
        byte: int,
        query: np.ndarray,     # [16] — the accumulated fiber at current horizon
        h: int,                # current horizon position
    ) -> tuple[int, float]:
        """
        Score all tokens sharing this byte, return best (token_id, score).

        score(t) = query · unembed_fields[t, h, :]
        """
        candidates = self.byte_to_tokens[byte]
        if not candidates:
            return 0, -1e10

        # Gather fibers: [n_candidates, 16]
        tids = np.array(candidates, dtype=np.int64)
        fibers = self.unembed_fields[tids, h, :]  # [n, 16]

        # Dot product with query
        scores = fibers @ query  # [n]

        best_idx = int(np.argmax(scores))
        return candidates[best_idx], float(scores[best_idx])

    def score_all_bytes(
        self,
        query: np.ndarray,
        h: int,
        temperature: float = 0.7,
    ) -> tuple[int, int, np.ndarray]:
        """
        Score all 256 bytes, select one, then resolve to best token within that byte.

        Returns (selected_byte, selected_token, byte_scores[256]).
        """
        byte_scores = np.zeros(256, dtype=np.float32)

        for b in range(256):
            candidates = self.byte_to_tokens[b]
            if not candidates:
                byte_scores[b] = -1e10
                continue

            tids = np.array(candidates, dtype=np.int64)
            fibers = self.unembed_fields[tids, h, :]
            scores = fibers @ query
            byte_scores[b] = float(np.max(scores))  # best token score for this byte

        # Softmax selection over bytes
        bs = byte_scores - byte_scores.max()
        bs /= max(temperature, 1e-8)
        probs = np.exp(bs)
        probs /= probs.sum() + 1e-18

        selected_byte = int(np.random.choice(256, p=probs))

        # Resolve to best token within selected byte
        best_token, best_score = self.score_byte(selected_byte, query, h)

        return selected_byte, best_token, byte_scores


def run_test(prompt: str, n_tokens: int = 50, seed: int = 42):
    from transformers import AutoTokenizer

    np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    vocab_size = tokenizer.vocab_size

    print("Loading embeddings and unembeddings as boundary fields...")
    embeddings = np.load(str(EMBEDDING_PATH)).astype(np.float32)
    unembeddings = np.load(str(UNEMBED_PATH)).astype(np.float32)
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Unembeddings: {unembeddings.shape}")

    nb, nf = 256, 16
    embed_fields = embeddings.reshape(vocab_size, nb, nf)

    byte_to_tokens = build_byte_to_tokens(vocab_size)

    scorer = FieldScorer(unembeddings, byte_to_tokens, nb, nf)

    # Kernel
    kernel = RouterKernel(atlas_dir=ATLAS_DIR)

    # M field: [256, 4, 16] — the Inference Function memory
    M = np.zeros((256, 4, nf), dtype=np.float32)

    # Process prompt
    prompt_ids = tokenizer.encode(prompt)
    print(f"\nPrompt: \"{prompt}\"")
    print(f"Prompt tokens ({len(prompt_ids)}): {prompt_ids}")

    for tid in prompt_ids:
        # View embedding as field, accumulate into M at kernel position
        field = embed_fields[tid]  # [256, 16]

        i = int(kernel.state_index[0])
        last_b = int(kernel.last_byte[0]) & 0xFF
        h = int(kernel.state_horizon[i])
        p = int(kernel.phase[i, last_b])

        # Accumulate the fiber at the current horizon into M
        alpha = 0.9
        M[h, p, :] = alpha * M[h, p, :] + (1.0 - alpha) * field[h, :]

        # Step kernel through all bytes of this token
        bs = list(tid.to_bytes(4, "big"))
        for b in bs:
            kernel.step_byte(b)

    sig = kernel.signature()
    print(f"After prompt: step={sig.step}, state=0x{sig.state_hex}")

    # Generate
    print(f"\nGenerating {n_tokens} tokens...")
    generated = []
    genealogy = []
    t0 = time.perf_counter()

    for step in range(n_tokens):
        i = int(kernel.state_index[0])
        last_b = int(kernel.last_byte[0]) & 0xFF
        h = int(kernel.state_horizon[i])
        p = int(kernel.phase[i, last_b])

        # Query = accumulated semantic fiber at current kernel position
        query = M[h, p, :].copy()

        # Normalize query
        qnorm = np.linalg.norm(query)
        if qnorm > 1e-8:
            query = query / qnorm

        # Score all bytes and select token
        sel_byte, sel_token, byte_scores = scorer.score_all_bytes(query, h)

        # Clamp to vocab
        if sel_token >= vocab_size:
            sel_token = 0

        generated.append(sel_token)

        # Accumulate selected token's embedding into M
        if sel_token < embeddings.shape[0]:
            field = embed_fields[sel_token]
            i2 = int(kernel.state_index[0])
            h2 = int(kernel.state_horizon[i2])
            p2 = int(kernel.phase[i2, last_b])
            M[h2, p2, :] = alpha * M[h2, p2, :] + (1.0 - alpha) * field[h2, :]

        # Advance kernel
        bs = list(sel_token.to_bytes(4, "big"))
        for b in bs:
            kernel.step_byte(b)
            genealogy.append(b)

        if step < 5 or step % 10 == 0 or step == n_tokens - 1:
            tok_str = tokenizer.decode([sel_token])
            print(f"  step {step:3d}: byte=0x{sel_byte:02x} token={sel_token:6d} \"{tok_str}\"")

        if tokenizer.eos_token_id is not None and sel_token == tokenizer.eos_token_id:
            print(f"  [EOS at step {step}]")
            break

    elapsed = time.perf_counter() - t0

    # Output
    full_text = tokenizer.decode(prompt_ids + generated, skip_special_tokens=True)
    print(f"\n{'='*60}")
    print("  OUTPUT")
    print(f"{'='*60}")
    print(full_text)

    # Stats
    print(f"\n{'='*60}")
    print("  STATISTICS")
    print(f"{'='*60}")
    O, E, par = trajectory_parity_commitment(genealogy)
    sig = kernel.signature()
    gen_arr = np.array(generated)
    unique = len(np.unique(gen_arr))
    M_energy = float(np.sqrt(np.mean(M ** 2)))
    M_nonzero = int(np.sum(np.abs(M) > 1e-6))

    print(f"  Tokens: {len(generated)}, unique: {unique}")
    print(f"  Rate: {len(generated)/max(elapsed,1e-9):.1f} tok/s")
    print(f"  Kernel: step={sig.step} state=0x{sig.state_hex}")
    print(f"  Parity: O=0x{O:03x} E=0x{E:03x} n%2={par}")
    print(f"  M energy: {M_energy:.6f}")
    print(f"  M coverage: {M_nonzero}/{M.size} ({100*M_nonzero/M.size:.1f}%)")


def main():
    extract_embeddings_if_needed()
    extract_unembeddings_if_needed()

    prompts = [
        "The purpose of good governance is",
        "In mathematics, a group is",
        "The relationship between economy and ecology",
    ]

    for prompt in prompts:
        run_test(prompt, n_tokens=50, seed=42)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
