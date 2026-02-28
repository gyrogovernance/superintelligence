"""
Standalone runner for the Router-native ASI model.

Supports two modes:
  1. Single-operator: router_operator.npz (90-dim full features)
  2. Multi-rate: fast_operator.npz + slow_operator.npz
     + candidate_scorer.npz (256-way candidate evaluator)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import torch

from src.tools.layers import create_default_four_layers
from src.tools.adaptors.state_encoder import (
    RouterFeatureBuilder,
    FastFeatureBuilder,
    SlowFeatureBuilder,
    walsh_expand,
)
from src.tools.adaptors.bolmo_vocab import BolmoVocabSpec
from src.tools.adaptors.candidate_features import candidate_features
from src.tools.tuning.router_operator import RouterOperator
from src.tools.tuning.operators import CompiledOperator, MultiRateOperator
from src.tools.tuning.inference_utils import boundary_prob_given_byte


# ─── UTF-8 Text Prior ─────────────────────────────────────────────────

def build_utf8_text_prior(strength: float = 5.0) -> torch.Tensor:
    """
    Build a 256-dim log-prior that biases toward valid English UTF-8 bytes.

    This encodes the structural fact that English text uses specific byte ranges.
    Bolmo was trained on English text, so its predictions heavily favor these ranges.
    The Router's operators don't encode this prior strongly enough after short-probe
    training, so we add it explicitly.

    strength=0 means no bias. strength=5 is moderate. strength=20 is very strong.
    """
    prior = torch.zeros(256, dtype=torch.float32)

    # Printable ASCII (space through tilde): very common in English
    for b in range(0x20, 0x7F):
        prior[b] = strength

    # Tab (0x09), newline (0x0A), carriage return (0x0D)
    prior[0x09] = strength * 0.5
    prior[0x0A] = strength * 0.8
    prior[0x0D] = strength * 0.3

    # UTF-8 continuation bytes and multi-byte starters: allow but don't prefer
    # (These appear in non-ASCII UTF-8 characters)
    for b in range(0x80, 0xC0):  # continuation bytes
        prior[b] = -strength * 0.5
    for b in range(0xC0, 0xE0):  # 2-byte starters
        prior[b] = -strength * 0.3
    for b in range(0xE0, 0xF0):  # 3-byte starters
        prior[b] = -strength * 0.5
    for b in range(0xF0, 0x100):  # 4-byte starters
        prior[b] = -strength * 0.8

    # Null and low control chars: strongly discourage
    for b in range(0x00, 0x09):
        prior[b] = -strength * 2.0
    for b in range(0x0B, 0x0D):
        prior[b] = -strength * 2.0
    for b in range(0x0E, 0x20):
        prior[b] = -strength * 1.5

    return prior


# ─── Candidate Scorer ─────────────────────────────────────────────────

class CandidateScorer:
    """Trained scorer: score(state, byte) = theta @ psi(state, byte)."""

    def __init__(self, theta: np.ndarray):
        self.theta = theta

    @classmethod
    def load(cls, path: Path) -> "CandidateScorer":
        data = np.load(path, allow_pickle=False)
        theta = data["theta"].astype(np.float32)
        return cls(theta=theta)

    def score(self, state24: int, O: int, E: int, parity: int, byte: int) -> float:
        psi = candidate_features(state24, O, E, parity, byte)
        return float(np.dot(self.theta, psi))


# ─── Single-operator inference ────────────────────────────────────────

def _run_inference_single(
    l3_path: Path,
    operator_path: Path,
    prompt: str,
    max_bytes: int = 64,
    temperature: float = 0.7,
    debug: bool = False,
    utf8_prior_strength: float = 5.0,
) -> None:
    op = RouterOperator.load(operator_path)
    W = op.W
    D = int(W.shape[1])
    vocab = BolmoVocabSpec()
    utf8_prior = build_utf8_text_prior(utf8_prior_strength)

    four = create_default_four_layers(l3_path=l3_path, build_l3_if_missing=False)

    prompt_bytes = prompt.encode("utf-8", errors="replace")
    last_byte = 0xAA
    for b in prompt_bytes:
        four.ingest_byte(b)
        last_byte = int(b) & 0xFF

    print(f"\nPrompt: {prompt!r}")
    print(f"Temperature: {temperature}, UTF-8 prior: {utf8_prior_strength}")
    print("\nGenerating...")

    generated: list[int] = []

    for step in range(max_bytes):
        regs = four.regs
        feat = RouterFeatureBuilder.build_raw(
            regs.l1_state8, regs.l2_state16, regs.l3_state24,
            regs.l4.O, regs.l4.E, regs.l4.parity, last_byte,
        )
        phi = RouterFeatureBuilder.walsh_expand(feat, D)
        logits = op.logits(phi)

        base = logits[vocab.base_start : vocab.base_end_exclusive]
        fused = logits[vocab.fused_start : vocab.fused_end_exclusive]
        byte_logits = torch.logsumexp(torch.stack([base, fused], dim=0), dim=0)

        # Apply UTF-8 text prior
        byte_logits = byte_logits + utf8_prior

        if temperature == 0.0:
            next_byte = int(torch.argmax(byte_logits).item())
        else:
            probs = torch.softmax(byte_logits / float(temperature), dim=0)
            next_byte = int(torch.multinomial(probs, num_samples=1).item())

        if debug:
            topv, topi = torch.topk(byte_logits, k=5)
            top_pairs = [
                f"0x{int(i):02x}({chr(int(i)) if 32 <= int(i) <= 126 else '.'})"
                for i in topi
            ]
            print(f"  step {step}: last=0x{last_byte:02x} top5={top_pairs} -> 0x{next_byte:02x}")

        four.ingest_byte(next_byte)
        last_byte = next_byte
        generated.append(next_byte)

        if next_byte == 0x00:
            break

    out_bytes = bytes(generated)
    decoded = out_bytes.decode("utf-8", errors="replace")
    print(f"\nGenerated {len(generated)} bytes")
    print(f"Raw hex: {' '.join(f'{b:02x}' for b in generated)}")
    print(f"\nDecoded: {decoded}")


# ─── Multi-rate inference ─────────────────────────────────────────────

def _run_inference_multirate(
    l3_path: Path,
    operator_dir: Path,
    prompt: str,
    max_bytes: int = 64,
    temperature: float = 0.7,
    debug: bool = False,
    boundary_threshold: float = 0.90,
    scout_topk: int = 0,
    scout_alpha: float = 0.15,
    scout_sample: bool = False,
    utf8_prior_strength: float = 5.0,
) -> None:
    fast_op = CompiledOperator.load(operator_dir / "fast_operator.npz")
    slow_op = CompiledOperator.load(operator_dir / "slow_operator.npz")
    mrop = MultiRateOperator(fast=fast_op, slow=slow_op)

    D_fast = fast_op.feature_dim
    D_slow = slow_op.feature_dim
    vocab = BolmoVocabSpec()
    utf8_prior = build_utf8_text_prior(utf8_prior_strength)

    scorer_path = operator_dir / "candidate_scorer.npz"
    scorer: CandidateScorer | None = None
    if scorer_path.exists():
        scorer = CandidateScorer.load(scorer_path)
        print(f"Loaded candidate_scorer.npz")

    four = create_default_four_layers(l3_path=l3_path, build_l3_if_missing=False)

    prompt_bytes = prompt.encode("utf-8", errors="replace")
    last_byte = 0xAA
    for b in prompt_bytes:
        four.ingest_byte(b)
        last_byte = int(b) & 0xFF

    regs = four.regs
    phi_s = walsh_expand(
        SlowFeatureBuilder.build(regs.l3_state24, regs.l4.O, regs.l4.E, regs.l4.parity),
        D_slow,
    )
    mrop.update_slow(phi_s)

    print(f"\nPrompt: {prompt!r}")
    print(f"Operator dir: {operator_dir} (multi-rate)")
    print(f"Temperature: {temperature}, UTF-8 prior: {utf8_prior_strength}")
    print(f"Fast dim: {D_fast}, Slow dim: {D_slow}")
    print(f"Scout top-k: {scout_topk}, alpha: {scout_alpha}, sample: {scout_sample}")
    print(f"Boundary threshold: {boundary_threshold}")
    print("\nGenerating...")

    generated: list[int] = []
    boundaries_fired = 0

    for step in range(max_bytes):
        regs = four.regs

        phi_f = walsh_expand(
            FastFeatureBuilder.build(regs.l2_state16, last_byte), D_fast
        )
        logits = mrop.combined_logits(phi_f)

        base = logits[vocab.base_start : vocab.base_end_exclusive]
        fused = logits[vocab.fused_start : vocab.fused_end_exclusive]
        byte_logits = torch.logsumexp(torch.stack([base, fused], dim=0), dim=0)

        # Apply UTF-8 text prior
        byte_logits = byte_logits + utf8_prior

        # ── Byte selection ──
        if scout_topk > 0:
            _, top_indices = torch.topk(byte_logits, min(scout_topk, 256))
            candidates = [int(i.item()) for i in top_indices]

            if scorer is not None:
                cand_logits = torch.tensor(
                    [float(byte_logits[b].item()) for b in candidates],
                    dtype=torch.float32,
                )
                cand_scores = np.array(
                    [scorer.score(regs.l3_state24, regs.l4.O, regs.l4.E, regs.l4.parity, b)
                     for b in candidates],
                    dtype=np.float32,
                )
                combined = cand_logits + float(scout_alpha) * torch.from_numpy(cand_scores)

                if temperature == 0.0 or not scout_sample:
                    next_byte = candidates[int(torch.argmax(combined).item())]
                else:
                    probs = torch.softmax(combined / float(temperature), dim=0)
                    next_byte = candidates[int(torch.multinomial(probs, 1).item())]
            else:
                if temperature == 0.0:
                    next_byte = candidates[0]
                else:
                    cand_logits = torch.tensor([float(byte_logits[b].item()) for b in candidates])
                    probs = torch.softmax(cand_logits / float(temperature), dim=0)
                    next_byte = candidates[int(torch.multinomial(probs, 1).item())]
        elif temperature == 0.0:
            next_byte = int(torch.argmax(byte_logits).item())
        else:
            probs = torch.softmax(byte_logits / temperature, dim=0)
            next_byte = int(torch.multinomial(probs, 1).item())

        # ── Boundary detection ──
        p_boundary = boundary_prob_given_byte(base, fused, next_byte)
        is_boundary = p_boundary >= boundary_threshold

        if debug:
            topv, topi = torch.topk(byte_logits, k=5)
            top_pairs = [
                f"0x{int(i):02x}({chr(int(i)) if 32 <= int(i) <= 126 else '.'})"
                for i in topi
            ]
            bnd_marker = f" [BND]" if is_boundary else ""
            print(f"  step {step}: last=0x{last_byte:02x} top5={top_pairs} -> 0x{next_byte:02x}{bnd_marker}")

        four.ingest_byte(next_byte)
        last_byte = next_byte
        generated.append(next_byte)

        if is_boundary:
            regs = four.regs
            phi_s = walsh_expand(
                SlowFeatureBuilder.build(
                    regs.l3_state24, regs.l4.O, regs.l4.E, regs.l4.parity,
                ),
                D_slow,
            )
            mrop.update_slow(phi_s)
            boundaries_fired += 1

        if next_byte == 0x00:
            break

    out_bytes = bytes(generated)
    decoded = out_bytes.decode("utf-8", errors="replace")
    print(f"\nGenerated {len(generated)} bytes, boundaries fired: {boundaries_fired}")
    print(f"Raw hex: {' '.join(f'{b:02x}' for b in generated)}")
    print(f"\nDecoded: {decoded}")


# ─── Dispatch ─────────────────────────────────────────────────────────

def run_inference(
    l3_path: Path,
    operator_path: Path | None,
    operator_dir: Path,
    prompt: str,
    max_bytes: int = 64,
    temperature: float = 0.7,
    debug: bool = False,
    boundary_threshold: float = 0.90,
    scout_topk: int = 0,
    scout_alpha: float = 0.15,
    scout_sample: bool = False,
    utf8_prior_strength: float = 5.0,
) -> None:
    if operator_path is not None and operator_path.exists():
        _run_inference_single(
            l3_path=l3_path, operator_path=operator_path,
            prompt=prompt, max_bytes=max_bytes,
            temperature=temperature, debug=debug,
            utf8_prior_strength=utf8_prior_strength,
        )
        return

    fast_p = operator_dir / "fast_operator.npz"
    slow_p = operator_dir / "slow_operator.npz"
    if fast_p.exists() and slow_p.exists():
        _run_inference_multirate(
            l3_path=l3_path, operator_dir=operator_dir,
            prompt=prompt, max_bytes=max_bytes,
            temperature=temperature, debug=debug,
            boundary_threshold=boundary_threshold,
            scout_topk=scout_topk,
            scout_alpha=scout_alpha,
            scout_sample=scout_sample,
            utf8_prior_strength=utf8_prior_strength,
        )
        return

    single_p = operator_dir / "router_operator.npz"
    if single_p.exists():
        _run_inference_single(
            l3_path=l3_path, operator_path=single_p,
            prompt=prompt, max_bytes=max_bytes,
            temperature=temperature, debug=debug,
            utf8_prior_strength=utf8_prior_strength,
        )
        return

    raise FileNotFoundError(
        "No operator found. Put fast_operator.npz+slow_operator.npz or "
        "router_operator.npz in --operator-dir, or use --operator"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Router-native ASI model")
    parser.add_argument("--operator-dir", type=Path, default=Path("data/router_operator"))
    parser.add_argument("--operator", type=Path, default=None)
    parser.add_argument("--l3", type=Path, default=Path("data/layers/l3_packed_u24.bin"))
    parser.add_argument("--prompt", type=str, default="Hello, world!")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--max-bytes", type=int, default=64)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--boundary-threshold", type=float, default=0.90)
    parser.add_argument("--scout-topk", type=int, default=0)
    parser.add_argument("--scout-alpha", type=float, default=0.15)
    parser.add_argument("--scout-sample", action="store_true")
    parser.add_argument("--utf8-prior", type=float, default=5.0,
                        help="Strength of UTF-8 text prior (0=off, 5=moderate, 20=strong)")
    args = parser.parse_args()

    l3 = args.l3 if args.l3.is_absolute() else _root / args.l3
    op_path = None
    if args.operator is not None:
        op_path = args.operator if args.operator.is_absolute() else _root / args.operator
    op_dir = args.operator_dir if args.operator_dir.is_absolute() else _root / args.operator_dir

    run_inference(
        l3_path=l3, operator_path=op_path, operator_dir=op_dir,
        prompt=args.prompt, max_bytes=args.max_bytes,
        temperature=args.temp, debug=args.debug,
        boundary_threshold=args.boundary_threshold,
        scout_topk=args.scout_topk,
        scout_alpha=args.scout_alpha,
        scout_sample=args.scout_sample,
        utf8_prior_strength=args.utf8_prior,
    )


if __name__ == "__main__":
    main()