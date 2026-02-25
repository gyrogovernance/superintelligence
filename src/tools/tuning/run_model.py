"""
Standalone runner for the Router-native ASI model.

Pipeline:
  FourLayers (L1-L4 FSM) 
    -> RouterFeatureBuilder (raw 4-layer features)
    -> Walsh expansion
    -> RouterOperator (W, b) 
    -> 520-way logits (Bolmo byte token space)
    -> merge base+fused to 256-way byte distribution
    -> sample next byte
    -> step FourLayers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch

from src.tools.layers import create_default_four_layers
from src.tools.adaptors.state_encoder import RouterFeatureBuilder
from src.tools.adaptors.bolmo_vocab import BolmoVocabSpec
from src.tools.tuning.router_operator import RouterOperator


def run_inference(
    l3_path: Path,
    operator_path: Path,
    prompt: str,
    max_bytes: int = 64,
    temperature: float = 0.7,
    debug: bool = False,
) -> None:
    # 1. Load compiled operator (the "brain")
    op = RouterOperator.load(operator_path)
    W = op.W
    D = int(W.shape[1])  # feature dimension (e.g. 2048)
    vocab = BolmoVocabSpec()

    # 2. Load Router physics (the 4-layer FSM)
    four = create_default_four_layers(l3_path=l3_path, build_l3_if_missing=False)

    # 3. Encode prompt bytes into Router state
    prompt_bytes = prompt.encode("utf-8", errors="replace")
    last_byte = 0xAA  # archetype for "no previous byte"
    for b in prompt_bytes:
        four.ingest_byte(b)
        last_byte = int(b) & 0xFF

    print(f"\nPrompt: {prompt!r}")
    print(f"L3 path: {l3_path}")
    print(f"Operator: {operator_path}")
    print(f"Temperature: {temperature}")
    print("\nResponse (hex + ASCII overview):")

    generated: list[int] = []
    step_count = 0
    non_ascii_count = 0

    for step in range(max_bytes):
        # A. Build raw Router features from full 4-layer state
        regs = four.regs
        feat = RouterFeatureBuilder.build_raw(
            regs.l1_state8,
            regs.l2_state16,
            regs.l3_state24,
            regs.l4.O,
            regs.l4.E,
            regs.l4.parity,
            last_byte,
        )

        # B. Expand to Walsh basis
        phi = RouterFeatureBuilder.walsh_expand(feat, D)  # [D]
        logits = op.logits(phi)  # [520]

        # C. Merge base + fused logits to 256-way byte distribution
        base = logits[vocab.base_start : vocab.base_end_exclusive]      # [256]
        fused = logits[vocab.fused_start : vocab.fused_end_exclusive]   # [256]
        byte_logits = torch.logsumexp(torch.stack([base, fused], dim=0), dim=0)  # [256]

        # D. Sample or greedy
        if temperature == 0.0:
            next_byte = int(torch.argmax(byte_logits).item())
        else:
            probs = torch.softmax(byte_logits / float(temperature), dim=0)
            next_byte = int(torch.multinomial(probs, num_samples=1).item())

        # E. Debug output
        if debug:
            topv, topi = torch.topk(byte_logits, k=5)
            top_bytes = [int(i.item()) for i in topi]
            top_pairs = [
                f"0x{b:02x}({chr(b) if 32 <= b <= 126 else '.'})"
                for b in top_bytes
            ]
            print(
                f"\n  step {step}: last=0x{last_byte:02x} "
                f"top5={top_pairs} -> next=0x{next_byte:02x}",
                end="",
                flush=True,
            )

        # F. Update Router state
        four.ingest_byte(next_byte)
        last_byte = next_byte
        generated.append(next_byte)
        step_count += 1

        # Track ASCII vs non-ASCII
        if not (32 <= next_byte <= 126):
            non_ascii_count += 1

        # Simple stopping condition: null byte as EOS
        if next_byte == 0x00:
            break

    # 5. Final reporting
    out_bytes = bytes(generated)
    decoded = out_bytes.decode("utf-8", errors="replace")

    print("\n")
    print(f"Generated {step_count} bytes, non-ASCII: {non_ascii_count}")
    print(f"Raw hex: {' '.join(f'{b:02x}' for b in generated)}")
    print("\nDecoded UTF-8 (errors='replace'):")
    print(decoded)
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Router-native ASI model")
    parser.add_argument(
        "--operator",
        type=Path,
        default=Path("data/router_operator/router_operator.npz"),
        help="Path to router_operator.npz compiled by SPC",
    )
    parser.add_argument(
        "--l3",
        type=Path,
        default=Path("data/layers/l3_packed_u24.bin"),
        help="Path to L3 packed FSM table",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, world!",
        help="Prompt string to seed the Router state",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=64,
        help="Maximum number of bytes to generate",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-step debug info (top-5 bytes & last state)",
    )
    args = parser.parse_args()

    # Resolve relative paths against project root
    l3 = args.l3 if args.l3.is_absolute() else _root / args.l3
    op_path = args.operator if args.operator.is_absolute() else _root / args.operator

    run_inference(
        l3_path=l3,
        operator_path=op_path,
        prompt=args.prompt,
        max_bytes=args.max_bytes,
        temperature=args.temp,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()