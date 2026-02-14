#!/usr/bin/env python3
# scripts/run_gyrolabe.py
"""
GyroLabe runner with full physics diagnostics.

Modes:
  Default:    Runs predefined prompts with baseline vs coordinated
  --chat:     Interactive loop with coordination active

Usage:
    python scripts/run_gyrolabe.py
    python scripts/run_gyrolabe.py --chat
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.gyrolabe import (
    GyroLabe,
    CouplingConfig,
    GenerationResult,
    detect_device,
    choose_dtype,
    generate,
)

logging.getLogger("transformers").setLevel(logging.ERROR)

DEFAULT_MODEL = Path("data/models/Olmo-3-7B-Instruct")
DEFAULT_ATLAS = Path("data/atlas")

PROMPTS = [
    "The purpose of good governance is",
    "Mathematics reveals that",
    "In three dimensions, the structure",
]


def circular_mean(angles_uint8: Sequence[int], n_bins: int = 256) -> float:
    if len(angles_uint8) == 0:
        return float("nan")
    thetas = np.array(angles_uint8, dtype=np.float64) * (2 * np.pi / n_bins)
    C = np.mean(np.cos(thetas))
    S = np.mean(np.sin(thetas))
    mu = np.arctan2(S, C) % (2 * np.pi)
    return mu * n_bins / (2 * np.pi)


def circular_std(angles_uint8: Sequence[int], n_bins: int = 256) -> float:
    if len(angles_uint8) == 0:
        return float("nan")
    thetas = np.array(angles_uint8, dtype=np.float64) * (2 * np.pi / n_bins)
    C = np.mean(np.cos(thetas))
    S = np.mean(np.sin(thetas))
    R = np.sqrt(C ** 2 + S ** 2)
    R = min(R, 1.0 - 1e-15)
    sigma_rad = np.sqrt(-2.0 * np.log(R))
    return sigma_rad * n_bins / (2 * np.pi)


@dataclass
class CoordinationDiagnostics:
    """Comprehensive diagnostics with physics metrics."""

    kernel_step: int = 0
    kernel_state_hex: str = "0000"
    kernel_a_hex: str = "0000"
    kernel_b_hex: str = "0000"

    unique_h: int = 0
    h_entropy: float = 0.0

    chi_dist: list[int] = field(default_factory=list)

    unique_bytes: int = 0
    b_entropy: float = 0.0
    mean_byte_weight: float = 0.0
    byte_weight_hist: list[int] = field(default_factory=list)

    charge_counts: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    charge_fracs: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    n_layers_routed: int = 0
    n_fibers: int = 0
    
    mean_dist_to_mu: float = float("nan")
    std_dist_to_mu: float = float("nan")
    
    mean_dist_to_qturn: float = float("nan")
    std_dist_to_qturn: float = float("nan")
    
    mean_code_dist: float = float("nan")
    std_code_dist: float = float("nan")

    layer_h_peaks: list[int] = field(default_factory=list)
    circ_mean_peak: float = float("nan")
    circ_std_peak: float = float("nan")

    gain_peak_mean: float = float("nan")
    gain_peak_std: float = float("nan")
    gain_at_mu_mean: float = float("nan")
    gain_at_qturn_mean: float = float("nan")

    correlation_mean: float = float("nan")
    correlation_std: float = float("nan")

    parity_O: int = 0
    parity_E: int = 0
    parity_n_mod_2: int = 0

    mean_logprob: float = float("nan")
    std_logprob: float = float("nan")
    min_logprob: float = float("nan")
    max_logprob: float = float("nan")


def extract_diagnostics(
    labe: GyroLabe,
    result: Optional[GenerationResult] = None,
) -> CoordinationDiagnostics:
    diag = CoordinationDiagnostics()
    s = labe.stats()

    if s["steps"] == 0:
        return diag

    sig = s["kernel_state"]
    diag.kernel_step = sig.step
    diag.kernel_state_hex = sig.state_hex
    diag.kernel_a_hex = sig.a_hex
    diag.kernel_b_hex = sig.b_hex

    diag.unique_h = s["unique_h"]
    diag.h_entropy = s["h_entropy"]

    diag.chi_dist = s["chi_dist"]

    diag.unique_bytes = s["unique_bytes"]
    diag.b_entropy = s["b_entropy"]
    diag.mean_byte_weight = s["mean_byte_weight"]
    diag.byte_weight_hist = list(s.get("byte_weight_hist", []))

    diag.parity_O = s["parity_O"]
    diag.parity_E = s["parity_E"]
    diag.parity_n_mod_2 = s["parity_n_mod_2"]

    diag.n_layers_routed = len(labe.routed_layers)
    diag.n_fibers = labe.n_fiber

    _extract_charge_distribution(labe, diag)

    diag.mean_dist_to_mu = float(s.get("mean_dist_to_mu", float("nan")))
    diag.std_dist_to_mu = float(s.get("std_dist_to_mu", float("nan")))
    diag.mean_dist_to_qturn = float(s.get("mean_dist_to_qturn", float("nan")))
    diag.std_dist_to_qturn = float(s.get("std_dist_to_qturn", float("nan")))
    diag.mean_code_dist = float(s.get("mean_code_dist", float("nan")))
    diag.std_code_dist = float(s.get("std_code_dist", float("nan")))

    diag.gain_peak_mean = float(s.get("mean_gain_at_peak", float("nan")))
    diag.gain_peak_std = float(s.get("std_gain_at_peak", float("nan")))
    diag.gain_at_mu_mean = float(s.get("mean_gain_at_mu", float("nan")))
    diag.gain_at_qturn_mean = float(s.get("mean_gain_at_qturn", float("nan")))

    diag.correlation_mean = float(s.get("mean_correlation", float("nan")))
    diag.correlation_std = float(s.get("std_correlation", float("nan")))

    layer_peaks = s.get("layer_h_peaks", [])
    diag.layer_h_peaks = layer_peaks
    if layer_peaks:
        diag.circ_mean_peak = circular_mean(layer_peaks, 256)
        diag.circ_std_peak = circular_std(layer_peaks, 256)

    if result is not None:
        _extract_logprob_summary(result, diag)

    return diag


def _extract_charge_distribution(labe: GyroLabe, diag: CoordinationDiagnostics) -> None:
    try:
        kernel = labe.kernel
        if hasattr(labe, 'byte_log') and labe.byte_log:
            counts = [0, 0, 0, 0]
            for b in labe.byte_log:
                c = kernel.byte_charge[b]
                if 0 <= c <= 3:
                    counts[c] += 1
            total = sum(counts)
            diag.charge_counts = counts
            diag.charge_fracs = [c / total if total > 0 else 0.0 for c in counts]
    except (AttributeError, TypeError):
        pass


def _extract_logprob_summary(result: GenerationResult, diag: CoordinationDiagnostics) -> None:
    logprobs = getattr(result, "logprobs", None)
    if logprobs is None or len(logprobs) == 0:
        if not math.isnan(result.perplexity) and result.n_tokens > 0:
            diag.mean_logprob = -math.log(result.perplexity)
        return

    arr = np.array(logprobs, dtype=np.float64)
    diag.mean_logprob = float(np.mean(arr))
    diag.std_logprob = float(np.std(arr))
    diag.min_logprob = float(np.min(arr))
    diag.max_logprob = float(np.max(arr))


def load_model(model_dir: Path, device: torch.device):
    dtype = choose_dtype(device)
    print(f"Loading model: {model_dir}")
    print(f"Device: {device}, dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    n_layers = len(model.model.layers)
    d_model = model.config.hidden_size
    print(f"Loaded: {d_model}d, {n_layers} layers\n")
    return model, tokenizer


def fmt(v: float, decimals: int = 3) -> str:
    if math.isnan(v):
        return "—"
    return f"{v:.{decimals}f}"


def print_generation(label: str, result: GenerationResult) -> None:
    print(
        f"{label} ({result.n_tokens} tok, {result.elapsed:.1f}s, "
        f"{result.tokens_per_second:.1f} tok/s, ppl={result.perplexity:.2f}):"
    )
    print(result.text)


def print_diagnostics(diag: CoordinationDiagnostics) -> None:
    if diag.kernel_step == 0 and diag.unique_h == 0:
        print("  (no steps recorded)")
        return

    print(f"  Kernel:       step={diag.kernel_step}  "
          f"state=0x{diag.kernel_state_hex}  "
          f"A=0x{diag.kernel_a_hex}  B=0x{diag.kernel_b_hex}")

    print(f"  Horizon:      {diag.unique_h}/256 unique  H(h)={diag.h_entropy} bits")

    if diag.chi_dist:
        print(f"  Vertex:       {diag.chi_dist}")

    print(f"  Bytes:        {diag.unique_bytes}/256 unique  "
          f"H(b)={diag.b_entropy} bits  mean_weight={diag.mean_byte_weight}")

    if any(c > 0 for c in diag.charge_counts):
        frac_str = " ".join(f"q{i}={diag.charge_fracs[i]:.2f}" for i in range(4))
        print(f"  Charge:       {frac_str}")

    print(f"  Parity:       O=0x{diag.parity_O:03x}  E=0x{diag.parity_E:03x}  n%2={diag.parity_n_mod_2}")

    print(f"  Layers:       {diag.n_layers_routed} routed  fibers={diag.n_fibers}")

    nonzero = [(w, c) for w, c in enumerate(diag.byte_weight_hist) if c > 0]
    if nonzero:
        hist_str = " ".join(f"w{w}:{c}" for w, c in nonzero)
        print(f"  Weights:      {hist_str}")

    print(f"  Alignment:")
    print(f"    to μ:       mean={fmt(diag.mean_dist_to_mu, 1)}  std={fmt(diag.std_dist_to_mu, 1)}")
    print(f"    to μ+π/2:   mean={fmt(diag.mean_dist_to_qturn, 1)}  std={fmt(diag.std_dist_to_qturn, 1)}  (quarter-turn)")
    print(f"    code dist:  mean={fmt(diag.mean_code_dist, 1)}  std={fmt(diag.std_code_dist, 1)}  (Hamming on masks)")

    if diag.layer_h_peaks:
        print(f"    peaks:      μ_circ={fmt(diag.circ_mean_peak, 1)}  σ_circ={fmt(diag.circ_std_peak, 1)}  "
              f"({len(diag.layer_h_peaks)} samples)")

    print(f"  Mask gain:")
    print(f"    at peaks:   mean={fmt(diag.gain_peak_mean)}  std={fmt(diag.gain_peak_std)}")
    print(f"    at μ:       {fmt(diag.gain_at_mu_mean)}  |  at μ+π/2: {fmt(diag.gain_at_qturn_mean)}")

    if not math.isnan(diag.gain_peak_mean):
        if diag.gain_peak_mean > 1.05:
            print(f"    → extraction IS tracking mask emphasis (gain > 1)")
        elif diag.gain_peak_mean < 0.95:
            print(f"    → extraction NOT tracking mask emphasis (gain < 1)")
        else:
            print(f"    → extraction neutral relative to mask (gain ≈ 1)")

    if not math.isnan(diag.correlation_mean):
        print(f"  Correlation:  mean={fmt(diag.correlation_mean)}  std={fmt(diag.correlation_std)}")

    if not math.isnan(diag.mean_logprob):
        print(f"  Logprobs:     mean={fmt(diag.mean_logprob)}  std={fmt(diag.std_logprob)}  "
              f"min={fmt(diag.min_logprob)}  max={fmt(diag.max_logprob)}")


def print_compact_diagnostics(diag: CoordinationDiagnostics) -> str:
    parts = [
        f"step={diag.kernel_step}",
        f"h={diag.unique_h}/256",
    ]
    if not math.isnan(diag.mean_dist_to_mu):
        parts.append(f"d_μ={fmt(diag.mean_dist_to_mu, 1)}")
    if not math.isnan(diag.mean_code_dist):
        parts.append(f"d_code={fmt(diag.mean_code_dist, 1)}")
    if not math.isnan(diag.gain_peak_mean):
        parts.append(f"gain={fmt(diag.gain_peak_mean)}")
    if not math.isnan(diag.correlation_mean):
        parts.append(f"corr={fmt(diag.correlation_mean)}")
    return "  " + "  ".join(parts)


def run_prompts(
    model, tokenizer, atlas_dir: Path,
    max_tokens: int, temperature: float, seed: int,
) -> None:
    """Run predefined prompts with baseline vs coordinated."""
    config = CouplingConfig()
    
    for prompt in PROMPTS:
        print("=" * 72)
        print(f"PROMPT: {prompt}\n")

        baseline = generate(
            model, tokenizer, labe=None,
            prompt=prompt, max_new_tokens=max_tokens,
            temperature=temperature, seed=seed,
        )
        print_generation("BASELINE", baseline)

        labe = GyroLabe(model, atlas_dir=atlas_dir, config=config)
        labe.install()
        try:
            coordinated = generate(
                model, tokenizer, labe=labe,
                prompt=prompt, max_new_tokens=max_tokens,
                temperature=temperature, seed=seed,
            )
        finally:
            labe.restore()

        print()
        print_generation("COORDINATED", coordinated)

        delta_ppl = coordinated.perplexity - baseline.perplexity
        sign = "+" if delta_ppl >= 0 else ""
        print(f"\n  Δppl: {sign}{delta_ppl:.2f}")

        baseline_diag = CoordinationDiagnostics()
        _extract_logprob_summary(baseline, baseline_diag)
        coord_diag = extract_diagnostics(labe, coordinated)

        if not math.isnan(baseline_diag.mean_logprob) and not math.isnan(coord_diag.mean_logprob):
            delta_lp = coord_diag.mean_logprob - baseline_diag.mean_logprob
            sign_lp = "+" if delta_lp >= 0 else ""
            print(f"  Δlogprob: {sign_lp}{delta_lp:.3f}")

        print(f"\nDIAGNOSTICS:")
        print_diagnostics(coord_diag)
        print()


def run_chat(
    model, tokenizer, atlas_dir: Path,
    max_tokens: int, temperature: float,
) -> None:
    """Interactive chat with coordination active."""
    config = CouplingConfig()
    labe = GyroLabe(model, atlas_dir=atlas_dir, config=config)
    labe.install()

    print("GyroLabe Chat")
    print(f"  Routed layers: {labe.routed_layers}")
    print(f"  d_model={labe.d_model}, fibers={labe.n_fiber}")
    print(f"  Commands: 'quit', 'reset', 'stats'\n")

    try:
        while True:
            try:
                prompt = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not prompt:
                continue
            
            cmd = prompt.lower()
            
            if cmd == "quit":
                break
            if cmd == "reset":
                labe.reset()
                print("Kernel reset.\n")
                continue
            if cmd in ("stats", "full"):
                diag = extract_diagnostics(labe)
                print_diagnostics(diag)
                print()
                continue

            result = generate(
                model, tokenizer, labe=labe,
                prompt=prompt, max_new_tokens=max_tokens,
                temperature=temperature,
            )

            print(f"\n{result.text}")
            print(f"\n[{result.n_tokens} tok, {result.elapsed:.1f}s, ppl={result.perplexity:.2f}]")

            diag = extract_diagnostics(labe, result)
            print(print_compact_diagnostics(diag))
            print()
    finally:
        labe.restore()
        print("Model restored.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GyroLabe: Holographic Coordination System"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--tokens", type=int, default=50)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}")
        sys.exit(1)
    if not args.atlas.exists():
        print(f"Atlas not found: {args.atlas}")
        sys.exit(1)

    device = torch.device(args.device) if args.device else detect_device()
    model, tokenizer = load_model(args.model, device)

    if args.chat:
        run_chat(model, tokenizer, args.atlas, args.tokens, args.temp)
    else:
        run_prompts(model, tokenizer, args.atlas, args.tokens, args.temp, args.seed)


if __name__ == "__main__":
    main()  