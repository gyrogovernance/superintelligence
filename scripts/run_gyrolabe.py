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
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.gyrolabe import (
    CouplingConfig,
    GenerationResult,
    GyroLabe,
    choose_dtype,
    detect_device,
    generate,
)

logging.getLogger("transformers").setLevel(logging.ERROR)

DEFAULT_MODEL = Path("data/models/Olmo-3-7B-Instruct")
DEFAULT_ATLAS = Path("data/atlas")

PROMPTS = [
    "The Source is Common. This is the foundational axiom from which all other structure derives. It establishes that operational structure must trace to a shared source while maintaining directional distinction. Without this asymmetry, no coherent observation would be possible, as there would be no way to distinguish different operational paths.",
    "Unity is Non-Absolute. Absolute unity (□E) would collapse all distinctions, making the system homogeneous and preventing any meaningful structure. Non-absolute unity (¬□E) ensures informational variety while maintaining traceability to the common source.",
    "Opposition is Non-Absolute. Absolute opposition (□¬E) would create irreconcilable contradictions, destroying coherence. Non-absolute opposition ensures accountability of inference, meaning that different operational paths remain comparable even when they yield different results.",
    "Balance is Universal. Perfect imbalance would make existence and freedom meaningless, since the memory of inferred information would have no reason to acquire substance and structure at all. Therefore, balance is the universal signature of alignment through integrity of intelligence: traceable inferential accountability of informational variety from a common source.",
]


# ---------------------------------------------------------------------------
# Circular statistics helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Diagnostics dataclass — every field from stats() is represented
# ---------------------------------------------------------------------------

@dataclass
class CoordinationDiagnostics:
    """Comprehensive diagnostics with physics metrics."""

    # Kernel identity
    kernel_step: int = 0
    kernel_state_hex: str = "0000"
    kernel_a_hex: str = "0000"
    kernel_b_hex: str = "0000"

    # Horizon coverage
    unique_h: int = 0
    h_entropy: float = 0.0

    # K4 vertex distribution
    chi_dist: list[int] = field(default_factory=list)

    # Byte diversity
    unique_bytes: int = 0
    b_entropy: float = 0.0
    mean_byte_weight: float = 0.0
    byte_weight_hist: list[int] = field(default_factory=list)

    # Charge distribution derived from byte_log
    charge_counts: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    charge_fracs: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    # Architecture
    n_layers_routed: int = 0
    n_fibers: int = 0

    # Alignment: distance to μ (current kernel focus point)
    mean_dist_to_mu: float = float("nan")
    std_dist_to_mu: float = float("nan")

    # Alignment: distance to μ + π/2 (quarter-turn)
    mean_dist_to_qturn: float = float("nan")
    std_dist_to_qturn: float = float("nan")

    # Alignment: Hamming distance on masks between kernel horizon and peak
    mean_code_dist: float = float("nan")
    std_code_dist: float = float("nan")

    # Per-layer energy peak positions (circular)
    layer_h_peaks: list[int] = field(default_factory=list)
    circ_mean_peak: float = float("nan")
    circ_std_peak: float = float("nan")

    # Per-layer raw values for distribution analysis
    layer_gains: list[float] = field(default_factory=list)
    layer_correlations: list[float] = field(default_factory=list)
    layer_dists_mu: list[int] = field(default_factory=list)
    layer_dists_qturn: list[int] = field(default_factory=list)
    layer_code_dists: list[int] = field(default_factory=list)

    # Mask gain at sampled points
    gain_peak_mean: float = float("nan")
    gain_peak_std: float = float("nan")
    gain_at_mu_mean: float = float("nan")
    gain_at_qturn_mean: float = float("nan")

    # --- NEW: gain extremes across steps ---
    gain_peak_min: float = float("nan")
    gain_peak_max: float = float("nan")

    # Mask–energy correlation
    correlation_mean: float = float("nan")
    correlation_std: float = float("nan")

    # --- NEW: correlation extremes ---
    correlation_min: float = float("nan")
    correlation_max: float = float("nan")

    # --- NEW: peak energy mass (how concentrated the energy peak is) ---
    mean_peak_mass: float = float("nan")
    std_peak_mass: float = float("nan")
    min_peak_mass: float = float("nan")
    max_peak_mass: float = float("nan")

    # Trajectory parity commitment
    parity_O: int = 0
    parity_E: int = 0
    parity_n_mod_2: int = 0

    # --- NEW: per-step worst-alignment tracking ---
    # Step index with highest dist_to_mu (least focused)
    worst_step_idx: int = -1
    worst_step_dist_mu: float = float("nan")
    worst_step_h: int = -1

    # Generation quality
    mean_logprob: float = float("nan")
    std_logprob: float = float("nan")
    min_logprob: float = float("nan")
    max_logprob: float = float("nan")

    # Speed
    tokens_per_second: float = float("nan")


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_diagnostics(
    labe: GyroLabe,
    result: GenerationResult | None = None,
) -> CoordinationDiagnostics:
    diag = CoordinationDiagnostics()
    s = labe.stats()

    if s["steps"] == 0:
        return diag

    # Kernel identity
    sig = s["kernel_state"]
    diag.kernel_step = sig.step
    diag.kernel_state_hex = sig.state_hex
    diag.kernel_a_hex = sig.a_hex
    diag.kernel_b_hex = sig.b_hex

    # Horizon
    diag.unique_h = s["unique_h"]
    diag.h_entropy = s["h_entropy"]

    # K4 vertex
    diag.chi_dist = s["chi_dist"]

    # Byte diversity
    diag.unique_bytes = s["unique_bytes"]
    diag.b_entropy = s["b_entropy"]
    diag.mean_byte_weight = s["mean_byte_weight"]
    diag.byte_weight_hist = list(s.get("byte_weight_hist", []))

    # Parity
    diag.parity_O = s["parity_O"]
    diag.parity_E = s["parity_E"]
    diag.parity_n_mod_2 = s["parity_n_mod_2"]

    # Architecture
    diag.n_layers_routed = len(labe.routed_layers)
    diag.n_fibers = labe.n_fiber

    # Charge distribution
    _extract_charge_distribution(labe, diag)

    # Alignment distances
    diag.mean_dist_to_mu = float(s.get("mean_dist_to_mu", float("nan")))
    diag.std_dist_to_mu = float(s.get("std_dist_to_mu", float("nan")))
    diag.mean_dist_to_qturn = float(s.get("mean_dist_to_qturn", float("nan")))
    diag.std_dist_to_qturn = float(s.get("std_dist_to_qturn", float("nan")))
    diag.mean_code_dist = float(s.get("mean_code_dist", float("nan")))
    diag.std_code_dist = float(s.get("std_code_dist", float("nan")))

    # Gain summaries
    diag.gain_peak_mean = float(s.get("mean_gain_at_peak", float("nan")))
    diag.gain_peak_std = float(s.get("std_gain_at_peak", float("nan")))
    diag.gain_at_mu_mean = float(s.get("mean_gain_at_mu", float("nan")))
    diag.gain_at_qturn_mean = float(s.get("mean_gain_at_qturn", float("nan")))

    # Correlation summaries
    diag.correlation_mean = float(s.get("mean_correlation", float("nan")))
    diag.correlation_std = float(s.get("std_correlation", float("nan")))

    # Layer-level raw arrays — now actually extracted
    layer_gains = s.get("layer_gains", [])
    layer_correlations = s.get("layer_correlations", [])
    layer_dists_mu = s.get("layer_dists_mu", [])
    layer_dists_qturn = s.get("layer_dists_qturn", [])
    layer_code_dists = s.get("layer_code_dists", [])
    layer_peaks = s.get("layer_h_peaks", [])

    diag.layer_gains = layer_gains
    diag.layer_correlations = layer_correlations
    diag.layer_dists_mu = layer_dists_mu
    diag.layer_dists_qturn = layer_dists_qturn
    diag.layer_code_dists = layer_code_dists
    diag.layer_h_peaks = layer_peaks

    # Circular peak stats
    if layer_peaks:
        diag.circ_mean_peak = circular_mean(layer_peaks, 256)
        diag.circ_std_peak = circular_std(layer_peaks, 256)

    # Gain extremes from raw layer data
    if layer_gains:
        arr = np.array(layer_gains, dtype=np.float64)
        diag.gain_peak_min = float(arr.min())
        diag.gain_peak_max = float(arr.max())

    # Correlation extremes from raw layer data
    if layer_correlations:
        arr = np.array(layer_correlations, dtype=np.float64)
        diag.correlation_min = float(arr.min())
        diag.correlation_max = float(arr.max())

    # Peak mass: pulled from per-layer data in trajectory
    # stats() exposes mean_peak_mass but not std/min/max — derive from trajectory
    pmass_vals = _collect_peak_mass(labe)
    if pmass_vals:
        arr = np.array(pmass_vals, dtype=np.float64)
        diag.mean_peak_mass = float(arr.mean())
        diag.std_peak_mass = float(arr.std())
        diag.min_peak_mass = float(arr.min())
        diag.max_peak_mass = float(arr.max())

    # Worst-alignment step: highest mean_dist_to_mu across trajectory steps
    _extract_worst_step(labe, diag)

    # Generation quality
    if result is not None:
        _extract_logprob_summary(result, diag)
        diag.tokens_per_second = result.tokens_per_second

    return diag


def _collect_peak_mass(labe: GyroLabe) -> list[float]:
    """Collect per-layer peak_mass values from the full trajectory."""
    vals: list[float] = []
    for t in labe.trajectory:
        if "layers" in t:
            for d in t["layers"]:
                pm = d.get("peak_mass")
                if pm is not None:
                    vals.append(float(pm))
    return vals


def _extract_worst_step(labe: GyroLabe, diag: CoordinationDiagnostics) -> None:
    """Find the step with highest mean_dist_to_mu (least kernel-focused)."""
    worst_dist = -1.0
    worst_idx = -1
    worst_h = -1
    for i, t in enumerate(labe.trajectory):
        d = t.get("mean_dist_to_mu")
        if d is not None and d > worst_dist:
            worst_dist = d
            worst_idx = i
            worst_h = t.get("h", -1)
    if worst_idx >= 0:
        diag.worst_step_idx = worst_idx
        diag.worst_step_dist_mu = worst_dist
        diag.worst_step_h = worst_h


def _extract_charge_distribution(labe: GyroLabe, diag: CoordinationDiagnostics) -> None:
    try:
        kernel = labe.kernel
        if hasattr(labe, "byte_log") and labe.byte_log:
            counts = [0, 0, 0, 0]
            for b in labe.byte_log:
                c = kernel.byte_charge[b]
                if 0 <= c <= 3:
                    counts[c] += 1
            total = sum(counts)
            diag.charge_counts = counts
            diag.charge_fracs = [
                c / total if total > 0 else 0.0 for c in counts
            ]
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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt(v: float, decimals: int = 3) -> str:
    if math.isnan(v):
        return "—"
    return f"{v:.{decimals}f}"


def _gain_interpretation(gain: float) -> str:
    if gain > 1.05:
        return "extraction IS tracking mask emphasis (gain > 1)"
    if gain < 0.95:
        return "extraction NOT tracking mask emphasis (gain < 1)"
    return "extraction neutral relative to mask (gain ≈ 1)"


def _corr_interpretation(corr: float) -> str:
    if corr > 0.5:
        return "strong mask–energy alignment"
    if corr > 0.2:
        return "moderate mask–energy alignment"
    if corr > 0.0:
        return "weak mask–energy alignment"
    return "no / negative mask–energy alignment"


def print_generation(label: str, result: GenerationResult) -> None:
    print(
        f"{label} ({result.n_tokens} tok, {result.elapsed:.1f}s, "
        f"{result.tokens_per_second:.1f} tok/s, ppl={result.perplexity:.2f}):"
    )
    print(result.text)


# ---------------------------------------------------------------------------
# Full diagnostics display
# ---------------------------------------------------------------------------

def print_diagnostics(diag: CoordinationDiagnostics) -> None:
    if diag.kernel_step == 0 and diag.unique_h == 0:
        print("  (no steps recorded)")
        return

    # --- Kernel identity ---
    print(f"  Kernel:       step={diag.kernel_step}  "
          f"state=0x{diag.kernel_state_hex}  "
          f"A=0x{diag.kernel_a_hex}  B=0x{diag.kernel_b_hex}")

    # --- Horizon ---
    print(f"  Horizon:      {diag.unique_h}/256 unique  H(h)={diag.h_entropy} bits")

    # --- K4 vertex ---
    if diag.chi_dist:
        total_chi = sum(diag.chi_dist)
        chi_labels = ["G(gov)", "I(inf)", "N(infer)", "B(bal)"]
        chi_parts = " ".join(
            f"{chi_labels[i]}={diag.chi_dist[i]}"
            + (f"({diag.chi_dist[i]/total_chi:.0%})" if total_chi > 0 else "")
            for i in range(min(4, len(diag.chi_dist)))
        )
        print(f"  Vertex χ:     {chi_parts}")

    # --- Bytes ---
    print(f"  Bytes:        {diag.unique_bytes}/256 unique  "
          f"H(b)={diag.b_entropy} bits  mean_weight={diag.mean_byte_weight}")

    # --- Charge ---
    if any(c > 0 for c in diag.charge_counts):
        frac_str = " ".join(f"q{i}={diag.charge_fracs[i]:.2f}" for i in range(4))
        print(f"  Charge:       {frac_str}")

    # --- Parity ---
    print(f"  Parity:       O=0x{diag.parity_O:03x}  "
          f"E=0x{diag.parity_E:03x}  n%2={diag.parity_n_mod_2}")

    # --- Architecture ---
    print(f"  Layers:       {diag.n_layers_routed} routed  fibers={diag.n_fibers}")

    # --- Byte weight histogram ---
    nonzero = [(w, c) for w, c in enumerate(diag.byte_weight_hist) if c > 0]
    if nonzero:
        hist_str = " ".join(f"w{w}:{c}" for w, c in nonzero)
        print(f"  Weights:      {hist_str}")

    # --- Alignment distances ---
    print("  Alignment:")
    print(f"    to μ:       mean={fmt(diag.mean_dist_to_mu, 1)}  "
          f"std={fmt(diag.std_dist_to_mu, 1)}")
    print(f"    to μ+π/2:   mean={fmt(diag.mean_dist_to_qturn, 1)}  "
          f"std={fmt(diag.std_dist_to_qturn, 1)}  (quarter-turn)")
    print(f"    code dist:  mean={fmt(diag.mean_code_dist, 1)}  "
          f"std={fmt(diag.std_code_dist, 1)}  (Hamming on masks)")

    # --- Circular peak stats ---
    if diag.layer_h_peaks:
        print(f"    peaks:      μ_circ={fmt(diag.circ_mean_peak, 1)}  "
              f"σ_circ={fmt(diag.circ_std_peak, 1)}  "
              f"({len(diag.layer_h_peaks)} samples)")

    # --- Worst step ---
    if diag.worst_step_idx >= 0:
        print(f"    worst step: idx={diag.worst_step_idx}  "
              f"h={diag.worst_step_h}  "
              f"d_μ={fmt(diag.worst_step_dist_mu, 1)}")

    # --- Mask gain ---
    print("  Mask gain:")
    print(f"    at peaks:   mean={fmt(diag.gain_peak_mean)}  "
          f"std={fmt(diag.gain_peak_std)}  "
          f"min={fmt(diag.gain_peak_min)}  max={fmt(diag.gain_peak_max)}")
    print(f"    at μ:       {fmt(diag.gain_at_mu_mean)}  "
          f"|  at μ+π/2: {fmt(diag.gain_at_qturn_mean)}")
    if not math.isnan(diag.gain_peak_mean):
        print(f"    → {_gain_interpretation(diag.gain_peak_mean)}")

    # --- Mask–energy correlation ---
    if not math.isnan(diag.correlation_mean):
        print(f"  Correlation:  mean={fmt(diag.correlation_mean)}  "
              f"std={fmt(diag.correlation_std)}  "
              f"min={fmt(diag.correlation_min)}  max={fmt(diag.correlation_max)}")
        print(f"    → {_corr_interpretation(diag.correlation_mean)}")

    # --- Peak mass (energy concentration) ---
    if not math.isnan(diag.mean_peak_mass):
        print(f"  Peak mass:    mean={fmt(diag.mean_peak_mass)}  "
              f"std={fmt(diag.std_peak_mass)}  "
              f"min={fmt(diag.min_peak_mass)}  max={fmt(diag.max_peak_mass)}")
        # Interpretation: fraction of total energy at peak boundary position
        if diag.mean_peak_mass > 0.05:
            print(f"    → energy is concentrated (peak captures "
                  f"{diag.mean_peak_mass:.1%} of boundary mass on average)")
        else:
            print(f"    → energy is diffuse across boundary positions")

    # --- Logprobs ---
    if not math.isnan(diag.mean_logprob):
        print(f"  Logprobs:     mean={fmt(diag.mean_logprob)}  "
              f"std={fmt(diag.std_logprob)}  "
              f"min={fmt(diag.min_logprob)}  max={fmt(diag.max_logprob)}")

    # --- Speed ---
    if not math.isnan(diag.tokens_per_second):
        print(f"  Speed:        {diag.tokens_per_second:.1f} tok/s")


# ---------------------------------------------------------------------------
# Compact diagnostics for chat mode
# ---------------------------------------------------------------------------

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
    # New additions to compact view
    if not math.isnan(diag.mean_peak_mass):
        parts.append(f"mass={fmt(diag.mean_peak_mass)}")
    if not math.isnan(diag.tokens_per_second):
        parts.append(f"{diag.tokens_per_second:.1f}tok/s")
    if diag.parity_n_mod_2 >= 0:
        parts.append(f"par={diag.parity_n_mod_2}")
    return "  " + "  ".join(parts)


# ---------------------------------------------------------------------------
# Prompt batch runner
# ---------------------------------------------------------------------------

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

        # Delta perplexity
        delta_ppl = coordinated.perplexity - baseline.perplexity
        sign = "+" if delta_ppl >= 0 else ""
        print(f"\n  Δppl:         {sign}{delta_ppl:.2f}")

        # Delta logprob
        baseline_diag = CoordinationDiagnostics()
        _extract_logprob_summary(baseline, baseline_diag)
        coord_diag = extract_diagnostics(labe, coordinated)

        if not math.isnan(baseline_diag.mean_logprob) and not math.isnan(coord_diag.mean_logprob):
            delta_lp = coord_diag.mean_logprob - baseline_diag.mean_logprob
            sign_lp = "+" if delta_lp >= 0 else ""
            print(f"  Δlogprob:     {sign_lp}{delta_lp:.3f}")

        # Speed comparison
        if not math.isnan(coord_diag.tokens_per_second):
            delta_speed = coord_diag.tokens_per_second - baseline.tokens_per_second
            sign_s = "+" if delta_speed >= 0 else ""
            print(f"  Δtok/s:       {sign_s}{delta_speed:.1f}  "
                  f"(baseline={baseline.tokens_per_second:.1f}  "
                  f"coordinated={coord_diag.tokens_per_second:.1f})")

        print("\nDIAGNOSTICS:")
        print_diagnostics(coord_diag)
        print()


# ---------------------------------------------------------------------------
# Interactive chat runner
# ---------------------------------------------------------------------------

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
    print("  Commands: 'quit', 'reset', 'stats', 'full'\n")

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
            if cmd == "stats":
                diag = extract_diagnostics(labe)
                print(print_compact_diagnostics(diag))
                print()
                continue
            if cmd == "full":
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
            print(f"\n[{result.n_tokens} tok, {result.elapsed:.1f}s, "
                  f"{result.tokens_per_second:.1f} tok/s, ppl={result.perplexity:.2f}]")

            diag = extract_diagnostics(labe, result)
            print(print_compact_diagnostics(diag))
            print()
    finally:
        labe.restore()
        print("Model restored.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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