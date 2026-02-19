#!/usr/bin/env python3
# scripts/run_gyrolabe_experiments.py
"""
GyroLabe Experiments Runner - Narrow Surface (Mask Layer) Decision Run.

This run tests ONLY the best-supported hypotheses on the *same* underlying object:
a radial mask on Hamming-distance geometry of the 12-bit mask code.

We keep:
  1) kraw_J4 radial profile (Krawtchouk projection of the CGM-sigma Gaussian LUT)
  2) sigma_focus dynamics (narrow the radial profile when code-space motion is large)
  3) alpha_CH regulator (gentle global scaling from correlation+entropy)

And we test them alone and in combinations (2^3 factorial design).
"""

from __future__ import annotations

import math
import sys
import traceback
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.gyrolabe import (
    CouplingConfig,
    GenerationResult,
    GyroLabe,
    N_BOUNDARY,
    QUARTER_TURN,
    _GAUSSIAN_LUT,
    choose_dtype,
    detect_device,
    generate,
    get_code_distance_matrix,
)
from src.router.constants import (
    C_PERP_12,
    mask12_for_byte,
    popcount,
    trajectory_parity_commitment,
)

MODEL_DIR = Path("data/models/Olmo-3-7B-Instruct")
ATLAS_DIR = Path("data/atlas")

MAX_TOKENS = 150
TEMPERATURE = 0.7
SEED = 42

PROMPTS = [
    ("governance", "The purpose of good governance is"),
    ("geometry", "In three dimensions, the structure"),
    ("math", "Mathematics reveals that"),
]

# ---------------------------------------------------------------------------
# CGM invariants (for printing + context)
# ---------------------------------------------------------------------------
_M_A = 1.0 / (2.0 * math.sqrt(2.0 * math.pi))  # ≈ 0.19947
DELTA_BU = 0.195342
A_STAR = 1.0 - DELTA_BU / _M_A                  # ≈ 0.0207
TARGET_CORR = 1.0 - A_STAR                      # ≈ 0.9793

# ---------------------------------------------------------------------------
# sigma_focus parameter (learned from experiments)
# ---------------------------------------------------------------------------
K_SIG = 0.40  # narrow when D_t > 0.5, widen when D_t < 0.5
SIG_CLIP = (0.30, 2.50)

# ---------------------------------------------------------------------------
# alpha_CH regulator parameters (gentle; stays near 1)
# ---------------------------------------------------------------------------
C_STAR = 0.48
H_STAR = 0.85
K_C = 2.0
K_H = 2.0
REG_CLIP = (0.50, 1.50)

CORR_WINDOW = 16


def sep(label: str) -> None:
    print(f"\n--- {label} ---")


def fmt(v: float, d: int = 3) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "—"
    return f"{v:.{d}f}"


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts.astype(np.float64) / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _f_quad(val: float, center: float, k: float) -> float:
    return float(np.clip(1.0 - k * (val - center) ** 2, REG_CLIP[0], REG_CLIP[1]))


# ============================================================================
# Krawtchouk (corrected) projection on Hamming spheres
# ============================================================================

def _kraw_poly(d: int, j: int, n: int = 12) -> float:
    return float(sum(
        (-1) ** s * comb(j, s) * comb(n - j, d - s)
        for s in range(d + 1)
        if 0 <= d - s <= n - j
    ))


def build_krawtchouk_basis(n: int = 12, J: int = 4) -> np.ndarray:
    K = np.zeros((J, n + 1), dtype=np.float64)
    for j in range(J):
        for d in range(n + 1):
            K[j, d] = _kraw_poly(d, j, n)
    return K


def build_kraw_lut_from_gaussian(J: int = 4) -> np.ndarray:
    """
    Construct a Krawtchouk-truncated LUT from the current _GAUSSIAN_LUT.
    This is NOT a competing physics; it is the same radial function expressed
    in the native Hamming spectral basis and low-pass truncated.

    Inner product uses binomial weights w(d)=C(12,d).
    """
    n = 12
    weights = np.array([comb(n, d) for d in range(n + 1)], dtype=np.float64)
    K = build_krawtchouk_basis(n=n, J=J)  # [J,13]
    norms = np.array([float(np.dot(K[j] * K[j], weights)) for j in range(J)], dtype=np.float64)

    lut = np.zeros((4, 4, 13), dtype=np.float32)
    for chi in range(4):
        for p in range(4):
            G = _GAUSSIAN_LUT[chi, p, :].astype(np.float64)  # [13]
            recon = np.zeros(13, dtype=np.float64)
            for j in range(J):
                a = float(np.dot(G * K[j], weights)) / (norms[j] + 1e-12)
                recon += a * K[j]
            lut[chi, p, :] = np.clip(recon, 0.0, None).astype(np.float32)
    return lut


def kraw_energy_error(lut_kraw: np.ndarray) -> float:
    """
    A meaningful global fit diagnostic on Hamming spheres:
    relative L2 error weighted by binomial sphere sizes and Gaussian energy.

      err = sqrt(sum w(d)*(G-K)^2) / sqrt(sum w(d)*G^2)

    aggregated over chi,p (simple mean).
    """
    n = 12
    weights = np.array([comb(n, d) for d in range(n + 1)], dtype=np.float64)
    errs = []
    for chi in range(4):
        for p in range(4):
            G = _GAUSSIAN_LUT[chi, p, :].astype(np.float64)
            K = lut_kraw[chi, p, :].astype(np.float64)
            num = float(np.dot((G - K) ** 2, weights))
            den = float(np.dot(G ** 2, weights)) + 1e-12
            errs.append(math.sqrt(num / den))
    return float(np.mean(errs))


# ============================================================================
# Mask variant: one class, three toggles
# ============================================================================

class BestMaskGyroLabe(GyroLabe):
    """
    A single implementation that can reproduce:
      - standard (Gaussian LUT)
      - kraw_J4 only
      - sigma_focus only
      - alpha_CH only
      - any combination of the three

    This keeps the comparison clean: same code path, different toggles.
    """

    def __init__(
        self,
        *args,
        radial_lut: np.ndarray,
        use_sigma_focus: bool,
        use_alpha_ch: bool,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._lut = radial_lut  # [4,4,13]
        self._use_sigma_focus = bool(use_sigma_focus)
        self._use_alpha_ch = bool(use_alpha_ch)

        self._cdist = get_code_distance_matrix()

        self._corr_hist: list[float] = []
        self._h_counts = np.zeros(256, dtype=np.int64)

    def end_step(self) -> dict[str, Any]:
        record = super().end_step()
        h = int(record.get("h", 0))
        self._h_counts[h] += 1

        c = record.get("mean_correlation", None)
        if isinstance(c, (int, float)):
            self._corr_hist.append(float(c))
            if len(self._corr_hist) > CORR_WINDOW:
                self._corr_hist.pop(0)
        return record

    def _alpha_reg(self) -> float:
        if not self._use_alpha_ch:
            return 1.0

        C = float(np.mean(self._corr_hist)) if self._corr_hist else C_STAR
        H_bits = _entropy_from_counts(self._h_counts)
        H = H_bits / 8.0

        fC = _f_quad(C, C_STAR, K_C)
        fH = _f_quad(H, H_STAR, K_H)
        return float(np.clip(fC * fH, REG_CLIP[0], REG_CLIP[1]))

    def _sigma_scale(self, h_prev: int | None, h_curr: int) -> float:
        if not self._use_sigma_focus or h_prev is None:
            return 1.0
        D = float(self._cdist[h_prev, h_curr]) / 12.0  # [0,1]
        # focus: when D > 0.5, scale < 1 (narrow); when D < 0.5, scale > 1 (widen)
        scale = 1.0 - K_SIG * (D - 0.5)
        return float(np.clip(scale, SIG_CLIP[0], SIG_CLIP[1]))

    def begin_step(self, sparse_width: int | None = None) -> None:
        # Read kernel observables
        self._step_h = int(self.kernel.current_horizon.item())
        self._step_chi = int(self.kernel.current_vertex.item())
        self._step_p = int(self.kernel.current_phase.item())
        self._step_mu = (self._step_h + self._step_p * QUARTER_TURN) % N_BOUNDARY
        self._step_mu_qturn = (self._step_mu + QUARTER_TURN) % N_BOUNDARY

        h = self._step_h
        chi = self._step_chi
        p = self._step_p

        last_b = int(self.kernel.last_byte[0])
        w_b = int(self.kernel.byte_weight[last_b])

        # Distances in code space
        distances = self._cdist[h, :].astype(np.float32)  # [256], values 0..12

        # sigma_focus implemented as distance scaling on the LUT sampling
        sscale = self._sigma_scale(self._prev_h, h)
        d_eff = distances / sscale  # sscale<1 => larger effective distance => narrower

        # LUT sampling (linear interp on d_eff in [0,12])
        chi_idx = min(max(chi, 0), 3)
        p_idx = min(max(p, 0), 3)
        base = self._lut[chi_idx, p_idx, :].astype(np.float32)  # [13]

        d0 = np.clip(d_eff.astype(np.int32), 0, 12)
        d1 = np.clip(d0 + 1, 0, 12)
        frac = (d_eff - d0.astype(np.float32)).astype(np.float32)
        mask_base = (1.0 - frac) * base[d0] + frac * base[d1]  # [256]

        # Wedge (unchanged)
        same_chi = (self.kernel.byte_charge == chi).astype(np.float32)
        mask_wedge = mask_base * (1.0 + 0.2 * same_chi)

        # Alpha (byte weight) + optional alpha_CH regulator
        alpha0 = 0.1 + 0.2 * (w_b / 12.0)
        alpha0 = float(np.clip(alpha0, 0.05, 0.35))
        alpha = float(np.clip(alpha0 * self._alpha_reg(), 0.03, 0.40))

        mask_np = 1.0 + alpha * (mask_wedge - 1.0)

        # Differential modulation (unchanged)
        if self._prev_h is not None:
            td = int(self._cdist[self._prev_h, h])
            diff_scale = 0.5 + 0.5 * (td / 12.0)
            mask_np = 1.0 + diff_scale * (mask_np - 1.0)

        # Mean normalization
        mask_np = (mask_np / (float(mask_np.mean()) + 1e-8)).astype(np.float32)
        self._mask_boundary = mask_np

        # Apply mask to wrapped layers
        mt = torch.from_numpy(mask_np).to(self._device, self._dtype)
        self._collector = []
        for w in self._wrapped.values():
            w.set_collector(self._collector)
            w.set_mask(mt)

        self._prev_h = h


# ============================================================================
# Metrics + printing
# ============================================================================

def monodromy_from_log(byte_log: list[int]) -> float:
    if len(byte_log) < 2:
        return float("nan")
    O, E, _ = trajectory_parity_commitment(byte_log)
    return (popcount(O) / 12.0 + popcount(E) / 12.0) / 2.0


def grammaticality_from_log(byte_log: list[int]) -> float:
    if not byte_log:
        return 1.0
    viol = 0
    for b in byte_log:
        m = mask12_for_byte(b)
        if any((popcount(m & v) & 1) for v in C_PERP_12):
            viol += 1
    return 1.0 - viol / len(byte_log)


@dataclass
class R:
    label: str
    ppl: float
    corr: float
    corr_std: float
    code_dist: float
    h_ent: float
    h_norm: float
    unique_h: int
    tok_s: float
    gain_peak: float
    gain_mu: float
    mono: float
    gram: float
    text: str
    error: str = ""


def row_from_labe(label: str, labe: GyroLabe, result: GenerationResult) -> R:
    s = labe.stats()
    return R(
        label=label,
        ppl=float(result.perplexity),
        corr=float(s.get("mean_correlation", float("nan"))),
        corr_std=float(s.get("std_correlation", float("nan"))),
        code_dist=float(s.get("mean_code_dist", float("nan"))),
        h_ent=float(s.get("h_entropy", float("nan"))),
        h_norm=float(s.get("h_entropy", 0.0)) / 8.0,
        unique_h=int(s.get("unique_h", 0)),
        tok_s=float(result.tokens_per_second),
        gain_peak=float(s.get("mean_gain_at_peak", float("nan"))),
        gain_mu=float(s.get("mean_gain_at_mu", float("nan"))),
        mono=float(monodromy_from_log(labe.byte_log)),
        gram=float(grammaticality_from_log(labe.byte_log)),
        text=result.text.replace("\n", " "),
    )


def print_row(r: R, baseline_ppl: float, standard_ppl: float) -> None:
    if r.error:
        print(f"  {r.label:<26} ERROR: {r.error}")
        return

    dp_bl = r.ppl - baseline_ppl
    dp_st = r.ppl - standard_ppl

    print(
        f"  {r.label:<26}"
        f" ppl={r.ppl:.3f}"
        f"  Δbase={dp_bl:+.3f}"
        f"  Δstd={dp_st:+.3f}"
        f"  corr={fmt(r.corr)}±{fmt(r.corr_std)}"
        f"  code={fmt(r.code_dist, 1)}"
        f"  H={fmt(r.h_ent, 2)}b({fmt(r.h_norm, 2)})"
        f"  h={r.unique_h}/256"
        f"  gain_peak={fmt(r.gain_peak, 3)}"
        f"  mono={fmt(r.mono, 3)}"
        f"  {r.tok_s:.1f}t/s"
    )
    print(f"    {r.text[:120]}...")
    print()


def print_physics_check(labe: GyroLabe) -> None:
    s = labe.stats()
    O, E, par = trajectory_parity_commitment(labe.byte_log)
    print("  Physics check:")
    print(f"    O=0x{O:03x}  E=0x{E:03x}  parity={par}")
    print(f"    corr={fmt(float(s.get('mean_correlation', float('nan'))))}  target={TARGET_CORR:.4f}")
    print(f"    H(h)={fmt(float(s.get('h_entropy', float('nan'))), 3)} bits"
          f"  unique_h={int(s.get('unique_h', 0))}/256")
    print(f"    gain_peak={fmt(float(s.get('mean_gain_at_peak', float('nan'))), 4)}"
          f"  gain_mu={fmt(float(s.get('mean_gain_at_mu', float('nan'))), 4)}")
    print(f"    monodromy={fmt(monodromy_from_log(labe.byte_log), 4)}"
          f"  gram={fmt(grammaticality_from_log(labe.byte_log), 4)}")
    print()


def print_summary(all_results: dict[str, list[R]]) -> None:
    sep("SUMMARY (avg Δstd across prompts)")
    # average Δstd for each label across prompts
    labels = list(dict.fromkeys(r.label for rs in all_results.values() for r in rs))
    # get standard per prompt
    std_by_prompt: dict[str, float] = {}
    for pk, rs in all_results.items():
        for r in rs:
            if r.label == "standard" and not r.error:
                std_by_prompt[pk] = r.ppl
    for label in labels:
        ds = []
        for pk, rs in all_results.items():
            std = std_by_prompt.get(pk, float("nan"))
            for r in rs:
                if r.label == label and not r.error and not math.isnan(std):
                    ds.append(r.ppl - std)
        if ds:
            print(f"  {label:<26} avg_Δstd={sum(ds)/len(ds):+.3f}  "
                  f"min={min(ds):+.3f}  max={max(ds):+.3f}")
    print()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    device = detect_device()
    dtype = choose_dtype(device)

    print(f"Loading {MODEL_DIR.name} on {device} ({dtype})")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    d = model.config.hidden_size
    n = len(model.model.layers)
    print(f"  {d}d, {n} layers, {d // N_BOUNDARY} fibers\n")

    sep("CGM constants (context)")
    print(f"  M_A={_M_A:.5f}  delta_BU={DELTA_BU:.6f}  A*={A_STAR:.4f}  target_corr={TARGET_CORR:.4f}")

    sep("Mask-layer choices under test")
    print("  radial: Gaussian LUT (CGM sigma) vs Krawtchouk J=4 projection of that LUT")
    print("  dynamics: sigma_focus (distance scaling) and alpha_CH (C,H regulator)")

    # Precompute code distance
    cdist = get_code_distance_matrix()
    print(f"  code_dist_matrix mean={cdist.mean():.2f}")

    # Precompute Kraw J=4 LUT from the current Gaussian LUT
    kraw4 = build_kraw_lut_from_gaussian(J=4)
    err = kraw_energy_error(kraw4)
    print(f"  kraw_J4 energy-weighted relative error vs Gaussian LUT: {err:.4f}")

    # Experiment set (ONLY best candidates, factorial)
    # label, use_kraw, sigma_focus, alpha_CH
    configs = [
        ("standard", False, False, False),
        ("kraw_J4", True, False, False),
        ("sigma_focus", False, True, False),
        ("alpha_CH", False, False, True),
        ("kraw_J4+sigma_focus", True, True, False),
        ("kraw_J4+alpha_CH", True, False, True),
        ("sigma_focus+alpha_CH", False, True, True),
        ("kraw_J4+sigma_focus+alpha_CH", True, True, True),
    ]

    all_results: dict[str, list[R]] = {}
    physics_done = False

    for prompt_key, prompt_text in PROMPTS:
        sep(f"PROMPT: {prompt_key}")
        print(f"  {prompt_text!r}\n")

        baseline = generate(
            model, tok, labe=None,
            prompt=prompt_text,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            seed=SEED,
        )
        bl_ppl = baseline.perplexity
        print(f"  baseline ppl={bl_ppl:.3f}  {baseline.tokens_per_second:.1f}t/s")
        print(f"    {baseline.text.replace(chr(10),' ')[:120]}...\n")

        rows: list[R] = []

        # We'll compute standard_ppl after the standard run
        standard_ppl = float("nan")

        for label, use_kraw, use_focus, use_alpha in configs:
            try:
                lut = kraw4 if use_kraw else _GAUSSIAN_LUT
                labe = BestMaskGyroLabe(
                    model,
                    atlas_dir=ATLAS_DIR,
                    config=CouplingConfig(),
                    radial_lut=lut,
                    use_sigma_focus=use_focus,
                    use_alpha_ch=use_alpha,
                )
                labe.install()
                try:
                    out = generate(
                        model, tok, labe=labe,
                        prompt=prompt_text,
                        max_new_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                        seed=SEED,
                    )
                finally:
                    labe.restore()

                r = row_from_labe(label, labe, out)
                rows.append(r)

                if label == "standard":
                    standard_ppl = r.ppl
                    if not physics_done:
                        print_physics_check(labe)
                        physics_done = True

            except Exception as e:
                r = R(
                    label=label,
                    ppl=float("nan"),
                    corr=float("nan"),
                    corr_std=float("nan"),
                    code_dist=float("nan"),
                    h_ent=float("nan"),
                    h_norm=float("nan"),
                    unique_h=0,
                    tok_s=0.0,
                    gain_peak=float("nan"),
                    gain_mu=float("nan"),
                    mono=float("nan"),
                    gram=float("nan"),
                    text="",
                    error=f"{e} [{traceback.format_exc().splitlines()[-1]}]",
                )
                rows.append(r)

        # Print rows in config order, now that standard_ppl is known
        for r in rows:
            print_row(r, bl_ppl, standard_ppl)

        all_results[prompt_key] = rows

    print_summary(all_results)

    sep("DONE")
    print(f"  {len(PROMPTS)} prompts x {len(configs)} variants x {MAX_TOKENS} tokens  temp={TEMPERATURE}  seed={SEED}")


if __name__ == "__main__":
    main()