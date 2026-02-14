# src/tools/gyrolabe.py
"""
GyroLabe: Holographic Coordination Logistics System.

Couples a generative AI model to the GGG ASI Alignment Router Kernel,
establishing a closed-loop coordination cycle between the model's
high-dimensional inference and the kernel's finite geometric reference frame.

The kernel is deterministic. The model remains stochastic. The kernel
shapes the landscape from which the model samples, without overriding
the model's generative capacity.

Coordination cycle (per inference step):
    1. Kernel exposes observables: horizon (h), vertex charge (chi), phase (p)
    2. Projection: observables become a mask on model activations
    3. Model processes masked input (stochastic inference preserved)
    4. Model samples token → token_id & 0xFF → byte advances kernel
    5. Extraction from activations is telemetry only (does not drive kernel)
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.router.kernel import RouterKernel
from src.router.constants import trajectory_parity_commitment, mask12_for_byte, vertex_charge_from_mask, popcount

N_BOUNDARY = 256
QUARTER_TURN = 64


@dataclass
class CouplingConfig:
    """Configuration for GyroLabe model coupling."""
    routed_layers: list[int] | None = None
    store_layer_telemetry: bool = True


def detect_device() -> torch.device:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(device: torch.device) -> torch.dtype:
    """Select appropriate dtype for a device."""
    if device.type == "cuda":
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except AttributeError:
            pass
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.bfloat16


def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy in bits from a count array."""
    p = counts.astype(np.float64)
    total = p.sum()
    if total == 0:
        return 0.0
    p = p / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def build_mask12_table() -> np.ndarray:
    """Build table of 12-bit masks for all 256 bytes."""
    return np.array([mask12_for_byte(b) for b in range(256)], dtype=np.uint16)


def build_byte_charge_table() -> np.ndarray:
    """Build table of vertex charges for all 256 bytes."""
    mask12_table = build_mask12_table()
    return np.array([vertex_charge_from_mask(int(m)) for m in mask12_table], dtype=np.uint8)


def build_code_distance_matrix(mask12_table: np.ndarray) -> np.ndarray:
    """Build 256x256 Hamming distance matrix on mask code."""
    dist = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            dist[i, j] = popcount(int(mask12_table[i]) ^ int(mask12_table[j]))
    return dist


_MASK12_TABLE: np.ndarray | None = None
_BYTE_CHARGE_TABLE: np.ndarray | None = None
_CODE_DISTANCE_MATRIX: np.ndarray | None = None


def get_mask12_table() -> np.ndarray:
    global _MASK12_TABLE
    if _MASK12_TABLE is None:
        _MASK12_TABLE = build_mask12_table()
    return _MASK12_TABLE


def get_byte_charge_table() -> np.ndarray:
    global _BYTE_CHARGE_TABLE
    if _BYTE_CHARGE_TABLE is None:
        _BYTE_CHARGE_TABLE = build_byte_charge_table()
    return _BYTE_CHARGE_TABLE


def get_code_distance_matrix() -> np.ndarray:
    global _CODE_DISTANCE_MATRIX
    if _CODE_DISTANCE_MATRIX is None:
        _CODE_DISTANCE_MATRIX = build_code_distance_matrix(get_mask12_table())
    return _CODE_DISTANCE_MATRIX


def _kernel_aperture_mass() -> float:
    """
    Kernel intrinsic aperture mass:
      A_kernel = P(weight <= 1) = (1 + 4)/256 = 5/256 ≈ 0.01953
    """
    m = get_mask12_table()
    w = np.array([popcount(int(x)) for x in m], dtype=np.int64)
    return float(np.mean(w <= 1))


_A_KERNEL: float = _kernel_aperture_mass()


def _rerank_topk_logits_kernel_native(
    topv: Tensor,
    topi: Tensor,
    kernel: RouterKernel,
) -> Tensor:
    """
    Kernel-native geometric re-ranking on the model's top-k logits.

    Uses the common language token -> byte (token_id & 0xFF), then scores each
    candidate byte using kernel invariants:
      - code-distance in the 12-bit mask code from current horizon index h
      - K₄ vertex charge compatibility with current χ

    The adjustment magnitude is aperture-scaled:
      Δlogit ~ A_kernel * (logit_spread_topk) * score
    """
    if topi.numel() == 0:
        return topv

    h = int(kernel.current_horizon)
    chi = int(kernel.current_vertex)

    code_dist = get_code_distance_matrix()

    toks = topi.detach().to("cpu").numpy().astype(np.int64)
    bytes_ = (toks & 0xFF).astype(np.int64)

    d = code_dist[h, bytes_].astype(np.float32)
    align = 1.0 - (d / 12.0)

    charges = kernel.byte_charge[bytes_].astype(np.int64)
    same_vertex = (charges == chi).astype(np.float32)

    weights = kernel.byte_weight[bytes_].astype(np.int64)
    gamma = kernel.gamma_table[chi, charges, weights]

    score = gamma * align * (1.0 + 0.2 * same_vertex)

    spread = float((topv.max() - topv.min()).detach().item())
    alpha = _A_KERNEL * spread

    adj = torch.from_numpy(score).to(device=topv.device, dtype=topv.dtype)
    return topv + alpha * adj


def compute_mask(
    device: torch.device,
    dtype: torch.dtype,
    h: int,
    chi: int,
    p: int,
    n_fiber: int,
    last_byte_weight: int,
) -> Tensor:
    """Compute projection mask using kernel-native geometry."""
    code_dist_matrix = get_code_distance_matrix()
    byte_charges = get_byte_charge_table()

    distances = code_dist_matrix[h, :].astype(np.float32)

    base_sigma_map = {0: 2.0, 1: 4.0, 2: 3.0, 3: 2.5}
    sigma = base_sigma_map.get(chi, 3.0)

    phase_factor = 1.0 + 0.1 * (p - 1.5) / 1.5
    sigma = max(0.5, sigma * phase_factor)

    mask_base = np.exp(-0.5 * (distances / sigma) ** 2)

    same_chi = (byte_charges == chi).astype(np.float32)
    wedge_factor = 1.0 + 0.2 * same_chi
    mask_wedge = mask_base * wedge_factor

    w = float(last_byte_weight)
    alpha = 0.1 + 0.2 * (w / 12.0)
    alpha = max(0.05, min(alpha, 0.35))

    mask_np = 1.0 + alpha * (mask_wedge - 1.0)
    mask_np = mask_np / (mask_np.mean() + 1e-8)

    mask = torch.from_numpy(mask_np).to(device=device, dtype=dtype)
    return mask.unsqueeze(-1).expand(-1, n_fiber).reshape(1, -1)


def extract_byte(x_out: Tensor, n_fiber: int) -> tuple[int, int, float, Tensor]:
    """Extract a byte from output activations via fiber sign pattern.

    This is telemetry only - the extracted byte does not drive the kernel.
    """
    with torch.no_grad():
        xf = x_out.float().view(-1, N_BOUNDARY, n_fiber)
        energy = (xf * xf).sum(dim=(0, 2))
        h_peak = int(torch.argmax(energy).item())
        total = energy.sum().item()
        peak_mass = float(energy[h_peak].item() / max(total, 1e-10))

        fiber = xf[:, h_peak, :].mean(dim=0)
        bits = 0
        for i in range(8):
            idx = (i * 2) % n_fiber
            if fiber[idx].item() > 0:
                bits |= (1 << i)
        return bits & 0xFF, h_peak, peak_mass, energy


class RoutedMLP(nn.Module):
    """Wraps a SwiGLU MLP layer with kernel-driven projection and extraction telemetry."""

    def __init__(
        self,
        mlp: nn.Module,
        kernel: RouterKernel,
        layer_idx: int,
        config: CouplingConfig,
        n_fiber: int,
    ):
        super().__init__()
        required = ("gate_proj", "up_proj", "down_proj")
        missing = [a for a in required if not hasattr(mlp, a)]
        if missing:
            raise TypeError(
                f"MLP missing {missing}. Expected SwiGLU architecture "
                f"with gate_proj, up_proj, down_proj."
            )
        self.mlp = mlp
        self.kernel = kernel
        self.layer_idx = layer_idx
        self.config = config
        self.n_fiber = n_fiber
        self._collector: list[tuple[int, int, int, float, Tensor]] | None = None
        self._mask: Tensor | None = None

    def set_collector(self, collector: list[tuple[int, int, int, float, Tensor]] | None) -> None:
        self._collector = collector

    def set_mask(self, mask: Tensor | None) -> None:
        self._mask = mask

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        d_model = shape[-1]
        x_flat = x.reshape(-1, d_model)

        gate = self.mlp.gate_proj(x_flat)
        up = self.mlp.up_proj(x_flat)
        z = F.silu(gate) * up

        hidden_dim = z.shape[-1]
        if hidden_dim % 256 != 0:
            raise ValueError(
                f"MLP hidden dim {hidden_dim} is not divisible by 256."
            )
        n_feat = hidden_dim // N_BOUNDARY

        last_byte_weight = int(self.kernel.byte_weight[self.kernel.last_byte])
        boundary_mask = compute_mask(
            z.device, z.dtype,
            self.kernel.current_horizon,
            self.kernel.current_vertex,
            self.kernel.current_phase,
            n_fiber=1,
            last_byte_weight=last_byte_weight,
        )

        boundary_mask = boundary_mask.view(N_BOUNDARY)

        z_view = z.view(-1, N_BOUNDARY, n_feat)
        z_masked = z_view * boundary_mask.view(1, N_BOUNDARY, 1)
        z_flat = z_masked.view(-1, hidden_dim)

        out = self.mlp.down_proj(z_flat)

        if self._collector is not None:
            b, h_peak, peak_mass, energy = extract_byte(out, self.n_fiber)
            self._collector.append((self.layer_idx, b, h_peak, peak_mass, energy))

        return out.view(*shape[:-1], d_model)


class GyroLabe:
    """Holographic Coordination Logistics System.

    Couples a generative AI model to the GGG ASI Alignment Router Kernel.
    
    The closed loop:
        Kernel state → Projection mask → Model inference → Token → Byte → Kernel state
    
    The kernel advances ONLY via tokens (the Common Language).
    Extraction from activations is telemetry for measuring alignment.
    """

    def __init__(
        self,
        model: Any,
        atlas_dir: str | Path,
        config: CouplingConfig | None = None,
    ):
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise TypeError(
                "Model architecture not supported. Expected model.model.layers "
                "(LLaMA, OLMo, Mistral family)."
            )

        self.model = model
        self.config = config or CouplingConfig()
        self.kernel = RouterKernel(atlas_dir=Path(atlas_dir))

        self.d_model: int = model.config.hidden_size
        if self.d_model % N_BOUNDARY != 0:
            raise ValueError(
                f"hidden_size {self.d_model} must be a multiple of {N_BOUNDARY}"
            )
        self.n_fiber: int = self.d_model // N_BOUNDARY

        if self.config.routed_layers is not None:
            self.routed_layers = list(self.config.routed_layers)
        else:
            self.routed_layers = _detect_routed_layers(model)

        self._originals: dict[int, nn.Module] = {}
        self._wrapped: dict[int, RoutedMLP] = {}
        self._installed = False
        self._collector: list[tuple[int, int, int, float, Tensor]] | None = None
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32

        self._step_h: int = 0
        self._step_chi: int = 0
        self._step_p: int = 0
        self._step_mu: int = 0
        self._step_mu_qturn: int = 0
        
        self._mask_boundary: np.ndarray | None = None

        self.byte_log: list[int] = []
        self.trajectory: list[dict[str, Any]] = []

    def install(self) -> None:
        """Install routing wrappers on the configured MLP layers."""
        if self._installed:
            return

        self._device = next(self.model.parameters()).device
        self._dtype = next(self.model.parameters()).dtype

        layers = self.model.model.layers
        for i in self.routed_layers:
            orig = layers[i].mlp
            self._originals[i] = orig
            wrapped = RoutedMLP(
                orig, self.kernel, i, self.config, self.n_fiber,
            )
            self._wrapped[i] = wrapped
            layers[i].mlp = wrapped

        self._installed = True

    def restore(self) -> None:
        """Remove routing wrappers and restore original MLP layers."""
        if not self._installed:
            return

        layers = self.model.model.layers
        for i, orig in self._originals.items():
            layers[i].mlp = orig

        self._originals.clear()
        self._wrapped.clear()
        self._installed = False

    def begin_step(self) -> None:
        """Begin a coordination step: read kernel state and prepare mask."""
        self._step_h = self.kernel.current_horizon
        self._step_chi = self.kernel.current_vertex
        self._step_p = self.kernel.current_phase
        self._step_mu = (self._step_h + self._step_p * QUARTER_TURN) % N_BOUNDARY
        self._step_mu_qturn = (self._step_mu + QUARTER_TURN) % N_BOUNDARY
        
        last_byte_weight = int(self.kernel.byte_weight[self.kernel.last_byte])

        mask = compute_mask(
            self._device, self._dtype,
            self._step_h, self._step_chi, self._step_p,
            self.n_fiber,
            last_byte_weight,
        )

        mask_reshaped = mask.view(N_BOUNDARY, self.n_fiber)
        self._mask_boundary = mask_reshaped[:, 0].float().cpu().numpy()

        self._collector = []
        for w in self._wrapped.values():
            w.set_collector(self._collector)
            w.set_mask(mask)

    def end_step(self) -> dict[str, Any]:
        """End a coordination step: collect telemetry from extraction."""
        extracted_byte = 0
        layer_data = []
        correlations = []

        if self._collector and self._mask_boundary is not None:
            mask_b = self._mask_boundary
            code_dist_matrix = get_code_distance_matrix()
            
            for layer_idx, b, h_peak, peak_mass, energy in self._collector:
                salt = (layer_idx * 29 + 17) & 0xFF
                extracted_byte ^= (b ^ salt)

                mu = self._step_mu
                dist_to_mu = min((h_peak - mu) % N_BOUNDARY, (mu - h_peak) % N_BOUNDARY)
                
                mu_qturn = self._step_mu_qturn
                dist_to_qturn = min((h_peak - mu_qturn) % N_BOUNDARY, (mu_qturn - h_peak) % N_BOUNDARY)

                code_dist = int(code_dist_matrix[self._step_h, h_peak])

                gain = float(mask_b[h_peak])

                energy_np = energy.cpu().numpy().astype(np.float64)
                mask_np = mask_b.astype(np.float64)
                
                e_norm = np.linalg.norm(energy_np)
                m_norm = np.linalg.norm(mask_np)
                if e_norm > 1e-10 and m_norm > 1e-10:
                    correlation = float(np.dot(energy_np, mask_np) / (e_norm * m_norm))
                else:
                    correlation = 0.0
                correlations.append(correlation)

                if self.config.store_layer_telemetry:
                    layer_data.append({
                        "layer_idx": layer_idx,
                        "extracted_byte": b,
                        "h_peak": h_peak,
                        "peak_mass": peak_mass,
                        "mu": mu,
                        "dist_to_mu": dist_to_mu,
                        "dist_to_qturn": dist_to_qturn,
                        "code_dist": code_dist,
                        "gain_at_peak": gain,
                        "correlation": correlation,
                    })

            extracted_byte &= 0xFF

        gain_at_mu = float(self._mask_boundary[self._step_mu]) if self._mask_boundary is not None else 1.0
        gain_at_qturn = float(self._mask_boundary[self._step_mu_qturn]) if self._mask_boundary is not None else 1.0

        step_record: dict[str, Any] = {
            "step": self.kernel.step,
            "h": self._step_h,
            "chi": self._step_chi,
            "p": self._step_p,
            "mu": self._step_mu,
            "mu_qturn": self._step_mu_qturn,
            "extracted_byte": extracted_byte,
            "gain_at_mu": gain_at_mu,
            "gain_at_qturn": gain_at_qturn,
        }
        
        if self.config.store_layer_telemetry and layer_data:
            step_record["layers"] = layer_data
        
        if layer_data:
            step_record["mean_dist_to_mu"] = float(np.mean([d["dist_to_mu"] for d in layer_data]))
            step_record["mean_dist_to_qturn"] = float(np.mean([d["dist_to_qturn"] for d in layer_data]))
            step_record["mean_code_dist"] = float(np.mean([d["code_dist"] for d in layer_data]))
            step_record["mean_gain_at_peak"] = float(np.mean([d["gain_at_peak"] for d in layer_data]))
            step_record["mean_correlation"] = float(np.mean(correlations))

        self.trajectory.append(step_record)

        for w in self._wrapped.values():
            w.set_collector(None)
            w.set_mask(None)
        self._collector = None
        self._mask_boundary = None

        return step_record

    def advance_with_token(self, token_id: int) -> int:
        """Advance the kernel using token ID as the byte source."""
        byte = int(token_id) & 0xFF
        self.kernel.step_byte(byte)
        self.byte_log.append(byte)
        
        if self.trajectory:
            self.trajectory[-1]["driving_byte"] = byte
            
        return byte

    def prime_from_tokens(self, token_ids: list[int]) -> None:
        """Prime the kernel from prompt tokens."""
        for tid in token_ids:
            byte = int(tid) & 0xFF
            self.kernel.step_byte(byte)
            self.byte_log.append(byte)

    def reset(self) -> None:
        """Reset kernel to archetype and clear trajectory logs."""
        self.kernel.reset()
        self.byte_log.clear()
        self.trajectory.clear()

    def stats(self) -> dict[str, Any]:
        """Compute trajectory statistics with full physics diagnostics."""
        if not self.trajectory:
            return {"steps": 0}

        h_arr = np.array([t["h"] for t in self.trajectory])
        chi_arr = np.array([t["chi"] for t in self.trajectory])
        b_arr = np.array([t.get("driving_byte", 0) for t in self.trajectory])

        h_counts = np.bincount(h_arr, minlength=256)
        chi_counts = np.bincount(chi_arr, minlength=4)
        b_counts = np.bincount(b_arr, minlength=256)

        byte_weights = self.kernel.byte_weight[b_arr]
        w_counts = np.bincount(byte_weights.astype(np.int64), minlength=13)

        O, E, parity = trajectory_parity_commitment(self.byte_log)

        dist_mu_data = [t.get("mean_dist_to_mu", 0.0) for t in self.trajectory if "mean_dist_to_mu" in t]
        dist_qturn_data = [t.get("mean_dist_to_qturn", 0.0) for t in self.trajectory if "mean_dist_to_qturn" in t]
        code_dist_data = [t.get("mean_code_dist", 0.0) for t in self.trajectory if "mean_code_dist" in t]
        gain_peak_data = [t.get("mean_gain_at_peak", 1.0) for t in self.trajectory if "mean_gain_at_peak" in t]
        gain_mu_data = [t.get("gain_at_mu", 1.0) for t in self.trajectory if "gain_at_mu" in t]
        gain_qturn_data = [t.get("gain_at_qturn", 1.0) for t in self.trajectory if "gain_at_qturn" in t]
        corr_data = [t.get("mean_correlation", 0.0) for t in self.trajectory if "mean_correlation" in t]

        layer_h_peaks = []
        layer_gains = []
        layer_dists_mu = []
        layer_dists_qturn = []
        layer_code_dists = []
        layer_correlations = []
        pmass_data = []

        for t in self.trajectory:
            if "layers" in t:
                for d in t["layers"]:
                    layer_h_peaks.append(d["h_peak"])
                    layer_gains.append(d["gain_at_peak"])
                    layer_dists_mu.append(d["dist_to_mu"])
                    layer_dists_qturn.append(d["dist_to_qturn"])
                    layer_code_dists.append(d["code_dist"])
                    layer_correlations.append(d.get("correlation", 0.0))
                    pmass_data.append(d["peak_mass"])

        return {
            "steps": len(self.trajectory),
            "unique_h": int(len(np.unique(h_arr))),
            "h_entropy": round(_entropy(h_counts), 2),
            "chi_dist": chi_counts.tolist(),
            "unique_bytes": int(len(np.unique(b_arr))),
            "b_entropy": round(_entropy(b_counts), 2),
            "mean_byte_weight": round(float(byte_weights.mean()), 2),
            "byte_weight_hist": w_counts.tolist(),
            "mean_dist_to_mu": round(float(np.mean(dist_mu_data)), 2) if dist_mu_data else 0.0,
            "std_dist_to_mu": round(float(np.std(dist_mu_data)), 2) if len(dist_mu_data) > 1 else 0.0,
            "mean_dist_to_qturn": round(float(np.mean(dist_qturn_data)), 2) if dist_qturn_data else 0.0,
            "std_dist_to_qturn": round(float(np.std(dist_qturn_data)), 2) if len(dist_qturn_data) > 1 else 0.0,
            "mean_code_dist": round(float(np.mean(code_dist_data)), 2) if code_dist_data else 0.0,
            "std_code_dist": round(float(np.std(code_dist_data)), 2) if len(code_dist_data) > 1 else 0.0,
            "mean_gain_at_peak": round(float(np.mean(gain_peak_data)), 4) if gain_peak_data else 1.0,
            "std_gain_at_peak": round(float(np.std(gain_peak_data)), 4) if len(gain_peak_data) > 1 else 0.0,
            "mean_gain_at_mu": round(float(np.mean(gain_mu_data)), 4) if gain_mu_data else 1.0,
            "mean_gain_at_qturn": round(float(np.mean(gain_qturn_data)), 4) if gain_qturn_data else 1.0,
            "mean_correlation": round(float(np.mean(corr_data)), 4) if corr_data else 0.0,
            "std_correlation": round(float(np.std(corr_data)), 4) if len(corr_data) > 1 else 0.0,
            "mean_peak_mass": round(float(np.mean(pmass_data)), 4) if pmass_data else 0.0,
            "parity_O": O,
            "parity_E": E,
            "parity_n_mod_2": parity,
            "kernel_state": self.kernel.signature(),
            "layer_h_peaks": layer_h_peaks,
            "layer_gains": layer_gains,
            "layer_dists_mu": layer_dists_mu,
            "layer_dists_qturn": layer_dists_qturn,
            "layer_code_dists": layer_code_dists,
            "layer_correlations": layer_correlations,
        }


def _detect_routed_layers(model: Any) -> list[int]:
    """Detect which layers to route based on model configuration."""
    layer_types = getattr(model.config, "layer_types", None)
    if isinstance(layer_types, list):
        full_attn = [
            i for i, t in enumerate(layer_types)
            if str(t) == "full_attention"
        ]
        if full_attn:
            return full_attn

    n_layers = len(model.model.layers)
    return list(range(3, n_layers, 4))


@dataclass
class GenerationResult:
    """Output of the generate function."""
    text: str
    elapsed: float
    n_tokens: int
    logprobs: list[float] = field(default_factory=list)

    @property
    def mean_logprob(self) -> float:
        if not self.logprobs:
            return 0.0
        return sum(self.logprobs) / len(self.logprobs)

    @property
    def perplexity(self) -> float:
        if not self.logprobs:
            return 1.0
        return math.exp(-self.mean_logprob)

    @property
    def tokens_per_second(self) -> float:
        return self.n_tokens / max(self.elapsed, 1e-9)


@torch.inference_mode()
def generate(
    model: Any,
    tokenizer: Any,
    labe: GyroLabe | None,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 40,
    seed: int | None = None,
) -> GenerationResult:
    """Generate text with optional GyroLabe coordination."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    prompt_ids = input_ids[0].tolist()
    generated: list[int] = []
    logprobs: list[float] = []
    past = None

    if labe is not None:
        labe.prime_from_tokens(prompt_ids)

    t0 = time.perf_counter()
    for step in range(max_new_tokens):
        if labe is not None:
            labe.begin_step()

        out = model(input_ids, use_cache=True, past_key_values=past)
        past = getattr(out, "past_key_values", None)

        if past is None and step == 0:
            warnings.warn(
                "Model did not return past_key_values. "
                "Generation will run without KV cache (slower).",
                stacklevel=2,
            )

        if labe is not None:
            labe.end_step()

        raw_logits = out.logits[0, -1]
        log_probs_full = torch.log_softmax(raw_logits, dim=-1)

        scaled_logits = raw_logits / max(temperature, 1e-8)
        if top_k > 0:
            k = min(top_k, scaled_logits.size(-1))
            topv, topi = torch.topk(scaled_logits, k=k)

            if labe is not None:
                topv = _rerank_topk_logits_kernel_native(topv, topi, labe.kernel)

            probs = torch.softmax(topv, dim=-1)
            sample_idx = torch.multinomial(probs, 1)
            next_tok = int(topi[sample_idx].item())
        else:
            probs = torch.softmax(scaled_logits, dim=-1)
            next_tok = int(torch.multinomial(probs, 1).item())

        token_logprob = float(log_probs_full[next_tok].item())
        logprobs.append(token_logprob)
        generated.append(next_tok)

        if labe is not None:
            labe.advance_with_token(next_tok)

        if tokenizer.eos_token_id is not None and next_tok == tokenizer.eos_token_id:
            break

        if past is not None:
            input_ids = torch.tensor(
                [[next_tok]], device=device, dtype=input_ids.dtype,
            )
        else:
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_tok]], device=device, dtype=input_ids.dtype),
            ], dim=1)

    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(prompt_ids + generated, skip_special_tokens=True)

    return GenerationResult(
        text=text,
        elapsed=elapsed,
        n_tokens=len(generated),
        logprobs=logprobs,
    )