# src/tools/gyrolabe.py
"""
GyroLabe: An AI Calibration Instrument.

Couples a generative AI model to the GGG ASI Alignment Router Kernel,
establishing a closed-loop coordination cycle between the model's
high-dimensional inference and the kernel's finite geometric reference frame.

The kernel is deterministic. The model remains stochastic.

Coordination cycle (per inference step):
    1. Kernel exposes observables: horizon (h), vertex charge (chi), phase (p)
    2. Projection: observables become a mask on model activations
    3. Model processes masked input (stochastic inference preserved)
    4. Model samples token -> token_id & 0xFF -> byte advances kernel
    5. Extraction from activations is telemetry only (does not drive kernel)
"""

from __future__ import annotations

import math as _math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

_M_A = 1.0 / (2.0 * _math.sqrt(2.0 * _math.pi))
_S_CS = (_math.pi / 2.0) / _M_A
_S_UNA = (1.0 / _math.sqrt(2.0)) / _M_A
_S_ONA = (_math.pi / 4.0) / _M_A
_S_BU_SIGMA = 24.0

from src.router.constants import mask12_for_byte, popcount, trajectory_parity_commitment
from src.router.kernel import RouterKernel

N_BOUNDARY = 256
QUARTER_TURN = 64

# ---------------------------------------------------------------------------
# Dynamic sigma focusing (kernel-native, CGM-compliant refinement)
#
# Uses the code-space motion between consecutive horizons:
#   D = code_dist(prev_h, h) / 12 in [0,1]
# and applies a small multiplicative sigma scale:
#   sigma_scale = 1 - K*(D - 0.5)
# so large jumps (D>0.5) narrow the mask and small jumps widen it slightly.
# ---------------------------------------------------------------------------
SIGMA_FOCUS_K: float = 0.40
SIGMA_FOCUS_CLIP = (0.30, 2.50)


@dataclass
class CouplingConfig:
    """Configuration for GyroLabe model coupling."""
    routed_layers: list[int] | None = None
    store_layer_telemetry: bool = True
    couple_local_decoder: bool = False
    couple_local_encoder: bool = False


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(device: torch.device) -> torch.dtype:
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


def build_code_distance_matrix(mask12_table: np.ndarray) -> np.ndarray:
    """Build 256x256 Hamming distance matrix on mask code."""
    dist = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            dist[i, j] = popcount(int(mask12_table[i]) ^ int(mask12_table[j]))
    return dist


_MASK12_TABLE: np.ndarray | None = None
_CODE_DISTANCE_MATRIX: np.ndarray | None = None
_P_CODE_PRIOR: np.ndarray | None = None
_MASK12_TO_BYTE: dict[int, int] | None = None
ACTIVE_BETA: float = 5.0 / 256.0


def get_mask12_table() -> np.ndarray:
    global _MASK12_TABLE
    if _MASK12_TABLE is None:
        _MASK12_TABLE = build_mask12_table()
    return _MASK12_TABLE


def get_code_distance_matrix() -> np.ndarray:
    global _CODE_DISTANCE_MATRIX
    if _CODE_DISTANCE_MATRIX is None:
        _CODE_DISTANCE_MATRIX = build_code_distance_matrix(get_mask12_table())
    return _CODE_DISTANCE_MATRIX


def get_code_prior() -> tuple[np.ndarray, dict[int, int]]:
    """Return (P_code, mask12_to_byte) for action-cost computation."""
    global _P_CODE_PRIOR, _MASK12_TO_BYTE
    if _P_CODE_PRIOR is None or _MASK12_TO_BYTE is None:
        mask_table = get_mask12_table()
        cdm = get_code_distance_matrix()
        counts = np.bincount(cdm.flatten(), minlength=13).astype(np.float64)
        P_code = counts / counts.sum()
        mask12_to_byte = {int(mask_table[b]): b for b in range(256)}
        _P_CODE_PRIOR = P_code
        _MASK12_TO_BYTE = mask12_to_byte
    assert _P_CODE_PRIOR is not None
    assert _MASK12_TO_BYTE is not None
    return _P_CODE_PRIOR, _MASK12_TO_BYTE


def _build_gaussian_lut() -> np.ndarray:
    """
    Precompute Gaussian base values for all (chi, p, distance) triples.
    4 chi x 4 phase x 13 distances = 208 floats.

    Sigma values are CGM-grounded from stage actions S_CS, S_UNA, S_ONA.
    Chi maps to K4 vertex which maps to CGM stage:
      chi=0 (Governance / CS):    tight coupling, sigma = S_CS / 2 ≈ 3.94
      chi=1 (Information / UNA):  orthogonal split, sigma = S_UNA ≈ 3.55
      chi=2 (Inference / ONA):    diagonal activation, sigma = S_ONA ≈ 3.94
      chi=3 (Intelligence / BU):  balance / near-uniform, sigma = large
    """
    cgm_base_sigma = {
        0: _S_CS / 2.0,
        1: _S_UNA,
        2: _S_ONA,
        3: _S_BU_SIGMA,
    }
    lut = np.zeros((4, 4, 13), dtype=np.float32)
    for chi in range(4):
        for p in range(4):
            sigma = cgm_base_sigma[chi]
            phase_factor = 1.0 + 0.1 * (p - 1.5) / 1.5
            sigma = max(0.5, sigma * phase_factor)
            for d in range(13):
                lut[chi, p, d] = float(np.exp(-0.5 * (d / sigma) ** 2))
    return lut


_GAUSSIAN_LUT: np.ndarray = _build_gaussian_lut()


def compute_mask(
    device: torch.device,
    dtype: torch.dtype,
    h: int,
    chi: int,
    p: int,
    last_byte_weight: int,
    byte_charge_table: np.ndarray,
    prev_h: int | None = None,
) -> Tensor:
    """Compute projection mask with differential modulation.

    Returns a tensor of shape (256,) — the boundary mask.
    Broadcasting across fibers/features is handled at the application site.
    """
    code_dist_matrix = get_code_distance_matrix()

    distances_i = code_dist_matrix[h, :].astype(np.int32)  # integer distances in {0..12}
    chi_idx = min(max(chi, 0), 3)
    p_idx = min(max(p, 0), 3)

    # Base radial profile row for this (chi, p)
    base_row = _GAUSSIAN_LUT[chi_idx, p_idx, :].astype(np.float32)  # shape (13,)

    # Dynamic sigma focusing: distance scaling + linear interpolation on the LUT row
    if prev_h is not None:
        td = int(code_dist_matrix[prev_h, h])        # 0..12
        D = float(td) / 12.0                         # 0..1
        sigma_scale = 1.0 - SIGMA_FOCUS_K * (D - 0.5)
        sigma_scale = float(max(SIGMA_FOCUS_CLIP[0], min(SIGMA_FOCUS_CLIP[1], sigma_scale)))
    else:
        sigma_scale = 1.0

    d_eff = distances_i.astype(np.float32) / sigma_scale   # float distances
    d0 = np.clip(d_eff.astype(np.int32), 0, 12)
    d1 = np.clip(d0 + 1, 0, 12)
    frac = (d_eff - d0.astype(np.float32)).astype(np.float32)

    mask_base = (1.0 - frac) * base_row[d0] + frac * base_row[d1]

    same_chi = (byte_charge_table == chi).astype(np.float32)
    wedge_factor = 1.0 + 0.2 * same_chi
    mask_wedge = mask_base * wedge_factor

    w = float(last_byte_weight)
    alpha = 0.1 + 0.2 * (w / 12.0)
    alpha = max(0.05, min(alpha, 0.35))

    mask_np = 1.0 + alpha * (mask_wedge - 1.0)

    if prev_h is not None:
        # Differential modulation remains as-is (strength scaling).
        # Note: td was computed above too; recompute for clarity and isolation.
        td = int(code_dist_matrix[prev_h, h])
        diff_scale = 0.5 + 0.5 * (td / 12.0)
        mask_np = 1.0 + diff_scale * (mask_np - 1.0)

    mask_np = mask_np / (mask_np.mean() + 1e-8)

    mask = torch.from_numpy(mask_np).to(device=device, dtype=dtype)
    return mask


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
    """Wraps a SwiGLU MLP layer. Mask is injected via set_mask."""

    def __init__(self, mlp: nn.Module, layer_idx: int, n_fiber: int):
        super().__init__()
        required = ("gate_proj", "up_proj", "down_proj")
        missing = [a for a in required if not hasattr(mlp, a)]
        if missing:
            raise TypeError(
                f"MLP missing {missing}. Expected SwiGLU architecture "
                f"with gate_proj, up_proj, down_proj."
            )
        self.mlp = mlp
        self.layer_idx = layer_idx
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
        n_feat = hidden_dim // N_BOUNDARY

        if self._mask is not None:
            mask_1d = self._mask.view(N_BOUNDARY)
            z_view = z.view(-1, N_BOUNDARY, n_feat)
            z_masked = z_view * mask_1d.view(1, N_BOUNDARY, 1)
            z = z_masked.view(-1, hidden_dim)

        out = self.mlp.down_proj(z)

        if self._collector is not None:
            b, h_peak, peak_mass, energy = extract_byte(out, self.n_fiber)
            self._collector.append((self.layer_idx, b, h_peak, peak_mass, energy))

        return out.view(*shape[:-1], d_model)


class SparseRoutedMLP(nn.Module):
    """
    Kernel-routed sparse MLP. Computes only selected boundary slices
    instead of the full 256-wide intermediate.

    Falls back to full computation if no mask is set.
    """

    def __init__(self, mlp: nn.Module, layer_idx: int, n_fiber: int, width: int = 64):
        super().__init__()
        required = ("gate_proj", "up_proj", "down_proj")
        missing = [a for a in required if not hasattr(mlp, a)]
        if missing:
            raise TypeError(f"MLP missing {missing}.")
        self.mlp = mlp
        self.layer_idx = layer_idx
        self.n_fiber = n_fiber
        self.width = width
        self._collector: list[tuple[int, int, int, float, Tensor]] | None = None
        self._mask: Tensor | None = None
        self._selected_positions: Tensor | None = None
        self._full_mode: bool = True

    def set_collector(self, collector: list[tuple[int, int, int, float, Tensor]] | None) -> None:
        self._collector = collector

    def set_mask(self, mask: Tensor | None) -> None:
        self._mask = mask
        self._selected_positions = None
        self._full_mode = True

    def set_sparse(self, mask: Tensor, positions: Tensor) -> None:
        self._mask = mask
        self._selected_positions = positions
        self._full_mode = False

    def set_full(self) -> None:
        self._selected_positions = None
        self._full_mode = True

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        d_model = shape[-1]
        x_flat = x.reshape(-1, d_model)
        batch = x_flat.shape[0]

        if self._full_mode or self._selected_positions is None:
            gate = self.mlp.gate_proj(x_flat)
            up = self.mlp.up_proj(x_flat)
            z = F.silu(gate) * up

            hidden_dim = z.shape[-1]
            n_feat = hidden_dim // N_BOUNDARY
            if self._mask is not None:
                mask_1d = self._mask.view(N_BOUNDARY)
                z_view = z.view(-1, N_BOUNDARY, n_feat)
                z = (z_view * mask_1d.view(1, N_BOUNDARY, 1)).view(-1, hidden_dim)

            out = self.mlp.down_proj(z)
        else:
            out = self._sparse_forward(x_flat, batch)

        if self._collector is not None:
            b, h_peak, peak_mass, energy = extract_byte(out, self.n_fiber)
            self._collector.append((self.layer_idx, b, h_peak, peak_mass, energy))

        return out.view(*shape[:-1], d_model)

    def _sparse_forward(self, x_flat: Tensor, batch: int) -> Tensor:
        positions = self._selected_positions
        mask = self._mask
        if positions is None or mask is None:
            raise RuntimeError("Sparse mode activated without positions or mask.")

        hidden_dim = self.mlp.gate_proj.weight.shape[0]
        n_feat = hidden_dim // N_BOUNDARY
        w = positions.shape[0]

        gate_w = self.mlp.gate_proj.weight
        up_w = self.mlp.up_proj.weight
        gate_dev = gate_w.device
        positions_dev = positions.to(gate_dev)

        offsets = positions_dev.unsqueeze(1) * n_feat
        offsets = offsets + torch.arange(n_feat, device=gate_dev)
        row_idx = offsets.view(-1)

        gate_sel = F.linear(x_flat, gate_w[row_idx, :])
        up_sel = F.linear(x_flat, up_w[row_idx, :])

        if self.mlp.gate_proj.bias is not None:
            gate_sel = gate_sel + self.mlp.gate_proj.bias[row_idx]
        if self.mlp.up_proj.bias is not None:
            up_sel = up_sel + self.mlp.up_proj.bias[row_idx]

        z_sel = F.silu(gate_sel) * up_sel
        z_view = z_sel.view(batch, w, n_feat)
        if mask is not None:
            mask_positions = positions.to(mask.device)
            mask_sel = mask[mask_positions].to(gate_dev)
            z_view = z_view * mask_sel.view(1, w, 1)
        z_sel = z_view.view(batch, w * n_feat)

        down_w_sel = self.mlp.down_proj.weight[:, row_idx]
        out = F.linear(z_sel, down_w_sel, bias=self.mlp.down_proj.bias)
        return out


class GyroLabe:
    """An AI Calibration Instrument.

    Couples a generative AI model to the GGG ASI Alignment Router Kernel.

    The closed loop:
        Kernel state -> Projection mask -> Model inference -> Token -> Byte -> Kernel state

    The kernel advances ONLY via tokens (the Common Language).
    Extraction from activations is telemetry for measuring alignment.
    """

    def __init__(
        self,
        model: Any,
        atlas_dir: str | Path,
        config: CouplingConfig | None = None,
        token_offset: int = 0,
    ):
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise TypeError(
                "Model architecture not supported. Expected model.model.layers "
                "(LLaMA, OLMo, Mistral family)."
            )

        self.model = model
        self.config = config or CouplingConfig()
        self.kernel = RouterKernel(atlas_dir=Path(atlas_dir))
        self.token_offset = token_offset

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

        self._originals: dict[Tuple[str, int], nn.Module] = {}
        self._wrapped: dict[Tuple[str, int], RoutedMLP | SparseRoutedMLP] = {}
        self._installed = False
        self._collector: list[tuple[int, int, int, float, Tensor]] | None = None
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32
        self._sparse_width: int | None = None

        self._step_h: int = 0
        self._step_chi: int = 0
        self._step_p: int = 0
        self._step_mu: int = 0
        self._step_mu_qturn: int = 0
        self._mask_boundary: np.ndarray | None = None
        self._prev_h: int | None = None

        self.byte_log: list[int] = []
        self.trajectory: list[dict[str, Any]] = []

    def install(self, sparse_width: int | None = None) -> None:
        """
        Install routing wrappers on the configured MLP layers.

        sparse_width: if set, use SparseRoutedMLP with this many positions.
                      If None, use standard RoutedMLP.
        """
        if self._installed:
            return

        self._device = next(self.model.parameters()).device
        self._dtype = next(self.model.parameters()).dtype
        self._sparse_width = sparse_width

        def wrap_mlp(orig: nn.Module, layer_idx: int) -> RoutedMLP | SparseRoutedMLP:
            if sparse_width is not None:
                return SparseRoutedMLP(orig, layer_idx, self.n_fiber, width=sparse_width)
            return RoutedMLP(orig, layer_idx, self.n_fiber)

        layers = self.model.model.layers
        for i in self.routed_layers:
            orig = layers[i].mlp
            key = ("layers", i)
            self._originals[key] = orig
            self._wrapped[key] = wrap_mlp(orig, i)
            layers[i].mlp = self._wrapped[key]

        if self.config.couple_local_decoder:
            local_decoder = getattr(self.model.model, "local_decoder", None)
            if local_decoder is not None and hasattr(local_decoder, "layers"):
                ld_layers = local_decoder.layers
                for i in range(len(ld_layers)):
                    orig = ld_layers[i].mlp
                    key = ("local_decoder", i)
                    self._originals[key] = orig
                    self._wrapped[key] = wrap_mlp(orig, 1000 + i)
                    ld_layers[i].mlp = self._wrapped[key]

        if self.config.couple_local_encoder:
            local_encoder = getattr(self.model.model, "local_encoder", None)
            if local_encoder is not None and hasattr(local_encoder, "layers"):
                le_layers = local_encoder.layers
                for i in range(len(le_layers)):
                    orig = le_layers[i].mlp
                    key = ("local_encoder", i)
                    self._originals[key] = orig
                    self._wrapped[key] = wrap_mlp(orig, 2000 + i)
                    le_layers[i].mlp = self._wrapped[key]

        self._installed = True

    def restore(self) -> None:
        """Remove routing wrappers and restore original MLP layers."""
        if not self._installed:
            return

        layers = self.model.model.layers
        for (path, i), orig in self._originals.items():
            if path == "layers":
                layers[i].mlp = orig
            elif path == "local_decoder":
                self.model.model.local_decoder.layers[i].mlp = orig
            elif path == "local_encoder":
                self.model.model.local_encoder.layers[i].mlp = orig

        self._originals.clear()
        self._wrapped.clear()
        self._installed = False

    def override_mask(self, mask: Tensor | None, mask_boundary: np.ndarray | None = None) -> None:
        """Override the mask on all routed layers."""
        for w in self._wrapped.values():
            w.set_mask(mask)
        if mask_boundary is not None:
            self._mask_boundary = mask_boundary

    def _select_positions(self, h: int, width: int) -> Tensor:
        """
        Select which boundary positions to compute, centered on horizon h.

        Uses code distance: pick the `width` positions closest to h.
        """
        code_dist_matrix = get_code_distance_matrix()
        distances = code_dist_matrix[h, :]  # [256]
        indices = np.argsort(distances)[:width]
        return torch.tensor(indices, dtype=torch.long, device=self._device)

    def begin_step(self, sparse_width: int | None = None) -> None:
        """
        Begin a coordination step.

        If sparse_width is set, only compute that many boundary positions
        in routed MLP layers (centered on current horizon in code distance).
        If None, compute all 256 (original behavior).
        """
        self._step_h = int(self.kernel.current_horizon.item())
        self._step_chi = int(self.kernel.current_vertex.item())
        self._step_p = int(self.kernel.current_phase.item())
        self._step_mu = (self._step_h + self._step_p * QUARTER_TURN) % N_BOUNDARY
        self._step_mu_qturn = (self._step_mu + QUARTER_TURN) % N_BOUNDARY

        last_byte_weight = int(self.kernel.byte_weight[self.kernel.last_byte.item()])

        mask = compute_mask(
            self._device, self._dtype,
            self._step_h, self._step_chi, self._step_p,
            last_byte_weight,
            self.kernel.byte_charge,
            prev_h=self._prev_h,
        )

        self._mask_boundary = mask.float().cpu().numpy()

        sparse_step_width = sparse_width if sparse_width is not None else self._sparse_width

        self._collector = []
        for w in self._wrapped.values():
            w.set_collector(self._collector)

            if sparse_step_width is not None and isinstance(w, SparseRoutedMLP):
                positions = self._select_positions(self._step_h, sparse_step_width)
                w.set_sparse(mask, positions)
            elif isinstance(w, SparseRoutedMLP):
                w.set_full()
                w.set_mask(mask)
            else:
                w.set_mask(mask)

        self._prev_h = self._step_h

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

    def token_to_byte(self, token_id: int) -> int:
        raw = int(token_id)
        if raw >= self.token_offset + 256:
            return raw - self.token_offset - 256
        if raw >= self.token_offset:
            return raw - self.token_offset
        return raw & 0xFF

    def adjust_logits_bu_ingress(self, token_ids: Tensor, logits: Tensor) -> Tensor:
        h_cur = int(self.kernel.current_horizon[0])
        mask_table = get_mask12_table()
        mh = int(mask_table[h_cur])
        P_code, mask12_to_byte = get_code_prior()
        cdm = get_code_distance_matrix()

        out = logits.clone()
        for j in range(int(token_ids.numel())):
            tok = int(token_ids[j].item())
            b = self.token_to_byte(tok)
            m_b = int(mask_table[b])
            m2 = mh ^ m_b
            h2 = mask12_to_byte.get(m2)
            if h2 is None:
                continue
            d = int(cdm[h_cur, h2])
            G = -_math.log(P_code[d] + 1e-12)
            out[j] = out[j] - ACTIVE_BETA * G
        return out

    def advance_with_token(self, token_id: int) -> int:
        """Advance the kernel using token ID as the byte source."""
        byte = self.token_to_byte(token_id)
        self.kernel.step_byte(byte)
        self.byte_log.append(byte)

        if self.trajectory:
            self.trajectory[-1]["driving_byte"] = byte

        return byte

    def prime_from_tokens(self, token_ids: list[int]) -> None:
        """Prime the kernel from prompt tokens."""
        for tid in token_ids:
            byte = self.token_to_byte(tid)
            self.kernel.step_byte(byte)
            self.byte_log.append(byte)

    def reset(self) -> None:
        """Reset kernel to archetype and clear trajectory logs."""
        self.kernel.reset()
        self.byte_log.clear()
        self.trajectory.clear()
        self._prev_h = None

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
        return _math.exp(-self.mean_logprob)

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
    sparse_width: int | None = None,
) -> GenerationResult:
    """Generate text with optional GyroLabe coordination and sparse MLP."""
    is_bolmo = hasattr(model, "model") and hasattr(model.model, "local_encoder")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    prompt_ids = input_ids[0].tolist()

    if is_bolmo and hasattr(model, "generate"):
        if labe is not None:
            labe.prime_from_tokens(prompt_ids)
        t0 = time.perf_counter()
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            labe=labe,
        )
        elapsed = time.perf_counter() - t0
        output_ids = output[0].tolist()
        gen_ids = output_ids[len(prompt_ids):]
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            elapsed=elapsed,
            n_tokens=len(gen_ids),
            logprobs=[],
        )

    use_cache = True
    generated: list[int] = []
    logprobs: list[float] = []
    past = None

    if labe is not None:
        labe.prime_from_tokens(prompt_ids)

    t0 = time.perf_counter()
    for step in range(max_new_tokens):
        if labe is not None:
            labe.begin_step(sparse_width=sparse_width)

        out = model(
            input_ids,
            use_cache=use_cache,
            past_key_values=past if use_cache else None,
        )
        past = getattr(out, "past_key_values", None) if use_cache else None

        if use_cache and past is None and step == 0:
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
                topv_adj = labe.adjust_logits_bu_ingress(topi, topv)
                probs = torch.softmax(topv_adj, dim=-1)
            else:
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

        if use_cache and past is not None:
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
