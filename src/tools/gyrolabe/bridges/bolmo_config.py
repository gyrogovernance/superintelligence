from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BOLMO_MODEL_PATH: Path = _REPO_ROOT / "data" / "models" / "Bolmo-1B"

_BOLMO_ROPE_PATCH_APPLIED = False


@dataclass(frozen=True)
class BolmoTokenLayout:
    offset: int = 4
    special_count: int = 4
    vocab_size: int = 520

    @property
    def boundary_offset(self) -> int:
        return self.offset + 256

    @property
    def normal_byte_low(self) -> int:
        return self.offset

    @property
    def normal_byte_high(self) -> int:
        return self.offset + 255

    @property
    def fused_special_low(self) -> int:
        return self.boundary_offset

    @property
    def fused_special_high(self) -> int:
        return self.boundary_offset + self.special_count - 1

    @property
    def fused_byte_low(self) -> int:
        return self.boundary_offset + self.offset

    @property
    def fused_byte_high(self) -> int:
        return self.boundary_offset + self.offset + 255


@dataclass(frozen=True)
class BolmoEncodeBridgeConfig:
    token_layout: BolmoTokenLayout = field(default_factory=BolmoTokenLayout)
    strict_cpu: bool = True
    embedding_scale: float = 1.0
    boundary_scale: float = 1.0

    boundary_mode: str = "hybrid"
    boundary_cosine_weight: float = 0.35
    boundary_structural_weight: float = 0.65
    target_bytes_per_patch: float | None = 5.5
    target_patch_gain: float = 0.35


def canonicalize_bolmo_ids(
    input_ids: torch.Tensor,
    *,
    layout: BolmoTokenLayout | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert Bolmo token IDs into canonical raw bytes [0..255].

    Returns:
      canonical_bytes: uint8, same shape as input_ids
      valid_mask: bool, True where token maps to a real byte
      boundary_mask: bool, True where token is fused-boundary phase
    """
    cfg = layout or BolmoTokenLayout()
    ids = input_ids.to(torch.int64)

    valid_normal = (ids >= cfg.normal_byte_low) & (ids <= cfg.normal_byte_high)
    valid_fused = (ids >= cfg.fused_byte_low) & (ids <= cfg.fused_byte_high)
    valid_mask = valid_normal | valid_fused

    boundary_mask = ids >= cfg.boundary_offset

    canonical = torch.zeros_like(ids, dtype=torch.int64)
    canonical = torch.where(valid_normal, ids - cfg.offset, canonical)
    canonical = torch.where(valid_fused, ids - cfg.fused_byte_low, canonical)

    return canonical.to(torch.uint8), valid_mask, boundary_mask


def _apply_bolmo_rope_patch() -> None:
    global _BOLMO_ROPE_PATCH_APPLIED
    if _BOLMO_ROPE_PATCH_APPLIED:
        return

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    def _compute_default_rope_parameters(
        config,
        device=None,
        seq_len=None,
        layer_type=None,
    ):
        if hasattr(config, "standardize_rope_params"):
            config.standardize_rope_params()

        rp = getattr(config, "rope_parameters", None)
        if rp is not None:
            rp = rp.get(layer_type, rp) if layer_type and isinstance(rp, dict) else rp
            if isinstance(rp, dict):
                base = rp.get("rope_theta", getattr(config, "rope_theta", 10000.0))
                partial = rp.get("partial_rotary_factor", 1.0)
            else:
                base = getattr(config, "rope_theta", 10000.0)
                partial = 1.0
        else:
            base = getattr(config, "rope_theta", 10000.0)
            partial = 1.0

        base = float(base.item()) if hasattr(base, "item") else float(base)
        partial = float(partial.item()) if hasattr(partial, "item") else float(partial)

        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        dim = int(head_dim * partial)

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, dim, 2, dtype=torch.int64, device="cpu").float() / dim
            )
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    from transformers.modeling_utils import PreTrainedModel

    if getattr(PreTrainedModel._init_weights, "__bolmo_rope_patch__", False):
        _BOLMO_ROPE_PATCH_APPLIED = True
        return

    _orig_init_weights = PreTrainedModel._init_weights

    def _patched_init_weights(self, module):
        if (
            "RotaryEmbedding" in module.__class__.__name__
            and getattr(module, "original_inv_freq", None) is not None
            and getattr(module, "rope_type", None) == "default"
            and "default" in ROPE_INIT_FUNCTIONS
        ):
            buf, _ = ROPE_INIT_FUNCTIONS["default"](module.config, None, None, None)
            with torch.no_grad():
                b = buf.to(module.inv_freq.device, dtype=module.inv_freq.dtype)
                module.inv_freq.copy_(b)
                module.original_inv_freq.copy_(b)
            return
        _orig_init_weights(self, module)

    _patched_init_weights.__bolmo_rope_patch__ = True
    PreTrainedModel._init_weights = _patched_init_weights

    from transformers.generation.utils import GenerationMixin

    if not getattr(
        GenerationMixin._prepare_generation_config,
        "__bolmo_patch__",
        False,
    ):
        _orig_gp = GenerationMixin._prepare_generation_config

        def _patched_gp(self, generation_config=None, *args, **kwargs):
            try:
                return _orig_gp(self, generation_config, *args, **kwargs)
            except TypeError as e:
                if "takes 2 positional arguments" in str(e) and len(args) > 0:
                    return _orig_gp(self, generation_config, **kwargs)
                raise

        _patched_gp.__bolmo_patch__ = True
        GenerationMixin._prepare_generation_config = _patched_gp

    _BOLMO_ROPE_PATCH_APPLIED = True


def load_base_bolmo(
    model_path: str | Path | None = None,
    **hf_kwargs: Any,
) -> Any:
    """
    Load raw Bolmo without the GyroLabe encode wrapper.
    """
    _apply_bolmo_rope_patch()

    path = model_path if model_path is not None else DEFAULT_BOLMO_MODEL_PATH

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if "local_files_only" not in hf_kwargs:
        hf_kwargs = {**hf_kwargs, "local_files_only": True}

    warnings.filterwarnings(
        "ignore",
        message=".*rope_config_validation.*",
        category=FutureWarning,
    )

    config = hf_kwargs.get("config")
    if config is None:
        config = AutoConfig.from_pretrained(
            str(path),
            trust_remote_code=True,
            local_files_only=hf_kwargs.get("local_files_only", True),
        )
        if getattr(config, "rope_scaling", None) and isinstance(config.rope_scaling, dict):
            rs = dict(config.rope_scaling)
            if "beta_fast" in rs and not isinstance(rs["beta_fast"], float):
                rs["beta_fast"] = float(rs["beta_fast"])
            if "beta_slow" in rs and not isinstance(rs["beta_slow"], float):
                rs["beta_slow"] = float(rs["beta_slow"])
            config.rope_scaling = rs
        hf_kwargs = {**hf_kwargs, "config": config}

    return AutoModelForCausalLM.from_pretrained(
        str(path),
        trust_remote_code=True,
        **hf_kwargs,
    )