from __future__ import annotations

import importlib
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from src.tools.gyrolabe import ops as gyrolabe_ops
from src.tools.gyrolabe.bridges.encode import (
    EncodedFields,
    hidden_geometry_distance6,
    extract_encoded_fields,
    patch_lengths_from_boundary_mask,
)


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
    chi_boundary_threshold: int = 2


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


def extract_bolmo_encode_fields(
    input_ids: torch.Tensor,
    *,
    layout: BolmoTokenLayout | None = None,
) -> EncodedFields:
    cfg = layout or BolmoTokenLayout()
    canonical_bytes, valid_mask, boundary_mask = canonicalize_bolmo_ids(
        input_ids,
        layout=cfg,
    )
    return extract_encoded_fields(
        canonical_bytes=canonical_bytes,
        valid_mask=valid_mask,
        boundary_mask=boundary_mask,
    )


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


class _GyroLabeBoundaryPredictor(nn.Module):
    owner: "GyroLabeBolmoEncodeBridge"

    def __init__(self, inner: nn.Module, owner: "GyroLabeBolmoEncodeBridge") -> None:
        super().__init__()
        self.inner = inner
        object.__setattr__(self, "owner", owner)

        model_mod = inner.__class__.__module__
        if not model_mod.endswith("modeling_bolmo"):
            raise RuntimeError(
                f"Unsupported Bolmo module path for boundary predictor: {model_mod!r}"
            )

        utils_mod_name = model_mod.rsplit(".", 1)[0] + ".utils_bolmo"
        utils_mod = importlib.import_module(utils_mod_name)
        self.compute_boundary_mask = getattr(utils_mod, "compute_boundary_mask")

    def forward(
        self,
        hidden_states: torch.Tensor,
        sequence_start_indices: torch.Tensor | None = None,
        epsilon: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fields = cast(EncodedFields | None, self.owner._ctx.get("fields"))

        if fields is None:
            return self.inner(
                hidden_states,
                sequence_start_indices=sequence_start_indices,
                epsilon=epsilon,
            )

        return self._exact_boundary_prediction(
            fields=fields,
            hidden_states=hidden_states,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            sequence_start_indices=sequence_start_indices,
        )

    def _exact_boundary_prediction(
        self,
        *,
        fields: EncodedFields,
        hidden_states: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        sequence_start_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Exact batch-safe boundary prediction.

        We preserve Bolmo's boundary question:
        where does the local hidden geometry change?

        But we answer it in the QuBEC chart using:
          - byte-state chirality distance
          - hidden-state sign-geometry distance
          - empirical support M2 modulation

        High empirical support (diffuse)  -> lower threshold
        Low empirical support (condensed) -> higher threshold
        """
        states = fields.states.to(torch.int32)
        valid = fields.valid_mask.to(torch.bool)

        if states.ndim == 1:
            states = states.unsqueeze(0)
            valid = valid.unsqueeze(0)

        batch, seq_len = states.shape
        chi_dist = torch.zeros((batch, seq_len), dtype=torch.int32, device="cpu")

        for b in range(batch):
            row_states = states[b].contiguous()
            row_dist = gyrolabe_ops.chirality_distance_adjacent(
                row_states, lookahead=1
            ).to(torch.int32)
            chi_dist[b] = row_dist

        hidden_dist = hidden_geometry_distance6(hidden_states)

        # preserve the same computational question:
        # boundary if either byte-transport or hidden geometry says so
        structural_dist = torch.maximum(chi_dist, hidden_dist)

        base_threshold = int(getattr(self.owner.config, "chi_boundary_threshold", 2))
        m2 = getattr(self.owner, "_source_m2", None)

        if m2 is not None and m2 > 0:
            m2_norm = max(0.0, min(1.0, float(m2) / 4096.0))
            # Stronger CE11 actuation so M2 feedback is measurably visible.
            # low M2 (condensed) -> higher threshold, high M2 -> base threshold
            boost = int(round((1.0 - m2_norm) * 2.0))
            effective_threshold = max(1, base_threshold + boost)
        else:
            effective_threshold = base_threshold

        exact_boundary = (structural_dist >= effective_threshold) & valid

        if sequence_start_indices is None:
            if seq_len > 0:
                exact_boundary[:, 0] = True
        else:
            for b in range(batch):
                idx = int(sequence_start_indices[b].item())
                if 0 <= idx < seq_len:
                    exact_boundary[b, idx] = True

        pos_val = torch.tensor(0.0, device=device, dtype=dtype)
        neg_val = torch.tensor(-100_000.0, device=device, dtype=dtype)

        boundary_logprobs = torch.where(
            exact_boundary.to(device=device),
            pos_val,
            neg_val,
        )

        self.owner._last_boundary_mask = exact_boundary.detach()
        self.owner._last_fields = fields

        return boundary_logprobs, exact_boundary


class GyroLabeBolmoEncodeBridge(nn.Module):
    """
    Encode-only Bolmo bridge. Exact mode only.

    Responsibilities:
      - canonicalize Bolmo token IDs into byte/gauge charts
      - extract q-class / family / micro-ref / signature / state fields
      - replace boundary prediction with exact chirality-distance thresholding
    """

    def __init__(
        self,
        base_model: nn.Module,
        *,
        config: BolmoEncodeBridgeConfig | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config or BolmoEncodeBridgeConfig()
        self._ctx: dict[str, Any] = {}
        self._handles: list[Any] = []
        self._original_boundary_predictor: nn.Module | None = None
        self._last_boundary_mask: torch.Tensor | None = None
        self._last_fields: EncodedFields | None = None
        self._source_m2: float | None = None

        self._validate_bolmo_structure()
        self.install()

    def _validate_bolmo_structure(self) -> None:
        if not hasattr(self.base_model, "model"):
            raise TypeError("Wrapped model does not look like Bolmo: missing .model")
        if not hasattr(self.base_model.model, "local_encoder"):
            raise TypeError("Wrapped model does not look like Bolmo: missing .model.local_encoder")
        if not hasattr(self.base_model.model.local_encoder, "byte_embedding"):
            raise TypeError("Wrapped model does not look like Bolmo: missing .model.local_encoder.byte_embedding")
        if not hasattr(self.base_model.model.local_encoder, "boundary_predictor_module"):
            raise TypeError("Wrapped model does not look like Bolmo: missing .model.local_encoder.boundary_predictor_module")
        if not hasattr(self.base_model, "config") or not hasattr(self.base_model.config, "hidden_size"):
            raise TypeError("Wrapped model does not expose .config.hidden_size")

    def runtime_capabilities(self) -> dict[str, Any]:
        return {
            "gyrolabe_native_available": gyrolabe_ops.native_available(),
            "strict_cpu": bool(self.config.strict_cpu),
            "token_layout": {
                "offset": self.config.token_layout.offset,
                "special_count": self.config.token_layout.special_count,
                "boundary_offset": self.config.token_layout.boundary_offset,
            },
        }

    def extract_fields(self, input_ids: torch.Tensor) -> EncodedFields:
        if self.config.strict_cpu and input_ids.device.type != "cpu":
            raise ValueError(
                f"GyroLabe encode bridge requires CPU input_ids, got device={input_ids.device}"
            )
        return extract_bolmo_encode_fields(
            input_ids.cpu() if input_ids.device.type != "cpu" else input_ids,
            layout=self.config.token_layout,
        )

    def install(self) -> None:
        if self._handles:
            return

        local_encoder = self.base_model.model.local_encoder

        def _local_encoder_pre_hook(
            module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if input_ids is None:
                self._ctx = {}
                return

            fields = self.extract_fields(input_ids)
            self._ctx = {"fields": fields}
            self._last_fields = fields

        self._handles.append(
            local_encoder.register_forward_pre_hook(_local_encoder_pre_hook, with_kwargs=True)
        )

        self._original_boundary_predictor = local_encoder.boundary_predictor_module
        inner_bp = cast(nn.Module, local_encoder.boundary_predictor_module)
        local_encoder.boundary_predictor_module = _GyroLabeBoundaryPredictor(inner_bp, self)

    def uninstall(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

        if self._original_boundary_predictor is not None:
            self.base_model.model.local_encoder.boundary_predictor_module = self._original_boundary_predictor
            self._original_boundary_predictor = None

        self._ctx = {}
        self._last_boundary_mask = None
        self._last_fields = None
        self._source_m2 = None

    def set_source_m2(self, m2: float) -> None:
        """Set M2 from decode cell for boundary threshold modulation (CE11)."""
        self._source_m2 = float(m2)

    def last_encoded_fields(self) -> EncodedFields | None:
        return self._last_fields

    def get_last_patch_geometry(self) -> dict[str, Any] | None:
        """Patch geometry from last exact boundary run (first row if batched)."""
        fields = self._last_fields
        mask = self._last_boundary_mask
        if fields is None or mask is None:
            return None

        valid_mask = fields.valid_mask
        if valid_mask.ndim == 2:
            valid_mask = valid_mask[0]
        if mask.ndim == 2:
            mask = mask[0]

        patch_lengths = patch_lengths_from_boundary_mask(mask, valid_mask)
        patch_count = len(patch_lengths)
        boundary_count = int((valid_mask.to(torch.bool) & mask.to(torch.bool)).sum().item())
        mean_bytes_per_patch = sum(patch_lengths) / len(patch_lengths) if patch_lengths else 0.0

        return {
            "patch_lengths": patch_lengths,
            "patch_count": patch_count,
            "boundary_count": boundary_count,
            "mean_bytes_per_patch": mean_bytes_per_patch,
            "attn_proxy": patch_count * patch_count,
            "kv_proxy": patch_count,
        }

    @torch.no_grad()
    def prefill_probe(
        self, input_ids: torch.Tensor
    ) -> tuple[EncodedFields, dict[str, Any] | None]:
        """Run forward far enough to populate bridge state; return fields and patch geometry."""
        if self.config.strict_cpu and input_ids.device.type != "cpu":
            input_ids = input_ids.cpu()
        _ = self.forward(input_ids)
        fields = self._last_fields or self.extract_fields(input_ids)
        geom = self.get_last_patch_geometry()
        return (fields, geom)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        base = cast(nn.Module, self.base_model)
        return base(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return cast(Any, self.base_model).generate(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path | None = None,
        *,
        config: BolmoEncodeBridgeConfig | None = None,
        **hf_kwargs: Any,
    ) -> "GyroLabeBolmoEncodeBridge":
        path = model_path if model_path is not None else DEFAULT_BOLMO_MODEL_PATH
        base_model = load_base_bolmo(
            model_path=path,
            **hf_kwargs,
        )
        return cls(
            base_model=cast(nn.Module, base_model),
            config=config,
        )

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def load_gyrolabe_bolmo_encode(
    model_path: str | Path | None = None,
    *,
    config: BolmoEncodeBridgeConfig | None = None,
    **hf_kwargs: Any,
) -> GyroLabeBolmoEncodeBridge:
    path = model_path if model_path is not None else DEFAULT_BOLMO_MODEL_PATH
    return GyroLabeBolmoEncodeBridge.from_pretrained(
        model_path=path,
        config=config,
        **hf_kwargs,
    )
