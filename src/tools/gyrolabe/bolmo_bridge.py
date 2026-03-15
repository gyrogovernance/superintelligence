# src/tools/gyrolabe/bolmo_bridge.py
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

# Default: use local data/models/Bolmo-1B; avoid HF cache/download.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BOLMO_MODEL_PATH: Path = _REPO_ROOT / "data" / "models" / "Bolmo-1B"

from src.tools.gyrolabe.ops import (
    chirality_distance_adjacent,
    extract_scan,
    qmap_extract,
    signature_scan,
    signatures_to_states,
)

_BOLMO_ROPE_PATCH_APPLIED = False


def _apply_bolmo_rope_patch() -> None:
    """Register ROPE_INIT_FUNCTIONS['default'] so Bolmo modeling_bolmo.py (rope_type='default') loads."""
    global _BOLMO_ROPE_PATCH_APPLIED
    if _BOLMO_ROPE_PATCH_APPLIED:
        return
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    def _compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
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
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64, device="cpu").float() / dim)
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

    if not getattr(GenerationMixin._prepare_generation_config, "__bolmo_patch__", False):
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


@dataclass
class GyroLabeSettings:
    enable_embedding_bias: bool = True
    enable_boundary_bias: bool = True
    enable_qclass_sparsity: bool = False  # reserved; intentionally not active here
    enable_decode_expand_cache: bool = False
    enable_decode_boundary_hook: bool = False
    strict_cpu: bool = True


class _DecodeExpandCache:
    """Cache for decode expansion. Output-neutral: identical results to non-cached path."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._cached_expanded: list[int] | None = None
        self._byte_ids: list[int] = []
        self._batch_size: int = 1

    def _expand(self, byte_ids: list[int], n_last: int | None = None) -> list[int]:
        expand = getattr(self._tokenizer, "expand_byte_ids", None)
        if expand is None:
            return byte_ids
        if n_last is not None:
            return expand(byte_ids, n_last=n_last)
        return expand(byte_ids)

    def get_hook(self) -> Any:
        def hook(mode: str, *args: Any, **kwargs: Any) -> Any:
            if mode == "init":
                generated = kwargs.get("generated")
                if generated is None:
                    generated = kwargs.get("byte_input_ids")
                batch_size = kwargs.get("batch_size", 1)
                self._batch_size = batch_size
                if generated is not None and hasattr(generated, "tolist"):
                    ids = generated[0].tolist() if batch_size >= 1 else []
                else:
                    ids = list(args[0]) if args else []
                self._byte_ids = ids
                self._cached_expanded = self._expand(self._byte_ids) if self._byte_ids else []
                return None
            if mode == "get_expanded":
                is_first = kwargs.get("is_first_forward", True)
                device = kwargs.get("device", "cpu")
                dtype = kwargs.get("dtype", torch.long)
                input_ids = kwargs.get("input_ids_for_model")
                batch_size = kwargs.get("batch_size", 1)
                if self._byte_ids:
                    n_last = input_ids.shape[1] if input_ids is not None else 1
                    expanded = self._expand(self._byte_ids, n_last=n_last)
                    return torch.tensor([expanded], device=device, dtype=dtype)
                return None
            if mode == "append":
                next_tokens = kwargs.get("next_tokens")
                batch_size = kwargs.get("batch_size", 1)
                if next_tokens is not None and batch_size >= 1:
                    tok = next_tokens[0].item()
                    self._byte_ids.append(int(tok))
                return None
            return None
        return hook


class _NoopLabe:
    """Minimal labe: records timing and chosen token, does not change logits."""

    def __init__(self) -> None:
        self.last_token: int | None = None
        self.step_times: list[float] = []
        self._step_start: float | None = None

    def begin_step(self) -> None:
        import time
        self._step_start = time.perf_counter()

    def end_step(self) -> None:
        import time
        if self._step_start is not None:
            self.step_times.append(time.perf_counter() - self._step_start)
            self._step_start = None

    def adjust_logits_bu_ingress(self, vocab_ids: Any, topv: Any) -> Any:
        return topv

    def advance_with_token(self, token: int) -> None:
        self.last_token = token


def canonicalize_bolmo_ids(
    input_ids: torch.Tensor,
    *,
    offset: int = 4,
    special_count: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert Bolmo byte-domain token IDs into canonical raw bytes [0..255].

    Returns:
      canonical_bytes: uint8, same shape as input_ids
      valid_mask: bool, True where token maps to a real byte
      boundary_mask: bool, True where token is a boundary-fused token
    """
    ids = input_ids.to(torch.int64)

    normal_byte_low = offset
    normal_byte_high = offset + 255

    fused_special_low = offset + 256
    fused_special_high = fused_special_low + special_count - 1

    fused_byte_low = fused_special_high + 1
    fused_byte_high = fused_byte_low + 255

    valid_normal = (ids >= normal_byte_low) & (ids <= normal_byte_high)
    valid_fused = (ids >= fused_byte_low) & (ids <= fused_byte_high)
    valid_mask = valid_normal | valid_fused

    boundary_mask = ids >= fused_special_low

    canonical = torch.zeros_like(ids, dtype=torch.int64)
    canonical = torch.where(valid_normal, ids - normal_byte_low, canonical)
    canonical = torch.where(valid_fused, ids - fused_byte_low, canonical)

    return canonical.to(torch.uint8), valid_mask, boundary_mask


class _GyroLabeBoundaryPredictor(nn.Module):
    owner: "GyroLabeBolmoBridge"

    def __init__(self, inner: nn.Module, owner: "GyroLabeBolmoBridge") -> None:
        super().__init__()
        self.inner = inner
        object.__setattr__(self, "owner", owner)  # do not register bridge as submodule (avoids train() recursion)

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
        boundary_logprobs, boundary_mask = self.inner(
            hidden_states,
            sequence_start_indices=sequence_start_indices,
            epsilon=epsilon,
        )

        if not self.owner.settings.enable_boundary_bias:
            return boundary_logprobs, boundary_mask

        bias = self.owner.make_boundary_bias(
            width=boundary_logprobs.shape[1],
            lookahead=int(getattr(self.inner, "boundary_predictor_lookahead", 1)),
            device=boundary_logprobs.device,
            dtype=boundary_logprobs.dtype,
        )
        if bias is None:
            return boundary_logprobs, boundary_mask

        boundary_logprobs = boundary_logprobs + bias
        boundary_mask = self.compute_boundary_mask(
            boundary_logprobs,
            getattr(self.inner, "boundary_threshold"),
        )
        return boundary_logprobs, boundary_mask


class GyroLabeBolmoBridge(nn.Module):
    """
    Lossless Bolmo wrapper.

    It does not patch vendor files.
    It composes with a loaded Bolmo HF model and injects exact algebraic side-information.
    """

    def __init__(
        self,
        base_model: nn.Module,
        settings: GyroLabeSettings | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.settings = settings or GyroLabeSettings()
        self._ctx: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._original_boundary_predictor: nn.Module | None = None

        self._validate_bolmo_structure()

        hidden_size = int(getattr(self.base_model.config, "hidden_size"))
        self.q_class_embedding = nn.Embedding(64, hidden_size)
        self.family_embedding = nn.Embedding(4, hidden_size)
        self.micro_ref_embedding = nn.Embedding(64, hidden_size)
        self.boundary_distance_bias = nn.Embedding(7, 1)

        self.reset_gyrolabe_parameters()
        self.install()

    def reset_gyrolabe_parameters(self) -> None:
        with torch.no_grad():
            self.q_class_embedding.weight.zero_()
            self.family_embedding.weight.zero_()
            self.micro_ref_embedding.weight.zero_()
            self.boundary_distance_bias.weight.zero_()

    def _validate_bolmo_structure(self) -> None:
        if not hasattr(self.base_model, "model"):
            raise TypeError("The wrapped model does not look like Bolmo: missing .model")
        if not hasattr(self.base_model.model, "local_encoder"):
            raise TypeError("The wrapped model does not look like Bolmo: missing .model.local_encoder")
        if not hasattr(self.base_model.model.local_encoder, "byte_embedding"):
            raise TypeError("The wrapped model does not look like Bolmo: missing .model.local_encoder.byte_embedding")
        if not hasattr(self.base_model.model.local_encoder, "boundary_predictor_module"):
            raise TypeError("The wrapped model does not look like Bolmo: missing .model.local_encoder.boundary_predictor_module")
        if not hasattr(self.base_model, "config") or not hasattr(self.base_model.config, "hidden_size"):
            raise TypeError("The wrapped model does not expose a usable .config.hidden_size")

    def install(self) -> None:
        if self._handles:
            return

        local_encoder = self.base_model.model.local_encoder

        def _local_encoder_pre_hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if input_ids is None:
                self._ctx = {}
                return

            if self.settings.strict_cpu and input_ids.device.type != "cpu":
                raise ValueError(
                    f"GyroLabe currently requires CPU input_ids, got device={input_ids.device}"
                )

            canonical_bytes, valid_mask, boundary_mask = canonicalize_bolmo_ids(
                input_ids,
                offset=4,
                special_count=4,
            )
            self._ctx = {
                "canonical_bytes": canonical_bytes,
                "valid_mask": valid_mask,
                "boundary_mask": boundary_mask,
            }

            if self.settings.enable_embedding_bias:
                try:
                    q_class, family, micro_ref, sigs, states = extract_scan(canonical_bytes)
                    self._ctx["q_class"] = q_class
                    self._ctx["family"] = family
                    self._ctx["micro_ref"] = micro_ref
                    self._ctx["signatures"] = sigs
                    self._ctx["states"] = states
                except Exception:
                    q_class, family, micro_ref = qmap_extract(canonical_bytes)
                    self._ctx["q_class"] = q_class
                    self._ctx["family"] = family
                    self._ctx["micro_ref"] = micro_ref

        def _byte_embedding_hook(module: nn.Module, args: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            if not self.settings.enable_embedding_bias:
                return output

            tokens = args[0] if args else None
            if tokens is None:
                return output

            if self._ctx.get("q_class") is not None and self._ctx.get("canonical_bytes") is not None:
                canonical_bytes = self._ctx["canonical_bytes"]
                valid_mask = self._ctx["valid_mask"]
                q_class = self._ctx["q_class"]
                family = self._ctx["family"]
                micro_ref = self._ctx["micro_ref"]
                if canonical_bytes.shape != tokens.shape:
                    canonical_bytes, valid_mask = canonicalize_bolmo_ids(tokens, offset=4, special_count=4)[:2]
                    q_class, family, micro_ref = qmap_extract(canonical_bytes)
            else:
                canonical_bytes, valid_mask = canonicalize_bolmo_ids(tokens, offset=4, special_count=4)[:2]
                q_class, family, micro_ref = qmap_extract(canonical_bytes)

            q_class = q_class.to(device=output.device, dtype=torch.long)
            family = family.to(device=output.device, dtype=torch.long)
            micro_ref = micro_ref.to(device=output.device, dtype=torch.long)
            valid = valid_mask.to(device=output.device, dtype=output.dtype).unsqueeze(-1)

            bias = (
                self.q_class_embedding(q_class)
                + self.family_embedding(family)
                + self.micro_ref_embedding(micro_ref)
            ).to(dtype=output.dtype)

            return output + (bias * valid)

        self._handles.append(
            local_encoder.register_forward_pre_hook(_local_encoder_pre_hook, with_kwargs=True)
        )
        self._handles.append(
            local_encoder.byte_embedding.register_forward_hook(_byte_embedding_hook)
        )

        self._original_boundary_predictor = local_encoder.boundary_predictor_module
        inner_bp = cast(nn.Module, local_encoder.boundary_predictor_module)
        local_encoder.boundary_predictor_module = _GyroLabeBoundaryPredictor(
            inner_bp,
            self,
        )

        if self.settings.enable_decode_expand_cache:
            self._install_decode_expand_cache()

    def _install_decode_expand_cache(self) -> None:
        tok = getattr(self.base_model.model, "tokenizer", None)
        if tok is None:
            tc = getattr(self.base_model.model, "tokenizer_config", None)
            if tc is not None and hasattr(tc, "build"):
                try:
                    tok = tc.build()
                except Exception:
                    tok = None
        if tok is not None:
            cache = _DecodeExpandCache(tok)
            setattr(tok, "_decode_expand_hook", cache.get_hook())

    def uninstall(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

        if self._original_boundary_predictor is not None:
            self.base_model.model.local_encoder.boundary_predictor_module = self._original_boundary_predictor
            self._original_boundary_predictor = None

    def freeze_base_model(self) -> None:
        for param in self.base_model.parameters():
            param.requires_grad = False

    def trainable_gyrolabe_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in (
            self.q_class_embedding,
            self.family_embedding,
            self.micro_ref_embedding,
            self.boundary_distance_bias,
        ):
            params.extend(list(module.parameters()))
        return params

    def make_boundary_bias(
        self,
        *,
        width: int,
        lookahead: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self._ctx:
            return None

        canonical = self._ctx["canonical_bytes"]
        valid_mask = self._ctx["valid_mask"]

        if canonical.ndim == 1:
            canonical = canonical.unsqueeze(0)
            valid_mask = valid_mask.unsqueeze(0)

        batch, seq_len = canonical.shape
        bias = torch.zeros((batch, width), dtype=dtype, device=device)

        cached_states = self._ctx.get("states")
        lookahead_val = max(1, lookahead) if lookahead >= 0 else 1

        for b in range(batch):
            row_valid = valid_mask[b]
            valid_idx = torch.nonzero(row_valid, as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue

            if cached_states is not None:
                states_b = cached_states[b] if cached_states.ndim > 1 else cached_states
                row_states = states_b[valid_idx].contiguous().to(torch.int32)
            else:
                row_bytes = canonical[b, valid_idx].contiguous()
                row_signatures = signature_scan(row_bytes)
                row_states = signatures_to_states(row_signatures)

            if row_states.numel() == 0:
                continue

            row_dist = torch.zeros(row_states.shape[0], dtype=torch.long, device="cpu")
            if row_states.shape[0] > lookahead_val:
                adj = chirality_distance_adjacent(row_states, lookahead_val).to(torch.long)
                if lookahead_val == 1:
                    row_dist[1:] = adj[:-1]
                else:
                    row_dist[:-lookahead_val] = adj[:-lookahead_val]

            row_bias = self.boundary_distance_bias(
                row_dist.to(self.boundary_distance_bias.weight.device)
            ).squeeze(-1)

            full_row = torch.zeros(seq_len, dtype=row_bias.dtype, device=row_bias.device)
            full_row[valid_idx.to(full_row.device)] = row_bias
            n = min(seq_len, width)
            bias[b, :n] = full_row[:n].to(device=device, dtype=dtype)

        return bias

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        base = cast(nn.Module, self.base_model)

        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]

        if self.settings.enable_qclass_sparsity and input_ids is not None:
            canonical_bytes, valid_mask, _ = canonicalize_bolmo_ids(
                input_ids, offset=4, special_count=4
            )
            try:
                q_class, _, _ = extract_scan(canonical_bytes)[:3]
            except Exception:
                q_class, _, _ = qmap_extract(canonical_bytes)

            if q_class is not None:
                if q_class.ndim == 1:
                    q_class = q_class.unsqueeze(0)
                batch_size, seq_len = q_class.shape
                q_match = (q_class.unsqueeze(2) == q_class.unsqueeze(1)).unsqueeze(1)
                causal_mask = torch.tril(
                    torch.ones((seq_len, seq_len), dtype=torch.bool, device=q_class.device)
                )
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                sparse_bool_mask = causal_mask & q_match
                target_dtype = next(base.parameters()).dtype
                sparse_float_mask = torch.zeros(
                    (batch_size, 1, seq_len, seq_len),
                    dtype=target_dtype,
                    device=q_class.device,
                )
                sparse_float_mask.masked_fill_(~sparse_bool_mask, -10000.0)
                kwargs["attention_mask"] = {
                    "full_attention": sparse_float_mask,
                    "sliding_attention": sparse_float_mask,
                }

        return base(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        base = cast(Any, self.base_model)
        return base.generate(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        settings: GyroLabeSettings | None = None,
        **hf_kwargs: Any,
    ) -> "GyroLabeBolmoBridge":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        path = str(model_path)
        if "local_files_only" not in hf_kwargs:
            hf_kwargs = {**hf_kwargs, "local_files_only": True}
        import warnings
        warnings.filterwarnings(
            "ignore",
            message=".*rope_config_validation.*",
            category=FutureWarning,
        )
        _apply_bolmo_rope_patch()

        config = hf_kwargs.get("config")
        if config is None:
            config = AutoConfig.from_pretrained(
                path, trust_remote_code=True, local_files_only=hf_kwargs.get("local_files_only", True)
            )
            if getattr(config, "rope_scaling", None) and isinstance(config.rope_scaling, dict):
                rs = dict(config.rope_scaling)
                if "beta_fast" in rs and not isinstance(rs["beta_fast"], float):
                    rs["beta_fast"] = float(rs["beta_fast"])
                if "beta_slow" in rs and not isinstance(rs["beta_slow"], float):
                    rs["beta_slow"] = float(rs["beta_slow"])
                config.rope_scaling = rs
            hf_kwargs = {**hf_kwargs, "config": config}

        base_model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            **hf_kwargs,
        )
        return cls(base_model=cast(nn.Module, base_model), settings=settings)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def load_gyrolabe_bolmo(
    model_path: str | Path | None = None,
    *,
    settings: GyroLabeSettings | None = None,
    **hf_kwargs: Any,
) -> GyroLabeBolmoBridge:
    path = model_path if model_path is not None else DEFAULT_BOLMO_MODEL_PATH
    return GyroLabeBolmoBridge.from_pretrained(
        path,
        settings=settings,
        **hf_kwargs,
    )


def load_base_bolmo(model_path: str | Path | None = None, **hf_kwargs: Any) -> Any:
    """Load raw Bolmo without bridge. Same ROPE patch, no GyroLabe hooks."""
    _apply_bolmo_rope_patch()
    path = model_path if model_path is not None else DEFAULT_BOLMO_MODEL_PATH
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if "local_files_only" not in hf_kwargs:
        hf_kwargs = {**hf_kwargs, "local_files_only": True}
    import warnings
    warnings.filterwarnings("ignore", message=".*rope_config_validation.*", category=FutureWarning)
    return AutoModelForCausalLM.from_pretrained(
        str(path),
        trust_remote_code=True,
        **hf_kwargs,
    )