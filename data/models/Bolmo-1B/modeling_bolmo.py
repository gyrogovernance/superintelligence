import copy
from typing import Any, Callable, Optional, Union, cast
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers.utils.generic import TransformersKwargs

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateOutput
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs

_DOLMA2_TOKENIZER_IDENTIFIER = "allenai/dolma2-tokenizer"
_DOLMA2_TOKENIZER_SUBDIR = "dolma2-tokenizer"


def _resolve_local_dolma2_tokenizer_path(identifier: str, model_path: str | None) -> str | None:
    if identifier != _DOLMA2_TOKENIZER_IDENTIFIER:
        return None

    candidates: list[Path] = []
    package_dir = Path(__file__).resolve().parent
    candidates.append(package_dir / _DOLMA2_TOKENIZER_SUBDIR)

    if model_path:
        candidates.append(Path(model_path) / _DOLMA2_TOKENIZER_SUBDIR)

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return str(candidate)
    return None

try:
    from .configuration_bolmo import BolmoConfig
    from .tokenization_bolmo import BolmoTokenizerConfig
    from .utils_bolmo import compute_boundary_mask, pad_right, pad_left, MaskState
except ImportError:
    from configuration_bolmo import BolmoConfig
    from tokenization_bolmo import BolmoTokenizerConfig
    from utils_bolmo import compute_boundary_mask, pad_right, pad_left, MaskState

try:
    from xlstm.xlstm_large.model import mLSTMLayer, mLSTMLayerConfig, mLSTMLayerStateType, soft_cap, mLSTMBackendConfig
except ImportError:
    raise ImportError("The `xlstm` package is required to use Bolmo. Please install it via `pip install xlstm`.")


@use_kernel_forward_from_hub("RMSNorm")
class BolmoRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        BolmoRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self._gyro_norm_hook = None

    def forward(self, hidden_states):
        hook = getattr(self, "_gyro_norm_hook", None)
        if hook is not None:
            out = hook(module=self, hidden_states=hidden_states)
            if out is not None:
                return out
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class _GroupNormWeightOnly(nn.Module):
    """GroupNorm with weight only (no bias) to match checkpoint that has no multihead_norm.bias."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.group_norm(x, self.num_groups, self.weight, None, self.eps)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)  # type: ignore
    value_states = repeat_kv(value, module.num_key_value_groups)  # type: ignore

    owner = getattr(module, "_gyrolabe_attention_owner", None)
    score_hook = None if owner is None else getattr(owner, "_attention_score_hook", None)

    raw_scores = None
    if score_hook is not None:
        raw_scores = score_hook(
            module=module,
            query=query,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )

    if raw_scores is None:
        raw_scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            raw_scores = raw_scores + causal_mask

    attn_weights = None
    softmax_hook = None if owner is None else getattr(owner, "_attention_softmax_hook", None)
    if softmax_hook is not None:
        attn_weights = softmax_hook(
            module=module,
            raw_scores=raw_scores,
            query=query,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )
    if attn_weights is None:
        attn_weights = nn.functional.softmax(raw_scores, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    reduce_hook = None if owner is None else getattr(owner, "_attention_reduce_hook", None)
    if reduce_hook is not None:
        reduced = reduce_hook(
            module=module,
            attn_weights=attn_weights,
            value_states=value_states,
            query=query,
            key=key_states,
            raw_scores=raw_scores,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )
        if reduced is not None:
            attn_output = reduced.transpose(1, 2).contiguous()
            return attn_output, attn_weights

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class BolmoAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: BolmoConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = BolmoRMSNorm(config.num_attention_heads * self.head_dim, config.rms_norm_eps)
        self.k_norm = BolmoRMSNorm(config.num_key_value_heads * self.head_dim, config.rms_norm_eps)
        assert config.layer_types is not None
        self.attention_type = config.layer_types[layer_idx]
        self.sliding_window = config.sliding_window if self.attention_type == "sliding_attention" else None

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.Tensor] = None,  # type: ignore
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class BolmoMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class BolmoDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: BolmoConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = BolmoAttention(config=config, layer_idx=layer_idx)

        self.mlp = BolmoMLP(config)
        self.post_attention_layernorm = BolmoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = BolmoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],  # type: ignore
    ) -> torch.Tensor:
        residual = hidden_states
        attn_out, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,  # type: ignore
        )
        hidden_states = self.post_attention_layernorm(attn_out)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        mlp_out = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(mlp_out)
        hidden_states = residual + hidden_states

        return hidden_states


class BolmoBoundaryPredictor(nn.Module):
    def __init__(self, config: BolmoConfig):
        super().__init__()

        self.d_model = config.hidden_size
        self.boundary_threshold = config.boundary_threshold
        self.boundary_predictor_lookahead = config.boundary_predictor_lookahead
        self.q_proj_layer = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj_layer = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        sequence_start_indices: Optional[torch.Tensor] = None,
        epsilon: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.boundary_predictor_lookahead == 0:
            # do not use the same rep for k and v, use current and one before as in H-Net + pad with negative to the left
            cos_sim = torch.cat([
                torch.ones((hidden_states.shape[0], 1), device=hidden_states.device, dtype=hidden_states.dtype) * -1,
                torch.einsum(
                    "b l d, b l d -> b l",
                    F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
                    F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
                )
            ], dim=1)
        else:
            cos_sim = torch.einsum(
                "b l d, b l d -> b l",
                F.normalize(self.q_proj_layer(hidden_states[:, :-self.boundary_predictor_lookahead]), dim=-1),
                F.normalize(self.k_proj_layer(hidden_states[:, self.boundary_predictor_lookahead:]), dim=-1),
            )
        boundary_logprobs = torch.log1p(-cos_sim.float().clip(max=1.0 - epsilon)) - math.log(2)
        POSITIVE_LOGPROB = 0.0
        NEGATIVE_LOGPROB = -100_000
        if boundary_logprobs.shape[1] > 0:
            if sequence_start_indices is None:
                boundary_logprobs[:, 0] = POSITIVE_LOGPROB
            else:
                pad_mask = torch.arange(boundary_logprobs.shape[1], device=boundary_logprobs.device)[None, :] < sequence_start_indices[:, None]
                boundary_logprobs = boundary_logprobs.masked_fill(pad_mask, NEGATIVE_LOGPROB)
                boundary_logprobs[torch.arange(len(boundary_logprobs), device=boundary_logprobs.device), sequence_start_indices] = POSITIVE_LOGPROB

        boundary_logprobs = F.pad(boundary_logprobs, (0, self.boundary_predictor_lookahead), "constant", NEGATIVE_LOGPROB)
        boundary_mask = compute_boundary_mask(boundary_logprobs, self.boundary_threshold)

        return boundary_logprobs, boundary_mask


class BolmoXLSTMLayer(mLSTMLayer):
    def __init__(self, config: BolmoConfig):
        self._gyro_xlstm_cuda_backend = mLSTMBackendConfig(
            chunkwise_kernel="chunkwise--triton_limit_chunk",
            sequence_kernel="native_sequence__triton",
            step_kernel="triton",
            mode="train",
            return_last_states=True,
            autocast_kernel_dtype="float32",
        )
        self._gyro_xlstm_cpu_backend = mLSTMBackendConfig(
            chunkwise_kernel="chunkwise--native_autograd",
            sequence_kernel="native_sequence__native",
            step_kernel="native",
            mode="train",
            return_last_states=True,
            autocast_kernel_dtype="float32",
        )
        self._gyro_xlstm_cuda_ready = bool(torch.cuda.is_available())

        super().__init__(mLSTMLayerConfig(
            embedding_dim=config.hidden_size,
            num_heads=config.num_local_heads,
            mlstm_backend=self._gyro_xlstm_cpu_backend,
        ))
        self._gyro_xlstm_preact_hook = None
        self._gyro_xlstm_soft_cap_hook = None
        self._gyro_xlstm_gate_act_hook = None
        self._gyro_xlstm_norm_hook = None
        self._gyro_xlstm_backend_hook = None

    def _original_forward(
        self, x: torch.Tensor,
        state: mLSTMLayerStateType | None = None,
        sequence_start_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, mLSTMLayerStateType | None]:
        use_xlstm = True
        if use_xlstm:
            assert x.ndim == 3, f"Input must have shape [B, S, D], got {x.shape}"
            B, S, _ = x.shape
            soft_cap_hook = getattr(self, "_gyro_xlstm_soft_cap_hook", None)

            def _apply_soft_cap(gate: str, x_raw: torch.Tensor) -> torch.Tensor:
                if soft_cap_hook is None:
                    return soft_cap(x_raw, cap_value=self.config.gate_soft_cap)
                x_new = soft_cap_hook(
                    module=self,
                    gate=gate,
                    preact=x_raw,
                    cap_value=self.config.gate_soft_cap,
                )
                if x_new is None:
                    return soft_cap(x_raw, cap_value=self.config.gate_soft_cap)
                return x_new

            if self.config.weight_mode == "single":
                q = self.q(x)
                k = self.k(x)
                v = self.v(x)
                o_preact = self.ogate_preact(x)
                i_preact = _apply_soft_cap("igate", self.igate_preact(x))
                f_preact = _apply_soft_cap("fgate", self.fgate_preact(x))
            elif self.config.weight_mode == "fused":
                qkv_opreact = self.qkv_opreact(x)
                q, k, v, o_preact = torch.tensor_split(
                    qkv_opreact,
                    (
                        self.qk_dim,
                        2 * self.qk_dim,
                        2 * self.qk_dim + self.v_dim,
                    ),
                    dim=-1,
                )
                if_preact = _apply_soft_cap("ifgate", self.ifgate_preact(x))
                i_preact, f_preact = torch.tensor_split(
                    if_preact, (self.config.num_heads,), dim=-1
                )
            else:
                raise ValueError(f"Unknown weight_mode: {self.config.weight_mode}")

            hook = getattr(self, "_gyro_xlstm_preact_hook", None)
            if hook is not None:
                patched = hook(
                    module=self,
                    x=x,
                    q=q,
                    k=k,
                    v=v,
                    o_preact=o_preact,
                    i_preact=i_preact,
                    f_preact=f_preact,
                    state=state,
                    sequence_start_indices=sequence_start_indices,
                )
                if patched is not None:
                    q, k, v, o_preact, i_preact, f_preact = patched

            q = q.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
            k = k.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
            v = v.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)

            if sequence_start_indices is not None:
                f_preact[torch.arange(B, device=f_preact.device), sequence_start_indices] = -100_000

            i_preact = i_preact.transpose(1, 2)
            f_preact = f_preact.transpose(1, 2)

            backend = cast(Any, self.mlstm_backend)
            selected_backend = (
                self._gyro_xlstm_cuda_backend
                if x.device.type == "cuda" and self._gyro_xlstm_cuda_ready
                else self._gyro_xlstm_cpu_backend
            )
            if hasattr(backend, "config"):
                backend.config = selected_backend
            if self.training:
                backend.config.mode = "train"
            else:
                backend.config.mode = "inference"
            backend_dtype = getattr(
                getattr(backend, "config", None),
                "autocast_kernel_dtype",
                None,
            )
            if backend_dtype is not None:
                target_dtype = (
                    torch.float32 if backend_dtype == "float32" else torch.float16 if backend_dtype == "float16" else None
                )
                if target_dtype is not None and q.dtype != target_dtype:
                    q = q.to(target_dtype)
                if target_dtype is not None and k.dtype != target_dtype:
                    k = k.to(target_dtype)
                if target_dtype is not None and v.dtype != target_dtype:
                    v = v.to(target_dtype)
                if target_dtype is not None and i_preact.dtype != target_dtype:
                    i_preact = i_preact.to(target_dtype)
                if target_dtype is not None and f_preact.dtype != target_dtype:
                    f_preact = f_preact.to(target_dtype)
                if target_dtype is not None and o_preact.dtype != target_dtype:
                    o_preact = o_preact.to(target_dtype)

            if state is None:
                c_initial, n_initial, m_initial = None, None, None
            else:
                c_initial, n_initial, m_initial = state

            backend_hook = getattr(self, "_gyro_xlstm_backend_hook", None)
            if backend_hook is not None:
                backend_result = backend_hook(
                    module=self,
                    q=q,
                    k=k,
                    v=v,
                    i=i_preact,
                    f=f_preact,
                    c_initial=c_initial,
                    n_initial=n_initial,
                    m_initial=m_initial,
                )
                if backend_result is None:
                    h, state = backend(
                        q=q, k=k, v=v, i=i_preact, f=f_preact,
                        c_initial=c_initial, n_initial=n_initial, m_initial=m_initial,
                    )
                else:
                    h, state = backend_result
            else:
                h, state = backend(
                    q=q, k=k, v=v, i=i_preact, f=f_preact,
                    c_initial=c_initial, n_initial=n_initial, m_initial=m_initial,
                )
            expected_h_shape = (B, self.config.num_heads, S, self.v_dim // self.config.num_heads)
            assert h.shape == expected_h_shape, f"Got {h.shape}, expected {expected_h_shape}"

            h = h.transpose(1, 2)
            norm_hook = getattr(self, "_gyro_xlstm_norm_hook", None)
            if norm_hook is None:
                h_norm = self.multihead_norm(h)
            else:
                h_norm = norm_hook(module=self, hidden_states=h)
                if h_norm is None:
                    h_norm = self.multihead_norm(h)
            h_norm = h_norm.reshape(B, S, -1)
            gate_act_hook = getattr(self, "_gyro_xlstm_gate_act_hook", None)
            if gate_act_hook is None:
                gate = self.ogate_act_fn(o_preact)
            else:
                gate = gate_act_hook(
                    module=self,
                    gate="ogate",
                    preact=o_preact,
                )
                if gate is None:
                    gate = self.ogate_act_fn(o_preact)
            h_out = gate * h_norm
            y = self.out_proj(h_out)
            return y, state
        else:
            return x, state

    def _has_bridge_hooks(self) -> bool:
        return (
            getattr(self, "_gyro_xlstm_preact_hook", None) is not None
            or getattr(self, "_gyro_xlstm_soft_cap_hook", None) is not None
            or getattr(self, "_gyro_xlstm_gate_act_hook", None) is not None
            or getattr(self, "_gyro_xlstm_norm_hook", None) is not None
            or getattr(self, "_gyro_xlstm_backend_hook", None) is not None
        )

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        past_key_values: Optional[dict] = None,
        use_cache: bool = False,
        sequence_start_indices: Optional[torch.Tensor] = None,
        cache_mask: Optional[MaskState] = None
    ):
        use_xlstm = True
        if use_xlstm:
            backend = cast(Any, self.mlstm_backend)
            selected_backend = (
                self._gyro_xlstm_cuda_backend
                if x.device.type == "cuda" and self._gyro_xlstm_cuda_ready
                else self._gyro_xlstm_cpu_backend
            )
            if hasattr(backend, "config"):
                backend.config = selected_backend
            if self.training:
                backend.config.mode = "train"
            else:
                backend.config.mode = "inference"

            if use_cache:
                assert past_key_values is not None
                prev_mode = backend.config.mode
                state = past_key_values.get("state", None)

                if cache_mask is not None:
                    state_for_model = cast(
                        mLSTMLayerStateType,
                        tuple(cache_mask.selective_get(x, inv=True) for x in state)
                        if state is not None else None
                    )
                else:
                    state_for_model = state

                h, new_state = self._original_forward(
                    x, state=state_for_model,
                    sequence_start_indices=sequence_start_indices
                )
                assert new_state is not None

                if state is None or cache_mask is None:
                    state = new_state
                else:
                    for i in range(len(state)):
                        cache_mask.selective_put(new_state[i], state[i], inv=True)

                past_key_values["state"] = state
                backend.config.mode = prev_mode
                return h
            else:
                if self._has_bridge_hooks():
                    h, _ = self._original_forward(
                        x, state=None, sequence_start_indices=sequence_start_indices
                    )
                    return h
                h, _ = super().forward(x)
                return h
        else:
            if use_cache and past_key_values is not None:
                past_key_values.setdefault("state", None)
            return x

class BolmoLocalLayer(nn.Module):
    def __init__(self, config: BolmoConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.act_fn = ACT2FN[config.hidden_act]

        self.xlstm = BolmoXLSTMLayer(config)

        local_mlp_config = copy.deepcopy(config)
        local_mlp_config.intermediate_size = config.local_intermediate_size
        self.mlp = BolmoMLP(local_mlp_config)

        self.pre_xlstm_layernorm = BolmoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = BolmoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        sequence_start_indices: Optional[torch.Tensor] = None,
        past_key_values: Optional[dict] = None,
        use_cache: Optional[bool] = False,
        cache_mask: Optional[MaskState] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        xlstm_out = self.xlstm(self.pre_xlstm_layernorm(hidden_states), sequence_start_indices=sequence_start_indices, past_key_values=past_key_values["xlstm"] if past_key_values is not None else None, use_cache=use_cache, cache_mask=cache_mask)
        hidden_states = residual + xlstm_out

        # Fully Connected
        residual = hidden_states
        ffn_out = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        hidden_states = residual + ffn_out

        return hidden_states


class BolmoLocalEncoder(nn.Module):
    def __init__(self, config: BolmoConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.add_expanded_embeddings = config.add_expanded_embeddings

        self.byte_embedding = nn.Embedding(
            config.vocab_size,
            self.hidden_size,
        )
        if self.add_expanded_embeddings:
            self.subword_embedding = nn.Embedding(
                config.subword_vocab_size,
                self.hidden_size,
            )
        else:
            self.subword_embedding = None

        self.layers = nn.ModuleList(
            [BolmoLocalLayer(config) for _ in range(config.num_local_encoder_layers)]
        )

        self.post_last_block_norm = BolmoRMSNorm(
            self.hidden_size,
            config.local_rms_norm_eps,
        )
        self.out_projection = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
        )

        self.boundary_predictor_module = BolmoBoundaryPredictor(config)

        self.has_cache = False

    def prepare_inference_cache(self, batch_size: int):
        device = next(self.parameters()).device
        self.has_cache = True

        self.cache_seqlens = 0
        self.last_h = torch.zeros((batch_size, self.hidden_size), dtype=self.out_projection.weight.dtype, device=device)
        self.layer_states = [{"xlstm": {}} for _ in range(len(self.layers))]

    def free_inference_cache(self):
        self.has_cache = False
        if hasattr(self, "cache_seqlens"):
            del self.cache_seqlens
        if hasattr(self, "last_h"):
            del self.last_h
        if hasattr(self, "layer_states"):
            del self.layer_states

    def _embed(self, tokens, expanded_input_ids: Optional[torch.Tensor] = None):
        embeddings = self.byte_embedding(tokens)
        if self.add_expanded_embeddings:
            assert expanded_input_ids is not None and self.subword_embedding is not None
            embeddings = embeddings + self.subword_embedding(expanded_input_ids)

        return embeddings

    def _pool(
        self,
        h: torch.Tensor,
        boundary_mask: torch.Tensor | None,
        n_patches: int,
        boundary_state: Optional[MaskState] = None,
    ):
        if self.has_cache and self.cache_seqlens > 0:
            assert boundary_state is not None
            if boundary_state.all():
                assert h.shape[1] == 1
                reduced_h = h
            else:
                reduced_h = h[[], :, :]
        else:
            assert boundary_mask is not None

            L = h.shape[1]
            token_idx = (
                torch.arange(L, device=h.device)[None, :] + (~boundary_mask).long() * L  # type: ignore
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)
            index = seq_sorted_indices[:, :n_patches, None].expand(
                -1, -1, h.shape[-1]
            )

            reduced_h = torch.gather(
                h,
                dim=1,
                index=index,
            )

        return reduced_h

    def forward(
        self,
        input_ids,
        true_boundary_mask: Optional[torch.Tensor] = None,
        boundary_state: Optional[MaskState] = None,
        pad_state: Optional[MaskState] = None,
        expanded_input_ids: Optional[torch.Tensor] = None,
        sequence_start_indices: Optional[torch.Tensor] = None,
    ):
        embeddings = self._embed(input_ids, expanded_input_ids)

        # pass through encoder layers
        if self.has_cache and self.cache_seqlens > 0:
            assert pad_state is not None

            # step those batch positions which are not currently idle (i.e. at a boundary position)
            # if all batch positions are idle, skip the step entirely
            # all positions being idle only happens if fuse_boundaries=False. In this case, the step where we
            # obtain a new representation from the global model will have all positions for the local encoder being idle.
            if not pad_state.all():
                h = pad_state.selective_get(embeddings, inv=True)

                for i, block in enumerate(self.layers):
                    h = block(h, past_key_values=self.layer_states[i], use_cache=True, cache_mask=pad_state)

                if self.post_last_block_norm is not None:
                    h = self.post_last_block_norm(h)

                pad_state.selective_put(h[:, -1, :], self.last_h, inv=True)

            h = self.last_h.unsqueeze(1)
        else:
            h = embeddings
            for i, block in enumerate(self.layers):
                if self.has_cache:
                    use_cache = True
                    past_key_values = self.layer_states[i]
                else:
                    use_cache = False
                    past_key_values = None

                h = block(h, past_key_values=past_key_values, use_cache=use_cache, sequence_start_indices=sequence_start_indices)

            if self.post_last_block_norm is not None:
                h = self.post_last_block_norm(h)

            if self.has_cache:
                self.last_h.copy_(h[:, -1, :])

        if not self.has_cache or self.cache_seqlens == 0: # only used for prefill
            boundary_logprobs, boundary_mask = self.boundary_predictor_module(
                h,
                sequence_start_indices=sequence_start_indices,
            )
            if boundary_state is not None:
                # can't predict through encoder - must be through prev local decoder step
                boundary_mask[:, -1] = boundary_state.mask
        else:
            boundary_logprobs = boundary_mask = None

        # overwrite with true boundaries
        if true_boundary_mask is not None:
            boundary_mask = true_boundary_mask

        patch_embeddings = self._pool(
            h=h,
            boundary_mask=boundary_mask,
            n_patches=int(cast(torch.Tensor, boundary_mask).sum(-1).max().item()) if boundary_mask is not None else 1,
            boundary_state=boundary_state,
        )
        patch_embeddings = self.out_projection(patch_embeddings)

        if self.has_cache:
            self.cache_seqlens += input_ids.shape[1]

        return h, patch_embeddings, boundary_logprobs, boundary_mask


class BolmoLocalDecoder(nn.Module):
    def __init__(self, config: BolmoConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.initial_norm = BolmoRMSNorm(
            self.hidden_size,
            eps=config.local_rms_norm_eps,
        )

        self.in_projection = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
        )

        self.layers = nn.ModuleList(
            [BolmoLocalLayer(config) for _ in range(config.num_local_decoder_layers)]
        )

        self.has_cache = False

    def prepare_inference_cache(self, batch_size: int):
        device = next(self.parameters()).device
        self.has_cache = True

        self.cache_seqlens = 0
        self.last_value = torch.zeros((batch_size, self.hidden_size), dtype=self.in_projection.weight.dtype, device=device)
        self.layer_states = [{"xlstm": {}} for _ in range(len(self.layers))]

    def free_inference_cache(self):
        self.has_cache = False
        if hasattr(self, "cache_seqlens"):
            del self.cache_seqlens
        if hasattr(self, "last_value"):
            del self.last_value
        if hasattr(self, "layer_states"):
            del self.layer_states

    def _depool(
        self,
        embeds: torch.Tensor,
        patch_embeds: torch.Tensor,
        boundary_mask: Optional[torch.Tensor],
        boundary_state: Optional[MaskState] = None,
        sequence_start_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = embeds
        if self.has_cache and self.cache_seqlens > 0:
            assert boundary_state is not None

            if patch_embeds.numel() > 0:
                # we got a new value from the global model, so must be at boundary position
                h_patch = patch_embeds[:, -1:, :]
                h = embeds + h_patch

                self.last_value.copy_(h_patch[:, -1])
            else:
                h = embeds + self.last_value.unsqueeze(1)

            # skip pad positions until we get a new value from the global model
            if patch_embeds.numel() == 0:
                h = boundary_state.selective_get(h, inv=True)
            else:
                boundary_state = None

            if h.shape[0] > 0:
                for i, layer in enumerate(self.layers):
                    h = layer(h, past_key_values=self.layer_states[i], use_cache=True, cache_mask=boundary_state)

            self.cache_seqlens += h.shape[1]

            return h
        else:
            assert boundary_mask is not None

            h_patch = patch_embeds 
            prepool_out = h_patch

        if prepool_out.shape[0] != embeds.shape[0]:
            depool_out = torch.zeros_like(embeds)
        elif prepool_out.shape[1] == 0:
            depool_out = torch.zeros_like(embeds)
        else:
            # TODO(benjaminm): clipping is problematic if it happens too much; track clip %.
            plug_back_idx = (
                torch.cumsum(boundary_mask, dim=1) - 1
            ).clip(min=0, max=prepool_out.shape[1] - 1)
            depool_out = torch.gather(
                prepool_out,
                dim=1,
                index=plug_back_idx.unsqueeze(-1).expand(-1, -1, self.hidden_size),
            )

            depool_out_modulated = depool_out
            h = depool_out_modulated + embeds

            for i, layer in enumerate(self.layers):
                if self.has_cache:
                    use_cache = True
                    past_key_values = self.layer_states[i]
                else:
                    use_cache = False
                    past_key_values = None

                h = layer(h, past_key_values=past_key_values, use_cache=use_cache, sequence_start_indices=sequence_start_indices)

            if self.has_cache:
                if prepool_out.shape[1] > 0:
                    self.last_value.copy_(prepool_out[:, -1])
                self.cache_seqlens += h.shape[1]

            return h

        return h

    def forward(
        self,
        embeds: torch.Tensor,
        patch_embeds: torch.Tensor,
        boundary_state: Optional[MaskState],
        boundary_mask: torch.Tensor | None,
        sequence_start_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.in_projection(embeds)
        h_patch = self.initial_norm(patch_embeds)

        return self._depool(
            embeds=h,
            patch_embeds=h_patch,
            boundary_mask=boundary_mask,
            boundary_state=boundary_state,
            sequence_start_indices=sequence_start_indices,
        )


class BolmoRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: BolmoConfig, device=None, rope_type: Optional[str] = None):
        super().__init__()
        if rope_type is not None:
            self.rope_type = rope_type
        elif hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            # BC: "rope_type" was originally "type"
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        assert self.rope_type is not None

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() * position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
            return cos, sin


class BolmoPreTrainedModel(PreTrainedModel):
    config: BolmoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BolmoDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": BolmoDecoderLayer,
        "attentions": BolmoAttention,
    }


class BolmoModel(BolmoPreTrainedModel):
    def __init__(self, config: BolmoConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.local_encoder = BolmoLocalEncoder(config)
        self.local_decoder = BolmoLocalDecoder(config)

        self.layers = nn.ModuleList(
            [BolmoDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = BolmoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.rotary_embs = nn.ModuleDict(
            {
                "sliding_attention": BolmoRotaryEmbedding(config=config, rope_type="default"),
                "full_attention": BolmoRotaryEmbedding(config=config),
            }
        )

        self.tokenizer_config = BolmoTokenizerConfig(**config.tokenizer_config)
        self._tokenizer = None
        self._resolve_local_tokenizer_config_path()

        # Initialize weights and apply final processing
        self.post_init()

    def _resolve_local_tokenizer_config_path(self) -> None:
        local_tokenizer_path = _resolve_local_dolma2_tokenizer_path(
            self.tokenizer_config.original_identifier or "",
            getattr(self, "name_or_path", None),
        )
        if local_tokenizer_path is not None:
            self.tokenizer_config.original_identifier = local_tokenizer_path

    def get_input_embeddings(self) -> nn.Module:  # type: ignore
        return self.local_encoder.byte_embedding

    def set_input_embeddings(self, value: nn.Embedding):  # type: ignore
        self.local_encoder.byte_embedding = value

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._resolve_local_tokenizer_config_path()
            self._tokenizer = self.tokenizer_config.build()

        return self._tokenizer

    def prefill_boundary_prediction_forward(
        self,
        input_ids: torch.Tensor,
        expanded_input_ids: Optional[torch.Tensor] = None,
        sequence_start_indices: Optional[torch.Tensor] = None,
        last_token_is_boundary: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        _, _, _, boundary_mask = self.local_encoder.forward(  # type: ignore
            input_ids,
            expanded_input_ids=expanded_input_ids,
            boundary_state=MaskState(torch.full((input_ids.shape[0],), fill_value=last_token_is_boundary, device=input_ids.device, dtype=torch.bool)),
            pad_state=MaskState(torch.zeros((input_ids.shape[0],), device=input_ids.device, dtype=torch.bool)),
            sequence_start_indices=sequence_start_indices,
        )

        return cast(torch.Tensor, boundary_mask)

    @check_model_inputs()
    def forward(
        self,
        input_ids: torch.Tensor,
        expanded_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        boundary_mask: Optional[torch.Tensor] = None,
        boundary_state: Optional[MaskState] = None,
        pad_state: Optional[MaskState] = None,
        sequence_start_indices: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],  # type: ignore
    ) -> BaseModelOutputWithPast:
        batch_size = input_ids.shape[0]
        device = input_ids.device

        if self.local_encoder.add_expanded_embeddings and expanded_input_ids is None and input_ids is not None:
            # not optimized
            expanded_input_ids_list: list[torch.Tensor] = []
            for example_idx in range(batch_size):
                expanded_input_ids_list.append(torch.tensor(self.tokenizer.expand_byte_ids(input_ids[example_idx].tolist()), dtype=torch.long, device=device))
            expanded_input_ids = pad_right(expanded_input_ids_list, value=self.tokenizer.pad_token_id, multiple_of=1)  # type: ignore

        h_byte, h_patch, _, boundary_mask = self.local_encoder(
            input_ids=input_ids,
            expanded_input_ids=expanded_input_ids,
            true_boundary_mask=boundary_mask,
            boundary_state=boundary_state,
            pad_state=pad_state,
        )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(  # type: ignore
                past_seen_tokens, past_seen_tokens + h_patch.shape[1], device=device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)  # type: ignore

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": h_patch,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        h_patch_after_global = h_patch
        position_embeddings_mapping = {
            "sliding_attention": self.rotary_embs["sliding_attention"](h_byte, position_ids),
            "full_attention": self.rotary_embs["full_attention"](h_byte, position_ids),
        }
        if h_patch.numel() > 0:
            # we need to convert from right-pad to left-pad and back for prefill
            # since flash attention expects left-pad and local/enc dec expect right-pad global tokens
            # should add better left-pad support but this only affects prefill so OK for now
            # although super inefficient!
            if boundary_mask is not None: # prefill
                n_boundaries = boundary_mask.sum(-1)

                for i, current_n_boundaries in enumerate(n_boundaries):
                    h_patch[i, -current_n_boundaries:] = h_patch[i, :current_n_boundaries].clone()

            for decoder_layer in self.layers[: self.config.num_hidden_layers]:
                attention_type = decoder_layer.self_attn.attention_type
                h_patch_after_global = decoder_layer(
                    h_patch_after_global,
                    attention_mask=causal_mask_mapping[attention_type],  # type: ignore
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings_mapping[attention_type],  # type: ignore
                    **kwargs,  # type: ignore
                )

            if boundary_mask is not None: # prefill
                n_boundaries = boundary_mask.sum(-1)

                for i, current_n_boundaries in enumerate(n_boundaries):
                    h_patch_after_global[i, :current_n_boundaries] = h_patch_after_global[i, -current_n_boundaries:].clone()

        h_out = self.local_decoder.forward(  # type: ignore
            embeds=h_byte,
            patch_embeds=h_patch_after_global,
            boundary_mask=boundary_mask,
            boundary_state=boundary_state,
            sequence_start_indices=sequence_start_indices,
        )
        h_out = self.norm(h_out)

        return BaseModelOutputWithPast(
            last_hidden_state=h_out,
            past_key_values=past_key_values,
        )


class BolmoForCausalLM(BolmoPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]  # type: ignore
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}  # type: ignore

    def __init__(self, config):
        super().__init__(config)
        self.model = BolmoModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self._decode_boundary_hook = None
        self._decode_token_hook = None
        self._decode_select_hook = None
        self._attention_score_hook = None
        self._attention_reduce_hook = None
        self._rms_norm_hook = None
        self._xlstm_preact_hook = None

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor,
        expanded_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        boundary_mask: Optional[torch.Tensor] = None,
        boundary_state: Optional[MaskState] = None,
        pad_state: Optional[MaskState] = None,
        sequence_start_indices: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],  # type: ignore
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, BolmoForCausalLM

        >>> model = BolmoForCausalLM.from_pretrained("meta-olmo3/Bolmo-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-olmo3/Bolmo-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            expanded_input_ids=expanded_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=use_cache,
            boundary_mask=boundary_mask,
            boundary_state=boundary_state,
            pad_state=pad_state,
            sequence_start_indices=sequence_start_indices,
            **kwargs,  # type: ignore
        )

        hidden_states = cast(torch.Tensor, outputs.last_hidden_state)
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(  # type: ignore
        self,
        inputs: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        use_model_defaults: Optional[bool] = None,
        labe: Any = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.Tensor]:
        # generic preprocessing

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config=generation_config, use_model_defaults=use_model_defaults, **kwargs
        )
        self._prepare_special_tokens(generation_config, device=self.model.device)  # type: ignore

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        # start of custom generate

        expand_input_ids = self.model.local_encoder.add_expanded_embeddings
        batch_size = len(inputs)
        _decode_expand_hook = getattr(self.model.tokenizer, "_decode_expand_hook", None)

        byte_input_ids = inputs
        if expand_input_ids:
            if _decode_expand_hook is not None:
                _decode_expand_hook(
                    "init",
                    byte_input_ids=byte_input_ids,
                    generated=byte_input_ids,
                    batch_size=batch_size,
                    tokenizer=self.model.tokenizer,
                )
                expanded_input_ids = _decode_expand_hook(
                    "get_expanded",
                    generated=byte_input_ids,
                    input_ids_for_model=byte_input_ids[:, -1:],
                    is_first_forward=True,
                    batch_size=batch_size,
                    device=self.device,
                    dtype=torch.long,
                )
                if not isinstance(expanded_input_ids, torch.Tensor):
                    expanded_input_ids = None
            else:
                expanded_input_ids = []

                for i in range(len(inputs)):
                    expanded_input_ids.append(torch.tensor(self.model.tokenizer.expand_byte_ids(inputs[i].tolist()), device=self.device, dtype=torch.long))

                expanded_input_ids = pad_left(expanded_input_ids, value=self.model.tokenizer.pad_token_id, multiple_of=1)  # type: ignore
        else:
            expanded_input_ids = None

        sequence_start_indices = (byte_input_ids == self.model.tokenizer.pad_token_id).sum(-1)
        batch_size = byte_input_ids.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        boundary_offset = self.model.tokenizer.offset + 256
        offset = self.model.tokenizer.offset
        eos = self.model.tokenizer.eos_token_id

        self.model.local_encoder.free_inference_cache()
        self.model.local_decoder.free_inference_cache()

        boundary_mask = self.model.prefill_boundary_prediction_forward(  # type: ignore
            byte_input_ids,
            expanded_input_ids=expanded_input_ids,
            sequence_start_indices=sequence_start_indices,
        )

        self.model.local_encoder.prepare_inference_cache(batch_size)
        self.model.local_decoder.prepare_inference_cache(batch_size)

        # roll back by one and force decoding to account for lookahead
        boundary_mask = boundary_mask[:, :-1]
        # need to roll one byte back and force decoding to detect whether the last byte is a boundary
        forced_decoding_ids = byte_input_ids[:, -1].cpu().tolist()
        byte_input_ids = byte_input_ids[:, :-1]
        expanded_input_ids = expanded_input_ids[:, :-1] if expanded_input_ids is not None else None
        # stays the same unless last token is pad.
        sequence_start_indices = (byte_input_ids == self.model.tokenizer.pad_token_id).sum(-1)

        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None  # type: ignore
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None  # type: ignore
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name="input_ids",
            inputs_tensor=byte_input_ids,
            input_ids_length=byte_input_ids.shape[1],
        )

        max_new_tokens_limit: Optional[int] = None
        max_total_length: Optional[int] = None
        if generation_config is not None:
            max_new_tokens_limit = generation_config.max_new_tokens
            max_total_length = generation_config.max_length

        logits_processor = self._get_logits_processor(
            generation_config=generation_config,  # type: ignore
            input_ids_seq_length=byte_input_ids.shape[1],
            encoder_input_ids=byte_input_ids,  # type: ignore
            logits_processor=logits_processor,
            device=byte_input_ids.device,  # type: ignore
            model_kwargs=model_kwargs,
        )
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,  # type: ignore
            stopping_criteria=stopping_criteria,
            tokenizer=self.model.tokenizer,
        )

        tokenizer = self.model.tokenizer
        bpe_token_end_id = tokenizer.bpe_token_end_id
        eos = tokenizer.eos_token_id

        max_n_prefill_patches = boundary_mask.sum(-1).max().item()
        tokens_generated_plus_prefilled = max_n_prefill_patches
        bytes_generated = 0
        decode_expand_hook = _decode_expand_hook

        def _grow_non_boundary_output(required_len: int) -> None:
            nonlocal generated_non_boundary
            if required_len < generated_non_boundary.shape[1]:
                return
            new_cap = generated_non_boundary.shape[1]
            while new_cap <= required_len:
                new_cap = max(1, new_cap * 2)
            expanded = torch.full(
                (batch_size, new_cap),
                tokenizer.pad_token_id,
                dtype=torch.long,
                device=generated_non_boundary.device,
            )
            expanded[:, :generated_non_boundary.shape[1]] = generated_non_boundary
            generated_non_boundary = expanded

        # generation state
        boundary_state = MaskState(boundary_mask[:, -1].clone())
        pad_state = MaskState(torch.zeros(batch_size, dtype=torch.bool, device=self.device))
        next_tokens = torch.full((batch_size,), bpe_token_end_id, device=self.device, dtype=torch.long)  # type: ignore
        prefix_non_boundary = byte_input_ids[:, :-1]
        prefix_len = prefix_non_boundary.shape[1]
        output_capacity = prefix_len + 1
        if max_new_tokens_limit is not None:
            output_capacity = output_capacity + max_new_tokens_limit
        elif max_total_length is not None:
            output_capacity = int(max_total_length)
        else:
            output_capacity = output_capacity + 32
        output_capacity = max(output_capacity, prefix_len + 2)
        # output container
        generated = torch.full(
            (batch_size, output_capacity),
            fill_value=tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        generated[:, :byte_input_ids.shape[1]] = byte_input_ids
        generated_len = byte_input_ids.shape[1]
        generated_non_boundary = torch.full(
            (batch_size, output_capacity),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=generated.device,
        )
        if prefix_len > 0:
            generated_non_boundary[:, :prefix_len] = prefix_non_boundary
        non_boundary_lengths = torch.full((batch_size,), prefix_len, dtype=torch.long, device=generated.device)
        non_boundary_last = byte_input_ids[:, -1]
        generated_non_boundary[
            torch.arange(batch_size, device=generated.device), non_boundary_lengths
        ] = non_boundary_last
        non_boundary_lengths += 1
        last_non_boundary_tokens = non_boundary_last.view(-1, 1)
        bytes_since_boundary = (boundary_mask.flip(1).cumsum(-1) == 0).sum(-1)
        is_first_forward = True
        global_past_key_values = None
        use_decode_expand_hook = _decode_expand_hook is not None and expand_input_ids

        while not finished.all():
            if labe is not None:
                labe.begin_step()
            input_ids_for_model = (
                generated[:, :generated_len]
                if is_first_forward
                else last_non_boundary_tokens
            )
            assert not (
                (input_ids_for_model == self.model.tokenizer.bpe_token_end_id) |
                (input_ids_for_model >= boundary_offset)
            ).any().item()  # type: ignore
            if expand_input_ids:
                expanded_input_ids_for_model = torch.empty_like(input_ids_for_model)
                if use_decode_expand_hook and decode_expand_hook is not None:
                    out = decode_expand_hook(
                        "get_expanded",
                        generated=generated,
                        input_ids_for_model=input_ids_for_model,
                        is_first_forward=is_first_forward,
                        batch_size=batch_size,
                        device=expanded_input_ids_for_model.device,
                        dtype=expanded_input_ids_for_model.dtype,
                    )
                    if out is not None:
                        expanded_input_ids_for_model = out
                    else:
                        for i in range(input_ids_for_model.shape[0]):
                            expanded_input_ids_for_model[i, :] = torch.tensor(
                                self.model.tokenizer.expand_byte_ids(
                                    generated[i, :generated_len].tolist(),
                                    n_last=input_ids_for_model.shape[1],
                                ),
                                device=expanded_input_ids_for_model.device,
                                dtype=expanded_input_ids_for_model.dtype,
                            )
                else:
                    for i in range(input_ids_for_model.shape[0]):
                        expanded_input_ids_for_model[i, :] = torch.tensor(
                            self.model.tokenizer.expand_byte_ids(
                                generated[i, :generated_len].tolist(),
                                n_last=input_ids_for_model.shape[1],
                            ),
                            device=expanded_input_ids_for_model.device,
                            dtype=expanded_input_ids_for_model.dtype,
                        )
            else:
                expanded_input_ids_for_model = None

            out = self.forward(  # type: ignore
                input_ids_for_model,
                expanded_input_ids=expanded_input_ids_for_model,
                boundary_mask=boundary_mask if is_first_forward else None,
                boundary_state=boundary_state,
                pad_state=pad_state,
                sequence_start_indices=sequence_start_indices,
                logits_to_keep=1,
                use_cache=True,
                past_key_values=global_past_key_values,
            )
            if labe is not None:
                labe.end_step()
            next_token_logits = cast(torch.Tensor, out.logits)
            _decode_boundary_hook = getattr(self, "_decode_boundary_hook", None)
            if _decode_boundary_hook is not None:
                last_bytes = [
                    int(generated_non_boundary[i, int(non_boundary_lengths[i].item()) - 1].item())
                    for i in range(batch_size)
                ]
                next_token_logits = _decode_boundary_hook(next_token_logits, last_bytes, boundary_offset)
            global_past_key_values = out.past_key_values

            if boundary_state.all():
                # new token, must not be boundary
                bytes_since_boundary[:] = 0
            else:
                boundary_state.selective_add(1, bytes_since_boundary, inv=True)

            if any(x is not None for x in forced_decoding_ids):
                # only supported for the first token atm, so len(next_token_logits) == batch_size
                assert len(next_token_logits) == batch_size and is_first_forward
                for example_idx in range(batch_size):
                    forced_decoding_id = forced_decoding_ids[example_idx]

                    if forced_decoding_id is not None:
                        no_boundary_logit = next_token_logits[example_idx, 0, forced_decoding_id].item()
                        boundary_logit = next_token_logits[example_idx, 0, forced_decoding_id + boundary_offset].item()

                        next_token_logits[example_idx, 0, :] = -100_000
                        next_token_logits[example_idx, 0, forced_decoding_id] = no_boundary_logit
                        next_token_logits[example_idx, 0, forced_decoding_id + boundary_offset] = boundary_logit

                        forced_decoding_ids[example_idx] = None  # only force once

            # passing input_ids to logit processor not implemented
            next_token_scores = logits_processor(None, next_token_logits[:, -1]) # type: ignore
            if labe is not None:
                boundary_offset = self.model.tokenizer.offset + 256
                offset = self.model.tokenizer.offset

                for b in range(batch_size):
                    scores = next_token_scores[b]
                    finite = torch.isfinite(scores)
                    cand = torch.nonzero(finite, as_tuple=False).squeeze(-1)
                    if cand.numel() == 0:
                        continue
                    k = min(64, int(cand.numel()))
                    topv, topi = torch.topk(scores[cand], k=k)
                    vocab_ids = cand[topi]

                    # Paired-candidate expansion: ensure both base and fused variants are present
                    partner_list = []
                    for tid in vocab_ids.tolist():
                        if offset <= tid < offset + 256:
                            partner = tid + 256
                            if partner < scores.numel():
                                partner_list.append(partner)
                        elif boundary_offset <= tid < boundary_offset + 256:
                            partner = tid - 256
                            if partner >= 0:
                                partner_list.append(partner)

                    if partner_list:
                        partner_t = torch.tensor(partner_list, device=vocab_ids.device, dtype=vocab_ids.dtype)
                        vocab_ids_expanded = torch.unique(torch.cat([vocab_ids, partner_t]))
                    else:
                        vocab_ids_expanded = vocab_ids

                    topv_expanded = scores[vocab_ids_expanded]
                    topv_adj = labe.adjust_logits_bu_ingress(vocab_ids_expanded, topv_expanded)
                    scores[vocab_ids_expanded] = topv_adj
                    next_token_scores[b] = scores

            _decode_select_hook = getattr(self, "_decode_select_hook", None)
            if callable(_decode_select_hook):
                new_next_tokens = _decode_select_hook(next_token_scores, boundary_offset)
                if new_next_tokens.ndim != 1:
                    new_next_tokens = new_next_tokens.reshape(-1)
                if new_next_tokens.shape[0] != next_token_scores.shape[0]:
                    raise RuntimeError(
                        "decode select hook returned batch size that does not match scores batch"
                    )
                new_next_tokens = new_next_tokens.to(
                    device=next_token_scores.device,
                    dtype=torch.long,
                )
            else:
                if generation_config is not None and generation_config.do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    new_next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    new_next_tokens = torch.argmax(next_token_scores, dim=-1)

            boundary_or_first = boundary_state.all() or is_first_forward
            boundary_offset = tokenizer.offset + 256
            next_non_boundary_tokens = torch.where(
                new_next_tokens >= boundary_offset,
                new_next_tokens - boundary_offset,
                new_next_tokens,
            )
            if boundary_or_first:
                tokens_generated_plus_prefilled += 1
                next_tokens = new_next_tokens
                append_mask = ~finished
            else:
                next_tokens[:] = bpe_token_end_id # type: ignore
                boundary_state.selective_put(new_next_tokens, next_tokens, inv=True)
                append_mask = (~boundary_state.cpu_mask) & (~finished.cpu())

            append_idx_cpu = torch.nonzero(append_mask, as_tuple=False).squeeze(-1)
            if append_idx_cpu.numel() > 0:
                append_idx = append_idx_cpu.to(generated_non_boundary.device)
                append_tokens = next_non_boundary_tokens[append_idx]
                write_pos = non_boundary_lengths[append_idx]
                _grow_non_boundary_output(int(write_pos.max().item()) + 1)
                generated_non_boundary[append_idx, write_pos] = append_tokens
                non_boundary_lengths[append_idx] = write_pos + 1
                last_non_boundary_tokens[append_idx] = append_tokens.view(-1, 1)
                if labe is not None:
                    for local_idx in range(int(append_idx_cpu.shape[0])):
                        example_idx = int(append_idx_cpu[local_idx].item())
                        next_token_cpu = int(append_tokens[local_idx].item())
                        if next_token_cpu != bpe_token_end_id:
                            labe.advance_with_token(next_token_cpu)

            is_first_forward = False

            boundary_state = MaskState(
                (next_tokens == self.model.tokenizer.bpe_token_end_id) |
                (next_tokens >= boundary_offset) |
                finished
            )  # type: ignore
            pad_state = MaskState(
                (next_tokens == self.model.tokenizer.bpe_token_end_id) |
                finished
            )

            _decode_token_hook = getattr(self, "_decode_token_hook", None)
            if _decode_token_hook is not None:
                _decode_token_hook(next_tokens.detach().clone(), boundary_offset)

            # Force EOS for (previously) finished sequences
            next_tokens = torch.where(finished, torch.full_like(next_tokens, eos), next_tokens)

            # Append next tokens
            if generated_len < generated.shape[1]:
                generated[:, generated_len] = next_tokens
            else:
                new_cap = max(2, generated.shape[1] * 2)
                expanded = torch.full(
                    (batch_size, new_cap),
                    tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=generated.device,
                )
                expanded[:, :generated.shape[1]] = generated
                generated = expanded
                generated[:, generated_len] = next_tokens
            generated_len += 1

            if _decode_expand_hook is not None and expand_input_ids:
                _decode_expand_hook("append", next_tokens=next_tokens, batch_size=batch_size, tokenizer=self.model.tokenizer)

            # Handle finished sequences
            stop_hit = next_tokens.eq(eos) | next_tokens.eq(eos + boundary_offset)

            for i in range(batch_size):
                # passing `scores` to stopping criteria not implemented
                token_count = int(non_boundary_lengths[i].item())
                if stopping_criteria(
                    generated_non_boundary[i, :token_count].unsqueeze(0),
                    None,
                ).squeeze(0).item():  # type: ignore
                    stop_hit[i] = True

            finished |= stop_hit
            bytes_generated += 1

            if max_new_tokens_limit is not None and bytes_generated >= max_new_tokens_limit:
                finished[:] = True
            if max_total_length is not None and byte_input_ids.shape[1] + bytes_generated >= max_total_length:
                finished[:] = True

        return pad_left(
            [
                generated_non_boundary[i, : int(non_boundary_lengths[i].item())]
                for i in range(batch_size)
            ],
            value=self.model.tokenizer.pad_token_id,
            multiple_of=1,
        )  # type: ignore

__all__ = ["BolmoForCausalLM", "BolmoModel", "BolmoPreTrainedModel"]