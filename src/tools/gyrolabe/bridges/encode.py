from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

from src.sdk import RuntimeOps
from src.tools.gyrolabe import ops as gyrolabe_ops
from .bolmo_config import (
    DEFAULT_BOLMO_MODEL_PATH,
    BolmoEncodeBridgeConfig,
    BolmoTokenLayout,
    canonicalize_bolmo_ids,
    load_base_bolmo,
)

_POPCOUNT6_LUT = torch.tensor([i.bit_count() for i in range(64)], dtype=torch.long)


def _popcount6_tensor(q6: torch.Tensor) -> torch.Tensor:
    """Element-wise popcount for 6-bit values (0..63)."""
    lut = _POPCOUNT6_LUT.to(q6.device)
    return lut[q6.to(torch.long).clamp(0, 63)]


def _logit_clamped(p: torch.Tensor) -> torch.Tensor:
    p = torch.clamp(p, 1e-6, 1.0 - 1e-6)
    return torch.log(p) - torch.log1p(-p)


@dataclass(frozen=True)
class BolmoEncodedFields:
    canonical_bytes: torch.Tensor
    valid_mask: torch.Tensor
    boundary_mask: torch.Tensor
    q_class: torch.Tensor
    family: torch.Tensor
    micro_ref: torch.Tensor
    signatures: torch.Tensor
    states: torch.Tensor
    omega12: torch.Tensor
    omega12_valid: torch.Tensor
    chirality6: torch.Tensor
    shell: torch.Tensor
    q_hist64: torch.Tensor
    family_hist4: torch.Tensor
    micro_hist64: torch.Tensor
    shell_hist7: torch.Tensor
    q_weight_hist7: torch.Tensor
    bit_excitation6: torch.Tensor
    boundary_valid_mask: torch.Tensor

    @property
    def valid_count(self) -> int:
        return int(self.valid_mask.sum().item())


@dataclass(frozen=True)
class BoundaryFieldRecord:
    cosine_boundary_probs: torch.Tensor
    structural_boundary_probs: torch.Tensor
    combined_boundary_probs: torch.Tensor
    boundary_mask: torch.Tensor
    valid_mask: torch.Tensor
    patch_lengths: tuple[int, ...]
    patch_count: int
    boundary_count: int
    mean_bytes_per_patch: float
    attn_proxy: int
    kv_proxy: int
    q_hist64: torch.Tensor
    shell_hist7: torch.Tensor
    boundary_q_hist64: torch.Tensor
    boundary_micro_hist64: torch.Tensor


def _histogram(values: torch.Tensor, mask: torch.Tensor, bins: int) -> torch.Tensor:
    flat_mask = mask.reshape(-1)
    if flat_mask.numel() == 0 or not bool(flat_mask.any().item()):
        return torch.zeros(bins, dtype=torch.int64, device=values.device)
    flat_vals = values.reshape(-1)[flat_mask]
    return torch.bincount(flat_vals.to(torch.long), minlength=bins)


def _patch_lengths_from_boundary_mask(
    boundary_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[int, ...]:
    """
    Only count valid byte positions; collect runs ending at boundary positions.
    If the final valid byte is not marked boundary, include the trailing open run.
    """
    b_mask = boundary_mask.reshape(-1).to(torch.bool)
    v_mask = valid_mask.reshape(-1).to(torch.bool)
    n = b_mask.shape[0]
    lengths: list[int] = []
    run = 0
    for i in range(n):
        if not v_mask[i].item():
            continue
        run += 1
        if b_mask[i].item():
            lengths.append(run)
            run = 0
    if run > 0:
        lengths.append(run)
    return tuple(lengths)


def _structural_boundary_probs(fields: BolmoEncodedFields) -> torch.Tensor:
    """
    Build an exact structural boundary probability field over valid positions.
    Components: adjacent chirality distance, q-class jump, family jump.
    Returns probs in [0,1], same shape as fields.valid_mask, on CPU.
    """
    states = fields.states.to(torch.int32).reshape(-1)
    q_class = fields.q_class.to(torch.long).reshape(-1)
    family = fields.family.to(torch.long).reshape(-1)
    valid = fields.valid_mask.reshape(-1).to(torch.bool)
    n = states.shape[0]
    probs = torch.zeros(n, dtype=torch.float32)
    if n == 0:
        return probs.reshape(fields.valid_mask.shape)

    d_chi = gyrolabe_ops.chirality_distance_adjacent(states, lookahead=1).to(torch.float32)
    d_chi = torch.clamp(d_chi / 6.0, 0.0, 1.0)

    q_next = torch.zeros_like(q_class)
    q_next[:-1] = q_class[1:]
    q_xor = torch.bitwise_xor(q_class & 0x3F, q_next & 0x3F)
    d_q = _popcount6_tensor(q_xor.to(torch.long)).to(torch.float32) / 6.0

    fam_next = torch.zeros_like(family)
    fam_next[:-1] = family[1:]
    fam_xor = torch.bitwise_xor(family & 0x3, fam_next & 0x3)
    d_fam = ((fam_xor & 0x1) != 0).to(torch.float32) + ((fam_xor & 0x2) != 0).to(torch.float32)
    d_fam = d_fam / 2.0

    score = (0.50 * d_chi) + (0.35 * d_q) + (0.15 * d_fam)
    score = torch.clamp(score, 1e-6, 1.0)
    score = torch.where(valid, score, torch.zeros_like(score))
    return score.reshape(fields.valid_mask.shape)


def extract_bolmo_encode_fields(
    input_ids: torch.Tensor,
    *,
    layout: BolmoTokenLayout | None = None,
) -> BolmoEncodedFields:
    cfg = layout or BolmoTokenLayout()

    canonical_bytes, valid_mask, boundary_mask = canonicalize_bolmo_ids(
        input_ids,
        layout=cfg,
    )

    q_class, family, micro_ref, signatures, states = gyrolabe_ops.extract_scan(
        canonical_bytes.contiguous().to(torch.uint8)
    )

    omega12, omega12_valid = RuntimeOps.omega12_and_valid_from_states(states.to(torch.int32))
    omega12 = omega12.to(torch.int32)
    omega12_valid = omega12_valid.to(torch.bool)

    chirality6 = torch.bitwise_and(
        torch.bitwise_xor(torch.bitwise_right_shift(omega12, 6), omega12),
        0x3F,
    ).to(torch.long)

    lut = _POPCOUNT6_LUT.to(chirality6.device)
    shell = lut[chirality6]

    effective_mask = valid_mask.to(torch.bool) & omega12_valid

    q_hist64 = _histogram(q_class, effective_mask, 64)
    family_hist4 = _histogram(family, effective_mask, 4)
    micro_hist64 = _histogram(micro_ref, effective_mask, 64)
    shell_hist7 = _histogram(shell, effective_mask, 7)

    q6 = (q_class.to(torch.long) & 0x3F).to(q_class.device)
    wt_q = _popcount6_tensor(q6).to(torch.long)
    q_weight_hist7 = _histogram(wt_q, effective_mask, 7)

    bit_excitation6 = torch.zeros(6, dtype=torch.int64, device=q_class.device)
    for j in range(6):
        bit_j = (torch.bitwise_right_shift(q6, j) & 1).to(torch.bool)
        bit_excitation6[j] = (effective_mask & bit_j).sum().to(torch.long)
    bit_excitation6 = bit_excitation6.cpu()

    boundary_valid_mask = valid_mask.to(torch.bool) & boundary_mask.to(torch.bool)

    return BolmoEncodedFields(
        canonical_bytes=canonical_bytes,
        valid_mask=valid_mask.to(torch.bool),
        boundary_mask=boundary_mask.to(torch.bool),
        q_class=q_class.to(torch.uint8),
        family=family.to(torch.uint8),
        micro_ref=micro_ref.to(torch.uint8),
        signatures=signatures.to(torch.int32),
        states=states.to(torch.int32),
        omega12=omega12,
        omega12_valid=omega12_valid,
        chirality6=chirality6.to(torch.uint8),
        shell=shell.to(torch.uint8),
        q_hist64=q_hist64.cpu(),
        family_hist4=family_hist4.cpu(),
        micro_hist64=micro_hist64.cpu(),
        shell_hist7=shell_hist7.cpu(),
        q_weight_hist7=q_weight_hist7.cpu(),
        bit_excitation6=bit_excitation6,
        boundary_valid_mask=boundary_valid_mask,
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
        boundary_logprobs, boundary_mask = self.inner(
            hidden_states,
            sequence_start_indices=sequence_start_indices,
            epsilon=epsilon,
        )

        fields = cast(BolmoEncodedFields | None, self.owner._ctx.get("fields"))
        if fields is None:
            return boundary_logprobs, boundary_mask

        mode = getattr(self.owner.config, "boundary_mode", "hybrid")

        cosine_probs = torch.exp(boundary_logprobs.detach().clamp(max=0.0))
        structural_probs = _structural_boundary_probs(fields).to(
            device=boundary_logprobs.device,
            dtype=boundary_logprobs.dtype,
        )

        if mode == "observe":
            combined_probs = cosine_probs
        elif mode == "exact":
            combined_probs = structural_probs
        else:
            wc = float(self.owner.config.boundary_cosine_weight)
            ws = float(self.owner.config.boundary_structural_weight)
            total = max(wc + ws, 1e-6)
            combined_probs = ((wc * cosine_probs) + (ws * structural_probs)) / total

        # optional global compression actuator:
        # calibrate boundary probabilities toward a target boundary rate = 1 / target_bytes_per_patch
        target_bpp = self.owner.config.target_bytes_per_patch
        if target_bpp is not None and float(target_bpp) > 0.0:
            valid = fields.valid_mask.to(device=combined_probs.device, dtype=torch.bool)
            valid_probs = combined_probs[valid]

            if valid_probs.numel() > 0:
                # target boundary rate corresponding to target mean bytes/patch
                target_rate = torch.tensor(
                    max(1e-4, min(1.0 - 1e-4, 1.0 / float(target_bpp))),
                    device=combined_probs.device,
                    dtype=combined_probs.dtype,
                )

                current_rate = torch.clamp(valid_probs.mean(), 1e-4, 1.0 - 1e-4)

                gain = float(self.owner.config.target_patch_gain)

                shift = _logit_clamped(target_rate) - _logit_clamped(current_rate)
                combined_logits = _logit_clamped(combined_probs)

                combined_probs = torch.sigmoid(combined_logits + (gain * shift))
                combined_probs = torch.clamp(combined_probs, 1e-6, 1.0 - 1e-6)

        combined_logprobs = torch.log(torch.clamp(combined_probs, 1e-6, 1.0))
        boundary_mask = self.compute_boundary_mask(
            combined_logprobs,
            getattr(self.inner, "boundary_threshold"),
        )
        self.owner._record_boundary_field(
            cosine_probs=cosine_probs,
            structural_probs=structural_probs,
            combined_probs=combined_probs,
            boundary_mask=boundary_mask,
        )
        return combined_logprobs, boundary_mask


class GyroLabeBolmoEncodeBridge(nn.Module):
    """
    Encode-only Bolmo bridge.

    Responsibilities:
      - canonicalize Bolmo token IDs into byte/gauge charts
      - extract q-class / family / micro-ref / signature / state fields
      - inject exact structural ingress bias into byte embeddings
      - inject exact chirality-distance bias into boundary prediction

    It intentionally does not own decode logic.
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
        self._last_boundary_field: BoundaryFieldRecord | None = None
        self._last_fields: BolmoEncodedFields | None = None

        self._validate_bolmo_structure()

        hidden_size = int(getattr(self.base_model.config, "hidden_size"))
        self.q_class_embedding = nn.Embedding(64, hidden_size)
        self.family_embedding = nn.Embedding(4, hidden_size)
        self.micro_ref_embedding = nn.Embedding(64, hidden_size)
        self.boundary_distance_bias = nn.Embedding(7, 1)

        self.reset_structural_parameters()
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

    def reset_structural_parameters(self) -> None:
        with torch.no_grad():
            self.q_class_embedding.weight.zero_()
            self.family_embedding.weight.zero_()
            self.micro_ref_embedding.weight.zero_()
            self.boundary_distance_bias.weight.zero_()

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

    def extract_fields(self, input_ids: torch.Tensor) -> BolmoEncodedFields:
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

        def _byte_embedding_hook(
            module: nn.Module,
            args: tuple[Any, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            fields = cast(BolmoEncodedFields | None, self._ctx.get("fields"))
            if fields is None:
                return output

            q_class = fields.q_class.to(device=output.device, dtype=torch.long)
            family = fields.family.to(device=output.device, dtype=torch.long)
            micro_ref = fields.micro_ref.to(device=output.device, dtype=torch.long)
            valid = fields.valid_mask.to(device=output.device, dtype=output.dtype).unsqueeze(-1)

            bias = (
                self.q_class_embedding(q_class)
                + self.family_embedding(family)
                + self.micro_ref_embedding(micro_ref)
            ).to(dtype=output.dtype)

            return output + (self.config.embedding_scale * bias * valid)

        self._handles.append(
            local_encoder.register_forward_pre_hook(_local_encoder_pre_hook, with_kwargs=True)
        )
        self._handles.append(
            local_encoder.byte_embedding.register_forward_hook(_byte_embedding_hook)
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
        self._last_boundary_field = None
        self._last_fields = None

    def last_boundary_field(self) -> BoundaryFieldRecord | None:
        return self._last_boundary_field

    def last_encoded_fields(self) -> BolmoEncodedFields | None:
        return self._last_fields

    def _record_boundary_field(
        self,
        cosine_probs: torch.Tensor,
        structural_probs: torch.Tensor,
        combined_probs: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> None:
        fields = cast(BolmoEncodedFields | None, self._ctx.get("fields"))
        if fields is None:
            return
        valid_mask = fields.valid_mask
        patch_lengths = _patch_lengths_from_boundary_mask(boundary_mask, valid_mask)
        patch_count = len(patch_lengths)
        boundary_count = int((valid_mask & boundary_mask).sum().item())
        mean_bytes_per_patch = (
            sum(patch_lengths) / len(patch_lengths) if patch_lengths else 0.0
        )
        attn_proxy = patch_count * patch_count
        kv_proxy = patch_count
        pred_boundary_valid = valid_mask.to(torch.bool) & boundary_mask.to(torch.bool)
        boundary_q_hist64 = _histogram(fields.q_class.to(torch.long), pred_boundary_valid, 64)
        boundary_micro_hist64 = _histogram(fields.micro_ref.to(torch.long), pred_boundary_valid, 64)
        self._last_boundary_field = BoundaryFieldRecord(
            cosine_boundary_probs=cosine_probs.detach(),
            structural_boundary_probs=structural_probs.detach(),
            combined_boundary_probs=combined_probs.detach(),
            boundary_mask=boundary_mask.detach(),
            valid_mask=valid_mask.detach(),
            patch_lengths=patch_lengths,
            patch_count=patch_count,
            boundary_count=boundary_count,
            mean_bytes_per_patch=mean_bytes_per_patch,
            attn_proxy=attn_proxy,
            kv_proxy=kv_proxy,
            q_hist64=fields.q_hist64,
            shell_hist7=fields.shell_hist7,
            boundary_q_hist64=boundary_q_hist64.cpu(),
            boundary_micro_hist64=boundary_micro_hist64.cpu(),
        )
        self._last_fields = fields

    @torch.no_grad()
    def prefill_probe(
        self, input_ids: torch.Tensor
    ) -> tuple[BolmoEncodedFields, BoundaryFieldRecord | None]:
        """Run forward far enough to populate bridge state; return field objects."""
        if self.config.strict_cpu and input_ids.device.type != "cpu":
            input_ids = input_ids.cpu()
        _ = self.forward(input_ids)
        return (
            self._last_fields or self.extract_fields(input_ids),
            self._last_boundary_field,
        )

    def freeze_base_model(self) -> None:
        for param in self.base_model.parameters():
            param.requires_grad = False

    def trainable_bridge_parameters(self) -> list[nn.Parameter]:
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
        fields = cast(BolmoEncodedFields | None, self._ctx.get("fields"))
        if fields is None:
            return None

        states = fields.states
        valid_mask = fields.valid_mask

        if states.ndim == 1:
            states = states.unsqueeze(0)
            valid_mask = valid_mask.unsqueeze(0)

        batch, seq_len = states.shape
        bias = torch.zeros((batch, width), dtype=dtype, device=device)

        lookahead_val = max(1, int(lookahead))

        for b in range(batch):
            row_valid = valid_mask[b]
            valid_idx = torch.nonzero(row_valid, as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue

            row_states = states[b, valid_idx].contiguous().to(torch.int32)
            if row_states.numel() == 0:
                continue

            row_dist = torch.zeros(row_states.shape[0], dtype=torch.long, device="cpu")
            if row_states.shape[0] > lookahead_val:
                adj = gyrolabe_ops.chirality_distance_adjacent(
                    row_states,
                    lookahead=lookahead_val,
                ).to(torch.long)

                if lookahead_val == 1:
                    row_dist[1:] = adj[:-1]
                else:
                    row_dist[:-lookahead_val] = adj[:-lookahead_val]

            row_bias = self.boundary_distance_bias(
                row_dist.to(self.boundary_distance_bias.weight.device)
            ).squeeze(-1)

            full_row = torch.zeros(
                seq_len,
                dtype=row_bias.dtype,
                device=row_bias.device,
            )
            full_row[valid_idx.to(full_row.device)] = row_bias

            n = min(seq_len, width)
            bias[b, :n] = full_row[:n].to(device=device, dtype=dtype)

        return bias

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