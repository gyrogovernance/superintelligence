"""
Holographic Grid Transformer for Causal Language Modeling.

Architecture:
  Physics -> BL1/TL1 -> Agent u1 -> BTL1_2 -> BL2/TL2 -> Agent u3 ->
  BTL2_3 -> BL3/TL3 -> Agent u5 -> HeadAgent -> 256 logits

The forward pass has two phases:
  1. PHYSICS PHASE: Compute exact FSM states from input bytes.
  2. NEURAL PHASE: Embed, attend, predict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, cast

import torch

try:
    from safetensors.torch import load_file, save_file
except ImportError:
    load_file = None
    save_file = None
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from .blocks import DirectionalAgentBlock, ByteBlock, TensorBlock, TransitionBlock
from .config import HGTConfig
from .embeddings import ByteLayer1, L4PositionEncoding, TensorLayer1
from .head import HeadAgent
from . import physics


class HGTForCausalLM(nn.Module):
    def __init__(self, config: HGTConfig):
        super().__init__()
        self.config = config
        dim0, dim1, dim2 = config.resolution_dims
        n0, n1, n2 = config.num_heads
        ffn0 = dim0 * config.ffn_multiplier
        ffn1 = dim1 * config.ffn_multiplier
        ffn2 = dim2 * config.ffn_multiplier

        self.register_buffer("mask12_table", physics.compute_mask12_table())
        with torch.no_grad():
            mtab = cast(torch.Tensor, getattr(self, "mask12_table")).long().unsqueeze(0)
            vtab = physics.compute_vertex_batch(
                mtab & 0xFFF, config.q0, config.q1
            )
        self.register_buffer("vertex_by_byte", vtab.squeeze(0).long())

        self.bl1 = ByteLayer1(config)
        self.tl1 = TensorLayer1(config)
        self.l4_pos = L4PositionEncoding(config)

        self.agent_1_bl = DirectionalAgentBlock(dim0, dim0, n0)
        self.agent_1_tl = DirectionalAgentBlock(dim0, dim0, n0)

        self.transition_1_2 = TransitionBlock(dim0, dim1, state_features=2)
        self.bl2 = ByteBlock(dim1, n1, ffn1)
        self.tl2 = TensorBlock(dim1, n1, ffn1)
        self.agent_3_bl = DirectionalAgentBlock(dim1, dim1, n1)
        self.agent_3_tl = DirectionalAgentBlock(dim1, dim1, n1)

        self.transition_2_3 = TransitionBlock(dim1, dim2, state_features=7)
        self.bl3 = ByteBlock(dim2, n2, ffn2)
        self.tl3 = TensorBlock(dim2, n2, ffn2)
        self.agent_5_bl = DirectionalAgentBlock(dim2, dim2, n2)
        self.agent_5_tl = DirectionalAgentBlock(dim2, dim2, n2)

        self.head = HeadAgent(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        batch, seq = input_ids.shape
        device = input_ids.device
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        with torch.no_grad():
            introns = (input_ids ^ self.config.gene_mic_s).to(torch.uint8)
            families = (torch.bitwise_right_shift(introns.int(), 6) & 0x3).long()
            micro_refs = (introns & 0x3F).long()
            mt: torch.Tensor = getattr(self, "mask12_table")
            mask12s = mt[input_ids.clamp(0, 255)].long() & 0xFFF

            l1_states = physics.compute_l1_trajectory(introns)
            vertices = physics.compute_vertex_batch(
                mask12s, self.config.q0, self.config.q1
            )
            l2_a8, l2_b8 = physics.compute_l2_trajectory(introns)
            l3_a12, l3_b12 = physics.compute_l3_trajectory(introns, mask12s)
            l4_O, l4_E = physics.compute_l4_commitments(mask12s)

        step_indices = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
        bl = self.bl1(input_ids, families, micro_refs)
        tl = self.tl1(l1_states.long(), vertices.long())
        l4_enc = self.l4_pos(l4_O, l4_E, step_indices)
        bl = bl + l4_enc
        tl = tl + l4_enc
        bl0, tl0 = bl, tl
        bl = self.agent_1_bl(bl0, tl0, key_padding_mask=key_padding_mask)
        tl = self.agent_1_tl(tl0, bl0, key_padding_mask=key_padding_mask)

        l2_feats = torch.stack([l2_a8.float() / 255.0, l2_b8.float() / 255.0], dim=-1)
        bl, tl = self.transition_1_2(bl, tl, l2_feats)
        bl = self.bl2(bl, key_padding_mask=key_padding_mask)
        tl = self.tl2(tl, key_padding_mask=key_padding_mask)
        bl0, tl0 = bl, tl
        bl = self.agent_3_bl(bl0, tl0, key_padding_mask=key_padding_mask)
        tl = self.agent_3_tl(tl0, bl0, key_padding_mask=key_padding_mask)

        l3_f0_a = (l3_a12 & 0x3F).float() / 63.0
        l3_f1_a = (torch.bitwise_right_shift(l3_a12, 6) & 0x3F).float() / 63.0
        l3_f0_b = (l3_b12 & 0x3F).float() / 63.0
        l3_f1_b = (torch.bitwise_right_shift(l3_b12, 6) & 0x3F).float() / 63.0
        horizon_dist = physics.compute_horizon_distance(l3_a12, l3_b12) / 12.0
        ab_dist = physics.compute_ab_distance(l3_a12, l3_b12) / 12.0
        state24 = torch.bitwise_or(
            torch.bitwise_left_shift(l3_a12.long(), 12), l3_b12.long()
        )
        archetype_dist = physics.compute_archetype_distance(state24) / 24.0
        l3_feats = torch.stack(
            [l3_f0_a, l3_f1_a, l3_f0_b, l3_f1_b, horizon_dist, ab_dist, archetype_dist],
            dim=-1,
        )
        bl, tl = self.transition_2_3(bl, tl, l3_feats)
        bl = self.bl3(bl, key_padding_mask=key_padding_mask)
        tl = self.tl3(tl, key_padding_mask=key_padding_mask)
        bl0, tl0 = bl, tl
        bl = self.agent_5_bl(bl0, tl0, key_padding_mask=key_padding_mask)
        tl = self.agent_5_tl(tl0, bl0, key_padding_mask=key_padding_mask)

        if labels is not None:
            byte_logits, fam_logits, mic_logits, vertex_logits = self.head(
                bl, tl, l4_O, l4_E, return_parts=True
            )
            shift_byte_logits = byte_logits[..., :-1, :].contiguous().view(-1, 256)
            shift_labels = labels[..., 1:].contiguous().view(-1)

            label_smoothing = float(kwargs.get("label_smoothing", 0.0))

            byte_loss = F.cross_entropy(
                shift_byte_logits, shift_labels, ignore_index=-100,
                label_smoothing=label_smoothing,
            )

            with torch.no_grad():
                next_ids = labels[:, 1:].clamp(0, 255)
                next_intr = (next_ids ^ self.config.gene_mic_s).to(torch.long)
                next_family = (torch.bitwise_right_shift(next_intr, 6) & 0x3).long()
                next_micro = (next_intr & 0x3F).long()
                vtab = cast(torch.Tensor, getattr(self, "vertex_by_byte"))
                next_vertex = vtab[next_ids]

            byte_probs = F.softmax(byte_logits[:, :-1, :], dim=-1)
            B, S1, _ = byte_probs.shape
            vtab = cast(torch.Tensor, getattr(self, "vertex_by_byte"))
            idx = vtab.view(1, 1, 256).expand(B, S1, 256).long()
            vertex_probs = torch.zeros(B, S1, 4, device=byte_probs.device, dtype=byte_probs.dtype)
            vertex_probs.scatter_add_(dim=2, index=idx, src=byte_probs)

            vtx_targets = next_vertex.clone()
            vtx_targets = torch.where(
                shift_labels.view(B, S1) == -100,
                torch.full_like(vtx_targets, -100),
                vtx_targets,
            )
            vertex_loss = F.nll_loss(
                torch.log(vertex_probs.clamp(min=1e-12)).view(-1, 4),
                vtx_targets.view(-1),
                ignore_index=-100,
            )

            shift_fam = fam_logits[:, :-1, :].contiguous().view(-1, 4)
            fam_targets = next_family.contiguous().view(-1)
            fam_targets = torch.where(
                shift_labels == -100,
                torch.full_like(fam_targets, -100),
                fam_targets,
            )
            fam_loss = F.cross_entropy(shift_fam, fam_targets, ignore_index=-100)

            shift_mic = mic_logits[:, :-1, :].contiguous().view(-1, 64)
            mic_targets = next_micro.contiguous().view(-1)
            mic_targets = torch.where(
                shift_labels == -100,
                torch.full_like(mic_targets, -100),
                mic_targets,
            )
            mic_loss = F.cross_entropy(shift_mic, mic_targets, ignore_index=-100)

            loss = byte_loss + 0.1 * fam_loss + 0.1 * mic_loss + 1.0 * vertex_loss
            logits = byte_logits
        else:
            logits = self.head(bl, tl, l4_O, l4_E)
            loss = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save model in HuggingFace-compatible format (config + safetensors or bin)."""
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(str(path))
        state = {k: v for k, v in self.state_dict().items()}
        if save_file is not None:
            save_file(state, str(path / "model.safetensors"))
        else:
            torch.save(state, path / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path) -> "HGTForCausalLM":
        """Load model from HuggingFace-compatible directory."""
        path = Path(pretrained_model_name_or_path)
        config = HGTConfig.from_pretrained(str(path))
        model = cls(config)
        sf_path = path / "model.safetensors"
        bin_path = path / "pytorch_model.bin"
        if sf_path.exists() and load_file is not None:
            state = load_file(str(sf_path))
        elif bin_path.exists():
            state = torch.load(bin_path, map_location="cpu")
        else:
            return model
        model.load_state_dict(state, strict=False)
        return model

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.0195,
        do_sample: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate bytes with FSM state tracking.
        Tracks L3 state across steps. BL1/TL1 handle validity natively.
        """
        self.eval()
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape")
        generated = input_ids.clone()
        batch_size = input_ids.size(0)
        device = input_ids.device

        with torch.no_grad():
            mt: torch.Tensor = getattr(self, "mask12_table")
            introns = (input_ids ^ self.config.gene_mic_s).to(torch.uint8)
            mask12s = mt[input_ids.clamp(0, 255)].long() & 0xFFF
            l3_a12, l3_b12 = self._last_l3_state(introns, mask12s, batch_size)
            state24_list = [
                int((l3_a12[i].item() << 12) | l3_b12[i].item())
                for i in range(batch_size)
            ]

        for _ in range(max_new_tokens):
            out = self.forward(
                input_ids=generated, attention_mask=attention_mask
            )
            logits = out.logits[:, -1, :]

            if attention_mask is not None:
                ext = torch.ones(
                    batch_size, 1, device=device, dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([attention_mask, ext], dim=1)

            next_bytes: list[int] = []
            for i in range(batch_size):
                lt = logits[i : i + 1]
                if do_sample:
                    probs = F.softmax(lt / temperature, dim=-1).clamp(min=1e-10)
                    next_byte = int(torch.multinomial(probs.squeeze(0), 1).item())
                else:
                    next_byte = int(lt.argmax().item())

                next_bytes.append(next_byte)
                state24_list[i] = physics.step_state_l3_scalar(
                    state24_list[i], next_byte
                )

            next_tokens = torch.tensor(
                [next_bytes], device=device, dtype=generated.dtype
            ).T
            generated = torch.cat([generated, next_tokens], dim=1)

        return generated

    def _last_l3_state(
        self, introns: torch.Tensor, mask12s: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute L3 trajectory and return last (l3_a12, l3_b12) per batch item."""
        l3_a12, l3_b12 = physics.compute_l3_trajectory(introns, mask12s)
        return l3_a12[:, -1], l3_b12[:, -1]
