"""
Base embeddings: BL1 (Byte Layer 1), TL1 (Tensor Layer 1), L4PositionEncoding.

BL1 embeds the GENE_Mic decomposition:
  - byte value (0-255) -> byte_embed
  - family (0-3) -> family_embed
  - micro_ref (0-63) -> micro_embed
  Output: sum of three embeddings, dim = resolution_dims[0]

TL1 embeds the GENE_Mac L1 state:
  - l1_state (0-255) -> l1_state_embed
  - vertex_charge (0-3) -> vertex_embed
  Output: sum of two embeddings, dim = resolution_dims[0]

Both BL1 and TL1 output tensors of shape [batch, seq, dim_0].
The L4 commitments (O, E) are added as a position encoding.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .config import HGTConfig


class ByteLayer1(nn.Module):
    """BL1: GENE_Mic embedding."""

    def __init__(self, config: HGTConfig):
        super().__init__()
        dim = config.resolution_dims[0]
        self.byte_embed = nn.Embedding(256, dim // 2)
        self.family_embed = nn.Embedding(4, dim // 4)
        self.micro_embed = nn.Embedding(64, dim // 4)

    def forward(
        self,
        input_ids: Tensor,
        families: Tensor,
        micro_refs: Tensor,
    ) -> Tensor:
        a = self.byte_embed(input_ids.clamp(0, 255))
        b = self.family_embed(families.clamp(0, 3))
        c = self.micro_embed(micro_refs.clamp(0, 63))
        return torch.cat([a, b, c], dim=-1)


class TensorLayer1(nn.Module):
    """TL1: GENE_Mac L1 state embedding."""

    def __init__(self, config: HGTConfig):
        super().__init__()
        dim = config.resolution_dims[0]
        self.l1_state_embed = nn.Embedding(256, dim - 4)
        self.vertex_embed = nn.Embedding(4, 4)

    def forward(
        self,
        l1_states: Tensor,
        vertex_charges: Tensor,
    ) -> Tensor:
        x = self.l1_state_embed(l1_states.clamp(0, 255))
        v = self.vertex_embed(vertex_charges.clamp(0, 3))
        return torch.cat([x, v], dim=-1)


class L4PositionEncoding(nn.Module):
    """
    Projects L4 commitments (O, E) into position vectors.
    Uses a learned gate conditioned on actual O/E state.
    The network learns WHEN closure has occurred (O~0, E~0) vs. when
    commitments should be amplified, rather than assuming a fixed 4-step cycle.
    """

    def __init__(self, config: HGTConfig):
        super().__init__()
        self.projection = nn.Linear(24, config.resolution_dims[0])
        self.gate = nn.Sequential(
            nn.Linear(24, 16),
            nn.GELU(),
            nn.Linear(16, 24),
            nn.Sigmoid(),
        )

    def _expand_12bit(self, x: Tensor) -> Tensor:
        """Expand 12-bit int to 12 binary features (one per bit)."""
        ar = torch.arange(12, device=x.device, dtype=torch.long)
        return (torch.bitwise_right_shift(x.unsqueeze(-1), ar) & 1).float()

    def forward(
        self,
        l4_O: Tensor,
        l4_E: Tensor,
        step_indices: Optional[Tensor] = None,
    ) -> Tensor:
        o_bits = self._expand_12bit(l4_O & 0xFFF)
        e_bits = self._expand_12bit(l4_E & 0xFFF)
        combined = torch.cat([o_bits, e_bits], dim=-1)
        if combined.size(-1) != 24:
            padded = torch.zeros(*combined.shape[:-1], 24, device=combined.device)
            padded[..., : combined.size(-1)] = combined
            combined = padded

        gate_values = self.gate(combined)
        combined = combined * gate_values

        return self.projection(combined)
