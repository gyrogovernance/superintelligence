"""
Head Agent: L4 closure -> next byte prediction.

Hierarchical 6+2: Family (4-way) + Micro (64-way) -> byte via intron mapping.
Enforces constitutional byte structure (Appendix G).
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import HGTConfig
from . import physics


class HeadAgent(nn.Module):
    def __init__(self, config: HGTConfig):
        super().__init__()
        dim = config.resolution_dims[2]
        self.l4_proj = nn.Linear(24, dim)
        self.combine = nn.Linear(dim * 3, dim)
        self.family_head = nn.Linear(dim, 4)
        self.micro_head = nn.Linear(dim, 64)
        self.vertex_head = nn.Linear(dim, 4)
        perm = torch.empty(256, dtype=torch.long)
        for intr_val in range(256):
            perm[intr_val] = intr_val ^ physics.GENE_MIC_S
        self.register_buffer("_intron_to_byte", perm)
        byte_to_vertex = torch.empty(256, dtype=torch.long)
        for b in range(256):
            intr = physics.intron(b)
            m12 = physics.expand_intron_to_mask12(intr)
            byte_to_vertex[b] = physics.vertex_charge(m12)
        self.register_buffer("_byte_to_vertex", byte_to_vertex)

    def _l4_features(self, l4_O: Tensor, l4_E: Tensor) -> Tensor:
        """Expand O, E to 24 binary features [batch, seq, 24]."""
        ar = torch.arange(12, device=l4_O.device, dtype=torch.long)
        o = (torch.bitwise_right_shift(l4_O.unsqueeze(-1), ar) & 1)
        e = (torch.bitwise_right_shift(l4_E.unsqueeze(-1), ar) & 1)
        feats = torch.cat([o.float(), e.float()], dim=-1)
        if feats.size(-1) < 24:
            pad = torch.zeros(*feats.shape[:-1], 24 - feats.size(-1), device=feats.device)
            feats = torch.cat([feats, pad], dim=-1)
        return feats

    def _combine_family_micro(
        self, family_logits: Tensor, micro_logits: Tensor
    ) -> Tensor:
        """
        Map family [B,S,4] x micro [B,S,64] -> byte [B,S,256].
        Broadcasting: joint[f,m] = family[f] + micro[m]
        Then permute from intron order to byte order.
        """
        joint = family_logits.unsqueeze(-1) + micro_logits.unsqueeze(-2)
        intron_logits = joint.reshape(
            family_logits.shape[0], family_logits.shape[1], 256
        )
        return intron_logits[:, :, self._intron_to_byte]

    def forward(
        self,
        bl3: Tensor,
        tl3: Tensor,
        l4_O: Tensor,
        l4_E: Tensor,
        return_parts: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor]:
        l4_feats = self._l4_features(l4_O & 0xFFF, l4_E & 0xFFF)
        l4_proj = self.l4_proj(l4_feats)
        combined = torch.cat([bl3, tl3, l4_proj], dim=-1)
        h = F.gelu(self.combine(combined))

        family_logits = self.family_head(h)
        micro_logits = self.micro_head(h)
        vertex_logits = self.vertex_head(h)
        byte_logits = self._combine_family_micro(family_logits, micro_logits)
        B, S, _ = byte_logits.shape
        b2v = cast(torch.Tensor, getattr(self, "_byte_to_vertex"))
        idx = b2v.view(1, 1, 256).expand(B, S, 256)
        vterm = vertex_logits.gather(dim=-1, index=idx)
        byte_logits = byte_logits + 0.0 * vterm  # gating disabled; vertex supervised via losses
        if return_parts:
            return byte_logits, family_logits, micro_logits, vertex_logits
        return byte_logits
