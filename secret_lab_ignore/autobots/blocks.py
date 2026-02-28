"""
Building blocks for the HGT architecture.

ByteBlock: Self-attention on the byte stream (BL2, BL3).
TensorBlock: Self-attention on the tensor stream (TL2, TL3).
DirectionalAgentBlock: Asymmetric cross-attention (source queries target).
TransitionBlock: Resolution transition (8->16->24 bit).

All blocks use standard nn.MultiheadAttention and nn.Linear.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import HGTConfig


class ByteBlock(nn.Module):
    """
    Self-attention on byte stream with causal masking.
    Position i cannot attend to positions j > i (autoregressive constraint).
    """

    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
        """Upper-triangular True mask: position i cannot attend to j > i."""
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        causal = self._causal_mask(x.size(1), x.device)
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=causal,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class TensorBlock(nn.Module):
    """
    Self-attention on tensor stream with causal masking.
    Topology is encoded via L3 state injection (TransitionBlock).
    """

    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        causal = self._causal_mask(x.size(1), x.device)
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=causal,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DirectionalAgentBlock(nn.Module):
    """
    Asymmetric cross-attention with causal masking.
    Query from source, Key/Value from target.
    Position i in source cannot attend to positions j > i in target.
    """

    def __init__(self, dim_source: int, dim_target: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            dim_source, num_heads, dropout=dropout, batch_first=True
        )
        self.proj_target = nn.Linear(dim_target, dim_source)
        self.norm = nn.LayerNorm(dim_source)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        target_proj = self.proj_target(target)
        causal = self._causal_mask(source.size(1), source.device)
        out, _ = self.cross_attn(
            source, target_proj, target_proj,
            attn_mask=causal,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.norm(source + self.dropout(out))


class TransitionBlock(nn.Module):
    """
    Resolution transition between FSM levels.
    BTL1_2: Projects resolution_dims[0] -> resolution_dims[1], injects L2 (A8, B8).
    BTL2_3: Projects resolution_dims[1] -> resolution_dims[2], injects L3 (A12, B12).
    """

    def __init__(self, dim_in: int, dim_out: int, state_features: int):
        super().__init__()
        self.proj_bl = nn.Linear(dim_in, dim_out)
        self.proj_tl = nn.Linear(dim_in, dim_out)
        self.state_inject = nn.Linear(state_features, dim_out)

    def forward(
        self,
        bl: Tensor,
        tl: Tensor,
        state_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        bl_out = self.proj_bl(bl) + self.state_inject(state_features)
        tl_out = self.proj_tl(tl) + self.state_inject(state_features)
        return bl_out, tl_out
