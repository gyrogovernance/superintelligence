"""
Forward pass through WHT-converted Bolmo weights using Router state as input.

Requires resonator converted with --profile full (all layers, norms, MLP).

EXPERIMENTAL: Not the Bolmo computation graph. SPC (tuning/run_tune.py)
compiles a Router-native operator that replaces this path. Prefer
tuning/operator.RouterOperator for production use.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .store import ResonatorStore


class SpectralForward:
    """
    Execute a forward pass through WHT-converted Bolmo weights
    using Router state as input.

    Single-vector attention: no sequence dimension; context is in Router state.
    """

    def __init__(
        self,
        store: ResonatorStore,
        config: dict[str, Any],
        *,
        lm_only: bool = False,
    ):
        self.store = store
        self.config = config
        self.lm_only = bool(lm_only)
        self.num_layers = int(config.get("num_hidden_layers", 16))
        self.hidden_size = int(config.get("hidden_size", 2048))
        self.num_heads = int(config.get("num_attention_heads", 16))
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = int(config.get("intermediate_size", 8192))
        self.rms_eps = float(config.get("rms_norm_eps", 1e-6))

    def forward(self, state_vector_2048: torch.Tensor) -> torch.Tensor:
        """
        state_vector_2048: [2048] float32, from FullStateEncoder
        Returns: logits [vocab_size]
        """
        h = state_vector_2048.clone().to(torch.float32)

        if not self.lm_only:
            for layer_idx in range(self.num_layers):
                h = self._transformer_layer(h, layer_idx)
            norm_w = self.store.get_tensor("model.norm.weight").to(torch.float32)
            h = self._rms_norm(h, norm_w)

        lm_head = self.store.get_tensor("lm_head.weight").to(torch.float32)
        logits = torch.matmul(lm_head, h)
        return logits

    def _transformer_layer(self, h: torch.Tensor, idx: int) -> torch.Tensor:
        prefix = f"model.layers.{idx}"

        residual = h

        q_w = self.store.get_tensor(f"{prefix}.self_attn.q_proj.weight").to(torch.float32)
        k_w = self.store.get_tensor(f"{prefix}.self_attn.k_proj.weight").to(torch.float32)
        v_w = self.store.get_tensor(f"{prefix}.self_attn.v_proj.weight").to(torch.float32)
        o_w = self.store.get_tensor(f"{prefix}.self_attn.o_proj.weight").to(torch.float32)
        qn_w = self.store.get_tensor(f"{prefix}.self_attn.q_norm.weight").to(torch.float32)
        kn_w = self.store.get_tensor(f"{prefix}.self_attn.k_norm.weight").to(torch.float32)

        q = torch.matmul(q_w, h)
        k = torch.matmul(k_w, h)
        v = torch.matmul(v_w, h)
        q = self._rms_norm(q, qn_w)
        k = self._rms_norm(k, kn_w)

        q = q.view(self.num_heads, self.head_dim)
        k = k.view(self.num_heads, self.head_dim)
        v = v.view(self.num_heads, self.head_dim)

        scores = (q * k).sum(dim=-1) / math.sqrt(float(self.head_dim))
        attn_out = (v * scores.unsqueeze(-1)).view(-1)

        h = torch.matmul(o_w, attn_out)

        norm_w = self.store.get_tensor(
            f"{prefix}.post_attention_layernorm.weight"
        ).to(torch.float32)
        h = self._rms_norm(h, norm_w)
        h = residual + h

        residual = h
        gate_w = self.store.get_tensor(f"{prefix}.mlp.gate_proj.weight").to(torch.float32)
        up_w = self.store.get_tensor(f"{prefix}.mlp.up_proj.weight").to(torch.float32)
        down_w = self.store.get_tensor(f"{prefix}.mlp.down_proj.weight").to(torch.float32)

        gate = torch.matmul(gate_w, h)
        up = torch.matmul(up_w, h)
        intermediate = torch.nn.functional.silu(gate) * up
        h = torch.matmul(down_w, intermediate)

        norm_w = self.store.get_tensor(
            f"{prefix}.post_feedforward_layernorm.weight"
        ).to(torch.float32)
        h = self._rms_norm(h, norm_w)
        h = residual + h

        return h

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean()
        x_norm = x * torch.rsqrt(variance + self.rms_eps)
        return weight * x_norm
