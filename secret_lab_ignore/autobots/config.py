"""
HGTConfig: physics constants + model hyperparameters.

The physics constants are exact integers. They are NOT learned.
They live in config.json alongside standard transformer hyperparams.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class HGTConfig(PretrainedConfig):
    model_type = "hgt"

    def __init__(
        self,
        gene_mic_s: int = 0xAA,
        q0: int = 0x033,
        q1: int = 0x0F0,
        archetype_state24: int = 0xAAA555,
        vocab_size: int = 256,
        family_size: int = 4,
        micro_ref_size: int = 64,
        resolution_dims: tuple[int, ...] = (64, 128, 256),
        num_heads: tuple[int, ...] = (4, 4, 8),
        ffn_multiplier: int = 4,
        max_position_embeddings: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gene_mic_s = gene_mic_s
        self.q0 = q0
        self.q1 = q1
        self.archetype_state24 = archetype_state24
        self.vocab_size = vocab_size
        self.family_size = family_size
        self.micro_ref_size = micro_ref_size
        self.resolution_dims = tuple(resolution_dims)
        self.num_heads = tuple(num_heads)
        self.ffn_multiplier = ffn_multiplier
        self.max_position_embeddings = max_position_embeddings
