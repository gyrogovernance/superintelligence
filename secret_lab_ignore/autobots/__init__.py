"""
Autobots: Holographic Grid Transformer (HGT).

Byte-native causal LM with lossless physics (L1/L2/L3/L4 FSM)
and learned neural representation.
"""

from transformers import AutoConfig

from .config import HGTConfig

AutoConfig.register("hgt", HGTConfig)
from .model import HGTForCausalLM
from .tokenizer import PhysicsTokenizer
from . import physics

from .blocks import DirectionalAgentBlock

__all__ = [
    "HGTConfig",
    "HGTForCausalLM",
    "PhysicsTokenizer",
    "DirectionalAgentBlock",
    "physics",
]
