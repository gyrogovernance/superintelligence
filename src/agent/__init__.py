"""
Gyroscopic ASI Agent module.

Components:
- constants: CGM invariants, phenomenology parameters, observables
- inference: Inference Function for phase-aware byte selection
- information: Agent constants and observables
- intelligence: Complete agent integration
- adapters: model-specific bindings (token/embedding) to the fixed atlas
"""

from __future__ import annotations

__all__ = [
    "InferenceFunction",
    "InferenceState",
    "GyroscopicAgent",
    "AgentConfig",
    "AgentState",
    "TokenBinding",
    "EmbeddingAdapter",
    "ByteGatingMasks",
]

from .inference import InferenceFunction, InferenceState
from .intelligence import GyroscopicAgent, AgentConfig, AgentState
from .adapters import TokenBinding, EmbeddingAdapter, ByteGatingMasks