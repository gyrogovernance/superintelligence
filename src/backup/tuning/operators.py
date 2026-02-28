"""
Multi-rate compiled operators for Router-native inference.

Three operators compiled by SPC:
  - FastOperator: local rhythm (boundary + byte prior), from L1/L2
  - SlowOperator: semantic correction, from L3/L4
  - MultiRateOperator: loads both and merges at inference time

Each CompiledOperator is a (W, b) linear map, compiled via ridge regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class CompiledOperator:
    """A single (W, b) linear map from features to logits."""

    W: torch.Tensor  # [V, D]
    b: torch.Tensor  # [V]
    name: str
    feature_dim: int
    vocab_size: int

    def logits(self, phi: torch.Tensor) -> torch.Tensor:
        return self.W @ phi + self.b

    def save(self, path: Path | str) -> None:
        path = Path(path)
        name_arr = np.frombuffer(self.name.encode("utf-8"), dtype=np.uint8)
        np.savez(
            path,
            W=self.W.detach().numpy().astype(np.float32),
            b=self.b.detach().numpy().astype(np.float32),
            name=name_arr,
            feature_dim=np.int32(self.feature_dim),
            vocab_size=np.int32(self.vocab_size),
        )

    @classmethod
    def load(cls, path: Path | str) -> CompiledOperator:
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        name = data["name"].tobytes().decode("utf-8")
        return cls(
            W=torch.from_numpy(data["W"].astype(np.float32)),
            b=torch.from_numpy(data["b"].astype(np.float32)),
            name=name,
            feature_dim=int(data["feature_dim"]),
            vocab_size=int(data["vocab_size"]),
        )


class MultiRateOperator:
    """
    Combines fast and slow operators for inference.

    At every byte: logits_fast = W_fast @ phi_fast + b_fast
    At patch boundaries: logits_slow = W_slow @ phi_slow + b_slow (cached)
    Combined: logits = logits_fast + logits_slow
    """

    def __init__(self, fast: CompiledOperator, slow: CompiledOperator) -> None:
        self.fast = fast
        self.slow = slow
        self._cached_slow_logits: torch.Tensor | None = None

    def update_fast(self, phi_fast: torch.Tensor) -> torch.Tensor:
        """Compute fast logits (every byte)."""
        return self.fast.logits(phi_fast)

    def update_slow(self, phi_slow: torch.Tensor) -> torch.Tensor:
        """Compute slow logits (at boundaries). Cached until next boundary."""
        self._cached_slow_logits = self.slow.logits(phi_slow)
        return self._cached_slow_logits

    def combined_logits(self, phi_fast: torch.Tensor) -> torch.Tensor:
        """Get combined logits using current fast + cached slow."""
        fast_logits = self.update_fast(phi_fast)
        if self._cached_slow_logits is not None:
            return fast_logits + self._cached_slow_logits
        return fast_logits
