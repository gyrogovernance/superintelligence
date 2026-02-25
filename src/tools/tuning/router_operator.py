"""
Compiled router operator (W, b) for logit prediction.
Produced by run_tune.py SPC; consumed by run_model.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class RouterOperator:
    """Load and apply compiled W, b from npz."""

    def __init__(self, W: torch.Tensor, b: torch.Tensor) -> None:
        self.W = W
        self.b = b

    @classmethod
    def load(cls, path: Path | str) -> RouterOperator:
        """Load W and b from router_operator.npz."""
        data: dict[str, Any] = np.load(path, allow_pickle=False)
        W = torch.from_numpy(data["W"].astype(np.float32))
        b = torch.from_numpy(data["b"].astype(np.float32))
        return cls(W=W, b=b)

    def logits(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute logits: logits = W @ phi + b."""
        return self.W @ phi + self.b
