"""
Shared inference utilities: boundary mass, etc.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

BOLMO_VOCAB_OFFSET = 4


def _boundary_mass_np(logits520: np.ndarray) -> np.ndarray:
    """
    Probability that the next output carries a boundary (fused token).

    Bolmo 512 = base(256) + fused(256). Boundary mass = sum of fused probs.
    logits520: [B, 520]
    Returns: [B]
    """
    x = logits520 - logits520.max(axis=1, keepdims=True)
    p = np.exp(np.clip(x, -50, 50))
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    fused_slice = p[:, BOLMO_VOCAB_OFFSET + 256 : BOLMO_VOCAB_OFFSET + 512]
    return fused_slice.sum(axis=1)


def _boundary_mass_torch(logits520: torch.Tensor) -> torch.Tensor:
    x = logits520 - logits520.max(dim=1, keepdim=True).values
    p = torch.exp(x.clamp(-50, 50))
    p = p / (p.sum(dim=1, keepdim=True) + 1e-12)
    fused_slice = p[:, BOLMO_VOCAB_OFFSET + 256 : BOLMO_VOCAB_OFFSET + 512]
    return fused_slice.sum(dim=1)


def boundary_mass_from_logits(logits520: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Boundary mass from 520-way logits. Handles np or torch."""
    if isinstance(logits520, torch.Tensor):
        return _boundary_mass_torch(logits520)
    return _boundary_mass_np(logits520)


def boundary_prob_given_byte(
    base256: Union[np.ndarray, torch.Tensor],
    fused256: Union[np.ndarray, torch.Tensor],
    byte: int,
) -> float:
    """
    P(boundary | chosen byte) = sigmoid(fused[byte] - base[byte]).

    Kernel-native, decode-consistent. Use this for boundary refresh, not
    marginal boundary mass.
    """
    b = int(byte) & 0xFF
    if isinstance(base256, torch.Tensor):
        d = float(fused256[b].item() - base256[b].item())
    else:
        d = float(fused256[b] - base256[b])
    if d >= 0:
        z = np.exp(-d)
        return 1.0 / (1.0 + z)
    z = np.exp(d)
    return z / (1.0 + z)
