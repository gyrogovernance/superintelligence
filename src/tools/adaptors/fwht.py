"""
Fast Walsh-Hadamard utilities for reversible basis transforms.
"""

from __future__ import annotations

import math

import torch


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def fwht_axis(tensor: torch.Tensor, axis: int) -> torch.Tensor:
    """
    Apply orthonormal FWHT along a single axis.

    The transform is its own inverse under orthonormal normalization.
    """
    if tensor.numel() == 0:
        return tensor

    axis = axis % tensor.ndim
    n = int(tensor.shape[axis])
    if not is_power_of_two(n):
        raise ValueError(f"FWHT axis length must be power-of-two, got {n}")

    x = tensor.transpose(axis, -1).contiguous()
    y = x.reshape(-1, n).to(torch.float32).clone()

    h = 1
    while h < n:
        y = y.view(y.shape[0], -1, 2 * h)
        a = y[:, :, :h]
        b = y[:, :, h:]
        tmp = a.clone()
        a.copy_(tmp + b)
        b.copy_(tmp - b)
        y = y.view(y.shape[0], n)
        h *= 2

    y = y * (1.0 / math.sqrt(float(n)))
    y = y.reshape(x.shape).transpose(axis, -1).contiguous()
    return y


def fwht_1d(vector: torch.Tensor) -> torch.Tensor:
    if vector.ndim != 1:
        raise ValueError(f"fwht_1d expects 1D tensor, got {tuple(vector.shape)}")
    return fwht_axis(vector, axis=0)


def fwht_right(matrix: torch.Tensor) -> torch.Tensor:
    """
    Right-transform a 2D matrix along its input axis.

    For Linear.weight with shape [out, in], this applies FWHT over axis=1.
    """
    if matrix.ndim != 2:
        raise ValueError(f"fwht_right expects 2D tensor, got {tuple(matrix.shape)}")
    return fwht_axis(matrix, axis=1)
