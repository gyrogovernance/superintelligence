"""
External hook installers for Bolmo runtime patching.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch.utils.hooks import RemovableHandle

from ..state_encoder import RouterStateEncoder2048
from .store import ResonatorStore


StateGetter = Callable[[int], tuple[int, int, int]]


def install_logits_prior_hook(
    *,
    model,
    store: ResonatorStore,
    get_router_state: StateGetter,
    encoder: RouterStateEncoder2048 | None = None,
    alpha: float = 0.05,
    tensor_key: str = "lm_head.weight",
) -> RemovableHandle:
    """
    Add an external prior to logits via model.lm_head forward hook.

    The prior is computed in transformed basis:
      prior = x_hat @ W_hat^T
    """
    if encoder is None:
        encoder = RouterStateEncoder2048(dim=2048)

    if not store.has(tensor_key):
        raise KeyError(f"Missing converted tensor key: {tensor_key}")
    lm_head_wht = store.get_tensor(tensor_key).to(torch.float32)

    def _hook(module, inputs, output):
        if not isinstance(output, torch.Tensor):
            return output
        if output.ndim != 3:
            return output

        batch_size = int(output.shape[0])
        device = output.device
        weight = lm_head_wht.to(device)

        priors: list[torch.Tensor] = []
        for b in range(batch_size):
            O, E, parity = get_router_state(b)
            x_hat = encoder.encode(O, E, parity, device=device)
            prior = torch.matmul(weight, x_hat)
            priors.append(prior)

        prior_batch = torch.stack(priors, dim=0).to(output.dtype)
        output[:, -1, :] = output[:, -1, :] + (float(alpha) * prior_batch)
        return output

    return model.lm_head.register_forward_hook(_hook)


def remove_hooks(handles: Sequence[RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()
