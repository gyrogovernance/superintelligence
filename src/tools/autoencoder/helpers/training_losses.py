"""Loss functions with individually logged components."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from src.tools.autoencoder.kernel import apply_k4_index, apply_signature_index

K4_GATE_IDS = (0, 1, 2, 3)  # id, S, C, F


@dataclass
class LossWeights:
    state_ce: float = 1.0
    recon_mse: float = 0.0
    state_bits: float = 0.0
    enc_eq: float = 0.0
    dec_eq: float = 0.0
    step: float = 0.0
    chi: float = 0.0
    shell: float = 0.0
    inverse: float = 0.0
    composition: float = 0.0
    signature: float = 0.0
    parity: float = 0.0
    tau_u: float = 0.0
    tau_v: float = 0.0
    commitment: float = 0.0
    end_state: float = 0.0
    provenance: float = 0.0
    rate: float = 0.0
    rank_ce: float = 0.0
    transition_ce: float = 0.0
    word_ce: float = 0.0


@dataclass
class LossReport:
    total: float
    components: dict[str, float] = field(default_factory=dict)


def state_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over 4096 state indices. logits [B, 4096], targets [B]."""
    return F.cross_entropy(logits, targets)


def state_bit_bce(
    bits_logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """BCE over 12 Omega coordinates. bits_logits [B, 12], targets [B, 12]."""
    return F.binary_cross_entropy_with_logits(bits_logits, targets.float())


def equivariance_defect(
    encoder: torch.nn.Module,
    batch: torch.Tensor,
    k4_perm: torch.Tensor,
    latent_action,
) -> torch.Tensor:
    """Mean encoder equivariance error over K4 for a state-index batch.

    batch: [B] int64 state indices. k4_perm: [4, 4096] long tensor mapping.
    latent_action: callable mapping (gate_i, z) -> rho(g) z. It is required:
    without a declared latent representation the equivariance target is
    undefined.
    """
    z = encoder(batch)
    total = batch.new_zeros((), dtype=torch.float32)
    for gate_i in K4_GATE_IDS:
        transformed = k4_perm[gate_i][batch]
        z_g = encoder(transformed)
        rho_z = latent_action(gate_i, z)
        total = total + (z_g - rho_z).pow(2).sum(dim=-1).mean()
    return total / len(K4_GATE_IDS)


def transition_loss(
    step_head_logits: torch.Tensor,
    next_state_index: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(step_head_logits, next_state_index)


def chi_transport_loss(
    chi_logits: torch.Tensor,
    chi_source: torch.Tensor,
    q6: torch.Tensor,
) -> torch.Tensor:
    """chi_pred should equal chi(x) XOR q(byte). chi_logits [B, 64]."""
    return F.cross_entropy(chi_logits, chi_source ^ q6)


def shell_loss(shell_logits: torch.Tensor, chi_target: torch.Tensor) -> torch.Tensor:
    """CE over 7 shells from a chirality target:
    L_shell = CE(shell_pred, popcount(chi))."""
    return F.cross_entropy(shell_logits, popcount_tensor(chi_target).long())


def popcount_tensor(x: torch.Tensor, d: int = 6) -> torch.Tensor:
    """Popcount over the ``d`` low bits; returns an integer tensor of the
    same dtype. The package is hQVM(6), so ``d`` defaults to 6; it is a
    keyword so callers that already generalize to ``hQVM(d)`` need no other
    change here."""
    out = torch.zeros_like(x)
    for bit in range(d):
        out += torch.bitwise_right_shift(x, bit) & 1
    return out


def inverse_consistency_loss(
    forward_fn,
    inverse_fn,
    state_index: torch.Tensor,
    byte: torch.Tensor,
) -> torch.Tensor:
    """distance(pred_inverse(pred_forward(x, b), b), x)."""
    mid = forward_fn(state_index, byte)
    back = inverse_fn(mid, byte)
    return (back.float() - state_index.float()).abs().mean()


def composition_loss(
    apply_latent_action,
    z: torch.Tensor,
    left_sig: torch.Tensor,
    right_sig: torch.Tensor,
    composed_sig: torch.Tensor,
) -> torch.Tensor:
    """distance(R(g)R(h)z, R(g*h)z) with signature ids [B]."""
    z_left = apply_latent_action(left_sig, z)
    z_both = apply_latent_action(right_sig, z_left)
    z_composed = apply_latent_action(composed_sig, z)
    return (z_both - z_composed).pow(2).sum(dim=-1).mean()


def word_semantic_losses(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """L_signature, L_parity, L_tau_u, L_tau_v, L_q_total, L_commitment."""
    out = {
        "signature": F.cross_entropy(pred["signature"], target["signature"]),
        "parity": F.cross_entropy(pred["parity"], target["parity"]),
        "tau_u": F.cross_entropy(pred["tau_u"], target["tau_u"]),
        "tau_v": F.cross_entropy(pred["tau_v"], target["tau_v"]),
    }
    if "q_total" in pred and "q_total" in target:
        out["q_total"] = F.cross_entropy(pred["q_total"], target["q_total"])
    if "commitment" in pred and "commitment" in target:
        out["commitment"] = F.cross_entropy(pred["commitment"], target["commitment"])
    return out


def weighted_total(
    components: dict[str, torch.Tensor], weights: LossWeights
) -> tuple[torch.Tensor, dict[str, float]]:
    total = None
    logs: dict[str, float] = {}
    for name, value in components.items():
        # Fail loudly on an unknown loss key: a misspelled or missing field
        # (e.g. "tau_ce" instead of "word_ce") would otherwise be silently
        # dropped with weight 0.0, so a task would "train" without a gradient.
        if not hasattr(weights, name):
            raise KeyError(
                f"Unknown loss component {name!r}: not a field of LossWeights. "
                f"Available fields: {[f for f in vars(weights)]}"
            )
        weight = float(getattr(weights, name))
        if weight == 0.0:
            continue
        term = weight * value
        total = term if total is None else total + term
        logs[name] = float(value.detach())
    if total is None:
        total = torch.zeros((), requires_grad=True)
    return total, logs