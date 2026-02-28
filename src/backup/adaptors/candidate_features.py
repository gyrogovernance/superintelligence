"""
Candidate-conditioned features for Router-native routing over 256 fanout.

At each step the Router provides 256 fully characterized candidate transitions.
This module builds features psi(state, byte) for scoring each candidate.

Expanded to 20+ dims: vertex charge, horizon distance, phase proxy, etc.
"""

from __future__ import annotations

import numpy as np

from src.router.constants import (
    ARCHETYPE_STATE24,
    LAYER_MASK_12,
    horizon_distance,
    mask12_for_byte,
    step_state_by_byte,
    unpack_state,
    vertex_charge_from_mask,
)


def step_l3_functional(state24: int, byte: int) -> int:
    """Hypothetical next L3 state if we take candidate byte (pure function)."""
    return step_state_by_byte(state24, byte)


def candidate_features(
    state24: int, O: int, E: int, parity: int, byte: int
) -> np.ndarray:
    """
    Features about candidate byte and its next-state effect.
    ~24 dims: weight, intron, deltas, vertex charge, horizon, O/E change.
    """
    ns = step_l3_functional(state24, byte)
    a, b = unpack_state(state24)
    na, nb = unpack_state(ns)

    m = mask12_for_byte(byte) & LAYER_MASK_12
    w = m.bit_count()

    if parity == 0:
        nO, nE = (O ^ m) & 0xFFF, E
    else:
        nO, nE = O, (E ^ m) & 0xFFF

    feat: list[float] = []

    feat.append((w / 6.0) - 1.0)
    intron_val = (byte & 0xFF) ^ 0xAA
    micro = intron_val & 0x3F
    family = (intron_val >> 6) & 0x3
    feat.append((micro / 31.5) - 1.0)
    feat.append((family / 1.5) - 1.0)

    feat.append((a ^ na).bit_count() / 12.0)
    feat.append((b ^ nb).bit_count() / 12.0)
    feat.append(((O ^ nO).bit_count() + (E ^ nE).bit_count()) / 24.0)

    vc = vertex_charge_from_mask(m)
    feat.append(1.0 if (vc & 1) else -1.0)
    feat.append(1.0 if ((vc >> 1) & 1) else -1.0)

    h_before = horizon_distance(a, b)
    h_after = horizon_distance(na, nb)
    feat.append(h_before / 12.0)
    feat.append(h_after / 12.0)
    feat.append((h_after - h_before) / 12.0)

    arch_dist_before = (state24 ^ ARCHETYPE_STATE24).bit_count() / 24.0
    arch_dist_after = (ns ^ ARCHETYPE_STATE24).bit_count() / 24.0
    feat.append(arch_dist_before)
    feat.append(arch_dist_after)
    feat.append(arch_dist_after - arch_dist_before)

    feat.append(1.0 if parity else -1.0)
    feat.append((O.bit_count() + E.bit_count()) / 24.0)
    feat.append((nO.bit_count() + nE.bit_count()) / 24.0)

    return np.array(feat, dtype=np.float32)


def candidate_features_dim() -> int:
    """Number of dimensions in candidate_features output."""
    return 17
