"""Thin adapter over the hQVM kernel.

All algebra comes from src.api / src.constants / src.family / src.sdk.
No transition formulas are duplicated here; this module only arranges kernel
calls into the array-shaped interface the package needs, and hosts the
signature id packing (the pack/unpack of a 13-bit OmegaSignature12).
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import Iterable

import numpy as np

from src import api, constants

STABILIZER_TYPES = ("equality", "complement", "bulk")


# ---------------------------------------------------------------------------
# Signature id packing (13-bit OmegaSignature12)
# ---------------------------------------------------------------------------


def signature_id(parity: int, tau_u6: int, tau_v6: int) -> int:
    """Pack (parity, tau_u6, tau_v6) -> 13-bit signature id."""
    return ((parity & 1) << 12) | ((tau_u6 & 63) << 6) | (tau_v6 & 63)


def signature_from_id(sig_id: int) -> tuple[int, int, int]:
    """Unpack a 13-bit signature id -> (parity, tau_u6, tau_v6)."""
    return (sig_id >> 12) & 1, (sig_id >> 6) & 63, sig_id & 63


def state_index(state24: int) -> int:
    """Canonical index of an Omega state: (u6 << 6) | v6."""
    omega = api.state24_to_omega12(state24)
    return (omega.u6 << 6) | omega.v6


def state24_from_index(index: int) -> int:
    omega = api.OmegaState12(u6=(index >> 6) & 63, v6=index & 63)
    return api.omega12_to_state24(omega)


def enumerate_state24_by_index() -> np.ndarray:
    """state24 values ordered by canonical state index."""
    out = np.empty(4096, dtype=np.uint32)
    for state24 in api.OMEGA_STATES_4096:
        omega = api.state24_to_omega12(state24)
        out[(omega.u6 << 6) | omega.v6] = state24
    return out


def step_index(index: int, byte: int) -> int:
    """Step via the kernel and return the canonical destination index."""
    return state_index(constants.step_state_by_byte(state24_from_index(index), byte))


def inverse_index(index: int, byte: int) -> int:
    return state_index(constants.inverse_step_by_byte(state24_from_index(index), byte))


def apply_k4_index(index: int, gate: str) -> int:
    omega = api.apply_omega_gate(api.OmegaState12(u6=(index >> 6) & 63, v6=index & 63), gate)
    return (omega.u6 << 6) | omega.v6


def apply_signature_index(index: int, sig_id: int) -> int:
    parity, tau_u6, tau_v6 = sig_id_parts(sig_id)
    sig = api.OmegaSignature12(parity, tau_u6, tau_v6)
    omega = api.apply_omega_signature(
        api.OmegaState12(u6=(index >> 6) & 63, v6=index & 63), sig
    )
    return (omega.u6 << 6) | omega.v6


def sig_id_parts(sig_id: int) -> tuple[int, int, int]:
    return signature_from_id(sig_id)


def signature_inverse_id(sig_id: int) -> int:
    parity, tau_u6, tau_v6 = sig_id_parts(sig_id)
    if parity == 0:
        return signature_id(0, tau_u6, tau_v6)
    return signature_id(1, tau_v6, tau_u6)


def enumerate_signature_ids() -> np.ndarray:
    return np.arange(8192, dtype=np.uint16)


@lru_cache(maxsize=2)
def k4_action_arrays() -> tuple[np.ndarray, np.ndarray]:
    """[4, 4096] action and fixed-flag arrays for gates (id, S, C, F)."""
    action = np.empty((4, 4096), dtype=np.uint16)
    fixed = np.zeros((4, 4096), dtype=np.uint8)
    for gate_i, gate in enumerate(("id", "S", "C", "F")):
        for index in range(4096):
            dest = apply_k4_index(index, gate)
            action[gate_i, index] = dest
            fixed[gate_i, index] = dest == index
    return action, fixed


def optical_fractions(shell_chi: int) -> tuple[Fraction, Fraction, Fraction]:
    return (
        Fraction(shell_chi, 6),
        Fraction(6 - shell_chi, 6),
        Fraction(2 * shell_chi - 6, 6),
    )


def popcount6(x: np.ndarray) -> np.ndarray:
    """Popcount over the 6 low bits of an integer array (int64 out)."""
    out = np.zeros(np.shape(x), dtype=np.int64)
    for bit in range(6):
        out += (x >> bit) & 1
    return out


def q_word6_for_indices(items: Iterable[int]) -> int:
    return api.q_word6_for_items(int(b) for b in items)


def word_signature_id(word: Iterable[int]) -> int:
    """13-bit packed OmegaSignature12 for a byte word (list of ints)."""
    sig = api.omega_word_signature([int(b) for b in word])
    return (sig.parity << 12) | (sig.tau_u6 << 6) | sig.tau_v6