"""
Router Signatures

Defines the immutable routing signature type used by the Router kernel.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingSignature:
    """
    Deterministic routing signature for a governance state.

    The signature is fully determined by the atlas and the current state and
    is replayable from the ledger without any probabilistic components.
    """

    state_index: int
    state_int_hex: str
    stage_profile: tuple[int, int, int, int]  # 0..12 per stage
    loop_defects: tuple[int, int, int]  # 0..48 per loop
    aperture: float
    si: float


