# src/router.py
"""
Gyroscopic ASI aQPU Kernel - Compact deterministic byte-driven coordination medium.

Maintains:
- state24: the 24-bit GENE_Mac tensor state
- step counter
- last_byte

All transitions go through step_state_by_byte from constants.
Depth-4 observables are measurements over any 4 consecutive steps,
not a special stepping mode.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.api import (
    depth4_intron_sequence32,
    depth4_mask_projection48,
)
from src.constants import (
    GENE_MAC_REST,
    GENE_MIC_S,
    MASK_STATE24,
    archetype_distance,
    byte_cgm_parities,
    byte_family,
    byte_micro_ref,
    byte_to_intron,
    component_density,
    horizon_distance,
    inverse_step_by_byte,
    is_on_horizon,
    single_step_trace,
    step_state_by_byte,
    unpack_state,
)


@dataclass(frozen=True)
class Signature:
    """Kernel state snapshot for audit and shared-moment verification."""

    step: int
    state24: int
    last_byte: int
    state_hex: str
    a_hex: str
    b_hex: str


class Gyroscopic:
    """Deterministic byte router over the GENE_Mac tensor state."""

    __slots__ = ("state24", "last_byte", "step")

    def __init__(self) -> None:
        self.state24: int = GENE_MAC_REST
        self.last_byte: int = GENE_MIC_S
        self.step: int = 0

    def reset(self) -> None:
        """Reset to GENE_Mac rest state."""
        self.state24 = GENE_MAC_REST
        self.last_byte = GENE_MIC_S
        self.step = 0

    # ---- stepping ----

    def step_byte(self, byte: int) -> None:
        """Single forward step."""
        b = int(byte) & 0xFF
        self.state24 = step_state_by_byte(self.state24, b)
        self.last_byte = b
        self.step += 1

    def step_bytes(self, payload: bytes | bytearray) -> None:
        """Bulk forward step."""
        n = len(payload)
        if n == 0:
            return
        s = self.state24
        for b in payload:
            s = step_state_by_byte(s, b)
        self.state24 = s
        self.last_byte = int(payload[-1]) & 0xFF
        self.step += n

    def step_payload(self, payload: bytes) -> Signature:
        """Bulk forward step, returns signature."""
        self.step_bytes(payload)
        return self.signature()

    def step_byte_inverse(self, byte: int) -> None:
        """Single inverse step."""
        self.state24 = inverse_step_by_byte(self.state24, byte)
        self.last_byte = GENE_MIC_S
        self.step = max(0, self.step - 1)

    def step_bytes_inverse(self, payload: bytes | bytearray) -> None:
        """Bulk inverse step."""
        n = len(payload)
        if n == 0:
            return
        s = self.state24
        for b in reversed(payload):
            s = inverse_step_by_byte(s, b)
        self.state24 = s
        self.last_byte = GENE_MIC_S
        self.step = max(0, self.step - n)

    # ---- state observables ----

    @property
    def current_archetype_distance(self) -> int:
        """Hamming distance from current state to GENE_Mac rest."""
        return archetype_distance(self.state24)

    @property
    def current_horizon_distance(self) -> int:
        """Distance to horizon set."""
        a, b = unpack_state(self.state24)
        return horizon_distance(a, b)

    @property
    def current_is_on_horizon(self) -> bool:
        """Whether current state is on the horizon."""
        return is_on_horizon(self.state24)

    @property
    def current_component_densities(self) -> tuple[float, float]:
        """(a_density, b_density) of current state."""
        a, b = unpack_state(self.state24)
        return component_density(a), component_density(b)

    # ---- depth-4 observables ----

    def depth4_mask48(self, b0: int, b1: int, b2: int, b3: int) -> int:
        """48-bit mask projection over 4 consecutive bytes."""
        return depth4_mask_projection48(b0, b1, b2, b3)

    def depth4_introns32(
        self, b0: int, b1: int, b2: int, b3: int
    ) -> int:
        """32-bit intron sequence over 4 consecutive bytes (bijective)."""
        return depth4_intron_sequence32(b0, b1, b2, b3)

    def trace_step(self, byte: int) -> dict[str, int]:
        """Trace the 4 internal CGM stages of processing one byte."""
        return single_step_trace(self.state24, byte)

    # ---- intron decomposition ----

    @staticmethod
    def byte_family(byte: int) -> int:
        """L0 boundary bits (0, 7) -> 2-bit family index."""
        return byte_family(byte)

    @staticmethod
    def byte_micro_ref(byte: int) -> int:
        """Payload bits (1-6) -> 6-bit micro-reference."""
        return byte_micro_ref(byte)

    @staticmethod
    def byte_cgm_parities(byte: int) -> dict[str, int]:
        """CGM stage parities for a byte."""
        return byte_cgm_parities(byte)

    # ---- signature ----

    def signature(self) -> Signature:
        """Current state snapshot."""
        a12, b12 = unpack_state(self.state24)
        s = int(self.state24) & MASK_STATE24
        return Signature(
            step=self.step,
            state24=s,
            last_byte=int(self.last_byte) & 0xFF,
            state_hex=f"{s:06x}",
            a_hex=f"{a12:03x}",
            b_hex=f"{b12:03x}",
        )

    def route_from_archetype(self, payload: bytes) -> Signature:
        """Route from GENE_Mac rest without disturbing current state."""
        saved = (self.state24, self.last_byte, self.step)
        try:
            self.reset()
            self.step_bytes(payload)
            return self.signature()
        finally:
            self.state24, self.last_byte, self.step = saved