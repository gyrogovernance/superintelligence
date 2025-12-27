"""
Coordinator: Kernel + App ledgers + plugins.

This is the "spine" that a future UI (React, CLI, etc.) will call.

Responsibilities:
- advance kernel state (shared moment) by bytes
- accept governance events from plugins/app
- update domain ledgers deterministically
- expose GGG apertures (ledger-based), plus kernel signature
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from src.router.kernel import RouterKernel

from .events import GovernanceEvent, Domain
from .ledger import DomainLedgers


@dataclass
class CoordinationStatus:
    kernel: Dict[str, Any]
    ledgers: Dict[str, Any]
    apertures: Dict[str, float]


class Coordinator:
    def __init__(self, atlas_dir: Path) -> None:
        self.kernel = RouterKernel(atlas_dir)
        self.ledgers = DomainLedgers()

        # Audit logs (kept simple; you can persist externally)
        self.byte_log: List[int] = []
        self.event_log: List[Dict[str, Any]] = []

    # -------------------------
    # Kernel stepping (shared moment)
    # -------------------------
    def step_byte(self, byte: int) -> None:
        b = int(byte) & 0xFF
        self.kernel.step_byte(b)
        self.byte_log.append(b)

    def step_bytes(self, payload: bytes) -> None:
        for b in payload:
            self.step_byte(b)

    # -------------------------
    # App events (ledger updates)
    # -------------------------
    def apply_event(self, ev: GovernanceEvent, bind_to_kernel_moment: bool = True) -> None:
        """
        Apply governance event to the appropriate domain ledger.
        Optionally bind it to the current kernel moment for replay.
        """
        if bind_to_kernel_moment:
            ev = GovernanceEvent(
                domain=ev.domain,
                edge_id=ev.edge_id,
                magnitude=ev.magnitude,
                confidence=ev.confidence,
                meta=dict(ev.meta),  # Copy dict to preserve audit trail immutability
                kernel_state_index=self.kernel.state_index,
                kernel_last_byte=self.kernel.last_byte,
            )

        self.ledgers.apply_event(ev)

        self.event_log.append(
            {
                "event_index": len(self.event_log),
                "kernel_state_index": ev.kernel_state_index,
                "kernel_last_byte": ev.kernel_last_byte,
                "event": ev.as_dict(),
            }
        )

    # -------------------------
    # Reporting
    # -------------------------
    def get_status(self) -> CoordinationStatus:
        sig = self.kernel.signature()

        kernel_info = {
            "state_index": sig.state_index,
            "state_hex": sig.state_hex,
            "a_hex": sig.a_hex,
            "b_hex": sig.b_hex,
            "last_byte": self.kernel.last_byte,
            "byte_log_len": len(self.byte_log),
            "event_log_len": len(self.event_log),
        }

        apertures = {
            "econ": self.ledgers.aperture(Domain.ECONOMY),
            "emp": self.ledgers.aperture(Domain.EMPLOYMENT),
            "edu": self.ledgers.aperture(Domain.EDUCATION),
        }

        ledgers = {
            "y_econ": self.ledgers.get(Domain.ECONOMY).tolist(),
            "y_emp": self.ledgers.get(Domain.EMPLOYMENT).tolist(),
            "y_edu": self.ledgers.get(Domain.EDUCATION).tolist(),
            "event_count": self.ledgers.event_count,
        }

        return CoordinationStatus(kernel=kernel_info, ledgers=ledgers, apertures=apertures)

    def reset(self) -> None:
        self.kernel.reset()
        self.ledgers = DomainLedgers()
        self.byte_log.clear()
        self.event_log.clear()

