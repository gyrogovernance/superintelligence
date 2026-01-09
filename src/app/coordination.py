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

from .events import GovernanceEvent, Domain, EdgeID
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
    def step_byte(self, byte: int, emit_system_event: bool = True) -> None:
        """
        Step the kernel by one byte.
        
        Per GGG hierarchy: Kernel = Economy domain (structural substrate).
        If emit_system_event is True, emits a small Economy domain event
        representing the structural change.
        """
        b = int(byte) & 0xFF
        self.kernel.step_byte(b)
        self.byte_log.append(b)
        
        # Emit Economy domain event for kernel structural activity
        if emit_system_event:
            # Small magnitude representing structural substrate evolution
            # Using GOV_INFO edge as primary structural coupling
            system_event = GovernanceEvent(
                domain=Domain.ECONOMY,
                edge_id=EdgeID.GOV_INFO,
                magnitude=0.01,  # Small structural change per byte
                confidence=1.0,
                meta={"type": "kernel_step", "byte": b},
            )
            self.apply_event(system_event, bind_to_kernel_moment=True)

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
                kernel_step=self.kernel.step,
                kernel_state_index=self.kernel.state_index,
                kernel_last_byte=self.kernel.last_byte,
            )

        self.ledgers.apply_event(ev)

        self.event_log.append(
            {
                "event_index": len(self.event_log),
                "kernel_step": ev.kernel_step,
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
            "step": sig.step,
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

    def derive_domain_counts(self) -> Dict[str, int]:
        """
        Derive domain_counts from the event log.
        
        Per GGG hierarchy:
        - Economy = Kernel (structural substrate)
        - Employment = Gyroscope (active work/principles)
        - Education = THM (measurements/displacements)
        
        Returns dict with keys: "economy", "employment", "education"
        """
        counts = {
            "economy": 0,
            "employment": 0,
            "education": 0,
        }
        
        for log_entry in self.event_log:
            event_dict = log_entry.get("event", {})
            domain_int = event_dict.get("domain")
            if domain_int is not None:
                domain = Domain(domain_int)
                if domain == Domain.ECONOMY:
                    counts["economy"] += 1
                elif domain == Domain.EMPLOYMENT:
                    counts["employment"] += 1
                elif domain == Domain.EDUCATION:
                    counts["education"] += 1
        
        return counts

