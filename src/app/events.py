"""
App-layer event types.

These are nonsemantic structural events that update per-domain K4 edge ledgers.

Key design choices:
- Edges are indexed in the canonical K₄ edge order (see System_Architecture.md Section 9.2):
  (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
  where vertices are (Gov, Info, Infer, Intel) by convention of CGM/GGG.
- Events do NOT change kernel physics; they are ordered by kernel "shared moment".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional


# Fixed-point micro-unit constant (1.0 = 1,000,000 micro-units)
MICRO = 1_000_000


class Domain(IntEnum):
    ECONOMY = 0
    EMPLOYMENT = 1
    EDUCATION = 2
    # Ecology is derived, not ledger-updated


class Vertex(IntEnum):
    GOV = 0
    INFO = 1
    INFER = 2
    INTEL = 3


class EdgeID(IntEnum):
    # Must match src.router.constants.K4.edges ordering
    GOV_INFO = 0   # (0,1)
    GOV_INFER = 1  # (0,2)
    GOV_INTEL = 2  # (0,3)
    INFO_INFER = 3 # (1,2)
    INFO_INTEL = 4 # (1,3)
    INFER_INTEL = 5# (2,3)


@dataclass(frozen=True)
class GovernanceEvent:
    """
    A single ledger update.

    magnitude_micro: signed integer (actual magnitude × MICRO).
    confidence_micro: integer from 0 to MICRO (0.0 to 1.0, typically MICRO for full confidence).
    meta: optional JSON-like dict for audit/debug (nonsemantic).
    """
    domain: Domain
    edge_id: EdgeID
    magnitude_micro: int
    confidence_micro: int = MICRO
    meta: Dict[str, Any] = field(default_factory=dict)

    # Optional: bind the event to a kernel "moment" (step, state_index, last_byte)
    # so the event log can be replayed deterministically.
    kernel_step: Optional[int] = None
    kernel_state_index: Optional[int] = None
    kernel_last_byte: Optional[int] = None

    def signed_value_micro(self) -> int:
        """Return signed value in micro-units: (magnitude_micro * confidence_micro) // MICRO"""
        return (self.magnitude_micro * self.confidence_micro) // MICRO

    def as_dict(self) -> Dict[str, Any]:
        return {
            "domain": int(self.domain),
            "edge_id": int(self.edge_id),
            "magnitude_micro": int(self.magnitude_micro),
            "confidence_micro": int(self.confidence_micro),
            "meta": dict(self.meta),
            "kernel_step": self.kernel_step,
            "kernel_state_index": self.kernel_state_index,
            "kernel_last_byte": self.kernel_last_byte,
        }


@dataclass(frozen=True)
class Grant:
    """
    Fiat substrate grant: MU allocation to an identity in a Shell.
    
    Fields:
    - identity: human-readable label (external identifier)
    - identity_id: SHA-256 hex (64 chars) - collision-resistant key
    - anchor: Router state_hex (6 chars) - structural coordinate
    - mu_allocated: MU allocated to this identity in a Shell
    """
    identity: str
    identity_id: str
    anchor: str
    mu_allocated: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "identity": str(self.identity),
            "identity_id": str(self.identity_id),
            "anchor": str(self.anchor),
            "mu_allocated": int(self.mu_allocated),
        }


@dataclass(frozen=True)
class Shell:
    """
    Fiat substrate Shell: time-bounded capacity window committed by the kernel.
    
    Fields:
    - header: contextual label (e.g. b"ecology:year:2026" as str)
    - seal: kernel state_hex of (header || sorted receipts)
    - total_capacity_MU: MU capacity available in this Shell (from physics)
    - used_capacity_MU: sum of all MU allocations in Grants
    - free_capacity_MU: remaining MU capacity (total - used)
    """
    header: str
    seal: str
    total_capacity_MU: int
    used_capacity_MU: int
    free_capacity_MU: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "header": str(self.header),
            "seal": str(self.seal),
            "total_capacity_MU": int(self.total_capacity_MU),
            "used_capacity_MU": int(self.used_capacity_MU),
            "free_capacity_MU": int(self.free_capacity_MU),
        }


@dataclass(frozen=True)
class Archive:
    """
    Fiat substrate Archive: accumulated capacity allocations across Shells.
    
    Fields:
    - per_identity_MU: total MU allocated per identity across all Shells
    - total_capacity_MU: sum of capacities in all Shells
    - used_capacity_MU: sum of used MU across all Shells
    - free_capacity_MU: remaining MU across all Shells
    """
    per_identity_MU: Dict[str, int]
    total_capacity_MU: int
    used_capacity_MU: int
    free_capacity_MU: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "per_identity_MU": dict(self.per_identity_MU),
            "total_capacity_MU": int(self.total_capacity_MU),
            "used_capacity_MU": int(self.used_capacity_MU),
            "free_capacity_MU": int(self.free_capacity_MU),
        }