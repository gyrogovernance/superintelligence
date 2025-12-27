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

    magnitude: signed float (positive or negative).
    confidence: [0..1] typically; not enforced here (policy decision).
    meta: optional JSON-like dict for audit/debug (nonsemantic).
    """
    domain: Domain
    edge_id: EdgeID
    magnitude: float
    confidence: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)

    # Optional: bind the event to a kernel "moment" (state_index and last_byte)
    # so the event log can be replayed deterministically.
    kernel_state_index: Optional[int] = None
    kernel_last_byte: Optional[int] = None

    def signed_value(self) -> float:
        return float(self.magnitude) * float(self.confidence)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "domain": int(self.domain),
            "edge_id": int(self.edge_id),
            "magnitude": float(self.magnitude),
            "confidence": float(self.confidence),
            "meta": dict(self.meta),
            "kernel_state_index": self.kernel_state_index,
            "kernel_last_byte": self.kernel_last_byte,
        }

