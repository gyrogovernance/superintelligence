"""
External connectors.

This file intentionally does NOT start a web server.
It provides thin adapters to:
- parse inbound JSON-like data into GovernanceEvents
- serialize status/events for external APIs (FastAPI/Flask/CLI later)

Keep this minimal and dependency-free.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.app.events import Domain, EdgeID, GovernanceEvent


def parse_domain(value: Any) -> Optional[Domain]:
    if isinstance(value, int):
        try:
            return Domain(int(value))
        except Exception:
            return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "economy":
            return Domain.ECONOMY
        if v == "employment":
            return Domain.EMPLOYMENT
        if v == "education":
            return Domain.EDUCATION
    return None


def parse_edge_id(value: Any) -> Optional[EdgeID]:
    if isinstance(value, int):
        try:
            return EdgeID(int(value))
        except Exception:
            return None
    if isinstance(value, str):
        v = value.strip().upper()
        try:
            return EdgeID[v]
        except Exception:
            return None
    return None


def event_from_dict(d: Dict[str, Any]) -> GovernanceEvent:
    dom = parse_domain(d.get("domain"))
    edge = parse_edge_id(d.get("edge_id"))
    if dom is None or edge is None:
        raise ValueError(f"Invalid event dict: domain={d.get('domain')}, edge_id={d.get('edge_id')}")

    magnitude = float(d.get("magnitude", 0.0))
    confidence = float(d.get("confidence", 1.0))
    meta = dict(d.get("meta", {}))

    # Optional kernel binding fields
    kernel_step = d.get("kernel_step")
    kernel_state_index = d.get("kernel_state_index")
    kernel_last_byte = d.get("kernel_last_byte")

    return GovernanceEvent(
        domain=dom,
        edge_id=edge,
        magnitude=magnitude,
        confidence=confidence,
        meta=meta,
        kernel_step=int(kernel_step) if kernel_step is not None else None,
        kernel_state_index=int(kernel_state_index) if kernel_state_index is not None else None,
        kernel_last_byte=int(kernel_last_byte) if kernel_last_byte is not None else None,
    )


def event_to_dict(ev: GovernanceEvent) -> Dict[str, Any]:
    return ev.as_dict()

