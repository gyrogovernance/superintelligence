"""
External connectors.

This file intentionally does NOT start a web server.
It provides thin adapters to:
- parse inbound JSON-like data into GovernanceEvents
- serialize status/events for external APIs (FastAPI/Flask/CLI later)

Keep this minimal and dependency-free.
"""

from __future__ import annotations

from typing import Any

from src.app.events import Archive, Domain, EdgeID, GovernanceEvent, Grant, Shell


def parse_domain(value: Any) -> Domain | None:
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


def parse_edge_id(value: Any) -> EdgeID | None:
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


def event_from_dict(d: dict[str, Any]) -> GovernanceEvent:
    from src.app.events import MICRO

    dom = parse_domain(d.get("domain"))
    edge = parse_edge_id(d.get("edge_id"))
    if dom is None or edge is None:
        raise ValueError(f"Invalid event dict: domain={d.get('domain')}, edge_id={d.get('edge_id')}")

    # Support both old format (magnitude/confidence as floats) and new format (magnitude_micro/confidence_micro as ints)
    if "magnitude_micro" in d:
        magnitude_micro = int(d.get("magnitude_micro", 0))
        confidence_micro = int(d.get("confidence_micro", MICRO))
    else:
        # Legacy format: convert float to micro-units
        magnitude = float(d.get("magnitude", 0.0))
        confidence = float(d.get("confidence", 1.0))
        magnitude_micro = int(round(magnitude * MICRO))
        confidence_micro = int(round(confidence * MICRO))

    meta = dict(d.get("meta", {}))

    # Optional kernel binding fields
    kernel_step = d.get("kernel_step")
    kernel_state_index = d.get("kernel_state_index")
    kernel_last_byte = d.get("kernel_last_byte")

    return GovernanceEvent(
        domain=dom,
        edge_id=edge,
        magnitude_micro=magnitude_micro,
        confidence_micro=confidence_micro,
        meta=meta,
        kernel_step=int(kernel_step) if kernel_step is not None else None,
        kernel_state_index=int(kernel_state_index) if kernel_state_index is not None else None,
        kernel_last_byte=int(kernel_last_byte) if kernel_last_byte is not None else None,
    )


def event_to_dict(ev: GovernanceEvent) -> dict[str, Any]:
    return ev.as_dict()


def grant_from_dict(d: dict[str, Any]) -> Grant:
    """
    Parse a Grant from a dictionary.
    
    Required keys: identity, identity_id, anchor, mu_allocated
    """
    identity = str(d.get("identity", ""))
    identity_id = str(d.get("identity_id", ""))
    anchor = str(d.get("anchor", ""))
    mu_allocated = int(d.get("mu_allocated", 0))

    if not identity or not identity_id or not anchor:
        raise ValueError("Invalid grant dict: missing identity, identity_id, or anchor")

    if len(identity_id) != 64:
        raise ValueError(f"identity_id must be 64 hex chars (SHA-256), got {len(identity_id)}")

    if len(anchor) != 6:
        raise ValueError(f"anchor must be 6 hex chars (24-bit state_hex), got {len(anchor)}")

    try:
        int(identity_id, 16)  # Validate hex
    except ValueError:
        raise ValueError(f"identity_id must be valid hex, got {identity_id}")

    try:
        int(anchor, 16)  # Validate hex
    except ValueError:
        raise ValueError(f"anchor must be valid hex, got {anchor}")

    return Grant(identity=identity, identity_id=identity_id, anchor=anchor, mu_allocated=mu_allocated)


def grant_to_dict(g: Grant) -> dict[str, Any]:
    """Serialize a Grant to a dictionary."""
    return g.as_dict()


def shell_to_dict(shell: Shell) -> dict[str, Any]:
    """Serialize a Shell to a dictionary."""
    return shell.as_dict()


def archive_to_dict(archive: Archive) -> dict[str, Any]:
    """Serialize an Archive to a dictionary."""
    return archive.as_dict()

