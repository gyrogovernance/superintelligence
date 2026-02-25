"""
src.tools.api
=============

External connectors and public API wiring for the 4-layer FSM + Agent architecture.

External connectors (events, grants):
    - Parse inbound JSON-like data into GovernanceEvents
    - Serialize status/events for external APIs (FastAPI/Flask/CLI later)
    - Does NOT start a web server

FSM + Agent wiring (tool-level glue):
    - Does NOT touch: ontology.npy, epistemology.npy, phenomenology.npz, RouterKernel
    - Wires: Layer1FSM, Layer2FSM, Layer3FSM, BULayer, SharedLayers, Agent, AgentPool
    - Process-global default pool: init_default_pool(), get_default_pool(), etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.app.events import Archive, Domain, EdgeID, GovernanceEvent, Grant, Shell

from .agents import Agent, AgentPool, ByteObservation, SharedLayers
from .layers import (
    BULayer,
    FourFSMs,
    Layer1FSM,
    Layer2FSM,
    Layer3FSM,
    create_default_four_fsms,
)


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


# ---------------------------------------------------------------------
# FSM + Agent: process-global default pool
# ---------------------------------------------------------------------

_DEFAULT_POOL: Optional[AgentPool] = None
_DEFAULT_SHARED: Optional[SharedLayers] = None
_DEFAULT_FSMS: Optional[FourFSMs] = None


def init_default_pool(
    l3_path: Path,
    build_l3_if_missing: bool = False,
) -> AgentPool:
    """
    Initialize the process-global AgentPool.

    Args:
        l3_path: Path to the Layer 3 FSM file (memmapped).
        build_l3_if_missing: If True, build L3 FSM when file does not exist.
                             WARNING: ~16.8 GB, takes time.

    Returns:
        The initialised AgentPool.

    L1 and L2 FSMs are built in RAM. L3 is memmapped from l3_path.
    """
    global _DEFAULT_POOL, _DEFAULT_SHARED, _DEFAULT_FSMS

    fsms: FourFSMs = create_default_four_fsms(
        l3_path=l3_path,
        build_l3_if_missing=build_l3_if_missing,
    )
    _DEFAULT_FSMS = fsms
    _DEFAULT_SHARED = SharedLayers.from_four_fsms(fsms)
    _DEFAULT_POOL = AgentPool(shared_layers=_DEFAULT_SHARED)

    return _DEFAULT_POOL


def get_default_pool() -> AgentPool:
    """
    Return the process-global AgentPool.

    Raises:
        RuntimeError if init_default_pool() has not been called yet.
    """
    if _DEFAULT_POOL is None:
        raise RuntimeError(
            "Default AgentPool is not initialised. "
            "Call init_default_pool(l3_path, ...) first."
        )
    return _DEFAULT_POOL


def new_agent(name: str) -> Agent:
    """Create a new agent in the default pool."""
    pool = get_default_pool()
    return pool.add_agent(name)


def ingest_bytes_for_agent(
    agent_name: str,
    payload: bytes | List[int],
    meta: Optional[Dict[str, Any]] = None,
) -> List[ByteObservation]:
    """
    Ingest a byte sequence for a specific agent in the default pool.

    Does NOT step all agents; only the named agent. For all agents:
        pool = get_default_pool()
        pool.ingest_bytes(payload)
    """
    pool = get_default_pool()
    agent = pool.get_agent(agent_name)
    obs_list: List[ByteObservation] = []
    for b in payload:
        obs_list.append(
            agent.ingest_byte(b, meta=meta if not obs_list else None)
        )
    return obs_list


def ingest_bytes_all_agents(
    payload: bytes | List[int],
) -> List[Dict[str, ByteObservation]]:
    """Ingest a byte sequence for all agents in the default pool."""
    pool = get_default_pool()
    return pool.ingest_bytes(payload)


def reset_all_agents() -> None:
    """Reset all agents in the default pool (clear logs + registers)."""
    pool = get_default_pool()
    pool.reset()


def api_summary() -> Dict[str, Any]:
    """High-level summary of the default pool and FSMs for debugging/UI."""
    pool = get_default_pool()
    fsms = _DEFAULT_FSMS

    fsm_summary: Dict[str, Any] = {}
    if fsms is not None:
        fsm_summary = {
            "L1": {
                "states": fsms.l1.spec.num_states,
                "params": fsms.l1.nparams,
                "bytes": fsms.l1.nbytes,
            },
            "L2": {
                "states": fsms.l2.spec.num_states,
                "params": fsms.l2.nparams,
                "bytes": fsms.l2.nbytes,
            },
            "L3": {
                "states": fsms.l3.spec.num_states,
                "params": fsms.l3.nparams,
                "bytes": fsms.l3.nbytes,
            },
        }

    return {
        "pool": pool.summary(),
        "fsms": fsm_summary,
    }


__all__ = [
    "Agent",
    "AgentPool",
    "ByteObservation",
    "SharedLayers",
    "init_default_pool",
    "get_default_pool",
    "new_agent",
    "ingest_bytes_for_agent",
    "ingest_bytes_all_agents",
    "reset_all_agents",
    "api_summary",
]

