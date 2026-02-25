"""
src.tools.agents
================

Agent framework over the 4-layer FSM memory substrate in src.tools.layers.

This DOES NOT use:
    - ontology.npy
    - epistemology.npy
    - phenomenology.npz
    - RouterKernel

Instead, it uses:
    - Layer1FSM  (8-bit FSM, 256×256 transitions)
    - Layer2FSM  (16-bit FSM, 65,536×256 transitions)
    - Layer3FSM  (24-bit FSM, 16,777,216×256 transitions)
    - BULayer    (closure / holography over byte sequences + 24-bit state)

All three FSM layers are real tables of next-state indices (parameters),
and BU is a small, purely functional layer.

Each Agent:
    - owns its own LayerRegisters + BUState (current states),
    - shares the FSM tables with all other agents,
    - maintains a byte_log and an observation_log.

AgentPool:
    - creates and manages multiple agents over the same shared FSMs,
    - steps all agents through a byte or byte sequence.

This is the “most important” piece: an ASI agent’s memory lives in these
4 layers rather than in a Transformer or in the legacy atlas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .layers import (
    BULayer,
    BUState,
    FourFSMs,
    Layer1FSM,
    Layer2FSM,
    Layer3FSM,
    LayerRegisters,
    create_default_four_fsms,
)


# =============================================================================
# Shared FSMs (tables) — created once, used by all agents
# =============================================================================


@dataclass
class SharedLayers:
    """
    Shared FSM tables and BU operator.

    All agents in a process should share the same SharedLayers instance, so that:
      - L1/L2 FSM tables live once in RAM
      - L3 FSM table (memmap) is shared
    """
    l1: Layer1FSM
    l2: Layer2FSM
    l3: Layer3FSM
    bu: BULayer

    @classmethod
    def from_four_fsms(cls, fsms: FourFSMs) -> "SharedLayers":
        return cls(l1=fsms.l1, l2=fsms.l2, l3=fsms.l3, bu=fsms.bu)


# =============================================================================
# Observation record
# =============================================================================


@dataclass
class ByteObservation:
    """
    Single-step observation of an Agent over all 4 layers.

    Stored per byte ingested so that we can later analyze trajectories.
    """
    step: int
    byte: int

    # Raw layer states after ingesting this byte
    l1_state8: int
    l2_state16: int
    l3_state24: int

    # BU holographic commitments
    bu_O: int
    bu_E: int
    bu_parity: int
    bu_length: int

    # Optional user metadata
    meta: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Agent
# =============================================================================


@dataclass
class Agent:
    """
    A single ASI agent with 4-layer structural memory.

    Memory = (L1 FSM, L2 FSM, L3 FSM, BU) + per-agent registers.

    Tables (parameters) are shared via SharedLayers; state registers are per-agent.
    """
    name: str
    layers: SharedLayers
    regs: LayerRegisters = field(default_factory=LayerRegisters)

    # Logs
    byte_log: List[int] = field(default_factory=list)
    observations: List[ByteObservation] = field(default_factory=list)

    def reset(self) -> None:
        """Reset registers and clear logs."""
        self.regs.reset()
        self.byte_log.clear()
        self.observations.clear()

    # ------------------------------
    # Ingestion
    # ------------------------------

    def ingest_byte(self, byte_val: int, meta: Optional[Dict[str, Any]] = None) -> ByteObservation:
        """
        Ingest a single byte into the agent's memory.

        This:
          - updates L1/L2/L3 state registers according to their FSM transitions
          - updates BUState holographic commitments
          - appends to byte_log
          - records an observation snapshot
        """
        b = int(byte_val) & 0xFF
        self.byte_log.append(b)

        # L1/L2/L3 FSM updates
        self.regs.l1_state8 = self.layers.l1.next_state(self.regs.l1_state8, b)
        self.regs.l2_state16 = self.layers.l2.next_state(self.regs.l2_state16, b)
        self.regs.l3_state24 = self.layers.l3.next_state(self.regs.l3_state24, b)

        # BU holography (egress update)
        bu_state: BUState = self.regs.bu
        self.layers.bu.egress_update(bu_state, b)

        obs = ByteObservation(
            step=len(self.byte_log) - 1,
            byte=b,
            l1_state8=self.regs.l1_state8,
            l2_state16=self.regs.l2_state16,
            l3_state24=self.regs.l3_state24,
            bu_O=bu_state.O,
            bu_E=bu_state.E,
            bu_parity=bu_state.parity,
            bu_length=bu_state.length,
            meta=meta or {},
        )
        self.observations.append(obs)
        return obs

    def ingest_bytes(self, payload: bytes | List[int]) -> List[ByteObservation]:
        """Ingest a full byte sequence, returning the list of observations."""
        return [self.ingest_byte(b) for b in payload]

    # ------------------------------
    # Memory queries
    # ------------------------------

    @property
    def steps(self) -> int:
        return len(self.byte_log)

    def current_states(self) -> Dict[str, int]:
        """Return a snapshot of the current FSM states."""
        return {
            "l1_state8": self.regs.l1_state8,
            "l2_state16": self.regs.l2_state16,
            "l3_state24": self.regs.l3_state24,
            "bu_O": self.regs.bu.O,
            "bu_E": self.regs.bu.E,
            "bu_parity": self.regs.bu.parity,
            "bu_length": self.regs.bu.length,
        }

    def holographic_commitment(self) -> tuple[int, int, int]:
        """
        Return (O, E, parity) for the entire byte_log.

        Uses BU's commitment function, which matches BUState up to
        constant-time differences.
        """
        return self.layers.bu.commitment(self.byte_log)

    def path_equivalent_to(self, other: "Agent") -> bool:
        """
        Are this agent's history and the other's equivalent under P8?

        True iff they share (O,E,parity), i.e. same effect from any start.
        """
        return self.holographic_commitment() == other.holographic_commitment()

    # ------------------------------
    # Simple stats
    # ------------------------------

    def bytes_seen(self) -> int:
        return len(self.byte_log)

    def first_n_bytes(self, n: int = 16) -> List[int]:
        return self.byte_log[:n]

    def last_n_bytes(self, n: int = 16) -> List[int]:
        return self.byte_log[-n:]

    def observation_summary(self) -> Dict[str, Any]:
        """
        Lightweight structural summary, useful for debugging or UI.
        """
        return {
            "name": self.name,
            "steps": self.steps,
            "current_states": self.current_states(),
        }


# =============================================================================
# AgentPool
# =============================================================================


class AgentPool:
    """
    A pool of Agents sharing the same FSM tables.

    Typical usage:
        - create_shared = create_default_four_fsms(...)
        - shared = SharedLayers.from_four_fsms(create_shared)
        - pool = AgentPool(shared)
        - pool.add_agent("alpha")
        - pool.add_agent("beta")
        - pool.ingest_bytes(b"...")  # both agents see the same bytes
    """

    def __init__(self, shared_layers: SharedLayers) -> None:
        self.layers = shared_layers
        self.agents: Dict[str, Agent] = {}
        self.global_step: int = 0

    def add_agent(self, name: str) -> Agent:
        if name in self.agents:
            raise ValueError(f"Agent '{name}' already exists")
        agent = Agent(name=name, layers=self.layers)
        self.agents[name] = agent
        return agent

    def get_agent(self, name: str) -> Agent:
        return self.agents[name]

    # ------------------------------
    # Global stepping
    # ------------------------------

    def ingest_byte(self, byte_val: int) -> Dict[str, ByteObservation]:
        """
        Feed one byte to all agents in the pool.

        Returns:
            dict[name -> ByteObservation]
        """
        results: Dict[str, ByteObservation] = {}
        for name, agent in self.agents.items():
            results[name] = agent.ingest_byte(byte_val)
        self.global_step += 1
        return results

    def ingest_bytes(self, payload: bytes | List[int]) -> List[Dict[str, ByteObservation]]:
        """
        Feed a sequence of bytes to all agents.

        Returns:
            list of per-step dict[name -> ByteObservation]
        """
        return [self.ingest_byte(b) for b in payload]

    # ------------------------------
    # Pool-level queries
    # ------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "num_agents": len(self.agents),
            "agents": {
                name: agent.observation_summary()
                for name, agent in self.agents.items()
            },
        }

    def reset(self) -> None:
        for agent in self.agents.values():
            agent.reset()
        self.global_step = 0


# =============================================================================
# Convenience factory
# =============================================================================


def create_default_pool(l3_path: Path, build_l3_if_missing: bool = False) -> AgentPool:
    """
    Convenience to build all FSM tables (L1,L2,L3) once, then create an AgentPool.

    L3 (~16.8 GB) will be loaded from `l3_path` if it exists, or built as a memmap
    if `build_l3_if_missing=True`.
    """
    fsms: FourFSMs = create_default_four_fsms(l3_path=l3_path, build_l3_if_missing=build_l3_if_missing)
    shared = SharedLayers.from_four_fsms(fsms)
    return AgentPool(shared_layers=shared)


__all__ = [
    "Agent",
    "AgentPool",
    "ByteObservation",
    "SharedLayers",
    "create_default_pool",
]