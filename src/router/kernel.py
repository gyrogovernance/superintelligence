"""
Router kernel.

Loads the three atlas artifacts and provides:
- deterministic stepping by bytes via epistemology
- O(1) current observables (horizon, vertex, phase)
- O(1) peek methods for next-state observables
- batch support (B >= 1)
- state signature
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .constants import (
    ARCHETYPE_STATE24,
    GENE_MIC_S,
    LAYER_MASK_12,
    mask12_for_byte,
    pack_state,
    unpack_state,
)


@dataclass(frozen=True)
class Signature:
    step: int
    state_index: int
    state_hex: str
    a_hex: str
    b_hex: str


class RouterKernel:
    """
    Router kernel with spectral atlas support.

    Provides O(1) access to current and next-state observables:
    - horizon: position on the 256-state holographic boundary
    - vertex: K4 vertex charge (0-3)
    - phase: position in byte-permutation cycle (0-3 typically)

    Supports batch inference (B >= 1). All state arrays have shape (B,).
    """

    def __init__(self, atlas_dir: Path, batch_size: int = 1):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.batch_size = batch_size

        self.ontology: NDArray[np.uint32] = np.load(atlas_dir / "ontology.npy", mmap_mode="r")
        self.epistemology: NDArray[np.uint32] = np.load(atlas_dir / "epistemology.npy", mmap_mode="r")

        with np.load(atlas_dir / "phenomenology.npz", allow_pickle=False) as phen:
            atlas_version = "1.0"
            if "atlas_version" in phen.files:
                atlas_version = str(phen["atlas_version"])
            if not atlas_version.startswith("2."):
                raise ValueError(
                    f"phenomenology.npz version {atlas_version} is incompatible. "
                    "Expected version 2.x. Rebuild atlas with: python -m src.router.atlas"
                )

            self.archetype_a12: int = int(phen["archetype_a12"])
            self.xform_mask_by_byte: NDArray[np.uint32] = phen["xform_mask_by_byte"]

            required = [
                "state_horizon", "state_vertex", "phase",
                "next_horizon", "next_vertex", "next_phase",
            ]
            for k in required:
                if k not in phen.files:
                    raise ValueError(
                        f"phenomenology.npz missing '{k}'. "
                        "Rebuild atlas with: python -m src.router.atlas"
                    )

            self.state_horizon: NDArray[np.uint8] = phen["state_horizon"].astype(np.uint8, copy=False)
            self.state_vertex: NDArray[np.uint8] = phen["state_vertex"].astype(np.uint8, copy=False)
            self.phase: NDArray[np.uint8] = phen["phase"].astype(np.uint8, copy=False)
            self.next_horizon: NDArray[np.uint8] = phen["next_horizon"].astype(np.uint8, copy=False)
            self.next_vertex: NDArray[np.uint8] = phen["next_vertex"].astype(np.uint8, copy=False)
            self.next_phase: NDArray[np.uint8] = phen["next_phase"].astype(np.uint8, copy=False)

            required_helpers = ["byte_weight", "byte_charge"]
            for k in required_helpers:
                if k not in phen.files:
                    raise ValueError(
                        f"phenomenology.npz missing '{k}'. "
                        "Rebuild atlas with: python -m src.router.atlas"
                    )

            self.byte_weight: NDArray[np.uint8] = phen["byte_weight"].astype(np.uint8, copy=False)
            self.byte_charge: NDArray[np.uint8] = phen["byte_charge"].astype(np.uint8, copy=False)

        archetype_indices = np.where(self.ontology == ARCHETYPE_STATE24)[0]
        if len(archetype_indices) == 0:
            raise ValueError(f"Archetype {ARCHETYPE_STATE24:06x} not found in ontology")
        self.archetype_index = int(archetype_indices[0])

        self.state_index: NDArray[np.int64] = np.full(batch_size, self.archetype_index, dtype=np.int64)
        self.last_byte: NDArray[np.int64] = np.full(batch_size, GENE_MIC_S, dtype=np.int64)
        self.step: int = 0

    # =========================================================================
    # Batch management
    # =========================================================================

    def resize_batch(self, new_size: int) -> None:
        """Resize batch dimension. Resets all sequences to archetype."""
        if new_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {new_size}")
        self.batch_size = new_size
        self.state_index = np.full(new_size, self.archetype_index, dtype=np.int64)
        self.last_byte = np.full(new_size, GENE_MIC_S, dtype=np.int64)
        self.step = 0

    # =========================================================================
    # Reset
    # =========================================================================

    def reset(self, state_index: int | None = None) -> None:
        """Reset all sequences to archetype (or specified state index)."""
        idx = self.archetype_index if state_index is None else int(state_index)
        self.state_index[:] = idx
        self.last_byte[:] = GENE_MIC_S
        self.step = 0

    # =========================================================================
    # Stepping
    # =========================================================================

    def step_byte(self, byte: int | np.ndarray) -> None:
        """Step by byte(s). Accepts scalar (broadcast) or array of shape (batch_size,)."""
        if isinstance(byte, (int, np.integer)):
            b = int(byte) & 0xFF
            self.last_byte[:] = b
            self.state_index[:] = self.epistemology[self.state_index, b]
        else:
            b = np.asarray(byte, dtype=np.int64) & 0xFF
            if b.shape != (self.batch_size,):
                raise ValueError(
                    f"byte array shape {b.shape} does not match batch_size {self.batch_size}"
                )
            self.last_byte[:] = b
            self.state_index[:] = self.epistemology[self.state_index, b]
        self.step += 1

    def step_payload(self, payload: bytes) -> Signature:
        """Step through a byte payload (broadcast to all sequences)."""
        for b in payload:
            self.step_byte(b)
        return self.signature()

    # =========================================================================
    # Current Observables - return arrays of shape (batch_size,)
    # =========================================================================

    @property
    def current_horizon(self) -> NDArray[np.uint8]:
        """Current horizon indices, shape (batch_size,)."""
        return self.state_horizon[self.state_index]

    @property
    def current_vertex(self) -> NDArray[np.uint8]:
        """Current K4 vertex charges, shape (batch_size,)."""
        return self.state_vertex[self.state_index]

    @property
    def current_phase(self) -> NDArray[np.uint8]:
        """Current phase values, shape (batch_size,)."""
        return self.phase[self.state_index, self.last_byte]

    # =========================================================================
    # Peek Methods - return arrays of shape (batch_size,)
    # =========================================================================

    def peek_next_state_index(self, byte: int | np.ndarray) -> NDArray[np.int64]:
        """Peek next state indices without stepping."""
        if isinstance(byte, (int, np.integer)):
            b = int(byte) & 0xFF
            return self.epistemology[self.state_index, b].astype(np.int64)
        b = np.asarray(byte, dtype=np.int64) & 0xFF
        return self.epistemology[self.state_index, b].astype(np.int64)

    def peek_next_horizon(self, byte: int | np.ndarray) -> NDArray[np.uint8]:
        """Peek next horizon indices without stepping."""
        if isinstance(byte, (int, np.integer)):
            b = int(byte) & 0xFF
            return self.next_horizon[self.state_index, b]
        b = np.asarray(byte, dtype=np.int64) & 0xFF
        return self.next_horizon[self.state_index, b]

    def peek_next_vertex(self, byte: int | np.ndarray) -> NDArray[np.uint8]:
        """Peek next vertex charges without stepping."""
        if isinstance(byte, (int, np.integer)):
            b = int(byte) & 0xFF
            return self.next_vertex[self.state_index, b]
        b = np.asarray(byte, dtype=np.int64) & 0xFF
        return self.next_vertex[self.state_index, b]

    def peek_next_phase(self, byte: int | np.ndarray) -> NDArray[np.uint8]:
        """Peek next phase values without stepping."""
        if isinstance(byte, (int, np.integer)):
            b = int(byte) & 0xFF
            return self.next_phase[self.state_index, b]
        b = np.asarray(byte, dtype=np.int64) & 0xFF
        return self.next_phase[self.state_index, b]

    # =========================================================================
    # Signature (uses sequence 0 for backward compatibility)
    # =========================================================================

    def signature(self, seq: int = 0) -> Signature:
        """Return signature for a specific sequence in the batch."""
        return self.signature_with_byte(int(self.last_byte[seq]), seq=seq)

    def signature_with_byte(self, byte: int, seq: int = 0) -> Signature:
        s = int(self.ontology[self.state_index[seq]])
        a, b = unpack_state(s)
        return Signature(
            step=self.step,
            state_index=int(self.state_index[seq]),
            state_hex=f"{s:06x}",
            a_hex=f"{a:03x}",
            b_hex=f"{b:03x}",
        )

    # =========================================================================
    # Single-sequence utilities
    # =========================================================================

    def route_from_archetype(self, payload: bytes, seq: int = 0) -> Signature:
        """Route payload from archetype without disturbing current state."""
        saved_state = self.state_index[seq]
        saved_byte = self.last_byte[seq]
        saved_step = self.step

        try:
            self.state_index[seq] = self.archetype_index
            self.last_byte[seq] = GENE_MIC_S
            self.step = 0
            for b in payload:
                bv = int(b) & 0xFF
                self.last_byte[seq] = bv
                self.state_index[seq] = int(self.epistemology[self.state_index[seq], bv])
                self.step += 1
            return self.signature(seq=seq)
        finally:
            self.state_index[seq] = saved_state
            self.last_byte[seq] = saved_byte
            self.step = saved_step

    def step_byte_inverse(self, byte: int, seq: int = 0) -> None:
        """Reverse a single byte step for one sequence."""
        current_state = int(self.ontology[self.state_index[seq]])
        a_prime, b_prime = unpack_state(current_state)

        m = mask12_for_byte(int(byte) & 0xFF)

        b_pred = (a_prime ^ LAYER_MASK_12) & LAYER_MASK_12
        a_pred = ((b_prime ^ m) ^ LAYER_MASK_12) & LAYER_MASK_12
        state24_pred = pack_state(a_pred, b_pred)

        idx = int(np.searchsorted(self.ontology, state24_pred))
        if idx >= len(self.ontology) or int(self.ontology[idx]) != state24_pred:
            raise ValueError(f"Predecessor state {state24_pred:06x} not found in ontology")
        self.state_index[seq] = idx

        self.last_byte[seq] = GENE_MIC_S
        self.step = max(0, self.step - 1)

    def step_payload_inverse(self, payload: bytes, seq: int = 0) -> None:
        """Reverse a payload for one sequence."""
        for b in reversed(payload):
            self.step_byte_inverse(b, seq=seq)
