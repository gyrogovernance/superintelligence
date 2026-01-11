"""
Router kernel.

Loads the three atlas artifacts and provides:
- deterministic stepping by bytes via epistemology
- state signature (step, state_index, state_hex, a_hex, b_hex)
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
    pack_state,
    unpack_state,
    mask12_for_byte,
)


@dataclass(frozen=True)
class Signature:
    step: int
    state_index: int
    state_hex: str
    a_hex: str
    b_hex: str


class RouterKernel:
    def __init__(self, atlas_dir: Path):
        self.ontology: NDArray[np.uint32] = np.load(atlas_dir / "ontology.npy", mmap_mode="r")
        self.epistemology: NDArray[np.uint32] = np.load(atlas_dir / "epistemology.npy", mmap_mode="r")

        phen = np.load(atlas_dir / "phenomenology.npz")
        self.archetype_a12: int = int(phen["archetype_a12"])
        self.xform_mask_by_byte: NDArray[np.uint32] = phen["xform_mask_by_byte"]

        archetype_indices = np.where(self.ontology == ARCHETYPE_STATE24)[0]
        if len(archetype_indices) == 0:
            raise ValueError(f"Archetype {ARCHETYPE_STATE24:06x} not found in ontology")
        self.archetype_index = int(archetype_indices[0])
        
        self.state_index = self.archetype_index
        self.last_byte: int = GENE_MIC_S
        self.step: int = 0

    def reset(self, state_index: int | None = None) -> None:
        if state_index is None:
            state_index = self.archetype_index
        self.state_index = int(state_index)
        self.last_byte = GENE_MIC_S
        self.step = 0

    def step_byte(self, byte: int) -> None:
        self.last_byte = int(byte) & 0xFF
        self.state_index = int(self.epistemology[self.state_index, self.last_byte])
        self.step += 1

    def step_payload(self, payload: bytes) -> Signature:
        """
        Step through a sequence of bytes and return the final signature.
        """
        for b in payload:
            self.step_byte(b)
        return self.signature()

    def signature(self) -> Signature:
        """
        Signature using last byte (defaults to GENE_MIC_S = 0xAA for neutral baseline).
        """
        return self.signature_with_byte(self.last_byte)

    def signature_with_byte(self, byte: int) -> Signature:
        """
        Get kernel signature for current state.
        The byte parameter is accepted for API compatibility but not used.
        """
        s = int(self.ontology[self.state_index])
        a, b = unpack_state(s)

        return Signature(
            step=self.step,
            state_index=self.state_index,
            state_hex=f"{s:06x}",
            a_hex=f"{a:03x}",
            b_hex=f"{b:03x}",
        )

    def route_from_archetype(self, payload: bytes) -> Signature:
        """
        Temporarily reset to archetype, step through payload, and return final signature.
        
        This method temporarily modifies kernel state (resets, steps payload), then
        restores the original state. The kernel's current state is unchanged after
        this method returns.
        """
        saved_state_index = self.state_index
        saved_last_byte = self.last_byte
        saved_step = self.step
        
        try:
            self.reset()
            for b in payload:
                self.step_byte(b)
            return self.signature()
        finally:
            self.state_index = saved_state_index
            self.last_byte = saved_last_byte
            self.step = saved_step

    def step_byte_inverse(self, byte: int) -> None:
        """
        Inverse step: compute predecessor state from current state and byte.
        
        Given current state (A', B') and byte b:
        - m = mask12_for_byte(b)
        - predecessor: B = A' ^ 0xFFF, A = (B' ^ m) ^ 0xFFF
        - pack predecessor state24
        - find its ontology index with np.searchsorted
        - set state_index to that index
        - decrement step
        
        This implements BU-Ingress in discrete form.
        """
        current_state = int(self.ontology[self.state_index])
        a_prime, b_prime = unpack_state(current_state)
        
        m = mask12_for_byte(int(byte) & 0xFF)
        
        # Compute predecessor state
        b_pred = (a_prime ^ LAYER_MASK_12) & LAYER_MASK_12
        a_pred = ((b_prime ^ m) ^ LAYER_MASK_12) & LAYER_MASK_12
        state24_pred = pack_state(a_pred, b_pred)
        
        # Find predecessor state index in ontology
        idx = int(np.searchsorted(self.ontology, state24_pred))
        if idx >= len(self.ontology) or int(self.ontology[idx]) != state24_pred:
            raise ValueError(f"Predecessor state {state24_pred:06x} not found in ontology")
        self.state_index = idx
        
        self.last_byte = GENE_MIC_S
        self.step = max(0, self.step - 1)

    def step_payload_inverse(self, payload: bytes) -> None:
        """
        Apply inverse steps for payload bytes in reverse order.
        
        This enables "audit rollback" and "undo last operation".
        """
        for b in reversed(payload):
            self.step_byte_inverse(b)