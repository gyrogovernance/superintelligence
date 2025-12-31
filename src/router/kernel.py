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