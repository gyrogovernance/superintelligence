"""
Router kernel.

Loads the three atlas artifacts and provides:
- deterministic stepping by bytes via epistemology
- per-step K4 aperture measurement via phenomenology graph
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
    signed_edge_value,
)


@dataclass(frozen=True)
class Signature:
    state_index: int
    state_hex: str
    a_hex: str
    b_hex: str
    aperture: float


class RouterKernel:
    def __init__(self, atlas_dir: Path):
        self.ontology: NDArray[np.uint32] = np.load(atlas_dir / "ontology.npy", mmap_mode="r")
        self.epistemology: NDArray[np.uint32] = np.load(atlas_dir / "epistemology.npy", mmap_mode="r")

        phen = np.load(atlas_dir / "phenomenology.npz")
        self.k4_edges: NDArray[np.uint8] = phen["k4_edges"]
        self.p_cycle: NDArray[np.float64] = phen["k4_p_cycle"]
        self.archetype_a12: int = int(phen["archetype_a12"])
        self.xform_mask_by_byte: NDArray[np.uint32] = phen["xform_mask_by_byte"]

        archetype_indices = np.where(self.ontology == ARCHETYPE_STATE24)[0]
        if len(archetype_indices) == 0:
            raise ValueError(f"Archetype {ARCHETYPE_STATE24:06x} not found in ontology")
        self.archetype_index = int(archetype_indices[0])
        
        self.state_index = self.archetype_index
        self.last_byte: int = GENE_MIC_S

    def reset(self, state_index: int | None = None) -> None:
        if state_index is None:
            state_index = self.archetype_index
        self.state_index = int(state_index)
        self.last_byte = GENE_MIC_S

    def step_byte(self, byte: int) -> None:
        self.last_byte = int(byte) & 0xFF
        self.state_index = int(self.epistemology[self.state_index, self.last_byte])

    def step(self, payload: bytes) -> Signature:
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
        Per-packet signature: aperture uses the instruction mask implied by the byte.
        
        K4 vertices:
        - V0: archetype A (reference)
        - V1: instruction mask (operation mask)
        - V2: current A AND mask
        - V3: current A
        """
        s = int(self.ontology[self.state_index])
        a, b = unpack_state(s)

        mask24 = int(self.xform_mask_by_byte[int(byte) & 0xFF])
        M = (mask24 >> 12) & 0xFFF  # instruction mask on A

        V0 = self.archetype_a12
        V1 = M
        V2 = a & M
        V3 = a

        y = np.empty(6, dtype=np.float64)
        verts = (V0, V1, V2, V3)
        for i, (u, v) in enumerate(self.k4_edges):
            y[i] = signed_edge_value(verts[int(u)], verts[int(v)])

        denom = float(np.dot(y, y))
        aperture = 0.0 if denom == 0.0 else float(np.dot(self.p_cycle @ y, self.p_cycle @ y) / denom)

        return Signature(
            state_index=self.state_index,
            state_hex=f"{s:06x}",
            a_hex=f"{a:03x}",
            b_hex=f"{b:03x}",
            aperture=aperture,
        )