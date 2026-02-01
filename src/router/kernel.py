"""
Router kernel.

Loads the three atlas artifacts and provides:
- deterministic stepping by bytes via epistemology
- O(1) current observables (horizon, vertex, phase)
- O(1) peek methods for next-state observables
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
    """
    Router kernel with spectral atlas support.
    
    Provides O(1) access to current and next-state observables:
    - horizon: position on the 256-state holographic boundary
    - vertex: K₄ vertex charge (0-3)
    - phase: position in byte-permutation cycle (0-3 typically)
    """
    
    def __init__(self, atlas_dir: Path):
        self.ontology: NDArray[np.uint32] = np.load(atlas_dir / "ontology.npy", mmap_mode="r")
        self.epistemology: NDArray[np.uint32] = np.load(atlas_dir / "epistemology.npy", mmap_mode="r")

        with np.load(atlas_dir / "phenomenology.npz", allow_pickle=False) as phen:
            # Validate atlas version
            atlas_version = "1.0"
            if "atlas_version" in phen.files:
                atlas_version = str(phen["atlas_version"])
            if not atlas_version.startswith("2."):
                raise ValueError(
                    f"phenomenology.npz version {atlas_version} is incompatible. "
                    "Expected version 2.x. Rebuild atlas with: python -m src.router.atlas"
                )
            
            # Load constants
            self.archetype_a12: int = int(phen["archetype_a12"])
            self.xform_mask_by_byte: NDArray[np.uint32] = phen["xform_mask_by_byte"]
            
            # Load per-state observables (required)
            required = ["state_horizon", "state_vertex", "phase", "next_horizon", "next_vertex", "next_phase"]
            for k in required:
                if k not in phen.files:
                    raise ValueError(
                        f"phenomenology.npz missing '{k}'. Rebuild atlas with: python -m src.router.atlas"
                    )
            
            self.state_horizon: NDArray[np.uint8] = phen["state_horizon"].astype(np.uint8, copy=False)
            self.state_vertex: NDArray[np.uint8] = phen["state_vertex"].astype(np.uint8, copy=False)
            self.phase: NDArray[np.uint8] = phen["phase"].astype(np.uint8, copy=False)
            self.next_horizon: NDArray[np.uint8] = phen["next_horizon"].astype(np.uint8, copy=False)
            self.next_vertex: NDArray[np.uint8] = phen["next_vertex"].astype(np.uint8, copy=False)
            self.next_phase: NDArray[np.uint8] = phen["next_phase"].astype(np.uint8, copy=False)
            
            # Load required helpers
            required_helpers = ["byte_weight", "byte_charge", "gamma_table"]
            for k in required_helpers:
                if k not in phen.files:
                    raise ValueError(
                        f"phenomenology.npz missing '{k}'. Rebuild atlas with: python -m src.router.atlas"
                    )
            
            self.byte_weight: NDArray[np.uint8] = phen["byte_weight"].astype(np.uint8, copy=False)
            self.byte_charge: NDArray[np.uint8] = phen["byte_charge"].astype(np.uint8, copy=False)
            self.gamma_table: NDArray[np.float32] = phen["gamma_table"].astype(np.float32, copy=False)
        
        # Find archetype index
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
        for b in payload:
            self.step_byte(b)
        return self.signature()

    # =========================================================================
    # Current Observables (O(1) via precomputed maps)
    # =========================================================================
    
    @property
    def current_horizon(self) -> int:
        """Current horizon index (0-255)."""
        return int(self.state_horizon[self.state_index])
    
    @property
    def current_vertex(self) -> int:
        """Current K₄ vertex charge (0-3)."""
        return int(self.state_vertex[self.state_index])
    
    @property
    def current_phase(self) -> int:
        """Current phase in the last_byte permutation cycle."""
        b = int(self.last_byte) & 0xFF
        return int(self.phase[self.state_index, b])

    # =========================================================================
    # Peek Methods (O(1) - no stepping)
    # =========================================================================
    
    def peek_next_state_index(self, byte: int) -> int:
        """Peek next state index without stepping."""
        return int(self.epistemology[self.state_index, int(byte) & 0xFF])
    
    def peek_next_horizon(self, byte: int) -> int:
        """Peek next horizon index without stepping."""
        return int(self.next_horizon[self.state_index, int(byte) & 0xFF])
    
    def peek_next_vertex(self, byte: int) -> int:
        """Peek next vertex charge without stepping."""
        return int(self.next_vertex[self.state_index, int(byte) & 0xFF])
    
    def peek_next_phase(self, byte: int) -> int:
        """Peek next phase without stepping."""
        return int(self.next_phase[self.state_index, int(byte) & 0xFF])

    # =========================================================================
    # Signature
    # =========================================================================
    
    def signature(self) -> Signature:
        return self.signature_with_byte(self.last_byte)

    def signature_with_byte(self, byte: int) -> Signature:
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
        current_state = int(self.ontology[self.state_index])
        a_prime, b_prime = unpack_state(current_state)
        
        m = mask12_for_byte(int(byte) & 0xFF)
        
        b_pred = (a_prime ^ LAYER_MASK_12) & LAYER_MASK_12
        a_pred = ((b_prime ^ m) ^ LAYER_MASK_12) & LAYER_MASK_12
        state24_pred = pack_state(a_pred, b_pred)
        
        idx = int(np.searchsorted(self.ontology, state24_pred))
        if idx >= len(self.ontology) or int(self.ontology[idx]) != state24_pred:
            raise ValueError(f"Predecessor state {state24_pred:06x} not found in ontology")
        self.state_index = idx
        
        self.last_byte = GENE_MIC_S
        self.step = max(0, self.step - 1)

    def step_payload_inverse(self, payload: bytes) -> None:
        for b in reversed(payload):
            self.step_byte_inverse(b)