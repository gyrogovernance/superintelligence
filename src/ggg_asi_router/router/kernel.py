"""
GGG ASI Alignment Router Kernel

Deterministic routing kernel implementing the CGM/THM/GGG coordination interface.

The kernel operates on a finite, precomputed atlas consisting of:
- Ontology: all lawful 48-bit states
- Epistemology: complete transition function for 256 action bytes
- Stage profile: stage-resolved distinction counts per layer (N×4 uint8)
- Loop defects: BU loop holonomy defects for three commutator loops (N×3 uint8)
- Aperture: Hodge-derived aperture from stage geometry and loop defects (N float32)

The kernel computes governance observables via a K4 Hodge decomposition derived
from stage profile and loop defects, providing BU-Egress and BU-Ingress physics.

No probabilistic scoring, sampling, or statistical estimation is used.
All outputs are deterministic and replayable from the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..physics import alignment, atlas_builder, atlas_loader, governance, router_maps_builder
from .ledger import Ledger
from .signature import RoutingSignature


@dataclass
class RouterKernel:
    """Deterministic Router kernel operating on a fixed atlas and router maps."""

    ontology: NDArray[np.uint64]
    epistemology: NDArray[np.int32]
    stage_profile: NDArray[np.uint8]
    loop_defects: NDArray[np.uint8]
    aperture: NDArray[np.float32]
    archetype_index: int
    state_index: int
    ledger: Optional[Ledger] = None

    @classmethod
    def from_directory(
        cls,
        base_dir: Path,
        with_ledger: bool = False,
        ledger_path: Optional[Path] = None,
    ) -> "RouterKernel":
        """
        Construct a RouterKernel from atlas and router maps stored in base_dir.

        If atlas or router maps are missing, they are built deterministically in-place.
        """
        base_dir.mkdir(parents=True, exist_ok=True)

        cfg = atlas_builder.AtlasConfiguration(output_directory=base_dir)
        atlas_paths = atlas_builder.AtlasPaths.from_directory(base_dir)
        if not (
            atlas_paths.ontology.exists()
            and atlas_paths.epistemology.exists()
        ):
            atlas_builder.AtlasBuilder(configuration=cfg).build_all()

        if not (
            atlas_paths.stage_profile.exists()
            and atlas_paths.loop_defects.exists()
            and atlas_paths.aperture.exists()
        ):
            router_maps_builder.build_router_maps(
                ontology_path=atlas_paths.ontology,
                epistemology_path=atlas_paths.epistemology,
                output_directory=base_dir,
            )

        atlas = atlas_loader.load_atlas(paths=atlas_paths)

        archetype_state = int(governance.tensor_to_int(governance.GENE_Mac_S))
        archetype_index = int(np.searchsorted(atlas.ontology, archetype_state))
        if archetype_index >= int(atlas.ontology.size) or int(atlas.ontology[archetype_index]) != archetype_state:
            raise RuntimeError(
                "Archetype state not found in ontology; atlas is inconsistent with governance physics"
            )

        ledger_obj: Optional[Ledger] = None
        if with_ledger:
            lp = ledger_path or (base_dir / "router_ledger.dat")
            ledger_obj = Ledger(path=lp)

        return cls(
            ontology=atlas.ontology,
            epistemology=atlas.epistemology,
            stage_profile=atlas.stage_profile,
            loop_defects=atlas.loop_defects,
            aperture=atlas.aperture,
            archetype_index=archetype_index,
            state_index=archetype_index,
            ledger=ledger_obj,
        )

    def reset(self, state_index: Optional[int] = None) -> None:
        """
        Reset the kernel session to a given atlas state.

        Args:
            state_index: Atlas state index. Defaults to the archetype index.
        """
        self.state_index = self.archetype_index if state_index is None else int(state_index)

    def _step_byte(self, byte: int) -> None:
        """Process a single byte through the BU egress pipeline."""
        action = governance.byte_to_action(byte)

        state_before_index = self.state_index

        self.state_index = int(self.epistemology[self.state_index, action])

        if self.ledger is not None:
            state_before_int = int(self.ontology[state_before_index])
            state_after_int = int(self.ontology[self.state_index])
            self.ledger.append_egress(
                state_before=state_before_int,
                action_byte=action,
                state_after=state_after_int,
            )

    def step_bytes(self, payload: bytes) -> RoutingSignature:
        """
        Process a payload of bytes and return the resulting routing signature.

        The BU egress mapping is byte → action → atlas transition.
        """
        for b in payload:
            self._step_byte(b)
        return self.get_signature()

    def get_signature(self) -> RoutingSignature:
        """
        Compute the routing signature for the current kernel state.

        This is a pure function of the current atlas state index and the
        precomputed router maps and does not modify kernel state.
        """
        i = self.state_index

        state_int = int(self.ontology[i])
        sp = self.stage_profile[i]
        stage_profile = (int(sp[0]), int(sp[1]), int(sp[2]), int(sp[3]))
        ld = self.loop_defects[i]
        loop_defects = (int(ld[0]), int(ld[1]), int(ld[2]))
        aperture = float(self.aperture[i])
        si = alignment.compute_si(aperture)

        return RoutingSignature(
            state_index=i,
            state_int_hex=f"{state_int:012x}",
            stage_profile=stage_profile,
            loop_defects=loop_defects,
            aperture=aperture,
            si=si,
        )


