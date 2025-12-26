"""
Atlas builder.

Running this module builds:
- ontology.npy: sorted unique reachable 24-bit states
- epistemology.npy: [N,256] next-state indices (fast router lookup)
- phenomenology.npz: capped measurement graph/constants
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
import numpy as np
from numpy.typing import NDArray

# Import constants - handle both direct execution and module import
try:
    # Try relative import first (when imported as module)
    from .constants import (
        ARCHETYPE_A12,
        ARCHETYPE_B12,
        ARCHETYPE_STATE24,
        GENE_MIC_S,
        K4,
        XFORM_MASK_BY_BYTE,
        step_state_by_byte,
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    # Add the project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from src.router.constants import (
        ARCHETYPE_A12,
        ARCHETYPE_B12,
        ARCHETYPE_STATE24,
        GENE_MIC_S,
        K4,
        XFORM_MASK_BY_BYTE,
        step_state_by_byte,
    )


@dataclass(frozen=True)
class AtlasPaths:
    base: Path

    @property
    def ontology(self) -> Path:
        return self.base / "ontology.npy"

    @property
    def epistemology(self) -> Path:
        return self.base / "epistemology.npy"

    @property
    def phenomenology(self) -> Path:
        return self.base / "phenomenology.npz"


def build_ontology(paths: AtlasPaths) -> NDArray[np.uint32]:
    """
    Discover all reachable 24-bit states using all 256 byte actions.
    
    Explores the full action space: each byte is a distinct operator.
    """
    print("Building ontology...")
    paths.base.mkdir(parents=True, exist_ok=True)

    # Use canonical archetype from constants
    archetype = ARCHETYPE_STATE24

    visited: set[int] = {archetype}
    frontier = [archetype]
    iterations = 0

    while frontier:
        iterations += 1
        nxt: list[int] = []
        for s in frontier:
            # Explore all 256 byte actions
            for byte in range(256):
                t = step_state_by_byte(s, byte)
                if t not in visited:
                    visited.add(t)
                    nxt.append(t)
        frontier = nxt
        if iterations <= 5 or iterations % 10 == 0:
            print(f"  Iteration {iterations}: {len(visited)} states discovered")

    ontology = np.array(sorted(visited), dtype=np.uint32)
    np.save(paths.ontology, ontology)
    
    file_size = paths.ontology.stat().st_size
    print(f"Ontology complete: {len(ontology):,} unique states")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  Explored in {iterations} iterations")
    
    # Verify byte sensitivity
    unique_from_archetype = len({step_state_by_byte(archetype, b) for b in range(256)})
    print(f"  Unique transitions from archetype: {unique_from_archetype} / 256 bytes")
    return ontology


def build_epistemology(paths: AtlasPaths, ontology: NDArray[np.uint32]) -> None:
    """
    Build epistemology as [N,256] indices using byte-sensitive transitions.
    
    Each column corresponds to an input byte (0-255), encoding the next state index.
    """
    print("Building epistemology...")
    n = int(ontology.size)

    from numpy.lib.format import open_memmap
    epi = open_memmap(str(paths.epistemology), mode="w+", dtype=np.uint32, shape=(n, 256))

    # Build lookup: state value -> ontology index
    state_to_idx = {int(ontology[i]): i for i in range(n)}

    # Compute transitions for each state and byte
    for i in range(n):
        s = int(ontology[i])
        
        for byte in range(256):
            t = step_state_by_byte(s, byte)
            if t not in state_to_idx:
                raise RuntimeError(f"Epistemology closure violation: state {hex(s)} + byte {byte} -> {hex(t)} not in ontology")
            epi[i, byte] = np.uint32(state_to_idx[t])

        if (i + 1) % 50_000 == 0:
            epi.flush()
            print(f"  Processed {i + 1:,} / {n:,} states ({100 * (i + 1) / n:.1f}%)")

    epi.flush()
    
    # Verify uniqueness: count distinct transitions per state
    unique_per_state = []
    for i in range(min(10, n)):  # Sample first 10 states
        unique = len(set(int(epi[i, b]) for b in range(256)))
        unique_per_state.append(unique)
    avg_unique = sum(unique_per_state) / len(unique_per_state) if unique_per_state else 0
    
    file_size = paths.epistemology.stat().st_size
    print(f"Epistemology complete: [{n:,}, 256] lookup table")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  Total entries: {n * 256:,}")
    if unique_per_state:
        print(f"  Avg unique transitions per state (sample): {avg_unique:.1f} / 256")


def build_phenomenology(paths: AtlasPaths) -> None:
    """
    Write capped measurement graph/constants.

    This file is independent of N. Stores byte→mask mapping and archetype.
    """
    print("Building phenomenology...")

    np.savez_compressed(
        paths.phenomenology,
        archetype_state24=np.uint32(ARCHETYPE_STATE24),
        archetype_a12=np.uint16(ARCHETYPE_A12),
        archetype_b12=np.uint16(ARCHETYPE_B12),
        gene_mic_s=np.uint8(GENE_MIC_S),
        xform_mask_by_byte=XFORM_MASK_BY_BYTE,
        k4_edges=np.array(K4.edges, dtype=np.uint8),
        k4_p_cycle=np.array(K4.p_cycle, dtype=np.float64),
    )
    
    file_size = paths.phenomenology.stat().st_size
    unique_masks = len(set(int(XFORM_MASK_BY_BYTE[b]) for b in range(256)))
    print(f"Phenomenology complete: measurement constants")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    print(f"  Unique masks: {unique_masks} / 256 bytes")
    print(f"  Contents: archetype_state24, archetype_a12, archetype_b12, gene_mic_s, xform_mask_by_byte, k4_edges, k4_p_cycle")


def build_all(base_dir: Path) -> None:
    paths = AtlasPaths(base=base_dir)
    print("Atlas Builder")
    print("----------")
    t0 = time.time()
    ontology = build_ontology(paths)
    print("")
    build_epistemology(paths, ontology)
    print("")
    build_phenomenology(paths)
    print("")
    elapsed = time.time() - t0
    
    total_size = (
        paths.ontology.stat().st_size +
        paths.epistemology.stat().st_size +
        paths.phenomenology.stat().st_size
    )
    
    print("Summary")
    print("----------")
    print(f"Total build time: {elapsed:.2f} seconds")
    print(f"Total atlas size: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")
    print(f"Output directory: {paths.base}")
    print("")
    print("Maps built:")
    print(f"  - ontology.npy: {len(ontology):,} states")
    print(f"  - epistemology.npy: [{len(ontology):,}, 256] lookup table")
    print(f"  - phenomenology.npz: measurement constants")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build ontology.npy, epistemology.npy, phenomenology.npz")
    parser.add_argument("--out", type=Path, default=Path("data/atlas"))
    args = parser.parse_args()
    build_all(args.out)


if __name__ == "__main__":
    main()