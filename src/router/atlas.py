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


# Atlas version - increment when build method or constants change incompatibly
ATLAS_VERSION = "1.0"


def build_ontology(paths: AtlasPaths) -> NDArray[np.uint32]:
    """
    Build ontology directly as A_set × B_set using proven closed-form algebra.
    
    By Property P3, Ω = A_set × B_set where:
    - A_set = {ARCHETYPE_A12 XOR m_b : b in [0,255]} (256 elements)
    - B_set = {ARCHETYPE_B12 XOR m_b : b in [0,255]} (256 elements)
    
    This is faster and more direct than BFS exploration.
    """
    print("Building ontology...")
    paths.base.mkdir(parents=True, exist_ok=True)

    # Extract A-masks from XFORM_MASK_BY_BYTE
    masks_a = np.array([(int(XFORM_MASK_BY_BYTE[b]) >> 12) & 0xFFF for b in range(256)], dtype=np.uint16)
    
    # Build A_set and B_set directly
    a_set = (ARCHETYPE_A12 ^ masks_a).astype(np.uint16) & 0xFFF
    b_set = (ARCHETYPE_B12 ^ masks_a).astype(np.uint16) & 0xFFF
    
    # Cartesian product: all combinations
    a_grid, b_grid = np.meshgrid(a_set, b_set, indexing='ij')
    
    # Pack into 24-bit states
    ontology = ((a_grid.astype(np.uint32) << 12) | b_grid.astype(np.uint32)).flatten()
    
    # Sort for consistent ordering and ensure uint32 dtype
    ontology = np.sort(ontology).astype(np.uint32)
    
    np.save(paths.ontology, ontology)
    
    file_size = paths.ontology.stat().st_size
    print(f"Ontology complete: {len(ontology):,} unique states")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  Built directly as 256 × 256 cartesian product")
    
    # Verify byte sensitivity
    unique_from_archetype = len({step_state_by_byte(ARCHETYPE_STATE24, b) for b in range(256)})
    print(f"  Unique transitions from archetype: {unique_from_archetype} / 256 bytes")
    return ontology


def build_epistemology(paths: AtlasPaths, ontology: NDArray[np.uint32]) -> None:
    """
    Build epistemology as [N,256] indices using vectorized column-wise construction.
    
    Each column corresponds to an input byte (0-255), encoding the next state index.
    Uses vectorized step law computation (same as test) for speed and correctness.
    """
    print("Building epistemology...")
    n = int(ontology.size)

    from numpy.lib.format import open_memmap
    epi = open_memmap(str(paths.epistemology), mode="w+", dtype=np.uint32, shape=(n, 256))

    # Ensure ontology is sorted for searchsorted
    assert np.all(ontology[:-1] <= ontology[1:]), "Ontology must be sorted"

    # Unpack all states into A and B components (vectorized)
    a = ((ontology >> 12) & 0xFFF).astype(np.uint32)
    b = (ontology & 0xFFF).astype(np.uint32)

    # Build column-wise (byte-by-byte) for better cache locality
    for byte in range(256):
        mask24 = int(XFORM_MASK_BY_BYTE[byte])
        m = (mask24 >> 12) & 0xFFF

        # Vectorized step law: compute next states for all ontology states at once
        new_a = (b ^ 0xFFF).astype(np.uint32)
        new_b = ((a ^ m) ^ 0xFFF).astype(np.uint32)
        new_state = ((new_a << 12) | new_b).astype(np.uint32)

        # Find indices using binary search (fast for sorted array)
        idx = np.searchsorted(ontology, new_state).astype(np.int64)

        # Verify membership (should always hold by closure)
        assert np.all(ontology[idx] == new_state), f"Byte {byte}: closure violation detected"

        # Write column
        epi[:, byte] = idx.astype(np.uint32)

        if (byte + 1) % 32 == 0:
            epi.flush()
            print(f"  Processed {byte + 1:,} / 256 bytes ({100 * (byte + 1) / 256:.1f}%)")

    epi.flush()
    
    # Verify uniqueness: count distinct transitions per state (sample)
    unique_per_state = []
    for i in range(min(10, n)):  # Sample first 10 states
        unique = len(np.unique(epi[i, :]))
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
        atlas_version=ATLAS_VERSION,
        archetype_state24=np.uint32(ARCHETYPE_STATE24),
        archetype_a12=np.uint16(ARCHETYPE_A12),
        archetype_b12=np.uint16(ARCHETYPE_B12),
        gene_mic_s=np.uint8(GENE_MIC_S),
        xform_mask_by_byte=XFORM_MASK_BY_BYTE,
    )
    
    file_size = paths.phenomenology.stat().st_size
    unique_masks = len(set(int(XFORM_MASK_BY_BYTE[b]) for b in range(256)))
    print(f"Phenomenology complete: measurement constants")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    print(f"  Unique masks: {unique_masks} / 256 bytes")
    print(f"  Atlas version: {ATLAS_VERSION}")
    print(f"  Contents: atlas_version, archetype_state24, archetype_a12, archetype_b12, gene_mic_s, xform_mask_by_byte")


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