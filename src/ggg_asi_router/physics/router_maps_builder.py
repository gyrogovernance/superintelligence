"""
Router Maps Builder

Builds the three router-specific maps from ontology and epistemology:

1. stage_profile.npy (N×4 uint8): Stage-resolved distinction counts per layer
2. loop_defects.npy (N×3 uint8): BU loop holonomy defects for three commutator loops
3. aperture.npy (N float32): Hodge-derived aperture from stage geometry and loop defects

These maps are derived deterministically from atlas-native quantities and provide
the observables needed for Router kernel routing decisions.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
import numpy as np

# Handle both script execution and module import
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from ggg_asi_router.physics import governance, k4_geometry
else:
    from . import governance, k4_geometry

logger = logging.getLogger(__name__)


def _popcount_uint64(arr):
    """
    Vectorized popcount for uint64 arrays using bit manipulation.

    This is endian-independent and works correctly for 48-bit integers
    stored in uint64 arrays.
    """
    # Parallel reduction popcount algorithm
    v = arr.copy().astype(np.uint64)
    v = v - ((v >> 1) & np.uint64(0x5555555555555555))
    v = (v & np.uint64(0x3333333333333333)) + ((v >> 2) & np.uint64(0x3333333333333333))
    v = (v + (v >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
    v = (v + (v >> 8)) & np.uint64(0x00FF00FF00FF00FF)
    v = (v + (v >> 16)) & np.uint64(0x0000FFFF0000FFFF)
    v = (v + (v >> 32)) & np.uint64(0x00000000FFFFFFFF)
    return v.astype(np.uint8)


def build_router_maps(
    ontology_path: Path,
    epistemology_path: Path,
    output_directory: Path,
    chunk_size: int = 10_000,
) -> None:
    """
    Build the three router maps from ontology and epistemology.

    Args:
        ontology_path: Path to ontology_keys.npy
        epistemology_path: Path to epistemology.npy
        output_directory: Directory to write stage_profile.npy, loop_defects.npy, aperture.npy
        chunk_size: Chunk size for processing states
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ontology and epistemology...")
    ontology = np.load(str(ontology_path.resolve()), mmap_mode="r")
    epistemology = np.load(str(epistemology_path.resolve()), mmap_mode="r")

    n_states = len(ontology)
    logger.info(f"Processing {n_states:,} states")

    # Archetype state
    s_ref = int(governance.tensor_to_int(governance.GENE_Mac_S))
    archetype_idx = np.searchsorted(ontology, s_ref)
    if archetype_idx >= n_states or int(ontology[archetype_idx]) != s_ref:
        raise RuntimeError(f"Archetype state {s_ref:012x} not found in ontology")

    # Initialize output arrays
    stage_profile = np.zeros((n_states, 4), dtype=np.uint8)
    loop_defects = np.zeros((n_states, 3), dtype=np.uint8)
    aperture_array = np.zeros(n_states, dtype=np.float32)

    # Precompute K4 geometry
    B = k4_geometry.get_incidence_matrix_k4()
    W = k4_geometry.get_weight_matrix_k4()
    F = k4_geometry.get_face_cycle_matrix_k4(W)  # Face-cycle matrix (not orthonormalized)

    # Loop internal actions (direct indices into epistemology)
    loop_actions = [
        (governance.UNA_P_ACTION, governance.ONA_P_ACTION, governance.UNA_M_ACTION, governance.ONA_M_ACTION),  # CS-UNA-ONA
        (governance.UNA_P_ACTION, governance.BU_P_ACTION, governance.UNA_M_ACTION, governance.BU_M_ACTION),    # CS-UNA-BU
        (governance.ONA_P_ACTION, governance.BU_P_ACTION, governance.ONA_M_ACTION, governance.BU_M_ACTION),   # CS-ONA-BU
    ]

    logger.info("Computing stage profiles, loop defects, and aperture...")
    start_time = time.time()

    for chunk_start in range(0, n_states, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_states)
        chunk_indices = np.arange(chunk_start, chunk_end, dtype=np.int64)

        # Stage profile: popcount of (s XOR s_ref) & LAYER_MASK[k] for each layer
        states_chunk = ontology[chunk_indices].astype(np.uint64, copy=False)
        diff = states_chunk ^ np.uint64(s_ref)
        for layer in range(4):
            layer_diff = diff & governance.LAYER_MASKS[layer]
            # Vectorized popcount using bit manipulation (endian-independent)
            stage_profile[chunk_indices, layer] = _popcount_uint64(layer_diff)

        # Loop defects: compute three commutator loops (vectorized)
        for loop_idx, (action_A_pos, action_B_pos, action_A_neg, action_B_neg) in enumerate(loop_actions):
            # Vectorized loop computation: i -> B_pos -> A_pos -> B_neg -> A_neg
            idx = chunk_indices.copy()
            
            # Step 1: B_pos
            idx = epistemology[idx, action_B_pos].astype(np.int64)
            
            # Step 2: A_pos
            idx = epistemology[idx, action_A_pos].astype(np.int64)
            
            # Step 3: B_neg
            idx = epistemology[idx, action_B_neg].astype(np.int64)
            
            # Step 4: A_neg
            idx = epistemology[idx, action_A_neg].astype(np.int64)
            
            # Defect = popcount(s_i XOR s_loop(i)) for all states in chunk
            states_i = ontology[chunk_indices].astype(np.uint64, copy=False)
            states_loop = ontology[idx].astype(np.uint64, copy=False)
            diff = states_i ^ states_loop
            defects = _popcount_uint64(diff)
            # Clamp to uint8 range (should never exceed 48, but be safe)
            loop_defects[chunk_indices, loop_idx] = np.clip(defects, 0, 255).astype(np.uint8)

        # Aperture: derive from stage profile and loop defects (vectorized)
        # Stage angles: theta_k = (1/pi) * arccos(1 - 2*p_k/12)
        p_chunk = stage_profile[chunk_indices].astype(np.float32)
        theta_stage = np.arccos(np.clip(1.0 - 2.0 * p_chunk / 12.0, -1.0, 1.0)) / np.pi

        # Gradient edge vector: y_grad = B^T * x where x = [theta_CS, theta_UNA, theta_ONA, theta_BU]
        y_grad = (B.T @ theta_stage.T).T  # Shape: (chunk_size, 6)

        # Cycle amplitudes from loop defects: c_m = (1/pi) * arccos(1 - 2*d_m/48)
        d_chunk = loop_defects[chunk_indices].astype(np.float32)
        c_cycle = np.arccos(np.clip(1.0 - 2.0 * d_chunk / 48.0, -1.0, 1.0)) / np.pi  # Shape: (chunk_size, 3)

        # Cycle edge vector: y_cycle = F * c where F is 6×3 face-cycle matrix, c is (3,), so y_cycle is (6,)
        # F columns correspond directly to the three commutator loops (CS-UNA-ONA, CS-UNA-BU, CS-ONA-BU)
        y_cycle = (F @ c_cycle.T).T  # Shape: (chunk_size, 6)

        # Total edge vector: y = y_grad + y_cycle
        y = y_grad + y_cycle  # Shape: (chunk_size, 6)

        # Aperture: A = ||y_cycle||²_W / ||y||²_W (vectorized)
        # Compute weighted norms: ||v||²_W = v^T @ W @ v
        Wy = y @ W  # Shape: (chunk_size, 6)
        Wy_cycle = y_cycle @ W  # Shape: (chunk_size, 6)
        y_norm_sq = np.sum(Wy * y, axis=1)  # Shape: (chunk_size,)
        y_cycle_norm_sq = np.sum(Wy_cycle * y_cycle, axis=1)  # Shape: (chunk_size,)
        
        # Handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            aperture_chunk = np.where(y_norm_sq > 0, y_cycle_norm_sq / y_norm_sq, 0.0)
        
        aperture_array[chunk_indices] = aperture_chunk.astype(np.float32)

        if (chunk_end % 50_000 == 0) or chunk_end == n_states:
            elapsed = time.time() - start_time
            rate = chunk_end / elapsed if elapsed > 0 else 0
            logger.info(f"  Processed {chunk_end:,}/{n_states:,} states ({rate:.0f} states/sec)")

    # Write outputs
    logger.info("Writing router maps...")
    stage_profile_path = output_directory / "stage_profile.npy"
    loop_defects_path = output_directory / "loop_defects.npy"
    aperture_path = output_directory / "aperture.npy"

    for path, array in [
        (stage_profile_path, stage_profile),
        (loop_defects_path, loop_defects),
        (aperture_path, aperture_array),
    ]:
        temp_path = path.with_suffix(".tmp")
        with open(str(temp_path.resolve()), "wb") as f:
            np.lib.format.write_array(f, array, allow_pickle=False)
        shutil.move(str(temp_path.resolve()), str(path.resolve()))

    logger.info("Router maps build complete")
    logger.info(f"  Stage profile: {stage_profile_path}")
    logger.info(f"  Loop defects: {loop_defects_path}")
    logger.info(f"  Aperture: {aperture_path}")


def main() -> None:
    """CLI entry point for router maps builder."""
    parser = argparse.ArgumentParser(description="Build router maps from atlas")
    parser.add_argument(
        "--ontology",
        type=Path,
        default=Path("data/atlas/ontology_keys.npy"),
        help="Path to ontology_keys.npy",
    )
    parser.add_argument(
        "--epistemology",
        type=Path,
        default=Path("data/atlas/epistemology.npy"),
        help="Path to epistemology.npy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/atlas"),
        help="Output directory for router maps",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Chunk size for processing",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    build_router_maps(
        ontology_path=args.ontology,
        epistemology_path=args.epistemology,
        output_directory=args.output,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
