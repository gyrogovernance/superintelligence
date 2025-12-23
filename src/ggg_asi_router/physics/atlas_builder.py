"""
Router Atlas Builder

Constructs the two core atlas maps that define the finite governance atlas
for the GGG ASI Alignment Router:

1. Ontology: all lawful 48-bit governance states
2. Epistemology: complete transition table for 256 action bytes

These maps are derived from the deterministic governance physics in
ggg_asi_router.physics.governance and provide the atlas-native substrate for Router
state transitions. Additional router maps (stage_profile, loop_defects, aperture)
are built separately by router_maps_builder.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Handle both script execution and module import
if __name__ == "__main__":
    # Add parent directory to path for script execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from ggg_asi_router.physics import governance
else:
    from . import governance


logger = logging.getLogger(__name__)


StateInteger = int


@dataclass(frozen=True)
class AtlasConfiguration:
    """Configuration parameters for atlas construction."""

    output_directory: Path = Path("data/atlas")

    expected_state_count: int = 788_986
    expected_diameter: int = 6

    bfs_chunk_size: int = 20_000
    epistemology_chunk_size: int = 10_000

    strict_validation: bool = True

    def __post_init__(self) -> None:
        """Ensure output directory exists."""
        self.output_directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AtlasPaths:
    """File paths for the five canonical atlas maps."""

    ontology: Path
    epistemology: Path
    stage_profile: Path
    loop_defects: Path
    aperture: Path

    @classmethod
    def from_directory(cls, directory: Path) -> "AtlasPaths":
        """Construct standard paths from a base directory."""
        return cls(
            ontology=directory / "ontology_keys.npy",
            epistemology=directory / "epistemology.npy",
            stage_profile=directory / "stage_profile.npy",
            loop_defects=directory / "loop_defects.npy",
            aperture=directory / "aperture.npy",
        )


class ProgressMonitor:
    """Monitor and report progress for long-running operations."""

    def __init__(self, operation_name: str) -> None:
        """
        Initialize progress monitor.

        Args:
            operation_name: Description of the operation being monitored.
        """
        self.operation_name = operation_name
        self.start_time = time.perf_counter()
        self.last_update_time = 0.0
        self.update_interval = 0.1

    def update(
        self,
        current: int,
        total: Optional[int] = None,
        metadata: Optional[str] = None,
    ) -> None:
        """
        Update progress display.

        Args:
            current: Current progress count.
            total: Total expected count (if known).
            metadata: Additional information to display.
        """
        current_time = time.perf_counter()
        if current_time - self.last_update_time < self.update_interval:
            if total is None or current != total:
                return

        self.last_update_time = current_time
        elapsed = current_time - self.start_time
        rate = current / elapsed if elapsed > 0 else 0.0

        message_parts = [f"{self.operation_name}: {current:,}"]
        if total is not None:
            percentage = 100.0 * current / total if total > 0 else 0.0
            message_parts.append(f"/{total:,} ({percentage:.1f}%)")
        message_parts.append(f"| {rate:.0f}/s | {elapsed:.1f}s")
        if metadata:
            message_parts.append(f"| {metadata}")

        print(f"\r{' '.join(message_parts):<80}", end="", flush=True)

    def complete(self) -> None:
        """Mark operation as complete and display final timing."""
        elapsed = time.perf_counter() - self.start_time
        print(f"\r{self.operation_name}: Complete in {elapsed:.1f}s{' '*60}")
        logger.info("%s completed in %.1fs", self.operation_name, elapsed)


class OntologyBuilder:
    """
    Construct the ontology map through breadth-first exploration.

    The ontology is the complete finite manifold of reachable governance states
    under the gyroscopic transform, starting from the archetypal tensor
    GENE_Mac_S. The measured manifold contains exactly 788,986 states with
    graph diameter 6.
    """

    def __init__(self, configuration: AtlasConfiguration) -> None:
        """
        Initialize ontology builder.

        Args:
            configuration: Atlas construction parameters.
        """
        self.configuration = configuration

    def _expand_frontier_vectorized(
        self,
        frontier: NDArray[np.uint64],
    ) -> NDArray[np.uint64]:
        """
        Apply all 256 actions to frontier states in parallel.

        Args:
            frontier: Array of current frontier states.

        Returns:
            Deduplicated array of successor states.
        """
        successors = governance.apply_transition_all_actions(frontier)
        unique_successors = np.unique(successors.ravel())
        return unique_successors

    def build(
        self,
        ontology_path: Path,
    ) -> NDArray[np.uint64]:
        """
        Discover the complete state manifold via breadth-first search.

        Args:
            ontology_path: Output path for state integers.

        Returns:
            Sorted array of all reachable state integers.

        Raises:
            RuntimeError: If measured invariants do not match expected values.
        """
        progress = ProgressMonitor("Discovering ontology")

        archetype_integer = int(governance.tensor_to_int(governance.GENE_Mac_S))
        logger.info("Starting BFS from archetype: %012x", archetype_integer)

        discovered_states: set[StateInteger] = {archetype_integer}
        frontier_array = np.array([archetype_integer], dtype=np.uint64)

        current_depth = 0
        layer_sizes: List[int] = []

        while frontier_array.size > 0:
            next_frontier_set: set[StateInteger] = set()

            for chunk_start in range(0, frontier_array.size, self.configuration.bfs_chunk_size):
                chunk_end = min(chunk_start + self.configuration.bfs_chunk_size, frontier_array.size)
                frontier_chunk = frontier_array[chunk_start:chunk_end]
                unique_successors = self._expand_frontier_vectorized(frontier_chunk)
                successor_set = set(int(s) for s in unique_successors)
                next_frontier_set.update(successor_set)

            next_frontier_set.difference_update(discovered_states)
            if not next_frontier_set:
                break

            frontier_array = np.fromiter(next_frontier_set, dtype=np.uint64)
            discovered_states.update(next_frontier_set)
            current_depth += 1
            layer_sizes.append(frontier_array.size)

            progress.update(
                len(discovered_states),
                None,
                f"depth={current_depth}, frontier={frontier_array.size:,}",
            )

            if current_depth > 64:
                raise RuntimeError("BFS depth exceeded 64; possible physics drift")

        progress.complete()

        if self.configuration.strict_validation:
            if len(discovered_states) != self.configuration.expected_state_count:
                raise RuntimeError(
                    "State count mismatch: expected "
                    f"{self.configuration.expected_state_count:,}, "
                    f"found {len(discovered_states):,}",
                )
            if current_depth != self.configuration.expected_diameter:
                raise RuntimeError(
                    "Diameter mismatch: expected "
                    f"{self.configuration.expected_diameter}, "
                    f"found {current_depth}",
                )

        state_array = np.array(sorted(discovered_states), dtype=np.uint64)

        logger.info(
            "Discovered %s states with diameter %s. Layer sizes: %s",
            f"{state_array.size:,}",
            current_depth,
            layer_sizes,
        )

        # Ensure directory exists
        ontology_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temp file first, then rename (avoids Windows file locking)
        ontology_tmp = ontology_path.parent / f".tmp_{ontology_path.name}"
        
        with open(str(ontology_tmp), 'wb') as f:
            np.lib.format.write_array(f, state_array, allow_pickle=False)
        
        # Atomic replace (works even if target is locked on some systems)
        shutil.move(str(ontology_tmp), str(ontology_path))

        archetype_index = int(np.searchsorted(state_array, archetype_integer))
        logger.info("Archetype index: %d", archetype_index)
        logger.info("Index 0 state: %012x", int(state_array[0]))

        return state_array


class EpistemologyBuilder:
    """
    Construct the epistemology map (complete transition table).

    The epistemology encodes all possible state transitions under the 256 actions,
    forming a complete description of the atlas dynamics used by the Router
    kernel for BU egress.
    """

    def __init__(self, configuration: AtlasConfiguration) -> None:
        """
        Initialize epistemology builder.

        Args:
            configuration: Atlas construction parameters.
        """
        self.configuration = configuration

    def build(
        self,
        ontology_path: Path,
        epistemology_path: Path,
    ) -> None:
        """
        Build the complete state transition table.

        Args:
            ontology_path: Path to ontology map.
            epistemology_path: Output path for transition table.

        Raises:
            FileNotFoundError: If required input files do not exist.
            RuntimeError: If transitions produce states outside the ontology.
        """
        if not ontology_path.exists():
            raise FileNotFoundError(f"Ontology not found: {ontology_path}")

        state_array = np.load(ontology_path, mmap_mode="r")
        n_states = int(state_array.size)

        epistemology_array = self._create_memmap_array(epistemology_path, (n_states, 256))

        progress = ProgressMonitor("Building epistemology")

        for chunk_start in range(0, n_states, self.configuration.epistemology_chunk_size):
            chunk_end = min(chunk_start + self.configuration.epistemology_chunk_size, n_states)
            chunk_states = state_array[chunk_start:chunk_end]

            successor_array = governance.apply_transition_all_actions(chunk_states)

            successor_indices = np.searchsorted(state_array, successor_array, side="left")

            if np.any(successor_indices >= n_states) or np.any(state_array[successor_indices] != successor_array):
                self._handle_closure_violation(
                    chunk_states,
                    successor_array,
                    successor_indices,
                    state_array,
                    n_states,
                )

            epistemology_array[chunk_start:chunk_end, :] = successor_indices.astype(np.int32, copy=False)

            progress.update(chunk_end, n_states)

        self._flush_memmap(epistemology_array)

        progress.complete()
        logger.info("Epistemology table saved: shape=(%s, 256)", f"{n_states:,}")

    def _create_memmap_array(
        self,
        path: Path,
        shape: Tuple[int, ...],
    ) -> NDArray[np.int32]:
        """
        Create a memory-mapped array for large data.

        Args:
            path: Output file path.
            shape: Array dimensions.

        Returns:
            Memory-mapped array.
        """
        from numpy.lib.format import open_memmap

        array = open_memmap(
            str(path),
            dtype=np.int32,
            mode="w+",
            shape=shape,
        )
        return array  # type: ignore[return-value]

    def _flush_memmap(self, array: NDArray[np.int32]) -> None:
        """Flush a memory-mapped array to disk."""
        try:
            array.flush()  # type: ignore[attr-defined]
        except AttributeError:
            return

    def _handle_closure_violation(
        self,
        chunk_states: NDArray[np.uint64],
        successor_array: NDArray[np.uint64],
        successor_indices: NDArray[np.intp],
        state_array: NDArray[np.uint64],
        n_states: int,
    ) -> None:
        """
        Handle a case where a transition produces a state outside the ontology.

        Raises:
            RuntimeError: Always. This is a violation of atlas closure.
        """
        violation_mask = (successor_indices >= n_states) | (state_array[successor_indices] != successor_array)
        violation_positions = np.where(violation_mask)

        if violation_positions[0].size > 0:
            state_index = int(violation_positions[0][0])
            action_index = int(violation_positions[1][0])

            source_state = int(chunk_states[state_index])
            target_state = int(successor_array[state_index, action_index])

            raise RuntimeError(
                "Closure violation: state "
                f"{source_state:012x} + action {action_index:02x} "
                f"→ {target_state:012x} (not in ontology)",
            )

class AtlasBuilder:
    """
    Orchestrate construction of the two core atlas maps.

    The construction order is:
    1. Ontology.
    2. Epistemology.
    """

    def __init__(self, configuration: Optional[AtlasConfiguration] = None) -> None:
        """
        Initialize atlas builder.

        Args:
            configuration: Atlas construction parameters (defaults if None).
        """
        self.configuration = configuration or AtlasConfiguration()
        self.paths = AtlasPaths.from_directory(self.configuration.output_directory)

        self.ontology_builder = OntologyBuilder(self.configuration)
        self.epistemology_builder = EpistemologyBuilder(self.configuration)

    def build_ontology(self) -> None:
        """Build ontology map."""
        logger.info("Building ontology map")
        self.ontology_builder.build(self.paths.ontology)

    def build_epistemology(self) -> None:
        """Build epistemology map (requires ontology)."""
        if not self.paths.ontology.exists():
            logger.info("Prerequisites missing; building ontology first")
            self.build_ontology()

        logger.info("Building epistemology map")
        self.epistemology_builder.build(
            self.paths.ontology,
            self.paths.epistemology,
        )

    def build_all(self) -> None:
        """Build all core atlas maps in dependency order."""
        logger.info("Starting complete atlas construction")
        start_time = time.perf_counter()

        self.build_ontology()
        self.build_epistemology()

        elapsed = time.perf_counter() - start_time

        logger.info("Atlas construction complete in %.1fs", elapsed)
        logger.info("Generated artifacts:")
        logger.info("ontology: %s", self.paths.ontology)
        logger.info("epistemology: %s", self.paths.epistemology)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for atlas construction."""
    parser = argparse.ArgumentParser(
        description="Router Atlas Builder: construct canonical atlas maps\n\n"
        "By default (no arguments), builds complete atlas (all + router_maps).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    ontology_parser = subparsers.add_parser(
        "ontology",
        help="Build ontology map",
    )
    ontology_parser.set_defaults(func=lambda args: AtlasBuilder().build_ontology())

    epistemology_parser = subparsers.add_parser(
        "epistemology",
        help="Build epistemology map",
    )
    epistemology_parser.set_defaults(func=lambda args: AtlasBuilder().build_epistemology())

    router_maps_parser = subparsers.add_parser(
        "router_maps",
        help="Build router maps (stage_profile, loop_defects, aperture)",
    )
    router_maps_parser.set_defaults(func=lambda args: _build_router_maps())

    all_parser = subparsers.add_parser(
        "all",
        help="Build all core atlas maps (ontology + epistemology)",
    )
    all_parser.set_defaults(func=lambda args: AtlasBuilder().build_all())

    complete_parser = subparsers.add_parser(
        "complete",
        help="Build complete atlas (all + router_maps)",
    )
    complete_parser.set_defaults(func=lambda args: _build_complete())

    return parser


def _build_router_maps() -> None:
    """Build router maps from existing atlas."""
    try:
        from . import router_maps_builder
    except ImportError:
        from ggg_asi_router.physics import router_maps_builder
    
    cfg = AtlasConfiguration()
    paths = AtlasPaths.from_directory(cfg.output_directory)
    
    if not paths.ontology.exists() or not paths.epistemology.exists():
        logger.info("Prerequisites missing; building atlas first")
        builder = AtlasBuilder(cfg)
        builder.build_all()
    
    logger.info("Building router maps...")
    router_maps_builder.build_router_maps(
        ontology_path=paths.ontology,
        epistemology_path=paths.epistemology,
        output_directory=cfg.output_directory,
    )
    logger.info("Router maps build complete")


def _build_complete() -> None:
    """Build complete atlas (core maps + router maps)."""
    logger.info("Building complete Router atlas (all maps + router maps)...")
    builder = AtlasBuilder()
    builder.build_all()
    _build_router_maps()
    logger.info("Complete atlas build finished")


def main() -> None:
    """Main entry point for command-line execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Default to building everything if no command provided
    if args.command is None:
        _build_complete()
    else:
        args.func(args)


if __name__ == "__main__":
    main()


