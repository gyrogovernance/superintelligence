# baby/constants/atlas_builder.py

"""
GyroSI Atlas Builder: Canonical Map Construction for Recursive Structural Intelligence

This module constructs the five fundamental maps that define the complete knowledge
structure of the GyroSI system:

1. Ontology (ontology_keys.npy): The finite manifold of all reachable states
2. Epistemology (epistemology.npy): The complete state transition table
3. Phenomenology (phenomenology_map.npy): The orbit partition structure
4. Orbit Sizes (orbit_sizes.npy): Cardinality measures for specificity
5. Theta (theta.npy): Angular divergence from the archetypal state

These maps emerge from the gyroscopic physics implemented in baby.kernel.governance
and represent the complete, measured ground truth of the system's state space.

References:
    - Common Governance Model (CGM): Axiomatic framework for recursive alignment
    - Ungar, A.A. (2008): Analytic Hyperbolic Geometry and gyrogroup structures

COMMANDS TO RUN:
  python -m baby.constants.atlas_builder ontology      # Build ontology and theta
  python -m baby.constants.atlas_builder epistemology  # Build epistemology 
  python -m baby.constants.atlas_builder phenomenology # Build phenomenology
  python -m baby.constants.atlas_builder all           # Build everything

"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Import physics engine (single source of truth for all transformations)
from baby.kernel import governance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type aliases for clarity
StateInteger = int
IntronValue = int
StateIndex = int
OrbitIndex = int


@dataclass(frozen=True)
class AtlasConfiguration:
    """Configuration parameters for atlas construction."""
    
    output_directory: Path = Path("memories/public/meta")
    
    # Expected invariants from measured ground truth
    expected_state_count: int = 788_986
    expected_diameter: int = 6
    expected_orbit_count: int = 256
    
    # Processing parameters
    bfs_chunk_size: int = 20_000
    epistemology_chunk_size: int = 10_000
    
    # Validation parameters
    strict_validation: bool = True
    
    def __post_init__(self) -> None:
        """Ensure output directory exists."""
        self.output_directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AtlasPaths:
    """File paths for the five canonical maps."""
    
    ontology: Path
    epistemology: Path
    theta: Path
    phenomenology: Path
    orbit_sizes: Path
    
    @classmethod
    def from_directory(cls, directory: Path) -> AtlasPaths:
        """Construct standard paths from base directory."""
        return cls(
            ontology=directory / "ontology_keys.npy",
            epistemology=directory / "epistemology.npy",
            theta=directory / "theta.npy",
            phenomenology=directory / "phenomenology_map.npy",
            orbit_sizes=directory / "orbit_sizes.npy"
        )


class ProgressMonitor:
    """Monitor and report progress for long-running operations."""
    
    def __init__(self, operation_name: str) -> None:
        """
        Initialize progress monitor.
        
        Args:
            operation_name: Description of the operation being monitored
        """
        self.operation_name = operation_name
        self.start_time = time.perf_counter()
        self.last_update_time = 0.0
        self.update_interval = 0.1  # seconds between updates
        
    def update(
        self,
        current: int,
        total: Optional[int] = None,
        metadata: Optional[str] = None
    ) -> None:
        """
        Update progress display.
        
        Args:
            current: Current progress count
            total: Total expected count (if known)
            metadata: Additional information to display
        """
        current_time = time.perf_counter()
        
        # Throttle updates to avoid console spam
        if current_time - self.last_update_time < self.update_interval:
            if total is None or current != total:
                return
                
        self.last_update_time = current_time
        elapsed = current_time - self.start_time
        rate = current / elapsed if elapsed > 0 else 0
        
        # Build progress message
        message_parts = [f"{self.operation_name}: {current:,}"]
        
        if total is not None:
            percentage = 100.0 * current / total if total > 0 else 0.0
            message_parts.append(f"/{total:,} ({percentage:.1f}%)")
            
        message_parts.append(f"| {rate:.0f}/s | {elapsed:.1f}s")
        
        if metadata:
            message_parts.append(f"| {metadata}")
            
        # Use carriage return for in-place updates
        print(f"\r{' '.join(message_parts):<80}", end="", flush=True)
        
    def complete(self) -> None:
        """Mark operation as complete and display final timing."""
        elapsed = time.perf_counter() - self.start_time
        print(f"\r{self.operation_name}: Complete in {elapsed:.1f}s{' '*60}")
        logger.info(f"{self.operation_name} completed in {elapsed:.1f}s")


class OntologyBuilder:
    """
    Constructs the ontology map through breadth-first exploration.
    
    The ontology represents the complete finite manifold of reachable states
    under the gyroscopic transform, starting from the archetypal tensor GENE_Mac_S.
    This manifold has been measured to contain exactly 788,986 states with
    diameter 6, representing the complete state space of the system.
    """
    
    def __init__(self, configuration: AtlasConfiguration) -> None:
        """
        Initialize ontology builder.
        
        Args:
            configuration: Atlas construction parameters
        """
        self.configuration = configuration
        
    def _expand_frontier_vectorized(
        self,
        frontier: NDArray[np.uint64]
    ) -> NDArray[np.uint64]:
        """
        Apply all 256 introns to frontier states in parallel.
        
        Args:
            frontier: Array of current frontier states
            
        Returns:
            Deduplicated array of successor states
        """
        # Apply all introns to all frontier states: shape (n_states, 256)
        successors = governance.apply_gyration_and_transform_all_introns(frontier)
        
        # Flatten and deduplicate
        unique_successors = np.unique(successors.ravel())
        
        return unique_successors
    
    def build(
        self,
        ontology_path: Path,
        theta_path: Path
    ) -> NDArray[np.uint64]:
        """
        Discover the complete state manifold via breadth-first search.
        
        Args:
            ontology_path: Output path for state integers
            theta_path: Output path for angular divergences
            
        Returns:
            Sorted array of all reachable state integers
            
        Raises:
            RuntimeError: If measured invariants don't match expected values
        """
        progress = ProgressMonitor("Discovering ontology")
        
        # Initialize from the archetypal state
        archetype_integer = int(governance.tensor_to_int(governance.GENE_Mac_S))
        logger.info(f"Starting BFS from archetype: {archetype_integer:012x}")
        
        # Track discovered states and current frontier
        discovered_states: set[StateInteger] = {archetype_integer}
        frontier_array = np.array([archetype_integer], dtype=np.uint64)
        
        # Track BFS statistics
        current_depth = 0
        layer_sizes: List[int] = []
        
        # Breadth-first exploration
        while frontier_array.size > 0:
            # Collect all unique successors from current frontier
            next_frontier_set: set[StateInteger] = set()
            
            # Process frontier in chunks to bound memory usage
            for chunk_start in range(0, frontier_array.size, self.configuration.bfs_chunk_size):
                chunk_end = min(chunk_start + self.configuration.bfs_chunk_size, frontier_array.size)
                frontier_chunk = frontier_array[chunk_start:chunk_end]
                
                # Apply all introns to chunk
                unique_successors = self._expand_frontier_vectorized(frontier_chunk)
                
                # Convert to Python set for efficient membership testing
                successor_set = set(int(s) for s in unique_successors)
                next_frontier_set.update(successor_set)
            
            # Remove already discovered states
            next_frontier_set.difference_update(discovered_states)
            
            if not next_frontier_set:
                break  # No new states discovered
                
            # Update for next iteration
            frontier_array = np.fromiter(next_frontier_set, dtype=np.uint64)
            discovered_states.update(next_frontier_set)
            current_depth += 1
            layer_sizes.append(frontier_array.size)
            
            # Update progress
            progress.update(
                len(discovered_states),
                None,  # Total not known a priori
                f"depth={current_depth}, frontier={frontier_array.size:,}"
            )
            
            # Safety check against infinite expansion
            if current_depth > 64:
                raise RuntimeError(f"BFS depth exceeded 64 - possible physics drift")
                
        progress.complete()
        
        # Validate measured invariants
        if self.configuration.strict_validation:
            if len(discovered_states) != self.configuration.expected_state_count:
                raise RuntimeError(
                    f"State count mismatch: expected {self.configuration.expected_state_count:,}, "
                    f"found {len(discovered_states):,}"
                )
            if current_depth != self.configuration.expected_diameter:
                raise RuntimeError(
                    f"Diameter mismatch: expected {self.configuration.expected_diameter}, "
                    f"found {current_depth}"
                )
        
        # Sort states for canonical ordering
        state_array = np.array(sorted(discovered_states), dtype=np.uint64)
        
        logger.info(
            f"Discovered {state_array.size:,} states with diameter {current_depth}. "
            f"Layer sizes: {layer_sizes}"
        )
        
        # Compute angular divergences from archetype
        theta_array = self._compute_theta_values(state_array, archetype_integer)
        
        # Save results
        np.save(ontology_path, state_array)
        np.save(theta_path, theta_array)
        
        # Log diagnostic information
        archetype_index = int(np.searchsorted(state_array, archetype_integer))
        logger.info(f"Archetype index: {archetype_index}, θ = {theta_array[archetype_index]:.6f}")
        logger.info(f"Index 0 state: {state_array[0]:012x}, θ = {theta_array[0]:.6f}")
        
        return state_array
    
    def _compute_theta_values(
        self,
        state_array: NDArray[np.uint64],
        archetype_integer: StateInteger
    ) -> NDArray[np.float32]:
        """
        Compute angular divergence from archetype for all states.
        
        The angular divergence θ = arccos(1 - 2H/48) where H is the Hamming
        distance between state tensors. This measures geometric alignment
        in the 48-dimensional space.
        
        Args:
            state_array: Array of state integers
            archetype_integer: The archetypal state integer
            
        Returns:
            Array of angular divergences in radians
        """
        n_states = state_array.size
        theta_array = np.empty(n_states, dtype=np.float32)
        
        # Precompute arccos lookup table for all possible Hamming distances
        hamming_to_theta = np.arccos(1 - 2 * np.arange(49) / 48.0).astype(np.float32)
        
        # Compute theta for each state
        for index, state in enumerate(state_array):
            hamming_distance = int(state ^ archetype_integer).bit_count()
            theta_array[index] = hamming_to_theta[hamming_distance]
            
        return theta_array


class EpistemologyBuilder:
    """
    Constructs the epistemology map (complete transition table).
    
    The epistemology encodes all possible state transitions under the 256 introns,
    forming a complete description of how knowledge changes in the system.
    This N×256 table is the fundamental dynamics of the gyroscopic transform.
    """
    
    def __init__(self, configuration: AtlasConfiguration) -> None:
        """
        Initialize epistemology builder.
        
        Args:
            configuration: Atlas construction parameters
        """
        self.configuration = configuration
        
    def build(
        self,
        ontology_path: Path,
        epistemology_path: Path,
        theta_path: Path
    ) -> None:
        """
        Build the complete state transition table.
        
        Args:
            ontology_path: Path to ontology map
            epistemology_path: Output path for transition table
            theta_path: Path to theta map (ensures dependency exists)
            
        Raises:
            FileNotFoundError: If required input files don't exist
            RuntimeError: If transitions produce states outside ontology
        """
        # Verify dependencies
        if not ontology_path.exists():
            raise FileNotFoundError(f"Ontology not found: {ontology_path}")
        if not theta_path.exists():
            raise FileNotFoundError(f"Theta map not found: {theta_path}")
            
        # Load ontology
        state_array = np.load(ontology_path, mmap_mode='r')
        n_states = int(state_array.size)
        
        # Create memory-mapped output array
        epistemology_array = self._create_memmap_array(epistemology_path, (n_states, 256))
        
        progress = ProgressMonitor("Building epistemology")
        
        # Process states in chunks for efficiency
        for chunk_start in range(0, n_states, self.configuration.epistemology_chunk_size):
            chunk_end = min(chunk_start + self.configuration.epistemology_chunk_size, n_states)
            chunk_states = state_array[chunk_start:chunk_end]
            
            # Compute all 256 successors for chunk
            successor_array = governance.apply_gyration_and_transform_all_introns(chunk_states)
            
            # Map successor states to ontology indices
            successor_indices = np.searchsorted(state_array, successor_array, side='left')
            
            # Validate closure (all successors must be in ontology)
            if np.any(successor_indices >= n_states) or np.any(state_array[successor_indices] != successor_array):
                self._handle_closure_violation(
                    chunk_states, successor_array, successor_indices, state_array, n_states
                )
                
            # Store in epistemology table
            epistemology_array[chunk_start:chunk_end, :] = successor_indices.astype(np.int32, copy=False)
            
            progress.update(chunk_end, n_states)
            
        # Ensure data is written to disk
        self._flush_memmap(epistemology_array)
        
        progress.complete()
        logger.info(f"Epistemology table saved: shape=({n_states:,}, 256)")
        
    def _create_memmap_array(
        self,
        path: Path,
        shape: Tuple[int, ...]
    ) -> NDArray[np.int32]:
        """
        Create a memory-mapped array for large data.
        
        Args:
            path: Output file path
            shape: Array dimensions
            
        Returns:
            Memory-mapped array
        """
        from numpy.lib.format import open_memmap
        
        array = open_memmap(
            str(path),
            dtype=np.int32,
            mode='w+',
            shape=shape
        )
        return array  # type: ignore[return-value]
        
    def _flush_memmap(self, array: NDArray[np.int32]) -> None:
        """
        Flush memory-mapped array to disk.
        
        Args:
            array: Memory-mapped array to flush
        """
        try:
            array.flush()  # type: ignore[attr-defined]
        except AttributeError:
            pass  # Not all array types have flush
            
    def _handle_closure_violation(
        self,
        chunk_states: NDArray[np.uint64],
        successor_array: NDArray[np.uint64],
        successor_indices: NDArray[np.intp],
        state_array: NDArray[np.uint64],
        n_states: int
    ) -> None:
        """
        Handle case where transition produces state outside ontology.
        
        Args:
            chunk_states: Current chunk of states
            successor_array: Computed successors
            successor_indices: Indices of successors in ontology
            state_array: Complete ontology
            n_states: Total number of states
            
        Raises:
            RuntimeError: Always (this is an error condition)
        """
        # Find first violation for detailed error message
        violation_mask = (successor_indices >= n_states) | (state_array[successor_indices] != successor_array)
        violation_positions = np.where(violation_mask)
        
        if violation_positions[0].size > 0:
            state_index = violation_positions[0][0]
            intron_index = violation_positions[1][0]
            
            source_state = chunk_states[state_index]
            target_state = successor_array[state_index, intron_index]
            
            raise RuntimeError(
                f"Closure violation: state {source_state:012x} + intron {intron_index:02x} "
                f"→ {target_state:012x} (not in ontology)"
            )


class PhenomenologyBuilder:
    """
    Constructs the phenomenology map (orbit partition structure).
    
    The phenomenology partitions the state space into 256 strongly connected
    components (orbits) under the full dynamics. Each orbit represents a
    phenomenological equivalence class where all states are mutually reachable.
    This structure encodes how things appear at the observable level.
    """
    
    def __init__(self, configuration: AtlasConfiguration) -> None:
        """
        Initialize phenomenology builder.
        
        Args:
            configuration: Atlas construction parameters
        """
        self.configuration = configuration
        
    def build(
        self,
        epistemology_path: Path,
        ontology_path: Path,
        phenomenology_path: Path,
        orbit_sizes_path: Path
    ) -> None:
        """
        Build phenomenology map and orbit size distribution.
        
        Args:
            epistemology_path: Path to epistemology table
            ontology_path: Path to ontology map
            phenomenology_path: Output path for phenomenology map
            orbit_sizes_path: Output path for orbit sizes
            
        Raises:
            FileNotFoundError: If required input files don't exist
            RuntimeError: If SCC computation fails
        """
        # Verify dependencies
        if not epistemology_path.exists() or not ontology_path.exists():
            raise FileNotFoundError("Missing epistemology and/or ontology")
            
        # Load required data
        epistemology_array = np.load(epistemology_path, mmap_mode='r')
        state_array = np.load(ontology_path, mmap_mode='r')
        n_states = int(state_array.size)
        
        logger.info("Computing strongly connected components...")
        
        # Run Tarjan's algorithm to find SCCs
        canonical_map, orbit_size_dict = self._compute_strongly_connected_components(
            epistemology_array, state_array
        )
        
        # Build orbit size array
        orbit_size_array = self._build_orbit_size_array(
            canonical_map, orbit_size_dict, n_states
        )
        
        # Save results
        np.save(phenomenology_path, canonical_map.astype(np.int32, copy=False))
        np.save(orbit_sizes_path, orbit_size_array)
        
        # Report statistics
        unique_representatives = np.unique(canonical_map)
        logger.info(f"Found {unique_representatives.size} orbits (strongly connected components)")
        
        if self.configuration.strict_validation:
            if unique_representatives.size != self.configuration.expected_orbit_count:
                logger.warning(
                    f"Orbit count mismatch: expected {self.configuration.expected_orbit_count}, "
                    f"found {unique_representatives.size}"
                )
                
    def _compute_strongly_connected_components(
        self,
        epistemology_array: NDArray[np.int32],
        state_array: NDArray[np.uint64]
    ) -> Tuple[NDArray[np.int32], Dict[int, int]]:
        """
        Find strongly connected components using Tarjan's algorithm.
        
        The representative for each SCC is chosen as the node with minimal
        48-bit state integer value (Traceable representative rule).
        
        Args:
            epistemology_array: Complete transition table
            state_array: Ontology of state integers
            
        Returns:
            Tuple of (canonical_map, orbit_size_dict) where:
                - canonical_map[i] gives representative index for state i
                - orbit_size_dict[rep] gives size of orbit with representative rep
        """
        n_states = int(epistemology_array.shape[0])
        
        # Tarjan algorithm state
        index_array = np.full(n_states, -1, dtype=np.int32)
        lowlink_array = np.zeros(n_states, dtype=np.int32)
        on_stack_array = np.zeros(n_states, dtype=bool)
        stack: List[StateIndex] = []
        
        # Results
        canonical_map = np.full(n_states, -1, dtype=np.int32)
        orbit_size_dict: Dict[StateIndex, int] = {}
        
        index_counter = 0
        
        def get_successors(state_index: StateIndex) -> NDArray[np.int32]:
            """Get all successor indices for a state."""
            return epistemology_array[state_index]
        
        # Process each unvisited node
        for root_index in range(n_states):
            if index_array[root_index] != -1:
                continue
                
            # Iterative depth-first search
            dfs_stack: List[Tuple[StateIndex, Iterator[int]]] = [
                (root_index, iter(get_successors(root_index)))
            ]
            
            index_array[root_index] = lowlink_array[root_index] = index_counter
            index_counter += 1
            stack.append(root_index)
            on_stack_array[root_index] = True
            
            while dfs_stack:
                current_index, successor_iterator = dfs_stack[-1]
                
                try:
                    # Process next successor
                    while True:
                        successor_index = int(next(successor_iterator))
                        
                        if index_array[successor_index] == -1:
                            # Unvisited successor - recurse
                            index_array[successor_index] = lowlink_array[successor_index] = index_counter
                            index_counter += 1
                            stack.append(successor_index)
                            on_stack_array[successor_index] = True
                            dfs_stack.append((successor_index, iter(get_successors(successor_index))))
                            break
                        elif on_stack_array[successor_index]:
                            # Back edge - update lowlink
                            if index_array[successor_index] < lowlink_array[current_index]:
                                lowlink_array[current_index] = index_array[successor_index]
                                
                except StopIteration:
                    # Finished with current node
                    dfs_stack.pop()
                    
                    if dfs_stack:
                        parent_index, _ = dfs_stack[-1]
                        if lowlink_array[current_index] < lowlink_array[parent_index]:
                            lowlink_array[parent_index] = lowlink_array[current_index]
                            
                    # Check if current node is SCC root
                    if lowlink_array[current_index] == index_array[current_index]:
                        # Extract strongly connected component
                        component: List[StateIndex] = []
                        
                        while True:
                            node_index = stack.pop()
                            on_stack_array[node_index] = False
                            component.append(node_index)
                            if node_index == current_index:
                                break
                                
                        # Find representative (minimal state integer)
                        component_array = np.array(component, dtype=np.int32)
                        component_states = state_array[component_array]
                        representative_local_index = int(np.argmin(component_states))
                        representative_index = int(component_array[representative_local_index])
                        
                        # Record mapping and size
                        canonical_map[component_array] = representative_index
                        orbit_size_dict[representative_index] = len(component)
                        
        # Validate completeness
        if not np.all(canonical_map >= 0):
            raise RuntimeError("Incomplete SCC computation - unassigned nodes remain")
            
        return canonical_map, orbit_size_dict
        
    def _build_orbit_size_array(
        self,
        canonical_map: NDArray[np.int32],
        orbit_size_dict: Dict[int, int],
        n_states: int
    ) -> NDArray[np.uint32]:
        """
        Build array of orbit sizes for all states.
        
        Args:
            canonical_map: Mapping from state to representative
            orbit_size_dict: Size of each orbit by representative
            n_states: Total number of states
            
        Returns:
            Array where element i gives size of orbit containing state i
        """
        orbit_size_array = np.zeros(n_states, dtype=np.uint32)
        
        for state_index in range(n_states):
            representative = int(canonical_map[state_index])
            orbit_size_array[state_index] = orbit_size_dict[representative]
            
        return orbit_size_array


class AtlasBuilder:
    """
    Main orchestrator for building all five canonical maps.
    
    The complete atlas construction follows a strict dependency order:
    1. Ontology and Theta (discovered together via BFS)
    2. Epistemology (requires ontology)
    3. Phenomenology and Orbit Sizes (requires epistemology)
    """
    
    def __init__(self, configuration: Optional[AtlasConfiguration] = None) -> None:
        """
        Initialize atlas builder.
        
        Args:
            configuration: Atlas construction parameters (uses defaults if None)
        """
        self.configuration = configuration or AtlasConfiguration()
        self.paths = AtlasPaths.from_directory(self.configuration.output_directory)
        
        self.ontology_builder = OntologyBuilder(self.configuration)
        self.epistemology_builder = EpistemologyBuilder(self.configuration)
        self.phenomenology_builder = PhenomenologyBuilder(self.configuration)
        
    def build_ontology(self) -> None:
        """Build ontology and theta maps."""
        logger.info("Building ontology and theta maps...")
        self.ontology_builder.build(self.paths.ontology, self.paths.theta)
        
    def build_epistemology(self) -> None:
        """Build epistemology map (requires ontology)."""
        if not self.paths.ontology.exists() or not self.paths.theta.exists():
            logger.info("Prerequisites missing - building ontology first...")
            self.build_ontology()
            
        logger.info("Building epistemology map...")
        self.epistemology_builder.build(
            self.paths.ontology,
            self.paths.epistemology,
            self.paths.theta
        )
        
    def build_phenomenology(self) -> None:
        """Build phenomenology and orbit size maps (requires epistemology)."""
        if not self.paths.epistemology.exists() or not self.paths.ontology.exists():
            logger.info("Prerequisites missing - building epistemology first...")
            self.build_epistemology()
            
        logger.info("Building phenomenology and orbit size maps...")
        self.phenomenology_builder.build(
            self.paths.epistemology,
            self.paths.ontology,
            self.paths.phenomenology,
            self.paths.orbit_sizes
        )
        
    def build_all(self) -> None:
        """Build all five canonical maps in dependency order."""
        logger.info("Starting complete atlas construction...")
        start_time = time.perf_counter()
        
        self.build_ontology()
        self.build_epistemology()
        self.build_phenomenology()
        
        elapsed = time.perf_counter() - start_time
        
        logger.info(f"Atlas construction complete in {elapsed:.1f}s")
        logger.info("Generated artifacts:")
        logger.info(f"  θ:            {self.paths.theta}")
        logger.info(f"  ontology:     {self.paths.ontology}")
        logger.info(f"  epistemology: {self.paths.epistemology}")
        logger.info(f"  phenomenology:{self.paths.phenomenology}")
        logger.info(f"  orbit_sizes:  {self.paths.orbit_sizes}")


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="GyroSI Atlas Builder: Construct canonical knowledge maps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m baby.constants.atlas_builder ontology      # Build ontology and theta
  python -m baby.constants.atlas_builder epistemology  # Build epistemology 
  python -m baby.constants.atlas_builder phenomenology # Build phenomenology
  python -m baby.constants.atlas_builder all           # Build everything
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Subcommand: ontology
    ontology_parser = subparsers.add_parser(
        'ontology',
        help='Build ontology and theta maps'
    )
    ontology_parser.set_defaults(func=lambda args: AtlasBuilder().build_ontology())
    
    # Subcommand: epistemology
    epistemology_parser = subparsers.add_parser(
        'epistemology',
        help='Build epistemology map'
    )
    epistemology_parser.set_defaults(func=lambda args: AtlasBuilder().build_epistemology())
    
    # Subcommand: phenomenology
    phenomenology_parser = subparsers.add_parser(
        'phenomenology',
        help='Build phenomenology and orbit size maps'
    )
    phenomenology_parser.set_defaults(func=lambda args: AtlasBuilder().build_phenomenology())
    
    # Subcommand: all
    all_parser = subparsers.add_parser(
        'all',
        help='Build all canonical maps'
    )
    all_parser.set_defaults(func=lambda args: AtlasBuilder().build_all())
    
    return parser


def main() -> None:
    """Main entry point for command-line execution."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Atlas construction failed: {e}")
        raise


if __name__ == "__main__":
    main()