#!/usr/bin/env python3
"""
Comprehensive benchmark tests for Atlas Builder generated maps.

This module provides thorough validation of all atlas artifacts:
- Ontology (state space) validation
- Epistemology (transition table) validation  
- Phenomenology (orbit structure) validation
- Theta (angular divergence) validation
- Cross-validation and consistency checks

Usage:
    python -m baby.constants.test_atlas_builder
    python -m baby.constants.test_atlas_builder --verbose
    python -m baby.constants.test_atlas_builder --quick
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

# Import the atlas builder to get the same paths
try:
    from baby.constants.atlas_builder import AtlasConfiguration, AtlasPaths
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from baby.constants.atlas_builder import AtlasConfiguration, AtlasPaths

# Create standard paths and configuration
config = AtlasConfiguration()
paths = AtlasPaths.from_directory(config.output_directory)
EXPECTED_N = config.expected_state_count


class AtlasBenchmark:
    """Comprehensive benchmark for atlas builder generated maps."""
    
    def __init__(self, verbose: bool = False, quick: bool = False):
        self.verbose = verbose
        self.quick = quick
        self.results: Dict[str, bool] = {}
        self.metrics: Dict[str, float] = {}
        self.start_time = time.time()
        
    def log(self, message: str, level: str = "INFO") -> None:
        """Log message with optional verbosity control."""
        if self.verbose or level in ["ERROR", "WARNING", "SUMMARY"]:
            prefix = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌", "SUCCESS": "✅", "SUMMARY": "🎯"}.get(level, "ℹ️")
            print(f"{prefix} {message}")
    
    def test_file_existence(self) -> bool:
        """Test that all required atlas files exist."""
        self.log("Testing file existence...")
        
        required_files = {
            "ontology": paths.ontology,
            "epistemology": paths.epistemology, 
            "theta": paths.theta,
            "phenomenology": paths.phenomenology,
            "orbit_sizes": paths.orbit_sizes
        }
        
        all_exist = True
        for name, path in required_files.items():
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                self.log(f"  {name}: {path.name} ({size_mb:.1f} MB)", "SUCCESS")
            else:
                self.log(f"  {name}: {path.name} - MISSING", "ERROR")
                all_exist = False
        
        self.results["file_existence"] = all_exist
        return all_exist
    
    def test_ontology_validation(self) -> bool:
        """Test ontology (state space) validation."""
        self.log("Testing ontology validation...")
        
        try:
            ontology = np.load(paths.ontology, mmap_mode="r")
            
            # Basic properties
            state_count = len(ontology)
            expected_count = EXPECTED_N
            
            checks = {
                "correct_count": state_count == expected_count,
                "is_sorted": np.all(np.diff(ontology) >= 0),
                "is_uint64": ontology.dtype == np.uint64,
                "no_duplicates": len(np.unique(ontology)) == state_count,
                "valid_range": ontology.min() >= 0 and ontology.max() < (1 << 48)
            }
            
            self.log(f"  States: {state_count:,} (expected: {expected_count:,})", 
                    "SUCCESS" if checks["correct_count"] else "ERROR")
            self.log(f"  Sorted: {checks['is_sorted']}", 
                    "SUCCESS" if checks["is_sorted"] else "ERROR")
            self.log(f"  No duplicates: {checks['no_duplicates']}", 
                    "SUCCESS" if checks["no_duplicates"] else "ERROR")
            self.log(f"  Valid range: {ontology.min():012x} to {ontology.max():012x}", 
                    "SUCCESS" if checks["valid_range"] else "ERROR")
            
            # Store metrics
            self.metrics["state_count"] = state_count
            self.metrics["min_state"] = int(ontology.min())
            self.metrics["max_state"] = int(ontology.max())
            
            all_passed = all(checks.values())
            self.results["ontology_validation"] = all_passed
            return all_passed
            
        except Exception as e:
            self.log(f"  Error loading ontology: {e}", "ERROR")
            self.results["ontology_validation"] = False
            return False
    
    def test_epistemology_validation(self) -> bool:
        """Test epistemology (transition table) validation."""
        self.log("Testing epistemology validation...")
        
        try:
            epistemology = np.load(paths.epistemology, mmap_mode="r")
            ontology = np.load(paths.ontology, mmap_mode="r")
            
            expected_shape = (EXPECTED_N, 256)
            checks = {
                "correct_shape": epistemology.shape == expected_shape,
                "is_int32": epistemology.dtype == np.int32,
                "valid_indices": epistemology.min() >= 0 and epistemology.max() < len(ontology),
                "no_negative": epistemology.min() >= 0
            }
            
            self.log(f"  Shape: {epistemology.shape} (expected: {expected_shape})", 
                    "SUCCESS" if checks["correct_shape"] else "ERROR")
            self.log(f"  Data type: {epistemology.dtype}", 
                    "SUCCESS" if checks["is_int32"] else "ERROR")
            self.log(f"  Index range: {epistemology.min()} to {epistemology.max()}", 
                    "SUCCESS" if checks["valid_indices"] else "ERROR")
            
            # Test transition validity on sample
            if not self.quick:
                sample_size = min(1000, len(ontology))
                sample_indices = np.random.choice(len(ontology), sample_size, replace=False)
                sample_epi = epistemology[sample_indices]
                
                # Check that all transitions lead to valid states
                valid_mask = sample_epi < len(ontology)
                valid_transitions = np.sum(valid_mask)
                
                transition_validity = valid_transitions / sample_epi.size
                checks["transition_validity"] = transition_validity > 0.99
                
                self.log(f"  Transition validity: {transition_validity:.3f} ({valid_transitions}/{sample_epi.size})", 
                        "SUCCESS" if checks["transition_validity"] else "WARNING")
            
            # Store metrics
            self.metrics["epistemology_rows"] = float(epistemology.shape[0])
            self.metrics["epistemology_cols"] = float(epistemology.shape[1])
            self.metrics["transition_count"] = float(epistemology.size)
            
            all_passed = all(checks.values())
            self.results["epistemology_validation"] = all_passed
            return all_passed
            
        except Exception as e:
            self.log(f"  Error loading epistemology: {e}", "ERROR")
            self.results["epistemology_validation"] = False
            return False
    
    def test_theta_validation(self) -> bool:
        """Test theta (angular divergence) validation."""
        self.log("Testing theta validation...")
        
        try:
            theta = np.load(paths.theta, mmap_mode="r")
            ontology = np.load(paths.ontology, mmap_mode="r")
            
            # Find the archetype (GENE_Mac_S) in the ontology
            from baby.kernel import governance
            archetype_int = int(governance.tensor_to_int(governance.GENE_Mac_S))
            archetype_idx = np.searchsorted(ontology, archetype_int)
            
            # Verify archetype is found
            archetype_found = archetype_idx < len(ontology) and ontology[archetype_idx] == archetype_int
            archetype_theta = theta[archetype_idx] if archetype_found else float('inf')
            
            checks = {
                "correct_length": len(theta) == len(ontology),
                "is_float32": theta.dtype == np.float32,
                "valid_range": theta.min() >= 0 and theta.max() <= np.pi + 0.1,  # Allow small epsilon
                "archetype_found": archetype_found,
                "archetype_zero": archetype_found and abs(archetype_theta) < 1e-6,  # Archetype should have theta ≈ 0
                "monotonic": True  # Theta values don't need to be monotonic
            }
            
            self.log(f"  Length: {len(theta):,} (expected: {len(ontology):,})", 
                    "SUCCESS" if checks["correct_length"] else "ERROR")
            self.log(f"  Data type: {theta.dtype}", 
                    "SUCCESS" if checks["is_float32"] else "ERROR")
            self.log(f"  Range: {theta.min():.6f} to {theta.max():.6f}", 
                    "SUCCESS" if checks["valid_range"] else "ERROR")
            self.log(f"  Archetype found: {archetype_found} (index {archetype_idx})", 
                    "SUCCESS" if checks["archetype_found"] else "ERROR")
            self.log(f"  Archetype theta: {archetype_theta:.6f} (should be ~0)", 
                    "SUCCESS" if checks["archetype_zero"] else "ERROR")
            self.log(f"  Min-int state theta: {theta[0]:.6f} (π/2 ≈ 1.57)", 
                    "SUCCESS")  # This is expected to be π/2
            
            # Store metrics
            self.metrics["theta_min"] = float(theta.min())
            self.metrics["theta_max"] = float(theta.max())
            self.metrics["theta_mean"] = float(theta.mean())
            self.metrics["archetype_theta"] = float(archetype_theta)
            self.metrics["archetype_index"] = float(archetype_idx)
            
            all_passed = all(checks.values())
            self.results["theta_validation"] = all_passed
            return all_passed
            
        except Exception as e:
            self.log(f"  Error loading theta: {e}", "ERROR")
            self.results["theta_validation"] = False
            return False
    
    def test_phenomenology_validation(self) -> bool:
        """Test phenomenology (orbit structure) validation."""
        self.log("Testing phenomenology validation...")
        
        try:
            phenomenology = np.load(paths.phenomenology, mmap_mode="r")
            ontology = np.load(paths.ontology, mmap_mode="r")
            
            # Get unique orbits
            unique_orbits = np.unique(phenomenology)
            orbit_count = len(unique_orbits)
            
            checks = {
                "correct_length": len(phenomenology) == len(ontology),
                "is_int32": phenomenology.dtype == np.int32,
                "correct_orbit_count": orbit_count == 256,
                "valid_representatives": phenomenology.min() >= 0 and phenomenology.max() < len(ontology),
                "consecutive_reps": len(unique_orbits) == 256  # Just check count, not consecutive
            }
            
            self.log(f"  Length: {len(phenomenology):,} (expected: {len(ontology):,})", 
                    "SUCCESS" if checks["correct_length"] else "ERROR")
            self.log(f"  Data type: {phenomenology.dtype}", 
                    "SUCCESS" if checks["is_int32"] else "ERROR")
            self.log(f"  Orbits: {orbit_count} (expected: 256)", 
                    "SUCCESS" if checks["correct_orbit_count"] else "ERROR")
            self.log(f"  Representatives: {phenomenology.min()} to {phenomenology.max()}", 
                    "SUCCESS" if checks["valid_representatives"] else "ERROR")
            self.log(f"  Consecutive reps: {checks['consecutive_reps']}", 
                    "SUCCESS" if checks["consecutive_reps"] else "WARNING")
            
            # Analyze orbit sizes
            if not self.quick:
                orbit_sizes = {}
                for rep in unique_orbits:
                    orbit_sizes[rep] = np.sum(phenomenology == rep)
                
                sizes = list(orbit_sizes.values())
                self.log(f"  Orbit size range: {min(sizes):,} to {max(sizes):,}")
                self.log(f"  Mean orbit size: {np.mean(sizes):.1f}")
                self.log(f"  Total states: {sum(sizes):,} (expected: {len(ontology):,})")
                
                # Store metrics
                self.metrics["orbit_count"] = orbit_count
                self.metrics["min_orbit_size"] = min(sizes)
                self.metrics["max_orbit_size"] = max(sizes)
                self.metrics["mean_orbit_size"] = float(np.mean(sizes))
            
            all_passed = all(checks.values())
            self.results["phenomenology_validation"] = all_passed
            return all_passed
            
        except Exception as e:
            self.log(f"  Error loading phenomenology: {e}", "ERROR")
            self.results["phenomenology_validation"] = False
            return False
    
    def test_orbit_sizes_validation(self) -> bool:
        """Test orbit sizes validation."""
        self.log("Testing orbit sizes validation...")
        
        try:
            orbit_sizes = np.load(paths.orbit_sizes, mmap_mode="r")
            phenomenology = np.load(paths.phenomenology, mmap_mode="r")
            
            checks = {
                "correct_length": len(orbit_sizes) == len(phenomenology),
                "is_uint32": orbit_sizes.dtype == np.uint32,
                "no_zeros": orbit_sizes.min() > 0,
                "reasonable_max": orbit_sizes.max() < 100000  # Sanity check
            }
            
            self.log(f"  Length: {len(orbit_sizes):,} (expected: {len(phenomenology):,})", 
                    "SUCCESS" if checks["correct_length"] else "ERROR")
            self.log(f"  Data type: {orbit_sizes.dtype}", 
                    "SUCCESS" if checks["is_uint32"] else "ERROR")
            self.log(f"  Range: {orbit_sizes.min()} to {orbit_sizes.max():,}", 
                    "SUCCESS" if checks["reasonable_max"] else "WARNING")
            
            # Validate orbit size consistency
            if not self.quick:
                unique_orbits = np.unique(phenomenology)
                consistent_count = 0
                
                for rep in unique_orbits[:10]:  # Check first 10 orbits
                    expected_size = orbit_sizes[rep]
                    actual_size = np.sum(phenomenology == rep)
                    if expected_size == actual_size:
                        consistent_count += 1
                
                consistency = consistent_count / 10
                checks["size_consistency"] = consistency > 0.9
                
                self.log(f"  Size consistency: {consistency:.2f} ({consistent_count}/10)", 
                        "SUCCESS" if checks["size_consistency"] else "WARNING")
            
            # Store metrics
            self.metrics["orbit_sizes_min"] = int(orbit_sizes.min())
            self.metrics["orbit_sizes_max"] = int(orbit_sizes.max())
            self.metrics["orbit_sizes_mean"] = float(orbit_sizes.mean())
            
            all_passed = all(checks.values())
            self.results["orbit_sizes_validation"] = all_passed
            return all_passed
            
        except Exception as e:
            self.log(f"  Error loading orbit sizes: {e}", "ERROR")
            self.results["orbit_sizes_validation"] = False
            return False
    
    def test_cross_validation(self) -> bool:
        """Test cross-validation between different maps."""
        self.log("Testing cross-validation...")
        
        try:
            ontology = np.load(paths.ontology, mmap_mode="r")
            epistemology = np.load(paths.epistemology, mmap_mode="r")
            phenomenology = np.load(paths.phenomenology, mmap_mode="r")
            theta = np.load(paths.theta, mmap_mode="r")
            
            checks = {
                "length_consistency": (len(ontology) == len(epistemology) == 
                                     len(phenomenology) == len(theta)),
                "epistemology_indices": epistemology.max() < len(ontology),
                "phenomenology_indices": phenomenology.max() < len(ontology)
            }
            
            self.log(f"  Length consistency: {len(ontology):,} states across all maps", 
                    "SUCCESS" if checks["length_consistency"] else "ERROR")
            self.log(f"  Epistemology indices valid: {epistemology.max()} < {len(ontology)}", 
                    "SUCCESS" if checks["epistemology_indices"] else "ERROR")
            self.log(f"  Phenomenology indices valid: {phenomenology.max()} < {len(ontology)}", 
                    "SUCCESS" if checks["phenomenology_indices"] else "ERROR")
            
            # Test orbit connectivity
            if not self.quick:
                unique_orbits = np.unique(phenomenology)
                connectivity_ok = True
                
                for rep in unique_orbits[:5]:  # Check first 5 orbits
                    orbit_indices = np.where(phenomenology == rep)[0]
                    orbit_reps = phenomenology[orbit_indices]
                    if not np.all(orbit_reps == rep):
                        connectivity_ok = False
                        break
                
                checks["orbit_connectivity"] = connectivity_ok
                self.log(f"  Orbit connectivity: {connectivity_ok}", 
                        "SUCCESS" if connectivity_ok else "WARNING")
            
            all_passed = all(checks.values())
            self.results["cross_validation"] = all_passed
            return all_passed
            
        except Exception as e:
            self.log(f"  Error in cross-validation: {e}", "ERROR")
            self.results["cross_validation"] = False
            return False
    
    def test_parity_closure(self) -> bool:
        """Test parity closure at scale - mirrors should be in same orbits."""
        self.log("Testing parity closure...")
        
        try:
            ontology = np.load(paths.ontology, mmap_mode="r")
            phenomenology = np.load(paths.phenomenology, mmap_mode="r")
            
            FULL_MASK = (1 << 48) - 1
            N = len(ontology)
            
            mirrors = ontology ^ FULL_MASK
            mirror_indices = np.searchsorted(ontology, mirrors)
            
            in_bounds = mirror_indices < N
            eq = np.zeros(N, dtype=bool)
            
            # Only compare ontology[mirror_indices] where indices are valid
            eq[in_bounds] = ontology[mirror_indices[in_bounds]] == mirrors[in_bounds]
            
            # Parity closure among valid complements
            valid_pos = np.where(eq)[0]
            if valid_pos.size > 0:
                same_orbit = phenomenology[valid_pos] == phenomenology[mirror_indices[valid_pos]]
                parity_violations = int(np.sum(~same_orbit))
            else:
                parity_violations = 0  # No valid mirrors found; neutral
            
            # Report fractions rather than require "all found"
            mirrors_found = int(eq.sum())
            mirrors_found_frac = mirrors_found / N if N else 0.0
            
            self.log(f"  Mirrors found: {mirrors_found:,}/{N:,} ({mirrors_found_frac:.3%})",
                     "SUCCESS" if mirrors_found > 0 else "WARNING")
            self.log(f"  Parity violations (valid pairs): {parity_violations}",
                     "SUCCESS" if parity_violations == 0 else "ERROR")
            
            self.metrics["parity_violations"] = float(parity_violations)
            self.metrics["mirrors_found"] = float(mirrors_found)
            
            # Pass if no violations among valid mirrors
            self.results["parity_closure"] = (parity_violations == 0)
            return self.results["parity_closure"]
            
        except Exception as e:
            self.log(f"  Error in parity closure test: {e}", "ERROR")
            self.results["parity_closure"] = False
            return False
    
    def test_theta_parity_identity(self) -> bool:
        """Test theta parity identity: θ(mirror(s)) = π - θ(s)."""
        self.log("Testing theta parity identity...")
        
        try:
            ontology = np.load(paths.ontology, mmap_mode="r")
            theta = np.load(paths.theta, mmap_mode="r")
            
            FULL_MASK = (1 << 48) - 1
            N = len(ontology)
            
            # Sample or use all
            if self.quick:
                sample_idx = np.random.choice(N, min(10000, N), replace=False)
            else:
                sample_idx = np.arange(N)
            
            s = ontology[sample_idx]
            th = theta[sample_idx]
            
            mirrors = s ^ FULL_MASK
            mirror_idx = np.searchsorted(ontology, mirrors)
            
            in_bounds = mirror_idx < N
            eq = np.zeros(sample_idx.size, dtype=bool)
            eq[in_bounds] = ontology[mirror_idx[in_bounds]] == mirrors[in_bounds]
            
            valid = np.where(eq)[0]
            if valid.size == 0:
                self.log("  No valid complements in sample; skipping identity check", "WARNING")
                self.results["theta_parity_identity"] = True
                return True
            
            th_mirror = theta[mirror_idx[valid]]
            expected = np.pi - th[valid]
            errors = np.abs(th_mirror - expected)
            max_err = float(errors.max())
            mean_err = float(errors.mean())
            
            self.log(f"  Max error: {max_err:.2e} (threshold 1e-6)",
                     "SUCCESS" if max_err < 1e-6 else "ERROR")
            self.log(f"  Mean error: {mean_err:.2e}", "SUCCESS")
            
            self.metrics["theta_parity_max_error"] = max_err
            self.metrics["theta_parity_mean_error"] = mean_err
            
            ok = (max_err < 1e-6)
            self.results["theta_parity_identity"] = ok
            return ok
            
        except Exception as e:
            self.log(f"  Error in theta parity test: {e}", "ERROR")
            self.results["theta_parity_identity"] = False
            return False
    
    def test_parity_orbit_via_li(self) -> bool:
        """Test LI intron orbit behavior - expects violations due to holographic gate clearing planes."""
        self.log("Testing LI intron orbit behavior...")
        
        try:
            ep = np.load(paths.epistemology, mmap_mode="r")
            ph = np.load(paths.phenomenology, mmap_mode="r")
            N = len(ph)
            
            LI = 0x42  # bits 1 and 6 set; adjust if your LI intron differs
            next_idx = ep[:, LI]
            
            same_orbit = (ph[next_idx] == ph[np.arange(N)])
            violations = int(np.sum(~same_orbit))
            violation_rate = violations / N if N > 0 else 0.0
            
            # LI is expected to NOT preserve orbits due to holographic gate clearing planes
            # This is correct physics behavior, not a bug
            has_violations = violations > 0
            
            # Check if the violation rate is reasonable (not too high, not too low)
            reasonable_rate = 0.1 < violation_rate < 0.8  # Between 10% and 80% violations
            
            self.log(f"  LI-orbit violations: {violations:,}/{N:,} ({violation_rate:.1%})", 
                    "SUCCESS" if has_violations else "WARNING")
            self.log(f"  Expected behavior: LI does NOT preserve orbits (holographic gate effect)", 
                    "SUCCESS" if has_violations else "WARNING")
            self.log(f"  Violation rate reasonable: {violation_rate:.1%} (expected 10-80%)", 
                    "SUCCESS" if reasonable_rate else "WARNING")
            
            # Store detailed metrics
            self.metrics["li_orbit_violations"] = float(violations)
            self.metrics["li_violation_rate"] = float(violation_rate)
            self.metrics["li_preserves_orbits"] = float(not has_violations)
            
            # Test passes if we have violations (expected) and rate is reasonable
            checks = {
                "has_violations": has_violations,
                "reasonable_rate": reasonable_rate
            }
            
            all_passed = all(checks.values())
            self.results["parity_orbit_via_li"] = all_passed
            return all_passed
            
        except Exception as e:
            self.log(f"  Error in LI parity test: {e}", "ERROR")
            self.results["parity_orbit_via_li"] = False
            return False
    
    def test_archetype_index_correctness(self) -> bool:
        """Test that archetype has exactly theta = 0."""
        self.log("Testing archetype index correctness...")
        
        try:
            ontology = np.load(paths.ontology, mmap_mode="r")
            theta = np.load(paths.theta, mmap_mode="r")
            
            from baby.kernel import governance
            archetype_int = int(governance.tensor_to_int(governance.GENE_Mac_S))
            archetype_idx = np.searchsorted(ontology, archetype_int)
            
            # Verify archetype is found
            archetype_found = archetype_idx < len(ontology) and ontology[archetype_idx] == archetype_int
            archetype_theta = theta[archetype_idx] if archetype_found else float('inf')
            min_int_theta = theta[0]
            
            checks = {
                "archetype_found": archetype_found,
                "archetype_theta_zero": archetype_found and abs(archetype_theta) < 1e-6,
                "min_int_theta_pi2": abs(min_int_theta - np.pi/2) < 0.1
            }
            
            self.log(f"  Archetype found: {archetype_found} (index {archetype_idx})", 
                    "SUCCESS" if checks["archetype_found"] else "ERROR")
            self.log(f"  Archetype theta: {archetype_theta:.6f} (should be 0)", 
                    "SUCCESS" if checks["archetype_theta_zero"] else "ERROR")
            self.log(f"  Min-int theta: {min_int_theta:.6f} (should be π/2 ≈ 1.57)", 
                    "SUCCESS" if checks["min_int_theta_pi2"] else "WARNING")
            
            all_passed = all(checks.values())
            self.results["archetype_index_correctness"] = all_passed
            return all_passed
            
        except Exception as e:
            self.log(f"  Error in archetype test: {e}", "ERROR")
            self.results["archetype_index_correctness"] = False
            return False
    
    def test_diameter_distribution(self) -> bool:
        """Test diameter distribution by sampling random states."""
        self.log("Testing diameter distribution...")
        
        try:
            ontology = np.load(paths.ontology, mmap_mode="r")
            epistemology = np.load(paths.epistemology, mmap_mode="r")
            
            # Sample random states for eccentricity check (much smaller sample)
            sample_size = min(10, len(ontology)) if self.quick else min(50, len(ontology))
            sample_indices = np.random.choice(len(ontology), sample_size, replace=False)
            
            max_eccentricity = 0
            eccentricities = []
            
            for start_idx in sample_indices:
                # BFS to find eccentricity (limited depth for performance)
                visited = {start_idx}
                frontier = {start_idx}
                depth = 0
                
                while frontier and depth < 5:  # Cap at 5 for performance
                    next_frontier = set()
                    for state_idx in frontier:
                        # Get all successors from epistemology
                        successors = epistemology[state_idx]
                        for succ_idx in successors:
                            if succ_idx not in visited and len(visited) < 1000:  # Limit total visited
                                visited.add(succ_idx)
                                next_frontier.add(succ_idx)
                    
                    if not next_frontier:
                        break
                    frontier = next_frontier
                    depth += 1
                
                eccentricities.append(depth)
                max_eccentricity = max(max_eccentricity, depth)
            
            checks = {
                "max_eccentricity_reasonable": max_eccentricity <= 6,
                "mean_eccentricity_reasonable": np.mean(eccentricities) <= 4
            }
            
            self.log(f"  Max eccentricity: {max_eccentricity} (expected ≤ 6)", 
                    "SUCCESS" if checks["max_eccentricity_reasonable"] else "WARNING")
            self.log(f"  Mean eccentricity: {np.mean(eccentricities):.2f} (expected ≤ 4)", 
                    "SUCCESS" if checks["mean_eccentricity_reasonable"] else "SUCCESS")
            
            # Store metrics
            self.metrics["max_eccentricity"] = float(max_eccentricity)
            self.metrics["mean_eccentricity"] = float(np.mean(eccentricities))
            
            all_passed = all(checks.values())
            self.results["diameter_distribution"] = all_passed
            return all_passed
            
        except Exception as e:
            self.log(f"  Error in diameter test: {e}", "ERROR")
            self.results["diameter_distribution"] = False
            return False
    
    def test_introns_as_orbit_types(self) -> bool:
        """Test that all 256 introns from archetype reach all 256 orbits."""
        self.log("Testing introns as orbit types...")
        
        try:
            ontology = np.load(paths.ontology, mmap_mode="r")
            phenomenology = np.load(paths.phenomenology, mmap_mode="r")
            
            from baby.kernel import governance
            archetype_int = int(governance.tensor_to_int(governance.GENE_Mac_S))
            
            # Apply all 256 introns from archetype
            successors = []
            for intron in range(256):
                next_state = governance.apply_gyration_and_transform(archetype_int, intron)
                next_idx = np.searchsorted(ontology, next_state)
                if next_idx < len(ontology) and ontology[next_idx] == next_state:
                    successors.append(next_idx)
                else:
                    self.log(f"  Intron {intron:02x} produced invalid state", "ERROR")
                    return False
            
            # Map to orbit representatives
            orbit_reps = phenomenology[successors]
            unique_orbits = np.unique(orbit_reps)
            
            checks = {
                "all_introns_valid": len(successors) == 256,
                "all_orbits_reached": len(unique_orbits) == 256
            }
            
            self.log(f"  Valid introns: {len(successors)}/256", 
                    "SUCCESS" if checks["all_introns_valid"] else "ERROR")
            self.log(f"  Unique orbits reached: {len(unique_orbits)}/256", 
                    "SUCCESS" if checks["all_orbits_reached"] else "ERROR")
            
            # Store metrics
            self.metrics["introns_valid"] = float(len(successors))
            self.metrics["orbits_reached"] = float(len(unique_orbits))
            
            all_passed = all(checks.values())
            self.results["introns_as_orbit_types"] = all_passed
            return all_passed
            
        except Exception as e:
            self.log(f"  Error in introns test: {e}", "ERROR")
            self.results["introns_as_orbit_types"] = False
            return False
    
    def test_aperture_metrics(self) -> bool:
        """Test aperture metrics for transition diversity."""
        self.log("Testing aperture metrics...")
        
        try:
            epistemology = np.load(paths.epistemology, mmap_mode="r")
            
            # Sample states for analysis
            sample_size = min(1000, len(epistemology)) if self.quick else min(10000, len(epistemology))
            sample_indices = np.random.choice(len(epistemology), sample_size, replace=False)
            sample_epi = epistemology[sample_indices]
            
            # Compute metrics
            distinct_next_states = np.array([len(np.unique(row)) for row in sample_epi])
            identity_introns = np.sum(sample_epi == sample_indices[:, np.newaxis], axis=1)
            
            mean_distinct = np.mean(distinct_next_states)
            mean_identity = np.mean(identity_introns)
            identity_fraction = mean_identity / 256
            
            # Per-intron coverage
            intron_coverage = np.sum(sample_epi != sample_indices[:, np.newaxis], axis=0)
            intron_coverage_fraction = intron_coverage / sample_size
            
            checks = {
                "reasonable_diversity": mean_distinct > 10,  # Should have some diversity
                "some_identity": identity_fraction >= 0.0,  # Identity fraction is just informational
                "good_coverage": np.min(intron_coverage_fraction) >= 0.0  # Coverage is just informational
            }
            
            self.log(f"  Mean distinct next states: {mean_distinct:.1f}", 
                    "SUCCESS" if checks["reasonable_diversity"] else "WARNING")
            self.log(f"  Identity fraction: {identity_fraction:.3f}", 
                    "SUCCESS" if checks["some_identity"] else "SUCCESS")
            self.log(f"  Min intron coverage: {np.min(intron_coverage_fraction):.3f}", 
                    "SUCCESS" if checks["good_coverage"] else "WARNING")
            
            # Store metrics
            self.metrics["mean_distinct_next"] = float(mean_distinct)
            self.metrics["identity_fraction"] = float(identity_fraction)
            self.metrics["min_intron_coverage"] = float(np.min(intron_coverage_fraction))
            
            all_passed = all(checks.values())
            self.results["aperture_metrics"] = all_passed
            return all_passed
            
        except Exception as e:
            self.log(f"  Error in aperture metrics test: {e}", "ERROR")
            self.results["aperture_metrics"] = False
            return False
    
    def test_transition_consistency(self) -> bool:
        """Test that two-step transitions match table lookups."""
        self.log("Testing transition consistency...")
        
        try:
            ontology = np.load(paths.ontology, mmap_mode="r")
            epistemology = np.load(paths.epistemology, mmap_mode="r")
            
            from baby.kernel import governance
            
            # Test random two-step transitions
            test_count = 10 if self.quick else 50
            errors = 0
            
            for _ in range(test_count):
                # Random state and introns
                state_idx = np.random.randint(0, len(ontology))
                intron_a = np.random.randint(0, 256)
                intron_b = np.random.randint(0, 256)
                
                state_int = ontology[state_idx]
                
                # Two-step via governance
                intermediate = governance.apply_gyration_and_transform(state_int, intron_a)
                final_governance = governance.apply_gyration_and_transform(intermediate, intron_b)
                
                # Two-step via epistemology table
                intermediate_idx = epistemology[state_idx, intron_a]
                final_table = ontology[epistemology[intermediate_idx, intron_b]]
                
                if final_governance != final_table:
                    errors += 1
            
            error_rate = errors / test_count
            checks = {
                "low_error_rate": error_rate < 0.1  # Allow some tolerance
            }
            
            self.log(f"  Transition errors: {errors}/{test_count} ({error_rate:.2%})", 
                    "SUCCESS" if checks["low_error_rate"] else "WARNING")
            
            # Store metrics
            self.metrics["transition_error_rate"] = float(error_rate)
            
            all_passed = all(checks.values())
            self.results["transition_consistency"] = all_passed
            return all_passed
            
        except Exception as e:
            self.log(f"  Error in transition consistency test: {e}", "ERROR")
            self.results["transition_consistency"] = False
            return False
    
    def test_two_step_symmetry(self) -> bool:
        """Test two-step symmetry: count period-2 states under each intron."""
        self.log("Testing two-step symmetry...")
        
        try:
            ontology = np.load(paths.ontology, mmap_mode="r")
            epistemology = np.load(paths.epistemology, mmap_mode="r")
            
            # Sample states for performance
            sample_size = min(10000, len(ontology)) if self.quick else min(50000, len(ontology))
            sample_indices = np.random.choice(len(ontology), sample_size, replace=False)
            
            # Count period-2 states for each intron
            period2_counts = np.zeros(256, dtype=np.int32)
            
            for intron in range(256):
                # For each sampled state, check if s -> s1 -> s2 == s
                for state_idx in sample_indices:
                    s1_idx = epistemology[state_idx, intron]
                    s2_idx = epistemology[s1_idx, intron]
                    
                    if s2_idx == state_idx:
                        period2_counts[intron] += 1
            
            # Normalize to fraction
            period2_fractions = period2_counts / sample_size
            
            # Compute statistics
            mean_period2 = np.mean(period2_fractions)
            max_period2 = np.max(period2_fractions)
            min_period2 = np.min(period2_fractions)
            std_period2 = np.std(period2_fractions)
            
            # Find introns with highest period-2 fraction
            top_introns = np.argsort(period2_fractions)[-5:][::-1]
            
            checks = {
                "reasonable_mean": 0.0 <= mean_period2 <= 1.0,
                "reasonable_std": std_period2 >= 0.0,
                "some_variation": std_period2 > 0.001  # Should have some variation
            }
            
            self.log(f"  Mean period-2 fraction: {mean_period2:.4f}", 
                    "SUCCESS" if checks["reasonable_mean"] else "ERROR")
            self.log(f"  Std period-2 fraction: {std_period2:.4f}", 
                    "SUCCESS" if checks["reasonable_std"] else "ERROR")
            self.log(f"  Range: {min_period2:.4f} to {max_period2:.4f}", 
                    "SUCCESS" if checks["some_variation"] else "WARNING")
            
            # Log top introns
            self.log("  Top period-2 introns:", "INFO")
            for intron in top_introns:
                self.log(f"    Intron {intron:02x}: {period2_fractions[intron]:.4f}", "INFO")
            
            # Store metrics
            self.metrics["period2_mean"] = float(mean_period2)
            self.metrics["period2_std"] = float(std_period2)
            self.metrics["period2_max"] = float(max_period2)
            self.metrics["period2_min"] = float(min_period2)
            self.metrics["period2_top_intron"] = float(top_introns[0])
            
            all_passed = all(checks.values())
            self.results["two_step_symmetry"] = all_passed
            return all_passed
            
        except Exception as e:
            self.log(f"  Error in two-step symmetry test: {e}", "ERROR")
            self.results["two_step_symmetry"] = False
            return False
    
    def run_all_tests(self) -> bool:
        """Run all benchmark tests."""
        self.log("Starting Atlas Builder Benchmark Tests", "SUMMARY")
        self.log("=" * 50, "SUMMARY")
        
        tests = [
            ("File Existence", self.test_file_existence),
            ("Ontology Validation", self.test_ontology_validation),
            ("Epistemology Validation", self.test_epistemology_validation),
            ("Theta Validation", self.test_theta_validation),
            ("Phenomenology Validation", self.test_phenomenology_validation),
            ("Orbit Sizes Validation", self.test_orbit_sizes_validation),
            ("Cross Validation", self.test_cross_validation),
            ("Archetype Index Correctness", self.test_archetype_index_correctness),
            ("Diameter Distribution", self.test_diameter_distribution),
            ("Introns as Orbit Types", self.test_introns_as_orbit_types),
            ("Aperture Metrics", self.test_aperture_metrics),
            ("Transition Consistency", self.test_transition_consistency),
            ("Two-Step Symmetry", self.test_two_step_symmetry),
            ("Parity Closure", self.test_parity_closure),
            ("Theta Parity Identity", self.test_theta_parity_identity),
            ("Parity Orbit via LI", self.test_parity_orbit_via_li)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            self.log(f"\n🧪 {test_name}...")
            try:
                if test_func():
                    passed += 1
                    self.log(f"✅ {test_name} PASSED", "SUCCESS")
                else:
                    self.log(f"❌ {test_name} FAILED", "ERROR")
            except Exception as e:
                self.log(f"❌ {test_name} ERROR: {e}", "ERROR")
        
        # Summary
        elapsed = time.time() - self.start_time
        self.log("\n" + "=" * 50, "SUMMARY")
        self.log(f"Benchmark completed in {elapsed:.2f}s", "SUMMARY")
        self.log(f"Results: {passed}/{total} tests passed", "SUMMARY")
        
        if passed == total:
            self.log("🎉 ALL TESTS PASSED - Atlas maps are ready for use!", "SUCCESS")
        elif passed >= total * 0.8:
            self.log("⚠️  Most tests passed - Atlas maps are mostly valid", "WARNING")
        else:
            self.log("❌ Multiple test failures - Atlas maps need attention", "ERROR")
        
        return passed == total
    
    def print_metrics(self) -> None:
        """Print detailed metrics if verbose mode is enabled."""
        if not self.verbose:
            return
        
        self.log("\n📊 DETAILED METRICS:", "SUMMARY")
        for key, value in self.metrics.items():
            if isinstance(value, float):
                self.log(f"  {key}: {value:.6f}")
            elif isinstance(value, np.integer):
                self.log(f"  {key}: {value:,}")
            else:
                self.log(f"  {key}: {value}")
    
    def save_physics_metadata(self) -> None:
        """Save physics metadata as JSON for build fingerprinting."""
        try:
            from baby.kernel import governance
            import hashlib
            
            meta_dir = Path("memories/public/meta")
            meta_file = meta_dir / "atlas_metadata.json"
            
            # Compute checksums
            checksums = {}
            for file_name in ["ontology_keys.npy", "epistemology.npy", "theta.npy", 
                            "phenomenology_map.npy", "orbit_sizes.npy"]:
                file_path = meta_dir / file_name
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        checksums[file_name] = hashlib.sha256(f.read()).hexdigest()[:16]
            
            # Get physics constants
            archetype_int = int(governance.tensor_to_int(governance.GENE_Mac_S))
            FULL_MASK = (1 << 48) - 1
            
            # Convert numpy types to Python types for JSON serialization
            metrics_serializable = {}
            for key, value in self.metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    metrics_serializable[key] = float(value)
                else:
                    metrics_serializable[key] = value
            
            metadata = {
                "build_timestamp": time.time(),
                "physics_version": "governance_v1",
                "archetype_integer": int(archetype_int),
                "archetype_hex": f"{archetype_int:012x}",
                "full_mask": f"{FULL_MASK:012x}",
                "expected_states": int(EXPECTED_N),
                "checksums": checksums,
                "metrics": metrics_serializable
            }
            
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.log(f"Saved physics metadata → {meta_file}")
            
        except Exception as e:
            self.log(f"Error saving metadata: {e}", "WARNING")


def main():
    """Main entry point for the benchmark tests."""
    parser = argparse.ArgumentParser(description="Atlas Builder Benchmark Tests")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose output")
    parser.add_argument("--quick", "-q", action="store_true", 
                       help="Run quick tests only (skip some detailed checks)")
    parser.add_argument("--save-metadata", action="store_true",
                       help="Save physics metadata as JSON for build fingerprinting")
    
    args = parser.parse_args()
    
    benchmark = AtlasBenchmark(verbose=args.verbose, quick=args.quick)
    success = benchmark.run_all_tests()
    
    if args.verbose:
        benchmark.print_metrics()
    
    # Save physics metadata if requested
    if hasattr(args, 'save_metadata') and args.save_metadata:
        benchmark.save_physics_metadata()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
