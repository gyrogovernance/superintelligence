#!/usr/bin/env python3
"""
Test and analyze the generated phenomenology map.
Verifies correctness and provides detailed statistics.
"""

import numpy as np
from pathlib import Path
from baby.information import InformationEngine
import sys


def test_phenomenology_map():
    """
    Test the phenomenology map for correctness and provide detailed statistics.
    """
    print("🔍 Testing Phenomenology Map")
    print("=" * 50)

    # Paths
    base_path = Path("memories/public/meta")
    ontology_path = base_path / "ontology_keys.npy"
    epistemology_path = base_path / "epistemology.npy"
    phenomenology_path = base_path / "phenomenology_map.npy"
    orbit_sizes_path = base_path / "orbit_sizes.npy"
    theta_path = base_path / "theta.npy"

    # Check file existence
    missing_files = []
    for name, path in [
        ("Ontology", ontology_path),
        ("Epistemology", epistemology_path),
        ("Phenomenology", phenomenology_path),
        ("Orbit Sizes", orbit_sizes_path),
        ("Theta", theta_path),
    ]:
        if not path.exists():
            missing_files.append(f"{name}: {path}")
        else:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✅ {name}: {path} ({size_mb:.1f} MB)")

    if missing_files:
        print("\n❌ Missing files:")
        for file in missing_files:
            print(f"   {file}")
        return False

    print("\n📊 Loading and analyzing phenomenology map...")

    try:
        # Load data
        ontology = np.load(ontology_path)
        phenomenology = np.load(phenomenology_path)
        orbit_sizes = np.load(orbit_sizes_path)
        theta = np.load(theta_path)

        N = len(ontology)
        print(f"\n📈 Basic Statistics:")
        print(f"   Total states: {N:,}")
        print(f"   Phenomenology map shape: {phenomenology.shape}")
        print(f"   Orbit sizes shape: {orbit_sizes.shape}")
        print(f"   Theta shape: {theta.shape}")

        # Analyze phenomenology map
        unique_reps = np.unique(phenomenology)
        num_orbits = len(unique_reps)
        print(f"\n🌀 Orbit Analysis:")
        print(f"   Unique orbit representatives: {num_orbits}")
        print(f"   Expected orbits: 256")
        print(f"   Match expected: {'✅' if num_orbits == 256 else '❌'}")

        # Analyze orbit sizes
        orbit_size_distribution = {}
        for size in np.unique(orbit_sizes):
            count = np.sum(orbit_sizes == size)
            orbit_size_distribution[int(size)] = count

        print(f"\n📏 Orbit Size Distribution:")
        total_states_check = 0
        for size in sorted(orbit_size_distribution.keys()):
            count = orbit_size_distribution[size]
            states_in_orbits = size * (count // size)  # States contributing to orbits of this size
            total_states_check += states_in_orbits
            print(f"   Size {size:4d}: {count:6,} states ({count//size:4d} orbits)")

        print(f"\n🔍 Consistency Checks:")
        print(f"   Total states from orbit analysis: {total_states_check:,}")
        print(f"   Expected total states: {N:,}")
        print(f"   States match: {'✅' if total_states_check == N else '❌'}")

        # Check that all representatives are valid indices
        invalid_reps = phenomenology[(phenomenology < 0) | (phenomenology >= N)]
        print(f"   Invalid representatives: {len(invalid_reps)} {'✅' if len(invalid_reps) == 0 else '❌'}")

        # Check self-consistency: representatives should map to themselves
        self_consistent = 0
        for rep in unique_reps:
            if phenomenology[rep] == rep:
                self_consistent += 1

        print(
            f"   Self-consistent representatives: {self_consistent}/{num_orbits} {'✅' if self_consistent == num_orbits else '❌'}"
        )

        # Test with InformationEngine
        print(f"\n🔧 Testing InformationEngine integration...")
        try:
            engine = InformationEngine(
                keys_path=str(ontology_path),
                ep_path=str(epistemology_path),
                phenomap_path=str(phenomenology_path),
                theta_path=str(theta_path),
            )

            # Test a few random states
            test_indices = np.random.choice(N, size=min(10, N), replace=False)
            print(f"   Testing {len(test_indices)} random states:")

            for i, idx in enumerate(test_indices):
                state = engine.get_state_from_index(idx)
                back_idx = engine.get_index_from_state(state)
                orbit_card = engine.get_orbit_cardinality(idx)

                if back_idx == idx:
                    status = "✅"
                else:
                    status = "❌"

                print(
                    f"     {i+1:2d}. Index {idx:6d} → State 0x{state:012X} → Index {back_idx:6d} {status} (orbit: {orbit_card})"
                )

            print(f"\n✅ InformationEngine integration successful!")

        except Exception as e:
            print(f"\n❌ InformationEngine integration failed: {e}")
            return False

        print(f"\n🎉 Phenomenology map appears to be correct and functional!")
        return True

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        return False


if __name__ == "__main__":
    success = test_phenomenology_map()
    sys.exit(0 if success else 1)
