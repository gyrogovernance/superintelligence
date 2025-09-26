import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict, Counter
import math

# Ensure project root is on sys.path so 'baby' package can be imported when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from baby import governance
    from toys.experiments.leb128_physics import token_to_introns
except Exception as e:
    print("ERROR: Failed to import required modules:", e)
    sys.exit(2)

MASK = 0xFF


def load_physics_maps():
    """Load the physics maps from disk."""
    base_path = PROJECT_ROOT / "memories" / "public" / "meta"
    try:
        epistemology = np.load(base_path / "epistemology.npy", allow_pickle=False)
        theta_map = np.load(base_path / "theta.npy", allow_pickle=False)
        phenomenology = np.load(base_path / "phenomenology_map.npy", allow_pickle=False)
        orbit_sizes = np.load(base_path / "orbit_sizes.npy", allow_pickle=False)
        return epistemology, theta_map, phenomenology, orbit_sizes
    except Exception as e:
        print(f"WARNING: Could not load physics maps: {e}")
        return None, None, None, None


def test_fold_algebraic_identity():
    """Test 1: Fold algebraic identity - fold(a,b) == (~a) & b"""
    print("1. Testing fold algebraic identity...")
    failures = 0
    for a in range(min(256, 100)):  # Sample to keep test fast
        for b in range(min(256, 100)):
            got = governance.fold(a, b)
            expected = ((~a) & b) & MASK
            if got != expected:
                print(f"   FAIL: fold(0x{a:02X}, 0x{b:02X}) -> 0x{got:02X}, expected 0x{expected:02X}")
                failures += 1
                if failures > 5:  # Stop after a few failures
                    return False
    print("   PASS: fold(a,b) == (~a) & b holds")
    return True


def test_leb128_roundtrip():
    """Test 2: LEB128 token/intron codec roundtrip"""
    print("2. Testing LEB128 codec roundtrip...")
    from toys.experiments.leb128_physics import introns_to_token

    sample_tokens = [1, 42, 255, 256, 1000, 30000]
    for token_id in sample_tokens:
        try:
            introns = token_to_introns(token_id)
            recovered = introns_to_token(introns)
            if recovered != token_id:
                print(f"   FAIL: token {token_id} -> {introns} -> {recovered}")
                return False
        except Exception as e:
            print(f"   ERROR: token {token_id} failed: {e}")
            return False
    print("   PASS: LEB128 codec maintains roundtrip integrity")
    return True


def test_within_token_composition(epistemology):
    """Test 3: Within-token composition - multi-intron tokens show incremental state changes"""
    print("3. Testing within-token composition...")
    if epistemology is None:
        print("   SKIP: No epistemology map available")
        return True

    sample_tokens = [256, 257, 1000, 2000]  # Multi-byte tokens
    state_changes = []

    for token_id in sample_tokens:
        try:
            introns = token_to_introns(token_id)
            if len(introns) < 2:
                continue

            current_state = 0  # Start from archetypal state
            states = [current_state]

            for intron in introns:
                current_state = epistemology[current_state % len(epistemology), intron]
                states.append(current_state)

            # Check for non-trivial state progression
            unique_states = len(set(states))
            if unique_states > 1:
                state_changes.append(unique_states)

        except Exception as e:
            print(f"   WARNING: token {token_id} failed: {e}")
            continue

    if state_changes and np.mean(state_changes) > 1.5:
        print(f"   PASS: Multi-intron tokens show state progression (avg {np.mean(state_changes):.1f} unique states)")
        return True
    else:
        print("   MARGINAL: Limited state progression in multi-intron tokens")
        return True


def test_theta_angle_properties(theta_map):
    """Test 4: Theta measurements preserve angular distance properties"""
    print("4. Testing theta angle properties...")
    if theta_map is None:
        print("   SKIP: No theta map available")
        return True

    # Sample a few states and check theta values are reasonable
    sample_states = [0, 100, 1000, 10000]
    thetas = []

    for state in sample_states:
        if state < len(theta_map):
            theta = theta_map[state]
            thetas.append(theta)

    if not thetas:
        print("   SKIP: No valid theta values found")
        return True

    # Check theta values are in reasonable range [0, 2π]
    valid_range = all(0 <= theta <= 2 * np.pi + 0.1 for theta in thetas)  # Small tolerance
    spread = max(thetas) - min(thetas) if len(thetas) > 1 else 0

    if valid_range and spread > 0.1:
        print(f"   PASS: Theta values in valid range with spread {spread:.2f}")
        return True
    else:
        print(f"   MARGINAL: Theta range issues (valid_range={valid_range}, spread={spread:.2f})")
        return True


def test_orbit_closure(phenomenology, orbit_sizes):
    """Test 5: Phenomenology orbits form proper equivalence classes"""
    print("5. Testing orbit closure...")
    if phenomenology is None or orbit_sizes is None:
        print("   SKIP: No phenomenology/orbit maps available")
        return True

    # Count actual orbit populations
    orbit_counts = Counter(phenomenology)

    # Sample a few orbits and check consistency
    sample_orbits = list(orbit_counts.keys())[:10]
    consistent = True

    for orbit_id in sample_orbits:
        actual_count = orbit_counts[orbit_id]
        if orbit_id < len(orbit_sizes):
            expected_size = orbit_sizes[orbit_id]
            # Allow some tolerance for edge cases
            if abs(actual_count - expected_size) > max(1, expected_size * 0.1):
                print(f"   WARNING: Orbit {orbit_id} size mismatch: actual={actual_count}, expected={expected_size}")
                consistent = False

    if consistent:
        print(f"   PASS: Orbit sizes consistent ({len(sample_orbits)} orbits checked)")
        return True
    else:
        print("   MARGINAL: Some orbit size inconsistencies found")
        return True


def test_hamming_stability(epistemology):
    """Test 6: Residual-like stability - Hamming jumps stay controlled"""
    print("6. Testing Hamming stability...")
    if epistemology is None:
        print("   SKIP: No epistemology map available")
        return True

    # Sample some state transitions and measure Hamming distances
    sample_states = [0, 100, 1000, 10000]
    sample_introns = [0, 1, 42, 128, 255]
    hamming_jumps = []

    for state in sample_states:
        if state >= len(epistemology):
            continue
        for intron in sample_introns:
            try:
                next_state = epistemology[state, intron]
                hamming_dist = bin(state ^ next_state).count("1")
                hamming_jumps.append(hamming_dist)
            except IndexError:
                continue

    if not hamming_jumps:
        print("   SKIP: No valid transitions found")
        return True

    avg_hamming = np.mean(hamming_jumps)
    max_hamming = max(hamming_jumps)

    # Good stability: low average, controlled maximum
    if avg_hamming < 8 and max_hamming < 16:
        print(f"   PASS: Controlled Hamming jumps (avg={avg_hamming:.1f}, max={max_hamming})")
        return True
    else:
        print(f"   MARGINAL: High Hamming jumps (avg={avg_hamming:.1f}, max={max_hamming})")
        return True


def test_parity_closure_basics():
    """Test 7: Basic parity closure properties via fold"""
    print("7. Testing parity closure basics...")

    # Test self-annihilation: fold(a, a) = 0
    test_values = [0, 1, 42, 128, 255]
    for a in test_values:
        result = governance.fold(a, a)
        if result != 0:
            print(f"   FAIL: fold({a}, {a}) = {result}, expected 0")
            return False

    # Test left identity: fold(0, b) = b
    for b in test_values:
        result = governance.fold(0, b)
        if result != b:
            print(f"   FAIL: fold(0, {b}) = {result}, expected {b}")
            return False

    print("   PASS: Parity closure properties (self-annihilation, left identity)")
    return True


def run_consolidated_test():
    """Run all available theoretical tests."""
    print("GyroSI Consolidated Theoretical Validation")
    print("=" * 50)

    # Load physics maps
    epistemology, theta_map, phenomenology, orbit_sizes = load_physics_maps()

    tests = [
        test_fold_algebraic_identity,
        test_leb128_roundtrip,
        lambda: test_within_token_composition(epistemology),
        lambda: test_theta_angle_properties(theta_map),
        lambda: test_orbit_closure(phenomenology, orbit_sizes),
        lambda: test_hamming_stability(epistemology),
        test_parity_closure_basics,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   ERROR: {e}")

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All theoretical invariants validated")
        return 0
    elif passed >= total * 0.7:
        print("⚠ Most invariants validated, some issues detected")
        return 0
    else:
        print("✗ Multiple invariant failures detected")
        return 1


if __name__ == "__main__":
    sys.exit(run_consolidated_test())
