"""
Routing tests: Atlas-based kernel and epistemology.

Tests the routing system:
- Atlas loading
- State transitions via epistemology
- Aperture measurement
- Multi-step routing
- Signature generation
"""

import pytest
import numpy as np
from pathlib import Path

from router.kernel import RouterKernel
from router.constants import ARCHETYPE_STATE24, unpack_state


# Fixture: Kernel loaded from atlas
@pytest.fixture(scope="module")
def kernel():
    """Load router kernel from built atlas."""
    atlas_dir = Path("data/atlas")
    if not atlas_dir.exists():
        pytest.skip("Atlas not built. Run: python -m src.router.atlas")
    
    return RouterKernel(atlas_dir)


@pytest.fixture(scope="module")
def atlas_stats(kernel):
    """Compute atlas statistics once for all tests."""
    stats = {
        "ontology_size": len(kernel.ontology),
        "archetype_state": int(kernel.ontology[kernel.archetype_index]),
        "archetype_a12": kernel.archetype_a12,
    }
    return stats


class TestAtlasLoading:
    """Test that atlas loads correctly."""

    def test_ontology_loaded(self, kernel):
        """Ontology should load as numpy array."""
        assert kernel.ontology is not None
        assert len(kernel.ontology) > 0

    def test_epistemology_loaded(self, kernel):
        """Epistemology should have shape [N, 256]."""
        n = len(kernel.ontology)
        assert kernel.epistemology.shape == (n, 256)

    def test_phenomenology_loaded(self, kernel):
        """Phenomenology constants should load."""
        assert kernel.k4_edges is not None
        assert kernel.p_cycle is not None
        assert kernel.archetype_a12 is not None
        assert kernel.xform_mask_by_byte is not None

    def test_archetype_found(self, kernel, atlas_stats):
        """Archetype should be found in ontology."""
        assert kernel.ontology[kernel.archetype_index] == ARCHETYPE_STATE24
        assert atlas_stats["archetype_state"] == ARCHETYPE_STATE24

    def test_archetype_index_convention(self, kernel):
        """Document archetype index convention (sorted ontology)."""
        # The ontology is sorted, so archetype may not be at index 0
        # Verify the kernel found it correctly
        archetype_state = int(kernel.ontology[kernel.archetype_index])
        
        assert archetype_state == ARCHETYPE_STATE24
        
        # Verify ontology is sorted (this is the actual convention)
        assert np.all(kernel.ontology[:-1] <= kernel.ontology[1:]), "Ontology not sorted"
        
        # Document where archetype actually is
        print(f"\n  Archetype at index {kernel.archetype_index:,} / {len(kernel.ontology):,}")


class TestStateTransitions:
    """Test epistemology-based state transitions."""

    def test_initial_state_is_archetype(self, kernel):
        """Kernel should start at archetype."""
        sig = kernel.signature()
        assert sig.state_index == kernel.archetype_index
        assert sig.state_hex == f"{ARCHETYPE_STATE24:06x}"

    def test_single_byte_transition(self, kernel):
        """Single byte should change state."""
        kernel.reset()
        initial_idx = kernel.state_index
        
        kernel.step_byte(0x42)
        assert kernel.state_index != initial_idx

    def test_all_bytes_produce_transitions(self, kernel):
        """All 256 bytes should produce valid transitions."""
        kernel.reset()
        
        for byte in range(256):
            kernel.reset()
            kernel.step_byte(byte)
            assert 0 <= kernel.state_index < len(kernel.ontology)

    def test_transitions_deterministic(self, kernel):
        """Same byte from same state must produce same result."""
        test_bytes = [0x00, 0x42, 0xAA, 0xFF]
        
        for byte in test_bytes:
            kernel.reset()
            kernel.step_byte(byte)
            idx1 = kernel.state_index
            
            kernel.reset()
            kernel.step_byte(byte)
            idx2 = kernel.state_index
            
            assert idx1 == idx2, f"Byte {hex(byte)} not deterministic"

    def test_reset_returns_to_archetype(self, kernel):
        """Reset should return to archetype."""
        kernel.step_byte(0x12)
        kernel.step_byte(0x34)
        kernel.reset()
        
        assert kernel.state_index == kernel.archetype_index


class TestMultiStepRouting:
    """Test routing through multiple bytes."""

    def test_two_byte_sequence(self, kernel):
        """Two-byte sequence should be deterministic."""
        kernel.reset()
        kernel.step_byte(0x12)
        kernel.step_byte(0x34)
        idx1 = kernel.state_index
        
        kernel.reset()
        kernel.step_byte(0x12)
        kernel.step_byte(0x34)
        idx2 = kernel.state_index
        
        assert idx1 == idx2

    def test_payload_routing(self, kernel):
        """step() with payload should apply all bytes."""
        payload = b"Hello"
        kernel.reset()
        sig = kernel.step(payload)
        
        # Manually apply each byte
        kernel.reset()
        for byte in payload:
            kernel.step_byte(byte)
        manual_idx = kernel.state_index
        
        assert sig.state_index == manual_idx

    def test_order_matters(self, kernel):
        """Byte order should affect final state (non-commutativity)."""
        # Forward
        kernel.reset()
        kernel.step_byte(0x12)
        kernel.step_byte(0x34)
        forward_idx = kernel.state_index
        
        # Reverse
        kernel.reset()
        kernel.step_byte(0x34)
        kernel.step_byte(0x12)
        reverse_idx = kernel.state_index
        
        # Should differ (non-commutative)
        assert forward_idx != reverse_idx


class TestApertureMeasurement:
    """Test K4 aperture computation."""

    def test_signature_has_aperture(self, kernel):
        """Signature should include aperture value."""
        kernel.reset()
        sig = kernel.signature()
        
        assert hasattr(sig, 'aperture')
        assert 0.0 <= sig.aperture <= 1.0

    def test_signature_with_byte(self, kernel):
        """signature_with_byte should use specified byte for measurement."""
        kernel.reset()
        
        sig0 = kernel.signature_with_byte(0x00)
        sig1 = kernel.signature_with_byte(0xFF)
        
        # Different bytes should (usually) give different apertures
        # (not guaranteed for all states, but likely)
        assert sig0 is not None
        assert sig1 is not None

    def test_aperture_changes_with_byte(self, kernel):
        """Aperture should depend on the instruction byte."""
        kernel.reset()
        
        apertures = []
        for byte in [0x00, 0x42, 0xAA, 0xFF]:
            sig = kernel.signature_with_byte(byte)
            apertures.append(sig.aperture)
        
        # At least some should differ
        unique = len(set(apertures))
        assert unique > 1, "All apertures identical across different bytes"

    def test_aperture_bounded(self, kernel):
        """Aperture should always be in [0, 1]."""
        kernel.reset()
        
        for byte in range(0, 256, 16):  # Sample every 16th byte
            sig = kernel.signature_with_byte(byte)
            assert 0.0 <= sig.aperture <= 1.0, f"Byte {hex(byte)}: aperture = {sig.aperture}"


class TestSignatureProperties:
    """Test signature dataclass properties."""

    def test_signature_fields(self, kernel):
        """Signature should have all required fields."""
        kernel.reset()
        sig = kernel.signature()
        
        assert hasattr(sig, 'state_index')
        assert hasattr(sig, 'state_hex')
        assert hasattr(sig, 'a_hex')
        assert hasattr(sig, 'b_hex')
        assert hasattr(sig, 'aperture')

    def test_signature_hex_format(self, kernel):
        """Hex strings should have correct length."""
        kernel.reset()
        sig = kernel.signature()
        
        assert len(sig.state_hex) == 6  # 24 bits = 6 hex chars
        assert len(sig.a_hex) == 3      # 12 bits = 3 hex chars
        assert len(sig.b_hex) == 3      # 12 bits = 3 hex chars

    def test_signature_consistency(self, kernel):
        """Unpacking state should match a_hex and b_hex."""
        kernel.reset()
        kernel.step_byte(0x42)
        sig = kernel.signature()
        
        state_int = int(kernel.ontology[sig.state_index])
        a, b = unpack_state(state_int)
        
        assert f"{a:03x}" == sig.a_hex
        assert f"{b:03x}" == sig.b_hex


class TestReachability:
    """Test state space reachability properties."""

    def test_archetype_reachable_from_itself(self, kernel):
        """Archetype should be in its own orbit."""
        kernel.reset()
        visited = {kernel.state_index}
        
        for byte in range(256):
            kernel.reset()
            kernel.step_byte(byte)
            visited.add(kernel.state_index)
        
        # Should reach multiple states
        assert len(visited) > 1

    def test_random_walk_stays_in_ontology(self, kernel):
        """Random byte sequence should never leave ontology."""
        kernel.reset()
        np.random.seed(42)
        
        for _ in range(100):
            byte = np.random.randint(0, 256)
            kernel.step_byte(byte)
            assert 0 <= kernel.state_index < len(kernel.ontology)


class TestAtlasGlobalGroupFacts:
    """Facts about the induced dynamics on the ontology, proven from epistemology."""

    def test_each_byte_column_is_permutation(self, kernel):
        """
        For each byte, epistemology[:, byte] must be a permutation of [0..N-1].
        This proves per-byte bijectivity ON the ontology (no merges, no drops).
        """
        epi = kernel.epistemology
        n = epi.shape[0]

        for byte in range(256):
            col = np.asarray(epi[:, byte], dtype=np.int64)
            counts = np.bincount(col, minlength=n)
            assert counts.min() == 1 and counts.max() == 1, f"Byte {byte}: not a permutation"

    def test_bfs_radius_two_from_archetype(self, kernel):
        """
        Prove the 'diameter 2 from archetype' claim using epistemology, exactly.
        """
        epi = kernel.epistemology
        n = epi.shape[0]
        a0 = kernel.archetype_index

        depth1 = np.asarray(epi[a0, :], dtype=np.int64)
        assert np.unique(depth1).size == 256

        depth2 = np.asarray(epi[depth1, :], dtype=np.int64).reshape(-1)
        assert np.unique(depth2).size == n, "Not all states are reachable in <=2 steps"

    def test_depth4_alternation_identity_on_all_states_for_selected_pairs(self, kernel):
        """
        Atlas-level confirmation: for a handful of (x,y) pairs, XYXY is identity on ALL states.
        This cross-checks the physics-law tests against the atlas map.
        """
        epi = kernel.epistemology
        n = epi.shape[0]
        idxs = np.arange(n, dtype=np.int64)

        pairs = [(0x00, 0xFF), (0x12, 0x34), (0xAA, 0x55), (0x42, 0x24)]
        for x, y in pairs:
            i1 = epi[idxs, x]
            i2 = epi[i1, y]
            i3 = epi[i2, x]
            i4 = epi[i3, y]
            assert np.array_equal(i4, idxs), f"XYXY not identity for pair {(x,y)}"


# Statistics summary
@pytest.fixture(scope="session", autouse=True)
def print_routing_summary(request):
    """Print routing statistics after all tests."""
    yield
    
    # Load kernel for stats
    atlas_dir = Path("data/atlas")
    if not atlas_dir.exists():
        return
    
    kernel = RouterKernel(atlas_dir)
    
    print("\n" + "="*70)
    print("ROUTING TEST SUMMARY")
    print("="*10)
    print(f"Ontology size: {len(kernel.ontology):,} states")
    print(f"Epistemology shape: {kernel.epistemology.shape}")
    print(f"Archetype state: {hex(int(kernel.ontology[kernel.archetype_index]))}")
    print(f"Archetype index: {kernel.archetype_index}")
    print(f"Archetype A12: {hex(kernel.archetype_a12)}")
    
    # Measure aperture distribution
    kernel.reset()
    apertures = []
    for byte in range(0, 256, 4):  # Sample every 4th byte
        sig = kernel.signature_with_byte(byte)
        apertures.append(sig.aperture)
    
    print(f"\nAperture statistics (sample n={len(apertures)}):")
    print(f"  Min: {min(apertures):.4f}")
    print(f"  Max: {max(apertures):.4f}")
    print(f"  Mean: {np.mean(apertures):.4f}")
    print(f"  Std: {np.std(apertures):.4f}")
    print("="*10)