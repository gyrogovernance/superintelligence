"""
Other tests: Edge cases, atlas building, validation.

Tests:
- Atlas building process
- Phenomenology validation
- Edge case handling
- Performance benchmarks
"""

import pytest
import numpy as np
from pathlib import Path
import time

from src.router.constants import (
    GENE_MIC_S,
    ARCHETYPE_STATE24,
    ARCHETYPE_A12,
    ARCHETYPE_B12,
    LAYER_MASK_12,
    XFORM_MASK_BY_BYTE,
)


class TestAtlasBuildingValidation:
    """Validate atlas building process (assumes atlas exists)."""

    @pytest.fixture(scope="class")
    def atlas_dir(self):
        """Atlas directory path."""
        return Path("data/atlas")

    def test_atlas_exists(self, atlas_dir):
        """Atlas directory should exist."""
        if not atlas_dir.exists():
            pytest.skip("Atlas not built. Run: python -m src.router.atlas")
        assert atlas_dir.is_dir()

    def test_ontology_file_exists(self, atlas_dir):
        """ontology.npy should exist."""
        if not atlas_dir.exists():
            pytest.skip("Atlas not built")
        assert (atlas_dir / "ontology.npy").exists()

    def test_epistemology_file_exists(self, atlas_dir):
        """epistemology.npy should exist."""
        if not atlas_dir.exists():
            pytest.skip("Atlas not built")
        assert (atlas_dir / "epistemology.npy").exists()

    def test_phenomenology_file_exists(self, atlas_dir):
        """phenomenology.npz should exist."""
        if not atlas_dir.exists():
            pytest.skip("Atlas not built")
        assert (atlas_dir / "phenomenology.npz").exists()

    def test_ontology_size(self, atlas_dir):
        """Ontology should have expected size."""
        if not atlas_dir.exists():
            pytest.skip("Atlas not built")
        
        ontology = np.load(atlas_dir / "ontology.npy")
        size = len(ontology)
        
        # Expected: 65,536 states
        expected = 65536
        assert size == expected, f"Ontology size {size:,} != {expected:,}"

    def test_ontology_sorted(self, atlas_dir):
        """Ontology should be sorted."""
        if not atlas_dir.exists():
            pytest.skip("Atlas not built")
        
        ontology = np.load(atlas_dir / "ontology.npy")
        assert np.all(ontology[:-1] <= ontology[1:]), "Ontology not sorted"

    def test_archetype_in_ontology(self, atlas_dir):
        """Archetype should be found in ontology."""
        if not atlas_dir.exists():
            pytest.skip("Atlas not built")
        
        ontology = np.load(atlas_dir / "ontology.npy")
        archetype_indices = np.where(ontology == ARCHETYPE_STATE24)[0]
        assert len(archetype_indices) > 0, f"Archetype {ARCHETYPE_STATE24:06x} not found in ontology"
        archetype_idx = archetype_indices[0]
        assert ontology[archetype_idx] == ARCHETYPE_STATE24


class TestOntologyStructure:
    """Prove ontology equals the exact cartesian product implied by the kernel physics."""

    def test_ontology_is_cartesian_product_of_two_256_sets(self):
        atlas_dir = Path("data/atlas")
        if not atlas_dir.exists():
            pytest.skip("Atlas not built")

        ontology = np.load(atlas_dir / "ontology.npy").astype(np.uint32)

        # Build the 256 A-values and 256 B-values that must appear after exactly 2 steps from archetype
        from src.router.constants import XFORM_MASK_BY_BYTE, LAYER_MASK_12, ARCHETYPE_A12, ARCHETYPE_B12

        def mask_a(byte: int) -> int:
            mask24 = int(XFORM_MASK_BY_BYTE[byte])
            return (mask24 >> 12) & LAYER_MASK_12

        A_set = np.array([ARCHETYPE_A12 ^ mask_a(b) for b in range(256)], dtype=np.uint16)
        B_set = np.array([ARCHETYPE_B12 ^ mask_a(b) for b in range(256)], dtype=np.uint16)

        assert len(set(int(x) for x in A_set)) == 256
        assert len(set(int(x) for x in B_set)) == 256

        # Cartesian product -> 65536 states
        states = ((A_set.astype(np.uint32)[:, None] << 12) | B_set.astype(np.uint32)[None, :]).reshape(-1)
        states_sorted = np.sort(states)

        assert ontology.shape[0] == 65536
        assert np.array_equal(ontology, states_sorted), "Ontology != A_set × B_set cartesian product"


class TestPhenomenologyValidation:
    """Validate phenomenology constants."""

    @pytest.fixture(scope="class")
    def phen(self):
        """Load phenomenology."""
        atlas_dir = Path("data/atlas")
        if not atlas_dir.exists():
            pytest.skip("Atlas not built")
        return np.load(atlas_dir / "phenomenology.npz")

    def test_archetype_stored_correctly(self, phen):
        """Archetype components should match constants."""
        assert int(phen["archetype_state24"]) == ARCHETYPE_STATE24
        assert int(phen["archetype_a12"]) == ARCHETYPE_A12
        assert int(phen["archetype_b12"]) == ARCHETYPE_B12

    def test_gene_mic_s_stored(self, phen):
        """GENE_MIC_S should be stored."""
        assert int(phen["gene_mic_s"]) == GENE_MIC_S

    def test_xform_masks_count(self, phen):
        """Should have 256 transformation masks."""
        masks = phen["xform_mask_by_byte"]
        assert len(masks) == 256

    def test_xform_masks_match_constants(self, phen):
        """Stored masks should match computed constants."""
        masks = phen["xform_mask_by_byte"]
        for i in range(256):
            assert int(masks[i]) == int(XFORM_MASK_BY_BYTE[i])



class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_zeros_state(self):
        """State with all zeros should be valid."""
        from src.router.constants import pack_state, unpack_state, step_state_by_byte
        
        state = pack_state(0x000, 0x000)
        next_state = step_state_by_byte(state, 0x00)
        a, b = unpack_state(next_state)
        
        assert 0 <= a <= LAYER_MASK_12
        assert 0 <= b <= LAYER_MASK_12

    def test_all_ones_state(self):
        """State with all ones should be valid."""
        from src.router.constants import pack_state, unpack_state, step_state_by_byte
        
        state = pack_state(0xFFF, 0xFFF)
        next_state = step_state_by_byte(state, 0xFF)
        a, b = unpack_state(next_state)
        
        assert 0 <= a <= LAYER_MASK_12
        assert 0 <= b <= LAYER_MASK_12

    def test_mask_boundary_values(self):
        """Mask expansion should handle all boundary values."""
        from src.router.constants import expand_intron_to_mask24
        
        boundary_introns = [0x00, 0xFF, 0xAA, 0x55]
        for intron in boundary_introns:
            mask = expand_intron_to_mask24(intron)
            assert 0 <= mask < (1 << 24)

    def test_repeated_byte_application(self):
        """Applying same byte repeatedly should stay in ontology."""
        from src.router.constants import step_state_by_byte
        
        state = ARCHETYPE_STATE24
        byte = 0x42
        
        for _ in range(100):
            state = step_state_by_byte(state, byte)
            # Should remain valid 24-bit state
            assert 0 <= state < (1 << 24)


class TestPerformance:
    """Performance benchmarks."""

    def test_step_performance(self, capsys):
        """Measure step_state_by_byte performance."""
        from src.router.constants import step_state_by_byte
        
        state = ARCHETYPE_STATE24
        n_steps = 10000
        
        start = time.perf_counter()
        for i in range(n_steps):
            byte = i % 256
            state = step_state_by_byte(state, byte)
        elapsed = time.perf_counter() - start
        
        steps_per_sec = n_steps / elapsed
        print(f"\n  Steps/sec: {steps_per_sec:,.0f}")
        print(f"  Time per step: {elapsed/n_steps*1e6:.2f} μs")
        
        # Should be fast (>100k steps/sec on modern hardware)
        assert steps_per_sec > 50000, f"Too slow: {steps_per_sec:.0f} steps/sec"

    def test_kernel_step_performance(self, capsys):
        """Measure kernel.step_byte performance."""
        atlas_dir = Path("data/atlas")
        if not atlas_dir.exists():
            pytest.skip("Atlas not built")
        
        from src.router.kernel import RouterKernel
        kernel = RouterKernel(atlas_dir)
        
        n_steps = 10000
        start = time.perf_counter()
        for i in range(n_steps):
            byte = i % 256
            kernel.step_byte(byte)
        elapsed = time.perf_counter() - start
        
        steps_per_sec = n_steps / elapsed
        print(f"\n  Kernel steps/sec: {steps_per_sec:,.0f}")
        print(f"  Time per step: {elapsed/n_steps*1e6:.2f} μs")
        
        # Epistemology lookup should also be fast
        assert steps_per_sec > 10000, f"Too slow: {steps_per_sec:.0f} steps/sec"

    def test_aperture_measurement_performance(self, capsys):
        """Measure signature_with_byte performance."""
        atlas_dir = Path("data/atlas")
        if not atlas_dir.exists():
            pytest.skip("Atlas not built")
        
        from src.router.kernel import RouterKernel
        kernel = RouterKernel(atlas_dir)
        
        n_measurements = 1000
        start = time.perf_counter()
        for i in range(n_measurements):
            byte = i % 256
            _ = kernel.signature_with_byte(byte)
        elapsed = time.perf_counter() - start
        
        measurements_per_sec = n_measurements / elapsed
        print(f"\n  Aperture measurements/sec: {measurements_per_sec:,.0f}")
        print(f"  Time per measurement: {elapsed/n_measurements*1e3:.2f} ms")


class TestInvariantValidation:
    """Validate global invariants."""

    def test_unique_mask_count(self):
        """Should have exactly 256 unique masks."""
        unique = len(set(int(XFORM_MASK_BY_BYTE[b]) for b in range(256)))
        assert unique == 256

    def test_all_b_masks_zero(self):
        """All B masks should be zero."""
        for byte in range(256):
            mask24 = int(XFORM_MASK_BY_BYTE[byte])
            mask_b = mask24 & LAYER_MASK_12
            assert mask_b == 0, f"Byte {hex(byte)} has non-zero B mask: {hex(mask_b)}"

    def test_a_mask_coverage(self):
        """A masks should cover reasonable range."""
        a_masks = set()
        for byte in range(256):
            mask24 = int(XFORM_MASK_BY_BYTE[byte])
            mask_a = (mask24 >> 12) & LAYER_MASK_12
            a_masks.add(mask_a)
        
        # Should have 256 unique A masks (since expansion is injective)
        assert len(a_masks) == 256



# Global summary
@pytest.fixture(scope="session", autouse=True)
def print_global_summary(request):
    """Print overall test summary."""
    yield
    
    print("\n" + "="*10)
    print("OVERALL TEST SUMMARY")
    print("="*10)
    
    atlas_dir = Path("data/atlas")
    if atlas_dir.exists():
        ontology = np.load(atlas_dir / "ontology.npy")
        epistemology = np.load(atlas_dir / "epistemology.npy")
        phen = np.load(atlas_dir / "phenomenology.npz")
        
        print(f"Atlas loaded: YES")
        print(f"  Ontology: {len(ontology):,} states")
        print(f"  Epistemology: {epistemology.shape}")
        print(f"  Phenomenology: {len(phen.files)} arrays")
    else:
        print(f"Atlas loaded: NO (run: python -m src.router.atlas)")
    
    print(f"\nConstants:")
    print(f"  GENE_MIC_S: {hex(GENE_MIC_S)}")
    print(f"  Archetype: {hex(ARCHETYPE_STATE24)}")
    print(f"  Unique masks: {len(set(int(XFORM_MASK_BY_BYTE[b]) for b in range(256)))}")
    
    print("="*10)

