import pytest
import numpy as np

from ggg_asi_router.physics import atlas_builder, governance


def _get_atlas_paths() -> atlas_builder.AtlasPaths:
    cfg = atlas_builder.AtlasConfiguration()
    paths = atlas_builder.AtlasPaths.from_directory(cfg.output_directory)
    if not all([paths.ontology.exists(), paths.epistemology.exists(),
                paths.stage_profile.exists(), paths.loop_defects.exists(), paths.aperture.exists()]):
        pytest.skip(
            f"Atlas not found at {cfg.output_directory}. "
            "Run: python -m ggg_asi_router.physics.atlas_builder complete"
        )
    return paths


def test_atlas_core_invariants() -> None:
    paths = _get_atlas_paths()

    ontology = np.load(paths.ontology, mmap_mode="r")
    epistemology = np.load(paths.epistemology, mmap_mode="r")
    stage_profile = np.load(paths.stage_profile, mmap_mode="r")
    loop_defects = np.load(paths.loop_defects, mmap_mode="r")
    aperture = np.load(paths.aperture, mmap_mode="r")

    n_states = int(ontology.size)

    assert n_states == 788_986
    assert epistemology.shape == (n_states, 256)
    assert stage_profile.shape == (n_states, 4)
    assert loop_defects.shape == (n_states, 3)
    assert aperture.shape == (n_states,)

    # Stage profile values must be in [0, 12]
    assert np.all(stage_profile >= 0)
    assert np.all(stage_profile <= 12)

    # Loop defects must be in [0, 48]
    assert np.all(loop_defects >= 0)
    assert np.all(loop_defects <= 48)

    # Aperture must be in [0, 1]
    assert np.all(aperture >= 0.0)
    assert np.all(aperture <= 1.0)

    archetype_state = int(governance.tensor_to_int(governance.GENE_Mac_S))
    archetype_index = int(np.searchsorted(ontology, archetype_state))
    assert archetype_index < n_states
    assert int(ontology[archetype_index]) == archetype_state

    # Archetype should have zero stage profile (all layers match reference)
    assert np.all(stage_profile[archetype_index] == 0)

    assert np.all(ontology < (1 << 48))


def test_loop_defects_closure() -> None:
    """
    Test that loop defects computation stays within ontology bounds.

    For each of the three commutator loops, verify that the loop computation
    produces valid state indices within [0, n_states).
    """
    paths = _get_atlas_paths()
    ontology = np.load(paths.ontology, mmap_mode="r")
    epistemology = np.load(paths.epistemology, mmap_mode="r")
    
    n_states = int(ontology.size)
    
    # Sample indices to test (every 1000th state for speed)
    sample_indices = np.arange(0, n_states, 1000, dtype=np.int64)
    
    loop_actions = [
        (governance.UNA_P_ACTION, governance.ONA_P_ACTION, governance.UNA_M_ACTION, governance.ONA_M_ACTION),
        (governance.UNA_P_ACTION, governance.BU_P_ACTION, governance.UNA_M_ACTION, governance.BU_M_ACTION),
        (governance.ONA_P_ACTION, governance.BU_P_ACTION, governance.ONA_M_ACTION, governance.BU_M_ACTION),
    ]
    
    for loop_idx, (action_A_pos, action_B_pos, action_A_neg, action_B_neg) in enumerate(loop_actions):
        idx = sample_indices.copy()
        
        # Compute loop: i -> B_pos -> A_pos -> B_neg -> A_neg
        idx = epistemology[idx, action_B_pos].astype(np.int64)
        idx = epistemology[idx, action_A_pos].astype(np.int64)
        idx = epistemology[idx, action_B_neg].astype(np.int64)
        idx = epistemology[idx, action_A_neg].astype(np.int64)
        
        # All loop end indices must be valid
        assert np.all((idx >= 0) & (idx < n_states)), (
            f"Loop {loop_idx} produced out-of-bounds indices: "
            f"min={idx.min()}, max={idx.max()}, expected range [0, {n_states})"
        )


def test_aperture_derivation_consistency() -> None:
    """
    Test that aperture is correctly derived from stage_profile and loop_defects.

    Recompute aperture for a sample of states using the same K4 geometry and
    verify it matches the saved aperture.npy within tolerance.
    """
    from ggg_asi_router.physics import k4_geometry
    
    paths = _get_atlas_paths()
    stage_profile = np.load(paths.stage_profile, mmap_mode="r")
    loop_defects = np.load(paths.loop_defects, mmap_mode="r")
    aperture = np.load(paths.aperture, mmap_mode="r")
    
    n_states = int(aperture.size)
    
    # Sample indices to test (every 5000th state for speed)
    sample_indices = np.arange(0, n_states, 5000, dtype=np.int64)
    
    # Precompute K4 geometry
    B = k4_geometry.get_incidence_matrix_k4()
    W = k4_geometry.get_weight_matrix_k4()
    F = k4_geometry.get_face_cycle_matrix_k4(W)
    
    # Recompute aperture for sample
    p_sample = stage_profile[sample_indices].astype(np.float32)
    theta_stage = np.arccos(np.clip(1.0 - 2.0 * p_sample / 12.0, -1.0, 1.0)) / np.pi
    
    y_grad = (B.T @ theta_stage.T).T  # Shape: (n_sample, 6)
    
    d_sample = loop_defects[sample_indices].astype(np.float32)
    c_cycle = np.arccos(np.clip(1.0 - 2.0 * d_sample / 48.0, -1.0, 1.0)) / np.pi  # Shape: (n_sample, 3)
    
    y_cycle = (F @ c_cycle.T).T  # Shape: (n_sample, 6)
    
    y = y_grad + y_cycle
    
    # Compute aperture: A = ||y_cycle||²_W / ||y||²_W
    # Compute weighted norms: ||v||²_W = v^T @ W @ v
    Wy = y @ W  # Shape: (n_sample, 6)
    Wy_cycle = y_cycle @ W  # Shape: (n_sample, 6)
    y_norm_sq = np.sum(Wy * y, axis=1)
    y_cycle_norm_sq = np.sum(Wy_cycle * y_cycle, axis=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        aperture_recomputed = np.where(y_norm_sq > 0, y_cycle_norm_sq / y_norm_sq, 0.0)
    
    aperture_saved = aperture[sample_indices].astype(np.float32)
    
    # Verify match within tolerance
    max_diff = np.max(np.abs(aperture_recomputed - aperture_saved))
    assert max_diff < 1e-5, (
        f"Aperture derivation mismatch: max difference = {max_diff:.2e} "
        f"(expected < 1e-5). First mismatch at index {sample_indices[0]}: "
        f"recomputed={aperture_recomputed[0]:.6f}, saved={aperture_saved[0]:.6f}"
    )


def test_layer_masks_correspond_to_tensor_slices() -> None:
    """
    Test that LAYER_MASKS correspond semantically to tensor layer slices.

    For each layer, verify that the popcount of (state XOR archetype) & LAYER_MASK[layer]
    equals the Hamming distance between the corresponding tensor slices.
    
    This ensures stage_profile columns (CS, UNA, ONA, BU) correspond to the actual
    tensor layers, not a permuted representation.
    """
    from ggg_asi_router.physics.router_maps_builder import _popcount_uint64
    
    paths = _get_atlas_paths()
    ontology = np.load(paths.ontology, mmap_mode="r")
    
    n_states = int(ontology.size)
    s_ref = int(governance.tensor_to_int(governance.GENE_Mac_S))
    T_ref = governance.GENE_Mac_S
    
    # Sample indices to test (every 1000th state for speed)
    sample_indices = np.arange(0, n_states, 1000, dtype=np.int64)
    
    violations = []
    for idx in sample_indices:
        state_int = int(ontology[idx])
        T = governance.int_to_tensor(state_int)
        
        # Compute stage profile via tensor comparison (ground truth)
        stage_profile_tensor = np.zeros(4, dtype=np.uint8)
        for layer in range(4):
            # Count mismatches: where T[layer] != T_ref[layer]
            layer_diff = (T[layer] != T_ref[layer]).astype(np.uint8)
            stage_profile_tensor[layer] = np.sum(layer_diff)
        
        # Compare with mask-based computation
        diff = state_int ^ s_ref
        stage_profile_mask = np.zeros(4, dtype=np.uint8)
        for layer in range(4):
            layer_diff = diff & governance.LAYER_MASKS[layer]
            stage_profile_mask[layer] = _popcount_uint64(np.array([layer_diff], dtype=np.uint64))[0]
        
        if not np.array_equal(stage_profile_tensor, stage_profile_mask):
            violations.append((idx, state_int, stage_profile_tensor, stage_profile_mask))
    
    assert len(violations) == 0, (
        f"Found {len(violations)} layer mask semantic violations. "
        f"LAYER_MASKS do not correspond to tensor layer slices. "
        f"First violation at index {violations[0][0]}: "
        f"tensor={violations[0][2]}, mask={violations[0][3]}"
    )
    
    # Note: saved stage_profile may have been built with old layer mask calculation
    # and will need to be rebuilt after fixing LAYER_MASKS


