"""
Module 2b: Algebraic decomposition of Bolmo's boundary surface.

THE KEY INSIGHT: The ontology has exactly 65,536 states.
The boundary matrix has exactly 65,536 entries (256×256).
By Property P4, every (b1, b2) pair from the archetype reaches
a unique ontology state. The epistemology maps (state, byte) → state.

The boundary score is therefore a FUNCTION ON THE ONTOLOGY.
The epistemology is the lookup that connects byte pairs to states.

If the boundary score correlates with ontology state properties,
then Bolmo learned a function on YOUR state space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.router.constants import (
    ARCHETYPE_STATE24,
    LAYER_MASK_12,
    mask12_for_byte,
    popcount,
    unpack_state,
    vertex_charge_from_mask,
)
from src.router.kernel import RouterKernel


# =====================================================================
# Kernel basis construction
# =====================================================================

def compute_kernel_bases(atlas_dir: Path) -> dict[str, np.ndarray]:
    """Build kernel bases including the epistemology-derived state map."""
    print("Loading kernel and epistemology...")
    kernel = RouterKernel(atlas_dir=atlas_dir)

    if not kernel.has_intron_priors:
        raise RuntimeError("Atlas v2.2 required. Run: python -m src.router.atlas")

    bases: dict[str, np.ndarray] = {}

    # Precompute per-byte
    masks_raw = np.array([mask12_for_byte(b) for b in range(256)], dtype=np.int32)
    micros = kernel.micro_ref_by_byte[:256].astype(np.int32)
    families = kernel.family_by_byte[:256].astype(np.int32)

    # --- Per-byte bases ---
    mask12 = np.zeros((256, 12), dtype=np.float32)
    for b in range(256):
        m = int(masks_raw[b])
        for i in range(12):
            mask12[b, i] = 1.0 if ((m >> i) & 1) else -1.0
    bases["mask12"] = mask12

    bases["intron"] = kernel.get_intron_feature_vector(np.arange(256))

    family_oh = np.zeros((256, 4), dtype=np.float32)
    for b in range(256):
        family_oh[b, int(families[b])] = 1.0
    bases["family_oh"] = family_oh

    micro_oh = np.zeros((256, 64), dtype=np.float32)
    for b in range(256):
        micro_oh[b, int(micros[b])] = 1.0
    bases["micro_oh"] = micro_oh

    micro_family_tensor = np.zeros((256, 256), dtype=np.float32)
    for b in range(256):
        tensor_idx = int(micros[b]) * 4 + int(families[b])
        micro_family_tensor[b, tensor_idx] = 1.0
    bases["micro_family_tensor"] = micro_family_tensor

    vertex_oh = np.zeros((256, 4), dtype=np.float32)
    for b in range(256):
        v = vertex_charge_from_mask(int(masks_raw[b]))
        vertex_oh[b, v] = 1.0
    bases["vertex_oh"] = vertex_oh

    bases["byte_onehot"] = np.eye(256, dtype=np.float32)

    # =================================================================
    # THE EPISTEMOLOGY MAP
    # For each (b1, b2), compute the ontology state reached from archetype.
    # By P4, this covers ALL 65,536 states exactly once.
    # =================================================================
    print("  Building epistemology state map for all 256x256 pairs...")

    arch_idx = kernel.archetype_index

    # state_index_after_b1[b1] = ontology index after stepping b1 from archetype
    state_after_b1 = np.array([
        int(kernel.epistemology[arch_idx, b1]) for b1 in range(256)
    ], dtype=np.int64)

    # state_index_after_b1b2[b1, b2] = ontology index after b1 then b2
    state_after_b1b2 = np.zeros((256, 256), dtype=np.int64)
    for b1 in range(256):
        si1 = int(state_after_b1[b1])
        for b2 in range(256):
            state_after_b1b2[b1, b2] = int(kernel.epistemology[si1, b2])

    # Verify P4: all 65536 states reached exactly once
    flat_states = state_after_b1b2.flatten()
    unique_states = np.unique(flat_states)
    n_unique = len(unique_states)
    print(f"  Unique states reached: {n_unique} (expected 65536)")
    assert n_unique == 65536, f"P4 violation: got {n_unique} unique states"
    bases["_state_after_b1b2"] = state_after_b1b2

    # =================================================================
    # STATE OBSERVABLES for each (b1, b2) pair
    # These are looked up from the phenomenology using the state index.
    # =================================================================
    print("  Extracting state observables for all pairs...")

    # State24 values
    state24_pairs = np.zeros((256, 256), dtype=np.int64)
    a12_pairs = np.zeros((256, 256), dtype=np.int32)
    b12_pairs = np.zeros((256, 256), dtype=np.int32)

    for b1 in range(256):
        for b2 in range(256):
            si = int(state_after_b1b2[b1, b2])
            s24 = int(kernel.ontology[si])
            a12 = (s24 >> 12) & 0xFFF
            b12_val = s24 & 0xFFF
            state24_pairs[b1, b2] = s24
            a12_pairs[b1, b2] = a12
            b12_pairs[b1, b2] = b12_val

    # Horizon index of the final state
    horizon_pairs = kernel.state_horizon[state_after_b1b2.flatten()].reshape(256, 256)
    bases["_horizon_of_state"] = horizon_pairs.astype(np.float32)

    # Vertex charge of the final state
    vertex_pairs = kernel.state_vertex[state_after_b1b2.flatten()].reshape(256, 256)
    bases["_vertex_of_state"] = vertex_pairs.astype(np.float32)

    # Phase of b2 in the state reached after b1
    phase_pairs = np.zeros((256, 256), dtype=np.float32)
    for b1 in range(256):
        si1 = int(state_after_b1[b1])
        for b2 in range(256):
            phase_pairs[b1, b2] = float(kernel.phase[si1, b2])
    bases["_phase_of_b2"] = phase_pairs

    # Horizon distance of the final state
    hd_pairs = np.zeros((256, 256), dtype=np.float32)
    for b1 in range(256):
        for b2 in range(256):
            a12 = int(a12_pairs[b1, b2])
            b12_val = int(b12_pairs[b1, b2])
            hd_pairs[b1, b2] = float(popcount(a12 ^ (b12_val ^ LAYER_MASK_12)))
    bases["_horizon_distance"] = hd_pairs

    # Archetype distance of the final state
    arch_dist = np.zeros((256, 256), dtype=np.float32)
    for b1 in range(256):
        for b2 in range(256):
            arch_dist[b1, b2] = float(popcount(int(state24_pairs[b1, b2]) ^ ARCHETYPE_STATE24))
    bases["_archetype_distance"] = arch_dist

    # A12 and B12 as bit vectors (the raw state)
    a12_bits = np.zeros((256, 256, 12), dtype=np.float32)
    b12_bits = np.zeros((256, 256, 12), dtype=np.float32)
    for b1 in range(256):
        for b2 in range(256):
            a = int(a12_pairs[b1, b2])
            b_val = int(b12_pairs[b1, b2])
            for i in range(12):
                a12_bits[b1, b2, i] = 1.0 if ((a >> i) & 1) else -1.0
                b12_bits[b1, b2, i] = 1.0 if ((b_val >> i) & 1) else -1.0
    bases["_a12_bits"] = a12_bits
    bases["_b12_bits"] = b12_bits

    # =================================================================
    # THE CLOSED FORM (Property P5)
    # After two steps from (A0, B0) with masks m1, m2:
    #   A_final = A0 XOR m1
    #   B_final = B0 XOR m2
    # So A depends ONLY on b1, B depends ONLY on b2.
    # The state is the Cartesian product of these.
    # =================================================================
    print("  Verifying P5 closed form...")

    # A should depend only on b1
    a_by_b1 = np.zeros((256, 12), dtype=np.float32)
    for b1 in range(256):
        a_by_b1[b1] = a12_bits[b1, 0, :]  # same for all b2

    a_varies_with_b2 = False
    for b1 in range(256):
        for b2 in range(1, 256):
            if not np.allclose(a12_bits[b1, b2], a12_bits[b1, 0]):
                a_varies_with_b2 = True
                break
        if a_varies_with_b2:
            break

    # B should depend only on b2
    b_by_b2 = np.zeros((256, 12), dtype=np.float32)
    for b2 in range(256):
        b_by_b2[b2] = b12_bits[0, b2, :]  # same for all b1

    b_varies_with_b1 = False
    for b2 in range(256):
        for b1 in range(1, 256):
            if not np.allclose(b12_bits[b1, b2], b12_bits[0, b2]):
                b_varies_with_b1 = True
                break
        if b_varies_with_b1:
            break

    bases["_a_by_b1"] = a_by_b1
    bases["_b_by_b2"] = b_by_b2

    print(f"  P5 verification: A varies with b2? {a_varies_with_b2}")
    print(f"  P5 verification: B varies with b1? {b_varies_with_b1}")

    # XOR popcount: mask12(b1) xor mask12(b2) weight per pair
    xor_popcount = np.zeros((256, 256), dtype=np.float32)
    for b1 in range(256):
        m1 = int(masks_raw[b1])
        for b2 in range(256):
            m2 = int(masks_raw[b2])
            xor_popcount[b1, b2] = float(popcount(m1 ^ m2))
    bases["_xor_popcount"] = xor_popcount

    summary = ", ".join(
        f"{k}[{v.shape[-1] if v.ndim == 2 else v.shape[0]}]"
        for k, v in bases.items() if not k.startswith("_")
    )
    print(f"Per-byte bases: {summary}")
    return bases


# =====================================================================
# Utilities
# =====================================================================

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0)) ** 2))
    if ss_tot < 1e-15:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _linear_probe(X: np.ndarray, Y: np.ndarray) -> float:
    X_b = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    W, _, _, _ = np.linalg.lstsq(X_b, Y, rcond=None)
    return _r2(Y, X_b @ W)


def _r2_scalar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def _wht_axis(a: np.ndarray, axis: int) -> np.ndarray:
    """Unnormalized Walsh-Hadamard transform along axis. Length 256 on that axis."""
    x = np.swapaxes(a, axis, -1).copy()
    n = x.shape[-1]
    assert n == 256
    x = x.reshape(-1, n)

    h = 1
    while h < n:
        x = x.reshape(-1, n // (2 * h), 2 * h)
        u = x[..., :h].copy()
        v = x[..., h : 2 * h].copy()
        x[..., :h] = u + v
        x[..., h : 2 * h] = u - v
        x = x.reshape(-1, n)
        h *= 2

    x = x.reshape(*np.swapaxes(a, axis, -1).shape)
    x = np.swapaxes(x, -1, axis)
    return x


def wht2(mat: np.ndarray) -> np.ndarray:
    """Unnormalized 2D WHT: apply along rows then cols."""
    out = _wht_axis(mat, axis=0)
    out = _wht_axis(out, axis=1)
    return out


# =====================================================================
# Main analysis
# =====================================================================

def correlate_kernel_with_bolmo(
    kernel_bases: dict[str, np.ndarray],
    bolmo_scores: np.ndarray,
    h_pos1: np.ndarray,
    h_pos2: np.ndarray,
    Wq: np.ndarray,
    Wk: np.ndarray,
    model_dir: Path | None = None,
    atlas_version: str = "2.2",
) -> dict[str, Any]:

    results: dict[str, Any] = {}
    n = 65536

    # =================================================================
    # Layer 0: Prepare
    # =================================================================
    print("Layer 0: Preparing...")

    eps = 1e-6
    scores_clipped = np.clip(bolmo_scores, eps, 1.0 - eps)
    logits = np.log(scores_clipped / (1.0 - scores_clipped)).astype(np.float32)
    flat_logits = logits.flatten()
    flat_scores = bolmo_scores.flatten()

    results["scores_mean"] = float(np.mean(bolmo_scores))
    results["logits_mean"] = float(np.mean(logits))

    # =================================================================
    # Layer 1: Additive decomposition
    # =================================================================
    print("Layer 1: Additive decomposition...")

    grand_mean = float(np.mean(logits))
    row_effects = logits.mean(axis=1) - grand_mean
    col_effects = logits.mean(axis=0) - grand_mean
    additive = grand_mean + row_effects[:, None] + col_effects[None, :]
    residual = logits - additive
    flat_residual = residual.flatten()

    var_total = float(np.var(logits))
    var_additive = float(np.var(additive))
    var_residual = float(np.var(residual))

    results["frac_additive"] = var_additive / (var_total + 1e-15)
    results["frac_residual"] = var_residual / (var_total + 1e-15)
    print(f"  Additive: {results['frac_additive']:.4f}  Residual: {results['frac_residual']:.4f}")

    # =================================================================
    # Layer 2: Marginal probing (established results)
    # =================================================================
    print("Layer 2: Marginal probing...")

    per_byte_bases = {k: v for k, v in kernel_bases.items()
                      if not k.startswith("_") and v.ndim == 2 and v.shape[0] == 256}

    for name, basis in per_byte_bases.items():
        r2_row = _linear_probe(basis, row_effects.reshape(-1, 1))
        r2_col = _linear_probe(basis, col_effects.reshape(-1, 1))
        results[f"r2_row_{name}"] = r2_row
        results[f"r2_col_{name}"] = r2_col

    # =================================================================
    # Layer 3: Tensor product completeness (established)
    # =================================================================
    print("Layer 3: Tensor product completeness...")
    tensor_basis = kernel_bases["micro_family_tensor"]
    results["r2_tensor_row"] = _linear_probe(tensor_basis, row_effects.reshape(-1, 1))
    results["r2_tensor_col"] = _linear_probe(tensor_basis, col_effects.reshape(-1, 1))
    perm_check = tensor_basis.T @ kernel_bases["byte_onehot"]
    results["tensor_is_byte_permutation"] = bool(
        np.allclose(perm_check.sum(axis=0), 1.0) and
        np.allclose(perm_check.sum(axis=1), 1.0)
    )

    # =================================================================
    # Layer 4: P5 CLOSED FORM — the fundamental decomposition
    # A_final depends ONLY on b1 (via mask m1)
    # B_final depends ONLY on b2 (via mask m2)
    # The state is (A, B) = independent product.
    # Therefore: boundary_score(b1, b2) should decompose as
    # f(A(b1)) + g(B(b2)) + interaction(A(b1), B(b2))
    # =================================================================
    print("Layer 4: P5 closed-form decomposition...")

    a_by_b1 = kernel_bases["_a_by_b1"]  # [256, 12] ±1 bits of A after b1
    b_by_b2 = kernel_bases["_b_by_b2"]  # [256, 12] ±1 bits of B after b2

    # A(b1) as marginal predictor of row effects
    r2_a_row = _linear_probe(a_by_b1, row_effects.reshape(-1, 1))
    results["r2_row_from_A_bits"] = r2_a_row
    print(f"  A12 bits -> row marginal: R2 = {r2_a_row:.6f}")

    # B(b2) as marginal predictor of col effects
    r2_b_col = _linear_probe(b_by_b2, col_effects.reshape(-1, 1))
    results["r2_col_from_B_bits"] = r2_b_col
    print(f"  B12 bits -> col marginal: R2 = {r2_b_col:.6f}")

    # Full state24 bits [n, 24] = A12[12] concat B12[12]
    state_bits = np.concatenate([
        kernel_bases["_a12_bits"].reshape(n, 12),
        kernel_bases["_b12_bits"].reshape(n, 12),
    ], axis=1)

    r2_state_logit = _linear_probe(state_bits, flat_logits.reshape(-1, 1))
    r2_state_resid = _linear_probe(state_bits, flat_residual.reshape(-1, 1))
    results["r2_logit_from_state24_bits"] = r2_state_logit
    results["r2_residual_from_state24_bits"] = r2_state_resid
    print(f"  State24 bits (24 feats) -> logits:   R2 = {r2_state_logit:.6f}")
    print(f"  State24 bits (24 feats) -> residual: R2 = {r2_state_resid:.6f}")

    # A×B cross products (the interaction term in the P5 decomposition)
    # This is 12×12 = 144 features: every A bit × every B bit
    ab_cross = np.zeros((n, 144), dtype=np.float32)
    a_flat = kernel_bases["_a12_bits"].reshape(n, 12)
    b_flat = kernel_bases["_b12_bits"].reshape(n, 12)
    for i in range(12):
        for j in range(12):
            ab_cross[:, i * 12 + j] = a_flat[:, i] * b_flat[:, j]

    r2_cross_logit = _linear_probe(ab_cross, flat_logits.reshape(-1, 1))
    r2_cross_resid = _linear_probe(ab_cross, flat_residual.reshape(-1, 1))
    results["r2_logit_from_AB_cross"] = r2_cross_logit
    results["r2_residual_from_AB_cross"] = r2_cross_resid
    print(f"  A*B cross (144 feats) -> logits:   R2 = {r2_cross_logit:.6f}")
    print(f"  A*B cross (144 feats) -> residual: R2 = {r2_cross_resid:.6f}")

    # Full P5 descriptor: A bits + B bits + A×B cross = 24 + 144 = 168 features
    full_p5 = np.concatenate([state_bits, ab_cross], axis=1)
    r2_p5_logit = _linear_probe(full_p5, flat_logits.reshape(-1, 1))
    r2_p5_resid = _linear_probe(full_p5, flat_residual.reshape(-1, 1))
    results["r2_logit_from_full_P5"] = r2_p5_logit
    results["r2_residual_from_full_P5"] = r2_p5_resid
    results["dims_full_P5"] = int(full_p5.shape[1])
    print(f"  Full P5 (168 feats) -> logits:   R2 = {r2_p5_logit:.6f}")
    print(f"  Full P5 (168 feats) -> residual: R2 = {r2_p5_resid:.6f}")

    # =================================================================
    # Layer 5: Ontology state index as predictor
    # The state index IS the 65536-dim one-hot.
    # If boundary score is a function on the ontology, this gets R²=1.
    # =================================================================
    print("Layer 5: Ontology state as predictor...")

    state_indices = kernel_bases["_state_after_b1b2"].flatten().astype(np.int64)

    # Since state_index is a permutation of [0, 65535],
    # a one-hot encoding would be 65536×65536 — too large.
    # Instead: the boundary score indexed by state should show structure.

    # Reorder boundary scores by state index (the ontology ordering)
    score_by_state = np.zeros(n, dtype=np.float32)
    logit_by_state = np.zeros(n, dtype=np.float32)
    for idx in range(n):
        b1 = idx // 256
        b2 = idx % 256
        si = int(state_indices[idx])
        score_by_state[si] = bolmo_scores[b1, b2]
        logit_by_state[si] = logits[b1, b2]

    # Check: is the mapping bijective? (P4 says yes)
    results["state_map_is_bijection"] = bool(len(np.unique(state_indices)) == n)

    # Variance of score when grouped by ontology observables
    horizon_flat = kernel_bases["_horizon_of_state"].flatten()
    vertex_flat = kernel_bases["_vertex_of_state"].flatten()

    # Group scores by horizon byte (256 groups of 256 states each)
    horizon_groups = {}
    for i in range(n):
        h = int(horizon_flat[i])
        if h not in horizon_groups:
            horizon_groups[h] = []
        horizon_groups[h].append(flat_logits[i])

    # Between-group variance / total variance = R² of horizon as predictor
    group_means = np.array([np.mean(horizon_groups[h]) for h in sorted(horizon_groups)])
    group_sizes = np.array([len(horizon_groups[h]) for h in sorted(horizon_groups)])
    overall_mean = np.mean(flat_logits)
    ss_between = float(np.sum(group_sizes * (group_means - overall_mean) ** 2))
    ss_total = float(np.sum((flat_logits - overall_mean) ** 2))
    r2_horizon_grouped = ss_between / (ss_total + 1e-15)
    results["r2_logit_horizon_grouped"] = r2_horizon_grouped
    print(f"  Horizon grouping (256 groups) -> logits: R2 = {r2_horizon_grouped:.6f}")

    # Same for vertex (4 groups)
    vertex_groups = {}
    for i in range(n):
        v = int(vertex_flat[i])
        if v not in vertex_groups:
            vertex_groups[v] = []
        vertex_groups[v].append(flat_logits[i])

    v_means = np.array([np.mean(vertex_groups[v]) for v in sorted(vertex_groups)])
    v_sizes = np.array([len(vertex_groups[v]) for v in sorted(vertex_groups)])
    ss_between_v = float(np.sum(v_sizes * (v_means - overall_mean) ** 2))
    r2_vertex_grouped = ss_between_v / (ss_total + 1e-15)
    results["r2_logit_vertex_grouped"] = r2_vertex_grouped
    print(f"  Vertex grouping (4 groups) -> logits:   R2 = {r2_vertex_grouped:.6f}")

    # =================================================================
    # Layer 6: State observables as pair features
    # =================================================================
    print("Layer 6: State observables as pair features...")

    obs_features = np.column_stack([
        horizon_flat,
        vertex_flat,
        kernel_bases["_phase_of_b2"].flatten(),
        kernel_bases["_horizon_distance"].flatten(),
        kernel_bases["_archetype_distance"].flatten(),
    ])

    r2_obs = _linear_probe(obs_features, flat_logits.reshape(-1, 1))
    r2_obs_res = _linear_probe(obs_features, flat_residual.reshape(-1, 1))
    results["r2_logit_state_observables"] = r2_obs
    results["r2_residual_state_observables"] = r2_obs_res
    print(f"  State observables (5 feats) -> logits:   R2 = {r2_obs:.6f}")
    print(f"  State observables (5 feats) -> residual: R2 = {r2_obs_res:.6f}")

    # Horizon one-hot (256 feats) — each horizon byte defines a partition
    horizon_oh = np.zeros((n, 256), dtype=np.float32)
    for i in range(n):
        horizon_oh[i, int(horizon_flat[i])] = 1.0

    r2_h_oh = _linear_probe(horizon_oh, flat_logits.reshape(-1, 1))
    r2_h_oh_res = _linear_probe(horizon_oh, flat_residual.reshape(-1, 1))
    results["r2_logit_horizon_oh"] = r2_h_oh
    results["r2_residual_horizon_oh"] = r2_h_oh_res
    print(f"  Horizon one-hot (256 feats) -> logits:   R2 = {r2_h_oh:.6f}")
    print(f"  Horizon one-hot (256 feats) -> residual: R2 = {r2_h_oh_res:.6f}")

    # =================================================================
    # Layer 7: Family pair structure (established baseline)
    # =================================================================
    print("Layer 7: Family pair baseline...")

    families = np.argmax(kernel_bases["family_oh"], axis=1)
    family_pair_oh = np.zeros((n, 16), dtype=np.float32)
    idx = 0
    for b1 in range(256):
        f1 = int(families[b1])
        for b2 in range(256):
            f2 = int(families[b2])
            family_pair_oh[idx, f1 * 4 + f2] = 1.0
            idx += 1

    r2_fam = _linear_probe(family_pair_oh, flat_logits.reshape(-1, 1))
    results["r2_logit_family_pair"] = r2_fam
    print(f"  Family pair (16 feats) -> logits: R2 = {r2_fam:.6f}")

    for f1 in range(4):
        for f2 in range(4):
            mask_f = (families[:, None] == f1) & (families[None, :] == f2)
            results[f"mean_score_f{f1}f{f2}"] = float(np.mean(bolmo_scores[mask_f]))

    # =================================================================
    # Layer A: Holographic "difference byte" b* = (b1^b2)^0xAA
    # =================================================================
    print("Walsh/Delta Layer A: delta-byte one-hot...")

    delta_oh = np.zeros((n, 256), dtype=np.float32)
    idx = 0
    for b1 in range(256):
        for b2 in range(256):
            db = (b1 ^ b2) ^ 0xAA
            delta_oh[idx, db] = 1.0
            idx += 1

    def _linear_probe_1d(X: np.ndarray, y: np.ndarray) -> float:
        Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
        w, _, _, _ = np.linalg.lstsq(Xb, y.reshape(-1, 1), rcond=None)
        pred = (Xb @ w).reshape(-1)
        return _r2_scalar(y, pred)

    results["r2_logit_delta_byte_oh"] = _linear_probe_1d(delta_oh, flat_logits)
    results["r2_residual_delta_byte_oh"] = _linear_probe_1d(delta_oh, flat_residual)

    print(f"  delta-byte one-hot -> logits:   R2={results['r2_logit_delta_byte_oh']:.6f}")
    print(f"  delta-byte one-hot -> residual: R2={results['r2_residual_delta_byte_oh']:.6f}")

    # =================================================================
    # Layer B: 2D Walsh-Hadamard spectrum of L in intron coords
    # =================================================================
    print("Walsh/Delta Layer B: 2D Walsh spectrum (intron-indexed)...")

    perm = np.arange(256, dtype=np.int64) ^ 0xAA
    Lx = logits[perm][:, perm].astype(np.float32, copy=True)

    F = wht2(Lx)

    energy_by_deg = np.zeros(17, dtype=np.float64)
    total_energy = float(np.sum(F.astype(np.float64) ** 2)) + 1e-12
    for u in range(256):
        wu = int(u).bit_count()
        for v in range(256):
            deg = wu + int(v).bit_count()
            if deg <= 16:
                energy_by_deg[deg] += float(F[u, v]) ** 2

    frac_by_deg = energy_by_deg / total_energy
    results["walsh_energy_frac_by_degree_0_16"] = frac_by_deg.tolist()

    best_r2_by_deg = {}
    for d in (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16):
        mask = np.zeros((256, 256), dtype=np.float32)
        for u in range(256):
            wu = int(u).bit_count()
            for v in range(256):
                if wu + int(v).bit_count() <= d:
                    mask[u, v] = 1.0
        Fm = F * mask
        Lx_approx = wht2(Fm) / (256.0 * 256.0)
        inv_perm = np.empty(256, dtype=np.int64)
        inv_perm[perm] = np.arange(256)
        L_approx = Lx_approx[inv_perm][:, inv_perm]
        best_r2_by_deg[f"deg_le_{d}"] = _r2_scalar(logits.flatten(), L_approx.flatten())

    results["walsh_r2_by_max_degree"] = best_r2_by_deg
    for k, v in best_r2_by_deg.items():
        print(f"  Walsh keep {k}: R2={v:.6f}")

    absF = np.abs(F).reshape(-1)
    order = np.argsort(-absF)
    for K in (16, 32, 64, 128, 256, 512, 1024, 2048):
        Fm = np.zeros_like(F)
        sel = order[:K]
        Fm.reshape(-1)[sel] = F.reshape(-1)[sel]
        Lx_approx = wht2(Fm) / (256.0 * 256.0)
        inv_perm = np.empty(256, dtype=np.int64)
        inv_perm[perm] = np.arange(256)
        L_approx = Lx_approx[inv_perm][:, inv_perm]
        results[f"walsh_r2_topK_{K}"] = _r2_scalar(logits.flatten(), L_approx.flatten())
        print(f"  Walsh topK={K}: R2={results[f'walsh_r2_topK_{K}']:.6f}")

    # Diagonal energy (u==v): functions of x1 xor x2 have Walsh support on diagonal
    total_F_sq = float(np.sum(F.astype(np.float64) ** 2)) + 1e-12
    diag_energy_logits = sum(float(F[u, u]) ** 2 for u in range(256))
    results["walsh_diagonal_energy_frac_logits"] = float(diag_energy_logits / total_F_sq)
    print(f"  Walsh diagonal (u==v) energy frac (logits): {results['walsh_diagonal_energy_frac_logits']:.6f}")

    # =================================================================
    # Layer C: 2D Walsh on RESIDUAL (interaction-only surface)
    # =================================================================
    print("Walsh/Delta Layer C: 2D Walsh on residual (intron-indexed)...")

    Rx = residual[perm][:, perm].astype(np.float32, copy=True)
    F_res = wht2(Rx)

    total_energy_res = float(np.sum(F_res.astype(np.float64) ** 2)) + 1e-12
    energy_by_deg_res = np.zeros(17, dtype=np.float64)
    for u in range(256):
        wu = int(u).bit_count()
        for v in range(256):
            deg = wu + int(v).bit_count()
            if deg <= 16:
                energy_by_deg_res[deg] += float(F_res[u, v]) ** 2

    frac_by_deg_res = energy_by_deg_res / total_energy_res
    results["walsh_residual_energy_frac_by_degree_0_16"] = frac_by_deg_res.tolist()

    best_r2_res_deg = {}
    for d in (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16):
        mask = np.zeros((256, 256), dtype=np.float32)
        for u in range(256):
            wu = int(u).bit_count()
            for v in range(256):
                if wu + int(v).bit_count() <= d:
                    mask[u, v] = 1.0
        Fm_res = F_res * mask
        Rx_approx = wht2(Fm_res) / (256.0 * 256.0)
        inv_perm = np.empty(256, dtype=np.int64)
        inv_perm[perm] = np.arange(256)
        R_approx = Rx_approx[inv_perm][:, inv_perm]
        best_r2_res_deg[f"deg_le_{d}"] = _r2_scalar(residual.flatten(), R_approx.flatten())

    results["walsh_residual_r2_by_max_degree"] = best_r2_res_deg
    for k, v in best_r2_res_deg.items():
        print(f"  Walsh residual keep {k}: R2={v:.6f}")

    absF_res = np.abs(F_res).reshape(-1)
    order_res = np.argsort(-absF_res)
    for K in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
        Fm_res = np.zeros_like(F_res)
        sel = order_res[:K]
        Fm_res.reshape(-1)[sel] = F_res.reshape(-1)[sel]
        Rx_approx = wht2(Fm_res) / (256.0 * 256.0)
        inv_perm = np.empty(256, dtype=np.int64)
        inv_perm[perm] = np.arange(256)
        R_approx = Rx_approx[inv_perm][:, inv_perm]
        results[f"walsh_residual_r2_topK_{K}"] = _r2_scalar(residual.flatten(), R_approx.flatten())
        print(f"  Walsh residual topK={K}: R2={results[f'walsh_residual_r2_topK_{K}']:.6f}")

    diag_energy_res = sum(float(F_res[u, u]) ** 2 for u in range(256))
    results["walsh_diagonal_energy_frac_residual"] = float(diag_energy_res / total_energy_res)
    print(f"  Walsh diagonal (u==v) energy frac (residual): {results['walsh_diagonal_energy_frac_residual']:.6f}")

    # Single bolmo_adaptor.npz: additive + full ranked Walsh residual (slice K at runtime)
    project_root = Path(__file__).resolve().parents[2]
    analysis_dir = project_root / "data" / "cache" / "blomo_port" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    u_all = (order_res // 256).astype(np.uint8)
    v_all = (order_res % 256).astype(np.uint8)
    c_all = F_res.reshape(-1)[order_res].astype(np.float32)
    adaptor_path = analysis_dir / "bolmo_adaptor.npz"
    model_id_str = str(model_dir.resolve()) if model_dir is not None else ""
    np.savez(
        adaptor_path,
        model_id=np.bytes_(model_id_str.encode("utf-8")),
        atlas_version=np.bytes_(atlas_version.encode("utf-8")),
        grand_mean=np.float32(grand_mean),
        row_effects=row_effects.astype(np.float32),
        col_effects=col_effects.astype(np.float32),
        u=u_all,
        v=v_all,
        coeffs=c_all,
    )
    results["bolmo_adaptor_path"] = str(adaptor_path)
    print(f"  Exported bolmo_adaptor.npz (ranked u,v,coeffs + additive) -> {adaptor_path}")

    # Evaluate all exported K levels from ranked slice (residual R2 + full-logit R2)
    b1_all = np.repeat(np.arange(256, dtype=np.int64), 256)
    b2_all = np.tile(np.arange(256, dtype=np.int64), 256)
    Ks = (2048, 4096, 8192, 16384, 32768)
    results["walsh_residual_r2_by_K"] = {}
    results["walsh_full_logit_r2_by_K"] = {}
    for K in Ks:
        u_k = u_all[:K]
        v_k = v_all[:K]
        c_k = c_all[:K]
        pred_res = np.asarray(predict_residual_walsh(b1_all, b2_all, u_k, v_k, c_k), dtype=np.float64)
        r2_res = _r2_scalar(flat_residual, pred_res)
        r2_full = _r2_scalar(flat_logits, additive.flatten() + pred_res)
        results["walsh_residual_r2_by_K"][K] = float(r2_res)
        results["walsh_full_logit_r2_by_K"][K] = float(r2_full)
        print(f"  K={K}: residual R2={r2_res:.6f}  full-logit R2={r2_full:.6f}")
    results["walsh_residual_evaluator_r2"] = results["walsh_residual_r2_by_K"][2048]
    results["walsh_full_logit_evaluator_r2"] = results["walsh_full_logit_r2_by_K"][2048]

    # =================================================================
    # Layer 8: FULL COMPARISON
    # =================================================================
    print("\nLayer 8: Full comparison...")

    # Old flat probe
    mask12_basis = kernel_bases["mask12"]
    mask_xor_bits = np.zeros((n, 12), dtype=np.float32)
    idx = 0
    for b1 in range(256):
        for b2 in range(256):
            mask_xor_bits[idx] = mask12_basis[b1] * mask12_basis[b2]
            idx += 1
    old_flat = np.concatenate([mask_xor_bits, family_pair_oh], axis=1)
    r2_old = _linear_probe(old_flat, flat_logits.reshape(-1, 1))
    results["r2_logit_old_flat"] = r2_old

    print(f"  === RESULTS ===")
    print(f"  Family pair alone   (16 feats):  R2 = {r2_fam:.6f}")
    print(f"  Old flat XOR+fam    (28 feats):  R2 = {r2_old:.6f}")
    print(f"  State24 bits        (24 feats):  R2 = {r2_state_logit:.6f}")
    print(f"  A*B cross          (144 feats):  R2 = {r2_cross_logit:.6f}")
    print(f"  Full P5            (168 feats):  R2 = {r2_p5_logit:.6f}")
    print(f"  State observables    (5 feats):  R2 = {r2_obs:.6f}")
    print(f"  Horizon one-hot    (256 feats):  R2 = {r2_h_oh:.6f}")
    print(f"  Horizon grouped    (256 groups): R2 = {r2_horizon_grouped:.6f}")
    print(f"  --- Walsh (bolmo_adaptor slice) ---")
    print(f"  K=2048:  residual R2 = {results['walsh_residual_r2_by_K'][2048]:.6f}  full-logit R2 = {results['walsh_full_logit_r2_by_K'][2048]:.6f}")
    print(f"  K=16384: residual R2 = {results['walsh_residual_r2_by_K'][16384]:.6f}  full-logit R2 = {results['walsh_full_logit_r2_by_K'][16384]:.6f}")
    print(f"  K=32768: residual R2 = {results['walsh_residual_r2_by_K'][32768]:.6f}  full-logit R2 = {results['walsh_full_logit_r2_by_K'][32768]:.6f}")

    print("\nAll layers computed.")
    return results


# =====================================================================
# Phenomenology: runtime evaluator from exported Walsh residual coeffs
# =====================================================================

# Popcount lookup 0..255 for fast parity in evaluator
_POP8 = np.array([int(i).bit_count() for i in range(256)], dtype=np.int32)


def predict_residual_walsh(
    b1: int | np.ndarray,
    b2: int | np.ndarray,
    u_idx: np.ndarray,
    v_idx: np.ndarray,
    coeffs: np.ndarray,
) -> float | np.ndarray:
    """
    Reconstruct residual (additive-free boundary logit component) from top-K
    Walsh coefficients. Intron coords: x = byte ^ 0xAA.
    Value = (1/65536) * sum_k coeffs[k] * (-1)^(popcount(u_k & x1) + popcount(v_k & x2)).
    """
    scalar = isinstance(b1, (int, np.integer))
    if scalar:
        b1 = np.array([int(b1)], dtype=np.int64)
        b2 = np.array([int(b2)], dtype=np.int64)
    b1 = np.asarray(b1, dtype=np.int64).ravel() & 0xFF
    b2 = np.asarray(b2, dtype=np.int64).ravel() & 0xFF
    x1 = (b1 ^ 0xAA).astype(np.int32)
    x2 = (b2 ^ 0xAA).astype(np.int32)

    K = coeffs.shape[0]
    out = np.zeros(b1.size, dtype=np.float64)
    for k in range(K):
        u = int(u_idx[k]) & 0xFF
        v = int(v_idx[k]) & 0xFF
        c = float(coeffs[k])
        parity = _POP8[(x1 & u)] + _POP8[(x2 & v)]
        sign = 1.0 - 2.0 * np.asarray(parity & 1, dtype=np.float64)
        out += c * sign

    out /= 256.0 * 256.0
    return float(out[0]) if scalar else out


def load_walsh_residual_phenomenology(npz_path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load u, v, coeffs from walsh_residual_topK.npz. Returns (u_idx, v_idx, coeffs)."""
    data = np.load(npz_path)
    return data["u"], data["v"], data["coeffs"]


def load_bolmo_adaptor(path: Path | str) -> dict[str, Any]:
    """
    Load bolmo_adaptor.npz. Returns dict with grand_mean, row_effects, col_effects, u, v, coeffs,
    and optionally model_id, atlas_version. u,v,coeffs are full ranked (length 65536); slice at runtime.
    """
    data = np.load(path)
    def _decode(b: np.ndarray) -> str:
        if b is None or b.size == 0:
            return ""
        x = b.item() if b.ndim == 0 else b.tobytes().rstrip(b"\x00")
        return x.decode("utf-8") if isinstance(x, bytes) else str(x)
    return {
        "grand_mean": float(data["grand_mean"]),
        "row_effects": data["row_effects"],
        "col_effects": data["col_effects"],
        "u": data["u"],
        "v": data["v"],
        "coeffs": data["coeffs"],
        "model_id": _decode(data["model_id"]) if "model_id" in data else "",
        "atlas_version": _decode(data["atlas_version"]) if "atlas_version" in data else "",
    }


def bolmo_boundary_logit_from_adaptor(
    b1: int | np.ndarray,
    b2: int | np.ndarray,
    adaptor: dict[str, Any] | Path | str,
    K: int = 16384,
) -> float | np.ndarray:
    """
    Reconstruct Bolmo boundary logit from adaptor: additive (P5) + top-K Walsh residual.
    adaptor: path to bolmo_adaptor.npz or dict from load_bolmo_adaptor.
    K: number of Walsh coefficients to use (default 16384 ~0.92 full-logit R2).
    """
    if isinstance(adaptor, (Path, str)):
        adaptor = load_bolmo_adaptor(adaptor)
    b1 = np.asarray(b1, dtype=np.uint8).ravel()
    b2 = np.asarray(b2, dtype=np.uint8).ravel()
    add = adaptor["grand_mean"] + adaptor["row_effects"][b1] + adaptor["col_effects"][b2]
    u = adaptor["u"][:K]
    v = adaptor["v"][:K]
    coeffs = adaptor["coeffs"][:K]
    res = predict_residual_walsh(b1, b2, u, v, coeffs)
    out = add + np.asarray(res, dtype=np.float64)
    return float(out[0]) if out.size == 1 else out.astype(np.float32)


def bolmo_boundary_prob_from_adaptor(
    b1: int | np.ndarray,
    b2: int | np.ndarray,
    adaptor: dict[str, Any] | Path | str,
    K: int = 16384,
) -> float | np.ndarray:
    """Boundary probability from adaptor: p = exp(logit) / (1 + exp(logit))."""
    logits = bolmo_boundary_logit_from_adaptor(b1, b2, adaptor, K=K)
    logits = np.asarray(logits, dtype=np.float64)
    p = np.exp(np.clip(logits, -50.0, 50.0)) / (1.0 + np.exp(np.clip(logits, -50.0, 50.0)))
    return float(p) if p.size == 1 else p.astype(np.float32)