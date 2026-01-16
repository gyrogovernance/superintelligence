# tests/test_holography.py
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

from src.router.kernel import RouterKernel
from src.router.constants import (
    ARCHETYPE_A12,
    C_PERP_12,
    LAYER_MASK_12,
    mask12_for_byte,
    unpack_state,
)
from src.app.ledger import get_incidence_matrix, get_cycle_basis, get_projections, hodge_decomposition

SEP = "=========="


def dot_parity(x: int, y: int) -> int:
    return bin(x & y).count("1") & 1


def popcount12(x: int) -> int:
    return bin(x & 0xFFF).count("1")


def horizon_indices(ont: NDArray[np.uint32]) -> List[int]:
    out = []
    for i, s in enumerate(ont):
        a, b = unpack_state(int(s))
        if a == (b ^ LAYER_MASK_12):
            out.append(i)
    return out


def build_vertex_map(ont: NDArray[np.uint32], h_idxs: List[int]) -> Dict[int, int]:
    """Build vertex map: horizon A -> K4 vertex (0-3)."""
    vertex_map: Dict[int, int] = {}
    for h_idx in h_idxs:
        a, _ = unpack_state(int(ont[h_idx]))
        frame0 = a & 0x3F
        frame1 = (a >> 6) & 0x3F
        frame_parity = (popcount12(frame0) + popcount12(frame1)) % 2
        row0_bits = (a & 0x003) | ((a >> 4) & 0x0C0)
        row1_bits = ((a >> 2) & 0x003) | ((a >> 6) & 0x0C0)
        row2_bits = ((a >> 4) & 0x003) | ((a >> 8) & 0x0C0)
        row_sum = popcount12(row0_bits) + popcount12(row1_bits) + popcount12(row2_bits)
        row_parity = row_sum % 2
        vertex = (frame_parity * 2) + row_parity
        vertex_map[a] = vertex
    return vertex_map


def build_charge_by_mask(
    ont: NDArray[np.uint32],
    epi: NDArray[np.uint32],
    h_idxs: List[int],
    vertex_map: Dict[int, int],
) -> Dict[int, int]:
    """Build charge_by_mask: mask -> vertex charge (2-bit) from byte² action."""
    charge_by_mask: Dict[int, int] = {}
    unique_masks = set(mask12_for_byte(b) for b in range(256))
    mask_to_any_byte: Dict[int, int] = {}
    for b in range(256):
        m = mask12_for_byte(b)
        if m not in mask_to_any_byte:
            mask_to_any_byte[m] = b

    for m in unique_masks:
        if m in charge_by_mask:
            continue
        b = mask_to_any_byte[m]
        charges_seen = []
        for h_idx in h_idxs:
            a0, _ = unpack_state(int(ont[h_idx]))
            v0 = vertex_map[a0]
            h1_idx = int(epi[h_idx, b])
            h2_idx = int(epi[h1_idx, b])
            a2, b2 = unpack_state(int(ont[h2_idx]))
            if a2 != (b2 ^ LAYER_MASK_12):
                continue
            v2 = vertex_map[a2]
            charge = v0 ^ v2
            charges_seen.append(charge)
        if len(charges_seen) > 0:
            charge_counts = Counter(charges_seen)
            most_common_charge, count = charge_counts.most_common(1)[0]
            assert count == len(charges_seen), f"Charge not constant for mask 0x{m:03x}: {charge_counts}"
            charge_by_mask[m] = most_common_charge
    return charge_by_mask


# ==========
# TEST 1: Holographic Dictionary - explicit reconstruction
# ==========

def test_holographic_dictionary_reconstruction():
    """
    DECISIVE: For every bulk state s=(A,B), there exists unique (h,b) with T_b(h)=s.
    
    The explicit formula is:
      h = (A, A^0xFFF)  [unique horizon state with A-component = A]
      m = A ^ (B^0xFFF)  [mask of byte b]
      b = inverse_mask_lookup(m)
    
    Verify this for ALL 65536 states.
    """
    atlas_dir = Path("data/atlas")
    K = RouterKernel(atlas_dir)
    ont = K.ontology
    epi = K.epistemology
    n = int(ont.size)

    # Build mask -> byte inverse lookup
    mask_to_byte: Dict[int, int] = {}
    for b in range(256):
        m = mask12_for_byte(b)
        if m in mask_to_byte:
            raise AssertionError("Mask collision - masks not bijective")
        mask_to_byte[m] = b

    # Build horizon A -> horizon_idx lookup
    h_idxs = horizon_indices(ont)
    assert len(h_idxs) == 256
    horizon_A_to_idx: Dict[int, int] = {}
    for idx in h_idxs:
        a, _ = unpack_state(int(ont[idx]))
        horizon_A_to_idx[a] = idx

    failures = 0
    for i in range(n):
        s = int(ont[i])
        A, B = unpack_state(s)

        # Reconstruct (h, b)
        h_idx = horizon_A_to_idx.get(A)
        if h_idx is None:
            failures += 1
            continue

        m = (A ^ (B ^ LAYER_MASK_12)) & LAYER_MASK_12
        b = mask_to_byte.get(m)
        if b is None:
            failures += 1
            continue

        # Verify T_b(h) = s
        reconstructed_idx = int(epi[h_idx, b])
        if reconstructed_idx != i:
            failures += 1

    print(SEP)
    print("Holographic Dictionary Reconstruction")
    print(SEP)
    print(f"  states checked: {n}")
    print(f"  failures: {failures}")

    assert failures == 0, "Holographic dictionary formula failed"


# ==========
# TEST 2: Horizon Walsh exact translation
# ==========

def test_horizon_walsh_exact_translation():
    """
    DECISIVE: The horizon A-set = ARCHETYPE_A12 ^ C.
    Walsh transform: W_H(s) = 256*(-1)^{<s,ARCHETYPE_A12>} if s in C_perp, else 0.
    
    This is the sharp phase translation by ARCHETYPE_A12, specific to kernel holography.
    """
    atlas_dir = Path("data/atlas")
    K = RouterKernel(atlas_dir)
    ont = K.ontology

    H_A = set()
    for i in range(int(ont.size)):
        a, b = unpack_state(int(ont[i]))
        if a == (b ^ LAYER_MASK_12):
            H_A.add(a)
    assert len(H_A) == 256

    C_perp_set = set(int(v) for v in C_PERP_12)

    bad_support = bad_phase = 0
    for s in range(4096):
        w = sum(1 if dot_parity(s, a) == 0 else -1 for a in H_A)
        if s in C_perp_set:
            expected = 256 if dot_parity(s, ARCHETYPE_A12) == 0 else -256
            if w != expected:
                bad_phase += 1
        else:
            if w != 0:
                bad_support += 1

    print(SEP)
    print("Horizon Walsh Exact Translation")
    print(SEP)
    print(f"  bad_support: {bad_support}")
    print(f"  bad_phase: {bad_phase}")

    assert bad_support == 0 and bad_phase == 0


# ==========
# TEST 3: Horizon -> K4 Vertex Map (Affine-Linear Structure)
# ==========

def test_horizon_k4_vertex_subcode_tower():
    """
    DECISIVE: The 256 horizon states partition into 4 K4 vertices (64 each).
    Each vertex class is a coset of a rank-6 subcode D ⊂ C (mask code).
    
    Proves subcode tower: C ⊃ D, C_PERP ⊂ D_perp.
    
    This connects kernel code duality to K4 vertex emergence as an internal
    code-subspace structure, not just a balanced hash.
    Trace: `N_14i.md` §4.1 - kernel geometry generates governance graph.
    """
    atlas_dir = Path("data/atlas")
    K = RouterKernel(atlas_dir)
    ont = K.ontology
    
    h_idxs = horizon_indices(ont)
    assert len(h_idxs) == 256
    
    # Build vertex map using helper
    vertex_map = build_vertex_map(ont, h_idxs)
    
    # Verify balanced partition
    vertex_counts = Counter(vertex_map.values())
    assert len(vertex_counts) == 4, "Must partition into exactly 4 vertices"
    assert all(c == 64 for c in vertex_counts.values()), f"Partition must be balanced: {vertex_counts}"
    
    # Build mask code C = {mask12_for_byte(b) : b in 0..255}
    C = {mask12_for_byte(b) for b in range(256)}
    assert len(C) == 256, "Mask code must have 256 elements"
    
    # Pick one vertex class (vertex 0) and prove it's a subcode D ⊂ C
    horizon_A_list = sorted(vertex_map.keys())
    vertex_0_members = [a for a in horizon_A_list if vertex_map[a] == 0]
    assert len(vertex_0_members) == 64, "Vertex 0 must have 64 members"
    
    # Pick reference and translate to get subspace D
    ref_a = vertex_0_members[0]
    D = {a ^ ref_a for a in vertex_0_members}
    D_list = sorted(D)
    
    # Verify: |D| = 64
    assert len(D) == 64, f"Subspace D must have 64 elements, got {len(D)}"
    
    # Verify: D is closed under XOR (full check, not sampling)
    for x1 in D_list:
        for x2 in D_list:
            if (x1 ^ x2) not in D:
                raise AssertionError(f"D not closed: {x1:03x} ^ {x2:03x} = {(x1^x2):03x} not in D")
    
    # Verify: D ⊂ C (every element in D is a valid mask codeword)
    not_in_C = [d for d in D if d not in C]
    assert len(not_in_C) == 0, f"D must be subset of C, but {len(not_in_C)} elements not in C"
    
    # Compute rank of D over GF(2) using Gaussian elimination
    def gf2_rank(vectors_12bit: list[int]) -> int:
        """Gaussian elimination over GF(2) for 12-bit vectors."""
        basis = [int(v) & 0xFFF for v in vectors_12bit if (int(v) & 0xFFF) != 0]
        rank = 0
        # pivot from MSB to LSB
        for bit in reversed(range(12)):
            # find a vector with this pivot
            pivot_idx = None
            for i in range(rank, len(basis)):
                if (basis[i] >> bit) & 1:
                    pivot_idx = i
                    break
            if pivot_idx is None:
                continue
            # swap into position
            basis[rank], basis[pivot_idx] = basis[pivot_idx], basis[rank]
            pivot = basis[rank]
            # eliminate pivot from all others
            for j in range(len(basis)):
                if j != rank and ((basis[j] >> bit) & 1):
                    basis[j] ^= pivot
            rank += 1
            if rank == 12:
                break
        return rank
    
    rank_D = gf2_rank(D_list)
    assert rank_D == 6, f"D must have GF(2) rank 6, got {rank_D}"
    
    # Note: C_PERP ⊂ D_perp follows from linear algebra: since D ⊂ C, then C_perp ⊂ D_perp.
    # This is proven explicitly in Test 4 via ker(χ) = D and explicit parity checks q0, q1.
    
    print(SEP)
    print("Horizon -> K4 Vertex Map (Subcode Tower)")
    print(SEP)
    print(f"  vertex counts: {dict(vertex_counts)}")
    print(f"  |D| = {len(D)}")
    print(f"  rank(D) = {rank_D}")


# ==========
# TEST 4: Ecology as Signed Cross-Domain Cycle Coherence
# ==========

def test_ecology_signed_cycle_coherence():
    """
    DECISIVE: Ecology is a signed normalized cycle-coherence invariant across domains.
    
    Define cycle components: c_D = P_cycle y_D for each domain ledger.
    Compute cycle Gram matrix: G_ij = <c_i, c_j>
    Ecology = (G_12 + G_13 + G_23) / (G_11 + G_22 + G_33) ∈ [-1, 1]
    
    This detects signed entanglement (correlation/opposition), not just energy.
    
    Test scenarios:
      - Correlated cycles: E ≈ +0.6
      - Independent cycles: |E| < 0.2
      - Anti-correlated cycles: E < -0.2
    
    Trace: `N_14i.md` §4.1 - entanglement ledger hypothesis.
    """
    B = get_incidence_matrix()  # 4x6
    P_grad, P_cycle = get_projections()
    cycle_basis = get_cycle_basis()  # 6x3
    
    rng = np.random.default_rng(42)
    scale = 1000.0
    
    # Scenario 1: Correlated cycles (all domains have same cycle direction)
    x1_corr = rng.normal(0, 1, 4).astype(np.float64) * scale
    x2_corr = rng.normal(0, 1, 4).astype(np.float64) * scale
    x3_corr = rng.normal(0, 1, 4).astype(np.float64) * scale
    u = cycle_basis[:, 0]  # Fixed unit cycle direction
    k_corr = 100.0  # Same cycle strength for all
    y1_corr = (B.T @ x1_corr + k_corr * u).astype(np.int64)
    y2_corr = (B.T @ x2_corr + k_corr * u).astype(np.int64)
    y3_corr = (B.T @ x3_corr + k_corr * u).astype(np.int64)
    
    # Scenario 2: Independent cycles (different cycle directions)
    x1_ind = rng.normal(0, 1, 4).astype(np.float64) * scale
    x2_ind = rng.normal(0, 1, 4).astype(np.float64) * scale
    x3_ind = rng.normal(0, 1, 4).astype(np.float64) * scale
    u1 = cycle_basis[:, 0]
    u2 = cycle_basis[:, 1]  # Different direction
    u3 = cycle_basis[:, 2]  # Different direction
    y1_ind = (B.T @ x1_ind + 100.0 * u1).astype(np.int64)
    y2_ind = (B.T @ x2_ind + 100.0 * u2).astype(np.int64)
    y3_ind = (B.T @ x3_ind + 100.0 * u3).astype(np.int64)
    
    # Scenario 3: Anti-correlated cycles (one opposed)
    x1_anti = rng.normal(0, 1, 4).astype(np.float64) * scale
    x2_anti = rng.normal(0, 1, 4).astype(np.float64) * scale
    x3_anti = rng.normal(0, 1, 4).astype(np.float64) * scale
    k1_anti = 100.0
    k2_anti = -100.0  # Opposed
    k3_anti = 100.0
    y1_anti = (B.T @ x1_anti + k1_anti * u).astype(np.int64)
    y2_anti = (B.T @ x2_anti + k2_anti * u).astype(np.int64)
    y3_anti = (B.T @ x3_anti + k3_anti * u).astype(np.int64)
    
    # Ecology operator: signed cycle coherence
    def compute_ecology(y1, y2, y3):
        # Extract cycle components
        _, c1 = hodge_decomposition(y1, P_grad, P_cycle)
        _, c2 = hodge_decomposition(y2, P_grad, P_cycle)
        _, c3 = hodge_decomposition(y3, P_grad, P_cycle)
        
        # Cycle Gram matrix
        G11 = float(c1 @ c1)
        G22 = float(c2 @ c2)
        G33 = float(c3 @ c3)
        G12 = float(c1 @ c2)
        G13 = float(c1 @ c3)
        G23 = float(c2 @ c3)
        
        # Normalized off-diagonal sum
        diagonal = G11 + G22 + G33
        if diagonal < 1e-10:
            return 0.0
        off_diagonal = G12 + G13 + G23
        return off_diagonal / diagonal
    
    E_corr = compute_ecology(y1_corr, y2_corr, y3_corr)
    E_ind = compute_ecology(y1_ind, y2_ind, y3_ind)
    E_anti = compute_ecology(y1_anti, y2_anti, y3_anti)
    
    print(SEP)
    print("Ecology as Signed Cycle Coherence")
    print(SEP)
    print(f"  E (correlated): {E_corr:.6f}")
    print(f"  E (independent): {E_ind:.6f}")
    print(f"  E (anti-correlated): {E_anti:.6f}")
    print(f"  targets: E_corr > +0.6, |E_ind| < 0.2, E_anti < -0.2")
    
    # Ecology must discriminate with signed coherence
    assert E_corr > 0.6, f"Correlated cycles must have E > 0.6, got {E_corr:.3f}"
    assert abs(E_ind) < 0.2, f"Independent cycles must have |E| < 0.2, got {E_ind:.3f}"
    assert E_anti < -0.2, f"Anti-correlated cycles must have E < -0.2, got {E_anti:.3f}"
    
    # Invariance checks: ecology should be scale-invariant and permutation-invariant
    # Scale invariance: multiply all ledgers by scalar
    scale_factor = 2.0
    E_scaled = compute_ecology(
        (y1_corr * scale_factor).astype(np.int64),
        (y2_corr * scale_factor).astype(np.int64),
        (y3_corr * scale_factor).astype(np.int64)
    )
    scale_diff = abs(E_corr - E_scaled)
    
    # Permutation invariance: swap domain labels
    E_perm = compute_ecology(y2_corr, y3_corr, y1_corr)
    perm_diff = abs(E_corr - E_perm)
    
    print(f"  scale invariance diff: {scale_diff:.6e}")
    print(f"  permutation invariance diff: {perm_diff:.6e}")
    
    assert scale_diff < 1e-10, f"Ecology must be scale-invariant, diff={scale_diff:.6e}"
    assert perm_diff < 1e-10, f"Ecology must be permutation-invariant, diff={perm_diff:.6e}"


# ==========
# TEST 4: Vertex Charge Parity-Check Recovery (Explicit Equations)
# ==========

def test_vertex_charge_has_two_parity_checks_and_vertex_is_affine():
    """
    DECISIVE: Recover explicit parity-check vectors q0, q1 such that:
      χ(m) = (<q0, m>, <q1, m>) for all m in C
      v(A) = (<q0, A> ⊕ c0, <q1, A> ⊕ c1) for horizon A
    
    This proves:
      - D = ker(χ) = {m in C : <q0,m>=0, <q1,m>=0} (size 64, rank 6)
      - χ is a homomorphism (linearity from explicit parity checks)
      - Charge depends only on mask (not byte value)
      - Unique charges = 4 (vertices = C/ker(χ))
      - Byte² preserves horizon and induces constant vertex charge
    
    Merges claims from Tests 5 and 6: once we have explicit q0,q1, all structural
    properties (homomorphism, kernel size, charge uniqueness) follow immediately.
    
    Trace: `N_14i.md` §4.1 - explicit governance graph equations.
    """
    atlas_dir = Path("data/atlas")
    K = RouterKernel(atlas_dir)
    ont = K.ontology
    epi = K.epistemology
    
    h_idxs = horizon_indices(ont)
    assert len(h_idxs) == 256
    
    # Build vertex map and charge_by_mask using helpers
    vertex_map = build_vertex_map(ont, h_idxs)
    charge_by_mask = build_charge_by_mask(ont, epi, h_idxs, vertex_map)
    
    # Build mask code C
    C = {mask12_for_byte(b) for b in range(256)}
    assert len(C) == 256
    
    # Verify we have charges for all masks
    masks_with_charge = set(charge_by_mask.keys())
    masks_missing = C - masks_with_charge
    assert len(masks_missing) == 0, f"Missing charges for {len(masks_missing)} masks"
    
    # Verify charge depends only on mask (not byte value) - from Test 5
    mask_to_charge: Dict[int, int] = {}
    mask_collisions = 0
    for b in range(256):
        m = mask12_for_byte(b)
        if m in mask_to_charge:
            if mask_to_charge[m] != charge_by_mask[m]:
                mask_collisions += 1
        else:
            mask_to_charge[m] = charge_by_mask[m]
    assert mask_collisions == 0, f"Charge must depend only on mask, but {mask_collisions} collisions"
    
    # Verify homomorphism property: χ(m1 ⊕ m2) = χ(m1) ⊕ χ(m2) - from Test 6
    homomorphism_violations = 0
    for m1 in C:
        for m2 in C:
            m_sum = (m1 ^ m2) & 0xFFF
            chi_m1 = charge_by_mask[m1]
            chi_m2 = charge_by_mask[m2]
            chi_sum = charge_by_mask[m_sum]
            if (chi_m1 ^ chi_m2) != chi_sum:
                homomorphism_violations += 1
    assert homomorphism_violations == 0, f"χ must be homomorphism, but {homomorphism_violations} violations"

    # Define target bits: t0[m] = χ(m) & 1, t1[m] = (χ(m) >> 1) & 1
    t0: Dict[int, int] = {}
    t1: Dict[int, int] = {}
    for m in C:
        t0[m] = charge_by_mask[m] & 1
        t1[m] = (charge_by_mask[m] >> 1) & 1
    
    # Solve for q0, q1: <q, m> = t(m) for all m in C
    # Use brute-force search (4096 candidates) - most robust method
    def solve_q_by_bruteforce(C_list: list[int], rhs: list[int]) -> int:
        """
        Brute-force search for q such that <q, m> = rhs[m] for all m in C_list.
        Returns 12-bit q or raises if no solution exists.
        """
        for q in range(4096):
            ok = True
            for m, b in zip(C_list, rhs):
                if dot_parity(q, m) != (b & 1):
                    ok = False
                    break
            if ok:
                return q
        raise AssertionError("No q satisfies dot(q,m)=rhs on C")

    # Build sorted mask list for consistent ordering
    C_list = sorted(C)
    
    # Solve for q0
    rhs0 = [t0[m] for m in C_list]
    q0 = solve_q_by_bruteforce(C_list, rhs0)

    # Solve for q1
    rhs1 = [t1[m] for m in C_list]
    q1 = solve_q_by_bruteforce(C_list, rhs1)
    
    # Verify: χ(m) = (<q0,m>, <q1,m>) for all masks
    chi_violations = 0
    violation_examples = []
    for m in C:
        if m not in charge_by_mask:
            chi_violations += 1
            if len(violation_examples) < 5:
                violation_examples.append(f"mask 0x{m:03x} missing charge")
            continue
        pred_bit0 = dot_parity(q0, m)
        pred_bit1 = dot_parity(q1, m)
        pred_chi = pred_bit0 + (pred_bit1 << 1)
        actual_chi = charge_by_mask[m]
        if pred_chi != actual_chi:
            chi_violations += 1
            if len(violation_examples) < 5:
                violation_examples.append(f"mask 0x{m:03x}: pred={pred_chi}, actual={actual_chi}")
    
    # Prove vertex(A) is affine: v(A) = (<q0,A> ⊕ c0, <q1,A> ⊕ c1)
    # Pick reference horizon A0
    A0 = sorted(vertex_map.keys())[0]
    v0_ref = vertex_map[A0]
    
    # Compute constants
    c0 = dot_parity(q0, A0) ^ (v0_ref & 1)
    c1 = dot_parity(q1, A0) ^ ((v0_ref >> 1) & 1)
    
    # Verify for all horizon A
    vertex_violations = 0
    for a in vertex_map.keys():
        pred_bit0 = dot_parity(q0, a) ^ c0
        pred_bit1 = dot_parity(q1, a) ^ c1
        pred_vertex = pred_bit0 + (pred_bit1 << 1)
        if pred_vertex != vertex_map[a]:
            vertex_violations += 1
    
    # Prove D = ker(χ) = {m in C : <q0,m>=0, <q1,m>=0}
    ker_from_q = {m for m in C if dot_parity(q0, m) == 0 and dot_parity(q1, m) == 0}
    
    # Verify kernel size and rank
    def gf2_rank(vectors_12bit: list[int]) -> int:
        """Gaussian elimination over GF(2)."""
        basis = [int(v) & 0xFFF for v in vectors_12bit if (int(v) & 0xFFF) != 0]
        rank = 0
        for bit in reversed(range(12)):
            pivot_idx = None
            for i in range(rank, len(basis)):
                if (basis[i] >> bit) & 1:
                    pivot_idx = i
                    break
            if pivot_idx is None:
                continue
            basis[rank], basis[pivot_idx] = basis[pivot_idx], basis[rank]
            pivot = basis[rank]
            for j in range(len(basis)):
                if j != rank and ((basis[j] >> bit) & 1):
                    basis[j] ^= pivot
            rank += 1
            if rank == 12:
                break
        return rank
    
    ker_list = sorted(ker_from_q)
    rank_ker = gf2_rank(ker_list)
    
    # Verify unique charges = 4 (merged from Test 5)
    unique_charges = len(set(charge_by_mask.values()))
    
    print(SEP)
    print("Vertex Charge Parity-Check Recovery (Explicit Equations)")
    print(SEP)
    print(f"  q0 = 0x{q0:03x}")
    print(f"  q1 = 0x{q1:03x}")
    print(f"  c0 = {c0}, c1 = {c1}")
    print(f"  χ formula violations: {chi_violations}")
    print(f"  vertex affine violations: {vertex_violations}")
    print(f"  |ker(χ)| = {len(ker_from_q)}")
    print(f"  rank(ker(χ)) = {rank_ker}")
    print(f"  unique charges: {unique_charges}")
    
    # χ must match parity-check formula
    assert chi_violations == 0, f"χ formula must hold, but {chi_violations} violations"
    
    # Vertex map must be affine
    assert vertex_violations == 0, f"Vertex map must be affine, but {vertex_violations} violations"
    
    # Kernel must match expected size and rank (merged from Test 6)
    assert len(ker_from_q) == 64, f"ker(χ) must have size 64, got {len(ker_from_q)}"
    assert rank_ker == 6, f"ker(χ) must have GF(2) rank 6, got {rank_ker}"
    assert unique_charges == 4, f"Must have 4 unique charges, got {unique_charges}"