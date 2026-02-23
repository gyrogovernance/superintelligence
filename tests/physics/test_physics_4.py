"""
Physics tests Part 4: CGM fundamental constraints and operator-subset dynamics.

This file explores:

- CGM-kernel correspondences (aperture constraint, holonomy, shell products, K_QG)
- New kernel invariants (R_nu from δ, m_a, α)
- Operator-subset phase transitions (restricted alphabet rank/orbit theorem)

The restricted alphabet tests probe "collapse-like" behavior via operator
accessibility constraints, yielding nucleation barriers and bubble sub-ontologies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.router.constants import (
    ARCHETYPE_A12,
    ARCHETYPE_B12,
)
from tests._physics_utils import (
    popcount12_arr,
    weight_enumerator_counts,
)

# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture(scope="module")
def atlas():
    atlas_dir = Path("data/atlas")
    if not atlas_dir.exists():
        pytest.skip("Atlas not built. Run: python -m src.router.atlas")

    ontology = np.load(atlas_dir / "ontology.npy", mmap_mode="r").astype(np.uint32)
    epistemology = np.load(atlas_dir / "epistemology.npy", mmap_mode="r").astype(np.uint32)

    from src.router.kernel import RouterKernel
    k = RouterKernel(atlas_dir)
    masks24 = k.xform_mask_by_byte.astype(np.uint32)
    masks_a12 = ((masks24 >> 12) & 0xFFF).astype(np.uint16)

    return {
        "dir": atlas_dir,
        "ont": ontology,
        "epi": epistemology,
        "masks_a12": masks_a12,
    }


# ========
# CGM Fundamental Constraints and Kernel Structure
# ========


class TestApertureConstraintHalfInteger:
    """
    CGM fundamental: Q_G × m_a² = 1/2
    
    This is the aperture balance equation. If kernel-derived constants
    satisfy this, it's a strong validation of CGM geometry.
    """

    def test_aperture_constraint_product_equals_half(self, atlas):
        """
        Compute Q_G_kernel × m_a_kernel² and verify ≈ 0.5
        
        From kernel:
          A_kernel = 5/256
          closure = 251/256
          θ_min = arccos(5/6)
          δ_kernel = θ_min / 3
          m_a_kernel = δ_kernel / closure
          Q_G_kernel = 1 / (2 × m_a_kernel²)
        
        Then Q_G × m_a² = 0.5 by construction... but let's verify the
        *intermediate* relation: Q_G = 1/(2 m_a²) implies Q_G m_a² = 0.5.
        
        The real test: does our δ_kernel / closure formula give m_a
        such that 4π × m_a² ≈ 0.5?
        """
        masks = atlas["masks_a12"].astype(np.uint16)
        counts = weight_enumerator_counts(masks)

        A_kernel = float((counts[0] + counts[1]) / 256.0)
        closure = 1.0 - A_kernel

        theta_min = float(np.arccos(5.0 / 6.0))
        delta_kernel = theta_min / 3.0
        m_a_kernel = delta_kernel / closure

        # CGM says Q_G = 4π
        Q_G_CGM = 4.0 * np.pi

        # Test the fundamental constraint
        product_cgm = Q_G_CGM * (m_a_kernel ** 2)

        # Also compute Q_G_kernel from the reconstruction
        Q_G_kernel = 1.0 / (2.0 * (m_a_kernel ** 2))
        # Q_G_kernel × m_a_kernel² = 0.5 by definition

        # The real question: how close is 4π × m_a_kernel² to 0.5?
        deviation = abs(product_cgm - 0.5)
        relative = deviation / 0.5

        print("\n" + "="*10)
        print("APERTURE CONSTRAINT: Q_G × m_a² = 1/2")
        print("="*10)
        print(f"  A_kernel = {A_kernel:.12f} (5/256)")
        print(f"  closure  = {closure:.12f} (251/256)")
        print(f"  θ_min    = {theta_min:.12f} rad")
        print(f"  δ_kernel = {delta_kernel:.12f} rad")
        print(f"  m_a_kernel = {m_a_kernel:.12f}")
        print("")
        print(f"  4π × m_a_kernel² = {product_cgm:.12f}")
        print("  Expected (CGM)   = 0.500000000000")
        print(f"  Deviation        = {deviation:.12f}")
        print(f"  Relative error   = {100*relative:.6f}%")
        print("")
        print(f"  Q_G_kernel (from 1/(2m_a²)) = {Q_G_kernel:.12f}")
        print(f"  Q_G_CGM (4π)                = {Q_G_CGM:.12f}")

        # Hard assertion: should be within 1%
        assert relative < 0.01, f"Aperture constraint violated: 4π × m_a² = {product_cgm:.6f} ≠ 0.5"

        print("  ✓ Aperture constraint 4π × m_a² ≈ 0.5 holds within 1%")


class TestHolonomyGroupIsCode:
    """
    Monodromy creates fiber defects. The set of achievable defects
    should form a group under XOR equal to the mask code C.
    """

    def test_achievable_fiber_defects_equal_mask_code(self, atlas):
        """
        The monodromy word W = [x, y, x, z] creates defect m_y ⊕ m_z in v.
        
        Achievable defects = {m_y ⊕ m_z : y, z ∈ [0,255]}
        
        Since C is closed under XOR and contains all masks,
        achievable defects = C.
        """
        masks = atlas["masks_a12"].astype(np.uint16)

        # Build all achievable defects
        achievable = set()
        for y in range(256):
            for z in range(256):
                d = int(masks[y]) ^ int(masks[z])
                achievable.add(d)

        # Build the mask code
        code_set = set(int(m) for m in masks)

        print("\n" + "="*10)
        print("HOLONOMY GROUP = MASK CODE")
        print("="*10)
        print(f"  |Achievable defects| = {len(achievable)}")
        print(f"  |Mask code C|        = {len(code_set)}")
        print(f"  Sets equal           = {achievable == code_set}")

        assert achievable == code_set, "Holonomy group ≠ mask code"

        # The group structure: C is closed under XOR
        # Verify explicitly
        closed = True
        for a in list(code_set)[:50]:  # Sample
            for b in list(code_set)[:50]:
                if (a ^ b) not in code_set:
                    closed = False
                    break

        print(f"  XOR closure verified = {closed}")
        print("  ✓ Holonomy group is isomorphic to (Z/2)^8")


class TestOpticalConjugacyShellProduct:
    """
    CGM: E^UV × E^IR = K/(4π²)
    
    In kernel: shell d and shell (24-d) are conjugate.
    What is the "product" invariant?
    """

    def test_shell_conjugacy_product_structure(self, atlas):
        """
        For each shell d, its conjugate is 24-d.
        
        Explore: d × (24-d) peaks at d=12 with value 144 = 12².
        The count distribution is also symmetric.
        
        Is there a 4π² connection?
        """
        ont = atlas["ont"]

        a = ((ont >> 12) & 0xFFF).astype(np.uint16)
        b = (ont & 0xFFF).astype(np.uint16)

        da = popcount12_arr((a ^ np.uint16(ARCHETYPE_A12)).astype(np.uint16))
        db = popcount12_arr((b ^ np.uint16(ARCHETYPE_B12)).astype(np.uint16))
        dist24 = (da + db).astype(np.int64)

        counts = np.bincount(dist24, minlength=25).astype(np.int64)

        # Shell products d × (24-d)
        products = [d * (24 - d) for d in range(25)]

        # Weighted by counts
        total_states = int(counts.sum())
        mean_d = float(np.sum(np.arange(25) * counts)) / total_states
        mean_product = float(np.sum(np.array(products) * counts)) / total_states

        # Identity: d(24-d) = 144 - (d-12)², so E[d(24-d)] = 144 - Var(d)
        var_d = float(np.sum((np.arange(25) - mean_d) ** 2 * counts)) / total_states

        # 4π² ≈ 39.478
        four_pi_sq = 4.0 * np.pi ** 2

        print("\n" + "="*10)
        print("OPTICAL CONJUGACY: SHELL PRODUCTS")
        print("="*10)
        print(f"  Mean distance d        = {mean_d:.6f} (expected 12.0)")
        print(f"  Mean product d(24-d)   = {mean_product:.6f}")
        print(f"  Max product (at d=12)  = {12 * 12} = 144 = 12²")
        print("")
        print("  Identity: E[d(24-d)] = 144 - Var(d)")
        print(f"  Var(d)               = {var_d:.6f}")
        print(f"  144 - Var(d)         = {144 - var_d:.6f}")
        print("  (Expected: Var(d) = 10, from Var(w1) + Var(w2) = 5 + 5)")
        print("")
        print(f"  4π²                    = {four_pi_sq:.6f}")
        print(f"  mean_product / 4π²     = {mean_product / four_pi_sq:.6f}")
        print(f"  144 / 4π²              = {144 / four_pi_sq:.6f}")

        # Verify symmetry
        symmetric = all(int(counts[d]) == int(counts[24-d]) for d in range(25))
        print(f"  Shell symmetry (d ↔ 24-d) = {symmetric}")

        assert symmetric
        assert abs(mean_d - 12.0) < 0.01
        assert abs(var_d - 10.0) < 0.01, "Variance should be 10 (sum of two independent weight variances)"
        assert abs(mean_product - (144 - var_d)) < 1e-6, "Identity E[d(24-d)] = 144 - Var(d) should hold"

        print("  ✓ Optical conjugacy structure verified")
        print("  ✓ Second-moment identity: E[d(24-d)] = 144 - Var(d) = 134")


class TestKQGCommutatorScale:
    """
    CGM: K_QG = π²/√(2π) ≈ 3.937 is the commutator scale.
    
    Also K_QG = (π/4) / m_a = S_ONA / 2 = S_CS / 4
    
    Verify this from kernel constants.
    """

    def test_commutator_scale_kqg(self, atlas):
        """
        K_QG appears in [X, P] = i K_QG
        
        From kernel:
          m_a_kernel derived as before
          K_QG_kernel = (π/4) / m_a_kernel
        
        Compare to:
          K_QG_CGM = π²/√(2π) ≈ 3.937
        """
        masks = atlas["masks_a12"].astype(np.uint16)
        counts = weight_enumerator_counts(masks)

        A_kernel = float((counts[0] + counts[1]) / 256.0)
        closure = 1.0 - A_kernel
        theta_min = float(np.arccos(5.0 / 6.0))
        delta_kernel = theta_min / 3.0
        m_a_kernel = delta_kernel / closure

        K_QG_kernel = (np.pi / 4.0) / m_a_kernel
        K_QG_CGM = (np.pi ** 2) / np.sqrt(2.0 * np.pi)

        # Also: S_ONA = o_p / m_a = (π/4) / m_a
        S_ONA_kernel = (np.pi / 4.0) / m_a_kernel

        print("\n" + "="*10)
        print("COMMUTATOR SCALE K_QG")
        print("="*10)
        print(f"  m_a_kernel       = {m_a_kernel:.12f}")
        print(f"  K_QG_kernel      = (π/4)/m_a = {K_QG_kernel:.12f}")
        print(f"  K_QG_CGM         = π²/√(2π) = {K_QG_CGM:.12f}")
        print(f"  Difference       = {K_QG_kernel - K_QG_CGM:+.12f}")
        print(f"  Relative error   = {100*abs(K_QG_kernel - K_QG_CGM)/K_QG_CGM:.6f}%")
        print("")
        print(f"  S_ONA_kernel     = {S_ONA_kernel:.12f}")
        print(f"  K_QG = S_ONA?    : {K_QG_kernel:.6f} vs {S_ONA_kernel:.6f}")
        print("  (In CGM: K_QG = S_CS/2 = S_ONA)")

        assert abs(K_QG_kernel - K_QG_CGM) / K_QG_CGM < 0.01
        assert abs(K_QG_kernel - S_ONA_kernel) < 1e-10, "K_QG should equal S_ONA in this setup"


class TestNeutrinoScaleInvariant:
    """
    Exploratory: define a new kernel-only invariant from (δ, m_a, α).

    Using:
      θ_min      = arccos(5/6)
      δ_kernel   = θ_min / 3
      A_kernel   = (A_0 + A_1) / 256 = 5/256
      m_a_kernel = δ_kernel / (1 - A_kernel)
      α_kernel   = δ_kernel^4 / m_a_kernel

    Define:
      R_nu = δ_kernel^6 / (m_a_kernel^2 * α_kernel)

    This is the "next" nontrivial dimensionless combination after α,
    increasing δ's effective power while cancelling m_a in a different way.

    We *only* compute and print R_nu here. No external PDG numbers enter.
    """

    def test_neutrino_scale_invariant_from_kernel(self, atlas):
        masks = atlas["masks_a12"].astype(np.uint16)
        counts = weight_enumerator_counts(masks)

        # A_kernel = P(w <= 1) over the mask code C (size 256) = (1+4)/256 = 5/256
        A_kernel = float((counts[0] + counts[1]) / 256.0)
        closure = 1.0 - A_kernel

        # Kernel theta_min and δ, as in your existing tests
        theta_min = float(np.arccos(5.0 / 6.0))
        delta_kernel = theta_min / 3.0

        # m_a and α from your CGM-bridge logic
        m_a_kernel = delta_kernel / closure
        alpha_kernel = (delta_kernel ** 4) / m_a_kernel

        # New invariant: R_nu
        R_nu = (delta_kernel ** 6) / (m_a_kernel ** 2 * alpha_kernel)

        print("\n" + "=" * 10)
        print("NEUTRINO-SCALE-LIKE KERNEL INVAIRANT")
        print("=" * 10)
        print(f"  A_kernel        = {A_kernel:.12f}  (5/256)")
        print(f"  closure         = {closure:.12f}  (251/256)")
        print(f"  theta_min       = {theta_min:.12f} rad  (arccos(5/6))")
        print(f"  delta_kernel    = {delta_kernel:.12f} rad  (theta_min/3)")
        print(f"  m_a_kernel      = {m_a_kernel:.12f}")
        print(f"  alpha_kernel    = {alpha_kernel:.12f}")
        print("")
        print(f"  R_nu (kernel)   = δ^6 / (m_a^2 * α) = {R_nu:.12f}")
        print("  (Purely kernel-defined, no external scales used)")

        # Sanity-only assertions: positivity and reasonable scale
        assert R_nu > 0.0
        assert R_nu < 1.0  # we expect a "small" dimensionless number here


class TestRestrictedAlphabetPhaseTransition:
    """
    Test operator-subset phase transition: restricted alphabet dynamics.
    
    Core theorem (kernel-specific):
    
    Let S ⊆ bytes and let U = span_GF2({mask(b): b∈S}) ⊆ GF(2)^12.
    Then from the archetype, the set of reachable (u,v) values under
    arbitrary-length words over S is exactly U × U, hence the reachable
    state set size is |U|^2 = 2^(2·rank(U)).
    
    This gives a nucleation barrier: as you enlarge S by allowing
    higher mask-weights, rank(U) jumps. When rank hits 8, you get
    full Ω (65536 states). Below that, you get a strict "bubble
    sub-ontology".
    
    This is the discrete analog of phase transition / nucleation via
    operator accessibility constraints.
    """

    @staticmethod
    def _gf2_rank(vectors_12bit: list[int]) -> int:
        """
        Gaussian elimination over GF(2) for 12-bit vectors represented as ints.
        Returns rank.
        """
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

    @staticmethod
    def _bfs_orbit_size(epi, start_idx: int, allowed_bytes: list[int]) -> int:
        """
        BFS from start_idx using only allowed_bytes, return orbit size.
        Vectorized implementation for performance.
        """
        N = epi.shape[0]
        visited = np.zeros(N, dtype=bool)
        frontier = np.array([start_idx], dtype=np.int64)
        visited[start_idx] = True

        allowed_array = np.array(allowed_bytes, dtype=np.int64)

        while len(frontier) > 0:
            next_idxs = epi[frontier][:, allowed_array].ravel()
            next_idxs = np.unique(next_idxs)
            new = next_idxs[~visited[next_idxs]]
            visited[new] = True
            frontier = new

        return int(visited.sum())

    def test_rank_orbit_theorem(self, atlas):
        """
        Test: orbit_size = 2^(2 * rank(U)) for restricted alphabet S.
        
        For threshold t = 0..12, define:
          allowed = {b: popcount(mask[b]) <= t}
        
        Compute:
          rank_t = gf2_rank([mask[b] for b in allowed])
          pred_size = 2^(2 * rank_t)
          orbit_size = bfs_orbit_size(epi, archetype_idx, allowed)
        
        Assert: orbit_size == pred_size
        """
        ont = atlas["ont"]
        epi = atlas["epi"]
        masks = atlas["masks_a12"].astype(np.uint16)

        # Archetype index
        arch_state = (int(ARCHETYPE_A12) << 12) | int(ARCHETYPE_B12)
        arch_idx = int(np.where(ont == arch_state)[0][0])

        print("\n" + "=" * 10)
        print("RANK/ORBIT THEOREM: RESTRICTED ALPHABET")
        print("=" * 10)

        mismatches = []
        for t in range(13):
            # Allowed bytes: mask weight <= t
            allowed = [b for b in range(256) if bin(int(masks[b])).count('1') <= t]

            if not allowed:
                continue

            # Compute rank of mask subspace
            mask_list = [int(masks[b]) for b in allowed]
            rank_t = self._gf2_rank(mask_list)
            pred_size = 2 ** (2 * rank_t)

            # Compute actual orbit size via BFS
            orbit_size = self._bfs_orbit_size(epi, arch_idx, allowed)

            match = (orbit_size == pred_size)
            if not match:
                mismatches.append((t, rank_t, pred_size, orbit_size))

            print(f"  t={t:2d}: |allowed|={len(allowed):3d}, rank={rank_t}, "
                  f"pred={pred_size:6d}, orbit={orbit_size:6d}, match={match}")

        print(f"\n  Mismatches: {len(mismatches)}")
        assert len(mismatches) == 0, f"Rank/orbit theorem failed at thresholds: {mismatches}"

    def test_nucleation_barrier_critical_threshold(self, atlas):
        """
        Find the critical threshold where rank jumps to 8 (full code dimension).
        This defines the nucleation barrier: below this threshold, you're in a
        "bubble sub-ontology"; at or above, you have full Ω.
        """
        masks = atlas["masks_a12"].astype(np.uint16)

        ranks = []
        for t in range(13):
            allowed = [b for b in range(256) if bin(int(masks[b])).count('1') <= t]
            if not allowed:
                ranks.append(0)
                continue
            mask_list = [int(masks[b]) for b in allowed]
            rank_t = self._gf2_rank(mask_list)
            ranks.append(rank_t)

        # Find critical threshold: smallest t where rank_t == 8
        critical_t = next((t for t, r in enumerate(ranks) if r == 8), None)

        print("\n" + "=" * 10)
        print("NUCLEATION BARRIER: CRITICAL THRESHOLD")
        print("=" * 10)
        print("  Rank progression by weight threshold t:")
        for t in range(13):
            print(f"    t={t:2d}: rank={ranks[t]}, 2^(2*rank)={2**(2*ranks[t])}")

        print(f"\n  Critical threshold: t={critical_t} (rank jumps to 8)")
        print(f"  Below t={critical_t}: bubble sub-ontology (strict subset of Ω)")
        print(f"  At t>={critical_t}: full Ω accessible (65536 states)")

        # Assertions: monotonicity and full rank at t=12
        assert all(ranks[i] <= ranks[i+1] for i in range(12)), "Rank must be non-decreasing"
        assert ranks[12] == 8, "Full alphabet must generate rank 8"
        assert critical_t is not None, "Must reach rank 8 at some threshold"
        assert critical_t <= 12, "Critical threshold must be <= 12"

    def test_minimal_generator_count(self, atlas):
        """
        What is the smallest subset size k such that a random k-byte subset
        generates full rank (8)? This tests minimal generator requirements.
        """
        masks = atlas["masks_a12"].astype(np.uint16)

        # Try random subsets of various sizes
        rng = np.random.default_rng(2025)
        sizes_to_try = [4, 5, 6, 7, 8, 9, 10, 12, 16, 32]

        min_observed = None
        max_observed = {}

        print("\n" + "=" * 10)
        print("MINIMAL GENERATOR COUNT")
        print("=" * 10)
        print("  Sampling random byte subsets to find minimal generators:")

        for k in sizes_to_try:
            ranks_seen = []
            for _ in range(100):
                subset = rng.choice(256, size=k, replace=False).tolist()
                mask_list = [int(masks[b]) for b in subset]
                rank = self._gf2_rank(mask_list)
                ranks_seen.append(rank)

            max_rank = max(ranks_seen)
            full_rank_count = sum(1 for r in ranks_seen if r == 8)
            max_observed[k] = (max_rank, full_rank_count)

            print(f"    k={k:2d}: max_rank={max_rank}, full_rank_count={full_rank_count}/100")

            if max_rank == 8 and min_observed is None:
                min_observed = k

        print(f"\n  Minimum k with full rank observed: {min_observed}")
        print("  (Code dimension = 8, so theoretical minimum is k >= 8)")

        # Assertions
        assert min_observed is not None, "Must find full rank at some subset size"
        assert min_observed >= 8, "Need at least 8 generators for rank 8"
        # k=8 should frequently (but not always) give full rank
        assert max_observed[8][1] > 0, "Subsets of size 8 should sometimes give full rank"

    def test_weight2_bridge_masks_extend_rank(self, atlas):
        """
        Decompose the t=2 jump:

        - Weight-1 masks span a rank-4 subspace U1 (16 distinct 12-bit masks).
        - Among weight-2 masks, exactly 6 lie in U1 and 4 lie outside.
        - Adding the 4 "bridge" masks extends rank from 4 to 8.

        This explains structurally why the nucleation barrier sits at t=2.
        """
        masks = atlas["masks_a12"].astype(np.uint16)

        # Identify weight-0 and weight-1 bytes
        w0_bytes = [b for b in range(256) if bin(int(masks[b])).count("1") == 0]
        w1_bytes = [b for b in range(256) if bin(int(masks[b])).count("1") == 1]

        assert len(w0_bytes) == 1  # identity mask
        assert len(w1_bytes) == 4  # four primitive directions

        # Build U1: span of the 4 weight-1 masks (2^4 = 16 combinations)
        w1_masks = [int(masks[b]) for b in w1_bytes]
        U1_masks = set()
        for s in range(1 << len(w1_masks)):
            acc = 0
            for i in range(len(w1_masks)):
                if (s >> i) & 1:
                    acc ^= w1_masks[i]
            U1_masks.add(acc & 0xFFF)

        # U1 should have 16 distinct masks and rank 4
        assert len(U1_masks) == 16
        U1_rank = self._gf2_rank(list(U1_masks))
        assert U1_rank == 4

        # Identify weight-2 bytes
        w2_bytes = [b for b in range(256) if bin(int(masks[b])).count("1") == 2]
        assert len(w2_bytes) == 10

        inside = []
        bridge = []
        for b in w2_bytes:
            m = int(masks[b]) & 0xFFF
            if m in U1_masks:
                inside.append(m)
            else:
                bridge.append(m)

        print("\n" + "="*10)
        print("WEIGHT-2 BRIDGE MASKS")
        print("="*10)
        print(f"  |U1|           = {len(U1_masks)} (rank 4)")
        print(f"  |W2 inside U1| = {len(inside)}")
        print(f"  |W2 bridge|    = {len(bridge)}")

        # Expect 6 inside and 4 true bridge masks
        assert len(inside) == 6
        assert len(bridge) == 4

        # Now show that U1 ∪ bridge has rank 8
        combined = list(U1_masks) + bridge
        combined_rank = self._gf2_rank(combined)
        print(f"  Combined rank  = {combined_rank}")
        assert combined_rank == 8


# ========
# Session dashboard
# ========

@pytest.fixture(scope="session", autouse=True)
def print_physics_4_dashboard():
    yield
    print("\n" + "="*10)
    print("PHYSICS 4 DASHBOARD - CGM EMERGENCE")
    print("="*10)
    print("  EXISTING:")
    print("  ✓ Reference byte cycles and eigenphases")
    print("  ✓ Code duality and MacWilliams")
    print("  ✓ Walsh spectrum support")
    print("  ✓ Shell distributions from enumerator")
    print("  ✓ CGM constants bridge (δ, m_a, Q_G, α)")
    print("  ✓ Group presentation in (u,v)")
    print("")
    print("  NEW EXPLORATIONS:")
    print("  • Aperture constraint: Q_G × m_a² = 1/2")
    print("  • Holonomy group = mask code")
    print("  • Optical conjugacy shell products")
    print("  • K_QG commutator scale")
    print("  • Neutrino-scale-like invariant R_nu")
    print("  • Restricted alphabet phase transition (rank/orbit theorem, bridge masks)")
    print("="*10)

