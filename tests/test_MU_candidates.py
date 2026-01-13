"""
Kernel-native MU capacity candidate tests.

This module tests kernel-derived capacity multipliers for the Moments Economy,
derived from kernel anatomy rather than hardware throughput.

The tests explore candidates for K_kernel such that:
    F_Moment = f_Cs × K_kernel

where K_kernel must come from atlas-real kernel structure, not CPU throughput.

Candidates tested:
1. K_open = A_kernel × |Ω| = |H| × |C_{<=1}| = 1280
   - "Open Horizon Modes" - kernel-native structural openness
2. K_Ω = |Ω| = 65536
   - "Full coherent modes" - every coherent state per atomic tick
3. K_invA = |Ω| / A_kernel = 2^24 / 5
   - "Inverse aperture capacity" - maximum addressable state-action microcells
4. K_QG = Q_G = 4π ≈ 12.566
   - "CGM Horizon Constant" - fundamental CGM invariant (total solid angle)

Run:
  python -m pytest -v -s tests/test_MU_candidates.py
"""

from __future__ import annotations
import math

from src.router.constants import mask12_for_byte, LAYER_MASK_12

ATOMIC_HZ_CS133 = 9_192_631_770
SECONDS_PER_YEAR = 365 * 24 * 60 * 60
ROUTER_ONTOLOGY_SIZE = 65_536
HORIZON_SIZE = 256

# CGM Constants
Q_G = 4 * math.pi  # Horizon constant from CGM

# Moments Economy Spec Requirements
WORLD_POP = 8_100_000_000
UHI_PER_YEAR_MU = 87_600
REQUIRED_CAPACITY_PER_YEAR = WORLD_POP * UHI_PER_YEAR_MU

# Moments Economy Spec Requirements
WORLD_POP = 8_100_000_000
UHI_PER_YEAR_MU = 87_600
REQUIRED_CAPACITY_PER_YEAR = WORLD_POP * UHI_PER_YEAR_MU


def test_cgm_kernel_open_horizon_modes_capacity_anchor():
    """
    Derive a kernel-native capacity multiplier K_open from kernel anatomy:

      - Horizon size |H| = 256 (fixed points of reference involution byte 0xAA)
      - Minimal openness sector size |C_{<=1}| = count of masks of weight 0 or 1
        (weight-0 identity + weight-1 primitive anatomical directions)

    Define:
      K_open := |H| * |C_{<=1}| = 256 * 5 = 1280

    Then propose a purely physics+kernel capacity rate:
      F_Moment = f_Cs * K_open   (units: "capacity quanta per second")

    This is *common* and hardware-independent: depends only on SI second + kernel code geometry.
    """
    weights = []
    for b in range(256):
        m = mask12_for_byte(b)
        w = bin(m & LAYER_MASK_12).count("1")
        weights.append(w)

    count_w0 = sum(1 for w in weights if w == 0)
    count_w1 = sum(1 for w in weights if w == 1)
    C_le1 = count_w0 + count_w1

    H = HORIZON_SIZE

    K_open = H * C_le1

    F_per_sec = ATOMIC_HZ_CS133 * K_open
    F_per_year = F_per_sec * SECONDS_PER_YEAR

    print("\n=== KERNEL-DERIVED MOMENT CAPACITY ANCHOR (C_{{<=1}} × H) ===")
    print(f"count(weight=0 masks): {count_w0}")
    print(f"count(weight=1 masks): {count_w1}")
    print(f"|C_{{<=1}}|             : {C_le1}  (expected 5)")
    print(f"|H| (horizon)         : {H}     (expected 256)")
    print(f"K_open = |H|*|C<=1|    : {K_open} (expected 1280)")
    print(f"F_Moment (/sec)        : {F_per_sec:,}")
    print(f"F_Moment (/year)       : {F_per_year:,}")

    assert count_w0 == 1
    assert count_w1 == 4
    assert C_le1 == 5
    assert K_open == 1280
    assert F_per_sec > 10**12


def test_cgm_kernel_open_modes_abundance_for_uhi():
    """
    Show that the kernel-derived, hardware-independent capacity anchor
    still funds global UHI for 1000 years with massive margin.
    """
    weights = [bin(mask12_for_byte(b) & LAYER_MASK_12).count("1") for b in range(256)]
    C_le1 = sum(1 for w in weights if w <= 1)
    H = HORIZON_SIZE
    K_open = H * C_le1

    capacity_per_year = ATOMIC_HZ_CS133 * K_open * SECONDS_PER_YEAR
    demand_per_year = WORLD_POP * UHI_PER_YEAR_MU

    utilization = demand_per_year / capacity_per_year

    print("\n=== UHI UTILIZATION UNDER KERNEL-DERIVED ANCHOR ===")
    print(f"capacity/year: {capacity_per_year:,}")
    print(f"demand/year  : {demand_per_year:,}")
    print(f"utilization  : {utilization:.3e} ({utilization*100:.8f}%)")

    assert utilization < 1e-5


def test_candidate_full_coherent_modes():
    """
    Candidate A: "full coherent modes" K_Ω = |Ω| = 65536
    
    F = f_Cs × |Ω|
    
    Interpretation: every coherent state is a mode per atomic tick.
    """
    K_omega = ROUTER_ONTOLOGY_SIZE

    F_per_sec = ATOMIC_HZ_CS133 * K_omega
    F_per_year = F_per_sec * SECONDS_PER_YEAR

    print("\n=== CANDIDATE A: FULL COHERENT MODES (K_Ω = |Ω|) ===")
    print(f"K_omega = |Ω|        : {K_omega:,}")
    print(f"F_Moment (/sec)       : {F_per_sec:,}")
    print(f"F_Moment (/year)      : {F_per_year:,}")

    assert K_omega == 65536
    assert F_per_sec > 10**14


def test_candidate_inverse_aperture_capacity():
    """
    Candidate B: "inverse aperture capacity" K_invA = |Ω| / A_kernel = 2^24 / 5
    
    F = f_Cs × (|Ω| / A_kernel)
    
    Interpretation: maximum addressable state-action microcells per condensate.
    This is less conservative and easier to criticize.
    """
    A_kernel = 5 / 256
    K_invA = int(ROUTER_ONTOLOGY_SIZE / A_kernel)

    F_per_sec = ATOMIC_HZ_CS133 * K_invA
    F_per_year = F_per_sec * SECONDS_PER_YEAR

    print("\n=== CANDIDATE B: INVERSE APERTURE CAPACITY ===")
    print(f"A_kernel             : {A_kernel}")
    print(f"K_invA = |Ω|/A_kernel : {K_invA:,}")
    print(f"F_Moment (/sec)       : {F_per_sec:,}")
    print(f"F_Moment (/year)      : {F_per_year:,}")

    assert K_invA == int(65536 / (5 / 256))
    assert F_per_sec > 10**16


def test_compare_all_candidates():
    """
    Compare all three kernel-native candidates side by side.
    """
    weights = [bin(mask12_for_byte(b) & LAYER_MASK_12).count("1") for b in range(256)]
    C_le1 = sum(1 for w in weights if w <= 1)
    A_kernel = 5 / 256

    K_open = HORIZON_SIZE * C_le1
    K_omega = ROUTER_ONTOLOGY_SIZE
    K_invA = int(ROUTER_ONTOLOGY_SIZE / A_kernel)

    F_open_per_sec = ATOMIC_HZ_CS133 * K_open
    F_omega_per_sec = ATOMIC_HZ_CS133 * K_omega
    F_invA_per_sec = ATOMIC_HZ_CS133 * K_invA

    print("\n=== COMPARISON OF KERNEL-NATIVE CAPACITY CANDIDATES ===")
    print(f"K_open  = |H| × |C_{{<=1}}| = {K_open:,}")
    print(f"K_omega = |Ω|              = {K_omega:,}")
    print(f"K_invA  = |Ω| / A_kernel   = {K_invA:,}")
    print()
    print(f"F_open  (/sec):  {F_open_per_sec:,}")
    print(f"F_omega (/sec):  {F_omega_per_sec:,}")
    print(f"F_invA  (/sec):  {F_invA_per_sec:,}")
    print()
    print(f"Ratio K_omega / K_open  : {K_omega / K_open:.2f}")
    print(f"Ratio K_invA / K_open   : {K_invA / K_open:.2f}")

    assert K_open == 1280
    assert K_omega == 65536
    assert K_invA > K_omega > K_open


def test_kernel_anatomy_verification():
    """
    Verify the kernel anatomy facts that underpin K_open:
    - Horizon cardinality |H| = 256
    - Minimal openness sector |C_{<=1}| = 5
    - The identity A_kernel × |Ω| = |H| × |C_{<=1}| = 1280
    """
    weights = []
    for b in range(256):
        m = mask12_for_byte(b)
        w = bin(m & LAYER_MASK_12).count("1")
        weights.append(w)

    count_w0 = sum(1 for w in weights if w == 0)
    count_w1 = sum(1 for w in weights if w == 1)
    C_le1 = count_w0 + count_w1

    A_kernel = C_le1 / 256
    H = HORIZON_SIZE

    identity_lhs = A_kernel * ROUTER_ONTOLOGY_SIZE
    identity_rhs = H * C_le1

    print("\n=== KERNEL ANATOMY VERIFICATION ===")
    print(f"A_kernel = P(w <= 1) = {C_le1}/256 = {A_kernel}")
    print(f"A_kernel × |Ω| = {A_kernel} × {ROUTER_ONTOLOGY_SIZE} = {identity_lhs}")
    print(f"|H| × |C_{{<=1}}| = {H} × {C_le1} = {identity_rhs}")
    print(f"Identity holds: {identity_lhs == identity_rhs}")

    assert count_w0 == 1
    assert count_w1 == 4
    assert C_le1 == 5
    assert abs(A_kernel - (5 / 256)) < 1e-10
    assert identity_lhs == identity_rhs == 1280


def test_all_candidates_sufficiency_for_global_uhi():
    """
    Compare all three candidates against the Moments Economy spec requirement.
    
    IMPORTANT: Capacity exists continuously per second. The formula is:
        F_Moment = f_Cs × K_kernel  (capacity per second)
    
    The per-year numbers shown are ONLY for comparison with annual demand.
    The actual capacity is available every second at the per-second rate.
    
    Required: 8.1 billion people × 87,600 MU/year = 7.10 × 10¹⁴ MU/year
    """
    weights = [bin(mask12_for_byte(b) & LAYER_MASK_12).count("1") for b in range(256)]
    C_le1 = sum(1 for w in weights if w <= 1)
    A_kernel = 5 / 256

    K_open = HORIZON_SIZE * C_le1
    K_omega = ROUTER_ONTOLOGY_SIZE
    K_invA = int(ROUTER_ONTOLOGY_SIZE / A_kernel)

    # Capacity per second (the actual continuous capacity)
    F_open_per_sec = ATOMIC_HZ_CS133 * K_open
    F_omega_per_sec = ATOMIC_HZ_CS133 * K_omega
    F_invA_per_sec = ATOMIC_HZ_CS133 * K_invA

    # Annual capacity (for comparison with annual demand only)
    F_open_per_year = F_open_per_sec * SECONDS_PER_YEAR
    F_omega_per_year = F_omega_per_sec * SECONDS_PER_YEAR
    F_invA_per_year = F_invA_per_sec * SECONDS_PER_YEAR

    required = REQUIRED_CAPACITY_PER_YEAR

    ratio_open = F_open_per_year / required
    ratio_omega = F_omega_per_year / required
    ratio_invA = F_invA_per_year / required

    print("\n" + "=" * 70)
    print("SUFFICIENCY ANALYSIS: ALL CANDIDATES vs. GLOBAL UHI REQUIREMENT")
    print("=" * 70)
    print("\nNOTE: Capacity exists continuously per second.")
    print("      Per-year numbers are shown only for comparison with annual demand.")
    print()
    print(f"Required capacity (8.1B people × 87,600 MU/year):")
    print(f"  {required:,} MU/year")
    print(f"  ({required:.2e} MU/year)")
    print()
    print("Candidate capacities (per second - the actual continuous rate):")
    print(f"  1. K_open  (1280):     {F_open_per_sec:,} MU/sec")
    print(f"                        ({F_open_per_sec:.2e} MU/sec)")
    print()
    print(f"  2. K_omega (65536):    {F_omega_per_sec:,} MU/sec")
    print(f"                        ({F_omega_per_sec:.2e} MU/sec)")
    print()
    print(f"  3. K_invA  (3,355,443): {F_invA_per_sec:,} MU/sec")
    print(f"                        ({F_invA_per_sec:.2e} MU/sec)")
    print()
    print("Annual capacity (for comparison only):")
    print(f"  1. K_open:  {F_open_per_year:,} MU/year ({F_open_per_year:.2e} MU/year)")
    print(f"             {ratio_open:,.0f}× required capacity")
    print()
    print(f"  2. K_omega: {F_omega_per_year:,} MU/year ({F_omega_per_year:.2e} MU/year)")
    print(f"             {ratio_omega:,.0f}× required capacity")
    print()
    print(f"  3. K_invA:  {F_invA_per_year:,} MU/year ({F_invA_per_year:.2e} MU/year)")
    print(f"             {ratio_invA:,.0f}× required capacity")
    print()
    print("=" * 70)
    print("CONCLUSION: ALL THREE CANDIDATES ARE SUFFICIENT")
    print("=" * 70)
    print(f"Even the smallest (K_open) provides {ratio_open:,.0f}× more than needed.")
    print(f"This means K_open could fund global UHI for {ratio_open:,.0f} years.")
    print()

    assert ratio_open > 1, "K_open must exceed requirement"
    assert ratio_omega > 1, "K_omega must exceed requirement"
    assert ratio_invA > 1, "K_invA must exceed requirement"
    
    assert ratio_open > 100_000, "K_open should provide massive margin"
    assert ratio_omega > 1_000_000, "K_omega should provide massive margin"
    assert ratio_invA > 10_000_000, "K_invA should provide massive margin"


def test_capacity_physical_meaning_and_cgm_relationship():
    """
    Address the assistant's questions about capacity:
    
    1. What does "capacity" mean in the Moments Economy?
    2. Is there a CGM derivation that gives a specific number?
    3. Is there a quantity in the physics tests that emerges as "the natural scale"?
    
    This test shows:
    - Capacity = issuance rate of novel structural degrees of freedom
    - The discrete quantum (5) emerges from kernel anatomy
    - The natural scale (1280) emerges from the kernel identity
    - K_open is the most physically justified candidate
    """
    weights = [bin(mask12_for_byte(b) & LAYER_MASK_12).count("1") for b in range(256)]
    C_le1 = sum(1 for w in weights if w <= 1)
    A_kernel = C_le1 / 256
    
    H = HORIZON_SIZE
    
    # The discrete quantum (from CGM constraint Q_G × m_a² = 1/2)
    discrete_quantum = H * A_kernel  # 256 × (5/256) = 5
    
    # The kernel identity
    K_open = H * C_le1  # 256 × 5 = 1280
    identity_lhs = A_kernel * ROUTER_ONTOLOGY_SIZE  # (5/256) × 65536 = 1280
    identity_rhs = H * C_le1  # 256 × 5 = 1280
    
    # Relationship between candidates
    K_omega = ROUTER_ONTOLOGY_SIZE
    K_invA = int(ROUTER_ONTOLOGY_SIZE / A_kernel)

    ratio_omega_to_open = K_omega / K_open  # Should be 1/A_kernel = 256/5 = 51.2
    ratio_invA_to_open = K_invA / K_open  # Should be approximately (1/A_kernel)² = (256/5)²
    
    # Note: K_invA uses integer truncation, so the ratio is approximate
    expected_ratio_invA = (1 / A_kernel) ** 2

    print("\n" + "=" * 70)
    print("CAPACITY PHYSICAL MEANING AND CGM RELATIONSHIP")
    print("=" * 70)
    print()
    print("1. WHAT IS CAPACITY?")
    print("   Capacity = issuance rate of novel structural degrees of freedom")
    print("   Each MU represents 1 minute of this issuance capacity at base rate")        
    print("   Capacity must be kernel-native and hardware-independent")
    print()
    print("2. THE DISCRETE QUANTUM (from CGM constraint Q_G × m_a² = 1/2):")
    print(f"   |H| × A_kernel = {H} × {A_kernel} = {discrete_quantum}")
    print(f"   This is the discrete realization of the CGM fundamental constraint")       
    print()
    print("3. THE KERNEL IDENTITY (exact, from kernel anatomy):")
    print(f"   A_kernel × |Ω| = {A_kernel} × {ROUTER_ONTOLOGY_SIZE} = {identity_lhs}")    
    print(f"   |H| × |C_{{<=1}}| = {H} × {C_le1} = {identity_rhs}")
    print(f"   Both equal: {K_open}")
    print()
    print("4. THE NATURAL SCALE:")
    print(f"   K_open = {K_open} emerges as the natural scale because:")
    print(f"   - It's an exact integer from kernel anatomy")
    print(f"   - It directly relates to the discrete quantum ({discrete_quantum})")       
    print(f"   - It satisfies the kernel identity")
    print(f"   - It represents horizon × minimal openness = issuance capacity")
    print()
    print("5. CANDIDATE RELATIONSHIPS (powers of 1/A_kernel):")
    print(f"   K_open  = {K_open:,} (base)")
    print(f"   K_omega = {K_omega:,} = K_open × (1/A_kernel) = K_open × {ratio_omega_to_open:.2f}")
    print(f"   K_invA  = {K_invA:,} ≈ K_open × (1/A_kernel)² = K_open × {ratio_invA_to_open:.2f}")
    print(f"            (expected: {expected_ratio_invA:.2f}, difference due to integer truncation)")
    print()
    print("6. RECOMMENDATION:")
    print(f"   K_open = {K_open} is the most physically justified because:")
    print(f"   - Kernel-native (derived from anatomy, not hardware)")
    print(f"   - Directly related to discrete quantum ({discrete_quantum})")
    print(f"   - Represents issuance capacity (novelty per second)")
    print(f"   - Satisfies exact kernel identity")
    print()

    assert discrete_quantum == 5, "Discrete quantum should be 5"
    assert identity_lhs == identity_rhs == K_open == 1280, "Kernel identity must hold"    
    assert abs(ratio_omega_to_open - (1 / A_kernel)) < 1e-10, "K_omega should be K_open × (1/A_kernel)"
    # K_invA uses integer truncation, so allow small tolerance
    assert abs(ratio_invA_to_open - expected_ratio_invA) < 1e-3, f"K_invA should be approximately K_open × (1/A_kernel)², got {ratio_invA_to_open:.6f} vs {expected_ratio_invA:.6f}"


def test_qg_capacity_candidate():
    """
    Test Q_G (4π) as a capacity multiplier candidate.
    
    Q_G = 4π ≈ 12.566 is the CGM horizon constant, representing the total
    solid angle (complete spherical closure). It's a fundamental CGM invariant
    that appears in the aperture constraint: Q_G × m_a² = 1/2
    
    This test checks if F_Moment = f_Cs × Q_G provides sufficient capacity
    for global UHI.
    """
    K_QG = Q_G
    
    F_per_sec = ATOMIC_HZ_CS133 * K_QG
    F_per_year = F_per_sec * SECONDS_PER_YEAR
    
    required = REQUIRED_CAPACITY_PER_YEAR
    ratio = F_per_year / required
    
    print("\n" + "=" * 70)
    print("Q_G (4π) CAPACITY CANDIDATE")
    print("=" * 70)
    print(f"\nQ_G (CGM horizon constant) = 4π = {Q_G:.12f}")
    print(f"K_QG multiplier             = {K_QG:.12f}")
    print()
    print("Capacity (per second - the actual continuous rate):")
    print(f"  F_Moment = f_Cs × Q_G = {F_per_sec:,.0f} MU/sec")
    print(f"                      = {F_per_sec:.2e} MU/sec")
    print()
    print("Annual capacity (for comparison only):")
    print(f"  F_Moment = {F_per_year:,} MU/year")
    print(f"          = {F_per_year:.2e} MU/year")
    print()
    print("Comparison with requirement:")
    print(f"  Required: {required:,} MU/year ({required:.2e} MU/year)")
    print(f"  Ratio:    {ratio:,.0f}× required capacity")
    print()
    print("=" * 70)
    if ratio > 1:
        print("CONCLUSION: Q_G PROVIDES SUFFICIENT CAPACITY")
        print(f"Q_G could fund global UHI for {ratio:,.0f} years.")
    else:
        print("CONCLUSION: Q_G DOES NOT PROVIDE SUFFICIENT CAPACITY")
    print("=" * 70)
    print()
    
    # Q_G is much smaller than the kernel candidates, so we need to check
    # if it's still sufficient (it should be, but let's verify)
    assert ratio > 1, f"Q_G capacity ({F_per_year:,}) must exceed requirement ({required:,})"
    
    # Note: Q_G is about 100× smaller than K_open, but should still be sufficient
    # Let's verify it provides at least 1000× margin
    assert ratio > 1000, f"Q_G should provide substantial margin, got {ratio:.0f}×"
