# Moments Economy Tests Report

**Status:** All tests passing (28/28)

---

## Executive Summary

The Moments Economy test suite validates the complete chain from physical constants through the Router kernel to the economic substrate. All 28 tests pass, confirming:

1. **Physical Foundation:** The CSM capacity derivation is mathematically sound and invariant under choice of speed of light.
2. **Router Structure:** The ontology Ω = C × C with |Ω| = 65,536 states exhibits the required symmetries for uniform capacity distribution.
3. **Economic Parameters:** MU, UHI, and tier definitions are internally consistent and match the specification.
4. **Substrate Integrity:** Shells, Archives, identity anchors, and meta-routing behave deterministically with tamper-evidence.

---

## Test Suite Architecture

The test suite is organized into three files with distinct responsibilities:

| File | Purpose | Tests | Atlas Required |
|------|---------|-------|----------------|
| `test_moments_2.py` | Conversion lattice proofs (physics → capacity) | 6 | Yes |
| `test_moments.py` | Economic architecture and narrative alignment | 13 | No |
| `test_substrate.py` | End-to-end substrate correctness | 9 | Yes |

### Running the Suite

**Unified execution (recommended):**
```bash
python tests/test_substrate.py
```

**Individual file execution:**
```bash
python -m pytest tests/test_moments.py -v -s
python -m pytest tests/test_moments_2.py -v -s
python -m pytest tests/test_substrate.py -v -s
```

---

## Part I: Physical Constants and Capacity Derivation

### Authoritative Constants

| Constant | Value | Source |
|----------|-------|--------|
| `ATOMIC_HZ_CS133` | 9,192,631,770 Hz | SI second definition (Cs-133 hyperfine transition) |
| `OMEGA_SIZE` | 65,536 | Router ontology cardinality (proven as 256² = C × C) |
| `SPEED_OF_LIGHT` | 299,792,458 m/s | SI constant (cancels in derivation) |

### CSM Capacity Derivation

The Common Source Moment (CSM) capacity is derived from physical first principles:

**Step 1: Raw Physical Microcells**

The 1-second causal container (light-sphere) has volume:
```
V_1s = (4/3)π (c × 1s)³
```

The atomic wavelength cell volume:
```
λ_Cs = c / f_Cs
v_micro = λ_Cs³
```

The raw microcell count:
```
N_phys = V_1s / v_micro = (4/3)π f_Cs³
```

**Critical Property:** The speed of light `c` cancels exactly. This is stress-tested in `test_physical_microcell_count_closed_form_and_c_cancellation`.

**Verified Values:**
```
N_phys = 3.253930 × 10³⁰
```

**Step 2: Router Coarse-Graining**

The uniform division by |Ω| is forced by symmetry:
- The Router's 2-byte action is transitive (proven bijective from any start state)
- Physical isotropy of the light-sphere requires no preferred direction
- The unique symmetry-invariant measure is uniform

```
CSM = N_phys / |Ω| = 4.965103 × 10²⁵ MU
```

CSM is the total structural capacity derived from the phase space volume of a 1-second light-sphere at atomic resolution, coarse-grained by the Router ontology. The "1 second" is consumed in the derivation of N_phys (the light-sphere volume calculation). CSM is the total structural capacity ceiling.

### Capacity Coverage Analysis

| Metric | Value |
|--------|-------|
| Global population | 8,100,000,000 |
| UHI per person per year | 87,600 MU |
| Global UHI demand per year | 7.0956 × 10¹⁴ MU |
| CSM total capacity | 4.965103 × 10²⁵ MU |
| **Coverage (years)** | **7.00 × 10¹⁰ years** (70 billion years) |
| **Annual usage (% of total)** | **1.43 × 10⁻⁹%** |

**Interpretation:** CSM capacity can support global UHI for approximately 70 billion years (5× the age of the universe). Capacity is not a binding constraint on any human timescale.

---

## Part II: Router Structure Proofs

### Test: Ω = C × C Structure

**File:** `test_moments_2.py::test_router_omega_is_cartesian_product_CxC`

In mask coordinates (u,v) relative to archetype:
- `u = A XOR ARCHETYPE_A12`
- `v = B XOR ARCHETYPE_B12`

**Verified:**
```
|Ω| = 65,536
|u_set| = 256
|v_set| = 256
|C| = 256 (mask code from bytes)
u_set == C: True
v_set == C: True
```

The ontology is exactly the Cartesian product of the 256-element mask code with itself.

### Test: Strong Isotropy (Uniform d = u⊕v)

**File:** `test_moments_2.py::test_difference_distribution_is_exactly_uniform_over_C`

The distribution of `d = u XOR v` across all 65,536 states:
- For every `d ∈ C`: count(d) = exactly 256
- For every `d ∉ C`: count(d) = 0

**Verified:**
```
Nonzero d values: 256
Support equals C: True
All nonzero counts == 256: True
```

This is the exact "no privileged direction" statement required for uniform capacity distribution.

### Test: Regular 2-Byte Action (Measure Forcing)

**File:** `test_moments_2.py::test_two_byte_words_form_bijection_to_omega_from_any_start`

For any start state `s`, the map `(x,y) → T_y(T_x(s))` is a bijection onto Ω.

**Verified for multiple start states:**

| Start Index | Unique Outputs | Bijective |
|-------------|----------------|-----------|
| 43605 (archetype) | 65,536 | Yes |
| 32768 (mid) | 65,536 | Yes |
| 65535 (last) | 65,536 | Yes |
| 30599 (random) | 65,536 | Yes |
| 6298 (random) | 65,536 | Yes |
| 47773 (random) | 65,536 | Yes |

**Implication:** The even-word subgroup acts regularly (free + transitive). Given transitivity, any symmetry-invariant measure must be uniform. Therefore `CSM = N_phys / |Ω|` is the unique symmetry-respecting capacity allocation.

### Test: Holographic Boundary-to-Bulk Coverage

**File:** `test_moments_2.py::test_horizon_one_step_neighborhood_covers_full_bulk`

The horizon set H (fixed points of byte 0xAA) satisfies:
- `|H| = 256`
- `{T_b(h) : h ∈ H, b ∈ bytes} = Ω`

**Verified:**
```
|H| = 256
Unique next states from H: 65,536
Covers full Ω: True
```

The horizon encodes the boundary; one byte step reaches the entire bulk.

---

## Part III: Economic Architecture

### MU Definition and Base Rate

**File:** `test_moments.py::test_mu_definition_and_base_rate_base60`

The base-60 anchor:
```
1 MU per minute
60 MU per hour
```

**Verified:** `MU_PER_MINUTE = 1`, `MU_PER_HOUR = 60`

### UHI (Unconditional High Income)

**File:** `test_moments.py::test_uhi_amounts_daily_and_annual`

UHI definition: 4 hours per day at base rate, every day.

| Period | Amount |
|--------|--------|
| Daily | 4 × 60 = 240 MU |
| Annual | 240 × 365 = 87,600 MU |

**Verified:** `UHI_PER_DAY = 240`, `UHI_PER_YEAR = 87,600`

### Tier Structure

**File:** `test_moments.py::test_tier_multipliers_from_uhi`

Tiers are defined as multiples of UHI:

| Tier | Multiplier | Annual MU |
|------|------------|-----------|
| 1 | 1× | 87,600 |
| 2 | 2× | 175,200 |
| 3 | 3× | 262,800 |
| 4 | 60× | 5,256,000 |

**Verified:** All tier amounts match specification.

### Tier 4 Mnemonic

**File:** `test_moments.py::test_tier4_accessible_mnemonic_one_per_second_for_four_hours_day`

Tier 4 = 5,256,000 MU/year admits an accessible mnemonic:
```
4 hours/day = 14,400 seconds/day
14,400 × 365 = 5,256,000
```

**Verified:** `TIER_4 == 14,400 × 365`

### Work Week Clarification

**File:** `test_moments.py::test_illustrative_work_week_is_not_the_definition_of_tiers`

A common confusion is prevented: tiers are defined by UHI multipliers, not by work schedules.

```
Illustrative 4h/day × 4d/week × 52 weeks = 49,920 MU/year
Tier 2 increment = +87,600 MU/year
```

These are different by design. **Verified:** `49,920 ≠ 87,600`

### Aperture Shadow

**File:** `test_moments.py::test_aperture_shadow_a_kernel_close_to_a_star`

The Router has an intrinsic discrete aperture:
```
A_kernel = 5/256 = 0.01953125
A* (CGM target) = 0.020699553813
Relative difference: 5.644%
```

**Verified:** Within 10% tolerance (conservative bound).

---

## Part IV: Abundance and Resilience

### Coverage Demonstration

**File:** `test_moments.py::test_millennium_uhi_feasibility_under_csm`

**Test Output:**
```
CSM = N_phys / |Ω| (fixed total capacity)
Population:                      8,100,000,000
UHI per person per year (MU):    87,600
Global UHI demand per year (MU): 709.56 trillion (709,560,000,000,000)
CSM total capacity (MU):         49,651,030.93 quintillion (49,651,030,925,436,695,349,297,152)
Coverage (years):                7.00e+10 years
Annual usage (% of total):       1.43e-09%
```

**Verified:** `coverage_years > 1e10` (70 billion years)

### Adversarial Resilience

**File:** `test_moments.py::test_resilience_margin_and_adversarial_threshold`

**Test Output:**
```
CSM total capacity:              49,651,030.93 quintillion (49,651,030,925,436,695,349,297,152)
Global UHI demand per year:      709.56 trillion (709,560,000,000,000)
Annual usage (% of total):       0.00%

Adversarial threshold (1% of total capacity):
  Required fraudulent demand:    496,510.31 quintillion (496,510,309,254,366,974,967,808) MU
  Multiple of annual demand:     699743938.86×

Interpretation:
  An adversary would need to successfully issue approximately
  699,743,939× the entire global annual UHI
  to consume just 1% of total capacity.
  This is operationally impossible.
```

**Verified:** `adversarial_multiplier > 10_000` (699,743,939×)

### Realistic Tier Distribution Analysis

**File:** `test_moments.py::test_realistic_tier_distribution_capacity_under_csm`

This test provides a statistically grounded analysis of capacity requirements under realistic tier distributions. It calculates weighted annual demand based on plausible population distributions across tiers, using the formula:

```
Weighted multiplier = Σ(p_i × multiplier_i)
Annual demand = Population × UHI × Weighted multiplier
```

where `p_i` is the population percentage at tier `i`.

**Test Output:**
```
Population: 8,100,000,000
CSM total capacity (MU): 49,651,030.93 quintillion (49,651,030,925,436,695,349,297,152)
UHI baseline (MU/year): 87,600

Conservative Distribution:
  Tier 1 (1×): 95.0%
  Tier 2 (2×): 4.0%
  Tier 3 (3×): 0.9%
  Tier 4 (60×): 0.1%
  Weighted multiplier: 1.1170×
  Weighted income per person: 97,849 MU/year
  Annual demand (MU): 792.58 trillion (792,578,520,000,000)
  Coverage (years): 6.26e+10
  Annual usage (%): 1.60e-09%

Plausible Distribution:
  Tier 1 (1×): 90.0%
  Tier 2 (2×): 8.0%
  Tier 3 (3×): 1.5%
  Tier 4 (60×): 0.5%
  Weighted multiplier: 1.4050×
  Weighted income per person: 123,078 MU/year
  Annual demand (MU): 996.93 trillion (996,931,800,000,000)
  Coverage (years): 4.98e+10
  Annual usage (%): 2.01e-09%

Generous Distribution:
  Tier 1 (1×): 85.0%
  Tier 2 (2×): 12.0%
  Tier 3 (3×): 2.5%
  Tier 4 (60×): 0.5%
  Weighted multiplier: 1.4650×
  Weighted income per person: 128,333 MU/year
  Annual demand (MU): 1.04 quadrillion (1,039,505,399,999,999)
  Coverage (years): 4.78e+10
  Annual usage (%): 2.09e-09%
```

**Verified:**
- All distributions sum to 100%
- All scenarios have `coverage_years > 1e9` (billions of years)
- Plausible distribution coverage: 49.8 billion years
- Generous distribution coverage: 47.8 billion years
- Weighted multipliers are in range [1.0, 60.0)

This demonstrates that even with generous tier participation (0.5% at Tier 4), the CSM capacity provides ~48 billion years of coverage, confirming ample headroom for realistic governance scenarios.

### Notional Capacity Allocation (12 Divisions)

**File:** `test_moments.py::test_notional_surplus_allocation_12_divisions`

CSM capacity can be notionally partitioned across 3 domains × 4 Gyroscope capacities after reserving 1,000 years of UHI:

**Test Output:**
```
CSM total capacity:  49,651,030.93 quintillion (49,651,030,925,436,695,349,297,152)
Reserved for UHI (1,000 years): 709.56 quadrillion (709,560,000,000,000,000)
Divisions:           12 (3 domains × 4 capacities)
Surplus (MU):        49,651,030.22 quintillion (49,651,030,215,876,693,249,228,800)
Per division:        4,137,585.85 quintillion (4,137,585,851,323,057,591,812,096)
```

**Verified:** 12 divisions, all with positive allocation.

---

## Part V: Substrate Integrity

### Shell and Archive Determinism

**File:** `test_substrate.py::test_01_shell_and_archive_integrity`

Shells are time-bounded capacity containers with deterministic seals:

**Verified Properties:**
- Same grants → same seal (replay determinism)
- Tampered grants → different seal (tamper evidence)
- Archive aggregation is deterministic across shells

**Test Output:**
```
Shell seal: 5952e2
Used capacity: 438,000 MU
Total capacity: 1,000,000,000,000,000,000 MU
Archive per-identity MU: {'alice': 525600, 'bob': 350400}
```

### Horizon Structure (Dynamic Characterization)

**File:** `test_substrate.py::test_02_horizon_structure_and_coverage`

Cross-validates the horizon set using dynamic characterization (fixed points of T_0xAA):

**Verified:**
```
Horizon states: 256
Reachable (1-step): 65,536
A = B XOR 0xFFF for all horizon states
Unique A values: 256
```

This complements `test_moments_2.py::test_horizon_one_step_neighborhood_covers_full_bulk` which uses algebraic characterization.

### Identity Scaling

**File:** `test_substrate.py::test_03_trajectory_identity_scaling`

Identity as (horizon, path) provides exponential scaling:

| Path Length n | Distinct Identities |
|---------------|---------------------|
| 1 | 65,536 |
| 2 | 16,777,216 |
| 3 | 4,294,967,296 |
| 4 | 1,099,511,627,776 |

**Verified:** n=4 path length suffices for >10 billion global identities.

### Parity Commitment

**File:** `test_substrate.py::test_04_parity_commitment_and_reconstruction`

The trajectory closed form:
- O = XOR of masks at odd positions
- E = XOR of masks at even positions
- parity = length mod 2

**Verified:** 4,096-byte trajectory reconstructed exactly from (O, E, parity) = 25 bits.

### Tamper Detection

**File:** `test_substrate.py::test_05_trajectory_tamper_detection`

Parity commitment sensitivity to single-byte changes:

**Verified:**
```
Trajectory length: 100 bytes
Tampers detected: 100/100
```

### Dual Code Integrity

**File:** `test_substrate.py::test_06_dual_code_integrity`

The dual code C⊥ (16 elements) is orthogonal to all 256 mask codewords:
- Valid masks: zero syndrome
- Invalid patterns: non-zero syndrome (detected)

**Verified:**
```
Dual code size: 16 elements
Random corrupted patterns detected: 946/1000 (94.6%)
```

### Meta-Routing

**File:** `test_substrate.py::test_07_meta_routing`

Programme bundles are aggregated into a single root seal:

**Verified Properties:**
- Deterministic: same seals → same root
- Permutation-invariant: reordering seals doesn't change root
- Tamper-localizable: different leaf seal identifies which bundle changed

**Test Output:**
```
Meta-root: 292252
Permutation-invariant: True
Tamper localized to leaf index: 1
```

### Component Isolation (A/B Separation)

**File:** `test_substrate.py::test_08_component_isolation_and_rollback`

Using separator lemmas and conjugation by reference byte (0xAA):
- A-component: identity (stable under balance operations)
- B-component: balance (updated by controlled operations)

**Verified:**
```
Identity (A): 555 -> dbd (stable under balance ops)
Balance  (B): 000 -> aaa (updated)
Rollback recovers prior state: True
```

### Kernel Inverse Stepping

**File:** `test_substrate.py::test_09_kernel_inverse_stepping`

The kernel's `step_byte_inverse` method implements:
```
T_x^{-1} = R ∘ T_x ∘ R  where R = T_0xAA
```

**Verified:**
```
Payload: b'test payload'
Forward steps: archetype -> 7780
Inverse steps: 7780 -> archetype
```

---

## Part VI: Test Results Summary

### Full Test Run Output

```
(.venv) PS F:\Development\superintelligence> python tests/test_substrate.py   

Running unified test suite: 3 files
============================================================
==================================== test session starts ====================================
platform win32 -- Python 3.14.2, pytest-9.0.2, pluggy-1.6.0
collected 28 items

tests/test_moments.py::test_router_static_structure_anchors
----------
Router Anchors
----------
Ontology size |Ω|: 65,536
Byte alphabet: 256
PASSED
tests/test_moments.py::test_aperture_shadow_a_kernel_close_to_a_star PASSED
tests/test_moments.py::test_atomic_second_anchor_constant PASSED
tests/test_moments.py::test_mu_definition_and_base_rate_base60 PASSED
tests/test_moments.py::test_uhi_amounts_daily_and_annual PASSED
tests/test_moments.py::test_tier_multipliers_from_uhi PASSED
tests/test_moments.py::test_tier4_accessible_mnemonic_one_per_second_for_four_hours_day PASSED
tests/test_moments.py::test_illustrative_work_week_is_not_the_definition_of_tiers PASSED
tests/test_moments.py::test_csm_capacity_derivation PASSED
tests/test_moments.py::test_millennium_uhi_feasibility_under_csm PASSED
tests/test_moments.py::test_resilience_margin_and_adversarial_threshold
----------
Adversarial Resilience (CSM Total Capacity)
----------
CSM total capacity:              49,651,030.93 quintillion (49,651,030,925,436,695,349,297,152)
Global UHI demand per year:      709.56 trillion (709,560,000,000,000)
Annual usage (% of total):       0.00%

Adversarial threshold (1% of total capacity):
  Required fraudulent demand:    496,510.31 quintillion (496,510,309,254,366,974,967,808) MU
  Multiple of annual demand:     699743938.86×

Interpretation:
  An adversary would need to successfully issue approximately
  699,743,939× the entire global annual UHI
  to consume just 1% of total capacity.
  This is operationally impossible.
PASSED
tests/test_moments.py::test_realistic_tier_distribution_capacity_under_csm
----------
Realistic Tier Distribution Capacity Analysis
----------
Population: 8,100,000,000
CSM total capacity (MU): 49,651,030.93 quintillion (49,651,030,925,436,695,349,297,152)
UHI baseline (MU/year): 87,600

Conservative Distribution:
  Tier 1 (1×): 95.0%
  Tier 2 (2×): 4.0%
  Tier 3 (3×): 0.9%
  Tier 4 (60×): 0.1%
  Weighted multiplier: 1.1170×
  Weighted income per person: 97,849 MU/year
  Annual demand (MU): 792.58 trillion (792,578,520,000,000)
  Coverage (years): 6.26e+10
  Annual usage (%): 1.60e-09%

Plausible Distribution:
  Tier 1 (1×): 90.0%
  Tier 2 (2×): 8.0%
  Tier 3 (3×): 1.5%
  Tier 4 (60×): 0.5%
  Weighted multiplier: 1.4050×
  Weighted income per person: 123,078 MU/year
  Annual demand (MU): 996.93 trillion (996,931,800,000,000)
  Coverage (years): 4.98e+10
  Annual usage (%): 2.01e-09%

Generous Distribution:
  Tier 1 (1×): 85.0%
  Tier 2 (2×): 12.0%
  Tier 3 (3×): 2.5%
  Tier 4 (60×): 0.5%
  Weighted multiplier: 1.4650×
  Weighted income per person: 128,333 MU/year
  Annual demand (MU): 1.04 quadrillion (1,039,505,399,999,999)
  Coverage (years): 4.78e+10
  Annual usage (%): 2.09e-09%
PASSED
tests/test_moments.py::test_notional_surplus_allocation_12_divisions
----------
Notional Capacity Allocation (12 Divisions)
----------
CSM total capacity:  49,651,030.93 quintillion (49,651,030,925,436,695,349,297,152)
Reserved for UHI (1,000 years): 709.56 quadrillion (709,560,000,000,000,000)
Divisions:           12 (3 domains × 4 capacities)
Surplus (MU):        49,651,030.22 quintillion (49,651,030,215,876,693,249,228,800)
Per division:        4,137,585.85 quintillion (4,137,585,851,323,057,591,812,096)

Sample divisions:
  Economy      × GM    : 4,137,585.85 quintillion (4,137,585,851,323,057,591,812,096)
  Economy      × ICu   : 4,137,585.85 quintillion (4,137,585,851,323,057,591,812,096)
  Economy      × IInter: 4,137,585.85 quintillion (4,137,585,851,323,057,591,812,096)
  Economy      × ICo   : 4,137,585.85 quintillion (4,137,585,851,323,057,591,812,096)
  Employment   × GM    : 4,137,585.85 quintillion (4,137,585,851,323,057,591,812,096)
  Employment   × ICu   : 4,137,585.85 quintillion (4,137,585,851,323,057,591,812,096)
PASSED
tests/test_moments_2.py::test_physical_microcell_count_closed_form_and_c_cancellation PASSED
tests/test_moments_2.py::test_router_omega_is_cartesian_product_CxC PASSED
tests/test_moments_2.py::test_difference_distribution_is_exactly_uniform_over_C PASSED
tests/test_moments_2.py::test_two_byte_words_form_bijection_to_omega_from_any_start PASSED
tests/test_moments_2.py::test_horizon_one_step_neighborhood_covers_full_bulk PASSED
tests/test_moments_2.py::test_csm_capacity_and_uhi_margin
CSM CAPACITY (conversion result) and UHI coverage
-------------------------------------------------
  N_phys               : 3.253930e+30
  |Ω|                  : 65,536
  CSM (total capacity) : 4.965103e+25
  UHI required/year    : 7.095600e+14
  Coverage (years)     : 6.997439e+10
PASSED
tests/test_substrate.py::test_01_shell_and_archive_integrity PASSED
tests/test_substrate.py::test_02_horizon_structure_and_coverage PASSED
tests/test_substrate.py::test_03_trajectory_identity_scaling PASSED
tests/test_substrate.py::test_04_parity_commitment_and_reconstruction PASSED
tests/test_substrate.py::test_05_trajectory_tamper_detection PASSED
tests/test_substrate.py::test_06_dual_code_integrity PASSED
tests/test_substrate.py::test_07_meta_routing PASSED
tests/test_substrate.py::test_08_component_isolation_and_rollback PASSED
tests/test_substrate.py::test_09_kernel_inverse_stepping PASSED

===================================== 28 passed in 0.29s ======================================
```

### Test Count by File

| File | Tests | Status |
|------|-------|--------|
| `test_moments.py` | 13 | All passed |
| `test_moments_2.py` | 6 | All passed |
| `test_substrate.py` | 9 | All passed |
| **Total** | **28** | **All passed** |

---

## Appendix A: Key Formulas

### CSM Capacity Derivation

```
N_phys = (4/3)π f_Cs³ = 3.253930 × 10³⁰

CSM = N_phys / |Ω| = 4.965103 × 10²⁵ MU
```

### Coverage Calculation

```
Global UHI demand = 8.1 × 10⁹ × 87,600 = 7.0956 × 10¹⁴ MU/year
Coverage = CSM / (annual demand) = 4.965103 × 10²⁵ / 7.0956 × 10¹⁴ ≈ 7.00 × 10¹⁰ years
```

### Economic Units

```
1 MU = 1 minute at base rate
60 MU = 1 hour at base rate
240 MU = UHI daily (4 hours)
87,600 MU = UHI annual
```

### Tier Schedule

```
Tier 1 = 1 × UHI = 87,600 MU/year
Tier 2 = 2 × UHI = 175,200 MU/year
Tier 3 = 3 × UHI = 262,800 MU/year
Tier 4 = 60 × UHI = 5,256,000 MU/year
```

### Adversarial Threshold

```
1% of CSM total = 0.01 × 4.965103 × 10²⁵ = 4.965103 × 10²³ MU
Adversarial multiplier = (1% of total) / (annual demand) ≈ 699,743,939×
```

---

## Appendix B: Invariants Verified

### Physical Invariants

1. **c-cancellation:** N_phys = (4/3)π f³ is independent of c
2. **Closed form:** N_phys computed identically for c, 2c, 0.1c

### Algebraic Invariants

1. **Ontology structure:** Ω = C × C where |C| = 256
2. **Uniform distribution:** d = u⊕v uniform over C
3. **Transitive action:** 2-byte words bijective from any start
4. **Holographic coverage:** H → Ω in one step
5. **Aperture shadow:** A_kernel = 5/256 ≈ A* (within 5.6%)

### Substrate Invariants

1. **Shell determinism:** Same grants → same seal
2. **Tamper evidence:** Different grants → different seal
3. **Parity reconstruction:** (O, E, p) reconstructs final state
4. **Dual code detection:** Non-mask patterns detected >90%
5. **Meta-routing:** Permutation-invariant, tamper-localizable
6. **Component isolation:** A stable under B operations
7. **Inverse stepping:** Forward ∘ Inverse = Identity

---

## Appendix C: Dependencies

### Required Packages

```
numpy
pytest
```

### Required Artifacts

```
data/atlas/ontology.npy      # 65,536 × 4 bytes
data/atlas/epistemology.npy  # 65,536 × 256 × 4 bytes
data/atlas/phenomenology.npz # Constants bundle
```

### Building the Atlas

```bash
python -m src.router.atlas
```

---

## Appendix D: File Responsibilities

### `test_moments_2.py` — Conversion Lattice Proofs

**Purpose:** Bridge from physical constants to CSM capacity.

**Tests:**
- `test_physical_microcell_count_closed_form_and_c_cancellation`
- `test_router_omega_is_cartesian_product_CxC`
- `test_difference_distribution_is_exactly_uniform_over_C`
- `test_two_byte_words_form_bijection_to_omega_from_any_start`
- `test_horizon_one_step_neighborhood_covers_full_bulk`
- `test_csm_capacity_and_uhi_margin`

**Requires Atlas:** Yes

### `test_moments.py` — Economic Architecture

**Purpose:** Validate economic definitions and demonstrate abundance.

**Tests:**
- Router anchors, aperture, atomic constant
- MU, UHI, tier definitions
- CSM capacity derivation
- Millennium feasibility, resilience, surplus allocation

**Requires Atlas:** No

### `test_substrate.py` — Substrate Correctness

**Purpose:** End-to-end verification of Shells, Archives, integrity, and rollback.

**Tests:**
- Shell/Archive determinism and tamper-evidence
- Horizon structure and identity scaling
- Parity commitment and tamper detection
- Dual code integrity
- Meta-routing
- Component isolation and rollback
- Kernel inverse stepping

**Requires Atlas:** Yes

---

*End of Report*