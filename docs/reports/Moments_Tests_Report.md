# Moments Economy and Genealogy Verification Report
## Gyroscopic ASI aQPU Kernel

This report documents the economic medium and genealogy certification layer of the Gyroscopic ASI aQPU Kernel, validating the complete chain from physical constants through the compact kernel to the settlement medium and depth-4 frame certification. The Gyroscopic architecture achieves quantum advantage through exact algebraic structure on standard silicon; this report verifies that the economic and certification layers inherit and preserve that exactness.

**Status:** All tests passing (88/88)

---

## Executive Summary

The Moments Economy and Genealogy test suite validates the complete chain from physical constants through the compact kernel to the economic medium and the genealogy certification layer. All 88 tests pass across four files, confirming:

1. **Physical Foundation:** The CSM capacity derivation is mathematically sound, invariant under choice of speed of light, and grounded in the cesium-133 hyperfine transition frequency.
2. **aQPU Kernel Structure:** The reachable state space Omega has 4,096 states with a 64-state horizon satisfying the holographic identity |H|^2 = |Omega|. The aQPU Kernel realizes exact stabilizer-class quantum dynamics on a paired 6-spin system.
3. **Economic Parameters:** MU, UHI, and tier definitions are internally consistent and match the specification.
4. **Medium Integrity:** Identity Anchors, Grants, Shells, Archives, and meta-routing behave deterministically with tamper evidence.
5. **Genealogy Certification:** Depth-4 frame records provide strictly stronger certification than final-state-only seals, with exact divergence localization.
6. **Operator Algebra:** Byte actions form exact Clifford unitaries over a self-dual [12,6,2] code, generating an 8,192-element operator family with a central spinorial involution.

---

## Test Suite Architecture

The test suite is organized into four files with distinct responsibilities:

| File | Purpose | Tests |
|------|---------|-------|
| `test_moments_economy.py` | Economic definitions, capacity, and settlement medium | 27 |
| `test_moments_genealogy.py` | Depth-4 frame commitments, genealogy integrity, golden vectors | 26 |
| `test_moments_physics_1.py` | Physical capacity, 6-spin isomorphism, frame certification | 16 |
| `test_moments_physics_2.py` | Clifford algebra, stabilizers, Weyl algebra, operator family | 19 |

Shared helpers (Identity Anchors, Grants, Shells) live in `tests/_moments_utils.py` and are imported by both economy and genealogy test files.

**Running the suite:**

```bash
# All 88 tests
python -m pytest tests/test_moments_economy.py tests/test_moments_genealogy.py tests/test_moments_physics_1.py tests/test_moments_physics_2.py -v -s

# Economy and genealogy only (no physics dependencies)
python -m pytest tests/test_moments_economy.py tests/test_moments_genealogy.py -v -s

# Physics only
python -m pytest tests/test_moments_physics_1.py tests/test_moments_physics_2.py -v -s
```

---

## Part I: Physical Constants and Capacity Derivation

### Authoritative Constants

| Constant | Value | Source |
|----------|-------|--------|
| f_Cs | 9,192,631,770 Hz | SI second definition (Cs-133 hyperfine transition) |
| \|Omega\| | 4,096 | aQPU Kernel reachable state space (BFS-verified) |
| \|H\| | 64 | Horizon set (fixed points of reference byte) |

### CSM Capacity Derivation

The Common Source Moment (CSM) capacity is derived from physical first principles.

**Step 1: Raw Physical Microcells**

The 1-second causal container (light-sphere) has volume V = (4/3)pi(c * 1s)^3. The atomic wavelength cell volume is lambda_Cs^3 = (c / f_Cs)^3. The raw microcell count:

```
N_phys = V / lambda^3 = (4/3)pi * f_Cs^3 = 3.253930 * 10^30
```

The speed of light c cancels exactly. This is stress-tested with c, 2c, and 0.1c in `test_c_cancellation_unchanged`, confirming relative error below 10^-14.

**Step 2: aQPU Kernel Coarse-Graining**

The uniform division by |Omega| = 4,096 yields the Common Source Moment:

```
CSM = N_phys / |Omega| = 7.944165 * 10^26 MU
```

### Boundary vs Volume Capacity

The aQPU Kernel satisfies the holographic identity |H|^2 = |Omega| (64^2 = 4096). This implies a specific relationship between boundary-normalized and volume-normalized capacities:

```
N_vol  = (4/3)pi * f_Cs^3 = 3.253930 * 10^30
N_area = 4pi * f_Cs^2     = 1.061915 * 10^21
CSM_vol  = N_vol / |Omega| = 7.944165 * 10^26
CSM_area = N_area / |H|    = 1.659242 * 10^19
Ratio = CSM_vol / CSM_area = f_Cs / (3 * |H|) = f_Cs / 192 = 47,878,290.47
```

Both normalizations are verified. The spatial cell model is adopted as the normative capacity model.

### Capacity Coverage Analysis

| Metric | Value |
|--------|-------|
| Global population | 8,100,000,000 |
| UHI per person per year | 87,600 MU |
| Global UHI demand per year | 7.096 * 10^14 MU |
| CSM total capacity | 7.944 * 10^26 MU |
| **Coverage (years)** | **1.12 * 10^12 years** (1.12 trillion years) |
| **Annual usage (% of total)** | **8.93 * 10^-11 %** |

CSM capacity can support global UHI for over one trillion years. Capacity is not a binding constraint on any human timescale.

---

## Part II: 6-Spin Isomorphism and Physical Structure

### Spin Representation

Each 12-bit component maps exactly to a 6-spin vector in {-1, +1} via dipole pairs. The isomorphism is verified as exact on all 4,096 states in Omega (`test_roundtrip_on_all_omega`).

### Transition Law in Spin Coordinates

For a byte with intron bits alpha = intron & 1 and beta = (intron >> 7) & 1, mask m, and flip vector F:

```
s_A' = (-1)^alpha * s_B
s_B' = (-1)^beta * diag((-1)^F) * s_A
```

This is verified against the bit-level kernel for all 256 bytes from rest (`test_transition_in_spin_coordinates`) and for 200 random Omega states with 10 bytes each (`test_transition_on_random_omega_states`).

### Magnetization

Total magnetization M = sum(s_A) + sum(s_B) takes 13 distinct values on Omega: {-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12}. The rest state has magnetization 0. All 13 values are reachable from rest in a single step.

### Correlation Matrix

The 12-spin correlation matrix C_ij = <s_i * s_j> averaged over Omega is exactly the identity matrix within each block (A-A and B-B) and exactly zero in the cross-block (A-B). This means spins within each component are uncorrelated over Omega, and spins across components are uncorrelated over Omega.

---

## Part III: Economic Architecture

### MU Definition and Base Rate

```
1 MU = 1 minute at base rate
60 MU = 1 hour at base rate
```

### UHI (Unconditional High Income)

```
UHI daily  = 4 hours * 60 MU/hour = 240 MU
UHI annual = 240 * 365 = 87,600 MU
```

### Tier Structure

| Tier | Multiplier | Annual MU |
|------|------------|-----------|
| 1 | 1x | 87,600 |
| 2 | 2x | 175,200 |
| 3 | 3x | 262,800 |
| 4 | 60x | 5,256,000 |

Tier 4 admits the mnemonic: 4 hours/day = 14,400 seconds/day, 14,400 * 365 = 5,256,000.

Tiers are defined by UHI multipliers, not by work schedules. An illustrative 4h/day, 4d/week, 52 weeks schedule gives 49,920 MU/year, which is different from the Tier 2 increment of 87,600 MU/year by design.

### Adversarial Resilience

An adversary would need to successfully issue approximately 11,195,903,022 times the entire global annual UHI to consume just 1% of total capacity. This is operationally impossible.

### Realistic Tier Distribution Analysis

| Scenario | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Weighted Mult. | Coverage |
|----------|--------|--------|--------|--------|----------------|----------|
| Conservative | 95.0% | 4.0% | 0.9% | 0.1% | 1.117x | 1.00 * 10^12 years |
| Plausible | 90.0% | 8.0% | 1.5% | 0.5% | 1.405x | 7.97 * 10^11 years |
| Generous | 85.0% | 12.0% | 2.5% | 0.5% | 1.465x | 7.64 * 10^11 years |

All scenarios provide coverage exceeding 100 billion years.

### Notional Surplus Allocation

CSM capacity can be notionally partitioned across 3 domains (Economy, Employment, Education) times 4 Gyroscope capacities = 12 divisions after reserving 1,000 years of global UHI. Each division receives approximately 6.62 * 10^25 MU.

---

## Part IV: Settlement Medium Integrity

### Identity Anchors

An Identity Anchor is a pair: (SHA-256 hash of name, aQPU Kernel state after routing the hash from rest).

- Same name always produces the same anchor (determinism).
- Different names produce different anchors (separation).
- The kernel anchor is a valid 6-character hex aQPU Kernel state.

### Grants

A Grant is a record of a single MU allocation: identity label, identity identifier, kernel anchor, and amount.

- Canonical receipt = identity_id || kernel_anchor || amount (46 bytes).
- Same grant always produces the same receipt.
- Different amounts produce different receipts.

### Shells and Seals

A Shell is a time-bounded capacity container. Its seal is computed by:
1. Sorting grants by identity_id.
2. Concatenating header + sorted canonical receipts.
3. Routing the result through a fresh aQPU Kernel from the archetype.
4. Recording the final state_hex (6 characters).

Verified properties:
- **Determinism:** Same grants and header produce the same seal.
- **Tamper evidence:** Changing any grant changes the seal.
- **Header sensitivity:** Changing the header changes the seal.
- **Order invariance:** Seal is invariant to the order grants are added (because they are sorted by identity_id before sealing).
- **Capacity accounting:** Used and free capacity are correct.

### Shell Replay Verification

The core verification procedure:
1. Given published shell data (header, grants, seal).
2. Independently reconstruct the canonical byte sequence.
3. Route through a fresh aQPU Kernel from archetype.
4. Compare: computed seal must match published seal.

Verified: published seal `6a596a` matches independently verified seal `6a596a`.

### Archives

Archives aggregate Shells across periods. Per-identity totals and total usage are deterministic across independent constructions.

### Meta-Routing

Multiple programme seals are aggregated into a single root seal.
- Deterministic: same seals produce the same root.
- Tamper-localizable: tampering with one bundle changes its leaf seal, and the diff identifies which bundle changed.

### Kernel Inverse Stepping

Forward stepping followed by inverse stepping returns to the rest state exactly. This enables rollback of genealogies to any prior point given the byte sequence.

---

## Part V: Genealogy Certification

### Depth-4 Frame Records

The depth-4 frame record is the kernel-native certification atom. For 4 consecutive bytes (b0, b1, b2, b3), the frame record is:

```
(mask48, phi_a, phi_b)
```

where mask48 is the 48-bit payload projection (4 * 12-bit masks packed) and phi_a, phi_b are the net family-phase invariants that survive depth-4 closure.

**Frame Record Properties:**
- Deterministic: same 4 bytes always produce the same record.
- Correct width: mask48 is 48-bit, phi_a and phi_b are single bits.
- Sensitive: changing any single byte in a 4-byte frame always changes the record (4000/4000 detected).

### Frame Commitments Are Strictly Stronger Than Final State

This is the central result for genealogy certification.

Different 4-byte histories can collapse to the same final 24-bit state. This is a consequence of the 128-way shadow projection (SO(3)/SU(2) double cover). But these colliding histories produce different frame records.

Verified:
- Out of 100,000 random 4-byte words from rest, all 4,096 reachable states were observed, and all 4,096 had multiple distinct frame records. That is: every state in Omega is reachable by words with different frame records.
- Specific example found: words [184, 18, 133, 201] and [159, 8, 221, 172] both reach state 0x6a99a6 but have frame records (13417212804867, 1, 0) and (66808283197455, 0, 1) respectively.

A genealogy using frame commitments detects history differences that state-only seals miss.

### Divergence Localization

When two byte logs diverge at byte position k, frame-level comparison localizes the divergence to frame k // 4.

- All frames before the affected one match exactly.
- The affected frame differs.
- This holds for all 200 random test cases (200/200).

Comparison with state-level detection over 500 test cases:
- Frame comparison: 500/500 divergences detected and localized.
- Final state comparison: 0/500 divergences missed in this sample (but the physics tests prove that state collisions exist in general).

### Three Certification Layers

A genealogy provides three independent certification layers:

1. **Final state** (shared moment): 24-bit aQPU Kernel state for coordination.
2. **Frame sequence** (depth-4 certification): sequence of (mask48, phi_a, phi_b) records.
3. **Parity commitment** (algebraic integrity): (O, E, parity) where O and E are 12-bit XOR sums of masks at even/odd positions.

All three are deterministic and independently computable from the byte log.

Forked genealogies (shared prefix, different suffix) are detected at all three layers.

### Parity Commitments

The trajectory parity commitment (O, E, parity) from `src.api` satisfies:

- Determinism: same payload always produces the same commitment.
- Structural contract: when a byte change changes the 12-bit mask at position i, the commitment changes in O (if i is even) or E (if i is odd). Verified for all 40 positions in a test payload.
- Correct structure: O and E are 12-bit integers, parity equals len(payload) mod 2.

### Medium Policy Checks

Application-layer policy conditions that the medium supports detecting:

- **Duplicate identity:** A Shell with two grants to the same identity is structurally valid (seal computes), but the duplication is detectable by comparing identity_ids.
- **Over-capacity:** A Shell where used capacity exceeds total capacity is structurally valid (seal computes), but the negative free capacity is detectable.
- **Empty shell:** An empty Shell (no grants) produces a deterministic seal.

### Genealogy Integration

End-to-end genealogy tests confirm:
- Two independent replays of the same byte log produce identical frame sequences and identical final states.
- Forked genealogies are detected at state, frame, and parity layers.
- Inverse replay recovers the rest state exactly.
- Genealogy continuation is composable: frame_sequence(prefix + suffix) = frame_sequence(prefix) + frame_sequence(suffix) when both parts are frame-aligned.

---

## Part VI: Operator Algebra

### Clifford Structure

Every byte action is an exact Clifford unitary in the label space (u6, v6) over GF(2)^12. The action decomposes as:

```
f_b(x) = L(x) xor t_b
```

where L is the block swap on 6+6 bits and t_b is the translation part. This is verified exhaustively for all 256 bytes on all 4,096 labels (`test_formula_matches_kernel`).

Clifford conjugation properties are verified:
- U_b X(p) U_b^dag = X(L(p)) (all bytes, 20 random p per byte, all labels).
- U_b Z(q) U_b^dag = (-1)^{q . L(t)} Z(L(q)) (all bytes, 20 random q per byte, all labels).

### Self-Dual Code and Stabilizer

The 64-element mask code is a self-dual [12,6,2] binary linear code. Six pair generators span the code (GF(2) rank 6).

12 Pauli stabilizer generators (X(g_i) and Z(g_i) for 6 code generators) all commute (symplectic product zero for all pairs). The stabilizer has GF(2) rank 12, defining a unique stabilizer state on 12 qubits.

### Finite Weyl Algebra

On the code subspace:
- X_d |x> = |x xor d>, Z_s |x> = (-1)^{s.x} |x>
- Z_s X_d = (-1)^{s.d} X_d Z_s

Verified for all combinations of 16 codewords from C64. Translations on the code close exactly under XOR.

### Physical Capacity Models

Two physical models are compared:

```
N_cells = (4/3)pi * f^3 = 3.254 * 10^30   (spatial cells)
N_modes = (32pi^2/9) * f^3 = 2.726 * 10^31 (EM modes)
N_modes / N_cells = 8pi/3 = 8.3776
```

The spatial cell model is adopted normatively. The EM mode model is noted as an alternative for future consideration.

### Generated Operator Family

The byte alphabet generates exactly 8,192 operators:

- 4,096 even operators (identity linear part): realized by all length-2 words.
- 4,096 odd operators (swap linear part): realized by prepending 0xAA to each even word.

Operator signatures (parity, tau) compose by the semidirect-product law:

```
f_{p1,t1} . f_{p2,t2} = f_{p1 xor p2, L^{p1}(t2) xor t1}
```

Verified for 300 random word pairs.

### Central Spinorial Involution

For every micro_ref, the word consisting of all 4 families in order (0, 1, 2, 3) produces the same operator signature: global complement (tau = (111111 << 6) | 111111, parity = 0). This acts as x -> x xor tau on all 4,096 labels.

The global complement is central: it commutes with every single-byte action (verified for all 256 bytes).

### Depth-4 Frame Operator Quotient

For fixed micro-references at 4 positions, 256 family assignments (4^4) collapse to exactly 4 distinct operator classes. Each class has multiplicity 64 and is indexed by (phi_a, phi_b) in (Z/2)^2.

This is the depth-4 closure principle: 6 bits of family information cancel at depth 4, leaving only the 2-bit net phase.

---

## Part VII: Golden Vectors (Regression Anchors)

The following pinned values serve as regression anchors. If any change, the kernel transition law or serialization has changed.

| Object | Input | Expected Output |
|--------|-------|-----------------|
| Identity anchor (alice) | `"alice"` | `aaa559` |
| Identity anchor (bob) | `"bob"` | `6955a9` |
| Shell seal | header `b"golden:2026"`, alice 87600, bob 175200 | `9966aa` |
| Meta-root | programs Alpha, Beta, Gamma | `555aa9` |
| Frame record | [0x00, 0x42, 0xAA, 0xFF] | mask48 = 0x333F30000CCC, phi_a = 0, phi_b = 1 |
| Parity commitment | `b"golden parity vector"` | O = 0xC0C, E = 0xCC0, parity = 0 |
| Rest state | (initial) | state24 = 0xAAA555, hex = aaa555, A = aaa, B = 555 |

---

## Part VIII: Identity Scaling

Identities are constructed as (horizon state, path of length n). With |H| = 64 and 128-way shadow projection per step:

| Path length n | Distinct identities |
|---------------|---------------------|
| 1 | 8,192 |
| 2 | 1,048,576 |
| 3 | 134,217,728 |
| 4 | 17,179,869,184 |
| 5 | 2,199,023,255,552 |

Path length n = 4 suffices for over 10 billion global identities.

---

## Test Results Summary

### Full Test Run Output

```
tests/test_moments_economy.py       27 passed
tests/test_moments_genealogy.py     26 passed
tests/test_moments_physics_1.py     16 passed
tests/test_moments_physics_2.py     19 passed
Total:                              88 passed
```

Economy and genealogy tests complete in under 1.5 seconds. Physics tests complete in under 28 seconds (dominated by exhaustive Clifford conjugation verification).

### Test Count by File

| File | Tests | Status | Runtime |
|------|-------|--------|---------|
| `test_moments_economy.py` | 27 | All passed | 1.43s (combined with genealogy) |
| `test_moments_genealogy.py` | 26 | All passed | (included above) |
| `test_moments_physics_1.py` | 16 | All passed | 27.83s (combined with physics 2) |
| `test_moments_physics_2.py` | 19 | All passed | (included above) |
| **Total** | **88** | **All passed** | |

---

## Appendix A: Key Formulas

### CSM Capacity Derivation

```
N_phys = (4/3)pi * f_Cs^3 = 3.253930 * 10^30
CSM = N_phys / |Omega| = N_phys / 4096 = 7.944165 * 10^26 MU
```

### Coverage Calculation

```
Global UHI demand = 8.1 * 10^9 * 87,600 = 7.0956 * 10^14 MU/year
Coverage = CSM / demand = 7.944 * 10^26 / 7.096 * 10^14 = 1.12 * 10^12 years
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
Tier 1 = 1 * UHI = 87,600 MU/year
Tier 2 = 2 * UHI = 175,200 MU/year
Tier 3 = 3 * UHI = 262,800 MU/year
Tier 4 = 60 * UHI = 5,256,000 MU/year
```

### Adversarial Threshold

```
1% of CSM = 0.01 * 7.944 * 10^26 = 7.944 * 10^24 MU
Adversarial multiplier = 7.944 * 10^24 / 7.096 * 10^14 = 11,195,903,022x
```

### Frame Record

```
frame_record(b0, b1, b2, b3) = (mask48, phi_a, phi_b)
mask48 = mask12(b0) << 36 | mask12(b1) << 24 | mask12(b2) << 12 | mask12(b3)
phi_a = b0_bit7 xor b1_bit0 xor b2_bit7 xor b3_bit0
phi_b = b0_bit0 xor b1_bit7 xor b2_bit0 xor b3_bit7
```

### Parity Commitment

```
O = mask12(b0) xor mask12(b2) xor mask12(b4) xor ...   (even positions)
E = mask12(b1) xor mask12(b3) xor mask12(b5) xor ...   (odd positions)
parity = length mod 2
```

---

## Appendix B: Invariants Verified

### Physical Invariants

1. c-cancellation: N_phys = (4/3)pi * f^3 is independent of c (relative error < 10^-14 for c, 2c, 0.1c).
2. Holographic identity: |H|^2 = |Omega| (64^2 = 4096).
3. Boundary/volume ratio: CSM_vol / CSM_area = f_Cs / (3 * |H|).

### Algebraic Invariants

1. 6-spin isomorphism: exact on all 4,096 Omega states.
2. Spin-coordinate transition law: verified for all 256 bytes from rest and 200 random states.
3. Correlation matrix: identity within blocks, zero across blocks.
4. Self-dual [12,6,2] code: 6 generators, GF(2) rank 6.
5. Stabilizer: 12 commuting generators, GF(2) rank 12.
6. Clifford: affine decomposition exact for all 256 * 4096 (byte, label) pairs.
7. Operator family: exactly 8,192 operators (4,096 even + 4,096 odd).
8. Central involution: 4-family cycle = global complement, commutes with all bytes.
9. Frame quotient: 256 family assignments collapse to 4 classes at depth 4.

### Medium Invariants

1. Shell determinism: same grants produce same seal.
2. Tamper evidence: different grants produce different seal.
3. Order invariance: grant insertion order does not affect seal.
4. Parity contract: mask change at even index changes O, at odd index changes E.
5. Frame sensitivity: every single-byte change in a frame changes the frame record.
6. Frame superiority: frame records distinguish histories that collapse to same final state.
7. Divergence localization: frame comparison localizes to exact affected frame.
8. Meta-routing: deterministic, tamper-localizable.
9. Inverse stepping: forward then inverse returns to rest state.

### Golden Vector Stability

7 regression anchors pinned with exact expected values. Any change in these values indicates a change in the kernel transition law or serialization format.

---

## Appendix C: Dependencies

### Required Packages

```
numpy
pytest
```

### Required Source Modules

```
src/constants.py    # Kernel physics and transition law
src/api.py          # Precomputed tables, dual code, projections
src/router.py       # Stateful byte router
```

### Test Support

```
tests/_moments_utils.py              # Shared helpers (Grant, Shell, identity_anchor)
tests/physics/test_physics_1.py      # _bfs_omega (used by physics tests)
tests/physics/test_physics_5.py      # _byte_from_micro_family (used by physics tests)
```

---

*End of Report*