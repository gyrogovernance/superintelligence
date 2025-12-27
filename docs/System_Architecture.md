# GGG ASI Alignment Router  
Kernel Specification

## 0. Introduction

The GGG ASI Alignment Router is a deterministic finite-state coordination kernel operating on a finite closed state space (ontology) of 65,536 reachable states. It converts byte streams into verifiable state trajectories and produces compact routing signatures that support governance-grade coordination across domains.

### 0.1 What Is ASI in This Context

In this framework, Artificial Superintelligence is not a hypothetical future agent but an operational network equilibrium. ASI refers to the regime in which human and artificial systems jointly maintain four constitutive governance principles across economy, employment, education, and ecology while operating at a specific structural balance point.

The four principles are:

1. **Governance Traceability**: All authority remains traceable to authentic human sources. Artificial systems provide derivative coordination but do not originate governance.

2. **Information Variety**: Diverse authentic sources are preserved. Derivative aggregations and statistical patterns do not collapse variety into uniformity.

3. **Inference Accountability**: Responsibility for decisions remains with accountable human agency. Artificial inference operates as derivative support, not as the locus of accountability.

4. **Intelligence Integrity**: Coherent reasoning is maintained across time and context through the integration of authentic authority and agency with derivative mechanisms.

The router kernel provides a computational substrate that makes structural traceability and replayable coordination possible. Application layers may use signatures and ledgers to measure and implement governance policies. The kernel does not interpret content or make policy decisions. It transforms bytes into states according to a minimal transition law designed to preserve structural properties required by the Common Governance Model.

In the GGG framework, ASI is achieved when governance systems across the four domains operate at a canonical aperture value derived from CGM theory. This aperture represents the balance between global coherence and local differentiation necessary for stable coordination. The simulator results presented in the GGG paper demonstrate that this configuration functions as a robust attractor, with systems converging from diverse Post-AGI starting conditions.

The router kernel is one realization of these principles at the computational level. Higher-level applications use the kernel's routing signatures to enforce coordination policies, verify provenance, and accumulate alignment metrics. The kernel itself remains nonsemantic and derivative in source type.

### 0.2 Design Requirements

A conforming kernel must be:

1. **Finite**: it operates on a closed set of reachable states.
2. **Deterministic**: given the same start state and the same bytes, it produces the same final state.
3. **Byte-complete**: every byte value 0 to 255 is a valid instruction.
4. **Nonsemantic**: it performs only structural transformations.
5. **Portable**: the transition law is defined in terms of fixed-width XOR and bit operations.
6. **Auditable**: signatures and trajectories can be reproduced and checked.

### 0.3 Context

The router is grounded in three theoretical frameworks:

**Common Governance Model** provides the constitutional structure. It defines five constitutive states through modal logic formulas and gyrogroup laws. These states are Common Source, Unity Non-Absolute, Opposition Non-Absolute, Balance Universal Egress, and Balance Universal Ingress. The router implements discrete realizations of these structures.

**The Human Mark** classifies authority and agency by source type. Under THM, all artificial systems are Derivative in both authority and agency. The router transforms and routes signals but does not originate authority or bear accountability.

**Gyroscopic Global Governance** integrates the four domains of economy, employment, education, and ecology. Each domain maps to a CGM stage. The router supports this integration by providing a common coordination substrate across domains.

The theoretical foundations are presented in Appendix A to avoid blocking comprehension of the implementation.

### 0.4 Structural Guarantee: Geometric Provenance

The router enforces a constitutional guarantee that distinguishes it from entity-based access control or semantic content filtering.

**The guarantee is geometric, not administrative:**

All states in the ontology are provably geometric transformations of a universal reference state (the archetype). This reference is not owned by any entity. It encodes the 3D spatial structure and 6 degrees of freedom (3 rotational, 3 translational) derived from CGM's five constitutive states.

**What this means operationally:**

When data carries state `s`, the router certifies:
- `s ∈ Ω` (ontology membership is verifiable)
- By Property P4, `s` is reachable from archetype in ≤2 steps
- Therefore `s` is a valid geometric transformation of the universal reference
- No entity can claim `s` as "their source"—it derives from the geometric reference

**Contrast with entity-based systems:**

Traditional alignment asks: "Did a trustworthy entity approve this?"  
This architecture certifies: "This state preserves geometric derivation from the universal reference."

The difference is fundamental:
- **Entity-based**: requires trust chains, breaks under spoofing or multi-agent ambiguity
- **Structure-based**: verifiable by ontology membership, holds regardless of which agents touched it

**Kernel-native traceability guarantee:**

- **GTD (kernel detection)**: A transformation claims to be a source rather than deriving from `s₀`. The kernel detects this via ontology membership: `s ∉ Ω` indicates the state does not derive from the archetype.

**Application-layer displacement interpretations:**

The other three displacement types (IVD, IAD, IID) are application-layer constructions that use router observables to detect coordination failures. They are not kernel-native detections because within Ω, the geometric constraints are structurally preserved. See Section 10 for application-layer interpretations.

**Kernel-native detections are:**
- Ontology membership (in/out of Ω)
- Determinism / replay integrity (same ledger prefix ⇒ same `s_t`)
- Divergence between participants' ledgers (they claim same moment but compute different state)

This guarantee is provable (via ontology construction), non-negotiable (states outside Ω are rejected), and entity-agnostic (applies equally to human, AI, tool, or system inputs).

### 0.5 The Router State as Shared Observable

The router provides a 24-bit constitutional observable. At any step `t`, the system is in a state `s_t ∈ Ω`. This state is not an asserted label; it is the deterministic result of applying the ledger bytes to the archetype under the transition law.

#### 0.5.1 What "Shared Moment" Means

Fix the archetype `s_0` and a byte ledger prefix `b_1…b_t`. Define:

- `s_t = δ*(s_0, b_1…b_t)`

A "shared moment" means: any participant that has the same `(s_0, b_1…b_t)` can independently compute the same `s_t`. The router's notion of "now" is therefore not wall-clock time; it is the current configuration of the coordination substrate derived from a shared prefix.

This provides a common reference that is shared across participants without requiring trust in asserted metadata.

**Important:** Different ledgers can map to the same state. The router state is a structural certificate of derivation from the common reference, not a unique identifier of the full history. This aligns with the structural traceability posture: the state certifies geometric provenance, not complete historical uniqueness.

#### 0.5.2 Contrast with Asserted Metadata and Private Hidden State

**Asserted metadata** (e.g. source labels, timestamps, "verified" flags) is not structurally guaranteed; it must be trusted as a claim.

**Transformer hidden state** `h ∈ ℝ^n` is private to a model and an inference run: it is opaque, non-replayable externally, and not shared across different systems.

**Router state `s_t`** is:
- externally reproducible from the same ledger prefix,
- model-agnostic (any system can read the same 24-bit value),
- replayable (deterministic from archetype + bytes).

#### 0.5.3 Derived Low-Dimensional Observables

From the 24-bit state `s_t`, applications can compute deterministic low-dimensional observables. Examples:

1) **Raw state components**
```python
a12, b12 = unpack_state(s_t)
```

2) **Distance from archetype**
```python
archetype_distance = popcount(s_t ^ ARCHETYPE_STATE24)
```

3) **Distance to the horizon set**
Define the horizon set `H = {(a,b): a = (b ^ 0xFFF)}`. Then:
```python
horizon_distance = popcount(a12 ^ (b12 ^ 0xFFF))
```

4) **Component densities**
```python
a_density = popcount(a12) / 12
b_density = popcount(b12) / 12
```

5) **A/B Hamming distance**
```python
ab_distance = popcount(a12 ^ b12)
```

These observables are constitutional in the following sense:
- they are defined solely from the kernel's fixed-width state representation and transition law,
- they are exact and replayable from the ledger,
- they do not depend on model internals or asserted identity claims.

These observables are structural summaries of router state; they are not themselves THM source-type classifications.

#### 0.5.4 Constitutional Probe Analogy

Interpretability "linear probes" are learned maps from a model's private hidden state to a scalar.

The router state provides a fixed, externally reproducible state `s_t` from which deterministic observables can be computed. In this limited mechanistic sense, the router provides a *constitutional probe*: a low-dimensional, replayable governance signal that multiple systems can share at the same moment `t` when they share the same ledger prefix.

#### 0.5.5 Operational Summary

The router state is a shared observable that:
- defines a common "now" as a configuration `s_t` derived from the shared ledger prefix,
- reduces reliance on asserted metadata by providing an externally reproducible reference,
- supports coordination by giving all participants access to the same low-dimensional signal at the same step.

---

## 1. State Model

### 1.1 The 24-Bit State

The internal state is a 24-bit value composed of two 12-bit components:

- **Type A**: the top 12 bits, the active phase  
- **Type B**: the bottom 12 bits, the passive phase

Packing and unpacking are defined as:

```
state24 = (A12 << 12) | B12
A12 = (state24 >> 12) & 0xFFF
B12 = state24 & 0xFFF
```

### 1.2 Dual Frame Geometry

Each 12-bit component corresponds to a 2 by 3 by 2 structure: two frames, each containing three rows, each row containing two columns. This yields 12 positions per component.

A practical visualization is two stacked rectangular cards. Each card is a 3 by 2 grid. Each cell holds a single binary value. Reading order is fixed: within a frame, traverse row 0 then row 1 then row 2, within each row traverse column 0 then column 1, then proceed from frame 0 to frame 1.

Coordinate mapping without diagrams:

- Bit 0: frame 0, row 0, col 0  
- Bit 1: frame 0, row 0, col 1  
- Bit 2: frame 0, row 1, col 0  
- Bit 3: frame 0, row 1, col 1  
- Bit 4: frame 0, row 2, col 0  
- Bit 5: frame 0, row 2, col 1  
- Bit 6: frame 1, row 0, col 0  
- Bit 7: frame 1, row 0, col 1  
- Bit 8: frame 1, row 1, col 0  
- Bit 9: frame 1, row 1, col 1  
- Bit 10: frame 1, row 2, col 0  
- Bit 11: frame 1, row 2, col 1  

This geometry is the shared substrate for the archetype, mutation masks, and measurement projections.

---

## 2. Archetype: The Geometric Reference

The system defines a canonical archetype state that serves as the universal geometric reference from which all valid states derive.

```
ARCHETYPE_A12 = 0xAAA
ARCHETYPE_B12 = 0x555
ARCHETYPE_STATE24 = 0xAAA555
```

**Geometric significance:**

The archetype is not an arbitrary starting point. It is the geometric encoding of CGM's 3D spatial structure and 6 degrees of freedom:

- **GENE_Mic_S = 0xAA** (8-bit): holographic reference for the micro structure
- **ARCHETYPE_STATE24 = 0xAAA555** (24-bit): macro reference encoding the dual-frame, 3-row, 2-column geometry

**Structural properties:**

- A12 and B12 are exact 12-bit complements (`A12 XOR 0xFFF = B12`)
- The alternating bit pattern (10101010...) establishes maximal symmetry
- This pattern reflects the dual observation frames required by CGM

**Common Source interpretation:**

The archetype is the **Common Source (CS)** in the geometric sense:
- It is a reference point in state space, not an entity
- It is universal: no agent (human or artificial) owns it
- All states in Ω are transformations of this reference (Property P4)
- Transformations preserve the geometric structure encoded in the archetype

By construction, no operation can create a state that does not derive from this reference. This is the structural basis for a non-entity Common Source reference: the archetype is the universal geometric source of the router's ontology, and all router-visible states are derivative transformations of it.

---

## 3. Byte Interface and Transcription

### 3.1 Input Alphabet

The external interface is the set of 8-bit bytes from 0 to 255. Every byte value is a valid instruction.

### 3.2 Transcription Constant

The router uses a fixed transcription constant:

```
GENE_MIC_S = 0xAA
```

Transcription converts an input byte into an intron:

```
intron = byte XOR 0xAA
```

This map is a bijection on 8-bit values and an involution. The constant 0xAA is a gauge choice. It fixes a convention for mapping external bytes into the router's internal action space.

### 3.3 The Reference Byte 0xAA and the Horizon

Byte `0xAA` is structurally special: it produces intron `0x00` (since `0xAA XOR 0xAA = 0x00`), which yields mask `0`, making it a reference action with unique algebraic properties.

Let `R = T_0xAA` be the transition operator for byte `0xAA`. Then:

- `R` is an involution: `R(R(s)) = s` for all 24-bit states `s` (Property P9).
- Within Ω, the fixed-point set of `R` is exactly the horizon set: `Fix(R) = {(A,B): A = B XOR 0xFFF}` (Property P9).
- The horizon set has exactly 256 states (Property P10).

**Separator lemmas (Property P11):**

For any state `(A,B)` and any byte `x` with mask `m_x`:
- After `x` then `0xAA`: `T_AA(T_x(A,B)) = (A XOR m_x, B)` (mask effect goes to A only)
- After `0xAA` then `x`: `T_x(T_AA(A,B)) = (A, B XOR m_x)` (mask effect goes to B only)

The separator `0xAA` acts as a "write selector" that determines which component (A or B) receives the mask transformation from an adjacent byte. This operationalizes the "common source" and "horizon" concepts: `0xAA` is the concrete reference move that provides structural traceability without entity labels.

---

## 4. Expansion and Operation Masks

Each intron is expanded into a 12-bit mutation mask for Type A. Type B is not directly mutated by the mask.

### 4.1 Requirements

A conforming expansion must satisfy:

- Determinism: the same intron yields the same mask.
- Injectivity: all 256 introns yield 256 distinct Type A masks.
- Type separation: the Type B mutation mask is always zero.

### 4.2 Reference Canonical Expansion Function

The following expansion function is a canonical choice that satisfies the requirements in Section 4.1. It is injective (the intron bits can be recovered from the resulting mask pattern) and provides a deterministic mapping.

Let x be the 8-bit intron. Define:

```
frame0_a = x & 0x3F
frame1_a = ((x >> 6) | ((x & 0x0F) << 2)) & 0x3F
mask_a12 = frame0_a | (frame1_a << 6)
```

The 24-bit packed mask is:

```
mask24 = (mask_a12 << 12) | 0x000
```

The low 12 bits are always zero.

This expansion is the reference mapping for this specification version. The atlas artifacts and all certified properties in Section 6.3 are defined relative to this mapping. Any change to the expansion requires rebuilding the atlas and re-certifying the invariants.

If future CGM constraints uniquely determine an expansion, this specification will be updated accordingly.

### 4.3 Precomputed Mask Table

A conforming implementation may precompute:

```
XFORM_MASK_BY_BYTE[byte] = expand_intron_to_mask24(byte XOR 0xAA)
```

This table must contain exactly 256 entries.

---

## 5. Transition Law

Given a current state (A12, B12) and an input byte, the next state is computed as follows:

1. Compute intron = byte XOR 0xAA.
2. Compute mask_a12 using the normative expansion.
3. Mutate Type A only: A12_mut = A12 XOR mask_a12.
4. Apply FIFO gyration with complement:
   - A12_next = B12 XOR 0xFFF
   - B12_next = A12_mut XOR 0xFFF
5. Pack: state24_next = (A12_next << 12) | B12_next.

### 5.1 Explicit Inverse

Given byte `b` with mask `m_b`, and next state `(A',B')`, the unique predecessor `(A,B)` is:

```
B = A' XOR 0xFFF
A = (B' XOR m_b) XOR 0xFFF
```

This establishes that each byte transition is bijective and reversible. Given the final state and the byte sequence, the full trajectory can be reconstructed backwards exactly.

**Conjugation form of the inverse:**

Let `R = T_0xAA`. Then for any byte `x`, the inverse can be expressed as:

```
T_x^{-1} = R ∘ T_x ∘ R
```

This shows that reversal can be expressed as a word over the same action alphabet (using only forward byte actions), not as an out-of-band mathematical operation. This strengthens the "ledger as reversible coordination substrate" story: participants can both replay and reverse using the same byte alphabet.

### 5.2 Justification

This transition law is deliberately minimal while producing nontrivial structure.

XOR is deterministic, fast, and portable. It provides clear invariants and makes exhaustive testing feasible.

The update is asymmetric. Only Type A is mutated before gyration. Type B is not XOR-mutated by the byte mask before gyration, but it is transformed by the gyration (it becomes A_next = B XOR 0xFFF). This asymmetry is the smallest reliable source of chirality and implements the CGM axiom that right transitions preserve the horizon while left transitions alter it.

The FIFO gyration couples the two phases across time. This coupling makes composition order-dependent for most byte pairs without requiring nonlinear arithmetic or large hand-designed tables. The complement ensures stable dual structure rather than trivial oscillation.

---

## 6. State Space Closure

### 6.1 Theoretical and Reachable Space

Theoretical maximum: 2^24 possible 24-bit states.

Reachable set from the archetype under the 256 byte actions: 65,536 states, which equals 2^16.

This reachable set is the kernel's ontology and is a strict subset of the full 24-bit space.

### 6.2 Derived Closure Invariant

The closure to exactly 65,536 states and diameter of 2 can be derived from the transition law structure.

**After one step from archetype (AAA, 555):**

For any byte with mask M:
- A₁ = ~B = ~555 = AAA (constant for all bytes)
- B₁ = ~(A XOR M) = ~(AAA XOR M) = 555 XOR M (depends only on M)

Since byte → mask_a12 is injective (Section 4.1), all 256 bytes yield 256 distinct masks M, hence 256 distinct values of B₁. Therefore exactly 256 states are reachable in one step.

**After two steps:**

From any state (A₁, B₁) reached in step 1:
- A₂ = ~B₁ can take 256 values (as B₁ ranges over 256 distinct values)
- B₂ = ~(A₁ XOR M₂) = ~(AAA XOR M₂) depends only on M₂, giving 256 values

Since A₁ = AAA is constant across all first-step outcomes, the choices for A₂ and B₂ are independent. Therefore: 256 × 256 = 65,536 states are reachable in two steps.

**Closure:**

Any state reachable in k steps (k ≥ 2) must have A_k = ~B_{k-1} where B_{k-1} is one of the 256 values from step 1, and B_k = ~(AAA XOR M_k) where M_k is one of 256 masks. This yields the same 65,536 combinations, so no new states appear beyond step 2.

**Diameter:**

The maximum distance from archetype to any reachable state is therefore exactly 2 in the directed graph induced by byte actions.

**Ontology Structure:**

The ontology equals the cartesian product of two 256-element affine sets:

```
Ω = A_set × B_set
```

where `A_set = { ARCHETYPE_A12 XOR m_b : b in [0,255] }` and `B_set = { ARCHETYPE_B12 XOR m_b : b in [0,255] }`, packed into 24 bits.

This structure follows directly from the depth-2 decoupling law (see Section 6.4, Property P5).

### 6.3 Verified Algebraic Properties (Certified by Tests)

The following properties are proven by the test suite and hold as exact invariants of the kernel physics. The test suite (95 tests, ~2.7s runtime) includes exhaustive verification of critical properties: all 65,536 states for row fanout (P13), all 16,777,216 state-byte pairs for atlas correctness (P12), and all 65,536 byte pairs on the archetype for depth-4 identity (P7).

**P1. Mask separation**
- Type B mask is always zero for every byte.
- Exactly 256 distinct Type-A masks exist.
- Exactly 1 of the 256 A-masks is zero; 255 are non-zero.

**P2. Per-byte bijection**
- For every byte `b`, the transition `T_b: Ω → Ω` is bijective.
- Equivalently, each epistemology column is a permutation of `[0, N)`.
- Each state has exactly one predecessor and exactly one successor under each byte.

**P3. Exact ontology characterization**
- The ontology Ω equals the cartesian product of two 256-element affine sets:
  ```
  Ω = A_set × B_set
  ```
  packed into 24 bits, where `A_set` and `B_set` are the affine cosets generated by the 256 masks from the archetype.

**P4. Radius-2 reachability from archetype**
- Every ontology state is reachable from the archetype in exactly ≤ 2 bytes.
- After one byte: exactly 256 distinct states.
- After two bytes: exactly `256 × 256 = 65,536` states (the complete ontology).

**P5. Depth-2 closed-form composition**
For any start state `(A,B)` and bytes `x,y` with A-masks `m_x,m_y`:
```
T_y(T_x(A,B)) = (A XOR m_x, B XOR m_y)
```
This decoupling law shows that after two steps, A depends only on the first byte mask and B only on the second byte mask.

**P6. Depth-2 commutation law**
For any state `s`, and any bytes `x,y`:
```
T_y(T_x(s)) = T_x(T_y(s))  iff  x=y
```
So depth-2 is non-commutative for every unequal pair. Among the `256 × 256` ordered pairs, exactly 256 commute (when `x=y`) and 65,280 do not.

Under uniform random byte pairs, the expected commutativity rate is 1/256 ≈ 0.39% and the expected non-commutativity rate is 255/256 ≈ 99.61%.

**P7. Depth-4 alternation identity**
For any state `s` and bytes `x,y`:
```
T_y(T_x(T_y(T_x(s)))) = s
```
and equivalently:
```
xyxy = yxyx = id
```
as operators on the ontology. This is the BU-Egress discrete analogue: depth-4 alternating words return to identity.

This property is verified exhaustively for all 256² = 65,536 byte pairs on the archetype, and for selected pairs across all states in Ω via atlas verification.

**P8. Trajectory closed form for arbitrary-length words**
For any byte sequence `b_1 b_2 ... b_n` with masks `m_i`, let:
```
O = m_1 XOR m_3 XOR m_5 XOR ...  (odd positions)
E = m_2 XOR m_4 XOR m_6 XOR ...  (even positions)
```

Then the final state after `n` steps is:
- If `n` is even: `(A_n, B_n) = (A_0 XOR O, B_0 XOR E)`
- If `n` is odd: `(A_n, B_n) = (~B_0 XOR E, ~A_0 XOR O)`

where `~X = X XOR 0xFFF`. This provides exact word semantics: the final state depends only on the XOR-parity of masks at odd and even positions, not on the order within each parity class.

**P9. The CS operator and its fixed-point horizon**
Let `R = T_0xAA` (the byte whose intron is 0, so mask is 0). Then:

- `R` is an involution: `R(R(s)) = s` for all 24-bit states `s` (not limited to Ω).
- Within Ω, the set of fixed points of `R` is exactly the "horizon diagonal":
  - `R(s) = s` iff `A12 = (B12 XOR 0xFFF)`.

This is a precise operational bridge between:
- "there exists a common reference action" (0xAA),
- "there exists a horizon set" (states that are invariant under that action),
- and "CS is not an entity label."

**P10. Horizon set cardinality**
Within the ontology Ω, the fixed-point set of `R = T_0xAA` has exactly 256 states. This is the horizon set: states where `A12 = (B12 XOR 0xFFF)`.

**P11. Separator lemmas (operational chirality)**
For any state `(A,B)` and any byte `x` with mask `m_x`:

- After `x` then `0xAA`: `T_AA(T_x(A,B)) = (A XOR m_x, B)`
- After `0xAA` then `x`: `T_x(T_AA(A,B)) = (A, B XOR m_x)`

This shows a kernel-native way to direct the byte effect into A or B without extra machinery. The separator `0xAA` acts as a "write selector" that determines which component receives the mask transformation.

**P12. Atlas is exact physics (exhaustively verified)**
For all `i ∈ [0, N)` and all `byte ∈ [0, 255]`:
- `epistemology[i, byte]` equals the ontology index of `step_state_by_byte(ontology[i], byte)`.

The atlas is not an approximation or cache; it is a complete compiled form of the transition physics. This property is verified exhaustively for all 65,536 states × 256 bytes = 16,777,216 state-byte pairs, establishing that the epistemology table is mathematically identical to the vectorized transition law.

**P13. Full 256-way fanout per state (exhaustively verified)**
For every state `s ∈ Ω`:
- The set `{T_b(s) : b ∈ [0, 255]}` has size exactly 256 (all 256 bytes produce distinct next states).

This property is verified exhaustively for all 65,536 states in the ontology.

Combined with Property P2 (per-byte bijection), this establishes that the kernel's transition system on Ω is a **256-regular directed graph** where:
- every node has outdegree 256 with distinct successors,
- and for each byte, the action is a permutation of Ω.

This graph structure is fully verified: every state has exactly 256 distinct successors, and every byte induces a bijection on the entire ontology.

*Test references: P1 (`test_all_b_masks_zero`, `test_unique_mask_count`), P2 (`test_each_byte_column_is_permutation`, `test_step_is_bijective_with_explicit_inverse`), P3 (`test_ontology_is_cartesian_product_of_two_256_sets`), P4 (`test_bfs_radius_two_from_archetype`), P5 (`test_depth2_decoupling_closed_form`), P6 (`test_depth2_commutes_iff_same_byte`), P7 (`test_depth4_alternation_is_identity`, `test_depth4_alternation_identity_on_all_states_for_selected_pairs`, `test_depth4_alternation_identity_all_pairs_on_archetype`), P8 (`test_trajectory_closed_form_arbitrary_length`), P9 (`test_R0xAA_is_involution_on_random_states`, `test_R0xAA_fixed_points_match_horizon_set_and_count`), P10 (`test_R0xAA_fixed_points_match_horizon_set_and_count`), P11 (`test_separator_lemma_x_then_AA_updates_A_only`, `test_separator_lemma_AA_then_x_updates_B_only`), P12 (`test_epistemology_matches_vectorized_step_for_all_states_all_bytes`), P13 (`test_row_fanout_is_256_for_all_states`).*

### 6.4 Geometric Closure and Structural Alignment

The ontology's closure at 65,536 states is not a computational accident but a geometric necessity.

**Why 256 × 256:**

The state space is the Cartesian product of two 256-element affine sets (Property P3):
- **256 Type A variations**: all possible XOR combinations of archetype A with the 256 distinct masks
- **256 Type B variations**: all possible XOR combinations of archetype B with the 256 distinct masks
- **Product**: 256 × 256 = 65,536 valid geometric transformations

**Geometric product structure:**

The 256 × 256 product structure reflects the independent variation of Type A and Type B components. This discrete structure is designed to reflect the CGM 3D spatial structure and 6 degrees of freedom (see Appendix A for the interpretive mapping to CGM theory).

**16-bit manifold in 24-bit space:**

Although the state is stored in 24 bits, the reachable manifold has size 2^16 = 65,536. Operationally, the router state carries 16 bits of structural information about the ledger prefix, expressed as two 256-element degrees of freedom (Type A choice and Type B choice). The state is a deterministic compression of the trajectory into a 16-bit affine structure (with a 24-bit carrier representation).

**Why diameter = 2:**

The radius-2 reachability (Property P4) is not a pathfinding result but a direct consequence of the transformation structure:
- After 1 step: one dimension of freedom is exercised (256 states)
- After 2 steps: both dimensions are exercised independently (256 × 256 states)
- Further steps recombine existing transformations via group relations

**Structural alignment:**

A state `s ∈ Ω` is "aligned" in the geometric sense when:
1. It is reachable from archetype (verified by ontology membership)
2. Its trajectory preserves the XOR-parity structure (Property P8)
3. It satisfies depth-4 closure (Property P7)

Misalignment occurs when a transformation violates these geometric constraints, which is structurally impossible within Ω but can be detected at the boundary when external inputs attempt to reference states outside the ontology.

---

## 7. The Atlas

The atlas is the persisted deterministic representation of the kernel's finite physics. It is built from the archetype and the transition law.

### 7.1 Ontology

File: ontology.npy  
Content: all reachable states as uint32, sorted ascending  
Expected size: 65,536 entries

### 7.2 Epistemology

File: epistemology.npy  
Shape: [N, 256] where N is ontology size  
Content: next-state indices, where epistemology[i, byte] returns the index of the next state

Epistemology is the complete deterministic dynamics of the kernel over the ontology.

**Normative constraint:** For each byte `b`, the column `epistemology[:, b]` must be a permutation of `[0, N)`. Consequently, each state has exactly one predecessor under each byte, and exactly one successor under each byte. This establishes that each byte induces a bijection on the ontology (Property P2).

### 7.3 Phenomenology

File: phenomenology.npz  
Content: fixed constants required for stepping

Phenomenology includes:

- archetype_state24
- archetype_a12
- archetype_b12
- gene_mic_s
- xform_mask_by_byte

These are the only normative arrays in phenomenology. The kernel does not include measurement scaffolding; GGG aperture is computed at the application layer (see Section 9.3).

---

## 8. Kernel Operation and Signature

A kernel instance maintains a current ontology index. For each byte:

```
state_index = epistemology[state_index, byte]
```

A routing signature is emitted on demand and must include at least:

- state_index
- state_hex (24-bit hex)
- a_hex (12-bit hex)
- b_hex (12-bit hex)

Applications may include additional observables as part of a signature, but the transition law and atlas are the stable core.

---

## 9. Replay and Audit

The system is deterministic. Given a start state and an input byte sequence, the trajectory and final state are uniquely determined. Recording the byte sequence is sufficient to reproduce the trajectory from the same start state.

**Memory and reversibility:** Given (final state, byte sequence), the full trajectory can be reconstructed backwards exactly using the explicit inverse formula (Section 5.1). However, given the final state alone, the past cannot be uniquely reconstructed, because different byte sequences can reach the same state due to group relations like `xyxy = id` (Property P7).

This supports auditability and cross-implementation verification.

### 9.1 Governance Event Ledger

GGG aperture is not a kernel-native observable. It is defined on an application-layer edge ledger y(t) ∈ ℝ⁶ over the K₄ tetrahedron.

The kernel provides the shared moment (state index and last byte) that orders events. The application layer maintains per-domain ledgers and updates them with nonsemantic GovernanceEvents.

**Domains:**
- Economy (CGM / CS stage)
- Employment (Gyroscope / UNA stage)
- Education (THM / ONA stage)

Ecology (BU) is derived from the three and is not ledger-updated.

A GovernanceEvent is a sparse update to exactly one edge coordinate of a domain ledger:

- domain ∈ {Economy, Employment, Education}
- edge_id ∈ {0..5} in canonical K₄ edge order (defined in Section 9.2)
- signed increment Δ = magnitude × confidence
- optional binding to kernel moment (state_index, last_byte) for replay/audit

**Ledger update rule (normative):**
```
y_D[edge_id] ← y_D[edge_id] + Δ
```

### 9.2 Canonical K₄ Edge Order

Vertices are ordered (Gov, Info, Infer, Intel) = (0,1,2,3).

Edges are ordered as:
- 0: (0,1) Gov–Info
- 1: (0,2) Gov–Infer
- 2: (0,3) Gov–Intel
- 3: (1,2) Info–Infer
- 4: (1,3) Info–Intel
- 5: (2,3) Infer–Intel

All ledgers y ∈ ℝ⁶ and all GovernanceEvents MUST use this edge order.

### 9.3 Hodge Decomposition on K₄

Let B be the 4×6 signed incidence matrix for K₄ in the above vertex/edge order:

```
B =
[[-1, -1, -1,  0,  0,  0],   # Gov
 [ 1,  0,  0, -1, -1,  0],   # Info
 [ 0,  1,  0,  1,  0, -1],   # Infer
 [ 0,  0,  1,  0,  1,  1]]   # Intel
```

We use the unweighted inner product (W = I₆). Event confidence is applied in the event update rule, not via a weight matrix.

**Define:**
- L = B Bᵀ  (4×4)
- P_grad = Bᵀ pinv(L) B  (6×6)  (general form)
- P_cycle = I₆ − P_grad  (6×6)

**For K₄ with W=I, closed form (audit-grade invariant):**
- P_grad = (1/4) × (BᵀB)  (exact, no pseudoinverse needed)
- P_cycle = I₆ − P_grad

Implementations MUST use this closed form to ensure deterministic, cross-platform identical results.

For any edge ledger y ∈ ℝ⁶:
- y_grad = P_grad y
- y_cycle = P_cycle y
- and y = y_grad + y_cycle

### 9.4 Aperture Definition

For each domain D ∈ {Economy, Employment, Education} with ledger y_D:

```
A_D = ||y_cycle||² / ||y||²
```

where ||v||² = vᵀ v is the standard Euclidean norm squared.  
If y = 0, define A_D = 0.

This A_D is the GGG aperture observable.

**The CGM-derived target is:**
A* = 0.0207  (canonical balance between closure and differentiation).

The router does not force A_D → A*. It makes A_D measurable, auditable, and replayable.

### 9.5 Replay Integrity for Aperture

**Kernel replay:**
Given the archetype s₀ and byte ledger b₁…b_t, all participants compute the same state s_t.

**Aperture replay:**
Given the same event log E₁…E_k (each event specifying domain, edge_id, magnitude, confidence), applied in the same order, all participants compute identical ledgers y_D and identical apertures A_D.

**Optional kernel binding:**
Events MAY record (kernel_state_index, kernel_last_byte). This binding is not required to compute aperture, but it is required for governance audit: it certifies at which shared moment an update was made.

---

## 10. Structural Displacement and Application-Layer Policy

The router provides a geometric substrate that makes displacement measurable. Application layers use the routing signature to enforce coordination policies aligned with the four CGM stages.

### 10.1 The Four Displacements (Geometric Interpretation)

**Governance Traceability Displacement (GTD)**  
A transformation claims to be a new source rather than deriving from the archetype `s₀`.  
*Kernel-native detection*: state `s ∉ Ω` (ontology membership violation)

**Information Variety Displacement (IVD)**  
**Inference Accountability Displacement (IAD)**  
**Intelligence Integrity Displacement (IID)**  

IVD, IAD, and IID are not kernel-native detections. They are application-layer diagnoses derived from:
- event provenance (THM source-type classification)
- ledger structure (y_grad vs y_cycle decomposition)
- aperture deviation from A*
- domain coupling rules (as specified by GGG)

This specification defines the measurable substrate (ledgers + aperture), not the full diagnosis policy. The kernel provides the deterministic substrate and shared moment ordering; applications implement displacement detection policies using ledger observables.

### 10.2 Application-Layer Coordination Modes

The router supports policy modes corresponding to CGM stages:

**Mode 1: Governance Management (CS)**  
Policy enforces ontology identity and ledger continuity.  
*Signature use*: verify `state_index ∈ [0, N)` and maintain append-only byte ledger

**Mode 2: Information Curation (UNA)**  
Policy preserves transformation variety and rotational degrees of freedom.  
*Signature use*: track which bytes have been applied from each state, ensure coverage of transformation space

**Mode 3: Inference Interaction (ONA)**  
Policy maintains differentiation across the 256 transformation paths.  
*Signature use*: compare state trajectories, detect when paths converge prematurely

**Mode 4: Intelligence Cooperation (BU)**  
Policy enforces closure and parity structure.  
*Signature use*: verify depth-4 alternation identity, check XOR-parity consistency via Property P8

These are application-level concerns. The kernel provides the deterministic substrate; applications implement the policy logic.

---

## 11. Conformance Requirements

A conforming implementation must satisfy all requirements below.

### 11.1 Representation

- Packing and unpacking must follow state24 = (A12 << 12) | B12 with 12-bit masking.
- Archetype must equal 0xAAA555.

### 11.2 Transcription

- For all bytes, intron = byte XOR 0xAA.

### 11.3 Expansion

- The expansion function must match the normative definition in Section 4.2.
- All 256 introns must yield distinct Type A masks.
- The Type B mask must always be zero.

### 11.4 Transition

- Type A must be mutated by XOR with the A mask prior to gyration.
- Type B must not be mutated prior to gyration.
- Gyration must set A_next = B XOR 0xFFF and B_next = A_mut XOR 0xFFF.

### 11.5 Atlas

- Ontology must contain exactly 65,536 states in the reference build.
- Ontology must be closed under all 256 byte actions.
- Epistemology must agree with direct stepping by the transition law for all (state, byte) pairs.

### 11.6 Dynamics

- The reachability diameter from the archetype must be exactly 2 (derived invariant, see Section 6.2, Property P4).
- **Depth-2 commutativity law**: For any state `s` and bytes `x,y`: `T_y(T_x(s)) = T_x(T_y(s))` iff `x=y`. Equivalently, among the `256 × 256` ordered pairs, exactly 256 commute (when `x=y`) and 65,280 do not (Property P6).
- **Byte-complete distinctness**: For any fixed current state (A, B), all 256 bytes must yield 256 distinct next states. This follows from: (1) byte → mask_a12 is injective, (2) A_next = ~B is constant across bytes, and (3) B_next = ~(A XOR mask_a12) is unique per byte since mask_a12 is unique per byte.
- **Per-byte bijection**: Each epistemology column must be a permutation (Property P2).

### 11.7 Application-Layer Conformance

A conforming GGG implementation MUST:

1. Maintain three domain ledgers y_D ∈ ℝ⁶ for D ∈ {Economy, Employment, Education}.
2. Update ledgers only by GovernanceEvents using the canonical edge order (Section 9.2):
   ```
   y_D[e] ← y_D[e] + magnitude × confidence
   ```
3. Compute P_grad and P_cycle exactly as in Section 9.3 using W = I.
4. Compute aperture A_D exactly as in Section 9.4.
5. Ensure replay integrity: same event sequence implies same ledgers and apertures.

A conforming implementation MUST NOT:
- compute "GGG aperture" from kernel state bits directly
- use a non-identity weight matrix W for aperture
- use event confidence as a second weighting mechanism beyond magnitude×confidence

---

## 12. Notes

**Disclaimer on Scope**

This specification defines the transition physics and atlas artifacts. It does not specify:

- Natural language processing or semantic interpretation.
- Policy decisions embedded in the transition function.
- Claims about cognitive abilities or consciousness.

These are application-layer concerns. The kernel provides deterministic structure that applications build upon.

**Source Type Classification**

Under The Human Mark, this kernel is Derivative in both authority and agency. It transforms and routes signals but does not originate authority or bear accountability.

---

## Appendix A. CGM Theoretical Foundation

This appendix states the CGM constitutive claims used as motivation. It is not required to implement the kernel but is relevant to the intended interpretation of chirality, noncommutativity, closure, and memory.

### A.1 CS: Common Source

Right transitions preserve the horizon; left transitions alter it.

Modal form: S implies that [R]S is equivalent to S, and [L]S is not equivalent to S.

Gyrogroup law: Left gyroassociativity where the right gyration is identity and the left gyration is not.

Router realization: The FIFO gyration implements this chirality. Type A is mutated before gyration, which corresponds to the left or altering modality. Type B is not mask-mutated before gyration, which corresponds to the right or preserving modality, though it is transformed by the gyration itself.

### A.2 UNA: Unity Non-Absolute

At depth two, order matters but not absolutely.

Modal form: S implies that it is not necessary that [L][R]S is equivalent to [R][L]S.

Gyrogroup law: Gyrocommutativity where a plus b equals gyration of a and b applied to b plus a.

Router realization: The XOR transcription introduces rotational degrees of freedom. The FIFO gyration makes composition noncommutative. Exactly: `T_y(T_x(s)) = T_x(T_y(s))` iff `x=y`. Among the `256 × 256` ordered pairs, exactly 256 commute (when `x=y`) and 65,280 do not (Property P6).

### A.3 ONA: Opposition Non-Absolute

Opposition occurs without absolute contradiction.

Modal form: S implies that it is not necessary that [L][R]S is not equivalent to [R][L]S.

Gyrogroup law: Bi-gyroassociativity where both left and right gyrations are active.

Router realization: The 256 distinct operation masks enable differentiated inference paths. Each input byte produces a unique transformation, ensuring that distinctions are preserved rather than collapsed.

**Interpretive mapping to 3D+6DoF structure:**

The discrete router structure (256 × 256 state space, depth-2 reachability, depth-4 closure) is designed to reflect CGM's 3D spatial structure and 6 degrees of freedom:
- **3 rotational DOF** (interpretive mapping from UNA): non-commutativity at depth-2 (Property P6) reflects rotational degrees of freedom
- **3 translational DOF** (interpretive mapping from ONA): 256 distinct differentiation paths reflect translational degrees of freedom
- **Closure constraint** (interpretive mapping from BU-Egress): depth-4 alternation returns to identity (Property P7)

This is an interpretive mapping from the discrete kernel structure to CGM theory. The kernel's verified properties (P1-P9) are exact invariants; the 3D+6DoF interpretation is a theoretical motivation, not a verified geometric property of the discrete system itself.

### A.4 BU-Egress: Balance Universal, Absorption Phase

Depth-four closure achieves coherent measurement.

Modal form: S implies it is necessary that the alternating depth-four words converge.

Router realization: The finite closed state space (ontology) of 65,536 states represents the absorptive orbit. Every trajectory remains within this orbit regardless of input sequence. This is a discrete closure substrate; depth-4 alternation identity (Property P7) is the exact kernel analogue verified here.

### A.5 BU-Ingress: Balance Universal, Memory Phase

The balanced state reconstructs prior distinctions.

Modal form: if closure holds, then CS, UNA, and ONA distinctions are recoverable at the level of traceability.

Router realization: Deterministic replay from the byte ledger reconstructs the full trajectory. The compact state is a summary. The boundary record preserves the history needed for reconstruction.

---

## Appendix B. Non-normative Numerics and Historical Notes

The numbers 256 and 65,536 emerge from the byte interface and the transition law structure. They also coincide with widely used scales in computing, which makes the system practical to enumerate, store, and test.

**256 = 2^8**
- Also equals 4^4 and ((2^2)^2)^2
- The latter form is called zenzizenzizenzic, meaning the square of the square of the square
- This recursive self-squaring structure mirrors the recursive depth structure of CGM

**65,536 = 2^16**
- A superperfect number: applying the sum-of-divisors function twice yields exactly twice the original number
- Specifically, the divisor sum of 65,536 is 131,071 (the sixth Mersenne prime)
- The divisor sum of 131,071 is 131,072, which equals 2 × 65,536

These properties are not the reason for choosing these numbers. The numbers emerge from the architecture. The fact that the emergent state space coincides with a superperfect number is a notable structural alignment with the recursive and self-referential character of CGM, but this is a historical observation, not a design requirement.

---

## Appendix C. Application-Layer GGG Aperture

GGG aperture is defined at the application layer, not in the kernel. This appendix specifies the normative definition.

### C.1 Scope

The kernel provides:
- Deterministic state transitions
- Shared moment ordering (state_index, last_byte)
- Replayable byte ledger

The application layer provides:
- Domain edge ledgers y_D ∈ ℝ⁶
- GovernanceEvent processing
- Hodge decomposition and aperture computation

### C.2 Normative Definition

GGG aperture is computed exactly as specified in Sections 9.3 and 9.4:
- Incidence matrix B (4×6) for K₄
- Unweighted projections: P_grad = Bᵀ pinv(B Bᵀ) B, P_cycle = I₆ − P_grad
- Aperture: A_D = ||y_cycle||² / ||y||²

This matches the GGG simulator export with W = I. Confidence is encoded in GovernanceEvent.signed_value() = magnitude × confidence, not via a weight matrix.

### C.3 Implementation Requirements

A conforming GGG implementation must:
- Use the canonical K₄ edge order (Section 9.2)
- Compute projections using W = I only
- Apply confidence via event updates, not via W
- Ensure deterministic replay (Section 9.5)

See Section 11.7 for complete conformance requirements.