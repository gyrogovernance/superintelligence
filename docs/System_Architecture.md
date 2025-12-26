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

The router kernel provides a computational substrate that makes these principles measurable and enforceable. It does not interpret content or make policy decisions. It transforms bytes into states according to a minimal transition law designed to preserve structural properties required by the Common Governance Model.

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

## 2. Archetype

The system defines a canonical archetype state used as the seed for ontology construction and as a stable reference for validation:

```
ARCHETYPE_A12 = 0xAAA
ARCHETYPE_B12 = 0x555
ARCHETYPE_STATE24 = 0xAAA555
```

These are exact 12-bit complements. The alternating pattern establishes maximal contrast between A and B at initialization.

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

This function is a convention that satisfies the required properties. If future CGM constraints uniquely determine an expansion, this specification will be updated accordingly.

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

The following properties are proven by the test suite and hold as exact invariants of the kernel physics.

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

*Test references: P1 (`test_all_b_masks_zero`, `test_unique_mask_count`), P2 (`test_each_byte_column_is_permutation`, `test_step_is_bijective_with_explicit_inverse`), P3 (`test_ontology_is_cartesian_product_of_two_256_sets`), P4 (`test_bfs_radius_two_from_archetype`), P5 (`test_depth2_decoupling_closed_form`), P6 (`test_depth2_commutes_iff_same_byte`), P7 (`test_depth4_alternation_is_identity`, `test_depth4_alternation_identity_on_all_states_for_selected_pairs`), P8 (`test_trajectory_closed_form_arbitrary_length`).*

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
Content: fixed constants required for stepping and for measurement scaffolding

At minimum, phenomenology must include:

- archetype_state24
- archetype_a12
- archetype_b12
- gene_mic_s
- xform_mask_by_byte

Measurement-specific arrays may be included but are not normative unless explicitly specified in a measurement appendix.

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

---

## 10. Routing Modes and Displacement Prevention

The router is intended to be used as a coordination primitive in higher layers. A common framing is to align policy modes to the CGM stages and to monitor displacement risks:

- **Governance Traceability Displacement**: loss of provenance and replay.
- **Information Variety Displacement**: collapse of distinguishability under transformations.
- **Inference Accountability Displacement**: inability to relate outcomes to inputs.
- **Intelligence Integrity Displacement**: failure to maintain stable closure properties across time.

These are application-level concerns. The kernel supports them by providing a closed deterministic substrate that can be audited and measured.

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

### A.4 BU-Egress: Balance Universal, Absorption Phase

Depth-four closure achieves coherent measurement.

Modal form: S implies it is necessary that the alternating depth-four words converge.

Router realization: The finite closed state space (ontology) of 65,536 states represents the absorptive orbit. Every trajectory remains within this orbit regardless of input sequence.

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

## Appendix C. Measurement Layer

The current implementation includes a measurement observable computed from a K4 graph-based correlation structure. This value is currently named **"K4 correlation cyclicity"** (not "GGG aperture") and is under active development.

The underlying theoretical intent is to measure balance and closure properties derived from CGM, but the specific graph construction and measurement definitions are expected to change.

**Current diagnostic statistics** (from test runs, not normative targets):
- Sample aperture range: approximately 0.12 to 0.64
- Mean: approximately 0.20
- Standard deviation: approximately 0.10

For this reason, measurement observables are not specified as a stable part of the kernel in this document. The normative scope of this specification is the transition law, atlas artifacts, and conformance requirements. Measurement definitions will be specified in a future revision once stabilized.