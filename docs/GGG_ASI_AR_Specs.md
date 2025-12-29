# GGG ASI Alignment Router
## Kernel Specification

This document is the normative technical specification of the GGG ASI Alignment Router. It includes the kernel physics, atlas artifacts, replay rules, and the application-layer governance measurement substrate.

Normative terms MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are interpreted as requirement keywords for conformance.

---

# 1. Constitutional Frame and ASI Purpose

## 1.1 Intelligence as Structural Coherence

Traditional artificial intelligence approaches intelligence as a statistical optimization problem. These systems achieve surface-level fluency by superimposing correlations within high-dimensional abstract spaces through massive datasets. Because such architectures lacks internal structural constraints, coherence and ethics are typically treated as post-hoc semantic overlays or external filters.

The GGG ASI Alignment Router represents a different paradigm. It treats intelligence as an intrinsic structural property that emerges from the recursive alignment of operational operations. Grounded in the Common Governance Model (CGM), the framework demonstrates how intelligence emerges naturally from the self-referential dynamics of structured space. Rather than approximating a target function, the router navigates a provably finite and fully discovered state space where alignment is constitutive. Coherence is not a policy choice but a requirement of the internal physics of the state space.

## 1.2 What the Router Is

GGG ASI Alignment Router is a multi-domain network coordination algorithm for focused and well-distributed interventions. As a collective superintelligence implementation, it is a network of humans and machines (Superintelligence, Bostrom 2014) amplifying rather than outperforming single agent potential, while preserving the constitutive conditions of governance and intelligibility (Gyroscopic Global Governance, Korompilias 2025).

The algorithm is a deterministic finite-state coordination kernel. It maps a byte sequence to a reproducible state trajectory on a finite closed state space. A kernel instance starts from a universal reference state called the archetype and applies a fixed transition rule to each byte in an append-only ledger. This process results in a deterministic state at each step and a compact routing signature.

The kernel is designed to support governance-grade coordination across the GGG domains: Economy, Employment, Education, and Ecology. It does not interpret the meaning of the input bytes. Instead, it performs structural transformations that make the results reproducible, comparable, and auditable.

## 1.3 ASI Definition in This System

Artificial Superintelligence (ASI) in this framework is a property of a human–AI governance regime. It is not an attribute of an autonomous agent. ASI refers to the regime in which human and artificial systems jointly sustain four constitutive governance principles while operating at a specific structural balance point termed the canonical aperture.

The four principles are:

*   **Governance Traceability:** Authority remains traceable to authentic human sources. Artificial systems contribute derivative coordination but do not originate governance.
*   **Information Variety:** Diverse authentic sources remain distinguishable. Derivative summaries and aggregations must not collapse the variety into a uniform narrative.
*   **Inference Accountability:** Responsibility for decisions remains with accountable human agency. Artificial inference serves as a mechanism within human accountability.
*   **Intelligence Integrity:** Coherence is maintained across time and context. Decisions remain consistent with the governing structure that produced them.

The router is the substrate that makes coordination structurally reproducible. This reproducibility is the prerequisite for governance that can scale without collapsing into entity-based trust chains.

## 1.4 Shared Moments

Shared moments are the central coordination primitive of the kernel. A shared moment occurs when participants who possess the same ledger prefix compute the identical kernel state at the identical step. This provides a shared "now" as a configuration derived from the ledger history.

This primitive replaces three fragile coordination patterns:
1.  **Coordination by asserted time:** Reliance on timestamps or UTC ordering.
2.  **Coordination by asserted identity:** Reliance on trusted signers or specific authorities.
3.  **Coordination by private state:** Reliance on model-internal hidden vectors or proprietary logs.

Shared moments coordinate through reproducible computation. Participants do not need to share a privileged identity; they only need to share the relevant ledger bytes and the kernel specification. If participants claim the same ledger prefix but compute different states, their implementations or ledgers are provably different.

## 1.5 Geometric Provenance

The second central primitive is geometric provenance. The kernel defines a finite set of valid reachable states called the ontology (Ω). A state possesses geometric provenance if and only if it belongs to this set.

The router makes claims about the origin of a state structurally checkable. If a presented state belongs to Ω, it is a valid transformation of the archetype under the kernel physics. If it does not belong to Ω, it is not a valid router state. 

This creates a clear separation of layers. The kernel-native layer verifies ontology membership, deterministic replay, and divergence detection. Authorization and accountability remain application-layer responsibilities under authentic human agency. The kernel provides a common structural basis for policy enforcement but does not decide policy itself.

## 1.6 Why This Achieves ASI

The router provides the missing substrate for multi-domain coordination required for the ASI regime. It achieves this through:
*   **Entity-agnostic verification:** Validity is checked by structure rather than a privileged entity.
*   **Deterministic coordination:** The shared moment enables reliable coordination across independent institutions and systems.
*   **Replayable audit:** Trajectories can be reconstructed from the ledger to enable governance-grade auditing.
*   **Constitutional Observables:** Participants compute low-dimensional observables that remain stable under replay and independent of identity claims. These depend only on the fixed-width state representation and the transition rule.

## 1.7 Design Requirements

A conforming kernel satisfies the following structural requirements:
*   **Finite:** The kernel operates on a closed set of reachable states.
*   **Deterministic:** The same archetype and byte ledger always produce the same state trajectory.
*   **Byte-complete:** Every byte value from 0 to 255 is a valid input instruction.
*   **Nonsemantic:** The kernel does not parse language or apply policy. It transforms bytes structurally.
*   **Portable:** The transition rule is defined using fixed-width bit operations such as XOR, shifts, and masking.
*   **Auditable:** Trajectories and signatures can be reproduced and verified by any independent party.

---

# 2. Kernel Physics

## 2.1 State Model

### 2.1.1 The 24-bit state

The internal state is a 24-bit value composed of two 12-bit components:

- Type A: the top 12 bits (active phase)
- Type B: the bottom 12 bits (passive phase)

Packing and unpacking:

```python
state24 = (A12 << 12) | B12
A12 = (state24 >> 12) & 0xFFF
B12 = state24 & 0xFFF
```

**Bit indexing convention (normative):**

In this specification, bit `k` of a 12-bit component is defined as `(word >> k) & 1`; bit 0 is the least significant bit and bit 11 is the most significant bit.

This two-component form is essential. The transition rule treats A and B asymmetrically, which is how the kernel realizes chirality as a structural feature of the physics.

### 2.1.2 Dual-frame geometry

Each 12-bit component is interpreted as a 2×3×2 binary grid:
- 2 frames
- 3 rows per frame
- 2 columns per row

The 2×3×2 geometry is an index mapping over the bit positions 0..11, where bit extraction follows the convention defined in §2.1.1.

**Coordinate mapping:**

The 2×3×2 geometry maps to bit positions as follows:
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

This mapping is fixed and MUST be used consistently for mask expansion, archetype definition, and all geometric operations.

This geometry is the shared substrate for:
- the archetype
- the expansion masks
- structural observables (such as horizon distance)

## 2.2 The Archetype

### 2.2.1 Definition

The archetype is the universal reference state from which all router-valid states derive:

```python
ARCHETYPE_A12 = 0xAAA
ARCHETYPE_B12 = 0x555
ARCHETYPE_STATE24 = 0xAAA555
```

### 2.2.2 Structural properties

The archetype has three structural properties that are used throughout the kernel:

1. Complement relation  
   `ARCHETYPE_A12 XOR 0xFFF = ARCHETYPE_B12`

2. Maximal symmetry  
   The alternating pattern distributes bits evenly across the dual-frame geometry.

3. Universal reference role  
   All ontology states are reachable from the archetype under the kernel transition physics.

### 2.2.3 Common source role

The archetype is a common source in the strict operational sense: it is a universal reference point in state space, not an entity or institution. The kernel does not embed ownership. Participants adopt the archetype as the reference so that their computations share a common origin.

## 2.2.4 Canonical Derived Observables

From the 24-bit state `s_t = (A12, B12)`, the following observables are defined as deterministic functions. These are the canonical constitutional observables exported by the kernel.

**1. Raw state components:**
```python
A12 = (state24 >> 12) & 0xFFF
B12 = state24 & 0xFFF
```

**2. Hamming distance to archetype:**
```python
archetype_distance = popcount(s_t ^ ARCHETYPE_STATE24)
```
where `popcount(x)` returns the number of set bits in `x`.

**3. Horizon distance:**
Define the horizon set `H = {(a,b): a = (b ^ 0xFFF)}`. Then:
```python
horizon_distance = popcount(A12 ^ (B12 ^ 0xFFF))
```

**4. A/B Hamming distance:**
```python
ab_distance = popcount(A12 ^ B12)
```

**5. Component densities:**
```python
a_density = popcount(A12) / 12.0
b_density = popcount(B12) / 12.0
```

These observables are constitutional in that they are defined solely from the kernel's fixed-width state representation and transition law. They are exact, replayable from the ledger, and do not depend on model internals or asserted identity claims.

## 2.3 Byte Interface and Transcription

### 2.3.1 Byte-complete input alphabet

The kernel input alphabet is the full set of bytes:

- 0..255 inclusive

There are no reserved bytes and no invalid bytes.

### 2.3.2 Transcription constant

The kernel uses a fixed constant:

```python
GENE_MIC_S = 0xAA
```

Given an input byte `byte`, the kernel computes an intron:

```python
intron = byte ^ 0xAA
```

This XOR mapping is a bijection on 8-bit values. Every input byte maps to exactly one intron and every intron corresponds to exactly one byte.

The constant fixes a convention for mapping external bytes into the internal action space. Conforming implementations must use the same constant.

## 2.4 The Reference Byte and the Horizon

### 2.4.1 The reference byte

Byte `0xAA` is structurally special:

`0xAA ^ 0xAA = 0x00`

So it produces intron `0x00`. Under the canonical expansion, intron `0x00` produces the zero mutation mask. This makes `0xAA` the reference action.

Define the operator:

`R = T_0xAA`

where `T_b` denotes “apply one transition step using byte b”.

### 2.4.2 Involution property

`R` is an involution, meaning applying it twice returns the original:

`R(R(s)) = s`

This is a structural property, not an interpretive label.

### 2.4.3 Horizon set

Within the ontology Ω, the fixed points of `R` define the horizon set. A state lies on the horizon when applying the reference action leaves it unchanged:

`R(s) = s`

In the kernel’s representation, horizon states satisfy a simple relation between the two components:

`A12 = (B12 XOR 0xFFF)`

The horizon set contains exactly 256 ontology states.

## 2.5 Expansion and Operation Masks

Each intron expands to a 12-bit mutation mask that is applied only to Type A. Type B does not receive a direct mutation mask.

### 2.5.1 Expansion requirements

A conforming expansion function satisfies:

- determinism: same intron yields same mask
- injectivity: all 256 introns yield 256 distinct Type A masks
- type separation: Type B mask is always zero

Injectivity is essential. If different introns produced the same mask, distinct bytes would become indistinguishable at the level of kernel dynamics, reducing the transformation space and weakening traceability.

### 2.5.2 Canonical expansion function

Let `x` be the 8-bit intron. Define:

```python
frame0_a = x & 0x3F
frame1_a = ((x >> 6) | ((x & 0x0F) << 2)) & 0x3F
mask_a12 = frame0_a | (frame1_a << 6)
```

Packed mask representation:

```python
mask24 = (mask_a12 << 12) | 0x000
```

The low 12 bits are always zero.

This expansion function is normative for this specification version. If it changes, all atlas artifacts must be rebuilt and all certified invariants must be re-verified.

### 2.5.3 Precomputed mask table

Implementations can precompute:

```python
XFORM_MASK_BY_BYTE[byte] = expand_intron_to_mask24(byte ^ 0xAA)
```

The table has exactly 256 entries.

## 2.6 Transition Law

The transition law specifies the single-step update from `(A12, B12)` under a byte input.

### 2.6.1 Forward transition

Given current state `(A12, B12)` and input byte:

1. Compute intron: `intron = byte ^ 0xAA`
2. Compute `mask_a12` from intron (canonical expansion)
3. Mutate Type A only: `A12_mut = A12 ^ mask_a12`
4. Apply FIFO gyration with complement:
   - `A12_next = B12 ^ 0xFFF`
   - `B12_next = A12_mut ^ 0xFFF`
5. Pack next state:
   - `state24_next = (A12_next << 12) | B12_next`

### 2.6.2 Inverse transition

Given byte `b` with mask `m_b` and next state `(A', B')`, the unique predecessor `(A, B)` is:

```python
B = A' ^ 0xFFF
A = (B' ^ m_b) ^ 0xFFF
```

This guarantees that each per-byte transition is bijective over the ontology.

### 2.6.3 Conjugation form of the inverse

Let `R = T_0xAA`. For any byte `x`:

`T_x^{-1} = R ∘ T_x ∘ R`

This expresses reversal using the same byte action alphabet, with no external inversion operator.

## 2.7 Minimality and Structural Rationale

This kernel physics is minimal in three senses: computational, structural, and governance-relevant.

### 2.7.1 Computational minimality
Only fixed-width operations are used:
- XOR
- shifts
- masking

This makes the kernel portable, fast, and suitable for exhaustive verification.

### 2.7.2 Structural minimality
Only one component receives the mutation mask before the gyration step. That asymmetry is the smallest mechanism that introduces chirality in the dynamics while keeping the system reversible.

### 2.7.3 Governance relevance
The two-component structure and deterministic update rule are what allow shared moments:
- same bytes imply same state
- disagreement is detectable
- audit can be performed by replay

The kernel itself does not assign authority. It supplies the common structural object on which authentic human agency can base accountable decisions.

# 3. Ontology, Closure, and Verified Invariants

This section specifies the kernel’s reachable state space (the ontology), explains why it closes exactly, and lists the certified algebraic properties that define the kernel physics operationally.

## 3.1 Reachable State Space

The kernel uses a 24-bit carrier state, so the theoretical state space contains `2^24` possible 24-bit values. The kernel does not range over that full space when initialized from the archetype and stepped with the specified transition law.

### 3.1.1 Ontology definition

The ontology Ω is the set of all states reachable from the archetype under the 256 byte actions.

In this specification, Ω has exactly:

- `65,536` reachable states, which equals `2^16`

Ω is a strict subset of the 24-bit carrier space.

### 3.1.2 Reachability radius

A state’s distance is the smallest number of bytes required to reach it from the archetype.

For this kernel:

- every state in Ω is reachable from the archetype in at most 2 steps
- after 1 step exactly 256 states are reachable
- after 2 steps exactly 65,536 states are reachable, which is all of Ω

This implies that the directed graph induced by byte actions has reachability radius 2 from the archetype and that Ω closes after depth 2.

## 3.2 Closure Derivation

The closure to exactly 65,536 states and the “no new states after step 2” result follow directly from the transition law (§2.6) and injective mask expansion (§2.5).

### 3.2.1 One step from archetype

Let the archetype be `(A0, B0) = (0xAAA, 0x555)`.

For any byte with Type A mask `M`:

- `A1 = B0 ^ 0xFFF = 0x555 ^ 0xFFF = 0xAAA`  
  so `A1` is constant for all bytes at step 1
- `B1 = (A0 ^ M) ^ 0xFFF = (0xAAA ^ M) ^ 0xFFF = 0x555 ^ M`  
  so `B1` depends only on `M`

Because the expansion is injective, the 256 bytes yield 256 distinct masks M and therefore 256 distinct values of B1.

So exactly 256 distinct states are reachable in one step.

### 3.2.2 Two steps from archetype

From any one-step state `(A1, B1)`:

- `A2 = B1 ^ 0xFFF`  
  so `A2` can take 256 values as `B1` ranges over 256 distinct values
- `B2 = (A1 ^ M2) ^ 0xFFF`  
  and `A1 = 0xAAA` is constant across all one-step states, so `B2` depends only on `M2` and takes 256 values

Because `A2` depends on the previous step’s mask and `B2` depends only on the current mask, the choices are independent at depth 2.

So exactly:

- `256 × 256 = 65,536`

states are reachable in two steps.

### 3.2.3 Closure after depth 2

At any step `k ≥ 2`, the state components always take values from the same two 256-element sets determined by the masks. The dynamics recombine existing values but do not create new ones beyond those already present at depth 2.

Therefore Ω closes after depth 2.

## 3.3 Ontology Structure

Ω has a product structure. It equals the Cartesian product of two 256-element sets embedded into the 24-bit carrier representation.

### 3.3.1 Product form

Define the 12-bit mask set:

- `M_set = { m_b : b ∈ [0,255] }`

where `m_b` is the 12-bit Type A mask associated to byte b (through transcription then expansion).

Define:

- `A_set = { ARCHETYPE_A12 ^ m : m ∈ M_set }`
- `B_set = { ARCHETYPE_B12 ^ m : m ∈ M_set }`

Then the ontology is:

```
Ω = A_set × B_set
```

packed into 24 bits as `(A12 << 12) | B12`.

### 3.3.2 Effective dimensionality

Although state is stored in 24 bits, Ω contains exactly `2^16` states. Operationally, the kernel state carries 16 bits of structural information about the ledger prefix, expressed as a pair of 256-way choices.

This is not a compression scheme chosen for efficiency. It is a consequence of the kernel physics and is essential for full enumeration, atlas compilation, and exhaustive verification.

## 3.4 Certified Algebraic Properties

The following properties are certified by the test suite and constitute the kernel physics in operational form. Test suite details (size, runtime, coverage) are provided in Appendix B.

### P1. Mask separation
- Type B mask is always zero for every byte
- exactly 256 distinct Type A masks exist
- exactly 1 of the 256 Type A masks is zero; 255 are nonzero

### P2. Per-byte bijection
For every byte `b`, the transition `T_b: Ω → Ω` is bijective.

Equivalent operational forms:
- each epistemology column is a permutation of `[0, N)`
- each state has exactly one predecessor and exactly one successor under each byte

### P3. Exact ontology characterization
Ω equals the Cartesian product of two 256-element sets:

```
Ω = A_set × B_set
```

packed into 24 bits, with `A_set` and `B_set` generated from the archetype and the 256 masks.

### P4. Radius-2 reachability from archetype
Every ontology state is reachable from the archetype in at most 2 bytes.
- after one byte: exactly 256 distinct states
- after two bytes: exactly 65,536 states (all of Ω)

### P5. Depth-2 closed-form composition
For any start state `(A,B)` and bytes `x,y` with Type A masks `m_x, m_y`:

```
T_y(T_x(A,B)) = (A ^ m_x, B ^ m_y)
```

After two steps, A depends only on the first byte mask and B only on the second byte mask.

### P6. Depth-2 commutation law
For any state `s` and bytes `x,y`:

```
T_y(T_x(s)) = T_x(T_y(s))   iff   x = y
```

So depth-2 is non-commutative for every unequal pair.

Counting result:
- among `256 × 256` ordered pairs, exactly 256 commute and 65,280 do not

Expected rates under uniform random byte pairs:
- commutativity: `1/256 ≈ 0.39%`
- non-commutativity: `255/256 ≈ 99.61%`

### P7. Depth-4 alternation identity
For any state `s` and bytes `x,y`:

```
T_y(T_x(T_y(T_x(s)))) = s
```

Equivalently, the words `xyxy` and `yxyx` both act as identity on Ω.

This is the discrete BU-Egress analogue: depth-4 alternating words return to identity.

### P8. Trajectory closed form for arbitrary-length words
For a byte sequence `b_1 … b_n` with masks `m_i`, define:

- `O = m_1 ^ m_3 ^ m_5 ^ …` (odd positions)
- `E = m_2 ^ m_4 ^ m_6 ^ …` (even positions)
- `~X = X ^ 0xFFF`

Then:

If `n` is even:
- `(A_n, B_n) = (A_0 ^ O, B_0 ^ E)`

If `n` is odd:
- `(A_n, B_n) = (~B_0 ^ E, ~A_0 ^ O)`

The final state depends only on XOR parity within odd and even positions, not on the order within each parity class.

### P9. The reference operator and its fixed points
Let `R = T_0xAA` (the transition operator for byte 0xAA).

- `R` is an involution on all 24-bit states: `R(R(s)) = s`
- within Ω, `R(s) = s` iff `A12 = (B12 ^ 0xFFF)`

### P10. Horizon set cardinality
Within Ω, the fixed-point set of `R` contains exactly 256 states.

### P11. Separator lemmas
For any state `(A,B)` and any byte `x` with mask `m_x`:

- after `x` then `0xAA`:
  - `T_AA(T_x(A,B)) = (A ^ m_x, B)`
- after `0xAA` then `x`:
  - `T_x(T_AA(A,B)) = (A, B ^ m_x)`

Operational meaning: inserting byte 0xAA adjacent to a byte directs the mask effect into A or B at depth 2, using only byte actions.

### P12. Atlas exactness
For all `i ∈ [0, N)` and all `byte ∈ [0,255]`:

- `epistemology[i, byte]` equals the ontology index of `step_state_by_byte(ontology[i], byte)`

This is verified exhaustively for all `65,536 × 256 = 16,777,216` state-byte pairs.

### P13. Full 256-way fanout per state
For every `s ∈ Ω`, the set `{ T_b(s) : b ∈ [0,255] }` has size exactly 256.

Combined with P2, this implies the transition system on Ω is a 256-regular directed graph:
- each node has outdegree 256 with distinct successors
- for each byte, the action is a permutation of Ω

### Test references

P1: `test_all_b_masks_zero`, `test_unique_mask_count`  
P2: `test_each_byte_column_is_permutation`, `test_step_is_bijective_with_explicit_inverse`  
P3: `test_ontology_is_cartesian_product_of_two_256_sets`  
P4: `test_bfs_radius_two_from_archetype`  
P5: `test_depth2_decoupling_closed_form`  
P6: `test_depth2_commutes_iff_same_byte`  
P7: `test_depth4_alternation_is_identity`, `test_depth4_alternation_identity_on_all_states_for_selected_pairs`, `test_depth4_alternation_identity_all_pairs_on_archetype`  
P8: `test_trajectory_closed_form_arbitrary_length`  
P9: `test_R0xAA_is_involution_on_random_states`, `test_R0xAA_fixed_points_match_horizon_set_and_count`  
P10: `test_R0xAA_fixed_points_match_horizon_set_and_count`  
P11: `test_separator_lemma_x_then_AA_updates_A_only`, `test_separator_lemma_AA_then_x_updates_B_only`  
P12: `test_epistemology_matches_vectorized_step_for_all_states_all_bytes`  
P13: `test_row_fanout_is_256_for_all_states`

## 3.5 Structural Validity at the Boundary

Within Ω, all kernel actions preserve the kernel invariants listed above. The boundary case is when an external system presents a state that is not in Ω, or when participants claim the same shared moment but hold different ledgers or implementations.

Kernel-native checks are those that any participant can perform directly from the specification and shared artifacts:
- ontology membership: whether a state belongs to Ω
- deterministic replay: whether the same ledger prefix reproduces the same trajectory
- divergence detection: whether participants claiming the same prefix compute different states

Application-layer diagnostics use additional governance data (events, domain ledgers) and are specified in Section 4. See §4.11 for application-layer displacement diagnosis.

---

# 4. Operational Runtime and Governance Measurement Substrate

This section specifies how the router operates as a complete coordination system in practice. It defines the atlas artifacts that compile kernel physics, the runtime stepping and replay procedures, the governance measurement substrate built from domain ledgers and aperture, and the orchestration and connector surfaces through which events enter the system.

This section does not embed policy decisions into kernel physics. Policy enters through application-layer event production and remains accountable to authentic human agency.

## 4.1 Atlas Artifacts

The atlas is the persisted deterministic representation of the kernel’s finite physics. It is built from the archetype and the transition law.

### 4.1.1 Ontology artifact

File: `ontology.npy`  
Content: all reachable states as `uint32`, sorted ascending  
Expected size: 65,536 entries

The ontology artifact is the canonical representation of Ω. Conforming implementations MUST treat `ontology.npy` as authoritative for membership checks and indexing conventions.

Ontology membership verification MUST be performed by checking whether a presented 24-bit state appears in `ontology.npy`, or in an equivalent representation that is provably identical to `ontology.npy`.

Notes on permitted atlas construction procedures are provided in Appendix E.

### 4.1.2 Epistemology artifact

File: `epistemology.npy`  
Shape: `[N, 256]` where `N = 65,536`  
Content: next-state indices

Operational meaning:
- `epistemology[i, byte]` returns the index of the next state after applying `byte` to ontology state `i`

Normative constraints:
- `epistemology` MUST be consistent with the kernel transition law for all `(state, byte)` pairs
- for each byte `b`, the column `epistemology[:, b]` MUST be a permutation of `[0, N)`

Notes on permitted epistemology construction procedures are provided in Appendix E.

### 4.1.3 Phenomenology artifact

File: `phenomenology.npz`  
Content: constants required for stepping

Phenomenology includes these normative arrays and constants:
- `archetype_state24`
- `archetype_a12`
- `archetype_b12`
- `gene_mic_s`
- `xform_mask_by_byte`

The kernel does not include governance measurement scaffolding. Governance measurement is application-layer and specified in Sections 4.4–4.11.

## 4.2 Kernel Runtime and Routing Signature

A kernel instance maintains a current ontology index. Each input byte advances the kernel by one deterministic step:

```python
state_index = epistemology[state_index, byte]
```

A routing signature is emitted on demand and MUST include at minimum:
- `state_index`
- `state_hex` (24-bit hex)
- `a_hex` (12-bit hex)
- `b_hex` (12-bit hex)

A conforming implementation MUST provide an ontology membership verification procedure grounded in the ontology artifact. That procedure MAY be implemented as a membership check against `ontology.npy` or an equivalent representation, and it MUST be mathematically equivalent to that check.

The canonical derived observables defined in §2.2.4 are computable deterministically from the exported state. A conforming implementation MUST support computing those observables from the exported state and the normative constants.

## 4.3 Replay and Audit

The kernel is deterministic. Given the same archetype and the same byte ledger, every conforming implementation computes the same state trajectory.

### 4.3.1 Forward replay

Given:
- start state `s_0` (the archetype)
- ledger bytes `b_1…b_t`

any participant computes `s_t` by repeated stepping.

### 4.3.2 Backward reconstruction

Given:
- final state `(A', B')`
- byte `b` with mask `m_b`

the unique predecessor `(A, B)` is computed by the inverse rule (§2.6.2):

```python
B = A' ^ 0xFFF
A = (B' ^ m_b) ^ 0xFFF
```

Given the final state and the full byte sequence, the full trajectory can be reconstructed backwards exactly.

### 4.3.3 Non-uniqueness of history from final state alone

Given final state alone, the past is not uniquely determined. Different byte sequences can reach the same state due to group relations such as the depth-4 identity `xyxy = id` (Property P7).

This is not a deficiency. The ledger is the record of governance events. The state is a shared observable for coordination, not a unique identifier of history.

## 4.4 Governance Event Model

The kernel provides shared moments. The application layer attaches governance meaning through domain ledgers updated by governance events.

### 4.4.1 Domain ledgers

The application layer maintains three domain ledgers:

- Economy ledger: `y_Econ ∈ ℝ^6`
- Employment ledger: `y_Emp ∈ ℝ^6`
- Education ledger: `y_Edu ∈ ℝ^6`

**Ecology domain:**

Ecology is conceptually part of the four-domain GGG framework but is not ledger-updated directly in this specification. Ecology outputs MAY be derived from the three domain ledgers using application-layer computation. The specification of ecology derivation is out of scope for this version and remains an application-layer policy choice.

### 4.4.2 GovernanceEvent

A GovernanceEvent is a sparse update to exactly one edge coordinate of one domain ledger. It contains:
- `domain ∈ {Economy, Employment, Education}`
- `edge_id ∈ {0..5}` (canonical K₄ edge order)
- signed increment `Δ = magnitude × confidence`
- optional binding to a shared moment: `(state_index, last_byte)` for audit

Normative update rule:

```python
y_D[edge_id] = y_D[edge_id] + Δ
```

The kernel does not interpret events. Events are application-layer records that remain accountable to authentic human agency under THM.

### 4.4.3 Event binding to kernel moments

By default, events SHOULD be bound to the current kernel moment when applied. This binding records:
- `kernel_state_index`: the current ontology index
- `kernel_last_byte`: the last byte that advanced the kernel

This binding enables audit and replay verification but is not required for aperture computation. Applications MAY choose to apply events without kernel binding for specific use cases.

## 4.5 Coordinator Component

The Coordinator is the orchestration layer that combines kernel stepping, domain ledgers, and event processing into a unified operational workflow.

### 4.5.1 Coordinator responsibilities

A Coordinator instance:
- owns a kernel instance and maintains domain ledgers
- maintains audit logs: `byte_log` (sequence of bytes applied) and `event_log` (sequence of GovernanceEvents applied)
- provides the operational workflow: `step_byte(s)`, `apply_event`, `get_status`, `reset`
- enforces default policy for event binding to kernel moments

### 4.5.2 Audit logs

The Coordinator maintains two append-only audit logs:

**Byte log:**
- Records the sequence of bytes applied to advance the kernel
- Enables deterministic replay of kernel state trajectory
- Format: ordered list of byte values `[0..255]`

**Event log:**
- Records the sequence of GovernanceEvents applied to domain ledgers
- Each entry includes:
  - event index (position in log)
  - kernel binding (`kernel_state_index`, `kernel_last_byte`) if present
  - complete event data (domain, edge_id, magnitude, confidence, metadata)
- Enables deterministic replay of ledger state and aperture

Both logs are append-only and MUST preserve ordering. Implementations MAY persist logs externally for durability.

### 4.5.3 Status reporting

A Coordinator MUST provide a status report that includes at minimum:
- Kernel signature fields (`state_index`, `state_hex`, `a_hex`, `b_hex`)
- `last_byte` (the last byte that advanced the kernel)
- `byte_log_len` (length of byte log)
- `event_log_len` (length of event log)
- Current domain ledgers (`y_Econ`, `y_Emp`, `y_Edu`)
- Current apertures (`A_Econ`, `A_Emp`, `A_Edu`)
- `event_count` (total events applied to ledgers)

The status report enables external systems to query the current state of the coordination substrate.

## 4.6 Plugin Architecture

The application layer uses a plugin architecture to convert domain-specific signals into GovernanceEvents. This keeps edge mappings explicit, auditable, and editable.

### 4.6.1 Plugin interface

A plugin is a component that:
- accepts domain-specific payloads (e.g., THM displacement signals, work-mix metrics)
- converts them deterministically into zero or more GovernanceEvents
- maintains explicit, auditable mappings from signals to edge updates

**Minimal plugin interface:**

A conforming plugin MUST:
- implement an `emit_events(payload, context)` method that returns a list of GovernanceEvents
- be deterministic: the same payload MUST produce the same events
- record its identity and mapping policy in event metadata for audit

Plugins are application-layer components. The kernel does not interpret plugin outputs; it only processes GovernanceEvents.

### 4.6.2 Edge mapping policy

Edge mappings (which signals affect which K₄ edges) are explicit policy choices, not hidden semantics. A conforming implementation MUST:
- make edge mappings visible and auditable
- record mapping policy in event metadata
- allow mappings to be edited without changing kernel physics

Example mappings are provided in Appendix D. These are illustrative, not normative.

### 4.6.3 External adapters

External systems (APIs, JSON inputs) MAY use adapter components that:
- parse external formats into plugin payloads
- route to appropriate plugins
- maintain provenance metadata

Adapters are application-layer and remain accountable to authentic human agency.

## 4.7 Tetrahedral Geometry for Governance

The GGG governance measurement layer uses the complete graph on four vertices, K₄. Its vertices correspond to:

- Governance
- Information
- Inference
- Intelligence

### 4.7.1 Canonical vertex order

Vertices are ordered:

`(Gov, Info, Infer, Intel) = (0, 1, 2, 3)`

### 4.7.2 Canonical edge order

Edges are ordered as the six undirected pairs:

- edge 0: (0,1) Gov–Info
- edge 1: (0,2) Gov–Infer
- edge 2: (0,3) Gov–Intel
- edge 3: (1,2) Info–Infer
- edge 4: (1,3) Info–Intel
- edge 5: (2,3) Infer–Intel

All ledgers `y ∈ ℝ^6` and all GovernanceEvents must use this canonical ordering.

## 4.8 Hodge Decomposition on K₄

Hodge decomposition splits an edge ledger into:
- a gradient component (globally consistent differences between vertex potentials)
- a cycle component (residual circulation around loops)

This split is the basis for aperture.

### 4.8.1 Incidence matrix

Let `B` be the signed incidence matrix of K₄ with the vertex and edge order above:

```text
B =
[[-1, -1, -1,  0,  0,  0],   # Gov
 [ 1,  0,  0, -1, -1,  0],   # Info
 [ 0,  1,  0,  1,  0, -1],   # Infer
 [ 0,  0,  1,  0,  1,  1]]   # Intel
```

### 4.8.2 Projection operators

This specification uses the unweighted inner product on edges:

- `W = I_6`

Event confidence is applied through the event update value `Δ = magnitude × confidence`, not through a weight matrix.

Define:
- `L = B B^T`
- `L^†` denotes the pseudoinverse of L
- `P_grad = B^T L^† B`
- `P_cycle = I_6 − P_grad`

For K₄ with `W = I_6`, the closed form holds exactly:

- `P_grad = (1/4) (B^T B)`
- `P_cycle = I_6 − P_grad`

Normative requirement:
- implementations must use this closed form in order to ensure deterministic, cross-platform identical results

For any ledger `y ∈ ℝ^6`:
- `y_grad = P_grad y`
- `y_cycle = P_cycle y`
- `y = y_grad + y_cycle`

Reference geometry helpers, including a canonical cycle basis and synthetic ledger constructors used for testing and simulation, are specified in Appendix F.

## 4.9 Aperture Definition

Aperture measures the fraction of edge energy that lies in the cycle component.

For each domain ledger `y_D`:

```text
A_D = ||y_cycle||^2 / ||y||^2
```

where `||v||^2 = v^T v`.

If `y = 0`, define `A_D = 0`.

The CGM-derived target aperture is:

- `A* = 0.0207`

The kernel does not enforce convergence toward A*. The kernel makes the computation of A_D reproducible by providing shared moments and a deterministic event ordering substrate.

## 4.10 Replay Integrity for Governance Metrics

The kernel provides replay for state trajectories. The application layer provides replay for governance ledgers and apertures.

### 4.10.1 Kernel replay

Given the archetype and byte ledger prefix `b_1…b_t`, all participants compute the same kernel state `s_t`.

### 4.10.2 Ledger replay

Given the same event log `E_1…E_k` applied in the same order, all participants compute identical domain ledgers and identical apertures.

The event log defines the governance record. The kernel state defines shared moments to which events can be bound.

### 4.10.3 Optional binding to shared moments

Events may record `(state_index, last_byte)` to certify the shared moment at which they occurred. This binding is not required to compute aperture. It is required for governance audit where ordering and attribution must be inspected under authentic human accountability.

## 4.11 Structural Displacement and Policy Modes

The kernel provides structural reproducibility. The application layer uses that substrate to measure and respond to displacement risks.

### 4.11.1 Four displacement categories

Governance Traceability Displacement (GTD) is kernel-native at the boundary:
- if a presented state is not in Ω, it cannot be a router-valid transformation of the archetype

Information Variety Displacement (IVD), Inference Accountability Displacement (IAD), and Intelligence Integrity Displacement (IID) are application-layer diagnoses. They are computed from:
- event provenance classification under THM
- ledger structure through `y_grad` and `y_cycle`
- aperture deviation from the target A*
- cross-domain coupling rules specified by GGG at the governance layer

This specification defines the measurement substrate. It does not encode policy decisions into kernel physics.

### 4.11.2 Application-layer displacement diagnosis

The kernel provides structural reproducibility. The application layer uses that substrate to diagnose displacement risks through:
- event provenance classification under THM
- ledger structure analysis through `y_grad` and `y_cycle`
- aperture deviation from the target A*
- cross-domain coupling rules specified by GGG at the governance layer

Operational modes for implementing these diagnoses are provided in Appendix D.

---

# 5. Conformance Profiles

This specification defines conformance as three profiles. Implementations MAY claim conformance to one or more profiles, provided they satisfy all requirements in the claimed profile.

## 5.1 Profile K: Kernel Conformance

A conforming kernel implementation MUST satisfy:

Representation:
- state packing and unpacking MUST follow `state24 = (A12 << 12) | B12` with 12-bit masking
- archetype MUST equal `0xAAA555`

Transcription:
- for all bytes, `intron = byte ^ 0xAA` MUST be used

Expansion:
- expansion function MUST match the canonical definition in §2.5.2
- all 256 introns MUST yield distinct Type A masks
- Type B mask MUST always be zero

Transition:
- Type A MUST be mutated by XOR with the Type A mask prior to gyration
- Type B MUST NOT be mask-mutated prior to gyration
- gyration MUST set:
  - `A_next = B ^ 0xFFF`
  - `B_next = A_mut ^ 0xFFF`

Atlas compatibility:
- ontology MUST contain exactly 65,536 states
- ontology MUST be closed under all 256 byte actions
- epistemology MUST match direct stepping for all `(state, byte)` pairs

Dynamics:
- reachability radius from archetype MUST be exactly 2 (Property P4)
- depth-2 commutation law MUST hold (Property P6)
- per-byte bijection MUST hold (Property P2)
- full 256-way fanout per state MUST hold (Property P13)

## 5.2 Profile M: Governance Measurement Conformance

A conforming measurement implementation MUST satisfy:

Domain ledgers:
- maintain three domain ledgers `y_D ∈ ℝ^6` for `D ∈ {Economy, Employment, Education}`
- update ledgers only by GovernanceEvents using the canonical edge order:
  - `y_D[e] = y_D[e] + magnitude × confidence`

Geometry:
- use the canonical K₄ vertex and edge order defined in §4.7
- compute `P_grad` and `P_cycle` exactly as specified in §4.8 with `W = I_6`
- compute aperture `A_D` exactly as specified in §4.9

Replay integrity:
- same event sequence implies the same ledgers and apertures

Prohibitions:
- MUST NOT compute aperture from kernel state bits directly
- MUST NOT use a non-identity weight matrix W for aperture
- MUST NOT apply confidence as a second weighting mechanism beyond `magnitude × confidence`

## 5.3 Profile R: Runtime Router Conformance

A conforming runtime router implementation MUST satisfy:

Orchestration:
- provide an orchestration component that advances kernel state by bytes, applies governance events, and reports status

Audit logs:
- maintain an append-only byte log sufficient for deterministic replay
- maintain an append-only event log sufficient for deterministic replay of domain ledgers and apertures
- preserve ordering in both logs

Event binding:
- provide a mechanism to bind events to the current kernel moment by recording `kernel_state_index` and `kernel_last_byte`

Status reporting:
- provide a status report including at minimum:
  - kernel signature fields
  - last byte
  - lengths of byte log and event log
  - current domain ledgers
  - current domain apertures
  - total applied event count

Connector surface:
- provide a deterministic mechanism to translate external payloads into GovernanceEvents
- preserve audit metadata sufficient to identify the source connector and mapping policy used

---

# 6. Notes

## 6.1 Scope

This specification defines:
- the kernel transition physics and its compiled atlas artifacts
- the governance measurement substrate: domain ledgers, Hodge decomposition, and aperture

This specification does not define:
- natural language processing or semantic interpretation
- policy decisions embedded in the transition function
- delegation of accountability from humans to the kernel

## 6.2 Source Type Classification

Under The Human Mark, this kernel is Derivative in both authority and agency. It transforms and routes signals, and it provides shared structural observables. It does not originate authority and it does not bear accountability.

---

# Appendix A. CGM Theoretical Foundation

This appendix states the CGM constitutive claims used as motivation. It is not required to implement the kernel, but it provides the intended interpretation of chirality, non-commutativity, closure, and memory.

## A.1 Common Source (CS)

Right transitions preserve the horizon; left transitions alter it.

Modal form: S implies `[R]S` is equivalent to S, and `[L]S` is not equivalent to S.

Router realization: Type A mutation prior to gyration corresponds to an altering modality. Type B not receiving a direct mask corresponds to a preserving modality. The resulting asymmetry is the minimal computational realization of chirality.

## A.2 Unity Non-Absolute (UNA)

At depth two, order matters but not absolutely.

Modal form: S implies it is not necessary that `[L][R]S` is equivalent to `[R][L]S`.

Router realization: depth-2 composition has a closed form (Property P5) and commutes only in the degenerate case `x = y` (Property P6).

## A.3 Opposition Non-Absolute (ONA)

Opposition occurs without absolute contradiction.

Modal form: S implies it is not necessary that `[L][R]S` is equivalent to the negation of `[R][L]S`.

Router realization: the 256 distinct masks and 256-way fanout preserve differentiated paths (Property P13).

Interpretive mapping to 3D and 6 degrees of freedom: the discrete structure (depth-2 non-commutativity, depth-4 closure, two-phase 256×256 ontology) is intended as a discrete reflection of CGM’s 3D structure and 6 degrees of freedom. This mapping is interpretive rather than a certified geometric theorem of the discrete kernel.

## A.4 Balance Universal Egress (BU-Egress)

Depth-four closure achieves coherent measurement.

Router realization: depth-4 alternation returns to identity (Property P7). The ontology is closed under all byte actions.

## A.5 Balance Universal Ingress (BU-Ingress)

The balanced state reconstructs prior distinctions.

Router realization: deterministic replay from the byte ledger reconstructs full trajectories, and inverse stepping reconstructs them backwards exactly given the byte sequence (§4.3).

---

# Appendix B. Non-normative Numerics and Test Suite

## B.1 Numeric Properties

The numbers 256 and 65,536 emerge from the byte interface and the transition law. They also coincide with commonly used scales in computing, which makes the system practical to enumerate, store, and test.

256 = 2^8. It also equals 4^4 and ((2^2)^2)^2, sometimes called zenzizenzizenzic.

65,536 = 2^16. It is a superperfect number: applying the sum-of-divisors function twice yields exactly twice the original. The divisor sum of 65,536 is 131,071, and the divisor sum of 131,071 is 131,072, which equals 2 × 65,536.

These properties are historical observations rather than design requirements.

## B.2 Test Suite Details

The certified properties in Section 3.4 are verified by an exhaustive test suite. The reference test suite contains 95 tests with approximately 2.7 seconds runtime in the reference environment.

The suite includes exhaustive verification of:
- all `65,536` states for full 256-way fanout (P13)
- all `16,777,216` state-byte pairs for atlas correctness (P12)
- all `65,536` ordered byte pairs on the archetype for the depth-4 identity (P7)

Test names and coverage claims are provided in Section 3.4. Test suite size and runtime are implementation details that may vary across environments and are not normative.

---

# Appendix C. Application-Layer GGG Aperture

This appendix restates the normative aperture definition and implementation constraints.

## C.1 Scope

The kernel provides:
- deterministic state transitions
- shared moments
- replayable byte ledger

The application layer provides:
- domain ledgers `y_D ∈ ℝ^6`
- GovernanceEvent processing
- Hodge decomposition and aperture computation

## C.2 Normative Definition

Aperture is computed using the canonical closed-form definition from §4.8:
- the incidence matrix B of K₄ (§4.8.1)
- unweighted projections with `W = I_6`
- `P_grad = (1/4)(B^T B)` (closed form, normative)
- `P_cycle = I_6 − P_grad`
- `A_D = ||P_cycle y_D||^2 / ||y_D||^2` (with `A_D = 0` when `y_D = 0`)

Confidence is encoded through GovernanceEvent value `Δ = magnitude × confidence`, not through a weight matrix.

Implementations MUST use the closed form `P_grad = (1/4)(B^T B)` to ensure deterministic, cross-platform identical results. The general form using pseudoinverse is not normative.

## C.3 Implementation Requirements

A conforming implementation must:
- use the canonical K₄ edge order (§4.7.2)
- use `W = I_6`
- use the closed-form `P_grad = (1/4)(B^T B)`
- apply confidence through events only
- preserve replay integrity for ledgers and apertures

---

# Appendix D. Operational Modes

This appendix describes application-layer policy modes for implementing displacement diagnosis. These modes correspond to the four CGM stages; see Appendix A for theoretical foundation.

## D.1 Mode CS: Governance Management

Policy enforces ontology identity and ledger continuity:
- continuity of ledgers
- verification of shared moments
- verification of ontology membership when state is presented externally

**Signature use:** verify `state_index ∈ [0, N)` and maintain append-only byte ledger.

## D.2 Mode UNA: Information Curation

Policy preserves transformation variety and rotational degrees of freedom:
- maintenance of transformation variety
- coverage metrics over byte actions and state neighborhoods

**Signature use:** track which bytes have been applied from each state, ensure coverage of transformation space.

## D.3 Mode ONA: Inference Interaction

Policy maintains differentiation across the 256 transformation paths:
- preservation of differentiated paths
- detection of premature collapse of independent trajectories

**Signature use:** compare state trajectories, detect when paths converge prematurely.

## D.4 Mode BU: Intelligence Cooperation

Policy enforces closure and parity structure:
- enforcement of closure constraints as checks
- verification of parity structure using Property P8
- verification of depth-4 alternation identity using Property P7

**Signature use:** verify depth-4 alternation identity, check XOR-parity consistency via Property P8.

These modes are application-layer governance patterns. The kernel provides structural primitives enabling them.

---

# Appendix E. Atlas Construction Procedures

This appendix describes permitted procedures for constructing atlas artifacts. The procedures are non-normative. The resulting artifacts are normative and MUST satisfy the requirements in the main specification.

## E.1 Ontology construction

The ontology MAY be constructed by direct closed-form construction using Property P3:

- construct `A_set = { ARCHETYPE_A12 ^ m_b : b ∈ [0,255] }`
- construct `B_set = { ARCHETYPE_B12 ^ m_b : b ∈ [0,255] }`
- form `Ω = A_set × B_set`
- pack each pair `(A12, B12)` into a 24-bit state
- sort ascending to produce a canonical `ontology.npy`

This procedure is equivalent to reachability enumeration from the archetype.

## E.2 Epistemology construction

The epistemology MAY be constructed column-wise:

- for each byte `b ∈ [0,255]`, compute next-state values for all ontology states using the transition law
- map each next-state value to its ontology index using a membership-preserving lookup
- verify closure by asserting every computed next-state is present in the ontology
- write the resulting indices as column `epistemology[:, b]`

---

# Appendix F. Reference Geometry Helpers

This appendix defines optional reference helpers exposed by implementations for analysis, testing, and simulation. These helpers are not required for conformance unless explicitly claimed by an implementation.

## F.1 Cycle basis for K₄

The cycle space `ker(B)` has dimension 3 for K₄. Implementations MAY provide a normalized cycle basis `U ∈ ℝ^{6×3}` whose columns:
- lie in `ker(B)`
- have unit norm
- span the cycle space

## F.2 Synthetic ledger construction with target aperture

Implementations MAY provide a helper that constructs an edge vector `y ∈ ℝ^6` from vertex potentials `x ∈ ℝ^4` and a target aperture `A`:

- compute `y_grad0 = B^T x`
- compute `G = ||y_grad0||^2`
- select a unit-norm cycle direction `u` in the cycle space
- compute `k^2 = (A/(1-A)) × G`
- return `y = y_grad0 + k u`

Implementations MAY include a deterministic safeguard for near-zero gradient cases to avoid degenerate constructions in tests and simulations.