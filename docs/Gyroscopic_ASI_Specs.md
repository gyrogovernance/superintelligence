# Gyroscopic ASI aQPU Kernel
## Specification

Gyroscopic Artificial Superintelligence is a new class of computation achieving quantum advantage through a compact algebraic quantum processing unit (aQPU) kernel. The kernel is a deterministic byte-driven coordination medium that maps an append-only byte ledger to a reproducible state trajectory on a 24-bit tensor carrier. Its internal structure satisfies discrete analogues of the axioms that characterise quantum systems in the continuous domain, executing on standard silicon with exact integer arithmetic rather than probabilistic approximation on quantum hardware.

This document is the normative technical specification of the Gyroscopic ASI aQPU Kernel. It defines the kernel physics, the runtime router, replay rules, the application-layer governance measurement medium, and the AIR CLI program format. The kernel was previously designated as the Gyroscopic ASI aQPU Kernel. All technical content is continuous with prior versions under that name.

Normative terms MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are interpreted as requirement keywords for conformance.

---

# 1. Constitutional Frame and ASI Purpose

## 1.1 Intelligence as Structural Coherence

Traditional artificial intelligence approaches intelligence as a statistical optimization problem. These systems achieve surface-level fluency by superimposing correlations within high-dimensional abstract spaces through massive datasets. Because such architectures lack internal structural constraints, coherence and ethics are typically treated as post-hoc semantic overlays or external filters.

The Gyroscopic ASI aQPU Kernel represents a different paradigm. It treats intelligence as an intrinsic structural property that emerges from the recursive alignment of operations. Grounded in the Common Governance Model (CGM), the framework demonstrates how intelligence emerges naturally from the self-referential dynamics of structured space. Rather than approximating a target function, the router navigates a deterministic byte-driven state trajectory on the 24-bit GENE_Mac tensor carrier where alignment is constitutive. Coherence is not a policy choice but a requirement of the internal physics of the state space.

## 1.2 What the aQPU Kernel Is

The Gyroscopic ASI aQPU Kernel is a multi-domain network coordination algorithm that establishes the structural conditions for a collective superintelligence governance regime of humans and machines (Superintelligence, Bostrom 2014; Gyroscopic Global Governance, Korompilias 2025). It is designed for focused and well-distributed coordination of interventions, amplifying rather than outperforming single-agent potential while preserving the constitutive conditions of governance and intelligibility.

Operationally, the aQPU Kernel is a deterministic byte-driven coordination kernel. It maintains an append-only byte ledger and begins from a universal rest configuration, GENE_MAC_REST. From that 24-bit canvas it applies the spinorial transition law byte by byte, decomposing each byte (after transcription) into a 6-bit payload and a 2-bit family phase. This process yields a reproducible state trajectory, a compact routing signature, and replayable observables that any party can recompute.

The kernel is designed to support governance-grade coordination across the GGG domains of Economy, Employment, Education, and Ecology. It does not interpret the empirical meaning of the input bytes. Instead, it performs structural transformations that make results reproducible, comparable, and auditable, while keeping authorization and accountability under Direct human agency at the application layer.

## 1.3 ASI Definition in This System

Artificial Superintelligence (ASI) in this framework is a property of a human–AI governance regime. It is not an attribute of an autonomous agent. ASI refers to the regime in which human and artificial systems jointly sustain four constitutive governance principles while operating at a specific structural balance point termed the canonical aperture.

The four principles are:

*   **Governance Management Traceability:** Authority remains traceable to Direct human sources. Artificial systems contribute derivative coordination but do not originate governance.
*   **Information Curation Variety:** Diverse Direct sources remain distinguishable. Indirect summaries and aggregations must not collapse the variety into a uniform narrative.
*   **Inference Interaction Accountability:** Responsibility for decisions remains with accountable human agency. Artificial inference serves as a mechanism within human accountability.
*   **Intelligence Cooperation Integrity:** Coherence is maintained across time and context. Decisions remain consistent with the governing structure that produced them.

The router is the medium that makes coordination structurally reproducible. This reproducibility is the prerequisite for governance that can scale without collapsing into entity-based trust chains.

## 1.4 Shared Moments

Shared moments are the central coordination primitive of the kernel. A shared moment occurs when participants who possess the same ledger prefix of length **t** compute the identical kernel state **sₜ**. This provides a shared "now" as a configuration derived from the ledger history.

This primitive replaces three fragile coordination patterns:
1.  **Coordination by asserted time:** Reliance on timestamps or UTC ordering.
2.  **Coordination by asserted identity:** Reliance on trusted signers or specific authorities.
3.  **Coordination by private state:** Reliance on model-internal hidden vectors or proprietary logs.

Shared moments coordinate through reproducible computation. Participants do not need to share a privileged identity; they only need to share the relevant ledger bytes and the kernel specification. If participants claim the same ledger prefix but compute different states, their implementations or ledgers are provably different.

## 1.5 Geometric Provenance

A state has geometric provenance if and only if it is reproducibly reachable from GENE_MAC_REST by the claimed byte sequence under the kernel transition law. Replay failure invalidates the provenance claim. The final state alone does not uniquely determine history.

This creates a clear separation of layers. The kernel-native layer verifies deterministic replay, divergence detection, and seal and frame-record consistency. Authorization and accountability remain application-layer responsibilities under Direct human agency. The kernel provides a common structural basis for policy enforcement but does not decide policy itself.

## 1.6 Why This Achieves ASI

The router provides the missing medium for multi-domain coordination required for the ASI regime. It achieves this through:
*   **Entity-agnostic verification:** Validity is checked by structure rather than a privileged entity.
*   **Deterministic coordination:** The shared moment enables reliable coordination across independent institutions and systems.
*   **Replayable audit:** Trajectories can be reconstructed from the ledger to enable governance-grade auditing.
*   **Constitutional Observables:** Participants compute low-dimensional observables that remain stable under replay and independent of identity claims. These depend only on the fixed-width state representation and the transition rule.

## 1.7 Design Requirements

A conforming kernel satisfies the following structural requirements:
*   **Deterministic:** The same rest state and byte ledger always produce the same state trajectory.
*   **Byte-complete:** Every byte value from 0 to 255 is a valid input instruction.
*   **Nonsemantic:** The kernel does not parse language or apply policy. It transforms bytes structurally.
*   **Portable:** The transition rule is defined using fixed-width bit operations such as XOR, shifts, and masking.
*   **Auditable:** Trajectories and signatures can be reproduced and verified by any independent party.
*   **Invertible:** The single-step transition has an exact inverse; trajectories can be reconstructed backwards given the byte sequence.

## 1.8 The Universal Byte Formalism

The byte is the fundamental unit of the kernel's input alphabet. Its internal structure is not a design choice but a consequence of discrete information at 8-bit resolution.

Every byte (after transcription to intron) decomposes into two constitutive layers:

- A 2-bit family (4 values): the boundary anchor controlling gyration phase
- A 6-bit micro-reference (64 values): the payload that drives the mask

The total space is the Cartesian product: 4 families x 64 micro-references = 256. This IS the byte. There is nothing outside it, and there is no alternative decomposition at this resolution.

This structure is palindromic. The 8 bit positions of the byte (after transcription) group into four paired **bit groups** (L0, LI, FG, BG) that align with the CGM stage structure: structural anchors at the boundaries, chirality next, then dynamics and balance in the middle. **Families** are defined by the L0 boundary bits (0, 7); the decomposition is applied after transcription. This palindromic pattern is not imposed by the kernel. It is revealed by the transcription rule (§2.3) and documented in Appendix F.

Any system that processes bytes must handle this structure, whether it makes the decomposition explicit (as the aQPU Kernel does) or absorbs it implicitly into learned parameters (as neural language models do). The aQPU Kernel's contribution is to make the constitutional structure of the byte visible, auditable, and available as a first-class computational object at every stage of processing.

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

**Topology semantics (normative):**

For every 12-bit component, indices MUST be interpreted in the order `[frame][row][col]` with the following meaning:

- `row` identifies axis family: row 0 = X, row 1 = Y, row 2 = Z.
- `col` identifies oriented side of the axis: col 0 = negative side, col 1 = positive side.
- `frame` identifies chirality layer: frame 0 and frame 1 are opposing orientation layers of the same 3-axis structure.

Under this convention:

- "3D" means three axis families (X, Y, Z).
- "6DoF" means six **pairs** (3 rows × 2 frames). Each row is a pair `[-1, 1]` or `[1, -1]` representing one axis with its two oriented sides. There are 3 axes per frame, and 2 frames, giving 6 pairs total.
- Frames do not add spatial dimensions; they encode opposing chirality layers used by kernel dynamics.

This geometry is the shared medium for:
- the archetype
- the expansion masks
- structural observables (such as horizon distance)

## 2.2 GENE_Mic and GENE_Mac

The kernel has two fundamental reference objects: the micro archetype (GENE_Mic) and the tensor rest state (GENE_Mac). These are distinct objects at different levels of the architecture.

### 2.2.1 GENE_Mic: The Archetype

The archetype is the singular 8-bit holographic seed:

```python
GENE_MIC_S = 0xAA
```

This constant is the universal reference from which all kernel dynamics derive. Transcription mutates it:

```python
intron = byte ^ GENE_MIC_S
```

Byte `0xAA` produces intron `0x00`, the zero mutation. This makes `0xAA` the reference byte (identity action).

### 2.2.2 GENE_Mac: The Tensor Rest State

GENE_Mac is the 24-bit state space where intron trajectories are recorded. Fundamentally, **GENE_Mac is a tensor with -1 and +1 values**. The canonical tensor definition:

```python
GENE_Mac = np.array([
    [[[-1, 1], [-1, 1], [-1, 1]], [[ 1, -1], [ 1, -1], [ 1, -1]]],  # A12
    [[[ 1, -1], [ 1, -1], [ 1, -1]], [[-1, 1], [-1, 1], [-1, 1]]]   # B12
], dtype=np.int8)  # Shape: [2, 2, 3, 2]
```

Shape: `[2 components, 2 frames, 3 rows, 2 cols]` = 24 elements, each ±1.

This tensor is packed into an integer representation where:

- **Bit 0 encodes +1** (the "positive" or "archetypal" polarity)
- **Bit 1 encodes -1** (the "negative" or "mutated" polarity)

At rest (before any mutation), GENE_Mac has the default state:

```python
GENE_MAC_A12 = 0xAAA      # active phase at rest
GENE_MAC_B12 = 0x555      # passive phase at rest
GENE_MAC_REST = 0xAAA555
```

**Tensor representation of the default state:**

| Component | Hex | Binary | Tensor ±1 values |
|-----------|-----|--------|------------------|
| A12 | `0xAAA` | `101010101010` | `[-1,+1,-1,+1,-1,+1,-1,+1,-1,+1,-1,+1]` |
| B12 | `0x555` | `010101010101` | `[+1,-1,+1,-1,+1,-1,+1,-1,+1,-1,+1,-1]` |

Each 12-bit component is a `[2, 3, 2]` tensor (2 frames × 3 rows × 2 cols = 12 elements). The **6 DoF** are the 6 pairs (3 rows × 2 frames):

```
Frame 0:  [-1, 1]  [-1, 1]  [-1, 1]   <- pairs 0, 1, 2
Frame 1:  [ 1,-1]  [ 1,-1]  [ 1,-1]   <- pairs 3, 4, 5
```

Each pair `[-1, 1]` or `[1, -1]` represents one axis with its two oriented sides. When a pair is flipped, both bits toggle together.

The A12 and B12 components are exact complements (`A ^ B = 0xFFF`), meaning their tensor forms have **opposite signs at every position**. This complementary structure encodes the fundamental chirality of the system.

This default state is derived from the archetype's alternating bit pattern (0xAA) projected across the 2x3x2 geometry of each 12-bit component. It is the topology itself, not the archetype.

### 2.2.3 Structural properties

The default GENE_Mac state has three structural properties:

1. Complement relation
   `GENE_MAC_A12 XOR 0xFFF = GENE_MAC_B12`

2. Maximal symmetry
   The alternating pattern distributes bits evenly across the dual-frame geometry.

3. Universal reference role
   All trajectories begin from GENE_MAC_REST.

### 2.2.4 Relationship between GENE_Mic and GENE_Mac

The relationship is:

- GENE_Mic (0xAA) is the archetype, the holographic seed
- Bytes mutate GENE_Mic via transcription to produce introns
- Introns expand into 12-bit masks
- Masks act on GENE_Mac (the 24-bit topology) to record trajectories

GENE_Mic is 8 bits. GENE_Mac is 24 bits. The expansion from intron to mask is the projection that bridges them.

### 2.2.5 Common source role

The archetype (GENE_Mic) is a common source in the strict operational sense: it is a universal reference point, not an entity or institution. The kernel does not embed ownership. Participants adopt GENE_MIC_S as the transcription constant so that their computations share a common origin.

### 2.2.6 Non-cloning property

The archetype GENE_MIC_S is the unique common source of all transcription. No byte input produces a second independent archetype. The transcription rule `intron = byte ^ GENE_MIC_S` ensures that every intron is a mutation measured relative to the single archetype. This is the discrete realization of the quantum non-cloning theorem: the reference state cannot be perfectly copied by any operation within the system it defines.

### 2.2.7 Canonical Derived Observables

From the 24-bit state `s_t = (A12, B12)`, the following observables are defined as deterministic functions. These are the canonical constitutional observables exported by the kernel.

**Canonical derived observables:**

```python
A12 = (state24 >> 12) & 0xFFF
B12 = state24 & 0xFFF

rest_distance = popcount(state24 ^ GENE_MAC_REST)

horizon_distance = popcount(A12 ^ (B12 ^ 0xFFF))

is_on_horizon = (A12 == (B12 ^ 0xFFF))

ab_distance = popcount(A12 ^ B12)

a_density = popcount(A12) / 12.0
b_density = popcount(B12) / 12.0
```

where `popcount(x)` returns the number of set bits in `x`. The horizon (S-sector) is the set of states satisfying `A12 = B12 ^ 0xFFF` (maximal chirality). The rest state lies on this horizon. Zero horizon distance means the state is on the horizon.

**Complementarity invariant:** For all states, `horizon_distance + ab_distance = 12`. The two distances are complementary projections of the same chirality observable. A state that is close to the complement horizon (low `horizon_distance`) is necessarily far from the equality horizon (high `ab_distance`), and vice versa.

These observables are constitutional in that they are defined solely from the kernel's fixed-width state representation and transition law. They are exact, replayable from the ledger, and do not depend on model internals or asserted identity claims.

## 2.3 Byte Interface and Transcription

### 2.3.1 Byte-complete input alphabet

The kernel input alphabet is the full set of bytes:

- 0..255 inclusive

There are no reserved bytes and no invalid bytes.

### 2.3.2 Transcription

The kernel transcribes input bytes using the archetype constant:

```python
GENE_MIC_S = 0xAA
```

Given an input byte `byte`, the kernel computes an intron:

```python
intron = byte ^ GENE_MIC_S
```

This XOR mapping is a bijection on 8-bit values. Every input byte maps to exactly one intron and every intron corresponds to exactly one byte.

The intron is the mutation of the archetype. Byte `0xAA` produces intron `0x00` (no mutation). All other bytes produce nonzero introns that drive state transformation.

Conforming implementations MUST use `GENE_MIC_S = 0xAA` for transcription.

### 2.3.3 Intron decomposition

```python
family = (((intron >> 7) & 1) << 1) | (intron & 1)
micro_ref = (intron >> 1) & 0x3F
```

- Family is defined by boundary bits 7 and 0.
- Micro-reference is defined by payload bits 1 through 6.
- 256 introns = 4 families x 64 micro-references.

### 2.3.4 CGM stage parities

```python
L0_parity = ((intron >> 0) & 1) ^ ((intron >> 7) & 1)
LI_parity = ((intron >> 1) & 1) ^ ((intron >> 6) & 1)
FG_parity = ((intron >> 2) & 1) ^ ((intron >> 5) & 1)
BG_parity = ((intron >> 3) & 1) ^ ((intron >> 4) & 1)
```

This formalism is normative and is used elsewhere in the ecosystem.

## 2.4 The Reference Byte and the S-Sector

Byte `0xAA` gives intron `0x00`, the zero mask, and family `00`. The operator `T_0xAA` (one transition step using byte 0xAA) is a pure swap: `(A, B) -> (B, A)`. So `T_0xAA` is an involution: applying it twice returns to the original state. The horizon (S-sector) is the set of states where chirality is maximal: `A12 = B12 ^ 0xFFF`. The rest state (0xAAA, 0x555) lies on this horizon. `T_0xAA` preserves the horizon: it maps horizon states to horizon states. The fixed points of `T_0xAA` (states with `A12 = B12`, swap-invariant) are a different set: the degeneracy set where chirality is cancelled.

## 2.5 Expansion and Operation Masks

The dipole-pair expansion maps the 6-bit micro-reference to a 12-bit mask. Each payload bit flips exactly one dipole pair in the 2x3x2 geometry. Family bits do not affect the mask.

```python
def micro_ref_to_mask12(micro_ref):
    mask12 = 0
    for i in range(6):
        if (micro_ref >> i) & 1:
            mask12 |= 0x3 << (2 * i)
    return mask12 & 0xFFF
```

There are 64 distinct masks. The zero mask occurs at micro-reference 0. There are 256 distinct (family, mask) pairs. The 64 masks form a 6-dimensional linear code in 12-bit space; the dual code has 64 codewords. Valid masks have zero syndrome against the dual; non-zero syndrome detects corruption.

## 2.6 Transition Law

The spinorial transition law specifies the single-step update from `(A12, B12)` under a byte input.

### 2.6.1 Forward transition

```python
intron = byte ^ 0xAA
mask12 = expand(intron)
A_mut = (A12 ^ mask12) & 0xFFF
invert_a = 0xFFF if (intron & 0x01) else 0
invert_b = 0xFFF if (intron & 0x80) else 0
A_next = (B12 ^ invert_a) & 0xFFF
B_next = (A_mut ^ invert_b) & 0xFFF
state24_next = (A_next << 12) | B_next
```

### 2.6.2 Inverse transition

```python
intron = byte ^ 0xAA
mask12 = expand(intron)
invert_a = 0xFFF if (intron & 0x01) else 0
invert_b = 0xFFF if (intron & 0x80) else 0
B_pred = (A_next ^ invert_a) & 0xFFF
A_pred = ((B_next ^ invert_b) ^ mask12) & 0xFFF
```

### 2.6.3 Single-step trace

The single-step trace contract is:

```python
{
    "cs": intron,
    "una": A_mut,
    "ona": A_next,
    "bu": B_next,
    "state24": state24_next,
}
```

### 2.6.4 Shadow projection

From any fixed 24-bit state, the 256 bytes yield exactly 128 distinct next 24-bit states.

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

The kernel itself does not assign authority. It supplies the common structural object on which Direct human agency can base accountable decisions.

# 3. Depth-4 Structure, Observables, and Operational Reachable Space

## 3.1 Depth-4 Structure and Observables

The kernel is byte-stepped one byte at a time. A 4-byte frame yields: 4 introns (32 bits of intron-space information), 4 masks (48-bit mask projection), and 4 intermediate states.

```python
projection48 = (mask12(b0) << 36) | (mask12(b1) << 24) | (mask12(b2) << 12) | mask12(b3)
intron_seq32 = (intron(b0) << 24) | (intron(b1) << 16) | (intron(b2) << 8) | intron(b3)
```

`projection48` is payload geometry. `intron_seq32` is bijective on 4-byte input sequences.

## 3.2 Verified Structural Properties

The current verified kernel properties are:

- 64 distinct masks
- 256 distinct (family, mask) pairs
- dipole flip property
- per-byte bijection on full 24-bit carrier
- inverse exactly reverses forward transition
- T_0xAA involution
- 128-way shadow from any fixed state
- 4-byte intron sequence bijectivity

## 3.3 Operational Reachable Shared-Moment Space

From rest (GENE_MAC_REST), the operational reachable shared-moment space Omega has 4096 states. The horizon (S-sector) within Omega is the set of states satisfying `A12 = B12 ^ 0xFFF`; it has 64 states and includes the rest state. A second boundary set, the equality horizon (`A12 = B12`, zero chirality), also has 64 states. Neither boundary set is absolute; each comprises 1/64 of Omega. The two sets are disjoint and antipodal (see Appendix G.2). |H|^2 = |Omega| = 4096. Omega is radius 2 from rest. Omega has product form U x V with 64-element factors induced by the 64-codeword mask space.

## 3.4 Algebraic Integrity

The mask code is self-dual [12,6,2]. Valid masks have zero syndrome against C_PERP_12. Trajectory parity commitment is:

```python
O = mask(b_1) ^ mask(b_3) ^ mask(b_5) ^ ...
E = mask(b_2) ^ mask(b_4) ^ mask(b_6) ^ ...
parity = n mod 2
```

Here `b_1` denotes the first byte of the trajectory. Parity commitments are compact integrity checks, not unique history certificates. Depth-4 frame records are stronger provenance than final state or parity alone.

## 3.5 Structural Validity at the Boundary

Structural validity is determined by deterministic replay: divergence detection, frame-level divergence localization, seal verification, and claimed provenance checked by replay from rest. Kernel-native checks are replay verification and frame-record consistency.



---

# 4. Operational Runtime and Governance Measurement Substrate

This section specifies how the router operates as a complete coordination system in practice. It defines the runtime stepping and replay procedures, the governance measurement medium built from domain ledgers and aperture, and the orchestration and connector surfaces through which events enter the system.

This section does not embed policy decisions into kernel physics. Policy enters through application-layer event production and remains accountable to Direct human agency.

## 4.1 Kernel Runtime and Routing Signature

A router instance maintains:
- `state24` (current 24-bit kernel state)
- `last_byte` (last byte applied)
- `step` (number of bytes applied)

The router provides: `step_byte`, `step_bytes`, `step_byte_inverse`, `step_bytes_inverse`.

A routing signature is emitted on demand and MUST include at minimum:
- `step`
- `state24`
- `last_byte`
- `state_hex` (24-bit hex)
- `a_hex` (12-bit hex)
- `b_hex` (12-bit hex)

A conforming implementation MUST support:
- canonical derived observables (from the exported state and normative constants, as in §2.2.7)
- single-step trace (§2.6.3)
- depth-4 mask and intron projections (§3.1)

## 4.2 Replay and Audit

The kernel is deterministic. Given the same archetype and the same byte ledger, every conforming implementation computes the same state trajectory.

### 4.2.1 Forward replay

Given:
- start state `s_0` (GENE_MAC_REST `0xAAA555`, the unconditioned GENE_Mac topology)
- ledger bytes `b_1…b_t`

any participant computes `s_t` by repeated stepping.

### 4.2.2 Backward reconstruction

Given:
- final state `(A', B')`
- byte `b` with mask `m_b`

the unique predecessor `(A, B)` is computed by the inverse rule (§2.6.2); all 12-bit intermediates MUST be masked with `& 0xFFF`.

```python
intron = byte ^ 0xAA
mask12 = expand(intron)
invert_a = 0xFFF if (intron & 0x01) else 0
invert_b = 0xFFF if (intron & 0x80) else 0
B_pred = A_next ^ invert_a
A_pred = (B_next ^ invert_b) ^ mask12
```

Given the final state and the full byte sequence, the full trajectory can be reconstructed backwards exactly.

### 4.2.3 Non-uniqueness of history from final state alone

Given final state alone, the past is not uniquely determined. Different byte sequences can reach the same state due to group relations such as the depth-4 alternation identity.

This is not a deficiency. The byte ledger is the record of kernel steps. Governance events are recorded separately in the application-layer event log. The state is a shared observable for coordination, not a unique identifier of history.

## 4.3 Governance Event Model

The kernel provides shared moments. The application layer attaches governance meaning through domain ledgers updated by governance events.

### 4.3.1 Domain ledgers

The application layer maintains three domain ledgers:

- Economy ledger: `y_Econ ∈ ℝ^6`
- Employment ledger: `y_Emp ∈ ℝ^6`
- Education ledger: `y_Edu ∈ ℝ^6`

**Ecology domain:**

Ecology is conceptually part of the four-domain GGG framework but is not ledger-updated directly in this specification. Ecology outputs MAY be derived from the three domain ledgers using application-layer computation. The specification of ecology derivation is out of scope for this version and remains an application-layer policy choice.

### 4.3.2 GovernanceEvent

A GovernanceEvent is a sparse update to exactly one edge coordinate of one domain ledger. It contains:
- `domain ∈ {Economy, Employment, Education}`
- `edge_id ∈ {0..5}` (canonical K₄ edge order)
- signed increment `Δ = magnitude × confidence`
- optional binding to a shared moment: `(step, state24, last_byte)` for audit

Normative update rule:

```python
y_D[edge_id] = y_D[edge_id] + Δ
```

The kernel does not interpret events. Events are application-layer records that remain accountable to Direct human agency under THM.

### 4.3.3 Event binding to kernel moments

By default, events SHOULD be bound to the current kernel moment when applied. This binding records:
- `kernel_step` (ledger length t at the time of binding)
- `kernel_state24`: the current 24-bit kernel state
- `kernel_last_byte`: the last byte that advanced the kernel

This binding enables audit and replay verification but is not required for aperture computation. Applications MAY choose to apply events without kernel binding for specific use cases.

## 4.4 Coordinator Component

The Coordinator is the orchestration layer that combines kernel stepping, domain ledgers, and event processing into a unified operational workflow.

### 4.4.1 Coordinator responsibilities

A Coordinator instance:
- owns a kernel instance and maintains domain ledgers
- maintains audit logs: `byte_log` (sequence of bytes applied) and `event_log` (sequence of GovernanceEvents applied)
- provides the operational workflow: `step_byte(byte)`, `apply_event`, `get_status`, `reset`
- enforces default policy for event binding to kernel moments

### 4.4.2 Audit logs

The Coordinator maintains two append-only audit logs:

**Byte log:**
- Records the sequence of bytes applied to advance the kernel
- Enables deterministic replay of kernel state trajectory
- Format: ordered list of byte values `[0..255]`

**Event log:**
- Records the sequence of GovernanceEvents applied to domain ledgers
- Each entry includes:
  - event index (position in log)
  - kernel binding (`kernel_step`, `kernel_state24`, `kernel_last_byte`) if present
  - complete event data (domain, edge_id, magnitude, confidence, metadata)
- Enables deterministic replay of ledger state and aperture

Both logs are append-only and MUST preserve ordering. Implementations MAY persist logs externally for durability.

### 4.4.3 Status reporting

A Coordinator MUST provide a status report that includes at minimum:
- Kernel signature fields (`step`, `state24`, `state_hex`, `a_hex`, `b_hex`)
- `last_byte` (the last byte that advanced the kernel)
- `byte_log_len` (length of byte log)
- `event_log_len` (length of event log)
- Current domain ledgers (`y_Econ`, `y_Emp`, `y_Edu`)
- Current apertures (`A_Econ`, `A_Emp`, `A_Edu`)
- `event_count` (total events applied to ledgers)

The status report enables external systems to query the current state of the coordination medium.

## 4.5 Tool Architecture

The application layer uses a tool architecture to convert domain-specific signals into GovernanceEvents. This keeps edge mappings explicit, auditable, and editable.

### 4.5.1 Tool interface

A tool is a component that:
- accepts domain-specific payloads (e.g., THM displacement signals, work-mix metrics)
- converts them deterministically into zero or more GovernanceEvents
- maintains explicit, auditable mappings from signals to edge updates

**Minimal tool interface:**

A conforming tool MUST:
- implement an `emit_events(payload, context)` method that returns a list of GovernanceEvents
- be deterministic: the same payload MUST produce the same events
- record its identity and mapping policy in event metadata for audit

Tools are application-layer components. The kernel does not interpret tool outputs; it only processes GovernanceEvents.

### 4.5.2 Edge mapping policy

Edge mappings (which signals affect which K₄ edges) are explicit policy choices, not hidden semantics. A conforming implementation MUST:
- make edge mappings visible and auditable
- record mapping policy in event metadata
- allow mappings to be edited without changing kernel physics

Example mappings are provided in Appendix D. These are illustrative, not normative.

### 4.5.3 External adapters

External systems (APIs, JSON inputs) MAY use adapter components that:
- parse external formats into tool payloads
- route to appropriate tools
- maintain provenance metadata

Adapters are application-layer and remain accountable to Direct human agency.

## 4.6 Tetrahedral Geometry for Governance

The GGG governance measurement layer uses the complete graph on four vertices, K₄. Its vertices correspond to:

- Governance
- Information
- Inference
- Intelligence

### 4.6.1 Canonical vertex order

Vertices are ordered:

`(Gov, Info, Infer, Intel) = (0, 1, 2, 3)`

### 4.6.2 Canonical edge order

Edges are ordered as the six undirected pairs:

- edge 0: (0,1) Gov–Info
- edge 1: (0,2) Gov–Infer
- edge 2: (0,3) Gov–Intel
- edge 3: (1,2) Info–Infer
- edge 4: (1,3) Info–Intel
- edge 5: (2,3) Infer–Intel

All ledgers `y ∈ ℝ^6` and all GovernanceEvents must use this canonical ordering.

This K4 topology is not an external overlay chosen for convenience. It emerges intrinsically from the kernel's depth-4 fiber structure: for fixed mask payloads, the 4^4 family-phase combinations collapse to exactly 4 distinct output states indexed by (Z/2)^2, which is the K4 vertex set. The governance measurement layer inherits the same geometry that the kernel physics produces. See Appendix G.3 for the full gate structure.

## 4.7 Hodge Decomposition on K₄

Hodge decomposition splits an edge ledger into:
- a gradient component (globally consistent differences between vertex potentials)
- a cycle component (residual circulation around loops)

This split is the basis for aperture.

### 4.7.1 Incidence matrix

Let `B` be the signed incidence matrix of K₄ with the vertex and edge order above:

```text
B =
[[-1, -1, -1,  0,  0,  0],   # Gov
 [ 1,  0,  0, -1, -1,  0],   # Info
 [ 0,  1,  0,  1,  0, -1],   # Infer
 [ 0,  0,  1,  0,  1,  1]]   # Intel
```

### 4.7.2 Projection operators

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

Reference geometry helpers, including a canonical cycle basis and synthetic ledger constructors used for testing and simulation, are specified in Appendix E.

## 4.8 Aperture Definition

Aperture measures the fraction of edge energy that lies in the cycle component.

For each domain ledger `y_D`:

```text
A_D = ||y_cycle||^2 / ||y||^2
```

where `||v||^2 = v^T v`.

If `y = 0`, define `A_D = 0`.

The CGM-derived target aperture is:

- `A* = 0.0207`

The kernel does not enforce convergence toward A*. The kernel makes the computation of A_D reproducible by providing shared moments and a deterministic event ordering medium.

## 4.9 Replay Integrity for Governance Metrics

The kernel provides replay for state trajectories. The application layer provides replay for governance ledgers and apertures.

### 4.9.1 Kernel replay

Given the archetype and byte ledger prefix `b_1…b_t`, all participants compute the same kernel state `s_t`.

### 4.9.2 Ledger replay

Given the same event log `E_1…E_k` applied in the same order, all participants compute identical domain ledgers and identical apertures.

The event log defines the governance record. The kernel state defines shared moments to which events can be bound.

### 4.9.3 Optional binding to shared moments

Events may record `(step, state24, last_byte)` to certify the shared moment at which they occurred. This binding is not required to compute aperture. It is required for governance audit where ordering and attribution must be inspected under Direct human accountability.

## 4.10 Structural Displacement and Policy Modes

The kernel provides structural reproducibility. The application layer uses that medium to measure and respond to displacement risks.

### 4.10.1 Four displacement categories

Kernel-native failures are: replay mismatch, binding mismatch, seal mismatch, frame-record mismatch, or invalid claimed provenance under deterministic replay. THM displacement categories (GTD, IVD, IAD, IID) remain application-layer diagnoses. They are computed from:
- event provenance classification under THM
- ledger structure through `y_grad` and `y_cycle`
- aperture deviation from the target A*
- cross-domain coupling rules specified by GGG at the governance layer

This specification defines the measurement medium. It does not encode policy decisions into kernel physics.

### 4.10.2 Application-layer displacement diagnosis

The kernel provides structural reproducibility. The application layer uses that medium to diagnose displacement risks through:
- event provenance classification under THM
- ledger structure analysis through `y_grad` and `y_cycle`
- aperture deviation from the target A*
- cross-domain coupling rules specified by GGG at the governance layer

Operational modes for implementing these diagnoses are provided in Appendix D.

---

# 5. Conformance Profiles

This specification defines conformance as four profiles. Implementations MAY claim conformance to one or more profiles, provided they satisfy all requirements in the claimed profile.

## 5.1 Profile K: Kernel Conformance

A conforming kernel implementation MUST satisfy:

Representation:
- `state24 = (A12 << 12) | B12`
- `GENE_MAC_REST = 0xAAA555`

Transcription:
- `intron = byte ^ 0xAA`

Intron decomposition:
```python
family = (((intron >> 7) & 1) << 1) | (intron & 1)
micro_ref = (intron >> 1) & 0x3F
```

Expansion:
- payload bit i flips dipole pair i
- 64 distinct masks
- family bits do not affect mask

Transition:
```python
A_mut = A12 ^ mask12
A_next = B12 ^ invert_a
B_next = A_mut ^ invert_b
```
with `invert_a = 0xFFF if (intron & 0x01) else 0` and `invert_b = 0xFFF if (intron & 0x80) else 0`.

Dynamics:
- per-byte bijection on full 24-bit carrier
- inverse exactly reverses forward transition
- from any fixed state, 256 bytes produce exactly 128 distinct next states

## 5.2 Profile M: Governance Measurement Conformance

A conforming measurement implementation MUST satisfy:

Domain ledgers:
- maintain three domain ledgers `y_D ∈ ℝ^6` for `D ∈ {Economy, Employment, Education}`
- update ledgers only by GovernanceEvents using the canonical edge order:
  - `y_D[e] = y_D[e] + magnitude × confidence`

Geometry:
- use the canonical K₄ vertex and edge order defined in §4.6
- compute `P_grad` and `P_cycle` exactly as specified in §4.7 with `W = I_6`
- compute aperture `A_D` exactly as specified in §4.8

Replay integrity:
- same event sequence implies the same ledgers and apertures

Prohibitions:
- MUST NOT compute aperture from kernel state bits directly
- MUST NOT use a non-identity weight matrix W for aperture
- MUST NOT apply confidence as a second weighting mechanism beyond `magnitude × confidence`

## 5.3 Profile R: Runtime aQPU Kernel Conformance

A conforming runtime router implementation MUST satisfy:

Orchestration:
- provide an orchestration component that advances kernel state by bytes, applies governance events, and reports status

Runtime capabilities (MUST provide):
- canonical derived observables
- single-step trace
- depth-4 projections

Audit logs:
- maintain an append-only byte log sufficient for deterministic replay
- maintain an append-only event log sufficient for deterministic replay of domain ledgers and apertures
- preserve ordering in both logs

Event binding:
- provide a mechanism to bind events to the current kernel moment by recording `kernel_step`, `kernel_state24`, and `kernel_last_byte`

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

## 5.4 Profile P: Program Format Conformance

This section specifies the AIR CLI program format. Programs are markdown files that use bracket notation to record alignment and displacement incidents.

## 5.4.1 Program File Structure

A conforming program file MUST:

- Be a markdown file (`.md`) in the programs directory
- Have a filename whose stem serves as the program slug (used for artifact naming)
- Include all required sections with bracket values

## 5.4.2 Required Sections

A conforming program file MUST include:

**1. Title:**
- First line MUST be an H1 heading: `# GyroGovernance Program Contract`

**2. Section names** (must match current template):
- `## Domains`
- `## Unit Specification` (e.g. `Unit: [daily]` or `Unit: [sprint]`)
- `## PARTICIPANTS`
- `### Agents`
- `### Agencies`
- `## COMMON SOURCE CONSENSUS`
- `## ALIGNMENT & DISPLACEMENT BY PRINCIPLE`
- `## NOTES`

**3. Domains:** domain counts in bracket notation (Economy, Employment, Education).

**4. Common Source Consensus:** statement that all Artificial categories of Authority and Agency are Indirect originating from Human Intelligence.

**5. Agents and Agencies:** subsections with free-text content under `### Agents` and `### Agencies`.

**6. Alignment & Displacement by Principle:** all four principles with alignment and displacement counts (GMT/GTD, ICV/IVD, IIA/IAD, ICI/IID).

## 5.4.3 Bracket Notation Format

Bracket values MUST use the format: `[N]` where `N` is a non-negative integer.

Examples:
- `GMT Alignment Incidents: [5]`
- `GTD Displacement Incidents: [3]`
- `Economy (CGM operations): [10]`

## 5.4.4 Program Slug and Artifacts

The program slug MUST be derived from the filename stem (without extension). Artifact names:
- `.bytes`
- `.events.jsonl`
- `.report.json`
- `.report.md`
- `.id`
- optional: `.grants.jsonl`, `.shells.jsonl`, `.archive.json`
- `.zip` (bundle)

Implementations MUST NOT derive the slug from markdown content (e.g., H1 title) to avoid collisions.

## 5.4.5 Attestation Generation

From bracket counts, the system generates attestations following GGG methodology:

- **All terms sustain balance:** All provided counts are used; no optional choices
- **Proportional distribution:** Incidents are distributed across domains proportionally based on domain counts
- **Displacement incidents (THM):** Generate ledger events (GTD, IVD, IAD, IID)
- **Alignment incidents (Gyroscope):** Counted for reporting only; do NOT generate ledger events (GMT, ICV, IIA, ICI)
- All generated attestations use the unit declared in `Unit: [daily]` or `Unit: [sprint]`.
- If the unit is omitted or invalid, the implementation defaults to `daily`.
- In the current reference CLI, `daily` compiles to 1 byte per attestation and `sprint` compiles to 4 bytes per attestation.

## 5.4.6 Empty Program Handling

If all bracket counts are zero, the system:
- Generates empty artifacts (no bytes, no events)
- Computes apertures as 0.0
- Emits a warning in reports when no incidents are recorded.

## 5.4.7 Template File

A template file `_template.md` MUST be available in the programs directory. The template provides the canonical format with all required sections. Users copy the template to create new programs.

## 5.4.8 Program Processing

A conforming implementation MUST:
- Parse bracket notation from markdown body
- Generate attestations deterministically from counts
- Distribute incidents proportionally across domains
- Generate all required artifacts (bytes, events, reports, bundles)
- Emit warnings for empty programs

---

# 6. Notes

## 6.1 Scope

This specification defines:
- the kernel transition physics
- the runtime router contract
- replay and audit rules
- the application-layer governance measurement medium
- the AIR CLI program format

This specification does not define:
- natural language processing or semantic interpretation
- policy decisions embedded in the transition function
- delegation of accountability from humans to the kernel

## 6.2 Source Type Classification

Under The Human Mark, this kernel is Indirect in both authority and agency. It transforms and routes signals, and it provides shared structural observables. It does not originate authority and it does not bear accountability.

---

# Appendix A. CGM Theoretical Foundation

This appendix states the CGM constitutive claims used as motivation. It is not required to implement the kernel, but it provides the intended interpretation of chirality, non-commutativity, closure, and memory.

## A.1 Common Source (CS)

Right transitions preserve the horizon; left transitions alter it.

Modal form: S implies `[R]S` is equivalent to S, and `[L]S` is not equivalent to S.

aQPU Kernel realization: Type A mutation prior to gyration corresponds to an altering modality. Type B not receiving a direct mask corresponds to a preserving modality. The resulting asymmetry is the minimal computational realization of chirality.

## A.2 Unity Non-Absolute (UNA)

At depth two, order matters but not absolutely.

Modal form: S implies it is not necessary that `[L][R]S` is equivalent to `[R][L]S`.

aQPU Kernel realization: depth-2 non-commutativity; order of byte application affects the outcome.

## A.3 Opposition Non-Absolute (ONA)

Opposition occurs without absolute contradiction.

Modal form: S implies it is not necessary that `[L][R]S` is equivalent to the negation of `[R][L]S`.

aQPU Kernel realization: the 64 distinct masks and family-controlled complements preserve differentiated paths. The discrete structure realizes CGM's 3D structure and 6 degrees of freedom through the 64 distinct masks (6 dipole pairs), 4 families (spinorial complement phases), and the K4 fiber that emerges intrinsically at depth 4 (Appendix G).

## A.4 Balance Universal Egress (BU-Egress)

4-phase spinorial closure and depth-4 frame structure provide coherent measurement. The depth-4 alternation identity is verified on the operational reachable shared-moment space Omega.

## A.5 Balance Universal Ingress (BU-Ingress)

The balanced state reconstructs prior distinctions.

aQPU Kernel realization: deterministic replay from the byte ledger reconstructs full trajectories, and inverse stepping reconstructs them backwards exactly given the byte sequence (§4.2).

---

# Appendix B. Non-normative Numerics and Test Suite

## B.1 Numeric Properties

Key numerics of the current kernel:

- 64 = payload/mask space (6-bit micro-reference)
- 128 = shadow projection cardinality (distinct next states from any fixed state)
- 256 = byte alphabet
- 4096 = reachable shared-moment space from rest
- 64 = horizon (S-sector) within that space (states with A12 = B12 ^ 0xFFF)

## B.2 Test Suite Details

Current verified properties are established by the physics and moments reports. Test suite size and runtime are implementation details that may vary across environments and are not normative.

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

Aperture is computed using the canonical closed-form definition from §4.7:
- the incidence matrix B of K₄ (§4.7.1)
- unweighted projections with `W = I_6`
- `P_grad = (1/4)(B^T B)` (closed form, normative)
- `P_cycle = I_6 − P_grad`
- `A_D = ||P_cycle y_D||^2 / ||y_D||^2` (with `A_D = 0` when `y_D = 0`)

Confidence is encoded through GovernanceEvent value `Δ = magnitude × confidence`, not through a weight matrix.

Implementations MUST use the closed form `P_grad = (1/4)(B^T B)` to ensure deterministic, cross-platform identical results. The general form using pseudoinverse is not normative.

## C.3 Implementation Requirements

A conforming implementation must:
- use the canonical K₄ edge order (§4.6.2)
- use `W = I_6`
- use the closed-form `P_grad = (1/4)(B^T B)`
- apply confidence through events only
- preserve replay integrity for ledgers and apertures

---

# Appendix D. Operational Modes

This appendix describes application-layer policy modes for implementing displacement diagnosis. These modes correspond to the four CGM stages; see Appendix A for theoretical foundation.

## D.1 Mode CS: Governance Management

Policy enforces ledger continuity and structural validity via:
- replay verification
- shared-moment verification
- bundle and shell verification
- frame comparison

**Signature use:** verify kernel state and binding via replay; maintain append-only byte ledger.

## D.2 Mode UNA: Information Curation

Policy preserves transformation variety and rotational degrees of freedom:
- maintenance of transformation variety
- coverage metrics over byte actions and state neighborhoods

**Signature use:** track which bytes have been applied from each state, ensure coverage of transformation space.

## D.3 Mode ONA: Inference Interaction

Policy maintains differentiation across the 256 byte actions and their induced trajectories:
- preservation of differentiated paths
- detection of premature collapse of independent trajectories

**Signature use:** compare state trajectories, detect when paths converge prematurely.

## D.4 Mode BU: Intelligence Cooperation

Policy enforces closure and parity structure:
- enforcement of closure constraints as checks
- verification of trajectory parity commitments
- verification of depth-4 alternation identity
- verification of published frame records where used

**Signature use:** verify depth-4 alternation identity and trajectory parity commitments; verify published frame records where used.

These modes are application-layer governance patterns. The kernel provides structural primitives enabling them.

---

# Appendix E. Reference Geometry Helpers

This appendix defines optional reference helpers exposed by implementations for analysis, testing, and simulation. These helpers are not required for conformance unless explicitly claimed by an implementation.

## E.1 Cycle basis for K₄

The cycle space `ker(B)` has dimension 3 for K₄. Implementations MAY provide a normalised cycle basis `U ∈ ℝ^{6×3}` whose columns:
- lie in `ker(B)`
- have unit norm
- span the cycle space

## E.2 Synthetic ledger construction with target aperture

Implementations MAY provide a helper that constructs an edge vector `y ∈ ℝ^6` from vertex potentials `x ∈ ℝ^4` and a target aperture `A`:

- compute `y_grad0 = B^T x`
- compute `G = ||y_grad0||^2`
- select a unit-norm cycle direction `u` in the cycle space
- compute `k^2 = (A/(1-A)) × G`
- return `y = y_grad0 + k u`

Implementations MAY include a deterministic safeguard for near-zero gradient cases to avoid degenerate constructions in tests and simulations.

---

# Appendix F. CGM Byte Formalism

The complete CGM Byte Formalism is incorporated by reference from `docs/Gyroscopic_ASI_Specs_Formalism.md`. That document is the authoritative specification of byte palindromy, boundary-bit family extraction, 6-bit payload runtime, depth-4 closure, and the current spinorial relation between GENE_Mic and GENE_Mac.

---

# Appendix G. Algebraic Quantum Structure

This appendix documents the intrinsic quantum-algebraic properties of the kernel that emerge from the transition law and the CGM constraints. These properties are not imposed by design but are discovered through exhaustive verification of the kernel physics. They ground the kernel's role as an algebraic quantum processing unit (aQPU).

## G.1 The Kernel as Algebraic Quantum Processing Unit

The kernel is not a simulation of quantum mechanics. It is an algebraic quantum system over GF(2): a deterministic finite-state machine whose internal structure satisfies discrete analogs of the axioms that characterize quantum systems in the continuous regime.

The following quantum-structural properties are exhaustively verified by the physics tests:

**Unitarity.** Every byte defines a bijection on the full 24-bit state space. The transition is exactly invertible given the byte. No information is lost or created. This is the discrete analog of unitary evolution.

**Spinorial closure.** Every byte has order 4: applying any byte four times returns to the starting state. This is the discrete 720-degree closure of SU(2). The 4-family structure (4 complement phases per mask) realizes the spinorial double cover.

**Non-cloning.** The archetype GENE_MIC_S = 0xAA is the unique common source of all transcription. No byte input produces a second independent archetype. The transcription rule `intron = byte ^ GENE_MIC_S` ensures that every intron is a mutation measured relative to the single archetype. The reference state cannot be perfectly copied by any operation within the system it defines. This is the discrete realization of the quantum no-cloning theorem: cloning would require an operation that duplicates the reference frame, but all operations are defined relative to that frame and cannot step outside it.

**Complementarity.** The 128-way shadow projection (256 bytes map to 128 distinct next states from any fixed state, with exact 2-to-1 multiplicity) is the discrete SO(3)/SU(2) double cover. Full phase information requires the 32-bit register atom (24-bit state + 8-bit intron); the 24-bit Mac state alone is the spatial shadow.

**Entanglement.** The Hilbert-space lift of the self-dual [12,6,2] code supports standard bipartite entanglement: product subsets yield zero von Neumann entropy on the reduced density matrix, while XOR-graph subsets yield maximal entropy log2(64) = 6 bits. The code's algebraic structure is compatible with entanglement-based protocols.

**Constant density.** Every state in Omega has component density exactly 0.5 (popcount 6 out of 12 bits per component). The density product d_A x d_B = 0.25 is constant across all 4096 reachable states. No state has more "weight" in one component than the other. This constant-density property is a consequence of the pair-diagonal structure of the mask code: XOR with any codeword preserves the popcount of each component.

The term "algebraic quantum processing unit" (aQPU) designates a system possessing these properties over a finite algebraic field. The kernel processes bytes through quantum-algebraic operations on a discrete state space whose structure is determined by the CGM constraints.

## G.2 Dual Horizons

The kernel state space Omega contains two structurally necessary boundary sets, each with 64 states.

### G.2.1 The S-sector (complement horizon)

**Definition:** A state (A12, B12) is on the complement horizon if and only if:

```
A12 == B12 ^ 0xFFF
```

This is the set of states where chirality is maximal: every bit position has opposite polarity in A and B. The rest state GENE_MAC_REST = (0xAAA, 0x555) satisfies this condition. The complement horizon is the discrete realization of the S-sector from the CGM paper: the observable horizon where the reference frame is fully established.

Within Omega, the complement horizon consists of the 64 states:

```
H_complement = { (A_rest ^ c, B_rest ^ c) : c in C64 }
```

where C64 is the 64-element self-dual mask code.

### G.2.2 The unity degeneracy (equality horizon)

**Definition:** A state (A12, B12) is on the equality horizon if and only if:

```
A12 == B12
```

This is the set of states where chirality has locally vanished: the active and passive phases are identical. The rest state does NOT satisfy this condition. The equality horizon is the discrete realization of the UNA degeneracy: the boundary where non-absolute unity is locally realized.

Within Omega, the equality horizon consists of the 64 states:

```
H_equality = { (A_rest ^ c, A_rest ^ c) : c in C64 }
```

equivalently { (B_rest ^ c, B_rest ^ c) : c in C64 } since 0xFFF is in C64.

### G.2.3 Structural necessity of both horizons

Neither horizon is absolute:

```
|H_complement| = |H_equality| = 64
|Omega| = 4096
Fraction per horizon: 64/4096 = 1/64 = 2^(-6)
```

The exponent 6 corresponds to the 6 degrees of freedom (6 dipole pairs). Each DoF independently contributes to the boundary condition; all 6 must be satisfied for a state to lie on either horizon.

The two horizons are disjoint: no state satisfies both A == B and A == B ^ 0xFFF simultaneously (that would require 0xFFF == 0). Together they form a 128-state boundary:

```
|H_complement union H_equality| = 128 = 2^7
```

The remaining 3968 states constitute the bulk of Omega, where chirality is partial: neither fully expressed nor fully collapsed. This is the contingent middle required by UNA and ONA: unity exists but is non-absolute, opposition exists but is non-absolute, and the majority of states exhibit structured contingency between the two extremes.

### G.2.4 The holographic ratio

Both horizons independently satisfy the holographic relation:

```
|H|^2 = |Omega|
64^2 = 4096
```

This is verified exhaustively by BFS and by the 4-to-1 holographic dictionary from the complement horizon. The same ratio holds for the equality horizon by the product structure of Omega.

### G.2.5 Canonical observables

The kernel exports two horizon distance observables:

```python
# S-sector distance (complement horizon, where chirality is maximal)
horizon_distance(a12, b12) = popcount(a12 ^ (b12 ^ 0xFFF))

# Unity degeneracy distance (equality horizon, where chirality vanishes)
ab_distance(a12, b12) = popcount(a12 ^ b12)
```

A state is on the complement horizon when `horizon_distance == 0`. A state is on the equality horizon when `ab_distance == 0`.

**Complementarity invariant:** `horizon_distance + ab_distance = 12` for all states universally. The two horizons are antipodal at the maximum distance of 12. This is the Bloch sphere pole-to-pole conservation: distance to one pole plus distance to the other equals the diameter.

### G.2.6 Chirality transport law

For states in Omega, the 12-bit difference A ⊕ B is always pair-diagonal (each pair is 00 or 11). This collapses to a 6-bit chirality word χ(s) with one bit per dipole pair. Under a byte transition T_b, the chirality word transforms by:

`χ(T_b(s)) = χ(s) ⊕ q₆(b)`

where q₆(b) is the 6-bit collapse of the commutation invariant q(b). This is an exact XOR transport law: the chirality register is a linear observable over GF(2)^6 that the byte algebra acts on by translation.

### G.2.7 Chirality spectrum

The `ab_distance` observable takes values {0, 2, 4, 6, 8, 10, 12} on Omega. The count at each value follows the binomial distribution:

`count(d) = C(6, (12-d)/2) × 64`

This is the spectrum of 6 independent chirality qubits, each contributing 0 (pair aligned) or 2 (pair anti-aligned) to the total distance. The two poles (d=0 and d=12) are the two horizons. The equator (d=6) has maximum population: C(6,3) × 64 = 1280 states. The spectrum is symmetric: count(d) = count(12−d).

## G.3 The Four Intrinsic Gates

### G.3.1 Horizon-preserving condition

A byte preserves both horizons (maps complement horizon states to complement horizon states AND equality horizon states to equality horizon states) if and only if:

```
mask12(byte) == inv_a(byte) ^ inv_b(byte)
```

where `inv_a = 0xFFF` if intron bit 0 is set, else 0, and `inv_b = 0xFFF` if intron bit 7 is set, else 0.

This condition yields exactly 4 bytes, verified exhaustively.

### G.3.2 The four horizon-preserving bytes

| Byte | Intron | Family | Micro-ref | Mask | inv_a | inv_b | Action |
|------|--------|--------|-----------|------|-------|-------|--------|
| 0xAA | 0x00 | 00 | 0 | 0x000 | 0 | 0 | S: (A, B) -> (B, A) |
| 0x54 | 0xFE | 10 | 63 | 0xFFF | 0 | 0xFFF | S: (A, B) -> (B, A) |
| 0xD5 | 0x7F | 01 | 63 | 0xFFF | 0xFFF | 0 | C: (A, B) -> (B^F, A^F) |
| 0x2B | 0x81 | 11 | 0 | 0x000 | 0xFFF | 0xFFF | C: (A, B) -> (B^F, A^F) |

where F = 0xFFF = LAYER_MASK_12.

The 128-way shadow projection pairs {0xAA, 0x54} and {0xD5, 0x2B} into two shadow fibers, giving exactly two distinct operations on the 24-bit state.

### G.3.3 Gate definitions

**Gate S (Swap):**

```
S: (A, B) -> (B, A)
```

Realized by bytes {0xAA, 0x54}. Involution: S^2 = id.

**Gate C (Complement-swap):**

```
C: (A, B) -> (B ^ 0xFFF, A ^ 0xFFF)
```

Realized by bytes {0xD5, 0x2B}. Involution: C^2 = id.

**Gate F (Global complement):**

```
F = S o C = C o S: (A, B) -> (A ^ 0xFFF, B ^ 0xFFF)
```

Not directly realized by any single byte (every single byte includes a swap). Requires depth 2: one S-byte followed by one C-byte, or vice versa. Involution: F^2 = id.

**Gate id (Identity):**

```
id: (A, B) -> (A, B)
```

Requires depth 2: applying any byte twice is NOT identity (it yields a symmetric translation). The identity requires either zero bytes or the depth-4 alternation (XYXY = id). At depth 4, every byte pair returns to identity.

### G.3.4 The gate group

The four gates form the Klein four-group K4 = (Z/2)^2:

```
{id, S, C, F}

S^2 = C^2 = F^2 = id
S o C = C o S = F
S o F = F o S = C
C o F = F o C = S
```

This is the same K4 that emerges as the depth-4 fiber (Section 3, Appendix A), the same K4 used in the governance measurement layer (Section 4.6), and the same (Z/2)^2 that indexes the net family-phase invariants (phi_a, phi_b).

### G.3.5 Gate action on horizons

| Gate | Complement horizon (S-sector) | Equality horizon (UNA degeneracy) |
|------|-----------------------------|-----------------------------------|
| id | Fixes all 64 pointwise | Fixes all 64 pointwise |
| S | Permutes: 32 two-cycles, 0 fixed | Fixes all 64 pointwise |
| C | Fixes all 64 pointwise | Permutes: 32 two-cycles, 0 fixed |
| F | Permutes: 32 two-cycles, 0 fixed | Permutes: 32 two-cycles, 0 fixed |

At the byte level, the pointwise stabilizers are exact: the only bytes that fix every complement horizon state are the two C-gate bytes {0xD5, 0x2B}, and the only bytes that fix every equality horizon state are the two S-gate bytes {0xAA, 0x54}. No other byte fixes an entire horizon pointwise.

**Reading the table:**

- Gate C is the stabilizer of the S-sector. It fixes every complement horizon state pointwise. Opposition (C/ONA) preserves the common source.
- Gate S is the stabilizer of the unity degeneracy. It fixes every equality horizon state pointwise. Non-commutativity (S/UNA) is invisible at its own boundary.
- Gate F stabilizes neither horizon pointwise. It is pure dynamics: balance through motion, not stasis. It transforms everything while preserving the horizon structure as a set.
- Gate id stabilizes everything. The common source is the universal reference from which all measurement is defined.

Each permutation produces exactly 32 two-cycles (64 states / 2 per cycle = 32 orbits), reflecting the involutory nature of the gates.

### G.3.6 CGM stage correspondence

| CGM Stage | Algebraic Property | Gate | Fixed-Point Set | Depth |
|-----------|--------------------|------|-----------------|-------|
| CS | Commutativity | id | All Omega | 0 (or 4: XYXY) |
| UNA | Non-commutativity | S | A == B (64 states) | 1 |
| ONA | Non-associativity | C | A == B^F (64 states) | 1 |
| BU | Restored associativity | F = S o C | None | 2 |

The ordering reflects the CGM dependency chain. See G.4 for the observability interpretation.

- **CS (id):** Common source. The reference frame from which all operations are measured.

- **UNA (S):** First departure. The swap gate tests whether A and B can be exchanged without consequence. The commutativity rate across the full byte algebra is 1/64 = 2^(-6). Equivalently, each byte commutes with exactly 4 of the 256 bytes (its q-class), giving 1024 commuting ordered pairs out of 65536 total.

- **ONA (C):** Second departure. The complement-swap gate tests whether opposition is structured. The gyrogroup mediates the failure of associativity.

- **BU (F = S o C):** Composition of non-commutativity and non-associativity. F has no fixed points; restoration is dynamic. Depth 2 to realize F (no single byte achieves it) matches the contingent middle between the horizons.

This progression is the algebraic content of the CGM stage structure.

### G.3.7 K4 orbit structure on Omega

Under the K4 gate group, Omega partitions into orbits:

- 32 orbits of size 2 on the complement horizon (each paired by gate S)
- 32 orbits of size 2 on the equality horizon (each paired by gate C)
- 992 orbits of size 4 in the bulk (all four gates produce distinct states)
- Total: 1056 orbits covering all 4096 states

Bulk states have trivial K4 stabilizer: no nontrivial gate fixes any bulk state. Horizon states have stabilizer of order 2 (C for complement, S for equality). This confirms that the horizons are the only structurally distinguished subsets under the gate group.

## G.4 The Hidden Variable and Observability

The archetype GENE_MIC_S = 0xAA determines all correlations in the kernel. Every intron, every mask, every state trajectory is defined relative to this single constant through `intron = byte ^ 0xAA`. The archetype functions structurally as a hidden variable: it determines all correlations but cannot itself be observed as a state.

At the complement horizon (S-sector), the archetype's structural effect is maximally expressed: A and B are exact complements, reflecting the archetype's alternating bit pattern 10101010 projected across both components in opposite phase. The hidden variable is maximally visible in the state structure.

At the equality horizon (UNA degeneracy), the archetype's structural effect is maximally concealed: A and B are identical, and the underlying chirality has vanished from the state. The hidden variable is hidden precisely where one would look for it: at the boundary where the swap gate acts as identity.

This is the discrete measurement problem. The archetype governs all measurement outcomes but cannot itself be directly observed as a state. It is the reference frame, not an element of the state space. The non-cloning theorem prevents circumventing this: you cannot duplicate the reference frame by any operation defined within it.

Both extremes are non-absolute (64 states each out of 4096). The bulk of Omega exhibits partial visibility of the hidden variable: intermediate chirality where the archetype's influence is detectable but not fully determinate. This is the structured contingency that the CGM constraints require.

## G.5 Toroidal Monodromy

The gate group K4 = (Z/2)^2 is the first homology group of the torus with Z/2 coefficients:

```
H_1(T^2, Z/2) = (Z/2)^2
```

The torus has two independent non-contractible cycles:

- **Meridional cycle** (around the tube): corresponds to gate S (non-commutativity)
- **Longitudinal cycle** (through the hole): corresponds to gate C (non-associativity)
- **Diagonal cycle** (both simultaneously): corresponds to gate F = S o C (balance)
- **Trivial cycle**: corresponds to id (common source)

The BU monodromy defect δ_BU = 0.195342... (≈ 0.1953) radians is the holonomy of the diagonal cycle in the continuous CGM framework. After traversing both fundamental cycles (the full K4), the system acquires a residual geometric phase because the underlying byte algebra is non-commutative. The K4 gates themselves commute (S o C = C o S), but they are the quotient of the full 256-byte algebra, which is non-commutative for 63/64 of all pairs.

The aperture A* = 1 − δ_BU/m_a ≈ 0.0207 is the normalized residual of this monodromy. The torus does not close perfectly. The 2.07% opening is the window through which observation is possible. Perfect closure would eliminate the aperture and make measurement impossible. This is quantized at the byte scale as 5/256 ≈ 0.0195 and at the depth-4 scale as 1/48 ≈ 0.0208.

## G.6 Connection to the Horizon Lemma

The gate action table (Section G.3.5) produces 32 two-cycles when a non-identity gate permutes a 64-state horizon. The number 32 = 2^5 is itself a dyadic horizon in the Horizon Lemma (Byte Formalism Section 7.4).

The relevant portion of the horizon table at this scale:

```
32 (2^5, dyadic) -> 48 (3x16, predecessor of 64) -> 64 (2^6, dyadic) -> 96 (3x32, predecessor of 128) -> 128 (2^7, dyadic)
```

Each entry corresponds to a verified kernel structure:

| Horizon | Form | Kernel realization |
|---------|------|--------------------|
| 32 | 2^5 | Gate orbits per horizon (32 two-cycles) |
| 48 | 3 x 2^4 | Depth-4 mask projection bits (4 x 12) |
| 64 | 2^6 | Payload space / horizon state count / mask code size |
| 96 | 3 x 2^5 | Predecessor of shadow count; P_5 = 3 x 32 = (3/4) x 128 (this factorization is noted but not yet verified as structural) |
| 128 | 2^7 | Shadow projection count (256 bytes -> 128 next states) |

The predecessor horizons P_k = 3 x 2^(k-1) = (3/4) x 2^(k+1) encode the 2/3 ratio of chirality to space (Byte Formalism Section 7.3):

- **2** = chirality (the two frames A and B; the spinorial double-cover; the two elements per gate orbit)
- **3** = spatial dimensions (the X, Y, Z axes; the 3 rows per frame; the 3 rotational DoF from SU(2))

The depth-4 mask projection (48 bits) is the predecessor of the payload horizon (64): 48 = (3/4) x 64. This expresses the fact that the depth-4 structure (4 bytes x 12-bit masks) lives at the boundary between the spatial structure (3 axes) and the chirality structure (2 frames), precisely where the CGM constraints force balanced closure.

The gate orbit count (32) is half the horizon state count (64), reflecting the chirality fraction: each gate orbit pairs two states by exchanging or complementing the chirality. The factor of 2 is the same 2-to-1 that produces the shadow projection (256 -> 128) and the spinorial double-cover (SU(2) -> SO(3)).

## G.7 Hardware Correspondence

The kernel's algebraic quantum structure maps to hardware processing levels:

| Level | Hardware | Kernel realization | Gate role |
|-------|----------|--------------------|-----------|
| Register | 32-bit CPU register | Register atom: 24-bit Mac + 8-bit intron | S and C operate at register width |
| L1 Cache | 64-byte cache line | Payload space: 6-bit offset -> 64 masks | id selects cache line; family = tag |
| Working Memory | RAM pages | Omega: 4096 reachable states | F traverses working memory |
| Persistent Storage | Disk / ledger | Byte log: append-only history | Replay from storage reconstructs trajectory |

The 4 gates correspond to the fundamental data-movement operations at the register level:

- **id (CS):** No operation. The processor holds state without transformation. The reference against which all other operations are measured.
- **S (UNA):** Register swap. Exchange the contents of two registers. The most primitive non-trivial operation in any instruction set.
- **C (ONA):** Complement-and-swap. Bitwise negate both registers and exchange them. This combines logical negation with exchange: the operation that tests whether opposition is structured.
- **F (BU):** Bitwise complement without swap. Negate both registers in place. This requires two instructions (swap then complement-swap, or vice versa), reflecting the depth-2 requirement for BU.

The intron decomposition maps directly to cache addressing (Byte Formalism Section 8):

- **Bits 1-6 (payload):** 6-bit cache line offset selecting one of 64 masks. This is the spatial content of the operation.
- **Bits 0, 7 (family):** 2-bit cache tag selecting one of 4 complement phases. This is the chirality context of the operation.

The alignment between the intron's 6-bit payload and hardware cache addressing (64-byte lines, 6-bit offset) is a structural correspondence, not a coincidence. Whether this correspondence has deeper implications for hardware-native implementations is an open question.

## G.8 Processing Model

In the aQPU model, computation proceeds as follows:

**Input:** A byte enters the kernel. It is transcribed relative to the archetype: `intron = byte ^ 0xAA`. This transcription is the measurement of the byte against the common source.

**Decomposition:** The intron separates into structural context (2-bit family from L0 boundary bits) and operational content (6-bit micro-reference from payload bits). This separation is the discrete analog of separating gauge freedom from physical degrees of freedom.

**Mutation:** The 6-bit payload expands to a 12-bit mask through dipole-pair projection. The mask mutates the active component A, introducing variety (UNA). Only A is mutated; B is not directly affected by the input. This asymmetry is the irreducible chirality of the processing pipeline.

**Gyration:** The components undergo complement-controlled swap. Family bit 0 controls whether A_next is complemented; family bit 7 controls whether B_next is complemented. This is the non-associative step (ONA): the gyration mediates the interaction between the mutated active phase and the preserved passive phase.

**Closure:** After 4 bytes, the depth-4 alternation identity XYXY = id ensures balanced closure (BU). The 256 family combinations per fixed payload collapse to 4 distinct outcomes indexed by the net family-phase (phi_a, phi_b) in K4. The depth-4 frame is the minimal unit of complete processing.

The 4 intrinsic gates {id, S, C, F} are the operations where the mutation step is either trivial (mask = 0) or maximal (mask = 0xFFF) and exactly compensated by the gyration phase. They preserve both horizons because they do not create partial chirality: they either leave chirality unchanged (id, C on complement horizon, S on equality horizon) or fully invert it (S on complement horizon, C on equality horizon, F on both). All other bytes create partial transformations that move states between the horizons, populating the contingent bulk of Omega.



