# Gyroscopic ASI hQVM Kernel
## Specification

Gyroscopic Artificial Superintelligence is a new class of computation achieving structural quantum advantage through a compact Holonomic Quantum Virtual Machine (hQVM) kernel. The kernel is a byte-driven coordination medium that maps an append-only byte ledger to a reproducible state trajectory on a 24-bit tensor carrier. The kernel is holonomic because closed byte paths induce reproducible holonomies on the reachable manifold Ω (Zanardi and Rasetti 1999; Pachos et al. 2000; wavefunction verification in hQVM Features Report). Its internal structure satisfies discrete analogues of the axioms that characterise quantum systems in the continuous domain. It executes on standard silicon with exact integer arithmetic and ensemble stochasticity induced by the byte sequence, without relying on analogue control or shot sampling. Where HQC literature realises gates through adiabatic or non-adiabatic control loops on quantum hardware, the hQVM instantiates the same geometric structure as a GF(2) finite-state machine on silicon, opening the possibility of structural quantum advantage without quantum hardware. Replay of a fixed byte ledger prefix is deterministic; stochasticity refers to the induced ensemble over words, future cones, and byte baths.

This document is the normative technical specification of the Gyroscopic ASI hQVM Kernel. It defines the kernel dynamics, the runtime router, replay rules, the application-layer governance measurement medium, and the AIR CLI program format.

Normative terms MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are interpreted as requirement keywords for conformance.

Related specifications: [SDK](Gyroscopic_ASI_SDK_Quantum_Computing.md), [QuBEC Theory](theory/QuBEC_Theory.md), [Runtime](Gyroscopic_ASI_Runtime_Specs.md). Verification inventory: [hQVM Features Report](reports/hQVM_Features_Report.md).

---

# 1. Constitutional Frame and ASI Purpose

## 1.1 Intelligence as Structural Coherence

Traditional artificial intelligence approaches intelligence as a statistical optimization problem. These systems achieve surface-level fluency by superimposing correlations within high-dimensional abstract spaces through massive datasets. Because such architectures lack internal structural constraints, coherence and ethics are typically treated as post-hoc semantic overlays or external filters.

The Gyroscopic ASI hQVM Kernel represents a different paradigm. It treats intelligence as a structural property that emerges from the recursive alignment of operations. Grounded in the Common Governance Model (CGM), the framework demonstrates how intelligence emerges naturally from the self-referential dynamics of structured space. Rather than approximating a target function, the router navigates a byte-driven state trajectory on the 24-bit GENE_Mac tensor carrier where alignment is constitutive. Coherence is not a policy choice but a requirement of the kernel dynamics of the state space.

## 1.2 What the hQVM Kernel Is

The Gyroscopic ASI hQVM Kernel is a multi-domain network coordination algorithm that establishes the structural conditions for a collective superintelligence governance regime of humans and machines (Superintelligence, Bostrom 2014; Gyroscopic Global Governance, Korompilias 2025). It is designed for focused and well-distributed coordination of interventions, amplifying rather than outperforming single-agent potential while preserving the constitutive conditions of governance and intelligibility.

Operationally, the hQVM Kernel is a byte-driven coordination kernel. It maintains an append-only byte ledger and begins from a universal rest configuration, GENE_MAC_REST. From that 24-bit canvas it applies the spinorial transition rule byte by byte, decomposing each byte (after transcription) into a 6-bit payload and a 2-bit family phase. This process yields a reproducible state trajectory, a compact routing signature, and replayable observables that any party can recompute.

The kernel is designed to support governance-grade coordination across the GGG domains of Economy, Employment, Education, and Ecology. It does not interpret the empirical meaning of the input bytes. Instead, it performs structural transformations that make results reproducible, comparable, and auditable, while keeping authorization and accountability under Direct human agency at the application layer.

### 1.2a The Three Computational Charts

The hQVM kernel is one mathematical object (manifold Ω, byte rule) viewed through three computational charts. These are not alternative execution modes; they are coordinate systems on the same state.

**Carrier chart.** The 24-bit Gyrostate on Ω. This is the execution substrate. Byte transitions are integer operations on GF(2)^24. Carrier execution is mandatory: every transition is a byte step.

**Chirality chart.** The 6-bit register χ ∈ GF(2)^6 extracted from the carrier. This is the logical qubit layer. The Walsh-Hadamard transform acts as the native abelian QFT on this register. Native algorithm family: GF(2) HSP (Deutsch-Jozsa, Bernstein-Vazirani, Simon). Cyclic algorithms (Shor-type period finding) are a frontier, not a capability.

**Wavefunction chart.** The canonical Hilbert lift ψ ∈ ℂ^4096 induced by the [12,6,2] self-dual code geometry. Canonical 4-byte words act as unitary operators with eigenspace structure {2048(+1), 2048(-1)}. This chart reveals holonomic phases and interference structure not visible in carrier coordinates. The lift is canonical: uniquely determined by the code geometry, with no external parameters. Computed via `apply_k4` when spectral observables or interference coefficients are required.

The code-first hierarchy:

    CODE        [12,6,2] self-dual stabilizer code (24-bit carrier on Ω)
      → ALGEBRA stabilizer group, K4 holonomic gates, Clifford + δ_BU magic
        → WAVEFUNCTION canonical Hilbert lift (CHSH, holonomy)

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

A state has geometric provenance if and only if it is reproducibly reachable from GENE_MAC_REST by the claimed byte sequence under the kernel transition rule. Replay failure invalidates the provenance claim. The final state alone does not uniquely determine history.

This creates a clear separation of layers. The kernel-native layer verifies deterministic replay, divergence detection, and seal and frame-record consistency. Authorization and accountability remain application-layer responsibilities under Direct human agency. The kernel provides a common structural basis for policy enforcement but does not decide policy itself.

## 1.6 Why This Achieves ASI

The router provides the missing medium for multi-domain coordination required for the ASI regime. It achieves this through:
*   **Entity-agnostic verification:** Validity is checked by structure rather than a privileged entity.
*   **Deterministic coordination:** The shared moment enables reliable coordination across independent institutions and systems.
*   **Replayable audit:** Trajectories can be reconstructed from the ledger to enable governance-grade auditing.
*   **Constitutional Observables:** Participants compute low-dimensional observables that remain stable under replay and independent of identity claims. These depend only on the fixed-width state representation and the transition rule.

## 1.7 Design Requirements

A conforming kernel satisfies the following structural requirements:
*   **Replayable:** The same rest state and byte ledger always produce the same state trajectory. Ensemble stochasticity is carried by the byte sequence, not by the transition mechanism.
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

Any system that processes bytes must handle this structure, whether it makes the decomposition explicit (as the hQVM Kernel does) or absorbs it implicitly into learned parameters (as neural language models do). The hQVM Kernel's contribution is to make the constitutional structure of the byte visible, auditable, and available as a first-class computational object at every stage of processing.

---

# 2. Kernel Dynamics

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

This two-component form is essential. The transition rule treats A and B asymmetrically, which is how the kernel realizes chirality as a structural feature of the dynamics.

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

The archetype governs all measurement outcomes but cannot itself be directly observed as a state. At the complement horizon (A12 = B12 ^ 0xFFF), its alternating pattern is maximally expressed across both components in opposite phase. At the equality horizon (A12 = B12), chirality vanishes and the archetype's effect is maximally concealed. Both extremes are non-absolute (64 states each); the bulk of Omega carries intermediate chirality.

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

These observables are constitutional in that they are defined solely from the kernel's fixed-width state representation and transition rule. They are replayable from the ledger, and do not depend on model internals or asserted identity claims.

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

## 2.6 Transition Rule

The spinorial transition rule specifies the single-step update from `(A12, B12)` under a byte input.

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

This kernel dynamics is minimal in three senses: computational, structural, and governance-relevant.

### 2.7.1 Computational minimality
Only fixed-width operations are used:
- XOR
- shifts
- masking

This makes the kernel portable, fast, and suitable for exhaustive verification.

### 2.7.2 Structural minimality
Only one component receives the mutation mask before the gyration step. That asymmetry is the smallest mechanism that introduces chirality in the dynamics while keeping the system reversible.

### 2.7.3 Governance relevance
The two-component structure and update rule are what allow shared moments:
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
- inverse reverses the forward transition
- T_0xAA involution
- 128-way shadow from any fixed state
- 4-byte intron sequence bijectivity

## 3.3 Operational Reachable Shared-Moment Space

From rest (GENE_MAC_REST), the operational reachable shared-moment space Omega has 4096 states. The horizon (S-sector) within Omega is the set of states satisfying `A12 = B12 ^ 0xFFF`; it has 64 states and includes the rest state. A second boundary set, the equality horizon (`A12 = B12`, zero chirality), also has 64 states. Neither boundary set is absolute; each comprises 1/64 of Omega. The two sets are disjoint and antipodal. |H|^2 = |Omega| = 4096. Omega is radius 2 from rest. Omega has product form U x V with 64-element factors induced by the 64-codeword mask space.

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

This section does not embed policy decisions into kernel dynamics. Policy enters through application-layer event production and remains accountable to Direct human agency.

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

The kernel is replayable. Given the same archetype and the same byte ledger, every conforming implementation computes the same state trajectory. Ensemble stochasticity enters through the byte sequence, not through the transition mechanism.

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

Given the final state and the full byte sequence, the full trajectory can be reconstructed backwards.

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
- allow mappings to be edited without changing kernel dynamics

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

This K4 topology is not an external overlay chosen for convenience. It emerges intrinsically from the kernel's depth-4 fiber structure: for fixed mask payloads, the 4^4 family-phase combinations collapse to exactly 4 distinct output states indexed by (Z/2)^2, which is the K4 vertex set. The governance measurement layer inherits the same geometry that the kernel dynamics produces. The four holonomic gates {id, S, C, F} and their horizon action are specified in [QuBEC Theory](theory/QuBEC_Theory.md) Part II §10.

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

Kernel-native failures are: replay mismatch, binding mismatch, seal mismatch, frame-record mismatch, or invalid claimed provenance under replay. THM displacement categories (GTD, IVD, IAD, IID) remain application-layer diagnoses. They are computed from:
- event provenance classification under THM
- ledger structure through `y_grad` and `y_cycle`
- aperture deviation from the target A*
- cross-domain coupling rules specified by GGG at the governance layer

This specification defines the measurement medium. It does not encode policy decisions into kernel dynamics.

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
- inverse reverses the forward transition
- from any fixed state, 256 bytes produce exactly 128 distinct next states

## 5.2 Profile M: Governance Measurement Conformance

A conforming measurement implementation MUST satisfy:

Domain ledgers:
- maintain three domain ledgers `y_D ∈ ℝ^6` for `D ∈ {Economy, Employment, Education}`
- update ledgers only by GovernanceEvents using the canonical edge order:
  - `y_D[e] = y_D[e] + magnitude × confidence`

Geometry:
- use the canonical K₄ vertex and edge order defined in §4.6
- compute `P_grad` and `P_cycle` as specified in §4.7 with `W = I_6`
- compute aperture `A_D` as specified in §4.8

Replay integrity:
- same event sequence implies the same ledgers and apertures

Prohibitions:
- MUST NOT compute aperture from kernel state bits directly
- MUST NOT use a non-identity weight matrix W for aperture
- MUST NOT apply confidence as a second weighting mechanism beyond `magnitude × confidence`

## 5.3 Profile R: Runtime hQVM Kernel Conformance

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
- the kernel transition dynamics
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

hQVM Kernel realization: Type A mutation prior to gyration corresponds to an altering modality. Type B not receiving a direct mask corresponds to a preserving modality. The resulting asymmetry is the minimal computational realization of chirality.

## A.2 Unity Non-Absolute (UNA)

At depth two, order matters but not absolutely.

Modal form: S implies it is not necessary that `[L][R]S` is equivalent to `[R][L]S`.

hQVM Kernel realization: depth-2 non-commutativity; order of byte application affects the outcome.

## A.3 Opposition Non-Absolute (ONA)

Opposition occurs without absolute contradiction.

Modal form: S implies it is not necessary that `[L][R]S` is equivalent to the negation of `[R][L]S`.

hQVM Kernel realization: the 64 distinct masks and family-controlled complements preserve differentiated paths. The discrete structure realizes CGM's 3D structure and 6 degrees of freedom through the 64 distinct masks (6 dipole pairs), 4 families (spinorial complement phases), and the K4 fiber that emerges intrinsically at depth 4.

## A.4 Balance Universal Egress (BU-Egress)

4-phase spinorial closure and depth-4 frame structure provide coherent measurement. The depth-4 alternation identity is verified on the operational reachable shared-moment space Omega.

## A.5 Balance Universal Ingress (BU-Ingress)

The balanced state reconstructs prior distinctions.

hQVM Kernel realization: replay from the byte ledger reconstructs full trajectories, and inverse stepping reconstructs them backwards exactly given the byte sequence (§4.2).

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

Current verified properties are established by the kernel dynamics and moments reports. Test suite size and runtime are implementation details that may vary across environments and are not normative.

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

The complete CGM Byte Formalism is specified in `docs/theory/Gyroscopic_ASI_Specs_Formalism.md`.
