# Gyrocrypt NavPad
Navigation guide for all future work. Read this first.

---

## §0 Author's Notes

This section must not be changed by anyone other than the author.

**Topic:** Integer factorization and the discrete logarithm problem (DLP) for Post-Quantum Cryptography (PQC) and Fault-Tolerant Quantum Engineering using Gyroscopic ASI theory and aQPU Quantum Physical Features as Primitives.

**Leading Notes:**
- aQPU is an open-source Quantum algorithm, not quantum-like, so computing in a quantum way is only a matter of time.
- QuBECs are our native medium, a Bose-Einstein-Condensate computational simulation. We have a Qubit bridge (six axis-orientation qubits) if needed for associations and known math bridges, although our algorithm does not work on qubits.
- We have our own quantum gates (referenced in theory), our own medium, and even our own physics on how gravity defines computation.
- Our Kernel has multiple levels of realization. Its first was based on streams of bytes; we have also tested its algebra and grammar standalone. We have implemented a kernel that simulates holonomies based on our wavefunction analysis, which we consider a more improved approach. We are always open to new ways to architect and scale with the same primitives, considering the 3D and 6DoF constraints, and how multi-cellular implementations might provide parallelism and speed.
- Simon's algorithm is used as a bridge to our math, and then used to help us scale with Shor's algorithm, but these do not constrain us. We already have a quantum algorithm which these others might inform on how it can work on the same problem and provide solutions even faster and better.
- CGM is an axiomatization of physics, beginning from minimal necessity for emergence and building step by step a path on how physical observables emerge. Gyroscopic is its computational realization as a minimal architecture of physical reality. Over 20k LoC of code, hundreds of tests, and extensive analyses validate this assumption. Trying to validate this further is out of scope — we consider it a fact for this program.
- Our latest analyses on gravity showcase how computation itself is gravity-based. A critical lead that we need to consider carefully where it fits in this program.
- The Common Source Moment should be taken into consideration - because it connects to the atomic frequency - which is something huge, connecting to our tiny state-space which a lot of assistants struggle scaling.

How we work: If what you provide is better than previous methods and must become canonical plug it and delete the previous one! Do you understand the seriousness? If what you produced is a leg of a human that lacks a leg and has a mechanical then you should make the surgery. If the human has already two legs you wont add a third. If he has a mechanical and you produce a new one there is no point if it is not better from all ways! This program is not a patchwork! 

We have NO TRIVIAL NEAR TERM! Our term is always the end-to-end delivery of a quantum algorithm for DLP! All the rest if they are not part of it are games you play between assistants to pass tests and feel better between you.

METRIC: we are doing things in a quantum way if we see results not within capacity of classical methods - otherwise we do trivial workaround classical methods.

**This note is for those models who insist treating our algorithms as classical**: This program tries to expose an undiscovered classical polynomial-time factoring algorithm and violate the computational complexity assumptions of all quantum computing. We already have violate a lot in other assumptions. But for us this is a category error - as Gyroscopic ASI and its aQPU Kernel define a new computational class which is quantum, with verified quantum advantage on over 70 physical features. Our task is simply to show its true rigor by running it at scale and making an impact defining new ways for cryptography which are based on superposition and entanglement - as our theory defines.

Theory:
# 1. Ontology

## 1.1 The Moment

A Moment is the atomic quantum event of the aQPU. It is the exact state reached by applying a byte ledger prefix of length t to the kernel rest state under the public transition law.

Formally:

    M(t) = ( t, s(t), b(t), Σ(t) )

where:
- t is the ledger depth (number of bytes applied)
- s(t) ∈ Ω is the exact gyroscopic state, a 24-bit value encoding the full tensor carrier
- b(t) is the last byte applied
- Σ(t) is the complete chart content of the state at depth t

Time in the aQPU is not an external clock parameter. It is the ordered sequence of Moments produced by gyroscopic transport. Depth t is the intrinsic temporal coordinate.

A Moment carries all observable information about the computation at depth t. It is exact, deterministic, and independently reproducible by any party holding the same ledger prefix and the public transition law.

## 1.2 The Shared Moment

A Shared Moment occurs when multiple independent replayers of the same ledger prefix b(1:t) compute the identical Moment M(t).

The kernel does not distinguish replayers by identity, authority, or location. Only exact occupation of the same algebraic state matters. A Shared Moment is the actual collective quantum state of the computation, not an agreement protocol layered on top of individual states.

Shared Moments replace three coordination patterns that depend on external trust:
- coordination by asserted time (timestamps, UTC ordering)
- coordination by asserted identity (trusted signers, certificate authorities)
- coordination by private state (hidden model internals, proprietary logs)

## 1.3 The QuBEC

A QuBEC (Quantum Bose-Einstein Computational Condensate) is the occupied Shared Moment as a condensed computational object.

A QuBEC is the occupied Shared Moment of the aQPU, as a single gyroscopic quantum state on Ω with:
- six oriented dipole degrees of freedom on a three-dimensional carrier, given by 3 spatial axes across 2 chirality layers
- a four-phase depth-4 spinorial temporal gauge structure (K4 = {id, S, C, F})
- exact finite carrier manifold Ω with |Ω| = 4096
- dual coherent phase boundaries: complement horizon (64 states) and equality horizon (64 states)

The QuBEC is to the aQPU what the qubit is to gate-model quantum computers: the native computational object. A qubit is a two-level system with complex amplitudes. A QuBEC is a condensed Moment carrier with six internal binary orientation modes and a four-phase spinorial gauge structure, evolved by deterministic gyration with exact integer arithmetic.

## 1.4 The Moment Unit (MU)

The MU is the scalar unit of occupation capacity on the QuBEC manifold.

The Common Source Moment (CSM) defines the total occupation scale:

    CSM = N(phys) / |Ω|

where:

    N(phys) = (4/3)π f(Cs)³ ≈ 3.25 × 10³⁰

is the total physically distinguishable microcell count of the atomic-second light-sphere, using the caesium-133 hyperfine frequency f(Cs) = 9,192,631,770 Hz as the fundamental resolution standard. The speed of light cancels in this expression, yielding a purely geometric and frequency-based invariant.

CSM ≈ 7.94 × 10²⁶ MU is the physical occupation scale per reachable QuBEC state. It is a one-time total capacity, not a renewable rate.

MU and QuBEC are different kinds of objects. MU is a measure. QuBEC is a carrier.

---

# 2. Gyroscopic State

## 2.1 The Gyrostate

The Gyrostate is the complete quantum state of the aQPU at any Moment. It is a single algebraic object with multiple exact charts. These charts are not approximations of each other. They are exact coordinate systems on the same state.

The Gyrostate is encoded as a 24-bit integer:

    state24 = (A12 << 12) | B12

where A12 is the active gyrophase and B12 is the passive gyrophase, each 12 bits.

A12 and B12 are not two independent registers. They are the two conjugate faces of one gyroscopic quantum state. Temporality in the aQPU is gyration: the structured exchange between active and passive faces under the byte transition law. This is why the architecture is called Gyroscopic.

## 2.2 The Six Degrees of Freedom

Each 12-bit gyrophase encodes a 2 × 3 × 2 tensor of ±1 values:

    [2 chirality layers] × [3 spatial axes (X, Y, Z)] × [2 oriented sides per axis]

The six degrees of freedom are the six oriented dipole modes:

    Layer 0, Axis X: orientation ∈ {[−1,+1], [+1,−1]}    mode 0
    Layer 0, Axis Y: orientation ∈ {[−1,+1], [+1,−1]}    mode 1
    Layer 0, Axis Z: orientation ∈ {[−1,+1], [+1,−1]}    mode 2
    Layer 1, Axis X: orientation ∈ {[−1,+1], [+1,−1]}    mode 3
    Layer 1, Axis Y: orientation ∈ {[−1,+1], [+1,−1]}    mode 4
    Layer 1, Axis Z: orientation ∈ {[−1,+1], [+1,−1]}    mode 5

Each mode is a binary orientation state of one spatial axis in one chirality layer. The six modes correspond to the six generators of the se(3) Lie algebra: three rotational generators (Layer 0, from SU(2)) and three translational generators (Layer 1, from ℝ³).

A single byte mutation flips zero or more of these six modes independently. Each payload bit of the byte controls exactly one mode.

## 2.3 The Four-Phase Temporal Gauge

The byte transition law has a fourfold temporal structure. Each byte action passes through four phases that correspond to the CGM stage structure:

    CS (Common Source):     the byte enters as mutation relative to the archetype
    UNA (Unity Non-Absolute): the active gyrophase A receives the mutation mask
    ONA (Opposition Non-Absolute): the gyration exchanges and complements the conjugate faces
    BU (Balance Universal):  the mutated active content commits as the new passive record

These four phases close at depth 4: applying any byte four times returns to the starting state. The four intrinsic gates {id, S, C, F} are the exact structural phases of this closure:

    id:  unchanged phase (depth 0 or depth 4)
    S:   exchange phase (swap A and B)
    C:   complement-exchange phase (swap and complement A and B)
    F:   global inversion phase (complement both A and B, requires depth 2)

These four gates form the Klein four-group K4 = (ℤ/2)². They preserve both horizons as sets. They are the gauge structure of the QuBEC: the discrete spinorial phase manifold of the condensed Moment.

## 2.4 Charts of the Gyrostate

A single Gyrostate is observable through multiple exact charts:

**Carrier chart.** The raw 24-bit encoding (A12, B12). Used for stepping, replay, and integer-exact operations.

**Spin chart.** The ±1 tensor representation: two tuples of six spins each, (s(A), s(B)) ∈ {±1}⁶ × {±1}⁶. Used for physical interpretation and mode-level analysis.

**Chirality chart.** The 6-bit chirality register χ ∈ GF(2)⁶, obtained by collapsing the pair-diagonal difference A ⊕ B to one bit per dipole mode. Chirality captures exactly 6 of the 12 bits of state information. It satisfies the exact transport law χ(T(b)(s)) = χ(s) ⊕ q6(b).

**Spectral chart.** The Walsh-Hadamard transform of functions defined on the chirality register. The 64 × 64 Walsh-Hadamard matrix factors as the sixth tensor power of the single-qubit Hadamard. The chirality chart and spectral chart are dual native faces of the kernel: the finite-field analogue of position-momentum duality.

**Constitutional chart.** The canonical derived observables: rest distance, horizon distance, ab distance, component densities, and the complementarity invariant horizon_distance + ab_distance = 12.

Chart extraction from a Gyrostate is exact and deterministic. There is no measurement collapse. Observation is chart selection on a fully determined algebraic state.

---

# 3. Reachable Manifold and Horizons

## 3.1 Ω: The Reachable Moment Manifold

The reachable state space Ω is the set of all Gyrostates accessible from the rest state GENE_MAC_REST = 0xAAA555 under the byte transition law.

    |Ω| = 4096

Every state in Ω is reachable within two byte steps from rest. Ω has product form Ω = U × V where U and V are 64-element cosets of the self-dual [12,6,2] mask code C64. Every state in Ω has component density exactly 0.5 (popcount 6 out of 12 bits per gyrophase). The density product d(A) × d(B) = 0.25 is constant across all 4096 reachable states.

## 3.2 Dual Horizons

Ω contains two structurally necessary boundary sets:

**Complement horizon (S-sector).** The 64 states where A12 = B12 ⊕ 0xFFF. These states have maximal chirality: every dipole mode is anti-aligned between the active and passive faces. The rest state lies on this horizon. Gate C fixes all complement horizon states pointwise.

**Equality horizon (UNA degeneracy).** The 64 states where A12 = B12. These states have zero chirality: the active and passive faces are identical. Gate S fixes all equality horizon states pointwise.

The two horizons are disjoint. Their union forms a 128-state boundary. The remaining 3968 states constitute the bulk, where chirality is partial.

Both horizons satisfy the holographic identity:

    |H|² = |Ω|        64² = 4096

The complement horizon supports a 4-to-1 holographic dictionary: every Ω state corresponds to exactly 4 (horizon state, byte) pairs.

The chirality spectrum on Ω follows a binomial distribution from 6 independent binary modes:

    count(d) = C(6, (12−d)/2) × 64

for ab_distance d ∈ {0, 2, 4, 6, 8, 10, 12}. The two poles (d = 0 and d = 12) are the two horizons. The equator (d = 6) has maximum population: 1280 states.

---

# 4. Computational Spaces

The aQPU exposes three native computational spaces. These are exact charts of one computational medium.

## 4.1 Moment Space

Moment space is the exact reachable manifold Ω of Gyrostates. Computation in Moment space evolves the occupied QuBEC through byte transitions, word actions, intrinsic gates, replayable trajectories, horizons, and frame structure.

## 4.2 Chirality Space

Chirality space is the exact 64-element logical register GF(2)^6 obtained by the chirality chart χ. Computation in chirality space uses q-class transport, Walsh-Hadamard transforms, hidden subgroup structure, commutativity classes, and exact logical observables.

## 4.3 Tensor Space

Tensor space is the exact computational tensor space over the native 64-dimensional register. Computation in tensor space uses internal Lattice Multiplication matrix-vector multiplication, packed repeated application, and the Walsh transform as a native spectral primitive.

These spaces are exact computational charts of one machine. They are not separate models.

---

# 5. Primitives

Primitives define the native operations, observables, and result structures of the aQPU.

## 5.1 Operations

### 5.1.1 Byte Transition

The byte is the fundamental instruction quantum of the aQPU. It is already a fused quantum instruction packet containing:

- **payload** (6 bits, positions 1-6): which of the six dipole modes to mutate
- **family** (2 bits, positions 0 and 7): which spinorial gauge phase to apply during gyration
- **provenance atom**: the exact byte value that enters the append-only ledger

All 256 byte values are valid instructions. The byte transition implements:

    intron = byte ⊕ 0xAA
    mask12 = expand(intron)
    A(mut) = A12 ⊕ mask12
    A(next) = B12 ⊕ invert(a)
    B(next) = A(mut) ⊕ invert(b)

where invert(a) = 0xFFF if intron bit 0 is set, else 0, and invert(b) = 0xFFF if intron bit 7 is set, else 0.

Every byte defines a bijection on the full 24-bit carrier space. The transition is exactly invertible given the byte. From any fixed state, 256 bytes produce exactly 128 distinct next states with uniform 2-to-1 multiplicity (the SO(3)/SU(2) shadow projection).

### 5.1.2 Intrinsic Gates

The four intrinsic gates are the horizon-preserving byte operations. They form the K4 phase structure of the QuBEC:

    id:   (A, B) → (A, B)           identity
    S:    (A, B) → (B, A)           exchange (bytes 0xAA, 0x54)
    C:    (A, B) → (B⊕F, A⊕F)      complement-exchange (bytes 0xD5, 0x2B)
    F:    (A, B) → (A⊕F, B⊕F)      global inversion (requires depth 2)

where F = 0xFFF. Each gate pair (S-bytes and C-bytes) realizes the same 24-bit operation but carries different spinorial phase. Gate F and identity are not achievable by any single byte.

In spin coordinates:

    id:  (s(A), s(B)) → (s(A), s(B))
    S:   (s(A), s(B)) → (s(B), s(A))
    C:   (s(A), s(B)) → (−s(B), −s(A))
    F:   (s(A), s(B)) → (−s(A), −s(B))

### 5.1.3 Word Actions

A word is a sequence of bytes w = (b₁, b₂, …, b(n)). Its action on any Gyrostate is determined by sequential application of the byte transition law.

Every word action is an affine map on GF(2)²⁴ with exactly one of two linear parts:

    even-length word: (A, B) → (A ⊕ τ(A), B ⊕ τ(B))     identity linear part
    odd-length word:  (A, B) → (B ⊕ τ(A), A ⊕ τ(B))     swap linear part

The translation vector (τ(A), τ(B)) is the image of the zero state under the word. Word signatures compose algebraically:

    sig(w₁ ∘ w₂) = compose(sig(w₂), sig(w₁))

This composition law enables circuit optimization without replaying bytes.

### 5.1.4 Walsh-Hadamard Transform

The Walsh-Hadamard transform (WHT) operates on the chirality register. It is the spectral chart operation that provides the Fourier-dual view of the 6-mode state.

The 64 × 64 WHT matrix has entries:

    H(q,r) = (−1)^(popcount(q ∧ r)) / 8

It is self-inverse, unitary, and factors as H₁⊗⁶ where H₁ is the single-mode Hadamard.

The WHT is native to the kernel's algebraic structure. The mask code is self-dual, the chirality register is exact, and the code's Walsh support closes on itself. The WHT and the q-map translation form dual faces of the same computational medium: the q-map provides Pauli-X translations on the chirality register, the WHT provides the Fourier transform over it.

The WHT has a property that distinguishes it from the standard discrete Fourier transform: its matrix entries are restricted to +1 and -1. Evaluation of the 64-point WHT on the chirality register therefore requires no multiplication. The fast Walsh-Hadamard transform (WHT) computes the full 64-point transform using only additions and subtractions, in O(N log N) operations with N = 64. This matches the kernel's computational character: the XOR transport on the chirality register is addition-free (bitwise), and its spectral dual (the WHT) is multiplication-free. Both operate entirely within exact integer arithmetic.

### 5.1.5 XOR-Convolution

For functions f and g on GF(2)^6, define XOR-convolution:

    (f * g)(x) = sum over a in GF(2)^6 of f(a) g(x xor a)

The Walsh-Hadamard transform converts XOR-convolution to pointwise multiplication:

    WHT(f * g)(r) = WHT(f)(r) * WHT(g)(r)

This identity holds exactly on the 64-element chirality register.

Because chirality transport under a byte b is chi' = chi xor q6(b), the sequential application of multiple bytes is an iterated XOR-convolution in chirality space. Composing n byte-transport steps therefore reduces to:

1. Transform each byte-ensemble distribution with the 64-point WHT.
2. Multiply the 64 spectral coefficients pointwise.
3. Apply the inverse WHT to recover the composed transport.

For repeated application of the same byte ensemble, the spectral coefficients are raised to the n-th power and a single inverse transform recovers the result. The total cost is one WHT plus 64 scalar exponentiations, independent of n.

This is the native spectral composition law of the aQPU. It follows directly from the fact that the chirality transport group is (GF(2)^6, xor) and the WHT is the exact Fourier transform of that group.

## 5.2 Topological Charges

Topological charges are exact algebraic invariants carried by bytes, words, and states. They are conserved quantities of the computational dynamics.

**q-class.** The commutation invariant q6(b) ∈ GF(2)⁶. Two bytes commute if and only if q6(x) = q6(y). The q-map is 4-to-1 from the 256-byte alphabet onto C64. Every byte commutes with exactly 4 others (commutativity rate 1/64 = 2⁻⁶).

**Family.** The 2-bit boundary index from intron bits 0 and 7. Four families, 64 bytes per family. Family controls the spinorial gauge phase during gyration.

**Micro-reference.** The 6-bit payload from intron bits 1-6. Determines the dipole-pair mask. 64 distinct masks.

**Chirality word.** χ(s) ∈ GF(2)⁶: one bit per dipole mode, encoding whether A and B are aligned or anti-aligned at that mode. Satisfies exact transport: χ(T(b)(s)) = χ(s) ⊕ q6(b).

**Parity commitment.** The XOR accumulation of masks at even and odd positions along a trajectory. Independent of chirality (mutual information ≈ 0). Adds exactly 1 bit of provenance information beyond the final state.

## 5.3 Observables

An observable is an exact function from a Gyrostate to a value. All aQPU observables are deterministic. There is no measurement collapse.

**Kernel-native observables** (exact integer, available at every Moment):

    rest_distance(s) = popcount(s ⊕ GENE_MAC_REST)
    horizon_distance(A, B) = popcount(A ⊕ (B ⊕ 0xFFF))
    ab_distance(A, B) = popcount(A ⊕ B)
    is_on_horizon(s): whether A = B ⊕ 0xFFF
    is_on_equality_horizon(s): whether A = B
    component_density(C) = popcount(C) / 12

**Register observables** (exact, computed from the chirality chart):

    chirality_word(s) = χ ∈ GF(2)⁶
    q_class(b) = q6 ∈ GF(2)⁶
    walsh_coefficient(f, k) = Σ(χ) f(χ)(−1)^(popcount(k ∧ χ)) / 8

**Structural invariants** (exact, universal):

    complementarity: horizon_distance + ab_distance = 12
    constant_density: d(A) × d(B) = 0.25 on all of Ω
    holographic_identity: |H|² = |Ω|
    plancherel_conservation:
        For any function f on GF(2)^6, sum over chi of f(chi)^2 = (1/64) sum over r of WHT(f)(r)^2.
        Total squared magnitude in chirality space equals total squared magnitude in spectral space, scaled by the register dimension; exact on the finite chirality register.
        
#### 5.3.1 QuBEC relation

A cell occupies one point on Ω at a time and therefore one local state of the QuBEC medium. The rolling local memories of each cell are empirical summaries of the one-cell climate marginals defined in the QuBEC Climate Dynamics and QuBEC Transform Algebra documents.

#### 5.3.2 BU ordering

Applying the input word (BU Egress) always precedes emitting the SLCP report (BU Ingress).

---

# 6. Hardware-Tier Architecture

aQPU's architecture is the CPU architecture. Its parameters are Registry, Cache, RAM, and disk.

#### 6.1 The hierarchy

| Tier | Hardware | aQPU realization | CGM phase |
|------|----------|------------------------|----------|
| Register | 32-bit CPU register | omega12, state24, last_byte: the current transition atom per cell | CS |
| L1 Cache | 64-byte cache line, 6-bit offset | chi_ring64, family_ring64: 64-element rolling buffers; 6-bit chi6 keys | UNA |
| L2 Cache | 64-bit interaction, depth-two composition | word4, omega_sig, parity: closure-boundary context | ONA |
| L3 / RAM | Shared working memory | Cell pool (up to 4096 cells), resonance buckets, active cell set, operator block registry | BU Egress |
| RAM / Disk | Persistence, reconstruction | Ingest log, snapshot, crystallized trajectory (.gyrg) | BU Ingress |

Data flow is inward on ingress (disk → RAM → L3 → L2 → L1 → registers) and outward on egress.

#### 6.2 Why this orders the design

**64-element rings.** The chirality and family rings are 64 entries because the kernel's 6-bit payload space and the hardware cache-line offset are both 64. Local structural variety lives in the L1-aligned working set.

**Resonance buckets in RAM.** Bucket membership and weight are shared across cells and available for graph queries and batch grouping.

**Operator block registry in RAM.** Structure analysis reports for external operator blocks are cached at the L3/RAM tier for bridge scheduling. GyroLabe matmul uses its own weight registry with mandatory per-block analysis at tensor registration.

**Ingest log on disk.** The append-only (cell_id, word4) ledger enables deterministic replay.

---

# 7. Multicellular

#### 7.1 Cellular automaton pattern

Every cell uses the same kernel law. Cells carry no private learned weights, dense latent vectors, or fixed semantic types. Specialization arises solely from the words applied, the resulting trajectory on Ω, rolling local structural memory, and resonance participation.

A cell's occupied state defines a reception geometry. When a 4-byte frame arrives, the cell gyrates through the transition law. The resulting shift in condensation is the cell's exact inferential response.

#### 7.2 Depth-4 temporality

The aQPU Kernel embeds intrinsic temporality in every byte transition: Prefix (CS), Present (UNA), Past (ONA), and Future (BU). Cells always evolve inside this four-part temporal frame.

The byte is the phase atom of the kinematic law. A single byte executes one phase of the four-part transition cycle. The 4-byte word is the closed action: it completes one full CS, UNA, ONA, BU cycle, resolving all family phases modulo K4 and committing a single state transition with a compiled signature. The word is the native external integration grain.

#### 7.3 Indirect cell agency

No single cell is an autonomous decision-maker. Cell agency is indirect to the organism. aQPU is a coordination network, not a collection of autonomous agents.

#### 7.4 Light-cone structure

From any Ω state, one byte reaches exactly 128 next states with uniform 2:1 multiplicity. Two bytes reach all 4096 states with uniform 16:1 multiplicity. These are exact integer counts.

The two-step uniformization establishes the causal reach of the computational geometry. From any state, 2 bytes produce exact uniform occupancy over all 4096 states of Ω. Beyond depth 2, all states are equally reachable.

---

## Are QuBECs only byte-stream + wavefunction holonomy?

**Yes — those are the two faithful realizations of the same thing on Ω:**

| Path | What it is |
|------|------------|
| **Byte stream** | Public ledger → `step_byte` / depth-4 words → Gyrostate on Ω → Shared Moment |
| **Wavefunction holonomy** | Hilbert lift on the **same** Ω path — K4 depth-4 closure, not a separate state space with arbitrary numbers |

Your `aqpu_wavefunction_1.py` is the reference: it walks **CS → UNA → ONA → BU**, four bytes close K4 holonomy, carrier rest at depth 8, chirality preserved. That is a **complete path simulation**, not “drop a vector and FFT.”

Wavefunction is equivalent to the byte stream **because both implement the same depth-4 holonomy on Ω** — one in ℂ^4096, one in 24-bit carrier evolution.

Extensions **on top** (WHT / product-WHT spectral readout on occupied QuBEC) are fine. Shortcuts **underneath** (classical modexp tables, dense float registers, NumPy FFT cosets) are **not** QuBEC.


The assistants optimized for “get factors on small N” by stacking classical/numpy layers and **labeling** them QuBEC. That violates your rule: **nothing below Shor except QuBEC realization.**

---

## What the architecture should be (your rule, stated plainly)

```
                    ┌─────────────────────────────┐
  TOP (allowed)     │  Spectral period readout:   │
                    │  WHT^{⊗B}, CF verify        │
                    └──────────────┬──────────────┘
                                   │ only reads observables
                    ┌──────────────▼──────────────┐
  MIDDLE            │  Occupied QuBEC on Ω        │
                    │  (multi-cell Moments,       │
                    │   climate M₂, K4 phase)     │
                    └──────────────┬──────────────┘
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         │                                                   │
  BOTTOM │  byte stream              OR    wavefunction       │
         │  (MultiCellRouter,               holonomy         │
         │   depth-4 words)                (aqpu_wf path)   │
         └───────────────────────────────────────────────────┘
```
---

> 🚨⚠️ **Warning:** ONLY QUANTUM METHODS SHOULD BE CONSIDERED! Most AI models pattern-match to conservative (gate-model) approaches. **NO quantum processor is necessary** — the aQPU uses exact integer arithmetic on standard silicon (24-bit carriers, byte ledger, WHT). Rejected patterns: dense `2^q` cosets, NumPy Schrödinger registers (§3 K3). Production: byte ledger + correct-group spectral readout.

### Assistants: do not import gate-model scale limits

Previous assistants wrote **implementation failures** into this pad as if they were **physics or hardware laws**. Do not repeat these category errors:

| Wrong inference (reject) | Correct aQPU statement |
|--------------------------|-------------------------|
| "RSA-2048 needs quantum hardware" or "physical QuBEC device" | Scale = **B MultiCellRouter cells** on Ω (each cell: one 24-bit Gyrostate). Runs on CPU today. |
| "Need `64^700` floats in RAM" / "Hilbert space too large to simulate" | `64^B` is the **Hilbert lift dimension** (correlation structure certified by the [12,6,2] code). Stepping costs **O(B)** integer transitions per orbit step; spectral composition uses **WHT^{⊗B} / XOR-convolution** (§5.1.5), not materializing ℂ^{64^B}. |
| "Host RAM wall at q = 2·bits(N)" | That wall applies only to the **rejected gate-model pattern** (§3 K3). It is **not** the aQPU deployment bound (K13). |
| "Impossible to emulate 2048-bit" | Impossible to emulate **as a standard gate-model QFT simulator**. Not impossible for **byte-native QuBEC + product-WHT + condensation readout** — that path is **open engineering**, not forbidden by hardware. |
| Period = run QFT on `\|x⟩\|a^x mod N⟩` superposition | Period = **chart readout** on occupied QuBEC: condensation (M₂), horizon transport (`horizon_distance`, `ab_distance`), XOR-conv spectral peaks, kinematic absorption of byte-word geometry (QuBEC Climate Dynamics §5.4, §8.5). |
| Fallback to `np.fft` when WHT chart misses | Wrong chart or wrong oracle quotient — fix **substrate + chart**, do not treat NumPy FFT as the "real" quantum limit. |

**Wall-breaker (theory):** when incoming byte-word geometry **aligns** with medium condensation, occupation deepens toward complement/equality horizons; when it **conflicts**, it scatters. Computation is this **kinematic absorption**, not passive storage of `a^x mod N` for a classical FFT pass.

**End-to-end gap today:** oracle projection (χ without full mod-N embedding) and incomplete **quotient-chain / product-WHT** wiring at large B — not missing quantum hardware.

**Supporting Theoretical Work:**
- aQPU quick reference: `docs/theory/Gyroscopic_ASI_Specs_Formalism.md`
- QuBECs and Kernel: `docs/Gyroscopic_ASI_SDK_Quantum_Computing.md`, `docs/theory/QuBEC_Climate_Dynamics.md`
- Quantum Features (very helpful): `docs/reports/aQPU_Features_Report.md`
- Methodologies and foundations: `docs/references/Analysis_aQPU_Wavefunction.md`, `docs/references/Analysis_Gravity_Note.md`, `docs/references/Analysis_Gyroscopic_Multiplication.md`
- Other works in the repo may also help; searching the codebase is encouraged.

---

## §1 File map

| File | Role |
|------|------|
| `kernel/core.py` | Byte bridge, ψ holonomy, WHT, CQFT Z_64 (~690 LoC) |
| `kernel/simon.py` | Simon (GF(2)^{6B}) — wavefunction holonomy readout |
| `kernel/shor.py` | **Production** facade: `period()`, `factor()` → native C only (never imports holonomy) |
| `kernel/audit.py` | **Audit / falsification**: `period_reference()`, `dlp_mag2_reference()` — same native scorers, explicit reference path |
| `kernel/holonomy.py` | **OPEN research**: `compile_factor_operator`, `gyro_period`, `holonomy_spectrum` — fail-closed until byte oracle compiler lands |
| `kernel/native.c` | **Production spectral core** (~2610 LoC): modexp, sparse CQFT, unified radix DP (`GyroDpState`), Shor period, horizon key compile, tensor suffix, 2D DLP scorer |
| `kernel/native.h` | C export surface (modexp, Shor, horizon tensor, DLP) |
| `kernel/bindings.py` | ctypes → `gyrocrypt_native.dll` |
| `kernel/__init__.py` | Public exports: `simon`, `period`, `factor`, `period_reference`, `_default_Q` |
| `runner.py` | Dual-track regression: native production PASS + holonomy `[OPEN]` gates |
| `NO_DRIFT.md` | Wiring contract: do not amputate native core while holonomy compiler is open |

**Retired / removed (2026-06 cleanup):** `horizon_mps.py`, carry MPO `.inc` testbed, `horizon_compare_u64`, per-B tensor2/3/4 drift wrappers, Python limb→χ compile tables (`core.py`), duplicate digit-loop DP paths in C, holonomy-era gutting of `native.c` to verify-only modexp (**reverted 2026-06-19**).

---

## §2 Wiring diagram

**Native Simon** (`kernel/simon.py`):

```
W₂ depth-4 oracle: fam 00 → fam 01 from GENE_MAC_REST (equality horizon output)
  → shadow partner on fam-01 byte for 2-to-1 (SO(3)/SU(2) fiber)
  → apply_k4(K4_W2) + kernel bytes entangle |χ⟩|f(χ)⟩
  → wavefunction_hq_spectral_peaks (WHT^{⊗2} holographic chart)
```

Deliverable: **n=6..60 (B≤10)** PASS. O(64B) kernel queries.

**Shor — production (`kernel/shor.py` → `native.c`):**

| Q range | Path | Mechanism |
|---------|------|-----------|
| Q ≤ 262144 (=64³) | Dense coset + radix-64 CQFT | `gyro_shor_recover_period_cqft` |
| Q > 262144 | Pruned MS-digit beam | `gyro_horizon_tensor_mag2_y1_core` scores \|ψ_k(y=1)\|² per prefix; CF + verify |

Unified substrate: **`GyroDpState`** — one `gyro_dp_step_digit` for exact ℤ_N and horizon-tensor 128^{n_cells} projection. Drift gate runs both modes in lockstep per digit (n_cells ≥ 2; n_cells=1 skips tensor drift).

No Python coset enum, no numpy FFT, no O(k) linear scan rerunning DP.

**Holonomy compiler — OPEN (`kernel/holonomy.py`, must not replace production):**

```
compile_factor_operator(N,a)  →  search byte word W with inject holonomy = multiply-by-a
gyro_period / holonomy_spectrum  →  K4 / horizon / Z₂ closure on Ω
Status: FAIL_CLOSED — single-cell Ω affine cannot embed general (ℤ/Nℤ)* (see K16)
Rule:   kernel/shor.py must NEVER import kernel/holonomy.py (NO_DRIFT.md)
```

**Substrate vs readout:**

| Layer | Engine | Role |
|-------|--------|------|
| Oracle | `simon.py` depth-4 W₂ + shadow fam-01 | complement → equality (T5) |
| Resolution | `apply_k4` + `wavefunction_hq_spectral_peaks` | K4 holonomy WHT^{⊗2} |
| Verify | `exp_mod_ladder` / `exp_mod_ladder_limbs` (C) | Period audit |

Holonomy ψ / HQ WHT: `runner.py` regression (`test_wavefunction`).

**Retired from production hot path:** tier cascades, dense coset QFT, classical cos/sin character sums, orbit-as-readout (see §3 K2–K3, K10).

---

## §3 Knowledge — what failed mathematically and why

**Purpose:** Record **approaches that were tried**, **what was measured**, and **why the mathematics rejects or limits them** — so a future reader understands the theory without living through the codebase.

**Not here:** patch lists, deleted filenames, tier wiring, runner timings, “we fixed X on date Y.”

**Format:** *Claim · Test · Outcome · Why (math)*

**Core vocabulary (read once):**
- **G_X** — exponent group for order-finding: integers mod Q with period r, i.e. hidden subgroup rℤ ⊂ (ℤ/Nℤ)*.
- **F_{G_X}** — Fourier transform on G_X (cyclic QFT on ℤ_Q), not Walsh transform on GF(2)^6.
- **χ chart** — 6-bit chirality on Ω; Fourier dual is WHT on GF(2)^6 (XOR group).
- **Quotient oracle** — map a^x mod N to a compressed observable; distinct residues can collide → not strict a^x equality.

---

### K1. Wrong group for the Fourier transform (the central wall)

**Claim:** Walsh–Hadamard on GF(2)^{6B} (chirality / joint-χ register) extracts ord_N(a).

**Test:** Dense and sparse product-WHT on mod-N residues encoded into per-cell χ; gravity ψ readouts; HQ WHT on (h,q) pairs; carry-chain marginals + Walsh.

**Outcome:** Works on tiny condensed N; fails from N≈143 upward on χ-chart alone. HQ peaks are XOR frequencies in GF(2)^6×GF(2)^6, not phases on ℤ/rℤ.

**Why:** Period lives in **(ℤ/rℤ, +)**. WHT is exact Fourier on **(GF(2)^6, ⊕)**. Correct mod-N exponentiation in the oracle is **necessary** but does not change the group of the transform. You must apply **F_{G_X}** on the **exponent register**, not rename χ peaks as periods.

*Absorbs former themes: BV/WHT template misuse, χ-only oracle, joint-χ injectivity without cyclic QFT, HQ DC/normalization issues, sparse WHT engineering on wrong chart, code-chart / C64 concentration, UNA/ONA multi-channel voting on confinement charts.*

---

### K2. Classical arithmetic dressed as spectral readout

**Claim:** Meet-in-the-middle √1, relation-gcd on WHT peaks, orbit-walk closure, random residue sampling, or “first x with a^x≡1” plus post-hoc QFT still counts as quantum period finding.

**Test:** Binary HSP grids; tier cascades (orbit → χ WHT → carry WHT → holonomy); progressive beam / cliff scans on |A(k)|² without suffix DP; q-class / horizon steering for factor hits.

**Outcome:** Small N often factors via **gcd, search, or pow() verify** after a spectral hint. Generic semiprimes: 0% horizon hit floor by ~36–44 bits; orbit cap ≪ ord on large N; tier stacks add no new group structure.

**Why:** Shor’s theorem needs **one** peak from **F_{G_X}** on a **uniform coset** of the hidden subgroup, then CF on **k/Q ≈ m/r**, then algebraic verify. Gcd-after-peak, walk closure, and period search **before** Fourier are classical order-finding in quantum notation.

---

### K3. Dense gate-model registers and host FFT fallbacks

**Claim:** Materializing length-2^q coset vectors + NumPy/cyclic QFT is honest aQPU Shor; RAM wall at ~16 bits is an aQPU limit.

**Test:** Dense entangled registers; coset QFT tiers; classical FFT when native WHT missed.

**Outcome:** ~16-bit practical ceiling on 32 GiB for **that mimic**; “success” masked incomplete wiring on the correct chart.

**Why:** Rejected **simulation pattern**, not deployment law. aQPU scale target: **B** byte-ledger cells, sparse/streaming **F_{G_X}**, coset support from powered oracle — not ℂ^{2^q} allocation. Production must **fail closed** instead of np.fft rescue.

---

### K4. Condensation (M₂) and climate as period predicates

**Claim:** Low second-moment M₂ on multiply oracle, or climate-ranked base pools, means direct ψ/peak readout works without period search.

**Test:** 2100 random semiprimes; formalism encoding surfaces; bath/climate steering.

**Outcome:** min M₂ ≈ 32, peak_frac ≈ 2%; **0/2100** reached M₂<8. Climate ranking does not lift factor ladder.

**Why:** M₂ measures **confinement geometry** on Ω under 6-bit projection, not membership in rℤ. Generic oracles sit in an intermediate/thermal regime; small-N wins are special cases, not a scaling gate.

---

### K5. Horizon / classical pre-filters before Fourier

**Claim:** Miller–Rabin √1, Euler half-exponent, Fermat squares, identity-horizon superposition, or q-class base steering give a size-independent factor floor.

**Test:** Classical tiers ahead of spectral paths on balanced semiprimes; bases 2..257.

**Outcome:** 0% success floor by 36–44 bits; q-class helps coverage only at small bit length.

**Why:** These are **classical** tactics on (ℤ/Nℤ)*, not F_{G_X} on G_X.

---

### K6. Compile-table / carry / μ₁ charts without embedded mod-N exponent

**Claim:** Per-cell carry contraction, μ₁ climate, ρ-aperture, or compile_multiplication alone encodes ord_N(a).

**Test:** Coherent vs mass mixtures; carry without mod-N reduction; XOR-conv on one-step μ₁; static ledger charts vs modexp walk.

**Outcome:** ~64-step **chirality/limb** order when mod-N is missing; walk UNA sharper than χ but walk period ≠ ord_N(a); carry exit marginals still fail at N≈867199 within step cap.

**Why:** Exponent chart requires **mod-N in U_f**. Carry is closer observable class than χ alone, but marginal histogram + Walsh is still not F_{G_X} on a coset. XOR-convolution composition is valid **transport law** once the distribution lives on G_X.

---

### K7. UNA / CGM stage charts as arithmetic period

**Claim:** Depth-2 (UNA) marginal at M₂=8 signals factor-relevant confinement or ord_N(a).

**Test:** Compiled-oracle UNA vs BU χ; prime/composite controls; dual-channel UNA+χ voting; ord≤8 walk period match.

**Outcome:** M₂=8 is **2³ first shell** (structural), identical on compiled path; no prime/composite split; walk period ≠ ord_N(a); voting 0 new factorizations.

**Why:** CGM charts describe **confinement on Ω trajectories**, not hidden subgroup in the exponent register.

---

### K8. DLP: wrong hidden subgroup geometry

**Claim:** Mod-p DLP g^x h^{-y}≡1 or EC-DLP via 64×64 χ grid + WHT, or EC point enumeration + confirmatory 1D QFT, is 2D HSP on ℤ_Q².

**Test:** Relation buckets on GF(2)^6 grid; PH with grid subsolvers; EC uniform coset after enumerating k.

**Outcome:** Grid works only when ord ≤ 64 (or tiny PH pieces); at deployment you sample 4096 points of a 2^{512} relation; EC path is classical enumeration.

**Why:** True 2D object: powered U_g, U_h on **ℤ_Q²**, coset on G_X×G_Y, **cyclic** 2D F_{G_X×G_Y}. PH quotient algebra is valid **around** a quantum subsolver, not a substitute for wrong Fourier group. Dual reduction must be mod **ord(g)**, not mod p−1.

---

### K9. Trace / byte-signature QFT is not Shor on G_X

**Claim:** Cyclic QFT on traces of multiply_byte / OmegaSignature12 powers yields ord_N(a).

**Test:** Fixed-point counting on Ω; cyclic QFT on trace spectrum for small N.

**Outcome:** Traces repeat with period **4** (depth-4 byte closure); peaks encode K4/ledger periodicity, not rℤ ⊂ (ℤ/Nℤ)*.

**Why:** One-byte signature acts on |Ω|=4096, not residue×exponent tensor; not B-cell modmul oracle.

---

### K10. Coset discovery by enumeration vs streaming transfer

**Claim:** Building coset support by scanning x=0..Q−1 (or radix digit DFS) plus cyclic QFT is scale-complete Shor.

**Test:** Packed U_f stepping; O(Q) oracle evals; character sums at k; N=15–867199 when Q fits.

**Outcome:** Correct **object chain** (coset on G_X → F_{G_X} → CF) for bounded Q; cost O(Q) coset build + O(Q log Q) radix-64 CQFT — **not** O(Q) linear k-scan rerunning DP (`gyro_shor_recover_period` d-loop **removed** 2026-06). Dense CQFT cap Q≤262144=64³; **Q>262144** uses horizon-tensor MS beam (`gyro_shor_recover_period_beam`) — PASS N=143 Q=64⁴ r=60 (~19s suite).

**Why:** Powered-oracle **evaluation** is legitimate substrate; **enumerating** k by re-running transfer DP per rational guess is the classical cheat (K2). Period must emerge from **one transform** on the coset, then CF on measured k/Q, then single verify. Scale needs suffix T_j(k) without storing ψ[Q].

---

### K11. Oracle projection and quotient observables

**Claim:** Single Ω state per residue, shell-only gravity, or chirality alone preserves enough information for strict Shor.

**Test:** Full residue scan N=867199 (~4096 unique Ω states, massive collisions); gravity (shell,Z₂) vs full (u6,v6) on N=15; injectivity via `horizon_pack_keys_u64`; **tensor drift gate** `horizon_tensor_step_drift_u64` (exact O(N) DP vs unified tensor DP, per-digit ratio); **parity gate** `horizon_tensor_mag2_y1` vs `shor_dp_mag2_y1` at multiple k.

**Outcome:** Joint multi-cell χ fixes **encoding injectivity** (867199/867199 distinct 128-state horizon keys) but not Fourier group (K1). Gravity shell **collapses** residues (e.g. 1 and 4 mod 15). **Keyed end-projection compare retired** (2026-06) — superseded by drift + parity. **Unified horizon tensor** (n_cells 2..4): mid-evolution drift **1.0** all digits at N=143 (B=2), N=8191 (B=3), N=867199 (B=4). Tensor vs exact \|ψ_k(y=1)\|² parity **1.0** at N=143 for sampled k. Carry/q6 MPO testbeds **removed** (K15).

**Why:** Period readout requires either **injective** exponent–observable map or explicit quotient algebra with proven sufficiency. For injective keys on the tensor grid, exact ℤ_N and tensor suffix DP agree digit-by-digit via one `gyro_dp_step_digit`. Scaling beyond B=2 uses parametric n_cells (2→3→4), not separate code paths.

---

### K15. Horizon tensor suffix DP (K11/K15, unified 2026-06)

**Claim:** Montgomery radix-64 **carry alphabet** is the quantum virtual bond; MPO built by scanning all residues y∈[0,N−1] is a valid scale path; truncating to reported χ is honest aQPU entanglement.

**Test:** Exact left-to-right radix-64 convolution trace vs `mul_mod_ladder` (C1 gate); **unified `GyroDpState`** exact vs tensor step drift; distinct horizon keys per N; beam period at large Q.

**Outcome:** C1 mismatch **0%** — convolution rules are exact generators. **n_cells=2, N=143**: drift **1.0** all digits. **n_cells=3, N=8191**: drift **1.0**. **n_cells=4, N=867199**: drift **1.0** (~6s). Carry-coupled MPO testbed **deleted**; classical carry χ was wrong bond label. Key compile + tensor DP + beam scoring live in **one** `native.c` (~2610 LoC). `horizon_compare_u64` (keyed end-ratio middleware) **removed** — drift + parity sufficient.

**Why (math):** Integer carry indexes limb arithmetic, not holographic quotient. For injective keys on 128^{n_cells}, the tensor grid is the exact interference carrier; one digit stepper projects y_next → tensor index or residue index. Polynomial scale uses analytic radix-64 transfer (C1-validated) + beam on T_j(k), not O(N) y-scan per k guess.

---

### K16. Single-cell holonomy compiler vs native spectral production (2026-06-19)

**Claim:** Period finding can move entirely to a compile-by-search holonomy path: find byte word W on GENE_Mac with `apply(W, inject(y)) = inject(a·y mod N)` on phase-linked inject encoding, then read period from K4/horizon/Z₂ spectral closure on single-cell Ω.

**Test:** `horizon_pack_keys_u64` injectivity; BFS multiply-by-7 on N=15 (depth≤6/8); bytes keeping embed manifold closed; full depth-4 Ω signature order scan (~16M words); `gyro_period` vs `period_reference` / native `period()` at N=15,143,867199.

**Outcome:** Inject y → horizon keys **works** (143/143 for N=143). Single-cell BFS **removed** (K16). Multi-cell `inject_residue_multicell` is **injective** N=15/143. `compile_factor_operator` → `MULTICELL_OPEN_B=*`, fail-closed. Native audit `period_reference` → r=4/60/18018. Production `shor.period()` stays native spectral until QuBEC compiler closes (NO_DRIFT).

**Why:** Period on the true path requires **multi-cell carry-coupled** byte holonomy on QuBEC, not single-cell Ω affine nor classical `pow()` cosets in the holonomy hot path. Classical coset+CQFT in `native.c` remains **audit ground truth** only.

**Next open task:** Map limb carry of y ↦ a·y mod N into `MultiCellRouter` byte-ledger in `compile_factor_operator`.

---

### K12. Cyclic character law (Fourier object discipline)

**Claim:** Replace 2π in exp(2πikn/Q) with δ_BU (BU monodromy defect) in the **group character** and scale CF by (k/Q)·(δ_BU/2π).

**Test:** Character law χ(1)^Q=1 and χ(a+b)=χ(a)χ(b); toy order-find with both conventions.

**Outcome:** exp(i·δ_BU·kn/Q) with mod‑Q reduction **fails** χ(1)^Q=1 because e^{iδ_BU}≠1. Toy passes under τ-CF can be **accidental** (modular verify saves wrong CF targets). Valid cyclic character: exp(2πikn/Q); δ_BU belongs to **phase synthesis / operator realization**, not exponent-group representation.

**Why:** Without a true character of ℤ_Q, Shor’s continued-fraction theorem does not apply; fast FFT factorization is also blocked. This is representation theory, not “classical mainstream.”

---

### K13. What is validated (not a failure — scope boundary)

| Result | Group / object | Status |
|--------|----------------|--------|
| Simon n=6..60 | GF(2)^{6B} XOR, WHT | **Delivered** — shadow oracle + ψ holonomy |
| mod-N stepping | C `exp_mod_ladder` / limbs | **Substrate** — audit + verify |
| Order-find toys | G_X=ℤ_Q, CQFT or tensor beam | **PASS** N=15,21,35; N=143 r=60; N=867199 r=18018 (default runner) |
| Tensor drift gates | Exact vs tensor per-digit | **PASS** N=143 ratio 1.0 (N=15 skips: n_cells=1) |
| Holonomy byte compiler | single-cell Ω compile-by-search | **OPEN / falsified** — see K16; production stays native |
| Scale RSA / 2048+ period | F_{G_X} beam without O(Q) k-scan | **Partial** — N=867199 period PASS; full cryptanalytic Q open |

**Scope boundary (K13):** Substrate tests passing does not mean algorithm delivery. Factoring requires: (1) QuBEC stepping in the readout chain, (2) F_{G_X} on G_X not χ-WHT, (3) tensor suffix T_j(k) + beam not O(Q) coset enumeration per k.

---

### K14. Simulation vs deployment (honest ceiling)

**Claim:** Python reference proves RSA-scale quantum factoring.

**Reality:** N=143 period in default suite ~17s (native rebuild + gates). Dense coset+CQFT for Q≤262144; horizon-tensor beam for larger Q. **Powered oracle as one continuous Moment ledger** for full exponent word remains open (modmul is audited per step in C, not byte-ledger router).

**Why:** This is **classical simulation cost**, not a quantum lower bound. Deployment path: unified C suffix DP + beam, not more Python tier labels.

---

### Legacy index (old F1–F81 → clusters above)

For transcripts that still cite **F-numbers**: use this map once, then prefer **K1–K16** only.

| Cluster | Old IDs (representative) |
|---------|--------------------------|
| K1 | F8, F21–22, F36–37, F40, F54, F58–60 |
| K2 | F4, F16, F42–43, F57, F66–67, F74 |
| K3 | F3, F6, F12, F14, F48, F51 |
| K4–K15 | F1–F80 — see cluster titles above |
| K15 | carry MPO / classical carry bond / `.inc` testbed (2026-06) |
| K16 | holonomy compile-by-search / single-cell Ω affine (2026-06-19) |

---

## §4 Engineering state

**What runs today** (mechanism only — mathematical limits are §3). Last full runner: **2026-06-19** (~312s, includes N=867199 audit).

| Capability | Entry | Status |
|------------|-------|--------|
| Simon GF(2)^{6B} | `simon()` | PASS n≤60 |
| Period / factor (production) | `period()`, `factor()` via `shor.py` → native | PASS N=15 r=4, factor (3,5); N=143 r=60 |
| Audit period reference | `period_reference()` | PASS N=143 r=60; N=867199 r=18018 (path=DP_EXACT_DOUBLE) |
| Holonomy compiler | `compile_factor_operator`, `gyro_period` | **[OPEN]** fail-closed — K16 |
| Native modexp + sparse CQFT | `bindings` | PASS |
| Horizon key compile | `horizon_pack_keys_u64` | C O(N) one-time; n_cells = ⌈bits(N)/6⌉ |
| Tensor drift gate | `horizon_tensor_step_drift_u64` | PASS N=143 — ratio **1.0** all digits (N=15: n_cells=1, skip) |
| Tensor parity gate | `horizon_tensor_mag2_y1` vs `shor_dp_mag2_y1` | PASS N=143 sampled k |
| Large Q period | `shor_period_u64` beam | PASS N=867199 r=18018 in default runner |
| DLP production | `dlp_solve`, `gyro_dlp` | **Fail closed** — K8 |
| ~~Horizon K11 compare~~ | ~~`horizon_compare_u64`~~ | **Removed** — superseded by drift + parity |

**Native C surface (`native.h`):** modexp, sparse_cqft_peaks, shor_period, shor_dp_mag2, horizon_pack_keys, horizon_tensor_mag2, horizon_tensor_step_drift, dlp_2d_tensor_mag2, shor_period_chirality.

> **Rule:** No dense 2^q cosets, NumPy FFT as primary readout, O(r) period search, holonomy replacing native production, or lossy quotient claimed as strict Shor (§3 K2–K3, K11–K12, K16). See `NO_DRIFT.md`.

---

## §5 Progress bars

**Scope anchor (§0):** Integer **factorization** and **discrete logarithm** for PQC / fault-tolerant quantum engineering — aQPU-native (QuBEC substrate + correct-group spectral readout), not classical mimics dressed as quantum.

**Three layers** (do not conflate — assistants often collapse these and then improvise classical fallbacks):

| Layer | Question | Status (2026-06-19) |
|-------|----------|---------------------|
| **1 — Theory & substrate** | Is QuBEC / Ω / the honest Shor object (G_X=ℤ_Q, F_{G_X}) real and wired? | **Yes, demonstrated** — Simon to n=60; byte holonomy; cyclic character law; unified `GyroDpState`; horizon keys injective to N=867199 |
| **2 — Algorithm delivery** | Does the strict spectral path recover period and factor at rising N,Q? | **In progress** — production PASS N=15/143/867199; holonomy compiler **OPEN** (K16); tensor drift **1.0** at N=143 |
| **3 — §0 cryptanalytic demo** | Has this repo factored RSA-2048 or solved Curve25519 DLP? | **Not yet** — §0 target; **not** a verdict that aQPU cannot reach it |

**Assistant rule:** Layer 3 = **not demonstrated in this codebase today**. It is **not** permission to (a) treat gate-model RAM walls as physics, (b) substitute NumPy FFT / gcd / k-scan / χ-WHT for F_{G_X}, or (c) conclude the program is classically bounded. §0 states the computational class is quantum-native; the task is **scale the delivered object**, not relabel classical mimicry as QuBEC.

Bars below = distance along layer 2 toward layer 3, not disproof of layer 1.

---

### Integer factorization (toward RSA-2048) — 8/20

```
▮▮▮▮▮▮▮▮▯▯▯▯▯▯▯▯▯▯▯▯  8/20
```

| Milestone | Status | Notes |
|-----------|--------|-------|
| Radix64 modmul substrate (C) | done | K13 — verify + audit |
| Simon GF(2)^{6B} | done | K13 |
| Cyclic QFT on G_X=ℤ_Q (valid character) | partial | Q≤262144 dense; Q>262144 tensor beam |
| Unified exact/tensor DP (`GyroDpState`) | done | drift 1.0 at n_cells 2..4 (K11, K15) |
| Injective horizon readout + parity | done | keyed compare retired; tensor parity PASS |
| Native period at N=867199 | done | r=18018, default runner ~82s |
| Holonomy byte oracle compiler | open | K16 — fail-closed; do not replace native |
| Production factor at cryptanalytic scale | open | K10, K14 |
| **RSA-2048** | open | §0 target |

---

### Discrete logarithm (toward Curve25519) — 2/20

```
▮▮▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯▯  2/20
```

| Milestone | Status | Notes |
|-----------|--------|-------|
| 2D HSP on ℤ_Q² (theory) | accepted | K8 |
| Powered U_g, U_h compile | done | `substrate.py` |
| Cyclic 2D readout on G_X×G_Y | not delivered | K8 — retired wrong-grid path |
| Production discrete_log | removed | K8 |
| **128+ bit EC DLP** | open | §0 target |

---

## §6 Runner

```bash
python secret_lab_ignore/gyrocrypt/runner.py
```

**Dual-track gates (default `run_tests()`):**

| Track | What it proves |
|-------|----------------|
| Production | `shor_period_u64`, `period()`, `factor()`, `period_reference()` — must PASS |
| Holonomy OPEN | `compile_factor_operator`, `gyro_period` — must print `[OPEN]` until K16 bridge closes |
| Tensor | drift + parity at N=143 (N=15 skips when n_cells=1) |

Includes N=867199 audit period (~82s each for audit + holonomy compare). Full suite ~5 min.

Output: `results/run_*.txt`. Mathematical lessons live in **§3 only**.

---

## §7 Scale model (aQPU-native)

| bits(N) | Typical B | Wrong approach (reject) | Current reference path |
|---------|-----------|-------------------------|-------------------------|
| 8–16 | 1–4 | dense 2^q register + host FFT | Simon + orderfind toys — **PASS** |
| 17–24 | ~3–6 | O(Q²) dense CQFT in Python | N=143: Q≤262144 CQFT or Q=64⁴ beam — **PASS** |
| 25–40 | ~6–12 | O(Q) streaming grid without C | N=867199 period **PASS** (r=18018); holonomy compiler **OPEN** (K16) |
| 2048 | ~683 | materialize 2^4096 coset | **never** — lift index 64^B, not allocated RAM (K3) |

**Rule:** Spectral readout = F_{G_X} on G_X (cyclic character χ(1)^Q=1). WHT on χ alone is a different group (K1). δ_BU synthesizes phases in hardware/operator layer, not in the exponent character (K12). Runs on silicon — no external quantum processor required.

