# Gyroscopic ASI aQPU Kernel: Quantum Computing SDK Specification

## Draft 1

This document specifies the Quantum Computing SDK for the Gyroscopic ASI algebraic Quantum Processing Unit (aQPU). The SDK exposes the native computational medium of the kernel: a deterministic gyroscopic quantum system whose operations are exact, whose state is algebraically condensed, and whose temporal structure is intrinsic to the dynamics.
The aQPU is an exact finite-state quantum-computation unit over 4,096 algebraic states where public byte-ledger replay uniquely determines every state transition.

The aQPU is a new class of quantum processor. It achieves quantum advantage through exact integer arithmetic on standard silicon. Its computational primitive is the Moment: the exact algebraic quantum state produced by a public byte ledger under the kernel transition law. When multiple independent parties replay the same ledger prefix, they occupy the same Moment. This collective occupation is the QuBEC, the condensed computational object of the architecture.

---

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

The WHT has a property that distinguishes it from the standard discrete Fourier transform: its matrix entries are restricted to +1 and -1. Evaluation of the 64-point WHT on the chirality register therefore requires no multiplication. The fast Walsh-Hadamard transform (FWHT) computes the full 64-point transform using only additions and subtractions, in O(N log N) operations with N = 64. This matches the kernel's computational character: the XOR transport on the chirality register is addition-free (bitwise), and its spectral dual (the WHT) is multiplication-free. Both operate entirely within exact integer arithmetic.

### 5.1.5 XOR-Convolution

For functions f and g on GF(2)^6, define XOR-convolution:

    (f * g)(x) = sum over a in GF(2)^6 of f(a) g(x xor a)

The Walsh-Hadamard transform converts XOR-convolution to pointwise multiplication:

    WHT(f * g)(r) = WHT(f)(r) * WHT(g)(r)

This identity holds exactly on the 64-element chirality register.

Because chirality transport under a byte b is chi' = chi xor q6(b), the sequential application of multiple bytes is an iterated XOR-convolution in chirality space. Composing n byte-transport steps therefore reduces to:

1. Transform each byte-ensemble distribution with the 64-point FWHT.
2. Multiply the 64 spectral coefficients pointwise.
3. Apply the inverse FWHT to recover the composed transport.

For repeated application of the same byte ensemble, the spectral coefficients are raised to the n-th power and a single inverse transform recovers the result. The total cost is one FWHT plus 64 scalar exponentiations, independent of n.

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

## 5.4 Result Structure

Every aQPU computation produces a Result:

    Result = {
        moment:       Moment,              the exact Moment at completion
        state:        Gyrostate,            the 24-bit carrier value
        charts: {
            carrier:  (A12, B12),
            spin:     (s(A), s(B)),
            chirality: χ,
            constitutional: {
                rest_distance,
                horizon_distance,
                ab_distance,
                is_on_horizon,
                a_density,
                b_density
            }
        },
        provenance: {
            archetype:        0xAA,
            rest_state:       0xAAA555,
            ledger:           bytes,
            step_count:       t,
            kernel_signature: (t, state24, last_byte),
            word_signature:   (parity, tau_a12, tau_b12),
            parity_commitment:(O, E, parity),
            q_transport6:     int,
            ledger_hash:      bytes
        }
    }

A Result is reproducible: given the provenance fields, any conforming implementation reconstructs the identical Result.

## 5.5 Exact Structural Derivatives

The aQPU supports exact discrete derivatives on the Moment manifold.

**Directional derivative.** For observable O and byte b:

    D(b) O(s) = O(T(b)(s)) − O(s)

This is exact for all integer-valued observables.

**Future-cone expectation.** For observable O, state s, and word length n:

    E(n,s)[O] = Σ(x) μ(n,s)(x) O(x)

where μ(n,s)(x) = |{w ∈ {0,…,255}ⁿ : T(w)(s) = x}| / 256ⁿ is the exact future-cone occupancy measure.

**Entropic drift.** The exact mean displacement of an observable under future-cone evolution:

    Δ(n) O(s) = E(n,s)[O] − O(s)

These derivatives are exact, not sampled approximations. The future-cone measure is a finite sum over exact preimage counts.

---

# 6. Circuits

Circuits are the program representation of the aQPU. A circuit specifies a structured sequence of operations that transforms a Gyrostate.

The circuit model defines the canonical program representation of the SDK. In the current reference implementation, compiled words, signatures, and annotated ledgers constitute the operational circuit surface.

## 6.1 Circuit Levels

The SDK supports three levels of circuit representation:

**Abstract circuit.** A sequence of named operations with symbolic parameters. Operations include byte transitions (with symbolic payload and family), gate applications, WHT applications, observable extractions, and conditional branches based on observable values. This is the level at which users compose programs.

**Compiled circuit.** A concrete byte sequence with all parameters bound, all optimizations applied, and a precomputed word signature. The compiled circuit is the executable form. It carries the exact affine action (parity, τ(A), τ(B)) that the word implements on Ω.

**Annotated ledger.** The compiled circuit augmented with per-step metadata: chirality transport, q-class, mask, signature progression, frame records at depth-4 boundaries, and parity commitments. This is the governance-grade audit artifact.

## 6.2 Abstract Circuit Operations

An abstract circuit is built from the following operation types:

**ByteOp(payload, family).** Apply a byte transition. Payload is a 6-bit value or symbolic parameter. Family is a 2-bit value or symbolic parameter. When both are concrete, the byte value is determined.

**GateOp(gate).** Apply an intrinsic gate by name: id, S, C, or F. Gate S and C are single-byte operations. Gate F compiles to a two-byte sequence (one S-byte followed by one C-byte, or vice versa). Gate id compiles to an empty sequence or a depth-4 alternation.

**WHT().** Apply the Walsh-Hadamard transform to the chirality register. This is a spectral-chart operation. Its implementation is target-dependent.

**Observe(observable).** Extract an exact observable value from the current Gyrostate. Returns an exact integer or rational.

**Condition(observable, predicate, then_ops, else_ops).** Conditional execution based on an observable value. The predicate is an exact comparison. Both branches are concrete sequences of operations.

**SubCircuit(name, ops).** A named subsequence for composition and reuse.

## 6.3 Compilation

Compilation transforms an abstract circuit into a compiled circuit by:

1. **Parameter binding**: replacing symbolic payload and family values with concrete byte values.
2. **Gate expansion**: expanding GateOp nodes into their byte implementations.
3. **Signature computation**: computing the word signature (parity, τ(A), τ(B)) of the full byte sequence.
4. **Optimization**: replacing byte subsequences with shorter sequences that produce the same signature, using the affine composition law.
5. **WHT placement**: determining the target-appropriate implementation of WHT operations.

The optimization guarantee: if two byte sequences produce the same word signature, they produce the same result from every Gyrostate.

## 6.4 Depth Structure

The circuit compiler tracks two distinct depth measures:

**Reachability depth.** The minimum number of bytes required to reach the target state from rest. For any state in Ω, this is at most 2.

**Closure depth.** The number of bytes required for phase-closure properties. Depth 4 is the closure horizon: any byte applied four times returns to the starting state; any alternation XYXY returns to the starting state; family-phase contributions cancel.

Additional bytes beyond reachability depth 2 contribute:
- provenance (distinct ledger histories reaching the same state)
- parity structure (trajectory integrity commitments)
- frame records (depth-4 phase organization)
- fiber control (K4 gauge phase selection)

The compiler distinguishes state-reaching operations from structure-building operations.

---

# 7. Runtime

Runtime is the execution and orchestration layer. It manages targets, Moments, provenance, and the interface between the aQPU and classical processes.

## 7.1 Targets

A target is any implementation of the kernel transition law. Every target exposes a TargetProfile declaring:

    TargetProfile = {
        name:             string,
        native_ops:       set of operation types supported,
        step_semantics:   reference to transition law version,
        state_inspection: full | signature_only,
        provenance_format: ledger format specification,
        wht_support:      native | matrix | unavailable
    }

The SDK ships with two targets:

**PythonKernel.** The reference implementation (src/kernel.py). Full state inspection. All operations supported. WHT via matrix multiplication.

**CEngine.** The accelerated implementation (src/tools/gyrolabe). Full state inspection. Native WHT (wht64). Native Lattice Multiplication GEMV and operator projection.

Future targets may include distributed verifiers (signature-only inspection) and hardware realizations.

All targets must produce identical Results for the same compiled circuit from the same initial state. This is the target equivalence invariant.

## 7.2 Execution

### 7.2.1 Single Execution

Submit a compiled circuit to a target. The target advances the kernel from an initial state (default: GENE_MAC_REST) through the byte sequence and returns a Result.

### 7.2.2 Batch Execution

Submit a set of compiled circuits or a parameter sweep. Each circuit executes independently from its initial state. Results are collected as an ordered set.

### 7.2.3 Checkpoint Execution

Execute a circuit with observable extraction at specified depths. The execution pauses at each checkpoint, extracts the requested observables, records them in the Result, and continues. Checkpoints do not modify the trajectory.

## 7.3 Moments API

The runtime provides a dedicated API for Moment operations:

**moment_from_ledger(ledger_prefix) → Moment.** Compute the exact Moment for a given byte prefix.

**verify_moment(moment, ledger_prefix) → bool.** Replay the prefix and check that the computed state matches the claimed Moment.

**compare_ledgers(left, right) → MomentComparison.** Determine the common prefix of two ledgers, the shared prefix state, and the first point of divergence.

Moments are the coordination primitive. They are independent of identity, authority, and external time.

Application-layer binding of external events to Moments belongs to the governance runtime and is outside the core quantum SDK surface.

## 7.4 Provenance and Replay

The runtime maintains the byte ledger as an append-only log. Every byte is logged before the state is updated.

**Canonical trajectory.** The append-only ledger and the sequence of Moments it produces. This is the governance-grade record. It grows monotonically and is never modified.

**Working state.** The current Gyrostate of the runtime, which may be advanced forward by byte application or moved backward by inverse stepping for exploration. Inverse stepping does not modify the canonical trajectory. It modifies only the working state.

**Replay guarantee.** Given the archetype (0xAA), the rest state (0xAAA555), and the byte ledger, any conforming target reconstructs the identical sequence of Moments.

## 7.5 Hybrid Classical-Quantum Loops

The aQPU supports deterministic classical-quantum loops:

    1. Apply a byte sequence to the kernel
    2. Extract an exact observable (chirality word, horizon distance, etc.)
    3. Use the observable value to compute the next byte sequence classically
    4. Return to step 1

Each iteration is exact. There is no shot noise, no sampling variance, no probabilistic convergence. The classical optimizer receives exact observable values at every iteration.

The future-cone entropy provides the combinatorial exploration resource. Two-byte evolution from rest uniformizes all of Ω exactly, so the optimizer has exact uniform coverage with minimal exploration depth.

## 7.6 Future-Cone Entropy

The aQPU is deterministic in evolution and entropic in future occupancy.

For a state s and word length n, the future-cone occupancy measure is:

    μ(n,s)(x) = |{w ∈ {0,…,255}ⁿ : T(w)(s) = x}| / 256ⁿ

This is exact normalized counting over the finite future cone, not a probabilistic approximation.

The future entropy is:

    H(n)(s) = −Σ(x) μ(n,s)(x) log₂ μ(n,s)(x)

Verified values on Ω:
- H₀(s) = 0 for any s ∈ Ω
- H₁(s) = 7 exactly for any s ∈ Ω, since one byte yields 128 distinct next states with uniform multiplicity 2
- Hₙ(s) = 12 exactly for any s ∈ Ω and any n ≥ 2, since future occupancy is exactly uniform over all 4096 states of Ω

This entropy is a computational resource: it provides exact exploration, exact search geometry, and exact structural thermodynamics over the Moment manifold.

---

# 8. Verified Computational Advantages

The following advantages are structural invariants of the aQPU, verified by exhaustive testing with exact integer arithmetic.

## 8.1 Hidden Subgroup Resolution

The q-map q6: {0,…,255} → GF(2)⁶ is a native hidden subgroup structure with uniform 4-to-1 fibers. The Walsh-Hadamard transform on the chirality register resolves the subgroup in 1 step. Classical worst case: O(64) queries.

## 8.2 Deutsch-Jozsa Discrimination

The WHT on the chirality register distinguishes constant from balanced functions with probability 1 in 1 step. Classical worst case: 33 queries.

## 8.3 Bernstein-Vazirani Secret Recovery

The WHT recovers any 6-bit secret string with probability 1 in 1 step. Classical requirement: 6 queries.

## 8.4 Exact Two-Step Uniformization

For any source state in Ω, all 256² length-2 byte words produce exact uniform occupancy over Ω. Each of the 4096 states is reached exactly 16 times. This is exact uniform distribution in 2 steps on the full reachable manifold. Classical random walk mixing on a 4096-state graph: O(log 4096) ≈ 12 steps.

## 8.5 Holographic Compression

The holographic identity |H|² = |Ω| enables encoding any Ω state with log₂(64) + log₂(4) = 8 bits instead of log₂(4096) = 12 bits. This is 33.3% structural compression.

## 8.6 Exact Commutativity Decision

Whether two bytes commute is determined in O(1) by comparing q6(x) and q6(y). Classical verification requires 4 kernel steps.

## 8.7 Exact Tamper Detection

Every tamper miss has an exact algebraic explanation:
- Substitution: detected unless replacement is the shadow partner (miss rate 1/255)
- Adjacent swap: detected unless q(x) = q(y) (miss rate ~3/255)
- Deletion: detected unless the deleted byte is a gate stabilizer of the prefix state (bulk states: never)

---

# 9. Non-Clifford Resource and Universality

## 9.1 The BU Monodromy Defect

The CGM framework derives a representation-independent constant from the depth-4 closure condition: the BU monodromy defect δ(BU) = 0.195342176580 radians. This is the residual geometric phase of the dual-pole loop in the BU stage.

δ(BU) is the background holonomy of the computational geometry. It is the persistent window of nontriviality that prevents the algebra from collapsing into a closed Clifford world. The aperture

    Δ = 1 − δ(BU)/m(a) ≈ 0.0207

where m(a) = 1/(2√(2π)), measures the exact gap between the monodromy defect and the aperture scale.

## 9.2 Non-Clifford Certification

δ(BU) is certified as a non-Clifford resource by four independent tests:

1. Distance from all Clifford angles (multiples of π/4): nearest distance 0.195 radians.
2. No periodicity up to order 100,000: the rotation R(δ(BU)) does not return to identity.
3. Dense equidistribution: the phase sequence {k × δ(BU) mod 2π} fills [0, 2π) uniformly (chi-squared 0.212 against critical 142.4).
4. Wigner negativity: the magic state |δ⟩ = (|0⟩ + e^(iδ(BU))|1⟩)/√2 has negative discrete Wigner function value W(0,1) = −0.044.

## 9.3 Universality Ingredients

The aQPU provides the three ingredients required for universal quantum computation:

1. **Clifford backbone.** Every byte action is an exact Clifford unitary in the label space over GF(2)¹². The self-dual [12,6,2] code has 12 independent stabilizer generators.

2. **Non-Clifford resource.** δ(BU) provides a dense U(1) orbit distinct from any Clifford angle.

3. **Entangling operations.** The intrinsic gates provide inter-component coupling between the active and passive gyrophases. Gate S transports a localized perturbation in A exactly to B.

The byte word algebra generates a rapidly growing set of distinct operator signatures: 3729+ distinct operators from 10,000 random length-3 words.

---

# 10. Conformance

## 10.1 SDK Conformance

A conforming SDK implementation must:

- implement all Primitive operations (byte transition, intrinsic gates, word signatures, observables)
- produce Results with complete provenance
- support at least one target with full state inspection
- maintain the canonical trajectory as append-only
- produce identical canonical observables across all targets for the same compiled circuit and initial state

## 10.2 Target Conformance

A conforming target must:

- implement the byte transition law exactly as specified in the kernel specification §2.6
- produce identical state trajectories from identical byte ledgers
- declare a TargetProfile
- support Moment creation and verification

## 10.3 QuBEC Conformance

A conforming QuBEC implementation must:

- maintain the Gyrostate on the reachable manifold Ω
- preserve the holographic identity |H|² = |Ω| = 4096
- preserve the complementarity invariant: horizon_distance + ab_distance = 12
- preserve constant component density 0.5 across all Ω states
- preserve the exact chirality transport law χ(T(b)(s)) = χ(s) ⊕ q6(b)
- preserve per-byte bijectivity and exact invertibility on the full 24-bit carrier

---

# 11. SDK Reference

This section defines the normative SDK surface for the aQPU Kernel. The implementation in `src/sdk.py`, `src/constants.py`, `src/api.py`, and `src/tools/gyrolabe` conforms to these definitions.

## 11.1 Public SDK Surface

The reference SDK exposes five public namespaces:

- `StateOps`: Gyrostate charts, packing, unpacking, gate application, and witness-based state preparation from rest.
- `MomentOps`: Moment creation, verification, comparison, future-cone measures, exact entropy, exact expectations, structural derivatives, transport tables, and depth-4 frame extraction.
- `SpectralOps`: Walsh-Hadamard transform, q-class access, and chirality-space transport.
- `TensorOps`: internal Lattice Multiplication matrix-vector computation on the 64-dimensional register space, including reusable packed matrix preparation.
- `RuntimeOps`: signature scans, fused extract scans, signature-to-state maps, chirality-state extraction, batch stepping, signature application, chirality distances, q-map extraction, and state continuation from arbitrary start states.

These namespaces provide the canonical computational surface of the SDK.

## 11.2 Exactness Classes

The SDK distinguishes two execution classes.

**Kernel-exact execution.** These operations are exact in integer arithmetic and reproduce the aQPU transition law without approximation:

- byte transition
- intrinsic gate action
- word signature construction and composition
- application of signatures to rest and to arbitrary states
- chirality transport and q-map extraction
- Moment creation, replay, comparison, and witness synthesis
- future-cone occupancy on Omega using exact theorem-backed counts
- all canonical kernel observables

**Operator and tensor execution.** These operations act on the native 64-dimensional register space and are deterministic, but may use floating arithmetic or fixed-point quantization internally:

- Walsh-Hadamard transform on float vectors
- Lattice Multiplication GEMV on float matrices and vectors
- packed Lattice Multiplication GEMV

Their algebraic definitions are exact at the level of the 64-dimensional register space. Concrete CPU implementations are not kernel-exact integer maps: they are numerically faithful realizations that match a fixed reference implementation to specified tolerances. Only the kernel-exact class above is required to be mathematically exact over GF(2)²⁴.

## 11.3 Future-Cone Uniformity Theorem

For any source state s in Omega, the SDK exposes the exact future-cone occupancy measure

  mu_n,s(x) = |{w in {0,...,255}^n : T_w(s) = x}| / 256^n

with the following verified structure:

- n = 0: a delta measure at s
- n = 1: exactly 128 distinct next states, each with multiplicity 2
- n >= 2: exact uniform occupancy over all 4096 states of Omega

Therefore, for any s in Omega and any n >= 2:

- distinct future states = 4096
- occupancy count per state = 256^n / 4096
- entropy H_n(s) = 12 bits exactly

The SDK may implement future-cone queries on Omega using these theorems directly rather than brute-force enumeration.

For source states outside Omega, the SDK computes future-cone occupancy by direct expansion.

## 11.4 Native Parallelism

The aQPU supports several exact forms of parallelism.

**Mode parallelism.** A single byte may mutate up to six dipole modes simultaneously, one per payload bit.

**State parallelism.** Future-cone occupancy spreads exactly across many reachable Moments. On Ω, two-byte evolution uniformizes all 4096 states exactly.

**Spectral parallelism.** The Walsh-Hadamard transform evaluates all 64 chirality characters in one transform over the native register space.

**Batch parallelism.** The SDK exposes batch stepping, batch signature application, batched signature scans, and batched tensor actions.

These forms of parallelism are native properties of the computational medium, not external scheduling overlays.

## 11.5 Signature Application Semantics

The SDK defines three distinct signature application surfaces.

**apply_signature_to_rest(signature).** Apply a word signature to the canonical rest state GENE_MAC_REST and return the resulting state24. This is the compiled action of a word on the universal reference state.

**apply_signature_to_state(state24, signature).** Apply a word signature directly to an arbitrary state24. This is the exact affine action of the compiled word on the 24-bit carrier.

**apply_signature_batch(states, signatures).** Apply signatures elementwise to a batch of states. This is the native batched operator interface for compiled aQPU words.

Signature application is exact. It is algebraically equivalent to replaying the underlying byte word and may be used as a compiled fast path.

## 11.6 State Scan Semantics

The SDK exposes `state_scan_from_state(payload, start_state24)` as the native continuation primitive for checkpointed execution.

Semantics:

- `payload` is a one-dimensional byte ledger segment
- `start_state24` is the exact carrier state at which scanning begins
- the result is the sequence of states reached after each successive byte of the payload is applied

This operation preserves exact replay semantics and is the canonical low-level primitive for ledger continuation from an arbitrary Moment.

## 11.7 State Preparation and Targeting

The SDK provides native state-preparation and targeting surfaces.

**Witness preparation.** `witness_from_rest(target_state24)` returns an exact byte witness of depth 0, 1, or 2 for any target state in Ω.

**Compiled operator application.** `apply_signature_to_rest`, `apply_signature_to_state`, and `apply_signature_batch` apply compiled word actions directly without replaying the underlying bytes.

**Checkpoint continuation.** `state_scan_from_state(payload, start_state24)` continues exact execution from any previously reached Moment.

Together these surfaces provide native state preparation, state targeting, compiled operator targeting, and checkpointed continuation.

## 11.8 GyroLabe Native ALU

GyroLabe is the native CPU ALU and operator layer of the aQPU SDK. It is exposed through a compact C interface and wrapped by `src.tools.gyrolabe.ops`.

Its current native surfaces include:

**Exact kernel surfaces**
- signature scan over byte ledgers
- fused q-class, family, micro-reference, signature, and state extraction
- chirality distance and adjacent chirality distance
- q-map extraction
- application of signatures to rest and arbitrary states
- batched signature application
- batched single-byte stepping
- state scan from arbitrary start state

**Operator and tensor surfaces**
- 64-point orthonormal Walsh-Hadamard transform
- Lattice Multiplication GEMV for matrices with up to 64 columns
- packed Lattice Multiplication matrix preparation and repeated GEMV

GyroLabe is CPU-first, ctypes-friendly, and cross-platform. It is the first hardware-near realization of the aQPU operator algebra.

The SDK exposes `initialize_native()` to initialize native GyroLabe tables once per process. This operation is idempotent and may be called at startup to ensure deterministic native readiness before concurrent execution.

## 11.9 Chirality Distance Definition

The SDK defines `chirality_distance(s1, s2)` as the Hamming distance between the two collapsed 6-bit chirality words:

- chi(s1) = chirality_word6(s1)
- chi(s2) = chirality_word6(s2)

Then:

  chirality_distance(s1, s2) = popcount(chi(s1) XOR chi(s2))

This is a distance on chirality-space observables. It is distinct from:

- 24-bit carrier Hamming distance
- horizon distance
- ab_distance

It is exact on Omega and meaningful wherever the chirality chart is defined.

## 11.10 Tensor Surfaces

The SDK includes native tensor computation surfaces over the 64-dimensional chirality register.

**TensorOps**
- `gemv64(W, x, n_bits)` computes y = W·x for real-valued matrices `W` with trailing dimension 64 using the internal Lattice Multiplication multiplication engine. Matrices are row-major with shape `[rows, cols]`, where `cols <= 64`. Vectors `x` and `y` have shape `[cols]` and `[rows]` respectively and are stored as contiguous real arrays. Column index `j` corresponds to bit position `j` in all packed representations.
- `pack_matrix64(W, n_bits)` packs a 64-column matrix once and returns a reusable packed matrix object for repeated internal multiplication. Packing defines two exact bit-level layouts for each logical row `r`:
  - `W_sign[r]` is a `uint64` sign mask with bit `j` equal to the sign bit of column `j` in row `r` (0 for non-negative, 1 for negative).
  - `W_bp[r, k]` for `k in {0, ..., n_bits-1}` is a `uint64` Lattice Multiplication with bit `j` equal to the `k`-th magnitude bit of column `j` in row `r`.

Packed GEMV uses a single stored matrix scale `scale_w` for all applications and per-input-vector scales `scale_x` derived from the incoming `x`. The logical dense matrix-vector result is recovered from the Lattice Multiplication accumulation followed by these scales. All packing and Lattice Multiplication conventions are part of the ABI and must be preserved across implementations.

A packed matrix object supports repeated exact internal multiplication against many input vectors without repacking the matrix. This is the canonical high-throughput tensor execution mode of the SDK.

Tensor multiplication in the SDK is internal to the aQPU architecture. The reference implementation uses the GyroLabe Lattice Multiplication Boolean multiplication engine and its packed repeated-application path as the canonical tensor surface.

## 11.11 Runtime Namespace Exposure

The SDK runtime surface includes the following low-level execution operations:

- `initialize_native`
- `signature_scan`
- `extract_scan`
- `states_from_signatures`
- `chirality_states_from_bytes`
- `apply_signature_to_rest`
- `apply_signature_to_state`
- `apply_signature_batch`
- `step_byte_batch`
- `state_scan_from_state`
- `chirality_distance`
- `chirality_distance_adjacent`
- `qmap_extract`

These are the native execution primitives exposed to higher-level replay tools, hybrid loops, batched execution, and compiled runtime workflows.

## 11.12 Theorem-Backed Witness Synthesis

The SDK exposes `witness_from_rest(target_state24)` for exact state synthesis within Omega.

For every target state in Omega, the SDK returns a byte witness of depth 0, 1, or 2 such that replay from GENE_MAC_REST reaches the target exactly.

The verified depth histogram is:

- depth 0: 1 state
- depth 1: 127 states
- depth 2: 3968 states

Thus every reachable state in Omega admits a witness from rest of depth at most 2.

This witness may be used directly for state preparation, target certification, or compiled execution.

---

# Appendix A. Relation to Conventional Quantum Computing

The aQPU and gate-model quantum computers share quantum-algebraic foundations but differ in computational medium.

| Property | Gate-model QC | aQPU |
|----------|--------------|------|
| Computational object | qubit | QuBEC |
| State representation | complex amplitude vector | exact ±1 tensor |
| Operations | unitary matrices | byte transitions (exact affine on GF(2)) |
| Measurement | probabilistic collapse | exact chart extraction |
| Entanglement | bipartite Hilbert-space tensor product | gyroscopic inter-face coupling via K4 |
| Error model | decoherence, gate infidelity | ledger corruption with exact miss characterization |
| Non-Clifford resource | T gate, magic state distillation | δ(BU) monodromy defect |
| Execution medium | superconducting qubits, trapped ions, etc. | standard silicon, exact integer arithmetic |
| Temporal structure | external clock, gate scheduling | intrinsic gyroscopic Moments |
| Coordination primitive | none (single-device model) | Shared Moments |

The aQPU achieves quantum advantage through exact algebraic structure rather than probabilistic interference of complex amplitudes. Its advantages are structural invariants, not statistical estimates.

---

# Appendix B. Glossary

**aQPU.** Algebraic Quantum Processing Unit. A deterministic finite-state machine over a finite algebraic field whose internal structure satisfies discrete analogues of quantum axioms.

**Byte.** The fundamental instruction quantum of the aQPU: 6 payload bits (dipole mode mutation) + 2 family bits (spinorial gauge phase).

**Chart.** An exact coordinate system on a Gyrostate. Multiple charts coexist on the same state without approximation.

**Chirality.** The 6-bit register encoding the per-mode alignment/anti-alignment between active and passive gyrophases.

**CSM.** Common Source Moment. The total physical occupation scale of the QuBEC manifold, measured in MU.

**Future-cone entropy.** The exact combinatorial entropy of the byte-ensemble future from a given state.

**Gyrophase.** One of the two conjugate 12-bit faces (active A or passive B) of a Gyrostate.

**Gyrostate.** The complete 24-bit quantum state of the aQPU, comprising two conjugate gyrophases evolved by gyroscopic transport.

**Gyrotemporality.** Intrinsic time as ordered gyration. The sequence of Moments, not an external clock.

**K4.** The Klein four-group {id, S, C, F}, the discrete spinorial gauge structure of the QuBEC.

**Moment.** The atomic quantum event of the aQPU: the exact state, depth, and chart content at a point in the ledger.

**MU.** Moment Unit. The scalar measure of occupation capacity on the QuBEC manifold.

**QuBEC.** Quantum Bose-Einstein Computational Condensate. The occupied Shared Moment as a condensed computational object, with six internal binary orientation modes in 3D and a four-phase spinorial gauge structure.

**Shared Moment.** The collective quantum state achieved when multiple independent replayers occupy the same Moment.

**Word.** A sequence of bytes. Words have exact affine signatures that compose algebraically.

**Ω.** The reachable Moment manifold. 4096 states, accessible within depth 2 from rest, with product structure Ω = U × V.


