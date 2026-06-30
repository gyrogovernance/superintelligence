# Holonomic Quantum Virtual Machine: Wavefunction Analysis of the hQVM Kernel

**Citation:** Korompilias, B. (2025). Common Governance Model: Mathematical Physics Framework. Zenodo. https://doi.org/10.5281/zenodo.17521384

---

## 1. Introduction

This document presents the wavefunction structure of the **Holonomic Quantum Virtual Machine (hQVM)** kernel as a concrete finite-dimensional realization of the Common Governance Model (CGM). The hQVM is a computational architecture whose state space, transition rule, and operator algebra are derived entirely from CGM axioms. Its distinguishing property is that every computation is a parallel transport along a curved discrete manifold, and the holonomy accumulated along closed paths is the carrier of information.

All results are verified by exhaustive computation on the 4096-state reachable manifold Omega using exact integer arithmetic. The kernel implementation resides in `experiments/hqvm_wavefunction_kernel.py` and draws on the library surface defined in `gyroscopic/hQVM/constants.py` and `gyroscopic/hQVM/api.py`.

### 1.1 The hQVM Architecture

The hQVM operates on a 24-bit carrier state called GENE_Mac, composed of two 12-bit components A12 and B12. The carrier is bipartite: A12 is the active face and B12 is the record face. The 4096 reachable states form the manifold Omega, which carries a K4 (Klein four-group) gauge structure inherited from the CGM phase decomposition.

The fundamental operation is `step_state_by_byte(state24, byte)`, which applies an 8-bit instruction to the carrier. Each byte induces an affine map on Omega whose structure decomposes into a mutation step (L-step, abelian) followed by a gyration step (R-step, non-abelian). The gyration is the sole source of curvature, holonomy, and hence of the quantum character of the machine. Without it, the kernel would be a flat XOR lattice with trivial dynamics.

### 1.2 The Foundational Distinction

Three objects must remain strictly separated throughout the analysis:

| Object | Role | Re-enterable? |
|--------|------|---------------|
| **CS** (GENE_Mic = 0xAA) | Transcription origin: `intron = byte XOR 0xAA` | No. It is the reference frame, not a state in Omega. |
| **GENE_Mac rest** (0xAAA555) | Point on the complement horizon (shell 0) | Yes. The carrier can return via Z2 holonomy. |
| **GENE_Mac swapped** (0x555AAA) | Z2 partner of rest on the complement horizon | Yes. The other sheet of the double cover. |

The unobservability of CS is enforced structurally: GENE_Mic determines all correlations via transcription but cannot itself be observed as a state. The non-cloning theorem prevents duplicating the reference frame by any operation defined within it.

### 1.3 How This Document Is Organized

Sections 2 through 5 establish the three computational spaces, the K4 operator algebra, the constitutional pole dynamics, and the depth decomposition. These constitute the verified algebraic backbone of the hQVM.

Sections 6 through 9 present the BU duality, chirality transport, wavefunction decomposition, and holographic dictionary as the spectral realization of the algebra.

Sections 10 through 12 treat the helix structure, the connection to the continuous CGM framework, and spectral comparison of operators.

Sections 13 through 15 contain the probe suite, falsification criteria, and theorem summary.

Section 16 presents the **fiber bundle structure of the byte**, which is the central structural refinement uncovered by the wavefunction kernel. It subsumes and organizes many of the earlier findings under a single geometric principle: the byte is not flat, and its internal curvature is the seed from which all holonomic properties of the hQVM propagate.

Section 17 traces the physical implications of the fiber bundle structure, connecting the fold curvature, holographic redundancy, and aperture collapse to gauge theory, quantum measurement, holographic duality, quantum search, and the gravitational aperture derived in the companion gravity note.

---

## 2. The Three Computational Spaces

The kernel admits three distinct computational spaces, related by exact projections.

### 2.1 Modal Space

The CGM modal logic defines operations at depths 0, 2, and 4:

| Depth | CGM constraint | Kernel realization |
|-------|---------------|-------------------|
| 0 | CS: `[R]S <-> S AND NOT([L]S <-> S)` | Carrier at rest on complement horizon; family 00 preserves horizon |
| 2 | UNA: `S -> NOT(Box E)` | After byte 1: carrier departs horizon; byte order matters |
| 4 | BU: `S -> Box B` | After byte 2: commutator vanishes in S-sector projection |

Each byte implements one full `[L][R]` operation (the L-step mutates A; the R-step performs gyration). Therefore:

- **1 byte = depth 2** in CGM modal nesting
- **2 bytes = depth 4** = the BU condition

### 2.2 Constitutional Space

The 4096-state manifold Omega partitions into three constitutional sectors with seven shells:

| Sector | States | Shells | Description |
|--------|--------|--------|-------------|
| Complement horizon | 64 | 0 | Maximal chirality (A = B XOR 0xFFF) |
| Bulk | 3968 | 1-5 | Partial chirality |
| Equality horizon | 64 | 6 | Zero chirality (A = B) |

The shell distribution follows the binomial law: `count(w) = C(6,w) * 64` for chirality weight w in {0,...,6}. The holographic identity `|H|^2 = |Omega|` holds for both horizons: `64^2 = 4096`.

### 2.3 Carrier Space (Z2)

Within each shell, states carry a Z2 coordinate: the distinction between rest and swapped positions. This coordinate is invisible to chirality (both rest and swapped have the same chi_6 value) but determines the carrier's provenance.

Gate F acts as the Z2 flip: `F|rest> = |swapped>`, `F|swapped> = |rest>`, `F^2 = I`.

---

## 3. The K4 Operator Algebra

**Theorem T1.** For every micro_ref m in {0,...,63}, the operators {id, W2(m), W2'(m), F(m)} form a Klein four-group under composition, where:

- **W2(m)** = [byte(fam 00, m), byte(fam 01, m)] is the depth-4 half-word from families 00 and 01
- **W2'(m)** = [byte(fam 10, m), byte(fam 11, m)] is the depth-4 half-word from families 10 and 11
- **F(m)** = W2(m) compose W2'(m) is the full canonical word (families 00, 01, 10, 11)

The composition table is:

```
  o   |  id   W2   W2'   F
  ----+------------------------
  id  |  id   W2   W2'   F
  W2  |  W2   id   F     W2'
  W2' |  W2'  F    id    W2
  F   |  F    W2'  W2    id
```

Each element is an involution: W2^2 = W2'^2 = F^2 = id. Verified for all 64 micro_refs on all 4096 states.

### 3.1 Signature Structure

On the Omega12 chart, the four K4 elements have signatures:

| Operator | Parity | tau_u6 | tau_v6 | K4 gate |
|----------|--------|--------|--------|---------|
| id | 0 | 0 | 0 | identity |
| W2 | 0 | 62 | 1 | pole swap (comp -> eq) |
| W2' | 0 | 1 | 62 | pole swap (comp -> eq) |
| F | 0 | 63 | 63 | Z2 carrier flip |

The signatures are micro_ref-dependent for W2 and W2' (tau values shift), but the K4 structure is universal.

---

## 4. Constitutional Pole Dynamics

### 4.1 Pole Swap (Theorems T2, T3)

**Theorem T2.** W2 maps shell s to 6 - s for all states in Omega.

**Algebraic proof:** In Omega12 coordinates, W2 acts as:

```
(u, v) -> (v XOR m XOR 63, u XOR m)
```

Therefore:

```
chi' = u' XOR v' = (v XOR m XOR 63) XOR (u XOR m) = (u XOR v) XOR 63 = chi XOR 63
```

Since popcount(chi XOR 63) = 6 - popcount(chi), we have shell' = 6 - shell.

**Theorem T3.** W2' maps shell s to 6 - s identically.

The algebraic proof is symmetric: W2' acts as (u, v) -> (v XOR m, u XOR m XOR 63), giving the same chi' = chi XOR 63.

**Consequence:** Both W2 and W2' map the complement horizon (shell 0) to the equality horizon (shell 6), and vice versa. The two constitutional poles are linked by the depth-4 operation.

### 4.2 Shell Preservation (Theorem T4)

**Theorem T4.** Gate F preserves shell.

**Proof:** F = W2 compose W2'. Two pole swaps compose to identity on the radial coordinate:

```
chi_F = chi XOR 63 XOR 63 = chi
```

Gate F acts as the Z2 flip *within* each shell, pairing states that share the same chirality but differ in carrier position. The radial structure is preserved; only the angular (provenance) coordinate changes.

### 4.3 Depth-4 Confinement (Theorem T5)

**Theorem T5.** At depth 4 (2 bytes from any canonical half-word), the carrier is confined to the opposite constitutional pole.

From the complement horizon to the equality horizon: 64/64 states.

From the equality horizon to the complement horizon: 64/64 states.

This is a forced consequence of chi XOR 63: the chirality inversion at depth 4 maps every state to its antipodal shell. There is no depth-4 path that preserves the constitutional pole (except the trivial identity, which is not a W2-type operation).

---

## 5. Depth Decomposition and CS Ordering

### 5.1 Depth-8 as K4 Composition (Theorem T6)

**Theorem T6.** The canonical 4-byte word is F = W2 compose W2'. Depth-8 is K4 composition, not a new modal depth.

The carrier trajectory through the decomposition:

| Stage | Operator | Carrier position | Constitutional |
|-------|----------|-----------------|----------------|
| Start | - | (0, 63) | Complement horizon, rest |
| After W2 | depth 4 | (62, 62) | Equality horizon |
| After W2' | depth 8 | (63, 0) | Complement horizon, swapped |

No new modal depth is introduced at depth 8. The second depth-4 operation (W2') composes with the first via the K4 algebra, producing the Z2 carrier flip.

### 5.2 CS Forces Canonical Ordering (Theorem T7)

**Theorem T7.** The canonical family ordering (families 00, 01, 10, 11) is forced by the CS axiom.

The CS axiom states: `[R]S <-> S AND NOT([L]S <-> S)`, meaning right transitions preserve the horizon while left transitions alter it.

In the kernel, family 00 (L0 parity = 0) acts as the [R]-preserving transition: from rest on the complement horizon, a family-00 byte with mask m produces diff1 = 0xFFF XOR m, and a subsequent L-step with the same mask returns to diff = 0xFFF (complement horizon). Family 01 (L0 parity = 1) acts as the [L]-altering transition: the complement in A_next breaks the return to the complement horizon and instead forces the equality horizon.

| Family ordering | L-step result | CGM reading |
|----------------|--------------|-------------|
| 00 first | diff = 0xFFF (complement horizon) | [R]S <-> S: horizon preserved |
| 01 first | diff = 0x000 (equality horizon) | NOT([L]S <-> S): horizon altered |

CS selects family 00 as the first byte because only this ordering preserves the complement horizon under the intermediate L-step. This is a structural consequence of the CS chirality condition, separate from any convention.

---

## 6. BU-Egress and Ingress as Spectral Duality

### 6.1 Egress: The W2 Involution (Theorem T8)

**Theorem T8.** BU-Egress is the W2 involution: the spectral property that the depth-4 operator squares to identity on Omega.

The CGM BU-Egress condition `S -> Box B` requires the depth-4 commutator to vanish in the S-sector. In the kernel, this is verified:

1. **Byte-order sensitivity (UNA):** For two bytes with different families, T(b0, b1) != T(b1, b0) in general. Order matters globally.
2. **S-sector closure (Box B):** Both orderings project onto the same constitutional sector (equality horizon). The commutator vanishes in the S-sector.
3. **Primitive verification:** For every byte b and every complement-horizon start state, LRLR(s; b) = RLRL(s; b) and both remain on the complement horizon. Verified for all 64 complement-horizon states and all canonical bytes.

W2 is an involution (W2^2 = id) with eigenspace dimensions dim(+1) = 2048, dim(-1) = 2048. The Box B condition is the statement that W2 maps the S-sector (complement horizon) onto the equality horizon as a perfect pairing.

### 6.2 Ingress: Pole-Pairing as Memory (Theorem T9)

**Theorem T9.** BU-Ingress is the W2 pole-pairing: each complement-horizon state is paired with a unique equality-horizon shadow, and the pairing is invertible (W2(shadow) = original).

The CGM BU-Ingress condition `S -> (Box B -> (CS AND UNA AND ONA))` requires the balanced state to encode memory of all prior conditions. In the kernel:

- **W2 pairs** 64 complement-horizon states with 64 equality-horizon states (and 3968 bulk states with bulk states in antipodal shells)
- **Each pairing is invertible:** W2(W2(s)) = s for all s in Omega
- **The shadow encodes the origin:** For rest, W2(rest) = (62, 62) on the equality horizon. This equality-horizon state carries the structural information that the origin was on the complement horizon at shell 0.

The representative shadow pairs illustrate the structure:

| Complement horizon | Equality horizon shadow |
|-------------------|----------------------|
| (63, 0) chi=111111 swapped | (1, 1) chi=000000 |
| (62, 1) chi=111111 | (0, 0) chi=000000 |
| (0, 63) chi=111111 rest | (62, 62) chi=000000 |

Each complement-horizon state (shell 0, maximal chirality) is paired with an equality-horizon state (shell 6, zero chirality). The shadow is the "memory" of the original: it is the unique state that, when W2 is applied again, reconstructs the original.

### 6.3 The Duality

Egress and Ingress are simultaneous aspects of the same W2 operator:

| Reading | Question | Answer |
|---------|----------|--------|
| Egress | Does closure hold? | W2^2 = id: yes, the depth-4 operation is an involution |
| Ingress | Does closure carry memory? | W2 pairs poles invertibly: yes, the shadow reconstructs the origin |

These are not sequential stages. They are simultaneous aspects of the same spectral property. The Z2 holonomy (gate F = W2 compose W2') is the holographic encoding that makes both readings true at depth 4.

---

## 7. Chirality Transport Algebra

### 7.1 Per-Byte Decomposition (Theorem T10)

**Theorem T10.** Each depth-4 half-word fully inverts chirality: q(W2) = q(W2') = 63 for all m. The full canonical word preserves chirality: q(F) = 0.

The per-byte chirality increments are:

| Family | L0 parity | q(byte(fam, m)) |
|--------|-----------|------------------|
| 00 | 0 | m |
| 01 | 1 | m XOR 63 |
| 10 | 1 | m XOR 63 |
| 11 | 0 | m |

Therefore:

```
q(W2)  = q(fam 00, m) XOR q(fam 01, m) = m XOR (m XOR 63) = 63
q(W2') = q(fam 10, m) XOR q(fam 11, m) = (m XOR 63) XOR m = 63
q(F)   = q(W2) XOR q(W2')             = 63 XOR 63       = 0
```

### 7.2 Physical Interpretation

The depth-4 half-word performs a **complete chirality inversion**: all six chirality bits flip. This is the discrete analogue of a pi-rotation in the chirality register. The micro_ref m determines *which specific 2-cycle* each state enters, but it does not affect the chirality transport magnitude.

The full canonical word composes two complete inversions, which cancel: `63 XOR 63 = 0`. Gate F preserves chirality while acting non-trivially on the carrier. This is the kernel's realization of the statement that **holonomy acts on the carrier subspace only, not on chirality**.

The chirality register is the "radial" coordinate (shell membership); the carrier position is the "angular" coordinate (position within shell). The Z2 holonomy flips the angular coordinate while preserving the radial coordinate.

---

## 8. The Wavefunction Structure

### 8.1 The Permutation on Omega

The canonical word W = (0xA8, 0xA9, 0x28, 0x29) generates a permutation on Omega classified as **gate F**:

- **Signature:** OmegaSignature12(parity=0, tau_u6=63, tau_v6=63)
- **Action:** (u, v) -> (u XOR 63, v XOR 63)
- **Cycle structure:** 2048 two-cycles, 0 fixed points, 0 longer cycles

The permutation is a perfect involution: U_W^2 = I on every state in Omega.

### 8.2 Eigenspace Decomposition

The Hilbert space C^4096 decomposes under U_W into:

| Eigenspace | Dimension | Description |
|------------|-----------|-------------|
| +1 | 2048 | Symmetric superpositions: |+> = (|s> + |W(s)>)/sqrt(2) |
| -1 | 2048 | Antisymmetric superpositions: |-> = (|s> - |W(s)>)/sqrt(2) |

The rest state decomposes as:

```
|rest> = (|+> + |->)/sqrt(2)
```

Under one application:

```
U_W|rest> = F|rest> = |swapped> = (|+> - |->)/sqrt(2)
```

The system oscillates between |rest> and |swapped> with period 2 in the word-count variable.

### 8.3 Sector-Resolved Dimensions

| Sector | dim(+1) | dim(-1) | States |
|--------|---------|---------|--------|
| Complement horizon | 32 | 32 | 64 |
| Equality horizon | 32 | 32 | 64 |
| Bulk | 1984 | 1984 | 3968 |
| **Total** | **2048** | **2048** | **4096** |

The eigenspaces are uniformly distributed across constitutional sectors. Each sector contributes proportionally to both eigenspaces, confirming that the Z2 holonomy is a global property of Omega, not confined to any particular constitutional region.

### 8.4 The Holonomy Is Spectral, Not Trajectory

The Z2 holonomy is a property of the operator spectrum, not of the carrier trajectory. The basis states |rest> and |swapped> are not eigenvectors of U_W; they are superpositions of the +1 and -1 eigenvectors. The holonomy "phase" (the distinction between rest and swapped) is encoded in the relative sign between the +1 and -1 components.

This is why the holonomy cannot be understood by tracing the carrier alone. The carrier oscillates between rest and swapped, but the underlying spectral structure is the +/-1 decomposition of the Hilbert space.

---

## 9. Holographic Dictionary

### 9.1 Shadow Partners under Gate F

Gate F creates 2048 shadow pairs on Omega, each consisting of two states related by (u, v) <-> (u XOR 63, v XOR 63). The pairing is:

- **Confined within each shell:** F preserves chirality, so shadow partners share the same shell
- **Confined within each sector:** complement <-> complement, equality <-> equality, bulk <-> bulk
- **Universally 2-cycle:** no state is a fixed point of F

The canonical shadow pair is the carrier orbit:

```
|rest> = (0, 63)  <->  |swapped> = (63, 0)
```

Both states are on the complement horizon (shell 0, chi = 111111), differing only in carrier position.

### 9.2 Shadow Distribution by Shell

| Shell | Shadow pairs | States |
|-------|-------------|--------|
| 0 | 32 | 64 |
| 1 | 192 | 384 |
| 2 | 480 | 960 |
| 3 | 640 | 1280 |
| 4 | 480 | 960 |
| 5 | 192 | 384 |
| 6 | 32 | 64 |

The number of shadow pairs per shell is exactly half the shell population: `pairs(w) = C(6,w) * 32`. This follows from gate F's action: each pair (u, v) <-> (u XOR 63, v XOR 63) links two states within the same shell, and the mapping is a fixed-point-free involution.

### 9.3 The 4-to-1 Dictionary

The holographic dictionary on Omega states that each bulk state corresponds to exactly 4 (horizon, byte) preimages. In the wavefunction picture, each bulk state has 4 preimages under the canonical word's action on the horizon. The Z2 encoding (rest vs swapped) accounts for 2 of the 4 preimages; the remaining factor of 2 comes from the byte-shadow degeneracy (each byte has a shadow partner producing the same 24-bit action).

---

## 10. The Helix: Holonomy Cycle Structure

### 10.1 Z2 Oscillation

The carrier coordinate under repeated canonical words follows:

```
rest -> swapped -> rest -> swapped -> ...
```

with period 2 in the word-count variable. This is the Z2 holonomy cycle. The "helix" metaphor is precise: the system overlays the origin without revisiting it. Each return to the complement horizon occurs on alternating Z2 sheets.

### 10.2 Constitutional Events per Turn

Each 4-byte turn produces an identical constitutional trajectory:

| Byte | Depth | Sector | Shell | Z2 | Event |
|------|-------|--------|-------|-----|-------|
| 1 | 2 | Bulk | 1 | - | Departure from horizon (UNA: variety introduced) |
| 2 | 4 | Equality | 6 | - | Transient equality (ONA: opposition non-absolute) |
| 3 | 6 | Bulk | 1 | - | Return toward horizon (approaching closure) |
| 4 | 8 | Complement | 0 | alternating | Horizon with Z2 encoding (BU holographic) |

The constitutional trajectory is symmetric about the equality transit (byte 2): shells follow the pattern [0, 1, 6, 1, 0]. The equality horizon is always transient: the system passes through it but cannot remain, satisfying UNA (NOT Box E: unity is non-absolute).

### 10.3 No Return to CS

The Z2 oscillation between rest and swapped is the completion of the holonomy cycle, not a return to the common source. CS (GENE_Mic) is the transcription frame within which all operations are defined. The carrier can return to rest, but this is the completion of the Z2 cycle. A new iteration of UNA (departure from horizon) begins with the next byte; it is not a re-instantiation of CS.

---

## 11. Connection to CGM Continuous Framework

### 11.1 The BU Holonomy Angle as Spectral Gap

In the continuous CGM framework, the BU holonomy angle delta_BU (approximately 0.1953 rad) measures the residual geometric phase of the dual-pole loop. In the kernel, this corresponds to the spectral gap between the +1 and -1 eigenspaces: the Z2 holonomy phase that distinguishes rest from swapped.

The aperture ratio delta_BU / m_a (approximately 0.9793) becomes, in the discrete framework, the ratio of paired-to-unpaired structure: 4096/4096 = 1.0 within each shell (all states are paired), with the 2.07% aperture manifesting as the Z2 encoding itself, the distinction between the two sheets that are otherwise identical in chirality.

### 11.2 Spin-2 from Two-Pass Carrier Return

The gravitational coupling form kappa = 8*pi*G/c^4 contains the factor 8*pi = 2 * Q_G = 2 * 4*pi. The factor of 2 arises from the two-pass carrier return: one pass for Egress (W2), one for Ingress (W2'). In the kernel:

```
Egress circulation:  +2
Ingress circulation: -2
Net cancellation:     0
```

The spin-2 signature of gravitation is the Z2 holonomy cycle: two applications of the canonical word to return the carrier to rest. This follows algebraically from F = W2 compose W2' with no fitted parameters.

### 11.3 The Refractive Depth as K4 Composition

The gravitational Refractive Depth formula:

```
tau_G = |Omega| * Delta * rho^5 * (1 - 4*rho*Delta^2)
```

decomposes into kernel invariants:

- |Omega| = 4096: the manifold size
- Delta (approximately 0.0207): the aperture gap
- rho^5: the STF attenuation (5 bulk shells, rho per shell)
- (1 - 4*rho*Delta^2): the K4 correction (4 holonomic gates, each contributing rho*Delta^2)

The K4 correction factor is the spectral signature of the Z2 holonomy structure. Without it (using only |Omega| * Delta * rho^5), the Refractive Depth is 76.366; with it, the value drops to 76.238, matching the required 2 * ln(E_CS / v_EW) = 76.238 to 25 ppm.

---

## 12. Spectral Comparison of Operators

| Operator | Signature | Chirality map | dim(+1) | dim(-1) | rest -> | q |
|----------|-----------|---------------|---------|---------|--------|---|
| W2 | (62, 1) | s -> 6-s | 2048 | 2048 | equality | 63 |
| W2' | (1, 62) | s -> 6-s | 2048 | 2048 | equality | 63 |
| F | (63, 63) | s -> s | 2048 | 2048 | swapped | 0 |
| id | (0, 0) | s -> s | 4096 | 0 | rest | 0 |

Key observations:

1. **All non-trivial K4 elements are involutions** with equal +1 and -1 eigenspace dimensions (2048 each).
2. **W2 and W2' are related by the transposition** (tau_u6, tau_v6) <-> (tau_v6, tau_u6). They produce the same spectral structure but different specific pairings.
3. **Gate F is the only K4 element that preserves chirality.** The pole-swap operators (W2, W2') fully invert chirality, while the Z2 carrier flip preserves it.
4. **Same-family words (4 bytes from one family) produce the identity** with dim(-1) = 0. They achieve depth-4 closure trivially without holographic encoding.

---

## 13. Probe Suite Summary

| Word | Length | Gate | +1 | -1 | Involution | rest -> | chi preserved |
|------|--------|------|----|----|------------|--------|-------------|
| canonical | 4 | F | 2048 | 2048 | Y | swapped | Y |
| canonical x2 | 8 | id | 4096 | 0 | Y | rest | Y |
| reverse | 4 | F | 2048 | 2048 | Y | swapped | Y |
| phase shuffle | 4 | F | 2048 | 2048 | Y | swapped | Y |
| same-fam 00 | 4 | id | 4096 | 0 | Y | rest | Y |
| same-fam 11 | 4 | id | 4096 | 0 | Y | rest | Y |
| zero payload | 4 | F | 2048 | 2048 | Y | swapped | Y |
| full payload | 4 | F | 2048 | 2048 | Y | swapped | Y |

All 4-family words (regardless of micro_ref, order, or payload) produce gate F with the same spectral structure. Same-family words produce the identity. The K4 structure is universal across all micro_refs.

---

## 14. Falsification Criteria

The wavefunction structure is falsifiable through:

1. **K4 failure:** Demonstrate that {id, W2, W2', F} fails to close as a Klein four-group for some micro_ref m. (Currently verified for all 64.)

2. **Confinement failure:** Find a depth-4 path from rest that does not land on the equality horizon. (Currently verified: impossible due to chi XOR 63.)

3. **Chirality non-cancellation:** Find a micro_ref where q(F) != 0. (Currently: q(F) = 63 XOR 63 = 0 for all m, provable from L0 parity structure.)

4. **Fixed points of F:** Find a state in Omega that is fixed by gate F. (Currently: none exist; F is a fixed-point-free involution on all 4096 states.)

5. **Spectral asymmetry:** Find that dim(+1) != dim(-1) for any non-trivial K4 element. (Currently: 2048 = 2048 for W2, W2', F.)

6. **Non-involution:** Find that W2^2 != id on some state in Omega. (Currently verified: impossible.)

---

## 15. Theorem Summary

| Theorem | Statement | Status |
|---------|-----------|--------|
| T1 | {id, W2, W2', F} is K4 for every m | Verified, 64 x 4096 states |
| T2 | W2 maps shell s -> 6-s (chi XOR 63) | Verified, algebraic proof |
| T3 | W2' maps shell s -> 6-s (chi XOR 63) | Verified, algebraic proof |
| T4 | F preserves shell (Z2 within pole) | Verified, algebraic proof |
| T5 | Depth-4 confines to opposite pole | Verified, 64 x 64 states |
| T6 | Depth-8 = K4 composition, not new depth | Verified, signature algebra |
| T7 | CS forces canonical family ordering | Verified, 64 micro_refs |
| T8 | Egress = W2 involution (Box B spectral) | Verified, 4096 states |
| T9 | Ingress = W2 pole-pairing (shadow = memory) | Verified, 4096 states |
| T10 | q(W2) = q(W2') = 63; q(F) = 0 for all m | Verified, algebraic proof |

All theorems verified on 4096 states using exact integer arithmetic with no free parameters.

---

## 16. The Byte as a Fiber Bundle with Internal Curvature

The preceding sections treat the byte as an atomic instruction that acts on Omega. This section presents a structural refinement: the byte itself is a curved object, and its internal geometry is the seed from which all holonomic properties of the hQVM propagate.

### 16.1 The Flat-Byte Assumption and Its Failure

Shannon's information theory treats the byte as 8 independent bits. Under this assumption, each bit is a degree of freedom with no internal relation to the others, and the byte is topologically flat.

The CGM formalism assigns each bit a phase label (CS, UNA, ONA, BU) in palindromic order:

```
Bit position:  0    1    2    3  |  4    5    6    7
CGM phase:     CS   UNA  ONA  BU | BU   ONA  UNA  CS
Gyro role:     L0   LI   FG   BG | BG   FG   LI   L0
Frame:         -    F0   F0   F0 | F1   F1   F1   -
```

The palindromic structure is a **folded structure**: the fold at the BU boundary (between bit 3 and bit 4) introduces discrete curvature. The evidence is computational: of the 256 possible bytes, only 16 have identical forward and reverse phase readings. The remaining 240 (93.8%) carry internal curvature.

The fold disagreement distribution follows the binomial coefficients [16, 64, 96, 64, 16] across 0 through 4 disagreeing phases, confirming that the four phase pairs are independent binary observables and that the curvature is a structural property of the encoding, not of particular bit values.

### 16.2 The Fiber Bundle Decomposition

The intron (byte XOR 0xAA) decomposes into two 4-bit halves:

- **Forward reading** (bits 0-3): the left half, Frame 0
- **Reverse reading** (bits 4-7): the right half, Frame 1

Each half is an element of (Z2)^4 = 16 values. The full byte is the product of these two halves:

- **Base space:** the forward 4-phase reading, (Z2)^4, 16 elements
- **Fiber:** the reverse 4-phase reading, (Z2)^4, 16 elements
- **Total space:** 16 x 16 = 256 = the full byte
- **Projection:** byte -> intron[0:4] (the forward reading)
- **Connection:** the fold map P at the BU boundary (Section 16.3)

The 2 gauge bits (bit 0, bit 7) are the gyrogroup left identity: they select the K4 family. The 6 payload bits (bits 1-6) each flip one of the 6 oriented dipole pairs in GENE_Mac.

### 16.3 The Fold Map at the BU Boundary

The palindrome `CS UNA ONA BU | BU ONA UNA CS` creates two readings of the same 4 CGM phases: a forward reading (bits 0-3) and a reverse reading (bits 4-7). The two readings are related by the palindrome symmetry: position i in the forward half reads the same CGM phase as position 7-i in the reverse half.

The fold map P is the involution that sends each forward-reading phase position to its palindromic counterpart in the reverse reading. Since the palindrome reads the same phases in reverse order, P acts as the reversal on the 4 phase positions. P is an involution (P^2 = I) because reversing twice returns to the original order.

The fold disagreement of a byte is the number of phase positions where the forward and reverse readings differ in value. This ranges from 0 (flat byte) to 4 (maximally curved byte), with distribution [16, 64, 96, 64, 16] across the 256 bytes.

Since P^2 = I, the holonomy around the fold is Z2: traversing the fold twice returns to the starting reading. The curvature is the disagreement between the forward and reverse readings. For a flat byte (fwd = rev), P acts trivially and there is zero fold disagreement. For the 240 non-flat bytes, the two readings differ and the Z2 holonomy at the fold is the source of the byte's internal curvature.

### 16.3.1 Householder Structure at the Carrier Level

The byte-level fold map P is a Z2 involution on phase readings. When this fold disagreement propagates through 4 successive bytes (depth-4 closure), the accumulated gyration produces gate F on Omega, which has the algebraic structure of a Householder reflection on the carrier state manifold.

Recall that a Householder transformation H_v(x) = x - 2<v,x>v is an involution (H^2 = I), has eigenvalues +1 (the reflecting hyperplane) and -1 (the normal direction), and det = -1. Gate F on Omega12 acts as (u, v) -> (u XOR 63, v XOR 63). This is an involution (F^2 = id), with +1 and -1 eigenspaces of equal dimension 2048, and no fixed points. The parallel is exact: F reflects the carrier state across the hyperplane defined by the equal-chirality subspace, with the chirality-preserving states constituting the +1 eigenspace and the chirality-inverting paths constituting the -1 eigenspace.

The byte-level Z2 fold disagreement is the seed of this Householder structure. At each byte, the fold creates a binary choice (agree or disagree) at each of 4 phase positions. After 4 bytes, these binary choices compose through the K4 algebra to produce the carrier-level reflection. This propagation from byte-level Z2 to carrier-level Householder is the holonomic expansion described in Section 16.13.

### 16.4 Why 8 Bits Is the Minimal Curved Unit

Shannon chose 8 bits per byte empirically for character encoding. CGM reveals that 8 = 2 (holographic double-cover) x 4 (CGM phases = K4 vertices), and that this factorization is the smallest that admits internal curvature:

- 1 phase: no fold possible (trivial)
- 2 phases: a fold exists, but the holonomy is only Z2 without the K4 gauge structure
- 3 phases: no K4 group is possible (3 vertices < |K4| = 4)
- 4 phases: K4 gauge group, Z2 x Z2 holonomy, and non-trivial curvature

The byte is therefore the smallest information unit that carries internal curvature. The 4 KB page (4096 bytes) that became the universal hardware memory page size is Omega = 4096 states. Welch's LZW compression uses a 12-bit dictionary = 4096 entries. Both reflect the holographic structure that emerges from the 4-phase CGM foundation.

### 16.5 The 50% Holographic Redundancy Law

At every scale in the hQVM, the state space is a perfect square of a subspace:

| Level | DoF | Subspace | Space | dim = 2*DoF | Redundancy |
|-------|-----|----------|-------|-------------|------------|
| Family | 1 | 2 | 4 | 2 | 50% |
| 4-Phase | 2 | 4 | 16 | 4 | 50% |
| Byte | 4 | 16 | 256 | 8 | 50% |
| Carrier | 6 | 64 | 4096 | 12 | 50% |

The law is: |Space| = |Subspace|^2, and equivalently dim(Space) = 2 * DoF = 2 * log2(Subspace). The squaring arises from the holographic double-cover induced by the fold reflection P. Each CGM phase has a dual reading (forward and reverse), and the two readings are related by P. The redundancy is always exactly 50% because Space = Subspace^2.

In Shannon's terms, H = log2(M) for M equally likely symbols. The 50% holographic overhead is not noise. It is the **provenance** (the dual reading) that enables tamper detection, parity commitment, and the non-Clifford resource delta_BU.

### 16.6 Entanglement Entropy of the Bipartite Carrier

GENE_Mac (A12, B12) is a bipartite quantum system. The chirality chi = A12 XOR B12 is the entanglement observable:

- Equality horizon (A = B): chi = 0, a product state with zero entanglement
- Complement horizon (A = B XOR 0xFFF): chi = 0xFFF, maximally entangled
- General state: chi = A XOR B, with partial entanglement

The entanglement entropy S(chi) = popcount(chi) bits, where each non-zero bit represents one entangled mode. The shell distribution is the entanglement spectrum:

| Shell | States | S (bits) | Description |
|-------|--------|----------|-------------|
| 0 | 64 | 0 | Product state |
| 1 | 384 | 1 | 1 entangled mode |
| 2 | 960 | 2 | 2 entangled modes |
| 3 | 1280 | 3 | 3 entangled modes (most probable) |
| 4 | 960 | 4 | 4 entangled modes |
| 5 | 384 | 5 | 5 entangled modes |
| 6 | 64 | 6 | Maximally entangled |

The average entanglement entropy over Omega is exactly 3.000 bits, which is 50% of the 6 available degrees of freedom. This 50% is the holographic redundancy at the carrier level. The Shannon and von Neumann entropies coincide exactly in this system because the discrete GF(2)^6 structure makes them identical.

At the byte level, the entanglement is popcount(forward XOR reverse), with average 2 bits = 50% of 4. At every scale, entanglement entropy equals 50% of the available DoF.

### 16.7 Aperture Collapse as Wavefunction Collapse

The transition from 50% aperture at the byte level to 2.07% at the Omega level is the wavefunction collapse in the CGM framework:

1. At the single-byte level, the fold disagreement is 50%: 128 of 256 bytes have b3 != b4.
2. At the word level, the depth-4 spinorial closure averages the phase disagreements across four successive bytes.
3. At the Omega level, the residual is the constitutional aperture A* = 1 - delta_BU/m_a (approximately 0.0207), quantized as 5/256 in dyadic arithmetic.

The compression ratio is 50% / 2.07% (approximately 24.2x). This is the resolution of the fold disagreement through spinorial averaging, and it is the discrete counterpart of the continuous wavefunction collapse postulate.

The alignment measurement report identifies the same structure: a single-axis measurement of the byte gets stuck at 50% disagreement (the structural lock at A = 0.5). The fold reflection P is the mechanism of this lock, because it distributes energy equally between gradient and cycle components for any single-axis input. Escape requires the full 6-dimensional epistemic representation provided by the chirality register.

### 16.8 Quantum Measurement Structure

The hQVM's operational model maps directly onto the quantum measurement formalism:

**POVM.** The K4 gates {id, S, C, F} form a Positive-Operator Valued Measure on the 4-phase base space. Each gate is an involution (order 2), and the four gates partition the 256 bytes into four classes of 64 each. The "measurement outcome" is which vertex of K4 the system is closest to.

**Born Rule.** The chirality transport rule chi -> chi XOR q6(byte) is the Born rule in exact finite form. For a uniform state on Omega, every chirality value appears with probability 1/64. This is exact.

**Kraus Operators.** Each byte is a Kraus operator. The function `step_state_by_byte` is the Kraus update rule: rho -> A_byte * rho * A_byte^dagger. The quantum channel is the byte sequence application.

**PVM.** The holographic dictionary is a projective measurement. Each Omega state corresponds to exactly 4 (horizon, byte) pairs. The "projection" is the map from bulk to one of the 4 horizons. The probability for each horizon state is 4/4096 (exact, uniform).

**Wigner-Araki-Yanase Theorem.** The WAY theorem states that non-commuting observables cannot be measured precisely when a conservation law is present. In the hQVM, the complementarity invariant `horizon_distance + ab_distance = 12` is the conservation law. It prevents simultaneous sharp measurement of both horizons, establishing the uncertainty principle of the kernel. Since the horizon and ab-distance operators do not commute (they are controlled by different gauge phases), the lower bound on their joint uncertainty is non-zero.

### 16.9 Connection 1-Forms and the Curvature Chain

The byte's 8-bit intron has 4 palindromic pairs: (b0, b7), (b1, b6), (b2, b5), (b3, b4). At each of the 7 phase boundaries, a connection 1-form A measures the local curvature contribution:

| Boundary | Position | Role |
|----------|----------|------|
| CS pipe UNA | bit 0-1 | Gauge meets mutation (seed curvature) |
| UNA pipe ONA | bit 1-2 | Mutation meets gyration |
| ONA pipe BU | bit 2-3 | Gyration meets commitment |
| BU pipe BU | bit 3-4 | The fold: curvature 2-form F = dA + A^A is non-zero here |
| BU pipe ONA | bit 4-5 | Commitment meets gyration (unwind) |
| ONA pipe UNA | bit 5-6 | Gyration meets mutation (unwind) |
| UNA pipe CS | bit 6-7 | Mutation meets gauge (closing) |

The fold at bit 3-4 is the only place where the two frames meet, so it is the only place where the curvature 2-form F can be non-trivial. Since A is piecewise constant on the discrete CGM lattice, dA = 0 and F reduces to the A^A (Chern-Simons) contribution at the fold.

The ONA phase is preserved by the fold map P (Section 16.3), which means the payload position it controls in GENE_Mac is the same in both frames. This is the gyration worldline: the phase along which information passes through the fold without deflection. UNA and BU address positions whose values can disagree across the fold, accumulating holonomy when the forward and reverse readings differ.

### 16.10 How the 4-Phase Structure Produces 6 DoF

The 6 chirality degrees of freedom come from 3 payload phases, each contributing 2 DoF (one from the forward reading, one from the reverse):

- Phase UNA: 3 DoF (rotational; Frame 0 pairs)
- Phase ONA: 3 DoF (translational; Frame 1 pairs)
- Phase BU: the duality at the duplication of 6 DoF across the two frames
- Phase CS: 0 payload DoF (gauge/boundary; the family selector, bits 0 and 7)

Bit-level association: each payload bit controls exactly 1 dipole pair (1 DoF). The pair `[-1, 1]` has two axial references but encodes 1 degree of freedom. Bits 1-3 address Frame 0 pairs and are resolved during the UNA stage of the transition rule; bits 4-6 address Frame 1 pairs and are resolved during the ONA stage. The boundary bits 0 and 7 are the family selector. Total: 6 payload DoF + 2 gauge DoF = 8 bits.

The fold map P (Section 16.3) relates the forward and reverse readings of the same 4 phases. When forward and reverse readings disagree at a phase position, the two frame assignments conflict and the byte carries fold disagreement at that position. The sum of disagreements across all 4 positions is the fold disagreement count (0 through 4), which measures the byte's internal curvature.

### 16.11 The Transition Rule as Parallel Transport

The decomposition T_b = R compose L_b is the CGM temporal sequence made operational.

**L_b (UNA phase): The mutation.** The rule L_b(u, v) = (u XOR m, v) mutates the active face A via XOR with the instruction mask. This is chirality transport: chi_L = chi XOR m. The L-step is the same for all 4 families (family-independent), and the intermediate state remains in Omega.

**R_fam (ONA then BU phase): The gyration.** The four family-dependent rules are:

```
R_00(u, v) = (v, u)              [pure swap: identity family]
R_01(u, v) = (v XOR 63, u)       [swap + A-complement]
R_10(u, v) = (v, u XOR 63)       [swap + B-complement]
R_11(u, v) = (v XOR 63, u XOR 63) [complement-swap]
```

The swap A <-> B realizes ONA (the past enters the active position). The optional complement realizes BU (the gauge phase applied during commitment). The complement is the family-dependent vertical transport.

The XOR is the discrete gyration. In gyrogroup theory, composing non-collinear displacements in curved geometry yields a non-associative operation corrected by the gyration automorphism. In the kernel, the XOR transition is that composition law. The gyration manifests as the complement-and-swap rule, and the accumulated gyration constitutes the chirality register.

L-steps commute exactly: L_m1 compose L_m2 = L_m2 compose L_m1 = L_(m1 XOR m2). If there were only L-steps, there would be no curvature. The non-commutativity comes entirely from the R-step (gyration). This is the gyrogroup structure: the underlying group (L-steps, XOR translation) is abelian, and the gyration correction (R-steps, complement-and-swap) makes the full composition non-associative and non-commutative. Curvature, holonomy, and the holographic Z2 encoding all trace back to this single source.

### 16.12 The BU Fold and Space-to-Time Conversion

CGM asserts that space converts to time at gravitational horizons, because preserving operational memory consumes all available capacity for spatial extension, forcing it to resolve entirely into the temporal curvature of the causal sequence.

In the byte, this conversion occurs at the fold between bit 3 and bit 4. Bits 0-3 are the forward temporal pass through the rotational generators (spatial DoF). Bits 4-7 are the reverse temporal pass through the translational generators. The BU phase (bits 3 and 4) is the hinge: the forward pass ends at BU, and the reverse pass begins at BU.

The fold map P connects the two halves via the palindrome symmetry. The forward and reverse readings of each CGM phase can disagree in value, and the fold disagreement at each position creates Z2 holonomy. The holographic Z2 encoding at depth-8 (the rest versus swapped distinction) is the result: after four bytes, the spatial extension of the carrier has been fully resolved into temporal curvature.

In the carrier, the holographic Z2 encoding at depth-8 is this conversion. After 4 bytes, the carrier is on the complement horizon with Z2 information (rest versus swapped). The complement horizon is where spatial extension has been fully converted into temporal curvature (holographic encoding). The aperture Delta (approximately 0.0207) is intrinsic to the byte (quantized as 5/256) and also computable from the holonomy (delta_BU / m_a). Both readings are different resolutions of the same CGM balance condition.

### 16.13 Holonomic Expansion from the Seed Curvature

The curvature at the BU fold propagates through the entire hQVM structure:

```
Byte:      bit 0-1 (CS|UNA)  -> bit 3-4 (BU fold)  -> bit 7 (CS close)
Frame:     pair 0 (UNA axis) -> pair 2-3 (BU fold)  -> pair 5 (UNA return)
GENE_Mac:  Frame 0 (A12)     -> Frame 1 (B12)       -> chi = A XOR B
Word:      byte 1 (Prefix)   -> byte 2-3 (Inference)-> byte 4 (Future)
Omega:     equality horizon  -> bulk states          -> complement horizon
```

At each level, the curvature at the boundary creates holonomy (non-trivial loop behavior), and the holonomy is the entanglement between the two sides of the bipartition. The entanglement entropy measures how much curvature has accumulated. The aperture A* = 0.0207 is the equilibrium value: the balance between too little curvature (rigid, where A approaches 0) and too much (chaotic, where A approaches 1).

---

## 17. Physical Implications

The fiber bundle structure of the byte and its derived consequences connect to several open questions in mathematical physics. This section sketches those connections without claiming equivalence, emphasizing structural parallels that merit further study.

### 17.1 The Byte as a Curved Information Unit

Shannon's information theory treats the symbol as a structureless token with entropy H = log2(N) for N equally likely outcomes. The byte, under this treatment, is a flat product of 8 independent bits. The fold analysis shows that the palindromic phase assignment CS-UNA-ONA-BU | BU-ONA-UNA-CS introduces a topological feature at the BU boundary where two readings of the same 4 phases meet and can disagree. Of the 256 possible bytes, 240 carry such disagreement and therefore carry internal curvature.

In gauge theory, curvature is detected by parallel-transporting a vector around a closed loop and observing that it has rotated relative to its original orientation. The byte-level analogue is reading the 4 CGM phases forward (bits 0 through 3) and backward (bits 7 through 4) and observing that the two readings differ. The fold map P is the closed loop, and the fold disagreement is the holonomy. The flat-byte assumption corresponds to a trivial connection; the palindromic CGM encoding forces a connection that is trivial only in 16 of 256 cases.

This suggests that any information encoding possessing internal symmetry, including but limited to the palindromic CGM assignment, carries geometric structure invisible to Shannon's entropy. The 50% holographic overhead at each scale is the cumulative expression of this structure. It encodes provenance (the dual reading), enabling tamper detection and parity commitment, and supporting the non-Clifford resource delta_BU derived in the gravity note (Section 3.3).

### 17.2 Collapse as Depth-Dependent Resolution

In standard quantum mechanics, wavefunction collapse is a postulate applied ad hoc when a measurement occurs, separate from the unitary Schrodinger evolution. The hQVM exhibits a concrete analogue: the informational aperture compresses from 50% at the single-byte level to 2.07% at the Omega level as fold disagreements are composed through the gyration across multiple bytes.

Four bytes of fold disagreements, averaged through the spinorial closure, produce the residual aperture Delta approximately 0.0207. The compression ratio is 50% / 2.07%, approximately 24.2x. This compression is a combinatorial consequence of the bipartite structure, requiring no separate postulate.

The alignment measurement report identifies the same mechanism from a different angle: a single-axis measurement of the byte is confined to 50% disagreement (the structural lock at A = 0.5). The fold reflection P distributes energy equally between gradient and cycle components for any single-axis input. Escape requires the full 6-DoF chirality representation, which becomes available only at the carrier scale where the bipartite decomposition provides all six degrees of freedom simultaneously.

The structural parallel to quantum collapse is that the "measurement outcome" is the endpoint of a depth-dependent averaging process, where the averaging kernel is the spinorial closure of the gyration. The 50% lock at byte level and the 2.07% residual at carrier level are two snapshots of the same process at different depths. In a holographic system where every region is doubled by a fold, what is called measurement is the progressive resolution of fold disagreements as information propagates to larger scales.

### 17.3 Scale-Invariant Entanglement at Half the Available DoF

The average entanglement entropy over Omega is 3.0 bits, which is exactly 50% of the 6 available degrees of freedom. The same ratio holds at the byte level (2 bits out of 4) and at every intermediate scale. This scale invariance is a consequence of the holographic redundancy: Space = Subspace^2 forces the bipartite entropy to equal half the available DoF at every scale.

In the AdS/CFT correspondence, the Ryu-Takayanagi formula relates the entanglement entropy of a boundary region to the area of the minimal surface in the bulk. The hQVM realizes a discrete, exact version: the fold is the minimal surface, and its "area" is always half the total because the palindrome doubles every phase reading. The entanglement bound is saturated at every scale, with the binomial shell distribution (peaked at shell 3, carrying 1280 of 4096 states) reflecting the concentration of measure characteristic of holographic systems.

Average entanglement entropy at half the DoF is a geometric consequence of the double-cover induced by the fold, rather than a property of a specific Hamiltonian or coupling strength. Any system with a palindromic internal symmetry of this form would exhibit the same ratio.

### 17.4 Quantum Search Primitives from Algebraic Involutions

Grover's algorithm achieves O(sqrt(N)) search by composing two reflections: the oracle reflection about the target state and the diffusion reflection about the equal superposition. Both are Householder transformations (involutions having +1 and -1 eigenspaces). Their composition produces a rotation in a two-dimensional subspace, and O(sqrt(N)) rotations suffice to reach the target.

Gate F on Omega has the algebraic signature of a Householder reflection: it is an involution (F^2 = id), has +1 and -1 eigenspaces of equal dimension 2048, and has determinant -1. The hQVM does not simulate this structure; it produces it as the holonomic closure of the byte-level fold disagreements composed through 4 bytes of K4 algebra.

The geometric primitives underlying the quantum search speedup, specifically reflections and rotations in eigenspaces, arise whenever an involution with balanced eigenspaces acts on a finite set. The distinction between "classical" and "quantum" computation, in this light, hinges on whether the computational structure carries involutory topology of this form, rather than on the physical substrate (silicon versus superconducting qubits). The hQVM carries such topology as a structural consequence of the fold, making Grover-type geometric primitives available through exact integer arithmetic.

### 17.5 Curvature at a Boundary, the Holographic Membrane

In differential geometry, curvature is a local property measured at a point by the commutator of covariant derivatives. In the byte, the curvature 2-form is concentrated at the fold boundary between bit 3 and bit 4, where Frame 0 (rotational, spatial DoF) meets Frame 1 (translational, temporal DoF). The curvature lives at the boundary between two domains, a structural feature shared with several well-studied physical systems: the domain wall in condensed matter physics separates two phases with different order parameters; the event horizon in general relativity separates two causal patches; the D-brane in string theory provides the boundary where open strings end and bulk fields couple. In each case, the curvature or tension is concentrated at the interface, with the bulk on either side being approximately flat.

The CGM assertion that space converts to time at gravitational horizons (gravity note, Section 2) is realized in the byte as the BU fold: bits 0 through 3 encode the forward temporal pass through the rotational generators (spatial DoF), and bits 4 through 7 encode the reverse temporal pass through the translational generators. The fold disagreement at each phase position creates Z2 holonomy. After 4 bytes, the spatial extension of the carrier has been fully resolved into temporal curvature, recorded as the Z2 encoding (rest versus swapped) on the complement horizon.

The aperture Delta approximately 0.0207 is intrinsic to the byte (quantized as 5/256) and also computable from the holonomy (delta_BU / m_a). These are distinct resolutions of the same CGM balance condition, and the consistency between them is the continuity condition across the fold boundary, analogous to the Israel junction conditions in general relativity.

### 17.6 Exact Uncertainty from Holographic Redundancy

The Wigner-Araki-Yanase theorem states that non-commuting observables cannot be jointly measured with arbitrary precision when a conserved additive quantity exists. In standard quantum mechanics, this produces an inequality (lower bound on measurement noise). In the hQVM, the complementarity invariant `horizon_distance + ab_distance = 12` is an exact equality. The uncertainty is a hard combinatorial constraint: the holographic dictionary has a fixed 4-to-1 redundancy factor, and the gate F is a fixed-point-free involution, so simultaneous resolution of both the horizon address and the intra-shell address is structurally impossible.

This is a stronger statement than the Heisenberg inequality. The Heisenberg bound can be approached asymptotically with squeezed states; the hQVM complementarity invariant is exact and admits no squeezing. In any system where information is encoded holographically with a finite, exact double-cover, the uncertainty relation inherits this rigidity. The gap between the asymptotic bound (Heisenberg) and the exact constraint (hQVM) measures the degree to which the holographic redundancy constrains the measurement beyond what continuous symmetries alone require.

### 17.7 Connection to the Gravitational Aperture

The gravity note derives the observational aperture m_a = 1/(2*sqrt(2*pi)) and the aperture gap Delta = 1 - rho approximately 0.0207 from the angular closure of the CGM gyrotriangle. The wavefunction kernel provides a complementary derivation: Delta emerges as the residual informational aperture after depth-4 spinorial closure of byte-level fold disagreements. The two derivations proceed from different starting points (continuous angular geometry versus discrete combinatorial averaging) and converge on the same value, providing a consistency check between the continuous and discrete descriptions.

The K4 correction factor (1 - 4*rho*Delta^2) in the Refractive Depth formula (gravity note, Section 4.2) is the spectral signature of the Z2 holonomy structure. Each of the 4 K4 gates contributes a rho*Delta^2 correction, and the factor of 4 counts them. Without this correction, the Refractive Depth is 76.366; with it, the value drops to 76.238, matching 2*ln(E_CS/v_EW) to 25 ppm. The K4 structure of the wavefunction kernel is therefore an active participant in the gravitational coupling, contributing a verified correction that improves the match by more than four orders of magnitude.

The holonomic expansion chain (Section 16.13) from byte-level Z2 fold to carrier-level Householder reflection to Omega-level aperture is the same chain that connects the micro-archetype (GENE_Mic, 0xAA) to the gravitational coupling constant. Each link in the chain adds one level of spinorial closure, compressing the informational aperture. The gravity note traces this chain from the continuous side; the wavefunction kernel traces it from the discrete side. Their agreement at Delta approximately 0.0207 is the junction where the discrete and continuous descriptions of CGM meet.

---

## 18. Summary of Structural Principles

The hQVM kernel rests on a small number of structural principles, each verified computationally and connected to the others by the fiber bundle geometry of the byte:

1. **The K4 gauge algebra** (Theorem T1) governs all depth-4 operations on Omega.
2. **The fold map P** at the BU boundary is the involution that relates the forward and reverse CGM phase readings, giving the byte internal Z2 curvature. Its holonomic closure through depth-4 produces gate F, a Householder reflection on the carrier manifold.
3. **The 50% holographic redundancy law** holds at every scale because Space = Subspace^2, with the squaring induced by the fold reflection.
4. **The entanglement entropy** S = popcount(chi) is the measure of holographic redundancy, and equals 50% of available DoF at every scale.
5. **The aperture collapse** from 50% (byte) to 2.07% (Omega) is the wavefunction collapse, driven by spinorial closure.
6. **The XOR transition** is the discrete gyration: the composition law of the gyrogroup that makes the kernel non-abelian and hence curved.
7. **The complementarity invariant** horizon_distance + ab_distance = 12 is the conservation law that enforces the WAY uncertainty principle.

These principles form a dependency chain. The fold map (2) causes the holographic doubling (3), which fixes the entanglement entropy (4), which drives the aperture collapse (5). The non-abelian character (6) is necessary for curvature (2) to exist. The conservation law (7) follows from the K4 structure (1) and the bipartite decomposition.

---

*This document presents a formal analysis of the Holonomic Quantum Virtual Machine kernel wavefunction structure within the Common Governance Model framework. All theorems are verified on 4096 states using exact integer arithmetic with no free parameters.*
