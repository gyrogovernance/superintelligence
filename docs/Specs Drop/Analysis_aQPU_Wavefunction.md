# CGM Wavefunction Analysis: Spectral Structure of the aQPU Kernel

**Citation:** Korompilias, B. (2025). Common Governance Model: Mathematical Physics Framework. Zenodo. https://doi.org/10.5281/zenodo.17521384

---

## 1. Introduction and Scope

This analysis presents the wavefunction structure of the Gyroscopic ASI aQPU kernel as a concrete finite-dimensional realization of the Common Governance Model (CGM) conditions. All results are verified by exhaustive computation on the 4096-state reachable manifold Ω using exact integer arithmetic.

The central finding is that the CGM constraint structure manifests in the kernel as a **Klein four-group (K4) of operators** acting on a Hilbert space over Ω. The BU-Egress/Ingress duality, far from being two sequential stages, emerges as two dual readings of a single depth-4 spectral property: the W₂ involution that pairs the two constitutional poles of Ω.

### 1.1 Corrected Principles

The analysis rests on three corrections to prior interpretations:

| Prior assumption | Corrected understanding |
|---|---|
| Carrier rest (0xAAA555) = CS | CS is GENE_Mic (0xAA), the transcription frame; carrier rest is a point on the complement horizon |
| BU-Egress then BU-Ingress as sequential | Egress and Ingress are dual readings of the same depth-4 event |
| Depth-8 = new modal depth | Depth-8 = K4 composition of two depth-4 involutions |

### 1.2 The Foundational Distinction

Three objects must remain strictly separated:

| Object | Role | Re-enterable? |
|--------|------|---------------|
| **CS** (horizon constant S / GENE_Mic) | Transcription origin: `intron = byte ⊕ 0xAA` | No - it is the reference frame, not a state in Ω |
| **GENE_Mac rest** (0xAAA555) | Point on complement horizon (shell 0) | Yes - carrier can return via Z2 holonomy |
| **GENE_Mac swapped** (0x555AAA) | Z₂ partner of rest on complement horizon | Yes - the other sheet of the double cover |

The unobservability of CS is enforced structurally: GENE_Mic determines all correlations via transcription but cannot itself be observed as a state. The non-cloning theorem prevents duplicating the reference frame by any operation defined within it.

---

## 2. The Three Computational Spaces

The kernel admits three distinct computational spaces, related by exact projections:

### 2.1 Modal Space

The CGM modal logic defines operations at depths 0, 2, and 4:

| Depth | CGM constraint | Kernel realization |
|-------|---------------|-------------------|
| 0 | CS: `[R]S ↔ S ∧ ¬([L]S ↔ S)` | Carrier at rest on complement horizon; family 00 preserves horizon |
| 2 | UNA: `S → ¬□E` | After byte 1: carrier departs horizon; byte order matters |
| 4 | BU: `S → □B` | After byte 2: commutator vanishes in S-sector projection |

Each byte implements one full `[L][R]` operation (the L-step mutates A; the R-step performs gyration). Therefore:
- **1 byte = depth 2** in CGM modal nesting
- **2 bytes = depth 4** = the BU condition

### 2.2 Constitutional Space

The 4096-state manifold Ω partitions into three constitutional sectors with seven shells:

| Sector | States | Shells | Description |
|--------|--------|--------|-------------|
| Complement horizon | 64 | 0 | Maximal chirality (A = B ⊕ 0xFFF) |
| Bulk | 3968 | 1-5 | Partial chirality |
| Equality horizon | 64 | 6 | Zero chirality (A = B) |

The shell distribution follows the binomial law: `count(w) = C(6,w) × 64` for chirality weight w ∈ {0,...,6}. The holographic identity `|H|² = |Ω|` holds for both horizons: `64² = 4096`.

### 2.3 Carrier Space (Z₂)

Within each shell, states carry a Z₂ coordinate: the distinction between rest and swapped positions. This coordinate is invisible to chirality (both rest and swapped have the same χ₆ value) but determines the carrier's provenance.

Gate F acts as the Z₂ flip: `F|rest⟩ = |swapped⟩`, `F|swapped⟩ = |rest⟩`, `F² = I`.

---

## 3. The K4 Operator Algebra

**Theorem T1.** For every micro_ref m ∈ {0,...,63}, the operators {id, W₂(m), W₂'(m), F(m)} form a Klein four-group under composition, where:

- **W₂(m)** = [byte(fam 00, m), byte(fam 01, m)] - depth-4 half-word (families 00, 01)
- **W₂'(m)** = [byte(fam 10, m), byte(fam 11, m)] - depth-4 half-word (families 10, 11)
- **F(m)** = W₂(m) ∘ W₂'(m) - full canonical word (families 00, 01, 10, 11)

The composition table is:

```
  ∘   |  id   W₂   W₂'   F
  ----+------------------------
  id  |  id   W₂   W₂'   F
  W₂  |  W₂   id   F     W₂'
  W₂' |  W₂'  F    id    W₂
  F   |  F    W₂'  W₂    id
```

Each element is an involution: W₂² = W₂'² = F² = id. Verified for all 64 micro_refs on all 4096 states.

### 3.1 Signature Structure

On the Omega12 chart, the four K4 elements have signatures:

| Operator | Parity | τ_u6 | τ_v6 | K4 gate |
|----------|--------|-------|-------|---------|
| id | 0 | 0 | 0 | identity |
| W₂ | 0 | 62 | 1 | pole swap (comp→eq) |
| W₂' | 0 | 1 | 62 | pole swap (comp→eq) |
| F | 0 | 63 | 63 | Z₂ carrier flip |

The signatures are micro_ref-dependent for W₂ and W₂' (τ values shift), but the K4 structure is universal.

---

## 4. Constitutional Pole Dynamics

### 4.1 Pole Swap (Theorems T2, T3)

**Theorem T2.** W₂ maps shell s → 6−s for all states in Ω.

**Algebraic proof:** In Omega12 coordinates, W₂ acts as:
```
(u, v) → (v ⊕ m ⊕ 63, u ⊕ m)
```
Therefore:
```
χ' = u' ⊕ v' = (v ⊕ m ⊕ 63) ⊕ (u ⊕ m) = (u ⊕ v) ⊕ 63 = χ ⊕ 63
```
Since popcount(χ ⊕ 63) = 6 − popcount(χ), we have shell' = 6 − shell.

**Theorem T3.** W₂' maps shell s → 6−s identically.

The algebraic proof is symmetric: W₂' acts as (u, v) → (v ⊕ m, u ⊕ m ⊕ 63), giving the same χ' = χ ⊕ 63.

**Consequence:** Both W₂ and W₂' map the complement horizon (shell 0) to the equality horizon (shell 6), and vice versa. The two constitutional poles are linked by the depth-4 operation.

### 4.2 Shell Preservation (Theorem T4)

**Theorem T4.** Gate F preserves shell.

**Proof:** F = W₂ ∘ W₂'. Two pole swaps compose to identity on the radial coordinate:
```
χ_F = χ ⊕ 63 ⊕ 63 = χ
```

Gate F acts as the Z₂ flip *within* each shell, pairing states that share the same chirality but differ in carrier position. The radial structure is preserved; only the angular (provenance) coordinate changes.

### 4.3 Depth-4 Confinement (Theorem T5)

**Theorem T5.** At depth 4 (2 bytes from any canonical half-word), the carrier is confined to the opposite constitutional pole.

From the complement horizon → equality horizon: 64/64 states.
From the equality horizon → complement horizon: 64/64 states.

This is a forced consequence of χ ⊕ 63: the chirality inversion at depth 4 maps every state to its antipodal shell. There is no depth-4 path that preserves the constitutional pole (except the trivial identity, which is not a W₂-type operation).

---

## 5. Depth Decomposition and CS Ordering

### 5.1 Depth-8 as K4 Composition (Theorem T6)

**Theorem T6.** The canonical 4-byte word is F = W₂ ∘ W₂'. Depth-8 is K4 composition, not a new modal depth.

The carrier trajectory through the decomposition:

| Stage | Operator | Carrier position | Constitutional |
|-------|----------|-----------------|----------------|
| Start | - | (0, 63) | Complement horizon, rest |
| After W₂ | depth 4 | (62, 62) | Equality horizon |
| After W₂' | depth 8 | (63, 0) | Complement horizon, swapped |

No new modal depth is introduced at depth 8. The second depth-4 operation (W₂') composes with the first via the K4 algebra, producing the Z₂ carrier flip.

### 5.2 CS Forces Canonical Ordering (Theorem T7)

**Theorem T7.** The canonical family ordering (families 00, 01, 10, 11) is forced by the CS axiom.

The CS axiom states: `[R]S ↔ S ∧ ¬([L]S ↔ S)` - right transitions preserve the horizon while left transitions alter it.

In the kernel, family 00 (L0 parity = 0) acts as the `[R]`-preserving transition: from rest on the complement horizon, a family-00 byte with mask m produces diff₁ = 0xFFF ⊕ m, and a subsequent L-step with the same mask returns to diff = 0xFFF (complement horizon). Family 01 (L0 parity = 1) acts as the `[L]`-altering transition: the complement in A_next breaks the return to the complement horizon and instead forces the equality horizon.

| Family ordering | L-step result | CGM reading |
|----------------|--------------|-------------|
| 00 first | diff = 0xFFF (complement horizon) | `[R]S ↔ S`: horizon preserved |
| 01 first | diff = 0x000 (equality horizon) | `¬([L]S ↔ S)`: horizon altered |

CS selects family 00 as the first byte because only this ordering preserves the complement horizon under the intermediate L-step. This is not a convention; it is a structural consequence of the CS chirality condition.

---

## 6. BU-Egress and Ingress as Spectral Duality

### 6.1 Egress: The W₂ Involution (Theorem T8)

**Theorem T8.** BU-Egress is the W₂ involution: the spectral property that the depth-4 operator squares to identity on Ω.

The CGM BU-Egress condition `S → □B` requires the depth-4 commutator to vanish in the S-sector. In the kernel, this is verified:

1. **Byte-order sensitivity (UNA):** For two bytes with different families, T(b₀, b₁) ≠ T(b₁, b₀) in general. Order matters globally.
2. **S-sector closure (□B):** Both orderings project onto the same constitutional sector (equality horizon). The commutator vanishes in the S-sector.
3. **Primitive verification:** For every byte b and every complement-horizon start state, LRLR(s; b) = RLRL(s; b) and both remain on the complement horizon. Verified for all 64 complement-horizon states × all canonical bytes.

W₂ is an involution (W₂² = id) with eigenspace dimensions dim(+1) = 2048, dim(−1) = 2048. The □B condition is the statement that W₂ maps the S-sector (complement horizon) onto the equality horizon as a perfect pairing.

### 6.2 Ingress: Pole-Pairing as Memory (Theorem T9)

**Theorem T9.** BU-Ingress is the W₂ pole-pairing: each complement-horizon state is paired with a unique equality-horizon shadow, and the pairing is invertible (W₂(shadow) = original).

The CGM BU-Ingress condition `S → (□B → (CS ∧ UNA ∧ ONA))` requires the balanced state to encode memory of all prior conditions. In the kernel:

- **W₂ pairs** 64 complement-horizon states with 64 equality-horizon states (and 3968 bulk states with bulk states in antipodal shells)
- **Each pairing is invertible:** W₂(W₂(s)) = s for all s ∈ Ω
- **The shadow encodes the origin:** For rest, W₂(rest) = (62, 62) on the equality horizon. This equality-horizon state *is* the Ingress memory - it carries the structural information that the origin was on the complement horizon at shell 0

The representative shadow pairs illustrate the structure:

| Complement horizon | Equality horizon shadow |
|-------------------|----------------------|
| (63, 0) χ=111111 swapped | (1, 1) χ=000000 |
| (62, 1) χ=111111 | (0, 0) χ=000000 |
| (0, 63) χ=111111 rest | (62, 62) χ=000000 |

Each complement-horizon state (shell 0, maximal chirality) is paired with an equality-horizon state (shell 6, zero chirality). The shadow is the "memory" of the original: it is the unique state that, when W₂ is applied again, reconstructs the original.

### 6.3 The Duality

Egress and Ingress are **the same W₂ operator read two ways**:

| Reading | Question | Answer |
|---------|----------|--------|
| Egress | Does closure hold? | W₂² = id: yes, the depth-4 operation is an involution |
| Ingress | Does closure carry memory? | W₂ pairs poles invertibly: yes, the shadow reconstructs the origin |

These are not sequential stages. They are simultaneous aspects of the same spectral property. The Z2 holonomy (gate F = W₂ ∘ W₂') is the holographic encoding that makes both readings true at depth 4.

---

## 7. Chirality Transport Algebra

### 7.1 Per-Byte Decomposition (Theorem T10)

**Theorem T10.** Each depth-4 half-word fully inverts chirality: q(W₂) = q(W₂') = 63 for all m. The full canonical word preserves chirality: q(F) = 0.

The per-byte chirality increments are:

| Family | L0 parity | q(byte(fam, m)) |
|--------|-----------|------------------|
| 00 | 0 | m |
| 01 | 1 | m ⊕ 63 |
| 10 | 1 | m ⊕ 63 |
| 11 | 0 | m |

Therefore:
```
q(W₂)  = q(fam 00, m) ⊕ q(fam 01, m) = m ⊕ (m ⊕ 63) = 63
q(W₂') = q(fam 10, m) ⊕ q(fam 11, m) = (m ⊕ 63) ⊕ m = 63
q(F)   = q(W₂) ⊕ q(W₂')             = 63 ⊕ 63       = 0
```

### 7.2 Physical Interpretation

The depth-4 half-word performs a **complete chirality inversion**: all six chirality bits flip. This is the discrete analogue of a π-rotation in the chirality register. The micro_ref m determines *which specific 2-cycle* each state enters, but it does not affect the chirality transport magnitude.

The full canonical word composes two complete inversions, which cancel: `63 ⊕ 63 = 0`. Gate F preserves chirality while acting non-trivially on the carrier. This is the kernel's realization of the statement that **holonomy acts on the carrier subspace only, not on chirality**.

The chirality register is the "radial" coordinate (shell membership); the carrier position is the "angular" coordinate (position within shell). The Z2 holonomy flips the angular coordinate while preserving the radial coordinate.

---

## 8. The Wavefunction Structure

### 8.1 The Permutation on Ω

The canonical word W = (0xA8, 0xA9, 0x28, 0x29) generates a permutation on Ω classified as **gate F**:

- **Signature:** OmegaSignature12(parity=0, τ_u6=63, τ_v6=63)
- **Action:** (u, v) → (u ⊕ 63, v ⊕ 63)
- **Cycle structure:** 2048 two-cycles, 0 fixed points, 0 longer cycles

The permutation is a perfect involution: U_W² = I on every state in Ω.

### 8.2 Eigenspace Decomposition

The Hilbert space ℂ⁴⁰⁹⁶ decomposes under U_W into:

| Eigenspace | Dimension | Description |
|------------|-----------|-------------|
| +1 | 2048 | Symmetric superpositions: `\|+⟩ = (\|s⟩ + \|W(s)⟩)/√2` |
| −1 | 2048 | Antisymmetric superpositions: `\|−⟩ = (\|s⟩ − \|W(s)⟩)/√2` |

The rest state decomposes as:
```
|rest⟩ = (|+⟩ + |−⟩)/√2
```

Under one application:
```
U_W|rest⟩ = F|rest⟩ = |swapped⟩ = (|+⟩ − |−⟩)/√2
```

The system oscillates between |rest⟩ and |swapped⟩ with period 2 in the word-count variable.

### 8.3 Sector-Resolved Dimensions

| Sector | dim(+1) | dim(−1) | States |
|--------|---------|---------|--------|
| Complement horizon | 32 | 32 | 64 |
| Equality horizon | 32 | 32 | 64 |
| Bulk | 1984 | 1984 | 3968 |
| **Total** | **2048** | **2048** | **4096** |

The eigenspaces are uniformly distributed across constitutional sectors. Each sector contributes proportionally to both eigenspaces, confirming that the Z2 holonomy is a global property of Ω, not confined to any particular constitutional region.

### 8.4 The Holonomy Is Spectral, Not Trajectory

The Z2 holonomy is a property of the operator spectrum, not of the carrier trajectory. The basis states |rest⟩ and |swapped⟩ are not eigenvectors of U_W; they are superpositions of the +1 and −1 eigenvectors. The holonomy "phase" (the distinction between rest and swapped) is encoded in the relative sign between the +1 and −1 components.

This is why the holonomy cannot be understood by tracing the carrier alone. The carrier oscillates between rest and swapped, but the underlying spectral structure is the ±1 decomposition of the Hilbert space.

---

## 9. Holographic Dictionary

### 9.1 Shadow Partners under Gate F

Gate F creates 2048 shadow pairs on Ω, each consisting of two states related by (u, v) ↔ (u ⊕ 63, v ⊕ 63). The pairing is:

- **Confined within each shell:** F preserves chirality, so shadow partners share the same shell
- **Confined within each sector:** complement ↔ complement, equality ↔ equality, bulk ↔ bulk
- **Universally 2-cycle:** no state is a fixed point of F

The canonical shadow pair is the carrier orbit:

```
|rest⟩ = (0, 63)  ↔  |swapped⟩ = (63, 0)
```

Both states are on the complement horizon (shell 0, χ = 111111), differing only in carrier position.

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

The number of shadow pairs per shell is exactly half the shell population: `pairs(w) = C(6,w) × 32`. This follows from gate F's action: each pair (u, v) ↔ (u ⊕ 63, v ⊕ 63) links two states within the same shell, and the mapping is a fixed-point-free involution.

### 9.3 The 4-to-1 Dictionary

The holographic dictionary on Ω states that each bulk state corresponds to exactly 4 (horizon, byte) preimages. In the wavefunction picture, this becomes: each bulk state has 4 preimages under the canonical word's action on the horizon. The Z₂ encoding (rest vs swapped) accounts for 2 of the 4 preimages; the remaining factor of 2 comes from the byte-shadow degeneracy (each byte has a shadow partner producing the same 24-bit action).

---

## 10. The Helix: Holonomy Cycle Structure

### 10.1 Z₂ Oscillation

The carrier coordinate under repeated canonical words follows:

```
rest → swapped → rest → swapped → ...
```

with period 2 in the word-count variable. This is the Z2 holonomy cycle. The "helix" metaphor is precise: the system overlays the origin without revisiting it. Each return to the complement horizon occurs on alternating Z₂ sheets.

### 10.2 Constitutional Events per Turn

Each 4-byte turn produces an identical constitutional trajectory:

| Byte | Depth | Sector | Shell | Z₂ | Event |
|------|-------|--------|-------|-----|-------|
| 1 | 2 | Bulk | 1 | - | Departure from horizon (UNA: variety introduced) |
| 2 | 4 | Equality | 6 | - | Transient equality (ONA: opposition non-absolute) |
| 3 | 6 | Bulk | 1 | - | Return toward horizon (approaching closure) |
| 4 | 8 | Complement | 0 | alternating | Horizon with Z₂ encoding (BU holographic) |

The constitutional trajectory is symmetric about the equality transit (byte 2): shells follow the pattern [0, 1, 6, 1, 0]. The equality horizon is always transient - the system passes through it but cannot remain, satisfying UNA (¬□E: unity is non-absolute).

### 10.3 No Return to CS

The Z₂ oscillation between rest and swapped is the completion of the holonomy cycle, not a return to the common source. CS (GENE_Mic) is the transcription frame within which all operations are defined. The carrier can return to rest, but this is the completion of the Z₂ cycle - a new iteration of UNA (departure from horizon) begins with the next byte, not a re-instantiation of CS.

---

## 11. Connection to CGM Continuous Framework

### 11.1 The BU Holonomy Angle as Spectral Gap

In the continuous CGM framework, the BU holonomy angle δ_BU ≈ 0.1953 rad measures the residual geometric phase of the dual-pole loop. In the kernel, this corresponds to the spectral gap between the +1 and −1 eigenspaces: the Z2 holonomy phase that distinguishes rest from swapped.

The aperture ratio δ_BU/m_a ≈ 0.9793 becomes, in the discrete framework, the ratio of paired-to-unpaired structure: 4096/4096 = 1.0 within each shell (all states are paired), with the 2.07% aperture manifesting as the Z₂ encoding itself - the distinction between the two sheets that are otherwise identical in chirality.

### 11.2 Spin-2 from Two-Pass Carrier Return

The gravitational coupling form κ = 8πG/c⁴ contains the factor 8π = 2 × Q_G = 2 × 4π. The factor of 2 arises from the two-pass carrier return: one pass for Egress (W₂), one for Ingress (W₂'). In the kernel:

```
Egress circulation:  +2
Ingress circulation: −2
Net cancellation:     0
```

The spin-2 signature of gravitation is the Z2 holonomy cycle: two applications of the canonical word to return the carrier to rest. This is not a fitted parameter; it is the algebraic consequence of F = W₂ ∘ W₂'.

### 11.3 The Refractive Depth as K4 Composition

The gravitational Refractive Depth formula:

```
τ_G = |Ω| · Δ · ρ⁵ · (1 − 4ρΔ²)
```

decomposes into kernel invariants:

- |Ω| = 4096: the manifold size
- Δ ≈ 0.0207: the aperture gap
- ρ⁵: the STF attenuation (5 bulk shells × ρ per shell)
- (1 − 4ρΔ²): the K4 correction (4 intrinsic gates, each contributing ρΔ²)

The K4 correction factor is the spectral signature of the Z2 holonomy structure. Without it (using only |Ω| · Δ · ρ⁵), the Refractive Depth is 76.366; with it, the value drops to 76.238, matching the required 2 ln(E_CS/v_EW) = 76.238 to 25 ppm.

---

## 12. Spectral Comparison of Operators

| Operator | Signature | Chirality map | dim(+1) | dim(−1) | rest → | q |
|----------|-----------|---------------|---------|---------|--------|---|
| W₂ | (62, 1) | s → 6−s | 2048 | 2048 | equality | 63 |
| W₂' | (1, 62) | s → 6−s | 2048 | 2048 | equality | 63 |
| F | (63, 63) | s → s | 2048 | 2048 | swapped | 0 |
| id | (0, 0) | s → s | 4096 | 0 | rest | 0 |

Key observations:

1. **All non-trivial K4 elements are involutions** with equal +1 and −1 eigenspace dimensions (2048 each).
2. **W₂ and W₂' are related by the transposition** (τ_u6, τ_v6) ↔ (τ_v6, τ_u6). They produce the same spectral structure but different specific pairings.
3. **Gate F is the only K4 element that preserves chirality.** The pole-swap operators (W₂, W₂') fully invert chirality, while the Z₂ carrier flip preserves it.
4. **Same-family words (4 bytes from one family) produce the identity** with dim(−1) = 0. They achieve depth-4 closure trivially without holographic encoding.

---

## 13. Probe Suite Summary

| Word | Length | Gate | +1 | −1 | Involution | rest → | χ preserved |
|------|--------|------|----|----|------------|--------|-------------|
| canonical | 4 | F | 2048 | 2048 | Y | swapped | Y |
| canonical ×2 | 8 | id | 4096 | 0 | Y | rest | Y |
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

1. **K4 failure:** Demonstrate that {id, W₂, W₂', F} fails to close as a Klein four-group for some micro_ref m. (Currently verified for all 64.)

2. **Confinement failure:** Find a depth-4 path from rest that does not land on the equality horizon. (Currently verified: impossible due to χ ⊕ 63.)

3. **Chirality non-cancellation:** Find a micro_ref where q(F) ≠ 0. (Currently: q(F) = 63 ⊕ 63 = 0 for all m, provable from L0 parity structure.)

4. **Fixed points of F:** Find a state in Ω that is fixed by gate F. (Currently: none exist; F is a fixed-point-free involution on all 4096 states.)

5. **Spectral asymmetry:** Find that dim(+1) ≠ dim(−1) for any non-trivial K4 element. (Currently: 2048 = 2048 for W₂, W₂', F.)

6. **Non-involution:** Find that W₂² ≠ id on some state in Ω. (Currently verified: impossible.)

---

## 15. Theorem Summary

| Theorem | Statement | Status |
|---------|-----------|--------|
| T1 | {id, W₂, W₂', F} is K4 for every m | Verified, 64 × 4096 states |
| T2 | W₂ maps shell s → 6−s (χ ⊕ 63) | Verified, algebraic proof |
| T3 | W₂' maps shell s → 6−s (χ ⊕ 63) | Verified, algebraic proof |
| T4 | F preserves shell (Z₂ within pole) | Verified, algebraic proof |
| T5 | Depth-4 confines to opposite pole | Verified, 64 × 64 states |
| T6 | Depth-8 = K4 composition, not new depth | Verified, signature algebra |
| T7 | CS forces canonical family ordering | Verified, 64 micro_refs |
| T8 | Egress = W₂ involution (□B spectral) | Verified, 4096 states |
| T9 | Ingress = W₂ pole-pairing (shadow = memory) | Verified, 4096 states |
| T10 | q(W₂) = q(W₂') = 63; q(F) = 0 for all m | Verified, algebraic proof |

All theorems verified on 4096 states using exact integer arithmetic with no free parameters.

---

*Document prepared as a formal analysis of the aQPU kernel wavefunction structure within the Common Governance Model framework.*