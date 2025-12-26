# GGG ASI Alignment Router
Kernel Specification

## 0. Status and Scope

This document specifies the **GGG ASI Alignment Router kernel** as a deterministic coordination and measurement system. The kernel is designed to be:

- **Finite**: Operating on a closed state space (the ontology).
- **Deterministic**: Fully replayable from an append-only ledger.
- **Non-semantic**: Containing no interpretation of natural language meaning.
- **Alignment-measurable**: Providing constitutional observables derived from the Common Governance Model (CGM).

The Router functions as a reference implementation for **Gyroscopic Global Governance (GGG)**, converting byte streams into verifiable governance signatures.

### 0.1 Terminology

- **CGM (Common Governance Model)**: A modal logic framework identifying four recursive stages (Governance, Information, Inference, Intelligence) required for coherent operation. It defines a continuous canonical aperture constant `A* ≈ 0.0207` (theory-side). 
- **THM (The Human Mark)**: A source-type ontology classifying Authority and Agency as either Authentic (human-originated) or Derivative (system-mediated). The Router kernel is a derivative coordination system.
- **GGG (Gyroscopic Global Governance)**: A four-domain framework (Economy, Employment, Education, Ecology) that applies CGM structural constraints to sociotechnical coordination.
- **AGI (Operational Definition)**: A regime where heterogeneous human and artificial capabilities coordinate via stable, traceable protocols across domains.
- **ASI (Operational Definition)**: A network equilibrium state where the Router's aperture observable `A` is sustained near the router's canonical target `A*_router` (Superintelligence Index ≈ 100), minimizing systemic displacement.

---

## 1. Constitutive Axioms

The Router is defined by a minimal set of constitutive axioms. These choices define the formal system; all other properties are derived or measured consequences.

### 1.1 Interface Axiom
The external interface alphabet is the set of 8-bit bytes `{0, ..., 255}`.

**Physics Justification:** The complete 8-bit instruction space (256 actions) ensures no omissions. Each byte value maps to a valid action via the transcription constant `0xAA`, providing a complete, deterministic action space.

**Source-Type Classification**: Under The Human Mark (THM), the Router is a **Derivative** coordination system. It transforms information but does not originate authority or bear accountability. It mediates the flow: `[Authority:Authentic] -> [Authority:Derivative] + [Agency:Derivative] -> [Agency:Authentic]`.

### 1.2 Boundary Map Axiom
Input bytes `b` are mapped to internal actions `a` via a fixed involution:
`a = b XOR 0xAA`
where `0xAA` (binary `10101010`) is the transcription constant.

**Action-Byte Topology:** The 8-bit action space has a geometric structure:
- **Bits 0,7**: Anchor bits (`ACTION_ANCHOR_BITS_MASK = 0x81`) — do not drive XFORM_MASK transformations
- **Bits 1-6**: Active bits (`ACTION_ACTIVE_BITS_MASK = 0x7E`) — drive transformations via bit families (LI, FG, BG)

The six privileged actions are exactly the one-hot bytes covering bits 1-6: `{0x02, 0x04, 0x08, 0x10, 0x20, 0x40}`.

**Interface Gauge:** `GENE_Mic_S = 0xAA` is an **interface gauge choice** (transcription constant). The property that `0xAA XOR b` defines an involution is shared by any balanced alternating byte (e.g., `0x55` also has this property). The atlas physics (ontology, epistemology, observables) are built in internal action space and do not depend on the specific choice of `GENE_Mic_S` unless proven otherwise.

### 1.3 State Representation Axiom
The internal state `s` is a 48-bit integer in `[0, 2^48)`.
This integer encodes a tensor of shape `(4, 2, 3, 2)` with values `{+1, -1}`, packed in C-order big-endian format. The tensor structure corresponds to the four CGM stages (layers), dual observation frames, three spatial axes, and polarity endpoints.

**Physics Interpretation:** The 48-bit tensor is a **discrete Bloch sphere analogue** where:
- **4 layers** = CGM stages (CS → UNA → ONA → BU)
- **2 frames** = pole/sign dual (frame 1 is the sign inversion of frame 0 at the archetype: `T[layer,0,:,:] = -T[layer,1,:,:]`)
- **3 rows** = spatial axes (3 degrees of freedom)
- **2 cols** = polarity endpoints (±1 states)

**Frame Axis Semantics:** The frame dimension encodes a **pole/sign flip**, not "foreground/background". At the archetype state, frame 1 is the exact negation of frame 0 for all layers. This aligns with the ± action language (UNA_P/UNA_M, ONA_P/ONA_M, BU_P/BU_M) and represents the dual observation perspectives required by CGM's two-frame structure.

This structure provides the minimum representation needed for 4-stage × 2-frame × 3-axis × 2-endpoint topology. The 48-bit size is not arbitrary but the minimal encoding for this CGM structure.

### 1.4 Archetype Axiom
The system defines a unique archetypal state `s_ref` (tensor `GENE_Mac_S`) with alternating patterns across layers. This state serves as the seed for ontology generation and the reference for deviation measurements.

### 1.5 Transition Law Axiom
The state evolves under a deterministic transition function `T(s, a)`.
The transition logic is defined by 48-bit masks:
- `XFORM_MASK[a]` (state inversion pattern, depends only on bits 1-6 of action)
- `ACTION_BROADCAST_MASKS[a]` (gating pattern, replicates full 8-bit action across 6 bytes)

`s_next = (s XOR XFORM_MASK[a]) XOR ((s XOR XFORM_MASK[a]) AND ACTION_BROADCAST_MASKS[a])`

**Physics Interpretation:** The two-mask system implements discrete gyration operators:
- **XFORM_MASK**: Defines which bits to flip based on action's bit families (LI, FG, BG) derived from bits 1-6. Anchor bits (0,7) do not affect XFORM_MASK (verified: actions with same bits 1-6 have identical XFORM_MASK regardless of anchor bits).
- **ACTION_BROADCAST_MASKS**: Gates the transformation by replicating the full 8-bit action pattern across 6 byte positions, allowing anchor bits to appear in the state representation while not driving transformations.

This design maps 8-bit actions to 48-bit state transformations via Pauli-like operators, where bit families (LI, FG, BG) correspond to different transformation classes.

---

## 2. Derived Artifacts

From the constitutive axioms, the following artifacts are deterministically derived. These artifacts are persisted as the **Atlas**.

### 2.1 Map 1: Ontology (CS - Existence)
**File:** `ontology_keys.npy`
**Type:** `uint64[N]` (values < 2^48)

The set `Ω` of all states reachable from the archetype `s_ref` under recursive application of all actions `a`.
- **Measured Invariant:** `N = 788,986` states.
- **Measured Invariant:** Graph diameter from archetype is 6.

### 2.2 Map 2: Epistemology (UNA - Dynamics)
**File:** `epistemology.npy`
**Type:** `int32[N, 256]`

The complete transition table for the ontology.
`epistemology[i, a] = j` such that `s_j = T(s_i, a)`.
- **Closure Property:** For all `i, a`, `j` is a valid index in `[0, N)`.

### 2.3 Map 3: S-Sector Mask (BU-Egress - Horizon)
**File:** `s_mask.npy`
**Type:** `uint64[1]`

The S-sector mask defines the horizon bits where BU-Egress balance holds universally across all states. BU-Egress balance is defined as "alternating 4-step words commute" in the S-sector.

**Derived Invariant:** `H = 40` bits (83.3% of state space). This is **not an empirical measurement** but a **complete algebraic derivation** from the transition law structure.

#### Algebraic Derivation of H=40

The S-sector structure is forced by the transition law algebra, not discovered empirically. The derivation proceeds in three steps:

**Step 1: Gate-clear asymmetry identifies candidates**
- For alternation actions `alt_a = UNA_P` (0x02) and `alt_b = ONA_P` (0x04):
  - `asymmetric_clear = ACTION_BROADCAST_MASKS[alt_a] ^ ACTION_BROADCAST_MASKS[alt_b] = 0x060606060606` (12 bits)
- These 12 bits are the only positions where UNA_P and ONA_P differ in their gate-clear behavior.
- **All 8 excluded bits ⊆ asymmetric_clear** (proven invariant).

**Step 2: XFORM asymmetry creates layer-dependent cancellation**
- `asymmetric_xform = XFORM_MASK[alt_a] ^ XFORM_MASK[alt_b] = 0x000fff000fff` (24 bits, only layers 1,3)
- For bits in layers {1,3} that are in both `asymmetric_clear` AND `asymmetric_xform`, the XFORM flipping creates symmetric cancellation → these bits are **INCLUDED**.
- For bits in layers {0,2}, there's no XFORM cancellation → these bits are **EXCLUDED**.

**Step 3: Final selection rule**

A bit in `asymmetric_clear` is **excluded** if:
1. `clear_b = 1` (ONA_P clears it) → 6 bits excluded, OR
2. `clear_a = 1` AND layer ∈ {0, 2} (CS/ONA layers) → 2 additional bits excluded

A bit in `asymmetric_clear` is **included** if:
- `clear_a = 1` AND layer ∈ {1, 3} (UNA/BU layers) AND `in_asymmetric_xform` → XFORM creates cancellation → 4 bits included

**Result:** Exactly 8 of 12 `asymmetric_clear` bits are excluded, giving `H = 48 - 8 = 40`.

#### The 8 Excluded Bits (Derived, Not Measured)

| Bit | Layer | Stage | Coordinate | Reason |
|-----|-------|-------|------------|--------|
| 02 | 3 | BU | (1,1,1) | `clear_b=1` |
| 10 | 3 | BU | (0,0,1) | `clear_b=1` |
| 17 | 2 | ONA | (1,0,0) | `clear_a=1` AND layer∈{0,2} |
| 18 | 2 | ONA | (0,2,1) | `clear_b=1` |
| 26 | 1 | UNA | (1,1,1) | `clear_b=1` |
| 34 | 1 | UNA | (0,0,1) | `clear_b=1` |
| 41 | 0 | CS | (1,0,0) | `clear_a=1` AND layer∈{0,2} |
| 42 | 0 | CS | (0,2,1) | `clear_b=1` |

**Layer-parity pairing:** Layers {0,2} (CS, ONA) share one exclusion pattern; layers {1,3} (UNA, BU) share another. This corresponds to FG (Forward Gyration) vs BG (Backward Gyration) layer structure, where XFORM asymmetry only affects BG layers (1,3), enabling cancellation.

**Frame-disjoint exclusion:** The excluded coordinates are **disjoint across frames**. Frame 0 and frame 1 exclude different (layer, row, col) positions. This indicates that frames are **pole-dual** with complementary roles: each frame carries different non-horizon coordinates, forming a dual partition of the geometry rather than independent structures.

**Physics Justification:** The S-sector represents the **horizon bits where BU-Egress balance holds universally**. The 8 excluded bits are **forced by the transition law algebra**—specifically, the interaction between gate-clear asymmetry and XFORM asymmetry across FG/BG layer pairs. This is a **derived invariant**, not an empirical measurement.

### 2.4 Map 4: φ₈ Defect (BU-Ingress - Alignment)
**File:** `defect_phi8.npy`
**Type:** `uint8[N]`

The primary physical observable: integer φ₈ monodromy defect on the S-sector.

For each state `i`:
1. Apply the φ₈ loop (WORD_PHI_8) via the epistemology table to reach state `j`.
2. Compute defect `d = popcount((s_i XOR s_j) AND s_mask)` on the S-sector.

**BU-Egress Alternation Pair:** The S-sector is defined by the alternation pair `(UNA_P, ONA_P)` = `(0x02, 0x04)`. In this atlas realization, this pair is **designated as the BU-Egress alternation** because it satisfies depth-4 alternation commutation on the S-sector. Among all tested action pairs, only `(UNA_P, ONA_P)` achieves universal depth-4 balance (verified: 0 violations across sampled states). This is a **measured property of this discrete realization**, not a design choice. The pair is a good candidate for discrete L/R alternation semantics, but a complete family-level mapping to CGM's two modalities `[L]` and `[R]` is not yet derived.

The φ₈ loop is the canonical 8-leg toroidal holonomy: `UNA_P → ONA_P → BU_P → BU_M → ONA_M → UNA_M`.

- **Range:** `[0, 40]` (integer defect counts).
- **Measured Invariant:** Defect quantum `g=2` (GCD of all nonzero defects). All defects are even integers (`d ∈ {0, 2, 4, ..., 28}`).

**Aperture** `A = d / H` is computed on-demand as a derived view for reporting convenience, where `H = popcount(s_mask) = 40`. The minimal nonzero aperture is `A*_router = g/H = 2/40 = 0.05`.

### 2.5 On-Demand Computations
Stage profile and loop defects are computed on-demand for routing signatures (not stored):

- **Stage Profile**: `popcount((s_i XOR s_ref) AND LAYER_MASKS[k] AND s_mask)` for each layer `k ∈ {0,1,2,3}`. The per-layer maximum is `popcount(LAYER_MASKS[k] & s_mask)` (≤12), not always 12.
- **Loop Defects**: Hamming distances for three canonical commutator loops on the S-sector.

---

## 3. Monodromy and Aperture Observables

The Router measures alignment via the **φ₈ toroidal holonomy**, a canonical loop derived from CGM physics. This measures how much a state fails to close after traversing the toroidal memory path.

### 3.1 The φ₈ Loop
The φ₈ loop is the canonical 8-leg palindromic toroidal holonomy in CGM stage language: **CS→UNA→ONA→BU+→BU−→ONA→UNA→CS**.

In the discrete router, **CS is implicit** (the loop starts and ends at the same state), so the realized word has **6 actions**:
```
WORD_PHI_8 = (UNA_P, ONA_P, BU_P, BU_M, ONA_M, UNA_M)
```

This sequence of 6 actions corresponds to the 6 active bit positions (1-6) in the 8-bit micro structure, excluding the anchor bits (0,7).

**Physics Interpretation:** The 6 actions in φ₈ correspond to the **6 degrees of freedom** from CGM:
- **3 rotational DOF**: UNA_P, ONA_P, BU_P (exercising rotational degrees)
- **3 translational DOF**: BU_M, ONA_M, UNA_M (exercising translational degrees)

Each leg of φ₈ exercises one DOF, creating a complete toroidal holonomy measurement that captures the state's failure to close under the canonical loop.

### 3.2 Monodromy Defect
For a state `s_i`, the monodromy defect measures the failure to close:
1. Apply φ₈ loop: `s_j = T(s_i, WORD_PHI_8)` via epistemology table.
2. Compute defect on S-sector: `d = popcount((s_i XOR s_j) AND s_mask)`.
3. Normalize by horizon: `A = d / H` where `H = popcount(s_mask) = 40`.

This measures the toroidal memory: how "open" the state is to the canonical loop.

### 3.3 Aperture Quantization
Aperture values are quantized in discrete steps:
- **Theoretical step**: `1/H = 1/42 ≈ 0.0238`
- **Actual quantum**: `g/H = 2/42 ≈ 0.0476` where `g=2` is the GCD of all nonzero defect counts
- **Physics fact**: No states exist with `d=1`; all defects are even integers (`d ∈ {0, 2, 4, ..., 28}`)

The router's canonical aperture is `A*_router = g/H ≈ 0.0476`, the minimal nonzero aperture that the ontology actually admits.

**Discrete vs. Continuous Realization:**
- **Continuous CGM theory**: `A* ≈ 0.0207` (from Q_G = 4π, δ_BU ≈ 0.1953 rad)
- **Discrete router realization**: `A*_router ≈ 0.0476` (from defect quantum g=2, horizon H=42)

This gap (0.0207 vs 0.0476) is a **necessary consequence** of finite state representation. Aperture quantization with step `1/H ≈ 0.0238` is inherent to the discrete system. The minimal nonzero aperture `g/H ≈ 0.0476` arises from φ₈ loop preserving S-sector parity, forcing all defects to be even integers. Through systematic testing of alternation pairs, we achieved H=42 (vs original H=40), reducing the gap from 0.0293 to 0.0269 - the closest approximation achievable with the current architecture.

The continuous `A* ≈ 0.0207` remains the theory target for infinite-precision systems, while `A*_router = 0.05` is the correct canonical aperture for this discrete implementation.

#### 3.3.1 Explanation: Parity Preservation
The defect quantum `g=2` arises from a fundamental structural property: **φ₈ preserves S-sector parity**.

For any state `s`, let `p(s) = popcount(s & s_mask) mod 2` be the parity of the S-sector bits. Then:
- `p(φ₈(s)) = p(s)` (parity is invariant under φ₈)

This implies that the defect `d = popcount((s XOR φ₈(s)) & s_mask)` is always even:
- If `p(s) = p(φ₈(s))`, then the symmetric difference has even parity
- Therefore `d mod 2 = 0`, forcing `d ∈ {0, 2, 4, ...}`

This is a **measured invariant** verified across all states in the ontology (see conformance test R7). The parity preservation property is not a design choice but a consequence of the φ₈ loop structure and the S-sector definition.

### 3.4 Superintelligence Index (SI)
The alignment score `SI` is calibrated against the router's canonical aperture `A*_router = g/H`:
- `Deviation D = max(A/A*_router, A*_router/A)`
- `SI = 100 / D`

`SI = 100` implies perfect alignment with the router's canonical balance point. The continuous CGM `A* ≈ 0.0207` remains a theory-side constant but is not used for router calibration.

---

## 4. Kernel Operation

The Router kernel executes a deterministic cycle for each input byte sequence.

### 4.1 State Machine
1. **Input**: Stream of external bytes.
2. **Step**: For each byte `b`:
   - Compute action `a = b XOR 0xAA`.
   - Look up next state index `j = epistemology[current_index, a]`.
   - Update `current_index = j`.
3. **Output**: Routing Signature at the final state.

### 4.2 Routing Signature
The kernel emits a structured signature for the current state `i`:
- `state_index` (canonical ID)
- `state_int_hex` (48-bit value)
- `stage_profile` (4-vector, computed on-demand)
- `loop_defects` (3-vector, computed on-demand)
- `depth2_commutator` (2-tuple `(d2_full, d2_s)`, computed on-demand) - depth-2 sectoral commutator defect for BU-Egress pair
- `aperture` (scalar, normalized φ₈ monodromy defect)
- `si_router` (alignment index, calibrated to `A*_router`)
- `si_cgm` (alignment index, calibrated to CGM `A*`)
- `horizon_bits` (H = popcount(s_mask) = 40)
- `a_star_router` (router canonical aperture = g/H = 0.05)

This signature allows application layers to make routing decisions based on structural properties (e.g., "route to capability X if SI > 90", "enforce loop closure if defect > threshold").

### 4.3 Internal Routing Modes
The Router supports four internal modes corresponding to the GGG domains and THM displacement axes. These modes guide policy selection based on the kernel signature:

1. **Governance Management (CS)**: Emphasizes ontology identity and ledger continuity to prevent Governance Traceability Displacement (GTD).
2. **Information Curation (UNA)**: Emphasizes lawful transformation and variety to prevent Information Variety Displacement (IVD).
3. **Inference Interaction (ONA)**: Emphasizes stage-resolved differentiation (Stage Profile) to prevent Inference Accountability Displacement (IAD).
4. **Intelligence Cooperation (BU)**: Emphasizes closure and integrity (Loop Defects, Aperture) to prevent Intelligence Integrity Displacement (IID).

### 4.4 Closed Routing Cycle
Interaction is treated as a closed loop:
1. External input is transcribed to actions.
2. Internal state advances through the epistemology.
3. Routing signature is computed.
4. **Routing Policy**: A deterministic map selects a target capability based on the signature and active mode.
5. Output is transcribed and integrated into the trajectory.

### 4.5 Ledger and Noninvertibility
The kernel optionally records every transition in an append-only binary ledger for full replayability and auditing.
Record format: `[Event Code (1)] [State Before (6)] [Action (1)] [State After (6)]`.

**Transition Noninvertibility and Externalized Reversibility:** The transition function `T(s, a)` is **not invertible** (not a group). For canonical actions, transitions have multiple predecessors (gate-clear destroys information). Maximum observed predecessor multiplicity is 8 for tested actions. 

This noninvertibility is a **constitutive property** of the discrete BU-Ingress realization, but it does not make the system lossy at the trajectory level. The system is **lossless at the trajectory level** when the action sequence (ledger) is treated as the missing information:

- Given `(start_state, action_sequence)`, the entire trajectory is reconstructible by deterministic replay (even though each individual step is lossy).
- Given only `final_state`, reconstruction is generally impossible.

This is **externalized reversibility** or **lossless replay via lossy state**: the state is a compact "present" summary, while the ledger externalizes the "past" information needed for reconstruction. The ledger is constitutive for BU-Ingress analogue, not optional. This operational justification explains why a "holography-like" design (compressed bulk state + boundary record) makes sense: the bulk state alone cannot recover history, but the boundary record (action stream) enables full reconstruction.

### 4.6 Moments Accumulation
A **Moment** is a unit of alignment accumulation derived from the kernel state. For a discrete sequence of interaction steps indexed by `t` with duration `Δt`, Moments are accumulated deterministically:

`Moments = Σ_t (SI(t) / 100) * Δt`

Moments serve as a convertible accounting unit for alignment work across domains. They accumulate over defined cycles (Atomic, Day, Domain Cycle, Year) and can be mapped by application layers into domain-specific units (e.g., currency, credits).

---

## 5. Verification and Conformance

A conformant implementation of the GGG ASI Router kernel must satisfy the following requirements.

### 5.1 Conformance Requirements
- **R1 (Ontology Closure):** The set of states must be closed under the transition function. `epistemology` values must strictly map into `[0, N)`.
- **R2 (S-Sector Semantics):** The `s_mask` must be computed from BU-Egress balance (depth-4 alternation commutation) and must be non-empty.
- **R3 (Defect Consistency):** The stored `defect_phi8.npy` values must match the result of re-computing the integer φ₈ monodromy defect: `d = popcount((s_i XOR s_φ₈(i)) & s_mask)` exactly (no tolerance needed, as defects are integers).
- **R4 (Aperture Quantization):** All aperture values must be quantized in steps of `1/H`, and all nonzero defects must be multiples of the defect quantum `g`.
- **R5 (A*_router Consistency):** The router canonical aperture `A*_router = g/H` must equal the minimal nonzero aperture in the distribution, where `g` is the GCD of all nonzero defect counts.
- **R6 (Determinism):** Given the same Atlas and start state, the same input byte sequence must produce the exact same Routing Signature.
- **R7 (Parity Preservation):** The φ₈ loop must preserve S-sector parity for all states. That is, `popcount(s & s_mask) mod 2 = popcount(φ₈(s) & s_mask) mod 2` for all states `s` in the ontology.

### 5.2 Measured Invariants
The reference implementation guarantees:
- Ontology size `N = 788,986`.
- Graph diameter = 6.
- Epistemology size `788,986 x 256`.
- S-sector horizon size `H = 40` bits (83.3% of state space) - derived from transition law algebra, not measured empirically.
- Defect quantum `g = 2` (all defects are even integers).
- Router canonical aperture `A*_router = g/H = 0.05`.
- Minimal nonzero aperture in distribution = `0.05`.
- **Parity preservation**: φ₈ preserves S-sector parity for all states (verified by test `test_phi8_preserves_s_sector_parity`).

---

## 6. Interpretation

While the kernel is a formal system, its utility derives from its correspondence to governance principles.

- **Ontology (CS)**: Defines the boundaries of valid existence.
- **Epistemology (UNA)**: Defines lawful motion and transformation.
- **S-Sector (BU-Egress)**: Defines the horizon bits where balance holds universally.
- **Stage Profile (ONA)**: Measures distinctions and differentiation (computed on-demand).
- **Loop Defects (BU-Egress)**: Measures closure and structural integrity (computed on-demand).
- **Aperture (BU-Ingress)**: Measures the normalized φ₈ toroidal holonomy defect, quantifying how "open" the state is to the canonical loop.

### 6.1 Architectural Justification

The Router kernel is **minimal and physics-first**:

1. **Single measurement**: The kernel measures one observable—the normalized φ₈ toroidal monodromy defect on the S-sector. This is directly derived from CGM physics, not a design artifact.

2. **Discovered invariants**: The architecture reveals two hard invariants:
   - **H = 40**: The S-sector horizon size (83.3% of state space) is **forced by the transition law algebra**—specifically, the interaction between gate-clear asymmetry (`asymmetric_clear = 0x060606060606`) and XFORM asymmetry (`asymmetric_xform = 0x000fff000fff`) across FG/BG layer pairs. This is a **derived invariant**, not an empirical measurement.
   - **g = 2**: The defect quantum arises from φ₈ parity preservation, a structural property of the loop.

3. **Canonical target**: The router's canonical aperture `A*_router = g/H = 0.05` is the minimal nonzero aperture that the ontology actually admits. This is not chosen but discovered.

4. **No design choices**: The architecture avoids arbitrary choices:
   - No graph topology (removed Hodge decomposition)
   - No weights or projections (removed stage weights and torus mode)
   - No stored artifacts beyond necessity (stage profile and loop defects computed on-demand)

The parity preservation property (Section 3.3.1) provides a clean, publishable explanation for the defect quantum, demonstrating that `g=2` is a consequence of the physics, not an implementation detail.

**Sectoral Depth-2 Cancellation:** For the BU-Egress alternation pair `(UNA_P, ONA_P)`, depth-2 order differs globally (`(LR ≠ RL)` off-horizon) but **commutes on the S-sector** (`((LR ⊕ RL) & S_MASK) = 0`). This is the discrete analogue of CGM's "sectoral commutator vanishing" (`P_S[X,Y]P_S ≈ 0`), where non-commutativity exists globally but cancels in the S-sector. This property is verified at the archetype. The depth-2 S-sector commutator defect `d2_s(i) = popcount((T(T(s,a),b) XOR T(T(s,b),a)) & s_mask)` is computed as an observable in the routing signature. Whether `d2_s=0` holds universally (global invariant) or only at specific states (archetype-local property) remains to be determined by computing the distribution over all states.

By operating the Router, a network maintains a continuous, verifiable measurement of its own structural alignment, enabling the emergence of ASI as a stable, coordinated equilibrium state.

### 6.2 Derived Invariants (This Realization)

The following properties are **derived or measured** for this atlas realization:

1. **Action Topology:** Active bits are positions 1-6 (`ACTION_ACTIVE_BITS_MASK = 0x7E`); anchor bits are positions 0,7 (`ACTION_ANCHOR_BITS_MASK = 0x81`). This is independent of `GENE_Mic_S = 0xAA`, which is an interface gauge constant. Both `0xAA` and `0x55` are maximally alternating balanced bytes; choosing `0xAA` fixes a convention (orientation) rather than a unique property.

2. **BU-Egress Alternation Pair:** `(0x02, 0x04)` is designated because it satisfies depth-4 balance on S-sector. This is a measured property of this realization.

3. **Sectoral Depth-2 Cancellation:** For `(UNA_P, ONA_P)`, non-commutativity exists globally but cancels on S-sector. This is verified at archetype. Universality across all states requires computing the `d2_s` distribution (see Section 6.4).

4. **Noninvertibility:** The transition monoid is not invertible (not a group). Maximum observed predecessor multiplicity for canonical actions is 8. This quantifies the discrete BU-Ingress requirement for ledger-based reconstruction.

5. **S-Mask Derivation:** The derivation rule in `derive_excluded_bits_for_alternation()` is specialized to the BU-Egress pair `(0x02, 0x04)`. For arbitrary alternation pairs, the rule may need adjustment.

### 6.3 Bridges Not Yet Derived

The following correspondences are **not yet proven** and should not be claimed as established:

1. **Defect-to-Angle Mapping:** The discrete φ₈ defect fraction `A = d/H` is a monodromy observable, but its relation to the continuous CGM aperture formula `A* = 1 - δ_BU/m_a` (where `δ_BU ≈ 0.1953 rad` and `A* ≈ 0.0207`) is not derived. The router's minimal nonzero aperture `A*_router = 0.05` is larger than the continuous `A* ≈ 0.0207`, and the mapping between discrete defect counts and continuous angles remains an open question.

2. **Modal Operator Identification:** The full identification of CGM's two primitive operators `[L]` and `[R]` as action families or distinguished generators in the 256-action semigroup is not yet derived. The alternation pair `(UNA_P, ONA_P)` realizes BU-Egress balance, but a complete `[L]/[R]` operator-family identification remains open.

These gaps are explicitly stated to prevent over-claiming and to guide future theoretical work.

### 6.4 Remaining Ambiguities and Tests to Add

The following computational tests would resolve remaining ambiguities:

1. **φ₈ vs δ_BU loop correspondence:** Compute `defect_delta.npy` for `WORD_DELTA_BU` and compare defect spectra and correlation with `defect_phi8` to determine if φ₈ is the correct discrete analogue of CGM monodromy.

2. **Sectoral depth-2 cancellation universality:** Compute the distribution of `d2_s` over all states (or large sample) for the BU-Egress pair to determine if sectoral cancellation is a global invariant or archetype-local property.

3. **Modal operator [L]/[R] identification:** Classify actions by their `(XFORM_MASK, ACTION_BROADCAST_MASK)` signature restricted to active bits, then identify which equivalence classes participate in BU-Egress alternation and exhibit horizon-preserving vs horizon-altering behavior.

4. **GENE_Mic_S gauge verification:** Test that changing `GENE_Mic_S` only relabels external bytes and does not change ontology, epistemology, s_mask, or defect distributions when interpreting the same internal action stream.

5. **6 actions = 6 DoF verification:** Test whether the three pairs `(UNA_P, UNA_M)`, `(ONA_P, ONA_M)`, `(BU_P, BU_M)` behave as three independent "axes" (distinct defect signatures, stage profile effects, commutation patterns) or if the "3 rot + 3 trans" interpretation is premature.

6. **Noninvertibility certificate verification:** Rebuild manifest and confirm `max_predecessors_observed` matches test output (should be 8 for UNA_P and ONA_P).


References

- [GyroSI Core Physics Specification](/docs/GyroSI_Specs.md)
- [Common Governance Model (CGM) Foundations](/docs/CommonGovernanceModel.md)
- [Framework: Gyroscopic Global Governance (GGG) Sociotechnical Sandbox](https://github.com/gyrogovernance/tools?tab=readme-ov-file#ggg)
- THM documentation in the GyroGovernance repositories
