# Gyroscopic ASI Runtime Specs
## Introduction

The **Gyroscopic runtime** is the multicellular AI runtime of the Gyroscopic ASI architecture. It is a **holonomic cellular automaton**: parallel finite-dimensional cells, discrete-time evolution under a shared byte rule, coupled by resonance rather than a fixed spatial grid. It realizes quantum cellular automaton structure through holonomic computation on a discrete GF(2) substrate executed on standard silicon (Zanardi and Rasetti 1999; Pachos et al. 2000). It composes the hQVM router and SDK primitives into an executable multicellular system. It provides structural observability and optimization surfaces for external actuators and AI runtimes.

The hQVM routes bytes on Ω; the Gyroscopic runtime is the multicellular execution layer built on that substrate. Where HQC literature realizes gates through adiabatic or non-adiabatic control loops on quantum hardware, the hQVM instantiates the same geometric structure as a GF(2) finite-state machine on silicon, opening the possibility of structural quantum advantage without quantum hardware.

QuBEC Theory maps to three implementation layers: the Python `src` semantic surface (normative exact semantics), the Gyroscopic SDK native backend (kernel-exact stepping, q-map, WHT, signature scans), and runtime rolling climate memories (`chi_hist64`, `shell_hist7`, `family_hist4`) with SLCP spectral output.

The Gyroscopic runtime operates in two valid modes. In **native mode**, cells evolve directly on Ω and interact by resonance, forming a standalone multicellular computational system. In **interoperability mode**, the runtime attaches to an external architecture, providing structural analysis, exact and hybrid operator routing, runtime scheduling guidance, and cache management support. Both modes use the same cell model, the same rolling climate memory, and the same spectral surfaces. The mode is a deployment choice, not an architectural distinction.

**Versioning note:** This specification is a living document. Changes to enum encodings, normalization contracts, or projection classes must include migration notes and conformance test updates.

---

## Glossary

| Term | Definition |
|------|------------|
| **Ω** | Reachable manifold of 4096 kernel states |
| **family_ring64** | Rolling buffer of recent family-phase values derived from ingested bytes |
| **family_hist4** | Distribution over the 4 family values in the current rolling window |
| **QuBEC** | Occupied computational object on Ω |
| **hQVM Kernel** | Exact byte rule governing state transitions on Ω |
| **Cell** | Single computational unit in the runtime cell pool, occupying one Ω point |
| **4-byte word** | Native input unit: four kernel bytes (b₀, b₁, b₂, b₃), each 0..255 |
| **omega12** | Packed Ω coordinate: u6 in bits 11..6, v6 in bits 5..0 |
| **chi6** | Chirality value: u6 ⊕ v6 |
| **Shell** | popcount(chi6), values 0..6 |
| **omega_sig** | Compiled Ω signature of a 4-byte word |
| **Resonance** | Co-occupation relation over a kernel-native observable |
| **SLCP** | Spectral Light-Cone Parametrization: the structured output record |
| **Gauge spectral** | 4-mode phase summary derived from family_hist4 via K4 character projection |
| **QuBEC climate** | Derived summary of occupation, shell balance, and support concentration from local cell memory |
| **Bridge** | Deployment-specific binding that maps runtime events into 4-byte words |
| **BU Egress** | Outward structural movement: applying the input word to the cell |
| **BU Ingress** | Structured return: emitting the SLCP report to an external actuator |
| **Projection energy ratio** | Fraction of operator energy in an exact quotient class |
| **Activation block** | A 64-wide hidden-state slice of an external model |
| **Weight block** | A 64-wide operator slice of an external model layer |
| **KV block** | A 64-wide cached key or value slice |

---

## Part I: Concepts

### 1. Introduction

#### 1.1 Scope

The Gyroscopic runtime transforms runtime traces into exact structural reports over the QuBEC medium:

```
runtime events → bridge serializer → 4-byte words → multicellular evolution on Ω
    → SLCP records + graph queries + interoperability outputs
```

In native mode, external actuators consume these reports and make runtime decisions. In interoperability mode, external runtimes additionally receive operator routing recommendations, exact block substitutions, and hybrid execution results.

#### 1.2 Classification

| The runtime is | The runtime is not |
|---|---|
| A multicellular holonomic AI model over Ω | A new kernel dynamics layer |
| An orchestration layer over the hQVM Kernel | A gradient-trained neural network |
| A resonance-defined graph without learned weights | A fixed-adjacency graph engine |
| A structural observability surface | A semantic parser of runtime events |
| An exact and hybrid optimization layer for external architectures | A replacement requiring external systems to adopt byte formalism internally |

#### 1.3 First coverage domains

| Domain | Description |
|--------|-------------|
| Applications | Program execution optimization (Python first) |
| Databases | Search, retrieval, indexing, query execution regularity |
| Networks | LLM serving, KV-cache pressure, batching, dispatch regularity |
| Transformers | Operator analysis, exact block substitution, hybrid lowering, KV management |

Any runtime that can map events into 4-byte words can be attached through a bridge. The transformer domain is treated as a first-class integration target given its strategic significance and the direct applicability of the QuBEC transform algebra to transformer weight and activation structure.

---

### 2. Position in the Stack

The Gyroscopic runtime sits above the hQVM router and SDK layers.

#### 2.1 Inherited kernel surfaces

- Byte transcription and mask expansion rules
- Spinorial transition rule
- Ω manifold structure
- Chirality register and K4 gate structure
- Shell algebra
- Walsh-Hadamard spectral surface
- Replay and Moment surfaces

#### 2.2 Inherited native surfaces

- Exact kernel stepping and extraction
- WHT64, Krawtchouk7, K4Char4 native transforms
- Structured operator analysis and quotient classification
- 64-wide lowering and hybrid application
- Packed arithmetic execution

The Gyroscopic runtime does not reimplement any of these. It calls SDK native operations directly for all stepping, transform, analysis, and lowering operations.

#### 2.3 QuBEC relation

A cell occupies one point on Ω at a time and therefore one local state of the QuBEC medium. The rolling local memories of each cell are empirical summaries of the one-cell climate marginals defined in QuBEC Theory Parts II and VI.

#### 2.4 BU ordering

Applying the input word (BU Egress) always precedes emitting the SLCP report (BU Ingress).

---

### 3. Hardware-Tier Architecture

The Gyroscopic runtime's architecture is the CPU architecture. Its parameters are Registry, Cache, RAM, and disk.

#### 3.1 The hierarchy

| Tier | Hardware | Runtime realization | CGM phase |
|------|----------|------------------------|----------|
| Register | 32-bit CPU register | omega12, state24, last_byte: the current transition atom per cell | CS |
| L1 Cache | 64-byte cache line, 6-bit offset | chi_ring64, family_ring64: 64-element rolling buffers; 6-bit chi6 keys | UNA |
| L2 Cache | 64-bit interaction, depth-two composition | word4, omega_sig, parity: closure-boundary context | ONA |
| L3 / RAM | Shared working memory | Cell pool (up to 4096 cells), resonance buckets, active cell set, operator block registry | BU Egress |
| RAM / Disk | Persistence, reconstruction | Ingest log, snapshot, crystallized trajectory (.gyrg) | BU Ingress |

Data flow is inward on ingress (disk → RAM → L3 → L2 → L1 → registers) and outward on egress.

#### 3.2 Why this orders the design

**64-element rings.** The chirality and family rings are 64 entries because the kernel's 6-bit payload space and the hardware cache-line offset are both 64. Local structural variety lives in the L1-aligned working set.

**Resonance buckets in RAM.** Bucket membership and weight are shared across cells and available for graph queries and batch grouping.

**Operator block registry in RAM.** Structure analysis reports for external operator blocks are cached at the L3/RAM tier for bridge scheduling. Native matmul uses its own weight registry with mandatory per-block analysis at tensor registration.

**Ingest log on disk.** The append-only (cell_id, word4) ledger enables deterministic replay.

---

### 4. Multicellular Holonomic AI Model

#### 4.1 Cellular automaton pattern

Every cell uses the same kernel rule. Cells carry no private learned weights, dense latent vectors, or fixed semantic types. Specialization arises solely from the words applied, the resulting trajectory on Ω, rolling local structural memory, and resonance participation.

A cell's occupied state defines a reception geometry. When a 4-byte frame arrives, the cell gyrates through the transition rule. The resulting shift in condensation is the cell's exact inferential response.

#### 4.2 Depth-4 temporality

The hQVM Kernel embeds temporal structure in every byte transition: Prefix (CS), Present (UNA), Past (ONA), and Future (BU). Cells always evolve inside this four-part temporal frame.

The byte is the phase atom of the kinematic law. A single byte executes one phase of the four-part transition cycle. The 4-byte word is the closed action: it completes one full CS, UNA, ONA, BU cycle, resolving all family phases modulo K4 and committing a single state transition with a compiled signature. The word is the native external integration grain.

#### 4.3 Indirect cell agency

No single cell is an autonomous decision-maker. Cell agency is indirect to the organism. The runtime is a coordination network, not a collection of autonomous agents.

#### 4.4 Light-cone structure

From any Ω state, one byte reaches exactly 128 next states with uniform 2:1 multiplicity. Two bytes reach all 4096 states with uniform 16:1 multiplicity. These are exact integer counts.

The two-step uniformization establishes the causal reach of the computational geometry. From any state, 2 bytes produce exact uniform occupancy over all 4096 states of Ω. Beyond depth 2, all states are equally reachable.

---

### 5. Operating Modes

#### 5.1 Native mode

In native mode, the Gyroscopic runtime operates as a standalone multicellular computational system. Cells evolve on Ω under the byte transition rule. The graph topology is defined by resonance over kernel-native observables. SLCP records are emitted to external actuators, which consume structural observables and make runtime decisions.

Native mode is the reference operating context for all cell mechanics, resonance, and SLCP emission defined in Parts II and III.

#### 5.2 Interoperability mode

In interoperability mode, the Gyroscopic runtime attaches to an external architecture. The cell model, SLCP emission, and resonance surfaces run as in native mode. Operator blocks additionally route through structured decomposition.

Two execution classes apply to weight matmul:

**Exact substitution.** When a 64-wide block has projection energy ratio 1.0 in an exact quotient class, `D_Q(W)` is zero and `W · x = P_Q(W) · x`.

**Hybrid exact-residual.** For blocks with nonzero defect:

```
W · x = P_Q(W) · x + D_Q(W) · x
```

The structured component uses native transforms. The defect uses the ternary residual surface. Both terms are required for correctness.

Structural summaries (decode monitoring, KV pressure, batch grouping) come from the cell model and SLCP records. They are independent of the matmul execution class.

#### 5.3 Routing contract

`describe_operator_route` selects exact substitution when `block_defect_norm == 0`, otherwise hybrid exact-residual. An operator report is required.

For llama.cpp matmul, the weight registry requires a completed registry entry for every executed weight block. Missing or unanalyzed blocks fail under strict policy.

---

### 6. Resonance and Graph Structure

#### 6.1 Core concept

Resonance is a relation of co-occupation over a kernel-native observable. Cells that share the same value of a chosen observable are co-resonant. No pairwise adjacency matrix is stored. Graph topology is dynamic, determined at runtime by the observable values cells occupy.

#### 6.2 Adaptation mechanism

Cells keep rolling local memories and participate in resonance profiles. Graph structure forms through repeated co-occupation under the active resonance profile. Bucket weight is the simplest measure of shared pattern strength.

---

## Part II: Specification

### 7. State Model

The state model is organized by the hardware-tier hierarchy.

#### 7.1 Primary state

The primary state of each cell is its packed Ω coordinate:

```
omega12 : int32
```

containing u6 in bits 11..6 and v6 in bits 5..0.

#### 7.2 Per-cell stored state

| Group | Field | Type | Description |
|-------|-------|------|-------------|
| Core | omega12 | int32 | Current Ω coordinate |
| Core | step | uint64 | Total bytes consumed |
| Core | last_byte | uint8 | Most recent byte |
| Word | word4[4] | uint8 | Most recent closed 4-byte word |
| Word | has_closed_word | bool | Whether at least one word has closed |
| Chirality memory | chi_ring64[64] | uint8 | Rolling buffer of last 64 chi6 values |
| Chirality memory | chi_ring_pos | uint8 | Current write position in ring |
| Chirality memory | chi_valid_len | uint8 | Valid entries in ring (0..64) |
| Distributions | chi_hist64[64] | uint16 | Histogram over 64 chi6 values in ring |
| Distributions | shell_hist7[7] | uint16 | Histogram over shells 0..6 in ring |
| Family memory | family_ring64[64] | uint8 | Rolling buffer of last 64 family values |
| Family memory | family_hist4[4] | uint16 | Histogram over the 4 family values in the ring |
| Compiled action | omega_sig | int32 | Ω signature of most recent closed word |
| Parity | parity_O12 | uint16 | Odd parity commitment |
| Parity | parity_E12 | uint16 | Even parity commitment |
| Parity | parity_bit | uint8 | Parity bit |
| Resonance | resonance_key | uint32 | Current key under active profile |

#### 7.3 Derived observables

Computed on demand:

| Observable | Source |
|------------|--------|
| u6, v6 | omega12 bit extraction |
| chi6 | u6 ⊕ v6 |
| shell | popcount(chi6) |
| state24 | omega12_to_state24 |
| horizon_distance | Kernel observable |
| ab_distance | Kernel observable |
| family, micro_ref, q6 | last_byte decomposition |
| charts | sdk.state_charts |
| future-cone measures | sdk.future_cone_measure |
| future shell measures | sdk.future_locus_measure |
| optical coordinates | sdk.optical_coordinates |
| stabilizer type | sdk.stabilizer_type |

---

### 8. Local Structural Memory

Local structural memory is the cache-tier realization of the runtime. The 64-element rings and histograms align to the L1 cache line.

#### 8.1 Chirality ring

Each cell maintains chi_ring64[64], a rolling buffer of the last 64 chi6 observations.

#### 8.2 Chirality histogram

chi_hist64[64] is the distribution over the 64 elements of GF(2)⁶ in the ring. It is the empirical chirality marginal of the one-cell climate. It supports 64-point Walsh-Hadamard spectral analysis via the WHT64 surface and fast structural similarity comparison.

#### 8.3 Shell histogram

shell_hist7[7] is the distribution over shell values 0..6 induced by the chirality ring. It supports Krawtchouk spectral decomposition via the Krawtchouk7 surface and horizon tendency analysis.

#### 8.4 Family ring and family histogram

family_ring64[64] stores the last 64 family values. family_hist4[4] stores the distribution over the 4 family values. This memory supports gauge-sensitive views of recent transport via the K4Char4 surface.

#### 8.5 Constant-time update rule

**Warmup (chi_valid_len < 64):** increment chi_hist64[chi_new] and shell_hist7[popcount(chi_new)]; no decrements.

**Full ring (chi_valid_len == 64):**

```
chi_old = chi_ring64[pos]
chi_new = current chi6

chi_hist64[chi_old] -= 1
chi_hist64[chi_new] += 1
shell_hist7[popcount(chi_old)] -= 1
shell_hist7[popcount(chi_new)] += 1

chi_ring64[pos] = chi_new
pos = (pos + 1) & 63
```

```
family_old = family_ring64[pos]
family_new = current family

family_hist4[family_old] -= 1
family_hist4[family_new] += 1

family_ring64[pos] = family_new
```

Family memory is updated at the same ring position and with the same valid-length semantics. This update is O(1).

#### 8.6 Spectral surfaces

Three spectral surfaces are derived from local memory via native transforms:

| Surface | Source | Transform | Output |
|---------|--------|-----------|--------|
| Chirality spectral | chi_hist64 | WHT64 | spectral64[64] |
| Shell spectral | shell_hist7 | Krawtchouk7 | shell_spectral[7] |
| Gauge spectral | family_hist4 | K4Char4 | gauge_spectral[4] |

These describe different inherited geometries and remain distinct in the SLCP record and in interoperability outputs.

---

### 9. Native Compiled Action

Each ingested 4-byte word has a compiled Ω action obtained through omega_word_signature(word4), stored as omega_sig. The word4 is retained because it is the exact depth-4 slice from which the Ω signature, parity commitments, and exact local provenance are derived.

---

### 10. Resonance Profiles

#### 10.1 Available profiles

| Profile ID | Name | Observable | Buckets | Key computation |
|------------|------|------------|---------|-----------------|
| 0 | Chirality | chi6 | 64 | Current chi6 |
| 1 | Shell | shell | 7 | popcount(chi6), values 0..6 |

Invalid profile IDs are rejected at runtime.

#### 10.2 Reference profile

The reference profile is chirality (profile 0). It is inherited directly from the kernel, compact, and cheap to compute.

#### 10.3 Resonance decay

Bucket weight is always an integer derived from cell membership. decay_resonance_buckets() shifts bucket values right by 1 without changing cell membership. Long-running deployments should apply deterministic decay or renormalization on a recorded schedule.

**Snapshot restriction.** Snapshots must be taken only when resonance bucket values represent true membership counts. On restore, resonance buckets are recomputed from stored resonance_key values.

---

### 11. Cell Lifecycle

#### 11.1 Pool management

A runtime instance is a finite pool of cells G = {c₁, c₂, …, cₙ}. Required operations: allocate, seed, free, and query active cells.

#### 11.2 Seeding options

| Seed method | Description |
|-------------|-------------|
| seed_rest | Rest state (complement-horizon representative) |
| seed_equality_horizon | Equality-horizon representative |
| seed_shell | Chosen shell representative |
| seed_omega | Arbitrary Ω coordinate |

#### 11.3 Cell-to-entity mapping

A bridge may assign one entity to one cell, one entity to several cells, or several entities to one cell. This mapping is bridge policy.

---

### 12. Ingestion Protocol

#### 12.1 Native input unit

The native input is the 4-byte word:

```
w = (b₀, b₁, b₂, b₃)
```

Each byte is a full kernel byte (0..255), aligned with the depth-4 closure structure.

#### 12.2 External packets

At the orchestration boundary, runtime systems feed packets:

```
P = (cell_id, word4, bridge_metadata)
```

bridge_metadata is not kernel state. It may include request IDs, program IDs, actor IDs, wall-clock timestamps, or bridge-local routing hints.

#### 12.3 Ingestion rule

**Byte cadence** (for each byte bₖ in word4):

1. Step: omega12 = step_omega12_by_byte(omega12, bₖ)
2. Increment step
3. Update last_byte
4. Compute chi6
5. Update chi_ring64, chi_hist64, shell_hist7, family_ring64, family_hist4

**Word closure** (after the fourth byte):

6. Store word4, set has_closed_word
7. Compute and store omega_sig
8. Compute and store parity commitment
9. Compute closure-boundary resonance key
10. Update resonance bucket membership
11. Optionally append ingest log record

Resonance updates occur only at word closure.

#### 12.4 Cadence summary

| Cadence | Trigger | Updates |
|---------|---------|---------|
| Byte | Each byte in word4 | omega12, step, last_byte, chi_ring64, chi_hist64, shell_hist7, family_ring64, family_hist4 |
| Word closure | After 4th byte | word4, has_closed_word, omega_sig, parity, resonance key and bucket weight |
| Emission | Bridge-controlled | SLCP record emitted, graph queries served, interoperability outputs emitted |

---

### 13. SLCP Record

#### 13.1 Role

The Spectral Light-Cone Parametrization is the structured output record delivered to external actuators and interoperability consumers. Field names, types, and exactness classes below are the normative contract; language bindings may wrap them but must not alter semantics.

#### 13.2 Standard fields

| Field | Type | Exactness class |
|-------|------|-----------------|
| cell_id | int | Identifier |
| step | uint64 | Integer exact |
| omega12 | int32 | Integer exact |
| state24 | int32 | Integer exact |
| last_byte | uint8 | Integer exact |
| family | int | Integer exact |
| micro_ref | int | Integer exact |
| q6 | int | Integer exact |
| chi6 | int | Integer exact |
| shell | int | Integer exact |
| horizon_distance | int | Integer exact |
| ab_distance | int | Integer exact |
| omega_sig | int32 | Integer exact |
| parity_O12 | uint16 | Integer exact |
| parity_E12 | uint16 | Integer exact |
| parity_bit | uint8 | Integer exact |
| resonance_key | uint32 | Integer exact |
| current_resonance | int | Runtime exact (resonance bucket weight at emission) |
| spectral64 | float32[64] | Numerically faithful (WHT64 of chi_hist64, normalized) |

**Pre-closure default.** Before the first word closure, omega_sig and all parity fields are reported as 0.

#### 13.3 Optional views

An implementation may also expose:

- shell_spectral[7]: Krawtchouk7 of shell_hist7
- gauge_spectral[4]: K4Char4 of family_hist4
- QuBEC climate summary (occupation density, effective support M₂, spectral damping η)
- optical coordinates
- stabilizer type
- future-cone summaries
- interoperability outputs (Section 17)

---

### 14. Graph Query Surface

Conforming implementations expose the queries below. Method names in bindings may differ; semantics must match.

#### 14.1 Resonance queries (minimum required)

| Query | Returns |
|-------|---------|
| get_co_resonant_cells(cell_id) | Cells sharing the same resonance key |
| get_bucket_population(key) | Current bucket value for key |
| get_bucket_cells(key) | All cells in a resonance bucket |

#### 14.2 Relation queries

| Query | Returns |
|-------|---------|
| get_cells_on_shell(shell) | Cells at a given shell value |
| get_cells_with_chi6(chi6) | Cells with a given chirality value |
| get_cells_with_signature(omega_sig) | Cells with a given Ω signature |
| chirality_distance_between_cells(a, b) | Chirality distance between two cells |

---

### 15. Ledger History and Replay

#### 15.1 Two memory types

| Type | Contents | Purpose |
|------|----------|---------|
| Local rolling memory | word4, chi_ring64, chi_hist64, shell_hist7, omega_sig, parity | Live runtime structure |
| Replayable ledger | Append-only (cell_id, word4) records | Replay, verification, audit |

#### 15.2 Replay surfaces

Replay, verification, and comparison use existing SDK surfaces: moment_from_ledger, verify_moment, compare_ledgers.

---

### 16. Persistence Format

#### 16.1 State file

Single file, e.g. data/models/gyroscopic/runtime.state.bin.

#### 16.2 Header

| Field | Type | Description |
|-------|------|-------------|
| magic | 4 bytes | GYRG |
| version | uint32 | Format version |
| capacity | uint32 | Cell pool capacity |
| active_count | uint32 | Currently active cells |
| profile_id | uint16 | Active resonance profile |
| flags | uint16 | Bit 0 = ingest logging enabled |
| created_unix_ns | uint64 | Creation timestamp |
| kernel_rule_hash | 32 bytes | SHA-256 of kernel rule surfaces |

A conforming implementation must reject snapshots where the stored kernel rule hash does not match the current kernel rule hash.

#### 16.3 Body

Arrays in C-contiguous layout, in order: allocated, has_closed_word, omega12, step, last_byte, word4, chi_ring64, family_ring64, chi_ring_pos, chi_valid_len, chi_hist64, shell_hist7, family_hist4, omega_sig, parity_O12, parity_E12, parity_bit, resonance_key, resonance_buckets.

#### 16.4 Ingest log

Persisted separately if enabled. Each record: (uint32 cell_id, 4 bytes word4).

---

## Part III: Transformer Interoperability

---

### 17. Transformer Integration Model

The Gyroscopic runtime provides structural analysis and exact or hybrid operator routing for transformer architectures. The integration does not require the transformer to adopt the byte formalism internally. External weight tensors, activation tensors, and KV cache entries are consumed as 64-wide blocks through the native lowering surface. Results are returned in the format expected by the external runtime.

The integration preserves external semantics. In exact substitution mode, the native path produces identical results to the dense path. In hybrid exact-residual mode, the result equals the full dense result up to the precision of the dense backend.

#### 17.1 External block projection contract

External tensors are not kernel states on Ω. When the runtime derives chirality, shell, boundary-anchor, or climate summaries from external tensors, it does so through a bridge-defined projection map Pi.

A projection map Pi is a deterministic bridge contract from an external 64-wide block into one of the native QuBEC coordinate charts or summaries. Every interoperability pathway that operates on external tensors must declare which projection map it uses.

Two projection classes are used for operator analysis and runtime summaries.

**Pi_basis.** Canonical basis-preserving projection for operator analysis. Preserves the 64-wide basis ordering used by quotient tests and transform surfaces. Used for exact and hybrid routing of weight blocks.

**Pi_summary.** Structural summary projection. Maps an external block or runtime summary into chirality, shell, or resonance summaries without claiming the block is a kernel state. Used for request grouping, decode monitoring, and cache-structure diagnostics.

#### 17.2 Interoperability classes

| Class | Condition | Result |
|---|---|---|
| Exact substitution | `block_defect_norm == 0` | `W · x = P_Q(W) · x` |
| Hybrid exact-residual | `block_defect_norm > 0` | `W · x = P_Q(W) · x + D_Q(W) · x` |

KV polar encode and polar attention scoring are auxiliary APIs (Sections 19–21). They are separate from weight matmul semantics.

#### 17.3 Transformer block objects

| Object | Description | Mapping |
|--------|-------------|---------|
| Activation block | 64-wide hidden-state slice | One block per 64 hidden dimensions |
| Weight block | 64-wide operator slice | One block per 64 columns of a weight matrix |
| KV block | 64-wide cached key or value slice | One block per 64 dimensions of KV embeddings |
| Request cell | One cell per active request or request role | Tracks per-request climate and resonance |
| Layer block cell | One cell per (layer, block) pair | Tracks per-block structural history |
| KV segment cell | One cell per cache segment class | Tracks per-segment occupancy and priority |

#### 17.4 Integration workflow overview

```
External weight matrix W (shape: rows × d)
    │
    ▼
Tile into 64-wide blocks (tile64)
    │
    ▼
Analyze each block (analyze_operator)
    │   Reports: class, projection energy ratio, eigenvalues, defect_norm
    ▼
Cache reports in operator block registry (RAM tier)
    │
    ▼
For each inference step:
    Input activation x (width d)
    │
    ▼
    Apply per block (apply_hybrid64)
    │   Always evaluate W·x = P_Q(W)·x + D_Q(W)·x
    │   If D_Q = 0 by value, only the P_Q branch remains
    │   Projection energy ratio is used for profiling and scheduling only
    ▼
    Concatenate results
    │
    ▼
    Update cell climate (chi_hist64, shell_hist7, family_hist4)
    │
    ▼
    Emit interoperability outputs (Section 22)
```

---

### 18. Operator Analysis and Routing

#### 18.1 Layer analysis

Layer analysis uses the SDK `analyze_operator` primitive on 64-wide blocks under `Pi_basis`. Registry layout and caching are specified below; operator class definitions are in QuBEC Theory Part IV.

```
Input:  W         weight matrix, shape (rows, d)
        threshold real, default 0.01

Output: block_registry

1. pad d to next multiple of 64
2. for each block b in range(d // 64):
       W_block ← W[:, b·64 : (b+1)·64]
       report  ← analyze_operator(W_block, threshold)
       block_registry[b] ← (W_block, report)

return block_registry
```

Reports are cached in RAM and reused across inference steps without re-analysis unless the weight changes.

#### 18.2 Block application

```
Input:  block_registry
        x_block   activation slice, width 64

Output: y_block   result, width rows

report ← block_registry[b].report

y_structured ← apply_native(report.class, x_block, report.eigenvalues)
y_defect     ← dot(report.defect, x_block)   # ordinary dot product on D_Q remainder
y_block      ← y_structured + y_defect

return y_block
```

#### 18.3 Full layer application

```
Input:  block_registry
        x   activation vector, width d

Output: y   result vector, width rows

y ← zeros(rows)
for each block b in block_registry:
    x_block ← x[b·64 : (b+1)·64]
    y_block ← apply_block(block_registry, b, x_block)
    y       += y_block

return y
```

#### 18.4 Quality preservation contracts

**Exact substitution contract.** When `report.proj_energy_ratio == 1.0`, `D_Q(W)` is zero by value, so `y = P_Q(W)·x` is exactly the same semantic decomposition, with no defect term to evaluate.

**Preservation contract (all projection energy ratio values).** For any `report.proj_energy_ratio`, the result satisfies:

```
y_hybrid = P_Q(W) · x + D_Q(W) · x = W · x
```

up to the declared numeric precision of the backend used for defect evaluation. projection energy ratio does not gate correctness.

**Low-structure contract.** When `report.proj_energy_ratio` is near zero, almost all cost sits in `D_Q`, but semantics remain `P_Q + D_Q`.

---

### 19. KV Cache Management

Auxiliary KV surfaces for experiments. Not part of the weight matmul path.

#### 19.1 KV block encoding

Each KV embedding of width d may be encoded into a polar summary per 64-wide block via `kv_polar_encode_block64`.

```
Input:  kv_vector   embedding, width d

Output: encoded_blocks

1. pad d to next multiple of 64
2. for each block b:
       kv_block ← kv_vector[b·64 : (b+1)·64]
       c    ← boundary_anchor(kv_block)    (6 bits)
       chi  ← chirality_word(kv_block)     (6 bits)
       N    ← popcount(chi)                (3 bits, shell 0..6)
       r    ← l2_norm(kv_block)            (float16, 16 bits)
       encoded_blocks[b] ← (c, chi, N, r)

return encoded_blocks
```

Storage per block: 31 bits total (6 + 6 + 3 + 16).
Uncompressed storage: 64 · 16 = 1024 bits per block.
Structural compression ratio: approximately 33×.

This encoding is a derived structural summary of external embeddings. It is not a universal or lossless representation theorem for arbitrary transformer embeddings.

#### 19.2 KV block decoding

```
Input:  encoded_blocks, b   block index
        decode_polar        bridge-defined reconstruction map

Output: kv_block_approx

(c, chi, N, r) ← encoded_blocks[b]
kv_block_approx ← decode_polar(c, chi, N, r)

return kv_block_approx
```

The reconstruction map is bridge-defined and approximate unless a semantics-preserving inverse is explicitly available for the chosen projection class.

#### 19.3 KV resonance tracking

Each KV segment is assigned to one KV segment cell. As KV entries are written and evicted, the cell ingests corresponding 4-byte words derived from the segment identifier and operation type.

The resulting chi6 and shell values characterize the structural occupancy of the KV segment. The resonance bucket gives the co-occupancy weight relative to other segments.

#### 19.4 KV eviction priority

```
Input:  kv_cell   KV segment cell SLCP record

Output: eviction_priority   real

priority ← weighted_combination(
    weight_resonance  * (1 − current_resonance / max_resonance),
    weight_shell      * |shell − 3| / 3,
    weight_horizon    * (1 − horizon_distance / 12),
    weight_spectral   * spectral_concentration(spectral64)
)

return priority
```

Higher priority indicates a candidate for eviction. The weights are bridge-level policy parameters. The inputs are exact kernel observables or deterministic spectral summaries.

---

### 20. Decode Batch Grouping

#### 20.1 Grouping principle

Requests whose cells share resonance keys or close chirality values produce structurally similar activation patterns. Grouping such requests into the same decode batch enables shared KV cache access and reduces memory bandwidth.

#### 20.2 Grouping algorithm

```
Input:  active_request_cells   list of (cell_id, SLCP record)
        max_batch_size         int

Output: batches   list of request-id lists

1. Sort cells by resonance_key (primary) and chi6 (secondary)
2. batch ← []
   batches ← []
   for each cell in sorted order:
       if len(batch) == 0:
           batch.append(cell)
       elif resonance_key(cell) == resonance_key(batch[0])
            or chirality_distance(cell, batch[0]) <= 2:
           batch.append(cell)
       else:
           batches.append(batch)
           batch ← [cell]
       if len(batch) == max_batch_size:
           batches.append(batch)
           batch ← []
   if len(batch) > 0:
       batches.append(batch)

return batches
```

Chirality distance between two cells is the Hamming distance between their chi6 values, computed via the chirality_distance surface.

#### 20.3 Batch resonance update

After each decode step, cells update their chi6 and resonance key. The grouping algorithm is re-run at bridge-controlled intervals. Cells that drift in chirality may join different batches in subsequent steps.

---

### 21. Attention Approximation

Polar attention scoring uses encoded KV summaries without reconstructing full embeddings. It is an auxiliary prefilter surface, not a substitute for exact matmul.

#### 21.1 Native attention score

```
Input:  q_encoded    projected query summary from Pi_polar
        k_encoded    projected key summary from Pi_polar

Output: score   real

chi_dist        ← popcount(chi_q ⊕ chi_k)
shell_sim       ← (6 − chi_dist) / 6
anchor_align    ← 1 − popcount(c_q ⊕ c_k) / 64
score           ← r_q * r_k * shell_sim * anchor_align

return score
```

This score uses popcount operations and one multiply. It is suitable for candidate scoring and speculative filtering. Final attention uses the full embedding path or semantics-preserving hybrid matmul.

#### 21.2 Approximate attention pass

```
Input:  q_encoded    encoded query blocks, one per 64-wide block
        K_encoded    encoded key blocks for all KV entries
        V_encoded    encoded value blocks for all KV entries
        d            embedding dimension

Output: output   approximate attention output

for each block b:
    scores_b ← [native_attention_score(q_encoded[b], k_encoded[b])
                 for k in K_encoded]
    weights_b ← softmax(scores_b / sqrt(d))
    output_b  ← weighted_sum(decode_kv_block(V_encoded, b), weights_b)

output ← concatenate(output_b for all b)
return output
```

#### 21.3 Accuracy characteristics

The polar attention score is a structural heuristic. Accuracy depends on embedding regularity. Final attention computation uses full embeddings or semantics-preserving hybrid matmul.

---

### 22. Interoperability Outputs

At each emission point, the runtime exposes the following interoperability outputs in addition to the standard SLCP record.

Outputs derived from exact kernel state and exact lowering are exact or semantics-preserving under the contracts of Sections 17 and 18. Outputs derived from Pi_summary are structural summaries. Polar KV outputs are auxiliary summaries.

| Output | Type | Description |
|--------|------|-------------|
| block_class | string | Quotient class of the most recently analyzed operator block |
| block_proj_energy_ratio | float | Projection energy ratio of the most recently analyzed block |
| block_eigenvalues | array | Eigenvalues of the structured component |
| block_defect_norm | float | Frobenius norm of the defect component |
| native_route | bool | Whether the native path was used for this block |
| kv_priority | float | Eviction priority for KV segment cells |
| batch_group_id | int | Assigned decode batch group |
| chi_anisotropy | float[6] | Per-axis damping estimates from chi_hist64 |
| gauge_anisotropy | float[2] | Per-axis gauge damping from family_hist4 |
| effective_support | float | Rényi-2 effective support M₂ estimated from chi_hist64 |
| spectral_damping | float | Isotropic spectral damping η estimated from shell_hist7 |

These outputs are consumed by bridge actuators, external runtime schedulers, and monitoring surfaces.

---

## Part IV: Bridge Architecture

---

### 23. Bridge Contract

Each bridge defines:

| Element | Description |
|---------|-------------|
| Source runtime | The system producing events |
| Serializer contract | How events become 4-byte words |
| Cell allocation policy | How entities map to cells |
| Active resonance profile | Which profile the bridge uses |
| Operating mode | Native or interoperability (exact substitution / hybrid per block) |
| Consumed SLCP fields | Which output fields the actuator reads |
| Consumed interoperability outputs | Which optimization outputs the actuator uses |
| Actuator decision surface | What decisions the actuator makes |

#### 23.1 Substitutional Upgrade Principle

A conforming bridge that operates in exact substitution or hybrid mode identifies exact or partially exact structural components of the external computation and routes those components through native QuBEC charts when doing so preserves the target semantics or preserves the full result through structured-plus-residual decomposition. The bridge does not replace the external model's semantic interfaces. It upgrades the execution of specific operator blocks within those interfaces.

---

### 24. Bridge Implementation Status

| Bridge | Status |
|--------|--------|
| Applications | Implemented |
| Databases | Reserved |
| Networks | Reserved |
| Transformers | Partially implemented (model-control bridge, decode surfaces) |

---

### 25. First Bridge Coverage

#### 25.1 Applications Bridge

**Scope.** Runtime execution traces. Python is the first binding.

**Serializer.** Fixed categorical vocabulary mapping event types to predetermined 4-byte words.

**Cell allocation.** One (entity_id, role) pair maps to one cell.

**Resonance profile.** Chirality (reference profile).

**Actuator scoring.** Two heuristic scores derived from exact structural inputs:

| Score | Inputs | Indicates |
|-------|--------|-----------|
| hot_loop_score | chi_support_ratio, shell_entropy, spectral_peak_ratio | Concentrated, repetitive occupation |
| contention_score | chi_support_ratio, shell_entropy, spectral_peak_ratio | Dispersed, irregular occupation |

profile_entity produces an ApplicationDecision with suggested action: specialize_hot_loop, mitigate_contention, or observe.

#### 25.2 Databases Bridge

**Scope.** Query planning, indexing, traversal, and cache-structure regularity.

**Status.** Reserved.

#### 25.3 Networks Bridge

**Scope.** Inference serving, request grouping, KV-cache pressure, and queue regularity.

**Status.** Reserved.

#### 25.4 Transformer Bridge

**Scope.** Transformer weight analysis, exact and hybrid block substitution, KV polar surfaces, decode batch grouping.

**Serializer.** The transformer bridge has two distinct interoperability channels.

**Runtime-event channel.** Maps transformer runtime events (layer invocation, token generation, KV write, KV evict) into deterministic 4-byte words. Drives request cells, KV segment cells, and resonance updates.

**Operator-analysis channel.** Consumes Q1_0 weight blocks and Q8_0 activation blocks as 64-wide tiles. Performs registry analysis, quotient classification, and exact or hybrid matmul.

**Cell allocation.**

| Entity | Cell type |
|--------|-----------|
| Active request | Request cell |
| (Layer, block) pair | Layer block cell |
| KV segment class | KV segment cell |

**Resonance profile.** Chirality (profile 0) for request grouping.

**Operator routing.** `describe_operator_route` selects exact substitution or hybrid exact-residual from `block_defect_norm`.

**Actuator decision surface.**

| Decision | Source |
|----------|--------|
| Operator routing | block_defect_norm, block_class |
| KV eviction priority | kv_priority, effective_support, current_resonance |
| Decode batch grouping | resonance_key, chi6, chirality_distance |
| KV prefilter (optional) | polar-encoded KV blocks, polar attention scores |
| Layer scheduling | shell spectral profile, spectral damping η |

---

## Part V: Conformance

---

### 26. Conformance Requirements

A conforming implementation:

1. Stores cell state primarily as omega12
2. Consumes input as packets whose native kernel content is a 4-byte word
3. Updates local rolling memories at byte cadence
4. Maintains both chirality and shell histories
5. Uses a declared resonance profile over a kernel-native observable
6. Emits SLCP records and exposes the graph query surface to external actuators
7. Uses SDK surfaces for stepping, extraction, replay, and spectral transforms
8. Keeps bridge policy outside the core machine
9. Rejects restored snapshots whose kernel rule hash does not match the current kernel rule hash
10. In interoperability mode, applies the exact substitution and hybrid contracts of Section 18.4
11. Declares external-tensor pathways with projection class Pi_basis or Pi_summary where applicable
12. Reports operating mode and consumed SLCP and interoperability output fields in the bridge contract declaration

The operational loop in all cases is:

```
packet input → Ω stepping → local memory update → resonance update
    → SLCP and graph queries → interoperability outputs
```

---

## Appendices

### Appendix A: Theoretical Classification

The Gyroscopic runtime is a **holonomic cellular automaton** that realizes quantum cellular automaton structure on a finite GF(2) substrate executed on standard silicon. This classification arises from the intersection of three model families:

| Model family | Key properties | Runtime realization |
|--------------|----------------|---------------------|
| Holonomic cellular automaton (realizes QCA on discrete GF(2)) | Identical finite-dimensional cells, time-independent rule, coupling by resonance, reversibility | Each cell has 4096 reachable states; all evolve under the same byte rule |
| Exact finite-field substrate | Exact integer arithmetic, spectral transforms for global property extraction | Chirality register is GF(2)⁶; WHT is exact; hidden subgroup resolution in one step |
| Group automaton | Transition bijections forming a finite group | Each byte is a bijection on Ω with period 4; two bytes reach all of Ω with exact uniform multiplicity |

Coupling differs from lattice QCA: cells interact by **resonance** over kernel-native observables, not a fixed spatial neighborhood. External AI systems attach through **bridges** without adopting the byte formalism internally.

#### Comparison with adjacent architectures

| Architecture | Key difference from this runtime |
|--------------|----------------------------------|
| Neural networks | Zero learned per-cell weights; specialization from trajectory and resonance |
| Classical cellular automata | Cells connect by resonance over algebraic observables, not fixed spatial grids |
| Graph neural networks | Edges are exact co-occupation relations; no learned edge functions |
| Reservoir computing | Exact structured dynamics with analytical spectral surfaces |

---

### Appendix B: Bridge Scenario Catalog

#### B.1 Applications

**A1: Hot-loop specialization**

| Step | Detail |
|------|--------|
| Runtime event | A Python code region repeatedly executes a stable loop |
| Runtime response | Repeated omega_sig; concentrated chi_hist64; elevated current_resonance |
| Actuator reads | omega_sig, spectral64, current_resonance |
| Actuator decision | Raise specialization priority for that code region |

**A2: Lock-contention detection**

| Step | Detail |
|------|--------|
| Runtime event | A thread alternates through wait, lock, wake, retry |
| Runtime response | Rapidly varying chi6; broad shell occupancy; unstable spectral64 |
| Actuator reads | chi6, shell, spectral64, co-resonant cell graph queries |
| Actuator decision | Adjust backoff or contention handling policy |

#### B.2 Transformer

**T1: Exact block substitution**

| Step | Detail |
|------|--------|
| Runtime event | Layer weight analysis at model load |
| Runtime response | Block registry populated; structure reports and defect mass identified |
| Actuator reads | block_proj_energy_ratio, block_class, block_eigenvalues |
| Actuator decision | Schedule native and defect costs by profile; preserve P_Q + D_Q semantics |

**T2: KV cache pressure management**

| Step | Detail |
|------|--------|
| Runtime event | KV cache approaches capacity |
| Runtime response | KV segment cells diverge in spectral64 and effective_support |
| Actuator reads | kv_priority, current_resonance, effective_support |
| Actuator decision | Evict segments with highest priority score |

**T3: Decode batch grouping**

| Step | Detail |
|------|--------|
| Runtime event | Multiple active requests await decode |
| Runtime response | Request cells carry live chirality and spectral profiles |
| Actuator reads | resonance_key, chi6, co-resonant cell queries |
| Actuator decision | Group requests with chirality distance ≤ 2 into shared decode batch |

**T4: Hybrid operator routing**

| Step | Detail |
|------|--------|
| Runtime event | Inference step on a layer with mixed-structure weight blocks |
| Runtime response | Block registry reports vary by projection energy ratio across blocks |
| Actuator reads | block_proj_energy_ratio, native_route, block_defect_norm |
| Actuator decision | Apply decomposition per block and sum native structured plus defect evaluation |

#### B.3 Networks

**N1: Decode batch grouping by serving cell**

| Step | Detail |
|------|--------|
| Runtime event | Many active requests await decode grouping |
| Runtime response | Each request cell carries a live chirality and spectral profile |
| Actuator reads | chi6, spectral64, get_co_resonant_cells results |
| Actuator decision | Group structurally close requests into the same decode batch |

**N2: KV-cache residency management**

| Step | Detail |
|------|--------|
| Runtime event | KV-cache pressure rises |
| Runtime response | Cells diverge in spectral64, horizon_distance, current_resonance |
| Actuator reads | current_resonance, horizon_distance, spectral64 |
| Actuator decision | Decide which request states remain resident |

---

### Appendix C: Implementation Notes

#### C.1 Implementation anchor

The reference stack lives under `src/tools/gyroscopic/` (Python semantic surface, ctypes bindings, native kernel) and `external/llama.cpp/ggml/src/ggml-gyroscopic/` (llama.cpp hook). Multicellular runtime bindings implement §11–§14: cell pool lifecycle, 4-byte ingestion, SLCP emission (§13), and graph queries (§14). Export paths, constructor options, and optional accelerator backends are implementation details, not part of this specification.

#### C.2 Native boundary

| Component | Responsibility |
|-------|----------------|
| Native execution | Kernel algebra, WHT64, Krawtchouk7, K4Char4, structured analysis, hybrid lowering |
| kernel.c | Native kernel stepping, K4 holonomy operators (wavefunction chart), gravity-scale metadata |
| ggml-gyroscopic | llama.cpp per-group gravity-scale hook |
| ops.py | ctypes bindings, native build automation |