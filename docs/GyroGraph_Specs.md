# GyroGraph
## Specification

GyroGraph is the **Multicellular Quantum AI model** of the Gyroscopic ASI architecture. Operating as an **Algebraic Quantum Cellular Automaton (aQCA)**, it composes the existing aQPU Kernel, SDK, and GyroLabe primitives into an executable model that provides structural observability to external actuators. It does not introduce new kernel physics.

This document is the normative specification. GyroGraph's architecture is the CPU architecture: Registry, Cache, RAM, and disk (Section 3). The specification defines the core machine, state model, ingestion protocol, local memories, resonance profiles, output surfaces, persistence, and first bridge coverage, all organized by that hardware-tier hierarchy.

Normative terms MUST, SHOULD, SHOULD NOT, and MAY are interpreted as requirement keywords for conformance.

---

## Glossary

| Term | Definition |
|------|------------|
| **Ω** | Reachable manifold of 4096 kernel states |
| **family_ring64** | Rolling buffer of recent family-phase values derived from ingested bytes |
| **family_hist4** | Distribution over the 4 family values in the current rolling window |
| **QuBEC** | Occupied computational object on Ω |
| **aQPU Kernel** | Exact byte law governing state transitions on Ω |
| **Cell** | Single computational unit in a GyroGraph pool, occupying one Ω point |
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
| **GyroLabe** | Native execution backend for kernel-level algebra and SDK surfaces |
| **BU Egress** | Outward structural movement: applying the input word to the cell |
| **BU Ingress** | Structured return: emitting the SLCP report to an external actuator |

---

## Part I: Concepts

### 1. Introduction

#### 1.1 Scope

GyroGraph transforms runtime traces into exact structural reports over the QuBEC medium:

```text
runtime events → bridge serializer → 4-byte words → multicellular evolution on Ω → SLCP records + graph queries
```

External actuators consume these reports and make runtime decisions. GyroGraph emits structural information; it does not make decisions.

#### 1.2 Classification

| GyroGraph is | GyroGraph is not |
|---|---|
| A Multicellular Quantum AI model over the algebraic quantum space (Ω) | A new kernel physics layer |
| An orchestration layer over the aQPU Kernel | A gradient-trained neural network |
| A resonance-defined graph without learned weights | A fixed-adjacency graph engine |
| A structural observability surface for external actuators | A semantic parser of runtime events |

#### 1.3 First coverage domains

| Domain | Description |
|--------|-------------|
| Applications | Program execution optimization (Python first) |
| Databases | Search, retrieval, indexing, query execution regularity |
| Networks | LLM serving, KV-cache pressure, batching, dispatch regularity |

Any runtime that can map events into 4-byte words can be attached through a bridge. These three are the first bindings, not the architectural limit.
The repository also includes a validated byte-native model-control bridge used as a direct stress test of the architecture. That bridge does not redefine the domain taxonomy above; it demonstrates that the same multicellular substrate can attach directly to AI decision surfaces.

---

### 2. Position in the Stack

GyroGraph sits above the existing aQPU Kernel, SDK, and GyroLabe layers.

#### 2.1 Inherited kernel surfaces

- Byte transcription and mask expansion rules
- Spinorial transition law
- Ω manifold structure
- Chirality register and K4 gate structure
- Shell algebra
- Walsh-Hadamard spectral surface
- Replay and Moment surfaces

#### 2.2 QuBEC relation

A cell occupies one point on Ω at a time and therefore one local state of the QuBEC medium.

#### 2.3 BU ordering

Applying the input word (BU Egress) always precedes emitting the SLCP report (BU Ingress).

---

### 3. Hardware-Tier Architecture

GyroGraph's architecture is the CPU architecture. Its parameters are Registry, Cache, RAM, and disk. There is no separate "model" to load; the multicellular substrate lives in the same memory hierarchy that executes it. This mapping is not cosmetic. It determines the size, placement, and purpose of every component.

#### 3.1 The hierarchy

| Tier | Hardware | GyroGraph realization | CGM phase |
|------|----------|------------------------|----------|
| Register | 32-bit CPU register | omega12, state24, last_byte: the current transition atom per cell | CS |
| L1 Cache | 64-byte cache line, 6-bit offset | chi_ring64, family_ring64: 64-element rolling buffers; 6-bit chi6 keys | UNA |
| L2 Cache | 64-bit interaction, depth-two composition | word4, omega_sig, parity: closure-boundary context | ONA |
| L3 / RAM | Shared working memory | Cell pool, resonance buckets, active cell set | BU Egress |
| RAM / Disk | Persistence, reconstruction | Ingest log, snapshot, crystallized trajectory (.gyrg) | BU Ingress |

Data flow is inward on ingress (disk -> RAM -> L3 -> L2 -> L1 -> registers) and outward on egress. GyroGraph ingests words into the register tier, updates cache-tier rolling memories, commits to RAM-tier resonance surfaces, and persists to disk when replay or indexing is required.

#### 3.2 Why this orders the design

- **64-element rings:** The chirality and family rings are 64 entries because the kernel's 6-bit payload space and hardware cache-line offset are both 64. Local structural variety lives in L1-aligned working set.

- **Resonance buckets in RAM:** Bucket membership and weight are shared across cells. They belong to L3/RAM: the tier where internal computation becomes auditable, visible to graph queries, and available for batch grouping.

- **Ingest log on disk:** The append-only (cell_id, word4) ledger enables deterministic replay. Disk is the persistence tier. Reconstruction flows inward from there.

- **No learned weights:** Embeddings and structural state are integrated within the cache layers. The 64-byte cache line, 6-bit chi6 space, and 4096-state Omega manifold are the geometry. GyroGraph does not approximate this geometry; it occupies it.

#### 3.3 Relation to kernel G.7

The aQPU Kernel specification (Gyroscopic_ASI_Specs Appendix G.7) maps the kernel law to Register, L1 Cache, Working Memory, and Persistent Storage. GyroGraph extends that mapping to the multicellular layer: each cell's register atom, each cell's cache-tier rolling memory, the shared RAM-tier resonance surface, and the disk-tier ingest log and snapshot. The correspondence is exact.

---

### 4. Multicellular Quantum AI Model

#### 4.1 Cellular automaton pattern

Every cell uses the same kernel law. Cells carry no private learned weights, dense latent vectors, or fixed semantic types. Specialization arises solely from the words applied, the resulting trajectory on Ω, rolling local structural memory, and resonance participation.

This follows a cellular automaton pattern: identical local rules, differentiation from history and context.

Because cells lack learned parameters, a cell's intelligence is strictly a function of its current occupation state (its condensation) and its rolling structural memory. The cell does not compute a probability distribution over incoming words. Its occupied state defines a reception geometry. When a 4-byte frame arrives, the cell gyrates through the transition law. The resulting shift in condensation—whether the cell absorbs the frame coherently or scatters toward thermalization—is the cell's exact inferential response.

#### 4.2 Depth-4 temporality

The aQPU Kernel embeds intrinsic temporality in every byte transition: Prefix (CS), Present (UNA), Past (ONA), and Future (BU). Cells always evolve inside this four-part temporal frame, which is why GyroGraph is a runtime intelligence layer rather than an offline state machine.

Within this temporal frame, the byte and the word serve distinct roles. The byte is the phase atom of the kinematic law. It executes one phase of the four-part transition cycle. A single byte has no closed inferential meaning; it is a kinematic phase, not a semantic event.

The 4-byte word is the closed action. It completes one full CS, UNA, ONA, BU cycle, resolving all family phases modulo K4 and committing a single state transition with a compiled signature (omega_sig). The word is the native external integration grain: the unit at which resonance updates, SLCP emission, and compiled action surfaces operate.

The byte-log is not a log of decisions. It is phase-resolved provenance. The useful computational surfaces (signatures, parity commitments, resonance keys, spectral records, climate observables) are compiled closures and exact quotients computed at word boundaries.

#### 4.3 Indirect cell agency

No single cell is an autonomous decision-maker. Cell agency is indirect to the organism. GyroGraph is a coordination network, not a collection of autonomous agents.

#### 4.4 Light-cone structure

From any Ω state, one byte reaches exactly 128 next states with uniform 2:1 multiplicity. Two bytes reach all 4096 states with uniform 16:1 multiplicity. These are exact integer counts with zero variance. Longer words preserve this uniform occupancy.

The rolling memories and spectral surfaces maintained by each cell are local projections of this global light-cone geometry. An SLCP record is therefore a precise coordinatization of where a cell sits inside the inherited light-cone, not an arbitrary feature vector.

The two-step uniformization establishes the causal reach of the computational geometry. From any state, 2 bytes produce exact uniform occupancy over all 4096 states of Omega. This is the discrete analogue of a light cone: the boundary within which the full manifold becomes causally accessible. The causal structure is nontrivial only within the first 2 byte steps, where the 128-state immediate future (with its 2:1 multiplicity cover) constrains which states are reachable. Beyond depth 2, all states are equally reachable.

The speed of light cancels exactly in the kernel's physical capacity derivation (CSM), leaving a purely geometric-frequency invariant. The kernel therefore operates in a computational geometry compatible with natural unit systems where c = 1. Time in this geometry is intrinsic ledger depth, not an external clock variable.

---

### 5. Resonance and Graph Structure

#### 5.1 Core concept

Resonance is a relation of co-occupation over a kernel-native observable. Cells that share the same value of a chosen observable are co-resonant. No pairwise adjacency matrix is stored. Graph topology is dynamic, determined at runtime by the observable values cells occupy.

#### 5.2 Adaptation mechanism

Cells keep rolling local memories and participate in resonance profiles. Graph structure forms through repeated co-occupation under the active resonance profile: cells that repeatedly share the same resonance key remain grouped in the same structural bucket. Bucket weight is the simplest measure of shared pattern strength (see 9.4).

Because the underlying observables are exact, there is a clear separation between structural and adaptive quantities. Adaptation focuses only on which structural configurations recur in a particular deployment.

---

## Part II: Specification

### 6. State Model

The state model is organized by the hardware-tier hierarchy (Section 3). Primary state belongs to the register tier; per-cell stored state spans cache and RAM tiers.

#### 6.1 Primary state

The primary state of each cell is its packed Ω coordinate, `omega12 : int32`. This is the register atom: the current transition context (24-bit Mac + 8-bit byte) per cell. This contains u6 in bits 11..6 and v6 in bits 5..0. The `int32` type matches the native GyroLabe batch interfaces for zero-copy compatibility.

The repository already exposes `pack_omega12`, `unpack_omega12`, `step_omega12_by_byte`, `omega12_to_state24`, `state24_to_omega12`, and native batch versions of the same.

#### 6.2 Per-cell stored state

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

This is the complete live hot-path state, including chirality memory, shell memory, and family-phase memory.

`current_resonance` is not stored per cell. It is derived from the global resonance bucket weight for the cell's key at emission time (see 9.4).

#### 6.3 Derived observables

These are computed on demand, not stored:

| Observable | Source |
|------------|--------|
| u6, v6 | omega12 bit extraction |
| chi6 | u6 ⊕ v6 |
| shell | popcount(chi6) |
| state24 | omega12_to_state24 |
| horizon_distance | Kernel observable (constants.horizon_distance) |
| ab_distance | Kernel observable (constants.ab_distance) |
| family, micro_ref, q6 | last_byte decomposition |
| charts | sdk.state_charts |
| future-cone measures | sdk.future_cone_measure |
| future shell measures | sdk.future_locus_measure |
| optical coordinates | sdk.optical_coordinates |
| stabilizer type | sdk.stabilizer_type |

---

### 7. Local Structural Memory

Local structural memory is the cache-tier realization of GyroGraph (Section 3.1). The 64-element rings and histograms align to the L1 cache line; the word4 and omega_sig closure context align to L2.

#### 7.1 Chirality ring

Each cell maintains `chi_ring64[64]`, a rolling buffer of the last 64 chi6 observations, each the exact 6-bit value u6 ⊕ v6 derived from omega12.

#### 7.2 Chirality histogram

`chi_hist64[64]` is the distribution over the 64 elements of GF(2)⁶ present in the ring. It supports 64-point Walsh-Hadamard spectral analysis and fast structural similarity comparison.

#### 7.3 Shell histogram

`shell_hist7[7]` is the distribution over shell values 0..6 induced by the chirality ring. It supports Krawtchouk spectral decomposition, shell regularity analysis, and horizon tendency analysis.

#### 7.4 Family ring and family histogram

Each cell also maintains a rolling family memory aligned to the same byte cadence as the chirality ring.

- `family_ring64[64]` stores the last 64 family values derived from the ingested bytes.
- `family_hist4[4]` stores the distribution over the 4 family values present in the ring.

This memory supports gauge-sensitive views of recent transport and bridge-level climate summaries derived from family-phase occupancy.

#### 7.5 Constant-time update rule

During warmup, while `chi_valid_len < 64`, the old value is not removed and histograms are only incremented; the decrement step begins only after the ring becomes full.

When a new chirality value enters the ring:

Family memory is updated at the same ring position and with the same valid-length semantics. Each ingested byte contributes both a `chi6` value and a `family` value, so chirality and family memories remain aligned across the rolling window.

**Warmup (valid_len < 64):** increment chi_hist64[chi_new] and shell_hist7[popcount(chi_new)]; no decrements.

**Full ring (valid_len == 64):**

```text
chi_old = chi_ring64[pos]
chi_new = current chi6

chi_hist64[chi_old] -= 1
chi_hist64[chi_new] += 1
shell_hist7[popcount(chi_old)] -= 1
shell_hist7[popcount(chi_new)] += 1

chi_ring64[pos] = chi_new
pos = (pos + 1) & 63
```

```text
family_old = family_ring64[pos]
family_new = current family

family_hist4[family_old] -= 1
family_hist4[family_new] += 1

family_ring64[pos] = family_new
```

This update is O(1).

#### 7.6 Spectral surfaces

Two distinct spectral surfaces are derived from local memory:

| Surface | Source | Transform | Output |
|---------|--------|-----------|--------|
| Chirality spectral | chi_hist64 | wht64 | spectral64 |
| Shell spectral | shell_hist7 | shell_krawtchouk_transform_exact | Exact shell spectral coefficients |
| Gauge spectral | family_hist4 | K4 character projection | 4 gauge-sector coefficients |

These describe different inherited geometries and remain distinct. Implementations MAY expose only the chirality spectral surface in the core SLCP record while keeping shell and gauge spectral surfaces available as derived views.

---

### 8. Native Compiled Action

Each ingested 4-byte word has a compiled Ω action obtained through `omega_word_signature(word4)`, stored as `omega_sig : int32`.

`word4` is also retained because it is the exact depth-4 slice from which the Ω signature, parity commitments, replay fragments, and exact local provenance are derived. Both `word4` and `omega_sig` are meaningful and available.
For a fixed closed 4-byte word, `parity_bit` is 0 by construction because the word length is even; it remains part of the stored commitment format for consistency with the broader trajectory parity contract.

---

### 9. Resonance Profiles

Resonance buckets live in the L3/RAM tier (Section 3.1): shared, visible across cells, and queriable for graph operations. Bucket membership is the structural commitment that makes the multicellular graph auditable.

#### 9.1 Available profiles

| Profile | Observable | Buckets | Key computation |
|---------|------------|---------|-----------------|
| Chirality | chi6 | 64 | Current chi6 |
| Shell | shell | 7 | Current shell |
| Horizon class | 3-way partition: equality (0), complement (1), bulk (2) | 3 | From chi6: 0 if chi6 == 0, 1 if chi6 == 0x3F, else 2 |
| Ω coincidence | omega12 | 4096 | Current omega12 |
| Signature | omega_sig | 8192 | Current omega_sig |
| Q-transport | q_word6 of most recent closed word | 64 | q_word6_for_items(word4) |

#### 9.2 Reference profile

The reference profile is **chirality resonance** (chi6 ∈ {0..63}, 64 buckets). It is the reference because it is inherited directly from the kernel, compact, cross-domain, cheap to compute, naturally shared across cells, and already supported by all code surfaces. Under the reference profile, two cells are adjacent if and only if they share the same chi6.

#### 9.3 Profile runtime state

Under the active profile:

- `resonance_key` is stored per cell
- `current_resonance` is derived at emission time, not stored per cell (see 6.2)

Alternative profiles MAY be selected by a bridge. If so, the profile identifier SHOULD be recorded in persisted state metadata, and the graph query surface SHOULD report which profile is active.

#### 9.4 Resonance decay

Bucket weight is always an integer derived from cell membership, equal to the exact membership count when no decay has been applied.

In the implementation, `_resonance_buckets[k]` is initially the membership count. `decay_resonance_buckets()` shifts bucket values right by 1 without changing cell membership. After decay, a bucket value is a decayed weight, not the current membership count.

Long-running deployments SHOULD apply deterministic decay or renormalization to resonance buckets. The schedule SHOULD be recorded in state metadata.

**Snapshot restriction:** Snapshots MUST be taken only when resonance bucket values represent true membership counts (no decay applied since last normalization). On restore, the implementation recomputes resonance buckets from the stored `resonance_key` values. Decayed bucket weights are therefore not preserved across restore and should not be relied on as persistent state.

---

### 10. Cell Lifecycle

#### 10.1 Pool management

A GyroGraph is a finite pool of cells: G = {c₁, c₂, …, cₙ}. Required operations: allocate, seed, free, and query active cells.

#### 10.2 Seeding options

| Seed method | Description |
|-------------|-------------|
| seed_rest | Rest state (complement-horizon representative) |
| seed_equality_horizon | Equality-horizon representative |
| seed_shell | Chosen shell representative |
| seed_omega | Arbitrary Ω coordinate |

A separate `seed_complement_horizon` is not exposed because rest is already a complement-horizon state. The SDK witness synthesis surface may be used when a replayable seed certificate is needed.

#### 10.3 Freeing

Freeing returns a cell to the inactive pool, clears its local memories, and removes its contribution from the resonance surface.

#### 10.4 Cell-to-entity mapping

A bridge may assign one entity to one cell, one entity to several cells, or several entities to one cell. This mapping is bridge policy. The bridge decides what a cell represents.

---

### 11. Ingestion Protocol

#### 11.1 Native input unit

The native input is the 4-byte word:

```text
w = (b₀, b₁, b₂, b₃)
```

Each byte is a full kernel byte (0..255). This aligns with the depth-4 closure structure already exposed by the repository (`depth4_mask_projection48`, `depth4_intron_sequence32`, `depth4_frame`).

GyroGraph consumes input as words, not as isolated bytes and not as classical field packets.

#### 11.2 External packets

At the orchestration boundary, runtime systems feed packets:

```text
P = (cell_id, word4, bridge_metadata)
```

`bridge_metadata` is not kernel state. It MAY include request IDs, program IDs, query IDs, actor IDs, wall-clock timestamps, or bridge-local routing hints.

#### 11.3 Event-to-word mapping

External runtime events do not define the model. A bridge MAY map one event to zero, one, or many words. The bridge owns that policy.

#### 11.4 Ingestion rule

Two distinct cadences govern the ingestion cycle:

**Byte cadence** (for each byte bₖ in word4):

1. Step: `omega12 = step_omega12_by_byte(omega12, bₖ)`
2. Increment step
3. Update last_byte
4. Compute chi6
5. Update chi_ring64, chi_hist64, shell_hist7, family_ring64, and family_hist4

**Word closure** (after the fourth byte):

6. Store word4, set has_closed_word
7. Compute and store omega_sig
8. Compute and store parity commitment
9. Compute closure-boundary resonance key
10. Update resonance bucket membership
11. Optionally append ingest log record

Resonance updates occur only at word closure because the resonance key may depend on the final omega12, word4, and omega_sig.

#### 11.5 Observation and emission

Cells are naturally observed at closure boundaries, after a full 4-byte word has been consumed. Internal evolution is byte-cadence; external reporting is word-boundary.

SLCP records are not automatically emitted after every word. Emission cadence belongs to the bridge or orchestrator. The core model maintains exact structural state continuously; bridges choose when to poll.

#### 11.6 Cadence summary

| Cadence | Trigger | Updates |
|---------|---------|---------|
| Byte | Each byte in word4 | omega12, step, last_byte, chi_ring64, chi_hist64, shell_hist7, family_ring64, family_hist4 |
| Word closure | After 4th byte | word4, has_closed_word, omega_sig, parity, resonance key and bucket weight |
| Emission | Bridge-controlled | SLCP record emitted, graph queries served |

---

### 12. SLCP Record

#### 12.1 Role

The Spectral Light-Cone Parametrization is the structured output record delivered to external actuators.

#### 12.2 Standard fields

Exactness categories:

- **Kernel-exact:** derived solely from exact kernel state and byte law (no approximations)
- **GyroGraph-exact:** exact integers from resonance surface and local memory
- **Deterministic numeric:** deterministic floating-point output (e.g. WHT)

| Field | Type | Exactness class |
|-------|------|-----------------|
| cell_id | int | Identifier |
| step | uint64 | Kernel-exact |
| omega12 | int32 | Kernel-exact |
| state24 | int32 | Kernel-exact |
| last_byte | uint8 | Kernel-exact |
| family | int | Kernel-exact |
| micro_ref | int | Kernel-exact |
| q6 | int | Kernel-exact |
| chi6 | int | Kernel-exact |
| shell | int | Kernel-exact |
| horizon_distance | int | Kernel-exact |
| ab_distance | int | Kernel-exact |
| omega_sig | int32 | Kernel-exact |
| parity_O12 | uint16 | Kernel-exact |
| parity_E12 | uint16 | Kernel-exact |
| parity_bit | uint8 | Kernel-exact |
| resonance_key | uint32 | Kernel-exact |
| current_resonance | int | GyroGraph-exact (resonance bucket weight at emission) |
| spectral64 | array[64] | Deterministic numeric (WHT of chi_hist64) |

**Pre-closure default:** Before the first word closure (`has_closed_word` is false), `omega_sig` and all parity fields MUST be reported as 0.

#### 12.3 Optional views

An implementation MAY also expose:

- Shell spectral coefficients
- Gauge spectral coefficients
- QuBEC climate summaries derived from chirality, shell, and family histories
- Optical coordinates
- Stabilizer type
- Future-cone summaries

 

These are derived views, not part of the minimum required record.

---

### 13. Graph Query Surface

#### 13.1 Resonance queries (minimum required)

| Query | Returns |
|-------|---------|
| get_co_resonant_cells(cell_id) | Cells sharing the same resonance key |
| get_bucket_population(key) | Current bucket value for key (membership count before decay, decayed weight after; see 9.4) |
| get_bucket_cells(key) | All cells in a resonance bucket |

#### 13.2 Relation queries (SHOULD expose)

| Query | Returns |
|-------|---------|
| get_cells_on_shell(shell) | Cells at a given shell value |
| get_cells_with_chi6(chi6) | Cells with a given chirality value |
| get_cells_with_signature(omega_sig) | Cells with a given Ω signature |
| chirality_distance_between_cells(a, b) | Chirality distance between two cells |

---

### 14. Ledger History and Replay

Ledger and ingest log belong to the RAM/disk tier (Section 3.1). The ingest log on disk is the persistence record that enables deterministic replay; reconstruction flows inward from there.

#### 14.1 Two memory types

| Type | Contents | Purpose |
|------|----------|---------|
| Local rolling memory | word4, chi_ring64, chi_hist64, shell_hist7, omega_sig, parity fields | Live runtime structure |
| Replayable ledger | Append-only (cell_id, word4) records | Replay, verification, audit |

#### 14.2 Ingest log

If replayability is required, the orchestrator records a global ingest log. A per-cell ledger is reconstructed by filtering by cell_id and concatenating word4 byte sequences.

#### 14.3 Replay surfaces

Replay, verification, and comparison use existing SDK surfaces: `moment_from_ledger`, `verify_moment`, `compare_ledgers`. No parallel replay subsystem is introduced.

---

### 15. Persistence Format

#### 15.1 State file

Single file, e.g. `data/models/gyrograph/gyrograph.state.bin`.

#### 15.2 Header (fixed struct)

| Field | Type | Description |
|-------|------|-------------|
| magic | 4 bytes | `GYRG` |
| version | uint32 | Format version |
| capacity | uint32 | Cell pool capacity |
| active_count | uint32 | Currently active cells |
| profile_id | uint16 | Active resonance profile |
| flags | uint16 | Bit 0 = ingest logging enabled |
| created_unix_ns | uint64 | Creation timestamp |
| kernel_law_hash | 32 bytes | SHA-256 of kernel law surfaces (see below) |

The kernel law hash is computed over the contents of: `src/constants.py`, `src/api.py`, and native law-carrying sources (e.g. `gyrolabe_codec.c`, `gyrolabe_mul.c`, `gyrolabe_opencl.c` if present). It does not include `gyrograph.c` or `gyrograph_opencl.c`.

A conforming implementation MUST reject snapshots where the stored kernel law hash does not match the current kernel law hash.

#### 15.3 Body

Arrays written in C-contiguous layout, in order:

| Array | Type | Shape |
|-------|------|-------|
| allocated | bool | capacity |
| has_closed_word | bool | capacity |
| omega12 | int32 | capacity |
| step | uint64 | capacity |
| last_byte | uint8 | capacity |
| word4 | uint8 | capacity × 4 |
| chi_ring64 | uint8 | capacity × 64 |
| family_ring64 | uint8 | capacity × 64 |
| chi_ring_pos | uint8 | capacity |
| chi_valid_len | uint8 | capacity |
| chi_hist64 | uint16 | capacity × 64 |
| shell_hist7 | uint16 | capacity × 7 |
| family_hist4 | uint16 | capacity × 4 |
| omega_sig | int32 | capacity |
| parity_O12 | uint16 | capacity |
| parity_E12 | uint16 | capacity |
| parity_bit | uint8 | capacity |
| resonance_key | uint32 | capacity |
| resonance_buckets | uint64 | bucket_count |

**Resonance bucket constraint:** Snapshots MUST be taken only when resonance bucket values represent true membership counts (see 9.4).

#### 15.4 Ingest log

Persisted separately if enabled. Each record: `(uint32 cell_id, 4 bytes word4)`.

---

## Part III: Deployment

### 16. Bridge Architecture

#### 16.1 Bridge contract

Each bridge defines:

| Element | Description |
|---------|-------------|
| Source runtime | The system producing events |
| Serializer contract | How events become 4-byte words |
| Cell allocation policy | How entities map to cells |
| Active resonance profile | Which profile the bridge uses |
| Consumed SLCP fields | Which output fields the actuator reads |
| Actuator decision surface | What decisions the actuator makes |
| Concrete scenarios | 1–3 illustrative use cases |

#### 16.2 Implementation status

| Bridge | Status |
|--------|--------|
| Applications | Implemented: event vocabulary, entity/role mapping, SLCP emission, actuator scoring |
| Databases | Reserved: module present, implementation pending |
| Networks | Reserved: module present, implementation pending |
| Model-control | Implemented: byte-native decode bridge, climate helpers, structural control surfaces |

#### 16.3 Substitutional Upgrade Principle

A conforming bridge MUST NOT treat GyroGraph as a passive observer of an external model's expensive computations. The scope of a model-control bridge is to upgrade the target architecture without breaking its semantic interfaces.

When wiring GyroGraph to a classical transformer, the bridge intercepts dense matrix multiplications and continuous activation functions at the L1/L2 cache boundaries. It routes those state updates through the exact discrete operations of the aQPU: Walsh-Hadamard spectral transforms, pointwise damping multipliers, and Plancherel condensation measures. The classical model proposes state updates; the GyroGraph cell applies its kinematic reception geometry to those updates. The exact structural response of the cell replaces the probabilistic guessing mechanism of the classical model.

---

### 17. First Bridge Coverage

#### 17.1 Applications Bridge

For detailed step-by-step narratives of the following scenarios, see Appendix B.

**Scope:** Runtime execution traces. Python is the first binding.

**Serializer:** Fixed categorical vocabulary mapping event types to predetermined 4-byte words. Similar event types share q-class structure. Unknown events map to identity word `0xAA 0xAA 0xAA 0xAA`.

**Cell allocation:** One (entity_id, role) pair maps to one cell, allocated on first reference. A bridge may assign one code region, one thread, one subsystem, or one program to one or more cells.

**Resonance profile:** Chirality (reference profile).

**Actuator scoring:** Two heuristic scores derived from exact structural inputs:

| Score | Inputs (weighted) | Indicates |
|-------|-------------------|-----------|
| hot_loop_score | (1 − chi_support_ratio), (1 − shell_entropy), spectral_peak_ratio | Concentrated, repetitive occupation |
| contention_score | chi_support_ratio, shell_entropy, (1 − spectral_peak_ratio) | Dispersed, irregular occupation |

`profile_entity` produces an `ApplicationDecision` with suggested action: `specialize_hot_loop`, `mitigate_contention`, or `observe`.

The structural inputs (chi_hist64, shell_hist7, spectral64, resonance bucket weight) are exact or deterministic. The scoring weights and action thresholds are bridge-level heuristics subject to revision.

#### 17.2 Databases Bridge

**Scope:** Query planning, indexing, traversal, and cache-structure regularity.

**Serializer:** Maps query-plan and storage events into deterministic 4-byte words.

**Cell allocation:** One query class, index path, query plan branch, or database workflow maps to one or more cells.

**Resonance profile:** Chirality, with optional signature resonance for specialized deployments.

**Status:** Reserved; implementation pending.

#### 17.3 Networks Bridge

**Scope:** Inference serving, request grouping, KV-cache pressure, and queue regularity.

**Serializer:** Maps serving events into deterministic 4-byte words.

**Cell allocation:** One request, batch fragment, or KV-cache segment class maps to one or more cells.

**Resonance profile:** Chirality, with optional signature or Q-transport resonance for specialized deployments.

**Status:** Reserved; implementation pending.

---

### 18. Conformance Requirements

A conforming implementation:

1. Stores cell state primarily as omega12
2. Consumes input as packets whose native kernel content is a 4-byte word
3. Updates local rolling memories at byte cadence
4. Maintains both chirality and shell histories
5. Uses a declared resonance profile over a kernel-native observable
6. Emits SLCP records and exposes the graph query surface to external actuators
7. Uses existing SDK and GyroLabe surfaces for stepping, extraction, replay, and spectral transforms
8. Keeps bridge policy outside the core machine
9. Treats the first bridge domains as coverage examples, not architectural limits
10. Rejects restored snapshots whose kernel law hash does not match the current kernel

The operational loop in all cases is (BU Egress first, BU Ingress second):

```text
packet input → Ω stepping → local memory update → resonance update → SLCP and graph queries
```

---

## Appendices

### Appendix A: Theoretical Classification

#### A.1 Computational model

GyroGraph is an Algebraic Quantum Cellular Automaton (aQCA) over GF(2). This classification arises from the intersection of three established model families:

| Model family | Key properties | GyroGraph realization |
|--------------|----------------|----------------------|
| Quantum Cellular Automaton | Identical finite-dimensional cells, time-independent law, locality, reversibility, local unitarity | Each cell has 4096 reachable states; all evolve under the same byte law; evolution is reversible per byte and per word; 24-bit carrier provides 2×3×2 lattice geometry |
| Algebraic Quantum Automaton over GF(2) | Exact finite field arithmetic, spectral transforms for global property extraction | Chirality register is GF(2)⁶; Walsh-Hadamard transforms are exact; hidden subgroup resolution in one spectral step gives O(1) answers where a classical DFA needs O(N) |
| Group Automaton | Transition bijections forming a finite group | Each byte is a bijection on Ω with period 4; two bytes reach all of Ω with exact uniform multiplicity |

"Algebraic" distinguishes this from QCA models using floating-point complex numbers. GyroGraph achieves quantum structure through exact integer arithmetic over finite fields on standard silicon.

A classical DFA classification would be inadequate: it ignores algorithmic efficiency and state geometry. The aQCA encodes discrete light-cone geometry, SE(3) Lie algebra structure, and SU(2) spinorial double-cover, none of which arise in standard DFA specifications.

#### A.2 Comparison with adjacent architectures

| Architecture | Key difference from GyroGraph |
|--------------|-------------------------------|
| Neural networks | GyroGraph has zero learned per-cell weights; specialization from trajectory and resonance, not gradient descent; most quantities NNs approximate are analytically available |
| Classical cellular automata | Classical CAs use fixed spatial grids; GyroGraph cells connect by resonance over algebraic observables, producing dynamic topology |
| Graph neural networks | GNN edges carry learned message-passing functions over continuous embeddings; GyroGraph edges are exact co-occupation relations with no adjacency matrix or learned edge function |
| Reservoir computing | Reservoirs use random fixed dynamics with a trained linear readout; GyroGraph uses exact structured dynamics with analytical spectral surfaces and exact SLCP output |

---

### Appendix B: Bridge Scenario Catalog

All scenarios follow the same structure:

| Step | Description |
|------|-------------|
| Runtime event | What happens in the source system |
| GyroGraph response | Observable structural changes in target cells |
| Actuator reads | SLCP fields and graph queries consumed |
| Actuator decision | External action taken |

The bridge serializer step (events → 4-byte words) is common to all scenarios and is omitted from individual entries.

#### B.1 Applications

**A1: Hot-loop specialization**

| Step | Detail |
|------|--------|
| Runtime event | A Python code region repeatedly executes a stable loop |
| GyroGraph response | Repeated omega_sig; concentrated chi_hist64 and shell_hist7; stable spectral64; elevated current_resonance |
| Actuator reads | omega_sig, spectral64, current_resonance, q6 |
| Actuator decision | Raise specialization or tier-up priority for that code region |

**A2: Lock-contention detection**

| Step | Detail |
|------|--------|
| Runtime event | A thread alternates through wait, lock, wake, retry |
| GyroGraph response | Rapidly varying chi6; unstable omega_sig; broad shell occupancy; unstable spectral64 |
| Actuator reads | chi6, shell, spectral64, co-resonant cell graph queries |
| Actuator decision | Adjust backoff, affinity, or contention handling policy |

#### B.2 Databases

**D1: Repeated index-seek path**

| Step | Detail |
|------|--------|
| Runtime event | Many similarly shaped index lookups |
| GyroGraph response | Repeated omega_sig; stable q6; concentrated spectral64; elevated current_resonance |
| Actuator reads | omega_sig, q6, current_resonance, spectral64 |
| Actuator decision | Keep seek path hot; favor corresponding index structures |

**D2: Scan/seek regime shift**

| Step | Detail |
|------|--------|
| Runtime event | Workload shifts from repeated seeks to broader scans |
| GyroGraph response | Drift in shell occupancy and spectral shape |
| Actuator reads | shell, spectral64, chi6, current_resonance |
| Actuator decision | Change scan strategy, cache policy, or prefetching mode |

#### B.3 Networks

**N1: Decode batch grouping**

| Step | Detail |
|------|--------|
| Runtime event | Many active requests await decode grouping |
| GyroGraph response | Each request cell carries a live chirality and spectral profile |
| Actuator reads | chi6, spectral64, get_co_resonant_cells results |
| Actuator decision | Group structurally close requests into the same decode batch |

**N2: KV-cache residency management**

| Step | Detail |
|------|--------|
| Runtime event | KV-cache pressure rises |
| GyroGraph response | Cells diverge in spectral64, horizon_distance, ab_distance, current_resonance |
| Actuator reads | current_resonance, horizon_distance, ab_distance, spectral64 |
| Actuator decision | Decide which request states remain resident |

**N3: TTFT pressure detection**

| Step | Detail |
|------|--------|
| Runtime event | Time-to-first-token rises under mixed prefill/decode pressure |
| GyroGraph response | Affected cells shift in shell occupancy and chirality regularity |
| Actuator reads | shell, spectral64, q6, graph queries |
| Actuator decision | Reshape request groups or separate service paths to recover TTFT |

---

### Appendix C: Implementation Notes

#### C.1 Repository file layout

```text
src/tools/gyrograph/
    __init__.py
    core.py
    profiles.py
    serializers.py
    ops.py
    gyrograph.c
    gyrograph_opencl.c
    bridges/
        __init__.py
        applications.py
        databases.py
        networks.py
        bolmo_config.py
        decode.py
    scripts/
        __init__.py
        _common.py
```

| File | Role |
|------|------|
| ops.py | ctypes loading, automatic C build, CPU/OpenCL dispatch, Python fallback, batched ingest and trace entry points |
| gyrograph.c | Native CPU batched helpers: Ω stepping, local-memory updates, omega signature, parity commitment |
| gyrograph_opencl.c | OpenCL backend: parallel omega_trace4 and chi_trace4 on GPU; histogram and ring updates on CPU |
| core.py | Cell pools, ingestion, SLCP emission, graph queries, lifecycle, snapshot/restore |
| serializers.py | Shared serializer helpers and deterministic word construction |
| profiles.py | Resonance profile definitions and keying functions |
| bridges/applications.py | Applications bridge |
| bridges/databases.py | Databases bridge (reserved) |
| bridges/networks.py | Networks bridge (reserved) |
| bridges/bolmo_config.py | Byte-native model-control bridge and decode report surfaces |
| bridges/decode.py | Climate helpers, gauge-spectrum helpers, and decode-side structural utilities |
| scripts/_common.py | Shared formatting and diagnostic helpers |

#### C.2 Python surface

**Exports from `__init__.py`:**

```python
from .core import GyroGraph, SLCPRecord
from .profiles import ResonanceProfile
from .serializers import pack_word4, ensure_word4
from .bridges.bolmo_config import (
    BolmoDecodeReport,
    GyroGraphBolmoDecodeBridge,
    PairedContentMetrics,
    PairedStepRecord,
    PatchRecord,
)
from .bridges.decode import (
    QuBECClimate,
    compute_qubec_climate,
    gauge_spectrum_from_family_hist,
)
```

**GyroGraph constructor:**

```python
def __init__(
    self,
    cell_capacity: int,
    profile: ResonanceProfile = ResonanceProfile.CHIRALITY,
    *,
    enable_ingest_log: bool = False,
    ingest_log_path: str | None = None,
    use_native_hotpath: bool = True,
    use_opencl_hotpath: bool = True,
    opencl_min_batch: int = 128,
    opencl_platform_index: int = 0,
    opencl_device_index: int = 0,
) -> None:
```

**Core methods:** allocate_cells, free_cells, seed_rest, seed_equality_horizon, seed_shell, seed_omega, ingest, emit_slcp, get_co_resonant_cells, get_bucket_population, get_bucket_cells, get_cells_on_shell, get_cells_with_chi6, get_cells_with_signature, chirality_distance_between_cells, snapshot, restore, decay_resonance_buckets, shell_spectral, set_ingest_log_path, iter_ingest_log, clear_ingest_log.

**Properties:** capacity, profile, active_cell_count, active_cell_ids, native_hotpath_enabled, opencl_hotpath_enabled, ingest_log_enabled.

**SLCPRecord dataclass:**

```python
@dataclass
class SLCPRecord:
    cell_id: int
    step: int
    omega12: int
    state24: int
    last_byte: int
    family: int
    micro_ref: int
    q6: int
    chi6: int
    shell: int
    horizon_distance: int
    ab_distance: int
    omega_sig: int
    parity_O12: int
    parity_E12: int
    parity_bit: int
    resonance_key: int
    current_resonance: int
    spectral64: np.ndarray

    def stabilizer_type(self): ...
    def charts(self): ...
    def future_cone(self, n: int): ...
    def future_locus(self, n: int): ...
    def optical_coordinates(self): ...
```

#### C.3 Native boundary

GyroGraph has its own native execution layer alongside GyroLabe:

| Component | Responsibility |
|-------|----------------|
| GyroLabe | Kernel-level algebra: signatures, chirality distance, WHT, Lattice Multiplication GEMV, SDK execution |
| gyrograph.c | Batched CPU: full ingestion cycle per cell in a single pass |
| gyrograph_opencl.c | GPU-parallel: 4-step Omega trace for all cells; histogram and ring updates on CPU |
| ops.py | Build automation, compiler detection, ctypes setup, dispatch between CPU/OpenCL/Python fallback |

Neither layer redefines kernel law. Both implement the same exact Ω stepping rule.


