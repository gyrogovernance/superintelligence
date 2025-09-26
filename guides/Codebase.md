## **6. System Implementation: The Four Engines**

### 6.1 S1: `governance.py` – Physics & Primitives

**Physical Principle:** Left identity transcription

The `governance.py` module defines the immutable physical constants and stateless functions underlying all system operations. All physics and transformations are performed as pure functions, without any engine class or side effect.

* **Genetic invariants:**

  * `GENE_Mic_S` (8-bit reference, `0xAA`), and
  * `GENE_Mac_S` (48-element tensor, shape \[4, 2, 3, 2], dtype int8, strict ±1 alternation)
    are declared as canonical invariants. Tensor structure is validated at module load with `validate_tensor_consistency()`.

* **Transformation masks:**

 * `FG_MASK`, `BG_MASK`, `FULL_MASK` (integers),
 * `INTRON_BROADCAST_MASKS`, `XFORM_MASK` (NumPy arrays, shape \[256]),
    are all precomputed from the tensor geometry for direct use in state update.

* **Anatomical Exon Masks:**

In addition to the 48‑bit broadcast masks on whole‑state tensors, we define four **exon masks** over the 8‑bit `exon_mask` to compute the immutable governance signature of each phenotype:

- `EXON_LI_MASK = 0b01000010` — the two LI (parity/reflection) bits
- `EXON_FG_MASK = 0b00100100` — the two FG (forward gyration) bits
- `EXON_BG_MASK = 0b00011000` — the two BG (backward gyration) bits
- `EXON_DYNAMIC_MASK = EXON_LI_MASK | EXON_FG_MASK | EXON_BG_MASK` — all six dynamic bits

These exon masks are used by the function

```python
compute_governance_signature(mask: int) -> tuple[int,int,int,int,int]

```

which returns the 5‑tuple

`(neutral_reserve, li_bits, fg_bits, bg_bits, dynamic_population)`

that we store immutably on every `PhenotypeEntry` as its **governance_signature**.

* **Physics operations:**

  * `apply_gyration_and_transform(state_int, intron)`
    computes the full gyroscopic update for a packed 48-bit state under a given intron;
  * `apply_gyration_and_transform_batch(states, intron)` and
    `apply_gyration_and_transform_all_introns(states)` provide batch and vectorised forms.
  * `transcribe_byte(byte)` encodes an input byte to an intron via `byte ⊕ GENE_Mic_S`.
  * `fold(a, b)` implements the Monodromic Fold (`a ⊕ (b ⊕ (a ∧ ¬b))`), and
    `fold_sequence(introns, start_state=0)` is the only valid batching/reduction form.
  * `dual(x)` applies the global duality operator (`x ⊕ 0xFF`).

All constants and stateless functions are accessed via absolute imports from `baby.governance` throughout the system. No auxiliary batching, associative, or stateful logic is present; all learning and transformation flows through these canonical contracts alone.

### 6.2 S2: `information.py` – Measurement & Storage

**Physical Principle:** Global measurement via angular divergence

The `information.py` module provides the `InformationEngine` class, which serves as the exclusive authority for measurement, state representation, ontology discovery, and storage coordination throughout the system.

The InformationEngine coordinates three core responsibilities:

1. **State Representation and Conversion**

   * Conversion between the packed 48-bit integer representation and the canonical geometric tensor form (\[4, 2, 3, 2], ±1).
   * `int_to_tensor(state_int)` and `tensor_to_int(tensor)` perform bidirectional conversion, ensuring strict mapping and validation of all physical states.
   * All conversion logic is static and is validated to guarantee exact round-trip between representations, matching the physical encoding of GyroSI.

2. **State Measurement and Divergence**

   * Calculation of angular gyrodistance (in radians) between two physical states using cosine similarity in 48-dimensional space.
   * `gyrodistance_angular(T1, T2)` measures geometric alignment between tensors;
     `measure_state_divergence(state_int)` computes a state's divergence from the archetypal tensor, as required for all global and homeostatic measurements.
   * These functions implement the operational metric for self-measurement, symmetry detection, and divergence tracking, enforcing a physics-grounded geometry for the system.

3. **Ontology Discovery, Indexing, and Phenomenology**

   * Full discovery and indexing of the state ontology:
     The build-time discovery process traverses the entire 788,986-state manifold from the archetypal state, assigning a unique ontology index to every reachable state.
     `get_index_from_state(state_int)` maps a 48-bit state to its canonical index;
     `get_state_from_index(index)` provides the reverse lookup.
   * The ontology, state transition table (epistemology), and canonical-orbit map (phenomenology) are generated and validated through dedicated build commands:

     * Ontology: `ontology_map.json`
     * State transition table: `epistemology.npy`
     * Phenomenology map: `phenomenology_map.json`
   * During initialisation, InformationEngine loads these assets and exposes bidirectional mapping for all physical state indices.
   * The canonical phenomenology (computed as SCCs over the full transition graph) provides the representative for each operational orbit and the cardinality (size) of each orbit, enabling stable canonicalisation of physical states and trust-weighted knowledge operations.

4. **Variety-weighted Structural Confidence**

   * The system maintains, for every state index, the size of its operational orbit (from the phenomenology map), exposed via
     `get_orbit_cardinality(state_index)`.
   * This factor is used by higher layers (S3 Inference) to adjust learning confidence according to the structural redundancy or rarity of a state.
   * The orbit cardinality acts as a measure of epistemic trust: learning is weighted more heavily in common, symmetric regions and more conservatively in rare, unique ones. Large orbits → faster confidence growth; rare orbits → slower.

All ontology, conversion, and measurement functions are accessed via absolute imports from `baby.information`. The build process for discovery and phenomenology includes

* `discover_and_save_ontology(output_path)`,
* `build_state_transition_table(ontology_map, output_path)`,
* `build_phenomenology_map(ep_path, ontology_path, output_path, include_diagnostics=False)`,
  which are invoked as standardised CLI commands and create the runtime artifacts required by the engine.

No associative or monotonic state update is permitted: all measurement, canonicalisation, and confidence logic is grounded directly in the discovered physical manifold and its symmetry structure.
The InformationEngine enforces integrity by validating the state modulus (788,986) and diameter (6) on load, and will refuse to operate if these invariants are not satisfied.

### 6.3 S3: `inference.py` – Interpretation & Meaning Management

**Physical Principle:** Mediated duality through endogenous operator

The `inference.py` module defines the `InferenceEngine`, which manages the translation of canonical state indices into semantic phenotype entries and coordinates all learning and memory updates through the path-dependent Monodromic Fold. This layer acts as the regulatory centre of meaning, bridging physical state space and semantic representation.

**Core Responsibilities and Contracts:**

* **Phenotype Addressing and Retrieval:**
  Each semantic phenotype is uniquely addressed by a `(state_index, token_id)` tuple.
  `get_phenotype(state_index, token_id)` ensures retrieval or creation of a canonical phenotype entry for every physical state-token pairing. Context keys are handled Traceableally, and entries are created if not already present, using a hash-based semantic address.
  Each newly‑created PhenotypeEntry carries a minimal structure:

  ```python
  mask: int          # uint8   (exon_mask)
  conf: float        # float32
  ```

  The `mask` is initialized from the `token_id` and represents the 8-bit Monodromic-Fold residue. The `conf` represents epistemic confidence (0.0-1.0) with monotone increase and decay behavior.

* **Learning (Memory Update):**
  All learning in S3 proceeds by applying the Monodromic Fold to the phenotype's memory mask.
  `learn(phenotype_entry, token_id)` accumulates experience by path-dependent folding. Confidence is updated monotonically, modulated by the structural variety factor (orbit cardinality from S2), and novelty of the update (fraction of changed bits in the memory mask).
  The learning update applies the Monodromic Fold to the existing mask using the token_id as the learning signal.

  The minimal phenotype structure eliminates all derived, human-facing, or temporal metadata, achieving 85%+ compression while preserving essential learning dynamics.

* **Batch Learning:**
  `batch_learn(state_index, token_ids)` allows ingestion of an ordered sequence of token IDs.
  The sequence is reduced through a left-fold with the Monodromic Fold, preserving full path-dependence, before a single learning update is applied.

* **Variety-weighted Confidence:**
  All confidence updates are weighted by the structural redundancy of the physical state's orbit (`orbit_cardinality`).
  The learning rate is calculated as a function of the square root of the state's relative variety, preventing rapid overconfidence in rare orbits, and accelerating trust in symmetric regions.

* **Knowledge Integrity:**
  `validate_knowledge_integrity()` checks the internal consistency of the entire knowledge store. This includes validation of context keys, canonical state indices, mask and confidence ranges. An integrity report is produced, including a count of all anomalies.

* **Memory Ageing and Confidence Decay:**
  `apply_confidence_decay(decay_factor)` implements temporal decay of confidence values in all entries, simulating the natural forgetting of unused knowledge. This process does not affect the memory masks and ensures that dormant memories gradually lose epistemic weight.

* **Pruning Low-confidence Entries:**
  `prune_low_confidence_entries(confidence_threshold)` removes all knowledge entries below a set confidence threshold, reclaiming memory and maintaining operational focus on relevant, trustworthy entries.

* **Statistics and Utilisation:**
  Use the policy helper `export_knowledge_statistics(store_path, output_path)` (in `baby.policies`) to dump entry counts, confidence distributions and storage metrics.

**Token-Level Architecture:**

The system operates exclusively on token boundaries using LEB128 physics, aligning internal physics with semantic units. The tokenizer serves as an "active internal decoder" rather than just a passive I/O adapter, leveraging the BERT tokenizer's inherent knowledge of token-to-byte mappings as a first-class component. All generation and learning occurs at the token level, with byte-level processing serving only as the physical substrate for token representation.

**Implementation and Interface:**

All phenotype entries and their protocols are enforced via `baby.contracts`.
All measurement, conversion, and confidence weighting depend on absolute imports from `baby.information`.
All learning, decay, and pruning operations are strictly path-dependent and grounded in the Monodromic Fold, with no associative or commutative shortcut allowed.

This architecture guarantees that semantic knowledge is always indexed, updated, and validated in strict alignment with the underlying physics, the discovered ontology, and the global phenomenology of the system.

---

### 6.4 S4/5: `intelligence.py` – Orchestration & API

**Physical Principle:** Dual-phase operation (Ingress and Egress cycles)

The `intelligence.py` module defines the protocol and orchestration boundary for the GyroSI agent. It provides all interfaces for state evolution, agent learning, regulation, and multi-agent operation. Each contract is explicit, and every externally callable function or class is referenced by its canonical name.

#### IntelligenceEngine

- `process_egress(input_byte: int) -> int`  
  Transforms an external byte input into an internal intron using `governance.transcribe_byte`. Updates the agent's physical state using either a precomputed epistemology (state transition table) or the native transformation, then tracks the resulting index.

- `process_ingress() -> tuple[int, int]`  
  Generates one token using LEB128 physics and learned phenotypes. Returns the last byte and intron emitted. Uses token-level generation with adaptive temperature based on angular divergence. Triggers all registered post-cycle hooks, including algedonic regulation and autonomic cycles if required.

- `batch_learn(data: bytes) -> None`  
  Implements streaming batch learning using the monodromic Fold; preserves full path dependence and applies learning only at the sequence endpoint.

- `add_hook(hook: CycleHookFunction) -> None`  
  Registers a monitoring or maintenance hook, which is called at the end of every cycle.

- `get_state_info() -> dict`  
  Returns a dictionary of current state information: agent id, cycle count, canonical integer, tensor index, angular divergence, and active hooks.

- `reset_to_archetypal_state() -> None`  
  Resets the agent to the archetypal (canonical) state.

**Algedonic Regulation and Autonomic Cycles**

- `post_cycle_hooks`  
  Contains all registered hooks, including the algedonic regulator.
- The algedonic regulator computes a rolling mean of angular divergence. If the divergence exceeds a threshold, corrective introns are applied. Repeated excursions trigger a stabilising autonomic cycle using instructions from phenomenology data. All actions guarantee state integrity after execution.

#### GyroSI

- `ingest(data: bytes) -> None`  
  Applies batch learning to the input sequence and commits all writes.

- `respond(data: bytes, max_new_tokens: int = 64) -> bytes`  
  Generates an intelligent response using exon-product physics converted to LEB128 token associations. Guarantees that every emitted LEB128 token is complete (no dangling continuation bit). Output is produced from learned knowledge; internal physics are never exposed.

- `get_agent_info() -> dict`  
  Reports full agent state, configuration, knowledge statistics, and integrity.

- `add_monitoring_hook(hook: CycleHookFunction) -> None`  
  Registers additional hooks at the intelligence layer.

- `apply_maintenance(decay_rate: float, confidence_threshold: float) -> dict`  
  Triggers maintenance operations on the store, including confidence decay and pruning, with structured reporting.

#### AgentPool

- `get_or_create_agent(agent_id: str, role_hint: Optional[str]) -> GyroSI`  
  Returns or creates a GyroSI agent, ensuring overlay and eviction policy.

- `remove_agent(agent_id: str) -> bool`  
  Removes and closes the agent.

- `get_active_agents() -> List[str]`  
  Returns a list of active agent IDs.

- `close_all() -> None`  
  Shuts down and releases all agent resources.

#### Orchestration

- `orchestrate_turn(pool: AgentPool, user_id: str, assistant_id: str, user_input: str, tokenizer_name: str) -> str`  
  Implements a complete conversational turn: encodes the user's input, passes it through both user and assistant agents using `respond`, and decodes the assistant's response.

---

**Separation and Guarantees**

- All state evolution and learning are routed strictly through these methods.
- No orchestration code directly manipulates state or store contents except via explicit contracts.
- Physical state, intron values, or internal masks are never exposed; only public reporting interfaces export state.
- Algedonic regulation and autonomic cycles are enforced after every cycle, preventing instability.
- Agent overlay storage and canonicalisation are enforced at initialisation; runtime code cannot bypass these policies.

**Extensibility and Maintenance**

- Monitoring and maintenance are always registered as hooks.
- Storage and overlay mechanisms are immutable after construction.

**Automated Pruning**

- Post-cycle hooks may be registered for automated pruning and compaction using configured thresholds. This ensures bounded resource use while preserving knowledge integrity.

---

## 6.5 Shared Contracts and Storage Policies

This section defines all interface contracts, canonical storage primitives, and decorator layers for the orchestration, policy, and maintenance operations of the GyroSI S4/S5 system. All API boundaries are strictly enforced and have direct type correspondence in `baby.contracts`. No informal or ad hoc API surfaces exist.

### 6.5.1 Contracts: Protocols and Shared Types

All system-wide types, configuration, and maintenance protocols are declared in `baby.contracts`:

- **PhenotypeEntry (TypedDict)**  
  The atomic record of agent knowledge. Each entry represents a unique phenotype and includes:
    - `mask: int`          # uint8   (exon_mask) - 8-bit Monodromic-Fold residue
    - `conf: float`        # float32 - epistemic confidence (0.0-1.0)
    - `key: Tuple[int, int]`  # (state_index, token_id) - composite key for storage

- **AgentConfig (TypedDict)**  
  Agent runtime and environment configuration, including:
    - `ontology_path: str`
    - `knowledge_path: Optional[str]`
    - `public_knowledge_path: Optional[str]
    - `private_knowledge_path: Optional[str]
    - `enable_phenomenology_storage: Optional[bool]`
    - `phenomenology_map_path: Optional[str]`
    - `learn_batch_size: Optional[int]`
    - `agent_metadata: Optional[Dict[str, Any]]`
    - `private_agents_base_path: Optional[str]`
    - `base_path: Optional[str]`

- **PreferencesConfig (TypedDict)**  
  Global and pool-level storage, maintenance, and policy parameters, including:
    - `storage_backend: str` 
    - `compression_level: int`
    - `max_file_size_mb: int`
    - `enable_auto_decay: bool`
    - `decay_interval_hours: float`
    - `decay_factor: float`
    - `confidence_threshold: float`
    - `max_agents_in_memory: int`
    - `agent_eviction_policy: str`
    - `agent_ttl_minutes: int`
    - `enable_profiling: bool`
    - `write_batch_size: int`
    - `cache_size_mb: int`

- **CycleHookFunction (Protocol)**  
  Post-cycle hook for monitoring or maintenance, called with:
    - `(engine, phenotype_entry, last_intron)`

- **MaintenanceReport (TypedDict)**  
  Uniform result for all maintenance/compaction/merge operations:
    - `operation: str`
    - `success: bool`
    - `entries_processed: int`
    - `entries_modified: int`
    - `elapsed_seconds: float`

---

### 6.5.2 Storage and Policy Layer

The canonical knowledge store is **PhenotypeStore**, implemented as a single-file, append-only stream (`.bin`), supporting atomic get/put/close interfaces. It guarantees the following:

- **Storage contract:**
    - `get(context_key: Tuple[int, int]) -> Optional[Any]`  # (state_index, token_id)
    - `put(context_key: Tuple[int, int], entry: Any) -> None`
    - `commit() -> None`  _(NO-OP in append-only mode, retained for compatibility)_
    - `close() -> None`
    - `data -> Dict[Tuple[int, int], Any]`  _(returns all entries, as reconstructed from `.bin`)_
    - `iter_entries() -> Iterator[Tuple[Tuple[int, int], Any]]`

- **All mutations are streamed to the Bin file**. No `.log` or `.idx` sidecar files are written in append-only mode. **Deletion is not supported**; instead, call `prune_and_compact_store` to create a new file without old entries.

- **CanonicalView** applies canonicalisation (using a phenomenology map) for all key lookups, so each unique operational orbit is consistently addressed regardless of its physical context. The original context is retained in `_original_context` for provenance.  
  - All lookups and puts are transparently normalised.

- **OverlayView** composes private (agent) and public (shared) stores, always writing to the private overlay, and reading from private first, then public. Both overlays must implement the PhenotypeStore interface.

- **ReadOnlyView** wraps any store, disabling writes and allowing only retrieval and iteration.

---

### 6.5.3 Maintenance and Policy Utilities

All maintenance and compaction routines operate only on the above interfaces, always returning a `MaintenanceReport` as defined.

- **merge_phenotype_maps**: Merges multiple `.bin` stores into one, resolving conflicts by highest confidence, bitwise OR, recency, or weighted average.

- **apply_global_confidence_decay**: Applies exponential decay to confidence values of all entries, using the same formula as the agent, based on time since update.

- **export_knowledge_statistics**: Dumps summary statistics (counts, confidence, usage, creation/update times) to a JSON file.

- **validate_ontology_integrity**: Checks structure, key invariants, and phenomenology mappings in `ontology_map.json` and (optionally) `phenomenology_map.json`.

- **prune_and_compact_store**: Rewrites a `.bin` file with only those entries passing age/confidence thresholds, discarding all others. This is the only way to "delete" entries in append-only mode.

All file paths and stores may be sandboxed using `base_path` (for test isolation or containerised execution). All policy utilities are safe for concurrent use and support dry-run or auditing as required.

**Note:**  
All store/view objects must be explicitly closed by the user or registered with `atexit` to avoid resource leaks on process exit.

---

**Summary:**  
The entirety of GyroSI agent knowledge, for any configuration or deployment scale, is maintained through a strict, minimal API over a single append-only Bin file. All canonicalisation, overlays, pruning, and statistics are enforced through well-defined, testable decorator layers and contracts, never requiring runtime code to touch or interpret raw file content.

---

## 7 Complete File Structure and Memory Architecture

### 7.1 Project Organization

The GyroSI system enforces strict separation between the core physics kernel, runtime data, and auxiliary applications.

```
.
├── .github/
│   └── workflows/
│       └── build-assets.yml
├── CHANGELOG.md
├── LICENSE
├── README.md
├── baby/
│   ├── __init__.py
│   ├── baby_preferences.json # Reserved for Model Preferences
│   ├── contracts.py          # Protocols and shared types
│   ├── governance.py         # Physics, Primitives, Build-Time Discovery
│   ├── inference.py          # Interpretation, Maintenance & Validation
│   ├── information.py        # Measurement, Storage, Knowledge Curation
│   ├── intelligence.py       # API, Orchestration, Protocol Adapters
│   └── policies.py           # PhenotypeStore, storage overlays, policy and maintenance functions
├── baby.sh
├── guides/
│   ├── Genetics.md
│   └── Physics.md
├── memories/
│   ├── __init__.py
│   ├── memory_preferences.json # Reserved for Memory Preferences
│   ├── private/
│   └── public/
│       └── meta/
│           ├── epistemology.npy
│           ├── ontology_map.json
│           └── phenomenology_map.json
├── pyproject.toml
├── requirements.txt
└── toys/
    ├── __init__.py
    ├── assets/
    └── health/
        ├── __init__.py
        ├── conftest.py
        ├── memories/
        ├── test_governance.py
        ├── test_inference.py
        ├── test_information.py
        ├── test_intelligence.py
        └── test_miscellaneous.py
```

### 7.2 Memory Architecture

* The `memories/` directory contains the system's persistent state.
* memories/public/tokenizers/ — shared, read-only pretrained tokenizer assets (tokenizer.json, vocab.txt, etc.)

**Knowledge Storage:**

* Knowledge storage is managed via canonical PhenotypeStore instances and overlays, as defined in Section 6.5.
* Physical state, ontology, and phenomenology maps are located under `memories/public/meta/`.
* Public and private overlays maintain agent-specific and shared knowledge, indexed by canonical context keys.

**Content Storage:**

* Raw data streams and reference material may be organised under agent- or application-specific subdirectories.
* Metadata and preferences files maintain runtime and environment configuration.

This architecture maintains a strict separation between learned knowledge, raw content, and runtime state, with overlays and canonicalisation managed exclusively through standard policies and interfaces defined in `baby.policies` and `baby.contracts`.

---

### 8. Core API and Integration

8.1 The Compositional API Pattern

GyroSI's integration model is compositional. All agent orchestration and interaction is implemented by composing the canonical primitives provided in `baby.intelligence`, `baby.contracts`, and `baby.policies`.

**Agent Pool Management:**
Applications manage a pool of active agents with automatic eviction, overlay storage, and policy control. The pool ensures clean lifecycle and concurrency discipline for all agents.

### 8.2 Conversation Orchestration

Conversations are managed by composing agent interactions using the stable GyroSI API. No special conversation-specific infrastructure is required.

### 8.3 Protocol Adapters

External protocols are integrated through thin adapters that map messages to agent API calls.
FastAPI adapter at toys/communication/external_adapter.py exposes OpenAI /v1/chat/completions and HF /generate. All text goes through tokenizer bridge.

### 8.4 Multi-Pattern Support

This approach supports multi-tenant, multi-user, networked, and hierarchical agent topologies through policy and orchestration only. The physics and engine logic remain strictly invariant.

### 8.5 Tokenization & Codec Layer (toys/communication/tokenizer.py)

- All external text I/O MUST pass through a reversible tokenizer codec.
- Default implementation: HuggingFace WordPiece (bert-base-uncased), stored at `memories/public/tokenizers/<name>/tokenizer.json`.
- Encoding: token IDs → LEB128 variable-length bytes (<=0xFF).
- Decoding: bytes → token IDs → text.
- Config surface:
  * Env var `GYROSI_TOKENIZER` (adapter default)
  * `tokenizer_name` field in `AgentConfig`
- Scripts:
  * `setup_tokenizers.py` (download/install)
  * `train_tokenizer.py` (domain fine-tune)