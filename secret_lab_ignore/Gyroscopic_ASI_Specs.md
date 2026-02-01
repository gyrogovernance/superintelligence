
```
┏━╸╻ ╻┏━┓┏━┓┏━┓┏━╸┏━┓┏━┓╻┏━╸                   
┃╺┓┗┳┛┣┳┛┃ ┃┗━┓┃  ┃ ┃┣━┛┃┃                     
┗━┛ ╹ ╹┗╸┗━┛┗━┛┗━╸┗━┛╹  ╹┗━╸                   
┏━┓┏━┓╺┳╸╻┏━╸╻┏━╸╻┏━┓╻                         
┣━┫┣┳┛ ┃ ┃┣╸ ┃┃  ┃┣━┫┃                         
╹ ╹╹┗╸ ╹ ╹╹  ╹┗━╸╹╹ ╹┗━╸                       
┏━┓╻ ╻┏━┓┏━╸┏━┓╻┏┓╻╺┳╸┏━╸╻  ╻  ╻┏━╸┏━╸┏┓╻┏━╸┏━╸
┗━┓┃ ┃┣━┛┣╸ ┣┳┛┃┃┗┫ ┃ ┣╸ ┃  ┃  ┃┃╺┓┣╸ ┃┗┫┃  ┣╸ 
┗━┛┗━┛╹  ┗━╸╹┗╸╹╹ ╹ ╹ ┗━╸┗━╸┗━╸╹┗━┛┗━╸╹ ╹┗━╸┗━╸                
```

# Gyroscopic Artificial Superintelligence Specifications

**Multi-Agent Architecture for Aligned Intelligence**

---

## 1. Architectural Overview

Traditional artificial intelligence systems operate through statistical approximation, predicting outputs based on patterns learned from data in unbounded high-dimensional spaces. This approach permits the generation of invalid states and requires continuous external correction to maintain coherence. Gyroscopic Artificial Superintelligence takes a fundamental departure from this paradigm by operating through geometric realisation.

The architecture is built upon a deterministic finite-state coordination kernel. This kernel, specified in the GGG ASI Alignment Router document and referred to throughout this specification simply as "the kernel", enforces a strict separation between coordination physics and semantic interpretation. The physics are handled by the kernel, which transforms input bytes into discrete states within a closed, finite manifold. Interpretation is handled by the agent, which uses a deterministic operator to select traversal paths based on external context.

The curriculum that shapes the agent before deployment, including language model weights, training data, tokeniser definitions, and the token-to-byte mapping, is fixed once the agent begins operation. The agent does not modify its weights during operation. All adaptation occurs through trajectory in the kernel state space, recorded in the Genealogy.

Open models such as OLMo may be used before deployment to produce trained parameter sets. At runtime the agent is not required to execute transformer-style forward passes. The trained parameters of such models are treated as a static parameter manifold that can be read on demand when constructing embedding vectors. No further training or gradient descent occurs after deployment; the only live dynamics are the kernel state, the Inference Function field, and the append-only Genealogy.

### The Five Constitutional Artefacts

The system is defined by five distinct artefacts that map the theoretical constraints of the Common Governance Model to executable software components.

**The Ontology** establishes reference. It is the complete enumeration of all valid coordination states, consisting of exactly 65,536 unique positions reachable from a universal reference state known as the Archetype (value 0xAAA555). Because the set is finite and enumerated, the system cannot enter invalid states; it can only occupy positions that exist within the Ontology. The Ontology artefact occupies 256 kilobytes and is built once and shared by all agents.

**The Epistemology** provides motion. It is the complete transition logic, stored as a precomputed lookup table with 65,536 rows and 256 columns, totalling 16,777,216 entries. Each entry maps a current state and input byte to the resulting next state. Each byte action is a bijection on the Ontology, meaning the transition is reversible: given a state and the byte that produced it, the predecessor state can be computed algebraically. The Epistemology artefact occupies 64 megabytes.

**The Phenomenology** governs interaction. It consists of two layers:

1. A **spectral atlas** (`phenomenology.npz`) generated entirely from the kernel physics. This atlas stores:
   - Per-state observables: `state_horizon[state]` and `state_vertex[state]`,
   - A spectral phase cube `phase[state, byte]` giving each state's position in the 4-cycle induced by each byte,
   - Backward-pass observables `next_horizon[state, byte]`, `next_vertex[state, byte]`, and `next_phase[state, byte]` used to "peek" the consequences of actions without stepping the kernel,
   - Helper tables for mask weights, K₄ charges, direction factors, and per-byte feature matrices for all supported K.
2. A dynamic **Inference Function** in the agent, which uses the atlas to perform phase-aware inference. It maintains a field M[h, p, :] for horizon h and phase p, and routes high-dimensional embeddings into byte actions by respecting the spectral geometry.

**The Genealogy** ensures memory. It is the complete, append-only record of all bytes processed by the agent. Unlike hidden state vectors in neural networks, the Genealogy provides a perfect audit trail. Any past state can be reconstructed by replaying the Genealogy from the Archetype.

The Genealogy is produced online. As the agent reads or generates, each byte is appended in real time. There is no separate offline “training” pass at the kernel level; reading and acting use the same loop, and the Genealogy is simply the record of that loop.

**The Common Source Moment** quantifies capacity. It is the fundamental unit of coordination capacity, derived from the caesium-133 hyperfine transition frequency (9,192,631,770 Hz) coarse-grained by the Ontology size. Each byte appended to the Genealogy consumes one Common Source Moment from a total capacity of approximately 4.96 × 10²⁵. A byte log of one million entries occupies approximately one megabyte. At a processing rate of one million bytes per second, exhausting the total capacity would require approximately 1.6 × 10¹² years.

---

## 2. Geometric Foundations

The architecture relies on specific geometric properties of the kernel to ensure alignment. These properties provide mathematical guarantees that replace statistical likelihoods.

### State Anatomy

The kernel operates on a 24-bit state, split into two 12-bit components designated A (active) and B (passive). Each component maps to a discrete 2×3×2 grid (two frames, three rows, two columns), representing a three-dimensional space with six degrees of freedom.

Each input byte maps to a 12-bit value called a mask through a fixed expansion function. The mask determines which bits of the A component are toggled during a transition. After mutation, the kernel performs a gyration step: the next A becomes the complement of the current B, and the next B becomes the complement of the mutated A. This asymmetric update creates fundamental chirality in the transition law.

The kernel exhibits a phase transition at mask weight two. When only bytes with mask weight zero or one are used, the reachable states form a restricted bubble of 256 states. Adding any byte with mask weight two or greater unlocks the full state space of 65,536 states.

### The Horizon

Within the Ontology, there exists a distinguished subset of 256 states called the Horizon. These are the states that remain invariant under the reference byte (0xAA), characterised by the condition that A equals the bitwise complement of B.

The Horizon acts as a holographic boundary. Every state in the full Ontology can be uniquely decomposed into a Horizon anchor and a single local byte. Given any state, the Horizon anchor shares its A component, and the byte is determined by the mask difference. This bijection allows the agent to navigate the large state space by referencing stable boundary markers. States that share a Horizon anchor are geometrically related and can be treated as variations of the same underlying configuration.

### The K₄ Quotient and Vertex Charge

The kernel exhibits a natural four-part symmetry based on the complete graph on four vertices (K₄). This graph has four vertices and six edges.

The 256 byte operations partition into four classes based on a property called vertex charge, calculated using two parity-check vectors (0x033 and 0x0F0). For any 12-bit mask, the vertex charge is a 2-bit value formed from the parity of the mask ANDed with each vector. This partitions the Ontology into four wedges of 16,384 states each.

The quotient induces a coarse 16-state view of the full dynamics, computed from the vertex charge pairs of the two components. Transitioning between wedges represents a shift in governance mode, allowing the agent to recognise distinct types of operation purely through geometry.

### Spectral Phase in Byte-Permutation Cycles

Each non-reference byte defines a permutation of the 65,536-state ontology. For `b ≠ 0xAA`, this permutation decomposes into disjoint cycles of length 4, giving four discrete eigenphases {1, i, −1, −i}. The spectral atlas assigns to each state and byte a **phase index** `p ∈ {0,1,2,3}` that tracks its position within the corresponding cycle, normalized by a canonical anchor. The reference byte `0xAA` acts as an involution, with fixed points on the horizon and 2-cycles elsewhere; its phases are restricted to {0,1}.

This phase coordinate plays a similar role to a frequency axis in a spectral image: it is neither a time parameter nor a rate, but a discrete spectral coordinate that the Inference Function uses as a second axis of memory alongside the horizon.

### Aperture

To measure alignment conceptually, vectors defined on the edges of K₄ are subjected to Hodge decomposition. This mathematical technique splits any flow of information into two orthogonal components: a gradient component representing flows consistent with a global potential, and a cycle component representing local circulation.

This quantity is referred to as aperture and is defined as the fraction of total energy contained in the cycle component. The Common Governance Model establishes a target aperture of approximately 0.0207. The kernel possesses an intrinsic aperture of approximately 0.0195 (specifically 5/256). The difference between these values, approximately 0.00117, serves as the canonical learning rate (denoted η) for the Phenomenology operator. This ties the maximum per-step adjustment to the geometric properties of the space rather than an arbitrary hyperparameter.

---

## 3. Kernel Tools

The kernel provides several verified invariants that support audit, integrity checking, and efficient reasoning about trajectories.

**Parity Commitment.** For any byte sequence, the trajectory admits a compact commitment consisting of the exclusive-or of all masks at odd positions, the exclusive-or of all masks at even positions, and the length parity. This allows comparison of long histories without full replay. Two histories with different commitments cannot have produced the same final state.

**Reversibility.** Every kernel transition has an algebraic inverse. Given the current state and the byte that was applied, the predecessor state can be computed directly. This supports rollback operations and precise attribution of state changes without requiring storage of intermediate states.

**Commutator Translation.** For any two bytes, the difference between applying them in one order versus the reverse order is a state-independent translation. This is a verified property of the kernel dynamics rather than a dedicated API call. It allows reasoning about operation reordering without simulating from each possible state.

**Syndrome Checking.** A 16-element dual code provides a fast syndrome test for accidental corruption of masks. A non-zero syndrome indicates data corruption.

---

## 4. The Inference Function (Spectral Phenomenology)

The Inference Function is the core of the agent's intelligence. It acts as the bridge between the continuous, high-dimensional embeddings of a language model and the discrete, finite spectral geometry of the kernel. It is deterministic in form and introduces no learned parameters beyond those fixed at deployment; instead, it accumulates a trajectory-dependent field.

### Dimensional Interface

The Inference Function accepts external numeric vectors (embeddings) of dimension D. To interface with the kernel's 256-state horizon, D must be a multiple of 256. The number of channels per horizon is defined as K such that D = 256 × K. If the external vector does not match the configured dimension, the Inference Function raises an error.

The Inference Function does not require that these vectors be token embeddings. Any external source that can present a deterministic vector of the appropriate dimension is admissible. This includes, in particular, vectors built from slices of open model parameters, as well as tool outputs, sensor data, or any other structured numeric input that can be mapped deterministically into a D-dimensional vector. All such vectors are treated uniformly as **fields** over the horizon: the embedding is reshaped to a real-valued array X of shape 256 × K, and the field is sampled at a specific horizon index and phase.

### Phase-Aware Memory

The agent maintains a dynamic field M of shape (256, 4, K). This field represents the agent's accumulated experience along its trajectory, indexed by:
- horizon index h ∈ {0..255},
- phase index p ∈ {0..3} (cycle position in byte permutations),
- channel index c ∈ {0..K−1}.

For each inference step, given the current (h, p), the Inference Function:

1. Extracts the current local activation a_curr = X[h, :].
2. Computes a direction factor γ from:
   - the mask weight of the most recent transition,
   - the vertex charges of the previous and current states,
   - using a precomputed `gamma_table[ χ_prev, χ_curr, weight ]`.
3. Applies a phase-aware Hebbian update:

   ΔM[h,p,:] = η · γ · (a_curr ⊙ a_prev)

   with clipping to ensure numerical stability.
4. Scores all 256 candidate bytes from:
   - the projection of `M[h,p,:] + a_curr` onto per-byte feature vectors,
   - a weight penalty that prefers lighter masks,
   - a K₄ vertex coherence term that rewards staying in or near the same wedge.

In deterministic mode, the Inference Function selects the argmax byte. In sampling mode, it draws from a softmax distribution over scores, optionally restricted to a subset of bytes determined by the token vocabulary size.

### Spectral Role

The inclusion of phase p as an index of M is the discrete analogue of adding a spectral axis to an image: the agent's memory is not only "where am I on the horizon?" but also "where am I within the byte-cycle phase?". This is the extension of the Inference Function that incorporates spectral phase alongside the Epistemology's geometric foundation; it raises the system from a three-degree-of-freedom motion to a six-degree-of-freedom spectral navigation.

---

## 5. Agent Operation

The agent operates in a continuous cycle of input processing (Egress) and output generation (Ingress). Both phases advance the kernel state by bytes and append to the Genealogy.

### Token Mapping

Tokens are treated as 16-bit coordinations in the internal ontology address space. For token identifiers t in the range 0 to 65,535 inclusive, the agent defines:

```python
b1 = (t >> 8) & 0xFF
b2 = t & 0xFF
```

Each token is realized as a path of length two in the 256-byte action alphabet. The kernel remains byte-native: it applies each of the two bytes in sequence, advancing the state via the Epistemology lookup. No modulo-256 collapse is used; the mapping from tokens to trajectories is lossless up to `vocab_size ≤ 65,536`.

### Input Processing (Egress)

When the agent receives a token from an external source, it:

1. Splits the token ID into `(b1, b2)`.
2. Applies each byte to the kernel in order:
   - `kernel.step_byte(b1)`,
   - `kernel.step_byte(b2)`.
3. Logs both bytes into the Genealogy.

This preserves a pure byte trajectory while ensuring that each token has a unique 16-bit internal coordination.

### Output Generation (Ingress): Palindromic Two-Byte Inference

To generate an output token, the agent:

1. Obtains an embedding vector for the current context from an embedding function and adapts it to dimension D = 256 × K. This embedding function may read from a pre-trained language model, from its static parameter tensors, or from any other deterministic source that provides a D-dimensional vector.
2. Reshapes the embedding into a field X ∈ ℝ^(256×K).
3. Reads the current spectral observables from the kernel:
   - horizon `h₀`,
   - vertex charge `χ₀`,
   - phase `p₀`,
   - and the last transition mask and previous vertex `(Δ₀, χ_prev0)` tracked by the agent.
4. Invokes the Inference Function to select the first byte `b1`, optionally restricted by the vocabulary size:

   ```python
   b1 = infer.step_with_field(
       state=inference_state,
       X_curr=X,
       h_curr=h0,
       p_curr=p0,
       delta_mask=Δ0,
       chi_prev=χ_prev0,
       chi_curr=χ0,
       deterministic=deterministic,
       allowed_max_byte=max_b1,  # derived from vocab_size
   )
   ```

5. Uses the spectral atlas **without stepping** to peek the next-state observables under `b1`:

   ```python
   h1   = kernel.peek_next_horizon(b1)
   χ1   = kernel.peek_next_vertex(b1)
   p1   = kernel.peek_next_phase(b1)
   Δ1 = mask12_for_byte(b1)
   ```

6. Invokes the Inference Function again to select the second byte `b2`, now constrained by the peeked `(h1, p1, χ₁)` and the vocabulary:

   ```python
   b2 = infer.step_with_field(
       state=inference_state,
       X_curr=X,
       h_curr=h1,
       p_curr=p1,
       delta_mask=Δ1,
       chi_prev=χ0,
       chi_curr=χ1,
       deterministic=deterministic,
       allowed_max_byte=max_b2,
   )
   ```

7. Only then applies both bytes to the kernel and logs them to the Genealogy.
8. Combines `(b1, b2)` into a 16-bit token:

   ```python
   token_id = (b1 << 8) | b2
   ```

This "palindromic" pattern reflects BU-Egress and BU-Ingress at the byte level: the first half of the loop chooses a coarse spectral move, while the second half uses a peeked view of the future state to refine the action cheaply and coherently.

### Tool Execution

Tool calls are handled as byte sequences. The identifier for a tool maps to a specific sequence of bytes. Executing a tool involves stepping the kernel through this sequence, with each byte appended to the Genealogy. The result of the tool is similarly encoded and processed as input. This ensures that all functional actions are recorded in the Genealogy and contribute to the evolution of the kernel state.

---

## 6. Memory and Generalisation

### The Genealogy as Memory

The Genealogy serves as the sole persistent memory store. It is a linear sequence of bytes. Because the kernel is deterministic, the kernel position can always be reconstructed by replaying the Genealogy from the Archetype. If the external model that provides embeddings is deterministic with respect to the history and its weights remain fixed, then the accumulated field M can also be reconstructed by replaying the Genealogy together with the same model.

Apart from the Genealogy, all other agent state, including the field M and the current kernel state, is either derived from or replayable from this byte log.

Given any byte in the history and the state it produced, the predecessor state can be recovered through algebraic inversion. This reversibility supports audit, rollback, and precise attribution of state changes.

The parity commitment (described in Section 3) provides efficient comparison of histories without full replay.

### Holographic Generalisation

Generalisation in this architecture arises from the holographic structure of the Horizon. Since every state decomposes into a horizon anchor and a local variation, states that share the same horizon anchor occupy related positions in the kernel's finite state space. The agent treats them as variations of the same underlying configuration. This allows the agent to apply patterns learned in one specific context to other contexts that share the same boundary conditions.

### History Degeneracy

A significant feature of the kernel is provenance degeneracy: many different sequences of bytes can lead to the same final state. Under a restricted generator alphabet of size 8, words of length 6 form 262,144 possible sequences that map onto only 4,096 final states from the Archetype, with a uniform preimage size of 64.

This provides native compression. The agent does not need to store every unique path as a distinct concept. Operationally equivalent histories naturally collapse into the same geometric position, allowing the agent to generalise across different experiences that yield the same result.

### Coarse Navigation

The 16-state quotient system (derived from vertex charge pairs) provides a macro-view of position. The agent can monitor which of the four K₄ vertices each component is associated with. Transitioning between coarse regions corresponds to shifting governance modes.

---

## 7. Implementation Reference

### Dependencies

Gyroscopic ASI requires the Router substrate and an embedding function. A language model is one possible source of embeddings but is not required at runtime:

```python
from src.agent.intelligence import GyroscopicAgent, AgentConfig
from src.agent.inference import InferenceFunction, InferenceState

# Example: external LM with its own tokenizer (optional source)
from transformers import AutoModelForCausalLM, AutoTokenizer

# The above example illustrates integration with a transformer model used as an 
# embedding source. In parameter-manifold configurations, the agent instead 
# constructs embedding vectors directly from fixed model parameters or other 
# static tensors, without executing a forward pass through a language model.
```

### Token Mapping

The interface between the language model and the kernel uses 16-bit tokens:


```python
def token_to_bytes(token_id: int) -> tuple[int, int]:
    b1 = (token_id >> 8) & 0xFF
    b2 = token_id & 0xFF
    return b1, b2
```

Each token is split into two bytes for processing through the kernel. The mapping is total and deterministic, preserving the full 16-bit token space up to `vocab_size ≤ 65536`.

### Inference Function Initialisation

```python
from src.agent.inference import InferenceFunction, InferenceState
from src.agent.information import ETA_DEFAULT

inference = InferenceFunction(K=3, eta=ETA_DEFAULT)
state = InferenceState.create(K=3)
```

The field M initialises to zero. The learning rate η defaults to the Aperture gap.

The Inference Function obtains its kernel-dependent tables (byte weights, byte charges, direction factors, and feature matrices) indirectly via the `RouterKernel` and the spectral atlas (`phenomenology.npz`), using:

```python
inference.set_kernel_tables(
    byte_weight=kernel.byte_weight,
    byte_charge=kernel.byte_charge,
    byte_features=features_K3,       # loaded from phenomenology.npz
    gamma_table=kernel.gamma_table,
)
```

### Agent Initialisation

```python
from src.agent.intelligence import GyroscopicAgent, AgentConfig
from pathlib import Path

config = AgentConfig(
    K=3,
    eta=0.00117,
    deterministic=True,
    atlas_dir=Path("data/atlas"),
)

agent = GyroscopicAgent(config=config)
```

### Generation Loop

```python
from pathlib import Path
from src.agent.intelligence import GyroscopicAgent, AgentConfig

config = AgentConfig(
    K=3,
    eta=ETA_DEFAULT,
    deterministic=True,
    atlas_dir=Path("data/atlas"),
    vocab_size=tokenizer.vocab_size,
)

agent = GyroscopicAgent(config=config)

# Process input tokens
for token_id in input_tokens:
    agent.step_input(token_id)

# Generate output
output_tokens = []
context = list(input_tokens)

for _ in range(max_output):
    last_token = context[-1] if context else 0
    embedding = get_embedding(last_token)    # Embedding function
    token = agent.step_output(embedding=embedding)
    output_tokens.append(token)
    context.append(token)
```

The function `get_embedding()` belongs to the embedding layer. It may, for example, look up a static embedding, read from pre-trained parameter tensors, or call a language model. The agent consumes embedding vectors and returns token identifiers; any additional use of logits or probabilistic information is handled entirely outside the agent or via a separate adapter.

### Agent State

The agent's complete state consists of: the kernel instance (providing the current state index), the Genealogy (byte log), the **Inference field M** (phase‑aware memory M[h,p,:]), and a reference to the embedding function for token generation.

### Verification

Any party with the Genealogy can verify the agent's state by initialising the kernel at the Archetype, replaying each byte through standard processing, and comparing the final state to the claimed position. Determinism guarantees identical replay produces identical results.

### Byte Feature Vector Computation

```python
def byte_feature_vector(byte: int, K: int) -> np.ndarray:
    m12 = mask12_for_byte(byte)
    bits = np.array([(m12 >> i) & 1 for i in range(12)], dtype=float)
    bits = 2.0 * bits - 1.0  # Map to {-1.0, +1.0}

    if K == 12:
        return bits

    if K == 3:
        row_groups = ((0, 1, 6, 7), (2, 3, 8, 9), (4, 5, 10, 11))
        return np.array([sum(bits[i] for i in group) / 4.0 for group in row_groups])

    if K == 6:
        frame_row_groups = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11))
        return np.array([sum(bits[i] for i in group) / 2.0 for group in frame_row_groups])

    if K == 16:
        out = np.zeros(16, dtype=float)
        out[:12] = bits
        out[12] = float(popcount(m12) & 1) * 2.0 - 1.0
        out[13] = float(popcount(m12 & 0x3F) & 1) * 2.0 - 1.0
        out[14] = float(popcount((m12 >> 6) & 0x3F) & 1) * 2.0 - 1.0
        out[15] = float(vertex_charge(m12) & 1) * 2.0 - 1.0
        return out

    raise ValueError(f"Unsupported K={K}")
```

The following example shows the construction for the recommended K values. The reference implementation additionally supports K = 1, 2, 4, and 8 using the same grouping principles.

In the reference implementation, these feature vectors are precomputed for all K and stored in `phenomenology.npz` as `features_K{K}`.

---

## 8. Performance Characteristics

The architecture is designed for constant-time operation relative to the history length.

### Time Complexity

All core operations complete in constant time per byte:

| Operation | Time |
|---|---|
| State transition | Single array lookup |
| Transcription (byte to intron) | Single exclusive-or operation |
| Token mapping | Bit-shift and mask (and optional O(1) table lookup via TokenBinding) |
| Genealogy append | Single write operation |
| Parity commitment update | Three exclusive-or operations |

The cost of constructing the embedding vector depends on the chosen embedding function. In the parameter-manifold configuration, this construction consists of reading selected rows or slices from fixed parameter tensors and mapping them into the D-dimensional field, without executing full matrix multiplications or attention mechanisms.

### Space Requirements

| Component | Size |
|---|---|
| Epistemology table (shared) | 64 megabytes |
| Ontology (shared) | 256 kilobytes |
| Spectral atlas (phenomenology.npz, shared) | ~64 megabytes |
| Kernel state | 3 bytes |
| Genealogy | Linear in history length |
| Accumulated field M (K=3) | ~12 kilobytes |
| Accumulated field M (K=16) | ~64 kilobytes |

### Throughput

On a modern processor with the Epistemology table cached:

| Operation | Rate |
|---|---|
| State transitions | Approximately 2.6 million per second |
| Complete operation cycles | Approximately 650,000 per second |

---

## 9. Alignment Properties

The architecture enforces alignment through geometric construction rather than external policy.

**Validity.** The agent cannot enter an undefined state. The Ontology is finite and closed; every possible transition leads to a valid known state within the enumerated set.

**Traceability.** The Genealogy provides a complete causal chain. Every output can be traced back to the specific inputs and internal states that generated it through deterministic replay.

**Accountability.** The reversibility of the kernel means that any sequence of actions can be unwound. Given a state and the byte that produced it, the predecessor state is recoverable. This supports rollback operations and precise attribution.

**Coherence.** The kernel's finite state space and the direction factor reduce the likelihood of incoherent transitions. The direction factor inhibits movement between opposing governance vertices, creating a constraint that guides the agent toward trajectories that respect the K₄ connectivity.

The design is intended to reduce the likelihood of common generation pathologies:

| Pathology | Reduction Mechanism |
|---|---|
| Output incoherence | Every token advances the kernel through a finite manifold. The accumulated field M creates attractors. Incoherent token sequences hit low-energy regions or cross wedge boundaries with negative direction factors, reducing their selection probability. |
| Verbatim repetition | M accumulates from the current trajectory, not from frozen training weights. The same tokens arriving from different prior states produce different updates to M and different byte selections. |
| Context loss | The Genealogy is append-only and permanent. The kernel state encodes the full trajectory. Replay reconstructs any prior state exactly. |

---

## 10. Viable System Model Mapping

The architecture admits a mapping to Beer's Viable System Model for organisational analysis.

| Subsystem | Function | Component |
|---|---|---|
| System 1 | Primary activities | Kernel stepping via Epistemology |
| System 2 | Coordination | Genealogy and parity commitment |
| System 3 | Control | Kernel invariants and reversibility |
| System 4 | Environment interface | Embedding input and token output |
| System 5 | Policy and identity | **Inference Function** and byte selection |

---

## 11. Conclusion

Gyroscopic ASI specifies an agent architecture in which a deterministic kernel provides a reproducible coordination substrate and an append-only byte history provides the basis for replay and reconstruction. Interpretation is constrained to the agent layer through a deterministic **Inference Function** and its spectral atlas (Phenomenology), which consume external embeddings and kernel observables to select output bytes.

These embeddings may be constructed from pre-trained open model parameters without executing forward passes at inference time.

The five artefacts (Ontology, Epistemology, Phenomenology, Genealogy, Common Source Moment) correspond to the five foundational constraints of the Common Governance Model. The Ontology establishes reference. The Epistemology provides motion. The Phenomenology governs interaction. The Genealogy ensures closure and reconstruction. The Common Source Moment quantifies capacity.

The architecture operates on a finite, complete, and closed manifold where every state is valid by construction. It resists invalid outputs because invalid states do not exist in the Ontology. It resists drift because the finite state space contains no undefined regions. It resists context loss because the Genealogy is permanent and replayable.

---

## Appendix A: Dimensional Reference

### Token Representations

| Type | Bits | Typical Use |
|---|---:|---|
| int8 | 8 | Small vocabularies |
| int16 | 16 | Medium vocabularies |
| int32 | 32 | Large vocabularies |
| float16 | 16 | Compressed activations |
| float32 | 32 | Standard activations |
| float64 | 64 | High-precision computation |

### Embedding Dimensions

| D | Factorisation | K = D / 256 |
|---:|---|---:|
| 256 | 2⁸ | 1 |
| 512 | 2⁹ | 2 |
| 768 | 2⁸ × 3 | 3 |
| 1024 | 2¹⁰ | 4 |
| 1536 | 2⁹ × 3 | 6 |
| 2048 | 2¹¹ | 8 |
| 3072 | 2¹⁰ × 3 | 12 |
| 4096 | 2¹² | 16 |

### Channel Dimensions (2ⁿ × 3ᵐ pattern)

16, 32, 48, 64, 96, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096

The implementation supports K = 1, 2, 3, 4, 6, 8, 12, and 16 (see Section 4).

---

## Appendix B: Reference Constants

### Aperture Values

| Quantity | Value |
|---|---|
| Target Aperture | 0.020699553813 |
| Kernel intrinsic Aperture | 0.01953125 (5/256) |
| Learning rate (η) | 0.001168303813 |
| Monodromy defect (reference) | 0.1953 |
| Aperture scale (reference) | 0.1995 |

### Kernel Constants

| Quantity | Value |
|---|---|
| Archetype state | 0xAAA555 |
| Reference byte | 0xAA |
| Ontology size | 65,536 states |
| Epistemology size | 16,777,216 transitions |
| Horizon size | 256 states |

### Parity-Check Vectors

| Vector | Value |
|---|---|
| Q0 | 0x033 |
| Q1 | 0x0F0 |

### Direction Factor Values

| Vertex Relationship | Factor |
|---|---|
| Same vertex | +0.5 |
| Adjacent vertex | +1.0 |
| Opposite vertex | −1.0 |
| Baseline (additive) | +0.1 |

---

**Repository:** github.com/gyrogovernance/superintelligence  
**Kernel Specification:** docs/GGG_ASI_AR_Specs.md  
**Contact:** basilkorompilias@gmail.com