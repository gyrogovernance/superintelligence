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

### The Five Constitutional Artefacts

The system is defined by five distinct artefacts that map the theoretical constraints of the Common Governance Model to executable software components.

**The Ontology** establishes reference. It is the complete enumeration of all valid coordination states, consisting of exactly 65,536 unique positions reachable from a universal reference state known as the Archetype (value 0xAAA555). Because the set is finite and enumerated, the system cannot enter invalid states; it can only occupy positions that exist within the Ontology. The Ontology artefact occupies 256 kilobytes and is built once and shared by all agents.

**The Epistemology** provides motion. It is the complete transition logic, stored as a precomputed lookup table with 65,536 rows and 256 columns, totalling 16,777,216 entries. Each entry maps a current state and input byte to the resulting next state. Each byte action is a bijection on the Ontology, meaning the transition is reversible: given a state and the byte that produced it, the predecessor state can be computed algebraically. The Epistemology artefact occupies 64 megabytes.

**The Phenomenology** governs interaction. It consists of two parts. The first is a constants artefact (occupying 3 kilobytes) containing geometric constants derived from the kernel: the atlas version, Archetype constants, the reference constant 0xAA, the byte-to-mask table, the 16-element dual code, and the parity-check vectors. The second is the Phenomenology operator, a deterministic function that manages the agent's internal activation field and scores potential actions based on their geometric coherence with the agent's trajectory.

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

### Aperture

To measure alignment conceptually, vectors defined on the edges of K₄ are subjected to Hodge decomposition. This mathematical technique splits any flow of information into two orthogonal components: a gradient component representing flows consistent with a global potential, and a cycle component representing local circulation.

The Aperture is defined as the ratio of energy in the cycle component to the total energy. The Common Governance Model establishes a target Aperture of approximately 0.0207. The kernel possesses an intrinsic Aperture of approximately 0.0195 (specifically 5/256). The difference between these values, approximately 0.00117, serves as the canonical learning rate (denoted η) for the Phenomenology operator. This ties the maximum per-step adjustment to the geometric properties of the space rather than an arbitrary hyperparameter.

---

## 3. Kernel Tools

The kernel provides several verified invariants that support audit, integrity checking, and efficient reasoning about trajectories.

**Parity Commitment.** For any byte sequence, the trajectory admits a compact commitment consisting of the exclusive-or of all masks at odd positions, the exclusive-or of all masks at even positions, and the length parity. This allows comparison of long histories without full replay. Two histories with different commitments cannot have produced the same final state.

**Reversibility.** Every kernel transition has an algebraic inverse. Given the current state and the byte that was applied, the predecessor state can be computed directly. This supports rollback operations and precise attribution of state changes without requiring storage of intermediate states.

**Commutator Translation.** For any two bytes, the difference between applying them in one order versus the reverse order is a state-independent translation. This is a verified property of the kernel dynamics rather than a dedicated API call. It allows reasoning about operation reordering without simulating from each possible state.

**Syndrome Checking.** A 16-element dual code provides a fast syndrome test for accidental corruption of masks. A non-zero syndrome indicates data corruption.

---

## 4. The Phenomenology Operator

The Phenomenology operator is the core of the agent's intelligence. It acts as the bridge between the continuous, high-dimensional embeddings of a language model and the discrete, finite geometry of the kernel. It is deterministic and introduces no learned parameters beyond those fixed at deployment; instead, it accumulates a trajectory-dependent field.

### Dimensional Interface

The operator accepts external numeric vectors (embeddings) of dimension D. To interface with the kernel's 256-state Horizon, D must be a multiple of 256. The number of channels per Horizon state is defined as K, such that D = 256 × K. If the external vector does not match the configured dimension, the operator raises an error.

The operator does not require that these vectors be token embeddings. Any external source that can present a deterministic vector of the appropriate dimension is admissible. This includes slices of model parameters, tool outputs, sensor data, or any other structured numeric input. Phenomenology treats all such vectors uniformly as channels over the Horizon.

A notable application of this interface is on‑demand parameter access. Model parameters can be flattened or partitioned into segments of length D and presented to Phenomenology as needed. The agent can select which parameter segment to inspect based on its current state and Genealogy, update its internal field M accordingly, and emit actions in the same loop. There is no separate offline parameter pass; reading and acting use the same mechanism.

The implementation supports K values of 1, 2, 3, 4, 6, 8, 12, and 16, corresponding to embedding dimensions of 256, 512, 768, 1024, 1536, 2048, 3072, and 4096. These align with the kernel anatomy:

| K | D | Interpretation |
|---:|---:|---|
| 1 | 256 | Global average |
| 2 | 512 | Global average per frame |
| 3 | 768 | One channel per spatial row |
| 4 | 1024 | One channel per frame-column pair |
| 6 | 1536 | One channel per frame-row |
| 8 | 2048 | Frame-rows plus frame parities |
| 12 | 3072 | One channel per component bit |
| 16 | 4096 | Bits plus parity features |

In practice, K = 3, 6, 12, and 16 are recommended when the full three‑dimensional structure of the kernel anatomy is required. Smaller K values provide more compact summaries and may be useful for lightweight or diagnostic configurations.

### The Accumulated Field

The agent maintains a dynamic field M of shape (256, K). This field represents the agent's accumulated experience along its trajectory. It initialises to zero at agent creation and evolves via an order-sensitive update rule.

For every step the agent takes, the operator updates M at the index corresponding to the current Horizon anchor. The update adds the element-wise product of the current and previous local activations, scaled by a direction factor and the learning rate η. The implementation applies numerical clipping to M to maintain stability.

### The Direction Factor

The direction factor determines whether an interaction strengthens or inhibits a pathway. It combines the normalised mask weight of the transition (popcount divided by 12) with the relationship between the vertex charges of the previous and current states.

The factor values are:
- Positive 0.5 for same-vertex transitions
- Positive 1.0 for adjacent-vertex transitions
- Negative 1.0 for opposite-vertex transitions (where the vertex charges differ by exclusive-or of 3)

A baseline term of 0.1 is added to prevent zero updates when the mask weight is zero.

This mechanism ensures that the agent naturally prefers transitions that respect the tetrahedral connectivity of the K₄ geometry, penalising abrupt jumps between opposing governance modes.

### Byte Feature Vectors

Each byte is assigned a deterministic feature vector of shape (K,). Feature vectors are derived solely from the 12-bit mask anatomy and contain no learned parameters.

For K=12, the feature vector contains the 12 mask bits mapped to the range negative one to positive one. For K=3, the vector contains three values (one per spatial row), each being the average of the four mask bits in that row. For K=6, the vector contains six values (one per frame-row combination). For K=16, the vector contains the 12 mask bits plus four parity-derived values.

### Byte Scoring and Selection

To generate output, the operator scores each of the 256 candidate bytes by combining three terms:

1. **Signal:** The dot product of (M at current Horizon plus current local activation) with the byte's feature vector
2. **Weight:** One minus the normalised mask weight, preferring lighter masks
3. **Wedge coherence:** One if the byte's vertex charge matches the current state's vertex charge, zero otherwise

The combination uses fixed coefficients: 0.5 for signal, 0.3 for weight, and 0.2 for wedge coherence.

In deterministic mode, the operator selects the byte with the highest score. In sampling mode, the operator applies softmax to the scores and samples from the resulting distribution.

---

## 5. Agent Operation

The agent operates in a continuous cycle of input processing (Egress) and output generation (Ingress). Both phases advance the kernel state and append to the Genealogy.

### Input Processing (Egress)

When the agent receives a token from an external source, it maps the token identifier to a byte using the modulo 256 operation. The kernel applies this byte to its current state using the Epistemology table. The byte is appended to the Genealogy. The agent and Phenomenology operator cache the mask and previous local activation required for the next update of the field M.

The internal kernel transition (transcription of byte to intron, expansion to mask, mutation of component A, and gyration) is specified in the Alignment Router document.

### Output Generation (Ingress)

When the agent needs to produce a token, it obtains an embedding vector for the current context from the language model. The Phenomenology operator reshapes this vector to (256, K), extracts the local activation at the current Horizon index, updates the field M, and selects the optimal output byte. The kernel applies this byte. The byte is appended to the Genealogy. The agent identifies the set of all tokens that map to this byte (all token identifiers where identifier modulo 256 equals the byte) and uses the language model's logits to select the specific token from within this valid set.

#### Router-native byte generation

The Phenomenology operator defines a complete probability distribution over the 256 byte actions at each step. The scores described in Section 4 are converted either to a deterministic choice (argmax) or to a stochastic choice (sampling from a softmax over scores). This defines a Router-native probabilistic model at the byte level.

Language generation can therefore proceed in two ways:

- Using the language model logits to choose a token within the class defined by a byte, as described above.
- Or using a fixed, invertible mapping between bytes and tokens or characters, without consulting external logits. In this case, the Phenomenology operator alone governs the stochastic language process.

### Tool Execution

Tool calls are handled as byte sequences. The identifier for a tool maps to a specific sequence of bytes. Executing a tool involves stepping the kernel through this sequence, with each byte appended to the Genealogy. The result of the tool is similarly encoded and processed as input. This ensures that all functional actions are recorded and affect the geometric state of the agent.

---

## 6. Memory and Generalisation

### The Genealogy as Memory

The Genealogy serves as the sole persistent memory store. It is a linear sequence of bytes. Because the kernel is deterministic, the kernel position can always be reconstructed by replaying the Genealogy from the Archetype. If the external model that provides embeddings is deterministic with respect to the history and its weights remain fixed, then the accumulated field M can also be reconstructed by replaying the Genealogy together with the same model.

Apart from the Genealogy, all other agent state, including the field M and the current kernel state, is either derived from or replayable from this byte log.

Given any byte in the history and the state it produced, the predecessor state can be recovered through algebraic inversion. This reversibility supports audit, rollback, and precise attribution of state changes.

The parity commitment (described in Section 3) provides efficient comparison of histories without full replay.

### Holographic Generalisation

Generalisation in this architecture arises from the holographic structure of the Horizon. Since every state decomposes into a Horizon anchor and a local variation, states that share a Horizon anchor are geometrically related. The agent treats them as variations of the same underlying configuration. This allows the agent to apply patterns learned in one specific context to other contexts that share the same boundary conditions.

### History Degeneracy

A significant feature of the kernel is provenance degeneracy: many different sequences of bytes can lead to the same final state. Under a restricted generator alphabet of size 8, words of length 6 form 262,144 possible sequences that map onto only 4,096 final states from the Archetype, with a uniform preimage size of 64.

This provides native compression. The agent does not need to store every unique path as a distinct concept. Operationally equivalent histories naturally collapse into the same geometric position, allowing the agent to generalise across different experiences that yield the same result.

### Coarse Navigation

The 16-state quotient system (derived from vertex charge pairs) provides a macro-view of position. The agent can monitor which of the four K₄ vertices each component is associated with. Transitioning between coarse regions corresponds to shifting governance modes.

---

## 7. Implementation Reference

### Dependencies

Gyroscopic ASI requires the Router substrate and a language model:

```python
from src.agent.intelligence import GyroscopicAgent, AgentConfig
from src.agent.inference import Phenomenology, PhenomenologyState
from src.agent.information import (
    horizon_index,
    vertex_charge_for_state,
    ETA_DEFAULT,
    K_MIN,
)

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("model-name")
tokenizer = AutoTokenizer.from_pretrained("model-name")
```

### Token Mapping

The interface between the language model and the kernel is a modulo operation:

```python
def token_to_byte(token_id: int) -> int:
    return token_id % 256
```

Token identifiers are mapped to bytes by reduction modulo 256. This mapping is total and deterministic. It is an interface convention between the tokeniser and the kernel and does not preserve token identity.

### Phenomenology Initialisation

```python
from src.agent.inference import Phenomenology, PhenomenologyState

phenomenology = Phenomenology(K=3, eta=0.00117)
state = PhenomenologyState.create(K=3)
```

The field M initialises to zero. The learning rate η defaults to the Aperture gap.

The `embedding_fn` passed to the agent need not be a token embedding function. It is any deterministic function that, given a context (for example a token identifier, a parameter address, or a tool state), produces a vector of dimension D = 256 × K. The agent treats this vector as the current external field for Phenomenology.

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
# Process input tokens
for token_id in input_tokens:
    byte = agent.step_input(token_id)

# Generate output
output_tokens = []
context = list(input_tokens)

for _ in range(max_output):
    embedding = get_embedding(context[-1])
    byte = agent.step_output(embedding=embedding)
    token = agent.select_token(byte, logits=get_logits(context))
    output_tokens.append(token)
    context.append(token)
```

The functions `get_embedding()` and `get_logits()` belong to the external model layer.

### Agent State

The agent's complete state consists of: the kernel instance (providing the current state index), the Genealogy (byte log), the Phenomenology field M, and a reference to the language model for token generation. No additional persistent data is required.

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

---

## 8. Performance Characteristics

The architecture is designed for constant-time operation relative to the history length.

### Time Complexity

All core operations complete in constant time per byte:

| Operation | Time |
|---|---|
| State transition | Single array lookup |
| Transcription (byte to intron) | Single exclusive-or operation |
| Token mapping | Single modulo operation |
| Genealogy append | Single write operation |
| Parity commitment update | Three exclusive-or operations |

### Space Requirements

| Component | Size |
|---|---|
| Epistemology table (shared) | 64 megabytes |
| Ontology | 256 kilobytes |
| Phenomenology constants | 3 kilobytes |
| Kernel state | 3 bytes |
| Genealogy | Linear in history length |
| Accumulated field M (K=3) | 3 kilobytes |
| Accumulated field M (K=16) | 16 kilobytes |

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

**Coherence.** The geometry reduces the likelihood of incoherent transitions. The direction factor inhibits movement between opposing governance vertices, creating a constraint that guides the agent toward trajectories that respect the K₄ connectivity.

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
| System 5 | Policy and identity | Phenomenology operator and byte selection |

---

## 11. Conclusion

Gyroscopic ASI specifies an agent architecture in which a deterministic kernel provides a reproducible coordination substrate and an append-only byte history provides the basis for replay and reconstruction. Interpretation is constrained to the agent layer through a deterministic Phenomenology operator that consumes external embeddings and kernel observables to select output bytes.

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