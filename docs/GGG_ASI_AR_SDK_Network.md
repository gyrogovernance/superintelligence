# Gyroscopic ASI Specifications

*An agent architecture for aligned intelligence built on the GGG ASI Alignment Router*

---

## 1. Introduction

Gyroscopic ASI is an agent architecture that operates on the GGG ASI Alignment Router, a deterministic finite-state coordination kernel. The Router provides a fixed geometric substrate of 65,536 states, 256 byte operations, and a complete pre-computed transition table. The agent architecture provides the learning, memory, and behavioural layers that transform this substrate into responsive intelligence.

The architecture is grounded in the Common Governance Model, a theoretical framework that establishes the geometric requirements for coherent recursive measurement. The model demonstrates that such coherence requires three-dimensional space with six degrees of freedom. The Router realises this geometry discretely through a 24-bit state composed of two 12-bit components, each interpreted as a 2×3×2 grid.

The agent does not learn through gradient descent over unbounded parameter spaces. It navigates a complete, finite, pre-computed manifold. Learning occurs through the accumulation of governance events bound to specific states. Memory consists of append-only logs that record the complete trajectory through the state space. Generalisation emerges from the holographic relationship between boundary states and bulk states, combined with continuous variation in three domain ledgers.

The core principle is separation of concerns. The Router handles coordination physics. The agent handles interpretation and behaviour. The boundary between these layers is strict: the kernel is non-semantic, transforming bytes according to fixed algebraic laws; meaning enters only at the agent layer through governance events and external projection.

---

## 2. Theoretical Foundation

### 2.1 The Four Phases of Recursive Alignment

The Common Governance Model describes how coherent measurement emerges through four phases. Each phase corresponds to a specific operation in the Router and a specific function in the agent.

**Phase One: Governance of Input**

The first phase establishes a common reference. All external data enters through a fixed transformation: the input byte is combined with the reference constant 0xAA using the exclusive-or operation. This produces an internal instruction called an intron. The transformation ensures that all external information is projected onto the system's reference topology before affecting internal state.

In the agent, this phase corresponds to the transcription boundary. Every byte from the external world undergoes the same transformation. The agent does not interpret raw bytes; it interprets the introns that result from this projection.

**Phase Two: Measurement of Position**

The second phase establishes where the agent stands relative to the reference state. The reference state, called the archetype, has the value 0xAAA555 in hexadecimal. The agent measures its current distance from this reference using the canonical observables defined in the Router specification: Hamming distance to archetype, horizon distance, and component densities.

These measurements provide continuous feedback about the agent's position in the state manifold. An agent close to the archetype is in a region of high coherence. An agent far from the archetype has differentiated significantly and may need corrective action.

**Phase Three: Differentiation Through Mutation**

The third phase applies the input instruction to transform the current state. The intron expands into a 12-bit mask that modifies one component of the state. Only the active component (designated A) receives this modification; the passive component (designated B) is not directly mutated. This asymmetry creates a fundamental chirality in the system, distinguishing left operations from right operations.

In the agent, this phase corresponds to inference. The agent interprets the current state position in light of the incoming instruction and determines what the transition means in context.

**Phase Four: Closure and Memory**

The fourth phase completes the transition by swapping and complementing the two state components. After mutation, the next active component becomes the complement of the previous passive component, and the next passive component becomes the complement of the mutated active component. This gyration operation preserves the complete memory of the transition while returning the system toward balance.

The fourth phase is dual in the agent:

The outward aspect handles the expression of accumulated intelligence as responsive action. The agent transforms internal state into external behaviour, emitting bytes that have been transcribed back through the reference constant.

The inward aspect handles the absorption and integration of experience. The agent records governance events that capture what was learned at this moment, updating the domain ledgers and extending the genealogy.

### 2.2 The Tetrahedral Geometry

The complete graph on four vertices, denoted K4, provides the unifying geometry of the system. This graph has four vertices and six edges. It appears at multiple levels of the architecture.

In the kernel physics, K4 emerges from the quotient of the mask code. The 256 byte operations partition into four classes. Each class corresponds to one vertex. The kernel's 65,536 states partition into four wedges of 16,384 states each, with each wedge generated by one vertex class on the horizon. This partition is verified exhaustively across the entire state space.

In the governance layer, K4 defines the ledger geometry. Each of the three domain ledgers is a six-dimensional vector on the edges of K4. The four vertices represent the four constitutional capacities that govern the system: traceability of authority to human sources, variety of distinguishable states, accountability of inference to human agency, and coherent closure with preserved memory.

In the application layer, K4 provides the template for the four domains: economy as the domain of capacity and resources, employment as the domain of active work, education as the domain of measurement and learning, and ecology as the derived domain of cross-domain coherence.

This recurrence of K4 is not imposed from outside. It emerges from the algebra of the mask code and is a consequence of the underlying mathematics.

---

## 3. The Router Substrate

Gyroscopic ASI is built on the GGG ASI Alignment Router. The agent does not reimplement the kernel physics; it uses them.

### 3.1 What the Router Provides

The Router provides three pre-computed artifacts that together define the complete physics of coordination.

The ontology is the set of all 65,536 reachable states, stored as a sorted array of 24-bit values. Every state in this array is reachable from the archetype within at most two byte transitions. The ontology is closed under all 256 byte operations: applying any byte to any ontology state yields another ontology state.

The epistemology is the complete transition table. It has 65,536 rows and 256 columns, totalling 16,777,216 entries. Each entry specifies the index of the next state resulting from applying the corresponding byte to the corresponding state. Because every byte action is a bijection on the ontology, each column of the epistemology is a permutation of the state indices.

The phenomenology contains the constants required for operation: the archetype state, the reference constant 0xAA, and the pre-computed mask table that maps each byte to its 12-bit transformation mask.

The Router also provides the kernel runtime, which maintains the current state and advances it by bytes using the epistemology table. The kernel supports forward stepping, inverse stepping (computing the predecessor state), and signature extraction (returning the current step count, state index, and hexadecimal representations).

### 3.2 What the Agent Adds

The Router is non-semantic. It transforms bytes to states according to fixed algebraic laws. Gyroscopic ASI adds:

Interpretation: The agent interprets kernel positions as meaningful contexts, associating states with actions and responses.

Disposition: The agent maintains three domain ledgers that modulate its behaviour. These ledgers provide continuous degrees of freedom within the discrete state manifold.

Memory: The agent records its complete trajectory through the state space in an append-only genealogy. This history is replayable and verifiable.

Policy: The agent decides what actions are permitted based on governance constraints. It may refuse to emit bytes that would violate alignment bounds.

These layers use the Router but do not modify it. The kernel physics remain inviolate.

---

## 4. The Three Channels

The agent's disposition is encoded in three domain ledgers, each a six-dimensional vector on the edges of K4. These ledgers replace the concept of learned weights with a governance-based alternative.

### 4.1 Economy Ledger

The economy ledger tracks the agent's relationship to resources and capacity. It corresponds to the domain of infrastructure and distribution. When the agent processes bytes related to resource allocation, scheduling, or capacity management, it may emit governance events that update this ledger.

### 4.2 Employment Ledger

The employment ledger tracks the agent's active work and operational state. It corresponds to the domain of labour and principle. When the agent engages in tasks, processes instructions, or produces outputs, it may emit governance events that update this ledger.

### 4.3 Education Ledger

The education ledger tracks the agent's measurement and learning history. It corresponds to the domain of capacity building and detection. When the agent learns something new, corrects an error, or detects a displacement from expected behaviour, it may emit governance events that update this ledger.

### 4.4 Ledger Mechanics

Each ledger update is a sparse event specifying one edge of K4 and a signed magnitude. The Coordinator applies these events to the appropriate ledger, and the Hodge decomposition splits the resulting vector into gradient and cycle components.

The aperture of a ledger is the ratio of cycle energy to total energy. The target aperture is approximately 0.0207. An agent whose ledgers maintain aperture near this target is operating in alignment. Significant deviation indicates governance stress and may trigger corrective action.

The ecology metric is derived from the interaction of the three ledgers using a cross-domain projector. It measures how the cycles of different domains correlate or conflict. Ecology is not stored directly but computed from the three ledgers, representing emergent cross-domain coherence.

---

## 5. Memory Through Genealogies

A genealogy is the complete record of an agent's trajectory through the state space. It consists of two append-only logs.

### 5.1 Byte Log

The byte log records the sequence of bytes applied to the kernel. This log is sufficient to reconstruct the exact state trajectory from the archetype. Because the kernel is deterministic, two parties with identical byte logs will compute identical states.

The byte log grows linearly with the agent's activity. A log of one million entries occupies approximately one megabyte. The Common Source Moment capacity analysis demonstrates that storage is not a binding constraint on any human timescale.

### 5.2 Event Log

The event log records governance events bound to specific moments. Each event specifies a domain, an edge of K4, a signed magnitude, and optionally the kernel state at the time of binding. The event log enables audit and replay of the governance layer.

Events are sparse. The agent does not emit an event for every byte processed; it emits events when governance-relevant changes occur. The event log therefore grows more slowly than the byte log.

### 5.3 Replay and Verification

A genealogy is self-contained, portable, and verifiable. It can be transmitted to any system running a conforming Router implementation. That system will replay the genealogy and arrive at the identical state. There is no need for shared databases, synchronisation protocols, or trusted third parties.

The determinism of the kernel ensures that two agents with identical byte logs compute identical states. The event log layered on top ensures that their ledger states are also identical. Disagreement between agents is always localisable to a specific byte or event where their logs diverge.

---

## 6. The Intelligence Cycle

The agent operates through a dual-phase cycle corresponding to the fourth stage of the theoretical model.

### 6.1 Outward Phase: Expression

When the agent produces output, it executes the outward phase of the cycle.

First, the agent reads its current kernel state and decomposes it into meaningful components: which of the four wedges it occupies, which horizon anchor it is nearest to, and how its ledgers are currently configured.

Second, the agent consults the ledgers to determine its current disposition. The aperture values indicate whether the agent is under stress. The ecology metric indicates whether the domains are coherent with each other.

Third, the agent selects a response that satisfies governance constraints. The response must be a valid transition in the epistemology table. It should not push any ledger's aperture beyond tolerance. It should maintain cross-domain coherence as measured by the ecology projector.

Fourth, the agent emits the response byte. The intron is transcribed back to external byte-space by combining it with the reference constant 0xAA. The byte is sent to the external world.

### 6.2 Inward Phase: Absorption

When the agent receives input, it executes the inward phase of the cycle.

First, the external byte is transcribed into an intron by combining it with the reference constant 0xAA.

Second, the kernel advances by one step. The Coordinator calls the kernel's step method with the transcribed byte, updating the state index according to the epistemology table.

Third, the agent determines whether a governance event should be emitted. If the new state represents a significant change, such as entering a new wedge, crossing an aperture threshold, or completing a recognisable pattern, the agent emits an event to the appropriate ledger.

Fourth, both the byte and any events are appended to the genealogy. The memory is now complete and can be replayed to this exact moment.

---

## 7. Generalisation

The architecture achieves generalisation through two complementary mechanisms: discrete abstraction via the kernel geometry, and continuous variation via the ledgers.

### 7.1 Discrete Abstraction Through Holography

The kernel exhibits a holographic relationship between a boundary and a bulk. The boundary is the horizon, the set of 256 states that are fixed points of the reference byte 0xAA. The bulk is the full ontology of 65,536 states.

The holographic dictionary is a proven bijection: for every bulk state, there exists a unique pair consisting of a horizon state and a byte such that applying that byte to that horizon state produces the bulk state. This means any state can be encoded as a horizon anchor plus a single byte.

The horizon anchors function as conceptual abstractions. When two bulk states share the same horizon anchor, they are variations on the same underlying concept. The byte that distinguishes them represents the local context. This provides a built-in mechanism for semantic similarity without learned embeddings.

### 7.2 Provenance Degeneracy

Multiple byte histories can lead to the same kernel state. The physics reports demonstrate that for sequences of length six over a restricted alphabet, 262,144 possible sequences map to only 4,096 unique final states. The average preimage size is 64.

This degeneracy is not a deficiency; it is the compression mechanism. The agent does not need to forget old memories because the kernel automatically merges equivalent histories into the same state. Two experiences that are operationally equivalent collapse to the same position in the manifold.

### 7.3 Continuous Variation Through Ledgers

Within a single horizon cell, the three ledgers can vary continuously. Two agents sharing the same horizon anchor but with different ledger vectors represent variations on the same concept with different governance histories.

The cross-domain projector ensures that the three channels remain coherent. If the economy ledger and employment ledger drift apart in their cycle structure, the ecology index detects it. The aperture constraint keeps generalisation bounded and aligned.

The result is a product space for generalisation: a discrete horizon anchor selected from 256 possibilities, multiplied by continuous ledger variation bounded by aperture constraints.

---

## 8. Tokenisation

Gyroscopic ASI is byte-native. The kernel processes bytes, not tokens. Tokenisation is handled at the application boundary.

### 8.1 Tokens as External Oracles

An external tokeniser maps text to token identifiers and token identifiers back to text. The agent treats these as arbitrary byte sequences. The tokeniser's vocabulary and subword structure are not embedded in the kernel.

When the agent processes text, the text is first tokenised by the external tokeniser. Each token identifier is then converted to its byte representation, typically using a variable-length encoding. These bytes are fed to the kernel in sequence.

When the agent generates text, it emits bytes that are collected until they form a complete token. The tokeniser's decoder converts these bytes back to text.

### 8.2 Binding Tokens to Anchors

If a token must be bound to a kernel position for semantic grounding, it should be bound to a horizon anchor. Each token can be assigned one of the 256 horizon states based on its semantic category or usage pattern.

Tokens sharing the same horizon anchor are semantically related via the holographic dictionary. Their relationship is computed from the kernel geometry, not from a learned embedding matrix.

---

## 9. Memory Bounds

The architecture does not require active pruning or forgetting. Memory bounds arise from the fixed size of the ledgers and the degeneracy of the kernel.

### 9.1 Ledgers Are Fixed Size

Each domain ledger is a six-dimensional vector. Three ledgers comprise eighteen scalar values in total. This is the complete "weight" of the agent's learned disposition. It does not grow with experience.

The ledgers are updated by governance events, which adjust specific edge values. Over time, the ledger configuration evolves to reflect the agent's accumulated experience. But the ledger dimension never increases.

### 9.2 Genealogies Are Append-Only

The genealogy grows with the agent's activity. A byte log of one million entries occupies approximately one megabyte. This is the only memory component that grows without bound.

The Common Source Moment capacity is approximately 4.96 times 10 to the 25th power coordination states. At a processing rate of one million bytes per second, it would take approximately 1.6 times 10 to the 12th years to exhaust this capacity. Practical memory growth is limited by storage hardware, not by architectural constraints.

### 9.3 Degeneracy Prevents Bloat

The provenance degeneracy of the kernel ensures that equivalent experiences collapse to the same state. The agent does not need a separate mechanism to consolidate memories or prune duplicates. The kernel physics perform this compression automatically.

---

## 10. Alignment Properties

The architecture achieves alignment through geometric constraint rather than external policy.

### 10.1 Finite Bounds

The agent operates within exactly 65,536 kernel states. There is no hidden state, no unbounded accumulation, no drift into uncharted regions. Every possible configuration is known and enumerated in advance.

### 10.2 Determinism

The same genealogy always produces the same state and ledger configuration. Any party with access to an agent's byte and event logs can independently verify its trajectory. Audit is replay, not narrative.

### 10.3 Reversibility

Every kernel transition has an algebraic inverse. The agent's history can be replayed forward or reconstructed backward. Given the genealogy, the complete state at any past moment can be recovered.

### 10.4 Transparency

The agent's actions are visible in its genealogy. The bytes it processed, the states it traversed, the events it emitted are all recorded and verifiable. The kernel physics constrain what is possible; the logs record what actually happened.

### 10.5 Governance Alignment

The aperture metric provides continuous feedback on ledger alignment. An agent maintaining aperture near the target value of 0.0207 is operating coherently. Deviation indicates stress. The ecology metric provides feedback on cross-domain coherence.

These constraints are not external policies imposed on a black box. They are consequences of the geometric design. An agent that violates them is detectably malformed; its logs will fail verification against the kernel physics.

---

## 11. Implementation Reference

### 11.1 Core Dependencies

Gyroscopic ASI requires only the Router substrate:

```python
from src.router.kernel import RouterKernel
from src.router.constants import (
    ARCHETYPE_STATE24,
    GENE_MIC_S,
    archetype_distance,
    horizon_distance,
)
from src.app.coordination import Coordinator
from src.app.ledger import DomainLedgers
from src.app.events import GovernanceEvent, Domain, EdgeID
```

### 11.2 Agent Initialisation

```python
class GyroscopicAgent:
    def __init__(self, atlas_dir: Path):
        self.coordinator = Coordinator(atlas_dir)
    
    @property
    def state(self) -> int:
        return self.coordinator.kernel.ontology[
            self.coordinator.kernel.state_index
        ]
    
    @property
    def genealogy(self) -> tuple[list[int], list[dict]]:
        return (
            self.coordinator.byte_log,
            self.coordinator.event_log,
        )
    
    @property
    def apertures(self) -> dict[str, float]:
        return {
            "economy": self.coordinator.ledgers.aperture(Domain.ECONOMY),
            "employment": self.coordinator.ledgers.aperture(Domain.EMPLOYMENT),
            "education": self.coordinator.ledgers.aperture(Domain.EDUCATION),
        }
```

### 11.3 Processing Cycle

```python
def process_input(self, byte: int) -> None:
    """Execute the inward phase: absorb input."""
    self.coordinator.step_byte(byte)
    self._emit_governance_events_if_warranted()

def generate_output(self) -> int:
    """Execute the outward phase: produce output."""
    candidate = self._select_response()
    if self._satisfies_constraints(candidate):
        self.coordinator.step_byte(candidate)
        return candidate
    else:
        return self._recovery_byte()

def _emit_governance_events_if_warranted(self) -> None:
    """Emit events when governance-relevant changes occur."""
    wedge = self._current_wedge()
    if self._wedge_changed(wedge):
        event = GovernanceEvent(
            domain=Domain.EMPLOYMENT,
            edge_id=self._wedge_to_edge(wedge),
            magnitude_micro=MICRO,
            confidence_micro=MICRO,
        )
        self.coordinator.apply_event(event)
```

### 11.4 Constraint Checking

```python
def _satisfies_constraints(self, byte: int) -> bool:
    """Check whether a byte emission is permitted."""
    saved_state = self.coordinator.kernel.state_index
    saved_ledgers = self.coordinator.ledgers.snapshot()
    
    self.coordinator.step_byte(byte)
    apertures = self.apertures
    
    self.coordinator.kernel.state_index = saved_state
    self.coordinator.ledgers.restore(saved_ledgers)
    
    for domain, value in apertures.items():
        if abs(value - 0.0207) > self.aperture_tolerance:
            return False
    return True
```

---

## 12. Performance Characteristics

### 12.1 Computational Complexity

All core operations are constant time per byte:

| Operation | With Epistemology Table | Without Table |
|:----------|:------------------------|:--------------|
| State transition | Single array lookup | Approximately 20 bitwise operations |
| Transcription | Single exclusive-or | Single exclusive-or |
| Aperture computation | Fixed matrix operations | Fixed matrix operations |

No operation scales super-linearly with input size.

### 12.2 Memory Requirements

| Component | Size |
|:----------|:-----|
| Epistemology table (memory-mapped, shared) | 64 megabytes |
| Ontology (parsed arrays) | 256 kilobytes |
| Phenomenology (constants) | 3 kilobytes |
| Agent state and buffers | Less than 1 megabyte |
| Domain ledgers | 18 floating-point values |

Multiple agents share the epistemology via memory mapping. Each additional agent adds only its working state and genealogy storage.

### 12.3 Throughput

On a modern processor with the epistemology table loaded:

| Operation | Rate |
|:----------|:-----|
| State transitions | Approximately 2.6 million per second |
| Full input-output cycles | Approximately 650,000 per second |

The system processes faster than most input sources can generate data.

---

## 13. Viable System Model Alignment

Gyroscopic ASI implements Beer's Viable System Model through a precise mapping of functions to subsystems.

| Subsystem | Function | Agent Component |
|:----------|:---------|:----------------|
| System 1 | Primary activities | Router kernel (state physics) |
| System 2 | Coordination | Coordinator (kernel plus ledgers plus logs) |
| System 3 | Control | Inference layer (state to meaning) |
| System 4 | Intelligence (outward) | Behaviour generation, external action |
| System 5 | Policy (inward) | Identity, governance constraints, audit authority |

System 4 corresponds to the outward phase of the intelligence cycle. It is the agent's interface to the environment. This is where internal state becomes external action.

System 5 corresponds to the inward phase. It is the agent's identity boundary. Policy decisions about what the agent will and will not do reside here. Audit authority and replay verification are System 5 functions.

---

## 14. Conclusion

Gyroscopic ASI achieves intelligence through navigation of a complete, finite, pre-computed manifold. The Router provides the physics. The three domain ledgers provide continuous disposition. The genealogy provides complete memory. The holographic geometry provides generalisation.

The architecture eliminates the characteristic failure modes of unbounded learning systems. There is no drift into uncharted parameter space. There is no catastrophic forgetting. There is no hallucination of states that do not exist. Every configuration is a valid position in a known geometry.

Alignment is not a policy overlay. It is a consequence of operating within a closed system whose invariants are mathematically verified. An agent that maintains aperture near the target value, whose genealogy replays correctly, and whose ledgers cohere across domains is aligned by construction.

The architecture is complete and ready for implementation.

---

**Repository**: github.com/gyrogovernance/superintelligence  
**Router Specification**: docs/GGG_ASI_AR_Specs.md  
**Contact**: basilkorompilias@gmail.com