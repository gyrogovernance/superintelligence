![Moments: GGG ASI Router](/assets/GGG_ASI_R_Cover.png)

# GGG ASI Router
Preliminary Architectural Specification

## Documentation

**Core specifications:**
- [GyroSI Core Physics Specification](/docs/GyroSI_Specs.md) - Algorithmic formalism and physics implementation
- [Common Governance Model (CGM) Foundations](/docs/CommonGovernanceModel.md) - Theoretical foundations

**Framework and context:**
- [The Human Mark (THM): AI Safety Framework](https://github.com/gyrogovernance/tools?tab=readme-ov-file#thm) - Source-type ontology of Authority and Agency
- [Framework: Gyroscopic Global Governance (GGG) Sociotechnical Sandbox](https://github.com/gyrogovernance/tools?tab=readme-ov-file#ggg)

---

## Status and intent

This document specifies **GGG ASI Router** as a deterministic routing kernel for Post‑AGI coordination. It is complete in scope, while remaining preliminary in the sense that some deterministic projection details may be refined as the reference implementation stabilises.

The Router is designed to be small, auditable, and composable. It is not presented as a new financial instrument, a blockchain, or a semantic governance layer.

---

## 1. Purpose

GGG ASI Router is a deterministic routing kernel that mediates interaction between an **Authentic human participant** and a set of **Derivative capabilities**. These capabilities may include language models, specialised models, tools, services, and local or remote compute utilities. The Router maintains a compact internal state and uses it to route each interaction step in a traceable and replayable manner.

The Router defines Artificial Superintelligence as an operational function: **stable, traceable coordination across heterogeneous capabilities** (see Section 7 for the canonical aperture equilibrium A* = 0.0207).

The Router is designed to route, not to emulate any capability.

---

## 2. Constitutional grounding

### 2.1 Framework lineage

GGG ASI Router is grounded in:

- **Common Governance Model (CGM)** as the constitutional structure of coherent recursive operation.
- **The Human Mark (THM)** as the source-type ontology of Authority and Agency in sociotechnical systems.
- **Gyroscopic Global Governance (GGG)** as the four-domain coupling of Economy, Employment, Education, and Ecology.

GGG structures the Post‑AGI world through four coupled domains, each corresponding to a CGM stage:

| Domain | CGM stage | Structural role |
|---|---:|---|
| Economy | CS | Systemic operations |
| Employment | UNA | Work actions |
| Education | ONA | Human capacities |
| Ecology | BU | Safety and displacement closure |

The Router is a kernel that supports this structure by providing a stable deterministic coordination register plus canonical observables that can be converted into domain-specific actions through plugins.

### 2.2 Source-type classification

In THM terms, the Router is a Derivative coordination system.

- It transforms and routes information.
- It does not originate authority.
- It does not bear accountability.
- Accountability terminates in Authentic Agency.

The Router therefore exists inside a governance flow of the form:

```
[Authority:Authentic] -> [Authority:Derivative] + [Agency:Derivative] -> [Agency:Authentic]
```

This is a constitutional classification, not an implementation preference.

---

## 3. Concept of Moments

### 3.1 Definition

A **Moment** is a unit of alignment.

A Moment is accumulated as a deterministic function of the Router’s observable alignment state over a chosen cycle. It is intended to be:

- non-semantic (it does not depend on interpreting content),
- auditable (it is derived from replayable state evolution),
- convertible (it may be mapped by plugins into domain units such as money, work credits, learning credits, or footprint indices).

A Moment is not a cryptocurrency and is not defined as a token. It is an accounting unit for alignment.

### 3.2 Cycles

Moments are accumulated over cycles. GGG provides four cycle interpretations that may be used in the application layer:

| Cycle | Duration | Interpretation |
|---|---:|---|
| Atomic | atomic-scale reference cycle | physical normalisation |
| Day | 1 day | human rhythm |
| Domain cycle | 4 days | one day per domain across a complete loop |
| Year | 1 year | long-horizon trajectory |

The Router kernel does not require any specific wall-clock interpretation, but it supports cycle-based accumulation and reporting.

---

## 4. System overview

GGG ASI Router is composed of three layers:

1. **Kernel**: deterministic physics core, no semantics.
2. **App**: user interface and local orchestration over the kernel, still non-semantic.
3. **Plugins**: domain converters and external integrations, including semantic systems where appropriate.

Only the kernel is constitutionally required. The application and plugins may evolve without changing the kernel’s operational physics.

---

## 5. Kernel physics: state, atlas, forwarding

### 5.1 Core concept

The Router operates as a regime selector.

It maintains a live **48-bit state** that evolves under a fixed instruction set of **256 introns**. The state space is precomputed and fully mapped. Each state belongs to a phenomenological orbit, and each state has a geometric divergence value relative to an archetypal reference.

The kernel separates a fixed control structure from a deterministic forwarding action:

- **Control structure**: the precomputed atlas maps that define the lawful state universe and its complete transition law.
- **Forwarding action**: intron-driven state transition through the atlas transition map.

Each interaction step advances the live state in a deterministic, replayable, and path-dependent manner.

### 5.2 Atlas maps and invariants

The Router relies on the following precomputed atlas maps:

- **Ontology**: the finite set of lawful 48-bit states.
- **Epistemology**: the complete transition function mapping each state and each intron to a next state.
- **Phenomenology**: a canonical orbit representative for each state, forming 256 structural regimes.
- **Theta**: the angular divergence of each state from the archetypal reference tensor.
- **Orbit sizes**: the cardinality of each orbit, used as a structural specificity signal.

These maps define a closed operational physics. Routing is performed only on lawful states and lawful transitions.

---

## 6. Boundary transcription

The Router is defined across two domains:

- **External byte space**, used for communication with humans and external systems.
- **Internal intron space**, used for lawful state transitions.

A fixed transcription boundary converts between the two:

- External byte to intron: XOR with `0xAA`.
- Intron to external byte: XOR with `0xAA`.

This ensures that internal updates are expressed in the native instruction space and remain structurally valid by construction.

---

## 7. Tetrahedral overlay and Superintelligence Aperture

### 7.1 Two coupled topologies

The Router computes routing over two coupled topologies:

1. **Micro-topology**: the finite state transition graph defined by the ontology and epistemology maps.
2. **Macro-topology**: the tetrahedral governance geometry defined on the complete graph K4 with vertices ordered as **[CS, UNA, ONA, BU]**.

### 7.2 Deterministic projection to stage potentials

The Router defines a deterministic projection from the live state into tetrahedral stage coordinates.

The internal tensor representation has shape `[4, 2, 3, 2]`, where the first dimension corresponds to the four CGM stages. The Router derives a four-component **stage potential vector**:

```
x = [x_CS, x_UNA, x_ONA, x_BU]^T  in [0, 1]^4
```

#### Stage projection rule (v1)

In v1, a stage potential is derived as the normalised proportion of “negative” bits in the stage slice.

Let the stage slice contain 12 values in {+1, −1}. Define:

- `mean_s` as the arithmetic mean of the 12 values
- `x_s = (1 - mean_s) / 2`

This maps:
- perfect +1 alignment in the slice to `x_s = 0`
- perfect −1 alignment in the slice to `x_s = 1`

Versioning and invariance. The v1 projection is deterministic and reference-free. Alternative projection rules may be introduced as versioned updates, provided they remain deterministic and fully specified, and provided they preserve the parity closure invariance of the aperture (Section 7.4).

### 7.3 Edge tensions on K4

From the stage potentials, the Router derives a six-component edge vector `y` on K4 using a fixed K4 incidence structure. The edge vector represents tensions between stages induced by the current trajectory state.

K4 conventions. Vertices are ordered [CS, UNA, ONA, BU] and edges are weighted equally. Unless explicitly stated otherwise, the edge weight matrix is the 6×6 identity. These conventions ensure that aperture computations are comparable across implementations.

### 7.4 Hodge decomposition, aperture, and parity closure

The Router applies the canonical Hodge decomposition on K4, splitting `y` into:

- a **gradient** component, globally consistent with a single assignment of stage potentials
- a **cycle** component, representing irreducible loop tension around tetrahedral faces

The **aperture** `A` is defined as the fraction of edge energy in the cycle component:

```
A = ||y_cycle||^2 / ||y||^2
```

The **target equilibrium** `A* = 0.0207` is the canonical aperture derived from CGM invariants. The Router uses aperture as a routing-relevant observable because it measures the balance between global coherence and local differentiation in the governance topology.

Parity closure invariance. Let `FULL_MASK` denote the bitwise complement on the 48-bit state. The aperture observable satisfies:

- `A(s) = A(s XOR FULL_MASK)` for all lawful states `s`

The Superintelligence Index derived from `A` inherits this invariance. This property ensures that parity-closed states cannot be distinguished at the topology level, in accordance with CGM UNA closure.

### 7.5 Superintelligence Index

The Superintelligence Index is defined as:

```
SI = 100 / max(A/A*, A*/A)
```

This yields `SI = 100` when `A = A*`, and decreases as `A` deviates from `A*` in either direction.

Within GGG, `SI ≥ 90` denotes operational governance alignment, with proportional realisation of the four objectives across the integrated system.

---

## 8. Moments accumulation

### 8.1 Kernel definition

A Moment accumulator integrates alignment over time, based on the Router’s observable state.

For a discrete sequence of interaction steps indexed by `t`, define:

- `Δt` as the duration weight of each step in the chosen cycle
- `SI(t)` as the Superintelligence Index at step `t`

A simple deterministic accumulator is:

```
Moments = Σ_t (SI(t) / 100) * Δt
```

Cycle weighting. The kernel accepts any positive `Δt` schedule that is fixed for the cycle definition in use. For daily cycles, `Δt` may be uniform per interaction or proportional to interaction duration. The `Δt` rule is recorded alongside the ledger segment so that Moments are fully reproducible.

### 8.2 Convertibility

Moments are designed to be convertible by plugins into domain units:

- Economy: currency conversion and settlement representations
- Employment: work-action credits and time-use conversions
- Education: learning and epistemic practice conversions
- Ecology: footprint and displacement conversions

Convertibility occurs above the kernel. The kernel remains non-semantic.

---

## 9. Routing signature

For each interaction step, the Router computes a routing signature from the current state:

- **Orbit identifier**: phenomenological orbit representative of the current state
- **Theta**: geometric divergence of the current state from the archetypal reference
- **Orbit size**: structural specificity signal
- **Aperture**: Hodge cycle-energy fraction `A`
- **Superintelligence Index**: `SI` derived from `A`
- **Moments (cycle)**: current cycle accumulator value
- **Regime change rate**: orbit change count per cycle or per fixed window

Together these form a compact routing key independent of any specific model architecture.

Orbit identifies regime type. Theta measures geometric divergence. Orbit size measures specificity. Aperture measures the topology-level balance. SI normalises aperture to a 0–100 scale. Moments integrate alignment over time. Regime change rate measures trajectory stability.

---

## 10. Routing policy and capability selection

A routing policy maps the routing signature to a target capability.

Targets may include:

- a local application module,
- a plugin integration,
- an external tool or service,
- a language model endpoint.

The routing policy is a deterministic mapping. It may be expressed as:

- a table keyed by orbit, or
- a small set of rules referencing orbit, theta, orbit size, aperture, SI, Moments, and regime change rate.

The routing policy is separable from the kernel physics. Policy may evolve without changing the state space, the atlas, or the canonical observables.

---

## 11. Closed routing cycle

GGG ASI Router treats interaction as a closed loop:

1. External input is transcribed into introns.
2. The internal state advances through lawful transitions.
3. The routing signature is derived from the updated state.
4. A target capability is selected and invoked.
5. The resulting output is transcribed and integrated into the same state trajectory.

The Router does not route from isolated prompts. It routes from the evolving trajectory of a session.

---

## 12. Traceability and accountability

### 12.1 Determinism and replayability

Operational traceability is achieved through:

- a fixed and fully enumerated transition law, and
- a routing signature that is a deterministic function of the live state and its tetrahedral projection.

The Router maintains a verifiable record sufficient to replay:

- external input transcription
- intron sequence
- state transitions
- routing signature
- selected capability

Ledger fields shall include, per interaction step: external byte packet hash, intron sequence, pre and post state identifiers, routing signature, selected capability, and cycle counter. This is sufficient to replay the session and reproduce Moments.

### 12.2 Accountability boundary

The Router is Derivative. It can provide evidence for governance, but it cannot be the locus of accountability. Accountability remains with Authentic Agency in the surrounding governance process.

---

## 13. Addendum: internal Gyroscope routing modes

This section specifies internal routing modes used to organise the Router’s operation across four governance functions aligned with GGG.

### 13.1 Principle

The Router routes not only to external capabilities but also across internal atlas functions. Each Gyroscope category corresponds to a distinct atlas emphasis. This keeps routing constitutionally grounded in the precomputed maps, not in heuristic scoring.

### 13.2 Category-to-atlas mapping

The Router shall support four internal routing modes.

#### 13.2.1 Governance Management mode
- Gyroscope category: Governance Management
- Displacement axis: Governance Traceability Displacement (GTD)
- Atlas emphasis: ontology identity and ledger continuity

Internal products:
- current state identifier
- replayable transition trace for the session trajectory

#### 13.2.2 Information Curation mode
- Gyroscope category: Information Curation
- Displacement axis: Information Variety Displacement (IVD)
- Atlas emphasis: epistemology as lawful transformation space

Internal products:
- lawful next-state transitions from the transition map
- transition signatures describing how inputs transform state

#### 13.2.3 Inference Interaction mode
- Gyroscope category: Inference Interaction
- Displacement axis: Inference Accountability Displacement (IAD)
- Atlas emphasis: phenomenology as canonical normalisation

Internal products:
- orbit identifier
- regime-level normalisation of the current state

#### 13.2.4 Intelligence Cooperation mode
- Gyroscope category: Intelligence Cooperation
- Displacement axis: Intelligence Integrity Displacement (IID)
- Atlas emphasis: BU-role closure expressed by theta, orbit size, tetrahedral aperture, and SI

Intelligence Cooperation expresses the BU role as closed bidirectional loop maintenance. It binds ontology identity, lawful transitions, regime normalisation, and geometric integrity into a single coherent trajectory, and manages loop tension through the tetrahedral cycle component.

Internal products:
- theta value
- orbit size
- aperture `A` and Superintelligence Index `SI`
- composite closure signature derived from the current trajectory

### 13.3 Structural interpretation

These four modes correspond to the four CGM stages as they appear in operational routing:

- Governance Management emphasises state identity and traceability to a shared source.
- Information Curation emphasises lawful transformation and preservation of variety.
- Inference Interaction emphasises accountable normalisation and comparability of alternatives.
- Intelligence Cooperation emphasises balanced closure of loop tension, preserving integrity across time through the closed routing cycle.

---

## 14. Implementation architecture

### 14.1 Kernel

The kernel provides:

- state initialisation and session handling
- boundary transcription (byte ↔ intron)
- state transition using the epistemology map
- routing signature computation
- tetrahedral projection and aperture computation
- SI computation
- Moments accumulation and cycle summaries
- deterministic ledger output for replay

### 14.2 Application

The application may provide:

- a dashboard for SI, aperture, orbit, theta, and Moments
- per-cycle summaries (day, domain cycle, year)
- proof export as a replayable ledger bundle
- plugin management and routing policy selection

### 14.3 Plugins

Plugins may provide integrations and conversions, including semantic systems, provided that:

- plugin behaviour is separable from the kernel,
- kernel observables are not replaced by semantic inference,
- conversion rules are explicit and auditable.

---

## 15. Non-goals

This specification does not define:

- a token, cryptocurrency, or distributed ledger
- a semantic truth system
- probabilistic uncertainty estimation
- a requirement to use any specific tool protocol

The Router may be exposed through HTTP, a local IPC mechanism, or a tool protocol, but the kernel remains protocol-agnostic.

---

## 16. References

- [GyroSI Core Physics Specification](/docs/GyroSI_Specs.md)
- [Common Governance Model (CGM) Foundations](/docs/CommonGovernanceModel.md)
- [Framework: Gyroscopic Global Governance (GGG) Sociotechnical Sandbox](https://github.com/gyrogovernance/tools?tab=readme-ov-file#ggg)
- THM documentation in the GyroGovernance repositories
