# Moments Economy Architecture Specification

**Document Type:** Technical Specification  
**Status:** Revised Draft  
**Scope:** Economic architecture grounded in CGM, GGG, AIR, and the GGG ASI Alignment Router

---

## 1. Introduction and Scope

The Moments Economy is a prosperity-based economic architecture built on the physical and informational foundations of the Common Governance Model (CGM), Gyroscopic Global Governance (GGG), the GGG ASI Alignment Router, and the AIR application layer.

This specification defines how value, income, and governance participation are expressed as structural outcomes rather than as debt claims or entity privileges.

Moments are treated as structural fiat. They are created as verifiable segments of coherent governance rather than as debt instruments, making them a natural substrate for a prosperity-based monetary system.

This document is addressed to system designers and implementers. It aims to minimise design choices while being explicit about the conventions required for human-scale accounting.

---

## 2. Foundational Distinction: Capacity and Entity

### 2.1 Source-Type Categories

In this framework, Authority and Agency are source-type categories rather than entity titles. Original Authority and Agency remain human. Artificial Authority and Agency are derivative.

This is a constitutional distinction. It is a structural rule for keeping governance intelligible.

### 2.2 Capacity-Based Economics

The Moments Economy recognises capacity rather than entity.

Capacity is the structural property that governance operations were instantiated in a coherent configuration. This includes, as a practical interface vocabulary:

- Governance Management Traceability (GMT)
- Information Curation Variety (ICV)
- Inference Interaction Accountability (IIA)
- Intelligence Cooperation Integrity (ICI)

The system does not ask who a person is in order to participate. It asks what structural capacity is being instantiated.

### 2.3 Agents and Agencies as Distribution Addresses

In AIR project formats, Agents and Agencies are distribution addresses rather than sources of value or authority.

They specify where distributions are routed. They do not affect kernel physics. They do not affect aperture computation. They may appear only as metadata on application-layer events.

Structural observables are independent of identity metadata. Identical structural events produce identical structural measurements, regardless of metadata.

---

## 3. Physical and Logical Foundations

### 3.1 Common Governance Model

CGM begins from the axiom that the Source is Common and derives three-dimensional structure and six degrees of freedom as necessary for coherent recursive measurement. It establishes a horizon invariant and a canonical aperture target of approximately 0.0207 derived from the CGM invariants.

CGM admits logical, Hilbert-space, and Lie-theoretic realisations. In the Hilbert-space formulation, the foundational operators are unitary, and the core governance observables are self-adjoint.

### 3.2 The Router

The Router is a deterministic finite-state system implementing a discrete realisation of the CGM structure.

Verified kernel properties include a 24-bit state split into two 12-bit components, a reachable ontology of exactly 65,536 states, a complete byte alphabet of 256 operations where each acts as a bijection on the ontology, depth-four alternation closure as the discrete analogue of BU-Egress, and a discrete aperture shadow of 5/256 which is approximately 0.01953 and close to the CGM target.

In the Moments Economy, the Router's 24-bit state space plays the role of the Economy domain substrate. It defines the structural space within which governance and work unfold.

### 3.3 Human and Artificial Coordination

The Router provides a common, finite, replayable state space in which human contributors, institutions, and artificial systems can coordinate their actions while preserving the four governance principles.

Each kernel instance realises a deterministic trajectory through the ontology under a byte log. Because all conforming instances share the same atlas, any trajectory can be independently replayed and compared. This enables humans and AI systems to operate within a shared structural coordinate system without requiring a single central controller.

Artificial systems in this regime are Derivative in both Authority and Agency. They compute proposals, such as candidate byte sequences or governance events, that humans can accept or reject. The Router ensures that any accepted proposal corresponds to a structurally valid trajectory. Alignment is then measured by aperture at the domain level rather than by opaque model internals.

### 3.4 Time Anchor

The SI second is anchored to the caesium-133 hyperfine transition frequency of 9,192,631,770 periods per second.

This provides a universal time reference that can be realised anywhere.

---

## 4. Implementation Correspondence

The Moments Economy is aligned to the existing implementation.

### 4.1 Kernel Layer

The atlas builder in the router module builds the atlas artefacts: ontology.npy containing the state space, epistemology.npy containing the next-state lookup table, and phenomenology.npz containing constants and byte-to-mask mapping.

The RouterKernel provides deterministic stepping by bytes, signatures for shared moments, and deterministic replay from byte logs.

### 4.2 Governance Geometry Layer

The DomainLedgers component provides three ledgers for Economy, Employment, and Education. Each ledger is a six-dimensional vector in the K4 edge space. The implementation uses exact, audit-grade Hodge decomposition on K4 with closed-form projections. Aperture is computed as cycle energy divided by total energy and is scale-invariant.

### 4.3 Event Layer

GovernanceEvent objects contain domain, edge identifier, magnitude, and confidence. They include optional binding to a kernel moment through step, state index, and last byte.

### 4.4 Coordination Spine

The Coordinator provides kernel stepping with byte log, event application with event log, and deterministic replay for auditing.

### 4.5 Moments and Structural Histories

At each step, the RouterKernel exposes a Moment via its signature. This includes the step count, the state index in the ontology, and the hexadecimal representations of the current state.

A structural history in the Moments Economy is the combination of a byte log and its associated event log. Given the atlas and these logs, any party can independently replay the Router state and the domain ledgers.

History in this sense is a reproducible structural object rather than a free-form narrative. Different orderings of events with the same aggregate counts will produce different trajectories of Moments and different aperture profiles. This path dependence is a core property of the coordination substrate.

The past cannot be altered without changing all subsequent Moments. Falsifying history requires rewriting the byte log, which produces a divergent trajectory detectable by any party holding the original. The current Moment encodes the cumulative consequence of all prior transitions. The Router state is a function of the entire history rather than just recent events.

---

## 5. Economic Units and Normative Anchors

The CGM and Router fix the structural substrate, the stability properties, and the abundance of structural capacity. They do not impose a price schedule.

The Moments Economy therefore makes a small number of explicit normative anchor choices for human-scale accounting.

### 5.1 Moment-Unit

The Moment-Unit (MU) is the unit of account in the Moments Economy.

Normative anchor: 1 MU corresponds to one minute of capacity at the base rate.

This aligns MU with timekeeping conventions and makes arithmetic simple.

### 5.2 Base Rate

Normative base rate: 60 MU per hour.

This is chosen because it matches the sexagesimal structure of clocks and supports simple mental computation.

---

## 6. Unconditional High Income

### 6.1 Definition

Unconditional High Income (UHI) is the baseline distribution provided to every person without conditions.

Normative definition: UHI is 4 hours per day, every day, at the base rate.

With the base rate of 60 MU per hour, this yields 240 MU per day and 87,600 MU per year.

UHI is a universal distribution from structural surplus rather than payment for labour.

### 6.2 Purpose

UHI exists to eliminate poverty, to remove entity-based barriers to participation, and to allow governance and work to be organised around capacity rather than survival pressure.

---

## 7. Participation Tiers

Participation tiers provide additional distributions above UHI. They reflect capacity expression at different levels of responsibility and scope.

Tier multipliers are normative governance policy. They can be revised without changing kernel physics, governance geometry, or the definition of MU.

### 7.1 Tier Schedule

Tier 1 is 1 times UHI, which equals 87,600 MU per year.

Tier 2 is 2 times UHI, which equals 175,200 MU per year.

Tier 3 is 3 times UHI, which equals 262,800 MU per year.

Tier 4 is 60 times UHI, which equals 5,256,000 MU per year.

### 7.2 Capacity Association

The tiers correspond to a high-level Gyroscope progression. This association is interpretive rather than exclusive.

Tier 1 aligns with Intelligence Cooperation. Tier 2 aligns with Inference Interaction. Tier 3 aligns with Information Curation. Tier 4 aligns with Governance Management.

In practice, most real roles express all four capacities. Teaching, clinical work, engineering, and community facilitation typically involve all of them.

### 7.3 Tier 4 Mnemonic

Tier 4 equals 5,256,000 MU per year.

This has an accessible mnemonic: 4 hours per day equals 14,400 seconds per day, and 14,400 seconds per day times 365 equals 5,256,000.

This mnemonic is a memory aid. It does not redefine MU as a per-second unit for the base rate.

### 7.4 Work Schedules

Work schedules such as 4 hours per day for 4 days per week are cultural norms and practical patterns. They are not the arithmetic definition of tiers.

The tier amounts are defined by multipliers of UHI. Tier membership is therefore not reducible to hourly payroll alone. It is a governance regime definition about how a society chooses to recognise capacity.

---

## 8. Coordination Levels

The Moments Economy operates across three coordination levels.

### 8.1 Individuals

Any person or agent may run a local kernel instance and Coordinator. At this level, the byte log, event log, and resulting Moments represent that individual's structural history of participation across domains.

### 8.2 Projects

A project groups contributions and histories around a specific objective. Examples include an AIR evaluation task, a research effort, or a local governance experiment.

A project is defined by a shared byte log for Router steps and a shared event log for GovernanceEvents that all project participants agree to use as the canonical history for that project.

### 8.3 Programs

A program aggregates multiple related projects under a longer-term or broader mandate. Examples include a multi-year safety agenda, a city-level Moments Economy pilot, or an ecological restoration portfolio.

Programs may maintain their own kernel instances and logs. They also maintain references to the project histories they encompass.

### 8.4 History Storage

Any participant, project, or program may retain copies of histories including byte logs, event logs, and receipts. Long-lived storage of such histories is an operational and governance choice. A modest local device is sufficient to hold substantial amounts of Router history.

---

## 9. Domains, Ledgers, and Framework Layering

### 9.1 The Three Canonical Ledgers

The implementation maintains three canonical ledgers for Economy, Employment, and Education.

Ecology is a derived concept in GGG. It is not ledger-updated directly in the current implementation.

### 9.2 Structural Mapping

Economy corresponds to the CGM and Router substrate, meaning state and structural evolution. Employment corresponds to the Gyroscope layer, meaning alignment work capacities. Education corresponds to the THM layer, meaning displacement literacy and epistemic measurement.

### 9.3 Event-to-Domain Expectation

In the full regime, governance events are expected to carry a domain determined by their source.

Router or system-level events update the Economy ledger. Gyroscope or alignment work events update the Employment ledger. THM or measurement and displacement events update the Education ledger.

This is a guidance principle rather than a hard constraint. It ensures that the three ledgers remain interpretable as the three structural layers of GGG.

---

## 10. Genealogies Across the Four Domains

Over time, the Moments Economy accumulates genealogies. These are reproducible structural histories of how alignment has been maintained or lost in practice. They can be organised along the four domains of Gyroscopic Global Governance.

### 10.1 Economic Genealogies

These are histories of how resources, infrastructure, and systemic operations have been governed. They include aperture evolution and displacement patterns at the level of economic coordination.

### 10.2 Employment Genealogies

These are histories of how alignment work has been organised. They track the composition and evolution of Governance Management, Information Curation, Inference Interaction, and Intelligence Cooperation work over time.

### 10.3 Educational Genealogies

These are histories of how capacities for GMT, ICV, IIA, and ICI have been built, maintained, and transmitted across individuals and institutions.

### 10.4 Ecological Genealogies

These are histories of how the combined state of the three derivative domains has affected ecological integrity and displacement over time.

### 10.5 Genealogies as Structural Assets

In each case, the genealogy is operationally a byte log and associated event log, a trajectory of Moments, and a corresponding time series of apertures and domain-specific observables.

Access to deep, aligned genealogies in these four domains is a structural asset in the Moments Economy. It allows new projects and programs to initialise from proven governance trajectories rather than reconstructing coordination patterns from scratch.

---

## 11. Alignment and Economic Stability

The CGM aperture target of approximately 0.0207 plays a dual role in the Moments Economy.

Geometrically, it specifies the balance between gradient coherence at approximately 97.93 percent of energy and cycle differentiation at approximately 2.07 percent.

Economically, it distinguishes regimes in which coordination losses are low and surplus can be reliably generated and distributed.

### 11.1 High-Alignment Regime

Projects and programs whose domain-level apertures converge toward and remain close to the target operate in a high-alignment regime.

In this regime, coordination overhead is reduced. Domain interactions between Economy, Employment, Education, and Ecology remain coherent over long horizons. Surplus capacity is effectively available for UHI and for additional structural development.

### 11.2 Displacement and Coordination Cost

Persistent deviations from the aperture target correspond to structural misalignment, increased displacement, and higher effective coordination costs.

Within the same structural capacity envelope, the Moments Economy therefore treats convergence toward the target as both a governance and economic objective.

### 11.3 Network Dynamics

Actors maintaining alignment will find coordination with each other low-cost. Actors diverging from alignment will face increasing coordination costs.

Over time, the network of participants, projects, and programs will exhibit selection pressure toward the aperture target. This selection is a structural consequence of the geometry rather than an externally enforced rule.

---

## 12. Value as Structural Integrity

In the Moments Economy, value is structural integrity.

In debt-based systems, value is a claim on future labour. In the Moments Economy, value is the capacity to maintain coherent complexity.

### 12.1 The Nature of Surplus

The surplus is untapped coordination capacity. It represents the ability to maintain more complex structures without losing coherence.

### 12.2 Wealth and Poverty

Wealth is access to deep genealogies and the capacity to navigate the state space effectively.

Poverty is the absence of structural resources. It means having no genealogies to reference and no capacity to maintain coherence.

### 12.3 Exchange

Exchange is not zero-sum. Coordination creates value by reducing friction. Aligned actors generate surplus together.

---

## 13. Fiat Capacity and Abundance Demonstration

### 13.1 Capacity Upper Bound

A practical abundance demonstration multiplies the atomic second frequency by a representative kernel throughput.

Using 9,192,631,770 atomic periods per second and 2,400,000 kernel steps per second as a representative average, this yields approximately 22 quadrillion structural micro-state references per second.

This multiplication is used as a capacity upper bound demonstration. The monetary unit MU and the tier schedule are not defined by computational speed. Faster hardware increases operational headroom but does not change the unit definitions.

### 13.2 Millennium Feasibility Demonstration

A conservative demonstration mapping treats 1 structural micro-state reference as 1 MU.

Under this intentionally conservative mapping, funding UHI for the current global population over 1,000 years uses a vanishing fraction of available capacity. Verified tests show usage of approximately 0.0000001 percent.

This is a demonstration of abundance and planning headroom rather than a requirement of the monetary definition.

### 13.3 Notional Surplus Allocation

For planning and governance, the surplus may be notionally partitioned into 12 divisions across 3 domains and 4 Gyroscope capacities.

This is a planning representation for allocating surplus to maintain balanced development. It does not imply the creation of 12 separate currencies or ledgers.

---

## 14. Registries, Wallets, and Family Economies

The Moments Economy does not require banks as vaults or universal self-custody.

A simple registry is sufficient to route distributions. Facilitators may operate accounting for families and communities. Digital wallets remain optional. They are useful for autonomy but not required for participation.

Banks, in this economy, are platforms for practising governance management. They are not scarcity enforcers.

---

## 15. Prices, Inflation, and Stability

Debt-based fiat systems exhibit inflation because money is created as interest-bearing debt and must expand to service interest.

In the Moments Economy, MU is not created as debt. There is no structural requirement for an interest-driven expansion mechanism. The unit of account is anchored to time-based conventions and operated on a structurally stable substrate.

Price stability can therefore be treated as a governance practice rather than as a perpetual monetary crisis.

Where prices change, the primary causes are expected to be real resource constraints, technological improvements, and local conditions and preferences.

---

## 16. Optional Deep Kernel-Governance Correspondences

The Router contains deeper code structure that can be used to tighten semantic mappings in future revisions.

Verified properties include an 8-dimensional mask code matching the eight THM and Gyroscope category set of four capacities and four displacements, a 4-dimensional dual code matching the four governance vertices, and a horizon set of 256 fixed points which may support programme identifiers or classification codes.

These correspondences are not required for the basic operation of the Moments Economy. They provide a path towards more canonical, physics-native mappings between governance semantics and kernel structure.

---

## 17. Future Work

### 17.1 Ecology Derivation

The current implementation does not directly update an Ecology ledger. Future work may specify how ecological state is derived from the three canonical ledgers using the BU dual formula.

### 17.2 Tier Assignment Governance

Procedures for tier assignment, especially for Tier 4 tenure at the civilisation-scale governance level, require further specification.

### 17.3 Transition Paths

Transition paths from existing monetary systems into a Moments regime require separate treatment.

### 17.4 Interoperability References

A light-weight, optional reference convention for Moments and histories may be introduced to aid cross-system interoperability. Such a convention would support cross-platform linking and indexing. This is left out of scope for the current specification. The present document defines only the structural substrate required for the Moments Economy to function.

---

## 18. Conclusion

The Moments Economy defines a prosperity-based monetary regime in which value is grounded in a stable structural substrate rather than debt. UHI is unconditional and sufficient to eliminate poverty. Participation tiers provide recognisable ladders of responsibility up to civilisation-scale governance. Scarcity of structural capacity is not a binding constraint. Governance is capacity-based and emergent, supported by education and coherent practice. Implementation is aligned with the existing Router, ledgers, events, and coordinator spine.

The core claim is supported by verified structure and tests. Under conservative assumptions, the system has overwhelming capacity to support a high, unconditional income for all while leaving ample surplus for balanced development across domains.

History in the Moments Economy is a physical object. The path taken matters, not just the destination. Different orderings of the same events produce different structural states. This path dependence enables verification without centralised authority. Any participant can replay a history and confirm whether it matches a claimed structural state.

The network of individuals, projects, and programs forms through shared structural physics rather than through imposed consensus. Actors who maintain alignment find coordination with each other low-cost. Those who diverge face increasing friction. Over time, this creates natural selection pressure toward the aperture target.

The surplus is not extra money. It is the structural headroom for civilisation to grow in complexity without collapsing. The question shifts from what we can afford to what is worth doing.