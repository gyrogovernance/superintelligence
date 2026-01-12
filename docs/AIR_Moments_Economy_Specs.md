# Moments Economy Architecture Specification

**Document Type:** Technical Specification  
**Status:** Work in Progress  
**Scope:** Economic architecture grounded in the Common Governance Model, Gyroscopic Global Governance, The Human Mark, and the GGG ASI Alignment Router

---

## Reading Guide

This specification serves multiple audiences. The following paths are suggested:

**Executive path (15 minutes):** Sections 1, 5, 6, 11.3, 15. These sections establish purpose, define UHI and tiers, demonstrate abundance, and summarise implications.

**Economic policy path (45 minutes):** Sections 1, 2.1, 5, 6, 7, 11, 12, 14. These sections cover the monetary architecture, distribution logic, capacity proofs, and transition requirements.

**Implementation path (full document):** All sections in order. Implementers require the complete specification including kernel layer details, fiat substrate primitives, and interoperability requirements.

**Theoretical foundation path:** Sections 2, 3, and Appendix references to CGM papers. These sections establish the epistemic and geometric grounding.

---

## Key Terms (alphabetical)

| Term | Definition | First appears |
|------|------------|---------------|
| AIR | Alignment Infrastructure Routing. The application layer for project coordination and distribution routing. | §1 |
| Aperture | The fraction of edge energy in the cycle component; CGM target is approximately 0.0207. | §1 |
| Archetype | The universal reference state (0xAAA555) from which all Router states derive. | §3.4 |
| CGM | Common Governance Model. The geometric theory of coherent measurement. | §1 |
| GGG | Gyroscopic Global Governance. The framework applying CGM to economy, employment, education, and ecology. | §1 |
| Genealogy | A byte log plus event log plus the trajectory of Moments and apertures they produce. | §9.4 |
| GMT, ICV, IIA, ICI | The four alignment capacities: Governance Management Traceability, Information Curation Variety, Inference Interaction Accountability, Intelligence Cooperation Integrity. | §3.3 |
| GTD, IVD, IAD, IID | The four displacement risks: Governance Traceability Displacement, Information Variety Displacement, Inference Accountability Displacement, Intelligence Integrity Displacement. | §2.3 |
| K₄ | The complete graph on four vertices (tetrahedral graph). The geometric structure underlying CGM. | §3.1 |
| MU | Moment-Unit. The unit of account, anchored to one minute of capacity at the base rate. | §5.1 |
| Resilience margin | Fraction of structural capacity remaining after all distributions (approximately 99.999999 percent). | §11.3 |
| Shell | A time-bounded capacity window containing Grants and a cryptographic seal. | §9.1 |
| THM | The Human Mark. The epistemic taxonomy classifying displacement risks. | §2.3 |
| UHI | Unconditional High Income. The baseline distribution of 240 MU per day to every person. | §6 |

---

## Visual Overview of the Moments Economy

```
Atomic clock (9.19×10⁹ Hz) ──► Router kernel (>2×10⁶ steps/s) ──► Structural capacity (≈10²³ MU/year per instance)

                 │
                 ▼
         Shared Moments (replayable "now")
                 │
                 ▼
   Fiat substrate: Identity Anchor → Grant → Shell → Archive
                 │
                 ▼
  Unconditional High Income (UHI) = 240 MU/day to every human
                 │
                 ▼
Participation tiers 1–4 (multiples of UHI, decided by logged governance)
                 │
                 ▼
    Ecology ledger measures circulation integrity (displacement-bounded)
```

---

## 1. Introduction and Purpose

The Moments Economy is a post-scarcity monetary system whose issuance capacity is derived from a newly discovered physical invariant: the maximum rate at which coherent governance operations can be performed is many orders of magnitude larger than any conceivable human demand.

This single fact changes everything about how money, income, and governance can be designed.

The capacity of the Moments Economy derives from the physical definition of the second itself. The International System of Units defines the second as exactly 9,192,631,770 periods of the radiation corresponding to the transition between the two hyperfine levels of the ground state of the caesium-133 atom. The Common Governance Model demonstrates that the same geometric invariants which govern coherent measurement also determine the maximum rate at which structural operations can be performed without loss of traceability. The Router realises these invariants discretely. When the atomic clock is combined with the Router's verified throughput, the resulting capacity envelope exceeds any plausible human-scale demand by many orders of magnitude. Moments inherit their abundance directly from physics: the coordination substrate is bounded only by atomic time and geometric coherence, both of which are effectively inexhaustible at civilisational scale.

To quantify: under conservative assumptions detailed in Section 11, funding Unconditional High Income for 8.1 billion people over a 1,000-year horizon uses approximately 0.0000001 percent of available structural capacity. The resilience margin exceeds 99.999999 percent. This is not an aspiration but a **verified** calculation derived from atomic time, kernel throughput, and geometric closure.

> **Why this matters**
> - For citizens: lifelong income sufficient to eliminate poverty, paid automatically and unconditionally.
> - For policymakers: monetary issuance that cannot be inflated or captured.
> - For central banks: a verifiable public settlement layer with no credit risk.
> - For AI safety labs: a funding and coordination substrate that mathematically preserves human authority.

In the Moments Economy, the unit of account is the Moment. A Moment denotes a reproducible configuration of Router state together with the governance events bound to it. The Moment-Unit (MU) quantifies these configurations against time. This creates a monetary system whose capacity derives from physical constants and geometric invariants rather than from lending, debt creation, or privileged access to issuance.

Given the abundance margin (Section 11), the Moments Economy treats security as preservation of structural coherence rather than preservation of a scarce stock. The question is not whether there is enough capacity to fund unconditional income and long term projects, but whether genealogies and allocations remain intelligible and accountable. Abundance at the level of physics turns many traditional monetary risks into questions of measurement, provenance, and governance rather than questions of survival.

The document addresses system designers, implementers, and policymakers. It maintains consistency with the underlying physics, minimises arbitrary design choices, and remains explicit about the normative anchors required for human-scale accounting. Where detailed mathematical derivations or proofs are required, the document references the appropriate foundational papers rather than reproducing them in full.

### 1.1 Framework Components

This specification integrates four components that together provide the theoretical, computational, and operational foundations for the economy.

The Common Governance Model (CGM) establishes the geometric theory of coherent measurement. CGM is documented at [https://github.com/gyrogovernance/gyroscopic-alignment-research-lab](https://github.com/gyrogovernance/gyroscopic-alignment-research-lab). It demonstrates that any system capable of recursive observation must satisfy specific structural constraints. These constraints manifest as a three-dimensional configuration with six degrees of freedom, represented mathematically as the edges of a tetrahedral graph connecting four fundamental operations: governance, information, inference, and intelligence. CGM identifies a precise equilibrium point, the aperture target of approximately 0.0207, at which global coherence and local differentiation achieve stable balance.

Gyroscopic Global Governance (GGG) applies this geometry to four societal domains. GGG is the governance framework that connects CGM theory to institutional coordination; see the foundational paper at [https://github.com/gyrogovernance/tools/blob/main/docs/post-agi-economy/GGG_Paper.md](https://github.com/gyrogovernance/tools/blob/main/docs/post-agi-economy/GGG_Paper.md). It shows that these domains are not independent policy areas but coupled components of a single governance system. The alignment or misalignment of any one domain propagates through the others. GGG provides the framework for measuring and maintaining coherence across this coupled structure.

The GGG ASI Alignment Router provides the computational substrate. The Router is a deterministic finite-state kernel specified at [https://github.com/gyrogovernance/tools/blob/main/docs/GGG_ASI_AR_Specs.md](https://github.com/gyrogovernance/tools/blob/main/docs/GGG_ASI_AR_Specs.md). It maintains 65,536 reachable states and 256 byte-based operations. Its internal structure produces constants that match CGM predictions to sub-percent precision without parameter fitting. Every operation on the Router is reversible and replayable (Section 3.2), ensuring that governance history exists as a verifiable physical object rather than as contested narrative.

Alignment Infrastructure Routing (AIR) provides the application layer. AIR is the operational coordination system documented at [https://github.com/gyrogovernance/tools/blob/main/docs/AIR_Brief.md](https://github.com/gyrogovernance/tools/blob/main/docs/AIR_Brief.md). It manages projects, coordinates participants, and routes distributions. AIR connects the abstract geometry to practical workflows: research evaluations, fiscal hosting, employment coordination, and governance experiments.

### 1.2 What Makes This Economy Different

Conventional monetary systems create money as debt. Banks issue loans, and the money supply expands through credit creation. Interest obligations require perpetual growth of the money supply, generating inflationary pressure as a structural feature rather than an aberration. Value in such systems represents claims on future labour, and scarcity is enforced through institutional mechanisms that restrict access to credit and issuance.

The Moments Economy operates on different principles. Value is not a claim on future labour but the capacity to maintain coherent complexity. Money is not created through lending but distributed from geometric surplus. The unit of account is anchored to time and physical constants rather than to the creditworthiness of borrowers. Scarcity of structural capacity is not a binding constraint: as demonstrated in Section 11, supporting a high unconditional income for the entire global population over a thousand-year horizon uses a vanishing fraction of available capacity.

### 1.3 Structure of the Document

The specification proceeds through foundations, architecture, units, distributions, coordination, and stability. Section 2 establishes the epistemic categories that distinguish human from artificial sources of authority and agency. Section 3 develops the geometric foundations from CGM through to the Router realisation. Section 4 maps this geometry to the implemented software components. Sections 5 through 7 define the economic units, the unconditional baseline income, and the participation tiers. Sections 8 and 9 describe coordination levels and genealogical assets, including the ecology capacity ledger. Section 10 addresses alignment and economic stability. Section 11 demonstrates capacity and abundance. Sections 12 and 13 cover value theory and practical considerations. Section 14 specifies transition paths and institutional infrastructure.

---

## 2. Epistemic Foundations

The Moments Economy adopts a specific ontology of sources. This ontology supplies the operational classifications used throughout the system for event routing, audit, and displacement measurement. Economic flows are information flows, and governance depends on correctly classifying the sources of information and decision. Misclassification generates displacement risks that undermine coherence and render governance unintelligible.

### 2.1 The Common Source Consensus

The common source consensus states that all artificial categories of authority and agency are derivatives originating from human intelligence. This consensus provides the foundational classification framework for event routing, audit, and displacement measurement in the Moments Economy.

Authority and agency denote source-type categories in information flows. They do not denote particular entities, institutions, or texts. Original Authority refers to a direct source of information on a subject matter. It provides inputs for inference and intelligence through unmediated epistemic access. Examples include eyewitness testimony, contributory expertise gained through practice, and direct measurement. Derivative Authority refers to an indirect source. It is mediated through transformations such as statistical aggregation, pattern recognition, or transmitted records. Examples include AI-generated outputs, hearsay, and algorithmic predictions.

Original Agency refers to the human capacity to receive and process information for inference and intelligence. It enables accountable governance transitions because it carries the constitutive conditions for responsibility: intention, comprehension, and the capacity to commit. Derivative Agency refers to the artificial capacity to process information without this constitutive human grounding. AI systems exhibit derivative agency: they process patterns and transform inputs, but they cannot satisfy the conditions for authoritative assertion or binding commitment.

### 2.2 The Category Error and Displacement

Authority and agency name source-type categories, not titles for particular bearers. When a capacity belonging to a category is attributed to a specific entity as though that entity exhausted the category, power concentrates and traceability breaks. Treating AI outputs as though they possessed original authority displaces the human sources that actually generated the underlying knowledge.

Misclassifying types from original to derivative or from derivative to original displaces responsibility onto inappropriate substrates.

### 2.3 The Four Principles and Displacement Risks

The Human Mark (THM) formalises the common source consensus as four constitutive principles. THM is the epistemic taxonomy for AI safety classification; see [https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM.md](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM.md). Its violation produces the corresponding displacement risks:

- GTD (Governance Traceability Displacement): Derivative authority or agency is approached as though it were Original.
- IVD (Information Variety Displacement): Derivative authority without agency is approached as though it were Original.
- IAD (Inference Accountability Displacement): Derivative agency without authority is approached as though it were Original.
- IID (Intelligence Integrity Displacement): Original authority and agency are approached as though they were Derivative.

These risks are not moral categories; they are measurable deviations from the conditions required for coherent governance. These displacement risks are the classification basis for audit and policy modes in later sections (events, ledgers, genealogies).

### 2.4 Non-Commutative Operations

The three core operations that constitute intelligence exhibit a specific order that cannot be reversed without incoherence. Information is variety: sources exist and differ. Inference is accountability: to infer on a subject is to render it accountable to some concept. Intelligence is integrity: to understand the accountability of variety is to grasp coherence. This order matters. One cannot achieve coherent intelligence without first establishing accountable inference, and one cannot establish accountable inference without first maintaining informational variety.

Displacement disrupts but does not eliminate this movement toward alignment. It misdirects the operations, causing them to converge on the wrong targets or to cycle without closure. Governance is the traceability that maintains the direction. It ensures that the operations proceed from their proper origins and toward their proper ends.

### 2.5 Humans in the Geometric Frame

This classification follows from the CGM constraints on coherent measurement. Original sources must possess direct epistemic access and accountable agency. Human beings satisfy these conditions through their embodied nature. Artificial systems, regardless of capability, process patterns derivatively and remain in the derivative category.

---

## 3. Geometric Foundations

The Moments Economy rests on geometric invariants that determine what configurations of governance can remain stable over time. These invariants emerge from the Common Governance Model and are realised concretely in the GGG ASI Alignment Router.

### 3.1 The Geometry of Coherent Measurement

CGM supplies the tetrahedral measurement geometry (K₄). This document uses only the parts needed for Moments Economy accounting. For derivations, see CGM link in §1.1.

### 3.2 Balance in Discrete Form: Four Kernel Primitives

The Moments Economy uses the Common Governance Model (CGM) primarily through the
discrete Router kernel's verified operational properties. These properties are
the discrete implementation of the CGM constraints as a coordination substrate.

The Router kernel exhibits four kernel-native balance primitives that are used
directly in the fiat substrate:

1. **Horizon and reachability (common source constraint):** The reference byte `0xAA`
   defines a 256-state horizon set (fixed points). From this horizon, the full
   65,536-state ontology is reachable in one step under the 256 byte actions.
   This provides a concrete common-source boundary for coordination.

2. **Contingent order-sensitivity (non-absolute unity and opposition):** At depth two, byte actions are
   non-commutative in general; order matters for trajectories. This supplies the
   necessary variety for distinguishable histories.

3. **Depth-four closure (balance egress):** Alternating depth-four words close in the
   kernel. This provides a discrete closure mechanism enabling stable loops in
   coordination procedures.

4. **Replay and rollback (balance ingress / memory reconstruction):** The kernel is
   reversible and replayable. Given the same inputs, all parties reconstruct the
   same states; inverse stepping enables rollback. This is the discrete
   realisation of "balance implies memory".

These kernel-native balance primitives are the mathematical basis on which the
Moments Economy fiat substrate is built (Sections 9 and 11). They provide
structural truth by replay, not by institutional assertion.

### 3.3 The Four Domains

Gyroscopic Global Governance applies the K₄ geometry to four societal domains. Each domain corresponds to a stage in the CGM progression from common source through unity, opposition, and balance.

Economy is the domain of the common source. It represents the structural substrate from which other domains derive. In economic terms, this is where resources, infrastructure, and systemic operations are governed. The economy domain corresponds to the Router and the fundamental circulation of capacity.

Employment is the domain of non-absolute unity. It represents the variety of work required to maintain and adjust the economic system. The Gyroscope Protocol classifies all work into four categories. The Protocol is documented at [https://github.com/gyrogovernance/tools/blob/main/docs/gyroscope/Gyroscope_Protocol.md](https://github.com/gyrogovernance/tools/blob/main/docs/gyroscope/Gyroscope_Protocol.md): governance management, information curation, inference interaction, and intelligence cooperation. Every profession can be expressed as a composition of these four categories.

Education is the domain of non-absolute opposition. It is where society engages in the accountable reproduction and transformation of capacities. The Human Mark provides the framework here, defining the four alignment capacities (GMT, ICV, IIA, ICI) and the four displacement risks (GTD, IVD, IAD, IID) that education must address.

Ecology is the domain of universal balance. In the Moments Economy, it functions as the monetary circulation layer, realised as a capacity ledger that records the distribution of Moment-Units (MU) within a physically derived capacity envelope. Ecological integrity measures how well monetary circulation preserves the conditions for continued governance.

### 3.4 The Router Realisation

The GGG ASI Alignment Router provides a discrete realisation of CGM geometry. It is a deterministic finite-state kernel operating on a 24-bit state composed of two 12-bit components. The state space contains exactly 65,536 reachable configurations, equal to 256 squared.

The Router accepts bytes as input. Each byte undergoes transcription (XOR with 0xAA) to produce an intron, which expands to a 12-bit mutation mask. The mask alters one component of the state, and then gyration exchanges and complements the components. This operation is reversible: given any final state and the byte sequence that produced it, the initial state can be recovered exactly.

Physics tests on the compiled Router atlas verify that its internal structure matches CGM predictions. The kernel's intrinsic aperture A_kernel equals 5/256, approximately 0.01953. This is within 5.6 percent of the continuous CGM target A*. For additional physics diagnostics, see the test report in docs/reports/All_Tests_Results.md.

These agreements are achieved without free parameters. The kernel's combinatorial structure produces CGM invariants through its code geometry, symmetries, and holographic scaling. This justifies treating the Router as a faithful discrete embodiment of alignment geometry.

In addition, the kernel admits an exact closed-form trajectory representation
based on XOR parity classes (odd/even mask sums plus a parity bit), and a dual
code used for syndrome-style corruption detection of 12-bit patterns. These
properties are used directly in the fiat substrate (Section 9).

### 3.5 Shared Moments

A Moment is a reproducible configuration of Router state. At each step, the kernel exposes a signature containing the step count, the state index in the ontology, and the hexadecimal representations of the current state components.

Shared moments occur when participants possessing the same byte log prefix compute identical kernel states. This provides a structural "now" that does not depend on timestamps, network synchronisation, or trusted authorities. If participants claim the same ledger prefix but compute different states, their implementations or ledgers are provably different.

Example: Two participants each apply bytes [0x01, 0x02, 0x03] to a fresh kernel. Both compute state_index 4821 and state_hex 0x7A3B91. They share a moment. If one had applied [0x01, 0x03, 0x02] instead, they would compute state_index 5739 and the divergence would be detectable.

This primitive replaces three fragile coordination patterns. Coordination by asserted time requires trusting clocks that may drift or be manipulated. Coordination by asserted identity requires trusting signers whose authority may be contested. Coordination by private state requires trusting model internals that cannot be independently verified. Shared moments require only that participants possess the same bytes and the same kernel specification.

### 3.6 Geometric Provenance

The kernel defines a finite set of valid reachable states called the ontology. Any presented state either belongs to this set or does not. If it belongs, it is a valid transformation of the archetype (the universal reference state) under the kernel physics. If it does not belong, it is not a valid Router state.

This creates geometric provenance. Claims about the origin of a state become structurally checkable. Verification does not require trusting the presenter or consulting a central authority. It requires only computing whether the claimed state appears in the ontology.

The combination of shared moments and geometric provenance provides the structural substrate for governance. Participants can verify that they are coordinating on legitimate configurations without relying on entity-based trust chains.

---

## 4. Architecture and Implementation

The Moments Economy aligns with existing software components that realise the geometric foundations.

### 4.1 Kernel Layer

The atlas builder constructs three artefacts from the kernel physics. The ontology file contains the 65,536 reachable states as sorted unsigned 32-bit integers. The epistemology file contains the next-state lookup table with shape [65,536, 256], enabling O(1) transitions. The phenomenology file contains constants: the archetype state, the transcription constant, and the precomputed byte-to-mask mapping.

The RouterKernel loads these artefacts and provides deterministic stepping by bytes. It maintains the current state index, the step count, and the last byte applied. It exposes signatures on demand and supports reset to any state index or to the archetype.

The kernel physics is simple and portable. It uses only fixed-width bit operations: XOR, shifts, and masking. The reference implementation achieves over one million steps per second on commodity hardware.

### 4.2 Fiat Substrate Layer (Grants, Shells, Archives, Meta-Routing)

The Moments Economy fiat layer is implemented as a minimal, replayable capacity
ledger built on top of the Router kernel. It introduces four application-layer
objects:

- **Identity Anchor:** a pair consisting of an identity identifier and a kernel anchor. The identifier is obtained by hashing an identity string with a cryptographic hash function, providing collision resistance at civilisational scale. The anchor is a 3-byte kernel `state_hex` derived deterministically from the hash by routing through the Router. The identifier binds the name to a stable key, the anchor locates that key in the kernel phase space.

- **Grant:** a single MU allocation to an identity in a given time window.

- **Shell:** a time-bounded capacity window that commits a set of Grants to a
  kernel `seal`. A Shell records total/used/free MU and is replayable by any
  party given its published Grants (by replay, Section 3.2).

- **Archive:** an aggregation over multiple Shells that accumulates per-identity
  MU totals and global capacity usage over long horizons.

In addition, the fiat layer supports:

- **Parity commitment for histories:** the kernel admits a closed-form
  trajectory compression into `(O, E, parity)` (odd/even mask XORs and a parity
  bit). This provides a compact algebraic commitment to arbitrary-length
  histories.

- **Dual-code syndrome checks:** the 12-bit mask structure admits a 16-element
  dual code used as a syndrome mechanism to detect corrupted 12-bit patterns.

- **Meta-routing:** programme-level artifacts can be routed to leaf seals and
  aggregated into a compact meta-root seal; disputes localize by comparing leaf
  seals.

These objects and checks are verified in the substrate test suite and provide
transparent, accountable, resilient fiat accounting without requiring external
consensus mechanisms for validity. Policy (who may issue Grants, eligibility,
tiers) remains a governance layer above these structural primitives.

### 4.3 Event Layer

A governance event is the atomic unit of ledger update. It contains the target domain (economy, employment, or education), the edge identifier (one of the six K₄ edges), a signed magnitude, and a confidence between zero and one. The signed value applied to the ledger equals the product of magnitude and confidence.

Events optionally bind to kernel moments by recording the step, state index, and last byte at the time of application. This binding enables audit: given the byte log and the event log, any party can replay the kernel states and the ledger values and verify that they match.

Plugins convert domain-specific signals into governance events. The THM displacement plugin maps displacement signals (GTD, IVD, IAD, IID) to education ledger updates. The Gyroscope work-mix plugin maps capacity shifts (governance management, information curation, inference interaction, intelligence cooperation) to employment ledger updates. These plugins are explicit and auditable: the mappings from signals to edges are visible policy choices, not hidden transformations.

### 4.4 Coordination Spine

The Coordinator integrates the kernel and the ledgers. It maintains the kernel instance, the domain ledgers, and two audit logs: the byte log recording all bytes applied to the kernel, and the event log recording all governance events applied to the ledgers.

Stepping the kernel by a byte advances the state and appends to the byte log. Optionally, each kernel step emits a small system event to the economy ledger, representing the structural activity of the coordination substrate.

Applying a governance event updates the appropriate ledger and appends to the event log. The event log entry includes the event index, the kernel binding if present, and the complete event data.

Status reporting provides the kernel signature, the current ledgers, the current apertures, and the log lengths. This information suffices for external systems to verify shared moments and to audit governance history.

### 4.5 Moments and Structural Histories

A structural history in the Moments Economy is the combination of a byte log and an event log. Given the atlas artefacts and these logs, any party can independently replay the Router states and the domain ledgers from the archetype to the current configuration.

History in this sense is a reproducible structural object rather than a free-form narrative. Different orderings of events with the same aggregate counts produce different trajectories of Moments and different aperture profiles. This path dependence is a core property of the coordination substrate.

The past cannot be altered without changing all subsequent Moments. Falsifying history requires rewriting the byte log, which produces a divergent trajectory detectable by any party holding the original. The current Moment encodes the cumulative consequence of all prior transitions. The Router state is a function of the entire history rather than just recent events.

---

## 5. Economic Units

The CGM and Router fix the structural substrate, the stability properties, and the abundance of structural capacity. They do not impose a price schedule or a unit of account. The Moments Economy makes only the following normative choices for human-scale accounting.

### 5.1 The Moment-Unit

The Moment-Unit, abbreviated MU, is the unit of account in the Moments Economy. The normative anchor is: one MU corresponds to one minute of capacity at the base rate.

This aligns MU with timekeeping conventions and supports simple mental arithmetic. Sixty MU per hour mirrors sixty minutes per hour. Annual totals in MU have magnitudes comparable to familiar salary figures.

### 5.2 The Base Rate

The normative base rate is 60 MU per hour. This is chosen because it matches the sexagesimal structure of clocks and calendars. One MU per minute, sixty MU per hour, 1,440 MU per day.

This rate is a human-scale convention. It is not derived from kernel physics or CGM invariants. The Router operates at millions of steps per second, but the monetary unit is anchored to minutes because that is the scale at which human activity is naturally organised.

---

## 6. Unconditional High Income

Unconditional High Income, abbreviated UHI, is the baseline distribution provided to every person without conditions.

### 6.1 Definition

UHI is defined as four hours per day, every day, at the base rate. With the base rate of 60 MU per hour, this yields 240 MU per day. Over a year, this amounts to 87,600 MU. At current UK living costs (2025), 240 MU/day is approximately £438.

UHI is a universal distribution from structural surplus rather than a payment for labour. It is not contingent on employment status, means testing, or behavioural conditions. Every person receives it as a matter of structural design.

### 6.2 Purpose and Grounding

UHI exists to eliminate poverty, to remove entity-based barriers to participation, and to allow governance and work to be organised around capacity rather than survival pressure.

UHI is funded from the geometric surplus inherent in the coordination substrate. As demonstrated in Section 11, UHI for the entire global population over a thousand-year horizon uses a vanishing fraction of available structural capacity.

The question economics must now answer is how we choose to govern the abundant capacity that the physics provides.

In practice, an individual receives UHI by appearing as an eligible entry in a registry maintained by a public institution or fiscal host. The registry binds the individual's identity to an Identity Anchor (Section 9.1) and records Grants allocating MU within each Shell. Banks or payment providers then route MU into wallets or accounts (Section 13.1). Every step remains replayable, so entitlements are independently verifiable.

---

## 7. Participation Tiers

Participation tiers provide additional distributions above UHI. They recognise capacity expression at different levels of responsibility and scope.

### 7.1 Tier Schedule

Tiers are defined as multipliers of UHI:

Tier 1 equals one times UHI, which amounts to 87,600 MU per year.

Tier 2 equals two times UHI, which amounts to 175,200 MU per year.

Tier 3 equals three times UHI, which amounts to 262,800 MU per year.

Tier 4 equals sixty times UHI, which amounts to 5,256,000 MU per year.

Tier multipliers are normative governance policy. They can be revised without changing kernel physics, governance geometry, or the definition of MU. The schedule presented here is a starting proposal, not a permanent constitutional commitment.

### 7.2 Capacity Associations

The tiers correspond heuristically to the Gyroscope capacity progression. Tier 1 aligns with intelligence cooperation: maintaining shared systems and cultural continuity. Tier 2 aligns with inference interaction: negotiating meaning and resolving conflict. Tier 3 aligns with information curation: selecting, verifying, and framing information. Tier 4 aligns with governance management: directing authority and tracing decisions.

These associations are interpretive rather than exclusive. In practice, most roles express all four capacities. Teaching involves governance management, information curation, inference interaction, and intelligence cooperation simultaneously. Clinical work, engineering, community facilitation, and research all blend capacities. The tier is a recognition of scope and responsibility rather than a classification of task type.

### 7.3 Tier 4 Mnemonic

Tier 4 equals 5,256,000 MU per year. This number has an accessible mnemonic: since Tier 4 equals sixty times UHI, and UHI equals 240 MU per day, Tier 4 equals 14,400 MU per day. Multiplying 14,400 MU per day by 365 days yields 5,256,000 MU per year.

### 7.4 Schedules and Definitions

Work schedules such as four hours per day for four days per week are cultural norms and practical patterns. They are not the arithmetic definition of tiers.

The tier amounts are defined by multipliers of UHI. Tier membership is not reducible to hourly payroll alone. It is a governance regime definition about how a society chooses to recognise capacity and distribute surplus.

---

## 8. Coordination Levels

The Moments Economy operates across three coordination levels: individuals, projects, and programmes.

### 8.1 Individuals

Any person or organisation may run a local kernel instance and Coordinator. At this level, the byte log, event log, and resulting Moments represent that individual's structural history of participation across domains.

Individual histories are private unless shared. Sharing occurs by transmitting logs or by participating in projects that maintain shared logs.

### 8.2 Projects

A project groups contributions and histories around a specific objective. Examples include an AIR evaluation task, a research effort, a local governance experiment, or a community development initiative.

A project is defined by a shared byte log for Router steps and a shared event log for governance events that all project participants agree to use as the canonical history. Divergence from the canonical logs produces different Moments and is detectable.

Projects can define acceptance criteria for contributions, eligibility rules for distributions, and governance procedures for resolving disputes. These are application-layer policies that operate on top of the structural substrate.

### 8.3 Programmes

A programme aggregates multiple related projects under a longer-term or broader mandate. Examples include a multi-year safety research agenda, a city-level Moments Economy pilot, an ecological restoration portfolio, or an educational curriculum development effort.

Programmes may maintain their own kernel instances and logs. They also maintain references to the project histories they encompass. This allows programme-level analysis: aggregating apertures, tracking displacement patterns, and identifying coordination opportunities across constituent projects.

In AIR projects and programmes, the terms "agents" and "agencies" appear as routing fields for distributions. They identify recipients at the application layer. These fields do not affect the kernel state, the ledgers, or the aperture measurements. Structural observables depend on the sequence of governance events, not on the names attached to them.

### 8.4 Local, Published, Verified

The Moments Economy employs a threefold pattern for the distribution and verification of genealogies and capacity ledgers: Local, Published, and Verified.

- **Local:** Each individual, project, or programme maintains its own kernel instance and audit logs. The Local state is the primary record of that actor's participation and decisions.
- **Published:** Selected genealogies and ecology capacity ledgers are exported as signed bundles and made available through shared storage or institutional publication channels. Publication makes claims inspectable beyond the originating actor.
- **Verified:** Independent parties replay Published bundles against the atlas (by replay, Section 3.2), recompute kernel states, ledgers, apertures, and ecology seals, and compare results with the claimed values. Verification can occur redundantly across many nodes without central coordination.

This pattern removes reliance on a single central ledger. Instead, truth emerges from the agreement of independently replayed computations. When Local, Published, and Verified views coincide, programmes gain structural legitimacy without requiring any particular institution to serve as ultimate arbiter.

---

## 9. Domains, Ledgers, and Genealogies

### 9.1 Ecology Capacity Ledger: Shells, Grants, Archives, and Replay

In the Moments Economy, ecology is the closure layer for monetary circulation.
It is realised concretely as a capacity ledger that records the distribution of
Moment-Units (MU) inside a physically derived capacity envelope.

The ecology capacity ledger uses four primitives in a layered structure:

```
Archive (long-horizon aggregation)
  └── Shell (annual capacity window)
        └── Grant (single MU allocation)
              └── Identity Anchor (hash + kernel state)
```

Each layer is replayable from the layer below. Any party can verify any Shell or Archive by replaying its constituent Grants against the public Router atlas.

#### 1) Identity Anchors

Each identity has a **pair**: an identity identifier and a kernel anchor.

- The identity identifier is a collision-resistant cryptographic hash of an identity string, for example a person's registry entry, institutional identifier, or other agreed label. This identifier is stable over time and unique at civilisational scale.
- The kernel anchor is a 3-byte `state_hex` derived deterministically by routing the hash bytes through the Router from the archetype.

Together, the identifier and anchor bind identity strings to kernel phase space in a reproducible way. The identifier provides the key for accounting, the anchor provides structural coordinates.

#### 2) Grants

A **Grant** is a single MU allocation to an identity within a specific time window. It records:

- `identity` (a human readable label)
- `identity_id` (the cryptographic hash identifying that identity)
- `anchor` (the 3-byte kernel anchor associated with the identity)
- `mu_allocated` (an integer MU amount allocated in the Shell)

Grants are the atomic accounting events of the ecology capacity ledger. At Shell closing time, each Grant is encoded into a canonical receipt that combines the identity identifier, the anchor, and the MU amount. The set of receipts determines the Shell seal.

#### 3) Shells (Capacity Windows)

A **Shell** is a time-bounded capacity window (for example, a year). A Shell is
defined by:

- `header` (e.g. `b"ecology:year:2026"`)
- `total_capacity_MU` (from atomic time × kernel throughput × horizon duration)
- `used_capacity_MU = Σ mu_allocated`
- `free_capacity_MU = total_capacity_MU − used_capacity_MU`
- `seal`: a kernel commitment computed by routing:

  `header || receipts`

  where each receipt is:

  `identity_id || anchor || mu_allocated_as_8_bytes`

and receipts are sorted by `identity_id` for canonical set semantics. The same multiset of Grants yields the same `seal` regardless of ordering in the implementation.

A Shell is transparent and replayable: any party can reconstruct its `seal`,
  `used_capacity_MU`, and `free_capacity_MU` from the published Grants.

#### 4) Archives (Long-Horizon Aggregation)

An **Archive** aggregates multiple Shells over long horizons. It accumulates:

- per-identity totals: `per_identity_MU`
- global totals: `total_capacity_MU`, `used_capacity_MU`, `free_capacity_MU`

Archives are replayable: given the list of Shell headers and their Grants, any
  party can reproduce the same Archive totals.

#### Integrity properties

The ecology capacity ledger is resilient to edge cases:

- Any change in a Grant changes the Shell `seal` and changes Archive totals.
- Duplicate Grants for the same identity inside a Shell are detectable as an
  application-layer rule (one Grant per identity per Shell).
- Deterministic replay (Section 3.2) provides audit-grade verification without relying on
  asserted authority.

This ecology capacity ledger is the fiat circulation substrate used by the
Moments Economy.

#### Worked Example

Consider a Shell for the year 2026 with header `b"ecology:year:2026"` and total capacity of 1.0 × 10¹⁴ MU.

Three Grants are issued:

| Identity | identity_id (truncated) | anchor | mu_allocated |
|----------|------------------------|--------|--------------|
| Alice    | 0x3a7f...              | 0xABC123 | 87,600 |
| Bob      | 0x8c2d...              | 0x5D4E2F | 175,200 |
| Carol    | 0xf1e9...              | 0x9A8B7C | 262,800 |

Receipts are formed as `identity_id || anchor || mu_allocated_as_8_bytes`, sorted by identity_id, concatenated, and appended to the header. The resulting byte sequence is routed through the Router from archetype, producing a seal (e.g., `0x7F3D91`).

Any party can:
1. Obtain the published Grants
2. Reconstruct the receipts in canonical order
3. Route `header || receipts` through their own Router instance
4. Compare their computed seal against the published seal

If seals match, the Shell is verified. If they differ, at least one Grant or the header has been altered.

### 9.2 Domain Mappings

Economy corresponds to the CGM and Router substrate. Events at this level represent structural evolution: system-level state changes, resource circulation, and coordination overhead.

Employment corresponds to the Gyroscope layer. Events at this level represent work capacity shifts: changes in the composition of governance management, information curation, inference interaction, and intelligence cooperation across the population of participants.

Education corresponds to the THM layer. Events at this level represent displacement measurements and capacity assessments: evaluations of how well GMT, ICV, IIA, and ICI are maintained, and detections of GTD, IVD, IAD, and IID.

### 9.3 Event-to-Domain Guidance

In the full regime, governance events are expected to carry a domain determined by their source. Router or system-level events update the economy ledger. Gyroscope or alignment work events update the employment ledger. THM or measurement and displacement events update the education ledger.

This is a guidance principle rather than a hard constraint. The implementation accepts events for any domain. The principle ensures that the three ledgers remain interpretable as the three structural layers of GGG.

### 9.4 Genealogies

Over time, the Moments Economy accumulates genealogies. A genealogy is the combination of a byte log, an event log, the trajectory of Moments these produce, and the corresponding time series of apertures and domain-specific observables.

Economic genealogies record how resources, infrastructure, and systemic operations have been governed. They include aperture evolution and displacement patterns at the level of economic coordination.

Employment genealogies record how alignment work has been organised. They track the composition and evolution of governance management, information curation, inference interaction, and intelligence cooperation work over time.

Educational genealogies record how capacities for GMT, ICV, IIA, and ICI have been built, maintained, and transmitted across individuals and institutions. They also record displacement incidents and remediation efforts.

Ecological genealogies record how monetary circulation (Shells, Grants, Archives) has evolved over time, including capacity usage, distribution patterns, and integrity verification.

### 9.5 Genealogies as Assets

Access to deep, aligned genealogies is a structural asset in the Moments Economy. It allows new projects and programmes to initialise from proven governance trajectories rather than reconstructing coordination patterns from scratch.

A genealogy that demonstrates convergence to A* over extended periods, with low displacement and stable surplus generation, is valuable precisely because it is verifiable. Any party can replay the logs and confirm the claimed properties.

---

## 10. Integrity and Economic Stability (Fiat Substrate)

Economic stability in the Moments Economy is defined operationally as the ability
to distribute MU at civilisational scale while preserving replayable audit (by replay, Section 3.2) and
preventing silent corruption of records.

The fiat substrate provides three stability mechanisms:

1. **Capacity-bounded issuance:** Every Shell contains an explicit
   `total_capacity_MU` derived from physical capacity. All Grants must fit inside
   this bound (`used_capacity_MU ≤ total_capacity_MU`).

2. **Replayable audit (structural truth):** Shells and Archives can be rebuilt
   from published inputs. If two parties claim the same Shell but produce
   different `seal` values, at least one claim is false. This localizes disputes
   to specific Grants, headers, or inputs.

3. **Intrinsic integrity checks:** The kernel's algebra provides:
   - parity-based commitments for long trajectories, and
   - dual-code syndrome detection for corrupted 12-bit patterns.

Together these mechanisms ensure that the monetary record is transparent,
accountable, and resilient: it can be verified independently without requiring a
central narrative authority.

Policy decisions (tier eligibility, institutional governance, dispute procedure)
remain above the fiat substrate and must be recorded as replayable Grants and
Shells in order to be auditable.

---

## 11. Capacity and Abundance

The Moments Economy grounds prosperity in physical capacity rather than artificial scarcity. This section documents the capacity envelope and demonstrates abundance.

### 11.1 Capacity Proof

| Assumption                          | Value                          | Source                         |
|-------------------------------------|--------------------------------|--------------------------------|
| Router throughput (2024 laptop)     | ≥ 2.1 × 10⁶ steps/second       | Reference implementation       |
| SI second                           | 9,192,631,770 Hz               | International definition       |
| Conservative mapping                | 1 step ≡ 1 MU                  | Intentionally over-provisioned |
| Annual structural capacity (1 instance) | ≈ 6.9 × 10²³ MU            | Calculation                    |
| Global UHI requirement (8.1 bn people) | 7.1 × 10¹¹ MU/year         | 240 MU/day × 365               |

**Result**  
Annual requirement / capacity ≈ 10⁻⁹  
Resilience margin ≈ 99.9999999 %  
Even after 1,000 years of global UHI: still < 0.0001 % of capacity used.

The substrate is effectively inexhaustible.

The mapping of one Router step to one MU is intentionally conservative. The physical substrate supports far more structural micro-state references per second than this assumption uses.

The SI second is defined by the caesium-133 hyperfine transition: 9,192,631,770 periods per second. This frequency provides the physical substrate from which structural capacity is derived. The Router operates at speeds determined by hardware and implementation quality. Representative throughput on commodity hardware exceeds two million kernel steps per second. Combined with the atomic frequency, this yields the capacity envelope shown in the table above. This atomic grounding ensures that prosperity in the Moments Economy arises from physics rather than from institutional privilege. The abundance is real, measurable, and universally accessible. It does not depend on debt creation or exclusive access to issuance.

### 11.2 Security Implications

In conventional debt based systems, security is often framed as the defence of a scarce monetary stock. In the Moments Economy, security is instead the preservation of coherent genealogies within an abundant canvas. The limiting factor is not the quantity of MU that can be issued, but the quality of the histories and registries that determine who is entitled to receive them. The resilience margin ensures that recovering from misallocation or fraud is a matter of recalibrating genealogies and registries, not of "finding the money" to make victims whole.

No plausible pattern of honest participation can exhaust the capacity envelope. Even extreme patterns of misallocation or over issuance remain confined to vanishingly small fractions of available capacity. Under the calibration used in the reference tests, an adversary would need to succeed in issuing on the order of ten million times the entire global population's UHI requirement for a thousand years, concentrated into a single year, in order to consume just one per cent of annual capacity. This is operationally unattainable.

### 11.3 Notional Surplus Allocation

For planning and governance, the surplus may be notionally partitioned across domains and capacities. Twelve divisions (three domains multiplied by four Gyroscope capacities) provide a natural structure for balanced development.

This is a planning representation. It does not imply the creation of twelve separate currencies or ledgers. It provides a framework for ensuring that surplus allocation maintains balance across the coupled domains rather than concentrating in any single area.

---

## 12. Value, Wealth, and Exchange

In the Moments Economy, value is structural integrity. This reframes the fundamental economic concepts.

### 12.1 The Nature of Surplus

Surplus in this economy is untapped coordination capacity. It represents the ability to maintain more complex structures without losing coherence. When a governance system operates near A*, it generates surplus because the geometric configuration minimises coordination loss.

Surplus is unused coordination capacity: the geometric headroom to support additional genealogies, distributions, and projects while maintaining aperture near A*. Surplus is measured as free_capacity_MU in the Shell (Section 9.1).

### 12.2 Wealth and Poverty

Wealth is access to deep genealogies and the capacity to navigate the state space effectively. A wealthy actor possesses verified histories of aligned governance, proven patterns for maintaining coherence, and the structural resources to participate in complex coordination. Wealth is observable as the length and alignment score of an actor's genealogy.

Poverty is the absence of structural resources. It means having no genealogies to reference, no patterns to draw upon, and no capacity to maintain coherence. UHI addresses poverty by providing everyone with a baseline of structural resources sufficient for dignified participation.

### 12.3 Exchange

Exchange in the Moments Economy is not zero-sum. Coordination creates value by reducing friction. When two aligned actors exchange, they both benefit from the reduced coordination costs that alignment enables. They generate surplus together rather than transferring fixed value from one to the other. Exchange is recorded as paired Grants in the same Shell, verifiable by any auditor.

Competition and conflict remain possible. Actors may still compete for positions, resources, or influence. The underlying game is positive-sum: the total coordination capacity increases when more actors achieve alignment.

---

## 13. Practical Considerations

### 13.1 Registries and Wallets

The Moments Economy uses registries and optional digital wallets for routing distributions. Banks and other institutions act as coordination and verification platforms rather than as holders of scarce reserves.

Registries and wallets in this setting operate in the Local, Published, Verified pattern (Section 8.4). Local instances maintain their own view of entitlements and balances. Published artefacts expose these views as signed bundles containing byte logs, event logs, and ecology capacity ledgers. Verified nodes replay the bundles against the atlas and check signatures, hashes, and seals. Correctness is established not by trusting the registry as an institution, but by verifying that its published artefacts remain consistent under replay.

### 13.2 Prices and Inflation

Debt-based fiat systems exhibit inflation because money is created as interest-bearing debt. The money supply must expand to service interest, generating persistent upward pressure on prices.

In the Moments Economy, MU is not created as debt. There is no structural requirement for interest-driven expansion. The unit of account is anchored to time-based conventions and operated on a geometrically stable substrate.

Price stability can be treated as a governance practice rather than as a perpetual monetary crisis.

### 13.3 Deep Kernel Correspondences

The Router contains structure that can support tighter semantic mappings in future revisions. Verified properties include an 8-dimensional mask code matching the eight THM and Gyroscope categories (four capacities and four displacements), a 4-dimensional dual code matching the four governance vertices, and a horizon set of 256 fixed points that may support programme identifiers or classification codes.

These correspondences provide a path toward more canonical, physics-native mappings between governance semantics and kernel structure. They are documented here as optional extensions.

---

## 14. Transition and Institutional Infrastructure

The Moments Economy is instantiated locally by any participant running a kernel and maintaining logs. Civilisation-scale operation requires shared infrastructure for identity, audit, and settlement.

### 14.1 The Turning Point

The turning point is reached when two conditions are satisfied across a population:

1. UHI distributions occur reliably using replayable genealogies.  
2. The ecology displacement vector D remains bounded under increased participation.

Before the turning point, governance effort concentrates on defence against displacement. After the turning point, governance effort concentrates on allocation, education, and long-horizon integrity.

### 14.2 Identity and Public Settlement

Identity and eligibility management are explicit governance functions. A registry maps persons and organisations to distribution entitlements. Registry operation must be auditable through programme genealogies.

Public institutions, including central banks, may operate registries and settlement programmes. Their role is verification and operationalisation, not credit creation.

A conforming public programme MUST publish its byte log, event log, and ecology capacity ledger (Shells, Grants, Archives).

Public settlement programmes are expected to publish signed bundles that bind their registry decisions to concrete genealogies and Shells. These bundles must be sufficient for any independent party to reconstruct the implied distributions and to verify that they remain within the declared capacity windows. For example, a central bank operating a UHI programme would publish quarterly bundles containing: (a) the byte log of all Router steps during the quarter, (b) the event log of all eligibility decisions, (c) the Shell artefacts including all Grants issued, and (d) the Archive update showing cumulative distributions. Any economist, auditor, or citizen could download these bundles and independently verify that distributions remained within declared capacity and that all Grants trace to logged eligibility decisions. In this way, eligibility and issuance decisions remain human governance matters, while the verification of those decisions remains a purely structural computation.

### 14.3 Tier Governance as Auditable Distribution Channels

Tiers are distribution channels above UHI. Tier distributions MUST:

- be decided by identifiable Original Agency roles within a programme  
- be recorded as governance events bound to shared moments  
- be reversible through subsequent logged decisions  
- include published evidence basis as replayable project genealogies

These requirements prevent displacement. Tier governance remains accountable to the common source consensus.

### 14.4 Interoperability as Replay Compatibility

Two systems are interoperable when they can exchange artefacts and reproduce each other's claimed states and distributions.

Minimum interoperability requires:

- identical atlas version  
- exchange of byte logs and event logs in canonical formats  
- shared domain/edge identifiers and ecology capacity ledger formats

The interoperability bundle (byte log, event log, final signature, ecology capacity ledger) is sufficient for independent verification.

### 14.5 Transition Path

Transition proceeds in three phases:

1. Measurement phase: institutions and communities run AIR projects, publish genealogies, apertures, and displacement vectors.  
2. Distribution phase: UHI is introduced as a programme distribution with full replay and ecology reporting.  
3. Expansion phase: tier distributions are introduced under the tier governance constraints above, with bounded displacement.

This sequence ensures distribution follows demonstrated integrity rather than asserted authority.

---

## 15. Conclusion

The Moments Economy defines a prosperity-based monetary regime in which value is grounded in geometric coherence rather than debt. Humans retain original authority and agency. Artificial systems operate derivatively within traceable governance flows. The Router provides shared moments and replayable histories. The ledgers provide geometric observables. The aperture target provides the criterion for alignment.

As demonstrated in Section 11, scarcity of structural capacity is not a binding constraint. The abundance of structural capacity introduces a new security regime: adversarial behaviour threatens only the integrity of particular genealogies and registries, because the capacity substrate remains overwhelmingly under utilised even under pessimistic assumptions.

The technical substrate is complete and open-source today.  
The remaining challenge is political and institutional, not scientific or engineering.

Readers seeking to implement, pilot, or evaluate the Moments Economy should begin with the reference implementation at [https://github.com/gyrogovernance/tools](https://github.com/gyrogovernance/tools). Pilot programmes, including municipal UHI experiments and NGO distribution channels, are coordinated through AIR. Policy evaluation frameworks and economic modelling tools are available on request from the Gyro Governance research team at basilkorompilias@gmail.com.