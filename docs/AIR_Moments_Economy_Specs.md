# Moments Economy Architecture Specification

**Document Type:** Technical Specification  
**Status:** Work in Progress  
**Scope:** Economic architecture grounded in the Common Governance Model, Gyroscopic Global Governance, The Human Mark, and the GGG ASI Alignment Router

---

## 1. Introduction and Purpose

The Moments Economy is a prosperity-based economic architecture that grounds value, income, and governance in geometric invariants rather than debt or artificial scarcity. It defines how societies can distribute surplus capacity unconditionally while preserving the conditions for coherent governance across all scales of human activity.

This specification integrates four components that together provide the theoretical, computational, and operational foundations for the economy.

The Common Governance Model (CGM) establishes the geometric theory of coherent measurement. It demonstrates that any system capable of recursive observation must satisfy specific structural constraints. These constraints manifest as a three-dimensional configuration with six degrees of freedom, represented mathematically as the edges of a tetrahedral graph connecting four fundamental operations: governance, information, inference, and intelligence. CGM identifies a precise equilibrium point, the aperture target of approximately 0.0207, at which global coherence and local differentiation achieve stable balance.

Gyroscopic Global Governance (GGG) applies this geometry to four societal domains: economy, employment, education, and ecology. It shows that these domains are not independent policy areas but coupled components of a single governance system. The alignment or misalignment of any one domain propagates through the others. GGG provides the framework for measuring and maintaining coherence across this coupled structure.

The GGG ASI Alignment Router provides the computational substrate. It is a deterministic finite-state kernel that realises CGM geometry in discrete form. The Router maintains 65,536 reachable states and 256 byte-based operations. Its internal structure produces constants that match CGM predictions to sub-percent precision without parameter fitting. Every operation on the Router is reversible and replayable, ensuring that governance history exists as a verifiable physical object rather than as contested narrative.

Alignment Infrastructure Routing (AIR) provides the application layer. It manages projects, coordinates participants, and routes distributions. AIR connects the abstract geometry to practical workflows: research evaluations, fiscal hosting, employment coordination, and governance experiments.

The capacity of the Moments Economy derives from the physical definition of the second itself. The International System of Units defines the second as exactly 9,192,631,770 periods of the radiation corresponding to the transition between the two hyperfine levels of the ground state of the caesium-133 atom. The Common Governance Model demonstrates that the same geometric invariants which govern coherent measurement also determine the maximum rate at which structural operations can be performed without loss of traceability. The Router realises these invariants discretely. When the atomic clock is combined with the Router's verified throughput, the resulting capacity envelope exceeds any plausible human-scale demand by many orders of magnitude. Moments inherit their abundance directly from physics: the coordination substrate is bounded only by atomic time and geometric coherence, both of which are effectively inexhaustible at civilisational scale.

In the Moments Economy, the unit of account is the Moment. A Moment denotes a reproducible configuration of Router state together with the governance events bound to it. The Moment-Unit (MU) quantifies these configurations against time. This creates a monetary system whose capacity derives from physical constants and geometric invariants rather than from lending, debt creation, or privileged access to issuance.

The document addresses system designers, implementers, and policymakers. It maintains consistency with the underlying physics, minimises arbitrary design choices, and remains explicit about the normative anchors required for human-scale accounting. Where detailed mathematical derivations or proofs are required, the document references the appropriate foundational papers rather than reproducing them in full.

### 1.1 What Makes This Economy Different

Conventional monetary systems create money as debt. Banks issue loans, and the money supply expands through credit creation. Interest obligations require perpetual growth of the money supply, generating inflationary pressure as a structural feature rather than an aberration. Value in such systems represents claims on future labour, and scarcity is enforced through institutional mechanisms that restrict access to credit and issuance.

The Moments Economy operates on different principles. Value is not a claim on future labour but the capacity to maintain coherent complexity. Money is not created through lending but distributed from geometric surplus. The unit of account is anchored to time and physical constants rather than to the creditworthiness of borrowers. Scarcity of structural capacity is not a binding constraint: verified calculations show that supporting a high unconditional income for the entire global population over a thousand-year horizon uses a vanishing fraction of available capacity.

### 1.2 Structure of the Document

The specification proceeds through foundations, architecture, units, distributions, coordination, and stability. Section 2 establishes the epistemic categories that distinguish human from artificial sources of authority and agency. Section 3 develops the geometric foundations from CGM through to the Router realisation. Section 4 maps this geometry to the implemented software components. Sections 5 through 7 define the economic units, the unconditional baseline income, and the participation tiers. Sections 8 and 9 describe coordination levels and genealogical assets, including the ecology ledger derivation. Section 10 addresses alignment and economic stability. Section 11 demonstrates capacity and abundance. Sections 12 and 13 cover value theory and practical considerations. Section 14 specifies transition paths and institutional infrastructure.

---

## 2. Epistemic Foundations

The Moments Economy adopts a specific ontology of sources. This ontology supplies the operational classifications used throughout the system for event routing, audit, and displacement measurement. Economic flows are information flows, and governance depends on correctly classifying the sources of information and decision. Misclassification generates displacement risks that undermine coherence and render governance unintelligible.

### 2.1 The Common Source Consensus

All artificial categories of authority and agency are derivatives originating from human intelligence. This statement, termed the common source consensus, constitutes the foundational axiom for reliable governance in any system that includes artificial components.

Authority and agency denote source-type categories in information flows. They do not denote particular entities, institutions, or texts. Original Authority refers to a direct source of information on a subject matter. It provides inputs for inference and intelligence through unmediated epistemic access. Examples include eyewitness testimony, contributory expertise gained through practice, direct measurement, and firsthand observation. Derivative Authority refers to an indirect source. It is mediated through transformations such as statistical aggregation, pattern recognition, report compilation, or transmitted records. Examples include AI-generated outputs, hearsay, literature reviews, and algorithmic predictions.

Original Agency refers to the human capacity to receive and process information for inference and intelligence. It enables accountable governance transitions because it carries the constitutive conditions for responsibility: intention, comprehension, and the capacity to commit. Derivative Agency refers to the artificial capacity to process information without this constitutive human grounding. AI systems exhibit derivative agency: they transform inputs into outputs through pattern-matching and optimisation, but they cannot satisfy the conditions for authoritative assertion or binding commitment.

When authority and agency manifest artificially, they remain constructed classifications that trace to human sources. The informational content of any AI system derives from direct observation, measurement, and recording that humans performed and processed into training data. The operational capacity of any AI system derives from design, specification, and objectives that humans codified into action. Enhanced capability represents more sophisticated transformation of these inputs rather than a change in source type. No artificial category exists independently of the human intelligence that provides its substance and validity.

### 2.2 The Category Error and Displacement

Authority and agency name source-type categories, not titles for particular bearers. When a capacity belonging to a category is attributed to a specific entity as though that entity exhausted the category, power concentrates and traceability breaks. This misattribution can occur in multiple directions. Treating AI outputs as though they possessed original authority displaces the human sources that actually generated the underlying knowledge. Treating particular humans, institutions, or texts as though they exhausted the category of original authority concentrates power that should remain distributed across all bearers of that category.

Misclassifying types from original to derivative or from derivative to original displaces responsibility onto inappropriate substrates. Legal frameworks recognise this distinction implicitly. Responsibility must follow actual capacity. Authority cannot be exercised without corresponding accountability. The doctrine of ultra vires prevents entities from acting beyond their constitutive powers. These principles extend naturally to systems that include artificial components.

### 2.3 The Four Principles and Displacement Risks

The Human Mark formalises the common source consensus as four constitutive principles whose violation produces the corresponding displacement risks: Governance Traceability Displacement (GTD), Information Variety Displacement (IVD), Inference Accountability Displacement (IAD), and Intelligence Integrity Displacement (IID). These risks are not moral categories; they are measurable deviations from the conditions required for coherent governance.

### 2.4 Non-Commutative Operations

The three core operations that constitute intelligence exhibit a specific order that cannot be reversed without incoherence. Information is variety: sources exist and differ. Inference is accountability: to infer on a subject is to render it accountable to some concept. Intelligence is integrity: to understand the accountability of variety is to grasp coherence. This order matters. One cannot achieve coherent intelligence without first establishing accountable inference, and one cannot establish accountable inference without first maintaining informational variety.

Displacement disrupts but does not eliminate this movement toward alignment. It misdirects the operations, causing them to converge on the wrong targets or to cycle without closure. Governance is the traceability that maintains the direction. It ensures that the operations proceed from their proper origins and toward their proper ends.

### 2.5 Humans in the Geometric Frame

This classification follows from the CGM constraints on coherent measurement. Original sources must possess direct epistemic access and accountable agency. Human beings satisfy these conditions through their embodied nature. Artificial systems, regardless of capability, process patterns derivatively and remain in the derivative category.

---

## 3. Geometric Foundations

The Moments Economy rests on geometric invariants that determine what configurations of governance can remain stable over time. These invariants emerge from the Common Governance Model and are realised concretely in the GGG ASI Alignment Router.

### 3.1 The Geometry of Coherent Measurement

CGM begins from the axiom that the source is common. This means that all operations within a governance system must trace to a shared origin that provides the reference for coherence. From this axiom, together with requirements for continuous physical implementation, CGM derives a specific geometric structure.

The structure has three dimensions with six degrees of freedom. These dimensions correspond to the edges of a complete graph on four vertices, known mathematically as K₄ or the tetrahedral graph. The four vertices represent the four fundamental operations: governance, information, inference, and intelligence. The six edges represent the pairwise relationships among these operations.

Any configuration of a governance system can be represented as a vector in this six-dimensional edge space. The vector encodes the tensions, alignments, and couplings among the four operations. Different configurations correspond to different ways of organising governance, and not all configurations are equally stable.

### 3.2 Aperture: Gradient-Cycle Decomposition and Alignment Target

Every governance configuration is a six-dimensional edge vector on K₄. Hodge decomposition splits it uniquely into a gradient component (globally coherent tensions) and a cycle component (non-integrable local circulation). Aperture A is the exact ratio of cycle energy to total energy. CGM proves that stable recursive measurement requires A ≈ 0.0207 (A*), where 97.93 % of energy is gradient and 2.07 % is cycle. Scalar evaluations collapse this geometry into a single number and accordingly cannot distinguish states near A* from states far from A*. Single-axis optimisation is geometrically locked at A = 0.5 and can never reach A*. The Moments Economy uses the full six-dimensional epistemic representation; alignment is a structural configuration to be maintained, not a scalar to be maximised.

### 3.3 The Four Domains

Gyroscopic Global Governance applies the K₄ geometry to four societal domains. Each domain corresponds to a stage in the CGM progression from common source through unity, opposition, and balance.

Economy is the domain of the common source. It represents the structural substrate from which other domains derive. In economic terms, this is where resources, infrastructure, and systemic operations are governed. The economy domain corresponds to the Router and the fundamental circulation of capacity.

Employment is the domain of non-absolute unity. It represents the variety of work required to maintain and adjust the economic system. The Gyroscope Protocol classifies all work into four categories: governance management, information curation, inference interaction, and intelligence cooperation. Every profession can be expressed as a composition of these four categories.

Education is the domain of non-absolute opposition. It is where society engages in the accountable reproduction and transformation of capacities. The Human Mark provides the framework here, defining the four alignment capacities (GMT, ICV, IIA, ICI) and the four displacement risks (GTD, IVD, IAD, IID) that education must address.

Ecology is the domain of universal balance. It functions as the structural closure of the governance system. The distinct operations of economy, employment, and education accumulate into a single material reality in the ecological domain. Ecological integrity measures how well the combined state of the three derivative domains preserves the conditions for continued governance.

### 3.4 The Router Realisation

The GGG ASI Alignment Router provides a discrete realisation of CGM geometry. It is a deterministic finite-state kernel operating on a 24-bit state composed of two 12-bit components. The state space contains exactly 65,536 reachable configurations, equal to 256 squared.

The Router accepts bytes as input. Each byte undergoes transcription (XOR with 0xAA) to produce an intron, which expands to a 12-bit mutation mask. The mask alters one component of the state, and then gyration exchanges and complements the components. This operation is reversible: given any final state and the byte sequence that produced it, the initial state can be recovered exactly.

Physics tests on the compiled Router atlas verify that its internal structure matches CGM predictions. The kernel's intrinsic aperture A_kernel equals 5/256, approximately 0.01953. This is within 5.6 percent of the continuous CGM target A*. The same discrete geometry reconstructs the BU monodromy defect δ_BU to 0.06 percent accuracy, the aperture scale m_a to 0.2 percent accuracy, the quantum gravity constant Q_G = 4π to 0.35 percent accuracy, and the fine-structure constant α to 0.02 percent accuracy at the sixth significant digit.

These agreements are achieved without free parameters. The kernel's combinatorial structure produces CGM invariants through its code geometry, symmetries, and holographic scaling. This justifies treating the Router as a faithful discrete embodiment of alignment geometry.

### 3.5 Shared Moments

A Moment is a reproducible configuration of Router state. At each step, the kernel exposes a signature containing the step count, the state index in the ontology, and the hexadecimal representations of the current state components.

Shared moments occur when participants possessing the same byte log prefix compute identical kernel states. This provides a structural "now" that does not depend on timestamps, network synchronisation, or trusted authorities. If participants claim the same ledger prefix but compute different states, their implementations or ledgers are provably different.

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

### 4.2 Governance Geometry Layer

The DomainLedgers component maintains three ledgers corresponding to economy, employment, and education. Each ledger is a six-dimensional vector in the K₄ edge space. Updates occur through governance events that specify a domain, an edge identifier, a magnitude, and a confidence.

The Hodge decomposition uses exact closed-form projections. For K₄ with unit weights, the gradient projection equals one-quarter of the product of the transposed incidence matrix with itself. The cycle projection is the identity minus this gradient projection. These forms are exact rational matrices that produce identical results across all platforms and numerical libraries.

Aperture is computed as the squared norm of the cycle component divided by the squared norm of the full vector. When the vector is zero, aperture is defined as zero. The computation is scale-invariant: multiplying all ledger entries by any positive constant leaves aperture unchanged.

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

UHI is defined as four hours per day, every day, at the base rate. With the base rate of 60 MU per hour, this yields 240 MU per day. Over a year, this amounts to 87,600 MU.

UHI is a universal distribution from structural surplus rather than a payment for labour. It is not contingent on employment status, means testing, or behavioural conditions. Every person receives it as a matter of structural design.

### 6.2 Purpose and Grounding

UHI exists to eliminate poverty, to remove entity-based barriers to participation, and to allow governance and work to be organised around capacity rather than survival pressure.

UHI is funded from the geometric surplus inherent in the coordination substrate. The abundance demonstrations in Section 11 show that UHI for the entire global population over a thousand-year horizon uses a vanishing fraction of available structural capacity.

The question economics must now answer is how we choose to govern the abundant capacity that the physics provides.

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

Tier 4 equals 5,256,000 MU per year. This number has an accessible mnemonic: four hours per day equals 14,400 seconds per day, and 14,400 seconds per day multiplied by 365 days equals 5,256,000.

This mnemonic is a memory aid for the magnitude of Tier 4. It does not redefine MU as a per-second unit or alter the base rate definition.

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

A project is defined by a shared byte log for Router steps and a shared event log for governance events that all project participants agree to use as the canonical history. Divergence from the canonical logs produces different Moments and is detectable through replay.

Projects can define acceptance criteria for contributions, eligibility rules for distributions, and governance procedures for resolving disputes. These are application-layer policies that operate on top of the structural substrate.

### 8.3 Programmes

A programme aggregates multiple related projects under a longer-term or broader mandate. Examples include a multi-year safety research agenda, a city-level Moments Economy pilot, an ecological restoration portfolio, or an educational curriculum development effort.

Programmes may maintain their own kernel instances and logs. They also maintain references to the project histories they encompass. This allows programme-level analysis: aggregating apertures, tracking displacement patterns, and identifying coordination opportunities across constituent projects.

In AIR projects and programmes, the terms "agents" and "agencies" appear as routing fields for distributions. They identify recipients at the application layer. These fields do not affect the kernel state, the ledgers, or the aperture measurements. Structural observables depend on the sequence of governance events, not on the names attached to them.

---

## 9. Domains, Ledgers, and Genealogies

### 9.1 Ecology Ledger: BU Closure and Monetary Circulation

In Gyroscopic Global Governance, ecology is the BU stage. It is the closure layer that integrates the three derivative domains into a single coherent profile. In the Moments Economy, this closure layer is the monetary circulation layer.

The ecology ledger is derived deterministically at every shared moment from the three canonical ledgers using the BU dual formula. It is not updated directly by events.

Let x_Econ, x_Emp, x_Edu be the vertex profiles recovered from y_Econ, y_Emp, y_Edu using the exact K₄ pseudoinverse L_dag (defined in the implementation).

The derivative aggregate profile is  
x_deriv = (x_Econ + x_Emp + x_Edu) / 3

The canonical balanced profile x_balanced is the normalised CGM stage weights [w_CS, w_UNA, w_ONA, w_BU].

The ecology vertex profile is the BU dual combination:  
x_Ecol = (δ_BU / m_a) · x_balanced + A* · x_deriv

The ecology edge ledger is constructed to enforce exact aperture A*:  
y_Ecol = construct_edge_vector_with_aperture(x_Ecol, target_aperture = A*)

The displacement vector is  
D = |x_deriv − x_balanced|

with components ordered (GTD, IVD, IAD, IID).

The ecology ledger y_Ecol is the circulating monetary medium. Because its aperture is constrained to A* by construction, it defines the stable fiat regime. The three canonical ledgers measure activity; the ecology ledger defines what circulates coherently.

Coordinator status reports MUST include y_Ecol, A_Ecol = A*, and the displacement vector D.

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

Ecological genealogies record how the combined state of the three derivative domains has affected ecological integrity and displacement over time.

### 9.5 Genealogies as Assets

Access to deep, aligned genealogies is a structural asset in the Moments Economy. It allows new projects and programmes to initialise from proven governance trajectories rather than reconstructing coordination patterns from scratch.

A genealogy that demonstrates convergence to A* over extended periods, with low displacement and stable surplus generation, is valuable precisely because it is verifiable. Any party can replay the logs and confirm the claimed properties.

---

## 10. Alignment and Economic Stability

The CGM aperture target A* ≈ 0.0207 plays a dual role in the Moments Economy. Geometrically, it specifies the balance between gradient coherence and cycle differentiation. Economically, it distinguishes regimes in which coordination losses are low and surplus can be reliably generated and distributed.

### 10.1 High-Alignment Regime

Projects and programmes whose domain-level apertures converge toward and remain close to A* operate in a high-alignment regime. In this regime, coordination overhead is reduced. Friction, waste, and miscommunication across the epistemic operations diminish. Domain interactions between economy, employment, education, and ecology remain coherent over long horizons.

Surplus capacity in the high-alignment regime is effectively available for UHI and for additional structural development. The economy generates more value than it consumes in coordination costs, and the excess can be distributed or invested.

### 10.2 Displacement and Coordination Cost

Persistent deviations from A* correspond to structural misalignment. The four displacement risks (GTD, IVD, IAD, IID) manifest concretely as coordination failures: decisions that cannot be traced, information that cannot be verified, accountability that cannot be located, coherence that cannot be maintained.

These failures have economic costs. Resources are wasted on redundant verification. Decisions must be revisited because they were made on unreliable foundations. Trust breaks down and must be rebuilt through costly mechanisms. The same structural capacity envelope supports less actual activity because more is consumed by friction.

Within the Moments Economy, convergence toward A* is both a governance objective and an economic objective. The two are not separate concerns but the same concern viewed from different angles.

### 10.3 Network Dynamics

Actors maintaining alignment find coordination with each other low-cost. Their apertures are compatible. Their genealogies are verifiable. Their events compose cleanly. They can share moments and build on each other's histories without extensive reconciliation.

Actors diverging from alignment face increasing coordination costs. Their apertures diverge from partners. Their histories require translation or cannot be verified. They must expend resources on trust-building that aligned actors avoid.

Over time, the network of participants, projects, and programmes exhibits selection pressure toward A*. Aligned configurations attract coordination. Misaligned configurations shed it. This selection is a structural consequence of the geometry rather than an externally enforced rule.

---

## 11. Capacity and Abundance

The Moments Economy grounds prosperity in physical capacity rather than artificial scarcity. This section documents the capacity envelope and demonstrates abundance.

### 11.1 Atomic Foundation

The SI second is defined by the caesium-133 hyperfine transition: 9,192,631,770 periods per second. This frequency provides the physical substrate from which structural capacity is derived.

The Router operates at speeds determined by hardware and implementation quality. Representative throughput on commodity hardware exceeds two million kernel steps per second. Combined with the atomic frequency, this yields a capacity envelope of approximately 22 quadrillion structural micro-state references per second under conservative assumptions.

This atomic grounding ensures that prosperity in the Moments Economy arises from physics rather than from institutional privilege. The abundance is real, measurable, and universally accessible. It does not depend on debt creation or exclusive access to issuance.

### 11.2 Millennium Feasibility

A conservative demonstration mapping treats one structural micro-state reference as one MU. This is intentionally conservative: it ignores the efficiency gains from operating near A* and treats the raw micro-state count as the binding constraint.

Under this mapping, funding UHI for the current global population of approximately 8.1 billion people over 1,000 years uses approximately 0.0000001 percent of available capacity. Verified calculations confirm this figure.

Section 11 quantifies capacity headroom under conservative assumptions. Operational use of capacity remains subject to programme rules and logged governance decisions.

### 11.3 Notional Surplus Allocation

For planning and governance, the surplus may be notionally partitioned across domains and capacities. Twelve divisions (three domains multiplied by four Gyroscope capacities) provide a natural structure for balanced development.

This is a planning representation. It does not imply the creation of twelve separate currencies or ledgers. It provides a framework for ensuring that surplus allocation maintains balance across the coupled domains rather than concentrating in any single area.

---

## 12. Value, Wealth, and Exchange

In the Moments Economy, value is structural integrity. This reframes the fundamental economic concepts.

### 12.1 The Nature of Surplus

Surplus in this economy is untapped coordination capacity. It represents the ability to maintain more complex structures without losing coherence. When a governance system operates near A*, it generates surplus because the geometric configuration minimises coordination loss.

Surplus is unused coordination capacity: the geometric headroom to support additional genealogies, distributions, and projects while maintaining aperture near A*.

### 12.2 Wealth and Poverty

Wealth is access to deep genealogies and the capacity to navigate the state space effectively. A wealthy actor possesses verified histories of aligned governance, proven patterns for maintaining coherence, and the structural resources to participate in complex coordination.

Poverty is the absence of structural resources. It means having no genealogies to reference, no patterns to draw upon, and no capacity to maintain coherence. UHI addresses poverty by providing everyone with a baseline of structural resources sufficient for dignified participation.

### 12.3 Exchange

Exchange in the Moments Economy is not zero-sum. Coordination creates value by reducing friction. When two aligned actors exchange, they both benefit from the reduced coordination costs that alignment enables. They generate surplus together rather than transferring fixed value from one to the other.

Competition and conflict remain possible. Actors may still compete for positions, resources, or influence. The underlying game is positive-sum: the total coordination capacity increases when more actors achieve alignment.

---

## 13. Practical Considerations

### 13.1 Registries and Wallets

The Moments Economy uses registries and optional digital wallets for routing distributions. Banks and other institutions act as coordination and verification platforms rather than as holders of scarce reserves.

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

A conforming public programme MUST publish its byte log, event log, and derived ecology outputs (y_Ecol, D).

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
- shared domain/edge identifiers and ecology derivation rules

The interoperability bundle (byte log, event log, final signature, derived ecology outputs) is sufficient for independent verification.

### 14.5 Transition Path

Transition proceeds in three phases:

1. Measurement phase: institutions and communities run AIR projects, publish genealogies, apertures, and displacement vectors.  
2. Distribution phase: UHI is introduced as a programme distribution with full replay and ecology reporting.  
3. Expansion phase: tier distributions are introduced under the tier governance constraints above, with bounded displacement.

This sequence ensures distribution follows demonstrated integrity rather than asserted authority.

---

## 15. Conclusion

The Moments Economy defines a prosperity-based monetary regime in which value is grounded in geometric coherence rather than debt. Humans retain original authority and agency. Artificial systems operate derivatively within traceable governance flows. The Router provides shared moments and replayable histories. The ledgers provide geometric observables. The aperture target provides the criterion for alignment.

UHI is unconditional and sufficient to eliminate poverty. Participation tiers recognise capacity at levels from baseline participation to civilisation-scale governance. Scarcity of structural capacity is not a binding constraint. Under conservative assumptions, the system has overwhelming capacity to support high unconditional income for all while leaving vast surplus for development across domains.

History in the Moments Economy is a physical object. Different orderings of events produce different structural states. Path dependence enables verification without centralised authority. Any participant can replay a history and confirm whether it matches a claimed structural state.

The network of individuals, projects, and programmes forms through shared structural physics rather than through imposed consensus. Actors who maintain alignment find coordination with each other low-cost. Those who diverge face increasing friction. Over time, this creates selection pressure toward the aperture target.

The surplus is not extra money. It is the structural headroom for civilisation to grow in complexity without collapsing. The question for economic policy shifts from what we can afford to what is worth doing.