# Alignment Infrastructure Routes (AIR)
## Global AI Governance Logistics Framework

---

### 1. Purpose and Scope

Alignment Infrastructure Routes (AIR) comprises protocols and software for recording and verifying coordination. An activity may involve information from several participants and successive transformations through artificial processing. The operational task is to retain the relationships between these contributions so that the resulting decision can be examined in context.

Participants need access to the relevant evidence and an account of how it was used, including the conditions for reviewing a conclusion. These requirements apply to local cooperation and larger programmes. The specifications listed below cover the classification of contributions and the computational procedures for checking their records.

The relevant specifications are organised as follows:

- Common Governance Model: the formal conditions for coherent governance through four capacities.
- The Human Mark: the AI safety and alignment framework used to classify relationships between information sources and the capacities for receiving or processing information. The terminology is introduced in Section 3, with the canonical block reproduced in the AIR Moments Economy Whitepaper, Appendix A.
- Gyroscope Protocol: the classification of contributions according to the capacities exercised.
- Gyroscopic Global Governance: application of these capacities across economy, employment, education, and ecology.
- Gyroscopic ASI hQVM Kernel: the deterministic computation used for recording and replay.
- Moments Economy: accounting and distribution within a common capacity derived from physical constants.

Participants use the classifications to describe an activity and the computational procedures to verify its recorded history. The distribution of capacities can then be examined through the contributions and relationships documented in that history.

---

### 2. Coordination requirements

In a decision process, information may be selected and transformed through several human and artificial contributions. Traceability requires a record of these relationships, including the observations available to providers and receivers of information. Reviewers can then examine the sources used in an inference and the processing applied to them.

Procedures for providing information and reviewing inferences form part of governance alongside the recorded outcome. The distribution of these capacities matters because participants contribute knowledge from different positions within the activity. In a public service, the recipient's observations and ability to contest an assessment remain relevant throughout the process.

Logistical implementation consists of recording the sequence of contributions, retaining their classifications, and making the relevant evidence available for review. Participants can use the same procedures across organisational boundaries while preserving the context of each contribution.

---

### 3. Canonical Ontology for Governance Logistics

For classification of the relationships described above, participants use The Human Mark (THM), an AI safety and alignment framework. Authority denotes the information available for inference and intelligence. Agency denotes the capacity to receive or process that information. They are epistemic capacities because they concern how information is available and used in forming knowledge, and they are exercised across providers and receivers within an activity.

Direct and Indirect classifications distinguish human capacities from artificial forms dependent on human intelligence. Ancestry denotes the relationship of dependence between them. Alignment requires that this relationship remain measurable through traceability of information variety, inference accountability, and intelligence integrity to Direct Authority and Agency. The following definitions establish the terms used in the recording requirements.

#### 3.1 Direct and Indirect Classifications

The classifications in The Human Mark distinguish Authority from Agency and Direct from Indirect forms.

**Direct Authority** refers to direct human access to a subject matter. Examples include an eyewitness observing an event, a clinician examining a patient, or a researcher conducting a measurement. The defining feature is unmediated epistemic access.

**Indirect Authority** refers to Indirect or processed information. Examples include reports, databases, statistical analyses, and model outputs. The defining feature is that the information has passed through one or more transformations from Direct Authority.

**Direct Agency** refers to human capacity to receive information, reason about it, and take decisions for which the person can be held accountable.

**Indirect Agency** refers to artificial capacity to process inputs and produce outputs. Artificial systems can transform and route information, but they do not constitute Direct Authority or Agency and cannot bear final accountability.

In this ontology, artificial intelligence systems are always Indirect Authority and Indirect Agency. Regardless of their capability, they remain constitutively dependent on human Direct Authority for the validity of their inputs and on human Direct Agency for the accountability of their outputs.

Power concentration arises when an epistemic category is attributed to a particular bearer as though that bearer exhausted it. Measurement of ancestry across providers and receivers is then lost. This error is possible in relation to an individual, a government or other institution, or an artificial system. Its four systematic forms are classified as follows:

- **Governance Traceability Displacement** occurs when Indirect Authority and Agency are treated as Direct, severing traceability to Direct Authority and Agency.
- **Information Variety Displacement** occurs when derivative outputs are mistaken for direct observations, collapsing the distinction between processed patterns and direct evidence.
- Inference Accountability Displacement occurs when Indirect Agency without Authority is treated as Direct, with loss of the ancestry of inference.
- **Intelligence Integrity Displacement** occurs when direct human capacity is devalued relative to derivative processing, eroding the foundation of governance itself.

Assessment of these risks examines the distribution and ancestry of capacities across the full process. The same assessment applies to human and artificial contributions within private or public arrangements.

#### 3.2 Four Governance Capacities

The four capacities are defined in relation to the formal conditions for coherent governance in the Common Governance Model:

- Governance Management Traceability: traceable ancestry of governance across providers and receivers.
- Information Curation Variety: preservation of distinguishable sources and forms of information.
- Inference Interaction Accountability: accountable relations between information and inference through Agency.
- Intelligence Cooperation Integrity: coherence of those relations over time and across contexts.

Participants may exercise all four capacities within a single activity. Several participants may contribute to the same capacity, and their records should preserve the relations among those contributions. Uniform power distribution follows from continued exercise of these capacities throughout the process.

#### 3.3 Four Application Domains

In Gyroscopic Global Governance, the four capacities are considered across four coupled domains:

- **Economy**: allocation of resources and settlement of value; the material medium for coordination.
- **Employment**: human work and contribution, through which the four capacities are maintained in practice.
- **Education**: formation and renewal of Direct Authority and Direct Agency.
- **Ecology**: overall balance among the other three domains and the sustainability of their combined operation.

Participants maintain records of activity in the first three domains. Ecology is derived from the combined state of the other three and emerges from cross-domain analysis.

The same capacities apply within households, informal groups, organisations, and wider programmes. Learning involves checking sources and revisiting beliefs. Everyday economic choices relate resources to commitments, while paid work and informal care combine the four capacities in practice. Ecological reflection relates the effects of those choices to the actions that produced them. Through AIR, participants can preserve these relationships as activities connect across contexts, including where service providers or institutional operators contribute tools and administration.

---

### 4. Core Components of AIR

Participants implement these requirements through the recording and verification procedures described below.

#### 4.1 The Gyroscopic ASI Kernel

Operators compute coordination states using the deterministic hQVM transition rule, with the following properties:

- It represents coordination as a sequence of states on a deterministic 24-bit carrier. From the rest condition, the shared-moment reachable space used operationally has 4,096 states, with two 64-state boundary horizons (the equality horizon where A = B, and the complement horizon where A = B XOR 0xFFF).
- It updates its state in response to single-byte inputs, with 256 possible input values.
- Given the same starting state and the same sequence of bytes, any conforming implementation will compute exactly the same trajectory of states.
- Every transition is reversible: given a final state and the bytes that led to it, the predecessor state can be reconstructed.

From any fixed state, the 256-byte alphabet produces 128 distinct next states with exact 2-to-1 multiplicity, reflecting the SO(3)/SU(2) double cover at the discrete level; full history is preserved byte-complete through replay.

The router does not interpret what the input bytes mean. It applies fixed transformation rules to move from one state to another. This property is essential for governance. Because the router does not embed interpretation, it cannot introduce hidden bias or drift. Interpretation happens at the application layer, where it is visible and governable. The router provides a neutral medium that records and routes without distortion.

In practical terms, the router provides a canonical coordination log. Each governance event corresponds to one or more bytes. The history of a project, organisation, or system corresponds to a sequence of bytes applied to the router. Anyone with access to that sequence can replay it from the starting state and arrive at exactly the same final state. This eliminates dependence on trusted intermediaries: verification is a matter of computation, not testimony.

The kernel's coordination medium has structural properties that strengthen its governance role. The self-dual [12,6,2] mask code detects all odd-weight bit errors in states unconditionally, providing intrinsic tamper detection. From any starting state, two consecutive byte steps distribute the coordination state exactly uniformly across all 4,096 reachable states, ensuring rapid structural convergence without central orchestration. The kernel also supports a 6-bit chirality register that tracks structural divergence between parties through an exact transport rule, enabling early detection of coordination drift before full state disagreement becomes visible.

#### 4.2 Genealogies

A **Genealogy** is a complete byte record that can be replayed end to end for an actor, project, or system. Its canonical kernel-native core is the byte log. Application-layer event logs may be bound to hQVM Kernel states or depth-4 frames, but they are not part of the kernel-native definition.

Because the router is deterministic, the genealogy can be replayed at any time. An auditor, regulator, or third party can load the byte log, run it through a conforming router implementation, and verify that the claimed trajectory is accurate. The event log can then be checked against this trajectory to confirm that events are correctly bound.

For stronger certification, genealogies SHOULD be segmented into depth-4 frames. Each frame yields a deterministic record (mask48, φ_a, φ_b). These frame records are strictly stronger than final-state-only certification, because different byte histories can collapse to the same final state while retaining different frame records.

Genealogies replace informal histories and narrative accounts with replayable records. They are portable: any system running the same router implementation can load a genealogy and reproduce its coordination history. They are also durable: because they consist only of byte sequences and event records, they can be stored indefinitely and verified at any future time.

When two parties share the same byte-log prefix, they compute the same hQVM Kernel state and therefore share the same moment. When they diverge, frame comparison localizes the divergence to the affected 4-byte frame. This gives AIR both shared coordination and precise fork localization.

#### 4.3 Physical Grounding of Capacity

In the Moments Economy, coordination capacity is derived from physical constants. The foundation is the caesium-133 hyperfine transition frequency, the atomic standard that also defines the SI second. This frequency establishes the physical resolution at which coordination events can be distinguished.

The Common Source Moment is calculated from this frequency. This represents the total coordination capacity of the light-sphere at atomic resolution, divided by the settlement system's 4,096 checkable states (reachable from rest under the public transition rule). The result is a fixed total capacity of approximately 7.94 × 10²⁶ Moment-Units.

Participants can inspect the capacity derivation as a common accounting reference. They govern its use through traceable allocations and agreed responsibilities, whether they keep records together locally or use administrative support from a larger organisation.

In practice, this capacity is inexhaustible on any human timescale. The Common Source Moment can support global baseline distribution for approximately 1.12 trillion years at current population and base-rate assumptions. The constraint on governance is therefore not capacity but quality: whether coordination events are correctly classified, properly routed, and coherently integrated.

Because baseline capacity is abundant, the primary operational risk is the exclusion of real humans through defensive access mechanisms, rather than the fraudulent claiming of excess capacity. The design order is therefore: accessibility first, coherence second, repair third, exclusion last.

#### 4.4 Shared Moments, Frame Commitments, Receipts, and Divergence Detection

Verification uses three kernel-native certification layers, together with a transport and archive profile.

First, the hQVM Kernel state gives a shared moment for coordination. When two parties share the same byte-log prefix, they compute the same hQVM Kernel state and therefore share a structural "now."

Second, depth-4 frame records (mask48, φ_a, φ_b) give stronger provenance and exact divergence localization. Each frame is computed from four consecutive bytes and is deterministic. Different byte histories can collapse to the same final hQVM Kernel state, but they produce different frame records. Frame comparison localizes divergence to the affected 4-byte frame.

Third, parity commitments provide compact algebraic integrity checks over longer trajectories. A trajectory parity commitment is a triple (O, E, parity), where O and E are 12-bit XOR sums of masks at even and odd byte positions, and parity is the trajectory length modulo 2.

The operational settlement object is the **moment receipt**: a short position on a deterministic identity trajectory, specified by an anchor, a depth, and a phase, with seal, parity, and event-class fields regenerable by replay. The manifold address inside the transport time field is state-derived and regenerable; sec32 recovery and discriminator allocation remain open items, and the compact depth-delta storage math depends on resolving them. Measured transport layouts occupy 16 to 20 bytes and fit commodity QR codes. An identity's archive stores the anchor once and one depth delta per receipt; each trajectory epoch carries a 512-byte occupancy bitmap for local duplicate detection. Transport layouts, QR enclosure, and the derived name layer for archive append control are **implementation profiles** layered above replay. Conformance remains defined by byte replay and canonical serialization (SHA-256 for Identity Identifier computation). The name layer names content for append control; it does not address the manifold.

These layers are replayable from the byte log and do not require an external ledger geometry to operate. Because each certification layer is computed from exact integer arithmetic on the byte log, verification is portable across implementations and platforms without numerical precision concerns. Publication in coordinate-ledger form (anchors, depth deltas, occupancy state) satisfies full-object publication requirements for kernel-native structural objects that replay regenerates; Event Logs, payloads, and policy bases must still be published as data. Measurements and open implementation items are recorded in the Moment Receipts, QR Transport, and FNV Profile analysis.

The four-domain AIR organisation remains valid, but AIR no longer depends on an externally imposed K₄ measurement layer or aperture computation for operational verification.

#### 4.5 Classification Protocols

Participants classify governance events using the following two protocols before recording them.

Under The Human Mark, participants classify the provenance and role of information and inference. Every input to the system is tagged according to whether it carries Direct Authority, Indirect Authority, Direct Agency, or Indirect Agency, preserving the distinction between human and artificial contributions under the Mark. This classification is recorded in the event log and bound to the corresponding router state. It ensures that the distinction between human and artificial roles is maintained throughout the coordination process.

Under the Gyroscope Protocol, participants classify contributions according to the four governance capacities:

- **Governance Management** work maintains traceability of authority. It includes leadership, oversight, administration, and resource allocation.

- **Information Curation** work maintains variety of Authority. It includes research, editing, data stewardship, and the design of measurement systems.

- **Inference Interaction** work maintains accountability of conclusions. It includes negotiation, care, teaching, and human review of artificial outputs.

- **Intelligence Cooperation** work maintains integrity over time. It includes engineering, institution building, and cultural preservation.

Participants classify the parts of an activity according to the capacities they support. One task can involve all four, including when someone performs it as unpaid care or informal collaboration. The classification makes those contributions visible and helps participants identify where further support is needed.

#### 4.6 Grants, Shells, and Moment Receipts

Economic and resource allocations are recorded through the following constructs:

A **Grant** is a record of a single allocation: a payment, a capacity assignment, or a resource transfer. It includes the identity of the recipient (linked to a kernel state via an Identity Anchor), the quantity allocated, and the genealogical binding that establishes when the allocation occurred. In canonical serialization, a Grant is encoded as `identity_id || kernel_anchor || amount_mu`. Grant fields, including the amount, are carried in the payload whose routed state forms the moment-receipt seal. The default payload schema is that canonical Grant receipt; other payload schemas are implementation profiles. The receipt position itself carries no amount field. Offline verification and counterparty amount-knowledge therefore require the payload to travel and archive alongside the 16-to-20-byte transport form.

An **Identity Anchor** links an identity to a fixed starting position on the deterministic settlement record. The identity bytes routed for anchor derivation are the SHA-256 Identity Identifier; verification regenerates receipt fields from those identity bytes and the payload.

A **Shell** is a container that groups grants over a defined scope, such as a time period or a programme. It carries a seal computed by routing its canonical contents through the public kernel. This seal binds the shell to a specific coordination state, making it tamper-evident. Anyone can verify a shell by replaying its contents and checking that the computed seal matches. Shell seals are order-invariant container commitments computed over canonically sorted Grant receipts; Grant insertion order does not affect the seal. Trajectory coordinates are order-sensitive per-identity positions. The two views are reconciled by replay.

A **Moment** is a reproducible hQVM Kernel state at a specific byte-log prefix. The **moment receipt** is its transport form: a regenerable coordinate specified by anchor, depth, and phase. For stronger certification, a published Moment MAY also include the current depth-4 frame record and a trajectory parity commitment. The receipt's event-class byte is a transport chirality/gauge field derived from the payload; it is distinct from the application-layer Event Log, which annotates meaning, decisions, and justifications.

These constructs enable verifiable settlement. Payments can be traced through genealogies. Shells can be validated through replay. Moment receipts provide portable anchors for offline presentation and later synchronisation. Participants verify the recorded computation by replaying it under the public rules. The normative economic layer is specified in the Moments Economy Architecture Specification.

---

### 5. Relation to Existing Standards and Regulations

Organisations can use replayable records as evidence when assessing compliance with applicable quality, security, or risk-management requirements.

Consider the difference between procedural and verifiable compliance:

- **Procedural compliance** means that an organisation has documented policies and can show evidence that policies were followed. Verification depends on trusting the organisation's records and the auditors who reviewed them.

- **Verifiable compliance** means that the actual sequence of governance events is recorded in a replayable form, and any party can independently reconstruct what occurred. Verification is computational rather than testimonial.

By recording governance events in genealogies, participants produce evidence that reviewers with access to the byte log can check computationally. Reviewers assess that evidence in relation to the applicable requirements.

Examples of record use in standards assessment:

**Quality management (such as ISO 9001):** The standard requires documented processes and evidence of their execution. Operators record process execution in genealogies for comparison with the specified procedures. Replayable genealogies, deterministic shell seals, and frame-level divergence localization provide quantitative and inspectable evidence of governance process integrity over time.

**Information security (such as ISO 27001):** The standard requires controls to protect information integrity. Operators compute Shell seals and replay genealogies to check record integrity. A claimed state, seal, or history can be independently checked by replay from rest under the public transition rule and canonical serialization rules.

**Artificial intelligence management (such as ISO 42001):** The standard requires accountability and transparency for AI systems. Participants classify Direct and Indirect Authority and Agency under The Human Mark and retain those classifications in the process record. Genealogies bind AI evaluations and outputs to specific router states, providing an audit trail.

**Regulatory regimes (such as the European Union Artificial Intelligence Act):** The regulation requires human oversight and documentation for high-risk AI systems. Operators retain replayable records of decisions and the available information, including the classification and use of AI outputs. Regulators can verify these records independently.

In each case, reviewers can inspect the recorded sequence alongside the applicable requirements. The computational checks concern record integrity, while assessment of the process also requires its documented context.

---

### 6. Practical Applications

The following examples concern the use of AIR records in processes involving human and artificial contributions.

#### 6.1 Model Evaluation and Deployment

**Without AIR:** An organisation develops and deploys AI models. Evaluation results are recorded in spreadsheets and documents. Deployment decisions are made in meetings and recorded in emails or tickets. When a deployed model behaves unexpectedly, tracing the decision to deploy it requires forensic investigation: gathering documents, interviewing people, and reconstructing a narrative.

**With AIR:** Each evaluation is a governance event bound to a router state. Each deployment approval is a governance event bound to a router state. The Human Mark classification tags model outputs as Indirect Authority. The entire sequence from evaluation through deployment is recorded in a genealogy. When unexpected behaviour occurs, the genealogy is replayed. The router state at deployment is identified. The events leading to that state are listed. The classification of each input is visible. Regulators or auditors can replay the same genealogy and verify the organisation's account.

#### 6.2 Research Provenance

**Without AIR:** A research paper claims to be based on experimental data. The data passed through several processing steps and was analysed using machine learning models. Reviewers and readers must trust that the authors correctly attributed their sources and did not confuse model outputs with primary observations.

**With AIR:** Each data collection event is classified as Direct Authority and bound to a router state. Each processing step is classified as Indirect Authority. Model outputs are classified as Indirect Authority and Indirect Agency. The genealogy records the full provenance chain. Reviewers can inspect the class classification of each input to the analysis. Readers can verify that claims about primary evidence are actually grounded in Direct Authority.

#### 6.3 Public Service Delivery

**Without AIR:** A government agency uses automated systems to assess eligibility for benefits. Caseworkers review edge cases. Payments are issued through a financial system. When errors occur, determining whether the fault lies with the automated system, the caseworker, or the payment system requires investigation.

**With AIR:** Eligibility assessments are governance events classified by Authority and Agency (automated systems as Indirect Agency, caseworker decisions as Direct Agency). Payments are grants within shells. The genealogy records the sequence from application through assessment through payment. When errors occur, the genealogy identifies exactly which event caused the error and what its classification was. Remediation can target the specific point of failure.

#### 6.4 Economic Distribution Programmes

**Without AIR:** An organisation implements an unconditional income programme. Payments are issued monthly. Recipients must trust that the organisation is calculating and issuing payments correctly. The organisation must maintain internal records and submit to periodic audits.

**With AIR:** Payments are grants within shells. Each shell carries a seal derived from the public kernel. Recipients receive not just payments but verifiable moment receipts (anchor, depth, phase) whose proof fields regenerate by replay. The organisation publishes shells, genealogies, and, where appropriate, coordinate-ledger archives. Any party can replay the genealogy to verify that the correct payments were issued. Audits become computational rather than investigative.

Where physical resources are constrained, participants document fair-use rules in Community Shells and retain the evidence and review procedures applicable to those rules.

---

### 7. Adoption and Next Steps

People can begin with a shared activity and agree on how to record its sources and decisions. A household coordinating care or a group reviewing AI outputs can use the same classifications and replay procedures as a larger programme. Adoption develops as participants find these records useful and connect them to further activities.

For local groups, a practical starting point is to identify each contribution, record the decisions made, and agree on how to review or correct them. Organisational deployments can build on the same practice in the following ways.

**For organisations deploying AI systems:** Begin by recording governance events in genealogies. Classify inputs using The Human Mark. Track replayable genealogies, shell seals, frame commitments, and moment receipts over time. Where inference hosts already run the kernel in the model execution path, receipt creation and local archive maintenance can attach to that installed base using the existing host infrastructure. Publish shells, genealogies, and coordinate-ledger archives for external verification. This provides an audit trail that can be inspected by regulators, partners, or the public.

**For regulators and auditors:** Request genealogies from regulated organisations. Replay them using conforming router implementations. Verify that classifications are consistent with claims. Compare replay integrity, shell verification results, and frame-localized divergences across organisations to identify outliers. This shifts regulatory practice from reviewing documents to verifying computations.

**For researchers and developers:** Extend the framework to new domains. Develop tools for genealogy analysis. Investigate the relationship between coordination structure and governance outcomes. Contribute to the open specifications.

The technical specifications for all components are published through the Gyro Governance repositories. The public kernel specification, The Human Mark classification system, the Gyroscope Protocol, and the Moments Economy architecture are documented in detail. Reference implementations are available for testing and integration.

---

### 8. Review of operational practice

An operational review should examine whether providers and receivers can exercise the four capacities throughout the activity. Relevant evidence includes the sources considered, the transformations applied to them, and the opportunities available to review or revise an inference. The same relationships apply as participation extends from local arrangements to larger programmes, including where institutional operators contribute tools or administration.

Participants can compare the documented procedures with the recorded history and identify where a category has been attributed exclusively to a particular bearer. Subsequent revisions should restore the distributed exercise of capacities and the measurement of their ancestry. The revised procedures and their observed effects can then be examined in further use.
