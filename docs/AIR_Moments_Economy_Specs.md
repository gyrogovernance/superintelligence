# Moments Economy Architecture Specification

## Introduction

The Moments Economy is a monetary and settlement architecture in which the ability to issue money is limited by a publicly verifiable physical capacity rather than by debt issuance, discretionary ledger authority, or institution-specific trust. It is grounded in the caesium-133 hyperfine transition frequency (`f_Cs = 9,192,631,770 Hz`), the physical frequency that defines the SI second. Because this atomic standard fixes how finely distinct physical events can be resolved, it also sets a physical bound on how much coordination can be recorded and settled within a shared causal region. In this architecture, money is not treated as an expandable claim created by a central issuer. It is treated as a recorded and replayable governance allocation within a bounded structural capacity.

That envelope is called the Common Source Moment. It is derived from the caesium-133 atomic-second standard together with the finite verification space of the settlement system. The result is a fixed one-time capacity for recording and settling coordination. This capacity is not a metaphor. It is the explicit physical and geometric basis on which issuance is bounded.

The settlement system used in this architecture is the Gyroscopic ASI aQPU Kernel. It is a deterministic verification system that maps append-only byte histories to reproducible state trajectories. Because replay is exact, independently held records can be checked without relying on a central ledger authority. Distribution, provenance, consultation, and correction can therefore be published as structural objects that any conforming party can verify by replay.

The Moments Economy is an institutional record architecture as well as a distribution system. The same infrastructure that supports monetary settlement also supports complete governance records, including who acted, what was issued, what evidence was referenced, what corrections were made, and where disputes arose. This is why the architecture matters beyond economic policy narrowly understood. It offers a common method for settlement, audit, provenance, and institutional memory within one replayable medium.

The architecture also has a staged adoption path. Institutions do not need to begin by replacing existing currencies or payment rails. They can first adopt the system for coordination, audit, compliance, and traceable programme administration. In doing so, they build the same byte-complete records that later make economic settlement in Moment-Units possible. The transition path is therefore cumulative. The infrastructure first functions as a coordination and verification system and only later, where conditions permit, as a monetary settlement architecture.

This document specifies that architecture. It defines the unit of account, the capacity envelope, the structural objects of settlement, the verification pattern, the domain model, the epistemic commitments, and the institutional requirements for transition.

### Why this matters

* **For individuals:** A guaranteed baseline distribution with additional tiered distributions for wider responsibility, delivered through verifiable records rather than debt-based issuance.
* **For policymakers:** Issuance limits derived from explicit physical and geometric assumptions that can be inspected, challenged, and revised through governance.
* **For institutions:** A settlement and audit method in which distributions and eligibility decisions are replayable records rather than opaque internal updates.
* **For AI safety:** A coordination medium that preserves human authority, traceability, and accountability in systems where artificial agents contribute to decisions and record-keeping.

### Two capabilities, one infrastructure

The same infrastructure provides two distinct capabilities.

First, it provides an immediate capability: verifiable coordination records for audit, safety, compliance, dispute resolution, and programme administration.

Second, it provides a latent capability: a complete replayable history of distributions and governance actions that can serve as the accounting basis for Moment-Units.

Adoption for the first purpose automatically builds the infrastructure for the second. The same byte logs and event bindings used for coordination and verification can also support economic settlement when the conditions for transition are met.

### Scope and relationship to AIR

The Gyroscopic ASI aQPU Kernel serves here as the settlement and verification layer for the Moments Economy. The same kernel also serves as the coordination backbone for Alignment Infrastructure Routing, a related coordination framework for grants, work receipts, and project histories. These uses are related but distinct. Institutions may adopt AIR for coordination without adopting the Moments Economy as a settlement architecture. This document specifies the additional economic layer that becomes possible when replayable coordination records are used as the basis for monetary distribution.

### Document structure

**Part I: The Economic Proposition** defines the Moment-Unit, the Common Source Moment, the baseline unconditional distribution, and the participation tiers.

**Part II: The Architecture** specifies the structural objects, the four domains, the verification pattern, and the role of genealogies.

**Part III: Foundations** explains the epistemic commitments and the geometric invariants that underpin the system.

**Part IV: Institutions and Transition** sets out registry, settlement, governance, interoperability, and transition requirements.

### Related frameworks

The architecture draws on five related internal frameworks and specifications.

**Common Governance Model (CGM):** A geometric theory of coherent measurement used in this body of work to describe an intrinsic fourfold governance structure.

**Gyroscopic Global Governance (GGG):** A governance framework developed in the surrounding research programme, applying four governance capacities across economy, employment, education, and ecology.

**The Human Mark (THM):** An epistemic taxonomy used here to distinguish Direct human sources from Indirect artificial and mediated sources.

**Gyroscopic ASI aQPU Kernel:** The deterministic coordination kernel used in this architecture for shared moments, provenance, and replay.

**Gyroscope Protocol:** A classification framework used to describe work in terms of governance management, information curation, inference interaction, and intelligence cooperation.

Normative requirements use **MUST**, **SHOULD**, and **MAY** as defined in RFC 2119.

---

# Part I: The Economic Proposition

## 1. The Moment-Unit

The unit of account is the Moment-Unit, or MU.

The MU is anchored to time for readability and practical accounting: **one MU corresponds to one minute of capacity at the base rate**.

The base rate is fixed at 60 MU per hour. This yields:

* 1,440 MU per day
* 525,600 MU per year

This convention makes annual magnitudes legible in familiar terms while avoiding dependence on commodity prices or debt instruments.

The MU is a scalar unit of account. As noted above, the minute denomination is for readability only; the system is not a time currency. The underlying capacity of the system is not generated by the passage of time. It is drawn from a fixed one-time capacity envelope, the Common Source Moment.

A Moment and a Moment-Unit are not the same kind of object. In this document, a **Moment** means a reproducible verification state at a given point in a replayable record. A **Shared Moment** means such a state when multiple parties reproduce it from the same record. A **Moment-Unit** is the accounting measure used to denominate distributions within this architecture.

## 2. The Common Source Moment

The system capacity is the Common Source Moment, or CSM. It is calculated as the phase-space volume of the atomic-second light-sphere, coarse-grained by the finite verification space of the settlement system. In plainer terms, the architecture takes the smallest shared physical timing reference in common scientific use and combines it with a finite replay space to define a bounded total settlement capacity. This is a one-time total, not a renewable rate.

### 2.1 Capacity derivation

**1. Physical capacity standard**

The International System of Units defines the atomic second via the caesium-133 hyperfine transition frequency:

`f_Cs = 9,192,631,770 Hz`

This frequency is used as the physical coordination standard because atomic timekeeping provides the most precise and internationally audited method currently available for distinguishing and synchronising physical events.

**2. Physical volume**

The causal container is the light-sphere whose radius equals the distance light travels in one second. At atomic wavelength `λ = c / f_Cs`, the raw physical microcell count is:

`N_phys = (4/3)π f_Cs³ ≈ 3.25 × 10³⁰`

The speed of light cancels in this expression, because `c` appears in both the volume and the wavelength expression, yielding a purely geometric and frequency-based invariant.

**3. Coarse-graining by reachable verification states**

The settlement system has a finite reachable verification space:

`|Ω| = 4,096`

Uniform coarse-graining by this verification space gives the Common Source Moment:

`CSM = N_phys / |Ω| ≈ 7.94 × 10²⁶ MU`

This is the fixed total capacity envelope for the Moments Economy.

### 2.2 Functional meaning of the capacity

The Common Source Moment serves two distinct functions.

**A. Monetary distribution**

It provides the total capacity envelope within which baseline and tiered distributions can be issued.

**B. Coordination records**

It provides sufficient capacity to preserve complete coordination records, including provenance, consultation histories, commitments, disputes, and corrections.

Because the capacity far exceeds foreseeable demand, these records do not need to be compressed into sparse summaries merely to conserve accounting space. Multiple institutions can maintain complete independent records without approaching saturation.

### 2.3 Capacity implications

The practical implications are as follows.

* Global UHI demand per year is `≈ 7.10 × 10¹⁴ MU`.
* The CSM supports global UHI for approximately `1.12 × 10¹² years`.
* Under realistic tier participation assumptions, the duration remains on the order of `10¹²` years.
* Adversarial exhaustion of even 1% of total capacity would require issuance on the order of `1.12 × 10¹⁰` times annual global UHI. On any human, institutional, or civilisational timescale relevant to settlement design, this is operationally impossible.

Capacity exhaustion is therefore not a credible attack vector for the Moments Economy. The real constraints are governance quality, registry integrity, and publication discipline.

## 3. Unconditional High Income

Unconditional High Income, or UHI, is the universal baseline distribution provided to every person recorded in a conforming registry under published eligibility rules.

### 3.1 Definition

UHI corresponds to four hours per day at the base rate.

* Daily: 240 MU
* Annual: 87,600 MU

### 3.2 Distinction from temporal currencies

The MU is denominated in minutes for readability, but value in this architecture does not derive from labour duration. It derives from structurally verifiable coordination within a fixed physical-capacity envelope.

The Moments Economy is therefore neither a time bank nor a labour-time currency. It is a settlement architecture whose accounting unit is made legible through time denomination while remaining grounded in a one-time geometric capacity.

### 3.3 Mechanism

Individuals receive UHI through public registries maintained by recognised institutions, fiscal hosts, or equivalent public-interest entities.

These registries:

* bind the individual to an Identity Anchor,
* issue Grants within time-bounded Shells,
* publish the relevant structural records,
* and route payment through banks or digital wallets where applicable.

Every step in this process produces a replayable audit trail.

## 4. Participation tiers

Participation tiers define the entitlement schedule of the Moments Economy.

Tier 1 is the universal baseline distribution: UHI.

Tiers 2 to 4 recognise progressively wider scope and higher responsibility. All tiers draw from the same Common Source Moment envelope.

### 4.1 Tier schedule

* **Tier 1:** 1× UHI = 87,600 MU annually
* **Tier 2:** 2× UHI = 175,200 MU annually
* **Tier 3:** 3× UHI = 262,800 MU annually
* **Tier 4:** 60× UHI = 5,256,000 MU annually

The Tier 4 multiplier reflects the outsized structural impact of governance-management roles in this architecture. It remains a governance parameter and is therefore subject to revision through the processes described in §4.3.

### 4.2 Capacity associations

The tier structure aligns with the four governance capacities used in the Gyroscope framework.

* **Tier 1: Intelligence Cooperation:** maintenance of shared systems and continuity of participation
* **Tier 2: Inference Interaction:** negotiation of meaning, mediation, and conflict resolution
* **Tier 3: Information Curation:** verification, selection, contextualisation, and stewardship of informational order
* **Tier 4: Governance Management:** direction of authority, traceability, and institutional continuity across systems

### 4.3 Governance of tier assignments

Tier 1 is universal.

Assignments for Tiers 2 to 4 are governance actions. They MUST:

* be made by identifiable human agents,
* be recorded as governance events bound to specific Moments,
* be reversible through subsequent logged events,
* and SHOULD reference the genealogical evidence on which the decision relied.

Tier multipliers are governance parameters. They may be revised through institutional process, but every revision MUST be published in a replayable form.

---

# Part II: The Architecture

## 5. The four domains

The architecture organises activity into four coupled domains drawn from Gyroscopic Global Governance, a governance framework developed within the same body of work as this specification.

The **economy** domain covers infrastructure, routing, settlement, and circulation. It includes the movement of Moment-Units, the publication of structural records, and the maintenance of the settlement environment.

The **employment** domain covers work and contribution. It includes the classification of activity into the four governance capacities used in the Gyroscope framework and the programme-level interpretation of contributions that may justify tiered allocations.

The **education** domain covers capacity formation. It includes the cultivation of the human capabilities required for governance, the development of alignment capacities, and the detection of displacement risks.

The **ecology** domain is the integrative domain of systemic balance. It reflects the accumulated state of the other three domains rather than operating as an independent edge ledger. It is computed from cross-domain records and may be summarised through Shells, Archives, and other aggregate capacity views.

These four domains are not arbitrary administrative categories. They correspond to the fourfold coordination structure assumed within the surrounding governance research. The economic architecture uses that correspondence directly.

## 6. Structural objects

Accounting and verification rely on six standardised structural objects.

The **Byte Log** is the canonical append-only sequence of bytes. It is the primary replay object. Every conforming verification procedure ultimately depends on the integrity of this log.

The **Event Log** is the application-layer annotation layer bound to specific verification states or to depth-4 frame records. It records meanings, decisions, classifications, references, and justifications that the kernel itself does not interpret.

An **Identity Anchor** links an identity to a structural coordinate within the settlement system. It consists of an Identity Identifier, which is a collision-resistant hash of the identity string, and a Kernel Anchor, which is the state obtained by routing that identifier from rest. In the reference implementation, the Identity Identifier is a SHA-256 digest and the Kernel Anchor is the resulting 6-hex-character kernel state.

A **Grant** is a record of a single MU allocation. It contains an identity label, an Identity Identifier, a Kernel Anchor, an MU amount, and the relevant shell context. The canonical Grant receipt is:

`identity_id || kernel_anchor || amount_mu`

where `||` denotes concatenation and `amount_mu` is encoded as an unsigned 8-byte big-endian integer.

A **Shell** is a time-bounded capacity container, typically annual or programme-bounded. It contains a contextual header, total, used, and free capacity metrics, a set of Grants, and a Seal. The Shell Seal is an order-invariant, deterministically computed commitment over the Shell's canonical contents. The computation procedure is specified in Appendix C.

An **Archive** is a long-horizon aggregation object. It aggregates Shells to track per-identity totals, programme totals, corrections, and overall capacity usage across multiple periods.

These objects form the minimum structural vocabulary required for settlement. Byte Logs and Event Logs preserve the replayable history. Identity Anchors and Grants define allocations. Shells and Archives make those allocations publishable and auditable across time.

## 7. Verification and replay

The defining feature of the Moments Economy is deterministic verification.

The settlement system verifies structural integrity and replay consistency. It does not determine eligibility, authorisation, or justice. Those remain institutional and human governance functions that must themselves be published in forms suitable for replay and audit.

Any party with access to the published artefacts can verify a Shell through the following procedure:

* Load the published Header and Grants.
* Reconstruct the canonical byte sequence by converting Grants into canonical receipts and sorting them.
* Route that sequence through a conforming aQPU Kernel instance from `GENE_MAC_REST`.
* Compare the resulting state with the published Seal.

A match confirms that the published structural object corresponds to its canonical contents. A mismatch proves that the Header or at least one Grant differs from the published claim.

Because replay uses exact integer arithmetic and fixed-width bit operations, conforming implementations produce identical results regardless of platform or language.

The system supports three layers of certification relevant to economic verification.

* **Final-state layer:** Shared Moments as reproducible 24-bit verification states.
* **Frame layer:** Depth-4 frame records `(mask48, φ_a, φ_b)`, providing stronger provenance than final states alone, because distinct histories can collapse to the same final state while still producing different frame records.
* **Parity layer:** Compact commitments `(O, E, parity)` for integrity checking and batch verification, but not unique history certificates.

A structurally correct seal does not by itself certify application-layer validity. Conforming settlement policy SHOULD reject or flag duplicate identity receipts within a Shell, SHOULD reject or flag Shells whose used capacity exceeds total capacity, SHOULD preserve correction histories rather than silently overwriting them, and SHOULD publish the policy basis for eligibility and dispute handling.

Public programmes MUST publish, at defined intervals, the Byte Logs, Event Logs, Shells, Archives, and any frame records or commitments required by their verification policy.

## 8. Coordination levels

At the **individual** level, any person or organisation may operate as a node. Each maintains a local aQPU Kernel instance and its own logs.

At the **project** level, a shared context of contribution is defined. Participants agree on a canonical Byte Log and Event Log. Divergence is detected by replay. Where distinct histories collapse to the same final state, frame records localise the divergence.

At the **programme** level, multiple projects are aggregated under a wider mandate. Programmes maintain references to project genealogies and produce programme-level Shells and Archives. Programme bundles MAY be aggregated by meta-routing, where leaf seals are themselves routed into a higher-level root seal. This enables deterministic multi-programme aggregation with tamper localisation.

## 9. Genealogies

A genealogy is the complete structural history of an actor, project, or programme within this architecture.

It consists of the Byte Log, any bound Event Log, the trajectory of Moments, the depth-4 frame sequence, and any optional compact integrity commitments attached to that trajectory.

A final state alone is not a unique history certificate. Genealogy-grade audit therefore compares frame sequences as well as final states.

Genealogies function as verifiable assets. A programme can prove its history of alignment and capacity usage by providing its genealogy for replay. New programmes may initialise from the final state of an existing verified genealogy and thereby preserve continuity.

Verification follows a three-stage social pattern. It begins locally, where each actor maintains its own kernel instance and logs. It extends through publication, where selected genealogies and structural objects are exported as signed bundles. It completes through independent verification, when other parties replay those bundles against the public specification. Truth emerges from the agreement of independently replayed computations.

---

# Part III: Foundations

## 10. Epistemic foundations

The Moments Economy relies on the distinction between human and artificial sources as set out in The Human Mark, an epistemic framework used throughout this wider body of work.

### 10.1 Common Source Consensus

The architecture operates on the principle that all artificial authority and agency are indirect and originate from human intelligence. Accordingly, every governance action above Tier 1 MUST trace to a Direct human source (§10.2).

### 10.2 Classifications

* **Direct Authority:** direct human access to a subject matter, such as observation, expertise, or measurement
* **Direct Agency:** human capacity for comprehension, intention, judgement, and accountable commitment
* **Indirect Authority:** mediated, processed, recorded, or model-generated information
* **Indirect Agency:** artificial processing capacity

Artificial systems may contribute to coordination, interpretation, and record production, but they do not originate authority and they do not bear final accountability.

### 10.3 Displacement risks

Misclassification between Direct and Indirect sources creates four named displacement risks in this framework.

* **GTD:** Governance Traceability Displacement
* **IVD:** Information Variety Displacement
* **IAD:** Inference Accountability Displacement
* **IID:** Intelligence Integrity Displacement

These categories are used to classify events and to audit automated contributions.

## 11. Geometric foundations

The settlement system used in this architecture has a reachable verification space of `4,096` states and a `64`-state horizon (`|H| = 64`) satisfying the holographic identity:

`|H|² = |Ω|`

The economic architecture relies on several properties of this verification space.

### 11.1 Shared moments

When two parties hold the same byte-log prefix and compute the same verification state, they share a structural present independent of external clocks or asserted authorities.

### 11.2 Exact uniformisation

From any state, two consecutive bytes distribute coordination state exactly uniformly across all 4,096 reachable states. Settlement convergence is therefore structurally guaranteed rather than probabilistically approximated.

### 11.3 Intrinsic error detection

The self-dual `[12,6,2]` mask code detects all odd-weight bit errors in states, giving intrinsic corruption detection for coordination records.

### 11.4 Chirality transport

The 6-bit chirality register satisfies an exact transport law that enables early detection of divergence between parties before full state disagreement becomes visible.

### 11.5 Provenance

A state or seal has valid provenance if and only if it is reproducibly reachable from the rest state by the claimed byte history under the public transition law and canonical serialization rules.

The economic architecture uses these invariants as the basis for settlement verification. It does not require semantic consensus in order to establish structural truth.

---

# Part IV: Institutions and Transition

## 12. Registries and settlement

Public programmes support the settlement architecture through three functions.

### 12.1 Registry operation

Programmes MUST maintain registries mapping persons and organisations to eligibility status. These registries bind entries to Identity Anchors.

### 12.2 Recording

Programmes MUST record all distributions as Grants within Shells.

### 12.3 Publication

Programmes MUST publish the associated logs and structural objects. This converts settlement from an internal ledger update into a public, verifiable act.

Banks, payment processors, and digital-wallet providers may act as routing layers for fiat or digital disbursement, but they do not replace the replayable record as the basis of settlement integrity.

## 13. Tier governance

Tier distributions above Tier 1 require higher scrutiny than the universal baseline.

These decisions:

* MUST be made by identifiable human agents,
* MUST be recorded as governance events bound to specific Moments,
* MUST be reversible through subsequent logged events,
* SHOULD reference the genealogical evidence used,
* and MUST preserve traceability from decision to authorising human source.

These rules ensure that tier assignments remain attached to human judgement rather than being silently delegated to opaque automation.

## 14. Interoperability

Interoperability is defined by the ability to replay. Systems are interoperable if they can exchange logs and reproduce one another’s states and structural objects.

Conforming systems MUST:

* use the shared aQPU Kernel specification,
* use canonical byte replay rules,
* use canonical serialization for Identity Anchors, Grants, Shells, and Archives,
* use consistent identifiers for domains, identities, programmes, and periods,
* and preserve sufficient information for independent verification.

## 15. Value and wealth

In this architecture, value is structural coherence rather than debt obligation.

**Wealth** is access to deep, verified genealogies and the ability to navigate coordination space effectively.

**Poverty** is the absence of structural resources, such as access to aligned programmes, registry recognition, or verified genealogical continuity.

**Exchange** within this architecture is a positive-sum coordination act: when aligned actors exchange, they generate shared structural surplus rather than merely transferring fixed value.

## 16. Transition path

A systemic turning point is reached when two conditions hold.

1. UHI distributions occur reliably using replayable genealogies.
2. Displacement remains bounded under increased participation.

Before this point, institutions focus on building replayable records, publication discipline, and verifiable settlement practice.

Transition from legacy systems to the Moments Economy typically follows three phases.

### Phase 1: Measurement

Institutions run pilots to build genealogies, test publication procedures, and establish replay discipline. Settlement may still occur in conventional currencies.

### Phase 2: Parallel distribution

UHI is introduced as a parallel distribution architecture alongside existing currency systems. Registries issue Grants, Shells are published, and the circulation loop becomes publicly auditable. Existing currencies may continue to be used for pricing, contracts, taxation, and banking interfaces.

### Phase 3: Expansion

Tiered distributions are introduced. Additional functions such as grants, pensions, scholarships, stipends, or programme entitlements migrate into MU channels, while legacy systems continue to interoperate where required.

The transition is therefore staged rather than abrupt. The same infrastructure that first supports audit and coordination can later support economic settlement.

---

## Conclusion

The Moments Economy establishes money as a function of coordination capacity rather than credit. Value derives from structural coherence rather than debt obligation. Human agents retain authority and accountability over governance decisions. Artificial systems contribute derivatively within auditable bounds.

The aQPU Kernel provides shared moments and deterministic replay. Grants, Shells, Archives, and genealogies provide verifiable records of distribution and continuity. The Common Source Moment provides an explicit physical capacity envelope within which these operations can occur.

Under the capacity analysis presented here, capacity is not a realistic limiting factor on human timescales. The central challenges are governance quality, registry integrity, publication discipline, and institutional design.

Implementation begins with the aQPU Kernel specification and reference implementation, the AIR coordination infrastructure, and the THM and Gyroscope frameworks referenced throughout this document.

Pilot programmes, public-interest fiscal hosts, NGO channels, research networks, and municipal or institutional experiments may adopt AIR first for coordination and verification, thereby establishing the records on which a Moments Economy can later settle.

**Contact:** [basilkorompilias@gmail.com](mailto:basilkorompilias@gmail.com)
**Repository:** [https://github.com/gyrogovernance](https://github.com/gyrogovernance)

---

## Appendix A: Glossary

**Archive:** A long-horizon aggregation of Shells recording per-identity totals and overall capacity usage.

**Byte Log:** The canonical append-only replay object.

**Common Source Moment (CSM):** The one-time total capacity envelope obtained by coarse-graining the physical microcell count by the settlement system’s reachable verification space.

**Depth-4 Frame Record:** A kernel-native certification atom of the form `(mask48, φ_a, φ_b)`, computed from four consecutive bytes.

**Direct:** A source type indicating direct human authority or agency.

**Event Log:** Application-layer annotations bound to specific states or frames.

**Genealogy:** The byte-complete replay history of an actor, project, or programme, optionally accompanied by event bindings, frame records, and integrity commitments.

**GENE_MAC_REST:** The universal tensor rest state `0xAAA555` from which all aQPU Kernel trajectories begin.

**GENE_MIC_S:** The transcription constant `0xAA` used to compute introns by `byte XOR 0xAA`.

**GGG:** Gyroscopic Global Governance, the governance framework referenced in this document.

**Grant:** A single MU allocation to an identity within a Shell.

**GTD, IVD, IAD, IID:** The four displacement risks.

**Identity Anchor:** A pair consisting of an Identity Identifier and a Kernel Anchor.

**Indirect:** A source type indicating mediated information or artificial processing capacity.

**Moment:** A reproducible verification state at a byte-log prefix.

**Moment-Unit (MU):** The scalar unit of account. One MU corresponds to one minute at the base rate for accounting readability.

**Parity Commitment:** A compact trajectory integrity commitment useful for integrity checking but not a unique history certificate.

**Seal:** A structural commitment for a Shell computed by replaying canonical contents through the aQPU Kernel.

**Shared Moment:** A reproducible verification state computed from a shared byte-log prefix.

**Shell:** A time-bounded capacity container containing Grants and a Seal.

**THM:** The Human Mark, the epistemic framework referenced in this document.

**UHI:** Unconditional High Income, the universal Tier 1 baseline distribution.

---

## Appendix B: Capacity Derivation

This appendix summarises the calculation supporting the capacity claims in Section 2.

### B.1 Verified constants

* `f_Cs = 9,192,631,770 Hz`
* `|Ω| = 4,096`
* `|H| = 64`
* `N_phys ≈ 3.254 × 10³⁰`
* `CSM ≈ 7.944 × 10²⁶ MU`

### B.2 Coverage proof

**Global demand:**

`8.1 × 10⁹ people × 87,600 MU/year ≈ 7.10 × 10¹⁴ MU/year`

**Coverage duration:**

`7.944 × 10²⁶ MU / 7.10 × 10¹⁴ MU/year ≈ 1.12 × 10¹² years`

**Conclusion:**

The fixed CSM supports global UHI for approximately 1.12 trillion years. Capacity is not a realistic limiting constraint on any human timescale relevant to settlement design.

---

## Appendix C: Kernel mechanics summary

The aQPU Kernel operates on a 24-bit state packed as two 12-bit components `(A, B)` from the rest state `0xAAA555`.

A byte transition consists of:

1. transcription by `byte XOR 0xAA`,
2. expansion of the 6-bit payload into a 12-bit mask,
3. mutation of the active component,
4. family-controlled gyration between active and passive components.

The kernel is deterministic, invertible, replayable, and exact under fixed-width integer arithmetic.

Its economic relevance lies in four properties:

* deterministic replay,
* exact reachable-state geometry,
* intrinsic corruption detection,
* and exact convergence under short byte sequences.

These properties make it suitable as a public settlement-verification medium.

### C.1 Shell Seal computation

The Shell Seal is computed from the Shell's canonical contents through the following procedure:

1. Convert each Grant into its canonical receipt: `identity_id || kernel_anchor || amount_mu`.
2. Sort the canonical receipts lexicographically by Identity Identifier.
3. Concatenate the Shell Header and the sorted receipts to form the canonical Shell byte sequence.
4. Route that sequence through a conforming aQPU Kernel instance beginning from `GENE_MAC_REST` (`0xAAA555`).
5. Record the resulting 3-byte state as the Shell Seal.

Because the canonical receipts are sorted before routing, the Shell Seal is invariant to the order in which Grants were originally added to the Shell.