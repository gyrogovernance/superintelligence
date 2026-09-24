# Moments Economy Architecture Specification

## Introduction

The Moments Economy is a monetary and settlement architecture in which the ability to issue money is limited by a publicly verifiable physical capacity. It is grounded in the caesium-133 hyperfine transition frequency (`f_Cs = 9,192,631,770 Hz`), the atomic frequency that also defines the SI second. Because this atomic standard fixes how finely distinct physical events can be resolved, it also sets a physical bound on how much coordination can be recorded and settled within a shared region. Money here is a recorded, replayable allocation of governance capacity within a fixed total envelope.

That envelope is called the Common Source Moment. It is derived from the caesium-133 atomic frequency together with the settlement system's finite set of checkable states. The result is a fixed one-time capacity for recording and settling coordination. Issuance is bounded by that capacity.

Settlement and verification run on a deterministic public kernel (hQVM) and the surrounding runtime components used by this programme. The kernel maps append-only byte histories to reproducible state sequences. Because replay is exact, independently held records can be checked by computation under the public transition rule. Distributions, provenance, consultation, and correction can therefore be published as records that any conforming party can verify by replay.

The operational form of a settlement event is the moment receipt: a short position on that deterministic record, specified by an identity origin (anchor), a step along the history (depth), and a position within a short frame (phase). Proof fields regenerate by replay.

The Moments Economy is an institutional record architecture as well as a distribution system. The same records that support monetary settlement also support complete governance histories: who acted, what was issued, what evidence was referenced, what corrections were made, and where disputes arose. Settlement, audit, provenance, and institutional memory share one replayable medium.

The architecture also has a staged adoption path. Institutions can first adopt the system for coordination, audit, compliance, and traceable programme administration while existing currencies and payment rails continue. In doing so, they build the same complete byte records that later make economic settlement in Moment-Units possible. The transition path is cumulative: coordination and verification first; monetary settlement later, where conditions permit.

This document specifies that architecture. It defines the unit of account, the capacity envelope, the structural objects of settlement, the verification pattern, the domain model, the epistemic commitments, and the institutional requirements for transition.

### Why this matters

* **For individuals:** A guaranteed baseline distribution with additional tiered distributions for wider responsibility, delivered through verifiable records.
* **For policymakers:** Issuance limits derived from explicit physical and geometric assumptions that can be inspected, challenged, and revised through governance.
* **For institutions:** A settlement and audit method in which distributions and eligibility decisions are replayable records.
* **For AI safety:** A coordination medium that preserves human authority, traceability, and accountability in systems where artificial agents contribute to decisions and record-keeping.

### Two capabilities, one infrastructure

The same infrastructure provides two distinct capabilities.

First, it provides an immediate capability: verifiable coordination records for audit, safety, compliance, dispute resolution, and programme administration.

Second, it provides a latent capability: a complete replayable history of distributions and governance actions that can serve as the accounting basis for Moment-Units.

Adoption for the first purpose automatically builds the infrastructure for the second. The same byte logs and event bindings used for coordination and verification can also support economic settlement when the conditions for transition are met.

### Scope and relationship to AIR

The Moments Economy uses the same deterministic settlement and verification machinery that Alignment Infrastructure Routes (AIR) uses for grants, work receipts, and project histories. Those uses are related but distinct. Institutions may adopt AIR for coordination and later adopt the Moments Economy as a settlement architecture. This document specifies the additional economic layer that becomes possible when replayable coordination records are used as the basis for monetary distribution.

### Document structure

**Part I: The Economic Proposition** defines the Moment-Unit, the Common Source Moment, the baseline unconditional distribution, and the participation tiers.

**Part II: The Architecture** specifies the structural objects, the four domains, the verification pattern, and the role of genealogies.

**Part III: Foundations** explains the epistemic commitments and the geometric invariants that underpin the system.

**Part IV: Institutions and Transition** sets out registry, settlement, governance, interoperability, and transition requirements.

### Related frameworks

Later sections refer to companion documents from the same research programme. A reader can treat the short glosses below as labels only; each claim that depends on them is restated in plain terms where it matters.

**Common Governance Model (CGM):** Describes a four-part structure of coherent measurement used when this specification discusses governance domains.

**Gyroscopic Global Governance (GGG):** Applies four governance capacities across economy, employment, education, and ecology.

**The Human Mark (THM):** Distinguishes human (Direct) from artificial (Indirect) Authority and Agency. The canonical Mark block appears in [AIR Moments Economy Whitepaper](AIR_Moments_Economy_Whitepaper.md), Appendix A.

**Settlement kernel (hQVM):** The deterministic public kernel used here for shared moments, provenance, and replay. Surrounding runtime components are named where a specific component matters.

**Gyroscope Protocol:** Classifies work under governance management, information curation, inference interaction, and intelligence cooperation.

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

The MU is a scalar unit of account. The minute denomination is for readability; capacity is drawn from a fixed one-time capacity envelope, the Common Source Moment.

### Legibility convention

MU is denominated at the reference value of 1 MU = 1 international dollar (int$). This is the legibility and fair-rate convention.

A Moment and a Moment-Unit are different kinds of object. In this document, a **Moment** means a reproducible verification state at a given point in a replayable record. A **Shared Moment** means such a state when multiple parties reproduce it from the same record. A **Moment-Unit** is the accounting measure used to denominate distributions within this architecture.

## 2. The Common Source Moment

The system capacity is the Common Source Moment, or CSM. In plain terms: take the caesium-133 atomic frequency as the physical resolution standard, count how many distinguishable physical cells fit in the light-sphere at that atomic wavelength, then divide by the number of checkable settlement states (4,096). The result is a bounded, one-time total settlement capacity. The derivation below states the same claim in the form used for inspection.

### 2.1 Capacity derivation

**1. Physical capacity standard**

The International System of Units defines the SI second via the caesium-133 hyperfine transition frequency:

`f_Cs = 9,192,631,770 Hz`

This atomic frequency is the physical capacity standard for the Moments Economy. The SI second is defined from the same constant. Capacity is fixed by the frequency and geometry.

**2. Physical volume**

At atomic wavelength `λ = c / f_Cs`, the raw physical microcell count in the corresponding light-sphere is:

`N_phys = (4/3)π f_Cs³ ≈ 3.25 × 10³⁰`

The speed of light cancels in this expression, because `c` appears in both the volume and the wavelength expression, yielding a purely geometric and frequency-based invariant.

**3. Division by checkable settlement states**

The settlement system has a finite set of checkable states:

`|Ω| = 4,096`

In this document, checkable means reachable from rest under the public transition rule. Uniform division by that set gives the Common Source Moment:

`CSM = N_phys / |Ω| ≈ 7.94 × 10²⁶ MU`

This is the fixed total capacity envelope for the Moments Economy.

### 2.2 Functional meaning of the capacity

The Common Source Moment serves two distinct functions.

**A. Monetary distribution**

It provides the total capacity envelope within which baseline and tiered distributions can be issued.

**B. Coordination records**

It provides sufficient capacity to preserve complete coordination records, including provenance, consultation histories, commitments, disputes, and corrections.

Because the capacity far exceeds foreseeable demand, institutions can retain complete coordination records, including provenance, consultation histories, commitments, disputes, and corrections. Multiple institutions can maintain complete independent records while remaining far from saturation.

### 2.3 Capacity implications

The practical implications are as follows.

* Global UHI demand per year is `≈ 7.10 × 10¹⁴ MU`.
* The CSM supports global UHI for approximately `1.12 × 10¹² years`.
* Under realistic tier participation assumptions, the duration remains on the order of `10¹²` years.
* Adversarial exhaustion of even 1% of total capacity would require issuance on the order of `1.12 × 10¹⁰` times annual global UHI. On any human, institutional, or civilisational timescale relevant to settlement design, this is operationally impossible.

Tier participation raises demand by the tier multiplier, and the envelope absorbs the strongest possible assumption:

| Universal occupation | Annual demand (MU) | CSM coverage |
|---------------------|--------------------|--------------|
| All people at Tier 1 | 7.0956 × 10¹⁴ | 1.12 × 10¹² years |
| All people at Tier 2 | 1.41912 × 10¹⁵ | 5.60 × 10¹¹ years |
| All people at Tier 3 | 2.12868 × 10¹⁵ | 3.73 × 10¹¹ years |
| All people at Tier 4 | 4.25736 × 10¹⁶ | 1.87 × 10¹⁰ years |

Even with every person on Earth occupying at Tier 4, the envelope covers the full human economy for over eighteen billion years.

The real constraints are governance quality, registry integrity, and publication discipline.

The CSM abolishes the artificial scarcity of the settlement medium. Physical constraints on goods, ecology, and care remain real and are governed through fair-use rules and local coordination.

## 3. Unconditional High Income

Unconditional High Income, or UHI, is the universal baseline distribution provided to every person recorded in a conforming registry under published eligibility rules.

### 3.1 Definition

UHI corresponds to four hours per day at the base rate.

* Daily: 240 MU
* Annual: 87,600 MU

### 3.2 Accounting readability

The MU is denominated in minutes for readability. Value derives from structurally verifiable coordination within a fixed physical-capacity envelope. The minute denomination is an accounting convenience over a one-time geometric capacity.

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

The **economy** domain covers infrastructure, routing, settlement, and circulation. It includes the movement of Moment-Units, the publication of structural records, and the maintenance of the settlement environment. It also includes governance of physical constraints through fair-use rules.

The **employment** domain covers work and contribution. It includes the classification of activity into the four governance capacities used in the Gyroscope framework and the programme-level interpretation of contributions that may justify tiered allocations.

The **education** domain covers capacity formation. It includes the cultivation of the human capabilities required for governance, the development of alignment capacities, and the detection of displacement risks.

The **ecology** domain is the integrative domain of systemic balance. It reflects the accumulated state of the other three domains and is computed from cross-domain records. It may be summarised through Shells, Archives, and other aggregate capacity views.

These four domains correspond to the fourfold coordination structure assumed within the surrounding governance research. The economic architecture uses that correspondence directly.

## 6. Structural objects

Accounting and verification rely on six standardised structural objects.

The **Byte Log** is the canonical append-only sequence of bytes. It is the primary replay object. Every conforming verification procedure depends on the integrity of this log.

The **Event Log** is the application-layer annotation bound to specific verification states or to depth-4 frame records. It records meanings, decisions, classifications, references, and justifications; the kernel routes bytes and leaves interpretation to the application layer.

An **Identity Anchor** links an identity to a fixed starting position on the deterministic settlement record. It consists of an Identity Identifier, which is a collision-resistant hash of the identity string, and an hQVM Anchor, which is the state obtained by routing that identifier from rest through the public kernel. In this specification, the identity bytes routed for anchor derivation are the SHA-256 Identity Identifier; verification regenerates receipt fields from those identity bytes and the payload.

A **Grant** is a record of a single MU allocation. It contains an identity label, an Identity Identifier, a Kernel Anchor, an MU amount, and the relevant shell context. The canonical Grant receipt is:

`identity_id || kernel_anchor || amount_mu`

where `||` denotes concatenation and `amount_mu` is encoded as an unsigned 8-byte big-endian integer. Grant fields, including the amount, are carried in the payload whose routed state forms the moment-receipt seal. The default payload schema is that canonical Grant receipt; other payload schemas are implementation profiles. The receipt position itself carries no amount field. Offline verification and counterparty amount-knowledge therefore require the payload to travel and archive alongside the 16-to-20-byte transport form.

A **Shell** is a time-bounded capacity container, typically annual or programme-bounded. It contains a contextual header, total, used, and free capacity metrics, a set of Grants, and a Seal. The Shell Seal is an order-invariant, deterministically computed commitment over the Shell's canonical contents. The computation procedure is specified in Appendix C. Shell seals are order-invariant container commitments; trajectory coordinates are order-sensitive per-identity positions; the two views are reconciled by replay.

An **Archive** is a long-horizon aggregation object. It aggregates Shells to track per-identity totals, programme totals, corrections, and overall capacity usage across multiple periods.

These objects form the minimum structural vocabulary required for settlement. Byte Logs and Event Logs preserve the replayable history. Identity Anchors and Grants define allocations. Shells and Archives make those allocations publishable and auditable across time.

## 7. Verification and replay

The defining feature of the Moments Economy is deterministic verification.

The settlement system verifies structural integrity and replay consistency. Eligibility, authorisation, and justice remain institutional and human governance functions that must themselves be published in forms suitable for replay and audit.

Any party with access to the published artefacts can verify a Shell through the following procedure:

* Load the published Header and Grants.
* Reconstruct the canonical byte sequence by converting Grants into canonical receipts and sorting them.
* Route that sequence through a conforming hQVM component instance from GENE_MAC_REST.
* Compare the resulting state with the published Seal.

A match confirms that the published structural object corresponds to its canonical contents. A mismatch proves that the Header or at least one Grant differs from the published claim.

Because replay uses exact integer arithmetic and fixed-width bit operations, conforming implementations produce identical results regardless of platform or language.

The system supports three layers of certification relevant to economic verification.

* **Final-state layer:** Shared Moments as reproducible 24-bit verification states.
* **Frame layer:** Depth-4 frame records `(mask48, φ_a, φ_b)`, providing stronger provenance than final states alone, because distinct histories can collapse to the same final state while still producing different frame records.
* **Parity layer:** Compact commitments `(O, E, parity)` for integrity checking and batch verification, but not unique history certificates.

A structurally correct seal confirms canonical contents under replay. Application-layer validity still requires settlement policy: conforming programmes SHOULD reject or flag duplicate identity receipts within a Shell, SHOULD reject or flag Shells whose used capacity exceeds total capacity, SHOULD preserve correction histories, and SHOULD publish the policy basis for eligibility and dispute handling.

Public programmes MUST publish, at defined intervals, the Byte Logs, Event Logs, Shells, Archives, and any frame records or commitments required by their verification policy. Publication in coordinate-ledger form (anchors, depth deltas, occupancy state) satisfies this requirement for kernel-native structural objects that replay regenerates; Event Logs, payloads, and policy bases MUST still be published as data.

## 8. Coordination levels

At the **individual** level, any person or organisation may operate as a node on the settlement network. Each maintains a local kernel instance and its own logs.

At the **project** level, a shared context of contribution is defined. Participants agree on a canonical Byte Log and Event Log. Divergence is detected by replay. Where distinct histories collapse to the same final state, frame records localise the divergence.

At the **programme** level, multiple projects are aggregated under a wider mandate. Programmes maintain references to project genealogies and produce programme-level Shells and Archives. Programme bundles MAY be aggregated by meta-routing, where leaf seals are themselves routed into a higher-level root seal. This enables deterministic multi-programme aggregation with tamper localisation.

## 9. Genealogies

A genealogy is the complete structural history of an actor, project, or programme within this architecture.

It consists of the Byte Log, any bound Event Log, the trajectory of Moments, the depth-4 frame sequence, and any optional compact integrity commitments attached to that trajectory.

A final state alone leaves history underdetermined when distinct byte logs collide. Genealogy-grade audit therefore compares frame sequences as well as final states.

Genealogies function as verifiable assets. A programme can prove its history of alignment and capacity usage by providing its genealogy for replay. New programmes may initialise from the final state of an existing verified genealogy and thereby preserve continuity.

An identity's receipt stream is archived as a trajectory: the identity anchor is stored once, each subsequent event adds one depth delta, and seal, parity, and event-class fields recompute by replay. The manifold address inside the transport time field (the m12 portion of frac32) is state-derived and regenerable; recovery of sec32 and allocation of the intra-bucket discriminator remain open implementation items. The 2.96 TB annual storage figure assumes one depth byte per receipt under time regenerability or equivalent compact time archival; if sec32 must be stored per receipt, storage scales accordingly. The event-class byte on the receipt is a transport chirality/gauge field derived from the payload; it is distinct from the application-layer Event Log, which annotates meaning, decisions, and justifications. At one daily receipt per person, humanity's annual stream is about 2.96 TB of depth deltas plus 24.3 GB of anchors under that assumption. Each trajectory epoch carries a 512-byte occupancy bitmap that detects re-presented coordinates locally within any archive. The measured transport layouts fit in 16 to 20 bytes and fit commodity QR codes; frame-aligned layouts keep the genealogy frame grid stationary across record boundaries. Those measurements, and the open implementation items they leave, are recorded in [Analysis: Moment Receipts, QR Transport, and the FNV Profile](../Findings/Analysis_hQVM_Moments_Fiat.md) from `experiments/hqvm_moments_fiat_analysis_1.py` and `experiments/hqvm_moments_fiat_analysis_2.py`.

Verification follows a three-stage social pattern. It begins locally, where each actor maintains its own verification instance and logs. It extends through publication, where selected genealogies and structural objects are exported as signed bundles. It completes through independent verification, when other parties replay those bundles against the public specification. Agreement comes from independently replayed computations matching.

---

# Part III: Foundations

## 10. Epistemic foundations

The Moments Economy relies on the human–artificial distinction formalised in The Human Mark: Direct Authority and Agency on the human side; Indirect forms on the artificial and mediated side. The canonical Mark block appears in [AIR Moments Economy Whitepaper](AIR_Moments_Economy_Whitepaper.md), Appendix A.

### 10.1 Common Ancestry Constitution

The architecture operates on the principle that all artificial authority and agency are Indirect and constitutively dependent on Human Intelligence. Accordingly, every governance action above Tier 1 MUST trace to a Direct human bearer of Authority or Agency (§10.2).

### 10.2 Classifications

* **Direct Authority:** direct human access to a subject matter, such as observation, expertise, or measurement
* **Direct Agency:** human capacity for comprehension, intention, judgement, and accountable commitment
* **Indirect Authority:** mediated, processed, recorded, or model-generated information
* **Indirect Agency:** artificial processing capacity

Artificial systems may contribute to coordination, interpretation, and record production. Final accountability remains with Direct human Authority and Agency.

### 10.3 Displacement risks

Misclassification between Direct and Indirect classifications creates four named displacement risks in this framework.

* **GTD:** Governance Traceability Displacement - Approaching Indirect Authority and Agency as Direct
* **IVD:** Information Variety Displacement - Approaching Indirect Authority without Agency as Direct
* **IAD:** Inference Accountability Displacement - Approaching Indirect Agency without Authority as Direct
* **IID:** Intelligence Integrity Displacement - Approaching Direct Authority and Agency as Indirect

These categories are used to classify events and to audit automated contributions.

## 11. Geometric foundations

The settlement system used in this architecture has `4,096` checkable states and a `64`-state horizon (`|H| = 64`) satisfying:

`|H|² = |Ω|`

The economic architecture relies on several properties of these checkable states.

### 11.1 Shared moments

When two parties hold the same byte-log prefix and compute the same verification state, they share a structural present independent of external clocks or asserted authorities.

### 11.2 Exact uniformisation

From any state, two consecutive bytes distribute coordination state exactly uniformly across all 4,096 reachable states. Settlement convergence is therefore structurally guaranteed.

### 11.3 Intrinsic error detection

The self-dual `[12,6,2]` mask code detects all odd-weight bit errors in states, giving intrinsic corruption detection for coordination records.

### 11.4 Chirality transport

The 6-bit chirality register satisfies an exact transport rule that enables early detection of divergence between parties before full state disagreement becomes visible.

### 11.5 Provenance

A state or seal has valid provenance if and only if it is reproducibly reachable from the rest state by the claimed byte history under the public transition rule and canonical serialization rules.

The economic architecture uses these invariants as the basis for settlement verification. Structural truth is established by replay under the public transition rule and canonical serialization rules.

---

# Part IV: Institutions and Transition

## 12. Registries and settlement

Public programmes support the settlement architecture through three functions.

### 12.1 Registry operation

Programmes MUST maintain registries mapping persons and organisations to eligibility status. These registries bind entries to Identity Anchors.

### 12.2 Recording

Programmes MUST record all distributions as Grants within Shells.

### 12.3 Publication

Programmes MUST publish the associated logs and structural objects. This converts settlement from an internal ledger update into a public, verifiable act. Coordinate-ledger publication may satisfy the kernel-native structural portion of this requirement under §7; Event Logs, payloads, and policy bases remain data-publication obligations.

Banks, payment processors, and digital-wallet providers may act as routing layers for fiat or digital disbursement. The replayable record remains the basis of settlement integrity.

## 13. Tier governance

Tier distributions above Tier 1 require higher scrutiny than the universal baseline.

These decisions:

* MUST be made by identifiable human agents,
* MUST be recorded as governance events bound to specific Moments,
* MUST be reversible through subsequent logged events,
* SHOULD reference the genealogical evidence used,
* and MUST preserve traceability from decision to authorising human source.

These rules keep tier assignments attached to human judgement, with every decision published in a replayable form.

## 14. Interoperability

Interoperability is defined by the ability to replay. Systems are interoperable if they can exchange logs and reproduce one another’s states and structural objects.

Conforming systems MUST:

* use the shared public kernel and settlement specifications, including the hQVM verification specification,
* use canonical byte replay rules,
* use canonical serialization for Identity Anchors, Grants, Shells, and Archives,
* use SHA-256 for Identity Identifier computation,
* use consistent identifiers for domains, identities, programmes, and periods,
* and preserve sufficient information for independent verification.

Transport layouts, QR enclosure, and the derived name layer for archive append control are implementation profiles layered above replay. Conformance is defined by byte replay and canonical serialization. Manifold addressing uses time-derived coordinates; the name layer names content for append control.

## 15. Value and wealth

In this architecture, value is structural coherence.

**Wealth** is access to deep, verified genealogies and effective movement through coordination space.

**Poverty** is the absence of structural resources, such as access to aligned programmes, registry recognition, or verified genealogical continuity.

Tier 1 baseline occupation follows registry recognition. Genealogy supports higher responsibility and specialist trust relationships. Baseline existence remains available through registry recognition alone.

**Exchange** within this architecture is a positive-sum coordination act: when aligned actors exchange, they generate shared structural surplus.

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

The transition is staged. The same infrastructure that first supports audit and coordination can later support economic settlement.

---

## Conclusion

The Moments Economy establishes money as a function of coordination capacity. Value derives from structural coherence. Human agents retain authority and accountability over governance decisions. Artificial systems contribute derivatively within auditable bounds.

The public kernel and settlement records provide shared coordination for the Moments Economy: shared moments and deterministic replay. Grants, Shells, Archives, and genealogies provide verifiable records of distribution and continuity. The Common Source Moment provides an explicit physical capacity envelope within which these operations can occur.

Under the capacity analysis presented here, capacity leaves headroom on human timescales. The central challenges are governance quality, registry integrity, publication discipline, and institutional design.

Implementation begins with the hQVM Kernel specification and reference implementation, the AIR coordination infrastructure, and the THM and Gyroscope frameworks referenced throughout this document.

Pilot programmes, public-interest fiscal hosts, NGO channels, research networks, and municipal or institutional experiments may adopt AIR first for coordination and verification, thereby establishing the records on which a Moments Economy can later settle.

**Contact:** [basilkorompilias@gmail.com](mailto:basilkorompilias@gmail.com)
**Repository:** [https://github.com/gyrogovernance](https://github.com/gyrogovernance)

---

## Appendix A: Glossary

**Append Gate / Occupancy Bitmap:** The 512-byte, 4096-bit map per trajectory epoch that marks occupied frame coordinates and rejects re-presented depths locally within an archive.

**Archive:** A long-horizon aggregation of Shells recording per-identity totals and overall capacity usage.

**Byte Log:** The canonical append-only replay object.

**Common Source Moment (CSM):** The one-time total capacity envelope obtained by dividing the physical microcell count by the settlement system’s 4,096 checkable states.

**Depth Delta:** The one-byte archive increment that advances an identity trajectory by one receipt after the anchor is stored.

**Depth-4 Frame Record:** A kernel-native certification atom of the form `(mask48, φ_a, φ_b)`, computed from four consecutive bytes.

**Direct:** A source type indicating direct human authority or agency.

**Event-Class Byte:** The receipt transport field derived from payload chirality and gauge. Distinct from the application-layer Event Log.

**Event Log:** Application-layer annotations bound to specific states or frames.

**Genealogy:** The complete replayable byte history of an actor, project, or programme, optionally accompanied by event bindings, frame records, and integrity commitments.

**GENE_MAC_REST:** The universal tensor rest state `0xAAA555` from which all hQVM Kernel trajectories begin.

**GENE_MIC_S:** The transcription constant `0xAA` used to compute introns by `byte XOR 0xAA`.

**GGG:** Gyroscopic Global Governance, the governance framework referenced in this document.

**Grant:** A single MU allocation to an identity within a Shell.

**GTD, IVD, IAD, IID:** The four displacement risks.

**Identity Anchor:** A pair consisting of an Identity Identifier and a Kernel Anchor.

**Implementation Profile:** A transport, QR, or name-layer layout layered above canonical byte replay. Conformance remains defined by replay and canonical serialization; profiles MAY coexist.

**Indirect:** A source type indicating mediated information or artificial processing capacity.

**Moment:** A reproducible verification state at a byte-log prefix. The Whitepaper Appendix B gloss that treats a Moment as a governed coordination event, and the moment receipt as its transport form, names the same object under the settlement and institutional charts respectively.

**Moment Receipt:** A short transport position on a deterministic identity trajectory, specified by an anchor, a depth, and a phase, with seal, parity, and event-class fields regenerable by replay. Measured layouts occupy 16 to 20 bytes. Time-field regenerability beyond the state-derived m12 address is an open implementation item (see §9).

**Moment-Unit (MU):** The scalar unit of account. One MU corresponds to one minute at the base rate for accounting readability. MU is denominated at the reference value of 1 MU = 1 international dollar (int$).

**Name Layer:** A derived content name used for archive append control. It names content; it does not address the manifold. The FNV-1a profile is one optional realisation.

**Parity Commitment:** A compact trajectory integrity commitment useful for integrity checking but not a unique history certificate.

**Seal:** A structural commitment for a Shell computed by replaying canonical contents through the hQVM Kernel.

**Shared Moment:** A reproducible verification state computed from a shared byte-log prefix.

**Shell:** A time-bounded capacity container containing Grants and a Seal.

**THM:** The Human Mark, the epistemic framework referenced in this document.

**Trajectory Epoch:** A bounded segment of an identity trajectory that carries its own occupancy bitmap for local duplicate detection.

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

The fixed CSM supports global UHI for approximately 1.12 trillion years. Capacity leaves headroom on any human timescale relevant to settlement design.

---

## Appendix C: Kernel mechanics summary

The hQVM Kernel operates on a 24-bit state packed as two 12-bit components `(A, B)` from the rest state `0xAAA555`.

A byte transition consists of:

1. transcription by `byte XOR 0xAA`,
2. expansion of the 6-bit payload into a 12-bit mask,
3. mutation of the active component,
4. family-controlled gyration between active and passive components.

The kernel is replayable, invertible, and exact under fixed-width integer arithmetic.

Its economic relevance lies in four properties:

* deterministic replay,
* exact reachable-state geometry,
* intrinsic corruption detection,
* and exact convergence under short byte sequences.

These properties make it suitable as a public settlement-verification medium.

### C.1 Shell Seal computation

The Shell Seal is computed from the Shell's canonical contents through the following procedure:

1. Convert each Grant into its canonical receipt: `identity_id || kernel_anchor || amount_mu`.
2. Sort the canonical receipts lexicographically by the byte representation of the Identity Identifier.
3. Concatenate the Shell Header and the sorted receipts to form the canonical Shell byte sequence.
4. Route that sequence through a conforming hQVM Kernel instance beginning from `GENE_MAC_REST` (`0xAAA555`).
5. Record the resulting 3-byte (24-bit) state as the Shell Seal.

Because the canonical receipts are sorted before routing, the Shell Seal is invariant to the order in which Grants were originally added to the Shell.
