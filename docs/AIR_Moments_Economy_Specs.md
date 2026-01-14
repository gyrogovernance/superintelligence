# Moments Economy Architecture Specification

**Scope:** Economic architecture for coordination-based monetary systems

---
## Introduction

The Moments Economy establishes a monetary system grounded in physical capacity rather than debt. Money in this architecture represents verified coordination that maintains good governance (alignment work). Issuance is therefore constrained not by institutional policy but by the volume and properties of the physical container defined by the atomic standard.

The system operationalises and routes alignment work through a holographic encoding algorithm that provides an efficient and distributable mechanism for global governance coordination. The GGG ASI Alignment Router is a multi-domain network coordination algorithm that establishes the structural conditions for a collective superintelligence governance regime of humans and machines (see Bostrom, Superintelligence, 2014; Korompilias, Gyroscopic Global Governance, 2025). Its kernel outputs are compact routing signatures and governance observables that any party can verify through replay. The Router performs structural transformations rather than interpreting empirical meaning. This ensures results are reproducible, comparable and auditable while keeping decision-making traceable to human agency.

Settlement in this economy does not depend on a central ledger keeper. All distributions are recorded in data structures that bind identity to specific coordinates within the Router's finite state space. Because the Router is deterministic, any party with the transaction logs can replay history to confirm exactly who received what and when. This removes the need for central custodians and replaces institutional trust with cryptographic proof.

The physical anchor for the system is the caesium-133 hyperfine transition frequency. This frequency is used because it is the most widely accepted, reproducible and internationally audited physical reference for measurement, independent of any institution's monetary policy. When this atomic resolution is coarse-grained by the Router's holographic state space, the result is a fixed total volume of distinct, verifiable coordination states. This one-time capacity is called the Common Source Moment (CSM).

The Common Source Moment constitutes the total volume available for these operations. This single capacity supports both the distribution of an Unconditional High Income and the governance of additional tiered distributions that recognise wider scope and higher responsibility. It also supports the preservation of complete governance records. These records include provenance, commitments, consultation, and disputes. Because the capacity is large, settlement does not require compressing or discarding coordination detail, and multiple independent parties can maintain complete records for verification. This supports uses beyond monetary distribution, including scientific research verification, AI model auditing, supply chain traceability, and personal consent tracking.

This document specifies the complete architecture, including the economic units, the structural objects for accounting, the geometric foundations of coordination, and the institutional requirements for transition.

### Why this matters

- **For individuals:** A guaranteed baseline income with additional tiered distributions that recognise wider scope and higher responsibility, delivered through verifiable records rather than debt-based issuance.
- **For policymakers:** Issuance limits derived from explicit physical and geometric assumptions, with parameters that can be inspected, tested, and revised through governance rather than opaque monetary policy.
- **For institutions:** A settlement and audit method where distributions and eligibility decisions are replayable records, reducing reliance on custodians and retrospective narrative dispute.
- **For AI safety:** A coordination substrate that preserves human authority, traceability, and accountability in systems where artificial agents contribute to decisions and record-keeping.

### Scope of the Router

The GGG ASI Alignment Router is used in this document as the settlement and verification layer for the Moments Economy. However, the Router also serves as the coordination backbone for the Alignment Infrastructure Routing (AIR) framework, where it provides coordination states and deterministic replay for grant distribution, work receipts, and project coordination. These uses are independent. Institutions may adopt AIR for coordination without committing to the Moments Economy. The economic architecture described here specifies how the Router can additionally serve as a monetary settlement layer when the conditions for the turning point (Section 16) are met.

### Document structure

**Part I: The Economic Proposition** defines the Moment-Unit, the baseline unconditional income, the participation tiers, and the capacity derivation that supports them.

**Part II: The Architecture** specifies the structural objects used for accounting, the domain structure (economy, employment, education, ecology), and the verification procedures.

**Part III: Foundations** explains the epistemic consensus on human authority and the geometric invariants (K₄ structure, Router properties) that underpin the system.

**Part IV: Institutions and Transition** outlines the requirements for registries, settlement systems, tier governance, and interoperability.

### Related frameworks

The architecture integrates four specifications. Each is referenced where relevant.

**Common Governance Model (CGM):** Geometric theory of coherent measurement identifying the tetrahedral structure (K₄) and target aperture (≈ 0.0207).  
[GitHub: Gyroscopic Alignment Research Lab](https://github.com/gyrogovernance/science)

**Gyroscopic Global Governance (GGG):** Framework applying CGM to the coupled domains of economy, employment, education, and ecology.  
[Document: GGG Paper](https://github.com/gyrogovernance/tools/blob/main/docs/post-agi-economy/GGG_Paper.md)

**The Human Mark (THM):** Epistemic taxonomy distinguishing Original (human) from Derivative (artificial) sources and defining displacement risks.  
[Document: The Human Mark](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM.md)

**GGG ASI Alignment Router:** Deterministic coordination kernel providing shared moments, provenance, and replay.  
[Document: Kernel Specifications](https://github.com/gyrogovernance/tools/blob/main/docs/GGG_ASI_AR_Specs.md)

**Gyroscope Protocol:** Classification framework for work into four categories (governance management, information curation, inference interaction, intelligence cooperation).  

Normative requirements use **MUST**, **SHOULD**, and **MAY** as defined in RFC 2119.

---

# Part I: The Economic Proposition

## 1. The Moment-Unit

The unit of account is the Moment-Unit (MU).

The MU is anchored to time: **one MU corresponds to one minute of capacity at the base rate.**

The base rate is fixed at 60 MU per hour. This definition yields:
- 1,440 MU per day
- 525,600 MU per year

This convention aligns monetary accounting with standard timekeeping. Annual magnitudes remain comparable to familiar salary figures, and the unit avoids dependence on volatile commodity prices or debt instruments.

## 2. Unconditional High Income

Unconditional High Income (UHI) is the baseline distribution provided to every person.

**Definition.** UHI corresponds to four hours per day at the base rate.
- Daily: 240 MU
- Annual: 87,600 MU

**Distinction from Temporal Currencies.** 
The MU is a scalar unit of account denominated in minutes for human readability. The system is not time-based; it is grounded in a fixed geometric volume (the Common Source Moment) derived from the atomic standard. This capacity is a one-time total derived from the phase space of the light-sphere, not a flow generated by the passage of time. Value derives from structural coherence, not labor duration.

**Funding.** UHI is issued within the Common Source Moment capacity envelope. Section 4 shows that the envelope supports this distribution for the entire global population with a resilience margin exceeding 99.999999 percent.

**Mechanism.** Individuals receive UHI through public registries maintained by recognised institutions or fiscal hosts. These registries bind the individual's identity to a structural anchor (an Identity Anchor) and issue Grants within time-bounded Shells. Payment providers then route these Grants into bank accounts or digital wallets. Every step produces a verifiable audit trail that can be independently replayed.

## 3. Participation tiers

Participation tiers provide distributions above the UHI baseline. They recognise contributions that involve wider scope or higher responsibility.

**Tier Schedule.** Tiers are defined as multiples of UHI:

| Tier | Multiple | Annual MU      |
|------|----------|----------------|
| 1    | 1×       | 87,600         |
| 2    | 2×       | 175,200        |
| 3    | 3×       | 262,800        |
| 4    | 60×      | 5,256,000      |

**Capacity Associations.** Tiers correspond to the four Gyroscope capacities:
- **Tier 1 (Intelligence Cooperation):** Maintenance of shared systems and cultural continuity.
- **Tier 2 (Inference Interaction):** Negotiation of meaning and conflict resolution.
- **Tier 3 (Information Curation):** Selection, verification, and contextualisation of information.
- **Tier 4 (Governance Management):** Direction of authority and traceability across systems.

**Governance.** Tier multipliers are governance parameters revisable through institutional processes. Tier assignments must be made by identifiable human agents and recorded in the audit log.

---

## 4. Coordination Capacity: The Common Source Moment

The system capacity is the Common Source Moment (CSM). It is calculated as the phase space volume of the atomic-second light-sphere, coarse-grained by the Router's finite state space. This is a one-time total, not a renewable rate.

### 4.1 Capacity Derivation

**1. The Physical Capacity Standard (f_Cs)**  
The International System of Units (SI) defines the atomic-second via the caesium-133 hyperfine transition. This sets the fundamental frequency reference for the system:  
`f_Cs = 9,192,631,770 Hz`

We use this constant because atomic timekeeping is the most precise, globally audited method humans have for synchronising physical processes. In metrology it quantifies how finely distinct events in a shared causal region can be distinguished and coordinated. The Moments Economy treats that same limit as a finite pool of distinguishable coordination states; alignment work is the verified occupation and transformation of those states.

**2. The Physical Volume (N_phys)**  
The causal container corresponding to the atomic-second is a light-sphere with volume `V = (4/3)π(c × 1s)³`. This volume determines the total capacity; it does not regenerate with each passing second. At the atomic wavelength `λ = c / f_Cs`, the raw physical microcell count is:  
`N_phys = V / λ³ = (4/3)π f_Cs³ ≈ 3.25 × 10³⁰`  
The speed of light cancels out in this equation, creating a purely geometric and frequency-based invariant (verified in `test_physical_microcell_count_closed_form_and_c_cancellation`).

**3. The Common Source Moment (CSM)**  
The Router kernel has ontology size |Ω| = 65,536 (`2¹⁶`). The uniform division CSM = N_phys / |Ω| is forced by symmetry: the Router's 2-byte action is transitive (verified in `test_two_byte_words_form_bijection_to_omega_from_any_start`), and physical isotropy admits no preferred direction. The Common Source Moment is:  
`CSM ≈ 4.96 × 10²⁵ MU`

**Capacity Functionality**

This capacity serves two distinct purposes:

**A. Monetary Distribution**  
- Global UHI demand per year: `≈ 7.10 × 10¹⁴ MU`  
- Coverage: The CSM pool supports global UHI for approximately 70 billion years.  
- Tier distributions: Under realistic tier participation scenarios (where most of the population receives baseline UHI, with smaller percentages at higher tiers), the weighted annual demand ranges from approximately 1.1× to 1.5× the baseline UHI demand. Even with generous tier participation (0.5% at Tier 4), the CSM capacity provides coverage exceeding 47.8 billion years (verified in `test_realistic_tier_distribution_capacity_under_csm`).  
- Adversarial safety: An adversary would need to issue approximately 70,000 times the annual global UHI to consume 1% of total capacity (verified in `test_resilience_margin_and_adversarial_threshold`). This is operationally impossible.

**B. Coordination Records**  
The capacity also supports complete coordination records. These records track:

- **Provenance:** Dependencies between documents, data, models, and decisions. Example: a scientific paper's byte log records which sources were consulted and in what order. Replay verifies the claim.
- **Commitments:** Claims that a dataset is valid, a model is safe, or a guideline is in force. These are bound to Router moments, making them verifiable and disputable.
- **Consultation:** What humans and machines actually used when making decisions. Example: a regulator's decision byte log shows which expert reports were routed through the kernel. Independent parties can replay to confirm.
- **Disputes:** Where institutions diverge. Because disagreement localises to specific byte log differences, disputes are resolvable by comparing logs rather than by adjudicating narratives.

In practice these records are maintained as append-only byte logs and event logs bound to Router moments, with external artefacts referenced by identifiers rather than embedded.

The depth of the capacity allows these records to remain complete rather than aggregated. This enables:

- **Complete genealogies:** No summarisation. Every decision retains its full provenance chain.
- **Independent redundancy:** Multiple institutions maintain their own complete records. No central custodian is required.
- **Fine granularity:** Every consultation event, every micro-decision, and every version is recorded without approaching saturation.

This dual function is what makes the Moments Economy secure. Monetary distributions are traceable because the coordination records are complete. Recovery from fraud involves replaying logs and republishing corrected Shells, not defending a scarce stock of money.

**Implication**  
Capacity scarcity is not a constraint. The limiting factors are governance quality and registry integrity: how genealogies are constructed, how events are classified, and how institutions publish sufficient information for independent verification.

---

# Part II: The Architecture

## 5. The four domains

The architecture organises activity into four coupled domains derived from the GGG framework.

**Economy.** The domain of infrastructure, routing, and settlement. It encompasses the Router, the capacity ledger, and the physical networks enabling circulation.

**Employment.** The domain of work and activity. It encompasses the classification of contributions into governance management, information curation, inference interaction, and intelligence cooperation.

**Education.** The domain of capacity building. It encompasses the development of alignment capacities (GMT, ICV, IIA, ICI) and the detection of displacement risks.

**Ecology.** The domain of circulation and balance. In GGG, ecology integrates the accumulated effects of the other three domains. In this architecture, ecology domain accounting is tracked via Shells and Archives that record the distribution of MU within the structural envelope and monitor the integrity of circulation. Unlike the other three domains, ecology is not maintained as a K₄ edge ledger but as aggregated capacity containers.

These domains map to the vertices of the K₄ graph described in Part III.

## 6. Structural objects

Accounting relies on four standardised data structures.

**Identity Anchor.** A data pair linking an identity to a structural coordinate.
- *Identity Identifier:* A collision-resistant hash of the identity string.
- *Kernel Anchor:* A three-byte Router state derived by routing the identifier from the archetype.

**Grant.** A record of a single MU allocation.
- Contains: Identity label, Identity Identifier, Kernel Anchor, and MU amount.
- Function: The atomic unit of distribution.

**Shell.** A time-bounded capacity container (typically annual).
- **Header:** Contextual label (e.g., `ecology:year:2026`).
- **Capacity Metrics:** Total Capacity (F_total), Used Capacity, and Free Capacity.
- **Seal:** A cryptographic commitment computed by:
  1. Converting all Grants into canonical receipts (Identity ID || Anchor || Amount).
  2. **Sorting** receipts lexicographically by Identity ID.
  3. Concatenating the Header and sorted receipts.
  4. Routing the result through the Router from the archetype.
  5. Recording the final 3-byte state.

**Archive.** A long-horizon aggregation object.
- Function: Aggregates Shells to track per-identity totals and global capacity usage over multiple periods.

## 7. Verification and replay

The defining feature of the Moments Economy is deterministic verification.

**The Replay Procedure.** Any party with access to the published artefacts can verify a Shell:
1. **Ingest:** Load the published Grants and Header.
2. **Reconstruct:** Generate the canonical byte sequence by sorting Grants by identifier.
3. **Execute:** Route the sequence through a conforming Router instance starting from the archetype.
4. **Compare:** Check if the resulting Router state matches the published Seal.

**Outcome.** A match confirms that the data is authentic and the circulation totals are correct. A mismatch indicates that the Header or at least one Grant has been altered.

**Requirement.** Public programmes MUST publish byte logs, event logs, Shells, and Archives at defined intervals to enable this verification.

## 8. Coordination levels

**Individuals.** Any person or organisation acts as an individual node. They run a local Router instance and maintain private logs.

**Projects.** A project defines a shared context for contribution. Participants agree on a canonical byte log and event log. Divergence from these logs results in different Router states, making fork detection automatic.

**Programmes.** A programme aggregates projects under a broader mandate. Programmes maintain references to project genealogies and produce programme-level Shells and Archives.

## 9. Genealogies

A genealogy is the complete structural history of an actor or programme. It consists of:
- The full Byte Log and Event Log.
- The trajectory of Moments (Router states) produced by those logs.
- The time series of derived observables (apertures, domain metrics).

Genealogies function as verifiable assets. A programme can prove its history of alignment and capacity usage by providing its genealogy for replay. New programmes can initialise from the final state of an existing verified genealogy, preserving continuity.

### 9.1 The verification pattern

Verification follows a three-stage pattern that replaces reliance on a central ledger:
- **Local:** Each actor maintains their own Router instance and logs.
- **Published:** Selected genealogies and ledgers are exported as signed bundles.
- **Verified:** Independent parties replay published bundles against the atlas to confirm states and seals.

Truth emerges from the agreement of independently replayed computations.

---

# Part III: Foundations

## 10. Epistemic foundations

The architecture relies on the distinction between human and artificial sources.

**Common Source Consensus.** The system operates on the principle that all artificial authority and agency are Derivative and originate from human intelligence.

**Classifications.**
- **Original Authority:** Direct epistemic access (e.g., eyewitness, expertise).
- **Original Agency:** Human capacity for comprehension, intention, and commitment.
- **Derivative Authority:** Mediated access (e.g., AI outputs, records).
- **Derivative Agency:** Artificial processing capacity.

**Displacement.** Misclassifying a Derivative source as Original (or vice versa) creates displacement risks.
- *GTD:* Governance Traceability Displacement.
- *IVD:* Information Variety Displacement.
- *IAD:* Inference Accountability Displacement.
- *IID:* Intelligence Integrity Displacement.

The system uses these categories to route events and audit automated contributions.

## 11. Geometric foundations

**K₄ Geometry and Exact Ledger.** The system uses the complete graph K₄. The reference implementation uses an exact closed-form projection where P_grad = (1/4) B^T B. Ledgers are stored in integer micro-units (MICRO = 1,000,000). Confidence is applied via fixed-point multiplication in the event update. Projections are computed from fixed constant matrices, avoiding pseudoinverse and SVD tolerance drift. This ensures that aperture calculations are deterministic and identical across all computing platforms.

**Router Properties.** The Router realises this geometry discretely. As verified in the test suite:
- **Intrinsic Aperture:** The kernel's discrete structure yields an intrinsic aperture A_kernel = 5/256 ≈ 0.01953, which approximates the CGM continuous target A* ≈ 0.0207 within 5.6% without parameter fitting (verified in `test_aperture_shadow_a_kernel_close_to_a_star`).
- **Integrity Checks:** The kernel physics supports algebraic tamper detection using trajectory parity (odd/even mask sums) and dual-code syndrome checks, which reliably detect corruption in ledger history.
- **Horizon Coverage:** The horizon set (256 states) has a 1-step neighborhood under all bytes that covers the full bulk Ω exactly (verified in `test_horizon_one_step_neighborhood_covers_full_bulk`).

**Shared Moments.** A shared moment occurs when two parties hold the same log prefix and compute the same Router state. This provides a structural "now" independent of external clocks or authorities.

**Provenance.** Geometric provenance allows verification of state validity. A state is valid if and only if it belongs to the Router's ontology. This provides a check against corruption or invalid transitions without requiring a central registry of all states.

---

# Part IV: Institutions and Transition

## 12. Registries and settlement

Public programmes facilitate settlement through three functions.

**Registry Operation.** Programmes MUST maintain registries mapping persons and organisations to eligibility status. These registries bind entries to Identity Anchors.

**Recording.** Programmes MUST record all distributions as Grants within Shells.

**Publication.** Programmes MUST publish the associated logs and structural objects. This converts settlement from an internal ledger update into a public, verifiable act.

Banks and payment providers act as routing layers, moving MU based on the verified Grants.

## 13. Tier governance

Tier distributions are governance actions. They require higher scrutiny than UHI.

**Requirements.**
- Decisions MUST be made by identifiable human agents (Original Agency).
- Decisions MUST be recorded as governance events bound to specific Moments.
- Decisions MUST be reversible through subsequent logged events.
- Decisions SHOULD reference the genealogical evidence used.

These rules ensure that tier assignments remain traceable to human judgement.

## 14. Interoperability

Interoperability is defined by the ability to replay. Systems are interoperable if they can exchange logs and reproduce each other's states.

**Standards.** Conforming systems MUST:
- Use the shared Router atlas (ontology and epistemology).
- Format byte logs and event logs canonically, as specified by the Router runtime and project format specifications (see `GGG_ASI_AR_Specs.md`).
- Use consistent identifiers for domains and edges.
- Support the standard format for Grants, Shells, and Archives.

## 15. Value and wealth

In this architecture, value is structural coherence rather than debt obligation.
- **Wealth** is access to deep, verified genealogies and the capacity to navigate the coordination space effectively.
- **Poverty** is the absence of structural resources: lacking access to aligned programmes or verified genealogies.
- **Exchange** is a positive-sum coordination act. When aligned actors exchange, they generate shared structural surplus rather than transferring fixed value.

## 16. Transition path

A systemic **turning point** is reached when two conditions hold:
1. UHI distributions occur reliably using replayable genealogies.
2. Displacement remains bounded under increased participation.

Before this point, governance focuses on measurement; after it, focus shifts to allocation and long-horizon integrity.

Transition from legacy systems to the Moments Economy typically follows three phases.

**Phase 1: Measurement.** Institutions run pilots to build genealogies. They publish apertures and displacement metrics but settle in conventional currency. This builds verification capacity.

**Phase 2: Distribution.** UHI is introduced as a parallel distribution. Registries issue Grants, and Shells are published. This establishes the circulation loop.

**Phase 3: Expansion.** Tier distributions are introduced. Additional functions—such as pensions, grants, and scholarships—migrate to MU channels, leveraging the established verification infrastructure.

---

## Conclusion

The Moments Economy establishes money as a function of coordination capacity rather than credit. Value derives from structural coherence rather than debt obligation. Humans retain authority and agency over all governance decisions; artificial systems contribute derivatively within auditable bounds.

The Router provides shared moments and deterministic replay. Shells and Archives provide verifiable records of circulation. Genealogies provide structural histories that can be independently confirmed.

Under the capacity analysis in this document, scarcity of structural capacity is not a binding constraint. The central challenges are governance quality, registry integrity, and institutional design: how genealogies are constructed, how displacement is measured, and how programmes publish sufficient information for independent verification.

The technical substrate is available in open-source form. Implementation begins with the Router specification and reference implementation, the AIR coordination infrastructure, and the THM and Gyroscope frameworks for classification and measurement.

Pilot programmes—including municipal experiments, NGO distribution channels, and research coordination initiatives—can be coordinated through AIR. Policy evaluation frameworks and economic modelling resources are available from the Gyro Governance research team.

**Contact:** basilkorompilias@gmail.com  
**Repository:** https://github.com/gyrogovernance

---

## Appendix A: Glossary

**Aperture:** The ratio of cycle energy to total energy in a K₄ ledger. The CGM target is approximately 0.0207.

**Archetype:** The universal reference state (0xAAA555) from which all Router states derive.

**Archive:** A long-horizon aggregation of Shells recording per-identity totals and global capacity usage.

**CGM (Common Governance Model):** The geometric theory identifying K₄ structure and aperture equilibrium.

**Derivative:** A source type indicating mediated epistemic access or artificial processing capacity.

**GGG (Gyroscopic Global Governance):** The framework applying CGM to economy, employment, education, and ecology.

**Genealogy:** The complete structural history of an actor: byte log, event log, trajectory of Moments, and derived observables.

**Grant:** A single MU allocation to an identity within a Shell.

**GTD, IVD, IAD, IID:** The four displacement risks (Governance Traceability, Information Variety, Inference Accountability, Intelligence Integrity).

**GMT, ICV, IIA, ICI:** The four alignment capacities (Governance Management Traceability, Information Curation Variety, Inference Interaction Accountability, Intelligence Cooperation Integrity).

**Identity Anchor:** A pair consisting of an Identity Identifier (hash) and a Kernel Anchor (Router state).

**K₄:** The tetrahedral graph (complete graph on four vertices) underlying the CGM domain model.

**Moment:** A reproducible Router state configuration, together with any governance events bound to it.

**MU (Moment-Unit):** The unit of account. One MU corresponds to one minute at the base rate (60 MU per hour).

**Original:** A source type indicating direct epistemic access or human agency.

**Router:** The GGG ASI Alignment Router, a deterministic finite-state coordination kernel.

**Seal:** A cryptographic commitment for a Shell, computed by routing the Shell contents through the Router.

**Shell:** A time-bounded capacity window containing Grants and a Seal.

**THM (The Human Mark):** The epistemic taxonomy classifying Original and Derivative sources and defining displacement risks.

**UHI (Unconditional High Income):** The baseline distribution of 240 MU per day to every person.

---

## Appendix B: Capacity Derivation

This appendix provides the detailed calculation supporting the capacity claims in Section 4. Full results in `docs/reports/Moments_Tests_Report.md` (all 27 tests pass).

### B.1 Verified Constants

| Parameter | Value | Source |
|-----------|-------|--------|
| Atomic Reference (f_Cs) | 9,192,631,770 Hz | Caesium-133 hyperfine transition |
| Router Ontology (|Ω|) | 65,536 | Verified in `test_moments_2.py` |
| N_phys (microcells) | 3.254 × 10³⁰ | Derived (4/3)π f_Cs³ |
| CSM (one-time total capacity) | 4.965 × 10²⁵ MU | N_phys / |Ω| (fixed pool, not a rate) |

### B.2 Coverage Proof

**Global Demand:**
8.1 × 10⁹ people × 87,600 MU/year ≈ 7.096 × 10¹⁴ MU/year

**Coverage Duration:**
4.965 × 10²⁵ MU / 7.096 × 10¹⁴ MU/year ≈ 7.0 × 10¹⁰ years

**Conclusion:**
The fixed CSM capacity can support global UHI for approximately 70 billion years. Capacity is not a binding constraint on any human timescale.

---

## Appendix C: Kernel Mechanics

This appendix provides the technical details of the Router kernel implementation.

### C.1 State Model

The Router operates on a 24-bit state packed as two 12-bit components (A, B):
- **Packing:** state24 = (A << 12) | B
- **Archetype:** 0xAAA555 (A=0xAAA, B=0x555)
- **Ontology:** Exactly 65,536 reachable states.

### C.2 Transition Law

The transition T_byte(A, B) is defined by:
1. **Transcription:** intron = byte XOR 0xAA
2. **Expansion:** Expand intron to 12-bit mask_A (Type B mask is always 0).
3. **Mutation:** A' = A XOR mask_A
4. **Gyration:**
   A_next = B XOR 0xFFF
   B_next = A' XOR 0xFFF

### C.3 Algebraic Integrity

The kernel supports fast integrity verification without full cryptographic hashing:
1. **Parity Commitment:** A trajectory is committed to a tuple (O, E, p) where O and E are XOR sums of masks at odd/even steps, and p is the length parity.
2. **Dual-Code Syndrome:** A 16-element dual code C^⊥ exists such that any valid mask m satisfies m · v = 0 for all v in C^⊥. Non-zero syndromes indicate data corruption.

These checks are designed for accidental corruption detection. Adversarial integrity, where an attacker deliberately falsifies records, requires cryptographic hashes and signature verification.