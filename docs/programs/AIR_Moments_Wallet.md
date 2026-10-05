# Pulse: People-Powered Capacity Wallet

*An interface for recognition, routing, and repair under a public event format*

## 1. Purpose and Core Distinction

Pulse is an interface for using Moment-Units (MU), the accounting unit of the Moments Economy. Participants use it to inspect allocations and route claims, with consent and settlement decisions retained in a replayable record. These functions can be accessed through individual devices or assisted and shared terminals.

A settlement may involve the person using an allocation and the counterparty receiving it, together with assistance from an operator or artificial processing. The record must distinguish their contributions and preserve the basis on which the settlement was accepted. Registry operators record identity and continuity for this verification. Shared rules and the procedures for reviewing their application are documented in containers called Shells.

The Human Mark (THM) is the AI safety and alignment framework used to classify these relationships. Authority denotes information available for inference, and Agency denotes the capacity to receive or process it. These are epistemic capacities distributed across providers and receivers throughout the interaction. Human forms are Direct. Artificial forms are Indirect and dependent on human intelligence, with the relationship of dependence termed ancestry.

Wallet procedures must preserve this distribution and ancestry as users access information or obtain assistance. Uniform power distribution follows from continued exercise of the relevant capacities across the interaction. A review examines any attribution of Authority or Agency to a particular bearer as though that bearer exhausted the category, with consequent loss of ancestry measurement. The same requirement applies in household use, assisted access, and institutional services.

## 2. Minimum Viable Pulse

The minimum viable Pulse is the recognition loop that follows the public event format. AI assistants, cloud inference, continuous internet access, fiat integration, biometric identity, and a global central ledger stay optional.

The minimum viable system requires:

1. A public event format.
2. A reference kernel implementation.
3. A pathway that recognises a Direct Source (a person whose observational access and accountable decision remain primary).
4. A Grant and Claim state model.
5. A Shell format for holding Grants, rules, receipts, and seals.
6. At least one acceptance circle.
7. At least one active terminal capable of scanning or recording passive access surfaces.
8. A local event log that can later be replayed and synchronised.
9. A public or community surface for reconciling accepted claims.
10. A repair pathway for duplicate, mistaken, or contested claims.

With these elements, Pulse can operate through paper QR codes, passive cards, shared terminals, or simple devices. AI improves interpretation and governance support. Baseline recognition, routing, and settlement run on the public-event recognition loop alone.

Acceptance Circle: A bounded group of counterparties, suppliers, community terminals, or institutions that agree to recognize claims that follow the public event format under shared Shell rules and repair procedures.

## 3. Protocol Event Format

The public event format defines the minimum structure required for replay, settlement, repair, and interoperability.

A Pulse event packet may include:

*   protocol version,
*   event type,
*   source anchor,
*   Shell reference,
*   Grant or Claim reference,
*   prior state reference,
*   routed amount where relevant,
*   recipient or counterparty anchor where relevant,
*   Offer Shell or resource reference where relevant,
*   applicable Community Shell rule reference,
*   consent scope where relevant,
*   expiry or challenge window,
*   witness or attestation method,
*   kernel transition output,
*   canonical byte serialisation,
*   event hash or packet identifier,
*   Shared Moment reference,
*   seal or signature.

The people affected by a decision assess it within the applicable shared arrangements. Shell records identify the rules and review procedures through which participants can examine its basis and consequences.

For replay to be reliable, the format must define canonical byte serialisation. The same event packet must produce the same kernel transition across independent implementations.

## 4. Core Wallet Objects

The wallet operates on a defined set of structural objects.

*   **Identity Anchor:** A persistent continuity commitment for a Direct Source (the recognised person).
*   **Grant:** MU capacity issued for a period or recognised occupation.
*   **Claim Record:** The operational state for one route-eligible portion of a Grant.
*   **Shell:** A container holding Grants, rules, routes, receipts, and seals.
*   **Decision Attestation:** A user-approved action bound to a Shared Moment (a verification state reproduced by the parties from the same record). It records the intent class, routed amount, recipient anchor, and resulting kernel transition.
*   **Settlement Receipt:** Evidence that the counterparty accepted routed capacity.
*   **Consent Attestation:** Scoped permission for data preservation or use.
*   **Community Shell:** A local governance container for fair-use rules and shared resources.
*   **Project Shell:** A time-bound container for building, repair, care, production, or infrastructure.
*   **Offer Shell:** A discoverable offer statement that binds conditions and constraints to a routeable capacity target.
*   **Genealogy:** The complete, replayable byte record of a user's decisions, attestations, and synchronisations over time.

## 5. Presence, Identity, and Continuity

Presence synchronisation makes current capacity operationally available inside the wallet. Tier 1 occupation follows registry recognition of the person for settlement. Direct Source names the person's existing capacity for observation and accountable decision, with registry entries preserving its continuity across recorded interactions.

### Entry

New users enter through controlled vouching, assisted entry, and provisional recognition pathways. Existing recognised Direct Sources attest to continuity and context, after which the user gains provisional operational status until continuity stabilises.

### Continuity

The wallet uses persistent anchors, key rotation, and migration-safe device binding to preserve continuity over device changes and key refresh events.

No identity mechanism may make Tier 1 occupation conditional on surrendering unnecessary personal data, biometric capture, state-issued documents, behavioural monitoring, or continuous device possession.

### Recovery

Recovery covers lost devices, compromised wallets, social recovery, and community recovery when continuity is disrupted. Recovery is bounded by challenge, review, and proportional restoration, with the presumption of restoring baseline occupation first, then validating continuity.

### Challenge

Duplicate claims, vouching abuse, and wrongful exclusion are handled as auditable dispute pathways. Challenge outcomes remain open to review and appeal before any lasting exclusion.

### Guardianship

Children, elders, disabled users, and people without devices may use assisted access. Guardianship has a defined scope and remains reviewable and revocable. The record must preserve the supported person's observations and expressed choices alongside the assistance provided.

## 6. Grant and Claim State Lifecycle

The wallet prepares MU claims from granted capacity. Occupation continues whether or not the app is opened frequently.

The wallet synchronises the user's Shell with Grants issued for the current period. If a Grant has not yet been made operational in the local Shell, the wallet prepares and records the claim state. Failure to open the wallet delays operational use until synchronisation occurs; baseline occupation remains in force.

Each claim follows a bounded state model. A typical successful path is:

`issued -> available -> reserved -> routed -> accepted -> retired`

*   `issued`: a Grant record is created according to the recognised issuance rule for the relevant Shell or occupation tier.
*   `available`: grant is ready in the local Shell and eligible for routing.
*   `reserved`: capacity has been allocated for a specific proposed settlement and cannot be spent elsewhere.
*   `routed`: counterparty has received the claim package and the settlement pathway is active.
*   `accepted`: counterparty has confirmed the settlement and acceptance has been recorded.
*   `suspended`: settlement is paused by dispute, risk signal, or governance hold.
*   `retired`: claim is completed, reversed by final settlement, refunded, expired, or otherwise removed from circulation.

The `suspended` state may branch from `available`, `reserved`, `routed`, or `accepted` when a dispute, risk signal, governance hold, or reconciliation conflict arises.

For Tier 1, issuance follows recognised Direct Source status. For higher tiers, issuance follows recognised occupation records, role attestations, Project Shell participation, or Community Shell rules.

## 7. Transaction Lifecycle

A settlement or governance action follows a defined sequence.

1.  The user expresses intent.
2.  The wallet interprets intent locally.
3.  The wallet checks available claims and relevant Shell rules.
4.  If capacity is insufficient or a rule is breached, routing is declined and the user is informed.
5.  If present, the AI explains consequences and action risk; otherwise the terminal or operator presents the rule consequences directly.
6.  The user creates a Decision Attestation.
7.  The local hQVM Kernel computes the event-state transition from the decision bytes that follow the public event format.
8.  The wallet updates the relevant Claim state locally to `reserved` or `routed` according to Shell rules, pending counterparty verification.
9.  The counterparty independently computes the event-state transition and verifies the Shared Moment.
10. A Settlement Receipt is created and linked to the claim state.
11. The counterparty and public surface reconcile the accepted claim.
12. The genealogy updates.

## 8. Settlement Finality and Offline Use

MU moves as a live claim state inside a Shell.

When capacity is routed, the relevant amount is reserved against the sender's Shell. Once counterparty acceptance occurs, the Settlement Receipt changes state from reserved to accepted.

A claim may be presented more than once in offline or disrupted conditions. Only one pathway receives final recognised settlement after replay and reconciliation. Local archives apply the occupancy bitmap append gate (512 bytes per trajectory epoch) to reject re-presented coordinates before wider reconciliation. Conflicting presentations become repair events. Repair restores continuity; it leaves punishment and exclusion as separate governance acts.

Offline settlements are possible using provisional states and proximity or local trust channels. Provisional settlements carry a time limit and a counterparty-risk flag. Final recognition occurs only after Shell state is synchronised and independently replayed in the public or community surface.

A passive surface requires a local confirmation method where possible, such as gesture, PIN, voice, witness attestation, assisted confirmation, or recognised local custom. The purpose is preservation of a traceable link between the settlement event and the Direct Source.

## 9. Structured Occupation

Through the wallet, people record allocations for the contributions they make. A person may exercise all four capacities within one task, including care and informal cooperation. The allocation schedule recognises their scope and continuing responsibility under shared rules.

*   **Tier 1 (Intelligence Cooperation):** The wallet receives and displays the daily unconditional capacity. It secures existence.
*   **Tier 2 (Inference Interaction):** The wallet routes capacity recognised through mediation, care, teaching, and human review of artificial outputs.
*   **Tier 3 (Information Curation):** The wallet routes capacity recognised through research, verification, data stewardship, and contextualisation.
*   **Tier 4 (Governance Management):** The wallet routes capacity recognised through traceability of decisions and coordination of shared responsibilities.

Tier 2, 3, and 4 streams are recognised through occupation records, accepted roles, contribution continuity, community attestations, and responsibility-bearing activity.

Integrity requirements are proportional to consequence. Tier 1 baseline occupation prioritises access and restoration. Higher-risk actions, such as guardianship changes, large Project Shell routing, Tier 3 or Tier 4 responsibility streams, identity recovery, or Community Shell rule approval, require stronger attestation and longer review paths.

The system is permissive at the level of survival access and stricter at the level of delegated responsibility.

## 10. Structural Wealth and Genealogy

Wealth in the Moments Economy is the depth and continuity of a user's genealogy. MU is continuously issued and weak for hoarding; genealogy is the durable structural resource.

Genealogy is continuity evidence used for specific roles, responsibilities, and trust relationships where replayable history is relevant.

Tier 1 occupation follows registry recognition. Different communities or Project Shells may recognise different parts of a genealogy for specific purposes, and baseline occupation remains available independently of genealogical depth.

Genealogy should support selective disclosure. Pulse discloses only the proof, receipt, role record, or Shell relation required for the specific context.

## 11. The Legibility Convention

Pulse may display MU using the reference denomination 1 MU = 1 international dollar (int$) for legibility and fair-rate presentation.

Settlement remains in MU under Shell rules.

The convention enables local pricing and fairness comparison in a familiar unit. Local prices may still reflect local production conditions, transport, ecology, scarcity, and fair-use rules. Local governance remains required.

## 12. Fair-Use Governance and Price

In the Moments Economy, price is an administrative and informational signal. Physical constraints on essentials, ecology, housing, and care are governed through Community Shell fair-use rules.

For essential resources, participants apply the fair-use rules recorded in the relevant Community Shell. During routing, the software checks the rules applicable to the resource and settlement. These may include maximum daily allocation limits, local-priority access during disruptions, household-based caps, ecological limits, reservation windows, and waiting lists.

Fair-use rules are held in Community Shells. Each rule has a scope, affected resource, duration, issuer, revision history, appeal process, and sunset condition. The wallet enforces only rules valid for the resource and transaction context.

## 13. Community, Project, and Offer Shells

Participants use Community Shells to record arrangements for shared resources and the responsibilities involved in managing them. People providing or receiving a service exercise authority and agency through their own contributions, with the Shell preserving the agreed scope of each decision and its review procedures.

### Distribution of capacities in shared rules

Participants must be able to exercise the relevant capacities in the adoption and review of Community Shell rules. Power concentration occurs when a rule assigns Authority or Agency to a particular bearer as though that bearer exhausted the category. The following conditions preserve access to the evidence and procedures needed for review. Community Shell rules must be:

*   Public and inspectable.
*   Scoped to explicit resource or context.
*   Time-bounded, with mandatory expiry and renewal conditions.
*   Appealable and replayable.
*   Auditable for source, revision history, and enforcement path.

Rules that affect essentials need stronger justification than rules on optional or surplus goods. Emergency rules must expire unless ratified. Supplier-specific rules must not override baseline human occupation. Any participant affected by a rule can inspect rule source, duration, revision history, and appeal path.

### Project Shells

Participants record production projects in Project Shells through the following lifecycle:

proposal -> capacity target -> participants -> milestones -> routing schedule -> attestations -> dispute path -> completion seal -> archive.

Projects can include housing builds, farm seasons, school repairs, energy microgrids, care rotas, local manufacturing, and ecological restoration.

Participants apply fair-use rules to immediate constraints and use Project Shells and Offer Shells to organise additional production. They can document an unmet need, invite contributions, and agree on allocations for the proposed work. The record should preserve the observations of those affected by the constraint alongside the production plan.

### Offer Shells

The wallet supports Offers as well as payments. Suppliers, workers, projects, and communities publish Offer Shells with price range, constraints, fair-rate assumptions, recipient conditions, and expiry. Users route MU in response to offers. Supply becomes visible through Offer Shells.

## 14. Consent as a Governance Act

Data preservation requires consent as a governance decision, separate from settlement.

When a project or external entity requests data access, the wallet presents the request clearly, specifying requested data, purpose, duration, steward, and withdrawal rights.

If approved, the wallet generates a Consent Attestation bound to a Shared Moment and recorded in the genealogy. Obligations include purpose limitation, access control, expiry, auditability, non-transfer without renewed consent, and withdrawal handling.

## 15. Wallet Architecture and AI Role

The wallet is a distributed interface stack that preserves explicit authority boundaries. Only the recognition and settlement loop that follows the public event format is required for minimum viable operation; AI and cloud services are optional support layers.

*   **Local settlement layer:** Holds Identity Anchor, local hQVM Kernel, and Shell state. It supports offline computation, signing, proximity exchange, and provisional settlement.
*   **Edge AI model:** Handles voice/text intent, action classification, rule checks, and consequences explanation.
*   **Routing layer:** Uses local protocols and tools for context-sensitive referrals and scheduling.
*   **Extended governance layer:** Escalates policy interpretation or multi-party negotiation to stronger models when needed, with minimum context only.
*   **Consent vault:** Optional encrypted local storage for user-chosen preservation.

AI components are strictly Indirect Agency and Indirect Authority. They may suggest, explain, route, summarise, and model consequences.

Action classes:

*   **Low-risk actions:** reminders, summaries, recurring small allocations within user-defined limits.
*   **Medium-risk actions:** purchases, data-sharing requests, role acceptances, Project contributions.
*   **High-risk actions:** identity recovery, large allocations, guardianship changes, long-term consent, dispute escalation, role escalation, and Community Shell rule approval.

Low-risk actions may be assisted when user policy allows.
Medium-risk actions require explicit confirmation.
High-risk actions require fresh Direct attestation and may require additional community or counterparty attestation.

The AI cannot:

*   originate authority,
*   finalise settlement,
*   override fair-use governance,
*   provide final attestation for routing decisions.

### Settlement layers

Pulse routes Moment-Units as occupation of the common capacity under Shell rules. Grants and Claims on the replayable record express that occupation; acceptance by counterparties makes it usable in practice.

Conventional currency and Fiat Pools handle obligations that still clear outside MU. Sponsors, banks, and fiscal hosts may disburse fiat while Shells and genealogies record the related MU entitlements and programme history. Application-specific artefacts, such as verified human oversight records sold into funded programmes, are priced and governed under those programmes; they are not the unit of the settlement medium itself.

Where artificial assistance is enabled, it remains Indirect Agency and Indirect Authority throughout routing and presentation. Optional inference or hosted weight builds may assist scan, interpret, or classify events when participants publish the metadata needed to reproduce a classification; sealed receipt coordinates still verify by public replay alone.

## 16. Fiat Boundary

A Fiat Pool is a boundary fund for obligations that still settle outside MU. Participation records MU contribution for governance and allocation. Fiat obligations are paid from separately held external funds under pool governance and legal responsibilities. MU contribution stays outside reserve accounting and outside backing of those fiat payment obligations.

The MU transfer to a Fiat Pool is an internal recognition of MU contribution for governance and allocation processes; the fiat payment is made from the pool's own external treasury.

MU routed to a Fiat Pool is retired from circulation or treated as a community expense for audit continuity. Pool fiat payments draw on the pool's external treasury.

## 17. Dispute and Repair

The wallet supports dispute and repair when settlement is contested. A dispute creates a review path where parties can attest, contest, amend, reverse, or compensate according to Community Shell and Project Shell rules.

Failed settlements, fraud claims, coercion, unfair enforcement, and corruption are treated as repair events, not as permanent exclusion.

If no local repair resolves the dispute, capacity remains in suspended state until a recognised governance process supplies a binding finality pathway.

## 18. Accessibility and Assisted Access

The wallet must support voice-first use, low-literacy interfaces, offline proximity settlement, shared community terminals, and fallback channels including paper or card receipts.

Assisted access must preserve Direct Source where possible. Guardians, carers, interpreters, or community operators may help operate the interface, but assistance is distinct from authority transfer.

Where substitute decision support is necessary, it must be scoped, reviewable, and revocable.

## 19. Implementation Plurality and Protocol Governance

Participants may use any wallet implementation conforming to the public protocol. Conformance includes preservation of the distributed exercise of capacities and traceability of their ancestry across providers and receivers. The event record documents the scope of each contribution and the applicable review procedures.

Protocol upgrades must be versioned, public, backwards-aware, and replayable. Each attestation must disclose protocol version used.

No private provider may silently change settlement semantics. Where protocol forks occur, Shells must expose which versions they recognise and which version is used for each attestation.
