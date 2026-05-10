# Pulse: People-Powered Capacity Wallet

*A grammar-compliant interface for recognition, routing, and repair*

## 1. Purpose and Core Distinction

Pulse is the civic interface through which recognised human capacity becomes operational. It does not create entitlement, issue value by discretion, or govern people from above. It synchronises a Direct Source with the Shared Moment, prepares decisions, records consent, routes Moment-Units (MU), applies fair-use rules, and preserves a replayable genealogy of agency. AI may assist interpretation and routing, but authority remains with the human and with the community Shells that coordinate shared constraints.

The wallet is not the source of MU. The wallet is the interface of occupation.

The registry does not create the human's entitlement. It only makes human entitlement operationally recognisable within the network.

Pulse is not designed around the question, "How do we protect scarce money from false claimants?" It is designed around the question, "How do we make human capacity accessible, replayable, and repairable without allowing institutions, platforms, or AI systems to become the source of authority?"

## 2. Minimum Viable Pulse

The minimum viable Pulse does not require an AI assistant, cloud inference, continuous internet access, fiat integration, biometric identity, or a global central ledger.

The minimum viable system requires:

1. A public event grammar.
2. A reference kernel implementation.
3. A Direct Source recognition pathway.
4. A Grant and Claim state model.
5. A Shell format for holding Grants, rules, receipts, and seals.
6. At least one acceptance circle.
7. At least one active terminal capable of scanning or recording passive access surfaces.
8. A local event log that can later be replayed and synchronised.
9. A public or community surface for reconciling accepted claims.
10. A repair pathway for duplicate, mistaken, or contested claims.

With these elements, Pulse can operate through paper QR codes, passive cards, shared terminals, or simple devices. AI improves interpretation and governance support, but it is not required for baseline recognition, routing, or settlement. The minimum viable system is the grammar-valid recognition loop.

Acceptance Circle: A bounded group of counterparties, suppliers, community terminals, or institutions that agree to recognize grammar-valid claims under shared Shell rules and repair procedures.

## 3. Protocol Event Grammar

Pulse events are valid when they comply with the public event grammar. The grammar defines the minimum structure required for replay, settlement, repair, and interoperability.

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

The grammar does not judge whether a decision is morally good, socially wise, or economically desirable. It ensures that the event has a replayable structure. Communities, counterparties, and Shells interpret the event according to their recognised rules.

For replay to be reliable, the grammar must define canonical byte serialisation. The same event packet must produce the same kernel transition across independent implementations.

## 4. Core Wallet Objects

The wallet operates on a defined set of structural objects.

*   **Identity Anchor:** A persistent continuity commitment for a Direct Source.
*   **Grant:** MU capacity issued for a period or recognised occupation.
*   **Claim Record:** The operational state for one route-eligible portion of a Grant.
*   **Shell:** A container holding Grants, rules, routes, receipts, and seals.
*   **Decision Attestation:** A user-approved action bound to a Shared Moment. It records the intent class, routed amount, recipient anchor, and resulting kernel transition.
*   **Settlement Receipt:** Evidence that the counterparty accepted routed capacity.
*   **Consent Attestation:** Scoped permission for data preservation or use.
*   **Community Shell:** A local governance container for fair-use rules and shared resources.
*   **Project Shell:** A time-bound container for building, repair, care, production, or infrastructure.
*   **Offer Shell:** A discoverable offer statement that binds conditions and constraints to a routeable capacity target.
*   **Genealogy:** The byte-complete, replayable record of a user's decisions, attestations, and synchronisations over time.

## 5. Presence, Identity, and Continuity

Presence synchronisation makes current capacity operationally available inside the wallet. Tier 1 occupation follows from recognised Direct Source status in the registry, not from app usage.

### Entry

New users enter through controlled vouching, assisted entry, and provisional recognition pathways. Existing recognised Direct Sources attest to continuity and context, after which the user gains provisional operational status until continuity stabilises.

### Continuity

The wallet uses persistent anchors, key rotation, and migration-safe device binding to preserve continuity over device changes and key refresh events.

No identity mechanism may make Tier 1 occupation conditional on surrendering unnecessary personal data, biometric capture, state-issued documents, behavioural monitoring, or continuous device possession.

### Recovery

Recovery covers lost devices, compromised wallets, social recovery, and community recovery when continuity is disrupted. Recovery is bounded by challenge, review, and proportional restoration, with the presumption of restoring baseline occupation first, then validating continuity.

### Challenge

Duplicate claims, vouching abuse, and wrongful exclusion are handled as auditable dispute pathways. Challenge outcomes do not convert into permanent exclusion without review and appeal.

### Guardianship

For children, elders, disabled users, or people without devices, the model supports substitute assistance. Guardianship is scoped, reviewable, and revocable, and it remains a support role rather than a transfer of Direct Source authority.

## 6. Grant and Claim State Lifecycle

The wallet prepares MU claims from granted capacity without making occupation depend on app-open frequency.

The wallet synchronises the user's Shell with Grants issued for the current period. If a Grant has not yet been made operational in the local Shell, the wallet prepares and records the claim state. Failure to open the wallet does not cancel baseline occupation; it only delays operational use until synchronisation occurs.

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
7.  The local aQPU Kernel computes the event-state transition from the grammar-valid decision bytes.
8.  The wallet updates the relevant Claim state locally to `reserved` or `routed` according to Shell rules, pending counterparty verification.
9.  The counterparty independently computes the event-state transition and verifies the Shared Moment.
10. A Settlement Receipt is created and linked to the claim state.
11. The counterparty and public surface reconcile the accepted claim.
12. The genealogy updates.

## 8. Settlement Finality and Offline Use

MU does not move as a loose object. It moves as a live claim state inside a Shell.

When capacity is routed, the relevant amount is reserved against the sender's Shell. Once counterparty acceptance occurs, the Settlement Receipt changes state from reserved to accepted.

A claim may be presented more than once in offline or disrupted conditions. The system does not pretend this is impossible. Instead, only one pathway can receive final recognised settlement after replay and reconciliation. Conflicting presentations become repair events. They do not automatically create punishment or exclusion.

Offline settlements are possible using provisional states and proximity or local trust channels. Provisional settlements carry a time limit and a counterparty-risk flag. Final recognition occurs only after Shell state is synchronised and independently replayed in the public or community surface.

A passive surface is not treated as sufficient authority by itself. Where possible, the terminal records a local confirmation method, such as gesture, PIN, voice, witness attestation, assisted confirmation, or recognised local custom. The purpose is not defensive identity control, but preservation of a traceable link between the settlement event and the Direct Source.

## 9. Structured Occupation

The wallet manages the full structure of occupation.

*   **Tier 1 (Intelligence Cooperation):** The wallet receives and displays the daily unconditional capacity. It secures existence.
*   **Tier 2 (Inference Interaction):** The wallet routes capacity recognised through mediation, care, teaching, and human review of artificial outputs.
*   **Tier 3 (Information Curation):** The wallet routes capacity recognised through research, verification, data stewardship, and contextualisation.
*   **Tier 4 (Governance Management):** The wallet routes capacity recognised through leadership, oversight, and institutional coordination.

Tier 2, 3, and 4 streams are recognised through occupation records, accepted roles, contribution continuity, community attestations, and responsibility-bearing activity.

Integrity requirements are proportional to consequence. Tier 1 baseline occupation prioritises access and restoration. Higher-risk actions, such as guardianship changes, large Project Shell routing, Tier 3 or Tier 4 responsibility streams, identity recovery, or Community Shell rule approval, require stronger attestation and longer review paths.

The system is permissive at the level of survival access and stricter at the level of delegated responsibility.

## 10. Structural Wealth and Genealogy

Wealth in the Moments Economy is not the accumulation of MU, which is continuously issued and structurally weak for hoarding. Wealth is the depth and continuity of a user's genealogy.

Genealogy is not a score. It is not a behavioural rating, credit profile, compliance record, or moral ranking. It is continuity evidence used for specific roles, responsibilities, and trust relationships where replayable history is relevant.

It does not affect Tier 1 occupation. Different communities or Project Shells may recognise different parts of a genealogy for specific purposes, but no genealogy may be used to deny baseline occupation.

Genealogy should support selective disclosure. A user should not need to reveal their full history to make a purchase, join a project, prove continuity, or access a service. Pulse should disclose only the proof, receipt, role record, or Shell relation required for the specific context.

## 11. The Legibility Convention

Pulse may display MU using a 1 MU = 1 international dollar reference convention for legibility and fair-rate dialogue.

This is a purchasing-power reference, not a settlement currency, exchange rate, peg, redemption promise, or fiat claim.

The convention enables local pricing and fairness dialogue without requiring everyone to learn a new unit. Local prices may still reflect local production conditions, transport, ecology, scarcity, and fair-use rules. The convention supports legibility; it does not eliminate the need for local governance.

## 12. Fair-Use Governance and Price

In the Moments Economy, price is demoted from the primary governor of access to an administrative and informational signal. Price inflation is treated as an insufficient governance response to physical constraint.

The wallet replaces exclusionary pricing with fair-use rules in essentials contexts. When constraints exist, it checks local Community Shells for routing rules. These may include maximum daily allocation limits, local-priority access during disruptions, household-based caps, ecological limits, reservation windows, and waiting lists.

Fair-use rules are held in Community Shells. Each rule has a scope, affected resource, duration, issuer, revision history, appeal process, and sunset condition. The wallet enforces only rules valid for the resource and transaction context.

## 13. Community, Project, and Offer Shells

Community Shells coordinate shared constraints, local priorities, supplier relations, ecological limits, and collective allocations. Community Shells coordinate constraints; they do not own persons, entitlements, or Direct agency.

### Anti-Capture Governance

Community Shell governance can itself be captured. To prevent local cartelisation, arbitrary exclusion, and community oligarchy, Community Shell rules must be:

*   Public and inspectable.
*   Scoped to explicit resource or context.
*   Time-bounded, with mandatory expiry and renewal conditions.
*   Appealable and replayable.
*   Auditable for source, revision history, and enforcement path.

Rules that affect essentials need stronger justification than rules on optional or surplus goods. Emergency rules must expire unless ratified. Supplier-specific rules must not override baseline human occupation. Any participant affected by a rule can inspect rule source, duration, revision history, and appeal path.

### Project Shells

Project Shells coordinate production and follow a defined lifecycle:

proposal -> capacity target -> participants -> milestones -> routing schedule -> attestations -> dispute path -> completion seal -> archive.

Projects can include housing builds, farm seasons, school repairs, energy microgrids, care rotas, local manufacturing, and ecological restoration.

Fair-use rules manage immediate constraints. Project Shells and Offer Shells convert unmet demand into production coordination. If food, housing, care, tools, or infrastructure are insufficient, Pulse should not merely deny excess routing. It should expose the constraint as a coordination opportunity: create offers, invite contributors, open Project Shells, route surplus MU, and organise the work needed to expand real capacity.

### Offer Shells

The wallet supports Offers as well as payments. Suppliers, workers, projects, and communities publish Offer Shells with price range, constraints, fair-rate assumptions, recipient conditions, and expiry. Users route MU in response to offers. This makes supply visible without depending on advertising, extraction, or speculative pricing.

## 14. Consent as a Governance Act

Data preservation is not inherently extractive, but it must be governed. The wallet treats consent as a specific governance decision, separate from settlement.

When a project or external entity requests data access, the wallet presents the request clearly, specifying requested data, purpose, duration, steward, and withdrawal rights.

If approved, the wallet generates a Consent Attestation bound to a Shared Moment and recorded in the genealogy. Obligations include purpose limitation, access control, expiry, auditability, non-transfer without renewed consent, and withdrawal handling.

## 15. Wallet Architecture and AI Role

The wallet is a distributed interface stack that preserves explicit authority boundaries. Only the grammar-compliant recognition and settlement loop is required for minimum viable operation; AI and cloud services are optional support layers.

*   **Local settlement layer:** Holds Identity Anchor, local aQPU Kernel, and Shell state. It supports offline computation, signing, proximity exchange, and provisional settlement.
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

AI tokens are not the currency of the Moments Economy.

## 16. Fiat Boundary

A Fiat Pool is not a convertibility service. It is a boundary fund for obligations that cannot yet be settled in MU.

Participation in a Fiat Pool does not create a right to redeem MU for fiat. Fiat obligations are paid from separately held external funds under pool governance and legal responsibilities.

The MU transfer to a Fiat Pool is an internal recognition of MU contribution for governance and allocation processes; the fiat payment is made from the pool's own external treasury.

MU routed to a Fiat Pool is retired from circulation or treated as a community expense for audit continuity. It does not sit in reserve and does not back fiat payment obligations.

## 17. Dispute and Repair

The wallet supports dispute and repair when settlement is contested. A dispute creates a review path where parties can attest, contest, amend, reverse, or compensate according to Community Shell and Project Shell rules.

Failed settlements, fraud claims, coercion, unfair enforcement, and corruption are treated as repair events, not as permanent exclusion.

If no local repair resolves the dispute, capacity remains in suspended state until a recognised governance process supplies a binding finality pathway.

## 18. Accessibility and Assisted Access

The wallet must support voice-first use, low-literacy interfaces, offline proximity settlement, shared community terminals, and fallback channels including paper or card receipts.

Assisted access must preserve Direct Source where possible. Guardians, carers, interpreters, or community operators may help operate the interface, but assistance is distinct from authority transfer.

Where substitute decision support is necessary, it must be scoped, reviewable, and revocable.

## 19. Implementation Plurality and Protocol Governance

Wallet implementations are plural. Any wallet may participate if it follows the public protocol, preserves Direct agency, supports replayable settlement, and does not subordinate users to provider authority.

Protocol upgrades must be versioned, public, backwards-aware, and replayable. Each attestation must disclose protocol version used.

No private provider may silently change settlement semantics. Where protocol forks occur, Shells must expose which versions they recognise and which version is used for each attestation.
