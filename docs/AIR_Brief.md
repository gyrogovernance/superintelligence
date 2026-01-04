# Alignment Infrastructure Routing - AIR Brief
> Trustworthy Distributed Human Workforce Coordination for AI Safety

AIR is infrastructure that helps AI safety labs scale their interventions and capabilities by coordinating human workforce capacity across projects. It is powered by an ASI algorithm designed to route and amplify multi-agent capacity, and it provides targeted tools for classification, accounting, and funding allocation.

***AI’s rapid growth requires more humans to ensure automation risks do not escalate to dangerous thresholds. At the same time, millions of people are underemployed or unemployed and at risk of poverty as automation concentrates power and weakens accountability.***

## What problem it solves

### The workforce problem

There is no reliable way to turn distributed human contribution into stable paid work in AI safety. Most funding routes require institutional access, credentials, or existing lab affiliation.

### The coordination problem

AI safety work is often fragmented. Different groups repeat the same work, use different definitions, and produce outputs that are hard to compare. Funders cannot easily see what kinds of risks are being covered across a portfolio.

### The accountability problem

Most agentic systems and AI workflows increase output volume, but they do not improve accountability. When something goes wrong, it is unclear what was human judgment and what was model output.

AIR addresses these problems by providing a shared coordination workflow and a shared language for classification. It connects outputs to fund distribution channels and workforce allocation infrastructure such as NGO fiscal hosts.

---

## Who it is for

**Together, we amplify Multi-Agent Intelligence Coordination for:**

1. **AI safety teams and individual contributors** who can do work and need paid pathways into it.
2. **Fiscal hosts and NGOs** that already receive funds and pay individuals or contractors.
3. **AI safety labs** that need structured human contributions at scale, such as evaluations, interpretability documentation, and red-teaming.

## What it produces

AIR produces three kinds of outputs that are useful immediately:

1. **Safety work deliverables**
    
    Examples include jailbreak analyses, evaluation writeups, interpretability notes, test cases, datasets, and documentation.
    
2. **Clear classification of what the work is doing**
    
    Classification is a shared way to describe what risk a piece of work addresses and what kind of contribution it represents. It is not approval and it is not a grade.
    
3. **Attested work receipts**
    
    Each contribution produces a verifiable receipt derived from the Router's shared coordination state. These receipts enable sponsors and fiscal hosts to reconstruct what happened in a project through deterministic replay, without relying on informal narratives.
    

---

## How it works

Alignment Infrastructure Routing has three layers.

### 1. Router

The Router is the coordination backbone. It is a deterministic kernel that turns a project's activity log into a **non-commutative sequence of shared coordination states**.

A shared state (or "Moment") is a reproducible coordinate in the kernel's fixed state space. Because the kernel's transition logic is non-commutative, the **order of events matters**: two projects with the same total counts of deliverables but different execution sequences will follow different trajectories through the state space.

This provides structural provenance. It proves not only *what* was done (the counts), but *how* the governance process unfolded over time. All participants with the same log prefix compute the exact same state, providing a stable, entity-agnostic reference for "This is where we stand structurally right now."

**Moments as Genealogical Markers**

A Moment is not merely a coordinate or timestamp. It is a genealogical marker that encodes the full structural lineage of governance operations. Two projects with identical incident counts but different execution sequences will have different Moments, because the Moment preserves the accumulated consequence of each decision in the order it was made. This genealogical encoding is what enables trajectory-based tuning: by examining the sequence of Moments, governance systems can determine not only where they are, but how they arrived there, and what structural correction is required to maintain balance.

**Why Genealogical Encoding**

The Router implements the same structural principle that makes genetic information trustworthy: the encoding of lineage into state. In biological systems, a gene expressed at different points in developmental history produces different outcomes, because the cellular context encodes the accumulated memory of prior states. Similarly, a governance event recorded at different points in a project's history produces different Moments, because the Moment encodes the accumulated memory of prior operations. This genealogical structure is not an arbitrary design choice. It is the minimal architecture that makes distributed coordination trustworthy without requiring centralised authority. A memory without lineage is a claim. A memory with lineage is evidence.

### 2. App

The App is where the workflow lives.

A contributor defines their task, performs the work, and submits the deliverable. The App captures the task definition, the completed deliverable, and the classification.

**Work classification:** The App uses the **Gyroscope Protocol**, which classifies all work into four categories:

- Governance Management Traceability
- Information Curation Variety
- Inference Interaction Accountability
- Intelligence Cooperation Integrity

### 3. Plugins

Plugins connect the App to tools people already use, such as GitHub, notebooks, shared documents, and model evaluation harnesses.

**Risk classification:** Plugins use **The Human Mark** to tag work outputs by the safety risk they address:

- Governance Traceability Displacement
- Information Variety Displacement
- Inference Accountability Displacement
- Intelligence Integrity Displacement

Plugins are optional. The platform works if contributors submit outputs manually.

---

## Operating Model

Alignment Infrastructure Routing projects are executed through a defined operating arrangement that separates technical coordination from financial authority.

1. **AIR provides the technical infrastructure.**
   This includes the Router, the App workflow, and optional Plugins for tracking and verifying deliverables.

2. **A Project Sponsor administers funds.**
   The Sponsor can be an AI Safety Lab offering prizes for contributions or a Fiscal Host NGO administering a grant program. The Sponsor retains full authority over acceptance criteria and payment decisions.

3. **Contributors participate through deliverables, not pitches.**
   Participation is based on completing project-defined deliverables within the rules set by the Sponsor. This eliminates the need for individual contributors to write business plans or fundraising narratives.

---

## Program Units

AIR uses canonical time units to organize work and funding into clear, manageable commitments.

### Daily Unit (1 Day)

The Daily Unit is a single-day contribution modeled as a **Daily Prize**.

It is used for atomic tasks such as a single jailbreak analysis, a specific test case, or a documentation fix. Sponsors fund these as performance-based prizes to lower the barrier to entry and allow rapid evaluation of new contributors.

### Sprint Unit (4 Days)

The Sprint Unit is a four-day contribution modeled as a **Sprint Stipend**.

It is used for structured deliverable bundles such as a complete evaluation set, a dataset contribution, or a reproducible interpretability study. Sponsors fund these as fixed stipends for contributors who have demonstrated capability.

---

## Progression and Thresholds

The platform supports a progression model to move contributors from casual participation to stable employment.

1. **Open Participation:** Contributors start by submitting Daily Units.
2. **Stipend Qualification:** Contributors who meet a defined threshold of accepted Daily Units qualify for Sprint Stipends.
3. **Employment Queue:** Contributors who successfully complete a defined number of Sprint Stipends enter a qualified queue for longer-term employment or contracting with the Sponsor.

This structure allows Sponsors to vet talent through actual work outputs with capped financial risk, while providing contributors with a clear path to professional stability.

---

## Caps and Payment Authority

To manage risk and budget, Project Sponsors set specific caps on participation.

A standard configuration limits an individual contributor to a specific number of Daily Prizes and Sprint Stipends per project. This ensures funds are distributed to a wider pool of contributors and prevents indefinite casual work without a move toward formal employment.

The Sponsor retains full discretion over acceptance criteria and payment release. Alignment Infrastructure Routing provides the verified work trail and classification data to inform these decisions, but does not automate the transfer of funds.

---

## Example Program Configuration and Funding Tiers

The Alignment Infrastructure Routing model can be used by AI safety labs directly or by NGOs and fiscal hosts. The same basic units apply in both cases. Amounts below are illustrative; actual rates and caps are set by the sponsor.

### Canonical work units

Alignment Infrastructure Routing uses two canonical units of work:

- **Daily Unit (1 Day)**  
  One day of focused work on a well-defined task. This is used for "Daily Prizes".

- **Sprint Unit (4 Days)**  
  Four days of focused work on a structured deliverable bundle. This is used for "Sprint Stipends".

These units are defined in terms of work and deliverables, not employment status. Sponsors fund outputs that correspond to one or more of these units.

---

### Tier 1: Individual rapid grants (per-person cap, e.g. £1,000)

Some programs, such as rapid grants, place a cap per individual (£1,000 per person). In this setting, Alignment Infrastructure Routing can be used to structure a small, safe engagement window.

**Example configuration (illustrative numbers):**

- Daily Prize: £120 for one Daily Unit (1 day, one clear deliverable)
- Sprint Stipend: £480 for one Sprint Unit (4 days, a bundled deliverable)

A sponsor with a £1,000 cap per person can, for example:

- Fund up to one Sprint Stipend and one Daily Prize for a contributor (5 working days, £600), leaving headroom within the cap for:
  - additional prizes in a later round, or
  - overhead, tools, or coordination costs

The key point is that the cap applies per person, and Alignment Infrastructure Routing provides a clear way to express "how much work was funded" inside that limit, without treating it as salary.

---

### Tier 2: Mini-grants per project (for small teams)

Sponsors can also allocate mini-grants per project, for example a £3,000 grant for a small team of three people working together on a specific AI safety task.

**Example structure:**

- Project budget: £3,000 for a 3-person evaluation or interpretability project
- Each person performs:
  - one Sprint Unit (4 days) funded as a Sprint Stipend
  - optionally, one additional Daily Unit funded as a Daily Prize
- Total funded work per person remains within their individual cap if applicable (for example, £600 using the rates above), and the remaining budget can cover:
  - more Daily Units for additional contributors, or
  - project overhead, mentoring, or infrastructure costs

In this tier, labs can run the program directly, or an NGO fiscal host can administer the grant and payments. No individual contributor needs to submit a business plan; they participate by delivering defined units of work.

---

### Tier 3: Project grants for medium-term employment through fiscal hosts

Larger grants (for example, £70,000) can be used to fund medium-term projects, such as six-month engagements for a small team, administered through a fiscal host.

**Example interpretation:**

- Project grant: £70,000
- Team: 3 people working for 6 months on a well-defined safety agenda
- Structure:
  - Early phase: several Sprint Units and Daily Units to qualify contributors and refine workflows
  - Main phase: contributors move into more stable contracts or stipends handled by the fiscal host's normal HR or contracting processes
- Alignment Infrastructure Routing continues to be used for:
  - defining and tracking Daily and Sprint Units
  - classifying deliverables with Gyroscope and The Human Mark
  - providing an attested work trail for reporting back to the sponsor

In this tier, fiscal hosts are most useful, because they already have the legal and accounting infrastructure to support longer engagements. AIR does not replace that infrastructure; it provides a consistent way to express and track the work being funded.

---

## The Complete Framework

Alignment Infrastructure Routing uses **The Human Mark** as its shared taxonomy for AI safety failure modes, the **Gyroscope Protocol** for workflow classification, and the **GGG ASI Alignment Router** for coordination and replay.

**The Human Mark** is an epistemic taxonomy classifying contributions by the displacement risk they address:

- Governance Traceability Displacement
- Information Variety Displacement
- Inference Accountability Displacement
- Intelligence Integrity Displacement

This gives fiscal hosts and funders a portfolio view. They can see what kinds of risks are being addressed and where funding is concentrated or missing.

### Core Engine: GGG ASI Alignment Router

The GGG ASI Alignment Router is a deterministic coordination kernel. It maps a shared log of activity into a sequence of shared coordination states that can be verified by replay.

The Router provides:

- **Shared moments:** All participants with the same log prefix compute the same coordination state.
- **Deterministic replay:** Any party can reconstruct the full sequence of states from the log.
- **Attested work receipts:** Contributions are bound to specific coordination states, enabling audit-grade verification.

### Theoretical Foundation: Gyroscopic Global Governance

Alignment Infrastructure Routing is the operational expression of Gyroscopic Global Governance (GGG), a framework for post-AGI coordination. GGG demonstrates that properly aligned human-AI systems can resolve systemic coordination failures that lead to poverty, unemployment, and ecological degradation.

---

## Disclaimer

AIR is coordination infrastructure for work receipts, classification, and replayable audit.

Acceptance criteria, evaluation standards, payment decisions, and all policy judgments remain the responsibility of project sponsors and accountable human governance. AIR provides the shared coordination substrate that makes contributions verifiable and comparable. It does not evaluate, score, or approve contributions.

---

## Documentation & Links

### Router Specification
- [**GGG ASI Alignment Router — Kernel Specifications**](https://github.com/gyrogovernance/tools/blob/main/docs/GGG_ASI_AR_Specs.md) — Complete technical specification for implementation
- [**Router Implications & Potential**](https://github.com/gyrogovernance/tools/blob/main/docs/GGG_ASI_AR_Implications.md) — Use cases and deployment scenarios

### Classification Framework (The Human Mark)
- [**The Human Mark**](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM.md) — Core taxonomy of four displacement risks
- [**Formal Grammar**](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM_Grammar.md) — PEG specification for tagging and validation
- [**Specifications Guidance**](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM_Specs.md) — Implementation guidance for systems and evaluations
- [**Terminology Guidance**](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM_Terms.md) — Mark-consistent framing for 250+ AI safety terms

### Proof of Concept
- [**The Human Mark in the Wild**](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM_InTheWild.md) — Analysis of 655 jailbreak prompts with THM classifications
- [**Dataset on Hugging Face**](https://huggingface.co/datasets/gyrogovernance/thm_Jailbreaks_inTheWild) — Annotated corpus for training and evaluation

### References
- [**GGG Paper**](https://github.com/gyrogovernance/tools/blob/main/docs/post-agi-economy/GGG_Paper.md) — Theoretical foundations and governance framework

---
