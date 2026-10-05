# Alignment Infrastructure Routes - AIR Brief
> Trustworthy Distributed Human Workforce Coordination for AI Safety

Alignment Infrastructure Routes (AIR) is a set of procedures and software for coordinating contributions to AI safety work. Participants record the task and its evidential basis alongside the resulting deliverable. These records can be independently checked and used to administer funded commitments through laboratories or fiscal hosts.

A contribution may involve observations from several people and processing by artificial systems. Review therefore requires an account of the sources used and the conclusions drawn from them, including the contributions of those receiving the result. The same recordkeeping is applicable to self-organised work and informal cooperation. The classification of these relationships is explained below, followed by the arrangements for funded programmes.

## What problem it solves

### The workforce problem

There is no reliable way to turn distributed human contribution into stable paid work in AI safety. Most funding routes require institutional access, credentials, or existing lab affiliation.

### The coordination problem

AI safety work is often fragmented. Different groups repeat the same work, use different definitions, and produce outputs that are hard to compare. Funders cannot easily see what kinds of risks are being covered across a portfolio.

### The accountability problem

Most agentic systems and AI workflows increase output volume, but they do not improve accountability. When something goes wrong, it is unclear what was human judgment and what was model output.

Participants use shared procedures to classify contributions and preserve their context. Sponsors and fiscal hosts can administer funding for work documented through these procedures.

---

## Who it is for

Participants include people organising informal reviews or mutual support, alongside the following funded-project participants:

1. **AI safety teams and individual contributors** who can do work and need paid pathways into it.
2. **Fiscal hosts and NGOs** that already receive funds and pay individuals or contractors.
3. **AI safety labs** that need structured human contributions at scale, such as evaluations, interpretability documentation, and red-teaming.

## What it produces

Participants produce the following records and deliverables:

1. **Safety work deliverables**
    
    Examples include jailbreak analyses, evaluation writeups, interpretability notes, test cases, datasets, and documentation.
    
2. **Clear classification of what the work is doing**
    
    Classification is a shared way to describe what risk a piece of work addresses and what kind of contribution it represents. It is not approval and it is not a grade.
    
3. **Attested work receipts**
    
    Each contribution produces a verifiable moment receipt: a short regenerable position on a deterministic kernel trajectory (anchor, depth, phase). Sponsors and fiscal hosts can reconstruct what happened by replaying the public transition rule, without relying on informal narratives. Transport layouts of 16 to 20 bytes fit commodity QR codes; the archive stores an identity anchor once and depth deltas thereafter.
    

---

## How it works

Implementation is organised in three layers.

### 1. hQVM Kernel

The hQVM Kernel is the coordination backbone. It provides a finite space of 4,096 reachable states that serve as universal coordination points for distributed work.

A shared state (or "Moment") is a reproducible coordinate that any participant can compute from the same activity log. The kernel's algebraic structure ensures that event order matters: two projects with identical deliverable counts but different execution sequences follow different trajectories through the state space. This non-commutativity provides structural provenance, proving not only *what* was done, but *how* the governance process unfolded over time.

The relevant computational properties are as follows:

- **Exact convergence:** Any two consecutive byte steps from the rest state distribute coordination positions exactly uniformly across all 4,096 reachable states, with each state receiving exactly 16 of the 65,536 possible two-byte words.
- **Tamper detection:** The self-dual [12,6,2] code structure detects corruption in coordination logs with mathematical certainty, catching almost all substitutions and deletions at the algebraic level.
- **Holographic verification:** The 64-state boundary horizons encode the full 4,096-state space, allowing efficient verification of large project histories through small boundary checks.

Participants with the same log prefix compute the same state and can use it as a common reference for subsequent coordination.

**Moments as Genealogical Markers**

A Moment is not merely a coordinate or timestamp. It is a genealogical marker that encodes the full structural lineage of governance operations. Two projects with identical incident counts but different execution sequences will have different Moments, because the Moment preserves the accumulated consequence of each decision in the order it was made. By examining the sequence of Moments, participants can see both the present state of an activity and the path through which it was reached, and can identify the structural correction needed to restore balance.

**Why Genealogical Encoding**

The hQVM Kernel implements the same structural principle that makes genetic information trustworthy: the encoding of lineage into state. In biological systems, a gene expressed at different points in developmental history produces different outcomes, because the cellular context encodes the accumulated memory of prior states. Similarly, a governance event recorded at different points in a project's history produces different Moments, because the Moment encodes the accumulated memory of prior operations. Participants preserve the sequence so that they can examine how a decision arose and how later actions changed its context. This gives them a common basis for checking the recorded history together.

### 2. App

Contributors use the application to record the task and submit their work for review.

A contributor defines their task, performs the work, and submits the deliverable. The App captures the task definition, the completed deliverable, and the classification.

Review involves examining the sources of information and the basis of conclusions, together with their relation to continuing work. Under the Gyroscope Protocol, these activities are classified according to four governance capacities. One person may exercise all four in a single task:

- Governance Management Traceability
- Information Curation Variety
- Inference Interaction Accountability
- Intelligence Cooperation Integrity

### 3. Tools

Tools connect the App to tools people already use, such as GitHub, notebooks, shared documents, and model evaluation harnesses.

The Human Mark (THM) is the AI safety and alignment framework used to classify relationships between the sources of information and the capacities through which it is received or processed. These are termed Authority and Agency, respectively. Participants exercise them across the activity, with Direct human and Indirect artificial forms distinguished by their ancestry.

For this work, alignment is the preservation of those relationships throughout production and review through traceability of information variety, inference accountability, and intelligence integrity to Direct Authority and Agency. Power concentration arises when a specific system, institution, or individual is treated as exhausting an entire category of Authority or Agency, with consequent loss of ancestry measurement. Participants use the following displacement-risk classifications to document the systematic forms of that error:

- Governance Traceability Displacement
- Information Variety Displacement
- Inference Accountability Displacement
- Intelligence Integrity Displacement

Tools are optional. The platform works if contributors submit outputs manually.

---

## Operating Model

In a sponsored project, participants define the work and its review procedures, while the sponsor administers the committed funds. The following arrangement describes this funded use of AIR. People can also use the same records for self-organised work and informal cooperation.

1. Technical implementation.
   This includes the hQVM Kernel, the App workflow, and optional Tools for tracking and verifying deliverables.

2. **A Project Sponsor administers funds.**
   The Sponsor can be an AI Safety Lab offering prizes for contributions or a Fiscal Host NGO administering a grant program. The Sponsor publishes acceptance criteria and payment terms for that funded programme. Contributors and reviewers remain responsible for their respective decisions, with the agreed review process recorded alongside the terms.

3. **Contributors participate through deliverables, not pitches.**
   Participation is based on completing project-defined deliverables within the rules set by the Sponsor. This eliminates the need for individual contributors to write business plans or fundraising narratives.

---

## Program Units

Project organisers use defined time units to specify funded work and the associated commitments.

### Daily Unit (1 Day)

The Daily Unit is a single-day contribution modeled as a **Daily Prize**.

It is used for atomic tasks such as a single jailbreak analysis, a specific test case, or a documentation fix. Sponsors fund these as performance-based prizes to lower the barrier to entry and allow rapid evaluation of new contributors.

### Sprint Unit (4 Days)

The Sprint Unit is a four-day contribution modeled as a **Sprint Stipend**.

It is used for structured deliverable bundles such as a complete evaluation set, a dataset contribution, or a reproducible interpretability study. Sponsors fund these as fixed stipends for contributors who have demonstrated capability.

---

## Progression and Thresholds

In a funded programme, contributors can progress through the following arrangements for longer-term work:

1. **Open Participation:** Contributors start by submitting Daily Units.
2. **Stipend Qualification:** Contributors who meet a defined threshold of accepted Daily Units qualify for Sprint Stipends.
3. **Employment Queue:** Contributors who successfully complete a defined number of Sprint Stipends enter a qualified queue for longer-term employment or contracting with the Sponsor.

This structure allows Sponsors to vet talent through actual work outputs with capped financial risk, while providing contributors with a clear path to professional stability.

---

## Caps and Payment Authority

To manage risk and budget, Project Sponsors set specific caps on participation.

A standard configuration limits an individual contributor to a specific number of Daily Prizes and Sprint Stipends per project. This ensures funds are distributed to a wider pool of contributors and prevents indefinite casual work without a move toward formal employment.

The Sponsor administers payment release under the programme's published terms. Participants use the verified work trail and classification data to review acceptance decisions and resolve disputes within that scope.

---

## Example Program Configuration and Funding Tiers

AI safety labs and fiscal hosts can use the following programme configurations. The same basic units apply in both cases. Amounts below are illustrative; actual rates and caps are set by the sponsor.

### Canonical work units

Two canonical units are used to specify funded work:

- **Daily Unit (1 Day)**  
  One day of focused work on a well-defined task. This is used for "Daily Prizes".

- **Sprint Unit (4 Days)**  
  Four days of focused work on a structured deliverable bundle. This is used for "Sprint Stipends".

These units are defined in terms of work and deliverables, not employment status. Sponsors fund outputs that correspond to one or more of these units.

In transition contexts, sponsors may express funding in conventional currency. Natively, within the Moments Economy framework, these units are denominated in Moment-Units (MU) at the reference value of 1 MU = 1 international dollar (int$), settled through the hQVM Kernel.

---

### Tier 1: Individual rapid grants (per-person cap, e.g. £1,000)

Some programs, such as rapid grants, place a cap per individual (£1,000 per person). In this setting, Alignment Infrastructure Routes can be used to structure a small, safe engagement window.

**Example configuration (illustrative numbers):**

- Daily Prize: £120 for one Daily Unit (1 day, one clear deliverable)
- Sprint Stipend: £480 for one Sprint Unit (4 days, a bundled deliverable)

A sponsor with a £1,000 cap per person can, for example:

- Fund up to one Sprint Stipend and one Daily Prize for a contributor (5 working days, £600), leaving headroom within the cap for:
  - additional prizes in a later round, or
  - overhead, tools, or coordination costs

The cap applies per person. Programme records specify the funded work and the payment basis within that limit.

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
- Alignment Infrastructure Routes continues to be used for:
  - defining and tracking Daily and Sprint Units
  - classifying deliverables with Gyroscope and The Human Mark
  - providing an attested work trail for reporting back to the sponsor

In this tier, fiscal hosts are most useful, because they already have the legal and accounting infrastructure to support longer engagements. Participants use AIR records to document the work administered through those arrangements.

---

## The Complete Framework

Participants use The Human Mark to classify displacement risks, the Gyroscope Protocol to classify contributions, and the hQVM Kernel to compute and replay coordination records.

The following displacement risks are defined in The Human Mark as systematic forms of the loss of ancestry measurement:

- Governance Traceability Displacement
- Information Variety Displacement
- Inference Accountability Displacement
- Intelligence Integrity Displacement

Contributors and reviewers can inspect which displacement risks are addressed by the work. Funders can use the same classifications to examine the distribution of support across activities.

### Core Engine: Gyroscopic ASI hQVM Kernel

The Holonomic Quantum Virtual Machine (hQVM) computation maps activity logs into sequences of states in a finite state space. Operators can reproduce those sequences by applying the published transition rule.

The computation is specified through the following properties:

- **4,096 shared moments:** A complete finite universe of coordination states reachable from a universal rest state.
- **Dual horizons:** 64-state equality and complement boundaries that encode provenance and chirality, enabling efficient verification of complex histories.
- **Single-step hidden patterns:** Operators can calculate the specified algebraic invariants in one computational step and inspect them for structural symmetries.
- **Exact replay:** Every coordination trajectory can be reconstructed forward or reversed algebraically from the activity log, so any party can verify the full sequence of states.

### Theoretical Foundation: Gyroscopic Global Governance

In Gyroscopic Global Governance (GGG), the four capacities are applied across economy, employment, education, and ecology. Through AIR, participants document their exercise in particular activities. A single person may exercise all four within a task, while the relationships among providers and receivers remain traceable across the wider process.

---

## Disclaimer

AIR is coordination infrastructure for work receipts, classification, and replayable audit.

Project procedures should preserve the distributed exercise of capacities throughout submission and review. Sponsors administer the agreed funding terms, and contributors retain access to the basis of an assessment and its review process. The record should make it possible to examine how evidence was considered and whether participants could exercise the relevant capacities.

---

## Documentation & Links

### hQVM Kernel Specification
- [**Gyroscopic ASI Foundations**](https://github.com/gyrogovernance/superintelligence/blob/main/docs/Gyroscopic_ASI_Foundations.md): Complete technical specification of the hQVM Kernel
- [**hQVM Kernel Implications & Potential**](https://github.com/gyrogovernance/tools/blob/main/docs/Gyroscopic_ASI_Implications.md): Use cases and deployment scenarios
- [**hQVM Verification Reports**](https://github.com/gyrogovernance/superintelligence/tree/main/docs/reports): Test results confirming the kernel's holonomic gate structure, CHSH saturation, and structural quantum advantages on Ω

### Classification Framework (The Human Mark)
- [**The Human Mark**](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM.md): Core taxonomy of four displacement risks
- [**Formal Grammar**](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM_Grammar.md): PEG specification for tagging and validation
- [**Specifications Guidance**](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM_Specs.md): Implementation guidance for systems and evaluations
- [**Terminology Guidance**](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM_Terms.md): Mark-consistent framing for 250+ AI safety terms

### Proof of Concept
- [**The Human Mark in the Wild**](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM_InTheWild.md): Analysis of 655 jailbreak prompts with THM classifications
- [**Dataset on Hugging Face**](https://huggingface.co/datasets/gyrogovernance/thm_Jailbreaks_inTheWild): Annotated corpus for training and evaluation

### References
- [**GGG Paper**](https://github.com/gyrogovernance/superintelligence/blob/main/docs/programs/GGG_Paper.md): Theoretical foundations and governance framework

---


