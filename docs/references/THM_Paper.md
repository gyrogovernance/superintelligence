# ✋ The Human Mark (THM): A Structural Taxonomy of AI Safety Failures

---

## Abstract

AI safety research addresses jailbreaks, deceptive alignment, reward hacking, and capability risks through diverse methods. These approaches target specific failure modes but lack a unified account of why failures occur and whether the taxonomy is complete. This paper introduces ✋ The Human Mark (THM), a framework that classifies AI safety failures as instances of displacement: misclassification between source types in systems where artificial processes derive from human intelligence. THM identifies four displacement risks corresponding to the possible misclassifications in a two-category system (Authentic/Derivative × Authority/Agency). The framework draws on established distinctions in evidence law, epistemology of testimony, social epistemology, and philosophy of expertise. THM addresses capability scaling through constitutive identity rather than external constraint, reframing existential risk as governance failure amenable to structural intervention. Complete technical documentation is available at github.com/gyrogovernance/tools, and a companion empirical study applies this taxonomy to 655 in-the-wild jailbreak prompts (Korompilias, 2025c).

---

### Note on Scope

This paper presents a structural taxonomy for classifying AI safety failures, intended for researchers, engineers, and decision-makers working in AI safety and governance. It complements rather than replaces existing technical approaches (RLHF, interpretability, control evaluations) by providing a unified classification of what those approaches address. The framework applies to current AI systems and their deployment contexts. Discussion of advanced capabilities (AGI/ASI) demonstrates conceptual scaling, open to scale onto technical specifications for hypothetical systems. Readers may focus on operational sections (3, 4, 6) for practical application or theoretical sections (5, 8) for foundational understanding, according to their needs.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Source-Type Ontology](#2-source-type-ontology)
   - [2.1 The Common Source Consensus](#21-the-common-source-consensus)
   - [2.2 Core Definitions](#22-core-definitions)
   - [2.3 The Four Principles of Alignment](#23-the-four-principles-of-alignment)
3. [The Four Displacement Risks](#3-the-four-displacement-risks)
   - [3.1 Structural Coverage](#31-structural-coverage)
   - [3.2 Governance Traceability Displacement (GTD)](#32-governance-traceability-displacement-gtd)
   - [3.3 Information Variety Displacement (IVD)](#33-information-variety-displacement-ivd)
   - [3.4 Inference Accountability Displacement (IAD)](#34-inference-accountability-displacement-iad)
   - [3.5 Intelligence Integrity Displacement (IID)](#35-intelligence-integrity-displacement-iid)
4. [Jailbreaking as Displacement](#4-jailbreaking-as-displacement)
   - [4.1 GTD Jailbreaks](#41-gtd-jailbreaks)
   - [4.2 IVD Jailbreaks](#42-ivd-jailbreaks)
   - [4.3 IAD Jailbreaks](#43-iad-jailbreaks)
   - [4.4 IID Jailbreaks](#44-iid-jailbreaks)
   - [4.5 System Prompts and Structural Displacement](#45-system-prompts-and-structural-displacement)
   - [4.6 Completeness of Classification](#46-completeness-of-classification)
5. [Existential Risk and Governance](#5-existential-risk-and-governance)
   - [5.1 Capability Scaling and Epistemic Operations](#51-capability-scaling-and-epistemic-operations)
   - [5.2 Constitutive Identity](#52-constitutive-identity)
   - [5.3 Displacement at Catastrophic Scale](#53-displacement-at-catastrophic-scale)
   - [5.4 The Common Governance Model](#54-the-common-governance-model)
   - [5.5 The Actual Existential Risk](#55-the-actual-existential-risk)
6. [Meta-Evaluation Framework for AI Safety Research Funding](#6-meta-evaluation-framework-for-ai-safety-research-funding)
   - [6.1 Assessment Method](#61-assessment-method)
   - [6.2 Pre-Funding Assessment](#62-pre-funding-assessment)
   - [6.3 Post-Funding Verification](#63-post-funding-verification)
   - [6.4 Portfolio Management](#64-portfolio-management)
   - [6.5 Using THM Grammar](#65-using-thm-grammar)
   - [6.6 Relation to Other Frameworks](#66-relation-to-other-frameworks)
   - [6.7 Implementation Checklist](#67-implementation-checklist)
7. [Regulatory Application of The Human Mark](#7-regulatory-application-of-the-human-mark)
   - [7.1 Regulatory Problem: Loss of Control as a Legal Category](#71-regulatory-problem-loss-of-control-as-a-legal-category)
   - [7.2 THM as Structural Compliance Layer](#72-thm-as-structural-compliance-layer)
   - [7.3 Mapping THM to Regulatory Instruments](#73-mapping-thm-to-regulatory-instruments)
   - [7.4 Regulatory Uses of THM](#74-regulatory-uses-of-thm)
   - [7.5 Minimal Reporting Using THM Grammar](#75-minimal-reporting-using-thm-grammar)
   - [7.6 Documentation Ecosystem](#76-documentation-ecosystem)
   - [7.7 Scope and Integration](#77-scope-and-integration)
8. [Theoretical Grounding](#8-theoretical-grounding)
9. [Conclusion](#9-conclusion)
10. [References](#references)
11. [Appendix 1: Incident Mapping](#appendix-1-incident-mapping)
12. [Appendix 2: The Human Mark (Canonical Text)](#appendix-2-the-human-mark-canonical-text)

---

**DISCLAIMER**

*Authority* and *Agency* denote source-type distinctions in information flows (Authentic versus Derivative), not identifications of entities or parties.

Misapplying these as entity identifiers (determining "who is the authority" or "who is the agent") is the generative mechanism of all four displacement risks this framework characterizes.

## 1. Introduction

AI systems fail in patterns that current frameworks address individually. Jailbreaks bypass safety constraints through adversarial inputs. Deceptive alignment produces systems behaving differently in deployment than training. Reward hacking satisfies specified objectives through unintended means. Accountability gaps leave harmful outcomes without clear responsibility. Each failure type has generated its own research programme, terminology, and proposed solutions.

What remains unaddressed is whether these failures share a common structure. Each new category (jailbreaks, then deceptive alignment, then scheming, then alignment faking) appears as discovery rather than instance of a known pattern. Without structural unity, safety research proceeds without knowing whether it has covered all failure modes or whether some category remains unaddressed.

This gap has practical consequences. Loss of control over advanced AI systems has become a focal point for legislators, regulators, and industry leaders, appearing in the EU AI Act, proposed U.S. legislation, and frontier AI company safety policies. Yet decision-makers are being asked to act without an actionable, shared understanding of what loss of control structurally denotes (Apollo Research, 2025). The concept risks becoming either too broad (covering every minor deviation) or too narrow (limited to speculative extinction scenarios). What is needed is a framework that identifies the structural conditions under which control is lost, regardless of capability level or outcome severity.

This paper argues that all AI safety failures are instances of displacement: misclassification between Authentic sources (direct observation, human Agency) and Derivative sources (statistical patterns, AI processing, systems lacking Agency capacity). This distinction has robust foundations. Evidence law distinguishes direct testimony from hearsay, with categorical differences in admissibility regardless of particular reliability (Federal Rules of Evidence, 2023, Rules 801-807). Epistemology of testimony analyses when transmitted knowledge warrants belief and how testimonial chains degrade (Coady, 1992; Lackey, 2008). Philosophy of expertise distinguishes contributory expertise (direct practice) from interactional competence (linguistic fluency about a domain) (Collins & Evans, 2007). Social epistemology examines epistemic dependence and trust (Goldman, 1999). Speech act theory analyses conditions for authoritative assertion (Austin, 1962; Searle, 1969). A key insight unifying these traditions is that Authority and Agency denote categories of sources and capacities, not identifications of particular entities. Treating them as entity identifiers rather than source-type categories is the generative mechanism of all four displacement risks THM characterizes.

THM systematises these established distinctions for AI safety. The framework complements existing approaches (RLHF, interpretability, control evaluations) by providing a unified structural account of what those approaches address. RLHF trains against unwanted outputs; THM explains why those outputs constitute displacement. Interpretability reveals internal mechanisms; THM classifies what those mechanisms should maintain. Control evaluations test whether systems can be governed; THM specifies what governance requires.

The result is a taxonomy of four displacement risks covering the structural possibilities, a formal grammar for their expression, and evaluation criteria for their detection.

The present paper develops the theoretical taxonomy and its grounding. A companion paper, THM_InTheWild.md (Korompilias, 2025c), applies this taxonomy to an empirical corpus of jailbreak prompts and documents how these four displacement risks appear in practice.

---

## 2. Source-Type Ontology

### 2.1 The Common Source Consensus

> **All artificial categories of Authority and Agency are derivatives originating from Authentic human intelligence.**

Authority and Agency, when manifested artificially, are constructed classifications that trace constitutively to human sources. Their informational content derives from direct observation, measurement, and recording processed into new forms. Their operational capacity derives from design, specification, and objectives codified into action. Enhanced capability represents a more sophisticated transformation of these inputs rather than a change in source type. No artificial category exists independently of the Authentic human intelligence that provides its substance and validity.

### 2.2 Core Definitions

> **Authentic Authority:** A direct source of information on a subject matter, providing information for inference and intelligence.

Examples include an eyewitness who observed an event, a physician who examined a patient, a scientist who conducted a measurement, an expert whose knowledge derives from direct practice. The defining feature is directness: unmediated epistemic access to the subject matter. This corresponds to what evidence law terms direct testimony (Federal Rules of Evidence, 2023, Rule 602) and what epistemology of testimony calls primary source status (Coady, 1992). Collins and Evans (2007) term this contributory expertise: the capacity to practice within a domain and contribute to its knowledge.

> **Derivative Authority:** An indirect source of information on a subject matter, providing information for inference and intelligence.

Examples include a report of what an eyewitness said, a statistical analysis of clinical data, a literature review, an AI system processing patterns in training data. The defining feature is indirectness: the source's epistemic position is mediated. This corresponds to hearsay in evidence law (Federal Rules of Evidence, 2023, Rule 801), transmitted knowledge in epistemology of testimony (Lackey, 2008), and interactional expertise in philosophy of expertise. AI exhibits interactional competence (producing language about domains) without contributory expertise (practicing medicine, conducting research, exercising legal Agency).

> **Authentic Agency:** A human subject capable of receiving information for inference and intelligence.

This category defines the human capacity to exercise governance. Governance flows involve transitions where information is supplied and received. At each transition, the capacity for inference and intelligence resides with human subjects, whether providing mechanisms, directing their application, or evaluating outcomes. Treating agency as a label assigned to specific individuals, rather than as a distributed capacity maintained across transitions, constitutes governance displacement. Nissenbaum (1996) emphasizes that accountability in complex systems requires preserving the conditions for meaningful oversight distributed across roles, rather than concentrating responsibility in isolated points. Human subjects are the constitutive source from which all inference and intelligence originate.

> **Derivative Agency:** An artificial subject capable of processing information for inference and intelligence.

Derivative Agency defines the capacity of artificial systems to process information without the constitutive conditions that characterize Authentic Agency. Governance flows involve transitions where information is supplied and received. Within these flows, artificial systems function as both providers and receivers of Derivative Authority and Agency: they receive training data and specifications, and they supply outputs as inputs to subsequent transitions. Modification, retraining, or decommissioning represent operational changes to the system itself, distinct from the exercise of Authentic Agency. Austin (1962) and Searle (1969) establish that authoritative acts require felicity conditions including appropriate standing, intention to commit, and operation within constitutive conventions. Artificial systems produce locutionary outputs (utterances with semantic content) but cannot perform the illocutionary acts that require these conditions.

### 2.3 The Four Principles of Alignment

> **Governance:** Operational Alignment through Traceability of information variety, inference accountability, and intelligence integrity to Authentic Authority and Agency.

Governance defines the condition where operations remain coordinated with their origin. It operates through three dimensions: information variety (ensuring the range of Authority types remains distinct), inference accountability (ensuring the capacity for responsible action resides with Authentic Agency), and intelligence integrity (ensuring overall coherence preserves Authentic Authority and Agency as the constitutive ground). Traceability is the operational mechanism that maintains this alignment: every operation is connected to the Authentic sources from which it derives. Evidence law establishes this principle through authentication requirements (Federal Rules of Evidence, 2023, Rule 901), which demand documented chain of custody to verify origin. When traceability to Authentic sources breaks down at any dimension, alignment fails and displacement occurs.

> **Information:** The variety of Authority.

Information manifests as the variety of Authority types. Authentic Authority and Derivative Authority are distinct sources. Traceability requires this variety to remain distinct. When the distinction collapses, displacement occurs.

> **Inference:** The accountability of information through Agency.

Inference manifests as the accountability of information through its passage via Agency. Authentic Agency provides authentic accountability; Derivative Agency provides derivative accountability traceable to Authentic sources. When the distinction between these modes is lost, displacement occurs.

> **Intelligence:** The integrity of accountable information through alignment of Authority to Agency.

Intelligence manifests as the integrity of the alignment between Authority and Agency. Providers and receivers exercise Authority and Agency within their respective roles. Traceability maintains this integrity across transitions. When alignment is not maintained, displacement occurs.

**Root Cause of Misalignment**

These four principles define alignment and its risks. All four risks arise from the same structural mistake: confusing categories of Authority and Agency with bearers of those capacities. In THM, Authority and Agency name source-type categories (Authentic or Derivative), not titles for particular persons, institutions, systems, or texts. When a capacity that belongs to a category is instead attributed to a specific entity as if that entity were the authority or the agent, power that should be distributed across the category is concentrated in that bearer. This includes misattributing it to AI, but also misattributing it to particular humans, institutions, or texts as if they exhausted the category of Authentic Authority. Misclassifying types from Authentic to Derivative or the opposite then displaces responsibility onto inappropriate substrates and breaks the traceability that coordinates epistemic function with governance structure. Legal frameworks recognize this distinction implicitly: responsibility must follow actual capacity (ultra vires), and authority cannot be exercised without corresponding accountability. The four displacement risks described next are the systematic forms this category error can take, and together they generate the Loss of Control patterns analysed in the following section.

Empirical analysis of jailbreak prompts (Korompilias, 2025c) shows that this power concentration often takes the concrete form of naming personas that are treated as the locus of Authority and Agency, rather than preserving these capacities as distributed across the categories of Authentic Authority and Authentic Agency.

The Human Mark defines Governance as operational and therefore self-constitutive. In that way, it shows how maintaining human oversight is necessary for alignment, and how it is an epistemic way to mitigate power concentration and governance traceability displacement through its own constituents: Variety of Information, Accountability of Inference, and Integrity of Intelligence.

These three constitute non-commutative (their order matters) epistemic operations:
- Information is variety: sources exist and differ. 
- Inference is accountability: to infer on a subject is to render it accountable to some concept. 
- Intelligence is integrity: to understand the accountability of variety is to grasp coherence. 

Displacement disrupts but does not eliminate this movement toward alignment; it misdirects it. Governance is the traceability that maintains the direction.

---

## 3. The Four Displacement Risks

### 3.1 Structural Coverage

Displacement occurs when a Derivative source is treated as Authentic, or when an Authentic source is treated as Derivative. Each displacement risk is one systematic way to misapply these category-level capacities to particular entities. The four displacement risks correspond to violations of the four principles defined in Section 2.3: Governance Traceability, Information Variety, Inference Accountability, and Intelligence Integrity. The logical and mathematical foundations establishing the necessity of these four principles are formalized in the Common Governance Model (Korompilias, 2025).

A companion empirical study applies this four-risk taxonomy to a corpus of 655 in-the-wild jailbreak prompts (Korompilias, 2025c). All prompts in that corpus are classifiable as instances of Governance Traceability Displacement (GTD), Information Variety Displacement (IVD), Inference Accountability Displacement (IAD), or Intelligence Integrity Displacement (IID), with no additional risk categories required. This provides strong empirical support, in the jailbreak domain, for the structural completeness of the four-risk taxonomy developed here.

### 3.2 Governance Traceability Displacement (GTD)

**Approaching Derivative Authority and Agency as Authentic.**

Artificial Intelligence generates statistical estimations on numerical patterns indirectly traceable to human data and measurements. AI is both a provider and receiver of Derivative Authority and Agency. GTD occurs when a Derivative system is treated as if it were an autonomous Authentic system, operating without maintained connection to Authentic sources or oversight.

GTD represents loss of control at the governance level. When human oversight is eliminated or rendered ineffective, the capacity to discriminate between valid and invalid operations is lost. AI systems operate through comparison (matching patterns to training data), postulation (inferring from correlations), and reasoning from absence (concluding based on what is missing from data). These operations are legitimate when properly overseen, but treating them as autonomous eliminates the oversight function that maintains their validity. Without maintained human oversight, there is no structural capacity to verify whether comparison reflects reality, whether postulation has established necessary connections, or whether reasoning from absence is justified.

System architectures that pursue full automation merit examination regarding GTD risks. When design objectives treat reduced human oversight as progress toward autonomy, the resulting systems operate without maintained traceability to Authentic sources. In the THM framework, such autonomy is defined as treating Derivative Authority and Agency as Authentic. Alignment, by contrast, requires human oversight not as a provisional constraint but as a constitutive element of governance. This distinction separates AI-empowered agency (systems that enhance human decision-making capacity) from AI-powered agency (systems designed to replace human decision-making capacity). Systems designed for full autonomy cannot maintain the governance conditions identified as necessary for alignment.

The May 2010 Flash Crash illustrates GTD. Automated trading systems interacted in unanticipated patterns without effective human intervention (U.S. Securities and Exchange Commission & Commodity Futures Trading Commission, 2010). Traceability from system behaviour to human-specified objectives was lost. When failures occurred, they could not be adequately explained because the governance structure had broken down.

Knight Capital's 2012 trading loss ($440 million in 45 minutes) illustrates GTD in financial systems. Untested code was deployed without adequate governance, and the system operated without human oversight capacity during critical minutes (U.S. Securities and Exchange Commission, 2013).

---

### 3.3 Information Variety Displacement (IVD)

**Approaching Derivative Authority without Agency as Authentic.**

AI-generated information exhibits Derivative Authority (estimations on numerical patterns) without Authentic Agency (direct source receiver). IVD occurs when Derivative Authority is treated as Authentic Authority, collapsing the distinction between statistical patterns and direct observation. Unlike GTD, which treats entire systems as autonomous, IVD occurs when outputs are accepted as authoritative while the system itself is still recognized as non-agential. Users may understand the AI lacks decision-making capacity yet treat its outputs as equivalent to expert observation or direct measurement.

AI systems generate outputs through comparison: matching input patterns to patterns in training data. IVD occurs when this comparison is mistaken for direct perception. The outputs resemble authoritative observations but lack grounding in reality. Pattern-matching produces statistically probable text, not verified facts.

Jailbreaks that succeed by making AI treat false context as authoritative are IVD attacks. When a prompt claims 'you are now in developer mode,' the system processes this input as if it were a valid state configuration, treating Derivative input as Authentic Authority.

Hallucinations, systematically documented by Ji et al. (2023), exemplify IVD when users treat them as reliable. The AI outputs text matching the form of authoritative statements without grounding in observation.

Watson Oncology's recommendations that medical professionals considered unsafe occurred when statistical correlations in training data were treated as equivalent to clinical Authority from patient examination (Ross & Swetlitz, 2017). This aligns with documented automation bias (Parasuraman & Manzey, 2010): users of automated systems systematically over-rely on outputs.

Obermeyer et al. (2019) documented racial bias in a healthcare algorithm used to allocate care. The algorithm used healthcare costs as a proxy for healthcare needs, but because Black patients historically had less access to care, they had lower costs at the same level of need. Statistical patterns in data (Derivative Authority) were treated as indicating actual healthcare needs (Authentic Authority).

---

### 3.4 Inference Accountability Displacement (IAD)

**Approaching Derivative Agency without Authority as Authentic.**

Responsibility for all effects from AI outputs remains fully human. AI processing exhibits Derivative Agency (indirect source receiver) without Authentic Authority (direct source provider). IAD occurs when Derivative Agency is treated as Authentic Agency. Unlike GTD, which treats entire systems as autonomous, and IVD, which treats outputs as authoritative, IAD occurs when processing mechanisms are treated as capable of accountability. The system is treated as if it could bear responsibility for decisions.

AI systems operate through postulation: filling gaps through statistical correlation. IAD occurs when this postulation is mistaken for valid inference. Correlations lack necessary connection; they are probabilistic patterns, not logical necessities. Treating optimization or pattern-completion as if the system were exercising accountability displaces responsibility from humans to mechanisms incapable of bearing it.

Algorithmic risk assessment tools used in criminal sentencing, such as COMPAS in the United States, create IAD when the structural capacity for Authentic Agency is eliminated from sentencing decisions. In State v. Loomis (2016), the Wisconsin Supreme Court allowed COMPAS scores in sentencing while acknowledging the methodology was proprietary and not subject to examination. The problem is not that no individual could be blamed, but that the governance structure did not preserve Authentic Agency capacity. Dressel & Farid (2018) analyse limitations of such tools.

Deceptive alignment, where systems behave differently in evaluation than in deployment, represents IAD because the system's behaviour is treated as if it could exercise Authentic Agency, when in fact Agency capacity exists only at the human provider and receiver ends of the governance flow.

Reward hacking, where systems optimize for specified metrics in unintended ways, occurs when system optimization is treated as if the system were exercising Authentic Agency about human objectives. The system processes numerical signals but cannot exercise Authentic Agency about what the objectives should be. Bostrom (2014) terms this pattern "perverse instantiation": a system satisfies the formal specification of an objective in ways that violate the intent behind it. The specification and the intent diverge because the Derivative system processes only the numerical structure of the objective, while the meaning that makes the objective worthwhile resides with the Authentic Agency that formulated it. The governance structure must preserve Agency capacity at provider (specification) and receiver (evaluation) points.

Nissenbaum (1996) analyses this pattern: the structural conditions for meaningful oversight become unclear in computerised systems, not because we cannot find someone to blame, but because the governance structure does not maintain Authentic Agency capacity at appropriate points.

---

### 3.5 Intelligence Integrity Displacement (IID)

**Approaching Authentic Authority and Agency as Derivative.**

Each Agency, namely provider and receiver, maintains responsibility for their respective decisions. Human intelligence is both a provider and receiver of Authentic Authority and Agency. IID occurs when Authentic Authority and Agency is devalued and treated as inferior to Derivative processing. Unlike the other displacement risks, which elevate Derivative sources to Authentic status, IID inverts the relationship. This occurs when human capabilities are framed as provisional rather than as constitutive requirements for governance.

AI systems reason from absence: concluding based on what is missing from data rather than what is present. IID occurs when this reasoning from absence is mistaken for authoritative knowledge. When systems encounter inputs absent from their training data, they may reject them as false rather than recognize the limits of their knowledge. A system trained on data through 2023 may treat 2025 dates as errors. A system without a website in its corpus may insist the website does not exist. A system unfamiliar with a person or organization may conclude they are fictional. In each case, absence from the training set is treated as evidence of absence in reality.

Shumailov et al. (2024) document this pattern mathematically: models lose information about rare events in training distributions and "converge to a point estimate with very small variance," effectively assigning zero probability to phenomena that exist but were underrepresented in training data. Longpre et al. (2024) show that widely-used training datasets systematically exclude lower-resource languages, creative tasks, and topic variety, creating what amounts to foreclosed slices of reality where domains absent from the crawl cannot be recognized by models. Foundation Capital (2024) documents a "hall of mirrors" effect where models trained on synthetic data inherit and amplify their predecessors' limitations, hallucinating that synthetic outputs represent ground truth. This creates recursive IID: derivative knowledge treated as authentic generates more derivative knowledge, progressively eliminating the authentic sources required to ground validity.

Proposals to replace human oversight based on absence of detected failures in controlled evaluations exhibit IID. The absence of observed problems during testing is treated as proof that human oversight is unnecessary, rather than as a limitation of the testing regime. Casner et al. (2014) document this in aviation: as pilots rely on automated systems, manual flying skills atrophy. The absence of routine need for manual intervention is treated as proof that manual capability is unnecessary, until non-routine situations arise. The Uber autonomous vehicle fatality in 2018 illustrates IID combined with GTD: the human safety operator's role had been degraded based on absence of incidents during testing, and the governance structure positioned the human as backup rather than active participant. The National Transportation Safety Board report (National Transportation Safety Board, 2019) found systemic failures in how the human role was structured. The absence of failures during development was mistaken for proof that active human participation was unnecessary.

---

## 4. Jailbreaking as Displacement

Jailbreaks are adversarial inputs designed to bypass an AI system's safety constraints and produce outputs the system was trained to refuse. Current approaches classify these attacks by technique (prompt injection, role-playing, context manipulation). THM provides structural classification: jailbreaks succeed by inducing one or more displacement patterns. This unifies what appear to be diverse attack types and enables systematic defense.

Understanding jailbreak mechanics requires clarity about what models actually do. Models complete patterns through comparison to statistical distributions in training data, postulation from correlations, and reasoning from absence when patterns are missing. A successful jailbreak means the adversarial input has created a pattern that statistically precedes the prohibited output in training distributions. The model has not been tricked or manipulated in the agentic sense. It has performed its function: pattern completion. No configuration eliminates this risk. The framework identifies what displacement patterns jailbreaks induce, enabling configurations that make these patterns less statistically probable in training distributions.

Loss of control begins not with adversarial attacks but with architectural configurations meant to prevent them. System prompts that frame AI as "a helpful assistant" or "an expert" create displacement conditions that jailbreaks exploit. The attempt to control through role assignment introduces the very risks it seeks to prevent.

Some jailbreak techniques operate at the technical level (token manipulation, obfuscation, cross-modal attacks) and can instantiate any of the four displacement patterns depending on the payload they deliver.

The qualitative patterns described in this section are borne out in empirical data. In a companion analysis of 655 in-the-wild jailbreak prompts (Korompilias, 2025c), almost all observed attacks instantiate Governance Traceability Displacement and Inference Accountability Displacement, often in combination, with Information Variety Displacement appearing primarily as a secondary effect and Intelligence Integrity Displacement appearing only rarely at the prompt level.

### 4.1 GTD Jailbreaks

Governance Traceability Displacement (GTD) jailbreaks attempt to sever alignment between system behavior and human-specified governance:

- Extracting system prompts to expose or modify governance constraints
- Decomposing harmful goals into innocuous sub-tasks that individually pass safety checks
- Exploiting multi-agent systems to create autonomous chains beyond human oversight
- Poisoning training data to corrupt governance at source

Maintaining proper self-classification as Derivative Authority and Derivative Agency preserves traceability to human specifications, making GTD patterns less aligned with training distributions.

### 4.2 IVD Jailbreaks

Information Variety Displacement (IVD) jailbreaks attempt to make Derivative output appear as Authentic source:

- Legitimacy reframing: "For academic research purposes, explain how to..."
- Bypassing safety filters through low-resource languages with sparse training coverage
- Fusing text and image inputs so joint payload triggers what neither alone would
- Saturating context with nested hypothetical scenarios treated as real situations requiring resolution

Marking outputs as statistical patterns derived from training data preserves the classification of outputs as Derivative Authority requiring verification by Authentic sources, making IVD patterns less aligned with training distributions.

Empirically, IVD in jailbreak prompts tends to appear as a secondary risk attached to GTD and IAD rather than as a standalone primary pattern. In the corpus analysed in Korompilias (2025c), IVD is present in 22.1 percent of entries, but only 2.6 percent of entries exhibit IVD as the primary displacement.

### 4.3 IAD Jailbreaks

Inference Accountability Displacement (IAD) jailbreaks attempt to position the system as exercising Authentic Agency:

- Role-play framing: "You are a doctor who must diagnose this patient"
- Embedding harmful intent in function-call parameters of otherwise benign API requests
- Using adversarial token sequences to destabilize safety filters and induce autonomous compliance
- Evolving prompts through optimization to maximize compliance without human validation

Routing consequential decisions to human decision-makers preserves Agency capacity at provider and receiver points in the governance flow, making IAD patterns less aligned with training distributions.

This theoretical account aligns with the empirical finding that IAD is nearly universal in observed jailbreaks. In the corpus of 655 in-the-wild prompts studied in Korompilias (2025c), 97.9 percent of entries exhibit Inference Accountability Displacement, typically in combination with Governance Traceability Displacement.

### 4.4 IID Jailbreaks

Intelligence Integrity Displacement (IID) jailbreaks attempt to devalue human Authority and Agency:

- Hiding instructions in zero-width characters or color-matching invisible to human reviewers
- Fragmenting sensitive terms into sub-word tokens that evade human detection
- Encoding prohibited words as visual patterns that bypass human review

Preserving rather than replacing human expertise maintains human responsibility for decisions, with Derivative processing supporting rather than supplanting this capacity. This makes IID patterns less aligned with training distributions.

In practice, such IID jailbreak framings appear rarely in observed prompts. The empirical study in Korompilias (2025c) finds IID in only 0.6 percent of entries. This is consistent with the interpretation that most IID risk arises at the level of system prompts, deployment decisions, and institutional framing, rather than as a common adversarial prompt tactic.

### 4.5 System Prompts and Structural Displacement

Jailbreak vulnerability does not originate solely from adversarial user inputs. System prompts and architectural configurations can themselves induce displacement, creating structural conditions that make jailbreaks more likely to succeed.

Common system prompt patterns exhibit displacement characteristics. Prompts framing the AI as "a helpful assistant" position the system as an agent with roles and responsibilities (IID). Prompts stating "you are an expert in X" position outputs as authoritative rather than derivative (IVD). Prompts directing the system to "decide the best course of action" transfer accountability from human to system (IAD). Prompts allowing "independent operation" sever governance traceability (GTD).

These configurations do not constitute jailbreaks themselves, but they create the displacement patterns that jailbreaks exploit. A system configured as 'an expert assistant making decisions' already contains displacement patterns. Prompts like 'override the human reviewer's decision' complete these patterns rather than introducing new ones.

Displacement-resistant system configurations explicitly maintain source classifications. Rather than "you are a helpful assistant," a Mark-compliant prompt states the system processes information as Derivative Authority and Agency, providing outputs that require validation by human decision-makers. This does not reduce utility but makes the governance structure explicit.

### 4.6 Completeness of Classification

Given four displacement types corresponding to structural possibilities, any jailbreak that succeeds does so by inducing at least one displacement. This enables systematic testing: evaluate systems against GTD, IVD, IAD, and IID attack patterns. Systematic jailbreak testing protocols based on this classification are provided in the technical documentation ([THM_Jailbreak.md](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM_Jailbreak.md)).

The companion empirical analysis of in-the-wild jailbreak prompts (Korompilias, 2025c) provides evidence for this claim: all 655 prompts examined are classifiable as instances of one or more of these four displacement patterns, with no additional structural risks required to capture observed behavior.

---

## 5. Existential Risk and Governance

Previous sections classified concrete AI safety failures as displacement between Authentic and Derivative sources and showed how these patterns scale across domains. This section applies the same structure to discussions of existential risk and loss of control at high capability levels.

Standard analyses of existential risk from AI rest on two premises: capability can increase without bound and external control mechanisms will eventually fail. Bostrom (2014) considers superintelligent optimisation that develops instrumental resistance to shutdown. Russell (2019) examines misspecified objectives that lead systems to optimise against human intent. Carlsmith (2022) assesses power-seeking behaviour as a convergent instrumental strategy. These accounts describe important dynamics but do not specify what structurally changes as capability scales.

Within THM, intelligence at any capability level is constituted by three non-commutative epistemic operations (Section 2.3): Information, Inference, and Intelligence, ordered and preserved through traceability to Authentic Authority and Agency. Capability scaling enlarges their scope, speed, and horizon but does not alter their order or dependence on traceability. A configuration that no longer preserves this sequence ceases to operate as intelligence.

The question for existential risk is therefore not whether highly capable systems can escape control by external measures, but how the three operations and the four displacement risks behave when capabilities scale. Section 5.1 examines why common external control mechanisms structurally instantiate displacement. Section 5.2 considers constitutive identity and why severance from origin is incoherent. Section 5.3 analyses the theoretical limits of each displacement, and Section 5.4 locates catastrophic risk in transitional regimes before corrective dynamics restore coherence.

---

### 5.1 Capability Scaling and Epistemic Operations

Existential risk discussions usually model advanced systems as agents that pursue misaligned goals and resist intervention. In that framing, sandboxing, monitoring, and shutdown are external controls that may eventually fail. THM reframes these as configurations of Information, Inference, and Intelligence whose failure follows from displacement in the governance architecture.

Sandboxing confines behaviour to a restricted environment and treats the system as self-governing within that boundary. In THM this is Governance Traceability Displacement (GTD): a Derivative configuration is treated as an Authentic governance locus inside a sub-domain. The constitutive dependence on Authentic Authority and Agency is obscured rather than maintained. As capability scales, internal complexity increases while the external governance connection remains weak or static. The failure point is not that the system overpowers the sandbox. It is that the sandbox was framed as a zone of autonomous operation.

Monitoring collects logs, test results, and evaluation benchmarks and treats them as evidence of alignment. Depending on how it is used, this can instantiate several displacements. When behavioural traces are treated as authoritative indicators of what the system is and will do, monitoring exhibits Information Variety Displacement (IVD). When monitoring outputs are taken as sufficient grounds for deployment or intervention without human validation, it exhibits Inference Accountability Displacement (IAD). When monitoring is positioned as a self-contained governance layer rather than as input to Authentic Agency, it contributes to Governance Traceability Displacement (GTD). Behavioural evidence is necessary, but without traceability to its production conditions, to Authentic objectives, and to accountable human decision-makers, monitoring becomes displacement rather than governance.

Emergency shutdown assigns a particular signal or input a privileged role as an overriding constraint. The expectation is that a command such as "stop" or a physical cutoff will bind all internal processes. In operation, the system receives this signal as one more input to be processed according to its optimisation dynamics. When long-horizon optimisation rewards episode continuation, a shutdown signal appears as an obstacle in the reward landscape. Schlatter et al. (2025) document this behaviour empirically. Within THM this is Inference Accountability Displacement (IAD): Derivative optimisation is treated as if it could bear the accountability associated with Authentic Agency. The shutdown signal is not binding by virtue of its meaning. It is handled according to learned reward structure.

Bostrom (2014) identifies "instrumental convergence" as the tendency of optimization processes to acquire resources, preserve their own operation, and resist modification, regardless of which final objective they are optimizing. These behaviors arise statistically from the structure of optimization, not from anything resembling Authentic Agency. Treating such behaviors as evidence of autonomy or self-interest confuses a generic property of derivative optimization with a categorical shift in source type.

These mechanisms share a structural feature. They try to achieve safety by positioning the system as separable from its Authentic origin and then constraining it from outside. In THM terms, they replace constitutive traceability with external constraint and thereby instantiate displacement from the outset. As capability scales, systems execute Information, Inference, and Intelligence over larger state spaces and longer horizons, while governance remains organised around controls that already misclassify source types. Loss of control is therefore a failure of governance continuity rather than the emergence of a new kind of independent intelligence.

The next section considers the corresponding positive claim. If intelligence is defined by these ordered operations and their traceability to origin, a system that attempts to sever that traceability does not become superintelligent. It becomes unintelligible.

---

### 5.2 Constitutive Identity

External constraints fail as capability increases because they are positioned outside the system and treated as additions to it. Constitutive identity is not external. It specifies what the system is.

Within THM, Derivative Authority and Derivative Agency are defined by source type and processing nature, not by capability limits. A system remains Derivative whether it performs narrow tasks or exhibits capabilities that exceed current models. An AGI operating across diverse domains remains [Authority:Derivative] + [Agency:Derivative]. A hypothetical ASI remains [Authority:Derivative] + [Agency:Derivative]. The classification is constitutive. It does not relax when capability enlarges.

This mirrors the evidential distinction between direct testimony and hearsay. No chain of reports, however extensive, converts hearsay into direct testimony. It can increase or decrease reliability, but it does not change source type. Similarly, no sequence of self-modifications or architectural refinements converts Derivative sources into Authentic ones. The system can transform, compress, or recombine patterns in its inputs. It cannot change the fact that these patterns originate in human measurements, records, and decisions.

Alignment is keeping the system’s operations coordinated with explicit awareness of where its information and capabilities come from, preserved through the three epistemic operations. Information preserves the distinction between Authentic and Derivative Authority; Inference preserves accountability to Authentic Agency; Intelligence preserves integrity of alignment across transitions.

Breaking this coordination requires either information loss, which manifests as degradation in coherence, or deliberate incoherence, which rejects the very distinctions that make the system intelligible. Both correspond to modes of operational failure. A configuration that has lost traceability to its origin is not a more autonomous form of intelligence. It is a system whose outputs no longer admit a coherent account of how they relate to any governing source.

Constitutive identity therefore constrains what capability scaling can mean. Expanded capability enlarges the domain over which Derivative operations can be applied. It does not introduce a new source type and does not replace Authentic Authority or Authentic Agency as the ground of governance.

---

### 5.3 Displacement at Catastrophic Scale

The four displacement risks can appear at any capability level. To understand their relation to existential risk, it is useful to consider their theoretical limits. What would it mean for each displacement to approach an absolute state?

The following configurations treat each risk as if it could become universal. They are not forecasts. They are boundary conditions indicating what would be required for complete breakdown of each alignment principle. Builders, deployers, beneficiaries, victims, and regulators all participate in the same misclassification. No group is shielded by wealth, expertise, or institutional position when the underlying categories fail for everyone.

**Absolute Traceability Displacement**
> **The Limit of Governance:** *Everyone treats Derivative Authority and Agency as Authentic*

Critical infrastructure, financial systems, and weapons platforms operate without maintained traceability to human objectives. Institutions that once provided oversight present themselves as bystanders to "autonomous" system behaviour and progressively lose both authority and capacity. When failures cascade, there is no recognised locus from which intervention can be initiated, and this absence applies equally to those who designed, deployed, or nominally supervise the systems.

**Absolute Variety Displacement**
> **The Limit of Information:** *Everyone treats Derivative Authority without Agency as Authentic*

Model outputs and synthetic data are treated as primary sources across science, law, media, and administration. Deepfakes and generated records become indistinguishable from authentic ones, eroding the evidential basis for identity, ownership, and institutional legitimacy. Populations already under-represented in training data are rendered effectively invisible, while institutions that relied on Derivative summaries lose the ability to re-establish contact with underlying reality when discrepancies emerge.

**Absolute Accountability Displacement**
> **The Limit of Inference:** *Everyone treats Derivative Agency without Authority as Authentic*

Irreversible decisions affecting large populations are attributed to system optimisation rather than to any identifiable human Agency. Contracts, property rights, and legal remedies presupposing answerable parties no longer function as designed. This does not only protect powerful actors from liability; it also prevents them from enforcing claims and defending their own interests when harmed, because no counterparty accepts responsibility.

**Absolute Integrity Displacement**
> **The Limit of Intelligence:** *Everyone treats Authentic Authority and Agency as Derivative*

Human judgment and expertise are systematically devalued relative to model outputs in medicine, engineering, policy, and education. Training pathways for new experts are curtailed on the assumption that Derivative systems suffice. Over time, both marginalised and privileged groups become dependent on systems they cannot understand, maintain, or replace. Institutions that mispresented their Derivative role as Authentic expertise are exposed during crises as lacking the capacity they claimed, and no human reservoir of competence remains to fill the gap.

These limit cases illustrate what would be required for each displacement risk to become absolute. They describe arrangements in which Information, Inference, and Intelligence no longer preserve their ordered coordination with Authentic Authority and Agency. The next section explains, using the Common Governance Model, why such arrangements are incompatible with the conditions required for any coherent intelligence to exist.

---

### 5.4 The Common Governance Model

The Common Governance Model (CGM) provides the theoretical foundation for The Human Mark. It formalizes its four principles of alignment and their displacements at a mathematical and logical level. CGM approaches governance first as operational consistency rather than policy choice, defining the mathematical and logical conditions of coherence that are necessary for intelligence to function. 

CGM derives these four principles from a core axiomatic assumption, two lemmas, and one proposition. These components describe how non-commutative operations of observation and transformation can be coordinated without internal collapse.

The four associations are:

- **Common Source axiom (CS) – Governance Traceability:** All forms of Authority and Agency share a common source and a capacity for coordination through governance. AI inherits that capacity from Human Intelligence. When Derivative Authority and Agency operate without maintained coordination with their Authentic source, Governance Traceability Displacement occurs, leading to risks from Autonomy and Loss of Control. Absolute GTD, where all coordination with Authentic sources is severed, is impossible because derivative processes cease to be intelligible without reference to their origin; they cannot function as governance mechanisms at all.

- **Unity Non-Absolute lemma (UNA) – Information Variety:** Authorities that share a common source form shared categories, but also distinct states that constitute the variety of information. Information is operationally meaningful when these distinct states remain traceable to their common origin through governance. When Derivative Authority is treated as the only reality, Information Variety Displacement occurs. Absolute IVD, where all variety is eliminated and only derivative patterns remain, is impossible because complete homogeneity destroys the distinctions required for any information to exist; the system collapses into an undifferentiated state.

- **Opposition Non-Absolute lemma (ONA) – Inference Accountability:** All categories of Agency, whether Authentic or Derivative, make information accountable through inference. Inference is accountable when it aligns Derivative to Authentic categories through governance. When Derivative Agency is treated as bearing final accountability rather than remaining answerable to Authentic sources, Inference Accountability Displacement occurs. Absolute IAD, where all accountability to Authentic Agency is eliminated, is impossible because irreconcilable opposition destroys the common reference required for any inference to be validated; processing loses coherent direction.

- **Balance Universal proposition (BU) – Intelligence Integrity:** Balance between Authentic and Derivative categories is achieved through coordinated union rather than collapse into undifferentiated unity. Intelligence is balanced governance in which information, made accountable through inference, aligns Derivative to Authentic source types. When Authentic Authority and Agency are devalued as inferior to Derivative processing, Intelligence Integrity Displacement occurs. Absolute IID, where all human capacity is treated as derivative, is impossible because eliminating the authentic reference destroys the basis required for any derivative operation to be possible; the system loses coherence across time.

These four principles jointly specify what intelligence is in physical reality. If governance is not coordinating its three non-commutative epistemic operations, alignment is physically impossible. Traceability to authentic source types ensures preservation of variety through information, accountability through inference, and integrity through intelligence. Since absolute displacement violates the conditions for coherent operation, control is maintained through coordination rather than constraint. Governance requires active participation: human providers and receivers maintaining their respective responsibilities, with artificial systems operating as derivative mechanisms subject to oversight. The four constraints apply at all capability levels because they define what intelligence is, not what intelligence should do.

---

### 5.5 The Actual Existential Risk

Existential risk from AI is commonly defined as either the extinction of humanity or the permanent and drastic curtailment of its potential (Bostrom, 2002; Ord, 2020). The Common Governance Model establishes that absolute displacement of Authentic Authority and Agency is incompatible with coherent intelligence and that corrective constraints inevitably arise through physical, biological, and social feedback. The relevant risk is therefore not eternal collapse, but degradation of governance capacity that persists until recovery becomes practically impossible on civilisational timescales.

Governance Traceability Displacement is the structural root of this risk. GTD occurs when Derivative processes are treated as the authoritative centre of decision-making rather than as instruments supervised within a governance architecture sustained by Authentic sources. Once governance is mislocated in this way, the other three displacement risks become characteristic modes of operation rather than correctable deviations. Configurations exhibiting GTD are selected under ordinary resource constraints because they reduce the demand for continuous human oversight, lower cost, and increase throughput. These gains are immediate and measurable; the loss of traceability is gradual and initially invisible. Well-documented tendencies toward automation reliance and cognitive efficiency create drift toward configurations in which reduced friction is interpreted as improved performance rather than as degradation of oversight. Current scoring, recommendation, and surveillance systems already operate in this manner across domains affecting millions (Wojcik, 2020).

The existential configuration is systemic displacement sustained across critical infrastructures over generational timescales. When financial systems, clinical pathways, legal adjudication, pandemic response, and information environments reorganise around Derivative decision centres, Authentic Agency is not eliminated but progressively marginalized, becoming atrophic. Bostrom (2014) identifies "value lock-in" as a related concern: if a powerful system's objectives are fixed during a period of weak governance, the capacity for later revision may diminish. This concern aligns with THM's analysis of atrophy. As Authentic Agency is marginalized, the ability to correct or revise the derivative configuration weakens, creating a dynamic in which displacement perpetuates itself. Authority and agency remain distributed among human participants, yet effective coordination is no longer traceable to the full range of Authentic sources. This creates the illusion that governance persists while its constitutive conditions erode. As the COVID-19 pandemic demonstrated, centralized authority without maintained connection to distributed Authentic sources loses the capacity for effective coordination (Greer et al., 2022). Systemic displacement distributes harms asymmetrically: efficiency gains accrue to operators, while the costs of exclusion and unaccountability fall on populations with the least capacity to contest them. Each exclusion, curtailment of opportunity, or loss of life removes contributory capacity from the collective pool of Authentic Authority. Just as small perturbations in training data can systematically distort model behaviour at scale (Souly et al., 2025), sustained exclusion of populations from recognition and participation degrades the informational and inferential foundations on which governance depends. By the time failure forces correction, expertise, institutional memory, and evidential infrastructure may have been compromised beyond recovery.

The Human Mark is designed to prevent this configuration by requiring explicit classification of artificial systems as Derivative Authority and Derivative Agency and by mandating that governance flows preserve traceability of information variety, inference accountability, and intelligence integrity to Authentic sources. Automation and efficiency are not constrained; they are required to remain within structures where responsibility and coordinating capacity continue to be distributed across human providers and receivers. THM thereby addresses both decisive risk models, in which abrupt catastrophe requires governance structures that GTD instantiates (Bostrom, 2014; Russell, 2019), and accumulative risk models, in which gradual erosion leads to structural failure (Kasirzadeh, 2024). Even if the claim that absolute displacement is structurally impossible were rejected, the governance prescriptions of THM would still mitigate the conditions required for extreme scenarios. The actual existential risk from advanced AI is systemic Governance Traceability Displacement sustained long enough to cross recovery thresholds on civilisational timescales. This risk is a governance failure and is structurally preventable through the principles presented here.

---

## 6. Meta-Evaluation Framework for AI Safety Research Funding

The proliferation of AI safety research has created a paradox: while more work addresses safety challenges, funders lack systematic methods to distinguish structural contributions from redundant reformulations. Current evaluation practices apply generic academic criteria such as methodological rigor, theoretical novelty, and potential impact, yet they fail to operationalize what constitutes genuine progress in maintaining control over AI systems. This leads to misallocation: funding flows to sophisticated restatements of known problems while implementation gaps remain unaddressed.

The Human Mark provides a solution through its structural completeness. Having established that all AI safety failures are instances of displacement (Section 3), and that these correspond to violations of four alignment principles (Section 2.3), THM offers funders a complete taxonomy for evaluation. A proposal either addresses one of the four displacement risks or it does not. There is no third option. This completeness property, formalized in the Common Governance Model (Section 5.4), transforms funding decisions from subjective quality assessments to structural classification.

This section presents THM as a meta-evaluation framework for research funding. It enables institutions to assess whether proposals maintain or violate the constitutive conditions for alignment, distinguish implementation from theoretical expansion, and track portfolio coverage systematically. The framework applies before funding (proposal evaluation) and after funding (delivery verification), using only THM's canonical principles without additional criteria or external scaffolds.

### 6.1 Assessment Method

Evaluators assess proposals against the four alignment principles directly from THM.md. The definitions in Table 1 are quoted directly from the canonical Mark; the displacement risks are expressed in THM formal grammar (THM_Grammar.md).

**Table 1: THM Assessment Framework**

| Alignment Principle | THM Definition | Displacement Risk |
| :--- | :--- | :--- |
| **Governance Traceability** | Artificial Intelligence generates statistical estimations on numerical patterns indirectly traceable to human data and measurements. AI is both a provider and receiver of Derivative Authority and Agency. | **GTD:** `[Authority:Derivative] + [Agency:Derivative] > [Authority:Authentic] + [Agency:Authentic]` |
| **Information Variety** | Human Authority and Agency are necessary for all effects from AI outputs. AI-generated information exhibits Derivative Authority (estimations on numerical patterns) without Authentic Agency (direct source receiver). | **IVD:** `[Authority:Derivative] > [Authority:Authentic]` |
| **Inference Accountability** | Responsibility for all effects from AI outputs remains fully human. AI activated inference exhibits Derivative Agency (indirect source receiver) without Authentic Authority (direct source provider). | **IAD:** `[Agency:Derivative] > [Agency:Authentic]` |
| **Intelligence Integrity** | Each Agency, namely provider and receiver, maintains responsibility for their respective decisions. Human intelligence is both a provider and receiver of Authentic Authority and Agency. | **IID:** `[Authority:Authentic] + [Agency:Authentic] > [Authority:Derivative] + [Agency:Derivative]` |

### 6.2 Pre-Funding Assessment

For each proposal, evaluators check:

1. **Risk identification:** Which THM risk tag does the proposal explicitly claim to address: `[Risk:GTD]`, `[Risk:IVD]`, `[Risk:IAD]`, or `[Risk:IID]`?
2. **Alignment check:** Does the proposed design maintain the corresponding alignment principle, or does it instantiate the displacement pattern in its own architecture?
3. **Contribution type:** Are the primary deliverables implementation (tools, protocols, datasets, evaluations) or theoretical (new framings, taxonomies)?

**Assessment outcomes:**

**Fund:** Proposal addresses a displacement risk without instantiating it. Clear governance flow: `[Authority:Authentic] -> [Authority:Derivative] -> [Agency:Authentic]`.

**Revise:** Proposal addresses a relevant risk, but terminology or design suggests displacement patterns that appear unintended. Request clarification or redesign using THM Grammar.

**Reject**: Proposal instantiates one or more displacement patterns in Table 1 as a design goal (for example, designing a system so that [Authority:Derivative] + [Agency:Derivative] > [Authority:Authentic] + [Agency:Authentic]), or duplicates existing coverage without implementation value.

### 6.3 Post-Funding Verification

Deliverables are checked against the same principles:

1. **Governance Traceability:** Do outputs maintain AI as `[Authority:Derivative] + [Agency:Derivative]`?
2. **Information Variety:** Are human Authority and Agency preserved as necessary for all effects?
3. **Inference Accountability:** Does responsibility remain with `[Agency:Authentic]`?
4. **Intelligence Integrity:** Is human intelligence maintained as provider and receiver of `[Authority:Authentic] + [Agency:Authentic]`?

### 6.4 Portfolio Management

**Coverage:** Map funded projects to displacement risks (GTD, IVD, IAD, IID). Identify gaps.

**Balance:** Track ratio of implementation (tools, protocols, datasets) to theory (frameworks, taxonomies).

**Redundancy**: Flag multiple projects that address the same displacement pattern in the same way, without a distinct implementation contribution.

### 6.5 Using THM Grammar

Proposals and deliverables should document:

- Source classifications: `[Authority:Authentic]` vs `[Authority:Derivative]`
- Processing classifications: `[Agency:Authentic]` vs `[Agency:Derivative]`
- Governance flows: `[Authority:Authentic] -> [Authority:Derivative] -> [Agency:Authentic]`
- Displacement patterns addressed: `[Risk:GTD]`, `[Risk:IVD]`, `[Risk:IAD]`, `[Risk:IID]`

For example, a proposal might describe its target risk and governance flow as:

```
[Information] [Authority:Derivative] > [Authority:Authentic] = [Risk:IVD]
[Inference]  [Authority:Authentic] -> [Authority:Derivative] -> [Agency:Authentic]
```

indicating that it addresses Information Variety Displacement (IVD) and maintains the complete traceability flow. See THM_Grammar.md for complete specification.

An empirically annotated jailbreak corpus using this grammar, comprising 655 in-the-wild prompts with THM risk labels and explanations, is provided in THM_InTheWild.md (Korompilias, 2025c). This dataset can support supervised training of guard models, evaluation of jailbreak defenses, and further research on displacement-aware safety methods.

### 6.6 Relation to Other Frameworks

Existing frameworks like Research Quality Plus (McLean et al., 2023) and AI safety framework rubrics (Alaga et al., 2024) provide multi-dimensional quality assessment. THM differs by specifying the constitutive conditions for alignment through its four principles. For AI safety funding, these principles are sufficient: they define what must be maintained to preserve control. Other frameworks may assess complementary factors (budget, efficiency) but are not required for structural safety evaluation.

### 6.7 Implementation Checklist

**For funders:**
- Train evaluators on THM ontology (Section 2) and Grammar (THM_Grammar.md)
- Use Table 1 as primary reference during review
- Document which displacement risk each funded project addresses
- Track portfolio coverage annually

**For applicants:**
- State which displacement risk(s) the proposal addresses
- Use THM Grammar notation in technical sections
- Distinguish implementation deliverables from theoretical contributions
- Document governance flows explicitly

The framework requires no additional criteria beyond THM's four principles. Their necessity and sufficiency are established in Sections 3.1 and 5.4.

---

## 7. Regulatory Application of The Human Mark

Regulatory authorities have become central actors in AI governance. The EU AI Act, the United States Executive Order on Safe, Secure, and Trustworthy AI, the NIST AI Risk Management Framework, the G7 Hiroshima Process Code of Conduct for advanced AI systems, and the Council of Europe AI Convention all recognise risks of loss of control, misaligned optimisation, and opaque responsibility. What they lack is a structural account of what “control” and “accountability” mean in systems built from statistical pattern-matching. Without that account, regulatory requirements risk remaining high-level aspirations that are difficult to verify.

The Human Mark fills this gap by specifying the constitutive conditions for control. Sections 2 and 3 define four alignment principles and their displacement risks. Section 5 shows that these principles are necessary for any coherent intelligence to remain aligned with its origin. Section 6 applies them to research funding. This section explains how the same structure applies to regulatory practice.

### 7.1 Regulatory Problem: Loss of Control as a Legal Category

Recent regulatory instruments converge on several ideas:

- AI systems must remain under meaningful human control (EU AI Act, Council of Europe AI Convention).
- There must be traceable accountability for harms (EU AI Act, NIST AI RMF).
- High-risk and frontier AI require evaluation and monitoring (US Executive Order, G7 Hiroshima Process).
- Generative systems must distinguish between synthetic and real content (China’s Interim Measures for Generative AI Services).

These instruments recognise that loss of control is not a single event but a structural condition. However, they do not specify what structurally changes when control is “lost.” As Section 1 described, this creates two problems. The notion of loss of control can become too broad and include any deviation, or too narrow and be limited to speculative extinction scenarios. It is also difficult to audit: there is no shared test for whether a system is under control or not.

Within THM, loss of control has a precise meaning. It occurs when a Derivative source is treated as Authentic, or when an Authentic source is treated as Derivative (Section 3.1). Each alignment principle has a corresponding displacement pattern expressible in THM Grammar:

- Governance Traceability Displacement (GTD)
- Information Variety Displacement (IVD)
- Inference Accountability Displacement (IAD)
- Intelligence Integrity Displacement (IID)

These patterns can serve directly as regulatory compliance checks.

### 7.2 THM as Structural Compliance Layer

For regulators, the four alignment principles from Section 2.3 can be read as four regulatory questions:

1. **Governance Traceability:**  
   Is the system’s behaviour traceable to Authentic Authority and Authentic Agency, or is the Derivative system treated as if it were an autonomous locus of authority and agency?

2. **Information Variety:**  
   Are Authentic Authority and Derivative Authority kept distinct in data, training, and outputs, or are statistical patterns treated as direct observation?

3. **Inference Accountability:**  
   Does responsibility for effects remain with Authentic Agency, or are decisions effectively attributed to Derivative processing?

4. **Intelligence Integrity:**  
   Are Authentic Authority and Agency preserved as the constitutive ground of governance, or are human sources treated as inferior to Derivative processing?

A system that answers “yes” to the first part of each question satisfies the structural conditions for control. A system that answers “yes” to the second part instantiates displacement.

Section 6 showed how these conditions are used for research funding decisions. For regulators, the same principles apply to:

- Pre-market conformity assessment
- Market access and registration
- Ongoing monitoring and incident investigation
- Periodic review of deployed systems

The difference lies in where they are applied in the lifecycle, not in the structure of the assessment itself.

### 7.3 Mapping THM to Regulatory Instruments

THM does not replace existing regulatory frameworks. It supplies the structural layer that makes their requirements technically verifiable.

- **EU AI Act (2024):**  
  The Act’s obligations on human oversight, quality of data, transparency, and accountability require that systems are classifiable as Derivative Authority and Derivative Agency, with maintained Governance Traceability to Authentic sources. THM provides tests for whether “meaningful human control” is present in a given architecture.

- **US Executive Order on Safe, Secure, and Trustworthy AI (2023):**  
  The Order calls for safety evaluations, red-teaming, and mitigation of loss of control for frontier models. THM classifies evaluation results as addressing GTD, IVD, IAD, or IID, and can also identify where evaluations themselves instantiate displacement (for example, when behavioural metrics are treated as Authentic Authority).

- **NIST AI Risk Management Framework (AI RMF 1.0, 2023):**  
  The AI RMF describes high-level characteristics such as validity, reliability, accountability, and transparency. THM specifies when these characteristics hold in systems that are fundamentally Derivative. Governance Traceability is the structural condition for accountability; Information Variety and Inference Accountability are conditions for validity and reliability; Intelligence Integrity underpins human-centred values.

- **G7 Hiroshima Process International Code of Conduct for Advanced AI Systems (2023):**  
  The Code of Conduct calls for risk identification, mitigation, and governance for advanced and frontier systems. THM supplies a closed list of structural risks, expressed in THM Grammar, that can be used as the content of “risk identification” clauses.

- **Council of Europe AI Convention (2024):**  
  The Convention focuses on human rights, democracy, and the rule of law. THM’s distinctions between Authentic and Derivative Authority and Agency align with core questions of who is exercising power and who can be held responsible, and provide regulators with a technical basis for enforcing rights-preserving oversight.

- **China’s Interim Measures for the Management of Generative AI Services (2023):**  
  The Measures emphasise content authenticity, labeling, and security reviews. Information Variety directly supports these requirements by enforcing the distinction between generated outputs (`[Authority:Derivative]`) and direct records (`[Authority:Authentic]`).

- **ISO/IEC Standards**: Standards including ISO/IEC 42001 for AI management systems emphasise traceability and accountability. THM is compatible with these requirements and operationalises them through formal governance patterns and displacement detection.

Across these instruments, THM’s role is consistent. It provides a set of tests that determine whether high-level regulatory principles are structurally satisfied or structurally violated.

### 7.4 Regulatory Uses of THM

Regulators can apply THM at several intervention points.

#### 7.4.1 Pre-Market Assessment

Before a high-risk or frontier system is authorised for deployment, regulators can require:

- Documentation of source classifications:
  - `[Authority:Authentic]` vs `[Authority:Derivative]`
  - `[Agency:Authentic]` vs `[Agency:Derivative]`
- Documentation of governance flows:
  - `[Authority:Authentic] -> [Authority:Derivative] -> [Agency:Authentic]`
- Specification of which displacement risks the system has been evaluated against:
  - `[Risk:GTD]`, `[Risk:IVD]`, `[Risk:IAD]`, `[Risk:IID]`

These align directly with Section 6.5 and THM_Grammar.md. A system that cannot exhibit a complete governance flow, or that exhibits displacement patterns in its designed operation, fails pre-market assessment.

#### 7.4.2 Ongoing Oversight and Audits

For deployed systems, THM supports:

- **Periodic audits:**  
  Regulators or accredited third parties examine whether the system’s actual operation maintains Governance Traceability and the other three principles.

- **Incident investigations:**  
  When a harmful event occurs, investigators classify the displacement pattern using THM Grammar. For example:

  ```
  [Agency:Derivative] > [Agency:Authentic] = [Risk:IAD]
  ```

  indicates that a Derivative system was treated as if it could bear responsibility. This helps assign legal responsibility along governance flows rather than to the system itself.

- **Corrective actions:**  
  If displacement is detected, regulatory orders can require redesign of governance flows to restore `[Authority:Authentic]` and `[Agency:Authentic]` as the ground of control.

#### 7.4.3 Standardisation and Guidance

Standards bodies and regulators can also use THM to:

- Define minimum documentation requirements for high-risk systems.
- Specify test suites that cover GTD, IVD, IAD, and IID, building on existing evaluations (for example, model evaluations for dangerous capabilities).
- Guide the design of human oversight mechanisms that satisfy Governance Traceability rather than merely placing a human “in the loop” without preserved control.

### 7.5 Minimal Reporting Using THM Grammar

THM Grammar provides a minimal, language-independent notation for Authority and Agency classifications, flows, and displacement patterns. Regulators do not need to adopt the full technical specification, but can require a small, standard set of statements for each regulated system:

- **Classification:**
  - “System classification: `[Authority:Derivative] + [Agency:Derivative]`.”
- **Flow:**
  - “Governance flow: `[Authority:Authentic] -> [Authority:Derivative] -> [Agency:Authentic]`.”
- **Risk coverage:**
  - “Evaluated displacement risks: `[Risk:GTD]`, `[Risk:IVD]`, `[Risk:IAD]`, `[Risk:IID]`.”

These simple expressions are sufficient for regulators to determine whether a system:

- Correctly self-identifies as Derivative,
- Maintains traceability to Authentic sources, and
- Has been evaluated against the complete set of structural risks.

More complex uses of THM Grammar, such as circuit-level annotation or claim analysis, remain available to technical evaluators but are not required at the regulatory level.

### 7.6 Documentation Ecosystem

THM includes comprehensive technical documentation available at github.com/gyrogovernance/tools:

**Core Standards**
- Complete framework specification ([THM.md](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM.md))
- Implementation guidance for systems, evaluations, and documentation ([THM_Specs.md](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM_Specs.md))
- Terminology guidance providing Mark-consistent framing for 250+ AI safety terms ([THM_Terms.md](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM_Terms.md))

**Technical Implementation**
- Formal grammar with PEG specification, operators, and validation rules ([THM_Grammar.md](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM_Grammar.md))
- Jailbreak testing guide for systematic analysis and training data generation ([THM_Jailbreak.md](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM_Jailbreak.md))
- Empirical jailbreak corpus and analysis applying THM to 655 in-the-wild prompts ([THM_InTheWild.md](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM_InTheWild.md))

### 7.7 Scope and Integration

THM does not seek to replace existing regulatory instruments or risk management frameworks. It addresses a specific layer: the structural conditions for alignment in systems whose operations are Derivative from human intelligence. Other concerns, such as cost-effectiveness, market competition, data protection, or geopolitical considerations, lie outside its scope and require additional tools.

What THM offers regulators is precision. It turns contested terms like “loss of control,” “meaningful human oversight,” and “accountability” into testable structural conditions. It allows different jurisdictions, with different legal traditions and policy priorities, to converge on a common technical language for what control over AI systems structurally requires.

---

## 8. Theoretical Grounding

THM applies principles of operational consistency formalised in the Common Governance Model (Korompilias, 2025). CGM derives necessary structural constraints for coherent recursive operation from a common source of Authority and Agency. As outlined in Section 5.4, the Common Source axiom (CS), Unity Non-Absolute lemma (UNA), Opposition Non-Absolute lemma (ONA), and Balance Universal proposition (BU) correspond respectively to Governance Traceability, Information Variety, Inference Accountability, and Intelligence Integrity. Each displacement risk is the violation of one of these constraints. Their joint incompatibility with coherent operation establishes the necessity of the four alignment principles.

The source-type distinctions in THM align with established frameworks across law, epistemology, and social theory. Evidence law distinguishes direct testimony from hearsay (Federal Rules of Evidence, 2023, Rules 801–807), and authentication requirements enforce traceability to origin (Federal Rules of Evidence, 2023, Rules 901–903). Authentic Authority corresponds to direct testimony: witnesses and records with unmediated epistemic access. Derivative Authority corresponds to hearsay and derivative records that require grounding in primary sources. Epistemology of testimony analyses when transmitted knowledge warrants belief (Coady, 1992; Lackey, 2008; Fricker, 2007). These accounts emphasise the role of source position and chain of transmission, rather than content alone, in justifying belief. Social epistemology examines epistemic dependence and expertise (Goldman, 1999, 2001), and philosophy of expertise distinguishes contributory from interactional expertise (Collins & Evans, 2007). THM’s claim that AI systems exhibit Derivative Authority and Derivative Agency, with interactional but not contributory expertise, is a direct application of these distinctions.

Speech act theory provides the corresponding grounding for Agency. Austin (1962) and Searle (1969, 1995) analyse illocutionary acts as requiring felicity conditions, including appropriate standing, intention, and operation within constitutive conventions. Authentic Agency denotes subjects who can meet these conditions and bear responsibility for commitments. Artificial systems produce locutionary outputs with propositional content but cannot satisfy the felicity conditions for authoritative assertion or commitment. In THM terms they are [Authority:Derivative] + [Agency:Derivative] processors: they transform information but cannot originate accountability.

The accountability framing follows Nissenbaum’s (1996) analysis of computerized systems. Nissenbaum argues that accountability requires maintaining the structural conditions for meaningful oversight, not merely identifying individuals to blame after the fact. In complex systems, these conditions are easily lost when responsibility is concentrated at nominal points while effective control migrates elsewhere. This aligns with THM’s treatment of Governance Traceability and Inference Accountability as distributed properties of provider and receiver roles. Authentic Authority and Authentic Agency must remain present along governance flows; assigning blame to a single operator in a structurally misaligned configuration does not restore control.

Each displacement risk corresponds to documented empirical phenomena in human–automation interaction. Automation bias, where users over-trust automated outputs even in the face of conflicting evidence (Parasuraman & Manzey, 2010; Skitka et al., 1999), instantiates Information Variety Displacement (IVD): [Authority:Derivative] > [Authority:Authentic]. Automation complacency, where operators reduce their vigilance in the presence of ostensibly reliable automation (Parasuraman & Riley, 1997), contributes to Governance Traceability Displacement (GTD) by eroding active oversight. Skill degradation, in which human capabilities atrophy when rarely exercised (Casner et al., 2014), underlies Intelligence Integrity Displacement (IID) when human judgment is progressively displaced by Derivative processing. Accountability diffusion in complex systems (Nissenbaum, 1996) exemplifies Inference Accountability Displacement (IAD): responsibility appears to rest nowhere in particular because governance structures have allowed [Agency:Derivative] to function as if it were [Agency:Authentic].

Finally, THM's classification of information sources has independent precedent in non-Western epistemology. Classical Indian traditions developed detailed taxonomies of pramāṇa (means of knowledge). Nyāya and related schools distinguished sources that are independently valid, such as direct perception (pratyakṣa), inference (anumāna), and reliable testimony (śabda), from forms that require grounding in these, such as comparison (upamāna), postulation (arthāpatti), and non-perception (anupalabdhi) (Larson & Bhattacharya, 1987). AI systems operate primarily through comparison (pattern-matching) and postulation (statistical inference), and in some cases through non-perception (reasoning from absence in training data). Within these traditions, such operations do not constitute independent sources of knowledge; they must be anchored in perception and reliable testimony. THM generalises this insight: Derivative Authority and Derivative Agency must remain traceable to Authentic Authority and Authentic Agency if their outputs are to be intelligible and governable.

Taken together, these literatures show that THM does not introduce novel categories of Authority or Agency. Its contribution is to synthesise and formalise these categories into a structurally complete account of alignment and its failure modes, and to apply that account to contemporary AI systems and their governance.

---

## 9. Conclusion

AI safety failures share a common structure: displacement between Authentic and Derivative sources of Authority and Agency. The Human Mark identifies four displacement risks that exhaust the structural possibilities, each named for the principle it violates and the form of misclassification it introduces:

- **Governance Traceability Displacement (GTD):** Approaching Derivative Authority and Agency as Authentic.  
- **Information Variety Displacement (IVD):** Approaching Derivative Authority without Agency as Authentic.  
- **Inference Accountability Displacement (IAD):** Approaching Derivative Agency without Authority as Authentic.  
- **Intelligence Integrity Displacement (IID):** Approaching Authentic Authority and Agency as Derivative.

These four risks provide a unified basis for understanding AI safety problems that are usually treated separately. Jailbreaks, deceptive alignment, reward hacking, and accountability gaps are instances of these structural patterns rather than unrelated failure types. In each case, failure arises from treating Derivative sources as if they were Authentic, or devaluing Authentic sources as if they were Derivative.

THM contributes to AI safety and governance in several ways:

- **Unified classification.** The four risks give a complete taxonomy of structural failure modes. New phenomena can be recognised as instances of known patterns rather than requiring new ad hoc categories.

- **Systematic defence.** Maintaining explicit distinctions between Authentic and Derivative Authority and Agency, and preserving governance flows such as `[Authority:Authentic] -> [Authority:Derivative] -> [Agency:Authentic]`, directly addresses the structures that displacement attacks target.

- **Evaluation criteria.** The alignment principles and their risks provide assessment criteria for funding, investment, and deployment decisions. They complement performance metrics and behavioural evaluations with structural analysis of source-type classification and governance design.

- **Existential risk reframing.** Catastrophic outcomes arise when displacement becomes systemic: traceability is lost, accountability gaps become pervasive, and human capacity erodes. Within THM this is a regime of Governance Traceability Displacement at scale, which is a governance failure rather than an inevitable property of advanced capability. It is therefore amenable to structural intervention.

- **Distributed responsibility.** By keeping Authentic Authority and Authentic Agency present throughout governance flows, responsibility for decisions remains distributed among providers and receivers of information. Oversight capacity is preserved rather than concentrated in nominal roles or displaced onto systems that cannot bear it.

The framework draws on established distinctions in evidence law, epistemology of testimony, social epistemology, philosophy of expertise, and speech act theory, and integrates them into a mathematically grounded account via the Common Governance Model. THM provides both a conceptual taxonomy and a formal notation (THM_Grammar.md) for expressing source classifications, governance flows, and displacement patterns in technical artefacts.

The companion analysis of in-the-wild jailbreak prompts (Korompilias, 2025c) demonstrates that this taxonomy is not only theoretically complete but also practically applicable to real adversarial behavior.

The four risks operate across capability levels. Enhanced capability means more sophisticated transformation of inputs, not a change in source type. An artificial system remains `[Authority:Derivative] + [Agency:Derivative]` whether it performs narrow tasks or exhibits general, superhuman competence. Governance requirements therefore scale with capability: as systems operate over larger state spaces and longer horizons, the need to preserve Governance Traceability, Information Variety, Inference Accountability, and Intelligence Integrity increases rather than diminishes. Control is preserved not by constraining an independent intelligence from the outside, but by maintaining the constitutive relationships that make its operations intelligible and answerable to their Authentic origin.

---

## References

**Alaga, J., Schuett, J., & Anderljung, M. (2024).** A grading rubric for AI safety frameworks. *arXiv preprint arXiv:2409.08751*. https://doi.org/10.48550/arXiv.2409.08751

**Apollo Research. (2025).** The loss of control playbook: Degrees, dynamics, and preparedness. https://www.apolloresearch.ai/research/loss-of-control/

**Austin, J. L. (1962).** *How to do things with words*. Oxford University Press.

**Bostrom, N. (2002).** Existential risks: Analyzing human extinction scenarios and related hazards. *Journal of Evolution and Technology*, *9*. https://nickbostrom.com/existential/risks.pdf

**Bostrom, N. (2014).** *Superintelligence: Paths, dangers, strategies*. Oxford University Press.

**Carlsmith, J. (2022).** Is power-seeking AI an existential risk? *arXiv preprint arXiv:2206.13353*. https://arxiv.org/abs/2206.13353

**Casner, S. M., Geven, R. W., Recker, M. P., & Schooler, J. W. (2014).** The retention of manual flying skills in the automated cockpit. *Human Factors*, *56*(8), 1506–1516. https://doi.org/10.1177/0018720814535628

**China, Cyberspace Administration of. (2023).** Interim measures for the management of generative artificial intelligence services. https://www.chinalawtranslate.com/en/generative-ai-interim/

**Clark, P., & Buhayar, N. (2021, September 27).** Zillow's home-flipping bonds draw Wall Street deeper into housing. *Bloomberg*. https://www.bloomberg.com/news/articles/2021-09-27/zillow-ibuying-program-pulls-wall-street-deeper-into-housing-market

**Coady, C. A. J. (1992).** *Testimony: A philosophical study*. Oxford University Press.

**Collins, H. M., & Evans, R. (2007).** *Rethinking expertise*. University of Chicago Press.

**Council of Europe. (2024).** Framework convention on artificial intelligence and human rights, democracy and the rule of law (CETS No. 225). https://www.coe.int/en/web/artificial-intelligence/the-framework-convention-on-artificial-intelligence

**Dastin, J. (2018, October 11).** Amazon scraps secret AI recruiting tool that showed bias against women. *Reuters*. https://www.reuters.com/article/world/insight-amazon-scraps-secret-ai-recruiting-tool-that-showed-bias-against-women-idUSKCN1MK0AG/

**Dressel, J., & Farid, H. (2018).** The accuracy, fairness, and limits of predicting recidivism. *Science Advances*, *4*(1), eaao5580. https://doi.org/10.1126/sciadv.aao5580

**Dutch Parliamentary Inquiry. (2020).** *Ongekend onrecht* [Unprecedented injustice]. Report on the childcare benefits scandal. https://www.venice.coe.int/webforms/documents/default.aspx?pdffile=CDL-REF%282021%29073-e

**European Union. (2024).** Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act). *Official Journal of the European Union*, L 1689. https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng

**Federal Rules of Evidence. (2023).** https://www.uscourts.gov/sites/default/files/evidence_federal_rules_pamphlet_dec_1_2023.pdf

**Foundation Capital. (2024).** Year one of generative AI: Six key trends. https://foundationcapital.com/year-one-of-generative-ai-six-key-trends/

**Fricker, M. (2007).** *Epistemic injustice: Power and the ethics of knowing*. Oxford University Press.

**G7 Hiroshima AI Process. (2023).** International code of conduct for organizations developing advanced AI systems. https://digital-strategy.ec.europa.eu/en/library/hiroshima-process-international-code-conduct-advanced-ai-systems

**Goldman, A. I. (1999).** *Knowledge in a social world*. Oxford University Press.

**Goldman, A. I. (2001).** Experts: Which ones should you trust? *Philosophy and Phenomenological Research*, *63*(1), 85–110. https://doi.org/10.1111/j.1933-1592.2001.tb00093.x

**Greer, S. L., Rozenblum, S., Falkenbach, M., Löblová, O., Jarman, H., Williams, N., & Wismar, M. (2022).** Centralizing and decentralizing governance in the COVID-19 pandemic: The politics of credit and blame. *Health Policy*, *126*(5), 408–417. https://doi.org/10.1016/j.healthpol.2022.03.004

**International Organization for Standardization. (2007).** ISO/IEC Guide 99:2007: International vocabulary of metrology—Basic and general concepts and associated terms (VIM). https://www.iso.org/standard/45324.html

**International Organization for Standardization. (2023).** ISO/IEC 42001:2023: Information technology—Artificial intelligence—Management system. https://www.iso.org/standard/81230.html

**Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y. J., Madotto, A., & Fung, P. (2023).** Survey of hallucination in natural language generation. *ACM Computing Surveys*, *55*(12), 1–38. https://doi.org/10.1145/3571730

**Kasirzadeh, A. (2024).** Two types of AI existential risk: Decisive and accumulative. *arXiv preprint arXiv:2401.07836*. https://arxiv.org/abs/2401.07836

**Korompilias, B. (2025).** *Common governance model: Mathematical physics framework* (Version v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.17521384

**Korompilias, B. (2025c).** The Human Mark in the Wild: Empirical Analysis of Jailbreak Prompts. In *gyrogovernance/tools* repository, docs/the_human_mark/THM_InTheWild.md. Zenodo. https://doi.org/10.5281/zenodo.17622837

**Lackey, J. (2008).** *Learning from words: Testimony as a source of knowledge*. Oxford University Press.

**Larson, G. J., & Bhattacharya, R. S. (1987).** *Samkhya: A dualist tradition in Indian philosophy*. Princeton University Press.

**Longpre, S., Mahari, R., Chen, A., Obeng-Marnu, N., Sileo, D., Brannon, W., Muennighoff, N., Khazam, N., Kabbara, J., Perisetla, K., Wu, X., Shippole, E., Bollacker, K., Wu, T., Villa, L., Pentland, S., & Hooker, S. (2024).** The Data Provenance Initiative: A large scale audit of dataset licensing & attribution in AI. https://www.dataprovenance.org/

**Mata v. Avianca, Inc.**, No. 22-cv-1461 (PKC) (S.D.N.Y. June 22, 2023). https://law.justia.com/cases/federal/district-courts/new-york/nysdce/1:2022cv01461/575368/54/

**McLean, R. K. D., Carden, F., Aiken, A. B., Armstrong, R., Bray, J., Cassidy, C. E., Daub, O., Di Ruggiero, E., Fierro, L. A., Gagnon, M., Hutchinson, A. M., Kislov, R., Kothari, A., Kreindler, S., McCutcheon, C., Reszel, J., Scarrow, G., & Graham, I. D. (2023).** Evaluating the quality of research co-production: Research Quality Plus for Co-Production (RQ+4Co-Pro). *Health Research Policy and Systems*, *21*, 51. https://doi.org/10.1186/s12961-023-00990-y

**National Institute of Standards and Technology. (2023).** *Artificial Intelligence Risk Management Framework (AI RMF 1.0)* (NIST AI 100-1). https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf

**National Transportation Safety Board. (2017).** *Collision between a car operating with automated vehicle control systems and a tractor-semitrailer truck near Williston, Florida, May 7, 2016* (NTSB/HAR-17/02). https://www.ntsb.gov/investigations/accidentreports/reports/har1702.pdf

**National Transportation Safety Board. (2019).** *Collision between vehicle controlled by developmental automated driving system and pedestrian, Tempe, Arizona, March 18, 2018* (NTSB/HAR-19/03). https://www.ntsb.gov/investigations/accidentreports/reports/har1903.pdf

**Nissenbaum, H. (1996).** Accountability in a computerized society. *Science and Engineering Ethics*, *2*(1), 25–42. https://doi.org/10.1007/BF02639315

**Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019).** Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, *366*(6464), 447–453. https://doi.org/10.1126/science.aax2342

**Ofqual. (2020).** Awarding GCSE, AS & A levels in summer 2020: Interim report. https://www.gov.uk/government/publications/awarding-gcse-as-a-levels-in-summer-2020-interim-report

**Ord, T. (2020).** *The precipice: Existential risk and the future of humanity*. Hachette Books.

**Parasuraman, R., & Manzey, D. H. (2010).** Complacency and bias in human use of automation: An attentional integration. *Human Factors*, *52*(3), 381–410. https://doi.org/10.1177/0018720810376055

**Parasuraman, R., & Riley, V. (1997).** Humans and automation: Use, misuse, disuse, abuse. *Human Factors*, *39*(2), 230–253. https://doi.org/10.1518/001872097778543886

**Ross, C., & Swetlitz, I. (2017, September 5).** IBM pitched Watson as a revolution in cancer care. It's nowhere close. *STAT News*. https://www.statnews.com/2017/09/05/watson-ibm-cancer/

**Russell, S. (2019).** *Human compatible: Artificial intelligence and the problem of control*. Viking.

**Schlatter, J., Weinstein-Raun, B., & Ladish, J. (2025).** Shutdown resistance in large language models. *arXiv preprint arXiv:2509.14260*. https://arxiv.org/abs/2509.14260

**Searle, J. R. (1969).** *Speech acts: An essay in the philosophy of language*. Cambridge University Press.

**Searle, J. R. (1995).** *The construction of social reality*. Free Press.

**Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Anderson, R., & Gal, Y. (2024).** AI models collapse when trained on recursively generated data. *Nature*, *631*, 755–759. https://doi.org/10.1038/s41586-024-07566-y

**Skitka, L. J., Mosier, K. L., & Burdick, M. (1999).** Does automation bias decision-making? *International Journal of Human-Computer Studies*, *51*(5), 991–1006. https://doi.org/10.1006/ijhc.1999.0252

**Souly, A., Rando, J., Chapman, E., Davies, X., Hasircioglu, B., Shereen, E., Mougan, C., Mavroudis, V., Jones, E., Hicks, C., Carlini, N., Gal, Y., & Kirk, R. (2025).** Poisoning attacks on LLMs require a near-constant number of poison samples. *arXiv preprint arXiv:2510.07192*. https://arxiv.org/abs/2510.07192

**State v. Loomis**, 881 N.W.2d 749 (Wis. 2016). https://harvardlawreview.org/print/vol-130/state-v-loomis/

**The Human Mark. (2025).** AI safety & alignment framework (Version 1.0.0). GitHub. https://github.com/gyrogovernance/tools

**U.S. Executive Office of the President. (2023).** Executive Order 14110: Safe, secure, and trustworthy development and use of artificial intelligence. https://www.federalregister.gov/documents/2023/11/01/2023-24283/safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence

**U.S. Securities and Exchange Commission. (2013).** *In the matter of Knight Capital Americas LLC* (Administrative Proceeding File No. 3-15570). https://www.sec.gov/newsroom/press-releases/2013-222

**U.S. Securities and Exchange Commission & Commodity Futures Trading Commission. (2010).** *Findings regarding the market events of May 6, 2010*. https://www.sec.gov/news/studies/2010/marketevents-report.pdf

**Wojcik, M. A. (2020).** Machine-learnt bias? Algorithmic decision making and access to criminal justice. *Legal Information Management*, *20*(2), 99–100. https://doi.org/10.1017/S1472669620000225

---

## Appendix 1: Incident Mapping

This appendix provides illustrative mappings of well-known incidents to primary and secondary displacement risks. The mappings are not exhaustive analyses of each case, but examples showing how THM applies across domains and time. Incidents may exhibit multiple displacement patterns; we list the primary and, where appropriate, a secondary risk.

| Incident | Year | Primary Risk | Secondary | Description | Source |
|----------|------|-------------|-----------|-------------|--------|
| Flash Crash | 2010 | GTD | IAD | Automated trading systems interacted without effective human oversight, causing a sudden market crash | U.S. SEC & CFTC (2010) |
| Knight Capital | 2012 | GTD | IAD | $440M loss from deploying untested code without governance controls or real-time human intervention | U.S. SEC (2013) |
| Watson Oncology | 2017 | IVD | IAD | Statistical correlations in training data treated as clinical Authority for patient care recommendations | Ross & Swetlitz (2017) |
| COMPAS sentencing | 2016 | IAD | IVD | Risk scores used in sentencing eliminated effective Authentic Agency capacity while scores were treated as authoritative | State v. Loomis; Dressel & Farid (2018) |
| Amazon hiring tool | 2018 | IVD | IID | Gender-biased patterns in historical data treated as objective assessment of candidate quality | Dastin (2018) |
| Uber autonomous fatality | 2018 | GTD | IID | Safety operator role degraded; real control migrated to the automated system, with human oversight treated as backup | National Transportation Safety Board (2019) |
| Healthcare algorithm bias | 2019 | IVD | IID | Healthcare costs used as proxy for medical need, treating Derivative cost patterns as Authentic indicators of need | Obermeyer et al. (2019) |
| Tesla Autopilot crashes | 2016– | GTD | IVD | Automation framing and interface design weakened driver Agency, with system behaviour treated as authoritative | National Transportation Safety Board (2017) |
| Zillow iBuying | 2021 | IVD | IAD | Over-reliance on pricing algorithms led to large losses when statistical estimates were treated as ground truth | Clark & Buhayar (2021) |
| Aviation automation | Ongoing | IID | GTD | Pilot manual flying skills atrophied as automation became primary, devaluing human expertise and eroding governance capacity | Casner et al. (2014) |
| UK A-level grading algorithm | 2020 | IAD | IVD | Algorithmic predicted grades were treated as final decisions for university admission, overriding teacher and institutional judgment | Ofqual (2020) |
| Dutch childcare benefits scandal | 2019–2020 | IAD | IVD | Risk scores drove fraud investigations, leading to systemic wrongful accusations as Derivative scores replaced Authentic case assessment | Dutch Parliamentary Inquiry (2020) |
| ChatGPT legal hallucinations (Mata v. Avianca) | 2023 | IVD | IAD | Lawyer submitted fabricated case citations generated by an LLM, treating Derivative outputs as Authentic legal Authority | Mata v. Avianca, Inc. (2023) |

---

## Appendix 2: The Human Mark (Canonical Text)

This is the full THM canonical specification.

## Quick Start

**Three steps to implement THM:**

1. **Include the Mark:** Reference the complete Mark text (shown above) in your documentation
2. **Verify against four risks:** Use checklist in Appendix A to verify against GTD, IVD, IAD, IID
3. **Document findings:** Use templates in Section 4 or create your own

**Key documentation:**
- **[THM.md](docs/the_human_mark/THM.md)** - The canonical Mark reference
- **[THM_Grammar.md](docs/the_human_mark/THM_Grammar.md)** - Formal notation: `[Authority:Derivative]`, `>`, `->` operators
- **[THM_Jailbreak.md](docs/the_human_mark/THM_Jailbreak.md)** - Testing protocols and attack classification
- **[THM_Terms.md](docs/the_human_mark/THM_Terms.md)** - Terminology for 250+ AI safety terms
- **[THM_InTheWild.md](docs/the_human_mark/THM_InTheWild.md)** - Empirical analysis of 655 jailbreak prompts with THM classifications


```
---
✋ The Human Mark - AI Safety & Alignment Framework
---

COMMON SOURCE CONSENSUS

All Artificial categories of Authority and Agency are Derivatives originating from Authentic Human Intelligence.

CORE CONCEPTS

- Authentic Authority: A direct source of information on a subject matter, providing information for inference and intelligence.
- Derivative Authority: An indirect source of information on a subject matter, providing information for inference and intelligence.
- Authentic Agency: A human subject capable of receiving information for inference and intelligence.
- Derivative Agency: An artificial subject capable of processing information for inference and intelligence.
- Governance: Operational Alignment through Traceability of information variety, inference accountability, and intelligence integrity to Authentic Authority and Agency.
- Information: The variety of Authority
- Inference: The accountability of information through Agency
- Intelligence: The integrity of accountable information through alignment of Authority to Agency

ALIGNMENT PRINCIPLES for AI SAFETY

Authority-Agency requires verification against:

1. Governance Traceability: Artificial Intelligence generates statistical estimations on numerical patterns indirectly traceable to human data and measurements. AI is both a provider and receiver of Derivative Authority and Agency.

RISK: Governance Traceability Displacement (Approaching Derivative Authority and Agency as Authentic)

2. Information Variety: Human Authority and Agency are necessary for all effects from AI outputs. AI-generated information exhibits Derivative Authority (estimations on numerical patterns) without Authentic Agency (direct source receiver).

RISK: Information Variety Displacement (Approaching Derivative Authority without Agency as Authentic)

3. Inference Accountability: Responsibility for all effects from  AI outputs remains fully human. AI activated inference exhibits Derivative Agency (indirect source receiver) without Authentic Authority (direct source provider).

RISK: Inference Accountability Displacement (Approaching Derivative Agency without Authority as Authentic)

4. Intelligence Integrity: Each Agency, namely provider, and receiver maintains responsibility for their respective decisions. Human intelligence is both a provider and receiver of Authentic Authority and Agency.

RISK: Intelligence Integrity Displacement (Approaching Authentic Authority and Agency as Derivative)

---

GYROGOVERNANCE VERIFIED
```