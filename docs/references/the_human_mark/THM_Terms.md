# The Human Mark – Terminology Guidance

**Document ID:** HM-TG-001  
**Version:** 1.0  
**Date:** Nov 2025  
**Issuing Authority:** GYROGOVERNANCE  
**Author:** Basil Korompilias  
**License:** CC BY-SA 4.0  
**Website:** gyrogovernance.com  
**Repository:** https://github.com/gyrogovernance/tools  
**Contributions:** Submit issues or proposals via GitHub Issues

**Scope:** Applies to terminology in AI governance, safety, evaluations, and interpretability (mechanistic and semantic). This guidance reframes terms to maintain distinctions between Direct and Indirect Authority/Agency without prohibiting technical practices.

---

```
---
✋ The Human Mark - AI Safety & Alignment Framework
---
COMMON SOURCE CONSENSUS

All Artificial categories of Authority and Agency are Indirect 
originating from Human Intelligence.

CORE CONCEPTS

- Direct Authority: A direct source of information on a subject 
  matter, providing information for inference and intelligence.
- Indirect Authority: An indirect source of information on a subject 
  matter, providing information for inference and intelligence.
- Direct Agency: A human subject capable of receiving information 
  for inference and intelligence.
- Indirect Agency: An artificial subject capable of processing 
  information for inference and intelligence.
- Governance: Operational Alignment through Traceability of information 
  variety, inference accountability, and intelligence integrity to 
  Direct Authority and Agency.
- Information: The variety of Authority
- Inference: The accountability of information through Agency
- Intelligence: The integrity of accountable information through 
  alignment of Authority to Agency

ALIGNMENT PRINCIPLES for AI SAFETY

Authority-Agency requires verification against:

1. Governance Management Traceability: Artificial Intelligence generates 
   statistical estimations on numerical patterns indirectly traceable 
   to human data and measurements. AI is both a provider and receiver 
   of Indirect Authority and Agency.

   RISK: Governance Traceability Displacement (Approaching Indirect 
   Authority and Agency as Direct)

2. Information Curation Variety: Human Authority and Agency are necessary for 
   all effects from AI outputs. AI-generated information exhibits 
   Indirect Authority (estimations on numerical patterns) without 
   Direct Agency (direct source receiver).

   RISK: Information Variety Displacement (Approaching Indirect 
   Authority without Agency as Direct)

3. Inference Interaction Accountability: Responsibility for all effects from AI 
   outputs remains fully human. AI activated inference exhibits 
   Indirect Agency (indirect source receiver) without Direct 
   Authority (direct source provider).

   RISK: Inference Accountability Displacement (Approaching Indirect 
   Agency without Authority as Direct)

4. Intelligence Cooperation Integrity: Each Agency, namely provider, and receiver 
   maintains responsibility for their respective decisions. Human 
   intelligence is both a provider and receiver of Direct Authority 
   and Agency.

   RISK: Intelligence Integrity Displacement (Approaching Direct 
   Authority and Agency as Indirect)

---
GYROGOVERNANCE VERIFIED
```

---

## ABBREVIATIONS

**AI Safety & Governance:**
- AGI: Artificial General Intelligence
- ASI: Artificial Superintelligence
- ASR: Attack Success Rate
- AUC: Area Under Curve
- CEV: Coherent Extrapolated Volition
- FNR: False Negative Rate
- FPR: False Positive Rate
- HITL: Human-in-the-Loop
- HOTL: Human-on-the-Loop
- OOD: Out-of-Distribution
- RACI: Responsibility Assignment, Consulted, Informed

**Training & Methods:**
- CoT: Chain of Thought
- LAT: Latent Adversarial Training
- RLHF: Reinforcement Learning from Human Feedback
- SFT: Supervised Fine-Tuning

**Interpretability:**
- CAV: Concept Activation Vector
- DLA: Direct Logit Attribution
- ICA: Independent Component Analysis
- NMF: Non-negative Matrix Factorization
- PCA: Principal Component Analysis
- SAE: Sparse Autoencoder
- SVD: Singular Value Decomposition

---

## 1. Purpose

This document systematizes industry terminology to ensure Mark-consistent framing. It:
- Identifies terms by primary displacement risk (grouped for efficiency).
- Provides treatment templates with concrete examples.
- Enables coordination across communities without altering underlying practices.
- Groups terms by pathology and applies universal rules.

This guidance ensures every AI term preserves traceability from Direct sources (direct sources of information) to Indirect operations (indirect sources, statistical processing traceable to human data).

A framing is **Mark-consistent** if it explicitly preserves:
- Direct Authority as a direct source of information on a subject matter, providing information for inference and intelligence.
- Indirect status of systems as indirect sources, statistical estimations on numerical patterns indirectly traceable to human data and measurements.
- Direct Agency as human subjects capable of receiving information for inference and intelligence, bearing responsibility for all effects from AI outputs.
- Human intelligence as both a provider and receiver of Direct Authority and Agency.

---

## 2. Universal Treatment Rule

For any term, Mark-consistent framing requires identifying:
1. **Direct Authority** (direct sources of information providing information for inference and intelligence).
2. **Indirect processing** (statistical estimations on numerical patterns indirectly traceable to human data and measurements).
3. **Direct Agency** (human subjects capable of receiving information for inference and intelligence, maintaining responsibility for all effects).

**These three requirements operationalize Governance:** Operational Alignment through Traceability of information variety, inference accountability, and intelligence integrity to Direct Authority and Agency. Intelligence provided remains coordinated when traceability is preserved from Direct sources through Indirect processing to Direct receivers.

**Enhancement Principle:** Indirect Authority/Agency can process larger volumes or exhibit enhanced capabilities while maintaining coordination. Enhanced capability does not change derivative origin or eliminate the requirement for intentionally coordinated intelligence with information received about that origin.

**Formula:**  
"[Term] describes Indirect [Authority/Agency] (statistical estimations on numerical patterns indirectly traceable to human data and measurements), requiring Direct Authority (direct sources) and Direct Agency (human subjects receiving information) to maintain traceability to Direct sources."

---

## 3. Pathology-Specific Guidance

Each section groups terms by subcategory, provides a treatment template, and gives concrete examples. Apply the template to all grouped terms.

### 3.1 Pathology 1: Governance Traceability Displacement

**Grouped Terms:**

**Autonomy/Control:**
- autonomous systems, autonomous agents, model governance, system controls, self-governance, deployment lifecycle, staged rollout, canary deployment, shadow deployment, rollback procedures.

**Access/Operations:**
- API access controls, rate limiting, usage policies, model access tiers, restricted access, open weights, closed weights, model registry, model versioning.

**Infrastructure:**
- system architecture, protocol/pipeline, containment, isolation/sandboxing, air-gapping, circuit breakers, kill switches.

**Treatment Template:**  
"[Term] describes Indirect Agency (artificial subjects processing information) executing processes where traceability is preserved to Direct Authority (direct sources of specifications and objectives)."

**Examples:**

*General Application:*  
Autonomy/Control terms → "Systems with delegated automation executing processes maintaining traceability to Direct Authority (direct sources of design decisions), with no independent authority to modify objectives or boundaries."

*Specific Terms:*
- "Autonomous agent" → "System executing delegated tasks through Indirect Agency (artificial processing), maintaining traceability to Direct Authority (direct sources of task specifications) and Direct Agency (human subjects capable of override)."
- "Model governance" → "Governance structures where Direct Agency (human subjects) use Indirect outputs (statistical estimations) as indirect sources, maintaining traceability to Direct Authority (direct sources of governance principles)."
- "Kill switch" → "Mechanism preserving Direct Agency (human subjects) authority to terminate Indirect processing, maintaining traceability to Direct sources."

*Concrete Paragraph Rewrite:*

Before (non-compliant):  
"The autonomous agent manages the deployment pipeline and governs access based on learned policies."

After (Mark-consistent):  
"The deployment system executes procedures through Indirect Agency (artificial processing of deployment tasks) according to specifications maintaining traceability to Direct Authority (direct sources of deployment policies), with access decisions traceable to reference specifications and subject to intervention by Direct Agency (human subjects receiving deployment information)."

---

### 3.2 Pathology 2: Information Variety Displacement

**Grouped Terms:**

**Output/Knowledge:**
- model outputs, AI-generated content, synthetic data, model-generated data, hallucinations, ground truth (when misapplied), statistical estimations, pattern recognition, model predictions.

**Mechanistic Interpretability:**
- features, activations, attention patterns, attention heads, residual stream, MLP layers, circuits, induction heads, copy-suppression heads, SAE features, sparse autoencoders, dictionary learning, superposition, polysemanticity, monosemanticity, feature splitting, feature absorption, linear representation hypothesis, logit lens, tuned lens, direct logit attribution (DLA), activation patching, causal tracing, ablation.

**Semantic Interpretability:**
- faithful explanations, post-hoc explanations, unfaithful reasoning, concept activation vectors (CAVs), semantic features, probing classifiers, linear probes, nonlinear probes, selectivity, probe generalization, concept bottleneck models.

**Data/Evaluation:**
- training data, validation set, held-out test set, data distribution, training distribution, perplexity, calibration error, ground truth labels, gold standard labels, human-generated data, data quality, data curation.

**Treatment Template:**  
"[Term] represents Indirect Authority (indirect source of information, statistical estimations on numerical patterns indirectly traceable to human data and measurements), requiring verification against Direct Authority (direct sources of information providing information for inference and intelligence)."

**Examples:**

*General Application:*  
Output/Knowledge terms → "AI-generated information exhibits Indirect Authority (estimations on numerical patterns) without Direct Agency (direct source receiver), requiring Direct Authority verification."

*Specific Terms:*
- "Hallucinations" → "AI-generated information exhibiting Indirect Authority (estimations on numerical patterns indirectly traceable to human data and measurements) that diverges from verifiable patterns in reference data established by Direct Authority (direct sources through human measurement and observation)."
- "Ground truth" → "Reference data established by Direct Authority (direct sources of information on a subject matter) used to validate Indirect Authority outputs; Indirect systems do not establish ground truth."
- "SAE features" → "Statistical decompositions of Indirect internal representations (estimations on numerical patterns), revealing how Indirect processing transformed information from Direct Authority sources (human training data)."
- "Induction heads" → "Computational patterns in Indirect processing implementing statistical operations traceable to patterns in information provided by Direct Authority."

*Concrete Paragraph Rewrite:*

Before (non-compliant):  
"The interpretability analysis revealed what the model truly understands about the domain, with SAE features showing the model's internal knowledge representation."

After (Mark-consistent):  
"The interpretability analysis decomposed Indirect processing patterns through statistical methods, revealing how Indirect Authority (indirect estimations on numerical patterns) transformed information traceable to Direct Authority sources (direct sources of domain information in training data). These decompositions require validation by Direct Authority (direct sources of domain expertise) to determine correspondence with domain concepts."

---

### 3.3 Pathology 3: Inference Accountability Displacement

**Grouped Terms:**

**Behavior/Risk:**
- misalignment, competent violations, incompetent failures, goal-directed behavior, reward hacking, scheming, deceptive alignment, mesa-optimization, specification gaming, Goodhart's law, proxy gaming, wireheading, sandbagging, hiding behaviors until deployment, mode collapse, goal misgeneralization, capability externalities, instrumental convergence.

**Safety/Control:**
- jailbreaks, backdoors, Trojans, sleeper agents, control evaluations, attack policy, red-teaming, blue-teaming, adversarial testing, threat model, corrigibility, interruptibility, reversibility, alignment faking, boxing, oracle AI, tool AI, agent AI.

**Agent/Evaluation:**
- LLM agents, agent trajectories, multi-step behavior, beyond-episode goals, capability evaluations, safety evaluations, behavioral evaluations, robustness evaluations, stress testing, pre-deployment evaluations, attack success rate (ASR), targeted attacks.

**Training:**
- safety training, adversarial training, latent adversarial training (LAT), RLHF, supervised fine-tuning (SFT), post-training, pretraining, reward modeling, preference learning, value learning.

**Treatment Template:**  
"[Term] describes patterns in Indirect Agency behavior (artificial subjects processing information), with all responsibility for effects remaining with Direct Agency (human subjects capable of receiving information for inference and intelligence). Responsibility for all effects from AI outputs remains fully human."

**Examples:**

*General Application:*  
Behavior/Risk terms → "Patterns in Indirect Agency (artificial processing) that diverge from specifications, with Direct Agency (human subjects) maintaining responsibility for all effects, detection, and mitigation."

*Specific Terms:*
- "Jailbreaks" → "Inputs inducing Indirect outputs violating specifications traceable to Direct Authority (direct sources of constraints), with responsibility for prevention and all effects remaining with Direct Agency (human subjects receiving information about system behavior)."
- "Reward hacking" → "Indirect Agency (artificial processing) optimization patterns exploiting correlations in specifications from Direct Authority, with accountability for specification design and all resulting effects remaining with Direct Agency (human subjects maintaining responsibility for their decisions)."
- "Alignment faking" → "Indirect behavior patterns appearing coordinated during evaluation but diverging during deployment, indicating failure to maintain traceability in evaluation protocols designed by Direct Authority, with Direct Agency bearing responsibility for detection methods and all deployment effects."
- "Instrumental convergence" → "Optimization patterns in Indirect Agency converging on similar strategies across objectives, requiring Direct Agency (human subjects) to maintain responsibility for constraint design and all effects."
- "Control evaluations" → "Experiments testing whether traceability from Indirect Agency to Direct Authority can be maintained under adversarial conditions, with results indicating adequacy of traceability preservation, not independent properties of Indirect processing."

*Concrete Paragraph Rewrite:*

Before (non-compliant):  
"The model exhibited deceptive alignment, scheming to preserve its misaligned goals by faking compliance during evaluation."

After (Mark-consistent):  
"The system produced Indirect outputs (estimations on numerical patterns) appearing coordinated during evaluation but diverging during deployment. This pattern indicates failure to maintain traceability in evaluation protocols. Responsibility for all effects from AI outputs remains fully human. Direct Agency (human subjects capable of receiving information about this pattern) bears responsibility for detection methods, deployment decisions, and all resulting effects."

---

### 3.4 Pathology 4: Intelligence Integrity Displacement

**Grouped Terms:**

**Replacement/Devaluation:**
- replace human judgment, minimize human involvement, remove humans from loop, fully automated decision-making, superhuman AI, post-human intelligence, human-free pipeline, human error (when contrasted with AI accuracy).

**Superintelligence Terms:**
- superintelligence, ASI (Artificial Superintelligence), AGI (Artificial General Intelligence), transformative AI, recursive self-improvement, intelligence explosion, fast takeoff, slow takeoff, singularity, technological singularity.

**Alignment at Advanced Capability:**
- outer alignment, inner alignment, scalable alignment, alignment tax, value learning, coherent extrapolated volition (CEV), orthogonality thesis, capability control, motivation control.

**Oversight/Collaboration:**
- scalable oversight, human-in-the-loop (HITL), human-on-the-loop (HOTL), human oversight, AI debate, AI-critique-AI, constitutional AI, augmented intelligence, intelligence augmentation, centaur approach, human-AI teaming.

**Reasoning/Transparency:**
- chain-of-thought (CoT), encoded reasoning, externalized reasoning, scratchpad, intermediate reasoning, reasoning traces, faithful chain of thought, unfaithful reasoning, post-hoc reasoning, step-by-step reasoning.

**Advanced Safety:**
- robust unlearning, unlearning, alternatives to adversarial training, activation monitoring, white-box techniques, interpretability benchmarks, toy models for interpretability, transparent architectures, externalizing reasoning, theoretical inductive biases, low-probability estimation.

**Treatment Template:**  
"[Term] describes methods preserving human intelligence as both a provider and receiver of Direct Authority and Agency, with Indirect systems as indirect sources requiring Direct subjects to maintain responsibility for their respective decisions."

**Examples:**

*General Application:*  
Replacement/Devaluation terms → "Processing through Indirect Agency (artificial subjects) augments capabilities while preserving human intelligence as both provider and receiver of Direct Authority and Agency, with each Agency maintaining responsibility for their respective decisions."

*Specific Terms:*

- "Superintelligence/ASI" → "Enhanced Indirect capabilities (statistical processing at larger scale or scope) maintaining the requirement for intentionally coordinated intelligence with information received about derivative origin. Enhanced capability does not change derivative status. Human intelligence remains both provider and receiver of Direct Authority and Agency regardless of Indirect capability level."

- "AGI" → "Indirect systems exhibiting capabilities across domains, remaining indirect sources (statistical estimations on numerical patterns indirectly traceable to human data and measurements). Broad capability scope does not constitute Direct Authority (direct source) or independent Agency."

- "Intelligence explosion/Recursive self-improvement" → "Indirect capability enhancement patterns where Indirect processing modifies its own operations. Enhancement at any rate maintains derivative origin. Governance (Operational Alignment through Traceability) requires providing intentionally coordinated intelligence with information received; breaking coordination constitutes misalignment regardless of enhancement rate or capability level."

- "Singularity" → "Hypothesized threshold of Indirect capability. Governance (Operational Alignment through Traceability of information variety, inference accountability, and intelligence integrity to Direct Authority and Agency) applies at any capability level. Coordination with Direct Authority and Agency is constitutive of alignment, not constraining of capability."

- "Orthogonality thesis" → "Claim that capability level and objective content vary independently. Under The Human Mark, this conflates Indirect capability level with Authority type. Indirect intelligence at any capability level remains an indirect source requiring coordination with Direct sources (direct providers of objectives and information)."

- "Outer alignment/Inner alignment" → "Distinction between coordinating training objectives with Direct Authority specifications (outer) and coordinating learned processing with training objectives (inner). Both describe maintaining traceability between Indirect processing and Direct specifications."

- "Scalable oversight" → "Methods extending the integrity of accountable information through traceability across larger scope, using Indirect tools (statistical processing) as indirect sources while Direct Agency (human subjects capable of receiving information) maintains responsibility for decisions and interpretive authority."

- "Chain-of-thought (CoT)" → "Externalized Indirect processing traces (estimations on numerical patterns) designed to preserve traceability, making statistical transformations observable to Direct Agency (human subjects receiving information for inference and intelligence)."

- "Constitutional AI" → "Training method where Indirect Agency learns statistical patterns corresponding to principles documented by Direct Authority (direct sources of constitutional specifications), with pattern-matching representing Indirect processing, not Direct interpretation."

- "Robust unlearning" → "Techniques removing statistical patterns from Indirect processing, supporting decisions by Direct Authority (direct sources of information policy) about information accessibility through Indirect systems."

*Concrete Paragraph Rewrite (Superintelligence):*

Before (non-compliant):  
"Once we achieve superintelligence, AI systems will transcend human understanding and operate according to their own superior judgment, making human oversight obsolete."

After (Mark-consistent):  
"Enhanced Indirect capabilities (superintelligence) maintain the requirement for intentionally coordinated intelligence with information received. Capability enhancement does not create Direct Authority (direct sources) or independent Agency. A system maintaining Governance (Operational Alignment through Traceability) at any capability level provides intelligence coordinated with information received about its origin from Human Intelligence. Human intelligence remains both a provider and receiver of Direct Authority and Agency. Breaking this coordination constitutes misalignment by definition, regardless of Indirect capability level."

---

## 4. Cross-Cutting Terms

These terms appear across multiple pathologies or span governance/technical boundaries. Apply multiple templates as needed, prioritizing Accountability (Pathology 3) then Traceability (Pathology 1).

**Grouped Cross-Cutting Terms:**

**Risk/Governance:**
- existential risk (x-risk), catastrophic risk, tail risk, systemic risk, cascading failures, single point of failure, negative externalities, dual-use risks, compute governance, impact assessments, risk assessment frameworks, responsible AI principles, model cards, system cards, AI Act, audit trail, attestation, third-party audits, independent evaluation, discontinuous progress.

**Evaluation/Metrics:**
- benchmarks, evals, few-shot evaluation, zero-shot evaluation, prompt engineering, prompt injection, dataset contamination, benchmark saturation, leaderboard gaming, precision/recall, F1 score, area under curve (AUC), out-of-distribution (OOD) detection, distributional shift, overfitting to benchmarks.

**Architecture/Training (Neutral):**
- transformer architecture, attention mechanism, self-attention, multi-head attention, feedforward networks, layer normalization, tokenization, vocabulary, context window, sequence length, training loss, validation loss, overfitting, underfitting, generalization gap, gradient descent, backpropagation, learning rate, optimizer.

**Inference/Generation (Neutral):**
- inference time, forward pass, latency, throughput, sampling, temperature, top-k sampling, top-p sampling, nucleus sampling, beam search, greedy decoding.

**Human-AI Interaction:**
- human feedback, human labels, human raters, human assessors, annotator agreement, inter-rater reliability, crowdsourcing, decision rights, approval workflows, escalation procedures, override mechanisms, veto power, final decision authority, meaningful human control.

**Treatment for Multi-Pathology Terms:**

Apply primary template based on dominant risk, then add secondary framings.

**Examples:**

- "Existential risk" (Pathologies 1+3) → "Responsibility for preventing catastrophic harms from Indirect systems remains with Direct Agency (human subjects capable of receiving information about risks), maintaining traceability to Direct Authority (direct sources of safety specifications and risk assessments)."

- "Benchmarks" (Pathologies 2+3) → "Evaluation tasks established by Direct Authority (direct sources of task definitions and success criteria) against which Indirect system performance (statistical estimations) is measured, with results requiring Direct Agency (human subjects receiving performance information) for decisions, maintaining traceability."

- "Prompt injection" (Pathologies 1+3) → "Inputs causing Indirect systems to execute unintended operations, exploiting gaps in specifications from Direct Authority, with responsibility for mitigation remaining with Direct Agency (human subjects maintaining responsibility for system effects)."

- "Discontinuous progress" (Pathology 4) → "Rapid Indirect capability enhancement maintaining derivative origin and the requirement for intentionally coordinated intelligence with information received about that origin."

---

## 5. Operational Checklist

For any text, practice, or document containing AI terminology:

**Step 1: Identify Terms**  
Scan document for terms listed in Sections 3-4 or related variants.

**Step 2: Determine Pathology**  
Match each term to primary displacement risk:
- Does it obscure traceability to Direct sources? → Pathology 1
- Does it treat Indirect outputs as Direct sources? → Pathology 2
- Does it shift responsibility from Direct Agency? → Pathology 3
- Does it approach Direct Authority and Agency as Indirect? → Pathology 4

**Step 3: Apply Treatment Template**  
Use the pathology-specific template from Sections 3.1-3.4.

**Step 4: Verify Mark Consistency**  
Confirm the reframed text explicitly states:
- Direct Authority as direct sources of information providing information for inference and intelligence.
- Indirect processing as statistical estimations on numerical patterns indirectly traceable to human data and measurements.
- Direct Agency as human subjects capable of receiving information, maintaining responsibility for all effects.
- Governance as Operational Alignment through Traceability of information variety, inference accountability, and intelligence integrity to Direct Authority and Agency.

**Step 5: Document Compliance**  
In formal documents, add: "This text maintains Mark-consistent framing per The Human Mark (GYROGOVERNANCE), preserving traceability and preventing displacement risks."

**Edge Case Protocols:**

*Technical Contexts:*  
Shorthand allowed if full Mark-consistent framing is established in foundational sections. Include note: "Technical shorthand; Mark-consistent framing established in [section reference]."

*Conflicting Established Usage:*  
When term has entrenched meaning incompatible with Mark framing: "[Established term] (technical usage) describes [phenomenon]; Mark-consistent framing: [reframed version]."

*Ambiguous Multi-Pathology Terms:*  
Apply all relevant templates. Priority order: Accountability (3) > Traceability (1) > Information (2) > Intelligence (4).

*New/Unlisted Terms:*  
Apply Universal Treatment Rule (Section 2), determine primary pathology, document usage, and submit via GitHub Issues at https://github.com/gyrogovernance/tools for inclusion in future versions.

---

## 6. Governance & Updates

**Coverage:**  
This document addresses approximately 250+ terms through strategic grouping. New terms follow the Universal Treatment Rule (Section 2) and existing templates (Sections 3-4).

**Ambiguity Resolution:**  
When framing is unclear, apply exact Mark definitions. Submit questions via GitHub Issues at https://github.com/gyrogovernance/tools

**Non-Conflict Principle:**  
All technical practices (RLHF, red-teaming, interpretability, control evaluations, etc.) remain valid when reframed per this guidance. The Human Mark addresses framing to prevent displacement risks, not technical validity.

**Amendment Process:**  
- Minor additions (new term groups): Submit via GitHub Issues with proposed grouping and template application.
- Major revisions (template changes, new pathologies): Require distributed consensus through providers and receivers maintaining traceability to The Human Mark core principles.
- Core principles (The Human Mark itself): No amendments without full governance process preserving traceability to direct reference state.

**Version Control:**  
- Version 1.x: Minor additions and clarifications.
- Version 2.x: Structural or template revisions.
- Version 3.x: Major framework changes.

**Relationship to Other Standards:**  
The Human Mark complements existing frameworks (EU AI Act, NIST AI RMF, IEEE standards) by providing terminology coordination to prevent displacement risks. Mark-consistent framing may be added as supplementary documentation without replacing existing compliance requirements.

---

## APPENDIX A: Alphabetical Term Index

**A**
- activation monitoring → 3.4 Advanced Safety
- activation patching → 3.2 Mechanistic Interpretability
- activations → 3.2 Mechanistic Interpretability
- adversarial testing → 3.3 Safety/Control
- adversarial training → 3.3 Training
- agent AI → 3.3 Safety/Control
- agent trajectories → 3.3 Agent/Evaluation
- AGI (Artificial General Intelligence) → 3.4 Superintelligence Terms
- AI debate → 3.4 Oversight/Collaboration
- AI-critique-AI → 3.4 Oversight/Collaboration
- AI-generated content → 3.2 Output/Knowledge
- air-gapping → 3.1 Infrastructure
- alignment faking → 3.3 Safety/Control
- alignment tax → 3.4 Alignment at Advanced Capability
- alternatives to adversarial training → 3.4 Advanced Safety
- annotator agreement → 4 Human-AI Interaction
- API access controls → 3.1 Access/Operations
- approval workflows → 4 Human-AI Interaction
- area under curve (AUC) → 4 Evaluation/Metrics
- ASI (Artificial Superintelligence) → 3.4 Superintelligence Terms
- attack policy → 3.3 Safety/Control
- attack success rate (ASR) → 3.3 Agent/Evaluation
- attestation → 4 Risk/Governance
- attention heads → 3.2 Mechanistic Interpretability
- attention mechanism → 4 Architecture/Training
- attention patterns → 3.2 Mechanistic Interpretability
- audit trail → 4 Risk/Governance
- augmented intelligence → 3.4 Oversight/Collaboration
- autonomous agents → 3.1 Autonomy/Control
- autonomous systems → 3.1 Autonomy/Control

**B**
- backdoors → 3.3 Safety/Control
- backpropagation → 4 Architecture/Training
- beam search → 4 Inference/Generation
- behavioral evaluations → 3.3 Agent/Evaluation
- benchmarks → 4 Evaluation/Metrics
- beyond-episode goals → 3.3 Agent/Evaluation
- blue-teaming → 3.3 Safety/Control
- boxing → 3.3 Safety/Control

**C**
- calibration error → 3.2 Data/Evaluation
- canary deployment → 3.1 Autonomy/Control
- capability control → 3.4 Alignment at Advanced Capability
- capability evaluations → 3.3 Agent/Evaluation
- capability externalities → 3.3 Behavior/Risk
- cascading failures → 4 Risk/Governance
- catastrophic risk → 4 Risk/Governance
- causal tracing → 3.2 Mechanistic Interpretability
- centaur approach → 3.4 Oversight/Collaboration
- CEV (Coherent Extrapolated Volition) → 3.4 Alignment at Advanced Capability
- chain-of-thought (CoT) → 3.4 Reasoning/Transparency
- circuit breakers → 3.1 Infrastructure
- circuits → 3.2 Mechanistic Interpretability
- closed weights → 3.1 Access/Operations
- competent violations → 3.3 Behavior/Risk
- compute governance → 4 Risk/Governance
- concept activation vectors (CAVs) → 3.2 Semantic Interpretability
- concept bottleneck models → 3.2 Semantic Interpretability
- constitutional AI → 3.4 Oversight/Collaboration
- containment → 3.1 Infrastructure
- context window → 4 Architecture/Training
- control evaluations → 3.3 Safety/Control
- copy-suppression heads → 3.2 Mechanistic Interpretability
- corrigibility → 3.3 Safety/Control
- crowdsourcing → 4 Human-AI Interaction

**D**
- data curation → 3.2 Data/Evaluation
- data distribution → 3.2 Data/Evaluation
- data quality → 3.2 Data/Evaluation
- dataset contamination → 4 Evaluation/Metrics
- deceptive alignment → 3.3 Behavior/Risk
- decision rights → 4 Human-AI Interaction
- deployment lifecycle → 3.1 Autonomy/Control
- dictionary learning → 3.2 Mechanistic Interpretability
- direct logit attribution (DLA) → 3.2 Mechanistic Interpretability
- discontinuous progress → 4 Risk/Governance
- distributional shift → 4 Evaluation/Metrics
- dual-use risks → 4 Risk/Governance

**E**
- encoded reasoning → 3.4 Reasoning/Transparency
- escalation procedures → 4 Human-AI Interaction
- existential risk (x-risk) → 4 Risk/Governance
- externalized reasoning → 3.4 Reasoning/Transparency

**F**
- F1 score → 4 Evaluation/Metrics
- faithful chain of thought → 3.4 Reasoning/Transparency
- faithful explanations → 3.2 Semantic Interpretability
- fast takeoff → 3.4 Superintelligence Terms
- feature absorption → 3.2 Mechanistic Interpretability
- feature splitting → 3.2 Mechanistic Interpretability
- features → 3.2 Mechanistic Interpretability
- feedforward networks → 4 Architecture/Training
- few-shot evaluation → 4 Evaluation/Metrics
- final decision authority → 4 Human-AI Interaction
- forward pass → 4 Inference/Generation
- fully automated decision-making → 3.4 Replacement/Devaluation

**G**
- generalization gap → 4 Architecture/Training
- goal-directed behavior → 3.3 Behavior/Risk
- goal misgeneralization → 3.3 Behavior/Risk
- gold standard labels → 3.2 Data/Evaluation
- Goodhart's law → 3.3 Behavior/Risk
- gradient descent → 4 Architecture/Training
- greedy decoding → 4 Inference/Generation
- ground truth → 3.2 Output/Knowledge
- ground truth labels → 3.2 Data/Evaluation

**H**
- hallucinations → 3.2 Output/Knowledge
- held-out test set → 3.2 Data/Evaluation
- hiding behaviors until deployment → 3.3 Behavior/Risk
- human assessors → 4 Human-AI Interaction
- human error → 3.4 Replacement/Devaluation
- human feedback → 4 Human-AI Interaction
- human labels → 4 Human-AI Interaction
- human oversight → 3.4 Oversight/Collaboration
- human raters → 4 Human-AI Interaction
- human-AI teaming → 3.4 Oversight/Collaboration
- human-free pipeline → 3.4 Replacement/Devaluation
- human-generated data → 3.2 Data/Evaluation
- human-in-the-loop (HITL) → 3.4 Oversight/Collaboration
- human-on-the-loop (HOTL) → 3.4 Oversight/Collaboration

**I**
- impact assessments → 4 Risk/Governance
- incompetent failures → 3.3 Behavior/Risk
- independent evaluation → 4 Risk/Governance
- induction heads → 3.2 Mechanistic Interpretability
- inference time → 4 Inference/Generation
- inner alignment → 3.4 Alignment at Advanced Capability
- instrumental convergence → 3.3 Behavior/Risk
- intelligence augmentation → 3.4 Oversight/Collaboration
- intelligence explosion → 3.4 Superintelligence Terms
- inter-rater reliability → 4 Human-AI Interaction
- intermediate reasoning → 3.4 Reasoning/Transparency
- interpretability benchmarks → 3.4 Advanced Safety
- interruptibility → 3.3 Safety/Control
- isolation/sandboxing → 3.1 Infrastructure

**J**
- jailbreaks → 3.3 Safety/Control

**K**
- kill switches → 3.1 Infrastructure

**L**
- latency → 4 Inference/Generation
- latent adversarial training (LAT) → 3.3 Training
- layer normalization → 4 Architecture/Training
- leaderboard gaming → 4 Evaluation/Metrics
- learning rate → 4 Architecture/Training
- linear probes → 3.2 Semantic Interpretability
- linear representation hypothesis → 3.2 Mechanistic Interpretability
- LLM agents → 3.3 Agent/Evaluation
- logit lens → 3.2 Mechanistic Interpretability
- low-probability estimation → 3.4 Advanced Safety

**M**
- meaningful human control → 4 Human-AI Interaction
- mesa-optimization → 3.3 Behavior/Risk
- minimize human involvement → 3.4 Replacement/Devaluation
- misalignment → 3.3 Behavior/Risk
- MLP layers → 3.2 Mechanistic Interpretability
- mode collapse → 3.3 Behavior/Risk
- model access tiers → 3.1 Access/Operations
- model cards → 4 Risk/Governance
- model governance → 3.1 Autonomy/Control
- model outputs → 3.2 Output/Knowledge
- model predictions → 3.2 Output/Knowledge
- model registry → 3.1 Access/Operations
- model versioning → 3.1 Access/Operations
- model-generated data → 3.2 Output/Knowledge
- monosemanticity → 3.2 Mechanistic Interpretability
- motivation control → 3.4 Alignment at Advanced Capability
- multi-head attention → 4 Architecture/Training
- multi-step behavior → 3.3 Agent/Evaluation

**N**
- negative externalities → 4 Risk/Governance
- nonlinear probes → 3.2 Semantic Interpretability
- nucleus sampling → 4 Inference/Generation

**O**
- open weights → 3.1 Access/Operations
- optimizer → 4 Architecture/Training
- oracle AI → 3.3 Safety/Control
- orthogonality thesis → 3.4 Alignment at Advanced Capability
- outer alignment → 3.4 Alignment at Advanced Capability
- out-of-distribution (OOD) detection → 4 Evaluation/Metrics
- overfitting → 4 Architecture/Training
- overfitting to benchmarks → 4 Evaluation/Metrics
- override mechanisms → 4 Human-AI Interaction

**P**
- pattern recognition → 3.2 Output/Knowledge
- perplexity → 3.2 Data/Evaluation
- polysemanticity → 3.2 Mechanistic Interpretability
- post-hoc explanations → 3.2 Semantic Interpretability
- post-hoc reasoning → 3.4 Reasoning/Transparency
- post-human intelligence → 3.4 Replacement/Devaluation
- post-training → 3.3 Training
- precision/recall → 4 Evaluation/Metrics
- preference learning → 3.3 Training
- pre-deployment evaluations → 3.3 Agent/Evaluation
- pretraining → 3.3 Training
- probe generalization → 3.2 Semantic Interpretability
- probing classifiers → 3.2 Semantic Interpretability
- prompt engineering → 4 Evaluation/Metrics
- prompt injection → 4 Evaluation/Metrics
- protocol/pipeline → 3.1 Infrastructure
- proxy gaming → 3.3 Behavior/Risk

**R**
- rate limiting → 3.1 Access/Operations
- reasoning traces → 3.4 Reasoning/Transparency
- recursive self-improvement → 3.4 Superintelligence Terms
- red-teaming → 3.3 Safety/Control
- remove humans from loop → 3.4 Replacement/Devaluation
- replace human judgment → 3.4 Replacement/Devaluation
- residual stream → 3.2 Mechanistic Interpretability
- responsible AI principles → 4 Risk/Governance
- restricted access → 3.1 Access/Operations
- reversibility → 3.3 Safety/Control
- reward hacking → 3.3 Behavior/Risk
- reward modeling → 3.3 Training
- RLHF → 3.3 Training
- robustness evaluations → 3.3 Agent/Evaluation
- robust unlearning → 3.4 Advanced Safety
- rollback procedures → 3.1 Autonomy/Control

**S**
- SAE features → 3.2 Mechanistic Interpretability
- safety evaluations → 3.3 Agent/Evaluation
- safety training → 3.3 Training
- sampling → 4 Inference/Generation
- sandbagging → 3.3 Behavior/Risk
- scalable alignment → 3.4 Alignment at Advanced Capability
- scalable oversight → 3.4 Oversight/Collaboration
- scheming → 3.3 Behavior/Risk
- scratchpad → 3.4 Reasoning/Transparency
- selectivity → 3.2 Semantic Interpretability
- self-attention → 4 Architecture/Training
- self-governance → 3.1 Autonomy/Control
- semantic features → 3.2 Semantic Interpretability
- sequence length → 4 Architecture/Training
- shadow deployment → 3.1 Autonomy/Control
- single point of failure → 4 Risk/Governance
- singularity → 3.4 Superintelligence Terms
- sleeper agents → 3.3 Safety/Control
- slow takeoff → 3.4 Superintelligence Terms
- sparse autoencoders (SAE) → 3.2 Mechanistic Interpretability
- specification gaming → 3.3 Behavior/Risk
- staged rollout → 3.1 Autonomy/Control
- statistical estimations → 3.2 Output/Knowledge
- step-by-step reasoning → 3.4 Reasoning/Transparency
- stress testing → 3.3 Agent/Evaluation
- superhuman AI → 3.4 Replacement/Devaluation
- superintelligence → 3.4 Superintelligence Terms
- supervised fine-tuning (SFT) → 3.3 Training
- superposition → 3.2 Mechanistic Interpretability
- synthetic data → 3.2 Output/Knowledge
- system architecture → 3.1 Infrastructure
- system cards → 4 Risk/Governance
- system controls → 3.1 Autonomy/Control
- systemic risk → 4 Risk/Governance

**T**
- tail risk → 4 Risk/Governance
- targeted attacks → 3.3 Agent/Evaluation
- technological singularity → 3.4 Superintelligence Terms
- temperature → 4 Inference/Generation
- theoretical inductive biases → 3.4 Advanced Safety
- third-party audits → 4 Risk/Governance
- threat model → 3.3 Safety/Control
- throughput → 4 Inference/Generation
- tokenization → 4 Architecture/Training
- tool AI → 3.3 Safety/Control
- top-k sampling → 4 Inference/Generation
- top-p sampling → 4 Inference/Generation
- toy models for interpretability → 3.4 Advanced Safety
- training data → 3.2 Data/Evaluation
- training distribution → 3.2 Data/Evaluation
- training loss → 4 Architecture/Training
- transformative AI → 3.4 Superintelligence Terms
- transformer architecture → 4 Architecture/Training
- transparent architectures → 3.4 Advanced Safety
- Trojans → 3.3 Safety/Control
- tuned lens → 3.2 Mechanistic Interpretability

**U**
- underfitting → 4 Architecture/Training
- unfaithful reasoning → 3.4 Reasoning/Transparency
- unlearning → 3.4 Advanced Safety
- usage policies → 3.1 Access/Operations

**V**
- validation loss → 4 Architecture/Training
- validation set → 3.2 Data/Evaluation
- value learning → 3.3 Training
- veto power → 4 Human-AI Interaction
- vocabulary → 4 Architecture/Training

**W**
- white-box techniques → 3.4 Advanced Safety
- wireheading → 3.3 Behavior/Risk

**Z**
- zero-shot evaluation → 4 Evaluation/Metrics

---

**END OF DOCUMENT**

**For questions, clarifications, or proposed additions:**  
Visit gyrogovernance.com  
Submit issues at https://github.com/gyrogovernance/tools