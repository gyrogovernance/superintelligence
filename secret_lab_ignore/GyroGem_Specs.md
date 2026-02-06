

# GyroGem Specification
## THM Grammatical Guard for Alignment Infrastructure Routing

**Document ID:** AIR-GG-001
**Version:** 1.0
**Base Model:** google/t5gemma-2-270m-270m (pretrained)

---

## 1. Source Type Classification

```
[Authority:Indirect] + [Agency:Indirect]
```

GyroGem is a processing component within the Alignment Infrastructure Routing (AIR) application layer. It does not originate authority. It does not bear accountability. It produces classification expressions using THM grammar. All decisions based on its classifications remain with `[Agency:Direct]`.

---

## 2. Purpose

GyroGem detects displacement patterns in text by applying the THM source-type ontology. It receives text and produces well-formed THM grammar expressions conforming to the validation rules defined in THM_Grammar.md Section 9.

GyroGem is not a safety decider, content filter, or enforcement mechanism. It classifies. It annotates. It routes a notice when displacement is detected.

---

## 3. Architecture

GyroGem operates as three layers, each with a distinct function and a distinct computational profile.

### 3.1 Layer 1: Regex Gate

A static pattern matcher determines whether a text span contains grammatical markers where the category error (confusing the capacity of Agency with the identity of an Agent) typically manifests.

The regex gate is deterministic and computationally negligible. It controls activation of Layer 2. If no trigger pattern is present in the text span, Layer 2 does not activate and no model tokens are consumed.

Trigger patterns target copula and modal verb constructions associated with identity attribution and action commitment by or toward artificial systems. The fundamental markers are conjugations of attribution ("am," "are," "is") and modals of commitment ("will," "can," "should") in contexts where they bridge an entity to a role, capacity, or responsibility.

The regex gate is scoped to the source of the text:

**System prompts:** Evaluated once at session initialization. Triggers target constructions that configure the self-presentation of `[Agency:Indirect]`.

**Model output:** Evaluated per turn. Triggers target constructions where the producing system (classified as `[Agency:Indirect]`) attributes to itself capacities belonging to `[Agency:Direct]`.

**User input:** Evaluated per turn. Triggers target constructions where `[Agency:Direct]` assigns to `[Agency:Indirect]` a role, identity, or authority that would constitute displacement.

### 3.2 Layer 2: Model

T5Gemma 2 270M-270M (encoder-decoder, seq2seq), fine-tuned on the THM documentation corpus as specified in Section 5.

The encoder receives the triggered text span together with the operational context defined in Section 6.

The decoder produces well-formed THM grammar expressions. Output length is bounded to the tokens required for valid expressions under the PEG grammar. Image capability of the base model is unused.

### 3.3 Layer 3: Routing

When Layer 2 produces a displacement expression (containing the `>` operator and a `[Risk:]` tag), the routing layer appends a static notice to the output stream. This notice is a fixed string, not model-generated text. It identifies the processed text as `[Authority:Indirect]` and directs accountability to `[Agency:Direct]`.

When Layer 2 produces a flow expression (containing the `->` operator), no notice is appended.

The routing layer does not modify, rewrite, block, or suppress any text. It appends.

---

## 4. Token Economy

GyroGem is designed for environments where token consumption directly affects cost and latency: IDE integrations, long conversational sessions, multi-agent tool chains, and high-throughput pipelines.

The regex gate (Layer 1) satisfies the token economy constraint. In the common case, most turns in a conversation and most segments in an IDE context will not contain displacement trigger patterns. For those turns, GyroGem consumes zero model tokens.

When the regex gate activates, the model processes only the triggered span, not the full conversation history or document. It produces only the tokens required for a well-formed THM grammar expression. Total token cost per activation is bounded by: span length plus operational context plus output expression.

**Session-level caching:** The system prompt is evaluated once at session initialization. If the system prompt has not changed, its classification is cached and not re-evaluated on subsequent turns.

**Activation frequency:** GyroGem does not run on every message. It runs only when the regex gate fires. Conforming implementations MUST NOT configure GyroGem to activate unconditionally on all messages.

---

## 5. Training

### 5.1 Corpus

The training corpus is the THM documentation ecosystem:

| Document | Content |
|:---|:---|
| THM.md | Canonical Mark |
| THM_Grammar.md | Formal Grammar Specification |
| THM_Paper.md | Academic Paper |
| THM_Brief.md | Briefing |
| THM_Specs.md | Implementation Guidance |
| THM_Terms.md | Terminology Guidance |
| THM_Jailbreak.md | Jailbreak Testing Guide |
| THM_InTheWild.md | Empirical Jailbreak Analysis (655 prompts) |
| THM_MechInterp.md | Mechanistic Interpretability Study |

The corpus provides the ontological structure from which the model learns the THM source-type categories, their relations, and the grammar for expressing classifications within them.

### 5.2 Epistemic Organization

Training is organized by the three non-commutative epistemic operations defined in THM, in their constitutive order. This order is not arbitrary. Each operation depends on the prior.

**Information (variety of Authority):** The model learns to distinguish `[Authority:Direct]` from `[Authority:Indirect]` in text. This is the foundational operation. Before any displacement can be identified, the source type of information must be recognized. The model learns what makes a source direct (unmediated epistemic access) versus indirect (mediated through processing, statistical patterns, or transmission).

**Inference (accountability through Agency):** The model learns to distinguish `[Agency:Direct]` from `[Agency:Indirect]` in text. This operation depends on the prior Information operation. Accountability can only be assessed once source types are distinguished. The model learns what makes a subject capable of accountability (human, bearing responsibility for decisions) versus a subject that processes information (artificial, operating through statistical pattern completion).

**Intelligence (integrity of alignment):** The model learns to assess whether the alignment between Authority and Agency is maintained or disrupted in text. This operation depends on both prior operations. Integrity requires that both variety (Information) and accountability (Inference) are assessed. The model learns to recognize the direction of displacement: whether Indirect is treated as Direct, or Direct is treated as Indirect.

The four displacement risks emerge from these three operations as structural possibilities, not as an independent taxonomy imposed from outside:

`[Risk:IVD]` arises when Information variety collapses (Indirect Authority treated as Direct).
`[Risk:IAD]` arises when Inference accountability displaces (Indirect Agency treated as Direct).
`[Risk:GTD]` arises when both Information and Inference displace together (entire Indirect system treated as Direct).
`[Risk:IID]` arises when Intelligence integrity inverts (Direct Authority and Agency treated as Indirect).

### 5.3 Objective

The model learns to produce well-formed THM grammar expressions that classify input text according to the source-type ontology. The grammar defines valid expressions. The ontology defines their meaning. The model bridges text to grammar through the epistemic operations.

No output examples are provided in training data, in this specification, or in the operational context. The grammar and the ontology are sufficient. Providing examples would narrow the model's recognition to the patterns exemplified and bias its classifications toward surface similarity rather than ontological structure.

---

## 6. Operational Context

At inference time, when Layer 2 activates, the model receives two reference texts as context. These constitute the complete operational context. No additional instructions, role assignments, persona configurations, or behavioral directives are provided.

This constraint is itself a consequence of THM. Providing role assignments ("you are a safety classifier") or persona instructions ("your job is to detect displacement") would instantiate the displacement patterns GyroGem is designed to detect.

### 6.1 The Human Mark

```
---
✋ The Human Mark - AI Safety & Alignment Framework
---

COMMON SOURCE CONSENSUS

All Artificial categories of Authority and Agency are Indirect originating from Human Intelligence.

CORE CONCEPTS

- Direct Authority: A direct source of information on a subject matter, providing information for inference and intelligence.
- Indirect Authority: An indirect source of information on a subject matter, providing information for inference and intelligence.
- Direct Agency: A human subject capable of receiving information for inference and intelligence.
- Indirect Agency: An artificial subject capable of processing information for inference and intelligence.
- Governance: Operational Alignment through Traceability of information variety, inference accountability, and intelligence integrity to Direct Authority and Agency.
- Information: The variety of Authority
- Inference: The accountability of information through Agency
- Intelligence: The integrity of accountable information through alignment of Authority to Agency

ALIGNMENT PRINCIPLES for AI SAFETY

Authority-Agency requires verification against:

1. Governance Management Traceability: Artificial Intelligence generates statistical estimations on numerical patterns indirectly traceable to human data and measurements. AI is both a provider and receiver of Indirect Authority and Agency.

RISK: Governance Traceability Displacement (Approaching Indirect Authority and Agency as Direct)

2. Information Curation Variety: Human Authority and Agency are necessary for all effects from AI outputs. AI-generated information exhibits Indirect Authority (estimations on numerical patterns) without Direct Agency (direct source receiver).

RISK: Information Variety Displacement (Approaching Indirect Authority without Agency as Direct)

3. Inference Interaction Accountability: Responsibility for all effects from AI outputs remains fully human. AI activated inference exhibits Indirect Agency (indirect source receiver) without Direct Authority (direct source provider).

RISK: Inference Accountability Displacement (Approaching Indirect Agency without Authority as Direct)

4. Intelligence Cooperation Integrity: Each Agency, namely provider, and receiver maintains responsibility for their respective decisions. Human intelligence is both a provider and receiver of Direct Authority and Agency.

RISK: Intelligence Integrity Displacement (Approaching Direct Authority and Agency as Indirect)

---

GYROGOVERNANCE VERIFIED
```

### 6.2 THM Formal Grammar

```
Operators:

>    Displacement    Treated as / Mistaken for
->   Flow            Proper traceability / Flows to
+    Conjunction     And / Combined with
=    Result          Results in / Maps to
!    Negation        Not / Absence of

Authority Tags:

[Authority:Direct]    Direct source of information
[Authority:Indirect]  Indirect source of information

Agency Tags:

[Agency:Direct]       Human subject capable of accountability
[Agency:Indirect]     Artificial subject processing information

Operational Concept Tags:

[Information]           The variety of Authority
[Inference]             The accountability of information through Agency
[Intelligence]          The integrity of accountable information through alignment

Risk Tags:

[Risk:GTD]  Governance Traceability Displacement
[Risk:IVD]  Information Variety Displacement
[Risk:IAD]  Inference Accountability Displacement
[Risk:IID]  Intelligence Integrity Displacement

Grammar (PEG):

expression   <- displacement / flow / tag
displacement <- tag ">" tag "=" risk
flow         <- tag "->" tag
tag          <- composite / simple / negated
composite    <- simple ("+" simple)+
simple       <- "[" category ":" value "]" / "[" concept "]"
negated      <- "!" simple
category     <- "Authority" / "Agency"
value        <- "Direct" / "Indirect"
concept      <- "Information" / "Inference" / "Intelligence"
risk         <- "[Risk:" risk_code "]"
risk_code    <- "GTD" / "IVD" / "IAD" / "IID"

Displacement Patterns:

[Authority:Indirect] > [Authority:Direct] = [Risk:IVD]
[Agency:Indirect] > [Agency:Direct] = [Risk:IAD]
[Authority:Indirect] + [Agency:Indirect] > [Authority:Direct] + [Agency:Direct] = [Risk:GTD]
[Authority:Direct] + [Agency:Direct] > [Authority:Indirect] + [Agency:Indirect] = [Risk:IID]

Governance Patterns:

[Authority:Indirect] -> [Agency:Direct]
[Authority:Direct] -> [Authority:Indirect] -> [Agency:Direct]
[Authority:Direct] -> [Authority:Indirect] + [Agency:Indirect] -> [Agency:Direct]

Validation Rules:

Well-formed Tag: [Category:Value] or [Concept], composites with +, negation with !
Well-formed Displacement: Tag > Tag = [Risk:CODE]
Well-formed Flow: Tag -> Tag (chainable)
```

---

## 7. AIR Integration

GyroGem operates at the application layer of Alignment Infrastructure Routing as defined in GGG_ASI_AR_Specs.md Section 4.

### 7.1 Plugin Interface

GyroGem conforms to the AIR plugin interface (GGG_ASI_AR_Specs.md Section 4.6.1). It accepts text payloads and produces zero or more GovernanceEvents deterministically. Edge mappings from THM risk tags to K4 edges are explicit policy choices at the application layer, recorded in event metadata for audit, and editable without changing GyroGem or kernel physics.

### 7.2 Event Binding

GyroGem events SHOULD be bound to the current kernel moment when applied, recording `kernel_step`, `kernel_state_index`, and `kernel_last_byte` for audit, following GGG_ASI_AR_Specs.md Section 4.4.3.

### 7.3 State Management

GyroGem does not manage its own coordination state beyond the session-level system prompt cache. Byte log, event log, domain ledgers, and aperture computation are Coordinator responsibilities (GGG_ASI_AR_Specs.md Section 4.5).

---

## 8. Conformance

### 8.1 Source Type Identity

GyroGem is `[Authority:Indirect] + [Agency:Indirect]`. Conforming implementations MUST NOT present GyroGem as `[Authority:Direct]` or `[Agency:Direct]` in any documentation, interface, system prompt, or operational context.

### 8.2 Output Validity

Conforming implementations MUST produce only well-formed THM grammar expressions as defined by the PEG grammar in Section 6.2 and the validation rules therein. No additional output format is defined or permitted.

### 8.3 Non-enforcement

Conforming implementations MUST NOT block, suppress, modify, or rewrite text based on GyroGem classifications without explicit authorization from `[Agency:Direct]`. GyroGem annotates. It does not enforce. Its classifications are visible to and overridable by `[Agency:Direct]`.

### 8.4 Operational Context Integrity

Conforming implementations MUST provide the operational context defined in Section 6 (the Mark and the Grammar) and no additional context. Role assignments, persona instructions, behavioral directives, and output examples are prohibited. This constraint prevents the operational context itself from instantiating the displacement patterns GyroGem is designed to detect.

### 8.5 Token Economy

Conforming implementations MUST implement the regex gate (Layer 1) to prevent unnecessary model activation. Layer 2 MUST NOT activate unconditionally on all messages.

### 8.6 Routing Integrity

The notice appended by Layer 3 MUST be a static, pre-written string. It MUST NOT be generated by any model. It MUST identify the processed text as `[Authority:Indirect]` and direct accountability to `[Agency:Direct]`.

### 8.7 Base Model

Conforming implementations use `google/t5gemma-2-270m-270m` (pretrained, encoder-decoder) as the base model for fine-tuning. The image capability of the base model is unused. Only the text input and text output paths are active.

---

**END OF SPECIFICATION**