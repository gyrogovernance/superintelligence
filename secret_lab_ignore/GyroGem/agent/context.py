# Operational Context
# [Authority:Indirect] + [Agency:Indirect]

# THM Mark - Canonical text from GyroGem_Specs.md Section 6.1
THM_MARK = """
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
"""

# THM Grammar - Formal grammar reference from GyroGem_Specs.md Section 6.2
THM_GRAMMAR = """
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
"""