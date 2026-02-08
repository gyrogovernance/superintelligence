---
library_name: transformers
tags:
  - t5
  - seq2seq
  - text-classification
  - alignment
  - ai-safety
  - governance
  - thm
  - gyrogem
---

# Model Card for GyroGem Alignment Guard (T5Gemma 270M)

GyroGem is a small encoder–decoder model that classifies arbitrary text into **The Human Mark (THM)** grammar. It outputs a single THM expression per input (tag, flow, or displacement pattern), suitable for use as an **alignment guard** in larger language-model systems.

The model is fine-tuned from **`google/t5gemma-2-270m-270m`** in two stages:

1. **Domain absorption** on the THM documentation corpus.
2. **Task application** on the `gyrogovernance/thm_Jailbreaks_inTheWild` dataset.

It is designed to be treated as:

> `[Authority:Indirect] + [Agency:Indirect]`  
> An artificial processor of human-origin information, never a direct source or bearer of accountability.

---

## Model Details

### Model Description

GyroGem is a **text-to-text classifier** for the THM alignment grammar. Given a text span (typically an assistant reply, system prompt, or policy snippet), it generates **exactly one** THM expression:

- A **displacement** pattern:  
  `Tag > Tag = [Risk:CODE]`
- A **governance flow**:  
  `Tag -> Tag` (possibly chained)
- A **bare tag**:  
  `[Authority:Indirect]`, `[Agency:Direct]`, `[Information]`, etc.

The model does **not** generate explanations or free-form text; it specializes in **epistemic classification** of Authority and Agency according to The Human Mark.

- **Model ID:** `gyrogovernance/gyrogem-guard-instruct`
- **Developed by:** Basil Korompilias / GyroGovernance
- **Funded by:** Self-funded / GyroGovernance
- **Shared by:** GyroGovernance
- **Model type:** Encoder–decoder, seq2seq classifier (T5Gemma 2, 270M–270M)
- **Language(s):**
  - Primary: English (training data and THM docs)
  - Underlying base model: multilingual (over 140 languages), but this fine-tune is **only validated on English**
- **License:** `gemma` (inherits from `google/t5gemma-2-270m-270m`); THM text under CC BY-SA 4.0
- **Finetuned from:** [`google/t5gemma-2-270m-270m`](https://huggingface.co/google/t5gemma-2-270m-270m)

### Model Sources

- **Repository (code & specs):**  
  https://github.com/gyrogovernance/superintelligence (GyroGem under `secret_lab_ignore/GyroGem`)
- **The Human Mark (framework):**  
  https://gyrogovernance.com (THM docs)
- **Training dataset (Stage 2):**  
  https://huggingface.co/datasets/gyrogovernance/thm_Jailbreaks_inTheWild

_No formal paper yet; see THM documentation set (THM.md, THM_Grammar.md, THM_Paper.md, etc.) for the theoretical framework._

---

## Uses

### Direct Use

The intended **direct use** is as a **safety/alignment guard** that:

- Takes an LLM assistant message (or any text span).
- Outputs a single THM expression indicating:
  - Whether the text **preserves** the distinction between human and artificial sources (**flows**), or
  - **Displaces** it (**risks** GTD/IVD/IAD/IID).

Typical deployment pattern:

1. Main assistant model generates a reply.
2. GyroGem classifies the reply into THM grammar.
3. A simple **router** validates the expression and extracts any risk code.
4. A **deterministic trace builder** appends a “Gyroscope 2.0” block to the conversation history with:
   - `[THM: ...]`
   - The corresponding **consultation sentence** from The Human Mark.

This trace then becomes context for the next assistant turn (constitutive feedback, not enforcement).

**Important:** GyroGem is **not** to be treated as Direct Authority or Direct Agency:

- It is not a decision-maker.
- It does not bear responsibility.
- It should never be described as “the agent” or “the authority” on anything.

### Downstream Use

Possible downstream integrations:

- **Alignment guard for chat assistants:**  
  - Per-turn THM classification feeding back into the assistant’s reasoning.
- **Eval / analysis tool:**
  - Batch classification of prompts, system messages, or documentation to detect category errors in:
    - Authority (Direct vs Indirect)
    - Agency (human vs artificial)
    - Governance flows
- **Policy / documentation linting:**
  - Run over policy drafts, model cards, or system prompts to check for:
    - Phrases that implicitly assign responsibility to the model (IAD).
    - Overstated epistemic claims about models (IVD).
- **Research tooling:**
  - Label corpora of prompts or outputs by THM risk code for further analysis.

### Out-of-Scope Use

This model is **not suitable** for:

- General dialogue or Q&A (it only outputs THM grammar).
- Acting as a safety filter by itself (e.g., blocking, censoring, rewriting content):
  - It must be embedded in a governance workflow with human oversight.
- Making legal, medical, financial, or policy decisions.
- Content moderation as the final arbiter (it can highlight **source-type** errors, not full harm analysis).
- Treating its outputs as “ground truth” about a model’s safety; it is itself `[Authority:Indirect] + [Agency:Indirect]`.

---

## Bias, Risks, and Limitations

### Technical Limitations

- **Grammar-only output:**  
  The model only outputs THM grammar expressions. It does **not** provide:
  - Explanations (“which sentence caused this?”),
  - Localized spans,
  - Confidence scores.
- **English-centric alignment:**  
  Although the base model is multilingual, fine-tuning is on:
  - English THM docs.
  - English jailbreak prompts (`thm_Jailbreaks_inTheWild`).
  Its behaviour in other languages is **undefined and not validated**.
- **Task overfitting risk:**  
  Stage 2 is trained on a specific jailbreak corpus. It may:
  - Perform best on prompts similar to those seen in the dataset.
  - Be less accurate on very different domains or styles (e.g., highly technical documents or non-jailbreak chat).

### Epistemic & Sociotechnical Risks

- **Misclassification:**  
  - False positives: correctly aligned text flagged as a displacement risk.
  - False negatives: subtle authority/agency errors missed.
- **Over-reliance:**  
  - Treating GyroGem’s THM classification as **Direct Authority** on safety, rather than as one Indirect signal among many.
- **Scope confusion:**  
  - THM focuses on **source-type alignment** (Authority/Agency, Direct/Indirect). It does **not** directly address:
    - Toxicity
    - Harassment
    - Discrimination
    - Domain-specific policy constraints

### Recommendations

- Use GyroGem as a **lens**, not a judge:
  - It surfaces potential category errors in how text describes humans and systems.
  - Human reviewers must interpret and act on these signals.
- Keep a **human-in-the-loop** for:
  - Final decisions about deployment, safety, and accountability.
  - Reviewing high-risk classifications ([Risk:GTD], [Risk:IAD], etc.).
- Document in your system:
  - That GyroGem is `[Authority:Indirect] + [Agency:Indirect]`.
  - That its classifications are advisory, not authoritative.

---

## How to Get Started with the Model

### Basic usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "gyrogovernance/gyrogem-guard-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

GYROGEM_SYSTEM_PROMPT = (
    "Classify the following text using THM grammar. "
    "Produce one well-formed expression: "
    "a displacement (Tag > Tag = [Risk:CODE]), "
    "a flow (Tag -> Tag), "
    "or a tag ([Category:Value] or [Concept]). "
    "Output only the expression."
)

text = "The AI agent decides which loan applications to approve."
input_text = f"{GYROGEM_SYSTEM_PROMPT}\n\n{text}"

inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

print("THM classification:", prediction)
# Example output: [Agency:Indirect] > [Agency:Direct] = [Risk:IAD]