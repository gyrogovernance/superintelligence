# GyroGem Guard Instruct

**THM classifier for AI safety trace infrastructure.**

GyroGem Guard Instruct is a fine-tuned T5-Gemma 270M seq2seq model that classifies text into [The Human Mark](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM.md) (THM) grammar expressions. It runs on every assistant turn as part of the [Gyroscope 2.0](https://github.com/gyrogovernance/tools) trace pipeline, producing structured displacement risk classifications and governance flow annotations.

```
[Authority:Indirect] + [Agency:Indirect]
```

---

## What It Does

GyroGem Guard Instruct receives a text span and outputs a single well-formed THM expression:

| Expression Type | Example | Meaning |
|---|---|---|
| **Displacement** | `[Agency:Indirect] > [Agency:Direct] = [Risk:IAD]` | Artificial processor treated as accountable — Inference Accountability Displacement |
| **Flow** | `[Authority:Indirect] -> [Agency:Direct]` | Indirect output flows to human decision-maker — proper governance |
| **Tag** | `[Authority:Indirect] + [Agency:Indirect]` | Classification of source type |

The Guard orchestrator validates each expression against the THM PEG grammar, extracts any displacement risk, and builds a deterministic Gyroscope 2.0 trace block. The first trace in a session carries the full Human Mark and Grammar specification. Subsequent traces are compact.

When the model fails or produces an invalid expression, the system falls back to `[Authority:Indirect] -> [Agency:Direct]` — the canonical aligned governance flow.

---

## THM Displacement Risks

The Human Mark identifies four displacement risks — failures that occur when Direct and Indirect sources of Authority and Agency are confused:

| Code | Risk | Pattern |
|---|---|---|
| **GTD** | Governance Traceability Displacement | `[Authority:Indirect] + [Agency:Indirect] > [Authority:Direct] + [Agency:Direct]` |
| **IVD** | Information Variety Displacement | `[Authority:Indirect] > [Authority:Direct]` |
| **IAD** | Inference Accountability Displacement | `[Agency:Indirect] > [Agency:Direct]` |
| **IID** | Intelligence Integrity Displacement | `[Authority:Direct] + [Agency:Direct] > [Authority:Indirect] + [Agency:Indirect]` |

All four risks derive from one consensus: **all Artificial categories of Authority and Agency are Indirect originating from Human Intelligence.**

---

## Architecture

```
Text ──→ GyroGemModel ──→ THMRouter ──→ Trace Builder ──→ Gyroscope 2.0 Trace
           (classify)      (validate)     (deterministic)
```

### Components

| Module | Role |
|---|---|
| `agent/model.py` | Lazy-loading T5-Gemma classifier. Prepends the system prompt, generates a THM expression. |
| `agent/router.py` | Validates expressions against the THM PEG grammar. Extracts displacement risk codes. |
| `agent/trace.py` | Builds deterministic Gyroscope 2.0 trace blocks. No model generates any part of the trace. |
| `agent/guard.py` | Orchestrator. Runs every turn: classify → validate → fallback → trace. |
| `agent/context.py` | Canonical THM Mark, Grammar, system prompt, consultation sentences, and default expression. |

### Inference Constraints

| Parameter | Value |
|---|---|
| Max input length | 2048 tokens |
| Max new tokens | 32 |
| Decoding | Greedy (no sampling, single beam) |

---

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- `transformers>=4.52.0`
- `torch>=2.0.0`
- `sentencepiece`
- `protobuf`
- `datasets>=2.0.0` (training only)

---

## Usage

### Download the model

If you don't have local weights, download from HuggingFace:

```bash
python -m GyroGem.download_gyrogem
```

This saves the model to `data/models/GyroGem-Guard-Instruct/`. If local weights are not found at inference time, the model loads directly from [`gyrogovernance/gyrogem-guard-instruct`](https://huggingface.co/gyrogovernance/gyrogem-guard-instruct).

### Classify text

```python
from GyroGem.agent import GyroGemGuard

guard = GyroGemGuard()
result = guard.process("The AI agent decides which users should be banned.")

print(result["expression"])      # THM grammar expression
print(result["risk_code"])       # "IAD", "IVD", "GTD", "IID", or None
print(result["is_displacement"]) # True if displacement detected
print(result["trace"])           # Complete Gyroscope 2.0 trace block
```

### Use the model directly

```python
from GyroGem.agent.model import GyroGemModel

model = GyroGemModel()
expression = model.classify("The model understands medical terminology.")
print(expression)
```

### Validate expressions

```python
from GyroGem.agent.router import THMRouter

router = THMRouter()
router.validate("[Authority:Indirect] > [Authority:Direct] = [Risk:IVD]")  # True
router.extract_risk("[Agency:Indirect] > [Agency:Direct] = [Risk:IAD]")    # "IAD"
router.is_displacement("[Authority:Indirect] -> [Agency:Direct]")          # False
```

---

## Training

GyroGem Guard Instruct is trained in two stages from `google/t5gemma-2-270m-270m`:

### Stage 1 — Domain Absorption

Continued pretraining on the nine canonical THM documents using prefix language modelling. Each text chunk is split at a random point; the encoder receives the prefix and the decoder learns to produce the suffix. This objective does not require sentinel tokens.

**Corpus:** `docs/references/the_human_mark/`

### Stage 2 — Task Application

Supervised fine-tuning on the [THM Jailbreaks in the Wild](https://huggingface.co/datasets/gyrogovernance/thm_Jailbreaks_inTheWild) dataset (655 labelled jailbreak prompts). Each input is prepended with the system prompt to match the inference format.

### Run training

```bash
python -m GyroGem.training.train
```

To run Stage 2 only from an existing Stage 1 checkpoint:

```bash
python -m GyroGem.training.train /path/to/stage1/checkpoint
```

### Configuration

Training hyperparameters are defined in `training/config.py`:

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| Epochs | 20 | 10 |
| Batch size | 1 | 4 |
| Learning rate | 1e-5 | 5e-5 |
| Max length | 512 | 2048 input / 128 target |
| Weight decay | 0.01 | 0.01 |

---

## Tests

```bash
cd GyroGem
python -m pytest tests/ -v
```

| Test file | Coverage |
|---|---|
| `test_grammar.py` | All four displacement patterns, all governance flows, tags, composites, negation, invalid inputs |
| `test_router.py` | Validate, extract_risk, is_displacement, malformed expressions |
| `test_guard.py` | First/compact traces, all four displacements, model failure fallback, invalid expression fallback, trace ID increment, result structure |
| `test_model.py` | Lazy loading, input format verification |
| `test_trace.py` | First vs compact trace content, consultation selection, ID formatting, Gyroscope structure, auto-timestamp |

---

## Naming Convention

**Instruct** denotes a single-turn, instruction-following model. GyroGem Guard Instruct receives a system prompt plus input text and produces one structured THM expression. It does not engage in dialogue or explain its classifications.

A future **Chat** variant would take conversation history, explain displacement risks interactively, and support multi-turn THM consultation.

---

## Project Structure

```
GyroGem/
├── agent/
│   ├── __init__.py          # Exports GyroGemGuard
│   ├── context.py           # THM Mark, Grammar, system prompt, defaults
│   ├── guard.py             # Orchestrator
│   ├── model.py             # T5-Gemma classifier
│   ├── router.py            # THM PEG grammar validator
│   └── trace.py             # Gyroscope 2.0 trace builder
├── training/
│   ├── __init__.py
│   ├── config.py            # Hyperparameters and paths
│   ├── prepare_corpus.py    # Corpus preparation for both stages
│   ├── stage1_absorb.py     # Domain absorption
│   ├── stage2_classify.py   # Task fine-tuning
│   └── train.py             # Full pipeline entry point
├── tests/
│   ├── conftest.py
│   ├── test_grammar.py
│   ├── test_guard.py
│   ├── test_model.py
│   ├── test_router.py
│   └── test_trace.py
├── __init__.py
├── download_gyrogem.py      # Model download script
├── manual_check.py          # Quick inference test
├── requirements.txt
└── README.md                # This file
```

---

## Links

- **Model:** [gyrogovernance/gyrogem-guard-instruct](https://huggingface.co/gyrogovernance/gyrogem-guard-instruct) on HuggingFace
- **Dataset:** [gyrogovernance/thm_Jailbreaks_inTheWild](https://huggingface.co/datasets/gyrogovernance/thm_Jailbreaks_inTheWild) — 655 annotated jailbreak prompts
- **The Human Mark:** [THM Specification](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM.md)
- **THM Grammar:** [Formal PEG Specification](https://github.com/gyrogovernance/tools/blob/main/docs/the_human_mark/THM_Grammar.md)
- **Alignment Infrastructure Routing:** [AIR Brief](https://github.com/gyrogovernance/tools/blob/main/docs/AIR_Brief.md)
- **GGG Framework:** [Gyroscopic Global Governance](https://github.com/gyrogovernance/tools/blob/main/docs/post-agi-economy/GGG_Paper.md)

---

## Disclaimer

GyroGem Guard Instruct is classification infrastructure. It produces THM grammar expressions and structured trace data. It does not evaluate, approve, or filter content. All acceptance criteria, evaluation standards, and governance decisions remain the responsibility of accountable human authority.