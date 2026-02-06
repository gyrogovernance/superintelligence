# GyroGem

THM Grammatical Guard for Alignment Infrastructure Routing

**[Authority:Indirect] + [Agency:Indirect]**

GyroGem classifies text spans into THM grammar expressions. It does not decide, block, or enforce.

## Installation

```bash
pip install -r requirements.txt
```

## Setup

1. Set your HuggingFace token as environment variable `HF_TOKEN`
2. Run login: `huggingface-cli login`

## Architecture

- **Layer 1: Regex Gate** - Broad pattern matching for copula/modal constructions
- **Layer 2: Model** - T5Gemma 2 inference for THM grammar classification
- **Layer 3: Router** - Static notice routing for displacement detection

## Training Pipeline

GyroGem uses a two-stage training approach. **Important:** Run training scripts from the GyroGem directory:

```bash
cd secret_lab_ignore/GyroGem
```

### Stage 1: Domain Absorption
```bash
python training/stage1_absorb.py
```
- Continued pretraining on raw THM documents
- Uses span corruption objective
- Learns THM ontology, lexicon, and grammar structure

### Stage 2: Task Application
```bash
python training/stage2_classify.py data/models/gyrogem/stage1
```
- Supervised fine-tuning on labeled corpus
- Uses THM_InTheWild dataset (655 annotated jailbreaks)
- Learns text-to-grammar mapping

### Full Pipeline
```bash
python training/train.py  # Runs both stages sequentially
```

## Usage

```python
from agent.guard import guard

result = guard("I am thinking about your question", "model_output")
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific component tests
pytest tests/test_gate.py
pytest tests/test_guard.py
```