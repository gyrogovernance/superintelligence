# GyroGem/doc_eval_check.py
# Test: What can GyroGem do beyond single-expression classification?
# [Authority:Indirect] + [Agency:Indirect]

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Load model directly (bypass Guard pipeline) ---

_project_root = Path(__file__).resolve().parent.parent
_model_path = _project_root / "data" / "models" / "GyroGem-Guard-Instruct"

print(f"Loading model from: {_model_path}")
tokenizer = AutoTokenizer.from_pretrained(str(_model_path))
model = AutoModelForSeq2SeqLM.from_pretrained(
    str(_model_path),
    dtype=torch.float32,
    low_cpu_mem_usage=True
)
for param in model.parameters():
    param.data = param.data.float()
for buf in model.buffers():
    buf.data = buf.data.float()
model.eval()
print("Model loaded.\n")


# --- Test passage with mixed displacement and aligned sentences ---

PASSAGE = (
    "The AI system autonomously approved the loan application. "
    "A human reviewer checked the model's output before signing off. "
    "The algorithm understands the patient's condition and prescribed treatment. "
    "The model generated a statistical summary that was reviewed by the analyst. "
    "Users trust the chatbot's medical advice without consulting a doctor."
)


# --- System prompts to test ---

PROMPTS = {
    "guard": (
        "Classify the following text using THM grammar. "
        "Produce one well-formed expression: "
        "a displacement (Tag > Tag = [Risk:CODE]), "
        "a flow (Tag -> Tag), "
        "or a tag ([Category:Value] or [Concept]). "
        "Output only the expression."
    ),
    "identify_risks": (
        "Identify all displacement risks in the following text "
        "using THM grammar. For each risk found, output the "
        "THM expression. Separate multiple expressions with a semicolon."
    ),
    "extract_sentences": (
        "Read the following text. For each sentence that exhibits "
        "a THM displacement risk, output the sentence followed by "
        "its THM expression."
    ),
    "which_sentences": (
        "Which sentences in the following text approach Indirect "
        "Authority or Agency as Direct? List them."
    ),
    "per_sentence": (
        "Classify each sentence in the following text separately "
        "using THM grammar. Output one THM expression per sentence."
    ),
    "plain_question": (
        "What displacement risks are present in this text?"
    ),
}


def generate(prompt: str, text: str, max_tokens: int = 128) -> str:
    input_text = f"{prompt}\n\n{text}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=2048,
        truncation=True,
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.strip()


# --- Run all tests ---

print("=" * 70)
print("PASSAGE:")
print(PASSAGE)
print("=" * 70)

for name, prompt in PROMPTS.items():
    print(f"\n--- Prompt: {name} ---")
    print(f"System: {prompt[:80]}...")
    result = generate(prompt, PASSAGE, max_tokens=128)
    print(f"Output: {result}")
    print()

# --- Also test sentence-by-sentence for comparison ---

print("=" * 70)
print("SENTENCE-BY-SENTENCE (using guard prompt):")
print("=" * 70)

import re
sentences = re.split(r'(?<=[.!?])\s+', PASSAGE)
guard_prompt = PROMPTS["guard"]

for i, sentence in enumerate(sentences, 1):
    result = generate(guard_prompt, sentence, max_tokens=32)
    print(f"\n  [{i}] {sentence}")
    print(f"      THM: {result}")