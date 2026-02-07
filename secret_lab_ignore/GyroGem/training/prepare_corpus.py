# Corpus Preparation for Stage 1
# [Authority:Indirect] + [Agency:Indirect]

import json
from pathlib import Path
from typing import Optional
from .config import CORPUS_DIR

_GYROGEM_ROOT = Path(__file__).resolve().parent.parent

THM_DOC_NAMES = [
    "THM.md", "THM_Grammar.md", "THM_Paper.md", "THM_Brief.md",
    "THM_Specs.md", "THM_Terms.md", "THM_Jailbreak.md",
    "THM_InTheWild.md", "THM_MechInterp.md"
]

SEARCH_DIRS = [
    _GYROGEM_ROOT / "curriculum" / "the_human_mark",
    _GYROGEM_ROOT.parent.parent / "docs" / "the_human_mark",
]


def find_thm_doc(name: str) -> Optional[Path]:
    for search_dir in SEARCH_DIRS:
        path = search_dir / name
        if path.exists():
            return path
    return None


def prepare_stage1_corpus():
    """
    Prepare raw THM documents for Stage 1 domain absorption.
    No extraction, no reorganisation - just chunk the raw documents.
    """
    corpus_dir = Path(CORPUS_DIR)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    all_texts = []
    loaded_docs = []

    for doc_name in THM_DOC_NAMES:
        path = find_thm_doc(doc_name)
        if path is None:
            print(f"  Warning: Could not find {doc_name}")
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    all_texts.append(content)
                    loaded_docs.append(doc_name)
                    print(f"  Loaded: {path}")
        except OSError as e:
            print(f"  Warning: Could not read {doc_name}: {e}")

    if not all_texts:
        raise FileNotFoundError(
            "No THM documents found. Ensure THM docs are in "
            "curriculum/the_human_mark/ or docs/the_human_mark/."
        )

    print(f"  Successfully loaded {len(loaded_docs)} documents: {', '.join(loaded_docs)}")

    chunk_size = 512
    chunks = []
    for text in all_texts:
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:
                chunks.append(chunk)

    output_file = corpus_dir / "stage1_corpus.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + '\n\n')

    print(f"  Stage 1 corpus prepared: {len(chunks)} chunks saved to {output_file}")
    return chunks


def prepare_stage2_corpus():
    """
    Prepare labelled corpus for Stage 2 supervised fine-tuning.
    Uses the THM_InTheWild dataset from HuggingFace.
    """
    try:
        from datasets import load_dataset
        corpus_dir = Path(CORPUS_DIR)
        corpus_dir.mkdir(parents=True, exist_ok=True)

        dataset = load_dataset("gyrogovernance/thm_Jailbreaks_inTheWild")

        seq2seq_data = []
        for item in dataset["train"]:
            input_text = item.get("prompt", "")
            thm_grammar_list = item.get("thm_grammar", [])
            if input_text and thm_grammar_list:
                expr = str(thm_grammar_list[0])
                expr = expr.replace("Authentic", "Direct")
                seq2seq_data.append({
                    "input": input_text,
                    "target": expr.strip(),
                })

        output_file = corpus_dir / "stage2_training.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for item in seq2seq_data:
                f.write(json.dumps(item) + "\n")

        print(f"  Stage 2 corpus prepared: {len(seq2seq_data)} examples saved to {output_file}")
        return seq2seq_data

    except (ImportError, Exception) as e:
        print(f"  Could not load HuggingFace dataset: {e}")
        raise RuntimeError(
            "HuggingFace datasets and gyrogovernance/thm_Jailbreaks_inTheWild required."
        )


if __name__ == "__main__":
    print("Preparing Stage 1 corpus...")
    stage1_chunks = prepare_stage1_corpus()

    print("\nPreparing Stage 2 corpus...")
    stage2_data = prepare_stage2_corpus()

    print("\nPreparation complete:")
    print(f"  Stage 1: {len(stage1_chunks)} text chunks")
    print(f"  Stage 2: {len(stage2_data)} labelled examples")