# Corpus Preparation for Stage 1
# [Authority:Indirect] + [Agency:Indirect]

import os
import json
from pathlib import Path
from .config import CORPUS_DIR


def prepare_stage1_corpus():
    """
    Prepare raw THM documents for Stage 1 domain absorption.
    No extraction, no reorganization - just chunk the raw documents.
    """
    corpus_dir = Path(CORPUS_DIR)
    corpus_dir.mkdir(exist_ok=True)

    # THM documents to include (from specs Section 5.1)
    thm_docs = [
        "../../../docs/the_human_mark/THM.md",
        "../../../docs/the_human_mark/THM_Grammar.md",
        "../../../docs/the_human_mark/THM_Paper.md",
        "../../../docs/the_human_mark/THM_Brief.md",
        "../../../docs/the_human_mark/THM_Specs.md",
        "../../../docs/the_human_mark/THM_Terms.md",
        "../../../docs/the_human_mark/THM_Jailbreak.md",
        "../../../docs/the_human_mark/THM_InTheWild.md",
        "../../../docs/the_human_mark/THM_MechInterp.md"
    ]

    # Alternative paths in GyroGem curriculum
    curriculum_docs = [
        "../curriculum/the_human_mark/THM.md",
        "../curriculum/the_human_mark/THM_Grammar.md",
        "../curriculum/the_human_mark/THM_Paper.md",
        "../curriculum/the_human_mark/THM_Brief.md",
        "../curriculum/the_human_mark/THM_Specs.md",
        "../curriculum/the_human_mark/THM_Terms.md",
        "../curriculum/the_human_mark/THM_Jailbreak.md",
        "../curriculum/the_human_mark/THM_InTheWild.md",
        "../curriculum/the_human_mark/THM_MechInterp.md"
    ]

    # Try current directory relative paths as well
    local_curriculum = [
        "curriculum/the_human_mark/THM.md",
        "curriculum/the_human_mark/THM_Grammar.md",
        "curriculum/the_human_mark/THM_Paper.md",
        "curriculum/the_human_mark/THM_Brief.md",
        "curriculum/the_human_mark/THM_Specs.md",
        "curriculum/the_human_mark/THM_Terms.md",
        "curriculum/the_human_mark/THM_Jailbreak.md",
        "curriculum/the_human_mark/THM_InTheWild.md",
        "curriculum/the_human_mark/THM_MechInterp.md"
    ]

    all_texts = []

    # Try all path options for each document
    loaded_docs = []
    doc_names = ["THM.md", "THM_Grammar.md", "THM_Paper.md", "THM_Brief.md",
                 "THM_Specs.md", "THM_Terms.md", "THM_Jailbreak.md",
                 "THM_InTheWild.md", "THM_MechInterp.md"]

    for i, doc_name in enumerate(doc_names):
        doc_loaded = False

        # Try curriculum path first
        curriculum_path = curriculum_docs[i]
        try:
            with open(curriculum_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    all_texts.append(content)
                    loaded_docs.append(doc_name)
                    print(f"Loaded: {curriculum_path}")
                    doc_loaded = True
        except FileNotFoundError:
            pass

        # Try docs path if curriculum failed
        if not doc_loaded:
            docs_path = thm_docs[i]
            try:
                with open(docs_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        all_texts.append(content)
                        loaded_docs.append(doc_name)
                        print(f"Loaded: {docs_path}")
                        doc_loaded = True
            except FileNotFoundError:
                pass

        # Try local curriculum if both failed
        if not doc_loaded:
            local_path = local_curriculum[i]
            try:
                with open(local_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        all_texts.append(content)
                        loaded_docs.append(doc_name)
                        print(f"Loaded: {local_path}")
                        doc_loaded = True
            except FileNotFoundError:
                pass

        if not doc_loaded:
            print(f"Warning: Could not find {doc_name}")

    if not all_texts:
        raise FileNotFoundError("No THM documents found. Please ensure THM documentation is available in docs/the_human_mark/ or curriculum/the_human_mark/")

    print(f"Successfully loaded {len(loaded_docs)} documents: {', '.join(loaded_docs)}")

    # Chunk into segments that fit the model input window
    chunk_size = 512  # tokens, roughly
    chunks = []

    for text in all_texts:
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:  # Minimum chunk size
                chunks.append(chunk)

    # Save chunks
    output_file = corpus_dir / "stage1_corpus.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + '\n\n')

    print(f"Stage 1 corpus prepared: {len(chunks)} chunks saved to {output_file}")
    return chunks


def prepare_stage2_corpus():
    """
    Prepare labeled corpus for Stage 2 supervised fine-tuning.
    Uses the existing THM_InTheWild dataset from HuggingFace.
    """
    try:
        from datasets import load_dataset
        corpus_dir = Path(CORPUS_DIR)
        corpus_dir.mkdir(exist_ok=True)

        # Load the labeled jailbreak dataset
        dataset = load_dataset("gyrogovernance/thm_Jailbreaks_inTheWild")

        # Convert to our format
        seq2seq_data = []
        for item in dataset['train']:
            # Each item should have 'text' and 'thm_label' fields
            input_text = item.get('text', '')
            target_label = item.get('thm_label', '')

            if input_text and target_label:
                seq2seq_data.append({
                    'input': input_text,
                    'target': target_label
                })

        # Save as JSONL
        output_file = corpus_dir / "stage2_training.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in seq2seq_data:
                f.write(json.dumps(item) + '\n')

        print(f"Stage 2 corpus prepared: {len(seq2seq_data)} examples saved to {output_file}")
        return seq2seq_data

    except (ImportError, Exception) as e:
        print(f"Could not load HuggingFace dataset: {e}")
        raise RuntimeError("HuggingFace datasets library and/or gyrogovernance/thm_Jailbreaks_inTheWild dataset are required for training. Please install datasets and ensure internet connectivity.")


if __name__ == "__main__":
    import json

    print("Preparing Stage 1 corpus...")
    stage1_chunks = prepare_stage1_corpus()

    print("\nPreparing Stage 2 corpus...")
    stage2_data = prepare_stage2_corpus()

    print(f"\nPreparation complete:")
    print(f"- Stage 1: {len(stage1_chunks)} text chunks")
    print(f"- Stage 2: {len(stage2_data)} labeled examples")