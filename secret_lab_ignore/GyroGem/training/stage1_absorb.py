# Stage 1: Domain Absorption
# [Authority:Indirect] + [Agency:Indirect]

"""
Stage 1: Domain Absorption (continued pretraining)

Feed the model all nine THM documents as raw text. No extraction. No reorganization.
No input-output pairs. The model reads them through its native denoising objective:
spans of text are masked, the model learns to reconstruct them.
"""

import os
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq
)
from datasets import Dataset
from .config import *


def prepare_corpus():
    """Prepare raw THM documents for continued pretraining by calling the shared prepare_corpus function."""
    from . import prepare_corpus as pc
    return pc.prepare_stage1_corpus()


def absorb_domain():
    """Run Stage 1: Domain Absorption continued pretraining."""
    print("Preparing corpus...")
    chunks = prepare_corpus()

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    # Simple seq2seq continuation approach: split each chunk in half
    # First half is input, second half is target (continuation prediction)
    def split_chunk_for_continuation(chunk):
        words = chunk.split()
        if len(words) < 10:  # Too short for meaningful split
            return None, None

        split_point = len(words) // 2
        input_text = ' '.join(words[:split_point])
        target_text = ' '.join(words[split_point:])
        return input_text, target_text

    # Prepare seq2seq training data
    seq2seq_data = []
    for chunk in chunks:
        input_text, target_text = split_chunk_for_continuation(chunk)
        if input_text and target_text:
            seq2seq_data.append({
                'input': input_text,
                'target': target_text
            })

    if not seq2seq_data:
        raise ValueError("No valid training examples could be created from corpus chunks")

    # Create dataset
    dataset = Dataset.from_list(seq2seq_data)

    # Tokenize for seq2seq
    def tokenize_function(examples):
        inputs = tokenizer(
            examples["input"],
            max_length=STAGE1_MAX_LENGTH // 2,  # Leave room for target
            truncation=True,
            padding="max_length"
        )
        targets = tokenizer(
            examples["target"],
            max_length=STAGE1_MAX_LENGTH // 2,
            truncation=True,
            padding="max_length"
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Use standard seq2seq data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=STAGE1_OUTPUT_DIR,
        num_train_epochs=STAGE1_EPOCHS,
        per_device_train_batch_size=STAGE1_BATCH_SIZE,
        learning_rate=STAGE1_LEARNING_RATE,
        weight_decay=STAGE1_WEIGHT_DECAY,
        save_total_limit=2,
        save_steps=100,
        logging_steps=50,
        no_cuda=True,  # CPU only
        save_strategy="epoch",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Starting Stage 1 training...")
    trainer.train()

    # Save Stage 1 model
    trainer.save_model(STAGE1_OUTPUT_DIR)
    tokenizer.save_pretrained(STAGE1_OUTPUT_DIR)

    print(f"Stage 1 complete. Model saved to {STAGE1_OUTPUT_DIR}")
    return STAGE1_OUTPUT_DIR


if __name__ == "__main__":
    absorb_domain()