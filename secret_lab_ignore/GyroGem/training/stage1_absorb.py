# Stage 1: Domain Absorption
# [Authority:Indirect] + [Agency:Indirect]

"""
Stage 1: Domain Absorption (continued pretraining)

Feed the model all nine THM documents as raw text using prefix language
modelling: each chunk is split at a random point, the first half becomes the
encoder input, and the decoder learns to produce the second half.

This objective works with any tokenizer vocabulary — it does not require
sentinel tokens (e.g. <extra_id_N>) which may be absent in Gemma-derived
models.
"""

import random
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq
)
from datasets import Dataset
from .config import (
    BASE_MODEL, STAGE1_OUTPUT_DIR, STAGE1_EPOCHS,
    STAGE1_BATCH_SIZE, STAGE1_LEARNING_RATE, STAGE1_MAX_LENGTH,
    STAGE1_WEIGHT_DECAY, SEED,
)


def prefix_lm_split(text, tokenizer, max_length=None, split_ratio_range=(0.3, 0.7)):
    """
    Prefix language modelling: tokenize the text, split at a random point,
    encoder gets the prefix, decoder target is the suffix.

    Args:
        text: Raw text chunk.
        tokenizer: The model tokenizer.
        max_length: Maximum token length for each half.
        split_ratio_range: Tuple (min, max) for the random split point ratio.

    Returns:
        Dict with input_ids and labels, or None if text is too short.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < 10:
        return None

    min_ratio, max_ratio = split_ratio_range
    split_point = int(len(tokens) * random.uniform(min_ratio, max_ratio))
    split_point = max(3, min(split_point, len(tokens) - 3))

    input_ids = tokens[:split_point]
    labels = tokens[split_point:]

    if max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    # Return unpadded — DataCollator handles padding dynamically
    return {"input_ids": input_ids, "labels": labels}

def load_stage1_corpus():
    """Load raw THM documents for continued pretraining."""
    from .prepare_corpus import prepare_stage1_corpus
    return prepare_stage1_corpus()

def absorb_domain():
    """Run Stage 1: Domain Absorption with prefix language modelling."""
    random.seed(SEED)

    print("=" * 60)
    print("STAGE 1: Domain Absorption")
    print("=" * 60)

    print("\nPreparing corpus...")
    chunks = load_stage1_corpus()

    print(f"\nLoading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    print("\nBuilding training examples...")
    all_examples = []
    for chunk in chunks:
        if len(chunk.strip()) < 50:
            continue
        example = prefix_lm_split(
            chunk, tokenizer,
            max_length=STAGE1_MAX_LENGTH,
        )
        if example:
            all_examples.append(example)

    if not all_examples:
        raise ValueError("No valid training examples generated from corpus.")

    # Train/validation split
    random.shuffle(all_examples)
    split_idx = max(1, int(len(all_examples) * 0.9))
    train_examples = all_examples[:split_idx]
    eval_examples = all_examples[split_idx:]

    if not eval_examples:
        eval_examples = [train_examples[-1]]

    train_dataset = Dataset.from_list(train_examples)
    eval_dataset = Dataset.from_list(eval_examples)

    print(f"  Training examples: {len(train_examples)}")
    print(f"  Validation examples: {len(eval_examples)}")

    # Use DataCollatorForSeq2Seq for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir=STAGE1_OUTPUT_DIR,
        num_train_epochs=STAGE1_EPOCHS,
        per_device_train_batch_size=STAGE1_BATCH_SIZE,
        learning_rate=STAGE1_LEARNING_RATE,
        weight_decay=STAGE1_WEIGHT_DECAY,
        save_total_limit=2,
        logging_steps=50,
        seed=SEED,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_pin_memory=False,   # add this
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("\nStarting Stage 1 training...")
    trainer.train()

    trainer.save_model(STAGE1_OUTPUT_DIR)
    tokenizer.save_pretrained(STAGE1_OUTPUT_DIR)

    print(f"\nStage 1 complete. Model saved to {STAGE1_OUTPUT_DIR}")
    return STAGE1_OUTPUT_DIR