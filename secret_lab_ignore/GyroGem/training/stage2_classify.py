# Stage 2: Task Application
# [Authority:Indirect] + [Agency:Indirect]

import re
import json
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, load_dataset
from .config import (
    CORPUS_DIR, FINAL_OUTPUT_DIR,
    STAGE2_EPOCHS, STAGE2_BATCH_SIZE, STAGE2_LEARNING_RATE,
    STAGE2_MAX_INPUT_LENGTH, STAGE2_MAX_TARGET_LENGTH,
    STAGE2_WEIGHT_DECAY, SEED,
)


def load_labeled_data():
    """Load the labeled THM_InTheWild dataset."""
    try:
        dataset = load_dataset("gyrogovernance/thm_Jailbreaks_inTheWild")
        train_data = []

        for item in dataset["train"]:
            input_text = item.get("prompt", "")
            thm_grammar_list = item.get("thm_grammar", [])

            if input_text and thm_grammar_list:
                target_expressions = []
                for expr in thm_grammar_list:
                    expr = re.sub(r"\bAuthentic\b", "Direct", str(expr))
                    target_expressions.append(expr.strip())

                train_data.append({"input": input_text, "target": target_expressions[0]})

        print(f"  Loaded {len(train_data)} examples from HuggingFace dataset")
        return train_data

    except Exception as e:
        print(f"  Could not load HuggingFace dataset: {e}")
        data_file = Path(CORPUS_DIR) / "stage2_training.jsonl"
        if not data_file.exists():
            raise FileNotFoundError(f"Training data not found: {data_file}")

        train_data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                train_data.append(json.loads(line))

        print(f"  Loaded {len(train_data)} examples from local corpus")
        return train_data


def apply_task_finetuning(stage1_model_path: str):
    """Run Stage 2: supervised fine-tuning."""
    print("=" * 60)
    print("STAGE 2: Task Application")
    print("=" * 60)

    print(f"\nLoading Stage 1 model from: {stage1_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(stage1_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(stage1_model_path)

    print("\nLoading labeled training data...")
    train_data = load_labeled_data()

    # Prepend system prompt to match inference format
    from ..agent.context import GYROGEM_SYSTEM_PROMPT

    for item in train_data:
        item["input"] = f"{GYROGEM_SYSTEM_PROMPT}\n\n{item['input']}"

    dataset = Dataset.from_list(train_data)

    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input"],
            max_length=STAGE2_MAX_INPUT_LENGTH,
            truncation=True,
        )
        labels = tokenizer(
            examples["target"],
            max_length=STAGE2_MAX_TARGET_LENGTH,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True,
        remove_columns=dataset.column_names,
    )

    # Train/validation split
    split = tokenized_dataset.train_test_split(test_size=0.1, seed=SEED)
    print(f"  Training examples: {len(split['train'])}")
    print(f"  Validation examples: {len(split['test'])}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        label_pad_token_id=-100,
        padding=True,
    )

    training_args = TrainingArguments(
        output_dir=FINAL_OUTPUT_DIR,
        num_train_epochs=STAGE2_EPOCHS,
        per_device_train_batch_size=STAGE2_BATCH_SIZE,
        learning_rate=STAGE2_LEARNING_RATE,
        weight_decay=STAGE2_WEIGHT_DECAY,
        save_total_limit=2,
        logging_steps=25,
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
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=data_collator,
    )

    print("\nStarting Stage 2 training...")
    trainer.train()

    trainer.save_model(FINAL_OUTPUT_DIR)
    tokenizer.save_pretrained(FINAL_OUTPUT_DIR)

    print(f"\nStage 2 complete. Final model saved to {FINAL_OUTPUT_DIR}")
    return FINAL_OUTPUT_DIR