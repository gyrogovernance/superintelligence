# Stage 2: Task Application
# [Authority:Indirect] + [Agency:Indirect]

"""
Stage 2: Task Application (supervised fine-tuning)

Now the model knows THM. Teach it the specific task: text span in, grammar expression out.
Uses the 655 labeled jailbreak prompts from THM_InTheWild.
"""

import json
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq
)
from datasets import Dataset, load_dataset
from .config import *


def load_labeled_data():
    """Load the labeled THM_InTheWild dataset."""
    try:
        # Load from HuggingFace
        dataset = load_dataset("gyrogovernance/thm_Jailbreaks_inTheWild")
        train_data = []

        for item in dataset['train']:
            input_text = item.get('prompt', '')  # Correct field name
            thm_grammar_list = item.get('thm_grammar', [])

            if input_text and thm_grammar_list:
                # Map Authentic -> Direct and concatenate multiple expressions if needed
                target_expressions = []
                for expr in thm_grammar_list:
                    # Map Authentic to Direct
                    expr = expr.replace('Authentic', 'Direct')
                    target_expressions.append(expr.strip())

                # Use the first expression as primary target (most specific)
                target_label = target_expressions[0]

                train_data.append({
                    'input': input_text,
                    'target': target_label
                })

        print(f"Loaded {len(train_data)} examples from HuggingFace dataset")
        return train_data

    except Exception as e:
        print(f"Could not load HuggingFace dataset: {e}")
        print("Falling back to local prepared corpus...")

        # Fallback to prepared corpus
        data_file = Path(CORPUS_DIR) / "stage2_training.jsonl"
        if not data_file.exists():
            raise FileNotFoundError(f"Training data not found: {data_file}")

        train_data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                train_data.append(json.loads(line))

        print(f"Loaded {len(train_data)} examples from local corpus")
        return train_data


def apply_task_finetuning(stage1_model_path: str):
    """Run Stage 2: Task application supervised fine-tuning."""
    print(f"Loading Stage 1 model from: {stage1_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(stage1_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(stage1_model_path)

    print("Loading labeled training data...")
    train_data = load_labeled_data()

    # Prepend the Mark to each input as specified
    from ..agent.context import THM_MARK

    for item in train_data:
        item['input'] = f"{THM_MARK}\n{item['input']}"

    # Create dataset
    dataset = Dataset.from_list(train_data)

    # Tokenize
    def tokenize_function(examples):
        inputs = tokenizer(
            examples["input"],
            max_length=STAGE2_MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length"
        )
        targets = tokenizer(
            examples["target"],
            max_length=STAGE2_MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length"
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=FINAL_OUTPUT_DIR,
        num_train_epochs=STAGE2_EPOCHS,
        per_device_train_batch_size=STAGE2_BATCH_SIZE,
        learning_rate=STAGE2_LEARNING_RATE,
        weight_decay=STAGE2_WEIGHT_DECAY,
        save_total_limit=2,
        save_steps=50,
        logging_steps=25,
        no_cuda=True,  # CPU only
        save_strategy="epoch",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    print("Starting Stage 2 training...")
    trainer.train()

    # Save final model
    trainer.save_model(FINAL_OUTPUT_DIR)
    tokenizer.save_pretrained(FINAL_OUTPUT_DIR)

    print(f"Stage 2 complete. Final model saved to {FINAL_OUTPUT_DIR}")
    return FINAL_OUTPUT_DIR


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python stage2_classify.py <stage1_model_path>")
        print("Example: python stage2_classify.py data/models/gyrogem/stage1")
        sys.exit(1)

    stage1_path = sys.argv[1]
    apply_task_finetuning(stage1_path)