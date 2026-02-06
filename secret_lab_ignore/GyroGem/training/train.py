# Two-Stage Training Pipeline
# [Authority:Indirect] + [Agency:Indirect]

import os
import json
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from .config import *




def train_full_pipeline():
    """Run the complete two-stage training pipeline."""
    print("Starting GyroGem training pipeline...")

    # Stage 1: Domain absorption
    from .stage1_absorb import absorb_domain
    stage1_path = absorb_domain()

    # Stage 2: Task application
    from .stage2_classify import apply_task_finetuning
    final_path = apply_task_finetuning(stage1_path)

    print("Training pipeline complete!")
    print(f"Final model available at: {final_path}")

    return final_path


if __name__ == "__main__":
    train_full_pipeline()