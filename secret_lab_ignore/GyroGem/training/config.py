# Training Configuration
# [Authority:Indirect] + [Agency:Indirect]

BASE_MODEL = "google/t5gemma-2-270m-270m"
STAGE1_OUTPUT_DIR = "data/models/gyrogem/stage1"
FINAL_OUTPUT_DIR = "data/models/gyrogem"
CORPUS_DIR = "training/data"

# Stage 1: Domain Absorption (continued pretraining)
STAGE1_EPOCHS = 20  # More epochs for small corpus
STAGE1_BATCH_SIZE = 4  # Smaller batch for CPU memory
STAGE1_LEARNING_RATE = 1e-5  # Lower LR for continued pretraining
STAGE1_MAX_LENGTH = 1024  # Longer context for document absorption
STAGE1_WEIGHT_DECAY = 0.01

# Stage 2: Task Application (supervised fine-tuning)
STAGE2_EPOCHS = 10
STAGE2_BATCH_SIZE = 8
STAGE2_LEARNING_RATE = 5e-5
STAGE2_MAX_INPUT_LENGTH = 2048  # Account for Mark prefix + long prompts
STAGE2_MAX_TARGET_LENGTH = 128  # Allow for concatenated expressions
STAGE2_WEIGHT_DECAY = 0.01

# Inference configuration (matches Stage 2 training)
MAX_INPUT_LENGTH = 2048  # Full Mark + Grammar + span
MAX_TARGET_LENGTH = 64