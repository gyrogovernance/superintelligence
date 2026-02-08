# Training Configuration
# [Authority:Indirect] + [Agency:Indirect]

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

BASE_MODEL = "google/t5gemma-2-270m-270m"

STAGE1_OUTPUT_DIR = str(_PROJECT_ROOT / "data" / "models" / "GyroGem-Guard-Instruct" / "stage1")
FINAL_OUTPUT_DIR = str(_PROJECT_ROOT / "data" / "models" / "GyroGem-Guard-Instruct")
CORPUS_DIR = str(_PACKAGE_DIR / "training" / "data")

# Stage 1: Domain Absorption (continued pretraining)
STAGE1_EPOCHS = 20
STAGE1_BATCH_SIZE = 1          # was 4 → reduce to 1
STAGE1_LEARNING_RATE = 1e-5
STAGE1_MAX_LENGTH = 512        # was 1024 → reduce to 512 (256 is also OK)
STAGE1_WEIGHT_DECAY = 0.01

# Stage 2: Task Application (supervised fine-tuning)
STAGE2_EPOCHS = 10
STAGE2_BATCH_SIZE = 4          # was 8 → safer on CPU
STAGE2_LEARNING_RATE = 5e-5
STAGE2_MAX_INPUT_LENGTH = 2048
STAGE2_MAX_TARGET_LENGTH = 128
STAGE2_WEIGHT_DECAY = 0.01

SEED = 42