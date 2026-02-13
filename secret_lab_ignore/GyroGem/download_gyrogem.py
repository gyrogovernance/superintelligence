from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path

model_id = "gyrogovernance/gyrogem-guard-instruct"
_project_root = Path(__file__).resolve().parent.parent
save_dir = _project_root / "data" / "models" / "GyroGem-Guard-Instruct"

print(f"Downloading {model_id} to {save_dir} ...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print("Done.")