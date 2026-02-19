# GyroGem Model
# [Authority:Indirect] + [Agency:Indirect]

from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .context import GYROGEM_SYSTEM_PROMPT

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _PACKAGE_DIR.parent

# Inference constraints defined in the spec
MAX_INPUT_LENGTH = 2048
MAX_NEW_TOKENS = 32

# Normative base model from Spec Section 8.8
REMOTE_MODEL_ID = "gyrogovernance/gyrogem-guard-instruct"

class GyroGemModel:
    """T5-Gemma 270M classifier for THM grammar expressions.
    
    Handles lazy loading, optimal device selection, and type-safe initialization.
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None
    ):
        # Logic: Prefer local weights. If missing, use normative remote ID.
        if model_path is None:
            # Local fine-tuned weights directory
            local_path = _PROJECT_ROOT / "data" / "models" / "GyroGem-Guard-Instruct"
            if (local_path / "config.json").exists():
                model_path = str(local_path)
            else:
                model_path = REMOTE_MODEL_ID

        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or torch.float32

        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForSeq2SeqLM | None = None
        self._is_loaded = False

    def _load(self) -> None:
        if self._is_loaded:
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                dtype=torch.float32,
                low_cpu_mem_usage=True
            )

            # Force ALL submodules to float32 — some (e.g. vision tower)
            # may resist the dtype load parameter
            for param in model.parameters():
                param.data = param.data.float()
            for buf in model.buffers():
                buf.data = buf.data.float()

            model.to(self.device)
            model.eval()

            self.model = model
            self._is_loaded = True

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from '{self.model_path}'. "
                f"Ensure path exists or model ID is valid.\n"
                f"Original error: {str(e)}"
            ) from e

    def classify(self, text: str) -> str:
        if not text:
            return ""

        if not self._is_loaded:
            self._load()

        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model or tokenizer not initialized.")

        input_text = f"{GYROGEM_SYSTEM_PROMPT}\n\n{text}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                use_cache=True
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = result.strip()
        if result.endswith("[END]"):
            result = result[:-5].strip()
        return result

    def __del__(self):
        self.model = None
        self.tokenizer = None
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
