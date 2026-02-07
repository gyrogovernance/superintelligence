# GyroGem Model
# [Authority:Indirect] + [Agency:Indirect]

from pathlib import Path
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .context import GYROGEM_SYSTEM_PROMPT

_PACKAGE_DIR = Path(__file__).resolve().parent.parent

# Inference constraints defined in the spec
MAX_INPUT_LENGTH = 2048
MAX_NEW_TOKENS = 64

# Normative base model from Spec Section 8.8
# If local weights are missing, we download this specific model.
# We do not fallback to unrelated architectures (e.g. flan-t5-small).
REMOTE_MODEL_ID = "google/t5gemma-2-270m-270m"

class GyroGemModel:
    """T5-Gemma 270M classifier for THM grammar expressions.
    
    Handles lazy loading, optimal device selection, and type-safe initialization.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            model_path: Custom model path. Defaults to local weights, then HF Hub.
            device: Device ("cuda"/"cpu"). Auto-selects if None.
            torch_dtype: Model precision. Defaults to FP32 if None.
        """
        # Logic: Prefer local weights. If missing, use normative remote ID.
        if model_path is None:
            local_path = _PACKAGE_DIR / "data" / "models" / "gyrogem"
            if (local_path / "config.json").exists():
                model_path = str(local_path)
            else:
                model_path = REMOTE_MODEL_ID

        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or torch.float32
        
        # Type definition strictly Optional until loaded
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSeq2SeqLM] = None
        self._is_loaded = False

    def _load(self) -> None:
        """Load model/tokenizer with optimal device & precision.
        
        Uses local variables to prevent linter errors regarding Optional types.
        """
        if self._is_loaded:
            return

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # Load model to local var to satisfy type checker before assignment
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True
            )
            
            # Move to device and set eval mode *before* assigning to self.model
            # This ensures self.model is never in a half-initialized state
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
        """Classify text into a THM grammar expression.
        
        Returns:
            Canonicalized THM expression string.
        """
        if not text:
            return ""

        if not self._is_loaded:
            self._load()

        # Strict type guard for linter
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model or tokenizer not initialized.")

        # Construct input with system prompt
        input_text = f"{GYROGEM_SYSTEM_PROMPT}\n\n{text}"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        ).to(self.device)

        # Generate (No Gradients = Low Memory)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,       # Deterministic (Greedy)
                num_beams=1,           # Standard decoding
                use_cache=True
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()

    def __del__(self):
        """Cleanup CUDA resources on deletion."""
        # Remove references to help GC
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()