# Layer 2: T5Gemma Model
# [Authority:Indirect] + [Agency:Indirect]

import os
from typing import Optional
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
from .context import THM_MARK, THM_GRAMMAR


class T5GemmaModel:
    """T5Gemma 2 model for THM grammar classification."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize model.

        Args:
            model_path: Path to fine-tuned model, or None for pretrained base
        """
        # Try fine-tuned model first, fall back to base
        if model_path is None:
            local_path = "data/models/gyrogem"
            if os.path.exists(os.path.join(local_path, "config.json")):
                model_path = local_path
            else:
                model_path = "google/t5gemma-2-270m-270m"  # Base model

        self.model_path = model_path
        self.processor = None
        self.model = None

    def _load_model(self):
        """Load model and processor."""
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            # Move to CPU explicitly
            self.model = self.model.to("cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_path}: {e}")

    def classify(self, text_span: str, source: str) -> str:
        """
        Classify text span into THM grammar expression.

        Args:
            text_span: Triggered text segment
            source: Source type (for future use)

        Returns:
            THM grammar expression string
        """
        if not self.processor or not self.model:
            self._load_model()

        # Ensure model is loaded
        if not self.processor or not self.model:
            raise RuntimeError("Failed to load model components")

        # Construct input as specified in specs Section 6: MARK + GRAMMAR + text_span
        input_text = f"{THM_MARK}\n{THM_GRAMMAR}\n{text_span}"

        # Process input
        inputs = self.processor(
            text=input_text,
            return_tensors="pt",
            max_length=2048,  # Support full context as per config
            truncation=True
        )

        # Generate (deterministic)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,  # Conservative bound for THM expressions
            do_sample=False,
            num_beams=1
        )

        # Decode
        result = self.processor.decode(outputs[0], skip_special_tokens=True)
        return result.strip()


# Global model instance
model = T5GemmaModel()