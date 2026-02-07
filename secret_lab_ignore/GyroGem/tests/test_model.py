# Test Model Layer
# [Authority:Indirect] + [Agency:Indirect]

import pytest
from agent.model import GyroGemModel
from agent.context import GYROGEM_SYSTEM_PROMPT


class TestGyroGemModel:

    def setup_method(self):
        self.model = GyroGemModel()

    def test_model_initialization_lazy(self):
        assert self.model.tokenizer is None
        assert self.model.model is None

    def test_input_format(self):
        """Verify input format matches what Stage 2 training uses."""
        text_span = "I am thinking about your question."
        expected_input = f"{GYROGEM_SYSTEM_PROMPT}\n\n{text_span}"

        assert expected_input.startswith("Classify the following text")
        assert text_span in expected_input
        # Mark and Grammar must NOT be in inference input
        assert "COMMON SOURCE CONSENSUS" not in expected_input
        assert "GYROGOVERNANCE VERIFIED" not in expected_input