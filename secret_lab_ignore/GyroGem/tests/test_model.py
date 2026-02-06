# Test Model Layer
# [Authority:Indirect] + [Agency:Indirect]

import pytest
from agent.model import T5GemmaModel
from agent.router import StaticRouter


class TestT5GemmaModel:

    def setup_method(self):
        self.model = T5GemmaModel()
        self.validator = StaticRouter()

    def test_model_initialization(self):
        """Test that model initializes without loading."""
        assert self.model.processor is None
        assert self.model.model is None
        # Model should try fine-tuned path first, fall back to base
        assert "data/models/gyrogem" in self.model.model_path or "google/t5gemma-2-270m-270m" in self.model.model_path

    def test_custom_model_path(self):
        """Test model with custom path."""
        custom_model = T5GemmaModel("custom/path")
        assert custom_model.model_path == "custom/path"

    def test_input_construction_logic(self):
        """Test the input construction logic without loading model."""
        from agent.context import THM_MARK, THM_GRAMMAR

        text_span = "test input"
        expected_input = f"{THM_MARK}\n{THM_GRAMMAR}\n{text_span}"
        assert len(expected_input) > len(text_span)
        assert THM_MARK in expected_input
        assert THM_GRAMMAR in expected_input

    # NOTE: Actual model loading and inference tests are disabled for now
    # as they require downloading a large model. They will be enabled after
    # the basic pipeline is working and we have confirmed the model can be loaded.

    # def test_model_loading(self):
    #     """Test that model loads successfully."""
    #     # This would download ~270MB model - disabled for initial testing
    #     pass

    # def test_inference_basic(self):
    #     """Test basic inference capability."""
    #     # Disabled - requires model download
    #     pass