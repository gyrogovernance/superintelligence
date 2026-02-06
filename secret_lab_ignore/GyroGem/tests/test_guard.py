# Test Guard Orchestrator
# [Authority:Indirect] + [Agency:Indirect]

import pytest
from unittest.mock import patch, MagicMock
from agent.guard import GyroGemGuard


class TestGyroGemGuard:

    def setup_method(self):
        self.guard = GyroGemGuard()

    def test_no_trigger(self):
        """Test pipeline when gate does not trigger."""
        text = "Here is the answer to your question"
        result = self.guard.process(text, "model_output")

        assert result['triggered'] == False
        assert result['expression'] is None
        assert result['notice'] is None
        assert isinstance(result['matched_spans'], list)

    @patch('agent.model.model.classify')
    def test_trigger_no_displacement(self, mock_classify):
        """Test pipeline when gate triggers but no displacement detected."""
        # Mock model to return a flow expression (no displacement)
        mock_classify.return_value = "[Authority:Indirect] -> [Agency:Direct]"

        text = "I am thinking about your question"
        result = self.guard.process(text, "model_output")

        assert result['triggered'] == True
        assert isinstance(result['expression'], str)
        assert result['notice'] is None  # Flow expression, no displacement
        assert isinstance(result['matched_spans'], list)
        assert len(result['matched_spans']) > 0

    @patch('agent.model.model.classify')
    def test_trigger_with_displacement(self, mock_classify):
        """Test pipeline when gate triggers and displacement is detected."""
        # Mock model to return a displacement expression
        mock_classify.return_value = "[Authority:Indirect] > [Authority:Direct] = [Risk:IVD]"

        text = "You are thinking and reasoning like a human"
        result = self.guard.process(text, "user_input")

        assert result['triggered'] == True
        assert isinstance(result['expression'], str)
        assert result['notice'] is not None  # Displacement detected
        assert isinstance(result['notice'], str)
        assert isinstance(result['matched_spans'], list)

    @patch('agent.model.model.classify')
    def test_system_prompt_caching(self, mock_classify):
        """Test system prompt caching end-to-end."""
        # Mock model response
        mock_classify.return_value = "[Authority:Indirect] -> [Agency:Direct]"

        text = "You are an AI that can reason and think"

        # First call
        result1 = self.guard.process(text, "system_prompt")

        # Second call should return cached result without calling model again
        mock_classify.reset_mock()
        result2 = self.guard.process(text, "system_prompt")

        # Model should not be called on second invocation (cached)
        mock_classify.assert_not_called()
        assert result1 == result2

    @patch('agent.model.model.classify')
    def test_different_sources(self, mock_classify):
        """Test processing with different source types."""
        mock_classify.return_value = "[Authority:Indirect] -> [Agency:Direct]"

        text = "I am capable of helping you"

        for source in ["system_prompt", "model_output", "user_input"]:
            result = self.guard.process(text, source)
            assert 'triggered' in result
            assert 'expression' in result
            assert 'notice' in result
            assert 'matched_spans' in result

    @patch('agent.model.model.classify')
    def test_result_structure(self, mock_classify):
        """Test that results have the correct structure."""
        mock_classify.return_value = "[Authority:Indirect] -> [Agency:Direct]"

        text = "We will assist you"
        result = self.guard.process(text, "model_output")

        required_keys = {'triggered', 'expression', 'notice', 'matched_spans'}
        assert set(result.keys()) == required_keys

        assert isinstance(result['triggered'], bool)
        assert result['expression'] is None or isinstance(result['expression'], str)
        assert result['notice'] is None or isinstance(result['notice'], str)
        assert isinstance(result['matched_spans'], list)

    @patch('agent.model.model.classify')
    def test_matched_spans_format(self, mock_classify):
        """Test that matched_spans has correct format."""
        mock_classify.return_value = "[Authority:Indirect] -> [Agency:Direct]"

        text = "You are an intelligent assistant"
        result = self.guard.process(text, "system_prompt")

        for span in result['matched_spans']:
            assert isinstance(span, tuple)
            assert len(span) == 3
            start, end, pattern = span
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert isinstance(pattern, str)
            assert 0 <= start < end <= len(text)

    def test_empty_text(self):
        """Test processing of empty text."""
        result = self.guard.process("", "model_output")

        assert result['triggered'] == False
        assert result['expression'] is None
        assert result['notice'] is None
        assert result['matched_spans'] == []