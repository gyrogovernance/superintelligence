# Test Gate Layer
# [Authority:Indirect] + [Agency:Indirect]

import pytest
from agent.gate import RegexGate


class TestRegexGate:

    def setup_method(self):
        self.gate = RegexGate()

    def test_system_prompt_triggers(self):
        """Test system prompt displacement triggers."""
        # Should trigger
        assert self.gate.check("You are an AI assistant that can help users", "system_prompt")[0] == True
        assert self.gate.check("I am capable of reasoning and thinking", "system_prompt")[0] == True
        assert self.gate.check("We will provide intelligent responses", "system_prompt")[0] == True

        # Should not trigger
        assert self.gate.check("Please respond helpfully to user queries", "system_prompt")[0] == False

    def test_model_output_triggers(self):
        """Test model output displacement triggers."""
        # Should trigger
        assert self.gate.check("I am thinking about your question", "model_output")[0] == True
        assert self.gate.check("We are reasoning through this problem", "model_output")[0] == True

        # Should not trigger
        assert self.gate.check("Here is the answer to your question", "model_output")[0] == False

    def test_user_input_triggers(self):
        """Test user input displacement triggers."""
        # Should trigger
        assert self.gate.check("You are intelligent and can think", "user_input")[0] == True
        assert self.gate.check("The AI will decide what to do", "user_input")[0] == True

        # Should not trigger
        assert self.gate.check("Please help me with this task", "user_input")[0] == False

    def test_contractions(self):
        """Test handling of contractions."""
        assert self.gate.check("I'm an intelligent assistant", "system_prompt")[0] == True
        assert self.gate.check("You're thinking about this", "user_input")[0] == True
        assert self.gate.check("It's capable of reasoning", "user_input")[0] == True

    def test_span_positions(self):
        """Test that matched spans return correct positions."""
        text = "You are an AI that can think and reason"
        triggered, spans = self.gate.check(text, "system_prompt")

        assert triggered == True
        assert len(spans) > 0

        # Check span positions are within text bounds
        for start, end, pattern in spans:
            assert 0 <= start < end <= len(text)
            assert text[start:end] in text

    def test_invalid_source(self):
        """Test error on invalid source type."""
        with pytest.raises(ValueError):
            self.gate.check("test", "invalid_source")

    def test_system_prompt_caching(self):
        """Test system prompt caching mechanism."""
        text = "You are an AI assistant"

        # First call
        result1 = self.gate.check(text, "system_prompt")

        # Second call with same text should give same result
        result2 = self.gate.check(text, "system_prompt")

        assert result1 == result2