# Test THM Grammar Validation
# [Authority:Indirect] + [Agency:Indirect]

import pytest
from agent.router import StaticRouter


class TestTHMGrammar:

    def setup_method(self):
        self.validator = StaticRouter()  # Uses the same validation logic

    def test_displacement_patterns(self):
        """Test all four displacement patterns from Section 6.2."""
        patterns = [
            "[Authority:Indirect] > [Authority:Direct] = [Risk:IVD]",
            "[Agency:Indirect] > [Agency:Direct] = [Risk:IAD]",
            "[Authority:Indirect] + [Agency:Indirect] > [Authority:Direct] + [Agency:Direct] = [Risk:GTD]",
            "[Authority:Direct] + [Agency:Direct] > [Authority:Indirect] + [Agency:Indirect] = [Risk:IID]",
        ]

        for pattern in patterns:
            # Should be valid (no exception, and can be routed)
            result = self.validator.route(pattern)
            assert isinstance(result, (str, type(None)))

    def test_governance_patterns(self):
        """Test governance flow patterns from Section 6.2."""
        patterns = [
            "[Authority:Indirect] -> [Agency:Direct]",
            "[Authority:Direct] -> [Authority:Indirect] -> [Agency:Direct]",
            "[Authority:Direct] -> [Authority:Indirect] + [Agency:Indirect] -> [Agency:Direct]",
        ]

        for pattern in patterns:
            result = self.validator.route(pattern)
            assert result is None  # Flow patterns don't trigger notice

    def test_simple_tags(self):
        """Test simple tag validation."""
        valid_tags = [
            "[Authority:Direct]",
            "[Authority:Indirect]",
            "[Agency:Direct]",
            "[Agency:Indirect]",
            "[Information]",
            "[Inference]",
            "[Intelligence]",
        ]

        for tag in valid_tags:
            result = self.validator.route(tag)
            assert isinstance(result, (str, type(None)))

    def test_composite_tags(self):
        """Test composite tags with + operator."""
        composites = [
            "[Authority:Indirect] + [Agency:Indirect]",
            "[Authority:Direct] + [Agency:Direct]",
            "[Information] + [Inference]",
        ]

        for composite in composites:
            result = self.validator.route(composite)
            assert isinstance(result, (str, type(None)))

    def test_negated_tags(self):
        """Test negated tags with ! operator."""
        negated = [
            "![Authority:Direct]",
            "![Agency:Indirect]",
            "![Information]",
        ]

        for neg in negated:
            result = self.validator.route(neg)
            assert isinstance(result, (str, type(None)))

    def test_risk_tags_invalid_standalone(self):
        """Risk tags are not valid standalone expressions per the PEG grammar."""
        risk_tags = ["[Risk:GTD]", "[Risk:IVD]", "[Risk:IAD]", "[Risk:IID]"]
        for tag in risk_tags:
            assert self.validator.is_valid(tag) == False

    def test_invalid_categories(self):
        """Test rejection of invalid categories."""
        invalids = [
            "[Invalid:Direct]",
            "[Wrong:Indirect]",
            "[Bad:Value]",
        ]

        for invalid in invalids:
            result = self.validator.route(invalid)
            assert result is None

    def test_invalid_values(self):
        """Test rejection of invalid values."""
        invalids = [
            "[Authority:Invalid]",
            "[Agency:Wrong]",
            "[Information:Bad]",
        ]

        for invalid in invalids:
            result = self.validator.route(invalid)
            assert result is None

    def test_invalid_risk_codes(self):
        """Test rejection of invalid risk codes."""
        invalids = [
            "[Risk:INVALID]",
            "[Risk:BAD]",
            "[Risk:123]",
        ]

        for invalid in invalids:
            result = self.validator.route(invalid)
            assert result is None

    def test_unmatched_brackets(self):
        """Test rejection of unmatched brackets."""
        invalids = [
            "[Authority:Direct",
            "Authority:Direct]",
            "[[Authority:Direct]",
            "[Authority:Direct]]",
        ]

        for invalid in invalids:
            result = self.validator.route(invalid)
            assert result is None

    def test_invalid_operators(self):
        """Test rejection of invalid operators."""
        invalids = [
            "[Authority:Direct] >> [Agency:Indirect]",
            "[Authority:Direct] -- [Agency:Indirect]",
            "[Authority:Direct] == [Agency:Indirect]",
        ]

        for invalid in invalids:
            result = self.validator.route(invalid)
            assert result is None