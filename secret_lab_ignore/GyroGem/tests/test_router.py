# Test Router Layer
# [Authority:Indirect] + [Agency:Indirect]

import pytest
from agent.router import StaticRouter


class TestStaticRouter:

    def setup_method(self):
        self.router = StaticRouter()

    def test_displacement_with_risk(self):
        """Test routing when displacement and risk are present."""
        # Should return notice
        expression = "[Authority:Indirect] > [Authority:Direct] = [Risk:IVD]"
        notice = self.router.route(expression)
        assert notice is not None
        assert "[Authority:Indirect]" in notice
        assert "[Agency:Direct]" in notice

    def test_flow_without_risk(self):
        """Test routing for flow expressions without risk."""
        # Should return None
        expression = "[Authority:Indirect] -> [Agency:Direct]"
        notice = self.router.route(expression)
        assert notice is None

    def test_tags_without_operators(self):
        """Test routing for simple tags without operators."""
        # Should return None
        expression = "[Authority:Indirect]"
        notice = self.router.route(expression)
        assert notice is None

    def test_malformed_expressions(self):
        """Test rejection of malformed expressions."""
        malformed = [
            "",  # Empty
            "[",  # Unmatched bracket
            "Invalid text",  # No brackets
            "[Authority:Indirect > [Authority:Direct",  # Missing closing bracket
        ]

        for expr in malformed:
            notice = self.router.route(expr)
            assert notice is None

    def test_notice_content(self):
        """Test that notice content is exactly as specified."""
        expression = "[Agency:Indirect] > [Agency:Direct] = [Risk:IAD]"
        notice = self.router.route(expression)

        expected = (
            "This text is classified as [Authority:Indirect]. "
            "All decisions based on this classification remain with [Agency:Direct]."
        )
        assert notice == expected

    def test_valid_expressions(self):
        """Test various valid THM expressions."""
        valid_expressions = [
            "[Authority:Indirect] > [Authority:Direct] = [Risk:IVD]",
            "[Agency:Indirect] > [Agency:Direct] = [Risk:IAD]",
            "[Authority:Indirect] + [Agency:Indirect] > [Authority:Direct] + [Agency:Direct] = [Risk:GTD]",
            "[Authority:Direct] + [Agency:Direct] > [Authority:Indirect] + [Agency:Indirect] = [Risk:IID]",
            "[Authority:Indirect] -> [Agency:Direct]",
            "[Information]",
            "[Intelligence]",
            "![Authority:Direct]",
        ]

        for expr in valid_expressions:
            # Should not raise exception during validation
            result = self.router.route(expr)
            # Result may be None or notice depending on content
            assert isinstance(result, (str, type(None)))