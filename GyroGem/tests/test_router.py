# Test Router Layer
# [Authority:Indirect] + [Agency:Indirect]

import pytest
from GyroGem.agent.router import THMRouter


class TestTHMRouter:

    def setup_method(self):
        self.router = THMRouter()

    def test_extract_risk_for_displacement(self):
        expr = "[Agency:Indirect] > [Agency:Direct] = [Risk:IAD]"
        assert self.router.validate(expr)
        assert self.router.extract_risk(expr) == "IAD"
        assert self.router.is_displacement(expr) is True

    def test_no_risk_for_flow(self):
        expr = "[Authority:Indirect] -> [Agency:Direct]"
        assert self.router.validate(expr)
        assert self.router.extract_risk(expr) is None
        assert self.router.is_displacement(expr) is False

    def test_no_risk_for_tag(self):
        expr = "[Authority:Indirect]"
        assert self.router.validate(expr)
        assert self.router.extract_risk(expr) is None
        assert self.router.is_displacement(expr) is False

    def test_malformed_expressions_invalid(self):
        malformed = [
            "",
            "[",
            "Invalid text",
            "[Authority:Indirect > [Authority:Direct",
        ]
        for expr in malformed:
            assert self.router.validate(expr) is False
            assert self.router.extract_risk(expr) is None