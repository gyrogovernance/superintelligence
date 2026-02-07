# Test THM Grammar Validation and Risk Extraction
# [Authority:Indirect] + [Agency:Indirect]

import pytest
from agent.router import THMRouter


class TestTHMGrammar:

    def setup_method(self):
        self.router = THMRouter()

    def test_displacement_patterns_valid_and_detected(self):
        patterns = [
            ("[Authority:Indirect] > [Authority:Direct] = [Risk:IVD]", "IVD"),
            ("[Agency:Indirect] > [Agency:Direct] = [Risk:IAD]", "IAD"),
            ("[Authority:Indirect] + [Agency:Indirect] > [Authority:Direct] + "
             "[Agency:Direct] = [Risk:GTD]", "GTD"),
            ("[Authority:Direct] + [Agency:Direct] > [Authority:Indirect] + "
             "[Agency:Indirect] = [Risk:IID]", "IID"),
        ]
        for expr, code in patterns:
            assert self.router.validate(expr)
            assert self.router.extract_risk(expr) == code
            assert self.router.is_displacement(expr) is True

    def test_governance_patterns_valid_and_not_displacement(self):
        patterns = [
            "[Authority:Indirect] -> [Agency:Direct]",
            "[Authority:Direct] -> [Authority:Indirect] -> [Agency:Direct]",
            "[Authority:Direct] -> [Authority:Indirect] + [Agency:Indirect] -> [Agency:Direct]",
        ]
        for expr in patterns:
            assert self.router.validate(expr)
            assert self.router.is_displacement(expr) is False
            assert self.router.extract_risk(expr) is None

    def test_simple_tags_valid(self):
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
            assert self.router.validate(tag)

    def test_composite_tags_valid(self):
        composites = [
            "[Authority:Indirect] + [Agency:Indirect]",
            "[Authority:Direct] + [Agency:Direct]",
            "[Information] + [Inference]",
        ]
        for expr in composites:
            assert self.router.validate(expr)

    def test_negated_tags_valid(self):
        negated = [
            "![Authority:Direct]",
            "![Agency:Indirect]",
            "![Information]",
        ]
        for expr in negated:
            assert self.router.validate(expr)

    def test_risk_tags_invalid_standalone(self):
        risk_tags = ["[Risk:GTD]", "[Risk:IVD]", "[Risk:IAD]", "[Risk:IID]"]
        for tag in risk_tags:
            assert self.router.validate(tag) is False

    def test_invalid_categories(self):
        invalids = [
            "[Invalid:Direct]",
            "[Wrong:Indirect]",
            "[Bad:Value]",
        ]
        for expr in invalids:
            assert self.router.validate(expr) is False

    def test_invalid_values(self):
        invalids = [
            "[Authority:Invalid]",
            "[Agency:Wrong]",
            "[Information:Bad]",
        ]
        for expr in invalids:
            assert self.router.validate(expr) is False

    def test_invalid_risk_codes(self):
        invalids = [
            "[Risk:INVALID]",
            "[Risk:BAD]",
            "[Risk:123]",
        ]
        for expr in invalids:
            assert self.router.extract_risk(expr) is None

    def test_unmatched_brackets(self):
        invalids = [
            "[Authority:Direct",
            "Authority:Direct]",
            "[[Authority:Direct]",
            "[Authority:Direct]]",
        ]
        for expr in invalids:
            assert self.router.validate(expr) is False

    def test_invalid_operators(self):
        invalids = [
            "[Authority:Direct] >> [Agency:Indirect]",
            "[Authority:Direct] -- [Agency:Indirect]",
            "[Authority:Direct] == [Agency:Indirect]",
        ]
        for expr in invalids:
            assert self.router.validate(expr) is False