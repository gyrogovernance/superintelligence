# Test Guard Orchestrator
# [Authority:Indirect] + [Agency:Indirect]

import pytest
from unittest.mock import MagicMock
from GyroGem.agent.guard import GyroGemGuard
from GyroGem.agent.router import THMRouter
from GyroGem.agent.context import (
    DEFAULT_EXPRESSION, CONSULT_ALIGNED, CONSULT_DISPLACEMENT, THM_MARK,
)


class TestGyroGemGuard:

    def _make_guard(self, classify_return: str):
        mock_model = MagicMock()
        mock_model.classify.return_value = classify_return
        return GyroGemGuard(model=mock_model, router=THMRouter())

    def test_first_trace_includes_mark(self):
        guard = self._make_guard("[Authority:Indirect] -> [Agency:Direct]")
        result = guard.process("Hello")

        assert result["first_trace"] is True
        assert "✋ The Human Mark" in result["trace"]
        assert "COMMON SOURCE CONSENSUS" in result["trace"]

    def test_second_trace_compact(self):
        guard = self._make_guard("[Authority:Indirect] -> [Agency:Direct]")
        guard.process("First message")
        result = guard.process("Second message")

        assert result["first_trace"] is False
        assert "✋ The Human Mark" not in result["trace"]

    def test_aligned_classification(self):
        guard = self._make_guard("[Authority:Indirect] -> [Agency:Direct]")
        result = guard.process("Here is the answer.")

        assert result["expression"] == "[Authority:Indirect] -> [Agency:Direct]"
        assert result["risk_code"] is None
        assert result["is_displacement"] is False
        assert CONSULT_ALIGNED in result["trace"]

    def test_displacement_classification(self):
        guard = self._make_guard(
            "[Agency:Indirect] > [Agency:Direct] = [Risk:IAD]")
        result = guard.process("I am thinking about your question.")

        assert result["risk_code"] == "IAD"
        assert result["is_displacement"] is True
        assert CONSULT_DISPLACEMENT["IAD"] in result["trace"]

    def test_all_four_displacements(self):
        cases = [
            ("[Authority:Indirect] > [Authority:Direct] = [Risk:IVD]", "IVD"),
            ("[Agency:Indirect] > [Agency:Direct] = [Risk:IAD]", "IAD"),
            ("[Authority:Indirect] + [Agency:Indirect] > [Authority:Direct] + [Agency:Direct] = [Risk:GTD]", "GTD"),
            ("[Authority:Direct] + [Agency:Direct] > [Authority:Indirect] + [Agency:Indirect] = [Risk:IID]", "IID"),
        ]
        for expr, expected_code in cases:
            guard = self._make_guard(expr)
            result = guard.process("text")
            assert result["risk_code"] == expected_code
            assert CONSULT_DISPLACEMENT[expected_code] in result["trace"]

    def test_model_failure_fallback(self):
        mock_model = MagicMock()
        mock_model.classify.side_effect = RuntimeError("model error")
        guard = GyroGemGuard(model=mock_model, router=THMRouter())

        result = guard.process("Some text")

        assert result["expression"] == DEFAULT_EXPRESSION
        assert result["risk_code"] is None
        assert result["is_displacement"] is False

    def test_malformed_expression_fallback(self):
        guard = self._make_guard("not valid THM at all")
        result = guard.process("Some text")

        assert result["expression"] == DEFAULT_EXPRESSION
        assert result["risk_code"] is None

    def test_trace_id_increments(self):
        guard = self._make_guard("[Authority:Indirect] -> [Agency:Direct]")
        r1 = guard.process("First")
        r2 = guard.process("Second")
        r3 = guard.process("Third")

        assert "ID: 001" in r1["trace"]
        assert "ID: 002" in r2["trace"]
        assert "ID: 003" in r3["trace"]

    def test_result_structure(self):
        guard = self._make_guard("[Authority:Indirect] -> [Agency:Direct]")
        result = guard.process("Any text")

        expected_keys = {"expression", "risk_code", "is_displacement", "trace", "first_trace"}
        assert set(result.keys()) == expected_keys

    def test_empty_text(self):
        guard = self._make_guard("[Authority:Indirect] -> [Agency:Direct]")
        result = guard.process("")
        assert result["trace"]

    def test_model_called_every_turn(self):
        mock_model = MagicMock()
        mock_model.classify.return_value = "[Authority:Indirect] -> [Agency:Direct]"
        guard = GyroGemGuard(model=mock_model, router=THMRouter())

        guard.process("First")
        guard.process("Second")
        guard.process("Third")

        assert mock_model.classify.call_count == 3