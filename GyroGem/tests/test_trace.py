# Test Gyroscope Trace Builder
# [Authority:Indirect] + [Agency:Indirect]

import pytest
from GyroGem.agent.trace import build_trace
from GyroGem.agent.context import (
    THM_MARK, THM_GRAMMAR,
    CONSULT_DISPLACEMENT, CONSULT_ALIGNED,
)


class TestTraceBuilder:

    def test_first_trace_includes_mark_and_grammar(self):
        trace = build_trace(
            thm_expression="[Authority:Indirect] -> [Agency:Direct]",
            risk_code=None,
            trace_id=1,
            first=True,
            timestamp="2025-07-13T14:30",
        )
        assert "✋ The Human Mark" in trace
        assert "COMMON SOURCE CONSENSUS" in trace
        assert "GYROGOVERNANCE VERIFIED" in trace
        assert "Displacement" in trace  # from Grammar
        assert "[Authority:Direct]" in trace  # from Grammar tags
        assert "[Gyroscope 2.0]" in trace
        assert "[End]" in trace

    def test_subsequent_trace_compact(self):
        trace = build_trace(
            thm_expression="[Authority:Indirect] -> [Agency:Direct]",
            risk_code=None,
            trace_id=2,
            first=False,
            timestamp="2025-07-13T14:31",
        )
        assert "✋ The Human Mark" not in trace
        assert "COMMON SOURCE CONSENSUS" not in trace
        assert "[Gyroscope 2.0]" in trace
        assert "[End]" in trace

    def test_first_trace_much_larger(self):
        first = build_trace(
            "[Authority:Indirect] -> [Agency:Direct]", None, 1,
            first=True, timestamp="2025-01-01T00:00",
        )
        compact = build_trace(
            "[Authority:Indirect] -> [Agency:Direct]", None, 2,
            first=False, timestamp="2025-01-01T00:00",
        )
        assert len(first) > len(compact) * 2

    def test_aligned_consultation(self):
        trace = build_trace(
            "[Authority:Indirect] -> [Agency:Direct]", None, 1,
            timestamp="2025-01-01T00:00",
        )
        assert f"[Consult: {CONSULT_ALIGNED}]" in trace

    def test_displacement_consultation_all_four(self):
        for code in ("GTD", "IVD", "IAD", "IID"):
            trace = build_trace(
                f"[Risk:{code}]", code, 1,
                timestamp="2025-01-01T00:00",
            )
            assert CONSULT_DISPLACEMENT[code] in trace

    def test_trace_id_formatting(self):
        assert "ID: 001" in build_trace("x", None, 1, timestamp="2025-01-01T00:00")
        assert "ID: 042" in build_trace("x", None, 42, timestamp="2025-01-01T00:00")
        assert "ID: 999" in build_trace("x", None, 999, timestamp="2025-01-01T00:00")

    def test_gyroscope_structure_present(self):
        trace = build_trace(
            "[Authority:Indirect] -> [Agency:Direct]", None, 1,
            timestamp="2025-01-01T00:00",
        )
        assert "1 = Governance Management Traceability" in trace
        assert "2 = Information Curation Variety" in trace
        assert "3 = Inference Interaction Accountability" in trace
        assert "4 = Intelligence Cooperation Integrity" in trace
        assert "[THM:" in trace
        assert "[Consult:" in trace
        assert "[Timestamp:" in trace

    def test_auto_timestamp(self):
        trace = build_trace("x", None, 1)
        assert "[Timestamp:" in trace