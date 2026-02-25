"""
Tool tests: analytics and framework tools.

Tests the tool system:
- Analytics helpers (Hodge decomposition)
- Framework tools (THM, Gyroscope)
- API adapters
"""

import numpy as np

from src.app.events import MICRO, Domain, EdgeID, GovernanceEvent
from src.app.ledger import DomainLedgers
from src.tools.analytics import hodge_decompose
from src.tools.api import event_from_dict, event_to_dict, parse_domain, parse_edge_id
from src.tools.frameworks import (
    GyroscopeWorkMixTool,
    ToolContext,
    THMDisplacementTool,
)


class TestAnalytics:
    """Test analytics helpers (Hodge decomposition)."""

    def test_tools_analytics_matches_domainledger_aperture(self):
        """hodge_decompose should match DomainLedgers.aperture for same vector."""
        ledgers = DomainLedgers()
        ledgers.apply_event(GovernanceEvent(domain=Domain.EDUCATION, edge_id=EdgeID.GOV_INTEL, magnitude_micro=int(round(1.25 * MICRO)), confidence_micro=MICRO))
        ledgers.apply_event(GovernanceEvent(domain=Domain.EDUCATION, edge_id=EdgeID.INFER_INTEL, magnitude_micro=int(round(-0.75 * MICRO)), confidence_micro=MICRO))

        y = ledgers.get(Domain.EDUCATION)
        a_led = ledgers.aperture(Domain.EDUCATION)
        # Convert int64 array to float64 for analytics tool (it expects float64)
        y_float = y.astype(np.float64)
        a_ana = hodge_decompose(y_float).aperture

        assert abs(a_led - a_ana) < 1e-12

    def test_hodge_decompose_reconstruction(self):
        """Hodge decomposition should reconstruct: y = y_grad + y_cycle."""
        y = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 2.5], dtype=np.float64)

        result = hodge_decompose(y)

        assert np.allclose(y, result.y_grad + result.y_cycle, atol=1e-12)
        assert abs(float(result.y_grad @ result.y_cycle)) < 1e-10

    def test_hodge_decompose_zero_vector(self):
        """Hodge decomposition of zero vector should return zero components and zero aperture."""
        y = np.zeros(6, dtype=np.float64)

        result = hodge_decompose(y)

        assert np.allclose(result.y_grad, 0.0)
        assert np.allclose(result.y_cycle, 0.0)
        assert result.aperture == 0.0


class TestAPIAdapters:
    """Test API parsing and serialization."""

    def test_parse_domain(self):
        """parse_domain should handle int and string inputs."""
        assert parse_domain(0) == Domain.ECONOMY
        assert parse_domain(1) == Domain.EMPLOYMENT
        assert parse_domain(2) == Domain.EDUCATION

        assert parse_domain("economy") == Domain.ECONOMY
        assert parse_domain("employment") == Domain.EMPLOYMENT
        assert parse_domain("education") == Domain.EDUCATION

        assert parse_domain("invalid") is None
        assert parse_domain(99) is None

    def test_parse_edge_id(self):
        """parse_edge_id should handle int and string inputs."""
        assert parse_edge_id(0) == EdgeID.GOV_INFO
        assert parse_edge_id(5) == EdgeID.INFER_INTEL

        assert parse_edge_id("GOV_INFO") == EdgeID.GOV_INFO
        assert parse_edge_id("INFER_INTEL") == EdgeID.INFER_INTEL

        assert parse_edge_id("INVALID") is None
        assert parse_edge_id(99) is None

    def test_event_from_dict(self):
        """event_from_dict should create GovernanceEvent from dict."""
        d = {
            "domain": "economy",
            "edge_id": "GOV_INFO",
            "magnitude": 1.5,
            "confidence": 0.8,
            "meta": {"test": "value"},
        }

        ev = event_from_dict(d)

        assert ev.domain == Domain.ECONOMY
        assert ev.edge_id == EdgeID.GOV_INFO
        # event_from_dict converts legacy float format to micro-units
        assert ev.magnitude_micro == int(round(1.5 * MICRO))
        assert ev.confidence_micro == int(round(0.8 * MICRO))
        assert ev.meta == {"test": "value"}

    def test_event_to_dict(self):
        """event_to_dict should serialize GovernanceEvent to dict."""
        ev = GovernanceEvent(
            domain=Domain.EMPLOYMENT,
            edge_id=EdgeID.INFO_INFER,
            magnitude_micro=int(round(-0.5 * MICRO)),
            confidence_micro=int(round(0.9 * MICRO)),
            meta={"key": "value"},
        )

        d = event_to_dict(ev)

        assert d["domain"] == 1
        assert d["edge_id"] == 3
        assert d["magnitude_micro"] == int(round(-0.5 * MICRO))
        assert d["confidence_micro"] == int(round(0.9 * MICRO))
        assert d["meta"] == {"key": "value"}


class TestFrameworkTools:
    """Test framework tools (THM, Gyroscope)."""

    def test_thm_displacement_tool(self):
        """THMDisplacementTool should emit events for displacement signals."""
        tool = THMDisplacementTool()
        ctx = ToolContext()

        payload = {
            "domain": "education",
            "GTD": 0.1,
            "IVD": -0.2,
            "IAD": 0.05,
            "IID": 0.0,
        }

        events = tool.emit_events(payload, ctx)

        assert len(events) == 3  # IID is 0.0, so not emitted

        # Check GTD -> GOV_INFO
        gtd_ev = next((e for e in events if e.meta.get("signal") == "GTD"), None)
        assert gtd_ev is not None
        assert gtd_ev.domain == Domain.EDUCATION
        assert gtd_ev.edge_id == EdgeID.GOV_INFO
        assert abs(gtd_ev.magnitude_micro - int(round(0.1 * MICRO))) < 10  # Allow for rounding differences

        # Check IVD -> INFO_INFER
        ivd_ev = next((e for e in events if e.meta.get("signal") == "IVD"), None)
        assert ivd_ev is not None
        assert ivd_ev.edge_id == EdgeID.INFO_INFER
        assert abs(ivd_ev.magnitude_micro - int(round(-0.2 * MICRO))) < 10  # Allow for rounding differences

    def test_thm_displacement_tool_ignores_domain_parameter(self):
        """THMDisplacementTool always emits to EDUCATION domain regardless of payload domain."""
        tool = THMDisplacementTool()
        ctx = ToolContext()

        payload = {"domain": "invalid", "GTD": 0.1}

        events = tool.emit_events(payload, ctx)

        # Tool ignores domain parameter and always emits to EDUCATION
        assert len(events) == 1
        assert events[0].domain == Domain.EDUCATION
        assert events[0].edge_id == EdgeID.GOV_INFO

    def test_gyroscope_workmix_tool(self):
        """GyroscopeWorkMixTool should emit events for work-mix shifts."""
        tool = GyroscopeWorkMixTool()
        ctx = ToolContext()

        payload = {
            "domain": "employment",
            "GM": 0.1,
            "ICu": -0.1,
            "IInter": 0.0,
            "ICo": 0.0,
        }

        events = tool.emit_events(payload, ctx)

        assert len(events) == 1  # Only GM-ICu delta is nonzero

        ev = events[0]
        assert ev.domain == Domain.EMPLOYMENT
        assert ev.edge_id == EdgeID.GOV_INFO
        assert abs(ev.magnitude_micro - int(round(0.2 * MICRO))) < 10  # 0.1 - (-0.1), allow for rounding
        assert ev.meta["metric"] == "GM-ICu"

    def test_gyroscope_workmix_tool_infer_intel(self):
        """GyroscopeWorkMixTool should emit INFER_INTEL events for IInter-ICo shifts."""
        tool = GyroscopeWorkMixTool()
        ctx = ToolContext()

        payload = {
            "domain": "employment",
            "GM": 0.0,
            "ICu": 0.0,
            "IInter": 0.3,
            "ICo": 0.1,
        }

        events = tool.emit_events(payload, ctx)

        assert len(events) == 1

        ev = events[0]
        assert ev.edge_id == EdgeID.INFER_INTEL
        assert abs(ev.magnitude_micro - int(round(0.2 * MICRO))) < 10  # 0.3 - 0.1, allow for rounding
        assert ev.meta["metric"] == "IInter-ICo"

    def test_tool_context_meta(self):
        """Tool context meta should be included in emitted events."""
        tool = THMDisplacementTool()
        ctx = ToolContext(meta={"session": "test123", "actor": "user1"})

        payload = {"domain": "economy", "GTD": 0.1}

        events = tool.emit_events(payload, ctx)

        assert len(events) == 1
        assert events[0].meta["session"] == "test123"
        assert events[0].meta["actor"] == "user1"
        assert events[0].meta["tool"] == "thm_displacement"

