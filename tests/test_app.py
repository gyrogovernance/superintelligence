"""
App-layer tests: coordination, ledgers, and GGG aperture.

Tests the app-layer coordination system:
- Domain ledgers and aperture computation
- Coordinator determinism and replay
- Event binding to kernel moments
"""

import numpy as np
from pathlib import Path

import pytest

from src.app.coordination import Coordinator
from src.app.events import Domain, EdgeID, GovernanceEvent
from src.app.ledger import DomainLedgers


@pytest.fixture(scope="module")
def atlas_dir():
    p = Path("data/atlas")
    if not p.exists():
        pytest.skip("Atlas not built. Run: python -m src.router.atlas")
    return p


class TestDomainLedgers:
    """Test domain ledger operations and aperture computation."""

    def test_aperture_zero_when_ledger_is_zero(self):
        """Aperture should be 0.0 when ledger vector is all zeros."""
        ledgers = DomainLedgers()
        for dom in (Domain.ECONOMY, Domain.EMPLOYMENT, Domain.EDUCATION):
            assert ledgers.aperture(dom) == 0.0

    def test_decompose_reconstructs_y(self):
        """Hodge decomposition should reconstruct: y = y_grad + y_cycle."""
        ledgers = DomainLedgers()

        # Put a nontrivial vector into Economy ledger via events
        for e, v in enumerate([1.0, -2.0, 0.5, 3.0, -1.5, 2.5]):
            ledgers.apply_event(GovernanceEvent(domain=Domain.ECONOMY, edge_id=EdgeID(e), magnitude=v))

        y = ledgers.get(Domain.ECONOMY)
        y_grad, y_cycle = ledgers.decompose(Domain.ECONOMY)

        # Reconstruction
        assert np.allclose(y, y_grad + y_cycle, atol=1e-12)

        # Orthogonality in the unweighted inner product (should hold for your projector)
        assert abs(float(y_grad @ y_cycle)) < 1e-10

    def test_aperture_scale_invariant(self):
        """Aperture should be scale-invariant (ratio property)."""
        ledgers = DomainLedgers()

        # Build a nonzero y
        ledgers.apply_event(GovernanceEvent(domain=Domain.ECONOMY, edge_id=EdgeID.GOV_INFO, magnitude=1.0))
        ledgers.apply_event(GovernanceEvent(domain=Domain.ECONOMY, edge_id=EdgeID.INFO_INFER, magnitude=2.0))

        a1 = ledgers.aperture(Domain.ECONOMY)

        # Scale y by 7 via replaying same events * 7 times
        ledgers2 = DomainLedgers()
        for _ in range(7):
            ledgers2.apply_event(GovernanceEvent(domain=Domain.ECONOMY, edge_id=EdgeID.GOV_INFO, magnitude=1.0))
            ledgers2.apply_event(GovernanceEvent(domain=Domain.ECONOMY, edge_id=EdgeID.INFO_INFER, magnitude=2.0))

        a2 = ledgers2.aperture(Domain.ECONOMY)

        assert abs(a1 - a2) < 1e-12


class TestCoordinator:
    """Test Coordinator determinism and event handling."""

    def test_coordinator_replay_determinism(self, atlas_dir):
        """Two coordinators with same bytes and events should produce identical state."""
        c1 = Coordinator(atlas_dir)
        c2 = Coordinator(atlas_dir)

        payload = b"Hello world"
        c1.step_bytes(payload)
        c2.step_bytes(payload)

        evs = [
            GovernanceEvent(domain=Domain.ECONOMY, edge_id=EdgeID.GOV_INFO, magnitude=1.0, confidence=0.8),
            GovernanceEvent(domain=Domain.EMPLOYMENT, edge_id=EdgeID.INFER_INTEL, magnitude=-0.5, confidence=1.0),
            GovernanceEvent(domain=Domain.EDUCATION, edge_id=EdgeID.INFO_INFER, magnitude=2.0, confidence=0.6),
        ]

        for ev in evs:
            c1.apply_event(ev, bind_to_kernel_moment=True)
            c2.apply_event(ev, bind_to_kernel_moment=True)

        s1 = c1.get_status()
        s2 = c2.get_status()

        assert s1.kernel["state_index"] == s2.kernel["state_index"]
        assert s1.kernel["state_hex"] == s2.kernel["state_hex"]

        assert np.allclose(np.array(s1.ledgers["y_econ"]), np.array(s2.ledgers["y_econ"]))
        assert np.allclose(np.array(s1.ledgers["y_emp"]), np.array(s2.ledgers["y_emp"]))
        assert np.allclose(np.array(s1.ledgers["y_edu"]), np.array(s2.ledgers["y_edu"]))

        assert abs(s1.apertures["econ"] - s2.apertures["econ"]) < 1e-12
        assert abs(s1.apertures["emp"] - s2.apertures["emp"]) < 1e-12
        assert abs(s1.apertures["edu"] - s2.apertures["edu"]) < 1e-12

    def test_event_binding_records_kernel_moment(self, atlas_dir):
        """Events bound to kernel moment should record state_index and last_byte."""
        c = Coordinator(atlas_dir)
        c.step_bytes(b"\x12\x34")  # advance kernel

        ev = GovernanceEvent(domain=Domain.ECONOMY, edge_id=EdgeID.GOV_INFO, magnitude=1.0)
        c.apply_event(ev, bind_to_kernel_moment=True)

        last = c.event_log[-1]["event"]
        assert last["kernel_state_index"] == c.kernel.state_index
        assert last["kernel_last_byte"] == c.kernel.last_byte

    def test_coordinator_reset(self, atlas_dir):
        """Reset should restore initial state and clear logs."""
        c = Coordinator(atlas_dir)
        c.step_bytes(b"test")
        c.apply_event(GovernanceEvent(domain=Domain.ECONOMY, edge_id=EdgeID.GOV_INFO, magnitude=1.0))

        assert len(c.byte_log) > 0
        assert len(c.event_log) > 0

        c.reset()

        assert c.kernel.state_index == c.kernel.archetype_index
        assert len(c.byte_log) == 0
        assert len(c.event_log) == 0
        assert c.ledgers.event_count == 0

    def test_coordinator_status_structure(self, atlas_dir):
        """get_status() should return properly structured CoordinationStatus."""
        c = Coordinator(atlas_dir)
        c.step_bytes(b"test")
        c.apply_event(GovernanceEvent(domain=Domain.ECONOMY, edge_id=EdgeID.GOV_INFO, magnitude=1.0))

        status = c.get_status()

        assert hasattr(status, "kernel")
        assert hasattr(status, "ledgers")
        assert hasattr(status, "apertures")

        assert "state_index" in status.kernel
        assert "state_hex" in status.kernel
        assert "a_hex" in status.kernel
        assert "b_hex" in status.kernel

        assert "y_econ" in status.ledgers
        assert "y_emp" in status.ledgers
        assert "y_edu" in status.ledgers

        assert "econ" in status.apertures
        assert "emp" in status.apertures
        assert "edu" in status.apertures


class TestHodgeProjections:
    """Test Hodge projection matrix invariants (audit-grade)."""

    def test_projector_identities(self):
        """Projectors must satisfy idempotence, symmetry, complementarity, and orthogonality."""
        from src.app.ledger import get_projections

        P_grad, P_cycle = get_projections()
        I = np.eye(6)

        # Symmetry
        assert np.allclose(P_grad, P_grad.T, atol=0, rtol=0)
        assert np.allclose(P_cycle, P_cycle.T, atol=0, rtol=0)

        # Idempotence
        assert np.allclose(P_grad @ P_grad, P_grad, atol=1e-15)
        assert np.allclose(P_cycle @ P_cycle, P_cycle, atol=1e-15)

        # Complementarity
        assert np.allclose(P_grad + P_cycle, I, atol=0, rtol=0)
        assert np.allclose(P_grad @ P_cycle, np.zeros((6, 6)), atol=1e-15)

    def test_cycle_component_in_kernel_of_B(self):
        """Cycle component must be in ker(B) for any edge vector."""
        from src.app.ledger import get_incidence_matrix, get_projections

        B = get_incidence_matrix()
        _, P_cycle = get_projections()

        y = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 2.5])
        y_cycle = P_cycle @ y

        # ker(B) check: B @ y_cycle must be zero
        assert np.allclose(B @ y_cycle, np.zeros(4), atol=1e-15)

    def test_cycle_basis_is_in_kernel_and_unit_norm(self):
        """Cycle basis columns must be in ker(B) and have unit norm."""
        from src.app.ledger import get_incidence_matrix, get_cycle_basis

        B = get_incidence_matrix()
        basis = get_cycle_basis()  # (6,3)

        assert basis.shape == (6, 3)
        for j in range(3):
            v = basis[:, j]
            # In kernel of B
            assert np.allclose(B @ v, np.zeros(4), atol=1e-15)
            # Unit norm
            assert abs(float(v @ v) - 1.0) < 1e-15

