import numpy as np

from ggg_asi_router.physics import k4_geometry


def test_projection_idempotence_and_completeness() -> None:
    B = k4_geometry.get_incidence_matrix_k4()
    W = k4_geometry.get_weight_matrix_k4()
    P_grad, P_cycle = k4_geometry.compute_projections_k4(B, W)

    I = np.eye(6)
    assert np.allclose(P_grad + P_cycle, I, atol=1e-8)
    assert np.allclose(P_grad @ P_grad, P_grad, atol=1e-8)
    assert np.allclose(P_cycle @ P_cycle, P_cycle, atol=1e-8)


def test_hodge_orthogonality_and_bounds() -> None:
    rng = np.random.default_rng(0)
    B = k4_geometry.get_incidence_matrix_k4()
    W = k4_geometry.get_weight_matrix_k4()
    P_grad, P_cycle = k4_geometry.compute_projections_k4(B, W)

    for _ in range(16):
        y = rng.standard_normal(6)
        y_grad, y_cycle = k4_geometry.hodge_decomposition(y, P_grad, P_cycle)
        inner = float(y_grad.T @ W @ y_cycle)
        assert abs(inner) < 1e-6

        A = k4_geometry.aperture(y, y_cycle, W)
        assert 0.0 <= A <= 1.0


def test_face_cycle_matrix_is_in_kernel_of_incidence() -> None:
    """Test that face-cycle matrix F columns are in the kernel of B (B @ F = 0)."""
    B = k4_geometry.get_incidence_matrix_k4()
    W = k4_geometry.get_weight_matrix_k4()
    F = k4_geometry.get_face_cycle_matrix_k4(W)
    assert np.allclose(B @ F, np.zeros((4, 3)), atol=1e-8)

