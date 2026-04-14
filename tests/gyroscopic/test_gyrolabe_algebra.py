"""GyroLabe algebra conformance: canonical, chi/phi, WHT, and structured operators."""

from __future__ import annotations

import numpy as np
import pytest

from src.tools.gyroscopic.ops import (
    GyroLabeOperatorReport,
    WHT_SIZE,
    apply_hybrid_blocks,
    gyrolabe_analyze_operator_64,
    gyrolabe_apply_structured_64,
    gyrolabe_canonical_decompose,
    gyrolabe_canonical_reconstruct,
    gyrolabe_chirality_evolve_n,
    gyrolabe_k4_dot,
    gyrolabe_shell_population,
    tile_external_tensor,
    topk_energy_fractions,
    wht_64_batch,
)


@pytest.fixture(scope="module")
def _native_loaded() -> None:
    try:
        from src.tools.gyroscopic import gyromatmul_runtime_caps

        gyromatmul_runtime_caps()
    except Exception as e:
        pytest.skip(f"native gyrolabe DLL not loadable: {e}")


def _fwht64(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float64).copy()
    h = 1
    while h < 64:
        for i in range(0, 64, 2 * h):
            for j in range(h):
                u = y[i + j]
                v = y[i + j + h]
                y[i + j] = u + v
                y[i + j + h] = u - v
        h <<= 1
    return y


def _apply_spectral_tile64_numpy(x64: np.ndarray, phi64: np.ndarray) -> np.ndarray:
    tmp = _fwht64(x64)
    tmp = tmp * phi64.astype(np.float64)
    tmp = _fwht64(tmp)
    return (tmp / 64.0).astype(np.float64)


def _chi_invariant_B_from_row0(row0: np.ndarray) -> np.ndarray:
    row0 = row0.astype(np.int64).reshape(64)
    B = np.zeros((64, 64), dtype=np.int64)
    for i in range(64):
        for j in range(64):
            B[i, j] = row0[j ^ i]
    return B


def test_canonical_decompose_0xAAA(_native_loaded: None) -> None:
    c, chi, n_shell = gyrolabe_canonical_decompose(0xAAA)
    assert c == 42
    assert chi == 0
    assert n_shell == 0


def test_canonical_roundtrip(_native_loaded: None) -> None:
    w = 0x123
    c, chi, _ = gyrolabe_canonical_decompose(w)
    assert gyrolabe_canonical_reconstruct(c, chi) == w


def test_shell_population_N0(_native_loaded: None) -> None:
    assert gyrolabe_shell_population(0) == 64


def test_chirality_evolve_n_uniform(_native_loaded: None) -> None:
    chi_hist = [1] * 64
    ensemble = [1] * 64
    out = gyrolabe_chirality_evolve_n(chi_hist, ensemble, 2)
    assert sum(out) == pytest.approx(64, rel=0, abs=4)


def test_k4_dot_single(_native_loaded: None) -> None:
    assert gyrolabe_k4_dot([3], [4]) == 12


def test_chi_phi_unnormalized_matches_dense_xor_circulant() -> None:
    rng = np.random.default_rng(0)
    row0 = rng.integers(-3, 4, size=64, dtype=np.int64)
    B = _chi_invariant_B_from_row0(row0)
    phi_unnorm = _fwht64(row0.astype(np.float64))
    x = rng.standard_normal(64).astype(np.float64)
    y_struct = _apply_spectral_tile64_numpy(x, phi_unnorm)
    y_dense = B.astype(np.float64) @ x
    np.testing.assert_allclose(y_struct, y_dense, rtol=1e-5, atol=1e-4)


def test_chi_phi_divided_by_64_is_inconsistent() -> None:
    rng = np.random.default_rng(1)
    row0 = rng.integers(-2, 3, size=64, dtype=np.int64)
    B = _chi_invariant_B_from_row0(row0)
    phi_unnorm = _fwht64(row0.astype(np.float64))
    phi_wrong = phi_unnorm / 64.0
    x = rng.standard_normal(64).astype(np.float64)
    y_bad = _apply_spectral_tile64_numpy(x, phi_wrong)
    y_dense = B.astype(np.float64) @ x
    err = float(np.max(np.abs(y_bad - y_dense)))
    assert err > 0.05


def test_fwht_involution_scaling() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal(64)
    y = _fwht64(_fwht64(x))
    np.testing.assert_allclose(y, 64.0 * x, rtol=1e-12, atol=1e-9)


def test_wht_is_orthonormal_involution() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, WHT_SIZE))
    c = wht_64_batch(x)
    x2 = wht_64_batch(c)
    np.testing.assert_allclose(x2, x, rtol=1e-9, atol=1e-9)


def test_parseval_energy() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((50, WHT_SIZE))
    c = wht_64_batch(x)
    ex = np.sum(x * x, axis=-1)
    ec = np.sum(c * c, axis=-1)
    np.testing.assert_allclose(ec, ex, rtol=1e-9, atol=1e-9)


def test_topk_energy_fractions() -> None:
    x = np.ones((3, WHT_SIZE), dtype=np.float64)
    c = wht_64_batch(x)
    fr = topk_energy_fractions(c, (1, 4))
    assert fr[1] == pytest.approx(1.0)
    assert fr[4] == pytest.approx(1.0)


def test_operator_analysis_circulant(_native_loaded: None) -> None:
    row0 = np.arange(64, dtype=np.float32)
    W = np.zeros((64, 64), dtype=np.float32)
    for i in range(64):
        for j in range(64):
            W[i, j] = row0[j ^ i]
    report = gyrolabe_analyze_operator_64(W, threshold=0.01)
    assert report.op_class == 3
    assert report.scr == pytest.approx(1.0)
    assert report.eigenvalues_valid == 1


def test_apply_hybrid_matches_dense_random_64x64(_native_loaded: None) -> None:
    rng = np.random.default_rng(7)
    W = rng.standard_normal((64, 64)).astype(np.float32)
    rep = gyrolabe_analyze_operator_64(W, threshold=0.01)
    if int(rep.eigenvalues_valid) != 0:
        pytest.skip("need generic dense path for this smoke check")
    blocks = [(W, rep)]
    x = rng.standard_normal(64).astype(np.float32)
    yh = apply_hybrid_blocks(blocks, x)
    np.testing.assert_allclose(yh, W @ x, rtol=1e-4, atol=1e-3)


def test_tile_mx64_hybrid_matches_dense(_native_loaded: None) -> None:
    rng = np.random.default_rng(44)
    W = rng.standard_normal((200, 128)).astype(np.float32)
    blocks = tile_external_tensor(W)
    assert len(blocks) == 2
    assert blocks[0][0].shape == (200, 64)
    assert blocks[1][0].shape == (200, 64)
    x = rng.standard_normal(128).astype(np.float32)
    yh = apply_hybrid_blocks(blocks, x)
    np.testing.assert_allclose(yh, W @ x, rtol=1e-4, atol=1e-3)


def test_apply_structured_chi_gauge_class(_native_loaded: None) -> None:
    rep = GyroLabeOperatorReport()
    rep.op_class = 4
    rep.scr = 1.0
    rep.defect_norm = 0.0
    rep.eigenvalues_valid = 1
    for i in range(64):
        rep.eigenvalues_256[i * 4 + 0] = 1.0
        rep.eigenvalues_256[i * 4 + 1] = -1.0
        rep.eigenvalues_256[i * 4 + 2] = -1.0
        rep.eigenvalues_256[i * 4 + 3] = 1.0
    x = np.ones(64, dtype=np.float32)
    y = np.zeros(64, dtype=np.float32)
    gyrolabe_apply_structured_64(rep, x, y)
    assert np.all(np.isfinite(y))
    assert y.shape == (64,)
