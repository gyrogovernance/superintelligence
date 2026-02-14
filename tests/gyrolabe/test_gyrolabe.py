"""
GyroLabe tests.

Unit tests verify projection, extraction, byte combination, and helpers
without requiring a model or atlas.

Integration tests verify kernel coupling and are skipped when the atlas
is not available.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.tools.gyrolabe import (
    N_BOUNDARY,
    QUARTER_TURN,
    CouplingConfig,
    RoutedMLP,
    compute_mask,
    extract_byte,
    _detect_routed_layers,
    _entropy,
    _kernel_aperture_mass,
    _rerank_topk_logits_kernel_native,
    get_mask12_table,
    get_byte_charge_table,
    get_code_distance_matrix,
)
from src.router.kernel import RouterKernel


@pytest.fixture
def atlas_dir():
    p = Path("data/atlas")
    if not (p / "ontology.npy").exists():
        pytest.skip("Atlas not built (run: python -m src.router.atlas)")
    return p


@pytest.fixture
def kernel(atlas_dir):
    return RouterKernel(atlas_dir)


# -- Constants --

class TestConstants:
    def test_n_boundary(self):
        assert N_BOUNDARY == 256

    def test_quarter_turn(self):
        assert QUARTER_TURN == 64
        assert QUARTER_TURN == N_BOUNDARY // 4


# -- Entropy helper --

class TestEntropy:
    def test_uniform(self):
        counts = np.ones(256)
        assert abs(_entropy(counts) - 8.0) < 0.001

    def test_single_bin(self):
        counts = np.zeros(256)
        counts[0] = 100
        assert _entropy(counts) == 0.0

    def test_empty(self):
        counts = np.zeros(256)
        assert _entropy(counts) == 0.0

    def test_two_equal_bins(self):
        counts = np.zeros(256)
        counts[0] = 50
        counts[1] = 50
        assert abs(_entropy(counts) - 1.0) < 0.001


# -- Precomputed tables --

class TestPrecomputedTables:
    def test_mask12_table_shape(self):
        table = get_mask12_table()
        assert table.shape == (256,)
        assert table.dtype == np.uint16

    def test_mask12_table_range(self):
        table = get_mask12_table()
        assert table.max() <= 0xFFF
        assert table.min() >= 0

    def test_byte_charge_table_shape(self):
        table = get_byte_charge_table()
        assert table.shape == (256,)
        assert table.dtype == np.uint8

    def test_byte_charge_table_range(self):
        table = get_byte_charge_table()
        assert table.max() <= 3
        assert table.min() >= 0

    def test_code_distance_matrix_shape(self):
        matrix = get_code_distance_matrix()
        assert matrix.shape == (256, 256)
        assert matrix.dtype == np.uint8

    def test_code_distance_matrix_symmetric(self):
        matrix = get_code_distance_matrix()
        assert np.array_equal(matrix, matrix.T)

    def test_code_distance_matrix_diagonal_zero(self):
        matrix = get_code_distance_matrix()
        assert np.all(np.diag(matrix) == 0)

    def test_code_distance_matrix_range(self):
        matrix = get_code_distance_matrix()
        assert matrix.max() <= 12
        assert matrix.min() >= 0


# -- Kernel aperture --

class TestKernelAperture:
    def test_aperture_value(self):
        a = _kernel_aperture_mass()
        assert abs(a - 5/256) < 0.001

    def test_aperture_positive(self):
        a = _kernel_aperture_mass()
        assert a > 0

    def test_aperture_less_than_one(self):
        a = _kernel_aperture_mass()
        assert a < 1.0


# -- Projection mask --

class TestProjection:
    def test_shape_standard(self):
        mask = compute_mask(
            torch.device("cpu"), torch.float32,
            h=100, chi=0, p=0, n_fiber=16, last_byte_weight=6,
        )
        assert mask.shape == (1, 256 * 16)

    def test_shape_small_fiber(self):
        mask = compute_mask(
            torch.device("cpu"), torch.float32,
            h=50, chi=1, p=2, n_fiber=4, last_byte_weight=6,
        )
        assert mask.shape == (1, 256 * 4)

    def test_all_positive(self):
        mask = compute_mask(
            torch.device("cpu"), torch.float32,
            h=50, chi=2, p=1, n_fiber=16, last_byte_weight=6,
        )
        assert (mask > 0).all()

    def test_no_nan_or_inf(self):
        mask = compute_mask(
            torch.device("cpu"), torch.float32,
            h=0, chi=3, p=3, n_fiber=16, last_byte_weight=6,
        )
        assert not torch.isnan(mask).any()
        assert not torch.isinf(mask).any()

    def test_boundary_mean_near_one(self):
        mask = compute_mask(
            torch.device("cpu"), torch.float32,
            h=100, chi=0, p=0, n_fiber=16, last_byte_weight=6,
        )
        boundary = mask.view(256, 16)[:, 0]
        assert abs(boundary.mean().item() - 1.0) < 0.01

    def test_varies_with_horizon(self):
        m1 = compute_mask(torch.device("cpu"), torch.float32, 0, 0, 0, 16, 6)
        m2 = compute_mask(torch.device("cpu"), torch.float32, 128, 0, 0, 16, 6)
        assert not torch.allclose(m1, m2)

    def test_varies_with_vertex(self):
        m1 = compute_mask(torch.device("cpu"), torch.float32, 100, 0, 0, 16, 6)
        m2 = compute_mask(torch.device("cpu"), torch.float32, 100, 1, 0, 16, 6)
        assert not torch.allclose(m1, m2)

    def test_varies_with_phase(self):
        m1 = compute_mask(torch.device("cpu"), torch.float32, 100, 0, 0, 16, 6)
        m2 = compute_mask(torch.device("cpu"), torch.float32, 100, 0, 2, 16, 6)
        assert not torch.allclose(m1, m2)

    def test_varies_with_weight(self):
        m1 = compute_mask(torch.device("cpu"), torch.float32, 100, 0, 0, 16, 0)
        m2 = compute_mask(torch.device("cpu"), torch.float32, 100, 0, 0, 16, 12)
        assert not torch.allclose(m1, m2)

    def test_all_observable_combinations(self):
        for h in range(0, 256, 64):
            for chi in range(4):
                for p in range(4):
                    mask = compute_mask(
                        torch.device("cpu"), torch.float32,
                        h=h, chi=chi, p=p, n_fiber=16, last_byte_weight=6,
                    )
                    assert mask.shape == (1, 4096)
                    assert not torch.isnan(mask).any()
                    assert (mask > 0).all()

    def test_all_weight_values(self):
        for w in range(13):
            mask = compute_mask(
                torch.device("cpu"), torch.float32,
                h=100, chi=0, p=0, n_fiber=16, last_byte_weight=w,
            )
            assert mask.shape == (1, 4096)
            assert not torch.isnan(mask).any()


# -- Extraction --

class TestExtraction:
    def test_valid_byte(self):
        x = torch.randn(1, 4096)
        b, h_peak, pm, energy = extract_byte(x, n_fiber=16)
        assert 0 <= b <= 255

    def test_returns_valid_h_peak(self):
        x = torch.randn(1, 4096)
        _, h_peak, _, _ = extract_byte(x, n_fiber=16)
        assert 0 <= h_peak <= 255

    def test_returns_valid_peak_mass(self):
        x = torch.randn(1, 4096)
        _, _, pm, _ = extract_byte(x, n_fiber=16)
        assert 0.0 <= pm <= 1.0

    def test_returns_energy_tensor(self):
        x = torch.randn(1, 4096)
        _, _, _, energy = extract_byte(x, n_fiber=16)
        assert energy.shape == (256,)
        assert (energy >= 0).all()

    def test_deterministic(self):
        x = torch.randn(1, 4096)
        r1 = extract_byte(x, 16)
        r2 = extract_byte(x, 16)
        assert r1[0] == r2[0]
        assert r1[1] == r2[1]
        assert r1[2] == r2[2]

    def test_small_fiber(self):
        x = torch.randn(1, 256 * 4)
        b, _, _, _ = extract_byte(x, n_fiber=4)
        assert 0 <= b <= 255

    def test_batched_input(self):
        x = torch.randn(3, 4096)
        b, _, _, _ = extract_byte(x, n_fiber=16)
        assert 0 <= b <= 255

    def test_many_random_inputs(self):
        for seed in range(50):
            torch.manual_seed(seed)
            x = torch.randn(1, 4096)
            b, h_peak, pm, energy = extract_byte(x, n_fiber=16)
            assert 0 <= b <= 255
            assert 0 <= h_peak <= 255
            assert 0.0 <= pm <= 1.0
            assert energy.shape == (256,)


# -- Extraction alignment --

class TestExtractionAlignment:
    def test_concentrated_input(self):
        n_fiber = 16
        x = torch.zeros(1, 256 * n_fiber)
        target_h = 42
        start = target_h * n_fiber
        x[0, start:start + n_fiber] = 10.0

        _, h_peak, pm, _ = extract_byte(x, n_fiber)
        assert h_peak == target_h
        assert pm > 0.9

    def test_uniform_input_low_mass(self):
        x = torch.ones(1, 4096)
        _, _, pm, _ = extract_byte(x, 16)
        assert pm < 0.01

    def test_two_peaks(self):
        n_fiber = 16
        x = torch.zeros(1, 256 * n_fiber)
        x[0, 10 * n_fiber: 11 * n_fiber] = 5.0
        x[0, 200 * n_fiber: 201 * n_fiber] = 5.0

        _, h_peak, pm, _ = extract_byte(x, n_fiber)
        assert h_peak in (10, 200)
        assert 0.45 < pm < 0.55

    def test_energy_peak_matches_h_peak(self):
        n_fiber = 16
        x = torch.zeros(1, 256 * n_fiber)
        target_h = 100
        x[0, target_h * n_fiber: (target_h + 1) * n_fiber] = 10.0

        _, h_peak, _, energy = extract_byte(x, n_fiber)
        assert h_peak == target_h
        assert energy.argmax().item() == target_h


# -- Byte combination --

class TestByteCombination:
    def test_salt_prevents_cancellation(self):
        b = 0x55
        combined = 0
        for i in [3, 7]:
            salt = (i * 29 + 17) & 0xFF
            combined ^= (b ^ salt)
        combined &= 0xFF
        assert combined != 0

    def test_always_valid_byte(self):
        layers = [3, 7, 11, 15, 19, 23, 27, 31]
        for trial in range(100):
            combined = 0
            for i in layers:
                b = (trial * 7 + i * 13) & 0xFF
                salt = (i * 29 + 17) & 0xFF
                combined ^= (b ^ salt)
            combined &= 0xFF
            assert 0 <= combined <= 255

    def test_different_salts_per_layer(self):
        layers = [3, 7, 11, 15, 19, 23, 27, 31]
        salts = [(i * 29 + 17) & 0xFF for i in layers]
        assert len(set(salts)) == len(salts)


# -- Reranking --

class TestReranking:
    def test_returns_same_shape(self, kernel):
        topv = torch.randn(40)
        topi = torch.arange(40)
        result = _rerank_topk_logits_kernel_native(topv, topi, kernel)
        assert result.shape == topv.shape

    def test_returns_tensor(self, kernel):
        topv = torch.randn(40)
        topi = torch.arange(40)
        result = _rerank_topk_logits_kernel_native(topv, topi, kernel)
        assert isinstance(result, torch.Tensor)

    def test_no_nan_or_inf(self, kernel):
        topv = torch.randn(40)
        topi = torch.arange(40)
        result = _rerank_topk_logits_kernel_native(topv, topi, kernel)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_empty_input(self, kernel):
        topv = torch.tensor([])
        topi = torch.tensor([], dtype=torch.long)
        result = _rerank_topk_logits_kernel_native(topv, topi, kernel)
        assert result.shape == (0,)

    def test_single_element(self, kernel):
        topv = torch.tensor([1.0])
        topi = torch.tensor([42])
        result = _rerank_topk_logits_kernel_native(topv, topi, kernel)
        assert result.shape == (1,)

    def test_adjustment_is_small(self, kernel):
        topv = torch.randn(40)
        topi = torch.arange(40)
        result = _rerank_topk_logits_kernel_native(topv, topi, kernel)
        diff = (result - topv).abs()
        spread = topv.max() - topv.min()
        assert diff.max() < spread

    def test_deterministic(self, kernel):
        topv = torch.randn(40)
        topi = torch.arange(40)
        r1 = _rerank_topk_logits_kernel_native(topv.clone(), topi.clone(), kernel)
        r2 = _rerank_topk_logits_kernel_native(topv.clone(), topi.clone(), kernel)
        assert torch.allclose(r1, r2)


# -- Configuration --

class TestCouplingConfig:
    def test_default_values(self):
        config = CouplingConfig()
        assert config.routed_layers is None
        assert config.store_layer_telemetry is True

    def test_custom_routed_layers(self):
        config = CouplingConfig(routed_layers=[0, 5, 10])
        assert config.routed_layers == [0, 5, 10]

    def test_disable_telemetry(self):
        config = CouplingConfig(store_layer_telemetry=False)
        assert config.store_layer_telemetry is False


# -- Architecture check --

class TestArchitectureCheck:
    def test_unsupported_mlp_raises(self, kernel):
        from torch import nn

        class FakeMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

        with pytest.raises(TypeError, match="gate_proj"):
            RoutedMLP(
                FakeMLP(), kernel=kernel, layer_idx=0,
                config=CouplingConfig(), n_fiber=16,
            )

    def test_valid_mlp_accepted(self, kernel):
        from torch import nn

        class FakeSwiGLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(10, 10)
                self.up_proj = nn.Linear(10, 10)
                self.down_proj = nn.Linear(10, 10)

        wrapped = RoutedMLP(
            FakeSwiGLU(), kernel=kernel, layer_idx=0,
            config=CouplingConfig(), n_fiber=16,
        )
        assert wrapped.layer_idx == 0


# -- Layer detection --

class TestLayerDetection:
    def test_with_layer_types(self):
        class Config:
            layer_types = [
                "sliding_window", "sliding_window",
                "sliding_window", "full_attention",
                "sliding_window", "sliding_window",
                "sliding_window", "full_attention",
            ]

        class Model:
            config = Config()
            class model:
                layers = [None] * 8

        result = _detect_routed_layers(Model())
        assert result == [3, 7]

    def test_fallback(self):
        class Config:
            pass

        class Model:
            config = Config()
            class model:
                layers = [None] * 32

        result = _detect_routed_layers(Model())
        assert result == [3, 7, 11, 15, 19, 23, 27, 31]

    def test_fallback_small_model(self):
        class Config:
            pass

        class Model:
            config = Config()
            class model:
                layers = [None] * 8

        result = _detect_routed_layers(Model())
        assert result == [3, 7]


# -- Integration tests (require atlas) --

class TestKernelIntegration:
    def test_observables_in_range(self, kernel):
        for b in [0, 42, 170, 255]:
            kernel.step_byte(b)
            assert 0 <= kernel.current_horizon <= 255
            assert 0 <= kernel.current_vertex <= 3
            assert 0 <= kernel.current_phase <= 3

    def test_deterministic_replay(self, atlas_dir):
        seq = [10, 20, 30, 170, 255, 0, 42]

        k1 = RouterKernel(atlas_dir)
        for b in seq:
            k1.step_byte(b)
        sig1 = k1.signature()

        k2 = RouterKernel(atlas_dir)
        for b in seq:
            k2.step_byte(b)
        sig2 = k2.signature()

        assert sig1.state_hex == sig2.state_hex
        assert sig1.step == sig2.step

    def test_projection_with_kernel(self, kernel):
        kernel.step_byte(42)

        mask = compute_mask(
            torch.device("cpu"), torch.float32,
            kernel.current_horizon,
            kernel.current_vertex,
            kernel.current_phase,
            n_fiber=16,
            last_byte_weight=int(kernel.byte_weight[42]),
        )
        assert mask.shape == (1, 4096)
        assert not torch.isnan(mask).any()

    def test_rerank_with_kernel_state(self, kernel):
        kernel.step_byte(100)
        kernel.step_byte(200)

        topv = torch.randn(40)
        topi = torch.arange(40)
        result = _rerank_topk_logits_kernel_native(topv, topi, kernel)

        assert result.shape == topv.shape
        assert not torch.isnan(result).any()

    def test_byte_weight_lookup(self, kernel):
        for b in range(256):
            w = kernel.byte_weight[b]
            assert 0 <= w <= 12

    def test_byte_charge_lookup(self, kernel):
        for b in range(256):
            c = kernel.byte_charge[b]
            assert 0 <= c <= 3

    def test_gamma_table_shape(self, kernel):
        assert kernel.gamma_table.shape == (4, 4, 13)

    def test_gamma_table_values(self, kernel):
        assert not np.isnan(kernel.gamma_table).any()
        assert not np.isinf(kernel.gamma_table).any()