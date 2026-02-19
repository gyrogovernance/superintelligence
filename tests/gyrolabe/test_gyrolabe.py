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

from src.router.kernel import RouterKernel
from src.tools.gyrolabe import (
    N_BOUNDARY,
    QUARTER_TURN,
    CouplingConfig,
    RoutedMLP,
    _detect_routed_layers,
    _entropy,
    compute_mask,
    extract_byte,
    get_code_distance_matrix,
    get_mask12_table,
)


@pytest.fixture
def atlas_dir():
    p = Path("data/atlas")
    if not (p / "ontology.npy").exists():
        pytest.skip("Atlas not built (run: python -m src.router.atlas)")
    return p


@pytest.fixture
def kernel(atlas_dir):
    return RouterKernel(atlas_dir)


@pytest.fixture
def byte_charge(kernel):
    """Provide byte_charge table from kernel for compute_mask calls."""
    return kernel.byte_charge


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

    def test_byte_charge_from_kernel(self, kernel):
        table = kernel.byte_charge
        assert table.shape == (256,)
        assert table.dtype == np.uint8
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


# -- Projection mask --

class TestProjection:
    def test_shape_standard(self, byte_charge):
        mask = compute_mask(
            torch.device("cpu"), torch.float32,
            h=100, chi=0, p=0, last_byte_weight=6,
            byte_charge_table=byte_charge,
        )
        assert mask.shape == (256,)

    def test_shape_consistent(self, byte_charge):
        mask = compute_mask(
            torch.device("cpu"), torch.float32,
            h=50, chi=1, p=2, last_byte_weight=6,
            byte_charge_table=byte_charge,
        )
        assert mask.shape == (256,)

    def test_all_positive(self, byte_charge):
        mask = compute_mask(
            torch.device("cpu"), torch.float32,
            h=50, chi=2, p=1, last_byte_weight=6,
            byte_charge_table=byte_charge,
        )
        assert (mask > 0).all()

    def test_no_nan_or_inf(self, byte_charge):
        mask = compute_mask(
            torch.device("cpu"), torch.float32,
            h=0, chi=3, p=3, last_byte_weight=6,
            byte_charge_table=byte_charge,
        )
        assert not torch.isnan(mask).any()
        assert not torch.isinf(mask).any()

    def test_boundary_mean_near_one(self, byte_charge):
        mask = compute_mask(
            torch.device("cpu"), torch.float32,
            h=100, chi=0, p=0, last_byte_weight=6,
            byte_charge_table=byte_charge,
        )
        assert abs(mask.mean().item() - 1.0) < 0.01

    def test_varies_with_horizon(self, byte_charge):
        m1 = compute_mask(torch.device("cpu"), torch.float32, 0, 0, 0, 6, byte_charge)
        m2 = compute_mask(torch.device("cpu"), torch.float32, 128, 0, 0, 6, byte_charge)
        assert not torch.allclose(m1, m2)

    def test_varies_with_vertex(self, byte_charge):
        m1 = compute_mask(torch.device("cpu"), torch.float32, 100, 0, 0, 6, byte_charge)
        m2 = compute_mask(torch.device("cpu"), torch.float32, 100, 1, 0, 6, byte_charge)
        assert not torch.allclose(m1, m2)

    def test_varies_with_phase(self, byte_charge):
        m1 = compute_mask(torch.device("cpu"), torch.float32, 100, 0, 0, 6, byte_charge)
        m2 = compute_mask(torch.device("cpu"), torch.float32, 100, 0, 2, 6, byte_charge)
        assert not torch.allclose(m1, m2)

    def test_varies_with_weight(self, byte_charge):
        m1 = compute_mask(torch.device("cpu"), torch.float32, 100, 0, 0, 0, byte_charge)
        m2 = compute_mask(torch.device("cpu"), torch.float32, 100, 0, 0, 12, byte_charge)
        assert not torch.allclose(m1, m2)

    def test_all_observable_combinations(self, byte_charge):
        for h in range(0, 256, 64):
            for chi in range(4):
                for p in range(4):
                    mask = compute_mask(
                        torch.device("cpu"), torch.float32,
                        h=h, chi=chi, p=p, last_byte_weight=6,
                        byte_charge_table=byte_charge,
                    )
                    assert mask.shape == (256,)
                    assert not torch.isnan(mask).any()
                    assert (mask > 0).all()

    def test_all_weight_values(self, byte_charge):
        for w in range(13):
            mask = compute_mask(
                torch.device("cpu"), torch.float32,
                h=100, chi=0, p=0, last_byte_weight=w,
                byte_charge_table=byte_charge,
            )
            assert mask.shape == (256,)
            assert not torch.isnan(mask).any()

    def test_differential_modulation(self, byte_charge):
        m_no_prev = compute_mask(
            torch.device("cpu"), torch.float32,
            h=100, chi=0, p=0, last_byte_weight=6,
            byte_charge_table=byte_charge, prev_h=None,
        )
        m_with_prev = compute_mask(
            torch.device("cpu"), torch.float32,
            h=100, chi=0, p=0, last_byte_weight=6,
            byte_charge_table=byte_charge, prev_h=50,
        )
        # With differential modulation the mask should differ
        assert not torch.allclose(m_no_prev, m_with_prev)

    def test_differential_small_transition_attenuates(self, byte_charge):
        # Same horizon as prev_h: td=0, diff_scale=0.5, mask closer to 1
        m_same = compute_mask(
            torch.device("cpu"), torch.float32,
            h=100, chi=0, p=0, last_byte_weight=6,
            byte_charge_table=byte_charge, prev_h=100,
        )
        m_none = compute_mask(
            torch.device("cpu"), torch.float32,
            h=100, chi=0, p=0, last_byte_weight=6,
            byte_charge_table=byte_charge, prev_h=None,
        )
        # Same-h mask should be closer to uniform than no-prev mask
        std_same = m_same.std().item()
        std_none = m_none.std().item()
        assert std_same < std_none


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
    def test_unsupported_mlp_raises(self):
        from torch import nn

        class FakeMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

        with pytest.raises(TypeError, match="gate_proj"):
            RoutedMLP(FakeMLP(), layer_idx=0, n_fiber=16)

    def test_valid_mlp_accepted(self):
        from torch import nn

        class FakeSwiGLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(10, 10)
                self.up_proj = nn.Linear(10, 10)
                self.down_proj = nn.Linear(10, 10)

        wrapped = RoutedMLP(FakeSwiGLU(), layer_idx=0, n_fiber=16)
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
            h = kernel.current_horizon
            v = kernel.current_vertex
            p = kernel.current_phase
            assert np.all((h >= 0) & (h <= 255))
            assert np.all((v >= 0) & (v <= 3))
            assert np.all((p >= 0) & (p <= 3))

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
            int(kernel.current_horizon[0]),
            int(kernel.current_vertex[0]),
            int(kernel.current_phase[0]),
            last_byte_weight=int(kernel.byte_weight[42]),
            byte_charge_table=kernel.byte_charge,
        )
        assert mask.shape == (256,)
        assert not torch.isnan(mask).any()

    def test_byte_weight_lookup(self, kernel):
        for b in range(256):
            w = kernel.byte_weight[b]
            assert 0 <= w <= 12

    def test_byte_charge_lookup(self, kernel):
        for b in range(256):
            c = kernel.byte_charge[b]
            assert 0 <= c <= 3

    def test_batch_kernel_step(self, atlas_dir):
        k = RouterKernel(atlas_dir, batch_size=3)
        k.step_byte(42)
        assert k.current_horizon.shape == (3,)
        assert np.all(k.current_horizon == k.current_horizon[0])

    def test_batch_kernel_per_sequence(self, atlas_dir):
        k = RouterKernel(atlas_dir, batch_size=2)
        k.step_byte(np.array([42, 100]))
        h = k.current_horizon
        assert h.shape == (2,)
        # Different bytes should generally produce different states
        # (not guaranteed but overwhelmingly likely)

    def test_batch_resize(self, atlas_dir):
        k = RouterKernel(atlas_dir, batch_size=1)
        k.step_byte(42)
        k.resize_batch(4)
        assert k.batch_size == 4
        assert k.state_index.shape == (4,)
        assert k.step == 0
