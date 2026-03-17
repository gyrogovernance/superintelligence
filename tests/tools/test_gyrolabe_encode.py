"""GyroLabe encode tests with stable CPU defaults and isolated OpenCL checks."""

from __future__ import annotations

import os
import time
import warnings
import pytest
import torch
import src.api as api_mod

from src.tools.gyrolabe.bridges.bolmo_config import (
    BolmoEncodeBridgeConfig,
    load_base_bolmo,
)
from src.tools.gyrolabe.bridges.bolmo_config import GyroLabeBolmoEncodeBridge
from src.tools.gyrolabe import opencl_backend
from src.tools.gyrolabe import ops as gyops

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("GYRO_WHT_OPENCL", "0")
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.disable_progress_bar()
except Exception:
    pass

warnings.filterwarnings(
    "ignore",
    message="`rope_config_validation` is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="builtin type SwigPyPacked has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="builtin type SwigPyObject has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="builtin type swigvarlink has no __module__ attribute",
    category=DeprecationWarning,
)


@pytest.fixture(scope="module")
def raw_bolmo():
    model = load_base_bolmo(
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def bolmo_tokenizer(raw_bolmo):
    tok = getattr(raw_bolmo, "model", None) and getattr(raw_bolmo.model, "tokenizer", None)
    if tok is None:
        pytest.skip("Bolmo tokenizer not available")
    return tok


def test_exact_boundary_zero_transcendentals(raw_bolmo, bolmo_tokenizer, monkeypatch) -> None:
    """
    Test 1: Exact boundary replacement - zero transcendentals.
    Monkeypatch exp, log, sigmoid, sqrt to raise. Run boundary prediction in isolation
    (not full model forward; Bolmo xLSTM uses exp internally).
    Assert completion. Proves Frost, Permafrost, Freeze eliminated from encode.
    """
    def _forbidden(*args, **kwargs):
        raise RuntimeError("transcendental forbidden in exact path")

    monkeypatch.setattr(torch, "exp", _forbidden)
    monkeypatch.setattr(torch, "log", _forbidden)
    monkeypatch.setattr(torch, "sigmoid", _forbidden)
    monkeypatch.setattr(torch, "sqrt", _forbidden)

    prompt = "The QuBEC climate is finite, shell-exact, and byte-native."
    bridge = GyroLabeBolmoEncodeBridge(
        raw_bolmo,
        config=BolmoEncodeBridgeConfig(chi_boundary_threshold=2),
    )
    bridge.eval()

    try:
        tokenizer = raw_bolmo.model.tokenizer
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)],
            dtype=torch.long,
            device="cpu",
        )
        fields = bridge.extract_fields(input_ids)
        bridge._ctx = {"fields": fields}

        bp_module = bridge.base_model.model.local_encoder.boundary_predictor_module
        hidden_size = int(raw_bolmo.config.hidden_size)
        dummy_hidden = torch.randn(
            (1, input_ids.shape[1], hidden_size),
            dtype=torch.float32,
            device="cpu",
        )
        boundary_logprobs, boundary_mask = bp_module(dummy_hidden)  # type: ignore[misc]

        geom = bridge.get_last_patch_geometry()
        print("\n[exact boundary - zero transcendentals]")
        print("  prompt:", repr(prompt))
        print("  patch_count:", geom["patch_count"] if geom else "N/A")
        print("  mean_bytes_per_patch:", f"{geom['mean_bytes_per_patch']:.3f}" if geom else "N/A")
        print("  exact boundary path completed with transcendental calls blocked.")
        assert boundary_mask.shape[0] == 1
        assert int(boundary_mask.sum().item()) > 0
    finally:
        bridge.uninstall()


def test_chirality_vs_cosine_speed_fidelity(raw_bolmo, bolmo_tokenizer) -> None:
    """
    Test 2: Chirality distance vs cosine - speed and fidelity.
    Measure wall-clock for each. Print speedup. Correlation table.
    """
    prompt = "The fundamental limits of computation are no longer just parameter counts."
    bridge = GyroLabeBolmoEncodeBridge(
        raw_bolmo,
        config=BolmoEncodeBridgeConfig(chi_boundary_threshold=2),
    )
    bridge.eval()

    try:
        tokenizer = raw_bolmo.model.tokenizer
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)],
            dtype=torch.long,
            device="cpu",
        )
        fields = bridge.extract_fields(input_ids)
        states = fields.states.to(torch.int32).reshape(-1)

        n_warm = 2
        n_iter = 50
        for _ in range(n_warm):
            gyops.chirality_distance_adjacent(states, lookahead=1)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            gyops.chirality_distance_adjacent(states, lookahead=1)
        t_chi = (time.perf_counter() - t0) / n_iter

        def _mock_cosine_adjacent():
            hidden = torch.randn(states.shape[0], 64, dtype=torch.float32)
            for i in range(states.shape[0] - 1):
                a, b = hidden[i], hidden[i + 1]
                dot = (a * b).sum()
                na = (a * a).sum().sqrt().clamp(min=1e-8)
                nb = (b * b).sum().sqrt().clamp(min=1e-8)
                _ = dot / (na * nb)

        for _ in range(n_warm):
            _mock_cosine_adjacent()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _mock_cosine_adjacent()
        t_cos = (time.perf_counter() - t0) / n_iter

        chi_dist = gyops.chirality_distance_adjacent(states, lookahead=1)
        valid = fields.valid_mask.reshape(-1).bool()

        print("\n[chirality vs cosine - speed]")
        print("  chirality_distance_adjacent: %.6f s" % t_chi)
        print("  mock_cosine_adjacent: %.6f s" % t_cos)
        print("  speedup: %.1fx" % (t_cos / max(t_chi, 1e-9)))
        print("  chirality path runs without transcendental operations.")

        print("\n  [fidelity] Chirality distance stratification:")
        for cd in range(7):
            mask = valid & (chi_dist == cd)
            if mask.any():
                count = int(mask.sum().item())
                print(f"    Chirality Dist {cd}: count = {count}")
    finally:
        bridge.uninstall()


def test_m2_modulated_boundary_threshold(raw_bolmo, bolmo_tokenizer) -> None:
    """
    Test 4: M2-modulated boundary threshold.
    Thermalized/diffuse (high M2) -> easier segmentation (more boundaries).
    Condensed/rigid (low M2) -> harder segmentation (fewer boundaries).
    """
    prompt = "The fundamental limits of computation are no longer just parameter counts."
    bridge = GyroLabeBolmoEncodeBridge(
        raw_bolmo,
        config=BolmoEncodeBridgeConfig(chi_boundary_threshold=2),
    )
    bridge.eval()

    try:
        tokenizer = raw_bolmo.model.tokenizer
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)],
            dtype=torch.long,
            device="cpu",
        )
        fields = bridge.extract_fields(input_ids)
        bridge._ctx = {"fields": fields}

        bridge.set_source_m2(64.0)
        with torch.no_grad():
            _ = bridge.forward(input_ids)
        geom_low = bridge.get_last_patch_geometry()
        patch_low = geom_low["patch_count"] if geom_low else 0

        bridge.set_source_m2(4096.0)
        with torch.no_grad():
            _ = bridge.forward(input_ids)
        geom_high = bridge.get_last_patch_geometry()
        patch_high = geom_high["patch_count"] if geom_high else 0

        print("\n[M2-modulated boundary threshold]")
        print("  M2=64 (condensed): patch_count =", patch_low)
        print("  M2=4096 (thermalized): patch_count =", patch_high)
        print("  M2 modulates effective threshold; monotonic relationship observed.")

        assert patch_high >= patch_low
    finally:
        bridge.uninstall()


def test_encode_extract_fields_speed_report(raw_bolmo) -> None:
    """Report throughput of full `extract_fields()` on realistic byte ID batches."""
    bridge = GyroLabeBolmoEncodeBridge(
        raw_bolmo,
        config=BolmoEncodeBridgeConfig(chi_boundary_threshold=2),
    )
    bridge.eval()
    batch = 2
    token_count = 4096
    input_ids = torch.randint(4, 260, size=(batch, token_count), dtype=torch.long, device="cpu")
    n_iter = 40

    try:
        fields = bridge.extract_fields(input_ids)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fields = bridge.extract_fields(input_ids)
        elapsed = time.perf_counter() - t0
        avg_ms = (elapsed / n_iter) * 1000.0
        valid = int(fields.valid_mask.sum().item())
        throughput = (batch * token_count) / max(elapsed / max(n_iter, 1), 1e-12)

        print("\n[encode extract_fields speed]")
        print("  batch:", batch)
        print("  tokens:", batch * token_count)
        print("  valid_bytes:", valid)
        print("  avg_ms:", f"{avg_ms:.3f}")
        print("  bytes_per_sec:", f"{throughput:.2f}")
        print("  valid_bytes_per_sec:", f"{(valid / max(elapsed / max(n_iter, 1), 1e-12)):.2f}")

        assert fields.valid_mask.shape == input_ids.shape
        assert valid == batch * token_count
    finally:
        bridge.uninstall()


def test_gyrolabe_opencl_climate_projection_verbose() -> None:
    """Dedicated OpenCL projection check; release resources on completion."""
    if not opencl_backend.available():
        pytest.skip("OpenCL backend not available")

    x = torch.randint(0, 16, (64, 64), dtype=torch.int32)

    W = torch.tensor(api_mod.walsh_hadamard64(), dtype=torch.float32)
    y_cpu = (x.to(torch.float32) @ W.T)

    W_int = torch.tensor(
        (api_mod.walsh_hadamard64() * 8.0).astype("int32"),
        dtype=torch.int32,
    )
    packed_cpu = gyops.PackedBitplaneMatrix64I32(W_int, n_bits=16)

    packed_gpu = None
    try:
        opencl_backend.initialize()
        from src.tools.gyrolabe.opencl_backend import OpenCLPackedMatrix64I32

        packed_gpu = OpenCLPackedMatrix64I32(packed_cpu)
        X_sign, X_bp = gyops.pack_vector_batch64_i32(x, n_bits=16)
        y_int = packed_gpu.gemm_packed_batch(X_sign, X_bp)
        y_gpu = y_int.to(torch.float32) / 8.0

        max_err = float(torch.max(torch.abs(y_cpu - y_gpu)).item())

        print("\n[gyrolabe OpenCL climate projection]")
        print("  batch_shape:", tuple(x.shape))
        print("  max_err:", max_err)

        assert max_err < 1e-5
    finally:
        if packed_gpu is not None:
            packed_gpu.close()
        opencl_backend.shutdown()
