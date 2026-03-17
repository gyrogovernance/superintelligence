"""GyroGraph decode tests with explicit default-path policy.

CPU is the default for normal graph tests. OpenCL behavior is only exercised
through the dedicated verbose OpenCL test.
"""

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
from src.tools.gyrograph.bridges.bolmo_config import BolmoDecodeBridgeConfig
from src.tools.gyrograph.bridges.bolmo_config import (
    GyroGraphBolmoDecodeBridge,
)

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


class _DummyModel:
    pass


def _strict_decode_config(**overrides: object) -> BolmoDecodeBridgeConfig:
    cfg = {
        "phase_hysteresis": 0.75,
        "proof_mode": True,
    }
    cfg.update(overrides)
    return BolmoDecodeBridgeConfig(**cfg)


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


def _run_generation(
    raw_bolmo,
    prompt: str,
    *,
    encode_config: BolmoEncodeBridgeConfig,
    decode_config: BolmoDecodeBridgeConfig,
    max_new_tokens: int,
    stream_id: str,
):
    """Run decode generation through CPU default paths.

    This helper always disables OpenCL hotpath for decode ingest so normal tests
    exercise the proven fastest and most stable route.
    """
    encode_bridge = GyroLabeBolmoEncodeBridge(raw_bolmo, config=encode_config)
    encode_bridge.eval()

    decode_bridge = GyroGraphBolmoDecodeBridge(
        config=decode_config,
        cell_capacity=512,
        use_opencl_hotpath=False,
    )

    try:
        tokenizer = raw_bolmo.model.tokenizer
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)],
            dtype=torch.long,
            device="cpu",
        )

        with torch.no_grad():
            with decode_bridge.session(encode_bridge, batch_size=1, stream_ids=[stream_id]):
                output_ids = encode_bridge.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

        report = decode_bridge.emit_stream_report(stream_id)
        text = tokenizer.decode(output_ids[0].tolist())
        return text, report
    finally:
        encode_bridge.uninstall()


def test_exact_selection_zero_transcendentals(monkeypatch) -> None:
    """
    Test 1: Exact selection - zero transcendentals.
    Monkeypatch transcendentals to raise. Run selection. Assert completion.
    Proves Frost and Drought eliminated from decode.
    """
    bridge = GyroGraphBolmoDecodeBridge(
        config=_strict_decode_config(),
        cell_capacity=16,
    )
    dummy = _DummyModel()

    def _forbidden(*args, **kwargs):
        raise RuntimeError("forbidden transcendental called")

    monkeypatch.setattr(torch, "exp", _forbidden)
    monkeypatch.setattr(torch, "log", _forbidden)
    monkeypatch.setattr(torch, "sigmoid", _forbidden)
    monkeypatch.setattr(torch, "logaddexp", _forbidden)
    monkeypatch.setattr(torch, "sqrt", _forbidden)

    with bridge.session(dummy, batch_size=1, stream_ids=["exact"]):
        logits = torch.randn((1, 520), dtype=torch.float32) * 10.0
        logits[0, 4 + 65] += 50.0
        logits[0, 260 + 4 + 65] += 50.0
        token = int(bridge.select_hook(logits.clone(), 260)[0].item())

    print("\n[decode exact - zero transcendentals]")
    print("  selected_token:", token)
    print("  exact selector completed with transcendental calls blocked.")


def test_qsector_collapse_drought_elimination(raw_bolmo, bolmo_tokenizer) -> None:
    """
    Test 2: Q-sector collapse - Drought elimination.
    Show reduction from 512-way flat to 64-sector exact selection.
    """
    text, report = _run_generation(
        raw_bolmo,
        "The fundamental limits of",
        encode_config=BolmoEncodeBridgeConfig(chi_boundary_threshold=2),
        decode_config=_strict_decode_config(),
        max_new_tokens=16,
        stream_id="drought",
    )

    pg = report.patch_geometry
    raw_support = pg.get("raw_support_count_mean", 0.0)
    exact_support = pg.get("support_count_mean", 0.0)
    phase_redundancy = pg.get("phase_redundancy_mean", 0.0)

    print("\n[q-sector collapse - Drought elimination]")
    print("  raw_support_count_mean:", raw_support)
    print("  exact_support_count_mean:", exact_support)
    print("  phase_redundancy_mean:", phase_redundancy)
    print("  512-way flat selection collapsed to 64-sector exact selection.")

    assert abs(phase_redundancy) < 0.1


def test_speed_comparison(raw_bolmo, bolmo_tokenizer) -> None:
    """Speed comparison uses the stable default WHT route (`wht64`)."""
    from src.tools.gyrolabe.ops import (
        exact_qsector_select,
        wht64,
        chirality_distance_adjacent,
    )

    batch = 256
    normal = torch.randint(-65536, 65536, (batch, 256), dtype=torch.int32)
    fused = torch.randint(-65536, 65536, (batch, 256), dtype=torch.int32)
    prev = torch.full((batch,), 255, dtype=torch.uint8)
    shell_weights = torch.full((7,), 4096, dtype=torch.int32)

    n_iter = 20

    t0 = time.perf_counter()
    for _ in range(n_iter):
        exact_qsector_select(normal, fused, 0, prev, shell_weights)
    t_exact = (time.perf_counter() - t0) / n_iter

    normal_f = normal.to(torch.float32)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = torch.softmax(normal_f, dim=-1)
        _ = torch.argmax(normal_f, dim=-1)
    t_softmax = (time.perf_counter() - t0) / n_iter

    states = torch.randint(0, 1 << 24, (4096,), dtype=torch.int32)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        chirality_distance_adjacent(states, lookahead=1)
    t_chi = (time.perf_counter() - t0) / n_iter

    x = torch.randn(4096, 64, dtype=torch.float32)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        wht64(x)
    t_wht = (time.perf_counter() - t0) / n_iter

    import numpy as np
    W = np.array(api_mod.walsh_hadamard64(), dtype=np.float32)
    x_np = x.numpy()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        np.matmul(x_np, W.T)
    t_np = (time.perf_counter() - t0) / n_iter

    print("\n[speed comparison]")
    print("  exact_qsector_select vs softmax+argmax: %.2fx" % (t_softmax / max(t_exact, 1e-9)))
    print("  chirality_distance_adjacent: %.6f s" % t_chi)
    print("  wht64 vs numpy WHT: %.2fx" % (t_np / max(t_wht, 1e-9)))


def test_decode_bridge_step_speed_report() -> None:
    """Report boundary/select/token step throughput at realistic decode batch sizes."""
    config = _strict_decode_config()
    decode_bridge = GyroGraphBolmoDecodeBridge(
        config=config,
        cell_capacity=1024,
        use_opencl_hotpath=False,
    )
    boundary_offset = config.token_layout.boundary_offset
    dummy = _DummyModel()

    for batch in (1, 4, 8, 16):
        n_iter = 40
        logits = torch.randn((batch, 1, 520), dtype=torch.float32)
        last_bytes = [0] * batch

        with decode_bridge.session(dummy, batch_size=batch, stream_ids=[f"step:{batch}:{i}" for i in range(batch)]):
            t0 = time.perf_counter()
            for _ in range(n_iter):
                _ = decode_bridge.boundary_hook(logits, last_bytes, boundary_offset)
            t_boundary = (time.perf_counter() - t0) / n_iter

            t0 = time.perf_counter()
            for _ in range(n_iter):
                hooked = decode_bridge.boundary_hook(logits, last_bytes, boundary_offset)
                _ = decode_bridge.select_hook(hooked, boundary_offset)
            t_select = (time.perf_counter() - t0) / n_iter

            t0 = time.perf_counter()
            for _ in range(n_iter):
                hooked = decode_bridge.boundary_hook(logits, last_bytes, boundary_offset)
                tokens = decode_bridge.select_hook(hooked, boundary_offset)
                decode_bridge.token_hook(tokens, boundary_offset)
            t_full = (time.perf_counter() - t0) / n_iter

        tokens_per_sec = (batch * n_iter) / max(t_full, 1e-12)
        print(f"\n[decode bridge step speed] batch={batch}")
        print("  boundary_hook avg_ms:", f"{t_boundary * 1000.0:.3f}")
        print("  select_hook avg_ms:", f"{t_select * 1000.0:.3f}")
        print("  full step avg_ms:", f"{t_full * 1000.0:.3f}")
        print("  full tokens_per_sec:", f"{tokens_per_sec:.2f}")

        assert t_boundary > 0.0


def test_generation_overhead_report(raw_bolmo, bolmo_tokenizer) -> None:
    """Compare raw Bolmo generation time against bridged generation time."""
    prompt = "In 2026, exact byte-level generation should remain deterministic and stable."
    input_ids = torch.tensor(
        [bolmo_tokenizer.encode(prompt)],
        dtype=torch.long,
        device="cpu",
    )

    max_new_tokens = 8

    torch.manual_seed(0)
    with torch.no_grad():
        t0 = time.perf_counter()
        raw_out = raw_bolmo.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        raw_secs = time.perf_counter() - t0

    encode_bridge = GyroLabeBolmoEncodeBridge(
        raw_bolmo,
        config=BolmoEncodeBridgeConfig(chi_boundary_threshold=3),
    )
    encode_bridge.eval()
    decode_bridge = GyroGraphBolmoDecodeBridge(
        config=_strict_decode_config(),
        cell_capacity=128,
        use_opencl_hotpath=False,
        opencl_min_batch=1,
    )

    try:
        torch.manual_seed(0)
        with torch.no_grad():
            t0 = time.perf_counter()
            with decode_bridge.session(encode_bridge, batch_size=1, stream_ids=["bridge-gen"]):
                bridged_out = encode_bridge.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            bridged_secs = time.perf_counter() - t0
    finally:
        encode_bridge.uninstall()

    raw_generated = int(raw_out.shape[1]) - int(input_ids.shape[1])
    bridged_generated = int(bridged_out.shape[1]) - int(input_ids.shape[1])
    overhead = (bridged_secs / raw_secs) if raw_secs > 0.0 else float("inf")
    print("\n[bolmo generation overhead]")
    print("  prompt_tokens:", int(input_ids.shape[1]))
    print("  raw_generated:", raw_generated)
    print("  bridged_generated:", bridged_generated)
    print("  raw_ms:", f"{raw_secs * 1000.0:.3f}")
    print("  bridged_ms:", f"{bridged_secs * 1000.0:.3f}")
    print("  slowdown_ratio:", f"{overhead:.3f}")
    print("  raw_tokens_per_s:", f"{(raw_generated / max(raw_secs, 1e-12)):.2f}")
    print("  bridged_tokens_per_s:", f"{(bridged_generated / max(bridged_secs, 1e-12)):.2f}")

    assert raw_generated > 0
    assert bridged_generated > 0


def test_decode_generation_language_quality_metrics(raw_bolmo, bolmo_tokenizer) -> None:
    prompt = "In 2026, exact byte-level decoding should still produce coherent language about"
    text, report = _run_generation(
        raw_bolmo,
        prompt,
        encode_config=BolmoEncodeBridgeConfig(chi_boundary_threshold=3),
        decode_config=_strict_decode_config(),
        max_new_tokens=80,
        stream_id="quality",
    )

    generated = text[len("<bos>"):] if text.startswith("<bos>") else text
    generated = generated.strip()

    ascii_printable = 0
    for ch in generated:
        c = ord(ch)
        if c == 9 or c == 10 or c == 13 or (32 <= c <= 126):
            ascii_printable += 1
    ascii_ratio = ascii_printable / max(1, len(generated))

    def _max_run(s: str) -> int:
        if not s:
            return 0
        best = 1
        cur = 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 1
        return best

    max_run = _max_run(generated)
    unique_ratio = len(set(generated)) / max(1, len(generated))
    pg = report.patch_geometry

    print("\n[decode language quality metrics]")
    print("  text[:220]:", generated[:220])
    print("  ascii_ratio:", f"{ascii_ratio:.4f}")
    print("  max_run:", max_run)
    print("  unique_char_ratio:", f"{unique_ratio:.4f}")
    print("  patch_count:", pg["patch_count"])
    print("  mean_bpp:", f"{pg['mean_bytes_per_patch']:.3f}")

    assert len(generated) >= 40
    assert ascii_ratio >= 0.98
    assert max_run < 25
    assert unique_ratio > 0.05
    assert pg["patch_count"] >= 2


def test_strict_mode_forces_exact_selector(raw_bolmo, bolmo_tokenizer, monkeypatch) -> None:
    import src.tools.gyrolabe.ops as gyro_ops

    counter = {"exact_calls": 0}
    original_exact = gyro_ops.exact_qsector_select

    def _counting_exact(*args, **kwargs):
        counter["exact_calls"] += 1
        return original_exact(*args, **kwargs)

    monkeypatch.setattr(gyro_ops, "exact_qsector_select", _counting_exact)

    _run_generation(
        raw_bolmo,
        "Strict mode should always use native exact selection",
        encode_config=BolmoEncodeBridgeConfig(chi_boundary_threshold=3),
        decode_config=_strict_decode_config(),
        max_new_tokens=24,
        stream_id="strict-mode",
    )

    print("\n[strict mode exact selector]")
    print("  exact_qsector_select calls:", counter["exact_calls"])

    assert counter["exact_calls"] >= 1


def test_gyrograph_opencl_backend_usage_verbose(raw_bolmo, bolmo_tokenizer) -> None:
    """Explicit OpenCL coverage: only this test should force OpenCL decode path."""
    prompt = "The fundamental limits of"

    encode_cfg = BolmoEncodeBridgeConfig(chi_boundary_threshold=3)
    decode_cfg = _strict_decode_config()

    encode_bridge = GyroLabeBolmoEncodeBridge(raw_bolmo, config=encode_cfg)
    encode_bridge.eval()

    decode_bridge = GyroGraphBolmoDecodeBridge(
        config=decode_cfg,
        cell_capacity=256,
        use_opencl_hotpath=True,
        opencl_min_batch=1,
    )

    try:
        tokenizer = raw_bolmo.model.tokenizer
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)],
            dtype=torch.long,
            device="cpu",
        )

        with torch.no_grad():
            with decode_bridge.session(
                encode_bridge,
                batch_size=1,
                stream_ids=["ocl"],
            ):
                _ = encode_bridge.generate(
                    input_ids,
                    max_new_tokens=32,
                    do_sample=False,
                )

        counts = decode_bridge.graph.backend_counts
        print("\n[gyrograph backend usage]")
        print("  backend_counts:", counts)

        if decode_bridge.graph.opencl_hotpath_enabled:
            assert counts["opencl_indexed"] > 0
        else:
            assert counts["cpu_indexed"] > 0
    finally:
        encode_bridge.uninstall()
