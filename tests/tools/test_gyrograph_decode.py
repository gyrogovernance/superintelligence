from __future__ import annotations

import pytest
import torch

from src.tools.gyrolabe.bridges.bolmo_config import (
    BolmoEncodeBridgeConfig,
    load_base_bolmo,
)
from src.tools.gyrolabe.bridges.encode import GyroLabeBolmoEncodeBridge
from src.tools.gyrograph.bridges.bolmo_config import BolmoDecodeBridgeConfig
from src.tools.gyrograph.bridges.decode import GyroGraphBolmoDecodeBridge


class _DummyModel:
    pass


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
    encode_bridge = GyroLabeBolmoEncodeBridge(raw_bolmo, config=encode_config)
    encode_bridge.eval()

    decode_bridge = GyroGraphBolmoDecodeBridge(
        config=decode_config,
        cell_capacity=512,
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


def test_gyrograph_decode_synthetic_selector_verbose() -> None:
    logits = torch.full((1, 520), -50.0, dtype=torch.float32)

    # Content 66 has the best flat score.
    logits[0, 4 + 66] = 9.30

    # Content 65 has the best paired content score after logaddexp(normal, fused).
    logits[0, 4 + 65] = 9.00
    logits[0, 260 + 4 + 65] = 9.20

    flat_bridge = GyroGraphBolmoDecodeBridge(
        config=BolmoDecodeBridgeConfig(
            control_mode="observe",
            selection_mode="flat",
        ),
        cell_capacity=16,
    )
    paired_bridge = GyroGraphBolmoDecodeBridge(
        config=BolmoDecodeBridgeConfig(
            control_mode="observe",
            selection_mode="paired",
            phase_threshold=0.0,
        ),
        cell_capacity=16,
    )

    dummy = _DummyModel()

    with flat_bridge.session(dummy, batch_size=1, stream_ids=["flat"]):
        flat_token = int(flat_bridge.select_hook(logits.clone(), 260)[0].item())

    with paired_bridge.session(dummy, batch_size=1, stream_ids=["paired"]):
        paired_token = int(paired_bridge.select_hook(logits.clone(), 260)[0].item())

    print("\n[gyrograph synthetic selector]")
    print("  flat_token:", flat_token)
    print("  paired_token:", paired_token)

    assert flat_token == (4 + 66)
    assert paired_token == (260 + 4 + 65)


def test_gyrograph_decode_real_baseline_vs_controlled(raw_bolmo, bolmo_tokenizer) -> None:
    prompts = [
        "The state",
        "The flowerbed was",
        "The fundamental limits of",
    ]

    baseline_encode = BolmoEncodeBridgeConfig(
        embedding_scale=0.0,
        boundary_scale=0.0,
        boundary_mode="observe",
        target_bytes_per_patch=None,
    )
    controlled_encode = BolmoEncodeBridgeConfig(
        embedding_scale=1.0,
        boundary_scale=1.0,
        boundary_mode="hybrid",
        boundary_cosine_weight=0.35,
        boundary_structural_weight=0.65,
        target_bytes_per_patch=6.0,
        target_patch_gain=0.90,
    )

    baseline_decode = BolmoDecodeBridgeConfig(
        control_mode="observe",
        selection_mode="flat",
        phase_threshold=0.0,
    )
    controlled_decode = BolmoDecodeBridgeConfig(
        control_mode="gauge_damp",
        selection_mode="paired",
        phase_threshold=0.0,
        phase_hysteresis=0.75,
        application_phase_damping=0.90,
        database_sector_bonus=0.0,
    )

    any_text_changed = False

    base_flip_vals = []
    ctrl_flip_vals = []

    base_bpp_vals = []
    ctrl_bpp_vals = []

    base_phase_redundancy_vals = []
    ctrl_phase_redundancy_vals = []

    base_attn_proxy_vals = []
    ctrl_attn_proxy_vals = []

    base_kv_proxy_vals = []
    ctrl_kv_proxy_vals = []

    base_support_count_vals = []
    ctrl_support_count_vals = []

    print("\n[gyrograph real decode climate]")

    for idx, prompt in enumerate(prompts):
        base_text, base_report = _run_generation(
            raw_bolmo,
            prompt,
            encode_config=baseline_encode,
            decode_config=baseline_decode,
            max_new_tokens=24,
            stream_id=f"base:{idx}",
        )
        ctrl_text, ctrl_report = _run_generation(
            raw_bolmo,
            prompt,
            encode_config=controlled_encode,
            decode_config=controlled_decode,
            max_new_tokens=24,
            stream_id=f"ctrl:{idx}",
        )

        base_pg = base_report.patch_geometry
        ctrl_pg = ctrl_report.patch_geometry

        base_flip_vals.append(float(base_pg["gauge_flip_rate"]))
        ctrl_flip_vals.append(float(ctrl_pg["gauge_flip_rate"]))

        base_bpp_vals.append(float(base_pg["mean_bytes_per_patch"]))
        ctrl_bpp_vals.append(float(ctrl_pg["mean_bytes_per_patch"]))

        base_phase_redundancy_vals.append(float(base_pg["phase_redundancy_mean"]))
        ctrl_phase_redundancy_vals.append(float(ctrl_pg["phase_redundancy_mean"]))

        base_attn_proxy_vals.append(float(base_pg["attn_proxy"]))
        ctrl_attn_proxy_vals.append(float(ctrl_pg["attn_proxy"]))

        base_kv_proxy_vals.append(float(base_pg["kv_proxy"]))
        ctrl_kv_proxy_vals.append(float(ctrl_pg["kv_proxy"]))

        base_support_count_vals.append(float(base_pg["support_count_mean"]))
        ctrl_support_count_vals.append(float(ctrl_pg["support_count_mean"]))

        text_changed = base_text != ctrl_text
        any_text_changed = any_text_changed or text_changed

        print(f"  prompt[{idx}]: {prompt!r}")
        print("    baseline   patch_count:", base_pg["patch_count"])
        print("    baseline   attn_proxy:", base_pg["attn_proxy"])
        print("    baseline   kv_proxy:", base_pg["kv_proxy"])
        print("    baseline   mean_bytes_per_patch:", base_pg["mean_bytes_per_patch"])
        print("    baseline   gauge_flip_rate:", base_pg["gauge_flip_rate"])
        print("    baseline   support_ratio_mean:", base_pg["support_ratio_mean"])
        print("    baseline   raw_support_ratio_mean:", base_pg["raw_support_ratio_mean"])
        print("    baseline   support_count_mean:", base_pg["support_count_mean"])
        print("    baseline   raw_support_count_mean:", base_pg["raw_support_count_mean"])
        print("    baseline   phase_redundancy_mean:", base_pg["phase_redundancy_mean"])
        print("    controlled patch_count:", ctrl_pg["patch_count"])
        print("    controlled attn_proxy:", ctrl_pg["attn_proxy"])
        print("    controlled kv_proxy:", ctrl_pg["kv_proxy"])
        print("    controlled mean_bytes_per_patch:", ctrl_pg["mean_bytes_per_patch"])
        print("    controlled gauge_flip_rate:", ctrl_pg["gauge_flip_rate"])
        print("    controlled support_ratio_mean:", ctrl_pg["support_ratio_mean"])
        print("    controlled raw_support_ratio_mean:", ctrl_pg["raw_support_ratio_mean"])
        print("    controlled support_count_mean:", ctrl_pg["support_count_mean"])
        print("    controlled raw_support_count_mean:", ctrl_pg["raw_support_count_mean"])
        print("    controlled phase_redundancy_mean:", ctrl_pg["phase_redundancy_mean"])
        print("    baseline   text:", base_text[:96])
        print("    controlled text:", ctrl_text[:96])

        assert base_report.network["order"]["samples"] > 0
        assert base_report.database["order"]["samples"] > 0
        assert base_report.application["order"]["samples"] > 0
        assert ctrl_report.network["order"]["samples"] > 0
        assert ctrl_report.database["order"]["samples"] > 0
        assert ctrl_report.application["order"]["samples"] > 0

    base_flip_mean = sum(base_flip_vals) / len(base_flip_vals)
    ctrl_flip_mean = sum(ctrl_flip_vals) / len(ctrl_flip_vals)
    base_bpp_mean = sum(base_bpp_vals) / len(base_bpp_vals)
    ctrl_bpp_mean = sum(ctrl_bpp_vals) / len(ctrl_bpp_vals)
    base_phase_redundancy_mean = sum(base_phase_redundancy_vals) / len(base_phase_redundancy_vals)
    ctrl_phase_redundancy_mean = sum(ctrl_phase_redundancy_vals) / len(ctrl_phase_redundancy_vals)

    base_attn_proxy_mean = sum(base_attn_proxy_vals) / len(base_attn_proxy_vals)
    ctrl_attn_proxy_mean = sum(ctrl_attn_proxy_vals) / len(ctrl_attn_proxy_vals)

    base_kv_proxy_mean = sum(base_kv_proxy_vals) / len(base_kv_proxy_vals)
    ctrl_kv_proxy_mean = sum(ctrl_kv_proxy_vals) / len(ctrl_kv_proxy_vals)

    base_support_count_mean = sum(base_support_count_vals) / len(base_support_count_vals)
    ctrl_support_count_mean = sum(ctrl_support_count_vals) / len(ctrl_support_count_vals)

    print("  baseline   gauge_flip_rate mean:", base_flip_mean)
    print("  controlled gauge_flip_rate mean:", ctrl_flip_mean)
    print("  baseline   mean_bpp mean:", base_bpp_mean)
    print("  controlled mean_bpp mean:", ctrl_bpp_mean)
    print("  baseline   phase_redundancy_mean:", base_phase_redundancy_mean)
    print("  controlled phase_redundancy_mean:", ctrl_phase_redundancy_mean)
    print("  baseline   attn_proxy mean:", base_attn_proxy_mean)
    print("  controlled attn_proxy mean:", ctrl_attn_proxy_mean)
    print("  baseline   kv_proxy mean:", base_kv_proxy_mean)
    print("  controlled kv_proxy mean:", ctrl_kv_proxy_mean)
    print("  baseline   support_count_mean:", base_support_count_mean)
    print("  controlled support_count_mean:", ctrl_support_count_mean)

    # Decode quotient selection + phase hysteresis should reduce gauge turbulence on average.
    assert ctrl_flip_mean <= base_flip_mean

    # Encode boundary control should coarsen patching on average.
    assert ctrl_bpp_mean >= base_bpp_mean

    # Flat 512-way competition must contain phase redundancy.
    assert base_phase_redundancy_mean > 0.0

    # Controlled paired selection must not destroy quotient reduction.
    assert ctrl_phase_redundancy_mean > 0.0

    # Coarser patching must reduce global attention / retrieval pressure proxies.
    assert ctrl_attn_proxy_mean <= base_attn_proxy_mean
    assert ctrl_kv_proxy_mean <= base_kv_proxy_mean

    # Quotient selection should reduce surviving content competition on average.
    assert ctrl_support_count_mean <= base_support_count_mean

    # The controlled law must materially alter actual continuation behavior.
    assert any_text_changed
