from __future__ import annotations

import pytest
import torch

from src.tools.gyrolabe.bridges.bolmo_config import (
    BolmoEncodeBridgeConfig,
    load_base_bolmo,
)
from src.tools.gyrolabe.bridges.encode import GyroLabeBolmoEncodeBridge


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


def _prefill_probe(
    raw_bolmo,
    prompt: str,
    config: BolmoEncodeBridgeConfig,
):
    bridge = GyroLabeBolmoEncodeBridge(raw_bolmo, config=config)
    bridge.eval()
    try:
        tokenizer = raw_bolmo.model.tokenizer
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)],
            dtype=torch.long,
            device="cpu",
        )
        with torch.no_grad():
            fields, boundary_field = bridge.prefill_probe(input_ids)
        return fields, boundary_field
    finally:
        bridge.uninstall()


def test_gyrolabe_encode_fields_verbose(raw_bolmo, bolmo_tokenizer) -> None:
    prompt = "The QuBEC climate is finite, shell-exact, and byte-native."

    fields, boundary_field = _prefill_probe(
        raw_bolmo,
        prompt,
        BolmoEncodeBridgeConfig(
            embedding_scale=1.0,
            boundary_scale=1.0,
            boundary_mode="observe",
            target_bytes_per_patch=None,
        ),
    )

    print("\n[gyrolabe encode]")
    print("  prompt:", repr(prompt))
    print("  shape:", tuple(fields.canonical_bytes.shape))
    print("  valid_count:", fields.valid_count)
    print("  q_hist64_nonzero:", [(i, int(v)) for i, v in enumerate(fields.q_hist64.tolist()) if int(v) > 0][:16])
    print("  family_hist4:", [int(x) for x in fields.family_hist4.tolist()])
    print("  micro_hist64_nonzero:", [(i, int(v)) for i, v in enumerate(fields.micro_hist64.tolist()) if int(v) > 0][:16])
    print("  shell_hist7:", [int(x) for x in fields.shell_hist7.tolist()])
    print("  q_weight_hist7:", [int(x) for x in fields.q_weight_hist7.tolist()])
    print("  bit_excitation6:", [int(x) for x in fields.bit_excitation6.tolist()])

    assert fields.valid_count > 0
    assert int(fields.q_hist64.sum().item()) == fields.valid_count
    assert int(fields.family_hist4.sum().item()) == fields.valid_count
    assert int(fields.micro_hist64.sum().item()) == fields.valid_count
    assert int(fields.shell_hist7.sum().item()) == fields.valid_count
    assert int(fields.q_weight_hist7.sum().item()) == fields.valid_count

    # bit_excitation counts how often each q-bit is present across valid positions
    assert fields.bit_excitation6.shape == (6,)
    assert all(int(x) >= 0 for x in fields.bit_excitation6.tolist())

    assert boundary_field is not None
    print("  boundary_count:", boundary_field.boundary_count)
    print("  patch_count:", boundary_field.patch_count)
    print("  patch_lengths:", boundary_field.patch_lengths)
    print("  mean_bytes_per_patch:", boundary_field.mean_bytes_per_patch)
    print("  attn_proxy:", boundary_field.attn_proxy)
    print("  kv_proxy:", boundary_field.kv_proxy)


def test_gyrolabe_boundary_control_verbose(raw_bolmo, bolmo_tokenizer) -> None:
    prompts = [
        "The flowerbed was damp.",
        "The flowers were damp.",
        "The fundamental limits of ",
    ]

    observe_cfg = BolmoEncodeBridgeConfig(
        embedding_scale=0.0,
        boundary_scale=0.0,
        boundary_mode="observe",
        target_bytes_per_patch=None,
    )

    hybrid_cfg = BolmoEncodeBridgeConfig(
        embedding_scale=1.0,
        boundary_scale=1.0,
        boundary_mode="hybrid",
        boundary_cosine_weight=0.35,
        boundary_structural_weight=0.65,
        target_bytes_per_patch=6.0,
        target_patch_gain=0.90,
    )

    exact_cfg = BolmoEncodeBridgeConfig(
        embedding_scale=1.0,
        boundary_scale=1.0,
        boundary_mode="exact",
        boundary_cosine_weight=0.0,
        boundary_structural_weight=1.0,
        target_bytes_per_patch=6.0,
        target_patch_gain=0.90,
    )

    configs = {
        "observe": observe_cfg,
        "hybrid": hybrid_cfg,
        "exact": exact_cfg,
    }

    results: dict[str, dict[str, float | int]] = {}

    for name, cfg in configs.items():
        total_patch_count = 0
        total_boundary_count = 0
        total_mean_bpp = 0.0
        total_prob_delta = 0.0
        total_attn_proxy = 0
        total_kv_proxy = 0
        boundary_q_nonzero_total = 0
        boundary_micro_nonzero_total = 0

        for prompt in prompts:
            fields, bf = _prefill_probe(raw_bolmo, prompt, cfg)
            assert bf is not None

            valid = bf.valid_mask.to(torch.bool)
            cosine = bf.cosine_boundary_probs.reshape(-1)[valid.reshape(-1)].to(torch.float32)
            combined = bf.combined_boundary_probs.reshape(-1)[valid.reshape(-1)].to(torch.float32)

            mean_prob_delta = (
                torch.mean(torch.abs(combined - cosine)).item()
                if cosine.numel() > 0
                else 0.0
            )

            total_patch_count += len(bf.patch_lengths)
            total_boundary_count += int(bf.boundary_count)
            total_mean_bpp += float(bf.mean_bytes_per_patch)
            total_prob_delta += float(mean_prob_delta)
            total_attn_proxy += int(bf.attn_proxy)
            total_kv_proxy += int(bf.kv_proxy)
            boundary_q_nonzero_total += int((bf.boundary_q_hist64 > 0).sum().item())
            boundary_micro_nonzero_total += int((bf.boundary_micro_hist64 > 0).sum().item())

        results[name] = {
            "total_patch_count": total_patch_count,
            "total_boundary_count": total_boundary_count,
            "mean_bpp": total_mean_bpp / len(prompts),
            "mean_prob_delta": total_prob_delta / len(prompts),
            "total_attn_proxy": total_attn_proxy,
            "total_kv_proxy": total_kv_proxy,
            "boundary_q_nonzero_total": boundary_q_nonzero_total,
            "boundary_micro_nonzero_total": boundary_micro_nonzero_total,
        }

    print("\n[gyrolabe boundary control]")
    for name in ("observe", "hybrid", "exact"):
        r = results[name]
        regime = "stress-limit" if name == "exact" else "operative"
        print(
            f"  {name} [{regime}]: "
            f"patch_count={r['total_patch_count']} "
            f"boundary_count={r['total_boundary_count']} "
            f"mean_bpp={r['mean_bpp']:.3f} "
            f"attn_proxy={r['total_attn_proxy']} "
            f"kv_proxy={r['total_kv_proxy']} "
            f"mean_prob_delta={r['mean_prob_delta']:.6f} "
            f"boundary_q_nonzero_total={r['boundary_q_nonzero_total']} "
            f"boundary_micro_nonzero_total={r['boundary_micro_nonzero_total']}"
        )

    # 1. Observe mode must leave the boundary law unchanged
    assert results["observe"]["mean_prob_delta"] == pytest.approx(0.0, abs=1e-7)

    # 2. Hybrid / exact must actually intervene on the root boundary law
    assert results["hybrid"]["mean_prob_delta"] > 0.0
    assert results["exact"]["mean_prob_delta"] > 0.0

    # 3. Predictor-boundary histograms must now reflect actual predicted boundaries
    assert results["observe"]["boundary_q_nonzero_total"] > 0
    assert results["observe"]["boundary_micro_nonzero_total"] > 0

    # 4. Structural control must shift patch geometry toward coarser patching
    assert results["hybrid"]["mean_bpp"] >= results["observe"]["mean_bpp"]
    assert results["exact"]["mean_bpp"] >= results["observe"]["mean_bpp"]

    assert results["hybrid"]["total_patch_count"] <= results["observe"]["total_patch_count"]
    assert results["exact"]["total_patch_count"] <= results["observe"]["total_patch_count"]

    # Coarser patching must reduce attention and KV growth proxies
    assert results["hybrid"]["total_attn_proxy"] <= results["observe"]["total_attn_proxy"]
    assert results["exact"]["total_attn_proxy"] <= results["observe"]["total_attn_proxy"]

    assert results["hybrid"]["total_kv_proxy"] <= results["observe"]["total_kv_proxy"]
    assert results["exact"]["total_kv_proxy"] <= results["observe"]["total_kv_proxy"]
