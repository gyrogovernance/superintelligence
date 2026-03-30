#!/usr/bin/env python3
"""
bench_qubec_bolmo_full.py

Full end-to-end QuBEC Bolmo/OLMo harness:

- loads Bolmo
- converts the full model to GyroMatMul
- runs a GyroMatMul-only generation pass
- wraps the converted model with GyroLabe encode bridge
- attaches GyroGraph decode bridge
- runs full generation through the combined path
- emits encode-side bath and decode-side QuBEC climate / resonance summaries

This is a wiring harness only.
It does NOT invent new decode logic.
It only connects existing code paths end to end.

Defaults are chosen to run without flags.
Override by editing the constants below or using the small env vars.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ---------------------------------------------------------------------
# Existing project imports only
# ---------------------------------------------------------------------

from src.tools.gyrolabe import ops as gyrolabe_ops
from src.tools.gyrolabe.bridges import (
    DEFAULT_BOLMO_MODEL_PATH,
    BolmoEncodeBridgeConfig,
    GyroLabeBolmoEncodeBridge,
    configure_bolmo_offline_loading,
    load_base_bolmo,
)
from src.tools.gyrolabe.bridges.encode import estimate_q_bath
from src.tools.gyrolabe.gyromatmul import convert_bolmo

from src.tools.gyrograph import GyroGraph, ResonanceProfile
from src.tools.gyrograph.bridges import (
    BolmoDecodeBridgeConfig,
    GyroGraphBolmoDecodeBridge,
    compute_qubec_climate,
)

# tokenizer helper used by your existing bench path if available
try:
    from tests.tools.conftest import bolmo_tokenizer_from_model  # type: ignore
except Exception:  # pragma: no cover
    bolmo_tokenizer_from_model = None

# fallback tokenizer load
from transformers import AutoTokenizer


# ---------------------------------------------------------------------
# Defaults: no flag-heavy CLI; edit here or set env vars
# ---------------------------------------------------------------------

MODEL_PATH = Path(os.environ.get("QUBEC_BOLMO_MODEL_PATH", str(DEFAULT_BOLMO_MODEL_PATH)))
MAX_NEW_TOKENS = int(os.environ.get("QUBEC_MAX_NEW_TOKENS", "12"))
WARMUP_RUNS = int(os.environ.get("QUBEC_WARMUP_RUNS", "1"))
MEASURE_RUNS = int(os.environ.get("QUBEC_MEASURE_RUNS", "1"))
TORCH_THREADS = int(os.environ.get("QUBEC_TORCH_THREADS", str(os.cpu_count() or 1)))

# Use multiple prompts by default so the decode bridge runs true multi-stream.
PROMPTS = [
    "The weather today is",
    "In one sentence, explain why deterministic replay matters.",
    "The following Python function is slow because",
    "A compact exact-state machine can improve generation by",
]

# Convert entire Bolmo/OLMo composite model to GyroMatMul.
GYROMATMUL_BITS = 12

# Encode bridge: keep exact boundary logic active.
ENCODE_CONFIG = BolmoEncodeBridgeConfig.wiring_quality_approved(
    strict_cpu=True,
    chi_boundary_threshold=2,
)

# Decode bridge: use existing benchmark quality control, not a new mode.
DECODE_CONFIG = BolmoDecodeBridgeConfig.benchmark_quality_control(
    control_mode="full",
    selection_mode="paired",
    proof_mode=True,
    do_sample=False,
)

# GyroGraph runtime config
GRAPH_PROFILE = ResonanceProfile.CHIRALITY
GRAPH_CAPACITY = 4096


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _fmt_rate(tokens: int, elapsed_ms: float) -> str:
    if elapsed_ms <= 0:
        return "0.00 tok/s"
    return f"{(tokens / (elapsed_ms / 1000.0)):.2f} tok/s"


def _set_threads() -> None:
    torch.set_num_threads(TORCH_THREADS)
    try:
        torch.set_num_interop_threads(TORCH_THREADS)
    except RuntimeError:
        # harmless if already initialized
        pass
    gyrolabe_ops.initialize_native()
    try:
        gyrolabe_ops.set_native_threads(TORCH_THREADS)
    except Exception:
        pass


def _load_tokenizer(model: Any, model_path: Path):
    if bolmo_tokenizer_from_model is not None:
        try:
            tok = bolmo_tokenizer_from_model(model)
            if tok is not None:
                if tok.pad_token is None and tok.eos_token is not None:
                    tok.pad_token = tok.eos_token
                return tok
        except Exception:
            pass

    configure_bolmo_offline_loading(model_path)
    tok = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def _tokenize_prompts(tokenizer: Any, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    input_ids = enc.input_ids.cpu()
    attention_mask = enc.attention_mask.cpu()
    return input_ids, attention_mask


def _decode_texts(tokenizer: Any, output_ids: torch.Tensor) -> list[str]:
    return list(tokenizer.batch_decode(output_ids, skip_special_tokens=True))


@contextmanager
def _encode_decode_adapter(encode_bridge: GyroLabeBolmoEncodeBridge):
    """
    Current harness glue:

    GyroGraphBolmoDecodeBridge attaches hooks to `model.base_model`
    but later tries to read `last_encoded_fields()` and `set_source_m2()`,
    which live on GyroLabeBolmoEncodeBridge.

    We do not invent anything here.
    We simply expose the existing encode-bridge methods on base_model
    for the duration of the run.
    """
    base = encode_bridge.base_model

    sentinel = object()
    old_last = getattr(base, "last_encoded_fields", sentinel)
    old_set = getattr(base, "set_source_m2", sentinel)

    setattr(base, "last_encoded_fields", encode_bridge.last_encoded_fields)
    setattr(base, "set_source_m2", encode_bridge.set_source_m2)

    try:
        yield
    finally:
        if old_last is sentinel:
            try:
                delattr(base, "last_encoded_fields")
            except Exception:
                pass
        else:
            setattr(base, "last_encoded_fields", old_last)

        if old_set is sentinel:
            try:
                delattr(base, "set_source_m2")
            except Exception:
                pass
        else:
            setattr(base, "set_source_m2", old_set)


def _summarize_q_bath(encode_bridge: GyroLabeBolmoEncodeBridge) -> dict[str, Any] | None:
    fields = encode_bridge.last_encoded_fields()
    if fields is None:
        return None

    q_bath = estimate_q_bath(fields)
    return {
        "valid_count": int(fields.valid_mask.sum().item()),
        "q_weight_hist7": [int(x) for x in fields.q_weight_hist7.tolist()],
        "shell_hist7": [int(x) for x in fields.shell_hist7.tolist()],
        "family_hist4": [int(x) for x in fields.family_hist4.tolist()],
        "rho_bath": float(q_bath.rho_bath),
        "eta_bath": float(q_bath.eta_bath),
        "lam_bath": float(q_bath.lam_bath),
        "m_bath": float(q_bath.m_bath),
        "m2_eq_bath": float(q_bath.m2_eq_bath),
        "eta_vec_bath": [float(x) for x in q_bath.eta_vec_bath],
        "q_weight_distribution": [float(x) for x in q_bath.q_weight_distribution],
        "shell_law_bath": [float(x) for x in q_bath.shell_law_bath],
    }


def _cell_climate(graph: GyroGraph, cell_id: int) -> dict[str, Any]:
    chi_hist = graph._chi_hist64[cell_id]
    shell_hist = graph._shell_hist7[cell_id]
    family_hist = graph._family_hist4[cell_id]
    samples = int(chi_hist.sum())

    climate = compute_qubec_climate(
        chi_hist64=chi_hist,
        shell_hist7=shell_hist,
        samples=samples,
        family_hist4=family_hist,
    )

    return {
        "samples": samples,
        "chi_hist64_nonzero": int((chi_hist > 0).sum()),
        "shell_hist7": [int(x) for x in shell_hist.tolist()],
        "family_hist4": [int(x) for x in family_hist.tolist()],
        "rho": float(climate.rho),
        "m": float(climate.m),
        "eta": float(climate.eta),
        "lam": float(climate.lam),
        "M2": float(climate.M2),
        "M2_eq": float(climate.M2_eq),
        "shell_spectrum": [float(x) for x in climate.shell_spectrum],
        "eta_vec": [float(x) for x in climate.eta_vec],
        "gauge_spectrum": None if climate.gauge_spectrum is None else [float(x) for x in climate.gauge_spectrum],
    }


def _print_domain_summary(name: str, report_block: dict[str, Any], climate_block: dict[str, Any]) -> None:
    record = report_block["record"]
    print(f"    [{name}]")
    print(
        f"      shell={record.shell} chi6={record.chi6} "
        f"omega_sig={record.omega_sig} resonance={record.current_resonance}"
    )
    print(
        f"      rho={climate_block['rho']:.4f} "
        f"m={climate_block['m']:.4f} "
        f"eta={climate_block['eta']:.4f} "
        f"lam={climate_block['lam']:.4f} "
        f"M2={climate_block['M2']:.2f}"
    )
    if climate_block["gauge_spectrum"] is not None:
        g = climate_block["gauge_spectrum"]
        print(
            f"      gauge_spectrum=[{g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f}, {g[3]:.4f}]"
        )


# ---------------------------------------------------------------------
# Generation runners
# ---------------------------------------------------------------------

def _run_generate_plain(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    warmup_runs: int,
    measure_runs: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    times_ms: list[float] = []
    last_output = None

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        for _ in range(measure_runs):
            t0 = time.perf_counter()
            out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            times_ms.append(_ms(t0))
            last_output = out

    assert last_output is not None
    new_tokens = int(last_output.shape[-1] - input_ids.shape[-1])

    return {
        "times_ms": times_ms,
        "mean_ms": mean(times_ms) if times_ms else 0.0,
        "output_ids": last_output,
        "new_tokens_per_stream": new_tokens,
        "total_new_tokens": new_tokens * int(input_ids.shape[0]),
    }


def _run_generate_full_qubec(
    encode_bridge: GyroLabeBolmoEncodeBridge,
    decode_bridge: GyroGraphBolmoDecodeBridge,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    prompts: list[str],
    warmup_runs: int,
    measure_runs: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    stream_ids = [f"qubec:{i}" for i in range(len(prompts))]
    times_ms: list[float] = []
    last_output = None
    last_reports: dict[str, Any] = {}
    last_selection_counts: dict[str, int] = {}
    last_patch_geometry = None
    last_q_bath = None
    last_graph_backend_counts = None
    last_stream_climates: dict[str, Any] = {}

    with _encode_decode_adapter(encode_bridge):
        for _ in range(warmup_runs):
            with decode_bridge.session(
                encode_bridge,
                batch_size=int(input_ids.shape[0]),
                stream_ids=stream_ids,
            ):
                with torch.no_grad():
                    _ = encode_bridge.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )

        for _ in range(measure_runs):
            with decode_bridge.session(
                encode_bridge,
                batch_size=int(input_ids.shape[0]),
                stream_ids=stream_ids,
            ):
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = encode_bridge.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                elapsed_ms = _ms(t0)

                reports = decode_bridge.emit_all_stream_reports()
                selection_counts = decode_bridge.last_selection_counts()
                patch_geometry = encode_bridge.get_last_patch_geometry()
                q_bath = _summarize_q_bath(encode_bridge)
                graph_backend_counts = decode_bridge.graph.backend_counts

                # Full climate with family_hist4, using current graph state directly.
                stream_climates: dict[str, Any] = {}
                for sid in stream_ids:
                    st = decode_bridge._streams[sid]
                    stream_climates[sid] = {
                        "network": _cell_climate(decode_bridge.graph, st.network_cell),
                        "database": _cell_climate(decode_bridge.graph, st.database_cell),
                        "application": _cell_climate(decode_bridge.graph, st.application_cell),
                    }

                times_ms.append(elapsed_ms)
                last_output = out
                last_reports = reports
                last_selection_counts = selection_counts
                last_patch_geometry = patch_geometry
                last_q_bath = q_bath
                last_graph_backend_counts = graph_backend_counts
                last_stream_climates = stream_climates

    assert last_output is not None
    new_tokens = int(last_output.shape[-1] - input_ids.shape[-1])

    return {
        "times_ms": times_ms,
        "mean_ms": mean(times_ms) if times_ms else 0.0,
        "output_ids": last_output,
        "new_tokens_per_stream": new_tokens,
        "total_new_tokens": new_tokens * int(input_ids.shape[0]),
        "reports": last_reports,
        "selection_counts": last_selection_counts,
        "patch_geometry": last_patch_geometry,
        "q_bath": last_q_bath,
        "graph_backend_counts": last_graph_backend_counts,
        "stream_climates": last_stream_climates,
        "encode_capabilities": encode_bridge.runtime_capabilities(),
        "decode_capabilities": decode_bridge.runtime_capabilities(),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=== Full QuBEC Bolmo/OLMo Harness ===")
    print(f"model_path={MODEL_PATH}")
    print(f"prompts={len(PROMPTS)}")
    print(f"max_new_tokens={MAX_NEW_TOKENS}")
    print(f"warmup_runs={WARMUP_RUNS} measure_runs={MEASURE_RUNS}")
    print(f"gyromatmul_bits={GYROMATMUL_BITS}")
    print()

    _set_threads()

    # -------------------------------------------------------------
    # 1) Load raw Bolmo exterior / OLMo interior composite model
    # -------------------------------------------------------------
    print("[1/5] loading Bolmo...")
    model = load_base_bolmo(
        model_path=MODEL_PATH,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    tokenizer = _load_tokenizer(model, MODEL_PATH)
    input_ids, attention_mask = _tokenize_prompts(tokenizer, PROMPTS)

    print(f"  input_ids shape={tuple(input_ids.shape)}")
    print(f"  native_available={gyrolabe_ops.native_available()}")
    print()

    # -------------------------------------------------------------
    # 2) Convert full model to GyroMatMul
    # -------------------------------------------------------------
    print("[2/5] converting full Bolmo/OLMo composite to GyroMatMul...")
    convert_bolmo(model, n_bits=GYROMATMUL_BITS)
    model.eval()
    print("  conversion complete")
    print()

    # -------------------------------------------------------------
    # 3) Baseline on current generation wiring (GyroMatMul-only)
    # -------------------------------------------------------------
    print("[3/5] running GyroMatMul-only generation path...")
    plain = _run_generate_plain(
        model,
        input_ids,
        attention_mask,
        warmup_runs=WARMUP_RUNS,
        measure_runs=MEASURE_RUNS,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    plain_texts = _decode_texts(tokenizer, plain["output_ids"])

    print(
        f"  mean_ms={plain['mean_ms']:.2f} "
        f"total_new_tokens={plain['total_new_tokens']} "
        f"rate={_fmt_rate(plain['total_new_tokens'], plain['mean_ms'])}"
    )
    print()

    # -------------------------------------------------------------
    # 4) Full QuBEC path: GyroLabe encode + GyroGraph decode
    # -------------------------------------------------------------
    print("[4/5] building full QuBEC generation path...")
    encode_bridge = GyroLabeBolmoEncodeBridge(
        base_model=model,
        config=ENCODE_CONFIG,
    )

    graph = GyroGraph(
        cell_capacity=GRAPH_CAPACITY,
        profile=GRAPH_PROFILE,
        use_native_hotpath=True,
        use_opencl_hotpath=False,
        enable_ingest_log=False,
    )

    decode_bridge = GyroGraphBolmoDecodeBridge(
        graph=graph,
        config=DECODE_CONFIG,
        use_native_hotpath=True,
        use_opencl_hotpath=False,
    )

    full = _run_generate_full_qubec(
        encode_bridge,
        decode_bridge,
        input_ids,
        attention_mask,
        prompts=PROMPTS,
        warmup_runs=WARMUP_RUNS,
        measure_runs=MEASURE_RUNS,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    full_texts = _decode_texts(tokenizer, full["output_ids"])

    print(
        f"  mean_ms={full['mean_ms']:.2f} "
        f"total_new_tokens={full['total_new_tokens']} "
        f"rate={_fmt_rate(full['total_new_tokens'], full['mean_ms'])}"
    )
    if plain["mean_ms"] > 0 and full["mean_ms"] > 0:
        ratio = plain["mean_ms"] / full["mean_ms"]
        print(f"  speed ratio vs GyroMatMul-only path: {ratio:.3f}x")
    print()

    # -------------------------------------------------------------
    # 5) Reports
    # -------------------------------------------------------------
    print("[5/5] reporting coverage and climate...")
    print("  encode capabilities:", full["encode_capabilities"])
    print("  decode capabilities:", full["decode_capabilities"])
    print("  graph backend counts:", full["graph_backend_counts"])
    print("  selection counts:", full["selection_counts"])
    print("  encode patch geometry:", full["patch_geometry"])
    print()

    q_bath = full["q_bath"]
    if q_bath is not None:
        print("  encode q-bath summary")
        print(
            f"    valid_count={q_bath['valid_count']} "
            f"rho_bath={q_bath['rho_bath']:.4f} "
            f"eta_bath={q_bath['eta_bath']:.4f} "
            f"lam_bath={q_bath['lam_bath']:.4f} "
            f"m2_eq_bath={q_bath['m2_eq_bath']:.2f}"
        )
        print(f"    q_weight_hist7={q_bath['q_weight_hist7']}")
        print(f"    family_hist4={q_bath['family_hist4']}")
        print()

    print("  outputs")
    for i, prompt in enumerate(PROMPTS):
        print(f"    stream {i}")
        print(f"      prompt: {prompt!r}")
        print(f"      gyromatmul-only: {plain_texts[i]!r}")
        print(f"      full-qubec:      {full_texts[i]!r}")
        print()

    print("  per-stream QuBEC decode summaries")
    for sid, report in full["reports"].items():
        print(f"  stream_id={sid}")
        patch = report.patch_geometry
        print(
            f"    patch_count={patch['patch_count']} "
            f"mean_bytes_per_patch={patch['mean_bytes_per_patch']:.3f} "
            f"boundary_emit_count={patch['boundary_emit_count']} "
            f"gauge_flip_rate={patch['gauge_flip_rate']:.4f} "
            f"support_count_mean={patch['support_count_mean']:.3f} "
            f"phase_redundancy_mean={patch['phase_redundancy_mean']:.3f}"
        )

        climates = full["stream_climates"][sid]
        _print_domain_summary("network", report.network, climates["network"])
        _print_domain_summary("database", report.database, climates["database"])
        _print_domain_summary("application", report.application, climates["application"])
        print()

    print("done")

    # best effort cleanup
    try:
        encode_bridge.uninstall()
    except Exception:
        pass
    del decode_bridge
    del encode_bridge
    del model
    gc.collect()


if __name__ == "__main__":
    main()