# src/tools/gyrolabe/helpers/benchmark.py
"""
Phase 4: Compute Benchmarking for GyroLabe ALU.

Benchmarks C primitives vs Python reference (src/api.py) and float baselines.
Run from repo root: python -m src.tools.gyrolabe.helpers.benchmark

Default: short smoke test only. Use --all for full run (~1m).
With --generate: generation exactness and speed (short by default, --all-prompts for long).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.api import walsh_hadamard64
from src.tools.gyrolabe import ops
from src.tools.gyrolabe.bridges import (
    DEFAULT_BOLMO_MODEL_PATH,
    BolmoEncodeBridgeConfig,
    load_base_bolmo,
    load_gyrolabe_bolmo_encode,
)


class _NoopLabe:
    """Minimal labe stub for benchmark timing; no GyroLabe logic."""
    step_times: list[float]

    def __init__(self) -> None:
        self.step_times = []


def _set_stable_threading() -> None:
    """Stable CPU threading for cleaner benchmark numbers (6600H: 6 cores, 12 threads)."""
    try:
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _get_tokenizer(model: Any) -> Any | None:
    try:
        tc = getattr(model.model, "tokenizer_config", None)
        if tc is not None and hasattr(tc, "build"):
            return tc.build()
    except Exception:
        pass
    return None


def _load_base_bolmo(model_path: Path | None = None, **hf_kwargs: Any) -> Any:
    """Load raw Bolmo without bridge."""
    path = str(model_path) if model_path is not None else None
    return load_base_bolmo(path, **hf_kwargs)


def _load_bridge_bolmo(
    config: BolmoEncodeBridgeConfig | None = None,
    model_path: Path | None = None,
    **hf_kwargs: Any,
) -> Any:
    """Load Bolmo with GyroLabe encode bridge."""
    path = str(model_path) if model_path is not None else None
    return load_gyrolabe_bolmo_encode(path, config=config or BolmoEncodeBridgeConfig(), **hf_kwargs)


_GEN_PROMPTS: list[tuple[str, str]] = [
    ("short", "The quick brown fox jumps over the lazy dog. " * 2),
    ("long", "The quick brown fox jumps over the lazy dog. " * 80),
]


def _bench_generate_equivalence(
    base_model: Any,
    bridge_model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 64,
    device: str = "cpu",
) -> dict:
    """Run deterministic generation on base and bridge, return comparison metrics."""
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"]
    if device != "cpu":
        input_ids = input_ids.to(device)
    prompt_byte_len = input_ids.numel()

    gen_kw = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=getattr(tokenizer, "pad_token_id", 0),
    )

    t0 = time.perf_counter()
    base_out = base_model.generate(input_ids.clone(), **gen_kw)
    t_base = time.perf_counter() - t0

    t0 = time.perf_counter()
    bridge_out = bridge_model.generate(input_ids.clone(), **gen_kw)
    t_bridge = time.perf_counter() - t0

    base_ids = base_out[0].cpu()
    bridge_ids = bridge_out[0].cpu()
    token_equal = torch.equal(base_ids, bridge_ids)
    base_str = tokenizer.decode(base_ids.tolist(), skip_special_tokens=True)
    bridge_str = tokenizer.decode(bridge_ids.tolist(), skip_special_tokens=True)
    str_equal = base_str == bridge_str

    first_diff = -1
    if not token_equal:
        for i in range(min(base_ids.numel(), bridge_ids.numel())):
            if base_ids[i].item() != bridge_ids[i].item():
                first_diff = i
                break

    return {
        "prompt_byte_len": prompt_byte_len,
        "base_sec": t_base,
        "bridge_sec": t_bridge,
        "token_equal": token_equal,
        "str_equal": str_equal,
        "first_diff": first_diff,
        "base_ids": base_ids,
        "bridge_ids": bridge_ids,
    }


def _bench_generate_speed(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 64,
    device: str = "cpu",
) -> dict:
    """Measure prefill, first-token, decode speed via timing labe."""
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"]
    if device != "cpu":
        input_ids = input_ids.to(device)

    gen_kw: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=getattr(tokenizer, "pad_token_id", 0),
    )

    timing_labe = _NoopLabe()
    gen_kw["labe"] = timing_labe

    t0 = time.perf_counter()
    out = model.generate(input_ids, **gen_kw)
    t_total = time.perf_counter() - t0

    step_times = getattr(timing_labe, "step_times", [])
    prefill_ms = step_times[0] * 1000 if step_times else 0
    first_token_ms = prefill_ms
    remaining_decode_ms = sum(step_times[1:]) * 1000 if len(step_times) > 1 else 0
    total_generate_ms = t_total * 1000

    new_tokens = out.shape[1] - input_ids.shape[1]
    decode_tok_s = new_tokens / t_total if t_total > 0 else 0
    total_tok_s = out.shape[1] / t_total if t_total > 0 else 0
    avg_decode_step_ms = (remaining_decode_ms / (len(step_times) - 1)) if len(step_times) > 1 else 0

    return {
        "prefill_ms": prefill_ms,
        "first_token_ms": first_token_ms,
        "remaining_decode_ms": remaining_decode_ms,
        "avg_decode_step_ms": avg_decode_step_ms,
        "total_generate_ms": total_generate_ms,
        "decode_tok_s": decode_tok_s,
        "total_tok_s": total_tok_s,
        "new_tokens": new_tokens,
        "decode_steps": len(step_times) - 1 if len(step_times) > 1 else 0,
    }


def _run_generate_benchmarks(
    model_path: Path | None,
    hf_kwargs: dict[str, Any] | None = None,
    all_prompts: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Run generation exactness and speed benchmarks. Returns (exactness, speed)."""
    model_path = model_path or DEFAULT_BOLMO_MODEL_PATH
    if not model_path.exists():
        return [], []
    hf_kwargs = hf_kwargs or {}

    prompts = _GEN_PROMPTS if all_prompts else [p for p in _GEN_PROMPTS if p[0] == "short"]

    _set_stable_threading()

    base = _load_base_bolmo(model_path, **hf_kwargs)
    base.eval()
    tokenizer = _get_tokenizer(base)
    if tokenizer is None:
        return [], []

    config_zero = BolmoEncodeBridgeConfig(
        embedding_scale=1.0,
        boundary_scale=1.0,
        strict_cpu=True,
    )
    bridge_zero = _load_bridge_bolmo(config_zero, model_path, **hf_kwargs)
    bridge_zero.reset_structural_parameters()
    bridge_zero.eval()

    config_cache = BolmoEncodeBridgeConfig(
        embedding_scale=1.0,
        boundary_scale=1.0,
        strict_cpu=True,
    )
    bridge_cache = _load_bridge_bolmo(config_cache, model_path, **hf_kwargs)
    bridge_cache.reset_structural_parameters()
    bridge_cache.eval()

    exactness: list[dict] = []
    speed: list[dict] = []

    for name, prompt in prompts:
        eq_base_bridge = _bench_generate_equivalence(
            base, bridge_zero, tokenizer, prompt, max_new_tokens=64
        )
        eq_base_bridge["mode"] = "base vs bridge zero-init"
        eq_base_bridge["prompt_class"] = name
        exactness.append(eq_base_bridge)

        eq_base_cache = _bench_generate_equivalence(
            base, bridge_cache, tokenizer, prompt, max_new_tokens=64
        )
        eq_base_cache["mode"] = "base vs bridge + decode cache"
        eq_base_cache["prompt_class"] = name
        exactness.append(eq_base_cache)

        sp_base = _bench_generate_speed(base, tokenizer, prompt, max_new_tokens=64)
        sp_base["mode"] = "base"
        sp_base["prompt_class"] = name
        speed.append(sp_base)

        sp_bridge = _bench_generate_speed(bridge_zero, tokenizer, prompt, max_new_tokens=64)
        sp_bridge["mode"] = "bridge zero-init"
        sp_bridge["prompt_class"] = name
        speed.append(sp_bridge)

        sp_cache = _bench_generate_speed(bridge_cache, tokenizer, prompt, max_new_tokens=64)
        sp_cache["mode"] = "bridge + decode cache"
        sp_cache["prompt_class"] = name
        speed.append(sp_cache)

    return exactness, speed


def _warmup() -> None:
    """Run a few ops to warm caches and JIT."""
    b = torch.randint(0, 256, (64,), dtype=torch.uint8)
    ops.signature_scan(b)
    s = ops.signatures_to_states(ops.signature_scan(b))
    if s.numel() >= 2:
        ops.chirality_distance(s[:-1], s[1:])
    if ops._get_lib() is not None:
        x = torch.randn(4, 64, dtype=torch.float32)
        ops.wht64(x)


def _timeit(
    fn,
    *args,
    warmup_runs: int = 3,
    timed_runs: int = 7,
    min_total_sec: float = 0.1,
    **kwargs,
) -> tuple[float, float]:
    """Return (median_sec, std_sec) over timed_runs."""
    for _ in range(warmup_runs):
        fn(*args, **kwargs)
    times: list[float] = []
    total = 0.0
    while len(times) < timed_runs or total < min_total_sec:
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        total += times[-1]
    times.sort()
    n = len(times)
    median = times[n // 2]
    variance = sum((t - median) ** 2 for t in times) / max(1, n - 1)
    return median, (variance**0.5) if variance > 0 else 0.0


def _bench_signature_scan(full: bool = False) -> list[dict]:
    """Benchmark signature_scan: C vs Python reference."""
    results: list[dict] = []
    lengths = [64, 256, 1024, 4096, 16384, 65536] if full else [1024]
    for n in lengths:
        bytes_t = torch.randint(0, 256, (n,), dtype=torch.uint8)
        py_out = ops._py_signature_scan(bytes_t)
        c_out = ops.signature_scan(bytes_t)
        if not torch.equal(py_out, c_out):
            raise AssertionError(f"signature_scan mismatch at n={n}")
        py_sec, py_std = _timeit(ops._py_signature_scan, bytes_t)
        c_sec, c_std = _timeit(ops.signature_scan, bytes_t)
        speedup = py_sec / c_sec if c_sec > 0 else float("inf")
        results.append({
            "op": "signature_scan",
            "n": n,
            "py_sec": py_sec,
            "py_std": py_std,
            "c_sec": c_sec,
            "c_std": c_std,
            "speedup": speedup,
            "py_mb_s": (n / 1e6) / py_sec if py_sec > 0 else 0,
            "c_mb_s": (n / 1e6) / c_sec if c_sec > 0 else 0,
        })
    return results


def _bench_chirality_distance(full: bool = False) -> list[dict]:
    """Benchmark chirality_distance: C vs Python reference."""
    results: list[dict] = []
    counts = [256, 1024, 4096, 16384, 65536, 262144] if full else [1024]
    for n in counts:
        bytes_t = torch.randint(0, 256, (n + 1,), dtype=torch.uint8)
        sigs = ops.signature_scan(bytes_t)
        states = ops.signatures_to_states(sigs)
        states_a = states[:-1].to(torch.int32)
        states_b = states[1:].to(torch.int32)
        py_out = ops._py_chirality_distance(states_a, states_b)
        c_out = ops.chirality_distance(states_a, states_b)
        if not torch.equal(py_out, c_out):
            raise AssertionError(f"chirality_distance mismatch at n={n}")
        py_sec, py_std = _timeit(ops._py_chirality_distance, states_a, states_b)
        c_sec, c_std = _timeit(ops.chirality_distance, states_a, states_b)
        speedup = py_sec / c_sec if c_sec > 0 else float("inf")
        pairs_per_sec_c = n / c_sec if c_sec > 0 else 0
        results.append({
            "op": "chirality_distance",
            "n_pairs": n,
            "py_sec": py_sec,
            "py_std": py_std,
            "c_sec": c_sec,
            "c_std": c_std,
            "speedup": speedup,
            "c_pairs_per_sec": pairs_per_sec_c,
        })
    return results


def _bench_chirality_vs_cosine(full: bool = False) -> list[dict]:
    """Benchmark chirality_distance vs F.cosine_similarity (what it replaces)."""
    results: list[dict] = []
    hidden = 2048
    counts = [256, 1024, 4096, 16384] if full else [1024]
    for n in counts:
        states_a = torch.randint(0, 1 << 24, (n,), dtype=torch.int32)
        states_b = torch.randint(0, 1 << 24, (n,), dtype=torch.int32)
        emb_a = torch.randn(n, hidden, dtype=torch.float32)
        emb_b = torch.randn(n, hidden, dtype=torch.float32)
        chir_sec, _ = _timeit(ops.chirality_distance, states_a, states_b)
        def _cosine_batch():
            torch.nn.functional.cosine_similarity(
                emb_a, emb_b, dim=1
            )
        cos_sec, _ = _timeit(_cosine_batch)
        speedup = cos_sec / chir_sec if chir_sec > 0 else float("inf")
        results.append({
            "op": "chirality_vs_cosine",
            "n_pairs": n,
            "hidden": hidden,
            "chirality_sec": chir_sec,
            "cosine_sec": cos_sec,
            "speedup": speedup,
        })
    return results


def _bench_wht64(full: bool = False) -> list[dict]:
    """Benchmark wht64: C vs Python WHT matmul vs dense 64x64 matmul."""
    if ops._get_lib() is None:
        return []
    H = torch.from_numpy(walsh_hadamard64().astype("float32"))
    W_dense = torch.randn(64, 64, dtype=torch.float32)
    results: list[dict] = []
    batch_sizes = [1, 16, 64, 256, 1024, 4096, 16384] if full else [64]
    for batch in batch_sizes:
        x = torch.randn(batch, 64, dtype=torch.float32)
        c_sec, c_std = _timeit(ops.wht64, x)
        def _py_wht():
            torch.mm(x, H.t())
        py_sec, py_std = _timeit(_py_wht)
        def _dense():
            torch.mm(x, W_dense)
        dense_sec, _ = _timeit(_dense)
        speedup_vs_py = py_sec / c_sec if c_sec > 0 else float("inf")
        speedup_vs_dense = dense_sec / c_sec if c_sec > 0 else float("inf")
        results.append({
            "op": "wht64",
            "batch": batch,
            "c_sec": c_sec,
            "c_std": c_std,
            "py_wht_sec": py_sec,
            "dense_sec": dense_sec,
            "speedup_vs_py": speedup_vs_py,
            "speedup_vs_dense": speedup_vs_dense,
        })
    return results


def _bench_qmap_extract(full: bool = False) -> list[dict]:
    """Benchmark qmap_extract: C vs Python."""
    results: list[dict] = []
    lengths = [256, 1024, 4096, 16384, 65536] if full else [1024]
    for n in lengths:
        bytes_t = torch.randint(0, 256, (n,), dtype=torch.uint8)
        py_q, py_f, py_m = ops._py_qmap_extract(bytes_t)
        c_q, c_f, c_m = ops.qmap_extract(bytes_t)
        if not (torch.equal(py_q, c_q) and torch.equal(py_f, c_f) and torch.equal(py_m, c_m)):
            raise AssertionError(f"qmap_extract mismatch at n={n}")
        py_sec, py_std = _timeit(ops._py_qmap_extract, bytes_t)
        c_sec, c_std = _timeit(ops.qmap_extract, bytes_t)
        speedup = py_sec / c_sec if c_sec > 0 else float("inf")
        results.append({
            "op": "qmap_extract",
            "n": n,
            "py_sec": py_sec,
            "c_sec": c_sec,
            "speedup": speedup,
        })
    return results


def _bench_aqpu_bitplane(full: bool = False) -> list[dict]:
    """Benchmark aQPU bitplane GEMV: torch.mv vs C unpacked vs packed vs OpenCL batch."""
    results: list[dict] = []
    n_bits = 16
    batch_sizes = [1, 16, 64, 256, 1024, 4096] if full else [64, 256]
    torch.manual_seed(42)
    W = torch.randn(64, 64, dtype=torch.float32) * 0.1

    for batch in batch_sizes:
        X = torch.randn(batch, 64, dtype=torch.float32) * 0.1
        r: dict[str, Any] = {"op": "aqpu_bitplane", "batch": batch}

        def _torch_mv():
            for i in range(batch):
                torch.mv(W, X[i])
        torch_sec, _ = _timeit(_torch_mv)
        r["torch_sec"] = torch_sec

        if ops._get_lib() is not None:
            packed = ops.PackedBitplaneMatrix64(W, n_bits=n_bits)
            y_cpu = packed.gemm_packed_batch(X)

            def _cpu_packed():
                packed.gemm_packed_batch(X)

            cpu_sec, _ = _timeit(_cpu_packed)
            r["cpu_packed_sec"] = cpu_sec
            r["speedup_vs_torch"] = torch_sec / cpu_sec if cpu_sec > 0 else 0

            # Integer-native CPU path (i32 -> i64 exact, then scaled back to float)
            W_i32 = (W * 128.0).round().to(torch.int32)
            X_i32 = (X * 128.0).round().to(torch.int32)
            packed_i32 = ops.PackedBitplaneMatrix64I32(W_i32, n_bits=n_bits)
            vecs_i32 = [ops.PackedBitplaneVector64I32(X_i32[b], n_bits=n_bits) for b in range(batch)]
            y_cpu_i32 = torch.stack([packed_i32.gemv_packed(v) for v in vecs_i32], dim=0).to(torch.float32) / (128.0 * 128.0)
            err_i32_cpu = (y_cpu - y_cpu_i32).abs().max().item()
            r["cpu_i32_err_vs_cpu_f32"] = err_i32_cpu

            def _cpu_i32():
                for v in vecs_i32:
                    packed_i32.gemv_packed(v)

            cpu_i32_sec, _ = _timeit(_cpu_i32)
            r["cpu_i32_sec"] = cpu_i32_sec

            try:
                from src.tools.gyrolabe import opencl_backend
                if opencl_backend.available():
                    opencl_backend.initialize()
                    cl_matrix = opencl_backend.OpenCLPackedMatrix64(packed)
                    try:
                        y_cl = cl_matrix.gemm_packed_batch(X)
                        err = (y_cpu - y_cl).abs().max().item()
                        if err > 1e-4:
                            r["opencl_ok"] = False
                            r["opencl_err"] = err
                        else:
                            def _cl_batch():
                                cl_matrix.gemm_packed_batch(X)

                            cl_sec, _ = _timeit(_cl_batch)
                            r["opencl_sec"] = cl_sec
                            r["opencl_speedup_vs_torch"] = torch_sec / cl_sec if cl_sec > 0 else 0
                            r["opencl_speedup_vs_cpu"] = cpu_sec / cl_sec if cl_sec > 0 else 0
                            r["opencl_ok"] = True

                        # Integer-native OpenCL path
                        cl_matrix_i32 = opencl_backend.OpenCLPackedMatrix64I32(packed_i32)
                        X_sign = torch.empty(batch, dtype=torch.uint64, device="cpu")
                        X_bp = torch.empty((batch, n_bits), dtype=torch.uint64, device="cpu")
                        for b in range(batch):
                            X_sign[b] = vecs_i32[b]._x_sign.to(torch.uint64)
                            X_bp[b].copy_(vecs_i32[b]._x_bp)
                        y_cl_i32 = cl_matrix_i32.gemm_packed_batch(X_sign, X_bp)
                        cl_matrix_i32.close()

                        y_cl_i32_scaled = y_cl_i32.to(torch.float32) / (128.0 * 128.0)
                        err_i32_cl = (y_cpu - y_cl_i32_scaled).abs().max().item()
                        r["opencl_i32_err_vs_cpu_f32"] = err_i32_cl

                        def _cl_i32_batch():
                            cl_i32 = opencl_backend.OpenCLPackedMatrix64I32(packed_i32)
                            try:
                                cl_i32.gemm_packed_batch(X_sign, X_bp)
                            finally:
                                cl_i32.close()

                        cl_i32_sec, _ = _timeit(_cl_i32_batch)
                        r["opencl_i32_sec"] = cl_i32_sec
                    finally:
                        cl_matrix.close()
                    opencl_backend.shutdown()
            except (ImportError, RuntimeError, OSError):
                r["opencl_ok"] = False
        results.append(r)
    return results


def _print_report(all_results: list[dict], use_c: bool) -> None:
    """Print formatted benchmark report."""
    print("\nGyroLabe Phase 4: Compute Benchmarking")
    print("C library:", "loaded" if use_c else "not available (Python fallback only)")
    print("-" * 60)
    for r in all_results:
        if r["op"] == "signature_scan":
            print(f"signature_scan n={r['n']:>6}: "
                  f"Python {r['py_sec']*1000:.3f}ms, C {r['c_sec']*1000:.3f}ms, "
                  f"speedup {r['speedup']:.1f}x")
        elif r["op"] == "chirality_distance":
            print(f"chirality_distance n={r['n_pairs']:>6}: "
                  f"Python {r['py_sec']*1000:.3f}ms, C {r['c_sec']*1000:.3f}ms, "
                  f"speedup {r['speedup']:.1f}x ({r['c_pairs_per_sec']/1e6:.2f}M pairs/s)")
        elif r["op"] == "chirality_vs_cosine":
            print(f"chirality vs cosine (2048d) n={r['n_pairs']:>5}: "
                  f"chirality {r['chirality_sec']*1000:.3f}ms, cosine {r['cosine_sec']*1000:.3f}ms, "
                  f"speedup {r['speedup']:.1f}x")
        elif r["op"] == "wht64":
            print(f"wht64 batch={r['batch']:>5}: "
                  f"C {r['c_sec']*1000:.3f}ms, py_wht {r['py_wht_sec']*1000:.3f}ms, "
                  f"dense {r['dense_sec']*1000:.3f}ms | vs_py {r['speedup_vs_py']:.1f}x, "
                  f"vs_dense {r['speedup_vs_dense']:.1f}x")
        elif r["op"] == "qmap_extract":
            print(f"qmap_extract n={r['n']:>6}: "
                  f"Python {r['py_sec']*1000:.3f}ms, C {r['c_sec']*1000:.3f}ms, "
                  f"speedup {r['speedup']:.1f}x")
        elif r["op"] == "aqpu_bitplane":
            print(f"aqpu_bitplane batch={r['batch']:>4}: torch {r['torch_sec']*1000:.2f}ms", end="")
            if "cpu_packed_sec" in r:
                print(f", CPU packed {r['cpu_packed_sec']*1000:.2f}ms ({r.get('speedup_vs_torch', 0):.1f}x vs torch)", end="")
            if "cpu_i32_sec" in r:
                print(f", CPU i32 {r['cpu_i32_sec']*1000:.2f}ms", end="")
            if r.get("opencl_ok"):
                print(f", OpenCL {r['opencl_sec']*1000:.2f}ms ({r.get('opencl_speedup_vs_torch', 0):.1f}x vs torch, {r.get('opencl_speedup_vs_cpu', 0):.1f}x vs CPU)", end="")
                if "opencl_i32_sec" in r:
                    print(f", OpenCL i32 {r['opencl_i32_sec']*1000:.2f}ms", end="")
            elif "opencl_ok" in r:
                print(f", OpenCL: unavailable", end="")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="GyroLabe Phase 4 compute benchmarks.")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup.")
    parser.add_argument("--report", type=Path, help="Write markdown report to file.")
    parser.add_argument("--skip-wht", action="store_true", help="Skip wht64 (needs C lib).")
    parser.add_argument("--skip-cosine", action="store_true", help="Skip chirality vs cosine.")
    parser.add_argument("--skip-aqpu", action="store_true", help="Skip aQPU bitplane benchmark.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Full compute benchmark (~1h). Default: short smoke test.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run generation exactness and speed benchmarks.",
    )
    parser.add_argument(
        "--all-prompts",
        action="store_true",
        help="With --generate: run short and long prompts. Default: short only.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_BOLMO_MODEL_PATH,
        help="Path to Bolmo model for --generate mode.",
    )
    parser.add_argument(
        "--bfloat16",
        action="store_true",
        help="Load model in bfloat16. May fail if mLSTM lacks bfloat16 support.",
    )
    args = parser.parse_args()

    _set_stable_threading()

    if not args.no_warmup:
        _warmup()

    use_c = ops._get_lib() is not None
    full = args.all
    all_results: list[dict] = []

    if args.generate:
        hf_kw = {"torch_dtype": torch.bfloat16} if args.bfloat16 else {}
        exactness, speed = _run_generate_benchmarks(
            args.model_path, hf_kwargs=hf_kw, all_prompts=args.all_prompts
        )
        _print_generate_reports(exactness, speed)
        if args.report:
            _write_generate_report(exactness, speed, args.report)
        return

    all_results.extend(_bench_signature_scan(full=full))
    all_results.extend(_bench_chirality_distance(full=full))
    if not args.skip_cosine:
        all_results.extend(_bench_chirality_vs_cosine(full=full))
    all_results.extend(_bench_qmap_extract(full=full))
    if not args.skip_wht and use_c:
        all_results.extend(_bench_wht64(full=full))
    if not args.skip_aqpu and use_c:
        all_results.extend(_bench_aqpu_bitplane(full=full))

    _print_report(all_results, use_c)

    if args.report:
        _write_markdown_report(all_results, use_c, args.report)
        print(f"\nReport written to {args.report}")


def _print_generate_reports(
    exactness: list[dict],
    speed: list[dict],
) -> None:
    """Print generation benchmark reports."""
    print("\n--- Generation Exactness Report ---")
    for r in exactness:
        eq = "yes" if r["token_equal"] else "no"
        sd = "yes" if r["str_equal"] else "no"
        fd = r.get("first_diff", -1)
        print(f"  {r['prompt_class']} ({r['mode']}): token_eq={eq}, str_eq={sd}, first_diff={fd}")

    print("\n--- Generation Speed Report ---")
    for r in speed:
        print(f"  {r['prompt_class']} {r['mode']}: prefill_ms={r['prefill_ms']:.1f}, "
              f"first_token_ms={r['first_token_ms']:.1f}, remaining_decode_ms={r['remaining_decode_ms']:.1f}, "
              f"total_generate_ms={r['total_generate_ms']:.1f}, decode_tok/s={r['decode_tok_s']:.1f}")


def _write_generate_report(
    exactness: list[dict],
    speed: list[dict],
    path: Path,
) -> None:
    """Write generation benchmark report to markdown."""
    lines = [
        "# GyroLabe Generation Benchmark Report",
        "",
        "## A. Generation Exactness Report",
        "",
        "| Prompt | Mode | Token Eq | Str Eq | First Diff |",
        "|--------|------|----------|--------|------------|",
    ]
    for r in exactness:
        eq = "yes" if r["token_equal"] else "no"
        sd = "yes" if r["str_equal"] else "no"
        fd = r.get("first_diff", -1)
        lines.append(f"| {r['prompt_class']} | {r['mode']} | {eq} | {sd} | {fd} |")
    lines.extend([
        "",
        "## B. Generation Speed Report",
        "",
        "| Prompt | Mode | Prefill ms | First tok ms | Decode ms | Total ms | Decode tok/s |",
        "|--------|------|------------|--------------|-----------|----------|--------------|",
    ])
    for r in speed:
        lines.append(f"| {r['prompt_class']} | {r['mode']} | {r['prefill_ms']:.1f} | "
                    f"{r['first_token_ms']:.1f} | {r['remaining_decode_ms']:.1f} | "
                    f"{r['total_generate_ms']:.1f} | {r['decode_tok_s']:.1f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_markdown_report(all_results: list[dict], use_c: bool, path: Path) -> None:
    """Write Phase 4 benchmark report in markdown."""
    lines = [
        "# GyroLabe Phase 4: Compute Benchmarking Report",
        "",
        f"**C library:** {'loaded' if use_c else 'not available'}",
        "",
        "## Results",
        "",
        "| Op | Size | Python (ms) | C (ms) | Speedup |",
        "|---|------|--------------|--------|---------|",
    ]
    for r in all_results:
        if r["op"] == "signature_scan":
            lines.append(f"| signature_scan | n={r['n']} | {r['py_sec']*1000:.2f} | {r['c_sec']*1000:.2f} | {r['speedup']:.1f}x |")
        elif r["op"] == "chirality_distance":
            lines.append(f"| chirality_distance | n={r['n_pairs']} | {r['py_sec']*1000:.2f} | {r['c_sec']*1000:.2f} | {r['speedup']:.1f}x |")
        elif r["op"] == "chirality_vs_cosine":
            lines.append(f"| chirality vs cosine (2048d) | n={r['n_pairs']} | - | chir {r['chirality_sec']*1000:.2f} / cos {r['cosine_sec']*1000:.2f} | {r['speedup']:.1f}x |")
        elif r["op"] == "wht64":
            lines.append(f"| wht64 | batch={r['batch']} | py_wht {r['py_wht_sec']*1000:.2f} | {r['c_sec']*1000:.2f} | vs_py {r['speedup_vs_py']:.1f}x, vs_dense {r['speedup_vs_dense']:.1f}x |")
        elif r["op"] == "qmap_extract":
            lines.append(f"| qmap_extract | n={r['n']} | {r['py_sec']*1000:.2f} | {r['c_sec']*1000:.2f} | {r['speedup']:.1f}x |")
        elif r["op"] == "aqpu_bitplane":
            extra = ""
            if r.get("opencl_ok"):
                extra = f", OpenCL {r['opencl_sec']*1000:.2f}ms ({r.get('opencl_speedup_vs_torch', 0):.1f}x vs torch)"
            lines.append(f"| aqpu_bitplane | batch={r['batch']} | torch {r['torch_sec']*1000:.2f} | CPU packed {r.get('cpu_packed_sec', 0)*1000:.2f} | {r.get('speedup_vs_torch', 0):.1f}x{extra} |")
    lines.extend(["", "## Exit Gate", "", "Documented orders-of-magnitude speedups for intercepted algebraic workloads.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
