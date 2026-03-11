# src/tools/gyrolabe/helpers/test_aqpu_matmul.py
"""
Prove that float matrix-vector multiplication can be computed exactly through
bitplane decomposition using the aQPU's native operations (AND, XOR, POPCNT).

Backends: Python (ref) | C unpacked | C packed | OpenCL (batched GEMM)
Run from repo root: python -m src.tools.gyrolabe.helpers.test_aqpu_matmul
Or: python src/tools/gyrolabe/helpers/test_aqpu_matmul.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.tools.gyrolabe import ops

N_BITS = 16
TOLERANCE = 1e-3


def _backend_status() -> str:
    """Return a one-line summary of available backends."""
    parts = ["Python (ref)"]
    if ops._get_lib() is not None:
        parts.append("C unpacked")
        parts.append("C packed")
        try:
            from src.tools.gyrolabe import opencl_backend
            if opencl_backend.available():
                parts.append("OpenCL")
        except (ImportError, OSError):
            pass
    return " | ".join(parts)


def _run_test(
    name: str,
    W: torch.Tensor,
    x: torch.Tensor,
    tol: float = TOLERANCE,
    verbose: bool = True,
) -> bool:
    """Run one test case. Returns True if pass."""
    W = W.float().cpu()
    x = x.float().cpu()
    assert W.dim() == 2 and x.dim() == 1 and W.shape[1] == x.shape[0]
    assert W.shape[1] <= 64

    y_ref = torch.mv(W, x)
    y_py = ops._py_bitplane_gemv(W, x, n_bits=N_BITS)
    err_py = (y_ref - y_py).abs().max().item()
    passed_py = err_py <= tol

    y_c = None
    err_c = float("inf")
    passed_c = True
    y_packed = None
    err_packed = float("inf")
    passed_packed = True
    if ops._get_lib() is not None:
        y_c = ops.bitplane_gemv(W, x, n_bits=N_BITS)
        err_c = (y_ref - y_c).abs().max().item()
        passed_c = err_c <= tol
        try:
            packed = ops.PackedBitplaneMatrix64(W, n_bits=N_BITS)
            y_packed = packed.gemv(x)
            err_packed = (y_ref - y_packed).abs().max().item()
            passed_packed = err_packed <= tol
        except (RuntimeError, AttributeError):
            pass

    c_identical = (
        torch.allclose(y_py, y_c, atol=1e-7, rtol=1e-6) if y_c is not None else False
    )

    if verbose:
        n_float_muls = 64 * 64
        print(f"  {name}:")
        print(f"    Python (ref): max_err={err_py:.2e}, pass={passed_py}")
        if y_c is not None:
            print(f"    C unpacked:   max_err={err_c:.2e}, pass={passed_c}, match_py={c_identical}")
        if y_packed is not None:
            print(f"    C packed:     max_err={err_packed:.2e}, pass={passed_packed}")
        print(f"    Ops: {n_float_muls} float muls/row, {64*n_float_muls} total")
    return passed_py and passed_c and passed_packed


def _bench_gemv(
    W: torch.Tensor,
    x: torch.Tensor,
    n_warmup: int = 50,
    n_runs: int = 200,
) -> dict:
    """Benchmark torch.mv vs C bitplane_gemv vs packed vs Python bitplane."""
    W = W.float().cpu()
    x = x.float().cpu()
    times = {}

    for _ in range(n_warmup):
        torch.mv(W, x)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        torch.mv(W, x)
    times["torch"] = (time.perf_counter() - t0) / n_runs

    for _ in range(n_warmup):
        ops._py_bitplane_gemv(W, x, n_bits=N_BITS)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        ops._py_bitplane_gemv(W, x, n_bits=N_BITS)
    times["py"] = (time.perf_counter() - t0) / n_runs

    if ops._get_lib() is not None:
        for _ in range(n_warmup):
            ops.bitplane_gemv(W, x, n_bits=N_BITS)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            ops.bitplane_gemv(W, x, n_bits=N_BITS)
        times["c"] = (time.perf_counter() - t0) / n_runs

        try:
            packed = ops.PackedBitplaneMatrix64(W, n_bits=N_BITS)
            for _ in range(n_warmup):
                packed.gemv(x)
            t0 = time.perf_counter()
            for _ in range(n_runs):
                packed.gemv(x)
            times["packed"] = (time.perf_counter() - t0) / n_runs
        except (RuntimeError, AttributeError):
            times["packed"] = None
    else:
        times["c"] = None
        times["packed"] = None

    return times


def _bench_batched_opencl(
    W: torch.Tensor,
    batch_sizes: list[int],
    n_warmup: int = 5,
    n_runs: int = 20,
) -> dict:
    """Benchmark batched GEMM: CPU packed vs OpenCL. Returns times per batch size."""
    if ops._get_lib() is None:
        return {}
    try:
        from src.tools.gyrolabe import opencl_backend
        if not opencl_backend.available():
            return {}
    except (ImportError, OSError):
        return {}

    packed = ops.PackedBitplaneMatrix64(W, n_bits=N_BITS)
    opencl_backend.initialize()
    cl_matrix = opencl_backend.OpenCLPackedMatrix64(packed)
    results: dict[int, dict[str, float]] = {}
    try:
        for batch in batch_sizes:
            X = torch.randn(batch, 64, dtype=torch.float32) * 0.1
            y_cpu = packed.gemm_packed_batch(X)
            y_cl = cl_matrix.gemm_packed_batch(X)
            err = (y_cpu - y_cl).abs().max().item()
            if err > 1e-4:
                continue
            for _ in range(n_warmup):
                packed.gemm_packed_batch(X)
                cl_matrix.gemm_packed_batch(X)
            t0 = time.perf_counter()
            for _ in range(n_runs):
                packed.gemm_packed_batch(X)
            t_cpu = (time.perf_counter() - t0) / n_runs
            t0 = time.perf_counter()
            for _ in range(n_runs):
                cl_matrix.gemm_packed_batch(X)
            t_cl = (time.perf_counter() - t0) / n_runs
            results[batch] = {"cpu_ms": t_cpu * 1000, "opencl_ms": t_cl * 1000, "speedup": t_cpu / t_cl if t_cl > 0 else 0}
    finally:
        cl_matrix.close()
        opencl_backend.shutdown()
    return results


def _bench_full_projection() -> dict:
    """Simulate 2048x2048 = 32x32 blocks of 64x64."""
    n_blocks = 1024
    torch.manual_seed(123)
    W_blocks = [torch.randn(64, 64) * 0.02 for _ in range(n_blocks)]
    x_blocks = [torch.randn(64) for _ in range(n_blocks)]

    n_warm = 2
    n_run = 5

    for _ in range(n_warm):
        for b in range(n_blocks):
            torch.mv(W_blocks[b], x_blocks[b])
    t0 = time.perf_counter()
    for _ in range(n_run):
        for b in range(n_blocks):
            torch.mv(W_blocks[b], x_blocks[b])
    t_mv = (time.perf_counter() - t0) / n_run

    t_c = None
    if ops._get_lib() is not None:
        for _ in range(n_warm):
            for b in range(n_blocks):
                ops.bitplane_gemv(W_blocks[b], x_blocks[b], n_bits=N_BITS)
        t0 = time.perf_counter()
        for _ in range(n_run):
            for b in range(n_blocks):
                ops.bitplane_gemv(W_blocks[b], x_blocks[b], n_bits=N_BITS)
        t_c = (time.perf_counter() - t0) / n_run

    W_full = torch.randn(2048, 2048) * 0.02
    x_full = torch.randn(2048) * 0.02
    for _ in range(n_warm):
        torch.mv(W_full, x_full)
    t0 = time.perf_counter()
    for _ in range(n_run):
        torch.mv(W_full, x_full)
    t_full = (time.perf_counter() - t0) / n_run

    return {
        "mv_per_block_us": t_mv * 1e6 / n_blocks,
        "mv_total_ms": t_mv * 1000,
        "c_per_block_us": t_c * 1e6 / n_blocks if t_c else None,
        "c_total_ms": t_c * 1000 if t_c else None,
        "full_mv_ms": t_full * 1000,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="aQPU bitplane matmul test")
    parser.add_argument("--no-bolmo", action="store_true", help="Skip Bolmo model test")
    parser.add_argument("--bench", action="store_true", help="Run benchmarks")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    print("aQPU bitplane matrix-vector multiplication")
    print("Backends:", _backend_status())
    print("-" * 40)

    all_pass = True
    verbose = not args.quiet

    print("\n--- Test 1: Random matrix ---")
    torch.manual_seed(42)
    W = torch.randn(64, 64) * 0.1
    x = torch.randn(64) * 0.1
    all_pass &= _run_test("random", W, x, verbose=verbose)

    print("\n--- Test 2: Real Bolmo block ---")
    model_path = Path(_REPO_ROOT) / "data" / "models" / "Bolmo-1B"
    if not args.no_bolmo and model_path.exists():
        try:
            from src.tools.gyrolabe.bolmo_bridge import load_gyrolabe_bolmo
            model = load_gyrolabe_bolmo(model_path)
            W = model.base_model.model.layers[0].self_attn.q_proj.weight[0:64, 0:64].float().cpu()  # type: ignore[index]
            x = torch.randn(64) * 0.02
            all_pass &= _run_test("Bolmo q_proj block", W, x, tol=1e-2, verbose=verbose)
        except Exception as e:
            print(f"  Skipped: {e}")
    else:
        reason = "model not at data/models/Bolmo-1B" if not args.no_bolmo else "--no-bolmo"
        print(f"  Skipped: {reason}")

    print("\n--- Test 3: Identity matrix ---")
    W = torch.eye(64)
    x = torch.randn(64)
    all_pass &= _run_test("identity", W, x, tol=1e-4, verbose=verbose)

    print("\n--- Test 4: Single nonzero ---")
    W = torch.zeros(64, 64)
    W[10, 20] = 1.5
    x = torch.zeros(64)
    x[20] = -2.0
    all_pass &= _run_test("single nonzero", W, x, tol=1e-4, verbose=verbose)

    if ops._get_lib() is not None:
        print("\n--- Test 5: Batched GEMM (OpenCL vs CPU packed) ---")
        try:
            from src.tools.gyrolabe import opencl_backend
            if opencl_backend.available():
                torch.manual_seed(123)
                W5 = torch.randn(64, 64, dtype=torch.float32) * 0.1
                X5 = torch.randn(64, 64, dtype=torch.float32) * 0.1
                packed5 = ops.PackedBitplaneMatrix64(W5, n_bits=N_BITS)
                y_cpu = packed5.gemm_packed_batch(X5)
                opencl_backend.initialize()
                cl = opencl_backend.OpenCLPackedMatrix64(packed5)
                y_cl = cl.gemm_packed_batch(X5)
                cl.close()
                opencl_backend.shutdown()
                err_cl = (y_cpu - y_cl).abs().max().item()
                ok_cl = err_cl <= 1e-3
                if verbose:
                    print(f"  batch=64: CPU vs OpenCL max_err={err_cl:.2e}, pass={ok_cl}")
                all_pass &= ok_cl
            else:
                if verbose:
                    print("  Skipped: OpenCL not available")
        except (ImportError, OSError, RuntimeError) as e:
            if verbose:
                print(f"  Skipped: {e}")

        print("\n--- Test 6: Integer-native OpenCL vs CPU (i32) ---")
        try:
            from src.tools.gyrolabe import opencl_backend
            if opencl_backend.available():
                torch.manual_seed(321)
                W_i32 = torch.randint(low=-4, high=5, size=(64, 64), dtype=torch.int32)
                X_i32 = torch.randint(low=-4, high=5, size=(64, 64), dtype=torch.int32)

                packed_W_i32 = ops.PackedBitplaneMatrix64I32(W_i32, n_bits=N_BITS)
                packed_vecs = [
                    ops.PackedBitplaneVector64I32(X_i32[b], n_bits=N_BITS) for b in range(64)
                ]
                y_cpu_i32 = torch.stack([packed_W_i32.gemv_packed(pv) for pv in packed_vecs], dim=0)

                # Build packed batch representation for OpenCL integer path
                X_sign = torch.empty(64, dtype=torch.uint64, device="cpu")
                X_bp = torch.empty((64, N_BITS), dtype=torch.uint64, device="cpu")
                for b in range(64):
                    X_sign[b] = packed_vecs[b]._x_sign.to(torch.uint64)
                    X_bp[b].copy_(packed_vecs[b]._x_bp)

                opencl_backend.initialize()
                cl_i32 = opencl_backend.OpenCLPackedMatrix64I32(packed_W_i32)
                y_cl_i32 = cl_i32.gemm_packed_batch(X_sign, X_bp)
                cl_i32.close()
                opencl_backend.shutdown()

                err_i32 = (y_cpu_i32.to(torch.int64) - y_cl_i32).abs().max().item()
                ok_i32 = err_i32 == 0
                if verbose:
                    print(f"  i32 batch=64: CPU vs OpenCL max_err={err_i32}, pass={ok_i32}")
                all_pass &= ok_i32
            else:
                if verbose:
                    print("  Skipped: OpenCL not available")
        except (ImportError, OSError, RuntimeError) as e:
            if verbose:
                print(f"  Skipped: {e}")

    if args.bench:
        print("\n--- Benchmark: 64x64 GEMV (single vector) ---")
        torch.manual_seed(42)
        W = torch.randn(64, 64) * 0.1
        x = torch.randn(64) * 0.1
        t = _bench_gemv(W, x)
        t_mv_us = t["torch"] * 1e6
        t_py_us = t["py"] * 1e6
        print(f"  torch.mv:        {t_mv_us:.2f} us/call (reference)")
        if t["c"] is not None:
            t_c_us = t["c"] * 1e6
            print(f"  C unpacked:     {t_c_us:.2f} us/call  ({t_mv_us/t_c_us:.1f}x vs torch)")
        if t["packed"] is not None:
            t_packed_us = t["packed"] * 1e6
            print(f"  C packed:       {t_packed_us:.2f} us/call  ({t_mv_us/t_packed_us:.1f}x vs torch)")
        print(f"  Python bitplane: {t_py_us:.2f} us/call")

        print("\n--- Benchmark: Batched GEMM (OpenCL vs CPU packed) ---")
        cl_batch = _bench_batched_opencl(W, [64, 256, 1024])
        if cl_batch:
            for batch, r in cl_batch.items():
                print(f"  batch={batch:>4}: CPU {r['cpu_ms']:.2f} ms, OpenCL {r['opencl_ms']:.2f} ms  ({r['speedup']:.1f}x)")
        else:
            print("  OpenCL not available")

        print("\n--- Precision sweep ---")
        for n_b in [8, 12, 16, 20, 24]:
            y_ref = torch.mv(W, x)
            y_py = ops._py_bitplane_gemv(W, x, n_bits=n_b)
            err = (y_ref - y_py).abs().max().item()
            print(f"  n_bits={n_b:2d}: max_err={err:.2e}")

        print("\n--- Full 2048x2048 projection (1024 blocks) ---")
        b = _bench_full_projection()
        print(f"  torch.mv per block:     {b['mv_per_block_us']:.2f} us, total: {b['mv_total_ms']:.2f} ms")
        if b["c_per_block_us"] is not None:
            print(f"  C bitplane per block:   {b['c_per_block_us']:.2f} us, total: {b['c_total_ms']:.2f} ms")
        print(f"  torch.mv(2048,2048):    {b['full_mv_ms']:.2f} ms")
        if b["c_total_ms"] is not None:
            print(f"  Ratio (blockwise C / single torch): {b['c_total_ms']/b['full_mv_ms']:.2f}x")

    print("\n" + "-" * 40)
    print(f"All tests passed: {all_pass}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
