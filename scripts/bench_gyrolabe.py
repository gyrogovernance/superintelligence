#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api import (
    EPS_A6_BY_BYTE,
    EPS_B6_BY_BYTE,
    MICRO_REF_BY_BYTE,
    OmegaSignature12,
    chirality_word6,
    compose_omega_signatures,
    pack_omega12,
    pack_omega_signature12,
    step_omega12_by_byte,
    step_state_by_byte,
    unpack_omega12,
    walsh_hadamard64,
)
from src.tools.gyrolabe import ops as gy_ops
from src.tools.gyrolabe import opencl_backend

torch.set_num_threads(1)

_WHT = torch.tensor(walsh_hadamard64(), dtype=torch.float32)
_LOOKAHEAD = 1


def _time_fn(repeats: int, fn) -> float:
    for _ in range(2):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return time.perf_counter() - start


def _seeded_randn(*shape: int, seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=dtype)


def _print_rate(label: str, n: int, elapsed: float, repeats: int) -> None:
    sec_per = elapsed / repeats
    rate = n / sec_per if sec_per > 0.0 else 0.0
    print(f"{label}: n={n}, avg {sec_per * 1000:.3f} ms, {rate:.2f} items/s")


def _check_eq(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    strict: bool = True,
) -> None:
    if a.is_floating_point() or b.is_floating_point():
        if not torch.allclose(a, b, rtol=rtol, atol=atol):
            if strict:
                raise RuntimeError(f"parity check failed: {name}")
            max_abs = (a - b).abs().max()
            print(f"parity warning for {name}: max_abs={max_abs.item():.6f}")
    elif not torch.equal(a, b):
        if strict:
            raise RuntimeError(f"parity check failed: {name}")
        print(f"parity warning for {name}: integer mismatch")


def _random_bytes(n: int, seed: int) -> torch.Tensor:
    return torch.randint(0, 256, size=(n,), dtype=torch.uint8, generator=torch.Generator().manual_seed(seed))


def _random_states(n: int, seed: int) -> torch.Tensor:
    return torch.randint(0, 1 << 24, size=(n,), dtype=torch.int32, generator=torch.Generator().manual_seed(seed + 1))


def _random_omega12(n: int, seed: int) -> torch.Tensor:
    return torch.randint(0, 1 << 12, size=(n,), dtype=torch.int32, generator=torch.Generator().manual_seed(seed + 2))


def _py_omega_signature_scan(bytes_tensor: torch.Tensor) -> torch.Tensor:
    acc = OmegaSignature12(parity=0, tau_u6=0, tau_v6=0)
    out = torch.empty_like(bytes_tensor, dtype=torch.int32)
    for i in range(int(bytes_tensor.numel())):
        b = int(bytes_tensor[i].item())
        cur = OmegaSignature12(
            parity=1,
            tau_u6=EPS_A6_BY_BYTE[b],
            tau_v6=MICRO_REF_BY_BYTE[b] ^ EPS_B6_BY_BYTE[b],
        )
        acc = compose_omega_signatures(cur, acc)
        out[i] = pack_omega_signature12(acc)
    return out


def _py_omega12_scan_from_omega12(bytes_tensor: torch.Tensor, start_omega12: int) -> torch.Tensor:
    n = int(bytes_tensor.numel())
    out = torch.empty(n, dtype=torch.int32)
    state = unpack_omega12(int(start_omega12) & 0xFFF)
    for i in range(n):
        state = step_omega12_by_byte(state, int(bytes_tensor[i].item()))
        out[i] = pack_omega12(state)
    return out


def _py_apply_signature_batch(states: torch.Tensor, signatures: torch.Tensor) -> torch.Tensor:
    flat_s = states.reshape(-1)
    flat_k = signatures.reshape(-1)
    out = torch.empty_like(flat_s, dtype=torch.int32)
    for i in range(flat_s.numel()):
        out[i] = gy_ops._py_apply_signature_to_state(int(flat_s[i].item()), int(flat_k[i].item()))
    return out.reshape(states.shape)


def _py_state_scan_from_state(bytes_tensor: torch.Tensor, start_state: int) -> torch.Tensor:
    out = torch.empty_like(bytes_tensor, dtype=torch.int32)
    state = int(start_state)
    for i in range(int(bytes_tensor.numel())):
        state = step_state_by_byte(state, int(bytes_tensor[i].item()))
        out[i] = state
    return out


def _py_shell_hist_state24(states: torch.Tensor) -> torch.Tensor:
    hist = torch.zeros(7, dtype=torch.int32)
    for value in states.reshape(-1):
        hist[chirality_word6(int(value.item())).bit_count()] += 1
    return hist


def _py_shell_hist_omega12(omega12: torch.Tensor) -> torch.Tensor:
    hist = torch.zeros(7, dtype=torch.int32)
    for value in omega12.reshape(-1):
        hist[unpack_omega12(int(value.item())).shell] += 1
    return hist


def _py_chirality_distance_adjacent(states: torch.Tensor, lookahead: int = 1) -> torch.Tensor:
    n = int(states.numel())
    if lookahead <= 0:
        raise ValueError("lookahead must be positive")
    out = torch.zeros_like(states, dtype=torch.uint8)
    if n <= lookahead:
        return out
    out[:-lookahead] = gy_ops._py_chirality_distance(
        states[:-lookahead],
        states[lookahead:],
    )
    return out


def _wht_ref(x: torch.Tensor) -> torch.Tensor:
    return x @ _WHT.T.to(x.device)


def _py_bitplane_gemv_batch(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return torch.stack([gy_ops._py_bitplane_gemv(W, X[i]) for i in range(X.shape[0])])


def benchmark_exact_ops(ns: list[int], repeats: int) -> None:
    print("=== exact kernel ops ===")
    for n in ns:
        print(f"--- n={n} ---")
        b = _random_bytes(n, seed=17 + n)
        states = _random_states(n, seed=19 + n)
        sig = _random_states(n, seed=23 + n)
        omega12 = _random_omega12(n, seed=29 + n)
        words = torch.randint(0, 255, size=(n,), dtype=torch.uint32, generator=torch.Generator().manual_seed(31 + n))

        sig_py = gy_ops._py_signature_scan(b)
        sig_native = gy_ops.signature_scan(b)
        _check_eq(f"signature_scan parity n={n}", sig_native, sig_py)
        _print_rate(
            "signature_scan python",
            n,
            _time_fn(repeats, lambda: gy_ops._py_signature_scan(b)),
            repeats,
        )
        _print_rate(
            "signature_scan native",
            n,
            _time_fn(repeats, lambda: gy_ops.signature_scan(b)),
            repeats,
        )

        q1, f1, m1 = gy_ops._py_qmap_extract(b)
        q2, f2, m2 = gy_ops.qmap_extract(b)
        _check_eq(f"qmap_extract q parity n={n}", q2, q1)
        _check_eq(f"qmap_extract f parity n={n}", f2, f1)
        _check_eq(f"qmap_extract m parity n={n}", m2, m1)
        _print_rate(
            "qmap_extract python",
            n,
            _time_fn(repeats, lambda: gy_ops._py_qmap_extract(b)),
            repeats,
        )
        _print_rate(
            "qmap_extract native",
            n,
            _time_fn(repeats, lambda: gy_ops.qmap_extract(b)),
            repeats,
        )

        q3, f3, m3, sig3, st3 = gy_ops.extract_scan(b)
        _check_eq(f"extract_scan q parity n={n}", q3, q1)
        _check_eq(f"extract_scan f parity n={n}", f3, f1)
        _check_eq(f"extract_scan m parity n={n}", m3, m1)
        _check_eq(f"extract_scan sig parity n={n}", sig3, sig_py)
        _check_eq(f"extract_scan state parity n={n}", st3, gy_ops.signatures_to_states(sig3))
        _print_rate(
            "extract_scan native",
            n,
            _time_fn(repeats, lambda: gy_ops.extract_scan(b)),
            repeats,
        )

        ch1 = gy_ops._py_chirality_distance(states, states ^ words.to(torch.int32))
        ch2 = gy_ops.chirality_distance(states, states ^ words.to(torch.int32))
        _check_eq(f"chirality_distance parity n={n}", ch2, ch1)
        _print_rate(
            "chirality_distance python",
            n,
            _time_fn(repeats, lambda: gy_ops._py_chirality_distance(states, states ^ words.to(torch.int32))),
            repeats,
        )
        _print_rate(
            "chirality_distance native",
            n,
            _time_fn(repeats, lambda: gy_ops.chirality_distance(states, states ^ words.to(torch.int32))),
            repeats,
        )

        adj_ref = _py_chirality_distance_adjacent(states, lookahead=_LOOKAHEAD)
        adj2 = gy_ops.chirality_distance_adjacent(states, lookahead=_LOOKAHEAD)
        _check_eq("chirality_distance_adjacent parity", adj2, adj_ref)
        _print_rate(
            "chirality_distance_adjacent python",
            n,
            _time_fn(repeats, lambda: _py_chirality_distance_adjacent(states, lookahead=_LOOKAHEAD)),
            repeats,
        )
        _print_rate(
            "chirality_distance_adjacent native",
            n,
            _time_fn(repeats, lambda: gy_ops.chirality_distance_adjacent(states, lookahead=_LOOKAHEAD)),
            repeats,
        )

        a1 = _py_apply_signature_batch(states, sig)
        a2 = gy_ops.apply_signature_batch(states, sig)
        _check_eq(f"apply_signature_batch parity n={n}", a2, a1)
        _print_rate(
            "apply_signature_batch python",
            n,
            _time_fn(repeats, lambda: _py_apply_signature_batch(states, sig)),
            repeats,
        )
        _print_rate(
            "apply_signature_batch native",
            n,
            _time_fn(repeats, lambda: gy_ops.apply_signature_batch(states, sig)),
            repeats,
        )

        start = int(torch.randint(0, 1 << 24, size=(1,), dtype=torch.int32).item())
        s1 = _py_state_scan_from_state(b, start)
        s2 = gy_ops.state_scan_from_state(b, start)
        _check_eq(f"state_scan_from_state parity n={n}", s2, s1)
        _print_rate(
            "state_scan_from_state python",
            n,
            _time_fn(repeats, lambda: _py_state_scan_from_state(b, start)),
            repeats,
        )
        _print_rate(
            "state_scan_from_state native",
            n,
            _time_fn(repeats, lambda: gy_ops.state_scan_from_state(b, start)),
            repeats,
        )

        o1 = _py_omega_signature_scan(b)
        o2 = gy_ops.omega_signature_scan(b)
        _check_eq(f"omega_signature_scan parity n={n}", o2, o1)
        _print_rate(
            "omega_signature_scan python",
            n,
            _time_fn(repeats, lambda: _py_omega_signature_scan(b)),
            repeats,
        )
        _print_rate(
            "omega_signature_scan native",
            n,
            _time_fn(repeats, lambda: gy_ops.omega_signature_scan(b)),
            repeats,
        )

        s_omega = int(omega12[0].item())
        o12_py = _py_omega12_scan_from_omega12(b, s_omega)
        o12_native = gy_ops.omega12_scan_from_omega12(b, s_omega)
        _check_eq(f"omega12_scan_from_omega12 parity n={n}", o12_native, o12_py)
        _print_rate(
            "omega12_scan_from_omega12 python",
            n,
            _time_fn(repeats, lambda: _py_omega12_scan_from_omega12(b, s_omega)),
            repeats,
        )
        _print_rate(
            "omega12_scan_from_omega12 native",
            n,
            _time_fn(repeats, lambda: gy_ops.omega12_scan_from_omega12(b, s_omega)),
            repeats,
        )

        h1 = _py_shell_hist_state24(states)
        h2 = gy_ops.shell_histogram_state24(states)
        _check_eq("shell_histogram_state24 parity", h2, h1)
        h3 = _py_shell_hist_omega12(omega12)
        h4 = gy_ops.shell_histogram_omega12(omega12)
        _check_eq("shell_histogram_omega12 parity", h4, h3)
        _print_rate(
            "shell_histogram_state24 python",
            n,
            _time_fn(repeats, lambda: _py_shell_hist_state24(states)),
            repeats,
        )
        _print_rate(
            "shell_histogram_state24 native",
            n,
            _time_fn(repeats, lambda: gy_ops.shell_histogram_state24(states)),
            repeats,
        )
        _print_rate(
            "shell_histogram_omega12 python",
            n,
            _time_fn(repeats, lambda: _py_shell_hist_omega12(omega12)),
            repeats,
        )
        _print_rate(
            "shell_histogram_omega12 native",
            n,
            _time_fn(repeats, lambda: gy_ops.shell_histogram_omega12(omega12)),
            repeats,
        )

        wf = _seeded_randn(n, 64, seed=41 + n)
        wref = _wht_ref(wf)
        wnat = gy_ops.wht64(wf)
        _check_eq(f"wht64 parity n={n}", wnat, wref)
        _check_eq(
            f"wht64_metal_first parity n={n}",
            gy_ops.wht64_metal_first(wf),
            wnat,
            rtol=1e-4,
            atol=1e-4,
            strict=False,
        )
        _print_rate(
            "wht64 python ref",
            n,
            _time_fn(repeats, lambda: _wht_ref(wf)),
            repeats,
        )
        _print_rate(
            "wht64 native",
            n,
            _time_fn(repeats, lambda: gy_ops.wht64(wf)),
            repeats,
        )
        _print_rate(
            "wht64_metal_first",
            n,
            _time_fn(repeats, lambda: gy_ops.wht64_metal_first(wf)),
            repeats,
        )


def benchmark_tensor_ops(batch_sizes: list[int], repeats: int, run_opencl: bool) -> None:
    print("=== tensor and operator ops ===")
    for n in batch_sizes:
        print(f"--- batch={n} ---")
        W = _seeded_randn(n, 64, seed=99 + n)
        x = _seeded_randn(64, seed=101 + n)
        y_ref = gy_ops._py_bitplane_gemv(W, x)
        _check_eq(
            f"bitplane_gemv parity batch={n}",
            gy_ops.bitplane_gemv(W, x),
            y_ref,
        )
        _print_rate(
            "bitplane_gemv python",
            n,
            _time_fn(repeats, lambda: gy_ops._py_bitplane_gemv(W, x)),
            repeats,
        )
        _print_rate(
            "bitplane_gemv native",
            n,
            _time_fn(repeats, lambda: gy_ops.bitplane_gemv(W, x)),
            repeats,
        )
        _print_rate("torch mv", n, _time_fn(repeats, lambda: W @ x), repeats)

        x8 = _seeded_randn(64, seed=103 + n)
        x8_ref = gy_ops._py_bitplane_gemv(W, x8)
        packed_f32 = gy_ops.PackedBitplaneMatrix64(W, n_bits=16)
        vx = gy_ops.PackedBitplaneVector64(x8, n_bits=16)
        _check_eq(
            f"PackedBitplaneMatrix64.gemv parity batch={n}",
            packed_f32.gemv_packed(vx),
            x8_ref,
            rtol=1e-4,
            atol=1e-4,
            strict=False,
        )
        _print_rate(
            "PackedBitplaneMatrix64.gemv",
            n,
            _time_fn(repeats, lambda: packed_f32.gemv_packed(vx)),
            repeats,
        )

        X = _seeded_randn(n, 64, seed=107 + n)
        packed_f32_b = gy_ops.PackedBitplaneMatrix64(W[:64, :], n_bits=16)
        X_c = X.contiguous()
        W_b = W[: packed_f32_b.rows, :].contiguous()
        W_b_t = W_b.t().contiguous()
        y_batch_ref = _py_bitplane_gemv_batch(X_c, W_b)
        _check_eq(
            f"PackedBitplaneMatrix64.gemm_packed_batch parity batch={n}",
            packed_f32_b.gemm_packed_batch(X_c),
            y_batch_ref[:, : packed_f32_b.rows],
            rtol=1e-4,
            atol=1e-4,
            strict=False,
        )
        _print_rate(
            "pack_vector_batch64",
            n,
            _time_fn(repeats, lambda: gy_ops.pack_vector_batch64(X_c, n_bits=packed_f32_b.n_bits)),
            repeats,
        )
        _print_rate(
            "PackedBitplaneMatrix64.gemm_packed_batch",
            n,
            _time_fn(repeats, lambda: packed_f32_b.gemm_packed_batch(X_c)),
            repeats,
        )
        _print_rate(
            "torch mm",
            n,
            _time_fn(
                repeats,
                lambda: torch.mm(X_c, W_b_t),
            ),
            repeats,
        )

        W_int = torch.randint(
            -8,
            9,
            size=W.shape,
            dtype=torch.int32,
            generator=torch.Generator().manual_seed(111 + n),
        )
        x_int = torch.randint(
            -8,
            9,
            size=(64,),
            dtype=torch.int32,
            generator=torch.Generator().manual_seed(113 + n),
        )
        ref_i32 = W_int.to(torch.int64) @ x_int.to(torch.int64)
        packed_i32 = gy_ops.PackedBitplaneMatrix64I32(W_int[:64, :], n_bits=16)
        vx_i32 = gy_ops.PackedBitplaneVector64I32(x_int, n_bits=16)
        _check_eq(
            f"PackedBitplaneMatrix64I32.gemv_packed parity batch={n}",
            packed_i32.gemv_packed(vx_i32),
            ref_i32[: packed_i32.rows],
        )
        _print_rate(
            "PackedBitplaneMatrix64I32.gemv_packed",
            n,
            _time_fn(repeats, lambda: packed_i32.gemv_packed(vx_i32)),
            repeats,
        )

        if run_opencl and opencl_backend.available():
            opencl_proj = opencl_backend.OpenCLPackedMatrix64(packed_f32_b)
            opencl_i32 = None
            try:
                opencl_i32 = opencl_backend.OpenCLPackedMatrix64I32(packed_i32)
            except Exception:
                opencl_i32 = None
            try:
                xcl = _seeded_randn(n, 64, seed=117 + n)
                xcl = xcl.contiguous()
                ycl = _py_bitplane_gemv_batch(xcl, W_b)
                _print_rate(
                    "OpenCLPackedMatrix64.gemm_packed_batch",
                    n,
                    _time_fn(repeats, lambda: opencl_proj.gemm_packed_batch(xcl)),
                    repeats,
                )
                _check_eq(
                    f"OpenCLPackedMatrix64 parity batch={n}",
                    opencl_proj.gemm_packed_batch(xcl),
                    ycl,
                    rtol=1e-4,
                    atol=1e-4,
                    strict=False,
                )

                if opencl_i32 is not None:
                    xbatch = torch.randint(
                        -8,
                        9,
                        size=(n, 64),
                        dtype=torch.int32,
                        generator=torch.Generator().manual_seed(131 + n),
                    )
                    xs, xb = gy_ops.pack_vector_batch64_i32(xbatch, n_bits=16)
                    run_opencl_i32 = opencl_i32.gemm_packed_batch
                    yi = run_opencl_i32(xs, xb)
                    yref_i = xbatch.to(torch.int64) @ W_int[: packed_i32.rows, :].to(torch.int64).T
                    _check_eq(
                        f"OpenCLPackedMatrix64I32 parity batch={n}",
                        yi,
                        yref_i,
                    rtol=0.0,
                    atol=0.0,
                    strict=False,
                    )
                    _print_rate(
                        "OpenCLPackedMatrix64I32.gemm_packed_batch",
                        n,
                        _time_fn(
                            repeats,
                            lambda: run_opencl_i32(xs, xb),
                        ),
                        repeats,
                    )
            finally:
                opencl_proj.close()
                if opencl_i32 is not None:
                    opencl_i32.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GyroLabe benchmark")
    parser.add_argument(
        "--repeats",
        type=int,
        default=8,
        help="Number of repeats per benchmark point",
    )
    parser.add_argument(
        "--exact-n",
        type=int,
        nargs="*",
        default=[256, 4096, 65536],
        help="Exact op sizes",
    )
    parser.add_argument(
        "--batch",
        type=int,
        nargs="*",
        default=[64, 256],
        help="Batch sizes for tensor ops",
    )
    parser.add_argument(
        "--skip-opencl",
        action="store_true",
        help="Skip OpenCL benchmarks even when available",
    )
    parser.add_argument(
        "--tensor-only",
        action="store_true",
        help="Run only tensor/operator benchmarks (skip exact kernel benchmarks).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repeats = args.repeats
    if not args.tensor_only:
        benchmark_exact_ops(args.exact_n, repeats)
    benchmark_tensor_ops(args.batch, repeats, run_opencl=not args.skip_opencl)


if __name__ == "__main__":
    main()
