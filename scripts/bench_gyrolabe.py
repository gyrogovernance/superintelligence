#!/usr/bin/env python3
"""GyroLabe benchmark: codec ops, OpenCL, Gyroscopic Matrix Multiplication, GyroMatMul kernels."""
from __future__ import annotations

import argparse
import gc
import io
import logging
import sys
import time
import os
import copy
import warnings
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from shutil import get_terminal_size
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
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
)
from src.constants import (
    LAYER_MASK_12,
    byte_to_intron,
    expand_intron_to_mask12,
    unpack_state,
)
from src.tools.gyrolabe.ops_build import get_lib, has_fn, native_available
from src.tools.gyrolabe.ops_codec import (
    apply_signature_batch,
    chirality_distance,
    chirality_distance_adjacent,
    extract_scan,
    omega12_scan_from_omega12,
    omega_signature_scan,
    get_native_threads,
    initialize_native,
    qmap_extract,
    shell_histogram_omega12,
    shell_histogram_state24,
    set_native_threads,
    signature_scan,
    signatures_to_states,
    state_scan_from_state,
)
from src.tools.gyrolabe.ops_mul import (
    gyromatmul_i32_dense,
    gyromatmul_i32,
    gyromatmul_i32_profile,
    gyromatmul_bmm_qk_f32,
    gyromatmul_bmm_av_f32,
)
from src.tools.gyrolabe import opencl_backend
from src.tools.gyrolabe.gyromatmul._build import get_ext

from tests.tools.conftest import bolmo_tokenizer_from_model, measure_generation_ms, random_i32_matrix
_LOOKAHEAD = 1


warnings.filterwarnings("ignore", message=".*Failed to find CUDA.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*rope_config_validation.*", category=FutureWarning)


def _silence_hf_and_cuda_loading() -> None:
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("fsspec").setLevel(logging.ERROR)

    if "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    if "TRANSFORMERS_VERBOSITY" not in os.environ:
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _benchmark_load_context(enabled: bool):
    if not enabled:
        from contextlib import nullcontext
        return nullcontext()

    @contextmanager
    def _ctx():
        _silence_hf_and_cuda_loading()

        sink = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with redirect_stdout(sink), redirect_stderr(sink):
                yield

    return _ctx()


def _collect_cpu_simd_caps() -> tuple[str, ...]:
    getter = getattr(torch.backends.cpu, "get_cpu_capability", None)
    if not callable(getter):
        return tuple()
    try:
        caps = getter()
    except Exception:
        return tuple()

    if isinstance(caps, str):
        tokens = [caps]
    elif isinstance(caps, (tuple, list, set)):
        tokens = [str(v) for v in caps]
    elif isinstance(caps, dict):
        tokens = [str(k) for k, v in caps.items() if v]
    else:
        return tuple()

    out: list[str] = []
    for token in tokens:
        out.extend(
            part
            for part in str(token).upper().replace("/", " ").replace(",", " ").split()
            if part
        )
    return tuple(sorted(set(out)))


def _supports_avx512(caps: tuple[str, ...]) -> bool:
    return any(cap.startswith("AVX512") for cap in caps)



def _configure_benchmark_threads(
    *,
    torch_threads: int,
    native_threads: int,
    omp_wait_policy: str | None,
) -> None:
    cpu_threads = os.cpu_count() or 1
    effective_torch_threads = torch_threads if torch_threads > 0 else cpu_threads
    requested_native_threads = native_threads if native_threads > 0 else effective_torch_threads

    torch.set_num_threads(effective_torch_threads)
    torch.set_num_interop_threads(effective_torch_threads)

    if omp_wait_policy:
        os.environ["OMP_WAIT_POLICY"] = omp_wait_policy

    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(requested_native_threads)
    elif native_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(requested_native_threads)

    initialize_native()
    set_native_threads(requested_native_threads)
    observed_native_threads = requested_native_threads
    try:
        observed_native_threads = get_native_threads()
    except Exception:
        pass

    simd_level = os.environ.get("GYROLABE_SIMD_LEVEL", "<unset>")
    cpu_simd_caps = _collect_cpu_simd_caps()
    avx2_support = "AVX2" in cpu_simd_caps
    avx512_support = _supports_avx512(cpu_simd_caps)
    cap_summary = ",".join(cpu_simd_caps) if cpu_simd_caps else "<unknown>"
    print(
        f"  threading: torch={effective_torch_threads} "
        f"native={observed_native_threads} "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<unset>')} "
        f"OMP_WAIT_POLICY={os.environ.get('OMP_WAIT_POLICY', 'default')} "
        f"GYROLABE_SIMD_LEVEL={simd_level}"
    )
    print(
        f"  simd: env={simd_level} AVX2={avx2_support} "
        f"AVX512={avx512_support} CPU_caps={cap_summary}"
    )
    if not avx512_support:
        print("  note: AVX-512 is not active. This build is AVX2-only here.")


DEFAULT_LINEAR_BITS = 12


def _effective_linear_n_bits(n_bits: int) -> int:
    return 18 if n_bits <= 16 else n_bits


def _effective_decode_n_bits(n_bits: int) -> int:
    return max(1, min(16, int(n_bits)))


# ---------------------------------------------------------------------------
# Timing / validation helpers
# ---------------------------------------------------------------------------


def _time_fn(repeats: int, fn) -> float:
    for _ in range(2):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return time.perf_counter() - start


def _rate(
    label: str, n: int, elapsed: float, repeats: int,
    *, vs: float | None = None, vs_label: str = "py",
) -> None:
    max_label_len = max(20, min(42, get_terminal_size((120, 20)).columns - 55))
    short_label = (label if len(label) <= max_label_len else f"{label[: max_label_len - 3]}...")
    sec = elapsed / repeats
    rate = n / sec if sec > 0 else 0.0
    suffix = ""
    if vs is not None and vs > 0 and sec > 0:
        mult = (vs / repeats) / sec
        suffix = f" | {mult:.0f}x vs {vs_label}" if mult >= 10 else f" | {100*mult:.0f}% vs {vs_label}"
    print(f"  {short_label:<{max_label_len}s} | {sec*1000:>7.3f}ms | {rate:>12.0f} items/s{suffix}")


def _check_eq(
    name: str, a: torch.Tensor, b: torch.Tensor,
    *, rtol: float = 1e-5, atol: float = 1e-5, strict: bool = True,
) -> None:
    if a.is_floating_point() or b.is_floating_point():
        if not torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol):
            d = (a.float() - b.float()).abs().max().item()
            if strict:
                raise RuntimeError(f"FAIL {name} (max_abs={d:.6f})")
            print(f"  WARN {name}: max_abs={d:.6f}")
    elif not torch.equal(a, b):
        if strict:
            raise RuntimeError(f"FAIL {name}")
        print(f"  WARN {name}: integer mismatch")


def _check_eq_np(name: str, a, b, *, strict: bool = True) -> None:
    a_np = a.numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    b_np = b.numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
    if not np.array_equal(a_np, b_np):
        d = int(np.max(np.abs(a_np.astype(np.int64) - b_np.astype(np.int64))))
        if strict:
            raise RuntimeError(f"FAIL {name} (max_diff={d})")
        print(f"  WARN {name}: max_diff={d}")


# ---------------------------------------------------------------------------
# Random data generators
# ---------------------------------------------------------------------------


def _random_bytes(n: int, seed: int) -> torch.Tensor:
    return torch.randint(0, 256, (n,), dtype=torch.uint8, generator=torch.Generator().manual_seed(seed))


def _random_states(n: int, seed: int) -> torch.Tensor:
    return torch.randint(0, 1 << 24, (n,), dtype=torch.int32, generator=torch.Generator().manual_seed(seed))


def _random_omega12(n: int, seed: int) -> torch.Tensor:
    return torch.randint(0, 1 << 12, (n,), dtype=torch.int32, generator=torch.Generator().manual_seed(seed))


def _seeded_randn(*shape: int, seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=dtype)


# ---------------------------------------------------------------------------
# Pure-Python references (for parity checks)
# ---------------------------------------------------------------------------


def _py_collapse_pairdiag12_to_word6(x12: int) -> int:
    out = 0
    for i in range(6):
        if (x12 >> (2 * i)) & 0x3 == 0x3:
            out |= 1 << i
    return out


def _py_pack_signature(parity: int, tau_a12: int, tau_b12: int) -> int:
    return ((parity & 1) << 24) | ((tau_a12 & LAYER_MASK_12) << 12) | (tau_b12 & LAYER_MASK_12)


def _py_unpack_signature(sig: int) -> tuple[int, int, int]:
    return (sig >> 24) & 1, (sig >> 12) & LAYER_MASK_12, sig & LAYER_MASK_12


def _py_compose_signatures(left: int, right: int) -> int:
    lp, lta, ltb = _py_unpack_signature(left)
    rp, rta, rtb = _py_unpack_signature(right)
    ra, rb = (rta, rtb) if lp == 0 else (rtb, rta)
    return _py_pack_signature(lp ^ rp, (ra ^ lta) & LAYER_MASK_12, (rb ^ ltb) & LAYER_MASK_12)


def _py_byte_signature(b: int) -> int:
    intron = byte_to_intron(b)
    mask12 = expand_intron_to_mask12(intron)
    invert_a = LAYER_MASK_12 if (intron & 1) else 0
    invert_b = LAYER_MASK_12 if (intron & 0x80) else 0
    return _py_pack_signature(1, invert_a, (mask12 ^ invert_b) & LAYER_MASK_12)


def _py_apply_signature_to_state(state24: int, signature: int) -> int:
    parity, tau_a12, tau_b12 = _py_unpack_signature(signature)
    a12, b12 = unpack_state(state24)
    if parity == 0:
        a12 = (a12 ^ tau_a12) & LAYER_MASK_12
        b12 = (b12 ^ tau_b12) & LAYER_MASK_12
    else:
        a12, b12 = (b12 ^ tau_a12) & LAYER_MASK_12, (a12 ^ tau_b12) & LAYER_MASK_12
    return ((a12 & LAYER_MASK_12) << 12) | (b12 & LAYER_MASK_12)


def _py_signature_scan(bytes_tensor: torch.Tensor) -> torch.Tensor:
    flat = bytes_tensor.reshape(-1)
    out = torch.empty(flat.shape, dtype=torch.int32)
    accum = 0
    for i in range(flat.numel()):
        accum = _py_compose_signatures(_py_byte_signature(int(flat[i]) & 0xFF), accum)
        out[i] = accum
    return out.reshape(bytes_tensor.shape)


def _py_chirality_distance(states_a: torch.Tensor, states_b: torch.Tensor) -> torch.Tensor:
    flat_a, flat_b = states_a.reshape(-1), states_b.reshape(-1)
    out = torch.empty(flat_a.shape, dtype=torch.uint8)
    for i in range(flat_a.numel()):
        sa, sb = int(flat_a[i]) & 0xFFFFFF, int(flat_b[i]) & 0xFFFFFF
        a12, b12 = unpack_state(sa)
        ca = _py_collapse_pairdiag12_to_word6((a12 ^ b12) & LAYER_MASK_12)
        a12b, b12b = unpack_state(sb)
        cb = _py_collapse_pairdiag12_to_word6((a12b ^ b12b) & LAYER_MASK_12)
        out[i] = bin(ca ^ cb).count("1")
    return out.reshape(states_a.shape)


def _py_qmap_extract(bytes_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from src.constants import byte_family, byte_micro_ref, l0_parity
    flat = bytes_tensor.reshape(-1)
    q = torch.empty(flat.shape, dtype=torch.uint8)
    f = torch.empty(flat.shape, dtype=torch.uint8)
    m = torch.empty(flat.shape, dtype=torch.uint8)
    for i in range(flat.numel()):
        bv = int(flat[i]) & 0xFF
        intron = byte_to_intron(bv)
        mask12 = expand_intron_to_mask12(intron)
        l0 = l0_parity(intron)
        q12 = (mask12 ^ (LAYER_MASK_12 if l0 else 0)) & LAYER_MASK_12
        q[i] = _py_collapse_pairdiag12_to_word6(q12)
        f[i] = byte_family(bv)
        m[i] = byte_micro_ref(bv)
    return q.reshape(bytes_tensor.shape), f.reshape(bytes_tensor.shape), m.reshape(bytes_tensor.shape)


def _py_apply_signature_batch(states: torch.Tensor, signatures: torch.Tensor) -> torch.Tensor:
    flat_s, flat_k = states.reshape(-1), signatures.reshape(-1)
    out = torch.empty_like(flat_s, dtype=torch.int32)
    for i in range(flat_s.numel()):
        out[i] = _py_apply_signature_to_state(int(flat_s[i]), int(flat_k[i]))
    return out.reshape(states.shape)


def _py_state_scan_from_state(bytes_tensor: torch.Tensor, start_state: int) -> torch.Tensor:
    out = torch.empty_like(bytes_tensor, dtype=torch.int32)
    state = int(start_state)
    for i in range(int(bytes_tensor.numel())):
        state = step_state_by_byte(state, int(bytes_tensor[i]))
        out[i] = state
    return out


def _py_omega_signature_scan(bytes_tensor: torch.Tensor) -> torch.Tensor:
    acc = OmegaSignature12(parity=0, tau_u6=0, tau_v6=0)
    out = torch.empty_like(bytes_tensor, dtype=torch.int32)
    for i in range(int(bytes_tensor.numel())):
        b = int(bytes_tensor[i])
        cur = OmegaSignature12(parity=1, tau_u6=EPS_A6_BY_BYTE[b], tau_v6=MICRO_REF_BY_BYTE[b] ^ EPS_B6_BY_BYTE[b])
        acc = compose_omega_signatures(cur, acc)
        out[i] = pack_omega_signature12(acc)
    return out


def _py_omega12_scan_from_omega12(bytes_tensor: torch.Tensor, start_omega12: int) -> torch.Tensor:
    n = int(bytes_tensor.numel())
    out = torch.empty(n, dtype=torch.int32)
    state = unpack_omega12(int(start_omega12) & 0xFFF)
    for i in range(n):
        state = step_omega12_by_byte(state, int(bytes_tensor[i]))
        out[i] = pack_omega12(state)
    return out


def _py_shell_hist_state24(states: torch.Tensor) -> torch.Tensor:
    hist = torch.zeros(7, dtype=torch.int32)
    for v in states.reshape(-1):
        hist[chirality_word6(int(v)).bit_count()] += 1
    return hist


def _py_shell_hist_omega12(omega12: torch.Tensor) -> torch.Tensor:
    hist = torch.zeros(7, dtype=torch.int32)
    for v in omega12.reshape(-1):
        hist[unpack_omega12(int(v)).shell] += 1
    return hist


def _py_chirality_distance_adjacent(states: torch.Tensor, lookahead: int = 1) -> torch.Tensor:
    n = int(states.numel())
    out = torch.zeros_like(states, dtype=torch.uint8)
    if lookahead <= 0 or n <= lookahead:
        return out
    out[:-lookahead] = _py_chirality_distance(states[:-lookahead], states[lookahead:])
    return out


def _py_gyromatmul_gemv(W: torch.Tensor, x: torch.Tensor, n_bits: int = 16) -> torch.Tensor:
    scale_max = (1 << (n_bits - 1)) - 1
    max_abs = max(W.abs().max().item(), x.abs().max().item(), 1e-12)
    scale = scale_max / max_abs
    W_int = torch.round(W * scale).to(torch.int64)
    x_int = torch.round(x * scale).to(torch.int64)
    return (W_int @ x_int).float() / (scale * scale)


def _py_gyromatmul_gemv_batch(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return torch.stack([_py_gyromatmul_gemv(W, X[i]) for i in range(X.shape[0])])


# ---------------------------------------------------------------------------
# Benchmark: exact codec ops
# ---------------------------------------------------------------------------


def benchmark_exact_ops(ns: list[int], repeats: int) -> None:
    print("\n=== Exact Codec Ops ===")
    for n in ns:
        print(f"\n  n={n}")
        b = _random_bytes(n, seed=17 + n)
        states = _random_states(n, seed=19 + n)
        sig = _random_states(n, seed=23 + n)
        omega12 = _random_omega12(n, seed=29 + n)
        words = torch.randint(0, 255, (n,), dtype=torch.int32, generator=torch.Generator().manual_seed(31 + n))

        sig_py = _py_signature_scan(b)
        sig_native = signature_scan(b)
        _check_eq(f"signature_scan n={n}", sig_native, sig_py)
        e = _time_fn(repeats, lambda: _py_signature_scan(b))
        _rate("signature_scan [py]", n, e, repeats)
        _rate("signature_scan [native]", n, _time_fn(repeats, lambda: signature_scan(b)), repeats, vs=e)

        q1, f1, m1 = _py_qmap_extract(b)
        q2, f2, m2 = qmap_extract(b)
        _check_eq(f"qmap q n={n}", q2, q1)
        _check_eq(f"qmap f n={n}", f2, f1)
        _check_eq(f"qmap m n={n}", m2, m1)
        e = _time_fn(repeats, lambda: _py_qmap_extract(b))
        _rate("qmap_extract [py]", n, e, repeats)
        _rate("qmap_extract [native]", n, _time_fn(repeats, lambda: qmap_extract(b)), repeats, vs=e)

        q3, f3, m3, sig3, st3 = extract_scan(b)
        _check_eq(f"extract_scan q n={n}", q3, q1)
        _check_eq(f"extract_scan sig n={n}", sig3, sig_py)
        _check_eq(f"extract_scan state n={n}", st3, signatures_to_states(sig3))
        _rate("extract_scan [native]", n, _time_fn(repeats, lambda: extract_scan(b)), repeats)

        perturbed = states ^ words
        ch1 = _py_chirality_distance(states, perturbed)
        ch2 = chirality_distance(states, perturbed)
        _check_eq(f"chirality_distance n={n}", ch2, ch1)
        e = _time_fn(repeats, lambda: _py_chirality_distance(states, perturbed))
        _rate("chirality_distance [py]", n, e, repeats)
        _rate("chirality_distance [native]", n, _time_fn(repeats, lambda: chirality_distance(states, perturbed)), repeats, vs=e)

        adj_ref = _py_chirality_distance_adjacent(states, lookahead=_LOOKAHEAD)
        adj2 = chirality_distance_adjacent(states, lookahead=_LOOKAHEAD)
        _check_eq("chir_dist_adjacent", adj2, adj_ref, strict=False)
        e = _time_fn(repeats, lambda: _py_chirality_distance_adjacent(states, lookahead=_LOOKAHEAD))
        _rate("chir_dist_adjacent [py]", n, e, repeats)
        _rate("chir_dist_adjacent [native]", n, _time_fn(repeats, lambda: chirality_distance_adjacent(states, lookahead=_LOOKAHEAD)), repeats, vs=e)

        a1 = _py_apply_signature_batch(states, sig)
        a2 = apply_signature_batch(states, sig)
        _check_eq(f"apply_sig_batch n={n}", a2, a1)
        e = _time_fn(repeats, lambda: _py_apply_signature_batch(states, sig))
        _rate("apply_sig_batch [py]", n, e, repeats)
        _rate("apply_sig_batch [native]", n, _time_fn(repeats, lambda: apply_signature_batch(states, sig)), repeats, vs=e)

        start = int(torch.randint(0, 1 << 24, (1,), dtype=torch.int32))
        s1 = _py_state_scan_from_state(b, start)
        s2 = state_scan_from_state(b, start)
        _check_eq(f"state_scan n={n}", s2, s1)
        e = _time_fn(repeats, lambda: _py_state_scan_from_state(b, start))
        _rate("state_scan [py]", n, e, repeats)
        _rate("state_scan [native]", n, _time_fn(repeats, lambda: state_scan_from_state(b, start)), repeats, vs=e)

        o1 = _py_omega_signature_scan(b)
        o2 = omega_signature_scan(b)
        _check_eq(f"omega_sig_scan n={n}", o2, o1)
        e = _time_fn(repeats, lambda: _py_omega_signature_scan(b))
        _rate("omega_sig_scan [py]", n, e, repeats)
        _rate("omega_sig_scan [native]", n, _time_fn(repeats, lambda: omega_signature_scan(b)), repeats, vs=e)

        s_omega = int(omega12[0])
        o12_py = _py_omega12_scan_from_omega12(b, s_omega)
        o12_native = omega12_scan_from_omega12(b, s_omega)
        _check_eq(f"omega12_scan n={n}", o12_native, o12_py)
        e = _time_fn(repeats, lambda: _py_omega12_scan_from_omega12(b, s_omega))
        _rate("omega12_scan [py]", n, e, repeats)
        _rate("omega12_scan [native]", n, _time_fn(repeats, lambda: omega12_scan_from_omega12(b, s_omega)), repeats, vs=e)

        h1 = _py_shell_hist_state24(states)
        h2 = shell_histogram_state24(states)
        _check_eq("shell_hist_state24", h2, h1)
        h3 = _py_shell_hist_omega12(omega12)
        h4 = shell_histogram_omega12(omega12)
        _check_eq("shell_hist_omega12", h4, h3)
        e = _time_fn(repeats, lambda: _py_shell_hist_state24(states))
        _rate("shell_hist_state24 [py]", n, e, repeats)
        _rate("shell_hist_state24 [native]", n, _time_fn(repeats, lambda: shell_histogram_state24(states)), repeats, vs=e)
        e = _time_fn(repeats, lambda: _py_shell_hist_omega12(omega12))
        _rate("shell_hist_omega12 [py]", n, e, repeats)
        _rate("shell_hist_omega12 [native]", n, _time_fn(repeats, lambda: shell_histogram_omega12(omega12)), repeats, vs=e)


# ---------------------------------------------------------------------------
# Benchmark: OpenCL
# ---------------------------------------------------------------------------


def benchmark_tensor_ops(batch_sizes: list[int], repeats: int, run_opencl: bool) -> None:
    print("\n=== OpenCL ===")
    has_gyromatmul = native_available() and has_fn("gyromatmul_i32")
    opencl_ready = False

    if run_opencl:
        if not opencl_backend.available():
            print("  runtime unavailable (OpenCL skipped)")
        elif not has_gyromatmul:
            print("  skipped (missing gyromatmul native symbols)")
        else:
            try:
                opencl_backend.initialize()
                opencl_ready = True
            except Exception as exc:
                print(f"  skipped (OpenCL init failed: {exc})")

    try:
        for n in batch_sizes:
            print(f"\n  batch={n}")
            W = _seeded_randn(n, 64, seed=99 + n)
            W_sub = W[:64, :].contiguous()

            if not run_opencl:
                print("    skipped (opencl disabled)")
                continue
            if not opencl_ready:
                print("    skipped (opencl unavailable)")
                continue

            packed_f32 = W_sub
            W_int = torch.randint(
                -8,
                9,
                size=W.shape,
                dtype=torch.int32,
                generator=torch.Generator().manual_seed(111 + n),
            )
            _benchmark_opencl(n, repeats, packed_f32, W_sub, W_int)
    finally:
        if opencl_ready:
            opencl_backend.shutdown()


def _benchmark_opencl(
    n: int, repeats: int,
    packed_f32: torch.Tensor,
    W_sub: torch.Tensor,
    W_int: torch.Tensor,
) -> None:
    scale = ((1 << 15) - 1) / max(float(packed_f32.abs().max().item()), 1e-12)
    packed_i32 = torch.round(packed_f32 * scale).to(torch.int32)
    opencl_proj = opencl_backend.OpenCLPackedMatrix64I32(packed_i32)
    W_i32 = W_int[:64, :].contiguous()
    opencl_i32 = opencl_backend.OpenCLPackedMatrix64I32(W_i32)

    try:
        xcl = _seeded_randn(n, 64, seed=117 + n).contiguous()
        ycl = _py_gyromatmul_gemv_batch(xcl, W_sub)
        xcl_i32 = torch.round(xcl * scale).to(torch.int32)
        ycl_opencl_i32 = opencl_proj.gemm_packed_batch(xcl_i32)
        ycl_opencl = ycl_opencl_i32.float() / (scale * scale)
        _rate(
            "OpenCL f32 gemm_batch",
            n,
            _time_fn(repeats, lambda: (opencl_proj.gemm_packed_batch(xcl_i32).float() / (scale * scale))),
            repeats,
        )
        _check_eq(
            f"OpenCL f32 batch={n}",
            ycl_opencl,
            ycl,
            rtol=1e-4,
            atol=1e-4,
            strict=False,
        )

        xbatch = torch.randint(-8, 9, size=(n, 64), dtype=torch.int32, generator=torch.Generator().manual_seed(131 + n))
        yi = opencl_i32.gemm_packed_batch(xbatch)
        yref = xbatch.to(torch.int64) @ W_i32.to(torch.int64).T
        _check_eq(f"OpenCL i32 batch={n}", yi, yref, rtol=0, atol=0, strict=False)
        _rate("OpenCL i32 gemm_batch", n, _time_fn(repeats, lambda: opencl_i32.gemm_packed_batch(xbatch)), repeats)
    finally:
        opencl_proj.close()
        opencl_i32.close()


# ---------------------------------------------------------------------------
# Benchmark: Gyroscopic Matrix Multiplication
# ---------------------------------------------------------------------------


def benchmark_gyroscopic(sizes: list[tuple[int, int]], repeats: int) -> None:
    print("\n=== Gyroscopic Matrix Multiplication ===")
    for rows, cols in sizes:
        print(f"\n  {rows}x{cols}")

        Q = random_i32_matrix(rows, cols, seed=200 + rows * cols)
        K = random_i32_matrix(rows, cols, seed=300 + rows * cols)
        ref = Q.astype(np.int64) @ K.astype(np.int64).T

        _check_eq_np(f"gyroscopic {rows}x{cols}", gyromatmul_i32(Q, K), ref)
        gyro_prof_out, profile = gyromatmul_i32_profile(Q, K)
        _check_eq_np(f"gyroscopic+profile {rows}x{cols}", gyro_prof_out, ref)
        _check_eq_np(f"gyromatmul_i32 {rows}x{cols}", gyromatmul_i32_dense(Q, K), ref)

        Q_t = torch.tensor(Q, dtype=torch.int32)
        K_t = torch.tensor(K, dtype=torch.int32)
        n_items = rows * rows

        e_np = _time_fn(repeats, lambda: Q.astype(np.int64) @ K.astype(np.int64).T)
        _rate("numpy i64", n_items, e_np, repeats)
        _rate("torch i64", n_items, _time_fn(repeats, lambda: Q_t.to(torch.int64) @ K_t.to(torch.int64).T), repeats, vs=e_np, vs_label="numpy")
        _rate("gyromatmul_i32_dense [matmul]", n_items, _time_fn(repeats, lambda: gyromatmul_i32_dense(Q, K)), repeats, vs=e_np, vs_label="numpy")
        _rate("gyromatmul_i32 [matmul]", n_items, _time_fn(repeats, lambda: gyromatmul_i32(Q, K)), repeats, vs=e_np, vs_label="numpy")
        _rate("torch i64 [matmul]", n_items, _time_fn(repeats, lambda: gyromatmul_i32(Q_t, K_t)), repeats, vs=e_np, vs_label="numpy")

        sym_ref = Q.astype(np.int64) @ Q.astype(np.int64).T
        _check_eq_np(f"gyroscopic symmetric {rows}x{cols}", gyromatmul_i32(Q, Q), sym_ref)
        _rate("gyromatmul_i32 symmetric [matmul]", n_items, _time_fn(repeats, lambda: gyromatmul_i32(Q, Q)), repeats, vs=e_np, vs_label="numpy")

        Q_mixed = random_i32_matrix(rows, cols, seed=400 + rows * cols, low=-100000, high=100001)
        K_mixed = random_i32_matrix(rows, cols, seed=500 + rows * cols, low=-100000, high=100001)
        ref_mixed = Q_mixed.astype(np.int64) @ K_mixed.astype(np.int64).T
        mixed_out, mixed_profile = gyromatmul_i32_profile(Q_mixed, K_mixed)
        _check_eq_np(f"gyroscopic mixed {rows}x{cols}", mixed_out, ref_mixed)

        for label, prof in [("standard", profile), ("mixed-range", mixed_profile)]:
            if not prof:
                continue
            total_pairs = sum(prof.get(k, 0) for k in (
                "bulk_bulk_pairs", "bulk_spin_pairs", "bulk_dense_pairs",
                "spin_spin_pairs", "spin_dense_pairs", "dense_dense_pairs",
            ))
            print(f"    [{label} profile]"
                  f" q: bulk={prof.get('q_bulk_rows',0)} spin={prof.get('q_spin_rows',0)} dense={prof.get('q_dense_rows',0)}"
                  f" | k: bulk={prof.get('k_bulk_rows',0)} spin={prof.get('k_spin_rows',0)} dense={prof.get('k_dense_rows',0)}"
                  f" | pairs={total_pairs}")


def benchmark_linear_movement_batch(
    ext: Any,
    repeats: int,
    *,
    n_bits: int,
    grouped_requests: int,
) -> None:
    print("\n  Linear movement batching (same shape + same compiled matrix)")
    bench_n_bits = _effective_linear_n_bits(n_bits)
    if grouped_requests <= 1:
        print(f"  grouped_requests={grouped_requests}, skip")
        return
    movement_cases = [(1, 128, 256), (1, 256, 256), (4, 128, 256)]
    for batch, in_f, out_f in movement_cases:
        weight_f = torch.randn(out_f, in_f)
        scale_max = float((1 << (bench_n_bits - 1)) - 1)
        weight_scale = scale_max / max(float(weight_f.abs().max().item()), 1e-8)
        weight_i32 = torch.round(weight_f * weight_scale).to(torch.int32)
        ptr = None
        rng_state = torch.get_rng_state()
        try:
            ptr = ext.compile_matrix(weight_i32)
            if ptr is None:
                raise RuntimeError("compile_matrix returned null pointer for movement benchmark")
            linear_ref = (weight_i32.to(torch.float32) / weight_scale).t()
            splits = []
            for seed_offset in range(grouped_requests):
                torch.manual_seed(7000 + seed_offset + batch * 100 + in_f)
                splits.append(torch.randn(batch, in_f))
            stacked = torch.cat(splits, dim=0).contiguous()
            tag = f"{batch}x{in_f}x{out_f}x{grouped_requests}"
            n_items = batch * grouped_requests * in_f * out_f

            torch_loop = _time_fn(
                repeats,
                lambda: torch.cat([x @ linear_ref for x in splits], dim=0),
            )
            gyro_loop = _time_fn(
                repeats,
                lambda: torch.cat(
                    [ext.linear_forward_compiled(x, cast(int, ptr), weight_scale, 0, bench_n_bits, out_f)
                     for x in splits],
                    dim=0,
                ),
            )
            torch_batched = _time_fn(
                repeats,
                lambda: stacked @ linear_ref,
            )
            gyro_batched = _time_fn(
                repeats,
                lambda: ext.linear_forward_compiled(stacked, cast(int, ptr), weight_scale, 0, bench_n_bits, out_f),
            )
        finally:
            torch.set_rng_state(rng_state)
            if ptr:
                ext.free_compiled_matrix(ptr)

        _rate(f"torch loop {tag} [compiled linear movement]", n_items, torch_loop, repeats)
        _rate(f"gyro loop {tag} [compiled linear movement]", n_items, gyro_loop, repeats, vs=torch_loop, vs_label="torch loop")
        _rate(f"torch batched {tag} [compiled linear movement]", n_items, torch_batched, repeats)
        _rate(f"gyro batched {tag} [compiled linear movement]", n_items, gyro_batched, repeats, vs=torch_batched, vs_label="torch batched")


def _throughput_rate(items: int, elapsed_sec: float, repeats: int) -> float:
    sec = elapsed_sec / max(1, int(repeats))
    return items / sec if sec > 0 else 0.0


def benchmark_bmm_movement_batch(
    repeats: int,
    *,
    n_bits: int,
    grouped_requests: int,
) -> None:
    print("\n  BMM movement batching (same shape + same cached inputs)")
    if grouped_requests <= 1:
        print(f"  grouped_requests={grouped_requests}, skip")
        return

    qk_cases = [
        ("decode", (1, 8, 1, 16, 128)),
        ("prefill", (2, 4, 32, 32, 64)),
    ]
    for phase, (b, h, tq, tk, d) in qk_cases:
        queries = []
        keys = []
        torch_state = torch.get_rng_state()
        base_seed = 9300 + b * 100 + h * 10 + d
        try:
            for idx in range(grouped_requests):
                torch.manual_seed(base_seed + idx)
                queries.append(torch.randn(b, h, tq, d))
                torch.manual_seed(base_seed + 10_000 + idx)
                keys.append(torch.randn(b, h, tk, d))

            q_stacked = torch.cat(queries, dim=0).contiguous()
            k_stacked = torch.cat(keys, dim=0).contiguous()
            n_items = grouped_requests * b * h * tq * tk
            tag = f"bmm_qk {phase} {b}x{h}x{tq}x{tk}x{d}x{grouped_requests}"
            e_torch_loop = _time_fn(
                repeats,
                lambda: torch.cat(
                    [query @ key.transpose(-1, -2) for query, key in zip(queries, keys)],
                    dim=0,
                ),
            )
            e_gyro_loop = _time_fn(
                repeats,
                lambda: torch.cat(
                    [gyromatmul_bmm_qk_f32(query, key, n_bits) for query, key in zip(queries, keys)],
                    dim=0,
                ),
            )
            e_torch_batched = _time_fn(
                repeats,
                lambda: q_stacked @ k_stacked.transpose(-1, -2),
            )
            e_gyro_batched = _time_fn(
                repeats,
                lambda: gyromatmul_bmm_qk_f32(q_stacked, k_stacked, n_bits),
            )
        finally:
            torch.set_rng_state(torch_state)

        _rate(f"torch bmm_qk_loop_{phase} {tag}", n_items, e_torch_loop, repeats)
        _rate(f"gyro bmm_qk_loop_{phase} {tag}", n_items, e_gyro_loop, repeats, vs=e_torch_loop, vs_label="torch loop")
        _rate(f"torch bmm_qk_batch_{phase} {tag}", n_items, e_torch_batched, repeats)
        _rate(f"gyro bmm_qk_batch_{phase} {tag}", n_items, e_gyro_batched, repeats, vs=e_torch_batched, vs_label="torch batched")

    av_cases = [
        ("decode", (1, 8, 1, 16, 128)),
        ("prefill", (2, 4, 32, 32, 64)),
    ]
    for phase, (b, h, tq, tk, d) in av_cases:
        queries = []
        keys = []
        values = []
        attns = []
        torch_state = torch.get_rng_state()
        base_seed = 9400 + b * 100 + h * 10 + d
        try:
            for idx in range(grouped_requests):
                torch.manual_seed(base_seed + idx)
                query = torch.randn(b, h, tq, d)
                torch.manual_seed(base_seed + 10_000 + idx)
                key = torch.randn(b, h, tk, d)
                raw = torch.matmul(query, key.transpose(-1, -2))
                attn = torch.nn.functional.softmax(raw, dim=-1).to(torch.float32)
                torch.manual_seed(base_seed + 20_000 + idx)
                value = torch.randn(b, h, tk, d)
                queries.append(query)
                keys.append(key)
                values.append(value)
                attns.append(attn)

            attn_stacked = torch.cat(attns, dim=0).contiguous()
            value_stacked = torch.cat(values, dim=0).contiguous()
            n_items = grouped_requests * b * h * tq * d
            tag = f"bmm_av {phase} {b}x{h}x{tq}x{d}x{grouped_requests}"
            e_torch_loop = _time_fn(
                repeats,
                lambda: torch.cat(
                    [torch.matmul(attn, value) for attn, value in zip(attns, values)],
                    dim=0,
                ),
            )
            e_gyro_loop = _time_fn(
                repeats,
                lambda: torch.cat(
                    [gyromatmul_bmm_av_f32(attn, value, n_bits) for attn, value in zip(attns, values)],
                    dim=0,
                ),
            )
            e_torch_batched = _time_fn(
                repeats,
                lambda: attn_stacked @ value_stacked,
            )
            e_gyro_batched = _time_fn(
                repeats,
                lambda: gyromatmul_bmm_av_f32(attn_stacked, value_stacked, n_bits),
            )
        finally:
            torch.set_rng_state(torch_state)

        _rate(f"torch bmm_av_loop_{phase} {tag}", n_items, e_torch_loop, repeats)
        _rate(f"gyro bmm_av_loop_{phase} {tag}", n_items, e_gyro_loop, repeats, vs=e_torch_loop, vs_label="torch loop")
        _rate(f"torch bmm_av_batch_{phase} {tag}", n_items, e_torch_batched, repeats)
        _rate(f"gyro bmm_av_batch_{phase} {tag}", n_items, e_gyro_batched, repeats, vs=e_torch_batched, vs_label="torch batched")


# ---------------------------------------------------------------------------
# Benchmark: GyroMatMul kernels (isolated timing vs torch)
# ---------------------------------------------------------------------------


def benchmark_gyromatmul_kernels(
    repeats: int,
    *,
    movement_batch_requests: int = 32,
) -> None:
    print("\n=== GyroMatMul Kernels ===")

    ext = get_ext()
    if ext is None:
        print("  skipped (extension unavailable)")
        return
    for attr in ("linear_forward_compiled", "bmm_qk", "bmm_av"):
        if not hasattr(ext, attr):
            print(f"  skipped (missing {attr})")
            return
    qk_prefill_rates: list[tuple[str, float]] = []
    qk_decode_rates: list[tuple[str, float]] = []
    av_prefill_rates: list[tuple[str, float]] = []
    av_decode_rates: list[tuple[str, float]] = []
    linear_n_bits = _effective_linear_n_bits(DEFAULT_LINEAR_BITS)
    bmm_n_bits = 12

    # --- GyroLinear (native kernel)
    benchmark_linear_movement_batch(
        ext,
        repeats,
        n_bits=linear_n_bits,
        grouped_requests=movement_batch_requests,
    )
    benchmark_bmm_movement_batch(
        repeats,
        n_bits=bmm_n_bits,
        grouped_requests=movement_batch_requests,
    )
    print("\n  GyroLinear (isolated kernel vs torch.nn.Linear)")
    linear_cases = [
        (1, 128, 256), (4, 256, 256), (16, 256, 512), (32, 512, 256),
        (16, 2048, 2048), (8, 2048, 8192), (4, 8192, 2048),
    ]
    for batch, in_f, out_f in linear_cases:
        weight_f = torch.randn(out_f, in_f)
        scale_max = float((1 << (linear_n_bits - 1)) - 1)
        weight_scale = scale_max / max(float(weight_f.abs().max().item()), 1e-8)
        weight_i32 = torch.round(weight_f * weight_scale).to(torch.int32)
        ptr = None
        try:
            ptr = ext.compile_matrix(weight_i32)
            if ptr is None:
                raise RuntimeError("compile_matrix returned null pointer for gyro linear")
            linear_ref = (weight_i32.to(torch.float32) / weight_scale).t()
            x = torch.randn(batch, in_f)
            n_items = batch * in_f * out_f
            e_torch = _time_fn(repeats, lambda: x @ linear_ref)
            e_gyro = _time_fn(
                repeats,
                lambda: ext.linear_forward_compiled(
                    x,
                    cast(int, ptr),
                    weight_scale,
                    0,
                    linear_n_bits,
                    out_f,
                ),
            )
        finally:
            if ptr:
                ext.free_compiled_matrix(ptr)
        tag = f"{batch}x{in_f}x{out_f}"
        _rate(f"torch {tag} [compiled linear]", n_items, e_torch, repeats)
        _rate(f"gyro {tag} [compiled linear]", n_items, e_gyro, repeats, vs=e_torch, vs_label="torch")

    # --- Block projection (former grouped linear bundle)
    print("\n  Block projection (native kernel vs concat torch)")
    grouped_cases = [
        (4, 64, [32, 16, 16]),
        (16, 64, [24, 24, 16, 16]),
        (32, 128, [64, 32, 32]),
        (2, 2048, [2048, 2048, 2048]),
        (2, 4096, [2048, 2048]),
    ]
    for batch, in_f, out_counts in grouped_cases:
        x = torch.randn(batch, in_f)
        block_parts = []
        ref_parts = []
        ptr = None
        n_items = batch * in_f * sum(out_counts)
        try:
            for out in out_counts:
                w = torch.randn(in_f, int(out))
                scale_max = float((1 << (linear_n_bits - 1)) - 1)
                scale = scale_max / max(float(w.abs().max().item()), 1e-8)
                w_i32 = torch.round(w * scale).to(torch.int32)
                block_parts.append(w_i32.t())
                ref_parts.append((w_i32.to(torch.float32) / scale))
            block_weight = torch.cat(block_parts, dim=0).contiguous()
            block_scale_max = float((1 << (linear_n_bits - 1)) - 1)
            block_scale = block_scale_max / max(float(block_weight.abs().max().item()), 1e-8)
            block_weight_i32 = torch.round(block_weight.to(torch.float32) * block_scale).to(torch.int32)
            ptr = ext.compile_matrix(block_weight_i32)
            if ptr is None:
                raise RuntimeError("compile_matrix returned null pointer for block projection")
            e_torch = _time_fn(repeats, lambda: torch.cat([x @ w for w in ref_parts], dim=-1))
            e_gyro = _time_fn(
                repeats,
                lambda: ext.linear_forward_compiled(
                    x,
                    cast(int, ptr),
                    block_scale,
                    0,
                    linear_n_bits,
                    int(block_weight_i32.shape[0]),
                ),
            )
        finally:
            if ptr:
                ext.free_compiled_matrix(ptr)
        tag = f"{batch}x{in_f}x{'+'.join(map(str, out_counts))}"
        _rate(f"torch {tag} [compiled linear]", n_items, e_torch, repeats)
        _rate(f"gyro {tag} [compiled linear]", n_items, e_gyro, repeats, vs=e_torch, vs_label="torch")

    # --- bmm_qk ---
    print("\n  bmm_qk (isolated kernel vs torch.matmul)")
    qk_cases = [
        (2, 4, 32, 32, 64), (1, 8, 1, 16, 128),
        (2, 8, 128, 128, 128), (1, 16, 256, 256, 128),
    ]
    for b, h, tq, tk, d in qk_cases:
        query = torch.randn(b, h, tq, d)
        key = torch.randn(b, h, tk, d)
        _ = gyromatmul_bmm_qk_f32(query, key, 12)
        n_items = b * h * tq * tk
        e_torch = _time_fn(repeats, lambda: torch.matmul(query, key.transpose(-1, -2)))
        e_gyro = _time_fn(repeats, lambda: gyromatmul_bmm_qk_f32(query, key, 12))
        tag = f"{b}x{h}x{tq}x{tk}x{d}"
        phase = "decode" if tq == 1 else "prefill"
        _rate(f"torch bmm_qk {tag} [fluid bmm] [{phase}]", n_items, e_torch, repeats)
        _rate(f"gyro bmm_qk {tag} [fluid bmm] [{phase}]", n_items, e_gyro, repeats, vs=e_torch, vs_label="torch")
        ratio = (e_gyro / e_torch * 100.0) if e_torch > 0 else 0.0
        if phase == "decode":
            qk_decode_rates.append((tag, ratio))
        else:
            qk_prefill_rates.append((tag, ratio))

    # --- bmm_av ---
    print("\n  bmm_av (isolated kernel vs torch.matmul)")
    av_cases = [
        (2, 4, 32, 32, 64), (1, 8, 1, 16, 128),
        (2, 8, 128, 128, 128), (1, 16, 256, 256, 128),
    ]
    for b, h, tq, tk, d in av_cases:
        query = torch.randn(b, h, tq, d)
        key = torch.randn(b, h, tk, d)
        raw = torch.matmul(query, key.transpose(-1, -2))
        attn = torch.nn.functional.softmax(raw, dim=-1).to(torch.float32)
        value = torch.randn(b, h, tk, d)
        n_items = b * h * tq * d
        e_torch = _time_fn(repeats, lambda: torch.matmul(attn, value))
        e_gyro = _time_fn(repeats, lambda: gyromatmul_bmm_av_f32(attn, value, 12))
        tag = f"{b}x{h}x{tq}x{d}"
        phase = "decode" if tq == 1 else "prefill"
        _rate(f"torch bmm_av {tag} [fluid bmm] [{phase}]", n_items, e_torch, repeats)
        _rate(f"gyro bmm_av {tag} [fluid bmm] [{phase}]", n_items, e_gyro, repeats, vs=e_torch, vs_label="torch")
        ratio = (e_gyro / e_torch * 100.0) if e_torch > 0 else 0.0
        if phase == "decode":
            av_decode_rates.append((tag, ratio))
        else:
            av_prefill_rates.append((tag, ratio))

    if qk_prefill_rates or qk_decode_rates or av_prefill_rates or av_decode_rates:
        print("\n  Prefill vs decode summary")
        print("  Prefill (Tq > 1):")
        for tag, ratio in qk_prefill_rates:
            print(f"    bmm_qk  {tag:<22s} gyro/torch = {ratio:>4.0f}%")
        for tag, ratio in av_prefill_rates:
            print(f"    bmm_av   {tag:<22s} gyro/torch = {ratio:>4.0f}%")
        if not qk_prefill_rates and not av_prefill_rates:
            print("    (none)")

        print("  Decode (Tq == 1):")
        for tag, ratio in qk_decode_rates:
            print(f"    bmm_qk  {tag:<22s} gyro/torch = {ratio:>4.0f}%")
        for tag, ratio in av_decode_rates:
            print(f"    bmm_av   {tag:<22s} gyro/torch = {ratio:>4.0f}%")
        if not qk_decode_rates and not av_decode_rates:
            print("    (none)")

def _make_timer() -> dict[str, float]:
    return {"calls": 0.0, "total_ms": 0.0, "last_ms": 0.0}


def _wrap_timed(
    fn: Callable[..., Any],
    key: str,
    stats: dict[str, dict[str, float]],
    on_exit: Callable[[float], None] | None = None,
) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        ms = (time.perf_counter() - t0) * 1000.0
        item = stats[key]
        item["calls"] += 1.0
        item["total_ms"] += ms
        item["last_ms"] = ms
        if on_exit is not None:
            on_exit(ms)
        return out

    return wrapped


def _summarize_hotspots(
    stats: dict[str, dict[str, float]],
    *,
    show_top: int = 12,
) -> None:
    if not stats:
        print("  no hotspot data captured")
        return

    rows = sorted(
        ((name, vals) for name, vals in stats.items() if vals["calls"] > 0),
        key=lambda item: item[1]["total_ms"],
        reverse=True,
    )
    for name, vals in rows[:show_top]:
        calls = int(vals["calls"])
        total_ms = float(vals["total_ms"])
        avg_ms = total_ms / calls if calls else 0.0
        print(
            f"  {name}: calls={calls:4d} total_ms={total_ms:.2f} "
            f"avg_ms={avg_ms:.4f} last_ms={vals['last_ms']:.4f}"
        )


def _summarize_linear_regime_by_shape(
    reg_stats: dict[str, dict[str, int]],
    *,
    show_top: int = 12,
) -> None:
    if not reg_stats:
        print("  no linear regime shape data captured")
        return

    print("  linear regime by source matrix (top modules)")
    rows = sorted(
        ((name, vals) for name, vals in reg_stats.items() if vals["calls"] > 0),
        key=lambda item: item[1]["calls"],
        reverse=True,
    )
    for name, vals in rows[:show_top]:
        calls = int(vals["calls"])
        dense = int(vals["dense"])
        bulk8 = int(vals["bulk8"])
        matmul_val = int(vals["matmul"])
        total = dense + bulk8 + matmul_val
        dense_pct = (dense / total * 100.0) if total else 0.0
        bulk_pct = (bulk8 / total * 100.0) if total else 0.0
        matmul_pct = (matmul_val / total * 100.0) if total else 0.0
        print(
            f"  {name:<58s} | calls={calls:4d} "
            f"bulk8={bulk8:6d} ({bulk_pct:4.1f}%) | "
            f"dense={dense:6d} ({dense_pct:4.1f}%) | "
            f"matmul={matmul_val:6d} ({matmul_pct:4.1f}%)"
        )


def _run_gen_with_hotspots(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    repeats: int,
    warmup: int = 2,
    do_profile: bool = False,
) -> tuple[float, int]:
    from src.tools.gyrolabe.gyromatmul._build import get_ext
    from src.tools.gyrolabe.gyromatmul import modules as gm
    from torch.profiler import ProfilerActivity, profile, record_function

    stats: dict[str, dict[str, float]] = defaultdict(_make_timer)
    module_wall: dict[str, dict[str, float]] = defaultdict(_make_timer)
    module_kernel: dict[str, dict[str, float]] = defaultdict(_make_timer)
    original: list[tuple[Any, str, Any]] = []
    linear_regime_by_shape: dict[str, dict[str, int]] = {}
    ext = get_ext()
    active_module_scopes: list[str] = []
    attention_decode_runtime_ptrs: list[tuple[str, int]] = []

    def _read_regime_counts() -> tuple[int, int, int] | None:
        if ext is None or not hasattr(ext, "get_linear_regime_counters"):
            return None
        try:
            return ext.get_linear_regime_counters()
        except Exception:
            return None

    def _record_linear_regime(tag: str, before: tuple[int, int, int] | None, after: tuple[int, int, int] | None) -> None:
        if before is None or after is None:
            return
        dense_delta = after[0] - before[0]
        bulk8_delta = after[1] - before[1]
        matmul_delta = after[2] - before[2]
        if dense_delta < 0 or bulk8_delta < 0 or matmul_delta < 0:
            return
        row = linear_regime_by_shape.setdefault(
            tag, {"calls": 0, "dense": 0, "bulk8": 0, "matmul": 0},
        )
        row["calls"] += 1
        row["dense"] += int(dense_delta)
        row["bulk8"] += int(bulk8_delta)
        row["matmul"] += int(matmul_delta)

    def _install_linear(target: Any, label: str, attr: str, source: str) -> None:
        base_fn = getattr(target, attr)
        is_linear_forward = attr == "_linear_forward"

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            call = base_fn
            if is_linear_forward:
                linear_forward = getattr(target, "_linear_ext", None)
                live = (
                    getattr(linear_forward, "linear_forward_compiled", None)
                    if linear_forward is not None
                    else None
                )
                if live is not None:
                    call = live

            active_module_scopes.append(source)
            t0 = time.perf_counter()
            before = _read_regime_counts()
            try:
                out = call(*args, **kwargs)
                ms = (time.perf_counter() - t0) * 1000.0
                after = _read_regime_counts()
                item = stats[label]
                item["calls"] += 1.0
                item["total_ms"] += ms
                item["last_ms"] = ms
                wall = module_wall[source]
                wall["calls"] += 1.0
                wall["total_ms"] += ms
                wall["last_ms"] = ms
                _record_linear_regime(source, before, after)
            finally:
                active_module_scopes.pop()
            return out
        original.append((target, attr, base_fn))
        setattr(target, attr, wrapped)

    def _record_ext_in_module(ms: float) -> None:
        if not active_module_scopes:
            return
        for scope in dict.fromkeys(active_module_scopes):
            item = module_kernel[scope]
            item["calls"] += 1.0
            item["total_ms"] += ms
            item["last_ms"] = ms

    def _install(target: Any, label: str, attr: str, source: str | None = None) -> None:
        base = cast(Callable[..., Any], getattr(target, attr))
        if source is None:
            original.append((target, attr, base))
            setattr(target, attr, _wrap_timed(base, label, stats))
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            active_module_scopes.append(source)
            t0 = time.perf_counter()
            try:
                out = base(*args, **kwargs)
                ms = (time.perf_counter() - t0) * 1000.0
            finally:
                active_module_scopes.pop()
            wall = module_wall[source]
            wall["calls"] += 1.0
            wall["total_ms"] += ms
            wall["last_ms"] = ms
            item = stats[label]
            item["calls"] += 1.0
            item["total_ms"] += ms
            item["last_ms"] = ms
            return out

        original.append((target, attr, base))
        setattr(target, attr, wrapped)

    for module_name, module in model.named_modules():
        if isinstance(module, gm.GyroMatMulAttention):
            runtime_ptr = getattr(module, "_native_decode_runtime", None)
            if isinstance(runtime_ptr, int) and runtime_ptr > 0:
                attention_decode_runtime_ptrs.append((module_name, int(runtime_ptr)))
            _install(module, "attn", "forward", source=f"attn:{module_name}")
        elif isinstance(module, gm.GyroMatMulMLP):
            _install(module, "mlp", "forward", source=f"mlp:{module_name}")
        elif isinstance(module, gm.GyroGroupedLinear):
            _install(module, "block_proj", "forward")
            source = f"block_proj_run:{module_name} out={module.out_features} in={module.in_features}"
            _install_linear(module, "block_proj_run", "_run_full", source)
            _install_linear(module, "block_proj_linear_forward", "_linear_forward", source)
        elif isinstance(module, gm.GyroLinear):
            source = f"linear:{module_name} out={module.out_features} in={module.in_features}"
            _install_linear(module, "linear_native", "_linear_forward", source)
            _install_linear(module, "linear", "forward", source)

    if ext is not None:
        ext_wrappers = (
            ("bmm_qk", "bmm_qk"),
            ("bmm_qk_gqa", "bmm_qk_gqa"),
            ("bmm_av", "bmm_av"),
            ("bmm_av_gqa", "bmm_av_gqa"),
            ("attention_decode_runtime_forward", "attention_decode_runtime_forward"),
            ("attention_decode_runtime_append_kv_f32", "attention_decode_runtime_append_kv"),
            ("mlstm_step", "mlstm_step"),
            ("linear_forward_compiled", "linear_forward_compiled:block"),
        )
        for item in ext_wrappers:
            attr, label = item
            if hasattr(ext, attr):
                original.append((ext, attr, cast(Callable[..., Any], getattr(ext, attr))))
                setattr(
                    ext,
                    attr,
                    _wrap_timed(
                        cast(Callable[..., Any], getattr(ext, attr)),
                        f"ext::{label}",
                        stats,
                        on_exit=_record_ext_in_module,
                    ),
                )

    def _summarize_module_breakdown() -> None:
        if not module_wall:
            print("  module wrapper vs kernel split: no scoped modules")
            return
        rows = sorted(module_wall.items(), key=lambda kv: kv[1]["total_ms"], reverse=True)
        print("  module wrapper vs kernel split")
        for name, wall in rows[:20]:
            kernel = module_kernel[name]["total_ms"]
            wall_ms = wall["total_ms"]
            calls = int(wall["calls"])
            py_ms = wall_ms - kernel
            kernel_pct = 100.0 * kernel / wall_ms if wall_ms > 0 else 0.0
            py_pct = 100.0 * py_ms / wall_ms if wall_ms > 0 else 0.0
            name_label = name if len(name) <= 74 else f"{name[:71]}..."
            print(
                f"    {name_label} c={calls:4d} wall={wall_ms:7.2f}ms "
                f"ext={kernel:7.2f}ms ({kernel_pct:4.1f}%) py={py_ms:7.2f}ms ({py_pct:4.1f}%)"
            )

    def _group_summary() -> None:
        stage_keys: dict[str, float] = {
            "attn": 0.0,
            "mlp": 0.0,
            "linear": 0.0,
            "ext": 0.0,
            "decode_runtime": 0.0,
            "other": 0.0,
        }
        for key, vals in stats.items():
            if vals["calls"] <= 0:
                continue
            if key.startswith("attn_") or key == "attn":
                stage_keys["attn"] += vals["total_ms"]
            elif key.startswith("mlp") or key == "mlp":
                stage_keys["mlp"] += vals["total_ms"]
            elif key == "linear" or key.startswith("linear"):
                stage_keys["linear"] += vals["total_ms"]
            elif key == "block_proj" or key == "block_proj_run":
                stage_keys["ext"] += vals["total_ms"]
            elif key.startswith("ext::"):
                stage_keys["ext"] += vals["total_ms"]
                if key.startswith("ext::attention_decode_runtime"):
                    stage_keys["decode_runtime"] += vals["total_ms"]
            else:
                stage_keys["other"] += vals["total_ms"]

        print("  stage summary")
        total = sum(item["total_ms"] for item in stats.values()) or 1.0
        top_name = ""
        top_ms = -1.0
        for name, total_ms in stage_keys.items():
            if total_ms <= 0.0:
                continue
            if total_ms > top_ms:
                top_ms = total_ms
                top_name = name
            pct = (total_ms / total) * 100.0
            print(f"  {name:13s}: {total_ms:8.2f}ms ({pct:5.1f}%)")
        if top_name:
            print(f"  bottleneck stage: {top_name} ({top_ms:.2f}ms, {(top_ms / total * 100.0):.1f}%)")
        if stage_keys["ext"] > 0.0:
            ext_pct = stage_keys["ext"] / total * 100.0
            py_pct = 100.0 - ext_pct
            print(f"  kernel dispatch: {stage_keys['ext']:.2f}ms ({ext_pct:5.1f}%)")
            print(f"  non-kernel:    {total - stage_keys['ext']:.2f}ms ({py_pct:5.1f}%)")
        if stage_keys["linear"] > 0.0 and stage_keys["ext"] > 0.0:
            print(
                "  note: linear block run is split between module wrapper (linear) "
                "and kernel path (ext::linear_forward_compiled:block)."
            )

    try:
        with torch.no_grad():
            for _ in range(max(0, warmup)):
                model.generate(input_ids, max_new_tokens=min(max_new_tokens, 2), do_sample=False)

        times: list[float] = []
        profile_decoders = (
            ext is not None
            and hasattr(ext, "attention_decode_runtime_reset_profile")
            and hasattr(ext, "attention_decode_runtime_get_profile")
            and attention_decode_runtime_ptrs
        )
        if profile_decoders:
            for _name, ptr in attention_decode_runtime_ptrs:
                try:
                    ext.attention_decode_runtime_reset_profile(int(ptr))
                except Exception:
                    pass
        if ext is not None:
            try:
                ext.reset_linear_regime_counters()
            except Exception:
                pass
        out = None
        if do_profile:
            with torch.no_grad():
                with profile(
                    activities=[ProfilerActivity.CPU],
                    record_shapes=False,
                    profile_memory=False,
                ) as prof:
                    with record_function("gyromatmul_generation_profiled"):
                        for _ in range(max(1, repeats)):
                            t0 = time.perf_counter()
                            out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
                            times.append((time.perf_counter() - t0) * 1000.0)
                print("  profiler top ops (self cpu time):")
                print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=80))
        else:
            with torch.no_grad():
                for _ in range(max(1, repeats)):
                    t0 = time.perf_counter()
                    out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
                    times.append((time.perf_counter() - t0) * 1000.0)

        print("  gyro generation detailed hotspots")
        _summarize_hotspots(stats, show_top=20)
        _summarize_linear_regime_by_shape(linear_regime_by_shape, show_top=8)
        _group_summary()
        _summarize_module_breakdown()
        print("  note: bulk8/dense/matmul here are runtime counts from linear_forward_compiled in this generation run.")
        if ext is not None and hasattr(ext, "attention_decode_runtime_get_profile"):
            phase_labels = (
                "calls",
                "total_ns",
                "qkv_proj_ns",
                "qk_k_norm_ns",
                "rope_ns",
                "cache_append_ns",
                "qk_decode_ns",
                "mask_apply_ns",
                "support_select_ns",
                "support_reduce_ns",
                "o_proj_ns",
            )
            phase_totals = [0.0 for _ in phase_labels]
            for _name, ptr in attention_decode_runtime_ptrs:
                try:
                    profile = ext.attention_decode_runtime_get_profile(int(ptr))
                    for i, v in enumerate(profile):
                        phase_totals[i] += float(v)
                except Exception:
                    pass
            if phase_totals[0] > 0.0:
                total_ns = phase_totals[1]
                print("  attention_decode_runtime native phase split")
                for idx, label in enumerate(phase_labels[1:], start=1):
                    ms = phase_totals[idx] / 1_000_000.0
                    pct = (phase_totals[idx] / total_ns) * 100.0 if total_ns > 0 else 0.0
                    print(f"    {label:<20s}: {ms:9.3f}ms ({pct:5.1f}%)")
                total_ms = total_ns / 1_000_000.0
                print(f"    {'calls':<20s}: {int(phase_totals[0])} invocations")
                print(f"    {'total':<20s}: {total_ms:9.3f}ms")
        if ext is not None:
            try:
                dense_count, bulk8_count, matmul_count = ext.get_linear_regime_counters()
                total = dense_count + bulk8_count + matmul_count
                print("  linear regime counters")
                print(
                    f"  dense: {dense_count:6d} | bulk8: {bulk8_count:6d} | matmul: {matmul_count:6d}"
                    f" (total: {total:6d})"
                )
            except Exception:
                pass
        if all(item["calls"] == 0 for item in stats.values()):
            print(
                "  warning: no instrumented GyroMatMul calls were observed; "
                "conversion or module matching may have failed."
            )
        times.sort()
        assert out is not None
        return times[len(times) // 2] / 1000.0, int(out.shape[-1] - input_ids.shape[-1])
    finally:
        for target, attr, value in reversed(original):
            setattr(target, attr, value)


# ---------------------------------------------------------------------------
# Benchmark: GyroMatMul Bolmo generation (torch vs gyromatmul ratio)
# ---------------------------------------------------------------------------


def benchmark_gyromatmul_bolmo_generation(
    repeats: int,
    *,
    run_decode_hotspot: bool = False,
) -> None:
    bench_n_bits = _effective_linear_n_bits(DEFAULT_LINEAR_BITS)
    decode_n_bits = _effective_decode_n_bits(bench_n_bits)
    print("\n=== GyroMatMul Bolmo Generation ===")
    from src.tools.gyrolabe.gyromatmul.convert import convert_bolmo as _convert_bolmo
    from src.tools.gyrolabe.bridges.bolmo_config import DEFAULT_BOLMO_MODEL_PATH as _path, load_base_bolmo as _load

    prompt = "The weather today is"
    max_new_tokens = 12
    gen_repeats = max(1, min(5, repeats))

    # --- torch baseline ---
    try:
        with _benchmark_load_context(True):
            raw = cast(Any, _load(str(_path), torch_dtype=torch.float32, low_cpu_mem_usage=True))
        raw.eval()
    except Exception as exc:
        print(f"  skipped (raw Bolmo load failed: {exc})")
        return

    try:
        tokenizer = bolmo_tokenizer_from_model(raw)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        torch_ms, torch_tokens = measure_generation_ms(
            raw, input_ids, max_new_tokens=max_new_tokens, repeats=gen_repeats, warmup=2,
        )
    except Exception as exc:
        print(f"  skipped (raw Bolmo benchmark failed: {exc})")
        del raw; gc.collect()
        return

    raw_for_gyro = copy.deepcopy(raw)
    del raw; gc.collect()

    # --- gyromatmul ---
    try:
        with _benchmark_load_context(True):
            gyro = _convert_bolmo(
                cast(Any, raw_for_gyro),
                n_bits=decode_n_bits,
            )
        gyro.eval()
        gyro_ms, gyro_tokens = measure_generation_ms(
            gyro,
            input_ids,
            max_new_tokens=max_new_tokens,
            repeats=gen_repeats,
            warmup=2,
        )
        try:
            if run_decode_hotspot:
                print("  generation hotspot debug (1-token decode)")
                _run_gen_with_hotspots(
                    gyro,
                    input_ids,
                    max_new_tokens=1,
                    repeats=1,
                    warmup=0,
                    do_profile=False,
                )
            _run_gen_with_hotspots(
                gyro,
                input_ids,
                max_new_tokens=max_new_tokens,
                repeats=1,
                warmup=0,
                do_profile=False,
            )
            print("  generation hotspot debug=separate diagnostic run")
        except Exception:
            print("  generation hotspot debug skipped")
    except Exception as exc:
        print(f"  skipped (gyromatmul Bolmo benchmark failed: {exc})")
        return

    ratio = torch_ms / max(gyro_ms, 1e-9)
    print(f"  prompt: {prompt!r}")
    print(f"  torch_ms/call={torch_ms*1000:.1f}ms | tokens={torch_tokens}")
    print(f"  gyromatmul_ms/call={gyro_ms*1000:.1f}ms | tokens={gyro_tokens}")
    print(f"  speedup ratio={ratio:.2f}x")
    if torch_tokens != gyro_tokens:
        print(f"  WARN token mismatch: torch={torch_tokens} gyro={gyro_tokens}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _verify_native_symbols() -> None:
    if not native_available():
        raise RuntimeError("Gyro native library required")
    lib = get_lib()
    if lib is None:
        raise RuntimeError("Gyro native library failed to load")
    critical = [
        "gyrolabe_signature_scan",
        "gyromatmul_i32",
        "gyromatmul_attention_decode_runtime_compile",
        "gyromatmul_attention_decode_runtime_free",
        "gyromatmul_attention_decode_runtime_reset_cache",
        "gyromatmul_attention_decode_runtime_append_kv_f32",
        "gyromatmul_attention_decode_runtime_forward_f32",
    ]
    missing = [fn for fn in critical if not has_fn(fn)]
    if missing:
        raise RuntimeError(f"Missing native symbols: {missing}")
    print(f"native: {len(critical)} critical symbols OK")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GyroLabe benchmark")
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--exact-n", type=int, nargs="*", default=[256, 4096, 65536])
    p.add_argument("--batch", type=int, nargs="*", default=[64, 256])
    p.add_argument("--gyroscopic-sizes", type=str, nargs="*", default=["64x64", "128x64", "256x64", "512x64"])
    p.add_argument("--torch-threads", type=int, default=0, help="Torch intraop/interop threads; 0 uses CPU count.")
    p.add_argument("--native-threads", type=int, default=0, help="Native OpenMP thread count; 0 follows torch setting.")
    p.add_argument(
        "--omp-wait-policy",
        default=None,
        help="Set OMP_WAIT_POLICY for native kernels.",
    )
    p.add_argument("--run-opencl", action="store_true", help="Run OpenCL tensor benchmark")
    p.add_argument("--exact-only", action="store_true")
    p.add_argument("--tensor-only", action="store_true")
    p.add_argument("--gyroscopic-only", action="store_true")
    p.add_argument("--gyromatmul", action="store_true", dest="gyromatmul_only",
                    help="Run only GyroMatMul kernel + Bolmo generation benchmarks")
    p.add_argument(
        "--movement-batch-requests",
        type=int,
        default=32,
        help="Grouped tiny movement calls (linear + bmm) to combine in movement benchmark",
    )
    p.add_argument(
        "--decode-hotspot",
        action="store_true",
        help="Run extra 1-token decode hotspot profile on gyromatmul generation",
    )
    return p.parse_args()


def _parse_gyroscopic_sizes(specs: list[str]) -> list[tuple[int, int]]:
    sizes = []
    for s in specs:
        parts = s.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid size spec: {s!r} (expected ROWSxCOLS)")
        sizes.append((int(parts[0]), int(parts[1])))
    return sizes


def main() -> None:
    args = parse_args()
    _verify_native_symbols()
    _configure_benchmark_threads(
        torch_threads=args.torch_threads,
        native_threads=args.native_threads,
        omp_wait_policy=args.omp_wait_policy,
    )
    print(f"GyroLabe benchmark  repeats={args.repeats}  native={'YES' if native_available() else 'NO'}")

    only_flags = [args.exact_only, args.tensor_only, args.gyroscopic_only, args.gyromatmul_only]
    any_only = any(only_flags)

    run_exact = args.exact_only if any_only else True
    run_tensor = args.tensor_only if any_only else True
    run_gyroscopic = args.gyroscopic_only if any_only else True
    run_gyromatmul = args.gyromatmul_only if any_only else True

    if run_exact:
        benchmark_exact_ops(args.exact_n, args.repeats)
    if run_tensor:
        benchmark_tensor_ops(
            args.batch,
            args.repeats,
            run_opencl=bool(args.run_opencl),
        )
    if run_gyroscopic:
        benchmark_gyroscopic(_parse_gyroscopic_sizes(args.gyroscopic_sizes), args.repeats)
    if run_gyromatmul:
        benchmark_gyromatmul_kernels(
            args.repeats,
            movement_batch_requests=max(1, int(args.movement_batch_requests)),
        )
        benchmark_gyromatmul_bolmo_generation(
            args.repeats,
            run_decode_hotspot=bool(args.decode_hotspot),
        )


if __name__ == "__main__":
    main()






