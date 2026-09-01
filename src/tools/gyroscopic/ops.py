"""ctypes bindings for the Gyroscopic kernel (offline / tests).

Builds ``kernel.c`` into a standalone native DLL for Python tests and helpers.
The llama.cpp inference hot path also links ``kernel.c`` (plus ``ledger.c``,
``attn.c``, ``codec.c``) into ggml-cpu — see ggml-gyroscopic CMakeLists.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

from .constants import (
    GAUGE_COUNT,
    HORIZON_SIZE,
    OMEGA_SIZE,
)

_PKG_DIR = Path(__file__).resolve().parent
_BUILD_DIR = _PKG_DIR / "_build"

K4_ID = 0
K4_W2 = 1
K4_W2P = 2
K4_F = 3

PATH_ISOTROPIC = 0
PATH_BULK_CS = 1
PATH_BULK_UNA = 2
PATH_BULK_ONA = 3
PATH_BULK_BU = 4

# GAUGE_COUNT mirrors K4 size; keep local aliases for callers.
assert GAUGE_COUNT == 4
assert HORIZON_SIZE == 64
assert OMEGA_SIZE == 4096


def _lib_name() -> str:
    if sys.platform == "win32":
        return "gyroscopic_native.dll"
    if sys.platform == "darwin":
        return "libgyroscopic_native.dylib"
    return "libgyroscopic_native.so"


def _lib_path() -> Path:
    return _BUILD_DIR / _lib_name()


_NATIVE_SRCS = (
    "kernel.c", "runtime.c", "codec.c", "attn.c", "ledger.c",
    "layer.c",
)
_NATIVE_HDRS = (
    "kernel.h", "runtime.h", "constants.h",
    "codec.h", "attn.h", "ledger.h", "layer.h",
)


def _needs_rebuild(lib: Path) -> bool:
    if not lib.is_file():
        return True
    deps = [_PKG_DIR / n for n in (*_NATIVE_SRCS, *_NATIVE_HDRS)]
    try:
        lib_m = lib.stat().st_mtime
        return any(p.stat().st_mtime > lib_m for p in deps if p.is_file())
    except OSError:
        return True


def _detect_c_compiler() -> list[str] | None:
    if sys.platform == "win32" and shutil.which("cl"):
        return ["cl", "/nologo", "/O2", "/MD", "/LD"]
    for cc in ("cc", "gcc", "clang"):
        if shutil.which(cc):
            return [cc, "-O2", "-fPIC", "-shared"]
    return None


def build_native(force: bool = False) -> Path:
    """Compile the full native tree (kernel, wave, runtime, codec, attn, ledger,
    layer) into the ctypes library."""
    lib = _lib_path()
    if not force and not _needs_rebuild(lib):
        return lib
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    srcs = [str(_PKG_DIR / n) for n in _NATIVE_SRCS]

    cc_argv = _detect_c_compiler()
    if cc_argv is None and sys.platform == "win32":
        ps1 = _PKG_DIR / "helpers" / "build_kernel_native.ps1"
        if ps1.is_file():
            cp = subprocess.run(
                ["powershell", "-NoProfile", "-File", str(ps1)],
                capture_output=True,
                text=True,
                check=False,
            )
            if cp.returncode == 0 and lib.is_file():
                return lib

    if cc_argv is None:
        raise RuntimeError("Gyroscopic: no C compiler (cl/cc/gcc/clang) found.")

    if cc_argv[0] == "cl":
        argv = cc_argv + [*srcs, f"/Fe:{lib}", f"/Fo:{_BUILD_DIR}\\"]
    else:
        argv = cc_argv + [*srcs, "-o", str(lib), "-lm"]

    cp = subprocess.run(argv, capture_output=True, text=True, cwd=str(_BUILD_DIR), check=False)
    if cp.returncode != 0:
        raise RuntimeError(
            f"Gyroscopic: kernel build failed (argv={argv!r}).\n"
            f"STDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
        )
    if not lib.is_file():
        raise FileNotFoundError(f"Gyroscopic: build finished but {lib} not found.")
    return lib


_LIB: ctypes.CDLL | None = None


class GateCounters(ctypes.Structure):
    """Mirror of hqvm_gate_counters (kernel codec gates)."""

    _fields_ = [
        ("matmul_calls", ctypes.c_uint64),
        ("matmul_pq_calls", ctypes.c_uint64),
        ("matmul_dq_calls", ctypes.c_uint64),
        ("norm_calls", ctypes.c_uint64),
        ("rope_calls", ctypes.c_uint64),
        ("attn_score_calls", ctypes.c_uint64),
        ("v_reduce_calls", ctypes.c_uint64),
        ("swiglu_calls", ctypes.c_uint64),
        ("not_implemented", ctypes.c_uint64),
    ]


class Dyad32(ctypes.Structure):
    """Mirror of hqvm_dyad32 (kernel dyadic float)."""

    _fields_ = [("bits", ctypes.c_uint32)]


class Q1Weight(ctypes.Structure):
    """Mirror of hqvm_q1_weight (kernel Q1-weight descriptor)."""

    _fields_ = [
        ("q1_data", ctypes.c_void_p),
        ("n_rows", ctypes.c_int64),
        ("n_cols", ctypes.c_int64),
        ("row_stride_bytes", ctypes.c_size_t),
    ]


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(str(build_native()))
        _bind(_LIB)
        _bind_runtime(_LIB)
        _bind_owners(_LIB)
    return _LIB


def _bind(lib: ctypes.CDLL) -> None:
    u8 = ctypes.c_uint8
    wf = ctypes.c_float * OMEGA_SIZE

    lib.gyroscopic_step_omega12.restype = ctypes.c_uint32
    lib.gyroscopic_step_omega12.argtypes = [ctypes.c_uint32, u8]

    lib.gyroscopic_apply_K4.restype = None
    lib.gyroscopic_apply_K4.argtypes = [wf, ctypes.c_int]

    lib.gyroscopic_chirality_from_signs64.restype = u8
    lib.gyroscopic_chirality_from_signs64.argtypes = [ctypes.c_uint64]

    lib.gyroscopic_signs64_from_f32.restype = ctypes.c_uint64
    lib.gyroscopic_signs64_from_f32.argtypes = [ctypes.POINTER(ctypes.c_float)]

    lib.gyroscopic_activation_chirality.restype = u8
    lib.gyroscopic_activation_chirality.argtypes = [ctypes.POINTER(ctypes.c_float)]

    lib.gyroscopic_chirality_distance.restype = ctypes.c_int
    lib.gyroscopic_chirality_distance.argtypes = [u8, u8]

    lib.gyroscopic_chirality_word6.restype = u8
    lib.gyroscopic_chirality_word6.argtypes = [ctypes.c_uint32]

    fptr = ctypes.POINTER(ctypes.c_float)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    lib.gyroscopic_kv_f32_to_word4.restype = None
    lib.gyroscopic_kv_f32_to_word4.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(u8)]

    lib.gyroscopic_word4_chirality.restype = u8
    lib.gyroscopic_word4_chirality.argtypes = [ctypes.POINTER(u8), u32p]

    lib.gyroscopic_kv_f32_block_chirality.restype = u8
    lib.gyroscopic_kv_f32_block_chirality.argtypes = [ctypes.POINTER(ctypes.c_float), u32p]

    hist64 = ctypes.c_uint32 * HORIZON_SIZE
    lib.gyroscopic_chi_hist_d_eff.restype = ctypes.c_int
    lib.gyroscopic_chi_hist_d_eff.argtypes = [hist64, u8, fptr, fptr]

    lib.gyroscopic_chi_hist_m2_eta.restype = None
    lib.gyroscopic_chi_hist_m2_eta.argtypes = [hist64, fptr, fptr]

    lib.gyroscopic_route_resonance.restype = ctypes.c_float
    lib.gyroscopic_route_resonance.argtypes = [
        u8, u8, ctypes.c_int, ctypes.c_int, u8, u8, ctypes.c_float,
    ]

    lib.gyroscopic_gravity_g1.restype = ctypes.c_float
    lib.gyroscopic_gravity_g1.argtypes = []

    lib.gyroscopic_gravity_scale.restype = ctypes.c_float
    lib.gyroscopic_gravity_scale.argtypes = [ctypes.c_int, ctypes.c_int, u8, u8]

    lib.gyroscopic_cyclic_qft.restype = None
    lib.gyroscopic_cyclic_qft.argtypes = [fptr, fptr, ctypes.c_int]

    u64 = ctypes.c_uint64
    lib.gyroscopic_mul_mod_ladder.restype = u64
    lib.gyroscopic_mul_mod_ladder.argtypes = [u64, u64, u64]
    lib.gyroscopic_exp_mod_ladder.restype = u64
    lib.gyroscopic_exp_mod_ladder.argtypes = [u64, u64, u64]
    lib.gyroscopic_multiplicative_period.restype = u64
    lib.gyroscopic_multiplicative_period.argtypes = [u64, u64, u64]
    lib.gyroscopic_comb_qft_peak.restype = ctypes.c_uint32
    lib.gyroscopic_comb_qft_peak.argtypes = [u64, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]

    tile = HORIZON_SIZE
    f32a = ctypes.c_float * (tile * tile)
    f32v = ctypes.c_float * tile

    class TileRatios(ctypes.Structure):
        _fields_ = [
            ("r_shell", ctypes.c_float),
            ("r_chi", ctypes.c_float),
            ("r_chi_minus_shell", ctypes.c_float),
            ("r_defect", ctypes.c_float),
            ("norm", ctypes.c_float),
        ]

    lib.gyroscopic_project_chi_coeffs.restype = None
    lib.gyroscopic_project_chi_coeffs.argtypes = [f32a, f32v]

    lib.gyroscopic_tile_decompose_ratios.restype = None
    lib.gyroscopic_tile_decompose_ratios.argtypes = [f32a, ctypes.POINTER(TileRatios)]

    lib.gyroscopic_chi_circulant_matvec.restype = None
    lib.gyroscopic_chi_circulant_matvec.argtypes = [f32v, f32v, f32v]

    lib.gyroscopic_tile_hybrid_matvec.restype = None
    lib.gyroscopic_tile_hybrid_matvec.argtypes = [f32a, f32v, f32v]

    lib.gyroscopic_tile_hybrid_dot_row.restype = ctypes.c_float
    lib.gyroscopic_tile_hybrid_dot_row.argtypes = [f32a, ctypes.c_int, f32v]

    lib.TileRatios = TileRatios

    u16 = ctypes.c_uint16
    u16p = ctypes.POINTER(u16)
    lib.hqvm_pack_state12.restype = u16
    lib.hqvm_pack_state12.argtypes = [u8, u8]
    lib.hqvm_step_state12_by_byte.restype = u16
    lib.hqvm_step_state12_by_byte.argtypes = [u16, u8]
    lib.hqvm_trace_word_state12.restype = u16
    lib.hqvm_trace_word_state12.argtypes = [u16, ctypes.POINTER(u8), ctypes.c_int]
    lib.hqvm_sig13_compile.restype = u16
    lib.hqvm_sig13_compile.argtypes = [ctypes.POINTER(u8), ctypes.c_int]
    lib.hqvm_sig13_compose.restype = u16
    lib.hqvm_sig13_compose.argtypes = [u16, u16]
    lib.hqvm_sig13_inv.restype = u16
    lib.hqvm_sig13_inv.argtypes = [u16]
    lib.hqvm_sig13_apply.restype = u16
    lib.hqvm_sig13_apply.argtypes = [u16, u16]
    lib.hqvm_sig13_apply_batch.restype = None
    lib.hqvm_sig13_apply_batch.argtypes = [u16p, ctypes.c_int, u16, u16p]
    lib.hqvm_route2_witnesses.restype = ctypes.c_int
    lib.hqvm_route2_witnesses.argtypes = [
        u16, u16, ctypes.POINTER(u8), ctypes.POINTER(u8),
    ]
    lib.hqvm_route2_synthesize.restype = ctypes.c_int
    lib.hqvm_route2_synthesize.argtypes = [
        u16, u16, ctypes.POINTER(u8), ctypes.POINTER(u8),
    ]
    lib.hqvm_sig13_cache_build.restype = None
    lib.hqvm_sig13_cache_build.argtypes = [u16p]
    lib.hqvm_sig13_cache_apply_batch.restype = None
    lib.hqvm_sig13_cache_apply_batch.argtypes = [
        u16p, ctypes.c_int, u16, u16p, u16p,
    ]
    lib.hqvm_sig13_apply_many_sigs.restype = None
    lib.hqvm_sig13_apply_many_sigs.argtypes = [
        u16p, ctypes.c_int, u16p, ctypes.c_int, u16p,
    ]
    lib.hqvm_sig13_compile_apply_many.restype = None
    lib.hqvm_sig13_compile_apply_many.argtypes = [
        u16p,
        ctypes.c_int,
        ctypes.POINTER(u8),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    lib.hqvm_wave_grammar_verify.restype = ctypes.c_int
    lib.hqvm_wave_grammar_verify.argtypes = [ctypes.c_void_p]

    f64p = ctypes.POINTER(ctypes.c_float)
    lib.gyroscopic_wht64_float.restype = None
    lib.gyroscopic_wht64_float.argtypes = [f64p]
    lib.gyroscopic_climate_dense_nstep.restype = None
    lib.gyroscopic_climate_dense_nstep.argtypes = [f64p, f64p, ctypes.c_int]
    lib.gyroscopic_climate_spectral_nstep.restype = None
    lib.gyroscopic_climate_spectral_nstep.argtypes = [f64p, f64p, ctypes.c_int]
    lib.gyroscopic_shell7_apply.restype = None
    lib.gyroscopic_shell7_apply.argtypes = [f64p, f64p]
    lib.gyroscopic_climate_from_kernel.restype = None
    lib.gyroscopic_climate_from_kernel.argtypes = [f64p, f64p, f64p]
    lib.hqvm_equiv2080_sector_index.restype = ctypes.c_int
    lib.hqvm_equiv2080_sector_index.argtypes = [u8, u8]
    lib.hqvm_equiv2080_apply.restype = None
    lib.hqvm_equiv2080_apply.argtypes = [f64p, f64p, f64p]
    lib.hqvm_dense4096_matvec.restype = None
    lib.hqvm_dense4096_matvec.argtypes = [f64p, f64p, f64p]


def step_omega12(state24: int, byte: int) -> int:
    return int(_lib().gyroscopic_step_omega12(state24 & 0xFFFFFF, byte & 0xFF))


def pack_state12(u6: int, v6: int) -> int:
    return int(_lib().hqvm_pack_state12(u6 & 0x3F, v6 & 0x3F))


def step_state12_by_byte(s12: int, byte: int) -> int:
    return int(_lib().hqvm_step_state12_by_byte(s12 & 0xFFF, byte & 0xFF))


def trace_word_state12(s12: int, word: bytes) -> int:
    if not word:
        return s12 & 0xFFF
    buf = (ctypes.c_uint8 * len(word))(*word)
    return int(_lib().hqvm_trace_word_state12(s12 & 0xFFF, buf, len(word)))


def sig13_compile(word: bytes) -> int:
    if not word:
        return 0
    buf = (ctypes.c_uint8 * len(word))(*word)
    return int(_lib().hqvm_sig13_compile(buf, len(word)))


def sig13_compose(left: int, right: int) -> int:
    return int(_lib().hqvm_sig13_compose(left & 0x1FFF, right & 0x1FFF))


def sig13_inv(sig: int) -> int:
    return int(_lib().hqvm_sig13_inv(sig & 0x1FFF))


def sig13_apply(s12: int, sig: int) -> int:
    return int(_lib().hqvm_sig13_apply(s12 & 0xFFF, sig & 0x1FFF))


def sig13_apply_batch(states: list[int], sig: int) -> list[int]:
    n = len(states)
    if n == 0:
        return []
    inp = (ctypes.c_uint16 * n)(*[s & 0xFFF for s in states])
    out = (ctypes.c_uint16 * n)()
    _lib().hqvm_sig13_apply_batch(inp, n, sig & 0x1FFF, out)
    return [int(out[i]) for i in range(n)]


def route2_witnesses(src12: int, tgt12: int) -> list[tuple[int, int]]:
    b1 = (ctypes.c_uint8 * 16)()
    b2 = (ctypes.c_uint8 * 16)()
    n = int(_lib().hqvm_route2_witnesses(src12 & 0xFFF, tgt12 & 0xFFF, b1, b2))
    if n != 16:
        raise RuntimeError(f"hqvm_route2_witnesses returned {n}, expected 16")
    return [(int(b1[i]), int(b2[i])) for i in range(16)]


def route2_synthesize(src12: int, tgt12: int) -> list[tuple[int, int]]:
    b1 = (ctypes.c_uint8 * 16)()
    b2 = (ctypes.c_uint8 * 16)()
    n = int(_lib().hqvm_route2_synthesize(src12 & 0xFFF, tgt12 & 0xFFF, b1, b2))
    if n != 16:
        raise RuntimeError(f"hqvm_route2_synthesize returned {n}, expected 16")
    return [(int(b1[i]), int(b2[i])) for i in range(16)]


_SIG13_CACHE = None


def sig13_cache_build() -> list[int]:
    global _SIG13_CACHE
    buf = (ctypes.c_uint16 * 8192)()
    _lib().hqvm_sig13_cache_build(buf)
    _SIG13_CACHE = buf
    return [int(buf[i]) for i in range(8192)]


def sig13_cache_apply_batch(states: list[int], sig: int, cache: list[int] | None = None) -> list[int]:
    n = len(states)
    if n == 0:
        return []
    inp = (ctypes.c_uint16 * n)(*[s & 0xFFF for s in states])
    out = (ctypes.c_uint16 * n)()
    if _SIG13_CACHE is not None and cache is None:
        cbuf = _SIG13_CACHE
    else:
        if cache is None or len(cache) != 8192:
            raise ValueError("cache must have 8192 entries (or call sig13_cache_build first)")
        cbuf = (ctypes.c_uint16 * 8192)(*[c & 0x1FFF for c in cache])
    _lib().hqvm_sig13_cache_apply_batch(inp, n, sig & 0x1FFF, cbuf, out)
    return [int(out[i]) for i in range(n)]


def sig13_apply_many_sigs(states: list[int], sigs: list[int], *, use_cache: bool = False) -> None:
    n = len(states)
    m = len(sigs)
    if n == 0 or m == 0:
        return
    inp = (ctypes.c_uint16 * n)(*[s & 0xFFF for s in states])
    sb = (ctypes.c_uint16 * m)(*[s & 0x1FFF for s in sigs])
    cbuf = _SIG13_CACHE if use_cache else None
    if use_cache and cbuf is None:
        raise RuntimeError("call sig13_cache_build() before use_cache=True")
    _lib().hqvm_sig13_apply_many_sigs(inp, n, sb, m, cbuf)


def sig13_compile_apply_many(states: list[int], words: list[bytes]) -> None:
    n = len(states)
    if n == 0 or not words:
        return
    inp = (ctypes.c_uint16 * n)(*[s & 0xFFF for s in states])
    flat = b"".join(words)
    fb = (ctypes.c_uint8 * len(flat))(*flat)
    lens = (ctypes.c_int * len(words))(*[len(w) for w in words])
    _lib().hqvm_sig13_compile_apply_many(inp, n, fb, lens, len(words))


def wht64(data: list[float]) -> list[float]:
    if len(data) != 64:
        raise ValueError("wht64 expects length 64")
    buf = (ctypes.c_float * 64)(*data)
    _lib().gyroscopic_wht64_float(buf)
    return [float(buf[i]) for i in range(64)]


def climate_dense_nstep(x: list[float], M: list[float], n_steps: int) -> list[float]:
    if len(x) != 64 or len(M) != 64 * 64:
        raise ValueError("climate_dense_nstep expects x[64], M[4096]")
    xb = (ctypes.c_float * 64)(*x)
    Mb = (ctypes.c_float * (64 * 64))(*M)
    _lib().gyroscopic_climate_dense_nstep(xb, Mb, int(n_steps))
    return [float(xb[i]) for i in range(64)]


def climate_spectral_nstep(x: list[float], phi: list[float], n_steps: int) -> list[float]:
    if len(x) != 64 or len(phi) != 64:
        raise ValueError("climate_spectral_nstep expects x[64], phi[64]")
    xb = (ctypes.c_float * 64)(*x)
    pb = (ctypes.c_float * 64)(*phi)
    _lib().gyroscopic_climate_spectral_nstep(xb, pb, int(n_steps))
    return [float(xb[i]) for i in range(64)]


def shell7_apply(chi: list[float], gains7: list[float]) -> list[float]:
    if len(chi) != 64 or len(gains7) != 7:
        raise ValueError("shell7_apply expects chi[64], gains[7]")
    xb = (ctypes.c_float * 64)(*chi)
    gb = (ctypes.c_float * 7)(*gains7)
    _lib().gyroscopic_shell7_apply(xb, gb)
    return [float(xb[i]) for i in range(64)]


def climate_from_kernel(f: list[float]) -> tuple[list[float], list[float]]:
    if len(f) != 64:
        raise ValueError("climate_from_kernel expects f[64]")
    fb = (ctypes.c_float * 64)(*f)
    Mb = (ctypes.c_float * (64 * 64))()
    pb = (ctypes.c_float * 64)()
    _lib().gyroscopic_climate_from_kernel(fb, Mb, pb)
    return [float(Mb[i]) for i in range(64 * 64)], [float(pb[i]) for i in range(64)]


EQUIV2080_GAINS = 2080


def equiv2080_sector_index(du: int, dv: int) -> int:
    return int(_lib().hqvm_equiv2080_sector_index(du & 0x3F, dv & 0x3F))


def equiv2080_apply(psi: list[float], gains: list[float]) -> list[float]:
    if len(psi) != OMEGA_SIZE or len(gains) != EQUIV2080_GAINS:
        raise ValueError(
            f"equiv2080_apply expects psi[{OMEGA_SIZE}], gains[{EQUIV2080_GAINS}]"
        )
    pb = (ctypes.c_float * OMEGA_SIZE)(*psi)
    out = (ctypes.c_float * OMEGA_SIZE)()
    gb = (ctypes.c_float * EQUIV2080_GAINS)(*gains)
    _lib().hqvm_equiv2080_apply(pb, out, gb)
    return [float(out[i]) for i in range(OMEGA_SIZE)]


def dense4096_matvec(M: list[float], x: list[float]) -> list[float]:
    if len(M) != OMEGA_SIZE * OMEGA_SIZE or len(x) != OMEGA_SIZE:
        raise ValueError("dense4096_matvec expects M[4096*4096], x[4096]")
    mb = (ctypes.c_float * (OMEGA_SIZE * OMEGA_SIZE))(*M)
    xb = (ctypes.c_float * OMEGA_SIZE)(*x)
    yb = (ctypes.c_float * OMEGA_SIZE)()
    _lib().hqvm_dense4096_matvec(mb, xb, yb)
    return [float(yb[i]) for i in range(OMEGA_SIZE)]


def wave_grammar_verify() -> bool:
    return int(_lib().hqvm_wave_grammar_verify(None)) == 0


def apply_K4(psi: list[float], gate: int) -> list[float]:
    if len(psi) != OMEGA_SIZE:
        raise ValueError(f"psi must be length {OMEGA_SIZE}, got {len(psi)}")
    buf = (ctypes.c_float * OMEGA_SIZE)(*psi)
    _lib().gyroscopic_apply_K4(buf, int(gate))
    return list(buf)


def chirality_from_signs64(signs: int) -> int:
    return int(_lib().gyroscopic_chirality_from_signs64(signs & 0xFFFFFFFFFFFFFFFF))


def activation_chirality(x: list[float]) -> int:
    if len(x) != 64:
        raise ValueError("activation vector must be length 64")
    buf = (ctypes.c_float * 64)(*x)
    return int(_lib().gyroscopic_activation_chirality(buf))


def chirality_distance(a: int, b: int) -> int:
    return int(_lib().gyroscopic_chirality_distance(a & 0xFF, b & 0xFF))


def chirality_word6(state24: int) -> int:
    return int(_lib().gyroscopic_chirality_word6(state24 & 0xFFFFFF))


def kv_f32_block_chirality(x: list[float], state24: int | None = None) -> tuple[int, int]:
    """Serialize 64-float block through word4→Ω; return (chi6, state24_out)."""
    if len(x) != 64:
        raise ValueError("block must be length 64")
    buf = (ctypes.c_float * 64)(*x)
    s = ctypes.c_uint32(state24 or 0)
    chi = int(_lib().gyroscopic_kv_f32_block_chirality(buf, ctypes.byref(s)))
    return chi, int(s.value)


def kv_f32_to_word4(x: list[float]) -> list[int]:
    """Map a 64-float block to its 4-byte holonomic word (bridge serializer)."""
    if len(x) != 64:
        raise ValueError("block must be length 64")
    buf = (ctypes.c_float * 64)(*x)
    out = (ctypes.c_uint8 * 4)()
    _lib().gyroscopic_kv_f32_to_word4(buf, out)
    return [int(out[i]) for i in range(4)]


def serialize_4096_to_hqvm_bytes(v: list[float]) -> list[int]:
    """Holonomic encoder: 4096-dim vector -> 256-byte stream (64 blocks x 4 bytes).

    Each 64-wide block maps to a depth-4 kernel word via kv_f32_to_word4; the
    resulting byte stream is exactly what step_omega12 walks to build the
    4096-cell Omega (u6*64+v6) occupation. Companion consumer: probe _omega_cell_after_bytes.
    """
    if len(v) != OMEGA_SIZE:
        raise ValueError(f"vector must be length {OMEGA_SIZE}")
    out: list[int] = []
    for b in range(0, OMEGA_SIZE, HORIZON_SIZE):
        out.extend(kv_f32_to_word4(v[b:b + HORIZON_SIZE]))
    return out


def chi_hist_d_eff(hist: list[int], chi_q: int) -> tuple[int, float, float]:
    """Percolation-aware Hamming aperture from 64-bin occupation histogram."""
    if len(hist) != 64:
        raise ValueError("hist must be length 64")
    hbuf = (ctypes.c_uint32 * 64)(*hist)
    m2 = ctypes.c_float()
    eta = ctypes.c_float()
    d = int(_lib().gyroscopic_chi_hist_d_eff(hbuf, chi_q & 0x3F, ctypes.byref(m2), ctypes.byref(eta)))
    return d, float(m2.value), float(eta.value)


def chi_hist_m2_eta(hist: list[int]) -> tuple[float, float]:
    """Rényi-2 effective support M̂₂ = W²/Σh² and spectral damping η from a 64-bin occupation histogram (hQVM_QuBEC_Theory.md §21.3)."""
    if len(hist) != 64:
        raise ValueError("hist must be length 64")
    hbuf = (ctypes.c_uint32 * 64)(*hist)
    m2 = ctypes.c_float()
    eta = ctypes.c_float()
    _lib().gyroscopic_chi_hist_m2_eta(hbuf, ctypes.byref(m2), ctypes.byref(eta))
    return float(m2.value), float(eta.value)


def route_resonance(
    chi_act: int,
    chi_weight: int,
    layer: int,
    total_layers: int,
    g_layer: float,
    *,
    k4_char: int = 0,
    shell: int = 0,
) -> float:
    return float(
        _lib().gyroscopic_route_resonance(
            chi_act & 0xFF,
            chi_weight & 0xFF,
            int(layer),
            int(total_layers),
            k4_char & 0xFF,
            shell & 0xFF,
            ctypes.c_float(g_layer),
        )
    )


def gravity_g1() -> float:
    return float(_lib().gyroscopic_gravity_g1())


def gravity_scale(layer: int, total_layers: int, k4_char: int = 0, shell: int = 0) -> float:
    return float(_lib().gyroscopic_gravity_scale(int(layer), int(total_layers), k4_char & 0xFF, shell & 0xFF))


def cyclic_qft(re: list[float], im: list[float], n_bits: int) -> tuple[list[float], list[float]]:
    """Native radix-2 cyclic QFT over Z_{2^n_bits} (WHT-atom butterflies)."""
    n = 1 << n_bits
    if len(re) != n or len(im) != n:
        raise ValueError(f"re/im must be length {n}")
    re_buf = (ctypes.c_float * n)(*re)
    im_buf = (ctypes.c_float * n)(*im)
    _lib().gyroscopic_cyclic_qft(re_buf, im_buf, int(n_bits))
    return list(re_buf), list(im_buf)


def mul_mod_ladder(y: int, multiplier: int, n: int) -> int:
    """Shift-add modular multiply (byte-ledger arithmetic primitive)."""
    return int(_lib().gyroscopic_mul_mod_ladder(y, multiplier, n))


def exp_mod_ladder(a: int, x: int, n: int) -> int:
    """Modular exponentiation via the multiply ladder."""
    return int(_lib().gyroscopic_exp_mod_ladder(a, x, n))


def multiplicative_period(a: int, n: int, max_len: int) -> int:
    """Steps until a^k == 1 mod n, or 0 if not found within max_len."""
    return int(_lib().gyroscopic_multiplicative_period(a, n, max_len))


def comb_qft_peak(period: int, q_bits: int) -> tuple[int, float] | None:
    """Build period comb, run native cyclic QFT, return (peak_index, amplitude)."""
    amp = ctypes.c_float()
    peak = int(_lib().gyroscopic_comb_qft_peak(period, q_bits, ctypes.byref(amp)))
    if peak == 0:
        return None
    return peak, float(amp.value)


TILE_SIZE = 64


def tile_hybrid_matvec(W: list[float], x: list[float]) -> list[float]:
    """64x64 hybrid matvec via native kernel (P_chi + defect)."""
    n = TILE_SIZE
    if len(W) != n * n or len(x) != n:
        raise ValueError(f"W must be {n*n} and x must be {n}")
    Wb = (ctypes.c_float * (n * n))(*[float(v) for v in W])
    xb = (ctypes.c_float * n)(*[float(v) for v in x])
    yb = (ctypes.c_float * n)()
    _lib().gyroscopic_tile_hybrid_matvec(Wb, xb, yb)
    return list(yb)


def tile_decompose_ratios(W: list[float]) -> dict[str, float]:
    n = TILE_SIZE
    if len(W) != n * n:
        raise ValueError(f"W must be length {n*n}")
    Wb = (ctypes.c_float * (n * n))(*[float(v) for v in W])
    out = _lib().TileRatios()
    _lib().gyroscopic_tile_decompose_ratios(Wb, ctypes.byref(out))
    return {
        "r_shell": float(out.r_shell),
        "r_chi": float(out.r_chi),
        "r_chi_minus_shell": float(out.r_chi_minus_shell),
        "r_defect": float(out.r_defect),
        "norm": float(out.norm),
    }


def ensure_ledger(*args, **kwargs):
    """Ensure the thin HQVMLEDS production ledger exists. See ``ledger``."""
    from .ledger import ensure_ledger as _ensure

    return _ensure(*args, **kwargs)


def write_ledger(*args, **kwargs):
    """Write the thin HQVMLEDS production ledger. See ``ledger``."""
    from .ledger import write_ledger as _write

    return _write(*args, **kwargs)


# ---------------------------------------------------------------------------
# Native Gyroscopic Inference Loop: request cells + genealogy (runtime.c).
# ---------------------------------------------------------------------------

RT_SEED_REST = 0
RT_SEED_EQUALITY_HORIZON = 1
RT_SEED_SHELL = 2
RT_SEED_OMEGA = 3

RT_LOG_CELL_RESET = 0xFFFFFFFF
RT_REQUEST_CELL_ID = 0

RT_PROFILE_CHIRALITY = 0
RT_PROFILE_SHELL = 1


class RtCell(ctypes.Structure):
    """Mirror of hqvm_rt_cell (packed layout; see runtime.h field order)."""

    _fields_ = [
        ("step", ctypes.c_uint64),
        ("resonance_key", ctypes.c_uint32),
        ("omega_sig", ctypes.c_int32),
        ("omega12", ctypes.c_uint16),
        ("chi_hist64", ctypes.c_uint16 * 64),
        ("shell_hist7", ctypes.c_uint16 * 7),
        ("family_hist4", ctypes.c_uint16 * 4),
        ("parity_O12", ctypes.c_uint16),
        ("parity_E12", ctypes.c_uint16),
        ("last_byte", ctypes.c_uint8),
        ("word4", ctypes.c_uint8 * 4),
        ("open_word", ctypes.c_uint8 * 4),
        ("word_len", ctypes.c_uint8),
        ("has_closed_word", ctypes.c_uint8),
        ("chi_ring64", ctypes.c_uint8 * 64),
        ("family_ring64", ctypes.c_uint8 * 64),
        ("ring_pos", ctypes.c_uint8),
        ("ring_valid_len", ctypes.c_uint8),
        ("parity_bit", ctypes.c_uint8),
    ]


class RtPool(ctypes.Structure):
    _fields_ = [
        ("capacity", ctypes.c_uint32),
        ("profile_id", ctypes.c_uint16),
        ("cells", ctypes.POINTER(RtCell)),
    ]


class RtSlcp(ctypes.Structure):
    """Mirror of hqvm_rt_slcp_t (Runtime 13.2)."""

    _fields_ = [
        ("cell_id", ctypes.c_uint32),
        ("step", ctypes.c_uint64),
        ("omega12", ctypes.c_int32),
        ("state24", ctypes.c_int32),
        ("last_byte", ctypes.c_uint8),
        ("_pad0", ctypes.c_uint8 * 3),
        ("family", ctypes.c_int32),
        ("micro_ref", ctypes.c_int32),
        ("q6", ctypes.c_int32),
        ("chi6", ctypes.c_int32),
        ("shell", ctypes.c_int32),
        ("horizon_distance", ctypes.c_int32),
        ("ab_distance", ctypes.c_int32),
        ("omega_sig", ctypes.c_int32),
        ("parity_O12", ctypes.c_uint16),
        ("parity_E12", ctypes.c_uint16),
        ("parity_bit", ctypes.c_uint8),
        ("_pad1", ctypes.c_uint8 * 3),
        ("resonance_key", ctypes.c_uint32),
        ("current_resonance", ctypes.c_int32),
        ("spectral64", ctypes.c_float * 64),
    ]


def _bind_runtime(lib: ctypes.CDLL) -> None:
    u8p = ctypes.POINTER(ctypes.c_uint8)
    lib.hqvm_rt_rule_hash.restype = ctypes.c_uint64
    lib.hqvm_rt_rule_hash.argtypes = []
    lib.hqvm_rt_enabled.restype = ctypes.c_int
    lib.hqvm_rt_enabled.argtypes = []
    lib.hqvm_rt_cell_init.restype = None
    lib.hqvm_rt_cell_init.argtypes = [ctypes.POINTER(RtCell), ctypes.c_int, ctypes.c_uint8, ctypes.c_uint8]
    lib.hqvm_rt_ingest_word.restype = None
    lib.hqvm_rt_ingest_word.argtypes = [ctypes.POINTER(RtCell), u8p]
    lib.hqvm_rt_ingest_bytes.restype = None
    lib.hqvm_rt_ingest_bytes.argtypes = [ctypes.POINTER(RtCell), u8p, ctypes.c_int]
    lib.hqvm_rt_resonance_key_of.restype = ctypes.c_uint32
    lib.hqvm_rt_resonance_key_of.argtypes = [ctypes.POINTER(RtCell), ctypes.c_uint16]
    lib.hqvm_rt_pool_create.restype = ctypes.POINTER(RtPool)
    lib.hqvm_rt_pool_create.argtypes = [ctypes.c_uint32, ctypes.c_uint16]
    lib.hqvm_rt_pool_free.restype = None
    lib.hqvm_rt_pool_free.argtypes = [ctypes.POINTER(RtPool)]
    lib.hqvm_rt_pool_cell.restype = ctypes.POINTER(RtCell)
    lib.hqvm_rt_pool_cell.argtypes = [ctypes.POINTER(RtPool), ctypes.c_uint32]
    lib.hqvm_rt_pool_ingest_word.restype = ctypes.c_int
    lib.hqvm_rt_pool_ingest_word.argtypes = [ctypes.POINTER(RtPool), ctypes.c_uint32, u8p]
    lib.hqvm_rt_request_reset.restype = None
    lib.hqvm_rt_request_reset.argtypes = [ctypes.c_int]
    lib.hqvm_rt_request_ingest_bytes.restype = None
    lib.hqvm_rt_request_ingest_bytes.argtypes = [u8p, ctypes.c_int]
    lib.hqvm_rt_request_cell.restype = ctypes.POINTER(RtCell)
    lib.hqvm_rt_request_cell.argtypes = []
    lib.hqvm_rt_log_configure.restype = ctypes.c_int
    lib.hqvm_rt_log_configure.argtypes = [ctypes.c_char_p]
    lib.hqvm_rt_log_close.restype = None
    lib.hqvm_rt_log_close.argtypes = []
    lib.hqvm_rt_log_events.restype = ctypes.c_uint64
    lib.hqvm_rt_log_events.argtypes = []
    lib.hqvm_rt_log_requests.restype = ctypes.c_uint64
    lib.hqvm_rt_log_requests.argtypes = []
    lib.hqvm_rt_snapshot_header_fill.restype = None
    lib.hqvm_rt_snapshot_header_fill.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.hqvm_rt_cell_checkpoint.restype = None
    lib.hqvm_rt_cell_checkpoint.argtypes = [ctypes.POINTER(RtCell), ctypes.c_void_p]
    lib.hqvm_rt_chi_distance.restype = ctypes.c_int
    lib.hqvm_rt_chi_distance.argtypes = [ctypes.POINTER(RtCell), ctypes.POINTER(RtCell)]
    lib.hqvm_rt_group_cells.restype = ctypes.c_int
    lib.hqvm_rt_group_cells.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32,
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint16),
    ]
    lib.hqvm_rt_polar_score.restype = ctypes.c_float
    lib.hqvm_rt_polar_score.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.hqvm_rt_stock_ops_add.restype = None
    lib.hqvm_rt_stock_ops_add.argtypes = [ctypes.c_uint32]
    lib.hqvm_rt_stock_ops_total.restype = ctypes.c_uint64
    lib.hqvm_rt_stock_ops_total.argtypes = []
    lib.hqvm_rt_prefilter_inc.restype = None
    lib.hqvm_rt_prefilter_inc.argtypes = []
    lib.hqvm_rt_prefilter_calls.restype = ctypes.c_uint64
    lib.hqvm_rt_prefilter_calls.argtypes = []
    lib.hqvm_rt_prefilter_skipped.restype = ctypes.c_uint64
    lib.hqvm_rt_prefilter_skipped.argtypes = []
    lib.hqvm_rt_prefilter_report.restype = None
    lib.hqvm_rt_prefilter_report.argtypes = [ctypes.c_int64, ctypes.c_int64]
    lib.hqvm_rt_group_enabled.restype = ctypes.c_int
    lib.hqvm_rt_group_enabled.argtypes = []
    lib.hqvm_rt_group_report.restype = None
    lib.hqvm_rt_group_report.argtypes = [ctypes.c_int64, ctypes.c_int64]
    lib.hqvm_rt_group_calls.restype = ctypes.c_uint64
    lib.hqvm_rt_group_calls.argtypes = []
    lib.hqvm_rt_group_rows.restype = ctypes.c_uint64
    lib.hqvm_rt_group_rows.argtypes = []
    lib.hqvm_rt_group_groups.restype = ctypes.c_uint64
    lib.hqvm_rt_group_groups.argtypes = []
    lib.hqvm_rt_counters_request_reset.restype = None
    lib.hqvm_rt_counters_request_reset.argtypes = []
    lib.hqvm_rt_log_begin_session.restype = ctypes.c_int
    lib.hqvm_rt_log_begin_session.argtypes = [ctypes.c_uint32]
    lib.hqvm_rt_slcp_fill.restype = None
    lib.hqvm_rt_slcp_fill.argtypes = [
        ctypes.POINTER(RtCell), ctypes.c_uint32, ctypes.POINTER(RtPool), ctypes.POINTER(RtSlcp),
    ]
    lib.hqvm_rt_bucket_population.restype = ctypes.c_int32
    lib.hqvm_rt_bucket_population.argtypes = [ctypes.POINTER(RtPool), ctypes.c_uint32]
    lib.hqvm_rt_bucket_cells.restype = ctypes.c_int
    lib.hqvm_rt_bucket_cells.argtypes = [
        ctypes.POINTER(RtPool), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
    ]
    lib.hqvm_rt_co_resonant_count.restype = ctypes.c_int
    lib.hqvm_rt_co_resonant_count.argtypes = [ctypes.POINTER(RtPool), ctypes.c_uint32]
    lib.hqvm_rt_cells_on_shell.restype = ctypes.c_int
    lib.hqvm_rt_cells_on_shell.argtypes = [
        ctypes.POINTER(RtPool), ctypes.c_int, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
    ]
    lib.hqvm_rt_cells_with_chi6.restype = ctypes.c_int
    lib.hqvm_rt_cells_with_chi6.argtypes = [
        ctypes.POINTER(RtPool), ctypes.c_uint8, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
    ]
    lib.hqvm_rt_cells_with_signature.restype = ctypes.c_int
    lib.hqvm_rt_cells_with_signature.argtypes = [
        ctypes.POINTER(RtPool), ctypes.c_int32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
    ]
    lib.hqvm_rt_medium_open.restype = ctypes.c_int
    lib.hqvm_rt_medium_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint32]
    lib.hqvm_rt_medium_ingest.restype = ctypes.c_int
    lib.hqvm_rt_medium_ingest.argtypes = [u8p, ctypes.c_int, ctypes.c_int]
    lib.hqvm_rt_medium_close.restype = ctypes.c_int
    lib.hqvm_rt_medium_close.argtypes = []
    lib.hqvm_rt_medium_last_slcp.restype = ctypes.POINTER(RtSlcp)
    lib.hqvm_rt_medium_last_slcp.argtypes = []
    lib.hqvm_rt_medium_cell.restype = ctypes.POINTER(RtCell)
    lib.hqvm_rt_medium_cell.argtypes = []


def _bind_owners(lib: ctypes.CDLL) -> None:
    """Bind codec/ledger/attn product faces formerly under hosting."""

    # GateCounters is defined at module level (see top of this module) so the
    # static type checker sees `lib.GateCounters` as a ctypes.Structure rather
    # than a _NamedFuncPointer attribute lookup.

    lib.hqvm_sidecar_ready.restype = ctypes.c_int
    lib.hqvm_sidecar_ready.argtypes = []
    lib.hqvm_sidecar_reset_session.restype = None
    lib.hqvm_sidecar_reset_session.argtypes = []
    lib.hqvm_gate_counters_reset.restype = None
    lib.hqvm_gate_counters_reset.argtypes = []
    lib.hqvm_gate_counters_snapshot.restype = None
    lib.hqvm_gate_counters_snapshot.argtypes = [ctypes.POINTER(GateCounters)]
    lib.hqvm_norm_ruler_dyad.restype = ctypes.c_int
    lib.hqvm_norm_ruler_dyad.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_float,
    ]
    lib.hqvm_rope_qk_dyad.restype = ctypes.c_int
    lib.hqvm_rope_qk_dyad.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    lib.hqvm_ffn_gate_dyad.restype = ctypes.c_int
    lib.hqvm_ffn_gate_dyad.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
        ctypes.c_uint8, ctypes.c_uint8,
    ]
    lib.hqvm_v_reduce_dyad.restype = ctypes.c_int
    lib.hqvm_v_reduce_dyad.argtypes = [
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
    ]
    lib.hqvm_attn_head_scores_dyad.restype = ctypes.c_int
    lib.hqvm_attn_head_scores_dyad.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int64,
        ctypes.c_uint8, ctypes.c_int, ctypes.c_int, ctypes.c_float,
    ]
    lib.hqvm_ffn_native_enabled.restype = ctypes.c_int
    lib.hqvm_ffn_native_enabled.argtypes = []
    lib.hqvm_vreduce_native_enabled.restype = ctypes.c_int
    lib.hqvm_vreduce_native_enabled.argtypes = []
    lib.hqvm_attn_scores_native_enabled.restype = ctypes.c_int
    lib.hqvm_attn_scores_native_enabled.argtypes = []
    lib.hqvm_dyad_q8_cache_row_score.restype = ctypes.c_float
    lib.hqvm_dyad_q8_cache_row_score.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float,
    ]

    # Dyad32 and Q1Weight are defined at module level (see top of this module) so
    # the static type checker sees `lib.Dyad32` as a ctypes.Structure rather
    # than a _NamedFuncPointer attribute lookup.

    lib.hqvm_matmul_dyad.restype = ctypes.c_int
    lib.hqvm_matmul_dyad.argtypes = [
        ctypes.POINTER(Q1Weight), ctypes.POINTER(Dyad32), ctypes.POINTER(Dyad32),
    ]
    lib.hqvm_matmul_dq_selftest.restype = ctypes.c_int
    lib.hqvm_matmul_dq_selftest.argtypes = [
        ctypes.POINTER(Q1Weight), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.hqvm_dyad32_from_f32.restype = Dyad32
    lib.hqvm_dyad32_from_f32.argtypes = [ctypes.c_float]
    lib.hqvm_dyad32_to_f32.restype = ctypes.c_float
    lib.hqvm_dyad32_to_f32.argtypes = [Dyad32]

    lib.Dyad32 = Dyad32
    lib.Q1Weight = Q1Weight
    lib.GateCounters = GateCounters
    # Compat aliases for older gate scripts.
    lib.HostCounters = GateCounters


def host_sidecar_ready() -> bool:
    return bool(_lib().hqvm_sidecar_ready())


def host_reset_session() -> None:
    _lib().hqvm_sidecar_reset_session()


def host_counters_reset() -> None:
    _lib().hqvm_gate_counters_reset()


def host_counters_snapshot() -> dict[str, int]:
    out = _lib().GateCounters()
    _lib().hqvm_gate_counters_snapshot(ctypes.byref(out))
    return {
        "matmul_calls": int(out.matmul_calls),
        "matmul_pq_calls": int(out.matmul_pq_calls),
        "matmul_dq_calls": int(out.matmul_dq_calls),
        "norm_calls": int(out.norm_calls),
        "rope_calls": int(out.rope_calls),
        "attn_score_calls": int(out.attn_score_calls),
        "v_reduce_calls": int(out.v_reduce_calls),
        "swiglu_calls": int(out.swiglu_calls),
        "not_implemented": int(out.not_implemented),
    }


def host_analyze_tile(W: list[float]) -> dict[str, float]:
    """Alias to kernel tile_decompose_ratios (ex-hosting probe)."""
    return tile_decompose_ratios(W)


def host_norm_stub_rc() -> int:
    """Null-arg probe: must reject cleanly."""
    return int(_lib().hqvm_norm_ruler_dyad(None, None, 0, None, 0.0))


def host_norm_ruler_dyad(
    x_floats: list[float],
    *,
    g: list[float] | None = None,
    g0: float = 1.0,
) -> tuple[int, dict[str, int], list[float]]:
    """Run codec H3 Delta-ruler norm; returns (rc, counters, y_out)."""
    lib = _lib()
    n = len(x_floats)
    xin = (Dyad32 * n)()
    xout = (Dyad32 * n)()
    for i, v in enumerate(x_floats):
        xin[i] = lib.hqvm_dyad32_from_f32(float(v))
    g_arr = None
    if g is not None:
        if len(g) != n:
            raise ValueError("g must match x length")
        g_arr = (ctypes.c_float * n)(*[float(v) for v in g])
    host_counters_reset()
    rc = int(lib.hqvm_norm_ruler_dyad(xin, xout, n, g_arr, float(g0)))
    snap = host_counters_snapshot()
    y = [float(lib.hqvm_dyad32_to_f32(xout[i])) for i in range(n)]
    return rc, snap, y


def host_rope_qk_dyad(
    q_floats: list[float],
    k_floats: list[float],
    *,
    n_heads: int,
    gqa_ratio: int,
    token_pos: int,
) -> tuple[int, dict[str, int], list[float], list[float]]:
    """Run codec H4 RoPE on dyad Q/K heads; returns (rc, counters, q_out, k_out)."""
    lib = _lib()
    head = 128
    n_kv = max(1, n_heads // gqa_ratio)
    if len(q_floats) != n_heads * head:
        raise ValueError(f"q must have length {n_heads * head}")
    if len(k_floats) != n_kv * head:
        raise ValueError(f"k must have length {n_kv * head}")
    Q = (Dyad32 * (n_heads * head))()
    K = (Dyad32 * (n_kv * head))()
    for i, v in enumerate(q_floats):
        Q[i] = lib.hqvm_dyad32_from_f32(float(v))
    for i, v in enumerate(k_floats):
        K[i] = lib.hqvm_dyad32_from_f32(float(v))
    host_counters_reset()
    rc = int(lib.hqvm_rope_qk_dyad(Q, K, int(n_heads), int(gqa_ratio), int(token_pos)))
    snap = host_counters_snapshot()
    q_out = [float(lib.hqvm_dyad32_to_f32(Q[i])) for i in range(n_heads * head)]
    k_out = [float(lib.hqvm_dyad32_to_f32(K[i])) for i in range(n_kv * head)]
    return rc, snap, q_out, k_out


def host_ffn_gate_dyad(
    gate_floats: list[float],
    up_floats: list[float],
    *,
    fam: int = 0,
    Nc: int = 3,
) -> tuple[int, dict[str, int], list[float]]:
    """H7 codec FFN on dyad lanes: product = stock SwiGLU unless GYRO_FFN_NATIVE=1."""
    lib = _lib()
    n = len(gate_floats)
    if len(up_floats) != n:
        raise ValueError("up must match gate length")
    G = (Dyad32 * n)()
    U = (Dyad32 * n)()
    D = (Dyad32 * n)()
    for i, v in enumerate(gate_floats):
        G[i] = lib.hqvm_dyad32_from_f32(float(v))
    for i, v in enumerate(up_floats):
        U[i] = lib.hqvm_dyad32_from_f32(float(v))
    host_counters_reset()
    rc = int(lib.hqvm_ffn_gate_dyad(G, U, D, n, int(fam) & 3, int(Nc) & 255))
    snap = host_counters_snapshot()
    dst = [float(lib.hqvm_dyad32_to_f32(D[i])) for i in range(n)]
    return rc, snap, dst


def host_v_reduce_dyad(
    weights: list[float],
    v_rows_f32: list[list[float]],
) -> tuple[int, dict[str, int], list[float]]:
    """H6 attn V-reduce face (stock float path when not native Q8)."""
    lib = _lib()
    n_k = len(weights)
    if n_k == 0 or len(v_rows_f32) != n_k:
        raise ValueError("weights and v_rows length must match and be >0")
    dv = len(v_rows_f32[0])
    if dv != 128:
        raise ValueError("V head dim must be 128")
    W = (Dyad32 * n_k)()
    for i, v in enumerate(weights):
        W[i] = lib.hqvm_dyad32_from_f32(float(v))
    flat = []
    for row in v_rows_f32:
        if len(row) != dv:
            raise ValueError("ragged V rows")
        flat.extend(float(x) for x in row)
    Vbuf = (ctypes.c_float * (n_k * dv))(*flat)
    Out = (Dyad32 * dv)()
    host_counters_reset()
    rc = int(lib.hqvm_v_reduce_dyad(
        Out, dv, W, n_k, Vbuf, dv * ctypes.sizeof(ctypes.c_float), 0))
    snap = host_counters_snapshot()
    out = [float(lib.hqvm_dyad32_to_f32(Out[i])) for i in range(dv)]
    return rc, snap, out


def host_matmul_dyad(
    q1_bytes: bytes,
    *,
    n_rows: int,
    n_cols: int,
    x_floats: list[float],
) -> tuple[int, dict[str, int]]:
    """Run hqvm_matmul_dyad on a dense Q1 row blob (row_stride = n_cols/128*20)."""
    lib = _lib()
    if len(x_floats) != n_cols:
        raise ValueError(f"x must have length {n_cols}")
    if (n_cols % 64) != 0 or (n_cols % 32) != 0:
        raise ValueError("n_cols must be a multiple of 64")
    stride = (n_cols // 128) * 20
    if len(q1_bytes) < n_rows * stride:
        raise ValueError("q1_bytes too small for shape")

    q1_buf = ctypes.create_string_buffer(q1_bytes, len(q1_bytes))
    W = lib.Q1Weight()
    W.q1_data = ctypes.cast(q1_buf, ctypes.c_void_p)
    W.n_rows = n_rows
    W.n_cols = n_cols
    W.row_stride_bytes = stride

    x_arr = (Dyad32 * n_cols)()
    y_arr = (Dyad32 * n_rows)()
    for i, v in enumerate(x_floats):
        x_arr[i] = lib.hqvm_dyad32_from_f32(float(v))

    lib.hqvm_gate_counters_reset()
    rc = int(lib.hqvm_matmul_dyad(ctypes.byref(W), x_arr, y_arr))
    snap = host_counters_snapshot()
    return rc, snap


def rt_rule_hash() -> int:
    return int(_lib().hqvm_rt_rule_hash())


def rt_enabled() -> bool:
    return bool(_lib().hqvm_rt_enabled())


def rt_cell_new(seed_mode: int = RT_SEED_REST, a: int = 0, b: int = 0) -> RtCell:
    cell = RtCell()
    _lib().hqvm_rt_cell_init(ctypes.byref(cell), int(seed_mode), a & 0xFF, b & 0xFF)
    return cell


def rt_ingest_word(cell: RtCell, word4: bytes) -> None:
    if len(word4) != 4:
        raise ValueError("word4 must be exactly 4 bytes")
    buf = (ctypes.c_uint8 * 4)(*word4)
    _lib().hqvm_rt_ingest_word(ctypes.byref(cell), buf)


def rt_ingest_bytes(cell: RtCell, data: bytes) -> None:
    if not data:
        return
    buf = (ctypes.c_uint8 * len(data))(*data)
    _lib().hqvm_rt_ingest_bytes(ctypes.byref(cell), buf, len(data))


def rt_resonance_key(cell: RtCell, profile_id: int = RT_PROFILE_CHIRALITY) -> int:
    return int(_lib().hqvm_rt_resonance_key_of(ctypes.byref(cell), profile_id & 0xFFFF))


def rt_cell_snapshot(cell: RtCell | None = None) -> dict:
    """Live record of a cell (request cell when None) for receipts/replay."""
    if cell is None:
        ptr = _lib().hqvm_rt_request_cell()
        if not ptr:
            raise RuntimeError("no live request cell (call rt_request_reset first)")
        c = ptr.contents
    else:
        c = cell

    def arr(a):
        return list(a)

    return {
        "step": int(c.step),
        "resonance_key": int(c.resonance_key),
        "omega_sig": int(c.omega_sig),
        "omega12": int(c.omega12),
        "chi_hist64": arr(c.chi_hist64),
        "shell_hist7": arr(c.shell_hist7),
        "family_hist4": arr(c.family_hist4),
        "parity_O12": int(c.parity_O12),
        "parity_E12": int(c.parity_E12),
        "parity_bit": int(c.parity_bit),
        "last_byte": int(c.last_byte),
        "word4": bytes(bytearray(c.word4)),
        "open_word": bytes(bytearray(c.open_word)),
        "word_len": int(c.word_len),
        "has_closed_word": int(c.has_closed_word),
        "chi_ring64": arr(c.chi_ring64),
        "family_ring64": arr(c.family_ring64),
        "ring_pos": int(c.ring_pos),
        "ring_valid_len": int(c.ring_valid_len),
    }


def rt_pool_create(capacity: int, profile_id: int = RT_PROFILE_CHIRALITY) -> RtPool:
    ptr = _lib().hqvm_rt_pool_create(int(capacity), profile_id & 0xFFFF)
    if not ptr:
        raise RuntimeError("hqvm_rt_pool_create failed")
    pool = ptr.contents
    # Keep the owner pointer alive so the C free() can be paired later.
    pool._owner = ptr  # type: ignore[attr-defined]
    return pool


def rt_pool_free(pool: RtPool) -> None:
    owner = getattr(pool, "_owner", None)
    if owner is not None:
        _lib().hqvm_rt_pool_free(owner)


def rt_pool_cell(pool: RtPool, cell_id: int) -> RtCell:
    if cell_id >= pool.capacity:
        raise IndexError("cell_id out of range")
    return pool.cells[cell_id]


def rt_pool_ingest_word(pool: RtPool, cell_id: int, word4: bytes) -> None:
    if len(word4) != 4:
        raise ValueError("word4 must be exactly 4 bytes")
    buf = (ctypes.c_uint8 * 4)(*word4)
    rc = int(_lib().hqvm_rt_pool_ingest_word(ctypes.byref(pool), int(cell_id), buf))
    if rc != 0:
        raise RuntimeError("hqvm_rt_pool_ingest_word failed")


def rt_request_reset(seed_mode: int = RT_SEED_REST) -> None:
    _lib().hqvm_rt_request_reset(int(seed_mode))


def rt_request_ingest_bytes(data: bytes) -> None:
    if not data:
        return
    buf = (ctypes.c_uint8 * len(data))(*data)
    _lib().hqvm_rt_request_ingest_bytes(buf, len(data))


def rt_request_cell() -> dict:
    return rt_cell_snapshot(None)


def rt_log_configure(path: str | None) -> int:
    return int(_lib().hqvm_rt_log_configure(path.encode() if path else None))


def rt_log_close() -> None:
    _lib().hqvm_rt_log_close()


def rt_log_stats() -> tuple[int, int]:
    return int(_lib().hqvm_rt_log_events()), int(_lib().hqvm_rt_log_requests())


def rt_snapshot_header(seed_mode: int = 0) -> dict:
    """Rule-hash snapshot header (Runtime 16.2) for receipt files."""
    buf = (ctypes.c_uint8 * 40)()
    _lib().hqvm_rt_snapshot_header_fill(buf, int(seed_mode))
    import struct

    magic, version, seed, reserved, n_ev, n_rq, rh = struct.unpack("<IIIIQQQ", bytes(buf))
    return {
        "magic": hex(magic),
        "version": version,
        "seed_mode": seed,
        "n_events": n_ev,
        "n_requests": n_rq,
        "rule_hash": rh,
    }


def rt_cell_checkpoint(cell: RtCell | None = None) -> bytes:
    """16-byte exact checkpoint of a cell (request cell when None)."""
    if cell is None:
        ptr = _lib().hqvm_rt_request_cell()
        if not ptr:
            raise RuntimeError("no live request cell")
        cell = ptr.contents
    assert cell is not None
    buf = (ctypes.c_uint8 * 16)()
    _lib().hqvm_rt_cell_checkpoint(ctypes.byref(cell), buf)
    return bytes(bytearray(buf))


class PolarSummary(ctypes.Structure):
    """Mirror of hqvm_rt_polar_summary (Runtime Specs 21.1)."""

    _fields_ = [
        ("chi6", ctypes.c_uint8),
        ("anchor64", ctypes.c_uint64),
        ("radius", ctypes.c_float),
    ]


def rt_chi_distance(a: RtCell, b: RtCell) -> int:
    return int(_lib().hqvm_rt_chi_distance(ctypes.byref(a), ctypes.byref(b)))


def rt_group_cells(
    cells: list[RtCell], max_batch: int
) -> list[int]:
    """Native decode-batch grouping (Runtime 20.2); returns dense group ids."""
    n = len(cells)
    if n == 0:
        return []
    arr = (ctypes.c_void_p * n)(*[ctypes.cast(ctypes.byref(c), ctypes.c_void_p) for c in cells])
    out = (ctypes.c_uint16 * n)()
    rc = _lib().hqvm_rt_group_cells(arr, n, max_batch, out)
    if rc < 0:
        raise RuntimeError("hqvm_rt_group_cells failed")
    return [int(out[i]) for i in range(n)]


def rt_polar_score(
    q_chi6: int, q_anchor: int,
    k_chi6: int, k_anchor: int,
    radius_q: float = 1.0, radius_k: float = 1.0,
) -> float:
    q = PolarSummary(q_chi6 & 0x3F, q_anchor & 0xFFFFFFFFFFFFFFFF, radius_q)
    k = PolarSummary(k_chi6 & 0x3F, k_anchor & 0xFFFFFFFFFFFFFFFF, radius_k)
    return float(_lib().hqvm_rt_polar_score(ctypes.byref(q), ctypes.byref(k)))


def rt_stock_ops_add(n: int) -> None:
    _lib().hqvm_rt_stock_ops_add(int(n))


def rt_counters() -> dict:
    return {
        "stock_ops_total": int(_lib().hqvm_rt_stock_ops_total()),
        "prefilter_calls": int(_lib().hqvm_rt_prefilter_calls()),
        "prefilter_skipped": int(_lib().hqvm_rt_prefilter_skipped()),
        "group_calls": int(_lib().hqvm_rt_group_calls()),
        "group_rows": int(_lib().hqvm_rt_group_rows()),
        "group_groups": int(_lib().hqvm_rt_group_groups()),
        "log_events": int(_lib().hqvm_rt_log_events()),
        "log_requests": int(_lib().hqvm_rt_log_requests()),
    }


def rt_counters_request_reset() -> None:
    _lib().hqvm_rt_counters_request_reset()


def rt_slcp_dict(slcp: RtSlcp) -> dict:
    return {
        "cell_id": int(slcp.cell_id),
        "step": int(slcp.step),
        "omega12": int(slcp.omega12),
        "state24": int(slcp.state24),
        "last_byte": int(slcp.last_byte),
        "family": int(slcp.family),
        "micro_ref": int(slcp.micro_ref),
        "q6": int(slcp.q6),
        "chi6": int(slcp.chi6),
        "shell": int(slcp.shell),
        "horizon_distance": int(slcp.horizon_distance),
        "ab_distance": int(slcp.ab_distance),
        "omega_sig": int(slcp.omega_sig),
        "parity_O12": int(slcp.parity_O12),
        "parity_E12": int(slcp.parity_E12),
        "parity_bit": int(slcp.parity_bit),
        "resonance_key": int(slcp.resonance_key),
        "current_resonance": int(slcp.current_resonance),
        "spectral64": [float(slcp.spectral64[i]) for i in range(64)],
    }


def rt_slcp_fill(cell: RtCell, cell_id: int = 0, pool: RtPool | None = None) -> dict:
    out = RtSlcp()
    pool_ptr = ctypes.byref(pool) if pool is not None else None
    _lib().hqvm_rt_slcp_fill(ctypes.byref(cell), int(cell_id), pool_ptr, ctypes.byref(out))
    return rt_slcp_dict(out)


def rt_bucket_population(pool: RtPool, key: int) -> int:
    return int(_lib().hqvm_rt_bucket_population(ctypes.byref(pool), int(key)))


def rt_bucket_cells(pool: RtPool, key: int, max_out: int = 256) -> list[int]:
    buf = (ctypes.c_uint32 * max_out)()
    n = int(_lib().hqvm_rt_bucket_cells(ctypes.byref(pool), int(key), buf, max_out))
    return [int(buf[i]) for i in range(max(0, n))]


def rt_medium_open(log_path: str | None, seed_mode: int = RT_SEED_REST, capacity: int = 64) -> None:
    path = log_path.encode() if log_path else None
    rc = int(_lib().hqvm_rt_medium_open(path, int(seed_mode), int(capacity)))
    if rc != 0:
        raise RuntimeError(f"hqvm_rt_medium_open failed rc={rc}")


def rt_medium_ingest(data: bytes, emit_slcp: bool = True) -> None:
    if not data:
        return
    buf = (ctypes.c_uint8 * len(data))(*data)
    rc = int(_lib().hqvm_rt_medium_ingest(buf, len(data), 1 if emit_slcp else 0))
    if rc != 0:
        raise RuntimeError("hqvm_rt_medium_ingest failed")


def rt_medium_close() -> None:
    _lib().hqvm_rt_medium_close()


def rt_medium_last_slcp() -> dict | None:
    ptr = _lib().hqvm_rt_medium_last_slcp()
    if not ptr:
        return None
    return rt_slcp_dict(ptr.contents)


def rt_medium_cell_snapshot() -> dict:
    ptr = _lib().hqvm_rt_medium_cell()
    if not ptr:
        raise RuntimeError("medium session not open")
    return rt_cell_snapshot(ptr.contents)
