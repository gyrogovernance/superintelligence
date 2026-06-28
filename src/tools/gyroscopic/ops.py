"""ctypes bindings for the Gyroscopic kernel.

Builds ``kernel.c`` for tests. The llama.cpp hot path uses ``gravity_scale`` via
TLS; this module also exposes step rule, K4, and chirality helpers.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_BUILD_DIR = _PKG_DIR / "_build"

OMEGA_SIZE = 4096
HORIZON_SIZE = 64

K4_ID = 0
K4_W2 = 1
K4_W2P = 2
K4_F = 3

PATH_ISOTROPIC = 0
PATH_BULK_CS = 1
PATH_BULK_UNA = 2
PATH_BULK_ONA = 3
PATH_BULK_BU = 4


def _lib_name() -> str:
    if sys.platform == "win32":
        return "gyroscopic_native.dll"
    if sys.platform == "darwin":
        return "libgyroscopic_native.dylib"
    return "libgyroscopic_native.so"


def _lib_path() -> Path:
    return _BUILD_DIR / _lib_name()


def _needs_rebuild(lib: Path) -> bool:
    if not lib.is_file():
        return True
    src = _PKG_DIR / "kernel.c"
    hdrs = [_PKG_DIR / "kernel.h", _PKG_DIR / "constants.h"]
    try:
        lib_m = lib.stat().st_mtime
        return any(p.stat().st_mtime > lib_m for p in [src, *hdrs] if p.is_file())
    except OSError:
        return True


def _detect_c_compiler() -> list[str] | None:
    if sys.platform == "win32" and shutil.which("cl"):
        return ["cl", "/nologo", "/O2", "/LD"]
    for cc in ("cc", "gcc", "clang"):
        if shutil.which(cc):
            return [cc, "-O2", "-fPIC", "-shared"]
    return None


def build_native(force: bool = False) -> Path:
    """Compile ``kernel.c`` into the standalone ctypes library."""
    lib = _lib_path()
    if not force and not _needs_rebuild(lib):
        return lib
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    src = str(_PKG_DIR / "kernel.c")

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
        argv = cc_argv + [src, f"/Fe:{lib}", f"/Fo:{_BUILD_DIR}\\"]
    else:
        argv = cc_argv + [src, "-o", str(lib), "-lm"]

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


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(str(build_native()))
        _bind(_LIB)
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

    hist64 = ctypes.c_uint32 * 64
    lib.gyroscopic_chi_hist_d_eff.restype = ctypes.c_int
    lib.gyroscopic_chi_hist_d_eff.argtypes = [hist64, u8, fptr, fptr]

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

    tile = 64
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


def step_omega12(state24: int, byte: int) -> int:
    return int(_lib().gyroscopic_step_omega12(state24 & 0xFFFFFF, byte & 0xFF))


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


def chi_hist_d_eff(hist: list[int], chi_q: int) -> tuple[int, float, float]:
    """Percolation-aware Hamming aperture from 64-bin occupation histogram."""
    if len(hist) != 64:
        raise ValueError("hist must be length 64")
    hbuf = (ctypes.c_uint32 * 64)(*hist)
    m2 = ctypes.c_float()
    eta = ctypes.c_float()
    d = int(_lib().gyroscopic_chi_hist_d_eff(hbuf, chi_q & 0x3F, ctypes.byref(m2), ctypes.byref(eta)))
    return d, float(m2.value), float(eta.value)


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
