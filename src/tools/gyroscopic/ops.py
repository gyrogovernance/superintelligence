"""ctypes bindings for the Gyroscopic kernel.

Builds ``kernel.c`` for tests. The llama.cpp hot path uses ``gravity_scale`` via
TLS; this module also exposes step law, K4, and chirality helpers.
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

    lib.gyroscopic_gravity_g1.restype = ctypes.c_float
    lib.gyroscopic_gravity_g1.argtypes = []

    lib.gyroscopic_gravity_scale.restype = ctypes.c_float
    lib.gyroscopic_gravity_scale.argtypes = [ctypes.c_int, ctypes.c_int, u8, u8]

    fptr = ctypes.POINTER(ctypes.c_float)
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
