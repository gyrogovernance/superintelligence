"""kernel/bindings.py — ctypes bridge to gyrocrypt_native (native.c)."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import threading
from pathlib import Path

_KERNEL_DIR = Path(__file__).resolve().parent
_BUILD_DIR = _KERNEL_DIR / "_build"
_SOURCES = ("native.c",)
_U64 = ctypes.c_uint64
_U32P = ctypes.POINTER(ctypes.c_uint32)


def _lib_name() -> str:
    if sys.platform == "win32":
        return "gyrocrypt_native.dll"
    if sys.platform == "darwin":
        return "libgyrocrypt_native.dylib"
    return "libgyrocrypt_native.so"


def _lib_path() -> Path:
    return _BUILD_DIR / _lib_name()


def _needs_rebuild(lib: Path) -> bool:
    if not lib.is_file():
        return True
    hdr = _KERNEL_DIR / "native.h"
    try:
        lib_m = lib.stat().st_mtime
        return any(
            p.stat().st_mtime > lib_m
            for p in [*(_KERNEL_DIR / s for s in _SOURCES), hdr]
            if p.is_file()
        )
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
    lib = _lib_path()
    if not force and not _needs_rebuild(lib):
        return lib
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    sources = [str(_KERNEL_DIR / s) for s in _SOURCES]

    cc_argv = _detect_c_compiler()
    if cc_argv is None and sys.platform == "win32":
        ps1 = _KERNEL_DIR.parent / "helpers" / "build_kernel.ps1"
        if ps1.is_file():
            cp = subprocess.run(
                ["powershell", "-NoProfile", "-File", str(ps1)],
                capture_output=True,
                text=True,
                check=False,
            )
            if cp.returncode == 0 and lib.is_file():
                return lib
            if cp.returncode != 0:
                raise RuntimeError(
                    f"gyrocrypt native build failed.\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
                )

    if cc_argv is None:
        raise RuntimeError("gyrocrypt: no C compiler (cl/cc/gcc/clang) found.")

    if cc_argv[0] == "cl":
        argv = cc_argv + sources + [f"/Fe:{lib}", f"/Fo:{_BUILD_DIR}\\"]
    else:
        argv = cc_argv + sources + ["-o", str(lib), "-lm"]

    cp = subprocess.run(
        argv, capture_output=True, text=True, cwd=str(_BUILD_DIR), check=False
    )
    if cp.returncode != 0:
        raise RuntimeError(
            f"gyrocrypt native build failed (argv={argv!r}).\n"
            f"STDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
        )
    if not lib.is_file():
        raise FileNotFoundError(f"gyrocrypt: build finished but {lib} not found.")
    return lib


_LIB_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        _LIB = ctypes.CDLL(str(build_native()))
        _bind(_LIB)
        return _LIB


def _bind(lib: ctypes.CDLL) -> None:
    lib.gyroscopic_mul_mod_ladder.restype = _U64
    lib.gyroscopic_mul_mod_ladder.argtypes = [_U64, _U64, _U64]
    lib.gyroscopic_exp_mod_ladder.restype = _U64
    lib.gyroscopic_exp_mod_ladder.argtypes = [_U64, _U64, _U64]
    lib.gyroscopic_exp_mod_ladder_limbs.restype = ctypes.c_int
    lib.gyroscopic_exp_mod_ladder_limbs.argtypes = [
        _U32P, ctypes.c_int, _U32P, ctypes.c_int, _U32P, ctypes.c_int, _U32P, ctypes.c_int,
    ]
    lib.gyroscopic_sparse_cqft_peaks.restype = ctypes.c_int
    lib.gyroscopic_sparse_cqft_peaks.argtypes = [
        _U32P, ctypes.c_int, ctypes.c_uint32, ctypes.c_int, _U32P,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ]
    lib.gyroscopic_shor_period_u64.restype = ctypes.c_uint32
    lib.gyroscopic_shor_period_u64.argtypes = [_U64, _U64, _U64]
    lib.gyroscopic_shor_last_path_tag.restype = ctypes.c_char_p
    lib.gyroscopic_shor_last_path_tag.argtypes = []
    lib.gyroscopic_shor_dp_mag2_y1_u64.restype = ctypes.c_double
    lib.gyroscopic_shor_dp_mag2_y1_u64.argtypes = [_U64, _U64, _U64, _U64]
    lib.gyroscopic_shor_period_chirality_u64.restype = ctypes.c_uint32
    lib.gyroscopic_shor_period_chirality_u64.argtypes = [_U64, _U64, ctypes.c_uint32]

    u64p = ctypes.POINTER(_U64)
    lib.gyroscopic_horizon_pack_keys_u64.restype = ctypes.c_int
    lib.gyroscopic_horizon_pack_keys_u64.argtypes = [_U64, u64p, ctypes.c_int]
    lib.gyroscopic_horizon_n_cells_u64.restype = ctypes.c_int
    lib.gyroscopic_horizon_n_cells_u64.argtypes = [_U64]
    lib.gyroscopic_horizon_key_u64.restype = _U64
    lib.gyroscopic_horizon_key_u64.argtypes = [_U64, _U64]
    lib.gyroscopic_horizon_tensor_step_drift_u64.restype = ctypes.c_int
    lib.gyroscopic_horizon_tensor_step_drift_u64.argtypes = [
        _U64, _U64, _U64, _U64, u64p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ]
    lib.gyroscopic_horizon_tensor_mag2_y1_u64.restype = ctypes.c_double
    lib.gyroscopic_horizon_tensor_mag2_y1_u64.argtypes = [
        _U64, _U64, _U64, _U64, u64p, ctypes.c_int,
    ]
    lib.gyroscopic_dlp_2d_tensor_mag2_u64.restype = ctypes.c_double
    lib.gyroscopic_dlp_2d_tensor_mag2_u64.argtypes = [
        _U64, _U64, _U64, _U64, _U64, _U64, u64p, ctypes.c_int,
    ]


def _u64(*vals: int) -> tuple[ctypes.c_uint64, ...]:
    return tuple(_U64(int(v)) for v in vals)


def _keys_buf(keys: list[int]) -> ctypes.Array:
    return (ctypes.c_uint64 * len(keys))(*(int(k) for k in keys))


def mul_mod_ladder(y: int, multiplier: int, n: int) -> int:
    return int(_lib().gyroscopic_mul_mod_ladder(*_u64(y, multiplier, n)))


def exp_mod_ladder(a: int, x: int, n: int) -> int:
    return int(_lib().gyroscopic_exp_mod_ladder(*_u64(a, x, n)))


def _int_to_limbs(value: int, limb_count: int) -> list[int]:
    x, limbs, i = int(value), [0] * limb_count, 0
    while x and i < limb_count:
        limbs[i] = x & 0xFFFFFFFF
        x >>= 32
        i += 1
    return limbs


def _limbs_to_int(limbs: list[int]) -> int:
    out = 0
    for i, limb in enumerate(limbs):
        out |= (int(limb) & 0xFFFFFFFF) << (32 * i)
    return out


def exp_mod_ladder_limbs(a: int, x: int, n: int) -> int:
    nn = int(n)
    xx = int(x)
    if nn <= 1:
        return 0
    aa = int(a) % nn
    if nn < (1 << 63):
        return exp_mod_ladder(aa, xx, nn)
    w = max(2, (nn.bit_length() + 31) // 32)
    a_limbs, x_limbs, n_limbs = _int_to_limbs(aa, w), _int_to_limbs(xx, w), _int_to_limbs(nn, w)
    out_buf = (ctypes.c_uint32 * w)()
    ok = int(
        _lib().gyroscopic_exp_mod_ladder_limbs(
            (ctypes.c_uint32 * w)(*a_limbs), w,
            (ctypes.c_uint32 * w)(*x_limbs), w,
            (ctypes.c_uint32 * w)(*n_limbs), w,
            out_buf, w,
        )
    )
    if ok != 0:
        return _limbs_to_int(list(out_buf))
    raise RuntimeError(f"gyroscopic_exp_mod_ladder_limbs failed for {aa}^{xx} mod {nn}")


def sparse_cqft_peaks(
    support: list[int], Q: int, *, k_top: int = 32
) -> list[tuple[int, float]]:
    if not support or int(Q) <= 1:
        return []
    n, cap = len(support), max(1, int(k_top))
    sup_arr = (ctypes.c_uint32 * n)(*(int(x) for x in support))
    out_k = (ctypes.c_uint32 * cap)()
    out_m = (ctypes.c_double * cap)()
    got = int(
        _lib().gyroscopic_sparse_cqft_peaks(
            sup_arr, n, ctypes.c_uint32(int(Q)), cap, out_k, out_m, cap
        )
    )
    return [(int(out_k[i]), float(out_m[i])) for i in range(got)]


def shor_period_u64(base: int, n: int, Q: int) -> int:
    return int(_lib().gyroscopic_shor_period_u64(*_u64(base, n, Q)))


def shor_period_chirality_u64(base: int, n: int, max_samples: int = 0) -> int:
    return int(
        _lib().gyroscopic_shor_period_chirality_u64(
            _U64(int(base)), _U64(int(n)), ctypes.c_uint32(int(max_samples))
        )
    )


def shor_last_path_tag() -> str:
    raw = _lib().gyroscopic_shor_last_path_tag()
    return raw.decode("ascii") if raw else "NONE"


def shor_dp_mag2_y1_u64(base: int, n: int, Q: int, k: int) -> float:
    return float(_lib().gyroscopic_shor_dp_mag2_y1_u64(*_u64(base, n, Q, k)))


def horizon_pack_keys_u64(n: int) -> tuple[list[int], int]:
    nn = int(n)
    if nn <= 1:
        raise ValueError("n must be > 1")
    keys_arr = (ctypes.c_uint64 * nn)()
    got = int(_lib().gyroscopic_horizon_pack_keys_u64(_U64(nn), keys_arr, nn))
    if got <= 0:
        raise RuntimeError("gyroscopic_horizon_pack_keys_u64 failed")
    return [int(keys_arr[y]) for y in range(nn)], got


def horizon_n_cells_u64(n: int) -> int:
    got = int(_lib().gyroscopic_horizon_n_cells_u64(_U64(int(n))))
    if got <= 0:
        raise RuntimeError("gyroscopic_horizon_n_cells_u64 failed")
    return got


def horizon_key_u64(n: int, y: int) -> int:
    return int(_lib().gyroscopic_horizon_key_u64(_U64(int(n)), _U64(int(y))))


def horizon_tensor_step_drift_u64(
    base: int, n: int, Q: int, k: int, keys_packed: list[int], n_cells: int
) -> list[tuple[int, float, float, float]]:
    cap, ex, tn = 24, (ctypes.c_double * 24)(), (ctypes.c_double * 24)()
    got = int(
        _lib().gyroscopic_horizon_tensor_step_drift_u64(
            *_u64(base, n, Q, k), _keys_buf(keys_packed), int(n_cells), ex, tn, cap
        )
    )
    if got == -2:
        raise RuntimeError(f"horizon_tensor_step_drift requires n_cells in 2..4, got {n_cells}")
    if got == -3:
        raise RuntimeError("horizon keys not injective on tensor grid")
    if got <= 0:
        raise RuntimeError(f"gyroscopic_horizon_tensor_step_drift_u64 failed ({got})")
    rows: list[tuple[int, float, float, float]] = []
    for j in range(got):
        e = float(ex[j])
        rows.append((j, e, float(tn[j]), float(tn[j]) / e if e > 1e-30 else 0.0))
    return rows


def horizon_tensor_mag2_y1_u64(
    base: int, n: int, Q: int, k: int, keys_packed: list[int], n_cells: int
) -> float:
    return float(
        _lib().gyroscopic_horizon_tensor_mag2_y1_u64(
            *_u64(base, n, Q, k), _keys_buf(keys_packed), int(n_cells)
        )
    )


def dlp_2d_tensor_mag2_u64(
    base_g: int,
    base_h: int,
    n: int,
    Q: int,
    k1: int,
    k2: int,
    keys_packed: list[int],
    n_cells: int,
) -> float:
    return float(
        _lib().gyroscopic_dlp_2d_tensor_mag2_u64(
            *_u64(base_g, base_h, n, Q, k1, k2),
            _keys_buf(keys_packed),
            int(n_cells),
        )
    )


__all__ = [
    "build_native",
    "exp_mod_ladder",
    "exp_mod_ladder_limbs",
    "mul_mod_ladder",
    "sparse_cqft_peaks",
    "shor_period_u64",
    "shor_period_chirality_u64",
    "shor_last_path_tag",
    "shor_dp_mag2_y1_u64",
    "horizon_pack_keys_u64",
    "horizon_n_cells_u64",
    "horizon_key_u64",
    "horizon_tensor_step_drift_u64",
    "horizon_tensor_mag2_y1_u64",
    "dlp_2d_tensor_mag2_u64",
]
