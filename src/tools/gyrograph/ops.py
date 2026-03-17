from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

from src.api import q_word6_for_items

_THIS_FILE = Path(__file__).resolve()
_GYROGRAPH_DIR = _THIS_FILE.parent
_REPO_ROOT = _THIS_FILE.parents[3]
_CSRC_FILE = _GYROGRAPH_DIR / "gyrograph.c"
_OPENCL_CSRC_FILE = _GYROGRAPH_DIR / "gyrograph_opencl.c"
_BUILD_DIR = _GYROGRAPH_DIR / "_build"

_CPU_LIB: ctypes.CDLL | None = None
_CPU_LIB_ATTEMPTED = False

_CL_LIB: ctypes.CDLL | None = None
_CL_LIB_ATTEMPTED = False
_CL_INITIALIZED = False


# ----------------------------------------------------------------------
# Build helpers
# ----------------------------------------------------------------------


def _shared_lib_suffix() -> str:
    if sys.platform.startswith("win"):
        return ".dll"
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


def _build_mode() -> str:
    return os.environ.get("GYRO_BUILD_MODE", "portable").strip().lower()


def _x86_64_host() -> bool:
    return platform.machine().lower() in {"x86_64", "amd64", "x64"}


def _simd_flags(kind: str) -> list[str]:
    if _build_mode() != "native" or not _x86_64_host():
        return []
    if kind == "msvc":
        return ["/arch:AVX2"]
    return ["-mavx2", "-mfma"]


def _hash_source(path: Path, *, compiler: str, kind: str, flags: list[str], opencl: bool) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    h.update(platform.platform().encode("utf-8"))
    h.update(platform.python_version().encode("utf-8"))
    h.update(compiler.encode("utf-8"))
    h.update(kind.encode("utf-8"))
    h.update(_build_mode().encode("utf-8"))
    h.update(("1" if opencl else "0").encode("utf-8"))
    for flag in flags:
        h.update(flag.encode("utf-8"))
    return h.hexdigest()[:16]


def _find_clang_windows() -> str | None:
    path = shutil.which("clang")
    if path:
        return path
    for candidate in (
        os.environ.get("LLVM_HOME"),
        os.environ.get("LLVM_PATH"),
        os.environ.get("LLVM_ROOT"),
    ):
        if candidate:
            p = Path(candidate).resolve() / "bin" / "clang.exe"
            if p.exists():
                return str(p)
    for prefix in (
        Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "LLVM",
        Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "LLVM",
    ):
        p = prefix / "bin" / "clang.exe"
        if p.exists():
            return str(p)
    return None


def _find_gcc_windows() -> str | None:
    path = shutil.which("gcc")
    if path:
        return path
    base = Path("C:/msys64")
    for sub in ("ucrt64", "mingw64"):
        p = base / sub / "bin" / "gcc.exe"
        if p.exists():
            return str(p)
    return None


def _find_compiler() -> tuple[str, str]:
    if os.name == "nt":
        gcc = _find_gcc_windows()
        if gcc:
            return gcc, "cc"

        for name in ("cl", "clang-cl", "clang"):
            path = shutil.which(name)
            if path:
                if name in ("cl", "clang-cl"):
                    return path, "msvc"
                return path, "cc"

        clang = _find_clang_windows()
        if clang:
            return clang, "cc"
    else:
        for name in ("cc", "clang", "gcc"):
            path = shutil.which(name)
            if path:
                return path, "cc"

    raise RuntimeError(
        "No suitable C compiler was found for GyroGraph.\n"
        "Windows: install Visual Studio Build Tools or GCC/Clang.\n"
        "Linux/macOS: install clang or gcc."
    )


def _dll_search_paths() -> list[Path]:
    if not os.name == "nt":
        return []
    paths: list[Path] = []
    compiler, _ = _find_compiler()
    comp_dir = Path(compiler).resolve().parent
    if comp_dir.exists():
        paths.append(comp_dir)
    base = Path("C:/msys64")
    for sub in ("ucrt64", "mingw64"):
        p = base / sub / "bin"
        if p.exists():
            paths.append(p)
    for env_key in ("LLVM_PATH", "LLVM_ROOT", "MINGW64_BIN", "MSYS2_PATH"):
        value = os.environ.get(env_key)
        if value:
            for candidate in (Path(value), Path(value) / "bin"):
                if candidate.exists():
                    paths.append(candidate)
    return paths


def _find_opencl_include_flags() -> list[str]:
    inc = os.environ.get("OPENCL_INCLUDE", "").strip()
    if inc:
        return [f"-I{inc}"]
    for candidate in (
        os.environ.get("OPENCL_SDK_PATH"),
        os.environ.get("OPENCL_ROOT"),
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/include",
        "C:/Program Files/AMD/OCL-SDK/include",
    ):
        if candidate and Path(candidate).exists():
            return [f"-I{candidate}"]
    return []


def _find_opencl_lib_flags() -> list[str]:
    lib = os.environ.get("OPENCL_LIB", "").strip()
    if lib:
        return [f"-L{lib}", "-lOpenCL"]

    for candidate in (
        os.environ.get("OPENCL_SDK_PATH"),
        os.environ.get("OPENCL_ROOT"),
    ):
        if candidate:
            p = Path(candidate) / "lib"
            if p.exists():
                for sub in ("x86_64", "x64", ""):
                    sp = p / sub if sub else p
                    if sp.exists():
                        return [f"-L{sp}", "-lOpenCL"]

    return ["-lOpenCL"]


def _build_shared_library(src: Path, *, opencl: bool = False) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    compiler, kind = _find_compiler()
    simd_flags = _simd_flags(kind)
    openmp_flags: list[str] = []
    if os.environ.get("GYROGRAPH_NO_OPENMP", "").strip() != "1":
        if kind == "cc":
            openmp_flags = ["-fopenmp"]
        elif kind == "msvc":
            openmp_flags = ["/openmp"]

    if kind == "msvc":
        compile_flags = ["/nologo", "/O2", *simd_flags, *openmp_flags, "/LD", "/TC"]
        hash_flags = compile_flags
        if opencl:
            hash_flags = [*hash_flags, "OpenCL.lib"]

        out_name = f"{src.stem}_{_hash_source(
            src,
            compiler=compiler,
            kind=kind,
            flags=hash_flags,
            opencl=opencl,
        )}{_shared_lib_suffix()}"
        out_path = _BUILD_DIR / out_name
        if out_path.exists():
            return out_path

        cmd = [
            compiler,
            "/nologo",
            "/O2",
            *simd_flags,
            *openmp_flags,
            "/LD",
            "/TC",
            str(src),
            f"/Fe:{out_path}",
        ]
        if opencl:
            cmd.append("OpenCL.lib")
        result = subprocess.run(
            cmd, cwd=str(_BUILD_DIR), capture_output=True, text=True, check=False
        )
    else:
        extra_flags: list[str] = []
        extra_libs: list[str] = []
        if opencl:
            extra_flags.extend(_find_opencl_include_flags())
            extra_libs.extend(_find_opencl_lib_flags())

        flags = [
            "-O3",
            "-std=c11",
            "-shared",
            *extra_flags,
            *simd_flags,
            *openmp_flags,
        ]
        hash_flags = [*flags, *extra_libs]
        if sys.platform == "darwin":
            flags[2] = "-dynamiclib"
        elif not sys.platform.startswith("win"):
            flags.append("-fPIC")

        out_name = f"{src.stem}_{_hash_source(
            src,
            compiler=compiler,
            kind=kind,
            flags=hash_flags,
            opencl=opencl,
        )}{_shared_lib_suffix()}"
        out_path = _BUILD_DIR / out_name
        if out_path.exists():
            return out_path

        run_cwd = str(_BUILD_DIR)
        run_env = None
        cmd = [compiler, *flags, "-o", str(out_path), str(src), *extra_libs]

        if os.name == "nt":
            comp_dir = str(Path(compiler).resolve().parent)
            run_cwd = str(_REPO_ROOT)
            run_env = {
                **os.environ,
                "PATH": comp_dir + os.pathsep + os.environ.get("PATH", ""),
            }
            rel_src = src.relative_to(_REPO_ROOT)
            rel_out = out_path.relative_to(_REPO_ROOT)
            cmd = [compiler, *flags, "-o", str(rel_out), str(rel_src), *extra_libs]

        result = subprocess.run(
            cmd,
            cwd=run_cwd,
            env=run_env,
            capture_output=True,
            text=True,
            check=False,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to compile {src.name}.\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"stdout:\n{result.stdout}\n\n"
            f"stderr:\n{result.stderr}"
        )

    if not out_path.exists():
        raise RuntimeError(f"Compiler reported success but output library was not created: {out_path}")

    return out_path


# ----------------------------------------------------------------------
# ctypes setup
# ----------------------------------------------------------------------


def _setup_cpu_argtypes(lib: ctypes.CDLL) -> None:
    if hasattr(lib, "gyrograph_init"):
        lib.gyrograph_init.argtypes = []
        lib.gyrograph_init.restype = None

    if hasattr(lib, "gyrograph_compute_m2_empirical"):
        lib.gyrograph_compute_m2_empirical.argtypes = [
            ctypes.c_void_p,  # chi_hist64 (64 uint16)
            ctypes.c_uint64,
        ]
        lib.gyrograph_compute_m2_empirical.restype = ctypes.c_double

    if hasattr(lib, "gyrograph_compute_m2_equilibrium"):
        lib.gyrograph_compute_m2_equilibrium.argtypes = [
            ctypes.c_void_p,  # shell_hist7 (7 uint16)
            ctypes.c_uint64,
        ]
        lib.gyrograph_compute_m2_equilibrium.restype = ctypes.c_double

    if hasattr(lib, "gyrograph_ingest_word4_batch_indexed"):
        lib.gyrograph_ingest_word4_batch_indexed.argtypes = [
            ctypes.c_void_p,  # cell_ids
            ctypes.c_void_p,  # omega12_io
            ctypes.c_void_p,  # step_io
            ctypes.c_void_p,  # last_byte_io
            ctypes.c_void_p,  # has_closed_word_io
            ctypes.c_void_p,  # word4_io
            ctypes.c_void_p,  # chi_ring64_io
            ctypes.c_void_p,  # chi_ring_pos_io
            ctypes.c_void_p,  # chi_valid_len_io
            ctypes.c_void_p,  # chi_hist64_io
            ctypes.c_void_p,  # shell_hist7_io
            ctypes.c_void_p,  # family_ring64_io
            ctypes.c_void_p,  # family_hist4_io
            ctypes.c_void_p,  # omega_sig_io
            ctypes.c_void_p,  # parity_O12_io
            ctypes.c_void_p,  # parity_E12_io
            ctypes.c_void_p,  # parity_bit_io
            ctypes.c_void_p,  # words4_in
            ctypes.c_void_p,  # resonance_key_io
            ctypes.c_ubyte,  # profile
            ctypes.c_int64,  # n
        ]
        lib.gyrograph_ingest_word4_batch_indexed.restype = None

    if hasattr(lib, "gyrograph_trace_word4_batch_indexed"):
        lib.gyrograph_trace_word4_batch_indexed.argtypes = [
            ctypes.c_void_p,  # cell_ids
            ctypes.c_void_p,  # omega12_in
            ctypes.c_void_p,  # words4_in
            ctypes.c_int64,  # n
            ctypes.c_void_p,  # omega_trace4_out
            ctypes.c_void_p,  # chi_trace4_out
        ]
        lib.gyrograph_trace_word4_batch_indexed.restype = None

    if hasattr(lib, "gyrograph_apply_trace_word4_batch_indexed"):
        lib.gyrograph_apply_trace_word4_batch_indexed.argtypes = [
            ctypes.c_void_p,  # cell_ids
            ctypes.c_void_p,  # omega12_io
            ctypes.c_void_p,  # step_io
            ctypes.c_void_p,  # last_byte_io
            ctypes.c_void_p,  # has_closed_word_io
            ctypes.c_void_p,  # word4_io
            ctypes.c_void_p,  # chi_ring64_io
            ctypes.c_void_p,  # chi_ring_pos_io
            ctypes.c_void_p,  # chi_valid_len_io
            ctypes.c_void_p,  # chi_hist64_io
            ctypes.c_void_p,  # shell_hist7_io
            ctypes.c_void_p,  # family_ring64_io
            ctypes.c_void_p,  # family_hist4_io
            ctypes.c_void_p,  # omega_sig_io
            ctypes.c_void_p,  # parity_O12_io
            ctypes.c_void_p,  # parity_E12_io
            ctypes.c_void_p,  # parity_bit_io
            ctypes.c_void_p,  # words4_in
            ctypes.c_void_p,  # omega_trace4_in
            ctypes.c_void_p,  # chi_trace4_in
            ctypes.c_void_p,  # resonance_key_io
            ctypes.c_ubyte,  # profile
            ctypes.c_int64,  # n
        ]
        lib.gyrograph_apply_trace_word4_batch_indexed.restype = None


def _setup_cl_argtypes(lib: ctypes.CDLL) -> None:
    if hasattr(lib, "gyrograph_cl_available"):
        lib.gyrograph_cl_available.argtypes = []
        lib.gyrograph_cl_available.restype = ctypes.c_int

    if hasattr(lib, "gyrograph_cl_init"):
        lib.gyrograph_cl_init.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.gyrograph_cl_init.restype = ctypes.c_int

    if hasattr(lib, "gyrograph_cl_shutdown"):
        lib.gyrograph_cl_shutdown.argtypes = []
        lib.gyrograph_cl_shutdown.restype = None

    if hasattr(lib, "gyrograph_cl_trace_word4_batch"):
        lib.gyrograph_cl_trace_word4_batch.argtypes = [
            ctypes.c_void_p,  # cell_ids
            ctypes.c_void_p,  # omega12_in
            ctypes.c_void_p,  # words4_in
            ctypes.c_int64,  # n
            ctypes.c_void_p,  # omega_trace4_out
            ctypes.c_void_p,  # chi_trace4_out
        ]
        lib.gyrograph_cl_trace_word4_batch.restype = ctypes.c_int


def _get_cpu_lib() -> ctypes.CDLL | None:
    global _CPU_LIB, _CPU_LIB_ATTEMPTED
    if _CPU_LIB is not None:
        return _CPU_LIB
    if _CPU_LIB_ATTEMPTED:
        return None
    _CPU_LIB_ATTEMPTED = True
    try:
        path = _build_shared_library(_CSRC_FILE, opencl=False)
        for dll_dir in _dll_search_paths():
            try:
                os.add_dll_directory(str(dll_dir))
            except OSError:
                pass
        _CPU_LIB = ctypes.CDLL(str(path))
        _setup_cpu_argtypes(_CPU_LIB)
        if hasattr(_CPU_LIB, "gyrograph_init"):
            _CPU_LIB.gyrograph_init()
        return _CPU_LIB
    except Exception as e:
        warnings.warn(
            f"GyroGraph CPU native library could not be loaded. Using Python fallback. {e}",
            UserWarning,
            stacklevel=2,
        )
        _CPU_LIB = None
        return None


def _get_cl_lib() -> ctypes.CDLL | None:
    global _CL_LIB, _CL_LIB_ATTEMPTED
    if _CL_LIB is not None:
        return _CL_LIB
    if _CL_LIB_ATTEMPTED:
        return None
    _CL_LIB_ATTEMPTED = True
    try:
        path = _build_shared_library(_OPENCL_CSRC_FILE, opencl=True)
        for dll_dir in _dll_search_paths():
            try:
                os.add_dll_directory(str(dll_dir))
            except OSError:
                pass
        _CL_LIB = ctypes.CDLL(str(path))
        _setup_cl_argtypes(_CL_LIB)
        return _CL_LIB
    except Exception as e:
        warnings.warn(
            f"GyroGraph OpenCL library could not be loaded. OpenCL path disabled. {e}",
            UserWarning,
            stacklevel=2,
        )
        _CL_LIB = None
        return None


# ----------------------------------------------------------------------
# Availability / initialization
# ----------------------------------------------------------------------


def native_available() -> bool:
    return _get_cpu_lib() is not None


def opencl_available() -> bool:
    lib = _get_cl_lib()
    if lib is None or not hasattr(lib, "gyrograph_cl_available"):
        return False
    try:
        return bool(lib.gyrograph_cl_available() == 1)
    except Exception:
        return False


def initialize_opencl(platform_index: int = 0, device_index: int = 0) -> None:
    global _CL_INITIALIZED
    lib = _get_cl_lib()
    if lib is None or not hasattr(lib, "gyrograph_cl_init"):
        raise RuntimeError("GyroGraph OpenCL backend not available")
    ok = lib.gyrograph_cl_init(int(platform_index), int(device_index))
    if ok != 1:
        _CL_INITIALIZED = False
        raise RuntimeError(
            f"gyrograph_cl_init({platform_index}, {device_index}) failed"
        )
    _CL_INITIALIZED = True


def shutdown_opencl() -> None:
    global _CL_INITIALIZED
    lib = _get_cl_lib()
    _CL_INITIALIZED = False
    if lib is not None and hasattr(lib, "gyrograph_cl_shutdown"):
        lib.gyrograph_cl_shutdown()


def compute_m2_empirical_from_chi_hist(
    chi_hist64: np.ndarray,
    total: int,
) -> float:
    """
    Empirical effective support on Ω from chirality occupancy.
    One chirality class occupied -> M2 = 64.
    All 64 equally occupied -> M2 = 4096.
    """
    lib = _get_cpu_lib()
    if lib is not None and hasattr(lib, "gyrograph_compute_m2_empirical"):
        arr = _as_read_c_array(
            chi_hist64, dtype=np.uint16, ndim=1, name="chi_hist64"
        )
        if arr.shape != (64,):
            raise ValueError(f"chi_hist64 must have shape (64,), got {arr.shape}")
        return float(
            lib.gyrograph_compute_m2_empirical(
                _void_p(arr),
                ctypes.c_uint64(total),
            )
        )

    total_f = float(total)
    if total_f <= 0.0:
        return 64.0

    arr = np.asarray(chi_hist64, dtype=np.float64)
    sumsq = float(np.dot(arr, arr))
    if sumsq <= 0.0:
        return 64.0

    return 64.0 * (total_f * total_f) / sumsq


def compute_m2_equilibrium_from_shell_hist(
    shell_hist7: np.ndarray,
    total: int,
) -> float:
    """
    Equation-of-state support from shell mean:
        M2_eq = 4096 / (1 + eta^2)^6
    """
    lib = _get_cpu_lib()
    if lib is not None and hasattr(lib, "gyrograph_compute_m2_equilibrium"):
        arr = _as_read_c_array(
            shell_hist7, dtype=np.uint16, ndim=1, name="shell_hist7"
        )
        if arr.shape != (7,):
            raise ValueError(f"shell_hist7 must have shape (7,), got {arr.shape}")
        return float(
            lib.gyrograph_compute_m2_equilibrium(
                _void_p(arr),
                ctypes.c_uint64(total),
            )
        )

    total_f = float(total)
    if total_f <= 0.0:
        return 4096.0

    shell = np.asarray(shell_hist7, dtype=np.float64) / total_f
    mean_N = sum(w * shell[w] for w in range(7))
    rho = mean_N / 6.0
    eta = 1.0 - 2.0 * rho
    return 4096.0 / ((1.0 + eta * eta) ** 6)


# ----------------------------------------------------------------------
# ndarray helpers
# ----------------------------------------------------------------------


def _as_writeable_c_array(
    x,
    *,
    dtype,
    ndim: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype, order="C")
    if arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {arr.shape}")
    if not arr.flags.c_contiguous or not arr.flags.writeable:
        arr = np.array(arr, dtype=dtype, order="C", copy=True)
    return arr


def _as_read_c_array(
    x,
    *,
    dtype,
    ndim: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype, order="C")
    if arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {arr.shape}")
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr, dtype=dtype)
    return arr


def _void_p(arr: np.ndarray) -> ctypes.c_void_p:
    return ctypes.c_void_p(arr.ctypes.data)


# ----------------------------------------------------------------------
# Python fallback logic
# ----------------------------------------------------------------------


def _py_step_packed_omega12(omega12: int, byte: int) -> int:
    x = int(omega12) & 0xFFF
    b = int(byte) & 0xFF
    intron = b ^ 0xAA
    micro = (intron >> 1) & 0x3F
    eps_a = 0x3F if (intron & 0x01) else 0
    eps_b = 0x3F if (intron & 0x80) else 0
    u6 = (x >> 6) & 0x3F
    v6 = x & 0x3F
    u_next = v6 ^ eps_a
    v_next = u6 ^ micro ^ eps_b
    return ((u_next & 0x3F) << 6) | (v_next & 0x3F)


def _py_mask12_for_byte(byte: int) -> int:
    intron = (int(byte) & 0xFF) ^ 0xAA
    micro = (intron >> 1) & 0x3F
    mask12 = 0
    for i in range(6):
        if (micro >> i) & 1:
            mask12 |= 0x3 << (2 * i)
    return mask12 & 0xFFF


def _py_omega_byte_signature(byte: int) -> int:
    intron = (int(byte) & 0xFF) ^ 0xAA
    micro = (intron >> 1) & 0x3F
    eps_a = 0x3F if (intron & 0x01) else 0
    eps_b = 0x3F if (intron & 0x80) else 0
    parity = 1
    tau_u6 = eps_a & 0x3F
    tau_v6 = (micro ^ eps_b) & 0x3F
    return ((parity & 1) << 12) | ((tau_u6 & 0x3F) << 6) | (tau_v6 & 0x3F)


def _py_compose_omega_signatures(left: int, right: int) -> int:
    lp = (left >> 12) & 1
    ltu = (left >> 6) & 0x3F
    ltv = left & 0x3F

    rp = (right >> 12) & 1
    rtu = (right >> 6) & 0x3F
    rtv = right & 0x3F

    if lp == 0:
        ru, rv = rtu, rtv
    else:
        ru, rv = rtv, rtu

    return (((lp ^ rp) & 1) << 12) | (((ru ^ ltu) & 0x3F) << 6) | ((rv ^ ltv) & 0x3F)


def _py_push_chi_row(
    ring_row: np.ndarray,
    ring_pos: int,
    valid_len: int,
    hist_row: np.ndarray,
    shell_row: np.ndarray,
    chi6: int,
) -> tuple[int, int]:
    chi = int(chi6) & 0x3F
    shell = chi.bit_count()

    if valid_len < 64:
        ring_row[ring_pos] = chi
        hist_row[chi] += 1
        shell_row[shell] += 1
        return (ring_pos + 1) & 63, valid_len + 1

    chi_old = int(ring_row[ring_pos])
    shell_old = chi_old.bit_count()

    hist_row[chi_old] -= 1
    shell_row[shell_old] -= 1

    ring_row[ring_pos] = chi
    hist_row[chi] += 1
    shell_row[shell] += 1

    return (ring_pos + 1) & 63, valid_len


def _py_push_family_row(
    family_ring_row: np.ndarray,
    family_pos: int,
    family_valid_len: int,
    family_hist_row: np.ndarray,
    family: int,
) -> tuple[int, int]:
    fam = int(family) & 3

    if family_valid_len < 64:
        family_ring_row[family_pos] = fam
        family_hist_row[fam] += 1
        return (family_pos + 1) & 63, family_valid_len + 1

    family_old = int(family_ring_row[family_pos])
    family_hist_row[family_old] -= 1

    family_ring_row[family_pos] = fam
    family_hist_row[fam] += 1
    return (family_pos + 1) & 63, family_valid_len


def _py_family_from_byte(byte: int) -> int:
    intron = int(byte) ^ 0xAA
    return (((intron >> 7) & 1) << 1) | (intron & 1)


def _py_trace_word4_batch(omega12: np.ndarray, words4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = int(omega12.shape[0])
    omega_trace4 = np.empty((n, 4), dtype=np.int32)
    chi_trace4 = np.empty((n, 4), dtype=np.uint8)

    for i in range(n):
        s = int(omega12[i])
        for k in range(4):
            s = _py_step_packed_omega12(s, int(words4[i, k]))
            omega_trace4[i, k] = s
            chi_trace4[i, k] = ((s >> 6) ^ s) & 0x3F

    return omega_trace4, chi_trace4


def _py_compute_resonance_key(profile: int, omega12: int, w4: np.ndarray, omega_sig: int) -> int:
    p = int(profile)
    o12 = int(omega12) & 0x0FFF
    sig = int(omega_sig)
    if p == 1:
        return int(((o12 >> 6) ^ o12) & 0x3F)
    if p == 2:
        return int(((o12 >> 6) ^ o12) & 0x3F).bit_count()
    if p == 3:
        chi6 = int((o12 >> 6) ^ o12) & 0x3F
        if chi6 == 0:
            return 0
        if chi6 == 0x3F:
            return 1
        return 2
    if p == 4:
        return o12
    if p == 5:
        return sig & 0x1FFF
    if p == 6:
        return int(q_word6_for_items(bytes(int(x) & 0xFF for x in w4.flatten()))) & 0x3F
    return 0


def _py_apply_trace_word4_batch_indexed(
    cell_ids: np.ndarray,
    omega12: np.ndarray,
    step: np.ndarray,
    last_byte: np.ndarray,
    has_closed_word: np.ndarray,
    word4_store: np.ndarray,
    chi_ring64: np.ndarray,
    chi_ring_pos: np.ndarray,
    chi_valid_len: np.ndarray,
    chi_hist64: np.ndarray,
    shell_hist7: np.ndarray,
    family_ring64: np.ndarray,
    family_hist4: np.ndarray,
    omega_sig: np.ndarray,
    parity_O12: np.ndarray,
    parity_E12: np.ndarray,
    parity_bit: np.ndarray,
    words4: np.ndarray,
    omega_trace4: np.ndarray,
    chi_trace4: np.ndarray,
    resonance_key: np.ndarray,
    profile: int,
) -> None:
    n = int(cell_ids.shape[0])
    for i in range(n):
        cid = int(cell_ids[i])
        sig = 0
        O12 = 0
        E12 = 0

        for k in range(4):
            b = int(words4[i, k])
            step[cid] += 1
            last_byte[cid] = b
            word4_store[cid, k] = b

            chi_pos_new, chi_valid_new = _py_push_chi_row(
                chi_ring64[cid],
                int(chi_ring_pos[cid]),
                int(chi_valid_len[cid]),
                chi_hist64[cid],
                shell_hist7[cid],
                int(chi_trace4[i, k]),
            )
            _, _ = _py_push_family_row(
                family_ring64[cid],
                int(chi_ring_pos[cid]),
                int(chi_valid_len[cid]),
                family_hist4[cid],
                _py_family_from_byte(b),
            )
            chi_ring_pos[cid] = chi_pos_new
            chi_valid_len[cid] = chi_valid_new

            sig = _py_compose_omega_signatures(_py_omega_byte_signature(b), sig)
            if (k & 1) == 0:
                O12 ^= _py_mask12_for_byte(b)
            else:
                E12 ^= _py_mask12_for_byte(b)

        omega12[cid] = int(omega_trace4[i, 3])
        has_closed_word[cid] = 1
        omega_sig[cid] = sig
        parity_O12[cid] = O12 & 0x0FFF
        parity_E12[cid] = E12 & 0x0FFF
        parity_bit[cid] = 0
        resonance_key[cid] = _py_compute_resonance_key(
            profile,
            omega12[cid],
            words4[i],
            sig,
        )


# ----------------------------------------------------------------------
# Public APIs
# ----------------------------------------------------------------------


def ingest_word4_batch_indexed(
    cell_ids,
    omega12,
    step,
    last_byte,
    has_closed_word,
    word4_store,
    chi_ring64,
    chi_ring_pos,
    chi_valid_len,
    chi_hist64,
    shell_hist7,
    family_ring64,
    family_hist4,
    omega_sig,
    parity_O12,
    parity_E12,
    parity_bit,
    resonance_key,
    words4,
    profile,
) -> None:
    cell_ids_a = _as_read_c_array(cell_ids, dtype=np.int64, ndim=1, name="cell_ids")
    n = int(cell_ids_a.shape[0])
    if n == 0:
        return

    omega12_a = _as_writeable_c_array(omega12, dtype=np.int32, ndim=1, name="omega12")
    step_a = _as_writeable_c_array(step, dtype=np.uint64, ndim=1, name="step")
    last_byte_a = _as_writeable_c_array(last_byte, dtype=np.uint8, ndim=1, name="last_byte")
    has_closed_word_a = _as_writeable_c_array(
        has_closed_word, dtype=np.uint8, ndim=1, name="has_closed_word"
    )
    word4_store_a = _as_writeable_c_array(word4_store, dtype=np.uint8, ndim=2, name="word4_store")
    chi_ring64_a = _as_writeable_c_array(chi_ring64, dtype=np.uint8, ndim=2, name="chi_ring64")
    chi_ring_pos_a = _as_writeable_c_array(chi_ring_pos, dtype=np.uint8, ndim=1, name="chi_ring_pos")
    chi_valid_len_a = _as_writeable_c_array(
        chi_valid_len, dtype=np.uint8, ndim=1, name="chi_valid_len"
    )
    chi_hist64_a = _as_writeable_c_array(chi_hist64, dtype=np.uint16, ndim=2, name="chi_hist64")
    shell_hist7_a = _as_writeable_c_array(shell_hist7, dtype=np.uint16, ndim=2, name="shell_hist7")
    family_ring64_a = _as_writeable_c_array(
        family_ring64, dtype=np.uint8, ndim=2, name="family_ring64"
    )
    family_hist4_a = _as_writeable_c_array(
        family_hist4, dtype=np.uint16, ndim=2, name="family_hist4"
    )
    omega_sig_a = _as_writeable_c_array(omega_sig, dtype=np.int32, ndim=1, name="omega_sig")
    parity_O12_a = _as_writeable_c_array(parity_O12, dtype=np.uint16, ndim=1, name="parity_O12")
    parity_E12_a = _as_writeable_c_array(parity_E12, dtype=np.uint16, ndim=1, name="parity_E12")
    parity_bit_a = _as_writeable_c_array(parity_bit, dtype=np.uint8, ndim=1, name="parity_bit")
    resonance_key_a = _as_writeable_c_array(
        resonance_key, dtype=np.uint32, ndim=1, name="resonance_key"
    )
    words4_a = _as_read_c_array(words4, dtype=np.uint8, ndim=2, name="words4")

    if words4_a.shape != (n, 4):
        raise ValueError(f"words4 must have shape {(n, 4)}, got {words4_a.shape}")
    if np.any(cell_ids_a < 0) or np.any(cell_ids_a >= omega12_a.shape[0]):
        raise ValueError("cell_ids contain out-of-range indices")

    lib = _get_cpu_lib()
    if lib is None or not hasattr(lib, "gyrograph_ingest_word4_batch_indexed"):
        omega_trace4, chi_trace4 = _py_trace_word4_batch(omega12_a[cell_ids_a], words4_a)
        _py_apply_trace_word4_batch_indexed(
            cell_ids_a,
            omega12_a,
            step_a,
            last_byte_a,
            has_closed_word_a,
            word4_store_a,
            chi_ring64_a,
            chi_ring_pos_a,
            chi_valid_len_a,
            chi_hist64_a,
            shell_hist7_a,
            family_ring64_a,
            family_hist4_a,
            omega_sig_a,
            parity_O12_a,
            parity_E12_a,
            parity_bit_a,
            words4_a,
            omega_trace4,
            chi_trace4,
            resonance_key_a,
            int(profile),
        )
        return

    lib.gyrograph_ingest_word4_batch_indexed(
        _void_p(cell_ids_a),
        _void_p(omega12_a),
        _void_p(step_a),
        _void_p(last_byte_a),
        _void_p(has_closed_word_a),
        _void_p(word4_store_a),
        _void_p(chi_ring64_a),
        _void_p(chi_ring_pos_a),
        _void_p(chi_valid_len_a),
        _void_p(chi_hist64_a),
        _void_p(shell_hist7_a),
        _void_p(family_ring64_a),
        _void_p(family_hist4_a),
        _void_p(omega_sig_a),
        _void_p(parity_O12_a),
        _void_p(parity_E12_a),
        _void_p(parity_bit_a),
        _void_p(words4_a),
        _void_p(resonance_key_a),
        ctypes.c_ubyte(int(profile)),
        ctypes.c_int64(n),
    )


def trace_word4_batch_indexed(
    cell_ids,
    omega12,
    words4,
) -> tuple[np.ndarray, np.ndarray]:
    cell_ids_a = _as_read_c_array(cell_ids, dtype=np.int64, ndim=1, name="cell_ids")
    n = int(cell_ids_a.shape[0])
    if n == 0:
        return np.empty((0, 4), dtype=np.int32), np.empty((0, 4), dtype=np.uint8)

    omega12_a = _as_read_c_array(omega12, dtype=np.int32, ndim=1, name="omega12")
    if np.any(cell_ids_a < 0) or np.any(cell_ids_a >= omega12_a.shape[0]):
        raise ValueError("cell_ids contain out-of-range indices")
    words4_a = _as_read_c_array(words4, dtype=np.uint8, ndim=2, name="words4")
    if words4_a.shape != (n, 4):
        raise ValueError(f"words4 must have shape {(n, 4)}, got {words4_a.shape}")

    omega_trace4 = np.empty((n, 4), dtype=np.int32)
    chi_trace4 = np.empty((n, 4), dtype=np.uint8)

    lib = _get_cpu_lib()
    if lib is None or not hasattr(lib, "gyrograph_trace_word4_batch_indexed"):
        return _py_trace_word4_batch(omega12_a[cell_ids_a], words4_a)

    lib.gyrograph_trace_word4_batch_indexed(
        _void_p(cell_ids_a),
        _void_p(omega12_a),
        _void_p(words4_a),
        ctypes.c_int64(n),
        _void_p(omega_trace4),
        _void_p(chi_trace4),
    )
    return omega_trace4, chi_trace4


def apply_trace_word4_batch_indexed(
    cell_ids,
    omega12,
    step,
    last_byte,
    has_closed_word,
    word4_store,
    chi_ring64,
    chi_ring_pos,
    chi_valid_len,
    chi_hist64,
    shell_hist7,
    family_ring64,
    family_hist4,
    omega_sig,
    parity_O12,
    parity_E12,
    parity_bit,
    words4,
    omega_trace4,
    chi_trace4,
    resonance_key,
    profile,
) -> None:
    cell_ids_a = _as_read_c_array(cell_ids, dtype=np.int64, ndim=1, name="cell_ids")
    n = int(cell_ids_a.shape[0])
    if n == 0:
        return

    omega12_a = _as_writeable_c_array(omega12, dtype=np.int32, ndim=1, name="omega12")
    step_a = _as_writeable_c_array(step, dtype=np.uint64, ndim=1, name="step")
    last_byte_a = _as_writeable_c_array(last_byte, dtype=np.uint8, ndim=1, name="last_byte")
    has_closed_word_a = _as_writeable_c_array(
        has_closed_word, dtype=np.uint8, ndim=1, name="has_closed_word"
    )
    word4_store_a = _as_writeable_c_array(word4_store, dtype=np.uint8, ndim=2, name="word4_store")
    chi_ring64_a = _as_writeable_c_array(chi_ring64, dtype=np.uint8, ndim=2, name="chi_ring64")
    chi_ring_pos_a = _as_writeable_c_array(chi_ring_pos, dtype=np.uint8, ndim=1, name="chi_ring_pos")
    chi_valid_len_a = _as_writeable_c_array(
        chi_valid_len, dtype=np.uint8, ndim=1, name="chi_valid_len"
    )
    chi_hist64_a = _as_writeable_c_array(chi_hist64, dtype=np.uint16, ndim=2, name="chi_hist64")
    shell_hist7_a = _as_writeable_c_array(shell_hist7, dtype=np.uint16, ndim=2, name="shell_hist7")
    family_ring64_a = _as_writeable_c_array(
        family_ring64, dtype=np.uint8, ndim=2, name="family_ring64"
    )
    family_hist4_a = _as_writeable_c_array(
        family_hist4, dtype=np.uint16, ndim=2, name="family_hist4"
    )
    omega_sig_a = _as_writeable_c_array(omega_sig, dtype=np.int32, ndim=1, name="omega_sig")
    parity_O12_a = _as_writeable_c_array(parity_O12, dtype=np.uint16, ndim=1, name="parity_O12")
    parity_E12_a = _as_writeable_c_array(parity_E12, dtype=np.uint16, ndim=1, name="parity_E12")
    parity_bit_a = _as_writeable_c_array(parity_bit, dtype=np.uint8, ndim=1, name="parity_bit")
    resonance_key_a = _as_writeable_c_array(
        resonance_key, dtype=np.uint32, ndim=1, name="resonance_key"
    )
    words4_a = _as_read_c_array(words4, dtype=np.uint8, ndim=2, name="words4")
    omega_trace4_a = _as_read_c_array(omega_trace4, dtype=np.int32, ndim=2, name="omega_trace4")
    chi_trace4_a = _as_read_c_array(chi_trace4, dtype=np.uint8, ndim=2, name="chi_trace4")

    if words4_a.shape != (n, 4):
        raise ValueError(f"words4 must have shape {(n, 4)}, got {words4_a.shape}")
    if omega_trace4_a.shape != (n, 4):
        raise ValueError(
            f"omega_trace4 must have shape {(n, 4)}, got {omega_trace4_a.shape}"
        )
    if chi_trace4_a.shape != (n, 4):
        raise ValueError(f"chi_trace4 must have shape {(n, 4)}, got {chi_trace4_a.shape}")
    if np.any(cell_ids_a < 0) or np.any(cell_ids_a >= omega12_a.shape[0]):
        raise ValueError("cell_ids contain out-of-range indices")

    lib = _get_cpu_lib()
    if lib is None or not hasattr(lib, "gyrograph_apply_trace_word4_batch_indexed"):
        _py_apply_trace_word4_batch_indexed(
            cell_ids_a,
            omega12_a,
            step_a,
            last_byte_a,
            has_closed_word_a,
            word4_store_a,
            chi_ring64_a,
            chi_ring_pos_a,
            chi_valid_len_a,
            chi_hist64_a,
            shell_hist7_a,
            family_ring64_a,
            family_hist4_a,
            omega_sig_a,
            parity_O12_a,
            parity_E12_a,
            parity_bit_a,
            words4_a,
            omega_trace4_a,
            chi_trace4_a,
            resonance_key_a,
            int(profile),
        )
        return

    lib.gyrograph_apply_trace_word4_batch_indexed(
        _void_p(cell_ids_a),
        _void_p(omega12_a),
        _void_p(step_a),
        _void_p(last_byte_a),
        _void_p(has_closed_word_a),
        _void_p(word4_store_a),
        _void_p(chi_ring64_a),
        _void_p(chi_ring_pos_a),
        _void_p(chi_valid_len_a),
        _void_p(chi_hist64_a),
        _void_p(shell_hist7_a),
        _void_p(family_ring64_a),
        _void_p(family_hist4_a),
        _void_p(omega_sig_a),
        _void_p(parity_O12_a),
        _void_p(parity_E12_a),
        _void_p(parity_bit_a),
        _void_p(words4_a),
        _void_p(omega_trace4_a),
        _void_p(chi_trace4_a),
        _void_p(resonance_key_a),
        ctypes.c_ubyte(int(profile)),
        ctypes.c_int64(n),
    )


def trace_word4_batch_indexed_opencl(
    cell_ids,
    omega12,
    words4,
) -> tuple[np.ndarray, np.ndarray]:
    global _CL_INITIALIZED
    cell_ids_a = _as_read_c_array(cell_ids, dtype=np.int64, ndim=1, name="cell_ids")
    n = int(cell_ids_a.shape[0])
    if n == 0:
        return np.empty((0, 4), dtype=np.int32), np.empty((0, 4), dtype=np.uint8)

    omega12_a = _as_read_c_array(omega12, dtype=np.int32, ndim=1, name="omega12")
    if np.any(cell_ids_a < 0) or np.any(cell_ids_a >= omega12_a.shape[0]):
        raise ValueError("cell_ids contain out-of-range indices")
    words4_a = _as_read_c_array(words4, dtype=np.uint8, ndim=2, name="words4")
    if words4_a.shape != (n, 4):
        raise ValueError(f"words4 must have shape {(n, 4)}, got {words4_a.shape}")

    omega_trace4 = np.empty((n, 4), dtype=np.int32)
    chi_trace4 = np.empty((n, 4), dtype=np.uint8)

    lib = _get_cl_lib()
    if lib is None or not hasattr(lib, "gyrograph_cl_trace_word4_batch"):
        return trace_word4_batch_indexed(cell_ids_a, omega12_a, words4_a)

    if not _CL_INITIALIZED:
        try:
            initialize_opencl()
        except Exception:
            return trace_word4_batch_indexed(cell_ids_a, omega12_a, words4_a)

    ok = lib.gyrograph_cl_trace_word4_batch(
        _void_p(cell_ids_a),
        _void_p(omega12_a),
        _void_p(words4_a),
        ctypes.c_int64(n),
        _void_p(omega_trace4),
        _void_p(chi_trace4),
    )
    if ok != 1:
        _CL_INITIALIZED = False
        return trace_word4_batch_indexed(cell_ids_a, omega12_a, words4_a)

    return omega_trace4, chi_trace4
