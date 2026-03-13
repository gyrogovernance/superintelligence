# src/tools/gyrolabe/ops.py
from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

import torch

from src.constants import (
    GENE_MAC_A12,
    GENE_MAC_B12,
    LAYER_MASK_12,
    byte_family,
    byte_micro_ref,
    byte_to_intron,
    expand_intron_to_mask12,
    l0_parity,
    step_state_by_byte,
    unpack_state,
)

_THIS_FILE = Path(__file__).resolve()
_GYROLABE_DIR = _THIS_FILE.parent
_REPO_ROOT = _THIS_FILE.parents[3]
_CSRC_FILE = _GYROLABE_DIR / "gyrolabe.c"
_BUILD_DIR = _GYROLABE_DIR / "_build"

_NAMESPACE: Final[str] = "gyro"
_OPS_DEFINED = False
_LIB_DEF: object | None = None
_LIB_IMPL: object | None = None


def _shared_lib_suffix() -> str:
    if sys.platform.startswith("win"):
        return ".dll"
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


def _hash_source() -> str:
    h = hashlib.sha256()
    h.update(_CSRC_FILE.read_bytes())
    h.update(platform.platform().encode("utf-8"))
    h.update(platform.python_version().encode("utf-8"))
    return h.hexdigest()[:16]


def _find_clang_windows() -> str | None:
    """On Windows, find clang.exe even when not on PATH (e.g. LLVM installed but terminal not restarted)."""
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
    """On Windows, find gcc.exe when not on PATH. Check C:\\msys64 (ucrt64/mingw64)."""
    if shutil.which("gcc"):
        return shutil.which("gcc")
    base = Path("C:/msys64")
    for sub in ("ucrt64", "mingw64"):
        p = base / sub / "bin" / "gcc.exe"
        if p.exists():
            return str(p)
    return None


def _find_compiler() -> tuple[str, str]:
    if os.name == "nt":
        # Prefer GCC (works without VS). Then MSVC cl. Then LLVM clang (needs VS headers).
        gcc = shutil.which("gcc") or _find_gcc_windows()
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
        "No suitable C compiler was found for GyroLabe.\n"
        "Windows: install Visual Studio Build Tools and run from a Developer Command Prompt,\n"
        "or install clang/gcc.\n"
        "Linux/macOS: install clang or gcc."
    )


def _build_shared_library() -> Path:
    if not _CSRC_FILE.exists():
        raise FileNotFoundError(f"GyroLabe source file not found: {_CSRC_FILE}")

    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"gyrolabe_{_hash_source()}{_shared_lib_suffix()}"
    out_path = _BUILD_DIR / out_name
    if out_path.exists():
        return out_path

    compiler, kind = _find_compiler()

    native_flags: list[str] = []
    if os.environ.get("GYROLABE_NO_NATIVE", "").strip() != "1" and not sys.platform.startswith("win"):
        if kind == "cc":
            native_flags = ["-march=native"]
        elif kind == "msvc":
            native_flags = []

    openmp_flags: list[str] = []
    if os.environ.get("GYROLABE_NO_OPENMP", "").strip() != "1":
        if kind == "cc":
            openmp_flags = ["-fopenmp"]
        elif kind == "msvc":
            openmp_flags = ["/openmp"]

    if kind == "msvc":
        if Path(compiler).name.lower() == "cl.exe" or Path(compiler).name.lower() == "cl":
            cmd = [
                compiler,
                "/nologo",
                "/O2",
                "/LD",
                "/TC",
                *openmp_flags,
                str(_CSRC_FILE),
                f"/Fe:{out_path}",
            ]
        else:
            cmd = [
                compiler,
                "/nologo",
                "/O2",
                "/LD",
                "/TC",
                *openmp_flags,
                str(_CSRC_FILE),
                f"/Fe:{out_path}",
            ]
    else:
        if sys.platform == "darwin":
            cmd = [
                compiler,
                "-O3",
                "-std=c11",
                "-dynamiclib",
                *native_flags,
                "-o",
                str(out_path),
                str(_CSRC_FILE),
            ]
        else:
            # Windows clang/gcc: -fPIC not used (unsupported for MSVC target)
            flags = ["-O3", "-std=c11", "-shared", *openmp_flags]
            if not sys.platform.startswith("win"):
                flags.append("-fPIC")
            cmd = [
                compiler,
                *flags,
                *native_flags,
                "-o",
                str(out_path),
                str(_CSRC_FILE),
            ]

    # On Windows with GCC from MSYS2, run from repo root and put compiler's bin on PATH
    run_env = None
    run_cwd = str(_BUILD_DIR)
    if os.name == "nt" and kind == "cc":
        run_cwd = str(_REPO_ROOT)
        comp_dir = str(Path(compiler).resolve().parent)
        run_env = {**os.environ, "PATH": comp_dir + os.pathsep + os.environ.get("PATH", "")}
        rel_src = _CSRC_FILE.relative_to(_REPO_ROOT)
        rel_out = out_path.relative_to(_REPO_ROOT)
        cmd = cmd[:-2] + [str(rel_out), str(rel_src)]
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
            "Failed to compile GyroLabe C library.\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"stdout:\n{result.stdout}\n\n"
            f"stderr:\n{result.stderr}"
        )

    if not out_path.exists():
        raise RuntimeError(f"Compiler reported success but output library was not created: {out_path}")

    return out_path


def build_gyrolabe_native() -> Path:
    """
    Build the GyroLabe C shared library (src/tools/gyrolabe/gyrolabe.c).
    Use this to compile before running the bridge so the C backend is used.
    Raises if no C compiler is found. Run from repo root.
    """
    return _build_shared_library()


_LIB: ctypes.CDLL | None = None
_LIB_PATH: Path | None = None
_LIB_LOAD_ATTEMPTED = False
_USE_PURE_PYTHON = False


def _dll_search_paths() -> list[Path]:
    """Paths to add to DLL search when loading GyroLabe (Windows: GCC runtime deps)."""
    if not sys.platform.startswith("win"):
        return []
    paths: list[Path] = []
    for candidate in (
        os.environ.get("LLVM_HOME"),
        os.environ.get("LLVM_PATH"),
        os.environ.get("LLVM_ROOT"),
    ):
        if candidate:
            p = Path(candidate) / "bin"
            if p.exists():
                paths.append(p)
    base = Path("C:/msys64")
    for sub in ("ucrt64", "mingw64"):
        p = base / sub / "bin"
        if p.exists():
            paths.append(p)
    return paths


def _get_lib() -> ctypes.CDLL | None:
    global _LIB, _LIB_PATH, _LIB_LOAD_ATTEMPTED, _USE_PURE_PYTHON
    if _USE_PURE_PYTHON:
        return None
    if _LIB is not None:
        return _LIB
    if _LIB_LOAD_ATTEMPTED:
        return None
    _LIB_LOAD_ATTEMPTED = True
    try:
        path = _build_shared_library()
        _LIB_PATH = path
        dll_dirs = _dll_search_paths()
        if sys.platform.startswith("win") and hasattr(os, "add_dll_directory"):
            for d in dll_dirs:
                try:
                    os.add_dll_directory(str(d))
                except OSError:
                    pass
        else:
            extra = os.pathsep.join(str(d) for d in dll_dirs)
            if extra:
                os.environ["PATH"] = extra + os.pathsep + os.environ.get("PATH", "")
        try:
            _LIB = ctypes.CDLL(str(path))
        finally:
            pass
        _setup_lib_argtypes(_LIB)
        return _LIB
    except Exception as e:
        _USE_PURE_PYTHON = True
        _LIB = None
        _LIB_PATH = None
        import warnings
        msg = str(e).strip()
        if "one of its dependencies" in msg or "dependencies" in msg.lower():
            hint = " Add the compiler's bin dir to PATH (e.g. C:\\msys64\\ucrt64\\bin for GCC)."
        else:
            hint = ""
        warnings.warn(
            f"GyroLabe C library could not be loaded. Using pure-Python fallbacks. {msg}{hint}",
            UserWarning,
            stacklevel=2,
        )
        return None


def _setup_lib_argtypes(lib: ctypes.CDLL) -> None:
    lib.gyro_signature_scan.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.gyro_signature_scan.restype = None
    lib.gyro_chirality_distance.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    lib.gyro_chirality_distance.restype = None
    lib.gyro_wht64_float.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
    ]
    lib.gyro_wht64_float.restype = None
    lib.gyro_qmap_extract.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.gyro_qmap_extract.restype = None
    lib.gyro_apply_signature_to_rest.argtypes = [ctypes.c_int32]
    lib.gyro_apply_signature_to_rest.restype = ctypes.c_int32
    lib.gyro_extract_scan.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.gyro_extract_scan.restype = None
    lib.gyro_chirality_distance_adjacent.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    lib.gyro_chirality_distance_adjacent.restype = None
    if hasattr(lib, "gyro_apply_signature_to_state"):
        lib.gyro_apply_signature_to_state.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.gyro_apply_signature_to_state.restype = ctypes.c_int32
    if hasattr(lib, "gyro_apply_signature_batch"):
        lib.gyro_apply_signature_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        lib.gyro_apply_signature_batch.restype = None
    if hasattr(lib, "gyro_step_byte_batch"):
        lib.gyro_step_byte_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_uint8,
            ctypes.c_void_p,
        ]
        lib.gyro_step_byte_batch.restype = None
    if hasattr(lib, "gyro_state_scan_from_state"):
        lib.gyro_state_scan_from_state.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_void_p,
        ]
        lib.gyro_state_scan_from_state.restype = None
    if hasattr(lib, "gyro_state24_to_omega12_batch"):
        lib.gyro_state24_to_omega12_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.gyro_state24_to_omega12_batch.restype = None
    if hasattr(lib, "gyro_omega12_to_state24_batch"):
        lib.gyro_omega12_to_state24_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        lib.gyro_omega12_to_state24_batch.restype = None
    if hasattr(lib, "gyro_step_omega12_batch"):
        lib.gyro_step_omega12_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_uint8,
            ctypes.c_void_p,
        ]
        lib.gyro_step_omega12_batch.restype = None
    if hasattr(lib, "gyro_apply_omega_signature_batch"):
        lib.gyro_apply_omega_signature_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        lib.gyro_apply_omega_signature_batch.restype = None
    if hasattr(lib, "gyro_shell_histogram_state24"):
        lib.gyro_shell_histogram_state24.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        lib.gyro_shell_histogram_state24.restype = None
    if hasattr(lib, "gyro_shell_histogram_omega12"):
        lib.gyro_shell_histogram_omega12.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        lib.gyro_shell_histogram_omega12.restype = None
    if hasattr(lib, "gyro_omega_signature_scan"):
        lib.gyro_omega_signature_scan.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        lib.gyro_omega_signature_scan.restype = None
    if hasattr(lib, "gyro_omega12_scan_from_omega12"):
        lib.gyro_omega12_scan_from_omega12.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_void_p,
        ]
        lib.gyro_omega12_scan_from_omega12.restype = None
    if hasattr(lib, "gyro_shell_histogram_state24_checked"):
        lib.gyro_shell_histogram_state24_checked.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        lib.gyro_shell_histogram_state24_checked.restype = ctypes.c_int64
    if hasattr(lib, "gyro_apply_omega_gate_batch"):
        lib.gyro_apply_omega_gate_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_uint8,
            ctypes.c_void_p,
        ]
        lib.gyro_apply_omega_gate_batch.restype = None
    if hasattr(lib, "gyro_bitplane_gemv_f32"):
        lib.gyro_bitplane_gemv_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_void_p,
        ]
        lib.gyro_bitplane_gemv_f32.restype = None
    if hasattr(lib, "gyro_pack_bitplane_matrix_f32"):
        lib.gyro_pack_bitplane_matrix_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.gyro_pack_bitplane_matrix_f32.restype = None
    if hasattr(lib, "gyro_bitplane_gemv_packed_f32"):
        lib.gyro_bitplane_gemv_packed_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_void_p,
        ]
        lib.gyro_bitplane_gemv_packed_f32.restype = None
    if hasattr(lib, "gyro_pack_bitplane_vector_f32"):
        lib.gyro_pack_bitplane_vector_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.gyro_pack_bitplane_vector_f32.restype = None
    if hasattr(lib, "gyro_bitplane_gemv_packed_x_f32"):
        lib.gyro_bitplane_gemv_packed_x_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_void_p,
        ]
        lib.gyro_bitplane_gemv_packed_x_f32.restype = None
    if hasattr(lib, "gyro_init"):
        lib.gyro_init.argtypes = []
        lib.gyro_init.restype = None
    if hasattr(lib, "gyro_pack_bitplane_matrix_i32"):
        lib.gyro_pack_bitplane_matrix_i32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.gyro_pack_bitplane_matrix_i32.restype = None
    if hasattr(lib, "gyro_pack_bitplane_vector_i32"):
        lib.gyro_pack_bitplane_vector_i32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.gyro_pack_bitplane_vector_i32.restype = None
    if hasattr(lib, "gyro_bitplane_gemv_packed_i32"):
        lib.gyro_bitplane_gemv_packed_i32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_void_p,
        ]
        lib.gyro_bitplane_gemv_packed_i32.restype = None
    if hasattr(lib, "gyro_pack_bitplane_vector_batch_f32"):
        lib.gyro_pack_bitplane_vector_batch_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.gyro_pack_bitplane_vector_batch_f32.restype = None
    if hasattr(lib, "gyro_bitplane_gemm_packed_x_batch_f32"):
        lib.gyro_bitplane_gemm_packed_x_batch_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_void_p,
        ]
        lib.gyro_bitplane_gemm_packed_x_batch_f32.restype = None


def _py_collapse_pairdiag12_to_word6(x12: int) -> int:
    out = 0
    for i in range(6):
        pair = (x12 >> (2 * i)) & 0x3
        if pair == 0x3:
            out |= 1 << i
    return out


def _py_pack_signature(parity: int, tau_a12: int, tau_b12: int) -> int:
    return ((parity & 1) << 24) | ((tau_a12 & LAYER_MASK_12) << 12) | (tau_b12 & LAYER_MASK_12)


def _py_unpack_signature(sig: int) -> tuple[int, int, int]:
    parity = (sig >> 24) & 1
    tau_a12 = (sig >> 12) & LAYER_MASK_12
    tau_b12 = sig & LAYER_MASK_12
    return parity, tau_a12, tau_b12


def _py_byte_signature(b: int, invert_a: int, mask12: int, invert_b: int) -> int:
    tau_a = invert_a
    tau_b = (mask12 ^ invert_b) & LAYER_MASK_12
    return _py_pack_signature(1, tau_a, tau_b)


def _py_compose_signatures(left: int, right: int) -> int:
    lp, lta, ltb = _py_unpack_signature(left)
    rp, rta, rtb = _py_unpack_signature(right)
    if lp == 0:
        ra, rb = rta, rtb
    else:
        ra, rb = rtb, rta
    return _py_pack_signature(lp ^ rp, (ra ^ lta) & LAYER_MASK_12, (rb ^ ltb) & LAYER_MASK_12)


def _py_signature_scan(bytes_tensor: torch.Tensor) -> torch.Tensor:
    x = _ensure_cpu_contiguous(bytes_tensor, dtype=torch.uint8, name="bytes")
    if x.ndim == 0:
        x = x.reshape(1)
    flat = x.reshape(-1)
    out = torch.empty(flat.shape, dtype=torch.int32, device="cpu")
    tables: list[tuple[int, int, int, int]] = []
    for b in range(256):
        intron = byte_to_intron(b)
        mask12 = expand_intron_to_mask12(intron)
        invert_a = LAYER_MASK_12 if (intron & 1) else 0
        invert_b = LAYER_MASK_12 if (intron & 0x80) else 0
        tables.append((invert_a, mask12, invert_b, _py_byte_signature(b, invert_a, mask12, invert_b)))
    accum = 0
    for i in range(flat.numel()):
        _, _, _, sig_b = tables[int(flat[i].item()) & 0xFF]
        accum = _py_compose_signatures(sig_b, accum)
        out[i] = accum
    return out.reshape(x.shape)


def _py_chirality_distance(states_a: torch.Tensor, states_b: torch.Tensor) -> torch.Tensor:
    a = _ensure_cpu_contiguous(states_a, dtype=torch.int32, name="states_a")
    b = _ensure_cpu_contiguous(states_b, dtype=torch.int32, name="states_b")
    if a.shape != b.shape:
        raise ValueError(f"states_a and states_b must have the same shape, got {a.shape} vs {b.shape}")
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    out = torch.empty(flat_a.shape, dtype=torch.uint8, device="cpu")
    for i in range(flat_a.numel()):
        sa = int(flat_a[i].item()) & 0xFFFFFF
        sb = int(flat_b[i].item()) & 0xFFFFFF
        a12, b12 = unpack_state(sa)
        ca = _py_collapse_pairdiag12_to_word6((a12 ^ b12) & LAYER_MASK_12)
        a12b, b12b = unpack_state(sb)
        cb = _py_collapse_pairdiag12_to_word6((a12b ^ b12b) & LAYER_MASK_12)
        out[i] = bin(ca ^ cb).count("1")
    return out.reshape(a.shape)


def _py_qmap_extract(bytes_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = _ensure_cpu_contiguous(bytes_tensor, dtype=torch.uint8, name="bytes")
    flat = x.reshape(-1)
    q = torch.empty(flat.shape, dtype=torch.uint8, device="cpu")
    f = torch.empty(flat.shape, dtype=torch.uint8, device="cpu")
    m = torch.empty(flat.shape, dtype=torch.uint8, device="cpu")
    for i in range(flat.numel()):
        b = int(flat[i].item()) & 0xFF
        intron = byte_to_intron(b)
        mask12 = expand_intron_to_mask12(intron)
        l0 = l0_parity(intron)
        q12 = (mask12 ^ (LAYER_MASK_12 if l0 else 0)) & LAYER_MASK_12
        q_class = _py_collapse_pairdiag12_to_word6(q12)
        q[i] = q_class
        f[i] = byte_family(b)
        m[i] = byte_micro_ref(b)
    return (
        q.reshape(x.shape),
        f.reshape(x.shape),
        m.reshape(x.shape),
    )


def _py_apply_signature_to_rest(signature: int) -> int:
    parity, tau_a12, tau_b12 = _py_unpack_signature(signature)
    if parity == 0:
        a12 = (GENE_MAC_A12 ^ tau_a12) & LAYER_MASK_12
        b12 = (GENE_MAC_B12 ^ tau_b12) & LAYER_MASK_12
    else:
        a12 = (GENE_MAC_B12 ^ tau_a12) & LAYER_MASK_12
        b12 = (GENE_MAC_A12 ^ tau_b12) & LAYER_MASK_12
    return (a12 << 12) | b12


def _py_apply_signature_to_state(state24: int, signature: int) -> int:
    parity, tau_a12, tau_b12 = _py_unpack_signature(signature)
    a12, b12 = unpack_state(state24)
    if parity == 0:
        a12 = (a12 ^ tau_a12) & LAYER_MASK_12
        b12 = (b12 ^ tau_b12) & LAYER_MASK_12
    else:
        a_in = a12
        a12 = (b12 ^ tau_a12) & LAYER_MASK_12
        b12 = (a_in ^ tau_b12) & LAYER_MASK_12
    return ((a12 & LAYER_MASK_12) << 12) | (b12 & LAYER_MASK_12)


def _ensure_cpu_contiguous(x: torch.Tensor, *, dtype: torch.dtype, name: str) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)!r}")
    if x.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU for GyroLabe C ops, got device={x.device}")
    if x.dtype != dtype:
        x = x.to(dtype=dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    return x


def _signature_scan_impl(bytes_tensor: torch.Tensor) -> torch.Tensor:
    lib = _get_lib()
    if lib is None:
        return _py_signature_scan(bytes_tensor)
    x = _ensure_cpu_contiguous(bytes_tensor, dtype=torch.uint8, name="bytes")
    if x.ndim == 0:
        x = x.reshape(1)

    if x.ndim == 1:
        out = torch.empty_like(x, dtype=torch.int32)
        lib.gyro_signature_scan(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_int64(x.numel()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(max(1, os.cpu_count() or 1)),
        )
        return out

    rows = x.reshape(-1, x.shape[-1])
    out = torch.empty(rows.shape, dtype=torch.int32, device="cpu")
    width = rows.shape[-1]
    for i in range(rows.shape[0]):
        row = rows[i]
        row_out = out[i]
        lib.gyro_signature_scan(
            ctypes.c_void_p(row.data_ptr()),
            ctypes.c_int64(width),
            ctypes.c_void_p(row_out.data_ptr()),
            ctypes.c_int(max(1, os.cpu_count() or 1)),
        )
    return out.reshape(x.shape)


def _chirality_distance_impl(states_a: torch.Tensor, states_b: torch.Tensor) -> torch.Tensor:
    lib = _get_lib()
    if lib is None:
        return _py_chirality_distance(states_a, states_b)
    a = _ensure_cpu_contiguous(states_a, dtype=torch.int32, name="states_a")
    b = _ensure_cpu_contiguous(states_b, dtype=torch.int32, name="states_b")
    if a.shape != b.shape:
        raise ValueError(f"states_a and states_b must have the same shape, got {a.shape} vs {b.shape}")

    out = torch.empty(a.shape, dtype=torch.uint8, device="cpu")
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    flat_out = out.reshape(-1)

    lib.gyro_chirality_distance(
        ctypes.c_void_p(flat_a.data_ptr()),
        ctypes.c_void_p(flat_b.data_ptr()),
        ctypes.c_int64(flat_a.numel()),
        ctypes.c_void_p(flat_out.data_ptr()),
    )
    return out


def _wht64_forward_impl(x: torch.Tensor) -> torch.Tensor:
    lib = _get_lib()
    if lib is None:
        raise RuntimeError("GyroLabe C library not available (no compiler). wht64 has no pure-Python fallback.")
    t = _ensure_cpu_contiguous(x, dtype=torch.float32, name="x")
    if t.shape[-1] != 64:
        raise ValueError(f"gyro::wht64 requires trailing dimension 64, got shape={tuple(t.shape)}")

    rows = t.reshape(-1, 64)
    out = torch.empty_like(rows)
    lib.gyro_wht64_float(
        ctypes.c_void_p(rows.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int64(rows.shape[0]),
    )
    return out.reshape(t.shape)


def _extract_scan_impl(bytes_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused: q_class, family, micro_ref, signatures, states in one pass."""
    lib = _get_lib()
    x = _ensure_cpu_contiguous(bytes_tensor, dtype=torch.uint8, name="bytes")
    if x.ndim == 0:
        x = x.reshape(1)
    flat = x.reshape(-1)
    n = flat.numel()
    q = torch.empty(n, dtype=torch.uint8, device="cpu")
    f = torch.empty(n, dtype=torch.uint8, device="cpu")
    m = torch.empty(n, dtype=torch.uint8, device="cpu")
    sigs = torch.empty(n, dtype=torch.int32, device="cpu")
    states = torch.empty(n, dtype=torch.int32, device="cpu")
    if lib is not None:
        lib.gyro_extract_scan(
            ctypes.c_void_p(flat.data_ptr()),
            ctypes.c_int64(n),
            ctypes.c_void_p(q.data_ptr()),
            ctypes.c_void_p(f.data_ptr()),
            ctypes.c_void_p(m.data_ptr()),
            ctypes.c_void_p(sigs.data_ptr()),
            ctypes.c_void_p(states.data_ptr()),
        )
    else:
        q_py, f_py, m_py = _py_qmap_extract(flat)
        sigs_py = _py_signature_scan(flat)
        q.copy_(q_py)
        f.copy_(f_py)
        m.copy_(m_py)
        sigs.copy_(sigs_py)
        states.copy_(signatures_to_states(sigs_py))
    return (
        q.reshape(x.shape),
        f.reshape(x.shape),
        m.reshape(x.shape),
        sigs.reshape(x.shape),
        states.reshape(x.shape),
    )


def _chirality_distance_adjacent_impl(
    states: torch.Tensor,
    lookahead: int,
) -> torch.Tensor:
    lib = _get_lib()
    s = _ensure_cpu_contiguous(states, dtype=torch.int32, name="states")
    flat = s.reshape(-1)
    out = torch.empty(flat.shape, dtype=torch.uint8, device="cpu")
    if lib is not None:
        lib.gyro_chirality_distance_adjacent(
            ctypes.c_void_p(flat.data_ptr()),
            ctypes.c_int64(flat.numel()),
            ctypes.c_int32(lookahead),
            ctypes.c_void_p(out.data_ptr()),
        )
    else:
        lah = max(1, lookahead) if lookahead >= 0 else 1
        n = flat.numel()
        if n > lah:
            d = _py_chirality_distance(flat[:-lah], flat[lah:])
            out[:-lah].copy_(d)
    return out.reshape(s.shape)


def _qmap_extract_impl(bytes_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lib = _get_lib()
    if lib is None:
        return _py_qmap_extract(bytes_tensor)
    x = _ensure_cpu_contiguous(bytes_tensor, dtype=torch.uint8, name="bytes")
    q = torch.empty_like(x, dtype=torch.uint8)
    f = torch.empty_like(x, dtype=torch.uint8)
    m = torch.empty_like(x, dtype=torch.uint8)

    flat_x = x.reshape(-1)
    flat_q = q.reshape(-1)
    flat_f = f.reshape(-1)
    flat_m = m.reshape(-1)

    lib.gyro_qmap_extract(
        ctypes.c_void_p(flat_x.data_ptr()),
        ctypes.c_int64(flat_x.numel()),
        ctypes.c_void_p(flat_q.data_ptr()),
        ctypes.c_void_p(flat_f.data_ptr()),
        ctypes.c_void_p(flat_m.data_ptr()),
    )
    return q, f, m


def _define_ops_once() -> None:
    global _OPS_DEFINED, _LIB_DEF, _LIB_IMPL
    if _OPS_DEFINED:
        return

    lib_def = torch.library.Library(_NAMESPACE, "DEF")
    _LIB_DEF = lib_def
    lib_def.define("signature_scan(Tensor bytes) -> Tensor")
    lib_def.define("chirality_distance(Tensor states_a, Tensor states_b) -> Tensor")
    lib_def.define("wht64_forward(Tensor x) -> Tensor")
    lib_def.define("qmap_extract(Tensor bytes) -> (Tensor, Tensor, Tensor)")

    lib_impl = torch.library.Library(_NAMESPACE, "IMPL", "CPU")
    _LIB_IMPL = lib_impl
    lib_impl.impl("signature_scan", _signature_scan_impl)
    lib_impl.impl("chirality_distance", _chirality_distance_impl)
    lib_impl.impl("wht64_forward", _wht64_forward_impl)
    lib_impl.impl("qmap_extract", _qmap_extract_impl)

    @torch.library.register_fake(f"{_NAMESPACE}::signature_scan")
    def _fake_signature_scan(bytes: torch.Tensor) -> torch.Tensor:
        return torch.empty(bytes.shape, dtype=torch.int32, device=bytes.device)

    @torch.library.register_fake(f"{_NAMESPACE}::chirality_distance")
    def _fake_chirality_distance(states_a: torch.Tensor, states_b: torch.Tensor) -> torch.Tensor:
        if states_a.shape != states_b.shape:
            raise ValueError("states_a and states_b must have identical shapes")
        return torch.empty(states_a.shape, dtype=torch.uint8, device=states_a.device)

    @torch.library.register_fake(f"{_NAMESPACE}::wht64_forward")
    def _fake_wht64_forward(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 64:
            raise ValueError("gyro::wht64_forward requires trailing dimension 64")
        return torch.empty_like(x, dtype=torch.float32)

    @torch.library.register_fake(f"{_NAMESPACE}::qmap_extract")
    def _fake_qmap_extract(bytes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = bytes.shape
        return (
            torch.empty(shape, dtype=torch.uint8, device=bytes.device),
            torch.empty(shape, dtype=torch.uint8, device=bytes.device),
            torch.empty(shape, dtype=torch.uint8, device=bytes.device),
        )

    _OPS_DEFINED = True


_define_ops_once()


def signature_scan(bytes_tensor: torch.Tensor) -> torch.Tensor:
    return torch.ops.gyro.signature_scan(bytes_tensor)


def chirality_distance(states_a: torch.Tensor, states_b: torch.Tensor) -> torch.Tensor:
    return torch.ops.gyro.chirality_distance(states_a, states_b)


class _WHT64Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        y = torch.ops.gyro.wht64_forward(x)
        return y

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        # The normalized WHT is orthonormal: backward = forward.
        return (torch.ops.gyro.wht64_forward(grad_output),)


def wht64(x: torch.Tensor) -> torch.Tensor:
    return _WHT64Function.apply(x)


def qmap_extract(bytes_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.gyro.qmap_extract(bytes_tensor)


def apply_signature_to_rest(signature: torch.Tensor) -> torch.Tensor:
    lib = _get_lib()
    sig = _ensure_cpu_contiguous(signature, dtype=torch.int32, name="signature")
    flat = sig.reshape(-1)
    out = torch.empty_like(flat, dtype=torch.int32)
    if lib is None:
        for i in range(flat.numel()):
            out[i] = _py_apply_signature_to_rest(int(flat[i].item()))
    else:
        for i in range(flat.numel()):
            out[i] = int(lib.gyro_apply_signature_to_rest(int(flat[i].item())))
    return out.reshape(sig.shape)


def apply_signature_to_state(
    state24: torch.Tensor | int,
    signature: torch.Tensor | int,
) -> torch.Tensor:
    lib = _get_lib()
    if not isinstance(state24, torch.Tensor):
        state = torch.tensor(state24, dtype=torch.int32, device="cpu")
    else:
        state = _ensure_cpu_contiguous(state24, dtype=torch.int32, name="state24")
    if not isinstance(signature, torch.Tensor):
        sig = torch.tensor(signature, dtype=torch.int32, device="cpu")
    else:
        sig = _ensure_cpu_contiguous(signature, dtype=torch.int32, name="signature")

    if state.shape != sig.shape:
        if sig.numel() == 1:
            sig = sig.reshape(1).expand_as(state)
        elif state.numel() == 1:
            state = state.reshape(1).expand_as(sig)
        else:
            raise ValueError(f"state24 and signature must have broadcastable shapes, got {state.shape} and {sig.shape}")

    flat_state = state.reshape(-1).contiguous()
    flat_sig = sig.reshape(-1).contiguous()
    out = torch.empty_like(flat_state, dtype=torch.int32)

    if lib is not None and hasattr(lib, "gyro_apply_signature_batch"):
        lib.gyro_apply_signature_batch(
            ctypes.c_void_p(flat_state.data_ptr()),
            ctypes.c_void_p(flat_sig.data_ptr()),
            ctypes.c_int64(flat_state.numel()),
            ctypes.c_void_p(out.data_ptr()),
        )
    elif lib is not None and hasattr(lib, "gyro_apply_signature_to_state"):
        for i in range(flat_state.numel()):
            out[i] = int(
                lib.gyro_apply_signature_to_state(
                    int(flat_state[i].item()),
                    int(flat_sig[i].item()),
                )
            )
    else:
        for i in range(flat_state.numel()):
            out[i] = _py_apply_signature_to_state(
                int(flat_state[i].item()),
                int(flat_sig[i].item()),
            )

    return out.reshape(state.shape)


def apply_signature_batch(states: torch.Tensor, signatures: torch.Tensor) -> torch.Tensor:
    lib = _get_lib()
    s = _ensure_cpu_contiguous(states, dtype=torch.int32, name="states")
    sig = _ensure_cpu_contiguous(signatures, dtype=torch.int32, name="signatures")
    if s.shape != sig.shape:
        raise ValueError(f"states and signatures must have the same shape, got {s.shape} and {sig.shape}")

    flat_s = s.reshape(-1)
    flat_sig = sig.reshape(-1)
    out = torch.empty_like(flat_s, dtype=torch.int32)

    if lib is None or not hasattr(lib, "gyro_apply_signature_batch"):
        for i in range(flat_s.numel()):
            out[i] = _py_apply_signature_to_state(int(flat_s[i].item()), int(flat_sig[i].item()))
    else:
        lib.gyro_apply_signature_batch(
            ctypes.c_void_p(flat_s.data_ptr()),
            ctypes.c_void_p(flat_sig.data_ptr()),
            ctypes.c_int64(flat_s.numel()),
            ctypes.c_void_p(out.data_ptr()),
        )

    return out.reshape(s.shape)


def step_byte_batch(states: torch.Tensor, byte: int) -> torch.Tensor:
    lib = _get_lib()
    s = _ensure_cpu_contiguous(states, dtype=torch.int32, name="states")
    flat_s = s.reshape(-1)
    out = torch.empty_like(flat_s, dtype=torch.int32)
    b = int(byte) & 0xFF

    if lib is None or not hasattr(lib, "gyro_step_byte_batch"):
        for i in range(flat_s.numel()):
            out[i] = step_state_by_byte(int(flat_s[i].item()), b)
    else:
        lib.gyro_step_byte_batch(
            ctypes.c_void_p(flat_s.data_ptr()),
            ctypes.c_int64(flat_s.numel()),
            ctypes.c_uint8(b),
            ctypes.c_void_p(out.data_ptr()),
        )

    return out.reshape(s.shape)


def state24_to_omega12_batch(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert state24 batch to packed omega12. Returns (omega12, valid).

    For non-Omega (invalid) inputs, the packed value is a placeholder (e.g. 0)
    and must not be inverted; only valid entries may be inverted meaningfully.
    """
    lib = _get_lib()
    s = _ensure_cpu_contiguous(states, dtype=torch.int32, name="states")
    flat = s.reshape(-1)
    omega = torch.empty_like(flat, dtype=torch.int32)
    valid = torch.empty(flat.shape, dtype=torch.uint8, device="cpu")

    if lib is None or not hasattr(lib, "gyro_state24_to_omega12_batch"):
        from src.api import try_state24_to_omega12, pack_omega12
        for i in range(flat.numel()):
            o = try_state24_to_omega12(int(flat[i].item()))
            if o is not None:
                omega[i] = pack_omega12(o)
                valid[i] = 1
            else:
                omega[i] = 0
                valid[i] = 0
    else:
        lib.gyro_state24_to_omega12_batch(
            ctypes.c_void_p(flat.data_ptr()),
            ctypes.c_int64(flat.numel()),
            ctypes.c_void_p(omega.data_ptr()),
            ctypes.c_void_p(valid.data_ptr()),
        )

    return omega.reshape(s.shape), valid.reshape(s.shape)


def omega12_to_state24_batch(omega12: torch.Tensor) -> torch.Tensor:
    """
    Convert packed omega12 batch to state24.

    Input must be from valid Omega states; packed values from invalid state24
    conversions are placeholders and produce undefined results when inverted.
    """
    lib = _get_lib()
    o = _ensure_cpu_contiguous(omega12, dtype=torch.int32, name="omega12")
    flat = o.reshape(-1)
    out = torch.empty_like(flat, dtype=torch.int32)

    if lib is None or not hasattr(lib, "gyro_omega12_to_state24_batch"):
        from src.api import unpack_omega12, omega12_to_state24
        for i in range(flat.numel()):
            out[i] = omega12_to_state24(unpack_omega12(int(flat[i].item())))
    else:
        lib.gyro_omega12_to_state24_batch(
            ctypes.c_void_p(flat.data_ptr()),
            ctypes.c_int64(flat.numel()),
            ctypes.c_void_p(out.data_ptr()),
        )

    return out.reshape(o.shape)


def step_omega12_batch(omega12: torch.Tensor, byte: int) -> torch.Tensor:
    """Step packed omega12 batch by one byte."""
    lib = _get_lib()
    o = _ensure_cpu_contiguous(omega12, dtype=torch.int32, name="omega12")
    flat = o.reshape(-1)
    out = torch.empty_like(flat, dtype=torch.int32)
    b = int(byte) & 0xFF

    if lib is None or not hasattr(lib, "gyro_step_omega12_batch"):
        from src.api import unpack_omega12, step_omega12_by_byte, pack_omega12
        for i in range(flat.numel()):
            out[i] = pack_omega12(step_omega12_by_byte(unpack_omega12(int(flat[i].item())), b))
    else:
        lib.gyro_step_omega12_batch(
            ctypes.c_void_p(flat.data_ptr()),
            ctypes.c_int64(flat.numel()),
            ctypes.c_uint8(b),
            ctypes.c_void_p(out.data_ptr()),
        )

    return out.reshape(o.shape)


def apply_omega_signature_batch(
    omega12: torch.Tensor,
    signatures: torch.Tensor,
) -> torch.Tensor:
    """Apply omega signatures to packed omega12 batch."""
    lib = _get_lib()
    o = _ensure_cpu_contiguous(omega12, dtype=torch.int32, name="omega12")
    sig = _ensure_cpu_contiguous(signatures, dtype=torch.int32, name="signatures")
    if o.numel() != sig.numel():
        raise ValueError(
            f"omega12 and signatures must have same size, got {o.numel()} vs {sig.numel()}"
        )
    flat_o = o.reshape(-1)
    flat_sig = sig.reshape(-1)
    out = torch.empty_like(flat_o, dtype=torch.int32)

    if lib is None or not hasattr(lib, "gyro_apply_omega_signature_batch"):
        from src.api import unpack_omega12, unpack_omega_signature12, apply_omega_signature, pack_omega12
        for i in range(flat_o.numel()):
            om = unpack_omega12(int(flat_o[i].item()))
            sg = unpack_omega_signature12(int(flat_sig[i].item()))
            out[i] = pack_omega12(apply_omega_signature(om, sg))
    else:
        lib.gyro_apply_omega_signature_batch(
            ctypes.c_void_p(flat_o.data_ptr()),
            ctypes.c_void_p(flat_sig.data_ptr()),
            ctypes.c_int64(flat_o.numel()),
            ctypes.c_void_p(out.data_ptr()),
        )

    return out.reshape(o.shape)


def shell_histogram_state24(states: torch.Tensor) -> torch.Tensor:
    """Shell histogram from state24 batch. Returns length-7 int32 tensor."""
    lib = _get_lib()
    s = _ensure_cpu_contiguous(states, dtype=torch.int32, name="states")
    flat = s.reshape(-1)
    out = torch.zeros(7, dtype=torch.int32, device="cpu")

    if lib is None or not hasattr(lib, "gyro_shell_histogram_state24"):
        from src.api import chirality_word6
        for i in range(flat.numel()):
            w = chirality_word6(int(flat[i].item())).bit_count()
            out[w] += 1
    else:
        lib.gyro_shell_histogram_state24(
            ctypes.c_void_p(flat.data_ptr()),
            ctypes.c_int64(flat.numel()),
            ctypes.c_void_p(out.data_ptr()),
        )

    return out


def shell_histogram_omega12(omega12: torch.Tensor) -> torch.Tensor:
    """Shell histogram from packed omega12 batch. Returns length-7 int32 tensor.
    Assumes values are valid packed Omega states."""
    lib = _get_lib()
    o = _ensure_cpu_contiguous(omega12, dtype=torch.int32, name="omega12")
    flat = o.reshape(-1)
    out = torch.zeros(7, dtype=torch.int32, device="cpu")

    if lib is None or not hasattr(lib, "gyro_shell_histogram_omega12"):
        from src.api import unpack_omega12
        for i in range(flat.numel()):
            om = unpack_omega12(int(flat[i].item()))
            out[om.shell] += 1
    else:
        lib.gyro_shell_histogram_omega12(
            ctypes.c_void_p(flat.data_ptr()),
            ctypes.c_int64(flat.numel()),
            ctypes.c_void_p(out.data_ptr()),
        )

    return out


def omega_signature_scan(
    bytes_tensor: torch.Tensor | bytes | bytearray | memoryview,
) -> torch.Tensor:
    """Scan byte stream to produce packed omega signatures. Returns int32 tensor of same length."""
    lib = _get_lib()
    if isinstance(bytes_tensor, torch.Tensor):
        x = _ensure_cpu_contiguous(bytes_tensor, dtype=torch.uint8, name="bytes")
        if x.ndim != 1:
            raise ValueError(f"bytes must be 1D for omega_signature_scan, got shape {tuple(x.shape)}")
        flat = x
        shape = x.shape
    else:
        flat = torch.tensor(list(bytes(bytes_tensor)), dtype=torch.uint8, device="cpu")
        shape = (flat.numel(),)

    n = flat.numel()
    out = torch.empty(n, dtype=torch.int32, device="cpu")

    if lib is None or not hasattr(lib, "gyro_omega_signature_scan"):
        from src.api import (
            EPS_A6_BY_BYTE,
            EPS_B6_BY_BYTE,
            MICRO_REF_BY_BYTE,
            OmegaSignature12,
            compose_omega_signatures,
            pack_omega_signature12,
        )

        acc = OmegaSignature12(parity=0, tau_u6=0, tau_v6=0)
        for i in range(n):
            b = int(flat[i].item()) & 0xFF
            sig_b = OmegaSignature12(
                parity=1,
                tau_u6=EPS_A6_BY_BYTE[b],
                tau_v6=MICRO_REF_BY_BYTE[b] ^ EPS_B6_BY_BYTE[b],
            )
            acc = compose_omega_signatures(sig_b, acc)
            out[i] = pack_omega_signature12(acc)
    else:
        lib.gyro_omega_signature_scan(
            ctypes.c_void_p(flat.data_ptr()),
            ctypes.c_int64(n),
            ctypes.c_void_p(out.data_ptr()),
        )

    return out.reshape(shape)


def omega12_scan_from_omega12(
    payload: torch.Tensor | bytes | bytearray | memoryview,
    start_omega12: int,
) -> torch.Tensor:
    """Omega-native continuation scan. Returns int32 tensor of omega12 states."""
    lib = _get_lib()
    if isinstance(payload, torch.Tensor):
        x = _ensure_cpu_contiguous(payload, dtype=torch.uint8, name="payload")
        if x.ndim != 1:
            raise ValueError(f"payload must be 1D for omega12_scan_from_omega12, got shape {tuple(x.shape)}")
        flat_bytes = x
        shape = x.shape
    else:
        flat_bytes = torch.tensor(list(bytes(payload)), dtype=torch.uint8, device="cpu")
        shape = (flat_bytes.numel(),)

    n = flat_bytes.numel()
    out = torch.empty(n, dtype=torch.int32, device="cpu")

    if lib is None or not hasattr(lib, "gyro_omega12_scan_from_omega12"):
        from src.api import unpack_omega12, step_omega12_by_byte, pack_omega12
        s = unpack_omega12(int(start_omega12) & 0xFFF)
        for i in range(n):
            s = step_omega12_by_byte(s, int(flat_bytes[i].item()) & 0xFF)
            out[i] = pack_omega12(s)
    else:
        lib.gyro_omega12_scan_from_omega12(
            ctypes.c_void_p(flat_bytes.data_ptr()),
            ctypes.c_int64(n),
            ctypes.c_int32(int(start_omega12) & 0xFFF),
            ctypes.c_void_p(out.data_ptr()),
        )

    return out.reshape(shape)


def shell_histogram_state24_checked(states: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Shell histogram from state24 with Omega check. Returns (hist, invalid_count)."""
    lib = _get_lib()
    s = _ensure_cpu_contiguous(states, dtype=torch.int32, name="states")
    flat = s.reshape(-1)
    out = torch.zeros(7, dtype=torch.int32, device="cpu")

    if lib is None or not hasattr(lib, "gyro_shell_histogram_state24_checked"):
        from src.api import try_state24_to_omega12
        invalid = 0
        for i in range(flat.numel()):
            omega = try_state24_to_omega12(int(flat[i].item()))
            if omega is None:
                invalid += 1
            else:
                out[omega.shell] += 1
    else:
        invalid = lib.gyro_shell_histogram_state24_checked(
            ctypes.c_void_p(flat.data_ptr()),
            ctypes.c_int64(flat.numel()),
            ctypes.c_void_p(out.data_ptr()),
        )
        invalid = int(invalid)
        if invalid < 0:
            raise RuntimeError("gyro_shell_histogram_state24_checked failed")

    return out, invalid


def apply_omega_gate_batch(
    omega12: torch.Tensor,
    gate_code: int,
) -> torch.Tensor:
    """Apply K4 gate to packed omega12 batch. gate_code: 0=id, 1=S, 2=C, 3=F."""
    g = int(gate_code)
    if g not in (0, 1, 2, 3):
        raise ValueError(f"gate_code must be 0,1,2,3; got {gate_code}")

    lib = _get_lib()
    o = _ensure_cpu_contiguous(omega12, dtype=torch.int32, name="omega12")
    flat = o.reshape(-1)
    out = torch.empty_like(flat, dtype=torch.int32)

    if lib is None or not hasattr(lib, "gyro_apply_omega_gate_batch"):
        from src.api import unpack_omega12, apply_omega_gate, pack_omega12
        gate_names = ("id", "S", "C", "F")
        for i in range(flat.numel()):
            om = unpack_omega12(int(flat[i].item()))
            out[i] = pack_omega12(apply_omega_gate(om, gate_names[g]))
    else:
        lib.gyro_apply_omega_gate_batch(
            ctypes.c_void_p(flat.data_ptr()),
            ctypes.c_int64(flat.numel()),
            ctypes.c_uint8(g),
            ctypes.c_void_p(out.data_ptr()),
        )

    return out.reshape(o.shape)


def state_scan_from_state(
    payload: torch.Tensor | bytes | bytearray | memoryview,
    start_state24: int,
) -> torch.Tensor:
    lib = _get_lib()
    if isinstance(payload, torch.Tensor):
        x = _ensure_cpu_contiguous(payload, dtype=torch.uint8, name="payload")
        if x.ndim != 1:
            raise ValueError(f"payload tensor must be 1D for state_scan_from_state, got shape {tuple(x.shape)}")
        flat_bytes = x
        shape = x.shape
    else:
        flat_bytes = torch.tensor(list(bytes(payload)), dtype=torch.uint8, device="cpu")
        shape = (flat_bytes.numel(),)

    n = flat_bytes.numel()
    out = torch.empty(n, dtype=torch.int32, device="cpu")

    if lib is None or not hasattr(lib, "gyro_state_scan_from_state"):
        s = int(start_state24)
        for i in range(n):
            s = step_state_by_byte(s, int(flat_bytes[i].item()) & 0xFF)
            out[i] = s
    else:
        lib.gyro_state_scan_from_state(
            ctypes.c_void_p(flat_bytes.data_ptr()),
            ctypes.c_int64(n),
            ctypes.c_int32(int(start_state24)),
            ctypes.c_void_p(out.data_ptr()),
        )

    return out.reshape(shape)


def signatures_to_states(signatures: torch.Tensor) -> torch.Tensor:
    sig = _ensure_cpu_contiguous(signatures, dtype=torch.int32, name="signatures").to(torch.int64)
    parity = torch.bitwise_and(torch.bitwise_right_shift(sig, 24), 1)
    tau_a12 = torch.bitwise_and(torch.bitwise_right_shift(sig, 12), 0xFFF)
    tau_b12 = torch.bitwise_and(sig, 0xFFF)

    a_even = tau_a12 ^ GENE_MAC_A12
    b_even = tau_b12 ^ GENE_MAC_B12
    a_odd = tau_a12 ^ GENE_MAC_B12
    b_odd = tau_b12 ^ GENE_MAC_A12

    a12 = torch.where(parity == 0, a_even, a_odd)
    b12 = torch.where(parity == 0, b_even, b_odd)
    state24 = torch.bitwise_or(
        torch.bitwise_left_shift(torch.bitwise_and(a12, 0xFFF), 12),
        torch.bitwise_and(b12, 0xFFF),
    )
    return state24.to(torch.int32)


def chirality_states_from_bytes(bytes_tensor: torch.Tensor) -> torch.Tensor:
    return signatures_to_states(signature_scan(bytes_tensor))


def native_available() -> bool:
    """True if the GyroLabe C library is loaded."""
    return _get_lib() is not None


def build_native() -> Path:
    """Build the GyroLabe C shared library. Returns path to the built library."""
    return build_gyrolabe_native()


def initialize_native() -> None:
    """
    Initialize GyroLabe native tables once per process.
    Safe to call multiple times; a no-op if the C library is unavailable.
    """
    lib = _get_lib()
    if lib is not None and hasattr(lib, "gyro_init"):
        lib.gyro_init()


def extract_scan(bytes_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused: (q_class, family, micro_ref, signatures, states) in one pass."""
    return _extract_scan_impl(bytes_tensor)


def chirality_distance_adjacent(states: torch.Tensor, lookahead: int = 1) -> torch.Tensor:
    """Chirality distance between states[i] and states[i+lookahead]."""
    return _chirality_distance_adjacent_impl(states, lookahead)


def _py_popcount64(x: int) -> int:
    return (x & 0xFFFFFFFFFFFFFFFF).bit_count()


def _py_bitplane_gemv(W: torch.Tensor, x: torch.Tensor, n_bits: int = 16) -> torch.Tensor:
    """
    Python reference: y = W @ x via bitplane AND + POPCNT.
    W: [rows, cols], x: [cols], cols <= 64.
    """
    W = _ensure_cpu_contiguous(W, dtype=torch.float32, name="W")
    x = _ensure_cpu_contiguous(x, dtype=torch.float32, name="x")
    if W.dim() != 2 or x.dim() != 1 or W.shape[1] != x.shape[0]:
        raise ValueError(f"Expected W [rows, cols] and x [cols], got {W.shape}, {x.shape}")
    rows, cols = W.shape
    if cols > 64:
        raise ValueError(f"cols must be <= 64, got {cols}")

    scale_max = (1 << (n_bits - 1)) - 1
    max_abs = max(W.abs().max().item(), x.abs().max().item(), 1e-12)
    scale = scale_max / max_abs
    W_int = torch.round(W * scale).to(torch.int32)
    x_int = torch.round(x * scale).to(torch.int32)
    W_mag = W_int.abs().to(torch.int64)
    W_sign = (W_int < 0).to(torch.int64)
    x_mag = x_int.abs().to(torch.int64)
    x_sign = (x_int < 0).to(torch.int64)

    y_out = torch.zeros(rows, dtype=torch.float32, device="cpu")
    for i in range(rows):
        pos_mask = 0
        neg_mask = 0
        for j in range(cols):
            xor_sign = int(W_sign[i, j].item()) ^ int(x_sign[j].item())
            if xor_sign == 0:
                pos_mask |= 1 << j
            else:
                neg_mask |= 1 << j

        W_bp: list[int] = []
        for m in range(n_bits):
            acc = 0
            for j in range(cols):
                bit = (int(W_mag[i, j].item()) >> m) & 1
                acc |= bit << j
            W_bp.append(acc)

        x_bp: list[int] = []
        for k in range(n_bits):
            acc = 0
            for j in range(cols):
                bit = (int(x_mag[j].item()) >> k) & 1
                acc |= bit << j
            x_bp.append(acc)

        pos_dot = 0
        neg_dot = 0
        for m in range(n_bits):
            for k in range(n_bits):
                pm = W_bp[m] & x_bp[k]
                partial_pos = _py_popcount64(pm & pos_mask)
                partial_neg = _py_popcount64(pm & neg_mask)
                shift = 1 << (m + k)
                pos_dot += partial_pos * shift
                neg_dot += partial_neg * shift

        signed_dot = pos_dot - neg_dot
        y_out[i] = signed_dot / (scale * scale)

    return y_out


def bitplane_gemv(W: torch.Tensor, x: torch.Tensor, n_bits: int = 16) -> torch.Tensor:
    """
    Fixed-point GEMV: y = W @ x via bitplane AND+POPCNT decomposition.
    Uses fixed-point quantization internally. Exact over the chosen fixed-point
    representation, not IEEE-754 exact.

    W: [rows, cols] float32, cols <= 64
    x: [cols] float32
    Returns: [rows] float32

    Uses C implementation if available, otherwise Python fallback.
    """
    W = _ensure_cpu_contiguous(W, dtype=torch.float32, name="W")
    x = _ensure_cpu_contiguous(x, dtype=torch.float32, name="x")
    if W.dim() != 2 or x.dim() != 1 or W.shape[1] != x.shape[0]:
        raise ValueError(f"Expected W [rows, cols] and x [cols], got {W.shape}, {x.shape}")
    rows, cols = int(W.shape[0]), int(W.shape[1])
    if cols > 64:
        raise ValueError(f"cols must be <= 64, got {cols}")

    lib = _get_lib()
    if lib is not None and hasattr(lib, "gyro_bitplane_gemv_f32"):
        y = torch.empty(rows, dtype=torch.float32, device="cpu")
        lib.gyro_bitplane_gemv_f32(
            ctypes.c_void_p(W.data_ptr()),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_int64(rows),
            ctypes.c_int64(cols),
            ctypes.c_int32(n_bits),
            ctypes.c_void_p(y.data_ptr()),
        )
        return y
    return _py_bitplane_gemv(W, x, n_bits)


class PackedBitplaneMatrix64:
    """
    Packed fixed-point matrix: pack W once, gemv many times.
    Uses fixed-point quantization internally. Exact over the chosen fixed-point
    representation, not IEEE-754 exact.
    W: [rows, cols] float32, cols <= 64.
    """

    def __init__(self, W: torch.Tensor, n_bits: int = 16) -> None:
        W = _ensure_cpu_contiguous(W, dtype=torch.float32, name="W")
        if W.dim() != 2 or W.shape[1] > 64:
            raise ValueError(f"W must be [rows, cols] with cols <= 64, got {W.shape}")
        self.rows = int(W.shape[0])
        self.cols = int(W.shape[1])
        self.n_bits = n_bits

        W_sign = torch.empty(self.rows, dtype=torch.uint64, device="cpu")
        W_bp = torch.empty((self.rows, n_bits), dtype=torch.uint64, device="cpu")
        scale_w = torch.empty((), dtype=torch.float32, device="cpu")

        lib = _get_lib()
        if lib is None or not hasattr(lib, "gyro_pack_bitplane_matrix_f32"):
            raise RuntimeError(
                "PackedBitplaneMatrix64 requires GyroLabe C library with gyro_pack_bitplane_matrix_f32"
            )
        lib.gyro_pack_bitplane_matrix_f32(
            ctypes.c_void_p(W.data_ptr()),
            ctypes.c_int64(self.rows),
            ctypes.c_int64(self.cols),
            ctypes.c_int32(n_bits),
            ctypes.c_void_p(scale_w.data_ptr()),
            ctypes.c_void_p(W_sign.data_ptr()),
            ctypes.c_void_p(W_bp.data_ptr()),
        )
        self.scale_w = float(scale_w.item())
        self._W_sign = W_sign
        self._W_bp = W_bp

    def gemv(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_cpu_contiguous(x, dtype=torch.float32, name="x")
        if x.dim() != 1 or x.shape[0] != self.cols:
            raise ValueError(f"x must be [cols] with cols={self.cols}, got {x.shape}")
        y = torch.empty(self.rows, dtype=torch.float32, device="cpu")
        lib = _get_lib()
        if lib is None or not hasattr(lib, "gyro_bitplane_gemv_packed_f32"):
            raise RuntimeError(
                "PackedBitplaneMatrix64.gemv requires gyro_bitplane_gemv_packed_f32"
            )
        lib.gyro_bitplane_gemv_packed_f32(
            ctypes.c_void_p(self._W_sign.data_ptr()),
            ctypes.c_void_p(self._W_bp.data_ptr()),
            ctypes.c_float(self.scale_w),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_int64(self.rows),
            ctypes.c_int64(self.cols),
            ctypes.c_int32(self.n_bits),
            ctypes.c_void_p(y.data_ptr()),
        )
        return y

    def gemm_packed_batch(self, X: torch.Tensor) -> torch.Tensor:
        """
        Batched GEMM: Y[b] = W @ X[b] for each batch item.
        X: [batch, cols] float32. Returns Y: [batch, rows] float32.
        """
        X = _ensure_cpu_contiguous(X, dtype=torch.float32, name="X")
        if X.dim() != 2 or X.shape[1] != self.cols:
            raise ValueError(
                f"X must be [batch, cols] with cols={self.cols}, got {X.shape}"
            )
        batch = int(X.shape[0])
        scale_x = torch.empty(batch, dtype=torch.float32, device="cpu")
        X_sign = torch.empty(batch, dtype=torch.uint64, device="cpu")
        X_bp = torch.empty(batch, self.n_bits, dtype=torch.uint64, device="cpu")
        Y = torch.empty(batch, self.rows, dtype=torch.float32, device="cpu")

        lib = _get_lib()
        if lib is None or not hasattr(lib, "gyro_pack_bitplane_vector_batch_f32"):
            raise RuntimeError(
                "PackedBitplaneMatrix64.gemm_packed_batch requires "
                "gyro_pack_bitplane_vector_batch_f32"
            )
        lib.gyro_pack_bitplane_vector_batch_f32(
            ctypes.c_void_p(X.data_ptr()),
            ctypes.c_int64(batch),
            ctypes.c_int64(self.cols),
            ctypes.c_int32(self.n_bits),
            ctypes.c_void_p(scale_x.data_ptr()),
            ctypes.c_void_p(X_sign.data_ptr()),
            ctypes.c_void_p(X_bp.data_ptr()),
        )

        if not hasattr(lib, "gyro_bitplane_gemm_packed_x_batch_f32"):
            raise RuntimeError(
                "PackedBitplaneMatrix64.gemm_packed_batch requires "
                "gyro_bitplane_gemm_packed_x_batch_f32"
            )
        lib.gyro_bitplane_gemm_packed_x_batch_f32(
            ctypes.c_void_p(self._W_sign.data_ptr()),
            ctypes.c_void_p(self._W_bp.data_ptr()),
            ctypes.c_float(self.scale_w),
            ctypes.c_void_p(scale_x.data_ptr()),
            ctypes.c_void_p(X_sign.data_ptr()),
            ctypes.c_void_p(X_bp.data_ptr()),
            ctypes.c_int64(self.rows),
            ctypes.c_int64(self.cols),
            ctypes.c_int64(batch),
            ctypes.c_int32(self.n_bits),
            ctypes.c_void_p(Y.data_ptr()),
        )
        return Y

    def gemv_packed(self, x: "PackedBitplaneVector64") -> torch.Tensor:
        """GEMV using pre-packed input vector. Packed-to-packed is the canonical high-throughput path."""
        if self.cols != x.cols:
            raise ValueError(
                f"Packed vector cols={x.cols} does not match matrix cols={self.cols}"
            )
        if self.n_bits != x.n_bits:
            raise ValueError(
                f"n_bits mismatch: matrix={self.n_bits}, vector={x.n_bits}"
            )

        y = torch.empty(self.rows, dtype=torch.float32, device="cpu")
        lib = _get_lib()
        if lib is None or not hasattr(lib, "gyro_bitplane_gemv_packed_x_f32"):
            raise RuntimeError(
                "PackedBitplaneMatrix64.gemv_packed requires gyro_bitplane_gemv_packed_x_f32"
            )

        lib.gyro_bitplane_gemv_packed_x_f32(
            ctypes.c_void_p(self._W_sign.data_ptr()),
            ctypes.c_void_p(self._W_bp.data_ptr()),
            ctypes.c_float(self.scale_w),
            ctypes.c_uint64(int(x._x_sign.item())),
            ctypes.c_void_p(x._x_bp.data_ptr()),
            ctypes.c_float(x.scale_x),
            ctypes.c_int64(self.rows),
            ctypes.c_int64(self.cols),
            ctypes.c_int32(self.n_bits),
            ctypes.c_void_p(y.data_ptr()),
        )
        return y


def pack_vector_batch64(
    x_batch: torch.Tensor, n_bits: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pack a batch of float32 vectors into fixed-point bitplanes for batched GEMM.
    Returns:
      scale_x [batch] float32
      X_sign  [batch] uint64
      X_bp    [batch, n_bits] uint64
    """
    x_batch = _ensure_cpu_contiguous(x_batch, dtype=torch.float32, name="x_batch")
    if x_batch.dim() != 2 or x_batch.shape[1] > 64:
        raise ValueError(
            f"x_batch must be [batch, cols] with cols <= 64, got {x_batch.shape}"
        )
    batch = int(x_batch.shape[0])
    cols = int(x_batch.shape[1])
    scale_x = torch.empty(batch, dtype=torch.float32, device="cpu")
    X_sign = torch.empty(batch, dtype=torch.uint64, device="cpu")
    X_bp = torch.empty(batch, n_bits, dtype=torch.uint64, device="cpu")
    lib = _get_lib()
    if lib is None or not hasattr(lib, "gyro_pack_bitplane_vector_batch_f32"):
        raise RuntimeError(
            "pack_vector_batch64 requires gyro_pack_bitplane_vector_batch_f32"
        )
    lib.gyro_pack_bitplane_vector_batch_f32(
        ctypes.c_void_p(x_batch.data_ptr()),
        ctypes.c_int64(batch),
        ctypes.c_int64(cols),
        ctypes.c_int32(n_bits),
        ctypes.c_void_p(scale_x.data_ptr()),
        ctypes.c_void_p(X_sign.data_ptr()),
        ctypes.c_void_p(X_bp.data_ptr()),
    )
    return scale_x, X_sign, X_bp


class PackedBitplaneVector64:
    """
    Packed fixed-point vector for repeated internal multiplication.
    Uses fixed-point quantization internally. Exact over the chosen fixed-point
    representation, not IEEE-754 exact.

    Vectors of length <= 64.
    """

    def __init__(self, x: torch.Tensor, n_bits: int = 16) -> None:
        x = _ensure_cpu_contiguous(x, dtype=torch.float32, name="x")
        if x.dim() != 1 or x.shape[0] > 64:
            raise ValueError(f"x must be [cols] with cols <= 64, got {x.shape}")
        self.cols = int(x.shape[0])
        self.n_bits = n_bits

        scale_x = torch.empty((), dtype=torch.float32, device="cpu")
        x_sign = torch.empty((), dtype=torch.uint64, device="cpu")
        x_bp = torch.empty((n_bits,), dtype=torch.uint64, device="cpu")

        lib = _get_lib()
        if lib is None or not hasattr(lib, "gyro_pack_bitplane_vector_f32"):
            raise RuntimeError(
                "PackedBitplaneVector64 requires gyro_pack_bitplane_vector_f32"
            )

        lib.gyro_pack_bitplane_vector_f32(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_int64(self.cols),
            ctypes.c_int32(n_bits),
            ctypes.c_void_p(scale_x.data_ptr()),
            ctypes.c_void_p(x_sign.data_ptr()),
            ctypes.c_void_p(x_bp.data_ptr()),
        )

        self.scale_x = float(scale_x.item())
        self._x_sign = x_sign
        self._x_bp = x_bp


class PackedBitplaneMatrix64I32:
    """
    Integer-native packed matrix. No quantization, no scale.
    Exact internal multiplication over int32 inputs, int64 outputs.
    """

    def __init__(self, W: torch.Tensor, n_bits: int = 16) -> None:
        W = _ensure_cpu_contiguous(W, dtype=torch.int32, name="W")
        if W.dim() != 2 or W.shape[1] > 64:
            raise ValueError(f"W must be [rows, cols] with cols <= 64, got {W.shape}")
        self.rows = int(W.shape[0])
        self.cols = int(W.shape[1])
        self.n_bits = n_bits

        W_sign = torch.empty(self.rows, dtype=torch.uint64, device="cpu")
        W_bp = torch.empty((self.rows, n_bits), dtype=torch.uint64, device="cpu")

        lib = _get_lib()
        if lib is None or not hasattr(lib, "gyro_pack_bitplane_matrix_i32"):
            raise RuntimeError(
                "PackedBitplaneMatrix64I32 requires gyro_pack_bitplane_matrix_i32"
            )
        lib.gyro_pack_bitplane_matrix_i32(
            ctypes.c_void_p(W.data_ptr()),
            ctypes.c_int64(self.rows),
            ctypes.c_int64(self.cols),
            ctypes.c_int32(n_bits),
            ctypes.c_void_p(W_sign.data_ptr()),
            ctypes.c_void_p(W_bp.data_ptr()),
        )
        self._W_sign = W_sign
        self._W_bp = W_bp

    def gemv_packed(self, x: "PackedBitplaneVector64I32") -> torch.Tensor:
        """Exact int64 output. No scaling."""
        if self.cols != x.cols or self.n_bits != x.n_bits:
            raise ValueError("cols and n_bits must match")
        y = torch.empty(self.rows, dtype=torch.int64, device="cpu")
        lib = _get_lib()
        if lib is None or not hasattr(lib, "gyro_bitplane_gemv_packed_i32"):
            raise RuntimeError(
                "PackedBitplaneMatrix64I32.gemv_packed requires gyro_bitplane_gemv_packed_i32"
            )
        lib.gyro_bitplane_gemv_packed_i32(
            ctypes.c_void_p(self._W_sign.data_ptr()),
            ctypes.c_void_p(self._W_bp.data_ptr()),
            ctypes.c_uint64(int(x._x_sign.item())),
            ctypes.c_void_p(x._x_bp.data_ptr()),
            ctypes.c_int64(self.rows),
            ctypes.c_int64(self.cols),
            ctypes.c_int32(self.n_bits),
            ctypes.c_void_p(y.data_ptr()),
        )
        return y


class PackedBitplaneVector64I32:
    """
    Integer-native packed vector. No quantization, no scale.
    Exact internal multiplication over int32 inputs.
    """

    def __init__(self, x: torch.Tensor, n_bits: int = 16) -> None:
        x = _ensure_cpu_contiguous(x, dtype=torch.int32, name="x")
        if x.dim() != 1 or x.shape[0] > 64:
            raise ValueError(f"x must be [cols] with cols <= 64, got {x.shape}")
        self.cols = int(x.shape[0])
        self.n_bits = n_bits

        x_sign = torch.empty((), dtype=torch.uint64, device="cpu")
        x_bp = torch.empty((n_bits,), dtype=torch.uint64, device="cpu")

        lib = _get_lib()
        if lib is None or not hasattr(lib, "gyro_pack_bitplane_vector_i32"):
            raise RuntimeError(
                "PackedBitplaneVector64I32 requires gyro_pack_bitplane_vector_i32"
            )
        lib.gyro_pack_bitplane_vector_i32(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_int64(self.cols),
            ctypes.c_int32(n_bits),
            ctypes.c_void_p(x_sign.data_ptr()),
            ctypes.c_void_p(x_bp.data_ptr()),
        )
        self._x_sign = x_sign
        self._x_bp = x_bp


def packed_gemv_packed_x(
    packed_W: PackedBitplaneMatrix64,
    packed_x: PackedBitplaneVector64,
) -> torch.Tensor:
    """
    GEMV using pre-packed weights and pre-packed input vector.
    Thin wrapper: prefer PackedBitplaneMatrix64.gemv_packed(x).
    """
    return packed_W.gemv_packed(packed_x)