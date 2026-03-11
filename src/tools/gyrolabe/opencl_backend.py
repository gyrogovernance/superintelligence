# src/tools/gyrolabe/opencl_backend.py
"""
OpenCL backend for GyroLabe packed tensor GEMM.
Accelerates only: packed matrix x packed batch of vectors -> output batch.
CPU keeps: Moments, signatures, q-map, control-plane.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.tools.gyrolabe.ops import PackedBitplaneMatrix64, PackedBitplaneMatrix64I32

_THIS_FILE = Path(__file__).resolve()
_GYROLABE_DIR = _THIS_FILE.parent
_REPO_ROOT = _THIS_FILE.parents[3]
_OPENCL_CSRC = _GYROLABE_DIR / "gyrolabe_opencl.c"
_BUILD_DIR = _GYROLABE_DIR / "_build"

_CL_LIB: object | None = None


def _shared_lib_suffix() -> str:
    if sys.platform.startswith("win"):
        return ".dll"
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


def _hash_opencl_source() -> str:
    h = hashlib.sha256()
    h.update(_OPENCL_CSRC.read_bytes())
    h.update(platform.platform().encode("utf-8"))
    return h.hexdigest()[:16]


def _find_opencl_include() -> list[str]:
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


def _find_opencl_lib() -> list[str]:
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
    if sys.platform.startswith("win"):
        return ["-lOpenCL"]
    return ["-lOpenCL"]


def _find_compiler() -> tuple[str, str]:
    if os.name == "nt":
        gcc = shutil.which("gcc")
        if gcc:
            return gcc, "cc"
        for sub in ("ucrt64", "mingw64"):
            p = Path("C:/msys64") / sub / "bin" / "gcc.exe"
            if p.exists():
                return str(p), "cc"
        cl = shutil.which("cl")
        if cl:
            return cl, "msvc"
    else:
        for name in ("cc", "clang", "gcc"):
            path = shutil.which(name)
            if path:
                return path, "cc"
    raise RuntimeError(
        "No C compiler found for OpenCL backend. Install GCC or MSVC."
    )


def _build_opencl_lib() -> Path | None:
    if not _OPENCL_CSRC.exists():
        return None
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"gyrolabe_opencl_{_hash_opencl_source()}{_shared_lib_suffix()}"
    out_path = _BUILD_DIR / out_name
    if out_path.exists():
        return out_path

    compiler, kind = _find_compiler()
    inc_flags = _find_opencl_include()
    lib_flags = _find_opencl_lib()

    run_cwd = str(_BUILD_DIR)
    run_env = None
    rel_src = _OPENCL_CSRC.relative_to(_REPO_ROOT)
    rel_out = out_path.relative_to(_REPO_ROOT)

    if kind == "msvc":
        cmd = [compiler, "/nologo", "/O2", "/LD", "/TC", str(_OPENCL_CSRC), f"/Fe:{out_path}"]
        for f in inc_flags:
            if f.startswith("-I"):
                cmd.insert(-2, f"/I{f[2:]}")
        cmd.append("OpenCL.lib")
    else:
        flags = ["-O3", "-std=c11", "-shared"] + inc_flags
        if not sys.platform.startswith("win"):
            flags.append("-fPIC")
        cmd = [compiler, *flags, "-o", str(out_path), str(_OPENCL_CSRC)] + lib_flags
        if os.name == "nt" and kind == "cc":
            run_cwd = str(_REPO_ROOT)
            comp_dir = str(Path(compiler).resolve().parent)
            run_env = {**os.environ, "PATH": comp_dir + os.pathsep + os.environ.get("PATH", "")}
            cmd = [compiler, *flags, "-o", str(rel_out), str(rel_src)] + lib_flags

    result = subprocess.run(
        cmd, cwd=run_cwd, env=run_env, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            "OpenCL backend build failed\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"stdout:\n{result.stdout}\n\n"
            f"stderr:\n{result.stderr}"
        )
    return out_path if out_path.exists() else None


def _get_cl_lib():
    global _CL_LIB
    if _CL_LIB is not None:
        return _CL_LIB
    path = _build_opencl_lib()
    if path is None:
        return None
    try:
        if os.name == "nt":
            os.add_dll_directory(r"C:\msys64\ucrt64\bin")
        _CL_LIB = ctypes.CDLL(str(path))
    except OSError:
        return None

    _CL_LIB.gyro_cl_available.restype = ctypes.c_int
    _CL_LIB.gyro_cl_available.argtypes = []

    _CL_LIB.gyro_cl_init.restype = ctypes.c_int
    _CL_LIB.gyro_cl_init.argtypes = [ctypes.c_int, ctypes.c_int]

    _CL_LIB.gyro_cl_shutdown.restype = None
    _CL_LIB.gyro_cl_shutdown.argtypes = []

    _CL_LIB.gyro_cl_create_packed_matrix_f32.restype = ctypes.c_uint64
    _CL_LIB.gyro_cl_create_packed_matrix_f32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int32,
    ]

    # Integer-native exact path
    if hasattr(_CL_LIB, "gyro_cl_create_packed_matrix_i32"):
        _CL_LIB.gyro_cl_create_packed_matrix_i32.restype = ctypes.c_uint64
        _CL_LIB.gyro_cl_create_packed_matrix_i32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int32,
        ]
    if hasattr(_CL_LIB, "gyro_cl_gemm_packed_x_batch_i32"):
        _CL_LIB.gyro_cl_gemm_packed_x_batch_i32.restype = ctypes.c_int
        _CL_LIB.gyro_cl_gemm_packed_x_batch_i32.argtypes = [
            ctypes.c_uint64,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_void_p,
        ]

    _CL_LIB.gyro_cl_release_packed_matrix.restype = None
    _CL_LIB.gyro_cl_release_packed_matrix.argtypes = [ctypes.c_uint64]

    _CL_LIB.gyro_cl_gemm_packed_x_batch_f32.restype = ctypes.c_int
    _CL_LIB.gyro_cl_gemm_packed_x_batch_f32.argtypes = [
        ctypes.c_uint64,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_void_p,
    ]
    return _CL_LIB


def available() -> bool:
    """Return True if OpenCL runtime is present and at least one device exists."""
    lib = _get_cl_lib()
    if lib is None:
        return False
    return lib.gyro_cl_available() == 1


def initialize(platform_index: int = 0, device_index: int = 0) -> None:
    """Create OpenCL context, queue, and compile kernels."""
    lib = _get_cl_lib()
    if lib is None:
        raise RuntimeError("OpenCL backend not built. Check compiler and OpenCL SDK.")
    if lib.gyro_cl_init(platform_index, device_index) != 1:
        raise RuntimeError(
            f"gyro_cl_init({platform_index}, {device_index}) failed. "
            "Check OpenCL runtime and device availability."
        )


def shutdown() -> None:
    """Release all OpenCL resources."""
    lib = _get_cl_lib()
    if lib:
        lib.gyro_cl_shutdown()


class OpenCLBatchWorkspace64:
    """
    Reusable OpenCL batch workspace.
    Owns no resources directly in Python, but provides a handle for
    future extensions such as double buffering and autotuning.
    """

    def __init__(self, max_batch: int, n_bits: int) -> None:
        if max_batch <= 0:
            raise ValueError("max_batch must be positive")
        if n_bits not in (8, 12, 16):
            raise ValueError("n_bits must be one of 8, 12, 16")
        self.max_batch = int(max_batch)
        self.n_bits = int(n_bits)


class OpenCLPackedMatrix64:
    """
    GPU-resident packed matrix for batched GEMM.
    Wraps a CPU-packed matrix and uploads it to the GPU.
    """

    def __init__(self, packed_cpu_matrix: "PackedBitplaneMatrix64") -> None:
        lib = _get_cl_lib()
        if lib is None:
            raise RuntimeError("OpenCL backend not built.")
        if not lib.gyro_cl_available():
            raise RuntimeError("OpenCL runtime not available.")

        self._packed = packed_cpu_matrix
        self._handle = lib.gyro_cl_create_packed_matrix_f32(
            ctypes.c_void_p(packed_cpu_matrix._W_sign.data_ptr()),
            ctypes.c_void_p(packed_cpu_matrix._W_bp.data_ptr()),
            ctypes.c_float(packed_cpu_matrix.scale_w),
            ctypes.c_int64(packed_cpu_matrix.rows),
            ctypes.c_int64(packed_cpu_matrix.cols),
            ctypes.c_int32(packed_cpu_matrix.n_bits),
        )
        if self._handle == 0:
            raise RuntimeError("gyro_cl_create_packed_matrix_f32 failed.")

    def gemm_packed_batch(
        self,
        X: torch.Tensor,
        workspace: OpenCLBatchWorkspace64 | None = None,
    ) -> torch.Tensor:
        """
        Batched GEMM: Y[b] = W @ X[b].
        X: [batch, cols] float32. Returns Y: [batch, rows] float32 on CPU.
        """
        from src.tools.gyrolabe import ops

        X = ops._ensure_cpu_contiguous(X, dtype=torch.float32, name="X")
        if X.dim() != 2 or X.shape[1] != self._packed.cols:
            raise ValueError(
                f"X must be [batch, cols] with cols={self._packed.cols}, got {X.shape}"
            )
        batch = int(X.shape[0])
        if workspace is not None and batch > workspace.max_batch:
            raise ValueError(
                f"workspace max_batch={workspace.max_batch} is smaller than batch={batch}"
            )
        scale_x, X_sign, X_bp = ops.pack_vector_batch64(X, n_bits=self._packed.n_bits)
        Y = torch.empty(batch, self._packed.rows, dtype=torch.float32, device="cpu")

        lib = _get_cl_lib()
        if lib is None:
            raise RuntimeError("OpenCL backend not loaded.")
        ok = lib.gyro_cl_gemm_packed_x_batch_f32(
            ctypes.c_uint64(self._handle),
            ctypes.c_void_p(scale_x.data_ptr()),
            ctypes.c_void_p(X_sign.data_ptr()),
            ctypes.c_void_p(X_bp.data_ptr()),
            ctypes.c_int64(batch),
            ctypes.c_void_p(Y.data_ptr()),
        )
        if ok != 1:
            raise RuntimeError("gyro_cl_gemm_packed_x_batch_f32 failed.")
        return Y

    def close(self) -> None:
        """Release GPU matrix handle."""
        if getattr(self, "_handle", 0) == 0:
            return
        lib = _get_cl_lib()
        if lib:
            lib.gyro_cl_release_packed_matrix(self._handle)
        self._handle = 0


class OpenCLPackedMatrix64I32:
    """
    GPU-resident integer-native packed matrix.
    Mirrors PackedBitplaneMatrix64I32 but runs GEMM on OpenCL.
    """

    def __init__(self, packed_cpu_matrix: "PackedBitplaneMatrix64I32") -> None:
        from src.tools.gyrolabe import ops

        lib = _get_cl_lib()
        if lib is None:
            raise RuntimeError("OpenCL backend not built.")
        if not lib.gyro_cl_available():
            raise RuntimeError("OpenCL runtime not available.")
        if not hasattr(lib, "gyro_cl_create_packed_matrix_i32"):
            raise RuntimeError("OpenCL library does not expose gyro_cl_create_packed_matrix_i32")

        if not isinstance(packed_cpu_matrix, ops.PackedBitplaneMatrix64I32):
            raise TypeError("packed_cpu_matrix must be PackedBitplaneMatrix64I32")

        self._packed = packed_cpu_matrix
        self._handle = lib.gyro_cl_create_packed_matrix_i32(
            ctypes.c_void_p(packed_cpu_matrix._W_sign.data_ptr()),
            ctypes.c_void_p(packed_cpu_matrix._W_bp.data_ptr()),
            ctypes.c_int64(packed_cpu_matrix.rows),
            ctypes.c_int64(packed_cpu_matrix.cols),
            ctypes.c_int32(packed_cpu_matrix.n_bits),
        )
        if self._handle == 0:
            raise RuntimeError("gyro_cl_create_packed_matrix_i32 failed.")

    def gemm_packed_batch(self, X_bp_sign: torch.Tensor, X_bp_bits: torch.Tensor) -> torch.Tensor:
        """
        Batched exact GEMM on already-packed integer vectors.

        X_bp_sign: [batch] uint64 tensor of sign words.
        X_bp_bits: [batch, n_bits] uint64 tensor of bitplane words.
        Returns int64 tensor Y: [batch, rows] on CPU.
        """
        X_bp_sign = X_bp_sign.contiguous()
        X_bp_bits = X_bp_bits.contiguous()

        if X_bp_sign.dtype != torch.uint64 or X_bp_bits.dtype != torch.uint64:
            raise ValueError("X_bp_sign and X_bp_bits must be uint64")
        if X_bp_bits.dim() != 2 or X_bp_sign.dim() != 1:
            raise ValueError("X_bp_sign must be [batch], X_bp_bits must be [batch, n_bits]")
        if X_bp_bits.shape[0] != X_bp_sign.shape[0]:
            raise ValueError("batch dimension mismatch between X_bp_sign and X_bp_bits")
        if int(X_bp_bits.shape[1]) != self._packed.n_bits:
            raise ValueError("X_bp_bits.shape[1] must equal packed n_bits")

        batch = int(X_bp_sign.shape[0])
        Y = torch.empty(batch, self._packed.rows, dtype=torch.int64, device="cpu")

        lib = _get_cl_lib()
        if lib is None or not hasattr(lib, "gyro_cl_gemm_packed_x_batch_i32"):
            raise RuntimeError("OpenCL library does not expose gyro_cl_gemm_packed_x_batch_i32")

        ok = lib.gyro_cl_gemm_packed_x_batch_i32(
            ctypes.c_uint64(self._handle),
            ctypes.c_void_p(X_bp_sign.data_ptr()),
            ctypes.c_void_p(X_bp_bits.data_ptr()),
            ctypes.c_int64(batch),
            ctypes.c_void_p(Y.data_ptr()),
        )
        if ok != 1:
            raise RuntimeError("gyro_cl_gemm_packed_x_batch_i32 failed.")
        return Y

    def close(self) -> None:
        """Release GPU matrix handle."""
        if self._handle == 0:
            return
        lib = _get_cl_lib()
        if lib:
            lib.gyro_cl_release_packed_matrix(self._handle)
        self._handle = 0
