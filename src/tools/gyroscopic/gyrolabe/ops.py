from __future__ import annotations

import ctypes as ct

from src.tools.gyroscopic.gyrolabe.ops_build import build_gyrolabe_native

_DLL = None


class GyroMatMulRuntimeCaps(ct.Structure):
    _fields_ = [
        ("avx2_enabled", ct.c_uint32),
        ("f16c_enabled", ct.c_uint32),
        ("fma_enabled", ct.c_uint32),
        ("reserved", ct.c_uint32),
    ]


class GyroMatMulBlockQ80(ct.Structure):
    _fields_ = [
        ("d", ct.c_uint16),
        ("qs", ct.c_int8 * 32),
    ]


def _load_dll() -> ct.CDLL:
    global _DLL
    if _DLL is not None:
        return _DLL

    dll_path = build_gyrolabe_native()
    lib = ct.CDLL(str(dll_path))

    lib.gyromatmul_runtime_query.argtypes = [ct.POINTER(GyroMatMulRuntimeCaps)]
    lib.gyromatmul_runtime_query.restype = None

    lib.gyromatmul_vec_dot_f32.argtypes = [
        ct.c_int,
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_float),
    ]
    lib.gyromatmul_vec_dot_f32.restype = ct.c_int

    lib.gyromatmul_vec_dot_q8_0_q8_0.argtypes = [
        ct.c_int,
        ct.POINTER(GyroMatMulBlockQ80),
        ct.POINTER(GyroMatMulBlockQ80),
        ct.POINTER(ct.c_float),
    ]
    lib.gyromatmul_vec_dot_q8_0_q8_0.restype = ct.c_int

    _DLL = lib
    return lib


def gyromatmul_runtime_caps() -> GyroMatMulRuntimeCaps:
    lib = _load_dll()
    caps = GyroMatMulRuntimeCaps()
    lib.gyromatmul_runtime_query(ct.byref(caps))
    return caps
