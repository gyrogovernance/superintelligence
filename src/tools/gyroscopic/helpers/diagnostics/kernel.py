"""Offline kernel ctypes bindings (not used in llama.cpp inference)."""

from __future__ import annotations

import ctypes

from src.tools.gyroscopic.ops import _lib


class GyroAccum(ctypes.Structure):
    _fields_ = [("a", ctypes.c_float), ("b", ctypes.c_float)]


gyro_accum_t = GyroAccum

_RESEARCH_BOUND = False


def _bind_research(lib: ctypes.CDLL) -> None:
    global _RESEARCH_BOUND
    if _RESEARCH_BOUND:
        return
    u8 = ctypes.c_uint8
    u8p = ctypes.POINTER(ctypes.c_uint8)
    f32 = ctypes.c_float

    lib.gyroscopic_analyze_q1_group.restype = None
    lib.gyroscopic_analyze_q1_group.argtypes = [u8p, u8p, u8p, u8p]

    lib.gyroscopic_extract_phase_native.restype = None
    lib.gyroscopic_extract_phase_native.argtypes = [u8p, u8p, u8p]

    lib.gyroscopic_k4_compose_gyroacc.restype = f32
    lib.gyroscopic_k4_compose_gyroacc.argtypes = [ctypes.POINTER(gyro_accum_t), f32]

    lib.gyroscopic_depth4_bu_factor.restype = f32
    lib.gyroscopic_depth4_bu_factor.argtypes = []

    lib.gyroscopic_route_path.restype = u8
    lib.gyroscopic_route_path.argtypes = [u8, u8]
    _RESEARCH_BOUND = True


def _native() -> ctypes.CDLL:
    lib = _lib()
    _bind_research(lib)
    return lib


def analyze_q1_group(signs16: bytes) -> tuple[int, int, int]:
    """Return (q_class, shell, k4_char) via WHT chirality."""
    if len(signs16) != 16:
        raise ValueError("signs16 must be exactly 16 bytes")
    buf = (ctypes.c_uint8 * 16)(*signs16)
    q = ctypes.c_uint8()
    sh = ctypes.c_uint8()
    k4 = ctypes.c_uint8()
    _native().gyroscopic_analyze_q1_group(buf, ctypes.byref(q), ctypes.byref(sh), ctypes.byref(k4))
    return int(q.value), int(sh.value), int(k4.value)


def extract_phase_native(signs16: bytes) -> tuple[int, int]:
    """O(1) (k4_char, shell_proxy) from 16 bytes of Q1_0 sign bits."""
    if len(signs16) != 16:
        raise ValueError("signs16 must be exactly 16 bytes")
    buf = (ctypes.c_uint8 * 16)(*signs16)
    k4 = ctypes.c_uint8()
    proxy = ctypes.c_uint8()
    _native().gyroscopic_extract_phase_native(buf, ctypes.byref(k4), ctypes.byref(proxy))
    return int(k4.value), int(proxy.value)


def k4_compose_gyroacc(sectors: list[tuple[float, float]], gravity: float = 1.0) -> float:
    """Compose four K4-sector gyrophase pairs via gate actions."""
    if len(sectors) != 4:
        raise ValueError("sectors must be four (a, b) pairs for CS, UNA, ONA, BU")
    arr = (GyroAccum * 4)(*(GyroAccum(a, b) for a, b in sectors))
    return float(_native().gyroscopic_k4_compose_gyroacc(arr, float(gravity)))


def depth4_bu_factor() -> float:
    """Depth-4 factor (1 - 4 rho Delta^2 + c4 Delta^4)."""
    return float(_native().gyroscopic_depth4_bu_factor())


def route_path(shell: int, k4_char: int) -> int:
    """Structural routing class from (shell, k4_char); magnitude-neutral."""
    return int(_native().gyroscopic_route_path(shell & 0xFF, k4_char & 0xFF))
