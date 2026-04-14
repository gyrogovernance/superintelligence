"""ctypes bindings for GyroGraph + GyroLabe native exports (runtime caps, graph stepping, SLCP)."""

from __future__ import annotations

import ctypes as ct
import os
import sys

import numpy as np

from .helpers.weight64_wht import (
    WHT_SIZE,
    wht_64_batch,
    topk_energy_fractions,
    topk_reconstruction_rel_l2,
)
from .ops_build import build_gyrolabe_native, repo_root

_DLL = None


def _require_batch_word4_n(n: int) -> None:
    if int(n) < 1:
        raise ValueError("gyrograph word4 batch n must be >= 1")


def _ctypes_array_len(arr: ct.Array) -> int:
    return int(arr._length_)


def _max_cell_id_zero_indexed(cell_ids: ct.Array, n: int) -> int:
    mx = -1
    for i in range(int(n)):
        cid = int(cell_ids[i])
        if cid < 0:
            raise ValueError(f"cell_ids[{i}] must be non-negative, got {cid}")
        if cid > mx:
            mx = cid
    return mx


def _validate_word4_batch_buffers(
    n: int,
    cell_ids: ct.Array,
    omega12_io: ct.Array,
    step_io: ct.Array,
    last_byte_io: ct.Array,
    has_closed_word_io: ct.Array,
    word4_io: ct.Array,
    chi_ring64_io: ct.Array,
    chi_ring_pos_io: ct.Array,
    chi_valid_len_io: ct.Array,
    chi_hist64_io: ct.Array,
    shell_hist7_io: ct.Array,
    family_ring64_io: ct.Array,
    family_hist4_io: ct.Array,
    omega_sig_io: ct.Array,
    parity_O12_io: ct.Array,
    parity_E12_io: ct.Array,
    parity_bit_io: ct.Array,
    resonance_key_io: ct.Array,
    words4_in: ct.Array,
    omega_trace4_in: ct.Array | None,
    chi_trace4_in: ct.Array | None,
) -> None:
    n = int(n)
    if _ctypes_array_len(cell_ids) < n:
        raise ValueError("cell_ids length must be >= n")
    mx = _max_cell_id_zero_indexed(cell_ids, n)
    nc = mx + 1
    if _ctypes_array_len(words4_in) < 4 * n:
        raise ValueError("words4_in length must be >= 4 * n")
    if omega_trace4_in is not None and _ctypes_array_len(omega_trace4_in) < 4 * n:
        raise ValueError("omega_trace4_in length must be >= 4 * n")
    if chi_trace4_in is not None and _ctypes_array_len(chi_trace4_in) < 4 * n:
        raise ValueError("chi_trace4_in length must be >= 4 * n")

    def need(name: str, arr: ct.Array, min_len: int) -> None:
        ln = _ctypes_array_len(arr)
        if ln < min_len:
            raise ValueError(
                f"{name} length must be >= {min_len} for max cell_id {mx}, got {ln}"
            )

    need("omega12_io", omega12_io, nc)
    need("step_io", step_io, nc)
    need("last_byte_io", last_byte_io, nc)
    need("has_closed_word_io", has_closed_word_io, nc)
    need("word4_io", word4_io, 4 * nc)
    need("chi_ring64_io", chi_ring64_io, 64 * nc)
    need("chi_ring_pos_io", chi_ring_pos_io, nc)
    need("chi_valid_len_io", chi_valid_len_io, nc)
    need("chi_hist64_io", chi_hist64_io, 64 * nc)
    need("shell_hist7_io", shell_hist7_io, 7 * nc)
    need("family_ring64_io", family_ring64_io, 64 * nc)
    need("family_hist4_io", family_hist4_io, 4 * nc)
    need("omega_sig_io", omega_sig_io, nc)
    need("parity_O12_io", parity_O12_io, nc)
    need("parity_E12_io", parity_E12_io, nc)
    need("parity_bit_io", parity_bit_io, nc)
    need("resonance_key_io", resonance_key_io, nc)


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


class GyroGraphMoment(ct.Structure):
    _fields_ = [
        ("step", ct.c_uint64),
        ("state24", ct.c_uint32),
        ("last_byte", ct.c_uint8),
        ("ledger_len", ct.c_uint8),
        ("ledger", ct.c_uint8 * 256),
        ("parity_O12", ct.c_uint16),
        ("parity_E12", ct.c_uint16),
        ("parity_bit", ct.c_uint8),
        ("omega_sig", ct.c_uint32),
        ("q_transport6", ct.c_uint8),
    ]


class GyroGraphSLCP(ct.Structure):
    _fields_ = [
        ("cell_id", ct.c_int64),
        ("step", ct.c_uint64),
        ("omega12", ct.c_int32),
        ("state24", ct.c_uint32),
        ("last_byte", ct.c_uint8),
        ("family", ct.c_uint8),
        ("micro_ref", ct.c_uint8),
        ("q6", ct.c_uint8),
        ("chi6", ct.c_uint8),
        ("shell", ct.c_uint8),
        ("horizon_distance", ct.c_uint16),
        ("ab_distance", ct.c_uint16),
        ("omega_sig", ct.c_int32),
        ("parity_O12", ct.c_uint16),
        ("parity_E12", ct.c_uint16),
        ("parity_bit", ct.c_uint8),
        ("resonance_key", ct.c_uint32),
        ("current_resonance", ct.c_uint32),
        ("spectral64", ct.c_float * 64),
        ("gauge_spectral", ct.c_float * 4),
        ("shell_spectral", ct.c_float * 7),
    ]


class GyroGraphWordSignature(ct.Structure):
    _fields_ = [
        ("parity", ct.c_uint8),
        ("tau_a12", ct.c_uint16),
        ("tau_b12", ct.c_uint16),
    ]


class GyroGraphConstitutional(ct.Structure):
    _fields_ = [
        ("rest_distance", ct.c_uint32),
        ("horizon_distance", ct.c_uint32),
        ("ab_distance", ct.c_uint32),
        ("on_complement_horizon", ct.c_uint8),
        ("on_equality_horizon", ct.c_uint8),
        ("a_density", ct.c_float),
        ("b_density", ct.c_float),
        ("complementarity_sum", ct.c_uint32),
    ]


class GyroLabeOperatorReport(ct.Structure):
    _fields_ = [
        ("op_class", ct.c_int),
        ("scr", ct.c_float),
        ("defect_norm", ct.c_float),
        ("eigenvalues_256", ct.c_float * 256),
        ("eigenvalues_valid", ct.c_uint8),
    ]


def _load_dll() -> ct.CDLL:
    global _DLL
    if _DLL is not None:
        return _DLL

    if sys.platform == "win32":
        llama = repo_root() / "external" / "llama.cpp"
        for sub in ("build/bin/Release", "build/bin/Debug", "build/bin"):
            d = llama / sub.replace("/", os.sep)
            if (d / "ggml.dll").is_file():
                os.add_dll_directory(str(d.resolve()))
                break

    dll_path = build_gyrolabe_native()
    lib = ct.CDLL(str(dll_path))

    lib.gyromatmul_runtime_query.argtypes = [ct.POINTER(GyroMatMulRuntimeCaps)]
    lib.gyromatmul_runtime_query.restype = None

    if hasattr(lib, "gyromatmul_vec_dot_f32"):
        lib.gyromatmul_vec_dot_f32.argtypes = [
            ct.c_int,
            ct.POINTER(ct.c_float),
            ct.POINTER(ct.c_float),
            ct.POINTER(ct.c_float),
        ]
        lib.gyromatmul_vec_dot_f32.restype = ct.c_int

    lib.gyrograph_init.argtypes = []
    lib.gyrograph_init.restype = None

    lib.gyrograph_trace_word4_batch_indexed.argtypes = [
        ct.POINTER(ct.c_int64),
        ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_uint8),
        ct.c_int64,
        ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_uint8),
    ]
    lib.gyrograph_trace_word4_batch_indexed.restype = None

    # Must match gyrograph.h gyrograph_apply_trace_word4_batch_indexed (order + pointee width).
    lib.gyrograph_apply_trace_word4_batch_indexed.argtypes = [
        ct.POINTER(ct.c_int64),  # cell_ids
        ct.POINTER(ct.c_int32),  # omega12_io
        ct.POINTER(ct.c_uint64),  # step_io
        ct.POINTER(ct.c_uint8),  # last_byte_io
        ct.POINTER(ct.c_uint8),  # has_closed_word_io
        ct.POINTER(ct.c_uint8),  # word4_io
        ct.POINTER(ct.c_uint8),  # chi_ring64_io
        ct.POINTER(ct.c_uint8),  # chi_ring_pos_io
        ct.POINTER(ct.c_uint8),  # chi_valid_len_io
        ct.POINTER(ct.c_uint16),  # chi_hist64_io
        ct.POINTER(ct.c_uint16),  # shell_hist7_io
        ct.POINTER(ct.c_uint8),  # family_ring64_io
        ct.POINTER(ct.c_uint16),  # family_hist4_io
        ct.POINTER(ct.c_int32),  # omega_sig_io
        ct.POINTER(ct.c_uint16),  # parity_O12_io
        ct.POINTER(ct.c_uint16),  # parity_E12_io
        ct.POINTER(ct.c_uint8),  # parity_bit_io
        ct.POINTER(ct.c_uint8),  # words4_in
        ct.POINTER(ct.c_int32),  # omega_trace4_in
        ct.POINTER(ct.c_uint8),  # chi_trace4_in
        ct.POINTER(ct.c_uint32),  # resonance_key_io
        ct.c_uint8,  # profile
        ct.c_int64,  # n
    ]
    lib.gyrograph_apply_trace_word4_batch_indexed.restype = None

    # Must match gyrograph.h gyrograph_ingest_word4_batch_indexed.
    lib.gyrograph_ingest_word4_batch_indexed.argtypes = [
        ct.POINTER(ct.c_int64),  # cell_ids
        ct.POINTER(ct.c_int32),  # omega12_io
        ct.POINTER(ct.c_uint64),  # step_io
        ct.POINTER(ct.c_uint8),  # last_byte_io
        ct.POINTER(ct.c_uint8),  # has_closed_word_io
        ct.POINTER(ct.c_uint8),  # word4_io
        ct.POINTER(ct.c_uint8),  # chi_ring64_io
        ct.POINTER(ct.c_uint8),  # chi_ring_pos_io
        ct.POINTER(ct.c_uint8),  # chi_valid_len_io
        ct.POINTER(ct.c_uint16),  # chi_hist64_io
        ct.POINTER(ct.c_uint16),  # shell_hist7_io
        ct.POINTER(ct.c_uint8),  # family_ring64_io
        ct.POINTER(ct.c_uint16),  # family_hist4_io
        ct.POINTER(ct.c_int32),  # omega_sig_io
        ct.POINTER(ct.c_uint16),  # parity_O12_io
        ct.POINTER(ct.c_uint16),  # parity_E12_io
        ct.POINTER(ct.c_uint8),  # parity_bit_io
        ct.POINTER(ct.c_uint8),  # words4_in
        ct.POINTER(ct.c_uint32),  # resonance_key_io
        ct.c_uint8,  # profile
        ct.c_int64,  # n
    ]
    lib.gyrograph_ingest_word4_batch_indexed.restype = None

    lib.gyrograph_compute_m2_empirical.argtypes = [
        ct.POINTER(ct.c_uint16),
        ct.c_uint64,
    ]
    lib.gyrograph_compute_m2_empirical.restype = ct.c_double

    lib.gyrograph_compute_m2_equilibrium.argtypes = [
        ct.POINTER(ct.c_uint16),
        ct.c_uint64,
    ]
    lib.gyrograph_compute_m2_equilibrium.restype = ct.c_double

    lib.gyrograph_pack_moment.argtypes = [
        ct.c_int64,
        ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_uint64),
        ct.POINTER(ct.c_uint8),
        ct.POINTER(ct.c_uint16),
        ct.POINTER(ct.c_uint16),
        ct.POINTER(ct.c_uint8),
        ct.POINTER(ct.c_int32),
        ct.c_void_p,
    ]
    lib.gyrograph_pack_moment.restype = ct.c_int

    lib.gyrograph_word_signature_from_bytes.argtypes = [
        ct.POINTER(ct.c_uint8),
        ct.c_int64,
        ct.c_void_p,
    ]
    lib.gyrograph_word_signature_from_bytes.restype = ct.c_int

    lib.gyrograph_compose_signatures.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_void_p,
    ]
    lib.gyrograph_compose_signatures.restype = ct.c_int

    lib.gyrograph_apply_signature.argtypes = [
        ct.c_uint32,
        ct.c_void_p,
        ct.POINTER(ct.c_uint32),
    ]
    lib.gyrograph_apply_signature.restype = ct.c_int

    lib.gyrograph_compute_constitutional.argtypes = [
        ct.c_uint32,
        ct.c_void_p,
    ]
    lib.gyrograph_compute_constitutional.restype = ct.c_int

    lib.gyrograph_emit_slcp.argtypes = [
        ct.c_int64,
        ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_uint64),
        ct.POINTER(ct.c_uint8),
        ct.POINTER(ct.c_uint8),
        ct.POINTER(ct.c_uint16),
        ct.POINTER(ct.c_uint16),
        ct.POINTER(ct.c_uint8),
        ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_uint16),
        ct.POINTER(ct.c_uint16),
        ct.POINTER(ct.c_uint8),
        ct.POINTER(ct.c_uint32),
        ct.c_void_p,
    ]
    lib.gyrograph_emit_slcp.restype = ct.c_int

    lib.gyrograph_emit_slcp_batch.argtypes = [
        ct.c_int64,
        ct.POINTER(ct.c_int64),
        ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_uint64),
        ct.POINTER(ct.c_uint8),
        ct.POINTER(ct.c_uint8),
        ct.POINTER(ct.c_uint16),
        ct.POINTER(ct.c_uint16),
        ct.POINTER(ct.c_uint8),
        ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_uint16),
        ct.POINTER(ct.c_uint16),
        ct.POINTER(ct.c_uint8),
        ct.POINTER(ct.c_uint32),
        ct.c_void_p,
    ]
    lib.gyrograph_emit_slcp_batch.restype = ct.c_int

    lib.gyrolabe_analyze_operator_64.argtypes = [
        ct.POINTER(ct.c_float),
        ct.c_float,
        ct.c_void_p,
    ]
    lib.gyrolabe_analyze_operator_64.restype = ct.c_int

    lib.gyrolabe_apply_structured_64.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_float),
    ]
    lib.gyrolabe_apply_structured_64.restype = ct.c_int

    lib.gyrolabe_krawtchouk7_float.argtypes = [
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_float),
    ]
    lib.gyrolabe_krawtchouk7_float.restype = None

    lib.gyrolabe_krawtchouk7_inverse_float.argtypes = [
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_float),
    ]
    lib.gyrolabe_krawtchouk7_inverse_float.restype = None

    lib.gyrograph_moment_from_ledger.argtypes = [
        ct.POINTER(ct.c_uint8),
        ct.c_int64,
        ct.c_void_p,
    ]
    lib.gyrograph_moment_from_ledger.restype = ct.c_int

    lib.gyrograph_verify_moment.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_uint8),
        ct.c_int64,
    ]
    lib.gyrograph_verify_moment.restype = ct.c_int

    lib.gyrograph_compare_ledgers.argtypes = [
        ct.POINTER(ct.c_uint8),
        ct.c_int64,
        ct.POINTER(ct.c_uint8),
        ct.c_int64,
        ct.POINTER(ct.c_int64),
    ]
    lib.gyrograph_compare_ledgers.restype = ct.c_int

    lib.gyrograph_step_state24_by_byte.argtypes = [ct.c_uint32, ct.c_uint8]
    lib.gyrograph_step_state24_by_byte.restype = ct.c_uint32

    lib.gyrograph_inverse_step_state24_by_byte.argtypes = [ct.c_uint32, ct.c_uint8]
    lib.gyrograph_inverse_step_state24_by_byte.restype = ct.c_uint32

    lib.gyrolabe_chirality_evolve_n.argtypes = [
        ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_int32),
        ct.c_uint64,
        ct.POINTER(ct.c_int32),
    ]
    lib.gyrolabe_chirality_evolve_n.restype = None

    lib.gyrolabe_shell_evolve_n.argtypes = [
        ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_float),
        ct.c_uint64,
        ct.POINTER(ct.c_int32),
    ]
    lib.gyrolabe_shell_evolve_n.restype = None

    lib.gyrolabe_horizon_proximity.argtypes = [
        ct.POINTER(ct.c_int32),
        ct.c_float,
    ]
    lib.gyrolabe_horizon_proximity.restype = ct.c_int

    lib.gyrolabe_anisotropy_extract.argtypes = [
        ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_float),
    ]
    lib.gyrolabe_anisotropy_extract.restype = None

    lib.gyrolabe_canonical_decompose.argtypes = [
        ct.c_uint32,
        ct.POINTER(ct.c_uint8),
        ct.POINTER(ct.c_uint8),
        ct.POINTER(ct.c_uint8),
    ]
    lib.gyrolabe_canonical_decompose.restype = None

    lib.gyrolabe_canonical_reconstruct.argtypes = [ct.c_uint8, ct.c_uint8]
    lib.gyrolabe_canonical_reconstruct.restype = ct.c_uint32

    lib.gyrolabe_shell_population.argtypes = [ct.c_uint8]
    lib.gyrolabe_shell_population.restype = ct.c_uint32

    lib.gyrolabe_chi_from_omega12.argtypes = [ct.c_uint32]
    lib.gyrolabe_chi_from_omega12.restype = ct.c_uint8

    lib.gyrolabe_shell_from_chi.argtypes = [ct.c_uint8]
    lib.gyrolabe_shell_from_chi.restype = ct.c_uint8

    lib.gyrolabe_k4_decompose_int32.argtypes = [
        ct.c_int32,
        ct.POINTER(ct.c_int16),
        ct.POINTER(ct.c_int16),
    ]
    lib.gyrolabe_k4_decompose_int32.restype = None

    lib.gyrolabe_k4_contract.argtypes = [
        ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_int32),
        ct.c_int,
        ct.POINTER(ct.c_int64),
        ct.POINTER(ct.c_int64),
        ct.POINTER(ct.c_int64),
        ct.POINTER(ct.c_int64),
    ]
    lib.gyrolabe_k4_contract.restype = None

    lib.gyrolabe_k4_dot.argtypes = [
        ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_int32),
        ct.c_int,
    ]
    lib.gyrolabe_k4_dot.restype = ct.c_int64

    lib.gyrolabe_k4char4_float.argtypes = [
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_float),
    ]
    lib.gyrolabe_k4char4_float.restype = None

    _DLL = lib
    return lib


def gyromatmul_runtime_caps() -> GyroMatMulRuntimeCaps:
    lib = _load_dll()
    caps = GyroMatMulRuntimeCaps()
    lib.gyromatmul_runtime_query(ct.byref(caps))
    return caps


def gyrograph_init() -> None:
    lib = _load_dll()
    lib.gyrograph_init()


def gyrograph_step_state24_by_byte(state24: int, byte: int) -> int:
    lib = _load_dll()
    s = int(state24) & 0xFFFFFF
    b = int(byte) & 0xFF
    return int(lib.gyrograph_step_state24_by_byte(ct.c_uint32(s), ct.c_uint8(b))) & 0xFFFFFF


def gyrograph_inverse_step_state24_by_byte(state24: int, byte: int) -> int:
    lib = _load_dll()
    s = int(state24) & 0xFFFFFF
    b = int(byte) & 0xFF
    return int(lib.gyrograph_inverse_step_state24_by_byte(ct.c_uint32(s), ct.c_uint8(b))) & 0xFFFFFF


def gyrograph_trace_word4_batch_indexed(
    cell_ids: ct.Array,
    omega12_in: ct.Array,
    words4_in: ct.Array,
    n: int,
    omega_trace4_out: ct.Array,
    chi_trace4_out: ct.Array,
) -> None:
    lib = _load_dll()
    lib.gyrograph_trace_word4_batch_indexed(
        cell_ids,
        omega12_in,
        words4_in,
        ct.c_int64(n),
        omega_trace4_out,
        chi_trace4_out,
    )


def gyrograph_apply_trace_word4_batch_indexed(
    cell_ids: ct.Array,
    omega12_io: ct.Array,
    step_io: ct.Array,
    last_byte_io: ct.Array,
    has_closed_word_io: ct.Array,
    word4_io: ct.Array,
    chi_ring64_io: ct.Array,
    chi_ring_pos_io: ct.Array,
    chi_valid_len_io: ct.Array,
    chi_hist64_io: ct.Array,
    shell_hist7_io: ct.Array,
    family_ring64_io: ct.Array,
    family_hist4_io: ct.Array,
    omega_sig_io: ct.Array,
    parity_O12_io: ct.Array,
    parity_E12_io: ct.Array,
    parity_bit_io: ct.Array,
    words4_in: ct.Array,
    omega_trace4_in: ct.Array,
    chi_trace4_in: ct.Array,
    resonance_key_io: ct.Array,
    profile: int,
    n: int,
) -> None:
    _require_batch_word4_n(n)
    _validate_word4_batch_buffers(
        n,
        cell_ids,
        omega12_io,
        step_io,
        last_byte_io,
        has_closed_word_io,
        word4_io,
        chi_ring64_io,
        chi_ring_pos_io,
        chi_valid_len_io,
        chi_hist64_io,
        shell_hist7_io,
        family_ring64_io,
        family_hist4_io,
        omega_sig_io,
        parity_O12_io,
        parity_E12_io,
        parity_bit_io,
        resonance_key_io,
        words4_in,
        omega_trace4_in,
        chi_trace4_in,
    )
    lib = _load_dll()
    lib.gyrograph_apply_trace_word4_batch_indexed(
        cell_ids,
        omega12_io,
        step_io,
        last_byte_io,
        has_closed_word_io,
        word4_io,
        chi_ring64_io,
        chi_ring_pos_io,
        chi_valid_len_io,
        chi_hist64_io,
        shell_hist7_io,
        family_ring64_io,
        family_hist4_io,
        omega_sig_io,
        parity_O12_io,
        parity_E12_io,
        parity_bit_io,
        words4_in,
        omega_trace4_in,
        chi_trace4_in,
        resonance_key_io,
        ct.c_uint8(profile),
        ct.c_int64(n),
    )


def gyrograph_ingest_word4_batch_indexed(
    cell_ids: ct.Array,
    omega12_io: ct.Array,
    step_io: ct.Array,
    last_byte_io: ct.Array,
    has_closed_word_io: ct.Array,
    word4_io: ct.Array,
    chi_ring64_io: ct.Array,
    chi_ring_pos_io: ct.Array,
    chi_valid_len_io: ct.Array,
    chi_hist64_io: ct.Array,
    shell_hist7_io: ct.Array,
    family_ring64_io: ct.Array,
    family_hist4_io: ct.Array,
    omega_sig_io: ct.Array,
    parity_O12_io: ct.Array,
    parity_E12_io: ct.Array,
    parity_bit_io: ct.Array,
    words4_in: ct.Array,
    resonance_key_io: ct.Array,
    profile: int,
    n: int,
) -> None:
    _require_batch_word4_n(n)
    _validate_word4_batch_buffers(
        n,
        cell_ids,
        omega12_io,
        step_io,
        last_byte_io,
        has_closed_word_io,
        word4_io,
        chi_ring64_io,
        chi_ring_pos_io,
        chi_valid_len_io,
        chi_hist64_io,
        shell_hist7_io,
        family_ring64_io,
        family_hist4_io,
        omega_sig_io,
        parity_O12_io,
        parity_E12_io,
        parity_bit_io,
        resonance_key_io,
        words4_in,
        None,
        None,
    )
    lib = _load_dll()
    lib.gyrograph_ingest_word4_batch_indexed(
        cell_ids,
        omega12_io,
        step_io,
        last_byte_io,
        has_closed_word_io,
        word4_io,
        chi_ring64_io,
        chi_ring_pos_io,
        chi_valid_len_io,
        chi_hist64_io,
        shell_hist7_io,
        family_ring64_io,
        family_hist4_io,
        omega_sig_io,
        parity_O12_io,
        parity_E12_io,
        parity_bit_io,
        words4_in,
        resonance_key_io,
        ct.c_uint8(profile),
        ct.c_int64(n),
    )


def gyrograph_compute_m2_empirical(chi_hist64: ct.Array, total: int) -> float:
    """Rényi-2 effective support (sum h)^2 / sum(h^2) on the 64-bin register, in [1, 64]."""
    lib = _load_dll()
    return float(lib.gyrograph_compute_m2_empirical(chi_hist64, ct.c_uint64(total)))


def gyrograph_compute_m2_equilibrium(shell_hist7: ct.Array, total: int) -> float:
    lib = _load_dll()
    return float(lib.gyrograph_compute_m2_equilibrium(shell_hist7, ct.c_uint64(total)))


def gyrograph_pack_moment(
    cell_id: int,
    omega12_io: ct.Array,
    step_io: ct.Array,
    last_byte_io: ct.Array,
    parity_O12_io: ct.Array | None = None,
    parity_E12_io: ct.Array | None = None,
    parity_bit_io: ct.Array | None = None,
    omega_sig_io: ct.Array | None = None,
) -> GyroGraphMoment:
    """Pack one cell; cell_id indexes parallel arrays (length is caller-sized, no fixed cap)."""
    lib = _load_dll()
    out = GyroGraphMoment()
    rc = lib.gyrograph_pack_moment(
        ct.c_int64(cell_id),
        omega12_io,
        step_io,
        last_byte_io,
        parity_O12_io,
        parity_E12_io,
        parity_bit_io,
        omega_sig_io,
        ct.byref(out),
    )
    if rc != 0:
        raise RuntimeError("gyrograph_pack_moment failed")
    return out


def gyrograph_word_signature_from_bytes(data: bytes) -> GyroGraphWordSignature:
    lib = _load_dll()
    n = len(data)
    buf = (ct.c_uint8 * max(n, 1))()
    if n:
        ct.memmove(buf, data, n)
    out = GyroGraphWordSignature()
    rc = lib.gyrograph_word_signature_from_bytes(buf, ct.c_int64(n), ct.byref(out))
    if rc != 0:
        raise RuntimeError("gyrograph_word_signature_from_bytes failed")
    return out


def gyrograph_compose_signatures(
    left: GyroGraphWordSignature,
    right: GyroGraphWordSignature,
) -> GyroGraphWordSignature:
    lib = _load_dll()
    out = GyroGraphWordSignature()
    rc = lib.gyrograph_compose_signatures(
        ct.byref(left),
        ct.byref(right),
        ct.byref(out),
    )
    if rc != 0:
        raise RuntimeError("gyrograph_compose_signatures failed")
    return out


def gyrograph_apply_signature(
    state24: int,
    sig: GyroGraphWordSignature,
) -> int:
    lib = _load_dll()
    out = ct.c_uint32()
    rc = lib.gyrograph_apply_signature(
        ct.c_uint32(state24 & 0xFFFFFF),
        ct.byref(sig),
        ct.byref(out),
    )
    if rc != 0:
        raise RuntimeError("gyrograph_apply_signature failed")
    return int(out.value)


def gyrograph_compute_constitutional(state24: int) -> GyroGraphConstitutional:
    lib = _load_dll()
    out = GyroGraphConstitutional()
    rc = lib.gyrograph_compute_constitutional(
        ct.c_uint32(state24 & 0xFFFFFF),
        ct.byref(out),
    )
    if rc != 0:
        raise RuntimeError("gyrograph_compute_constitutional failed")
    return out


def gyrograph_emit_slcp(
    cell_id: int,
    omega12_io: ct.Array,
    step_io: ct.Array,
    last_byte_io: ct.Array,
    word4_io: ct.Array,
    chi_hist64_io: ct.Array,
    shell_hist7_io: ct.Array,
    family_hist4_io: ct.Array,
    omega_sig_io: ct.Array,
    parity_O12_io: ct.Array,
    parity_E12_io: ct.Array,
    parity_bit_io: ct.Array,
    resonance_key_io: ct.Array,
) -> GyroGraphSLCP:
    lib = _load_dll()
    out = GyroGraphSLCP()
    rc = lib.gyrograph_emit_slcp(
        ct.c_int64(cell_id),
        omega12_io,
        step_io,
        last_byte_io,
        word4_io,
        chi_hist64_io,
        shell_hist7_io,
        family_hist4_io,
        omega_sig_io,
        parity_O12_io,
        parity_E12_io,
        parity_bit_io,
        resonance_key_io,
        ct.byref(out),
    )
    if rc != 0:
        raise RuntimeError("gyrograph_emit_slcp failed")
    return out


def gyrograph_emit_slcp_batch(
    n_cells: int,
    cell_ids: ct.Array,
    omega12_io: ct.Array,
    step_io: ct.Array,
    last_byte_io: ct.Array,
    word4_io: ct.Array,
    chi_hist64_io: ct.Array,
    shell_hist7_io: ct.Array,
    family_hist4_io: ct.Array,
    omega_sig_io: ct.Array,
    parity_O12_io: ct.Array,
    parity_E12_io: ct.Array,
    parity_bit_io: ct.Array,
    resonance_key_io: ct.Array,
) -> list[GyroGraphSLCP]:
    lib = _load_dll()
    if n_cells <= 0:
        return []
    outs = (GyroGraphSLCP * n_cells)()
    rc = lib.gyrograph_emit_slcp_batch(
        ct.c_int64(n_cells),
        cell_ids,
        omega12_io,
        step_io,
        last_byte_io,
        word4_io,
        chi_hist64_io,
        shell_hist7_io,
        family_hist4_io,
        omega_sig_io,
        parity_O12_io,
        parity_E12_io,
        parity_bit_io,
        resonance_key_io,
        outs,
    )
    if rc != 0:
        raise RuntimeError("gyrograph_emit_slcp_batch failed")
    return [outs[i] for i in range(n_cells)]


def gyrolabe_analyze_operator_64(
    W_block: np.ndarray,
    threshold: float = 0.01,
) -> GyroLabeOperatorReport:
    lib = _load_dll()
    w = np.ascontiguousarray(W_block, dtype=np.float32).ravel()
    if w.size != 4096:
        raise ValueError("W_block must contain 4096 float32 values")
    out = GyroLabeOperatorReport()
    rc = lib.gyrolabe_analyze_operator_64(
        w.ctypes.data_as(ct.POINTER(ct.c_float)),
        ct.c_float(threshold),
        ct.byref(out),
    )
    if rc != 0:
        raise RuntimeError("gyrolabe_analyze_operator_64 failed")
    return out


def gyrolabe_apply_structured_64(
    report: GyroLabeOperatorReport,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    lib = _load_dll()
    xa = np.ascontiguousarray(x, dtype=np.float32).ravel()
    ya = np.ascontiguousarray(y, dtype=np.float32).ravel()
    if xa.size != 64 or ya.size != 64:
        raise ValueError("x and y must each contain 64 float32 values")
    rc = lib.gyrolabe_apply_structured_64(
        ct.byref(report),
        xa.ctypes.data_as(ct.POINTER(ct.c_float)),
        ya.ctypes.data_as(ct.POINTER(ct.c_float)),
    )
    if rc != 0:
        raise RuntimeError("gyrolabe_apply_structured_64 failed")


def gyrolabe_chirality_evolve_n(
    chi_hist: list[int] | tuple[int, ...],
    byte_ensemble: list[int] | tuple[int, ...],
    n_steps: int,
) -> list[int]:
    if len(chi_hist) != 64 or len(byte_ensemble) != 64:
        raise ValueError("chi_hist and byte_ensemble must have length 64")
    lib = _load_dll()
    chi = (ct.c_int32 * 64)(*chi_hist)
    ens = (ct.c_int32 * 64)(*byte_ensemble)
    out = (ct.c_int32 * 64)()
    lib.gyrolabe_chirality_evolve_n(
        chi, ens, ct.c_uint64(n_steps), out
    )
    return list(out)


def gyrolabe_shell_evolve_n(
    shell_hist: list[int] | tuple[int, ...],
    eigenvals: list[float] | tuple[float, ...],
    n_steps: int,
) -> list[int]:
    if len(shell_hist) != 7 or len(eigenvals) != 7:
        raise ValueError("shell_hist and eigenvals must have length 7")
    lib = _load_dll()
    sh = (ct.c_int32 * 7)(*shell_hist)
    ev = (ct.c_float * 7)(*eigenvals)
    out = (ct.c_int32 * 7)()
    lib.gyrolabe_shell_evolve_n(sh, ev, ct.c_uint64(n_steps), out)
    return list(out)


def gyrolabe_horizon_proximity(
    chi_hist: list[int] | tuple[int, ...],
    threshold: float,
) -> int:
    if len(chi_hist) != 64:
        raise ValueError("chi_hist must have length 64")
    lib = _load_dll()
    chi = (ct.c_int32 * 64)(*chi_hist)
    return int(lib.gyrolabe_horizon_proximity(chi, ct.c_float(threshold)))


def gyrolabe_anisotropy_extract(
    byte_ensemble: list[int] | tuple[int, ...],
) -> list[float]:
    if len(byte_ensemble) != 256:
        raise ValueError("byte_ensemble must have length 256")
    lib = _load_dll()
    arr = (ct.c_int32 * 256)(*byte_ensemble)
    eta = (ct.c_float * 6)()
    lib.gyrolabe_anisotropy_extract(arr, eta)
    return list(eta)


def gyrolabe_canonical_decompose(omega12: int) -> tuple[int, int, int]:
    lib = _load_dll()
    c = ct.c_uint8()
    chi = ct.c_uint8()
    n_shell = ct.c_uint8()
    lib.gyrolabe_canonical_decompose(
        ct.c_uint32(omega12 & 0xFFF),
        ct.byref(c),
        ct.byref(chi),
        ct.byref(n_shell),
    )
    return (int(c.value), int(chi.value), int(n_shell.value))


def gyrolabe_canonical_reconstruct(c: int, chi: int) -> int:
    lib = _load_dll()
    return int(
        lib.gyrolabe_canonical_reconstruct(
            ct.c_uint8(c & 0xFF), ct.c_uint8(chi & 0xFF)
        )
    )


def gyrolabe_shell_population(N: int) -> int:
    lib = _load_dll()
    return int(lib.gyrolabe_shell_population(ct.c_uint8(N & 0xFF)))


def gyrolabe_chi_from_omega12(omega12: int) -> int:
    lib = _load_dll()
    return int(lib.gyrolabe_chi_from_omega12(ct.c_uint32(omega12 & 0xFFF)))


def gyrolabe_shell_from_chi(chi: int) -> int:
    lib = _load_dll()
    return int(lib.gyrolabe_shell_from_chi(ct.c_uint8(chi & 0xFF)))


def gyrolabe_k4_decompose_int32(v: int) -> tuple[int, int]:
    lib = _load_dll()
    lo = ct.c_int16()
    hi = ct.c_int16()
    lib.gyrolabe_k4_decompose_int32(ct.c_int32(v), ct.byref(lo), ct.byref(hi))
    return (int(lo.value), int(hi.value))


def gyrolabe_k4_contract(
    q: list[int] | tuple[int, ...],
    k: list[int] | tuple[int, ...],
) -> tuple[int, int, int, int]:
    if len(q) != len(k):
        raise ValueError("q and k must have the same length")
    lib = _load_dll()
    n = len(q)
    qa = (ct.c_int32 * n)(*q)
    ka = (ct.c_int32 * n)(*k)
    d00 = ct.c_int64()
    d01 = ct.c_int64()
    d10 = ct.c_int64()
    d11 = ct.c_int64()
    lib.gyrolabe_k4_contract(
        qa,
        ka,
        ct.c_int(n),
        ct.byref(d00),
        ct.byref(d01),
        ct.byref(d10),
        ct.byref(d11),
    )
    return (int(d00.value), int(d01.value), int(d10.value), int(d11.value))


def gyrolabe_k4_dot(
    q: list[int] | tuple[int, ...],
    k: list[int] | tuple[int, ...],
) -> int:
    if len(q) != len(k):
        raise ValueError("q and k must have the same length")
    lib = _load_dll()
    n = len(q)
    qa = (ct.c_int32 * n)(*q)
    ka = (ct.c_int32 * n)(*k)
    return int(lib.gyrolabe_k4_dot(qa, ka, ct.c_int(n)))


def gyrolabe_k4char4_float(
    family_hist: list[float] | tuple[float, ...],
) -> list[float]:
    if len(family_hist) != 4:
        raise ValueError("family_hist must have length 4")
    lib = _load_dll()
    inp = (ct.c_float * 4)(*[float(x) for x in family_hist])
    out = (ct.c_float * 4)()
    lib.gyrolabe_k4char4_float(inp, out)
    return [float(out[i]) for i in range(4)]


def gyrolabe_krawtchouk7_float(shell_hist: list[float] | tuple[float, ...]) -> list[float]:
    if len(shell_hist) != 7:
        raise ValueError("shell_hist must have length 7")
    lib = _load_dll()
    h = (ct.c_float * 7)(*shell_hist)
    out = (ct.c_float * 7)()
    lib.gyrolabe_krawtchouk7_float(h, out)
    return list(out)


def gyrolabe_krawtchouk7_inverse_float(spectral: list[float] | tuple[float, ...]) -> list[float]:
    if len(spectral) != 7:
        raise ValueError("spectral must have length 7")
    lib = _load_dll()
    spec = (ct.c_float * 7)(*spectral)
    out = (ct.c_float * 7)()
    lib.gyrolabe_krawtchouk7_inverse_float(spec, out)
    return list(out)


def gyrograph_moment_from_ledger_native(ledger: bytes) -> GyroGraphMoment:
    lib = _load_dll()
    n = len(ledger)
    buf = (ct.c_uint8 * max(n, 1))()
    if n:
        ct.memmove(buf, ledger, n)
    out = GyroGraphMoment()
    rc = lib.gyrograph_moment_from_ledger(buf, ct.c_int64(n), ct.byref(out))
    if rc != 0:
        raise RuntimeError("gyrograph_moment_from_ledger failed")
    return out


def gyrograph_verify_moment_native(moment: GyroGraphMoment, ledger: bytes) -> bool:
    lib = _load_dll()
    n = len(ledger)
    buf = (ct.c_uint8 * max(n, 1))()
    if n:
        ct.memmove(buf, ledger, n)
    v = lib.gyrograph_verify_moment(ct.byref(moment), buf, ct.c_int64(n))
    if v < 0:
        raise RuntimeError("gyrograph_verify_moment failed")
    return bool(v)


def gyrograph_compare_ledgers_native(a: bytes, b: bytes) -> tuple[int, int]:
    lib = _load_dll()
    na = len(a)
    nb = len(b)
    bufa = (ct.c_uint8 * max(na, 1))()
    bufb = (ct.c_uint8 * max(nb, 1))()
    if na:
        ct.memmove(bufa, a, na)
    if nb:
        ct.memmove(bufb, b, nb)
    prefix = ct.c_int64()
    rc = lib.gyrograph_compare_ledgers(
        bufa, ct.c_int64(na), bufb, ct.c_int64(nb), ct.byref(prefix)
    )
    return int(rc), int(prefix.value)


def _operator_report_for_tiled_weight(
    wb: np.ndarray, threshold: float
) -> GyroLabeOperatorReport:
    """Pad Mx64 to 64x64 for native analyze (top min(M,64) rows; heuristic if M>64)."""
    rows, cols = int(wb.shape[0]), int(wb.shape[1])
    if cols != 64:
        raise ValueError("internal: tiled block width must be 64")
    if rows == 64:
        return gyrolabe_analyze_operator_64(wb, threshold=threshold)
    w64 = np.zeros((64, 64), dtype=np.float32)
    ncopy = min(rows, 64)
    if ncopy:
        w64[:ncopy, :] = np.ascontiguousarray(wb[:ncopy, :], dtype=np.float32)
    return gyrolabe_analyze_operator_64(w64, threshold=threshold)


def tile_external_tensor(W: np.ndarray) -> list[tuple[np.ndarray, GyroLabeOperatorReport]]:
    """Column tiles of width 64; row count M is arbitrary (reports use padded 64x64 when M!=64)."""
    w = np.ascontiguousarray(W, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError("W must be 2-D")
    _, d = int(w.shape[0]), int(w.shape[1])
    pad = (64 - d % 64) % 64
    if pad:
        w = np.pad(w, ((0, 0), (0, pad)), mode="constant")
    blocks: list[tuple[np.ndarray, GyroLabeOperatorReport]] = []
    ncol = w.shape[1]
    for b in range(ncol // 64):
        sl = slice(b * 64, (b + 1) * 64)
        wb = np.ascontiguousarray(w[:, sl], dtype=np.float32)
        rep = _operator_report_for_tiled_weight(wb, threshold=0.01)
        blocks.append((wb, rep))
    return blocks


def apply_hybrid_blocks(
    blocks: list[tuple[np.ndarray, GyroLabeOperatorReport]],
    x: np.ndarray,
) -> np.ndarray:
    if not blocks:
        raise ValueError("blocks must be non-empty")
    x = np.ascontiguousarray(x, dtype=np.float32).ravel()
    rows = int(blocks[0][0].shape[0])
    y = np.zeros(rows, dtype=np.float32)
    nblk = len(blocks)
    need = nblk * 64
    if x.size < need:
        x = np.pad(x, (0, need - x.size), mode="constant")
    for b, (wb, rep) in enumerate(blocks):
        xb = x[b * 64 : (b + 1) * 64]
        use_struct = int(rep.eigenvalues_valid) != 0 and int(wb.shape[0]) == 64
        if use_struct:
            yb = np.zeros(64, dtype=np.float32)
            gyrolabe_apply_structured_64(rep, xb, yb)
            y += yb
        else:
            y += wb @ xb
    return y


__all__ = [
    "GyroGraphConstitutional",
    "GyroGraphMoment",
    "GyroGraphSLCP",
    "GyroGraphWordSignature",
    "GyroLabeOperatorReport",
    "GyroMatMulBlockQ80",
    "GyroMatMulRuntimeCaps",
    "WHT_SIZE",
    "wht_64_batch",
    "topk_energy_fractions",
    "topk_reconstruction_rel_l2",
    "gyrograph_apply_signature",
    "gyrograph_apply_trace_word4_batch_indexed",
    "gyrograph_compose_signatures",
    "gyrograph_compute_constitutional",
    "gyrograph_compute_m2_empirical",
    "gyrograph_compute_m2_equilibrium",
    "gyrograph_compare_ledgers_native",
    "gyrograph_emit_slcp",
    "gyrograph_emit_slcp_batch",
    "gyrograph_init",
    "gyrograph_inverse_step_state24_by_byte",
    "gyrograph_ingest_word4_batch_indexed",
    "gyrograph_moment_from_ledger_native",
    "gyrograph_pack_moment",
    "gyrograph_step_state24_by_byte",
    "gyrograph_trace_word4_batch_indexed",
    "gyrograph_verify_moment_native",
    "gyrograph_word_signature_from_bytes",
    "apply_hybrid_blocks",
    "gyrolabe_analyze_operator_64",
    "gyrolabe_anisotropy_extract",
    "gyrolabe_apply_structured_64",
    "gyrolabe_canonical_decompose",
    "gyrolabe_canonical_reconstruct",
    "gyrolabe_chi_from_omega12",
    "gyrolabe_chirality_evolve_n",
    "gyrolabe_krawtchouk7_float",
    "gyrolabe_krawtchouk7_inverse_float",
    "gyrolabe_horizon_proximity",
    "gyrolabe_k4_contract",
    "gyrolabe_k4_decompose_int32",
    "gyrolabe_k4_dot",
    "gyrolabe_k4char4_float",
    "gyrolabe_shell_evolve_n",
    "gyrolabe_shell_from_chi",
    "gyrolabe_shell_population",
    "gyromatmul_runtime_caps",
    "tile_external_tensor",
]
