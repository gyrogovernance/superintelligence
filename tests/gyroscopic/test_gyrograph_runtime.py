"""GyroGraph runtime conformance: batch ingest, climate, persist, and SDK surfaces."""

from __future__ import annotations

import ctypes as ct
import tempfile

import numpy as np
import pytest

from src.constants import GENE_MAC_REST, MASK_STATE24, step_state_by_byte
from src.sdk import (
    apply_packed64_gemv,
    dyadic_wht64_normalized,
    pack_matrix64,
    state_scan_from_state,
)
from src.tools.gyroscopic.bridge import build_result, get_interoperability_outputs
from src.tools.gyroscopic.circuit import ByteOp, GateOp, compile_circuit
from src.tools.gyroscopic.climate import (
    cell_climate_from_histograms,
    m2_empirical_from_chi_hist,
    m2_equilibrium_from_shell_hist,
    shell_order_parameters_from_hist,
)
from src.tools.gyroscopic.ops import (
    GyroGraphSLCP,
    gyrograph_apply_trace_word4_batch_indexed,
    gyrograph_emit_slcp,
    gyrograph_emit_slcp_batch,
    gyrograph_ingest_word4_batch_indexed,
    gyrograph_moment_from_ledger_native,
    gyrograph_pack_moment,
    gyrolabe_krawtchouk7_float,
    gyrolabe_krawtchouk7_inverse_float,
    gyromatmul_runtime_caps,
)
from src.tools.gyroscopic.persist import (
    GyrographSnapshot,
    compute_kernel_digest,
    compute_kernel_law_digest,
    restore,
    snapshot,
)


@pytest.fixture(scope="module")
def _native() -> None:
    try:
        gyromatmul_runtime_caps()
    except Exception as e:
        pytest.skip(f"native DLL: {e}")


def _alloc_word4_buffers() -> dict[str, ct.Array]:
    z = lambda n: (ct.c_uint8 * n)(*([0] * n))
    zu16 = lambda n: (ct.c_uint16 * n)(*([0] * n))
    zi32 = lambda n: (ct.c_int32 * n)(*([0] * n))
    zq = lambda n: (ct.c_uint64 * n)(*([0] * n))
    zi64 = lambda n: (ct.c_int64 * n)(*([0] * n))
    zu32 = lambda n: (ct.c_uint32 * n)(*([0] * n))
    return {
        "cell_ids": zi64(1),
        "omega12_io": zi32(256),
        "step_io": zq(256),
        "last_byte_io": z(256),
        "has_closed_word_io": z(256),
        "word4_io": z(1024),
        "chi_ring64_io": z(256 * 64),
        "chi_ring_pos_io": z(256),
        "chi_valid_len_io": z(256),
        "chi_hist64_io": zu16(256 * 64),
        "shell_hist7_io": zu16(256 * 7),
        "family_ring64_io": z(256 * 64),
        "family_hist4_io": zu16(256 * 4),
        "omega_sig_io": zi32(256),
        "parity_O12_io": zu16(256),
        "parity_E12_io": zu16(256),
        "parity_bit_io": z(256),
        "words4_in": z(4),
        "omega_trace4_in": zi32(4),
        "chi_trace4_in": z(4),
        "resonance_key_io": zu32(256),
    }


def test_shell_order_parameters_uniform_shell_3() -> None:
    h = [0, 0, 0, 64, 0, 0, 0]
    d = shell_order_parameters_from_hist(h)
    assert d["N_mean"] == pytest.approx(3.0)
    assert d["rho"] == pytest.approx(0.5)
    assert d["eta"] == pytest.approx(0.0)
    assert d["m"] == pytest.approx(0.0)
    assert d["var_N"] == pytest.approx(1.5)


def test_shell_order_parameters_empty_total() -> None:
    d = shell_order_parameters_from_hist([0] * 7)
    assert d["eta"] == 1.0
    assert d["m"] == -1.0


def test_m2_empirical_point_mass(_native: None) -> None:
    chi = [100] + [0] * 63
    assert m2_empirical_from_chi_hist(chi) == pytest.approx(64.0)


def test_m2_empirical_uniform_chi(_native: None) -> None:
    chi = [10] * 64
    assert m2_empirical_from_chi_hist(chi) == pytest.approx(4096.0)


def test_m2_equilibrium_all_shell_zero(_native: None) -> None:
    sh = [64, 0, 0, 0, 0, 0, 0]
    assert m2_equilibrium_from_shell_hist(sh) == pytest.approx(64.0)


def test_cell_climate_from_histograms_shape(_native: None) -> None:
    chi = [1] + [0] * 63
    sh = [64, 0, 0, 0, 0, 0, 0]
    fam = [16, 16, 16, 16]
    out = cell_climate_from_histograms(chi, sh, fam)
    assert set(out.keys()) == {
        "N_mean",
        "rho",
        "eta",
        "m",
        "var_N",
        "M2_empirical",
        "M2_equilibrium",
        "shell_spectral",
        "gauge_spectral",
        "byte_anisotropy",
    }
    assert out["byte_anisotropy"] is None
    assert len(out["shell_spectral"]) == 7
    assert len(out["gauge_spectral"]) == 4


def test_cell_climate_with_byte_anisotropy(_native: None) -> None:
    chi = [1] + [0] * 63
    sh = [64, 0, 0, 0, 0, 0, 0]
    fam = [0, 0, 0, 0]
    ens = [1] + [0] * 255
    out = cell_climate_from_histograms(chi, sh, fam, byte_ensemble_256=ens)
    assert out["byte_anisotropy"] is not None
    assert len(out["byte_anisotropy"]) == 6


def test_ingest_word4_batch_indexed_smoke(_native: None) -> None:
    b = _alloc_word4_buffers()
    b["cell_ids"][0] = 0
    for i in range(4):
        b["words4_in"][i] = (0x10 + i) & 0xFF
    gyrograph_ingest_word4_batch_indexed(
        b["cell_ids"],
        b["omega12_io"],
        b["step_io"],
        b["last_byte_io"],
        b["has_closed_word_io"],
        b["word4_io"],
        b["chi_ring64_io"],
        b["chi_ring_pos_io"],
        b["chi_valid_len_io"],
        b["chi_hist64_io"],
        b["shell_hist7_io"],
        b["family_ring64_io"],
        b["family_hist4_io"],
        b["omega_sig_io"],
        b["parity_O12_io"],
        b["parity_E12_io"],
        b["parity_bit_io"],
        b["words4_in"],
        b["resonance_key_io"],
        0,
        1,
    )
    assert int(b["step_io"][0]) == 4
    assert int(b["last_byte_io"][0]) == 0x13
    assert int(b["has_closed_word_io"][0]) == 1


def test_apply_trace_word4_batch_indexed_smoke(_native: None) -> None:
    b = _alloc_word4_buffers()
    b["cell_ids"][0] = 0
    for i in range(4):
        b["words4_in"][i] = (0x20 + i) & 0xFF
    b["omega_trace4_in"][3] = 0x155
    gyrograph_apply_trace_word4_batch_indexed(
        b["cell_ids"],
        b["omega12_io"],
        b["step_io"],
        b["last_byte_io"],
        b["has_closed_word_io"],
        b["word4_io"],
        b["chi_ring64_io"],
        b["chi_ring_pos_io"],
        b["chi_valid_len_io"],
        b["chi_hist64_io"],
        b["shell_hist7_io"],
        b["family_ring64_io"],
        b["family_hist4_io"],
        b["omega_sig_io"],
        b["parity_O12_io"],
        b["parity_E12_io"],
        b["parity_bit_io"],
        b["words4_in"],
        b["omega_trace4_in"],
        b["chi_trace4_in"],
        b["resonance_key_io"],
        0,
        1,
    )
    assert int(b["step_io"][0]) == 4
    assert int(b["last_byte_io"][0]) == 0x23
    assert int(b["has_closed_word_io"][0]) == 1


def test_ingest_minimal_one_cell_arrays(_native: None) -> None:
    n_cells = 1
    cell_ids = (ct.c_int64 * 1)(0)
    omega12_io = (ct.c_int32 * n_cells)(0)
    step_io = (ct.c_uint64 * n_cells)(0)
    last_byte_io = (ct.c_uint8 * n_cells)(0)
    has_closed = (ct.c_uint8 * n_cells)(0)
    word4_io = (ct.c_uint8 * (4 * n_cells))(*([0] * (4 * n_cells)))
    chi_ring64 = (ct.c_uint8 * (64 * n_cells))(*([0] * (64 * n_cells)))
    chi_pos = (ct.c_uint8 * n_cells)(0)
    chi_valid = (ct.c_uint8 * n_cells)(0)
    chi_hist = (ct.c_uint16 * (64 * n_cells))(*([0] * (64 * n_cells)))
    shell_hist = (ct.c_uint16 * (7 * n_cells))(*([0] * (7 * n_cells)))
    family_ring = (ct.c_uint8 * (64 * n_cells))(*([0] * (64 * n_cells)))
    family_hist = (ct.c_uint16 * (4 * n_cells))(*([0] * (4 * n_cells)))
    omega_sig = (ct.c_int32 * n_cells)(0)
    pO = (ct.c_uint16 * n_cells)(0)
    pE = (ct.c_uint16 * n_cells)(0)
    pbit = (ct.c_uint8 * n_cells)(0)
    words_in = (ct.c_uint8 * 4)(1, 2, 3, 4)
    res_key = (ct.c_uint32 * n_cells)(0)
    gyrograph_ingest_word4_batch_indexed(
        cell_ids,
        omega12_io,
        step_io,
        last_byte_io,
        has_closed,
        word4_io,
        chi_ring64,
        chi_pos,
        chi_valid,
        chi_hist,
        shell_hist,
        family_ring,
        family_hist,
        omega_sig,
        pO,
        pE,
        pbit,
        words_in,
        res_key,
        0,
        1,
    )
    assert int(step_io[0]) == 4
    assert int(last_byte_io[0]) == 4


def test_words4_in_length_validated(_native: None) -> None:
    b = _alloc_word4_buffers()
    b["cell_ids"][0] = 0
    short_words = (ct.c_uint8 * 3)(0, 0, 0)
    with pytest.raises(ValueError, match="words4_in"):
        gyrograph_ingest_word4_batch_indexed(
            b["cell_ids"],
            b["omega12_io"],
            b["step_io"],
            b["last_byte_io"],
            b["has_closed_word_io"],
            b["word4_io"],
            b["chi_ring64_io"],
            b["chi_ring_pos_io"],
            b["chi_valid_len_io"],
            b["chi_hist64_io"],
            b["shell_hist7_io"],
            b["family_ring64_io"],
            b["family_hist4_io"],
            b["omega_sig_io"],
            b["parity_O12_io"],
            b["parity_E12_io"],
            b["parity_bit_io"],
            short_words,
            b["resonance_key_io"],
            0,
            1,
        )


def test_batch_n_rejects_zero(_native: None) -> None:
    b = _alloc_word4_buffers()
    with pytest.raises(ValueError, match=">= 1"):
        gyrograph_ingest_word4_batch_indexed(
            b["cell_ids"],
            b["omega12_io"],
            b["step_io"],
            b["last_byte_io"],
            b["has_closed_word_io"],
            b["word4_io"],
            b["chi_ring64_io"],
            b["chi_ring_pos_io"],
            b["chi_valid_len_io"],
            b["chi_hist64_io"],
            b["shell_hist7_io"],
            b["family_ring64_io"],
            b["family_hist4_io"],
            b["omega_sig_io"],
            b["parity_O12_io"],
            b["parity_E12_io"],
            b["parity_bit_io"],
            b["words4_in"],
            b["resonance_key_io"],
            0,
            0,
        )


def test_gyrograph_pack_moment_cell_id_above_255(_native: None) -> None:
    n = 512
    omega12 = (ct.c_int32 * n)(0)
    step = (ct.c_uint64 * n)(0)
    last_byte = (ct.c_uint8 * n)(0)
    cid = 400
    omega12[cid] = 0x155
    step[cid] = 9
    last_byte[cid] = 0xAB
    m = gyrograph_pack_moment(cid, omega12, step, last_byte)
    assert int(m.step) == 9
    assert int(m.last_byte) == 0xAB


def test_slcp_emission(_native: None) -> None:
    omega12 = (ct.c_int32 * 256)(*([0] * 256))
    step = (ct.c_uint64 * 256)(*([10] * 256))
    last_byte = (ct.c_uint8 * 256)(*([0] * 256))
    word4 = (ct.c_uint8 * 1024)(*([0] * 1024))
    chi_hist = (ct.c_uint16 * 16384)(*([1] * 16384))
    shell_hist = (ct.c_uint16 * 1792)(*([1] * 1792))
    family_hist = (ct.c_uint8 * 1024)(*([0] * 1024))
    omega_sig = (ct.c_int32 * 256)(*([0] * 256))
    parity_O = (ct.c_uint16 * 256)(*([0] * 256))
    parity_E = (ct.c_uint16 * 256)(*([0] * 256))
    parity_bit = (ct.c_uint8 * 256)(*([0] * 256))
    res_key = (ct.c_uint32 * 256)(*([0] * 256))
    slcp = gyrograph_emit_slcp(
        0,
        omega12,
        step,
        last_byte,
        word4,
        chi_hist,
        shell_hist,
        family_hist,
        omega_sig,
        parity_O,
        parity_E,
        parity_bit,
        res_key,
    )
    assert slcp.cell_id == 0
    assert slcp.step == 10
    assert slcp.chi6 == 0
    assert slcp.shell == 0
    assert sum(slcp.spectral64) == pytest.approx(1.0)
    sh = [float(shell_hist[i]) for i in range(7)]
    exp_shell = gyrolabe_krawtchouk7_float(sh)
    for i in range(7):
        assert slcp.shell_spectral[i] == pytest.approx(exp_shell[i])


def test_slcp_emit_batch_matches_single(_native: None) -> None:
    omega12 = (ct.c_int32 * 256)(*([0] * 256))
    step = (ct.c_uint64 * 256)(*([10] * 256))
    last_byte = (ct.c_uint8 * 256)(*([0] * 256))
    word4 = (ct.c_uint8 * 1024)(*([0] * 1024))
    chi_hist = (ct.c_uint16 * 16384)(*([0] * 16384))
    shell_hist = (ct.c_uint16 * 1792)(*([0] * 1792))
    family_hist = (ct.c_uint8 * 1024)(*([0] * 1024))
    omega_sig = (ct.c_int32 * 256)(*([0] * 256))
    parity_O = (ct.c_uint16 * 256)(*([0] * 256))
    parity_E = (ct.c_uint16 * 256)(*([0] * 256))
    parity_bit = (ct.c_uint8 * 256)(*([0] * 256))
    res_key = (ct.c_uint32 * 256)(*([0] * 256))
    for cid in range(24):
        for j in range(64):
            chi_hist[cid * 64 + j] = (cid + j) & 0xFFFF
        shell_hist[cid * 7 + (cid % 7)] = 1 + cid
        family_hist[cid * 4 + (cid % 4)] = cid % 256
    ids = (ct.c_int64 * 24)(*range(24))
    batch_out = gyrograph_emit_slcp_batch(
        24,
        ids,
        omega12,
        step,
        last_byte,
        word4,
        chi_hist,
        shell_hist,
        family_hist,
        omega_sig,
        parity_O,
        parity_E,
        parity_bit,
        res_key,
    )
    assert len(batch_out) == 24
    for cid in range(24):
        one = gyrograph_emit_slcp(
            cid,
            omega12,
            step,
            last_byte,
            word4,
            chi_hist,
            shell_hist,
            family_hist,
            omega_sig,
            parity_O,
            parity_E,
            parity_bit,
            res_key,
        )
        b = batch_out[cid]
        assert b.cell_id == one.cell_id
        assert b.step == one.step
        assert b.omega12 == one.omega12
        assert b.state24 == one.state24
        assert b.current_resonance == one.current_resonance
        for k in range(64):
            assert b.spectral64[k] == one.spectral64[k]
        for k in range(4):
            assert b.gauge_spectral[k] == one.gauge_spectral[k]
        for k in range(7):
            assert b.shell_spectral[k] == one.shell_spectral[k]


def test_snapshot_restore_roundtrip(_native: None) -> None:
    class Cell:
        omega12 = 0x155
        step = 3
        last_byte = 0x42
        chi_hist64 = [i & 0xFFFF for i in range(64)]
        shell_hist7 = [1, 0, 2, 0, 0, 0, 0]
        family_hist4 = [1, 2, 3, 4]
        omega_sig = 0x123
        parity_O12 = 1
        parity_E12 = 2
        parity_bit = 1
        resonance_key = 0xABCDEF01

    with tempfile.NamedTemporaryFile(delete=False, suffix=".gyrg") as tmp:
        path = tmp.name
    try:
        snapshot(path, [Cell()])
        snap = restore(path, verify_kernel=True)
        assert isinstance(snap, GyrographSnapshot)
        assert snap.version == 1
        assert snap.kernel_digest == compute_kernel_digest()
        assert compute_kernel_law_digest() != b""
        assert len(snap.cells) == 1
        c0 = snap.cells[0]
        assert c0["omega12"] == Cell.omega12
        assert c0["step"] == Cell.step
        assert c0["last_byte"] == Cell.last_byte
        assert c0["chi_hist64"][:4] == [0, 1, 2, 3]
        assert c0["shell_hist7"] == [1, 0, 2, 0, 0, 0, 0]
        assert c0["family_hist4"] == [1, 2, 3, 4]
        assert c0["omega_sig"] == Cell.omega_sig
        assert c0["parity_O12"] == 1
        assert c0["parity_E12"] == 2
        assert c0["parity_bit"] == 1
        assert c0["resonance_key"] == Cell.resonance_key
    finally:
        import os

        os.unlink(path)


def test_pack_matrix64_roundtrip() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((3, 64)).astype(np.float64)
    x = rng.standard_normal(64).astype(np.float64)
    packed = pack_matrix64(W, n_bits=8)
    y = apply_packed64_gemv(packed, x)
    y_ref = W @ x
    np.testing.assert_allclose(y, y_ref, rtol=0.15, atol=0.05)


def test_dyadic_wht_involution() -> None:
    f = [1] + [0] * 63
    d1 = dyadic_wht64_normalized(f)
    assert len(d1.numerators) == 64
    from src.tools.gyroscopic.dyadic_wht import wht64_int_forward

    w1 = wht64_int_forward(f)
    w2 = wht64_int_forward(w1)
    assert w2[0] == 64


def test_state_scan_matches_stepping() -> None:
    payload = bytes([0, 1, 0xAA])
    s0 = GENE_MAC_REST & MASK_STATE24
    seq = state_scan_from_state(payload, s0)
    assert len(seq) == 3
    s = s0
    for b, t in zip(payload, seq, strict=True):
        s = step_state_by_byte(s, b) & MASK_STATE24
        assert t == s


def test_interoperability_thirteen_keys(_native: None) -> None:
    slcp = GyroGraphSLCP()
    slcp.cell_id = 0
    slcp.step = 1
    slcp.omega12 = 0
    slcp.state24 = 0
    slcp.last_byte = 0
    slcp.family = 0
    slcp.micro_ref = 0
    slcp.q6 = 0
    slcp.chi6 = 3
    slcp.shell = 2
    slcp.horizon_distance = 0
    slcp.ab_distance = 0
    slcp.omega_sig = 0
    slcp.parity_O12 = 0
    slcp.parity_E12 = 0
    slcp.parity_bit = 0
    slcp.resonance_key = 0
    slcp.current_resonance = 10
    for i in range(64):
        slcp.spectral64[i] = 1.0 if i == 0 else 0.0
    for i in range(4):
        slcp.gauge_spectral[i] = float(i)
    for i in range(7):
        slcp.shell_spectral[i] = float(i)
    ens = [0] * 256
    ens[0] = 100
    ens[255] = 100
    out = get_interoperability_outputs(slcp, byte_ensemble_256=ens)
    assert len(out) == 13
    assert set(out.keys()) == {
        "block_class",
        "block_scr",
        "block_defect_norm",
        "native_route",
        "kv_priority",
        "batch_group_id",
        "gauge_anisotropy",
        "spectral_damping",
        "chi_anisotropy",
        "effective_support",
        "shell_spectral",
        "gauge_spectral",
        "operator_class_id",
    }


def test_krawtchouk7_float_roundtrip(_native: None) -> None:
    rng = np.random.default_rng(0)
    for _ in range(5):
        h = rng.standard_normal(7).astype(np.float32)
        spec = gyrolabe_krawtchouk7_float(list(h))
        h2 = np.array(gyrolabe_krawtchouk7_inverse_float(spec), dtype=np.float32)
        np.testing.assert_allclose(h2, h, rtol=1e-4, atol=1e-3)


def test_compile_circuit(_native: None) -> None:
    b, sig, ir = compile_circuit([ByteOp(5, 1), GateOp("S")])
    assert len(b) >= 1
    assert int(sig.tau_a12) >= 0
    assert ir == []


def test_build_result_constitutional_fields(_native: None) -> None:
    ledger = b"\x01"
    m = gyrograph_moment_from_ledger_native(ledger)
    r = build_result(m, ledger)
    c = r["charts"]["constitutional"]
    assert "is_on_equality_horizon" in c
    assert "complementarity_sum" in c
