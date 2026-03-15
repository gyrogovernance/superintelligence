# tests/tools/test_gyrograph_1.py
from __future__ import annotations

import itertools
import struct
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from src.api import (
    omega12_to_state24,
    omega_word_signature,
    pack_omega_signature12,
    q_word6,
    q_word6_for_items,
    state24_to_omega12,
    trajectory_parity_commitment,
    unpack_omega12,
)
from src.constants import GENE_MAC_REST, GENE_MIC_S
from src.sdk import moment_from_ledger
from src.tools.gyrograph import GyroGraph, ResonanceProfile, ensure_word4, pack_word4
from src.tools.gyrograph.core import step_packed_omega12
from src.tools.gyrograph.profiles import (
    bucket_count,
    chi6_from_omega12,
    key_for_closed_word,
    shell_from_omega12,
)

_INGEST_REC = struct.Struct("<I4s")


def _pack_omega12_from_state24(state24: int) -> int:
    om = state24_to_omega12(state24)
    return ((om.u6 & 0x3F) << 6) | (om.v6 & 0x3F)


def _state24_from_packed_omega12(omega12: int) -> int:
    return omega12_to_state24(unpack_omega12(int(omega12) & 0xFFF))


def _slcp_brief(r) -> str:
    return (
        f"cell={r.cell_id} "
        f"step={r.step} "
        f"omega12=0x{r.omega12:03x} "
        f"state24=0x{r.state24:06x} "
        f"last=0x{r.last_byte:02x} "
        f"chi6=0x{r.chi6:02x} "
        f"shell={r.shell} "
        f"q6=0x{r.q6:02x} "
        f"sig=0x{r.omega_sig:04x} "
        f"res_key={r.resonance_key} "
        f"res_pop={r.current_resonance}"
    )


def _find_q_transport_witness() -> bytes:
    """
    Find a 4-byte word for which:
      q_word6_for_items(word4) != q_word6(last_byte)
    so we can explicitly test Q_TRANSPORT profile semantics.
    """
    sample = [0xAA, 0x54, 0xD5, 0x2B, 0x00, 0xFF, 0x33, 0xCC, 0x5A, 0xA5]
    for b0, b1, b2, b3 in itertools.product(sample, repeat=4):
        w = pack_word4(b0, b1, b2, b3)
        if q_word6_for_items(w) != q_word6(b3):
            return w
    raise AssertionError("Could not find a q-transport witness word")


def _parse_ingest_log(path: Path) -> list[tuple[int, bytes]]:
    data = path.read_bytes()
    assert len(data) % _INGEST_REC.size == 0
    out: list[tuple[int, bytes]] = []
    for i in range(0, len(data), _INGEST_REC.size):
        cid, word4 = _INGEST_REC.unpack_from(data, i)
        out.append((cid, word4))
    return out


def _reconstruct_cell_ledger(records: list[tuple[int, bytes]], cell_id: int) -> bytes:
    parts = [word4 for cid, word4 in records if cid == cell_id]
    return b"".join(parts)


def test_gyrograph_smoke_and_bootstrap_state():
    print("\n[gyrograph smoke / bootstrap]")

    g = GyroGraph(cell_capacity=8, profile=ResonanceProfile.CHIRALITY)
    ids = g.allocate_cells(3)

    print(f"capacity={g.capacity}")
    print(f"profile={g.profile.name}")
    print(f"active_cell_ids={g.active_cell_ids}")

    assert g.active_cell_count == 3
    assert ids == [0, 1, 2]

    rest_omega12 = _pack_omega12_from_state24(GENE_MAC_REST)
    print(f"rest_omega12=0x{rest_omega12:03x}")

    # Newly allocated cells should have zero local history and no closed word yet.
    for cid in ids:
        assert int(g._omega12[cid]) == rest_omega12
        assert int(g._step[cid]) == 0
        assert int(g._last_byte[cid]) == GENE_MIC_S
        assert bool(g._has_closed_word[cid]) is False
        assert int(g._chi_valid_len[cid]) == 0
        assert int(g._chi_hist64[cid].sum()) == 0
        assert int(g._shell_hist7[cid].sum()) == 0

    recs = g.emit_slcp(ids)
    for r in recs:
        print(_slcp_brief(r))
        print(f"  spectral64_norm={float(np.linalg.norm(r.spectral64)):.6f}")
        assert r.omega12 == rest_omega12
        assert r.state24 == GENE_MAC_REST
        assert r.last_byte == GENE_MIC_S
        assert r.omega_sig == 0
        assert r.parity_O12 == 0
        assert r.parity_E12 == 0
        assert r.parity_bit == 0
        assert r.current_resonance == g.get_bucket_population(r.resonance_key)

    # All newly allocated cells on chirality profile should be co-resonant at rest.
    print(f"bucket_population(rest_key)={g.get_bucket_population(recs[0].resonance_key)}")
    assert set(g.get_co_resonant_cells(ids[0])) == {ids[1], ids[2]}


def test_gyrograph_word_ingest_matches_sdk_moment_replay():
    print("\n[gyrograph ingest vs sdk replay]")

    g = GyroGraph(cell_capacity=2, profile=ResonanceProfile.CHIRALITY)
    [cid] = g.allocate_cells(1)

    words = [
        pack_word4(0xAA, 0x54, 0xD5, 0x2B),
        pack_word4(0x00, 0x11, 0x22, 0x33),
        pack_word4(0xFE, 0x81, 0x7C, 0xA5),
    ]
    ledger = b"".join(words)

    print("words:")
    for i, w in enumerate(words):
        print(
            f"  w{i}: {[f'0x{b:02x}' for b in w]} "
            f"q_word6_for_items=0x{q_word6_for_items(w):02x}"
        )

    g.ingest([(cid, w) for w in words])
    rec = g.emit_slcp([cid])[0]
    print("SLCP:", _slcp_brief(rec))

    moment = moment_from_ledger(ledger)
    print(
        f"SDK moment: step={moment.step} state24=0x{moment.state24:06x} "
        f"q_transport6=0x{moment.q_transport6:02x}"
    )

    expected_omega12 = _pack_omega12_from_state24(moment.state24)
    expected_sig = pack_omega_signature12(omega_word_signature(words[-1]))
    expected_parity = trajectory_parity_commitment(words[-1])

    assert rec.step == len(ledger)
    assert rec.state24 == moment.state24
    assert rec.omega12 == expected_omega12
    assert rec.last_byte == ledger[-1]
    assert rec.family == g.emit_slcp([cid])[0].family
    assert rec.micro_ref == g.emit_slcp([cid])[0].micro_ref
    assert rec.q6 == q_word6(ledger[-1])
    assert rec.chi6 == chi6_from_omega12(expected_omega12)
    assert rec.shell == shell_from_omega12(expected_omega12)
    assert rec.omega_sig == expected_sig
    assert rec.parity_O12 == expected_parity[0]
    assert rec.parity_E12 == expected_parity[1]
    assert rec.parity_bit == expected_parity[2]

    # state24 and omega12 are mutually consistent
    assert _state24_from_packed_omega12(rec.omega12) == rec.state24


def test_gyrograph_history_warmup_and_ring_replacement():
    print("\n[gyrograph history warmup / ring replacement]")

    g = GyroGraph(cell_capacity=4, profile=ResonanceProfile.CHIRALITY)
    [cid] = g.allocate_cells(1)

    first_word = pack_word4(0x10, 0x20, 0x30, 0x40)
    g.ingest([(cid, first_word)])

    print("after 1 word:")
    print(f"  chi_valid_len={int(g._chi_valid_len[cid])}")
    print(f"  chi_ring_pos={int(g._chi_ring_pos[cid])}")
    print(f"  chi_hist_sum={int(g._chi_hist64[cid].sum())}")
    print(f"  shell_hist_sum={int(g._shell_hist7[cid].sum())}")

    assert int(g._chi_valid_len[cid]) == 4
    assert int(g._chi_ring_pos[cid]) == 4
    assert int(g._chi_hist64[cid].sum()) == 4
    assert int(g._shell_hist7[cid].sum()) == 4
    assert bool(g._has_closed_word[cid]) is True

    # Push beyond 64 bytes total to verify proper replacement behavior.
    more_words = [
        pack_word4((i * 17) & 0xFF, (i * 17 + 1) & 0xFF, (i * 17 + 2) & 0xFF, (i * 17 + 3) & 0xFF)
        for i in range(1, 21)
    ]
    g.ingest([(cid, w) for w in more_words])

    print("after 21 total words (84 bytes):")
    print(f"  chi_valid_len={int(g._chi_valid_len[cid])}")
    print(f"  chi_ring_pos={int(g._chi_ring_pos[cid])}")
    print(f"  chi_hist_sum={int(g._chi_hist64[cid].sum())}")
    print(f"  shell_hist_sum={int(g._shell_hist7[cid].sum())}")

    top_chi = sorted(
        [(i, int(v)) for i, v in enumerate(g._chi_hist64[cid]) if int(v) > 0],
        key=lambda x: (-x[1], x[0]),
    )[:10]
    print("  top chi buckets:", top_chi)
    print("  shell histogram:", [int(x) for x in g._shell_hist7[cid]])

    assert int(g._chi_valid_len[cid]) == 64
    assert int(g._chi_hist64[cid].sum()) == 64
    assert int(g._shell_hist7[cid].sum()) == 64


@pytest.mark.parametrize(
    "profile",
    [
        ResonanceProfile.CHIRALITY,
        ResonanceProfile.SHELL,
        ResonanceProfile.HORIZON_CLASS,
        ResonanceProfile.OMEGA_COINCIDENCE,
        ResonanceProfile.SIGNATURE,
        ResonanceProfile.Q_TRANSPORT,
    ],
)
def test_gyrograph_resonance_profiles_closed_word_keys(profile):
    print(f"\n[profile key test] profile={profile.name}")

    if profile == ResonanceProfile.Q_TRANSPORT:
        word = _find_q_transport_witness()
    else:
        word = pack_word4(0xAA, 0x5A, 0x3C, 0xE1)

    print("word:", [f"0x{b:02x}" for b in word])
    print(f"q(last_byte)=0x{q_word6(word[-1]):02x}")
    print(f"q(word4)=0x{q_word6_for_items(word):02x}")

    g = GyroGraph(cell_capacity=2, profile=profile)
    [cid] = g.allocate_cells(1)
    g.ingest([(cid, word)])

    rec = g.emit_slcp([cid])[0]
    expected_key = key_for_closed_word(
        profile,
        omega12=rec.omega12,
        word4=word,
        omega_sig=rec.omega_sig,
    )

    print("SLCP:", _slcp_brief(rec))
    print(f"expected_key={expected_key} bucket_count={bucket_count(profile)}")

    assert rec.resonance_key == expected_key
    assert rec.current_resonance == 1
    assert g.get_bucket_population(expected_key) == 1

    if profile == ResonanceProfile.Q_TRANSPORT:
        # Explicitly verify we are using the word transport, not the last byte q-class.
        assert rec.resonance_key == q_word6_for_items(word)
        assert rec.resonance_key != q_word6(word[-1])


def test_gyrograph_graph_queries_and_bucket_consistency():
    print("\n[graph queries / bucket consistency]")

    g = GyroGraph(cell_capacity=6, profile=ResonanceProfile.CHIRALITY)
    c0, c1, c2, c3 = g.allocate_cells(4)

    w_same = pack_word4(0x00, 0x11, 0x22, 0x33)
    w_alt1 = pack_word4(0xAA, 0x54, 0xD5, 0x2B)
    w_alt2 = pack_word4(0xFE, 0x81, 0x7C, 0xA5)

    g.ingest(
        [
            (c0, w_same),
            (c1, w_same),
            (c2, w_alt1),
            (c3, w_alt2),
        ]
    )

    recs = g.emit_slcp([c0, c1, c2, c3])
    for r in recs:
        bucket_cells = g.get_bucket_cells(r.resonance_key)
        print(_slcp_brief(r), "bucket_cells=", bucket_cells)
        assert r.current_resonance == len(bucket_cells)
        assert r.cell_id in bucket_cells

    co0 = g.get_co_resonant_cells(c0)
    print(f"co_resonant({c0})={co0}")
    assert c1 in co0

    sig_same = recs[0].omega_sig
    sig_cells = g.get_cells_with_signature(sig_same)
    print(f"cells_with_signature(0x{sig_same:04x})={sig_cells}")
    assert c0 in sig_cells and c1 in sig_cells

    shell0_cells = g.get_cells_on_shell(recs[0].shell)
    chi0_cells = g.get_cells_with_chi6(recs[0].chi6)
    print(f"cells_on_shell({recs[0].shell})={shell0_cells}")
    print(f"cells_with_chi6(0x{recs[0].chi6:02x})={chi0_cells}")
    assert c0 in shell0_cells
    assert c0 in chi0_cells

    d01 = g.chirality_distance_between_cells(c0, c1)
    d02 = g.chirality_distance_between_cells(c0, c2)
    print(f"chirality_distance({c0},{c1})={d01}")
    print(f"chirality_distance({c0},{c2})={d02}")
    assert d01 == 0


def test_gyrograph_error_handling():
    print("\n[error handling / safeguards]")

    g = GyroGraph(cell_capacity=4, profile=ResonanceProfile.CHIRALITY)
    [cid] = g.allocate_cells(1)

    with pytest.raises(ValueError, match="Expected exact 4-byte word"):
        g.ingest([(cid, b"\x00\x01\x02")])
    with pytest.raises(ValueError, match="Expected exact 4-byte word"):
        g.ingest([(cid, b"\x00\x01\x02\x03\x04")])

    unallocated = 3
    while unallocated < g.capacity and g._allocated[unallocated]:
        unallocated += 1
    if unallocated < g.capacity:
        with pytest.raises(ValueError, match="not allocated"):
            g.ingest([(unallocated, pack_word4(0, 0, 0, 0))])
        with pytest.raises(ValueError, match="not allocated"):
            g.emit_slcp([unallocated])
        with pytest.raises(ValueError, match="not allocated"):
            g.get_co_resonant_cells(unallocated)

    with pytest.raises(ValueError, match="shell must be in 0..6"):
        g.get_cells_on_shell(8)
    with pytest.raises(ValueError, match="shell must be in 0..6"):
        g.seed_shell([cid], shell=7)


def test_gyrograph_resonance_decay():
    print("\n[resonance decay]")

    g = GyroGraph(cell_capacity=4, profile=ResonanceProfile.CHIRALITY)
    c0, c1 = g.allocate_cells(2)
    w = pack_word4(0x11, 0x22, 0x33, 0x44)

    for _ in range(100):
        g.ingest([(c0, w), (c1, w)])

    recs = g.emit_slcp([c0, c1])
    assert recs[0].resonance_key == recs[1].resonance_key
    pop_before = recs[0].current_resonance
    assert pop_before == 2

    g.decay_resonance_buckets()
    recs_after = g.emit_slcp([c0, c1])
    pop_after = recs_after[0].current_resonance
    print(f"current_resonance before decay={pop_before} after={pop_after}")
    assert pop_after == pop_before >> 1


def test_gyrograph_native_batch_equivalence():
    print("\n[native batch vs Python step_packed_omega12]")

    import torch

    import src.sdk as sdk

    n = 100
    rest_o12 = _pack_omega12_from_state24(GENE_MAC_REST)
    initial = [(rest_o12 + i) & 0xFFF for i in range(n)]
    byte = 0x7E

    py_out = [step_packed_omega12(x, byte) for x in initial]

    tensor = torch.tensor(initial, dtype=torch.int32, device="cpu")
    native_out = sdk.RuntimeOps.step_omega12_batch(tensor, byte)

    py_list = [int(x) & 0xFFF for x in py_out]
    native_list = [int(x) & 0xFFF for x in native_out.tolist()]
    print(f"byte=0x{byte:02x} first 5 python={py_list[:5]} native={native_list[:5]}")
    assert py_list == native_list


def test_gyrograph_snapshot_restore_roundtrip(tmp_path: Path):
    print("\n[snapshot / restore roundtrip]")

    snap_path = tmp_path / "gyrograph.state.bin"

    g = GyroGraph(cell_capacity=8, profile=ResonanceProfile.SHELL)
    ids = g.allocate_cells(3)
    g.seed_shell([ids[1]], shell=3)
    g.seed_omega(ids[2], 0x155)

    packets = [
        (ids[0], pack_word4(0x01, 0x02, 0x03, 0x04)),
        (ids[1], pack_word4(0xAA, 0xBB, 0xCC, 0xDD)),
        (ids[2], pack_word4(0x10, 0x20, 0x30, 0x40)),
        (ids[0], pack_word4(0x11, 0x22, 0x33, 0x44)),
    ]
    g.ingest(packets)

    recs_before = g.emit_slcp(ids)
    print("before snapshot:")
    for r in recs_before:
        print(" ", _slcp_brief(r))

    g.snapshot(str(snap_path))
    print(f"snapshot_path={snap_path}")
    print(f"snapshot_size={snap_path.stat().st_size} bytes")

    g2 = GyroGraph(cell_capacity=1, profile=ResonanceProfile.CHIRALITY)
    g2.restore(str(snap_path))

    assert g2.capacity == g.capacity
    assert g2.profile == g.profile
    assert g2.active_cell_ids == g.active_cell_ids

    np.testing.assert_array_equal(g2._allocated, g._allocated)
    np.testing.assert_array_equal(g2._has_closed_word, g._has_closed_word)
    np.testing.assert_array_equal(g2._omega12, g._omega12)
    np.testing.assert_array_equal(g2._step, g._step)
    np.testing.assert_array_equal(g2._last_byte, g._last_byte)
    np.testing.assert_array_equal(g2._word4, g._word4)
    np.testing.assert_array_equal(g2._chi_ring64, g._chi_ring64)
    np.testing.assert_array_equal(g2._chi_ring_pos, g._chi_ring_pos)
    np.testing.assert_array_equal(g2._chi_valid_len, g._chi_valid_len)
    np.testing.assert_array_equal(g2._chi_hist64, g._chi_hist64)
    np.testing.assert_array_equal(g2._shell_hist7, g._shell_hist7)
    np.testing.assert_array_equal(g2._omega_sig, g._omega_sig)
    np.testing.assert_array_equal(g2._parity_O12, g._parity_O12)
    np.testing.assert_array_equal(g2._parity_E12, g._parity_E12)
    np.testing.assert_array_equal(g2._parity_bit, g._parity_bit)
    np.testing.assert_array_equal(g2._resonance_key, g._resonance_key)
    np.testing.assert_array_equal(g2._resonance_buckets, g._resonance_buckets)

    recs_after = g2.emit_slcp(ids)
    print("after restore:")
    for r in recs_after:
        print(" ", _slcp_brief(r))

    for a, b in zip(recs_before, recs_after):
        assert a.cell_id == b.cell_id
        assert a.step == b.step
        assert a.omega12 == b.omega12
        assert a.state24 == b.state24
        assert a.last_byte == b.last_byte
        assert a.family == b.family
        assert a.micro_ref == b.micro_ref
        assert a.q6 == b.q6
        assert a.chi6 == b.chi6
        assert a.shell == b.shell
        assert a.horizon_distance == b.horizon_distance
        assert a.ab_distance == b.ab_distance
        assert a.omega_sig == b.omega_sig
        assert a.parity_O12 == b.parity_O12
        assert a.parity_E12 == b.parity_E12
        assert a.parity_bit == b.parity_bit
        assert a.resonance_key == b.resonance_key
        assert a.current_resonance == b.current_resonance
        np.testing.assert_allclose(a.spectral64, b.spectral64, atol=1e-6)


def test_gyrograph_ingest_log_and_replay(tmp_path: Path):
    print("\n[ingest log / per-cell replay]")

    log_path = tmp_path / "gyrograph.ingest.log"

    g = GyroGraph(
        cell_capacity=6,
        profile=ResonanceProfile.CHIRALITY,
        enable_ingest_log=True,
        ingest_log_path=str(log_path),
    )
    c0, c1 = g.allocate_cells(2)

    packets = [
        (c0, pack_word4(0x00, 0x01, 0x02, 0x03)),
        (c1, pack_word4(0x10, 0x11, 0x12, 0x13)),
        (c0, pack_word4(0x20, 0x21, 0x22, 0x23)),
        (c1, pack_word4(0x30, 0x31, 0x32, 0x33)),
        (c0, pack_word4(0x40, 0x41, 0x42, 0x43)),
    ]
    g.ingest(packets)

    mem_records = list(g.iter_ingest_log())
    disk_records = _parse_ingest_log(log_path)

    print("memory ingest log:")
    for rec in mem_records:
        print(" ", rec[0], [f"0x{b:02x}" for b in rec[1]])

    print("disk ingest log:")
    for rec in disk_records:
        print(" ", rec[0], [f"0x{b:02x}" for b in rec[1]])

    assert mem_records == disk_records
    assert log_path.stat().st_size == len(mem_records) * _INGEST_REC.size

    for cid in [c0, c1]:
        ledger = _reconstruct_cell_ledger(disk_records, cid)
        moment = moment_from_ledger(ledger)
        slcp = g.emit_slcp([cid])[0]
        print(
            f"cell={cid} reconstructed_ledger_len={len(ledger)} "
            f"moment_state24=0x{moment.state24:06x} slcp_state24=0x{slcp.state24:06x}"
        )
        assert slcp.state24 == moment.state24
        assert slcp.step == len(ledger)


def test_gyrograph_shell_spectral_and_state_views():
    print("\n[shell spectral / sdk view integration]")

    g = GyroGraph(cell_capacity=4, profile=ResonanceProfile.CHIRALITY)
    [cid] = g.allocate_cells(1)

    words = [
        pack_word4(0xAA, 0x54, 0xD5, 0x2B),
        pack_word4(0x01, 0x23, 0x45, 0x67),
        pack_word4(0x89, 0xAB, 0xCD, 0xEF),
    ]
    g.ingest([(cid, w) for w in words])

    rec = g.emit_slcp([cid])[0]
    shell_spec = g.shell_spectral(cid)
    charts = rec.charts()
    cone1 = rec.future_cone(1)
    cone2 = rec.future_cone(2)
    locus1 = rec.future_locus(1)
    optical = rec.optical_coordinates()
    stab = rec.stabilizer_type()

    print("SLCP:", _slcp_brief(rec))
    print("shell spectral:", shell_spec)
    print("charts.state_hex:", charts.state_hex)
    print("future_cone(1): distinct_states=", cone1.distinct_states, "entropy_bits=", cone1.entropy_bits)
    print("future_cone(2): distinct_states=", cone2.distinct_states, "entropy_bits=", cone2.entropy_bits, "exact_uniform=", cone2.exact_uniform)
    print("future_locus(1):", locus1)
    print("optical_coordinates:", optical)
    print("stabilizer_type:", stab)

    assert all(isinstance(x, Fraction) for x in shell_spec)
    assert charts.state24 == rec.state24
    assert cone1.distinct_states == 128
    assert cone2.distinct_states == 4096
    assert cone2.exact_uniform is True
    assert sum(locus1.values(), Fraction(0, 1)) == Fraction(1, 1)
    assert len(optical) == 3
    assert stab in ("equality", "complement", "bulk")
    assert rec.spectral64.shape == (64,)
    assert np.isfinite(rec.spectral64).all()


def test_gyrograph_performance_exploration():
    print("\n[performance exploration]")

    rng = np.random.default_rng(12345)
    n_cells = 256
    n_packets = 256

    g = GyroGraph(cell_capacity=n_cells, profile=ResonanceProfile.CHIRALITY)
    ids = g.allocate_cells(n_cells)

    packets = []
    for cid in ids[:n_packets]:
        word = bytes(rng.integers(0, 256, size=4, dtype=np.uint8).tolist())
        packets.append((cid, word))

    t0 = time.perf_counter()
    g.ingest(packets)
    t1 = time.perf_counter()

    sample_ids = ids[:32]
    t2 = time.perf_counter()
    recs = g.emit_slcp(sample_ids)
    t3 = time.perf_counter()

    ingest_dt = t1 - t0
    emit_dt = t3 - t2

    ingest_rate = n_packets / ingest_dt if ingest_dt > 0 else float("inf")
    emit_rate = len(sample_ids) / emit_dt if emit_dt > 0 else float("inf")

    print(f"packets_ingested={n_packets} ingest_time={ingest_dt:.6f}s ingest_rate={ingest_rate:.1f} packets/s")
    print(f"slcp_emitted={len(sample_ids)} emit_time={emit_dt:.6f}s emit_rate={emit_rate:.1f} records/s")
    print("sample records:")
    for r in recs[:5]:
        print(" ", _slcp_brief(r))

    assert len(recs) == len(sample_ids)
    assert ingest_dt >= 0.0
    assert emit_dt >= 0.0


def test_gyrograph_bridge_stubs_import():
    print("\n[bridge stubs import]")

    import src.tools.gyrograph.bridges.applications as applications_bridge
    import src.tools.gyrograph.bridges.databases as databases_bridge
    import src.tools.gyrograph.bridges.networks as networks_bridge

    doc = (applications_bridge.__doc__ or "").strip().splitlines()
    print("applications doc:", doc[0] if doc else "")
    doc = (databases_bridge.__doc__ or "").strip().splitlines()
    print("databases doc:", doc[0] if doc else "")
    doc = (networks_bridge.__doc__ or "").strip().splitlines()
    print("networks doc:", doc[0] if doc else "")

    assert applications_bridge is not None
    assert databases_bridge is not None
    assert networks_bridge is not None