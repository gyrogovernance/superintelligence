from __future__ import annotations

import math

import numpy as np
import pytest

from src.tools.gyrograph.bridges.applications import (
    ApplicationDecision,
    ApplicationEvent,
    ApplicationsBridge,
)


def _decision_brief(d: ApplicationDecision) -> str:
    return (
        f"entity={d.entity_id!r} role={d.role!r} cell={d.cell_id} "
        f"hot_loop_score={d.hot_loop_score:.4f} "
        f"contention_score={d.contention_score:.4f} "
        f"action={d.suggested_action} "
        f"res_key={d.resonance_key} res_pop={d.current_resonance} "
        f"shell={d.shell} chi6=0x{d.chi6:02x} "
        f"spec_norm={d.spectral_norm:.4f} "
        f"spec_peak_ratio={d.spectral_peak_ratio:.4f} "
        f"chi_support_ratio={d.chi_support_ratio:.4f} "
        f"chi_peak_ratio={d.chi_peak_ratio:.4f} "
        f"shell_entropy={d.shell_entropy:.4f}"
    )


def test_applications_bridge_mock_stream_clustering():
    print("\n[applications bridge: mock stream clustering]")

    bridge = ApplicationsBridge(
        cell_capacity=64,
        use_native_hotpath=True,
        use_opencl_hotpath=True,
        opencl_min_batch=32,
    )
    print("runtime capabilities:", bridge.runtime_capabilities())

    # One structurally repetitive stream, one broader GC-style stream.
    hot_events = [
        ApplicationEvent(
            entity_id="hot_loop",
            event_type="hot_loop",
            index=i,
            role="main",
            region="loop_body",
            metadata=(("scenario", "mock_stream"),),
        )
        for i in range(50)
    ]

    gc_events = [
        ApplicationEvent(
            entity_id="gc_worker",
            event_type=("gc_start", "mark", "scan", "sweep", "finalize", "idle")[i % 6],
            index=i,
            role="main",
            region="gc_cycle",
            metadata=(("scenario", "mock_stream"),),
        )
        for i in range(50)
    ]

    bridge.ingest_events(hot_events, phase_mod=4)
    bridge.ingest_events(gc_events, phase_mod=6)

    hot_rec = bridge.emit_entity_slcp("hot_loop")
    gc_rec = bridge.emit_entity_slcp("gc_worker")

    hot_dec = bridge.profile_entity("hot_loop")
    gc_dec = bridge.profile_entity("gc_worker")

    spectral_l2 = float(
        np.linalg.norm(
            hot_rec.spectral64.astype(np.float64) - gc_rec.spectral64.astype(np.float64)
        )
    )

    print("hot_loop SLCP:",
          f"res_key={hot_rec.resonance_key} step={hot_rec.step} shell={hot_rec.shell} "
          f"spectral_norm={float(np.linalg.norm(hot_rec.spectral64)):.4f}")
    print("gc_worker SLCP:",
          f"res_key={gc_rec.resonance_key} step={gc_rec.step} shell={gc_rec.shell} "
          f"spectral_norm={float(np.linalg.norm(gc_rec.spectral64)):.4f}")
    print(f"spectral64 L2 difference={spectral_l2:.4f}")
    print("hot_loop decision:", _decision_brief(hot_dec))
    print("gc_worker decision:", _decision_brief(gc_dec))
    print(
        "spectral64 peak indices:",
        f"hot={int(np.argmax(np.abs(hot_rec.spectral64)))}",
        f"gc={int(np.argmax(np.abs(gc_rec.spectral64)))}",
    )

    assert hot_rec.step == 200
    assert gc_rec.step == 200
    assert np.isfinite(hot_rec.spectral64).all()
    assert np.isfinite(gc_rec.spectral64).all()
    assert spectral_l2 > 0.1
    assert hot_dec.hot_loop_score > gc_dec.hot_loop_score


def test_applications_bridge_hot_loop_vs_contention_classifier():
    print("\n[applications bridge: hot loop vs contention classifier]")

    bridge = ApplicationsBridge(
        cell_capacity=64,
        use_native_hotpath=True,
        use_opencl_hotpath=True,
        opencl_min_batch=32,
    )
    print("runtime capabilities:", bridge.runtime_capabilities())

    bridge.feed_hot_loop("jit_candidate", iterations=80, role="main", region="vectorized_loop")
    bridge.feed_lock_contention("thrashing_lock", cycles=40, role="lock", region="mutex_A")

    loop_dec = bridge.profile_entity("jit_candidate", "main")
    lock_dec = bridge.profile_entity("thrashing_lock", "lock")

    loop_rec = bridge.emit_entity_slcp("jit_candidate", "main")
    lock_rec = bridge.emit_entity_slcp("thrashing_lock", "lock")

    cmp = bridge.compare_entities("jit_candidate", "thrashing_lock", role_a="main", role_b="lock")

    print("jit_candidate:", _decision_brief(loop_dec))
    print("thrashing_lock:", _decision_brief(lock_dec))
    print("compare_entities:", cmp)

    print("jit_candidate spectral head:", np.round(loop_rec.spectral64[:8], 4).tolist())
    print("thrashing_lock spectral head:", np.round(lock_rec.spectral64[:8], 4).tolist())

    # Physics: loop has lower contention score than lock (tighter limit cycle).
    assert loop_dec.contention_score < lock_dec.contention_score
    assert lock_dec.contention_score > lock_dec.hot_loop_score

    # Lock should be classified as contention; loop may be "specialize_hot_loop" or "observe" depending on calibration.
    assert lock_dec.suggested_action == "mitigate_contention"
    assert loop_dec.suggested_action in ("specialize_hot_loop", "observe")

    assert cmp["spectral_l2"] > 0.1
    assert np.isfinite(loop_rec.spectral64).all()
    assert np.isfinite(lock_rec.spectral64).all()


def test_applications_bridge_multi_role_mapping_and_usage_surface():
    print("\n[applications bridge: multi-role mapping / usage surface]")

    bridge = ApplicationsBridge(
        cell_capacity=128,
        use_native_hotpath=True,
        use_opencl_hotpath=True,
        opencl_min_batch=32,
    )
    print("runtime capabilities:", bridge.runtime_capabilities())

    entity = "python_service_A"

    # Same runtime entity, two different roles / cells.
    bridge.feed_hot_loop(entity, iterations=40, role="main", region="main_loop")
    bridge.feed_lock_contention(entity, cycles=20, role="lock", region="allocator_lock")
    bridge.feed_gc_cycle(entity, cycles=10, role="gc", region="heap_gc")

    role_map = bridge.entity_cells(entity)
    print("role_map:", role_map)

    records = bridge.emit_entity_records(entity)
    for role, rec in records.items():
        print(
            f"role={role!r} cell={role_map[role]} "
            f"step={rec.step} res_key={rec.resonance_key} shell={rec.shell} "
            f"chi6=0x{rec.chi6:02x} spec_norm={float(np.linalg.norm(rec.spectral64)):.4f}"
        )

    decisions = {role: bridge.profile_entity(entity, role) for role in role_map}
    for role, dec in decisions.items():
        print("decision:", _decision_brief(dec))

    assert set(role_map.keys()) == {"main", "lock", "gc"}
    assert len(set(role_map.values())) == 3

    # iterations=40 -> 40 events -> 40 words -> 160 bytes
    assert records["main"].step == 160
    # cycles=20 * pattern(8) -> 160 events -> 160 words -> 640 bytes
    assert records["lock"].step == 640
    # cycles=10 * pattern(6) -> 60 events -> 60 words -> 240 bytes
    assert records["gc"].step == 240

    assert np.isfinite(records["main"].spectral64).all()
    assert np.isfinite(records["lock"].spectral64).all()
    assert np.isfinite(records["gc"].spectral64).all()

    # exploratory but meaningful:
    # main should skew more hot-loop-like than lock
    assert decisions["main"].hot_loop_score > decisions["lock"].hot_loop_score
    # lock should skew more contention-like than main
    assert decisions["lock"].contention_score > decisions["main"].contention_score