"""Smoke tests for the Gyroscopic kernel."""

from __future__ import annotations

import random

from src import constants as kpy
from src.tools.gyroscopic import ops
from src.tools.gyroscopic.helpers import diagnostics as diag


def test_step_law_matches_python() -> None:
    rng = random.Random(1234)
    for _ in range(2000):
        state = rng.randint(0, 0xFFFFFF)
        byte = rng.randint(0, 255)
        assert ops.step_omega12(state, byte) == kpy.step_state_by_byte(state, byte)


def test_k4_involutions_and_group_law() -> None:
    rng = random.Random(99)
    psi = [rng.uniform(-1.0, 1.0) for _ in range(ops.OMEGA_SIZE)]
    base = ops.apply_K4(psi, ops.K4_ID)
    for gate in (ops.K4_W2, ops.K4_W2P, ops.K4_F):
        once = ops.apply_K4(base, gate)
        twice = ops.apply_K4(once, gate)
        assert twice == base, f"gate {gate} is not an involution"
    f_direct = ops.apply_K4(base, ops.K4_F)
    f_compose = ops.apply_K4(ops.apply_K4(base, ops.K4_W2), ops.K4_W2P)
    assert f_direct == f_compose


def test_gravity_scale_never_zero() -> None:
    g1 = ops.gravity_g1()
    assert -0.8 < g1 < -0.5, f"g1 out of band: {g1}"
    for layer in range(0, 37):
        for k4 in range(4):
            for shell in range(7):
                s = ops.gravity_scale(layer, 36, k4, shell)
                assert s > 0.0, f"scale zeroed at layer={layer} k4={k4} shell={shell}"


def test_gravity_scale_monotonic() -> None:
    """Depth attenuation decreases (or stays) as layer index increases."""
    total = 36
    prev = ops.gravity_scale(0, total)
    for layer in range(1, total + 1):
        cur = ops.gravity_scale(layer, total)
        assert cur <= prev + 1e-6, f"scale increased at layer {layer}: {prev} -> {cur}"
        prev = cur


def test_depth4_bu_factor_positive() -> None:
    bu = diag.depth4_bu_factor()
    assert 0.99 < bu < 1.0


def test_k4_compose_gyroacc_is_nontrivial() -> None:
    """Gate composition must differ from naive sum of all sector components."""
    sectors = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
    composed = diag.k4_compose_gyroacc(sectors, gravity=1.0)
    naive = sum(a + b for a, b in sectors)
    assert composed != naive
    assert composed == -16.0


def test_extract_phase_native_parity_bits() -> None:
    """k4_char from sign parity matches Z2 x Z2 homomorphism."""
    rng = random.Random(42)
    for _ in range(500):
        sig = bytes(rng.randint(0, 255) for _ in range(16))
        a = int.from_bytes(sig[:8], "little")
        b = int.from_bytes(sig[8:], "little")
        k4, proxy = diag.extract_phase_native(sig)
        pa = bin(a).count("1") & 1
        pb = bin(b).count("1") & 1
        assert k4 == (pa | (pb << 1))
        assert 0 <= proxy <= 7
        assert proxy == (bin(a ^ b).count("1") >> 4)


def test_analyze_is_address_invariant() -> None:
    rng = random.Random(7)
    sig = bytes(rng.randint(0, 255) for _ in range(16))
    a = diag.analyze_q1_group(sig)
    b = diag.analyze_q1_group(sig)
    assert a == b
    q, shell, k4 = a
    assert shell == bin(q).count("1")
    assert 0 <= k4 <= 3
