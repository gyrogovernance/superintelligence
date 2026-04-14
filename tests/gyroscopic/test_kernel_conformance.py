"""
aQPU kernel conformance (GyroLabe / GyroGraph / QuBEC SDK).

Maps to the implementation guide: omega reachability, horizons, byte law,
WHT involution, signature composition, ledger compare, two-step mixing.
"""

from __future__ import annotations

import ctypes as ct
import random
from collections import deque
from collections.abc import Callable

import numpy as np
import pytest

from src.constants import (
    GENE_MAC_REST,
    MASK_STATE24,
    OMEGA_SIZE,
    ab_distance,
    complementarity_invariant,
    horizon_distance,
    is_on_equality_horizon,
    is_on_horizon,
    unpack_state,
)
from src.tools.gyroscopic.bridge import verify_native_against_reference
from src.tools.gyroscopic.target import test_target_equivalence as target_moments_equivalent


def _reachable_from_rest(step24: Callable[[int, int], int]) -> set[int]:
    start = int(GENE_MAC_REST) & MASK_STATE24
    seen = {start}
    q = deque([start])
    while q:
        s = q.popleft()
        for b in range(256):
            ns = step24(s, b) & MASK_STATE24
            if ns not in seen:
                seen.add(ns)
                q.append(ns)
    return seen


@pytest.fixture(scope="module")
def native_gyrograph_byte_ops() -> tuple[
    Callable[[int, int], int], Callable[[int, int], int]
]:
    try:
        from src.tools.gyroscopic import gyromatmul_runtime_caps

        gyromatmul_runtime_caps()
    except Exception as e:
        pytest.skip(f"native gyrolabe DLL not loadable: {e}")
    from src.tools.gyroscopic.ops import (
        gyrograph_inverse_step_state24_by_byte,
        gyrograph_step_state24_by_byte,
    )

    return gyrograph_step_state24_by_byte, gyrograph_inverse_step_state24_by_byte


def test_omega_bfs_size_matches_4096(
    native_gyrograph_byte_ops: tuple[
        Callable[[int, int], int], Callable[[int, int], int]
    ],
) -> None:
    step24, _ = native_gyrograph_byte_ops
    seen = _reachable_from_rest(step24)
    assert len(seen) == OMEGA_SIZE


def test_horizons_within_reachable_component(
    native_gyrograph_byte_ops: tuple[
        Callable[[int, int], int], Callable[[int, int], int]
    ],
) -> None:
    step24, _ = native_gyrograph_byte_ops
    seen = _reachable_from_rest(step24)
    eq_n = sum(1 for s in seen if is_on_equality_horizon(s))
    ch_n = sum(1 for s in seen if is_on_horizon(s))
    assert eq_n == 64
    assert ch_n == 64
    for s in seen:
        a12, b12 = unpack_state(s)
        assert horizon_distance(a12, b12) + ab_distance(a12, b12) == 12


def test_complementarity_sum_12_for_all_a12_b12() -> None:
    for a in range(4096):
        for b in range(4096):
            assert complementarity_invariant(a, b)


def test_two_step_uniformization_from_rest(
    native_gyrograph_byte_ops: tuple[
        Callable[[int, int], int], Callable[[int, int], int]
    ],
) -> None:
    step24, _ = native_gyrograph_byte_ops
    s0 = int(GENE_MAC_REST) & MASK_STATE24
    hist: dict[int, int] = {}
    for b1 in range(256):
        for b2 in range(256):
            s1 = step24(s0, b1) & MASK_STATE24
            s2 = step24(s1, b2) & MASK_STATE24
            hist[s2] = hist.get(s2, 0) + 1
    assert len(hist) == OMEGA_SIZE
    assert all(c == 16 for c in hist.values())


def test_wht_involution_exact_integer_random() -> None:
    from src.api import walsh_sign6

    h = np.array(
        [[walsh_sign6(q, r) for r in range(64)] for q in range(64)],
        dtype=np.int64,
    )
    rng = np.random.default_rng(7)
    for _ in range(10000):
        f0 = rng.integers(-500, 500, size=64, dtype=np.int64)
        w1 = h @ f0
        w2 = h @ w1
        np.testing.assert_array_equal(w2, 64 * f0)


def test_signature_composition_law_concat(
    native_gyrograph_byte_ops: tuple[
        Callable[[int, int], int], Callable[[int, int], int]
    ],
) -> None:
    _ = native_gyrograph_byte_ops
    from src.tools.gyroscopic.ops import (
        gyrograph_compose_signatures,
        gyrograph_word_signature_from_bytes,
    )

    rng = random.Random(31)
    for _ in range(128):
        la = rng.randint(1, 5)
        lb = rng.randint(1, 5)
        ba = bytes(rng.randint(0, 255) for _ in range(la))
        bb = bytes(rng.randint(0, 255) for _ in range(lb))
        concat = ba + bb
        s_ab = gyrograph_word_signature_from_bytes(concat)
        s_a = gyrograph_word_signature_from_bytes(ba)
        s_b = gyrograph_word_signature_from_bytes(bb)
        composed = gyrograph_compose_signatures(s_b, s_a)
        assert composed.parity == s_ab.parity
        assert composed.tau_a12 == s_ab.tau_a12
        assert composed.tau_b12 == s_ab.tau_b12


def test_native_byte_step_matches_moment_from_ledger(
    native_gyrograph_byte_ops: tuple[
        Callable[[int, int], int], Callable[[int, int], int]
    ],
) -> None:
    from src.tools.gyroscopic.ops import gyrograph_moment_from_ledger_native

    step24, _ = native_gyrograph_byte_ops
    s = GENE_MAC_REST & MASK_STATE24
    for b in (0, 1, 0xAA, 0xFF, 0x54):
        m = gyrograph_moment_from_ledger_native(bytes([b]))
        assert m.state24 == step24(s, b) & MASK_STATE24


def test_byte_step_inverse_roundtrip_random(
    native_gyrograph_byte_ops: tuple[
        Callable[[int, int], int], Callable[[int, int], int]
    ],
) -> None:
    step24, inv24 = native_gyrograph_byte_ops
    rng = random.Random(99)
    for _ in range(5000):
        s = rng.randint(0, MASK_STATE24)
        b = rng.randint(0, 255)
        t = step24(s, b) & MASK_STATE24
        assert inv24(t, b) & MASK_STATE24 == s


def test_compare_ledgers_pair_native(
    native_gyrograph_byte_ops: tuple[
        Callable[[int, int], int], Callable[[int, int], int]
    ],
) -> None:
    _ = native_gyrograph_byte_ops
    from src.tools.gyroscopic.ops import gyrograph_compare_ledgers_native

    rc, pref = gyrograph_compare_ledgers_native(b"abc", b"abd")
    assert pref == 2
    assert rc != 0
    rc2, pref2 = gyrograph_compare_ledgers_native(b"x", b"x")
    assert rc2 == 0
    assert pref2 == 1


def test_targets_match(
    native_gyrograph_byte_ops: tuple[
        Callable[[int, int], int], Callable[[int, int], int]
    ],
) -> None:
    _ = native_gyrograph_byte_ops
    assert target_moments_equivalent(b"\x00\x01\x02\x03") is True
    assert target_moments_equivalent(b"") is True


def test_moment_verify_native_against_reference(
    native_gyrograph_byte_ops: tuple[
        Callable[[int, int], int], Callable[[int, int], int]
    ],
) -> None:
    _ = native_gyrograph_byte_ops
    omega12 = (ct.c_int32 * 256)(*([0] * 256))
    step = (ct.c_uint64 * 256)(*([2] * 256))
    last_byte = (ct.c_uint8 * 256)(*([0] * 256))
    result = verify_native_against_reference(0, omega12, step, last_byte, b"\x00\x01")
    assert result["step_match"] is True
    assert isinstance(result["state_match"], bool)
    assert "native" in result and "reference" in result
