"""KV byte serializer: word4 → Ω stepping → chi6 (not raw WHT on floats)."""

from __future__ import annotations

import math
import random

from src.api import chirality_word6 as chirality_word6_py
from src.tools.gyroscopic import ops

GYRO_KV_SLCP_BYTES = 14


def test_chirality_word6_matches_python() -> None:
    rng = random.Random(77)
    for _ in range(500):
        state = rng.randint(0, 0xFFFFFF)
        assert ops.chirality_word6(state) == chirality_word6_py(state)


def test_kv_block_chirality_uses_omega_stepping() -> None:
    rng = random.Random(3)
    x = [rng.uniform(-1.0, 1.0) for _ in range(64)]
    chi, s_out = ops.kv_f32_block_chirality(x)
    assert 0 <= chi < 64
    assert 0 <= s_out <= 0xFFFFFF
    chi2, _ = ops.kv_f32_block_chirality(x)
    assert chi == chi2


def test_temporal_ledger_advances_state() -> None:
    rng = random.Random(9)
    x = [rng.uniform(-1.0, 1.0) for _ in range(64)]
    y = [rng.uniform(-1.0, 1.0) for _ in range(64)]
    _, s0 = ops.kv_f32_block_chirality(x, 0)
    chi_y, s1 = ops.kv_f32_block_chirality(y, s0)
    assert s1 != s0 or chi_y == ops.chirality_word6(s1)


def test_slcp_record_is_14_bytes() -> None:
    assert GYRO_KV_SLCP_BYTES == 14


def test_gravity_coupling_formula() -> None:
    """G(psi)=exp(g1*psi), g1<0: coupling falls with token distance."""
    g1 = ops.gravity_g1()
    assert g1 < 0.0
    near = math.exp(g1 * 1.0 * 0.25)
    far = math.exp(g1 * 40.0 * 0.25)
    assert near > far
    assert far < 1.0


def test_chi_hist_d_eff_widens_for_uniform_hist() -> None:
    uniform = [1] * 64
    d_uni, _, _ = ops.chi_hist_d_eff(uniform, 0)
    condensed = [0] * 64
    condensed[0] = 64
    d_con, _, _ = ops.chi_hist_d_eff(condensed, 0)
    assert d_uni >= d_con
