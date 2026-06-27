#!/usr/bin/env python3
"""Chirality-resonance routing on GF(2)^6 (Simon/WHT peak match)."""

from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("repo root not found")


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.gyroscopic import ops


def test_activation_chirality_sign_bits() -> None:
    ops.build_native(force=True)
    x = [1.0 if i % 3 == 0 else -1.0 for i in range(64)]
    chi = ops.activation_chirality(x)
    assert 0 <= chi <= 63


def test_resonance_identity_matches() -> None:
    ops.build_native(force=True)
    chi = 0b101010
    g = ops.gravity_scale(10, 36)
    assert ops.route_resonance(chi, chi, 10, 36, g) == g


def test_resonance_blocks_distant_chirality() -> None:
    ops.build_native(force=True)
    g = ops.gravity_scale(10, 36)
    chi_a = 0
    chi_b = 0b111111
    assert ops.chirality_distance(chi_a, chi_b) == 6
    assert ops.route_resonance(chi_a, chi_b, 10, 36, g) == 0.0


def test_resonance_threshold_two() -> None:
    ops.build_native(force=True)
    g = ops.gravity_scale(5, 36)
    assert ops.route_resonance(0, 0, 5, 36, g) == g
    assert ops.route_resonance(0, 1, 5, 36, g) == g
    assert ops.route_resonance(0, 3, 5, 36, g) == g
    assert ops.route_resonance(0, 7, 5, 36, g) == 0.0
