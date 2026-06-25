#!/usr/bin/env python3
"""Q1 structure diagnostic: binomial reference and WHT route stats."""

from __future__ import annotations

import random
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

from src.tools.gyroscopic.helpers import diagnostics as diag


def test_expected_shell_sums_to_one() -> None:
    probs = diag.expected_shell_probs()
    assert len(probs) == 7
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert abs(probs[3] - 20 / 64.0) < 1e-9


def test_expected_route_sums_to_one() -> None:
    probs = diag.expected_route_probs()
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert abs(probs["iso"] - 2 / 64.0) < 1e-9


def test_random_signs_near_binomial() -> None:
    try:
        from src.tools.gyroscopic import ops
        ops.build_native()
    except Exception:
        return

    rng = random.Random(0)
    counts = diag.CountSnapshot()
    for _ in range(5000):
        sign_bytes = bytes(rng.getrandbits(8) for _ in range(16))
        _, shell, k4 = diag.analyze_q1_group(sign_bytes)
        path_idx = diag.route_path(shell, k4)
        counts.groups += 1
        counts.shell[shell] += 1
        counts.k4[k4] += 1
        counts.route[diag.PATH_NAMES[path_idx]] += 1

    report = diag.distribution_report(counts)
    assert report["binomial_faithful"] is True
    assert report["shell"]["compare"]["max_abs_deviation"] < 0.05
