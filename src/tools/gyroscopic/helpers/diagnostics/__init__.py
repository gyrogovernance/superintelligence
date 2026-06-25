"""Bench and offline diagnostics for Q1_0 GGUF structure."""

from __future__ import annotations

from . import gguf, kernel, route, tiles
from .gguf import (
    GGML_TYPE_Q1_0,
    GgufFile,
    GgufTensor,
    Q1_0_BLOCK_BYTES,
    open_gguf,
    iter_q1_sign_bytes,
    iter_q1_tiles,
)
from .kernel import (
    GyroAccum,
    analyze_q1_group,
    depth4_bu_factor,
    extract_phase_native,
    gyro_accum_t,
    k4_compose_gyroacc,
    route_path,
)
from .route import (
    PATH_NAMES,
    CountSnapshot,
    distribution_report,
    expected_route_probs,
    expected_shell_probs,
    run_route_diagnostic,
)
from .tiles import (
    decompose_tile,
    decompose_tile_gyro_halves,
    project_chi,
    project_shell,
    random_tile_reference,
    summarize_gyro_halves,
    summarize_ratios,
)

__all__ = (
    "GGML_TYPE_Q1_0",
    "GgufFile",
    "GgufTensor",
    "GyroAccum",
    "PATH_NAMES",
    "Q1_0_BLOCK_BYTES",
    "CountSnapshot",
    "analyze_q1_group",
    "decompose_tile",
    "decompose_tile_gyro_halves",
    "depth4_bu_factor",
    "distribution_report",
    "expected_route_probs",
    "expected_shell_probs",
    "extract_phase_native",
    "gguf",
    "gyro_accum_t",
    "iter_q1_sign_bytes",
    "iter_q1_tiles",
    "k4_compose_gyroacc",
    "kernel",
    "open_gguf",
    "project_chi",
    "project_shell",
    "random_tile_reference",
    "route",
    "route_path",
    "run_route_diagnostic",
    "summarize_gyro_halves",
    "summarize_ratios",
    "tiles",
)
