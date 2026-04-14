"""Gyroscopic LLM helpers and domain layout.

Core package for Gyroscopic execution. Two operational domains:

Domain 1: GyroLabe (numeric transforms / structural operations / matmul)
 - Responsibilities: native transforms, structural operator analysis, QuBEC matmul
   bridge, and low-level vec-dot kernels.
 - Python functions: gyrolabe_<op> (for domain helpers) and gyromatmul_<op>
   (for kernels).
 - Python classes: GyroLabe<Name>.
 - C functions: gyrolabe_<op> or gyromatmul_<op>.
 - Error messages: "GyroLabe: ..."

GyroMatMul is a responsibility namespace inside GyroLabe.

Domain 2: GyroGraph (multicellular telemetry model)
 - Responsibilities: token -> word4 ingestion, Ω stepping, rolling χ/shell/family
   memories, M2 / η climate updates, and trace/replay surfaces.
 - Python functions: gyrograph_<op>.
 - Python classes: GyroGraph<Name>.
 - C functions: gyrograph_<op>.
 - Error messages: "GyroGraph: ..."

Prohibited patterns
 - No public Python name prefixed with bare "gyro_".
 - No public API names using intent-repeating prefixes from old drafts.
 - No public aliases that duplicate an existing canonical name.

Shared C header (`gyrolabe.h`)
 - Shared inline helpers use the "gyro_" prefix and remain C-only.
 - "gyro_" is not part of the Python public API.

This subpackage also wires the native llama.cpp C backend (llama-cli).
QuBEC fast-path behavior is implemented in native ggml/llama.cpp and exposed via
runtime diagnostics, not as a public Python matmul API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import (
    GyroscopicLLMConfig,
    get_gyroscopic_llm_config,
    repo_root,
    resolve_gguf_path,
    resolve_llama_cli_path,
)
from .climate import (
    cell_climate_from_histograms,
    m2_empirical_from_chi_hist,
    m2_equilibrium_from_shell_hist,
    shell_order_parameters_from_hist,
)
from .loader import (
    build_llama_cli_command,
    run_llama_cli,
    run_llama_cli_smoke,
    run_llama_cli_version,
)

if TYPE_CHECKING:
    from .ops import (
        GyroGraphMoment,
        GyroGraphSLCP,
        GyroLabeOperatorReport,
        gyrograph_emit_slcp,
        gyrograph_pack_moment,
        gyrolabe_analyze_operator_64,
    )


def __getattr__(name: str):
    if name in (
        "GyroGraphMoment",
        "GyroGraphSLCP",
        "GyroLabeOperatorReport",
        "gyrograph_emit_slcp",
        "gyrograph_pack_moment",
        "gyrolabe_analyze_operator_64",
    ):
        from . import ops as _ops

        return getattr(_ops, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def gyromatmul_runtime_caps():
    from . import ops as _ops

    return _ops.gyromatmul_runtime_caps()


__all__ = [
    "GyroGraphMoment",
    "GyroGraphSLCP",
    "GyroLabeOperatorReport",
    "GyroscopicLLMConfig",
    "cell_climate_from_histograms",
    "m2_empirical_from_chi_hist",
    "m2_equilibrium_from_shell_hist",
    "shell_order_parameters_from_hist",
    "build_llama_cli_command",
    "get_gyroscopic_llm_config",
    "gyrograph_emit_slcp",
    "gyrograph_pack_moment",
    "gyrolabe_analyze_operator_64",
    "gyromatmul_runtime_caps",
    "repo_root",
    "resolve_gguf_path",
    "resolve_llama_cli_path",
    "run_llama_cli",
    "run_llama_cli_smoke",
    "run_llama_cli_version",
]
