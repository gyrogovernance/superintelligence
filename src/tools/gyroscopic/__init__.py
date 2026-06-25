"""Gyroscopic: kernel and llama.cpp gravity-scale hook."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import (
        GyroscopicLLMConfig,
        get_gyroscopic_llm_config,
        repo_root,
        resolve_gguf_path,
        resolve_llama_cli_path,
        resolve_llama_perplexity_path,
    )
    from .loader import (
        build_llama_cli_command,
        build_llama_perplexity_command,
        parse_llama_perplexity_output,
        run_llama_cli,
        run_llama_cli_smoke,
        run_llama_cli_version,
        run_llama_perplexity,
    )
    from .ops import (
        apply_K4,
        build_native,
        chirality_from_signs64,
        gravity_g1,
        gravity_scale,
        step_omega12,
    )

_LAZY_SPEC: tuple[tuple[str, str], ...] = (
    ("GyroscopicLLMConfig", "config"),
    ("get_gyroscopic_llm_config", "config"),
    ("repo_root", "config"),
    ("resolve_gguf_path", "config"),
    ("resolve_llama_cli_path", "config"),
    ("resolve_llama_perplexity_path", "config"),
    ("build_llama_cli_command", "loader"),
    ("build_llama_perplexity_command", "loader"),
    ("parse_llama_perplexity_output", "loader"),
    ("run_llama_cli", "loader"),
    ("run_llama_cli_smoke", "loader"),
    ("run_llama_cli_version", "loader"),
    ("run_llama_perplexity", "loader"),
    ("apply_K4", "ops"),
    ("build_native", "ops"),
    ("chirality_from_signs64", "ops"),
    ("gravity_g1", "ops"),
    ("gravity_scale", "ops"),
    ("step_omega12", "ops"),
)

_LAZY_EXPORTS: dict[str, str] = dict(_LAZY_SPEC)


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module_name}"), name)


__all__ = (
    "GyroscopicLLMConfig",
    "get_gyroscopic_llm_config",
    "repo_root",
    "resolve_gguf_path",
    "resolve_llama_cli_path",
    "resolve_llama_perplexity_path",
    "build_llama_cli_command",
    "build_llama_perplexity_command",
    "parse_llama_perplexity_output",
    "run_llama_cli",
    "run_llama_cli_smoke",
    "run_llama_cli_version",
    "run_llama_perplexity",
    "apply_K4",
    "build_native",
    "chirality_from_signs64",
    "gravity_g1",
    "gravity_scale",
    "step_omega12",
)

assert frozenset(__all__) == frozenset(_LAZY_EXPORTS)
