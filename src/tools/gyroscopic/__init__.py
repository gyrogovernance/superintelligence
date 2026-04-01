"""Gyroscopic LLM helpers: native llama.cpp C backend (``llama-cli``)."""

from src.tools.gyroscopic.config import (
    GyroscopicLLMConfig,
    get_gyroscopic_llm_config,
    repo_root,
    resolve_gguf_path,
    resolve_llama_cli_path,
)
from src.tools.gyroscopic.loader import (
    build_llama_cli_command,
    run_llama_cli,
    run_llama_cli_smoke,
    run_llama_cli_version,
)

__all__ = [
    "GyroscopicLLMConfig",
    "build_llama_cli_command",
    "get_gyroscopic_llm_config",
    "repo_root",
    "resolve_gguf_path",
    "resolve_llama_cli_path",
    "run_llama_cli",
    "run_llama_cli_smoke",
    "run_llama_cli_version",
]
