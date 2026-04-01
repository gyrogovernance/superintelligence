"""Integration tests: native llama-cli (llama.cpp C backend), GGUF, and config."""

from __future__ import annotations

import pytest

from src.tools.gyroscopic.config import (
    get_gyroscopic_llm_config,
    resolve_gguf_path,
    resolve_llama_cli_path,
)
from src.tools.gyroscopic.loader import run_llama_cli_smoke, run_llama_cli_version


def test_gyroscopic_config_resolves_paths():
    """Config should resolve default GGUF path under repo root."""
    cfg = get_gyroscopic_llm_config()
    path = resolve_gguf_path(cfg)
    assert path.name.endswith(".gguf")
    assert "Qwen3.5-4B" in path.name or path.suffix == ".gguf"


def test_llama_cli_version_when_binary_present():
    """``llama-cli --version`` runs when the C binary exists."""
    cfg = get_gyroscopic_llm_config()
    try:
        exe = resolve_llama_cli_path(cfg)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    assert exe.is_file()
    proc = run_llama_cli_version(cfg)
    assert proc.returncode == 0, proc.stderr
    assert "version" in (proc.stdout + proc.stderr).lower()


def test_llama_cli_smoke_loads_gguf_when_binary_and_weights_present():
    """One token of generation after load (slow on CPU; long timeout)."""
    cfg = get_gyroscopic_llm_config()
    try:
        resolve_llama_cli_path(cfg)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    path = resolve_gguf_path(cfg)
    if not path.is_file():
        pytest.skip(f"GGUF not present: {path}")

    proc = run_llama_cli_smoke(cfg, timeout_sec=900.0)
    assert proc.returncode == 0, f"stderr={proc.stderr!r} stdout={proc.stdout!r}"
