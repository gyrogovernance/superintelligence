"""Pytest configuration for kernel tests."""

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on path (imports are src.api, src.constants, etc.)
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))


def _configure_offline_bolmo_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    try:
        from src.tools.gyrolabe.bridges.bolmo_config import (
            DEFAULT_BOLMO_MODEL_PATH,
            configure_bolmo_offline_loading,
        )
        if DEFAULT_BOLMO_MODEL_PATH.exists():
            configure_bolmo_offline_loading(DEFAULT_BOLMO_MODEL_PATH)
    except Exception:
        pass


def pytest_configure(config):
    _configure_offline_bolmo_env()
    config.addinivalue_line("markers", "research: research/diagnostic tests (run with -m research)")

