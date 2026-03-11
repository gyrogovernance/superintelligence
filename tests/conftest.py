"""Pytest configuration for kernel tests."""

import sys
from pathlib import Path

import pytest

# Ensure project root is on path (imports are src.api, src.constants, etc.)
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))


def pytest_configure(config):
    config.addinivalue_line("markers", "research: research/diagnostic tests (run with -m research)")

