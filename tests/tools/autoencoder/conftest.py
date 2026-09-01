"""Test configuration for the autoencoder test suite.

Registers the ``slow`` marker and skips slow tests by default; run them with
``pytest --runslow``.
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    # triton emits "Failed to find CUDA" on CPU-only machines; silence it so it
    # does not clutter the warnings summary during local (no-GPU) test runs.
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Failed to find CUDA:UserWarning",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
