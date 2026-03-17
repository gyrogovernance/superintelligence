from __future__ import annotations

import warnings


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*rope_config_validation.*:FutureWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*SwigPyPacked has no __module__ attribute.*:DeprecationWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*SwigPyObject has no __module__ attribute.*:DeprecationWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*swigvarlink has no __module__ attribute.*:DeprecationWarning",
    )

    warnings.filterwarnings(
        "ignore",
        message=".*rope_config_validation.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*SwigPyPacked has no __module__ attribute.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*SwigPyObject has no __module__ attribute.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*swigvarlink has no __module__ attribute.*",
        category=DeprecationWarning,
    )
