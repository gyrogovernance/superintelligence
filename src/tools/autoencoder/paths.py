"""Define the autoencoder package data locations.

All autoencoder data live under ``src/tools/autoencoder/data/``. The repository
root also has a ``data/`` directory, so using this module keeps package inputs
separate from unrelated project data.

Each generated dataset has its own ``dataset_<word>/`` directory. Model
checkpoints and evaluation reports have dedicated directories. Temporary work
is stored under ``data/tmp/``.

The repository includes the frozen production checkpoints and evaluation
reports needed by genomics synthesis and topology. Source catalogs and derived
datasets remain local and can be regenerated with the relevant ingest or
autoencoder commands.
"""

from __future__ import annotations

from pathlib import Path

# This package's own data directory: src/tools/autoencoder/data/.
DATA_HOME = Path(__file__).resolve().parent / "data"


def dataset_dir(name: str) -> Path:
    """Directory for a labeled dataset, e.g. ``dataset_dir("states")``.

    The folder is ``data/dataset_<name>/``; one word per dataset."""
    return DATA_HOME / f"dataset_{name}"


def checkpoints_dir() -> Path:
    """Trained weights directory: ``data/checkpoints/``."""
    return DATA_HOME / "checkpoints"


def reports_dir() -> Path:
    """Report directory: ``data/reports/``."""
    return DATA_HOME / "reports"


def tmp_dir() -> Path:
    """Scratch directory: ``data/tmp/``."""
    return DATA_HOME / "tmp"


def ensure() -> None:
    """Create the top-level data folders if they do not exist yet."""
    for d in (checkpoints_dir(), reports_dir(), tmp_dir()):
        d.mkdir(parents=True, exist_ok=True)
    for name in (
        "bytes",
        "states",
        "transitions",
        "signatures",
        "actions",
        "embeddings",
        "ensembles",
        "null",
        "genomics",
        "riboseq",
        "topology",
    ):
        dataset_dir(name).mkdir(parents=True, exist_ok=True)
