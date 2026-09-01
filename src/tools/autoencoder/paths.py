"""Single source of truth for where the autoencoder keeps its data on disk.

Everything lives under this package's own ``data/`` directory
(``src/tools/autoencoder/data/``) - never the repo-root ``data/``, which is
shared with the rest of the project. Nothing is nested beyond one folder per
class. The layout:

- ``data/dataset_<word>/``  - labeled arrays (bytes, states, transitions,
  signatures, actions, embeddings, ensembles). Files only, manifest beside.
- ``data/checkpoints/``     - trained weights (``<run>_<model>.pt``) + logs.
- ``data/reports/``         - eval / verify / audit JSON, flat.
- ``data/tmp/``             - anything temporary or scratch.

Model-made datasets join the ``dataset_<word>`` family; ``checkpoints/`` holds
weights and logs only. A manifest already records which weights produced a
dataset (``checkpoint_hash``), so provenance is not lost.

Everything here is gitignored and deterministically regenerable, so moving a
folder is a one-command operation, never a file-by-file migration.
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
    ):
        dataset_dir(name).mkdir(parents=True, exist_ok=True)
