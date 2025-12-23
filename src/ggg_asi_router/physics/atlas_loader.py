"""
Router Atlas Loader

Provides utilities to load the canonical atlas artifacts required by the
GGG ASI Alignment Router kernel, using memory-mapped arrays where appropriate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .atlas_builder import AtlasConfiguration, AtlasPaths


@dataclass(frozen=True)
class AtlasHandle:
    """Container for loaded atlas artifacts."""

    ontology: NDArray[np.uint64]
    epistemology: NDArray[np.int32]
    stage_profile: NDArray[np.uint8]
    loop_defects: NDArray[np.uint8]
    aperture: NDArray[np.float32]


def get_default_paths(configuration: Optional[AtlasConfiguration] = None) -> AtlasPaths:
    """
    Return canonical atlas paths for the given configuration.

    If no configuration is supplied, the default Router atlas configuration is
    used, which places artifacts under data/atlas.
    """
    cfg = configuration or AtlasConfiguration()
    return AtlasPaths.from_directory(cfg.output_directory)


def load_atlas(paths: Optional[AtlasPaths] = None, base_dir: Optional[Path] = None) -> AtlasHandle:
    """
    Load atlas artifacts from disk using memory mapping where helpful.

    Args:
        paths: Explicit atlas paths. If None, they are derived from base_dir
            or from the default AtlasConfiguration.
        base_dir: Optional base directory for atlas files.

    Returns:
        AtlasHandle with ontology, epistemology, stage_profile, loop_defects, aperture.
    """
    if paths is None:
        if base_dir is not None:
            paths = AtlasPaths.from_directory(base_dir)
        else:
            paths = get_default_paths()

    ontology = np.load(paths.ontology, mmap_mode="r")
    epistemology = np.load(paths.epistemology, mmap_mode="r")
    stage_profile = np.load(paths.stage_profile, mmap_mode="r")
    loop_defects = np.load(paths.loop_defects, mmap_mode="r")
    aperture = np.load(paths.aperture, mmap_mode="r")

    return AtlasHandle(
        ontology=ontology,
        epistemology=epistemology,
        stage_profile=stage_profile,
        loop_defects=loop_defects,
        aperture=aperture,
    )


