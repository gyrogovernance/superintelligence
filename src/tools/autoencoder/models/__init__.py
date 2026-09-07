"""Model registry and plug.

This file is the single entry point for constructing any model. It owns
``build_model``, the constructor the CLI and tests use.

The three tier files hold the model classes:

- ``narrow``: codecs, MLP, byte-mechanism predictors, percolation learner
- ``general``: K4Autoencoder, AffineSpectralCodec
- ``super``: Super
"""

from __future__ import annotations

import numpy as np
from torch import nn

from . import general as _general
from . import super as _super

N_STATES = 4096
N_BLOCKS = 2080

MODEL_KINDS = (
    "mlp",
    "k4",
    "super",
    "transition",
    "rawbyte",
    "word",
    "percolation",
)

TIER_MEMBERS = {
    "narrow": ("mlp", "transition", "rawbyte", "word", "percolation"),
    "general": ("k4",),
    "super": ("super",),
}
TIER_MEMBERS["all"] = tuple(MODEL_KINDS)

_HIDDEN_DEFAULTS = {
    "mlp": 64,
    "k4": 64,
    "transition": 128,
    "rawbyte": 256,
    "word": 64,
    "percolation": 128,
    "super": 128,
}


def build_model(
    kind: str,
    symmetry: str | None = None,
    ladder: str | None = None,
    hidden_dim: int | None = None,
    latent_dim: int = 8,
    heads: tuple[str, ...] | list[str] | None = None,
    n_trivial: int | None = None,
    n_sign: int | None = None,
    sector_mask: np.ndarray | None = None,
    orbit_index: np.ndarray | None = None,
    context_dim: int = 64,
    atom_dim: int = 32,
    frame_dim: int = 64,
    signature_mode: str = "exact",
    analytic_grammar: bool = True,
) -> nn.Module:
    """Construct a model instance by kind.

    ``kind`` is one of ``MODEL_KINDS`` or a tier selector
    (``narrow`` / ``general`` / ``super`` / ``all``). For a tier selector this
    returns the first member of that tier.
    """
    del symmetry, heads, sector_mask, orbit_index
    from .narrow import (
        MLPAutoencoder,
        PercolationLearner,
        RawByteTransitionModel,
        TransitionModel,
        WordActionModel,
    )

    if kind in TIER_MEMBERS:
        kind = TIER_MEMBERS[kind][0]

    h = hidden_dim if hidden_dim is not None else _HIDDEN_DEFAULTS.get(kind, 128)
    if kind == "mlp":
        return MLPAutoencoder(latent_dim=latent_dim, hidden_dim=h)
    if kind == "k4":
        return _general.K4Autoencoder(
            n_trivial=n_trivial if n_trivial is not None else 2,
            n_sign=n_sign if n_sign is not None else 2,
            hidden_dim=h,
        )
    if kind == "super":
        return _super.Super(
            context_dim=context_dim,
            atom_dim=atom_dim,
            frame_dim=frame_dim,
            signature_mode=signature_mode,
            analytic_grammar=analytic_grammar,
        )
    if kind == "transition":
        return TransitionModel(hidden_dim=h)
    if kind == "rawbyte":
        return RawByteTransitionModel(hidden_dim=h)
    if kind == "word":
        return WordActionModel(hidden_dim=h)
    if kind == "percolation":
        return PercolationLearner(hidden_dim=h)
    raise ValueError(f"unknown model kind {kind!r}")
