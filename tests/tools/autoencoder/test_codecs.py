"""Tests for the deterministic chart codecs (spec 6.1)."""

from __future__ import annotations

import numpy as np

from src.tools.autoencoder.models.narrow import (
    BoundaryChiralityCodec,
    ChiralityOnlyCodec,
    ExactUVCodec,
    ShellOnlyCodec,
)


def test_exact_uv_codec_lossless() -> None:
    codec = ExactUVCodec()
    out = codec.reconstruct_all()
    assert np.array_equal(out, np.arange(4096))


def test_boundary_chirality_codec_lossless() -> None:
    codec = BoundaryChiralityCodec()
    out = codec.reconstruct_all()
    assert np.array_equal(out, np.arange(4096))


def test_chirality_only_codec_64_state_ambiguity() -> None:
    codec = ChiralityOnlyCodec()
    dist = codec.reconstruct_distribution(1234)
    assert abs(float(dist.sum()) - 1.0) < 1e-6
    assert int((dist > 0).sum()) == 64
    # any two states with the same chirality share the same distribution
    chi = codec.encode(1234)
    fiber = codec.fiber(chi)
    assert np.array_equal(codec.reconstruct_distribution(int(fiber[7])), dist)


def test_shell_only_codec_binomial_fibers() -> None:
    codec = ShellOnlyCodec()
    sizes = [len(codec.shell_states(w)) for w in range(7)]
    from math import comb

    assert sizes == [64 * comb(6, w) for w in range(7)]
    dist = codec.reconstruct_distribution(0)
    assert abs(float(dist.sum()) - 1.0) < 1e-6
