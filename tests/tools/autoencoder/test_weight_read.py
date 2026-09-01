"""Tests for weight_read (Super Walsh operative P_Q + D_Q)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tools.autoencoder.helpers.weight_read import (
    HIDDEN,
    apply_operative,
    embed_tile_4096,
    read_tile_embedded,
    read_weight,
    read_weight_matrix,
    verify_read,
    walsh_forward_vec,
    walsh_inverse_vec,
)
from src.tools.autoencoder.models.super import SpectralAutoencoder


def test_walsh_roundtrip_matches_super() -> None:
    model = SpectralAutoencoder()
    rng = np.random.default_rng(1)
    x = rng.standard_normal(HIDDEN).astype(np.float32)
    with torch.no_grad():
        onehot = torch.zeros(1, HIDDEN, dtype=torch.float32)
        onehot[0] = torch.as_tensor(x, dtype=torch.float32)
        coeff_t = model.walsh_coefficients(onehot)[0].numpy()
        back_t = model.inverse_walsh(torch.as_tensor(coeff_t[None])).numpy()[0]
    coeff = walsh_forward_vec(x)
    back = walsh_inverse_vec(coeff)
    assert np.max(np.abs(coeff - coeff_t)) < 1e-4
    assert np.max(np.abs(back - back_t)) < 1e-4
    assert np.max(np.abs(back - x)) < 1e-4


def test_pq_plus_defect_exact_random_tile() -> None:
    rng = np.random.default_rng(2)
    tile = rng.standard_normal((64, 64))
    W = embed_tile_4096(tile)
    rep = read_tile_embedded(tile)
    stats = verify_read(rep, W, n_random=64)
    assert stats["max_abs_err"] < 1e-9


def test_apply_operative_matches_dense() -> None:
    rng = np.random.default_rng(3)
    W = rng.standard_normal((HIDDEN, HIDDEN)) * 0.01
    rep = read_weight_matrix(W)
    x = rng.standard_normal(HIDDEN)
    y0 = W @ x
    y1 = apply_operative(rep, x)
    assert np.max(np.abs(y0 - y1)) < 1e-8


def test_rectangular_read_exact() -> None:
    rng = np.random.default_rng(4)
    W = rng.standard_normal((128, HIDDEN)) * 0.01
    rep = read_weight(W)
    x = rng.standard_normal(HIDDEN)
    assert np.max(np.abs(W @ x - apply_operative(rep, x))) < 1e-8


def test_embedding_corpus_smoke() -> None:
    from pathlib import Path

    from src.tools.gyroscopic.config import get_gyroscopic_llm_config, resolve_gguf_path

    from src.tools.autoencoder.helpers.weight_read import (
        bonsai_dequant_tensor,
        embedding_corpus,
        evaluate_on_corpus,
        read_weight,
    )

    gguf = resolve_gguf_path(get_gyroscopic_llm_config())
    if not Path(gguf).is_file():
        pytest.skip("Bonsai GGUF not on disk")
    W = bonsai_dequant_tensor(gguf, "blk.0.attn_q.weight", max_rows=256)
    rep = read_weight(W)
    corpus = embedding_corpus(gguf, [100, 200, 300, 400])
    stats = evaluate_on_corpus(rep, W, corpus)
    assert stats["max_rel_err"] < 1e-4


@pytest.mark.skipif(
    not (
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "data"
        / "models"
        / "Bonsai-8B-gguf"
        / "Bonsai-8B-Q1_0.gguf"
    ).is_file(),
    reason="Bonsai GGUF not on disk",
)
def test_bonsai_tile_read() -> None:
    from pathlib import Path

    from src.tools.gyroscopic.config import get_gyroscopic_llm_config, resolve_gguf_path

    from src.tools.autoencoder.helpers.weight_read import bonsai_tile, read_bonsai_tensor_tile

    gguf = resolve_gguf_path(get_gyroscopic_llm_config())
    if not Path(gguf).is_file():
        pytest.skip("Bonsai GGUF not on disk")
    tile = bonsai_tile(gguf, "blk.0.attn_q.weight", 0, 0)
    W = embed_tile_4096(tile)
    rep = read_bonsai_tensor_tile(gguf, "blk.0.attn_q.weight", 0, 0)
    stats = verify_read(rep, W, n_random=32)
    assert stats["max_abs_err"] < 1e-4
