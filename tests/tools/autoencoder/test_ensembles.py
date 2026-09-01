"""Tests for lambda ensembles and the never-broken group (evals_datasets.py)."""

from __future__ import annotations

import numpy as np

from src.tools.autoencoder.helpers.evals_datasets import (
    chi_fiber,
    corpus_shell_histogram,
    is_stabilizer_of_lambda_ensemble,
    lambda_chirality_distribution,
    never_broken_group,
    sample_lambda_corpus,
    w2_not_in_never_broken,
    w2_word_signature_ids,
)


def test_lambda_chirality_distribution_normalizes() -> None:
    for lam in (0.25, 1.0, 4.0):
        p = lambda_chirality_distribution(lam)
        assert np.isclose(p.sum(), 1.0)
        assert (p >= 0).all()
    # lambda=1 is uniform
    p = lambda_chirality_distribution(1.0)
    assert np.allclose(p, 1 / 64)
    # lambda<1 concentrates on wt=0 (the single equality-horizon word)
    p_low = lambda_chirality_distribution(0.01)
    assert p_low[0] > 0.5


def test_chi_fiber_partitions_omega() -> None:
    seen = set()
    for chi in range(64):
        fiber = chi_fiber(chi)
        assert len(fiber) == 64
        for u, v in fiber:
            assert (u ^ v) == chi
            seen.add((u << 6) | v)
    assert len(seen) == 4096


def test_sample_lambda_corpus_matches_expected_shell() -> None:
    """Sampled shells must match the exact binomial law of the ensemble."""
    lam = 0.3
    n = 200_000
    corpus = sample_lambda_corpus(lam, n, seed=1)
    hist = corpus_shell_histogram(corpus) / n
    p = lam / (1 + lam)
    from math import comb

    expected = np.array([comb(6, s) * p**s * (1 - p) ** (6 - s) for s in range(7)])
    # large-sample statistical agreement
    assert np.allclose(hist, expected, atol=5e-3), (hist, expected)


def test_stabilizer_characterization() -> None:
    """Diagonal signatures (t, t) stabilize; off-diagonal ones do not."""
    for t in (0, 1, 17, 63):
        for parity in (0, 1):
            sig_id = (parity << 12) | (t << 6) | t
            assert is_stabilizer_of_lambda_ensemble(sig_id, None), (parity, t)
    # an off-diagonal translation redistributes shells
    sig_id = (0 << 12) | (3 << 6) | 0
    assert not is_stabilizer_of_lambda_ensemble(sig_id, None)
    # the W2(m=0) word signature (0, 63, 0) is not a stabilizer
    assert not is_stabilizer_of_lambda_ensemble((0 << 12) | (63 << 6) | 0, None)


def test_never_broken_group_exhaustive() -> None:
    """Brute force over all 8192 signatures: H = {(0,t,t),(1,t,t)}, order
    128, closed under kernel composition, and W2 never in H."""
    report = never_broken_group()
    assert report["n_signatures"] == 8192
    assert report["order"] == 128
    assert report["matches_diagonal_form"]
    assert report["is_group"]
    assert w2_not_in_never_broken()


def test_w2_signature_ids_all_broken_shape() -> None:
    ids = w2_word_signature_ids()
    assert len(ids) == 64
    # W2(m) = translation (63^m, m): tau_u ^ tau_v = 63 != 0, never diagonal
    for m, sid in enumerate(ids):
        parity, tu, tv = (sid >> 12) & 1, (sid >> 6) & 63, sid & 63
        assert parity == 0
        assert (tu, tv) == (63 ^ m, m)
        assert tu ^ tv == 63
