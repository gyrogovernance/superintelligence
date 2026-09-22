"""Unit tests for genomics synthesis helpers (no full-genome runs)."""

from __future__ import annotations

import numpy as np

from src.tools.autoencoder.programs.genomics.genomics import (
    is_pair_inversion_encoding,
    pair_inversion_encoding,
)
from src.tools.autoencoder.programs.genomics.synthesis.censuses import (
    GateResult,
    collision_pairs,
    format_gate,
    permutation_pairs,
)


def test_pair_inversion_encoding_is_certified_orbit():
    enc = pair_inversion_encoding()
    assert is_pair_inversion_encoding(enc)


def test_collision_pairs_share_signature():
    from src.tools.autoencoder.kernel import word_signature_id

    rng = np.random.default_rng(0)
    stream = rng.integers(0, 256, size=5000).tolist()
    pairs = collision_pairs(stream, n_pairs=20, seed=1)
    assert len(pairs) >= 1
    for a, b in pairs:
        assert len(a) == 4 and len(b) == 4
        assert word_signature_id(a) == word_signature_id(b)


def test_permutation_pairs_fix_sig_and_multiset():
    from src.tools.autoencoder.kernel import word_signature_id

    rng = np.random.default_rng(0)
    stream = rng.integers(0, 256, size=8000).tolist()
    pairs = permutation_pairs(stream, n_pairs=40, seed=1)
    assert len(pairs) >= 10
    for a, b in pairs:
        assert a != b
        assert sorted(a) == sorted(b)
        assert word_signature_id(a) == word_signature_id(b)


def test_format_gate_reports_pass_fail():
    g = GateResult(
        name="g4_ecoli",
        n=10,
        trained=0.3,
        random=0.1,
        margin=0.2,
        compile_floor=None,
        passed=True,
        detail="unit",
    )
    lines = format_gate(g)
    assert any("outcome: PASS" in line for line in lines)
    assert any("trained=0.3000" in line for line in lines)


def test_random_constellation_matches_architecture():
    from src.tools.autoencoder.programs.genomics.synthesis.constellation import (
        load_frozen_constellation,
        production_checkpoint_paths,
        random_constellation,
    )

    needed = production_checkpoint_paths()
    if not all(p.exists() for p in needed.values()):
        return  # skip when production weights are absent
    frozen = load_frozen_constellation(device="cpu")
    rnd = random_constellation(seed=0, device="cpu")
    assert frozen.k4.get_config() == rnd.k4.get_config()
    assert frozen.mlp.get_config() == rnd.mlp.get_config()
    assert frozen.super_model.get_config() == rnd.super_model.get_config()
    # At least one Super parameter differs from the trained checkpoint.
    same = True
    for a, b in zip(frozen.super_model.parameters(), rnd.super_model.parameters()):
        if a.shape == b.shape and not bool((a == b).all().item()):
            same = False
            break
    assert same is False
