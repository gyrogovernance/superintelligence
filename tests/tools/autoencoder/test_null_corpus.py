"""Stage-0 invariants for the null corpus (permutation atlas)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.constants import GENE_MAC_REST, GENE_MIC_S
from src.tools.autoencoder.datasets import (
    GENE_MAC_SWAPPED,
    build_byte_fiber,
    build_canonical_cycles,
    build_collisions,
    build_depth2_witnesses,
    build_signature_words,
    generate_null_dataset,
    run_null_invariants,
)


@pytest.fixture(scope="module")
def null_arrays():
    byte_fiber = build_byte_fiber()
    cycles = build_canonical_cycles()
    depth2 = build_depth2_witnesses()
    signature_words = build_signature_words()
    collisions = build_collisions(depth2, signature_words)
    return byte_fiber, cycles, depth2, collisions, signature_words


def test_null_stage0_invariants_all_green(null_arrays) -> None:
    byte_fiber, cycles, depth2, collisions, signature_words = null_arrays
    checks = run_null_invariants(
        byte_fiber, cycles, depth2, collisions, signature_words
    )
    failed = [k for k, v in checks.items() if not v]
    assert not failed, failed


def test_archetype_is_flat_zero_intron(null_arrays) -> None:
    byte_fiber, *_ = null_arrays
    aa = byte_fiber[byte_fiber["byte"] == GENE_MIC_S][0]
    assert int(aa["intron"]) == 0
    assert int(aa["is_flat"]) == 1


def test_canonical_cycle_returns_to_rest(null_arrays) -> None:
    _, cycles, *_ = null_arrays
    for m in (0, 1, 7, 63):
        block = cycles[cycles["micro_ref"] == m]
        assert int(block[3]["state_after"]) == GENE_MAC_SWAPPED
        assert int(block[7]["state_after"]) == GENE_MAC_REST


def test_depth2_uniform_witness_count(null_arrays) -> None:
    _, _, depth2, *_ = null_arrays
    _, counts = np.unique(depth2["s2"], return_counts=True)
    assert len(counts) == 4096
    assert np.all(counts == 16)


def test_signature_words_cover_group(null_arrays) -> None:
    *_, signature_words = null_arrays
    assert len(signature_words) == 8192
    assert set(int(x) for x in signature_words["sig_id"]) == set(range(8192))


def test_collision_replay_class_semantics(null_arrays) -> None:
    *_, collisions, _sw = null_arrays
    kinds = {str(k) for k in collisions["kind"]}
    assert kinds >= {"shadow", "same_sig", "same_end"}
    assert "length_a" in collisions.dtype.names


def test_generate_null_writes_manifest(tmp_path: Path) -> None:
    out = generate_null_dataset(tmp_path / "dataset_null")
    assert (out / "manifest.json").exists()
    assert (out / "byte_fiber.npy").exists()
    assert (out / "canonical_cycles.npy").exists()
    assert (out / "depth2_witnesses.npy").exists()
    assert (out / "signature_words.npy").exists()
    assert (out / "collisions.npy").exists()
    assert (out / "measures.json").exists()
