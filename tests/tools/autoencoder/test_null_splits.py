"""NullCorpus structured splits and ExactHQVMScan leakage guard."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tools.autoencoder.datasets import (
    NullCorpus,
    verify_ledger_disjoint_splits,
)
from src.tools.autoencoder.models.super import ExactHQVMScan


@pytest.fixture(scope="module")
def corpus() -> NullCorpus:
    return NullCorpus()


def test_micro_holdout_sizes(corpus: NullCorpus) -> None:
    assert len(corpus.holdout_micros) == 16
    assert len(corpus.train_micros) == 48
    assert set(corpus.train_micros).isdisjoint(set(corpus.holdout_micros))
    frames_tr = corpus.canonical_frames("train")
    frames_ho = corpus.canonical_frames("holdout")
    assert frames_tr.shape == (48, 4)
    assert frames_ho.shape == (16, 4)
    cycles_tr = corpus.canonical_cycles("train")
    assert cycles_tr.shape == (48, 8)


def test_collision_ledgers_disjoint_across_splits(corpus: NullCorpus) -> None:
    assert verify_ledger_disjoint_splits(
        corpus.collisions_split, corpus.ledger_split
    )
    train_keys = set()
    hold_keys = set()
    for split, bucket in (("train", train_keys), ("holdout", hold_keys)):
        for row in corpus.collision_pairs(split):
            from src.tools.autoencoder.datasets import _row_ledger_keys

            ka, kb = _row_ledger_keys(row)
            assert corpus.ledger_split[ka] == split
            assert corpus.ledger_split[kb] == split
            bucket.add(ka)
            bucket.add(kb)
    assert train_keys.isdisjoint(hold_keys)


def test_exact_scan_mask_leakage_guard_full_fields() -> None:
    """Masked steps publish zeros for every byte-derived field."""
    scan = ExactHQVMScan()
    ledger = torch.tensor([0xAA, 0x55, 0xF0, 0x0F], dtype=torch.long)
    mask = torch.tensor([False, True, False, False])
    traj = scan(ledger, mask=mask)
    assert int(traj.state_after[1].item()) == 0
    assert int(traj.chi_shell[1].item()) == 0
    assert int(traj.arch_shell[1].item()) == 0
    assert int(traj.prefix_sig[1].item()) == 0
    assert int(traj.intron[1].item()) == 0
    assert int(traj.family_phase[1].item()) == 0
    assert int(traj.micro_ref[1].item()) == 0
    assert int(traj.q6[1].item()) == 0
    assert int(traj.fold_disagreement[1].item()) == 0
    visible = traj.visible_bytes(mask)
    assert int(visible[1].item()) == 0


def test_flip_masked_byte_visible_invariant() -> None:
    scan = ExactHQVMScan()
    base = torch.tensor([0x12, 0x34, 0x56, 0x78], dtype=torch.long)
    mask = torch.tensor([False, True, False, False])
    flipped = base.clone()
    flipped[1] = 0xAB
    a = scan(base, mask=mask)
    b = scan(flipped, mask=mask)
    for name in (
        "intron",
        "family_phase",
        "micro_ref",
        "q6",
        "state_before",
        "state_after",
        "chi_shell",
        "arch_shell",
        "prefix_sig",
        "fold_disagreement",
    ):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert torch.equal(a.visible_bytes(mask), b.visible_bytes(mask))


def test_corruption_transfer_levels(corpus: NullCorpus) -> None:
    assert corpus.corruption_n_bytes("train") == 1
    assert corpus.corruption_n_bytes("holdout") == 2


def test_frame_position_stratification(corpus: NullCorpus) -> None:
    idx = corpus.frame_position_indices(2)
    assert len(idx) == 64 * 2
    values = np.arange(len(corpus.cycles))
    by_pos = corpus.stratify_by_frame_position(
        values, corpus.cycles["phase_position"]
    )
    assert set(by_pos) == {0, 1, 2, 3}


def test_signature_words_holdout_varies(corpus: NullCorpus) -> None:
    hold = corpus.signature_words_split("holdout")
    assert len(hold) > 0
    assert len(np.unique(hold["parity"])) >= 2
    assert len(np.unique(hold["tau_u6"])) >= 2
    assert len(np.unique(hold["tau_v6"])) >= 2
