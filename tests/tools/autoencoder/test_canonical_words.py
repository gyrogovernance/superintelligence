"""Tests for the canonical K4/BU word dataset (Dataset D) and the
percolation dataset (Dataset F) - autoencoder-facing contracts only.

These tests assert the dataset's own structure and label-contract
(its rows, its replay against the kernel for a sampled word, and the
signature ids the dataset records). They are not a re-proof of the
kernel's word-group algebra, which lives in the kernel's own suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import api

from src.tools.autoencoder.kernel import word_signature_id
from src.tools.autoencoder.helpers.evals_datasets import (
    WORD_TYPES,
    canonical_word_dataset,
    canonical_words,
    word_replay_effects,
)


@pytest.fixture(scope="module")
def dataset() -> dict[str, np.ndarray]:
    return canonical_word_dataset()


def test_canonical_words_rows_and_types() -> None:
    rows = canonical_words()
    # 7 word types x 64 micro_refs
    assert len(rows) == 7 * 64
    for wt in WORD_TYPES:
        assert sum(1 for r in rows if WORD_TYPES[r["word_type"]] == wt) == 64


def test_word_k4_differs_from_gate_k4_except_f() -> None:
    """Documented discrepancy: the canonical word K4 {id, W2, W2', F} shares
    {id, F} with the kernel gate K4 {id, S, C, F}, but W2/W2' are parity-0
    translations whereas S/C are parity-1 swap/complement-swap on the Omega
    chart. The prose claim 'W2 acts as gate S' is loose; only the chirality
    action (inversion) and the group structure coincide."""
    from src.tools.autoencoder.kernel import sig_id_parts

    gate_s = sig_id_parts(4096)  # S = (1, 0, 0) swap
    rows = canonical_words()
    w2_sigs = {
        (r["parity"], r["tau_u6"], r["tau_v6"])
        for r in rows
        if WORD_TYPES[r["word_type"]] == "W2"
    }
    assert gate_s not in w2_sigs


def test_word_replay_effects_consistent(dataset) -> None:
    # spot-check one W2 word's replay directly against the kernel
    row = next(r for r in canonical_words() if WORD_TYPES[r["word_type"]] == "W2")
    word = list(row["word"])
    effects = word_replay_effects(word)
    probe = 1234
    omega = api.state24_to_omega12(api.omega12_to_state24(api.OmegaState12(u6=probe >> 6, v6=probe & 63)))
    for byte in word:
        omega = api.step_omega12_by_byte(omega, byte)
    assert int(effects["dest"][probe]) == (omega.u6 << 6) | omega.v6


def test_sig_ids_replay(dataset) -> None:
    for row in canonical_words()[:: 37]:
        assert word_signature_id(list(row["word"])) == row["sig_id"]
