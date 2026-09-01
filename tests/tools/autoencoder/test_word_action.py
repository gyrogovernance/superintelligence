"""Tests for the transition model, word/signature datasets, and splits
(spec 4.6, 5, 6.6, 6.7)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src import api

from src.tools.autoencoder.kernel import sig_id_parts, word_signature_id
from src.tools.autoencoder.helpers.evals_datasets import (
    held_out_q_class_split,
    minimal_representative_words,
    shadow_pair_split,
    stratified_word_sample,
)
from src.tools.autoencoder.models.narrow import (
    TransitionModel,
    WordActionModel,
    compositional_consistency,
)


# ---------------------------------------------------------------------------
# Word/signature datasets
# ---------------------------------------------------------------------------


def test_word_signature_id_convention() -> None:
    word = (0x12, 0x34)
    sig_id = word_signature_id(word)
    parity, tau_u6, tau_v6 = sig_id_parts(sig_id)
    sig = api.omega_word_signature(list(word))
    assert (parity, tau_u6, tau_v6) == (sig.parity, sig.tau_u6, sig.tau_v6)


def test_minimal_representative_words_covers_group() -> None:
    reps = minimal_representative_words(max_len=4)
    assert len(reps) == 8192
    # every representative replays to its signature (decisive check: the
    # stored byte sequence must compile to exactly the key it is filed under)
    for sig_id, word in reps.items():
        assert word_signature_id(list(word)) == sig_id
    # minimality: length-0 only for identity
    assert reps[word_signature_id(())] == b""
    # BFS gives minimal length: no representative has a shorter prefix-word
    # with the same signature (sampled check)
    for sig_id, word in list(reps.items())[100:200]:
        if word:
            for cut in range(1, len(word)):
                shorter = word[cut:]
                if word_signature_id(list(shorter)) == sig_id:
                    pytest.fail(f"non-minimal representative {word!r}")


def test_stratified_word_sample_labels_exact() -> None:
    rows = stratified_word_sample(n_words=256, seed=3)
    assert len(rows) == 256
    for row in rows[:32]:
        word = row["word"]
        assert row["length"] == len(word)
        sig = api.omega_word_signature(list(word))
        assert row["sig_id"] == (sig.parity << 12) | (sig.tau_u6 << 6) | sig.tau_v6
        assert row["q_total"] == int(api.q_word6_for_items(list(word)))
        # end index from start 0 via kernel replay
        omega = api.OmegaState12(u6=0, v6=0)
        for byte in word:
            omega = api.step_omega12_by_byte(omega, int(byte))
        assert row["end_index"] == (omega.u6 << 6) | omega.v6


def test_stratified_sample_is_deterministic() -> None:
    a = stratified_word_sample(n_words=64, seed=5)
    b = stratified_word_sample(n_words=64, seed=5)
    assert [r["word"] for r in a] == [r["word"] for r in b]


def test_stratified_sample_commitment_fields() -> None:
    rows = stratified_word_sample(n_words=128, seed=9)
    for row in rows[:40]:
        word = list(row["word"])
        O, E, tp = api.trajectory_parity_commitment(word)
        assert row["commitment_O"] == int(O)
        assert row["commitment_E"] == int(E)
        assert row["commitment_parity"] == int(tp)
        assert row["provenance_needed"] == 1


def test_same_signature_different_ledger_pairs() -> None:
    from src.tools.autoencoder.helpers.evals_datasets import (
        same_signature_different_ledger_pairs,
    )

    pairs = same_signature_different_ledger_pairs(n_pairs=64, seed=2)
    assert len(pairs) > 0
    for pair in pairs:
        left, right = list(pair["word_left"]), list(pair["word_right"])
        assert word_signature_id(left) == word_signature_id(right)
        assert pair["signature_equal"] == 1
        # the words are genuinely different
        assert left != right
        # commitments are exact kernel values
        O_l, E_l, p_l = api.trajectory_parity_commitment(left)
        assert pair["commitment_left"] == (int(O_l), int(E_l), int(p_l))
        O_r, E_r, p_r = api.trajectory_parity_commitment(right)
        assert pair["commitment_right"] == (int(O_r), int(E_r), int(p_r))


# ---------------------------------------------------------------------------
# Split regimes
# ---------------------------------------------------------------------------


def test_shadow_pair_split_no_leakage() -> None:
    from src.tools.autoencoder.datasets import byte_census_arrays

    census = byte_census_arrays()
    split = shadow_pair_split(census, seed=0)
    train, test = set(split["train"].tolist()), set(split["test"].tolist())
    # no shadow pair spans train and test
    for byte in range(256):
        partner = api.shadow_partner_byte(byte)
        assert not (byte in train and partner in test)
        assert not (byte in test and partner in train)
    assert len(train) + len(test) + len(split["val"].tolist()) == 256


def test_held_out_q_class_split_matches_q_table() -> None:
    split = held_out_q_class_split(n_holdout=8, seed=1)
    assert len(split["train"]) + len(split["test"]) == 256
    q6 = np.array([api.q_word6(b) for b in range(256)])
    train_q = set(q6[split["train"]].tolist())
    test_q = set(q6[split["test"]].tolist())
    assert not (train_q & test_q)
    assert len(test_q) == 8
    # held-out classes keep all 4 members together
    for q in test_q:
        members = [b for b in range(256) if q6[b] == q]
        assert all(b in split["test"].tolist() for b in members)


# ---------------------------------------------------------------------------
# Transition model
# ---------------------------------------------------------------------------


def test_byte_features_shape_and_determinism() -> None:
    from src.tools.autoencoder.models.narrow import byte_features

    byte = torch.arange(8, dtype=torch.long)
    feats = byte_features(byte)
    assert feats.shape == (8, 14)
    assert torch.equal(feats, byte_features(byte))


def test_transition_model_forward_shape() -> None:
    model = TransitionModel(hidden_dim=32)
    idx = torch.arange(16, dtype=torch.long)
    byte = torch.randint(0, 256, (16,))
    logits = model(idx, byte)
    assert logits.shape == (16, 4096)


def test_transition_model_one_step_smoke_training() -> None:
    """The model should fit exact transitions on a small subset quickly."""
    from src.tools.autoencoder.datasets import transition_table

    table = transition_table()
    torch.manual_seed(0)
    model = TransitionModel(hidden_dim=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    idx = torch.arange(0, 4096, 64, dtype=torch.long)  # 64 states
    byte = torch.full((len(idx),), 0x54, dtype=torch.long)
    target = torch.as_tensor(table[idx.numpy(), 0x54].astype(np.int64))
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(idx, byte), target)
        loss.backward()
        opt.step()
    acc = float((model(idx, byte).argmax(-1) == target).float().mean())
    assert acc > 0.95


# ---------------------------------------------------------------------------
# Word/action model
# ---------------------------------------------------------------------------


def test_word_action_model_parity_is_exact() -> None:
    model = WordActionModel(hidden_dim=16)
    words = [b"", b"\x12", b"\x12\x34", b"\x01\x02\x03"]
    preds = model(words)
    parity = preds["parity_logits"].argmax(dim=-1)
    assert parity.tolist() == [0, 1, 0, 1]


def test_word_action_model_learns_single_byte_taus() -> None:
    """Per-byte translation heads must fit the exact per-byte increments."""
    torch.manual_seed(0)
    model = WordActionModel(hidden_dim=64)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    single_bytes = torch.arange(256, dtype=torch.long)
    targets_u = torch.zeros(256, dtype=torch.long)
    targets_v = torch.zeros(256, dtype=torch.long)
    for b in range(256):
        sig = api.omega_word_signature([b])
        targets_u[b] = sig.tau_u6
        targets_v[b] = sig.tau_v6
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        logits = model.byte_logits(single_bytes)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :64], targets_u
        ) + torch.nn.functional.cross_entropy(logits[:, 64:], targets_v)
        loss.backward()
        opt.step()
    logits = model.byte_logits(single_bytes)
    acc_u = float((logits[:, :64].argmax(-1) == targets_u).float().mean())
    acc_v = float((logits[:, 64:].argmax(-1) == targets_v).float().mean())
    assert acc_u > 0.9 and acc_v > 0.9


def test_compositional_consistency_with_exact_heads() -> None:
    """With per-byte memorized exactly, concatenations compose exactly."""
    torch.manual_seed(0)
    model = WordActionModel(hidden_dim=128)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    single_bytes = torch.arange(256, dtype=torch.long)
    targets_u = torch.zeros(256, dtype=torch.long)
    targets_v = torch.zeros(256, dtype=torch.long)
    for b in range(256):
        sig = api.omega_word_signature([b])
        targets_u[b] = sig.tau_u6
        targets_v[b] = sig.tau_v6
    for _ in range(500):
        opt.zero_grad(set_to_none=True)
        logits = model.byte_logits(single_bytes)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :64], targets_u
        ) + torch.nn.functional.cross_entropy(logits[:, 64:], targets_v)
        loss.backward()
        opt.step()
    pairs = [
        (bytes([0x12]), bytes([0x34])),
        (bytes([0xAA]), bytes([0x54])),
        (bytes([0x01, 0x02]), bytes([0x03])),
        (bytes([0xD5]), bytes([0x2B, 0x7E])),
    ]
    consistency = compositional_consistency(model, pairs)
    assert consistency == 1.0
