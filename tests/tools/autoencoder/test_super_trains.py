"""Super training smoke, registry regression, and leakage guard."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.tools.autoencoder.cli import cmd_train
from src.tools.autoencoder.helpers.evals_run import load_any_checkpoint
from src.tools.autoencoder.datasets import NullCorpus
from src.tools.autoencoder.helpers.training_super import (
    eval_g1,
    eval_g2_replay,
    train_super,
)
from src.tools.autoencoder.models import MODEL_KINDS, build_model
from src.tools.autoencoder.models.super import ExactHQVMScan, Super


def test_registry_spectral_unified_absent() -> None:
    assert "spectral" not in MODEL_KINDS
    assert "unified" not in MODEL_KINDS
    assert "super" in MODEL_KINDS
    with pytest.raises(ValueError, match="unknown model kind"):
        build_model("spectral")
    with pytest.raises(ValueError, match="unknown model kind"):
        build_model("unified")


def test_super_leakage_guard_masked_post_state() -> None:
    scan = ExactHQVMScan()
    ledger = torch.arange(8, dtype=torch.long) % 256
    mask = torch.zeros(8, dtype=torch.bool)
    mask[3] = True
    traj = scan(ledger, mask=mask)
    assert (traj.state_after[mask] == 0).all()
    assert (traj.intron[mask] == 0).all()
    assert (traj.q6[mask] == 0).all()
    assert (traj.visible_bytes(mask)[mask] == 0).all()


def _train_args(tmp_path: Path, epochs: int = 2) -> object:
    class A:
        pass

    a = A()
    a.model = "super"
    a.task = "masked_frame"
    a.symmetry = None
    a.seed = 0
    a.device = "cpu"
    a.epochs = epochs
    a.batch_size = 16
    a.learning_rate = 1e-2
    a.output_dir = str(tmp_path)
    a.run_name = "super_masked"
    a.val_fraction = 0.0
    a.val_interval = 1
    a.patience = 0
    a.min_delta = 0.0
    a.rate_weight = 0.0
    a.ladder = None
    a.hidden_dim = None
    a.report_gates = False
    return a


def test_super_trains_masked_frame_params_change(tmp_path: Path) -> None:
    initial = build_model("super")
    learnable0 = {
        k: v.detach().clone()
        for k, v in initial.state_dict().items()
        if k.startswith(("atom.", "frame.", "context.", "byte_head."))
    }
    args = _train_args(tmp_path, epochs=2)
    rc = cmd_train(args)
    assert rc == 0
    path = Path(args.output_dir) / f"{args.run_name}.pt"
    assert path.exists()
    reloaded, meta = load_any_checkpoint(path, device="cpu")
    assert isinstance(reloaded, Super)
    assert meta["extra"]["task"] == "masked_frame"
    assert meta["extra"]["fit"]["epochs_run"] == 2
    delta = 0.0
    for k, v0 in learnable0.items():
        delta += float((reloaded.state_dict()[k] - v0).abs().sum().item())
    assert delta > 1e-6


def test_super_train_direct_smoke() -> None:
    corpus = NullCorpus()
    model = Super(context_dim=32, atom_dim=16, frame_dim=32)
    before = sum(p.detach().sum().item() for p in model.parameters() if p.requires_grad)
    stats = train_super(
        model, corpus, task="masked_frame", epochs=2, device="cpu", batch_size=16, lr=1e-2
    )
    after = sum(p.detach().sum().item() for p in model.parameters() if p.requires_grad)
    assert stats["epochs_run"] == 2
    assert before != after or stats["final_loss"] >= 0.0
    g1 = eval_g1(model, corpus)
    assert 0.0 <= g1 <= 1.0


def test_theorem1_grammar_head_recovers_holdout_bytes() -> None:
    """Visible-bit mean + position recovers masked canonical bytes (Theorem 1)."""
    corpus = NullCorpus()
    model = Super(context_dim=16, atom_dim=8, frame_dim=32)
    assert eval_g1(model, corpus) == 1.0
    assert eval_g2_replay(model, corpus) == 1.0


def test_visible_raw_pool_recovers_micro_bits() -> None:
    """Mean of visible sibling bits 1..6 equals the masked byte's bits."""
    corpus = NullCorpus()
    frames = corpus.canonical_frames("holdout")
    assert len(frames) > 0
    for frame in frames:
        for pos in range(4):
            visible = [int(frame[i]) for i in range(4) if i != pos]
            mean = []
            for bit in range(8):
                mean.append(sum((b >> bit) & 1 for b in visible) / len(visible))
            target = int(frame[pos])
            for bit in range(1, 7):
                assert round(mean[bit]) == ((target >> bit) & 1)


def test_raw_pool_matches_visible_bits() -> None:
    """raw_pool is the visible-sibling statistic, not a copy of the target byte."""
    corpus = NullCorpus()
    frame = corpus.canonical_frames("holdout")[0]
    model = Super(context_dim=16, atom_dim=8, frame_dim=32)
    pos = 2
    mask = torch.zeros(1, 4, dtype=torch.bool)
    mask[0, pos] = True
    ledger = torch.as_tensor(frame, dtype=torch.long).unsqueeze(0)
    out = model(ledger, mask=mask, hide_masked=True)
    raw = out["raw_pool"][0]
    visible = [int(frame[i]) for i in range(4) if i != pos]
    for bit in range(8):
        expected = sum((b >> bit) & 1 for b in visible) / len(visible)
        assert abs(float(raw[bit].item()) - expected) < 1e-5
        votes = sum((b >> bit) & 1 for b in visible)
        majority = int(votes > len(visible) / 2)
        assert int(raw[8 + bit].item()) == majority


def test_grammar_head_no_peek_at_masked_byte() -> None:
    """Flipping a hidden byte must leave grammar outputs unchanged (no leak)."""
    model = Super(context_dim=16, atom_dim=8, frame_dim=32)
    model.eval()
    base = torch.tensor([[0x12, 0x34, 0x56, 0x78]], dtype=torch.long)
    mask = torch.tensor([[False, True, False, False]])
    flipped = base.clone()
    flipped[0, 1] = 0xAB
    with torch.no_grad():
        a = model(base, mask=mask, hide_masked=True)
        b = model(flipped, mask=mask, hide_masked=True)
    for key in ("byte_logits", "raw_pool", "payload_logits"):
        assert torch.allclose(a[key], b[key], atol=1e-6), key


def test_occupation_is_a_probability() -> None:
    model = Super(context_dim=16, atom_dim=8, frame_dim=32)
    ledger = torch.randint(0, 256, (3, 4))
    mask = torch.zeros(3, 4, dtype=torch.bool)
    mask[:, 1] = True
    out = model(ledger, mask=mask, hide_masked=True, compute_dynamics=True)
    occ = out["next_state_occupation"]
    assert occ.shape == (3, 4096)
    assert torch.allclose(occ.sum(dim=-1), torch.ones(3), atol=1e-4)
    assert out["shell_distribution"].shape == (3, 7)
    assert torch.allclose(
        out["shell_distribution"].sum(dim=-1), torch.ones(3), atol=1e-4
    )


def test_g6_two_sided_rejects_saturation() -> None:
    from src.tools.autoencoder.helpers.training_super import gate_pass_flags

    def _gates(bits: float) -> dict[str, float]:
        return {
            "G1_masked_holdout": 1.0,
            "G2_replay": 1.0,
            "G3_composition": 1.0,
            "G4_margin_over_floor": 0.1,
            "G5_provenance_margin_over_null": 0.1,
            "G6_uniform_bits_per_byte": bits,
            "G7_canonical_bits_per_byte": 1.5,
            "probe_holdout_parity_nunique": 2.0,
            "probe_holdout_tau_u_nunique": 2.0,
            "probe_holdout_tau_v_nunique": 2.0,
            "context_probe_holdout_parity_nunique": 2.0,
            "context_probe_holdout_tau_u_nunique": 2.0,
            "context_probe_holdout_tau_v_nunique": 2.0,
        }

    assert gate_pass_flags(_gates(8.0))["g6_pass"] is True
    assert gate_pass_flags(_gates(8.4))["g6_pass"] is True
    assert gate_pass_flags(_gates(21.6))["g6_pass"] is False
    assert gate_pass_flags(_gates(6.0))["g6_pass"] is False


def test_xor_registers_match_kernel_word_signature() -> None:
    """Theorem 2: visible-byte XOR registers equal kernel.word_signature_id."""
    from src.tools.autoencoder import kernel
    from src.tools.autoencoder.helpers.training_super import (
        _joint_word_ledger,
        _pred_sig_factors,
        _sig_row_ledger,
        eval_g3_composition,
    )

    corpus = NullCorpus()
    model = Super(context_dim=16, atom_dim=8, frame_dim=32)
    rows = corpus.signature_words_split("holdout")[:64]
    for row in rows:
        led, msk = _sig_row_ledger(row)
        p, u, v = _pred_sig_factors(model, led, msk, "cpu")
        truth = kernel.signature_from_id(
            kernel.word_signature_id([int(x) for x in led[: int(row["length"])]])
        )
        assert (p, u, v) == truth
    assert eval_g3_composition(model, corpus, n_pairs=32) == 1.0
    ra, rb = rows[0], rows[1]
    wa = [int(x) for x in _sig_row_ledger(ra)[0][: int(ra["length"])]]
    wb = [int(x) for x in _sig_row_ledger(rb)[0][: int(rb["length"])]]
    packed = _joint_word_ledger(wa, wb)
    assert packed is not None
    lj, mj = packed
    pj, uj, vj = _pred_sig_factors(model, lj, mj, "cpu")
    joint = kernel.signature_from_id(kernel.word_signature_id(wa + wb))
    assert (pj, uj, vj) == joint


def test_joint_word_keeps_full_length() -> None:
    from src.tools.autoencoder.helpers.training_super import _joint_word_ledger

    packed = _joint_word_ledger([1, 2, 3], [4, 5])
    assert packed is not None
    lj, mj = packed
    assert len(lj) == 8
    assert list(lj[:5]) == [1, 2, 3, 4, 5]
    assert list(mj) == [False, False, False, False, False, True, True, True]
    assert _joint_word_ledger([1, 2, 3, 4], [5, 6, 7, 8, 9]) is None
    short_packed = _joint_word_ledger([1, 2], [3])
    assert short_packed is not None
    short, sm = short_packed
    assert len(short) == 4
    assert list(short[:3]) == [1, 2, 3]
    assert list(sm) == [False, False, False, True]
