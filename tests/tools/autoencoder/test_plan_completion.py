"""Pins the autoencoder completion plan: LUT scan, posterior, caches, ladders."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.constants import GENE_MAC_REST, step_state_by_byte
from src.tools.autoencoder import kernel
from src.tools.autoencoder.datasets import (
    NullCorpus,
    _build_inverse_transition_table,
    inverse_transition_table,
    transition_table,
)
from src.tools.autoencoder.helpers.evals_metrics import probe_from_latent
from src.tools.autoencoder.helpers.training_super import train_super
from src.tools.autoencoder.models.general import AffineSpectralCodec, resolve_ladder
from src.tools.autoencoder.models.super import (
    ExactHQVMScan,
    Super,
    prefix_factor_table,
    signature_posterior_uv,
)


def test_import_datasets_does_not_build_inverse_table() -> None:
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.tools.autoencoder.datasets as d; "
            "assert d._build_inverse_transition_table.cache_info().currsize == 0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    inverse_transition_table()
    assert _build_inverse_transition_table.cache_info().currsize == 1


def test_transition_tables_are_write_protected() -> None:
    table = transition_table()
    with pytest.raises(ValueError):
        table[0, 0] = 1
    inv = inverse_transition_table()
    with pytest.raises(ValueError):
        inv[0, 0] = 1


def test_lut_scan_matches_kernel_and_word_signature() -> None:
    scan = ExactHQVMScan()
    rng = np.random.default_rng(0)
    ledgers = rng.integers(0, 256, size=(4, 8), dtype=np.int64)
    traj = scan(torch.as_tensor(ledgers, dtype=torch.long))
    for b in range(4):
        state = int(GENE_MAC_REST)
        word = []
        for t in range(8):
            byte = int(ledgers[b, t])
            word.append(byte)
            state = step_state_by_byte(state, byte)
            assert int(traj.state_after[b, t].item()) == int(state)
            assert int(traj.prefix_sig[b, t].item()) == int(kernel.word_signature_id(word))


def test_prefix_factor_table_matches_kernel() -> None:
    rng = np.random.default_rng(1)
    ledgers = rng.integers(0, 256, size=(5, 4), dtype=np.int64)
    tbl = prefix_factor_table(ledgers)
    for i in range(5):
        for t in range(4):
            p, u, v = kernel.signature_from_id(
                kernel.word_signature_id([int(x) for x in ledgers[i, : t + 1]])
            )
            assert (int(tbl[i, t, 0]), int(tbl[i, t, 1]), int(tbl[i, t, 2])) == (p, u, v)


def test_compute_dynamics_default_omits_occupation() -> None:
    model = Super(context_dim=16, atom_dim=8, frame_dim=32)
    ledger = torch.randint(0, 256, (2, 4))
    out = model(ledger)
    assert "next_state_occupation" not in out


def test_signature_posterior_matches_short_word_enumeration() -> None:
    bytes_t = torch.tensor([[10, 20]], dtype=torch.long)
    mask = torch.tensor([[False, True]])
    probs = torch.zeros(1, 256)
    probs[0, 7] = 0.25
    probs[0, 9] = 0.75
    post = signature_posterior_uv(bytes_t, mask, probs)[0]
    tbl = {}
    from src.tools.autoencoder.models.super import _byte_tables

    t = _byte_tables()
    mass = torch.zeros(64, 64)
    for b, w in ((7, 0.25), (9, 0.75)):
        p, u, v = 0, 0, 0
        for bb in (10, b):
            u, v, p = v ^ int(t["tau_u"][bb]), u ^ int(t["tau_v"][bb]), p ^ 1
        mass[u, v] += w
    assert torch.allclose(post, mass, atol=1e-5)


def test_dual_form_probe_when_rows_lt_cols() -> None:
    x = torch.randn(8, 32)
    w = torch.randn(32, 3)
    y = x @ w
    pred = probe_from_latent(x, y)
    assert torch.allclose(pred, y.float(), atol=1e-4)


def test_ladder_aliases_and_shell_climate() -> None:
    assert resolve_ladder("diagonal") == "chirality"
    assert resolve_ladder("shell_radial") == "diagonal_translation_radial"
    assert resolve_ladder("shell") == "w2_invariant"
    climate = AffineSpectralCodec(ladder="shell_climate")
    assert climate.ladder == "shell_climate"
    assert int(climate.bottleneck.sector_mask[:64].sum().item()) == 64
    radial = AffineSpectralCodec(ladder="shell_radial")
    assert radial.ladder == "diagonal_translation_radial"


def test_super_all_completes_one_epoch_and_writes_checkpoint(tmp_path) -> None:
    corpus = NullCorpus()
    model = Super(context_dim=16, atom_dim=8, frame_dim=32)
    stats = train_super(
        model,
        corpus,
        task="super_all",
        epochs=1,
        device="cpu",
        batch_size=16,
        lr=1e-2,
    )
    assert stats["epochs_run"] == 1
    path = tmp_path / "super_all.pt"
    torch.save(
        {
            "extra": {"model_kind": "super", "model_config": model.get_config()},
            "model_state": model.state_dict(),
        },
        path,
    )
    assert path.exists() and path.stat().st_size > 0


def test_encode_decode_returns_byte_logits() -> None:
    from src.tools.autoencoder.models.super import SuperCode

    model = Super(context_dim=16, atom_dim=8, frame_dim=32)
    ledger = torch.randint(0, 256, (4,))
    encoded = model.encode(ledger)
    assert isinstance(encoded, SuperCode)
    assert "ledger" not in encoded.__dataclass_fields__
    logits = model.decode(encoded, length=4)
    assert logits.shape[-1] == 256


def test_seq_head_excludes_climate() -> None:
    from src.tools.autoencoder.helpers.training_super import _loss_from_heads
    from src.tools.autoencoder.helpers.training_losses import LossWeights

    prior = torch.zeros(2, 256)
    climate = torch.full((2, 256), 9.0)
    seq = torch.zeros(2, 256)
    seq[0, 3] = 4.0
    seq[1, 5] = 4.0
    out = {
        "exact_prior_logits": prior,
        "climate_logits": climate,
        "seq_logits": seq,
        "markov_logits": torch.zeros(2, 256),
        "residual_logits": climate + seq,
        "family_logits": torch.zeros(2, 4),
        "payload_logits": torch.zeros(2, 6),
        "parity_logits": torch.zeros(2, 2),
        "tau_u_logits": torch.zeros(2, 64),
        "tau_v_logits": torch.zeros(2, 64),
    }
    targets = {
        "byte": torch.tensor([3, 5]),
        "family": torch.zeros(2, dtype=torch.long),
        "payload": torch.zeros(2, 6),
        "parity": torch.zeros(2, dtype=torch.long),
        "tau_u": torch.zeros(2, dtype=torch.long),
        "tau_v": torch.zeros(2, dtype=torch.long),
    }
    w = LossWeights(state_ce=0.0, byte_ce=1.0, family_ce=0.0, payload_ce=0.0, signature=0.0)
    loss, _ = _loss_from_heads(out, targets, w, arm="seq")
    loss_climate, _ = _loss_from_heads(out, targets, w, arm="climate")
    assert float(loss) < float(loss_climate)


def test_incomplete_prior_frames_shape() -> None:
    corpus = NullCorpus()
    frames = corpus.incomplete_prior_frames(n=12, seed=0, split="train")
    hold = corpus.incomplete_prior_frames(n=12, seed=0, split="holdout")
    assert frames.shape == (12, 4)
    assert hold.shape == (12, 4)
    assert not np.array_equal(frames, hold)


def test_lambda_ceiling_closed_form() -> None:
    from src.tools.autoencoder.helpers.evals_datasets import lambda_ceiling_bits

    assert lambda_ceiling_bits(1.0) == 0.0
    assert abs(lambda_ceiling_bits(4.0) - 1.67) < 0.05


def test_signature_posterior_opt_in() -> None:
    model = Super(context_dim=8, atom_dim=8, frame_dim=32)
    ledger = torch.randint(0, 256, (2, 4))
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[:, 1] = True
    out = model(ledger, mask=mask, hide_masked=True)
    assert "signature_posterior" not in out
    out2 = model(
        ledger, mask=mask, hide_masked=True, compute_signature_posterior=True
    )
    assert out2["signature_posterior"].shape[-2:] == (64, 64)


def test_cli_super_verify_evaluate_return_gates(tmp_path) -> None:
    from src.tools.autoencoder.cli import cmd_evaluate, cmd_verify_equivariance
    from src.tools.autoencoder.helpers.training_super import train_super
    from src.tools.autoencoder.models.super import Super
    from src.tools.autoencoder.datasets import NullCorpus

    corpus = NullCorpus()
    model = Super(context_dim=16, atom_dim=8, frame_dim=32)
    fit = train_super(
        model, corpus, task="masked_frame", epochs=1, device="cpu", batch_size=16, lr=1e-2
    )
    path = tmp_path / "super_cli.pt"
    torch.save(
        {
            "extra": {
                "model_kind": "super",
                "model_config": model.get_config(),
                "fit": fit,
                "gates": {"G1_masked_holdout": 1.0},
                "gate_flags": {"g1_pass": True},
            },
            "model_state": model.state_dict(),
        },
        path,
    )

    class A:
        checkpoint = str(path)
        seed = 0
        report_file = None
        task = None

    assert cmd_verify_equivariance(A()) == 0
    assert cmd_evaluate(A()) == 0


def test_evaluate_gates_lite_runs_measure_channels() -> None:
    """Capacity sweep uses lite=True; λ/Markov/boundary must not stay at defaults."""
    from src.tools.autoencoder.helpers.training_super import evaluate_gates

    model = Super(context_dim=8, atom_dim=8, frame_dim=32)
    corpus = NullCorpus()
    gates = evaluate_gates(
        model, corpus, device="cpu", lite=True, include_probe=False
    )
    assert gates["lambda_ceiling_bits"] > 0.0
    assert "markov_bound_bits" in gates
    assert gates["markov_bound_bits"] > 0.0
    assert "boundary_mean_ambiguity" in gates
