"""Tests for the byte-mechanism module (spec section 1.4).

Covers the raw-byte ablation, closed-form factorization probes, the
shadow-invariance metric, and the depth-4 frame dataset with the real frame
head. These are the learning/measurement tests; the module's kernel-geometry
facts (pair-diagonal masks, flat-byte census, frame parity-zero) are asserted
by the kernel's own suite, not here.
"""

from __future__ import annotations

import numpy as np
import torch

from src import api
from src.tools.autoencoder.kernel import word_signature_id
from src.tools.autoencoder.helpers.evals_metrics import (
    byte_factorization_targets,
    factorization_target_matrix,
    probe_from_latent,
    shadow_invariance_error,
)
from src.tools.autoencoder.helpers.evals_datasets import depth4_frame_dataset
from src.tools.autoencoder.models.narrow import (
    FrameHead,
    RawByteTransitionModel,
    signature_to_bits,
)


# ---------------------------------------------------------------------------
# 1.4a Raw-byte ablation
# ---------------------------------------------------------------------------


def test_raw_byte_model_forward_shape() -> None:
    model = RawByteTransitionModel(hidden_dim=32)
    idx = torch.arange(8, dtype=torch.long)
    byte = torch.randint(0, 256, (8,))
    logits = model(idx, byte)
    assert logits.shape == (8, 4096)


def test_raw_byte_model_learns_transitions_smoke() -> None:
    """At smoke budget the raw-byte model should reduce loss and learn the
    table. This is the 'encodable -> learnable' experiment."""
    from src.tools.autoencoder.datasets import transition_table

    table = transition_table()
    torch.manual_seed(0)
    model = RawByteTransitionModel(hidden_dim=256)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    idx = torch.arange(0, 4096, 16, dtype=torch.long)
    byte = torch.full((len(idx),), 0x54, dtype=torch.long)
    target = torch.as_tensor(table[idx.numpy(), 0x54].astype(np.int64))
    # baseline loss before training
    with torch.no_grad():
        base_loss = float(
            torch.nn.functional.cross_entropy(model(idx, byte), target)
        )
    for _ in range(150):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(idx, byte), target)
        loss.backward()
        opt.step()
    with torch.no_grad():
        final_loss = float(
            torch.nn.functional.cross_entropy(model(idx, byte), target)
        )
        acc = float((model(idx, byte).argmax(-1) == target).float().mean())
    assert final_loss < base_loss  # training loss decreases
    assert acc > 0.5  # raw-byte model learns a meaningful fraction


# ---------------------------------------------------------------------------
# 1.4b Factorization probes (closed-form)
# ---------------------------------------------------------------------------


def test_factorization_targets_shape() -> None:
    t = byte_factorization_targets()
    assert t["family"].shape == (256,)
    assert t["mask12"].shape == (256,)
    assert set(t.keys()) >= {
        "family",
        "micro",
        "q6",
        "mask12",
        "intron",
        "l0_parity",
    }


def test_factorization_probe_recovers_pairs_from_oracle_latent() -> None:
    """The 2+6 split and pair-diagonal mask are recoverable exactly from a
    latent that already carries the oracle census columns (closed form).

    The oracle latent is the bit-expanded census (the 2+6+6+12+8+1 columns),
    so a linear probe over the 256-row census recovers every target bit
    exactly. This is the closed-form 'encodable -> recoverable' check.
    """
    targets = factorization_target_matrix()  # [256, 35], bit-expanded
    latent = torch.as_tensor(targets, dtype=torch.float64)
    pred = probe_from_latent(latent, torch.as_tensor(targets))
    pred_bits = (pred > 0.5).long()
    # all 35 columns (family 2, micro 6, q6 6, mask12 12, intron 8, l0 1)
    # recovered exactly from the oracle latent
    assert torch.equal(pred_bits, torch.as_tensor(targets).long())


# ---------------------------------------------------------------------------
# 1.4d Shadow-invariance metric
# ---------------------------------------------------------------------------


def test_shadow_invariance_zero_for_exact_table() -> None:
    """The exact kernel transition table is shadow-invariant by construction."""
    from src.tools.autoencoder.datasets import transition_table

    table = transition_table()

    # model that returns one-hot next-state logits (exact table lookup)
    class ExactTableModel(torch.nn.Module):
        def __init__(self, table: np.ndarray) -> None:
            super().__init__()
            self.table = table

        def __call__(self, state_index: torch.Tensor, byte: torch.Tensor):
            out = torch.full(
                (len(state_index), 4096), float("-inf")
            )
            for i, (s, b) in enumerate(zip(state_index.tolist(), byte.tolist())):
                out[i, int(self.table[s, b])] = 0.0
            return out

    model = ExactTableModel(table)
    idx = torch.arange(0, 4096, 137, dtype=torch.long)
    byte = torch.full((len(idx),), 0x51, dtype=torch.long)
    err = shadow_invariance_error(model, idx, byte)
    assert err == 0.0


def test_shadow_invariance_nonzero_for_untrained_mlp() -> None:
    model = RawByteTransitionModel(hidden_dim=16)
    torch.manual_seed(1)
    idx = torch.arange(0, 4096, 200, dtype=torch.long)
    byte = torch.randint(0, 256, (len(idx),))
    err = shadow_invariance_error(model, idx, byte)
    assert err > 0.0


# ---------------------------------------------------------------------------
# 1.4f Depth-4 frames + real frame head
# ---------------------------------------------------------------------------


def test_depth4_frame_signature_consistency() -> None:
    """The compiled frame signature equals the kernel word signature, and the
    frame strictly determines the final state."""
    frames = depth4_frame_dataset(64, seed=7)
    for row in range(len(frames["bytes"])):
        b = [int(x) for x in frames["bytes"][row]]
        assert frames["frame_signature"][row] == word_signature_id(b)
        # staged final state matches kernel replay
        omega = api.OmegaState12(u6=0, v6=0)
        for byte in b:
            omega = api.step_omega12_by_byte(omega, byte)
        assert frames["final_state"][row] == (omega.u6 << 6) | omega.v6


def test_frame_head_learns_signature_smoke() -> None:
    """Real frame head at smoke budget: predicts the 13-bit frame signature
    (parity, tau_u6, tau_v6) from the 32-bit intron sequence, which is the
    identifying frame record (Formalism 6.3)."""
    frames = depth4_frame_dataset(512, seed=11)
    torch.manual_seed(0)
    model = FrameHead(hidden_dim=128)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    i32 = torch.as_tensor(frames["intron_seq32"], dtype=torch.long)
    sig_bits = torch.as_tensor(
        np.array([signature_to_bits(int(s)) for s in frames["frame_signature"]]),
        dtype=torch.float32,
    )
    for _ in range(700):
        opt.zero_grad(set_to_none=True)
        out = model(i32)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out["signature_logits"], sig_bits
        )
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (model(i32)["signature_logits"] > 0).long().numpy()
    truth = sig_bits.long().numpy()
    bit_acc = float((pred == truth).mean())
    assert bit_acc > 0.95
