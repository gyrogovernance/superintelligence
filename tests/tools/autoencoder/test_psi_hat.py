"""Tests for the Agrawal psi-hat symmetry readout (evals_metrics.psi_hat)."""

from __future__ import annotations

import numpy as np
import torch

from src import api

from src.tools.autoencoder.helpers.evals_metrics import psi_hat
from src.tools.autoencoder.models.general import K4Autoencoder


def _k4_perm() -> np.ndarray:
    perm = np.empty((4, 4096), dtype=np.int64)
    for i in range(4096):
        s24 = api.omega12_to_state24(api.OmegaState12(u6=(i >> 6) & 63, v6=i & 63))
        for gate_i, gate in enumerate(("id", "S", "C", "F")):
            d = api.state24_to_omega12(api.apply_gate(s24, gate))
            perm[gate_i, i] = (d.u6 << 6) | d.v6
    return perm


def test_psi_hat_unit_for_exact_encoder() -> None:
    """For the exactly equivariant K4 encoder, rho(g) is a signed diagonal
    whose blocks carry the four K4 characters, so psi_hat(g) must equal the
    exact identity psi = sum_blocks w_block * chi_block(g) with w_block the
    latent energy fraction of the block. This makes psi_hat a direct readout
    of the K4 character energy of the latent (the diagnostic the notes
    request) - and |psi| = 1 exactly when the latent carries a single
    character."""
    torch.manual_seed(0)
    model = K4Autoencoder(n_trivial=2, n_sign=2)
    perm = _k4_perm()
    corpus = np.arange(0, 4096, 7)
    signature_perm = {gate_i: perm[gate_i][corpus] for gate_i in range(4)}
    out = psi_hat(model.encoder_eval(), corpus, signature_perm)

    idx = torch.tensor(corpus)
    with torch.no_grad():
        z = model.encode(idx)
        total = z.pow(2).sum(-1)
        weights = {
            name: float((z[:, sl].pow(2).sum(-1) / total).mean())
            for name, sl in model.z_slices.items()
        }
    chars = {
        # block -> (chi(id), chi(S), chi(C), chi(F))
        "z_inv": (1, 1, 1, 1),
        "z_chi": (1, -1, 1, -1),
        "z_shell": (1, 1, -1, -1),
        "z_irrep": (1, -1, -1, 1),
    }
    for gate_i in range(1, 4):
        predicted = sum(weights[name] * chars[name][gate_i] for name in chars)
        assert abs(out[gate_i] - predicted) < 1e-4, (gate_i, out[gate_i], predicted)


def test_psi_hat_trivial_generator_is_plus_one() -> None:
    """The identity is always +1 for any encoder."""
    torch.manual_seed(1)
    model = K4Autoencoder()
    corpus = np.arange(0, 4096, 11)
    out = psi_hat(model.encoder_eval(), corpus, {0: corpus})
    assert out[0] > 0.9999


def test_psi_hat_single_character_encoder_is_signed_one() -> None:
    """A latent carrying exactly one character must give psi = chi(g): the
    chirality indicator is odd under the W2 word (chi -> chi ^ 63), so
    psi = -1 there, while any K4 gate preserves chi and gives +1."""
    from src.family import byte_from_family_micro
    from src.tools.autoencoder.kernel import apply_signature_index

    class ChiEncoder(torch.nn.Module):
        """z(x) = signed chirality-bit sum; odd under chi -> 63 ^ chi."""

        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            chi = torch.bitwise_xor(
                torch.bitwise_right_shift(idx, 6) & 63, idx & 63
            )
            bits = torch.zeros((idx.shape[0], 6))
            for b in range(6):
                bits[:, b] = (
                    (torch.bitwise_right_shift(chi, b) & 1).float() * 2.0 - 1.0
                )
            return bits.sum(dim=-1, keepdim=True)

    enc = ChiEncoder()
    corpus = np.arange(0, 4096, 5)
    # K4 gates all preserve chi on the Omega chart: psi = +1
    perm = _k4_perm()
    for gate_i in range(4):
        out = psi_hat(enc, corpus, {gate_i: perm[gate_i][corpus]})
        assert out[gate_i] > 0.9999, (gate_i, out[gate_i])
    # the W2(m=0) word inverts chi: psi = -1
    w2 = [byte_from_family_micro(0, 0, 6), byte_from_family_micro(1, 0, 6)]
    sig = api.omega_word_signature(w2)
    sig_id = (sig.parity << 12) | (sig.tau_u6 << 6) | sig.tau_v6
    assert (sig.parity, sig.tau_u6, sig.tau_v6) == (0, 63, 0)
    perm_w2 = np.array(
        [apply_signature_index(int(i), sig_id) for i in corpus], dtype=np.int64
    )
    out = psi_hat(enc, corpus, {sig_id: perm_w2})
    assert out[sig_id] < -0.9999


def test_psi_hat_unstructured_encoder_near_zero() -> None:
    """A random untrained MLP gives directionless latents; its psi values sit
    well below the |psi| = 1 signature of a symmetry-carrying encoder."""
    from src.tools.autoencoder.models.narrow import MLPAutoencoder

    torch.manual_seed(3)
    model = MLPAutoencoder(latent_dim=8, hidden_dim=32)
    corpus = np.arange(0, 4096, 13)
    perm = _k4_perm()
    out = psi_hat(model.encoder_fn(), corpus, {3: perm[3][corpus]})
    assert abs(out[3]) < 0.6
