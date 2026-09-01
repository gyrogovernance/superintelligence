"""Full-G spectral equivariance tests (spec 6.5)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tools.autoencoder.kernel import apply_signature_index
from src.tools.autoencoder.models.super import (
    SpectralAutoencoder,
    full_g_equivariance_error,
    irrep_block_index,
    walsh_matrix_64,
)


def test_walsh_matrix_orthogonal() -> None:
    W = torch.as_tensor(walsh_matrix_64())
    product = W @ W.T
    assert torch.allclose(product, 64.0 * torch.eye(64), atol=1e-4)


def test_irrep_blocks_cover_all_frequencies() -> None:
    bid, pos = irrep_block_index()
    assert bid.shape == (64, 64)
    # 64 diagonal blocks with position 0
    for a in range(64):
        assert bid[a, a] == a and pos[a, a] == 0
    # off-diagonal pairs share blocks; (a,b) and (b,a) at positions 0/1
    ids = bid[np.triu_indices(64, k=1)]
    assert len(set(ids.tolist())) == 2016
    for a, b in ((0, 1), (5, 63), (31, 32)):
        assert bid[a, b] == bid[b, a]
        assert pos[a, b] != pos[b, a]


def test_walsh_transform_of_onehot_is_kernel_sign() -> None:
    model = SpectralAutoencoder()
    x = torch.tensor([0, 1, 65, 4095], dtype=torch.long)
    onehot = torch.zeros((4, 4096))
    onehot[torch.arange(4), x] = 1.0
    coeff = model.walsh_coefficients(onehot)
    # C(a, b)(state (u,v)) = (-1)^(dot(a,u)+dot(b,v)) - verify against direct
    W = torch.as_tensor(walsh_matrix_64())
    for row, idx in enumerate(x.tolist()):
        u, v = idx >> 6, idx & 63
        for a in (0, 1, 63):
            for b in (0, 1, 63):
                expected = (W[a, u] * W[b, v]).item()
                assert abs(float(coeff[row, a * 64 + b]) - expected) < 1e-5


def test_roundtrip_onehot_through_transform() -> None:
    model = SpectralAutoencoder()
    x = torch.arange(4096, dtype=torch.long)
    onehot = torch.zeros((4096, 4096))
    onehot[torch.arange(4096), x] = 1.0
    recon = model.inverse_walsh(model.walsh_coefficients(onehot))
    # with all gains open the pipeline must reproduce the input function
    err = (recon - onehot).abs().max()
    assert float(err) < 1e-4


def test_spectral_action_sign_only_matches_translation() -> None:
    model = SpectralAutoencoder()
    # parity-0 signature is a pure translation: coefficient signs only
    x = torch.tensor([0, 123, 4095], dtype=torch.long)
    onehot = torch.zeros((3, 4096))
    onehot[torch.arange(3), x] = 1.0
    c = model.walsh_coefficients(onehot)
    sig_id = (0 << 12) | (0x0F << 6) | 0x21  # parity 0
    transformed = torch.tensor(
        [apply_signature_index(int(i), sig_id) for i in x.tolist()], dtype=torch.long
    )
    onehot_g = torch.zeros((3, 4096))
    onehot_g[torch.arange(3), transformed] = 1.0
    c_g = model.walsh_coefficients(onehot_g)
    rho_c = model.spectral_action(sig_id, c)
    assert torch.allclose(c_g, rho_c, atol=1e-4)


def test_full_g_equivariance_sampled() -> None:
    model = SpectralAutoencoder()
    states = torch.arange(0, 4096, 128, dtype=torch.long)
    sig_ids = torch.tensor([0, 1, 64, 4131, 8191], dtype=torch.long)
    report = full_g_equivariance_error(model, states, sig_ids)
    assert report["max"] < 1e-3


def test_signature_composition_in_spectrum() -> None:
    """rho(g) rho(h) == rho(g*h) on coefficients (group homomorphism)."""
    model = SpectralAutoencoder()
    from src.tools.autoencoder.kernel import sig_id_parts
    from src import api

    x = torch.tensor([0, 777, 2089], dtype=torch.long)
    onehot = torch.zeros((3, 4096))
    onehot[torch.arange(3), x] = 1.0
    c = model.walsh_coefficients(onehot)

    g, h = 4131, 275  # arbitrary signatures
    # compose via kernel
    parity_g, tu_g, tv_g = sig_id_parts(g)
    parity_h, tu_h, tv_h = sig_id_parts(h)
    composed = api.compose_omega_signatures(
        api.OmegaSignature12(parity_g, tu_g, tv_g),
        api.OmegaSignature12(parity_h, tu_h, tv_h),
    )
    packed = (composed.parity << 12) | (composed.tau_u6 << 6) | composed.tau_v6

    rho_gh = model.spectral_action(packed, c)
    rho_g_rho_h = model.spectral_action(g, model.spectral_action(h, c))
    # rho must be a homomorphism: rho(g o h) == rho(g) o rho(h), with the
    # kernel convention that compose(g, h) applies h first.
    assert torch.allclose(rho_g_rho_h, rho_gh, atol=1e-4)


def test_bottleneck_preserves_equivariance() -> None:
    """Sector gains commute with rho(g): gating then action == action then gating."""
    model = SpectralAutoencoder()
    with torch.no_grad():
        model.bottleneck.gain.copy_(torch.rand(2080))
    x = torch.tensor([0, 100, 4000], dtype=torch.long)
    onehot = torch.zeros((3, 4096))
    onehot[torch.arange(3), x] = 1.0
    c = model.walsh_coefficients(onehot)
    sig_id = (1 << 12) | (0x33 << 6) | 0x0F
    transformed = torch.tensor(
        [apply_signature_index(int(i), sig_id) for i in x.tolist()], dtype=torch.long
    )
    onehot_g = torch.zeros((3, 4096))
    onehot_g[torch.arange(3), transformed] = 1.0
    c_g = model.walsh_coefficients(onehot_g)

    gated_then_action = model.spectral_action(
        sig_id, model.bottleneck(c, model.block_id)
    )
    action_then_gated = model.bottleneck(c_g, model.block_id)
    assert torch.allclose(gated_then_action, action_then_gated, atol=1e-4)


# ---------------------------------------------------------------------------
# Frozen sector masks: the lossy-codec ladder
# ---------------------------------------------------------------------------


def test_codec_ladder_masks_structure() -> None:
    from src.tools.autoencoder.models.super import codec_ladder_mask

    full = codec_ladder_mask("full")
    assert full.shape == (2080,) and int(full.sum()) == 2080

    diag = codec_ladder_mask("diagonal")
    assert int(diag.sum()) == 64 and diag[64:].sum() == 0

    trivial = codec_ladder_mask("trivial")
    assert int(trivial.sum()) == 1 and trivial[0] == 1.0

    offdiag = codec_ladder_mask("offdiagonal")
    assert int(offdiag.sum()) == 2016 and offdiag[:64].sum() == 0

    shell = codec_ladder_mask("shell")
    # even-weight a only: wt(0)=0 kept, wt(1)=1 dropped, wt(3)=2 kept,
    # wt(7)=3 dropped, wt(63)=6 kept
    assert int(shell.sum()) == 32
    assert shell[0] == 1.0 and shell[1] == 0.0 and shell[3] == 1.0
    assert shell[7] == 0.0 and shell[63] == 1.0

    with pytest.raises(ValueError):
        codec_ladder_mask("nonsense")


def test_frozen_mask_zeros_sectors_regardless_of_gain() -> None:
    from src.tools.autoencoder.models.super import codec_ladder_mask

    model = SpectralAutoencoder(ladder="trivial")
    with torch.no_grad():
        model.bottleneck.gain.copy_(torch.rand(2080) + 0.5)
    bid = model.block_id
    out_with_mask = model.bottleneck(torch.ones(2, 4096), bid)
    # only block 0 passes
    kept = (bid == 0).nonzero().squeeze(-1)
    assert torch.count_nonzero(out_with_mask[:, kept]) == 2 * len(kept)
    dropped = (bid != 0).nonzero().squeeze(-1)
    assert float(out_with_mask[:, dropped].abs().max().detach()) == 0.0
    # mask wins over gains: removing the mask would give nonzero output
    model.bottleneck.sector_mask.fill_(1.0)
    out_unmasked = model.bottleneck(torch.ones(2, 4096), bid)
    assert float(out_unmasked[:, dropped].abs().max().detach()) > 0.0


def test_ladder_model_poses_rate_distortion_question() -> None:
    """With all gains at 1, the full model is the exact identity; a masked
    model is not, so CE training actually has something to trade off."""
    x = torch.arange(0, 4096, 97)

    full = SpectralAutoencoder()
    recon_full = full(x)
    truth = torch.zeros((len(x), 4096))
    truth[torch.arange(len(x)), x] = 1.0
    assert float((recon_full - truth).abs().max().detach()) < 1e-4

    diag = SpectralAutoencoder(ladder="diagonal")
    recon_diag = diag(x)
    # 64 of 2080 sectors retained: reconstruction must be lossy
    assert float((recon_diag - truth).abs().max().detach()) > 0.1


def test_rate_penalty_ignores_masked_blocks() -> None:
    from src.tools.autoencoder.models.super import codec_ladder_mask

    mask = codec_ladder_mask("trivial")
    model = SpectralAutoencoder(sector_mask=mask)
    with torch.no_grad():
        model.bottleneck.gain.copy_(torch.arange(1.0, 2081.0))
    # only block 0 (gain 1.0) is free; penalty must be 1.0
    assert float(model.bottleneck.rate_penalty().detach()) == 1.0


def test_diag_model_preserves_equivariance_exactly() -> None:
    """The masked codec remains exactly equivariant under the full group."""
    model = SpectralAutoencoder(ladder="diagonal")
    x = torch.tensor([0, 100, 4000], dtype=torch.long)
    onehot = torch.zeros((3, 4096))
    onehot[torch.arange(3), x] = 1.0
    c = model.walsh_coefficients(onehot)
    sig_id = (1 << 12) | (0x33 << 6) | 0x0F
    transformed = torch.tensor(
        [apply_signature_index(int(i), sig_id) for i in x.tolist()], dtype=torch.long
    )
    onehot_g = torch.zeros((3, 4096))
    onehot_g[torch.arange(3), transformed] = 1.0
    c_g = model.walsh_coefficients(onehot_g)
    gated_then_action = model.spectral_action(
        sig_id, model.bottleneck(c, model.block_id)
    )
    action_then_gated = model.bottleneck(c_g, model.block_id)
    assert torch.allclose(gated_then_action, action_then_gated, atol=1e-4)
