"""Tests for readouts.py (spec section 2.1-2.10).

Each readout returns kernel-exact targets; tests assert the closed forms.
"""

from __future__ import annotations

import numpy as np
import torch

from src.tools.autoencoder.helpers.evals_metrics import (
    anisotropy_readout,
    climate_readout,
    climate_synthesizer,
    code_readout,
    exact_denoiser_multipliers,
    gauge_character_readout,
    lift32_readout,
    plancherel_consistency,
    z2_sheet_readout,
)
from src.tools.autoencoder.datasets import state_census_arrays


# ---------------------------------------------------------------------------
# 2.1 Climate
# ---------------------------------------------------------------------------


def test_climate_readout_rho_eta_m2() -> None:
    lam = [0.125, 0.5, 1.0, 2.0, 8.0]
    c = climate_readout(lam)
    for i, l in enumerate(lam):
        assert abs(c["rho"][i] - l / (1 + l)) < 1e-9
        assert abs(c["eta"][i] - (1 - l) / (1 + l)) < 1e-9
        # M2 = 64 x participation ratio
        weights = np.array([(chi).bit_count() for chi in range(64)], dtype=np.float64)
        p = np.power(l, weights)
        p /= p.sum()
        m2 = 64.0 / float((p * p).sum())
        assert abs(c["M2"][i] - m2) < 1e-6
    assert c["ensemble_shell_histogram"].shape == (5, 7)
    # the ensemble law is the binomial: for lambda=1, C(6,s)/64
    assert abs(np.asarray(c["ensemble_shell_histogram"][2, 6]) - 1.0 / 64.0) < 1e-12
    assert abs(np.asarray(c["ensemble_shell_histogram"]).sum(axis=1) - 1.0).max() < 1e-9
    assert c["census_shell_histogram"].shape == (7,)
    assert abs(c["census_shell_histogram"].sum() - 1.0) < 1e-9
    assert c["krawtchouk_A"].shape == (5, 7)


def test_plancherel_consistency_exact() -> None:
    census = state_census_arrays()
    hist = np.bincount(census["shell_chi"].astype(np.int64), minlength=7).astype(np.float64)
    hist /= hist.sum()
    err = plancherel_consistency(hist)
    assert abs(err) < 1e-9


# ---------------------------------------------------------------------------
# 2.2 Anisotropy
# ---------------------------------------------------------------------------


def test_anisotropy_eta_wt_law() -> None:
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    a = anisotropy_readout(probs)
    chars = np.array([[(c >> i) & 1 for i in range(6)] for c in range(64)])
    for ch in range(64):
        expected = 1.0
        for i in range(6):
            if chars[ch, i]:
                expected *= 1.0 - 2.0 * probs[i]
        assert abs(a["damping_eta_wt"][ch] - expected) < 1e-9


# ---------------------------------------------------------------------------
# 2.3 Gauge characters
# ---------------------------------------------------------------------------


def test_gauge_character_readout() -> None:
    idx = torch.arange(0, 4096, 137, dtype=torch.long)
    g = gauge_character_readout(idx)
    from src.tools.autoencoder.kernel import k4_action_arrays

    action, fixed = k4_action_arrays()
    # fixed_flags reproduce the kernel K4 fixed-point table exactly
    assert torch.equal(
        g["fixed_flags"], torch.as_tensor(fixed[:, idx.numpy()]).long()
    )
    # identity fixes every state; orbit sizes divide 4
    assert (g["fixed_flags"][0] == 1).all()
    assert set(g["orbit_size"].tolist()) <= {1, 2, 4}


# ---------------------------------------------------------------------------
# 2.4 Z2 sheet
# ---------------------------------------------------------------------------


def test_z2_sheet_readout() -> None:
    idx = torch.arange(0, 4096, 311, dtype=torch.long)
    z = z2_sheet_readout(idx)
    # shell is invariant under the swap gate F (chirality invariant)
    assert np.array_equal(z["shell"], z["shell_swapped"])
    # fixed flags are exact 0/1
    assert set(z["fixed_by_swap"].tolist()) <= {0, 1}
    # over the full register: S fixes 64 diagonal states; the other 4032
    # states form exactly 2016 two-cycles
    full = z2_sheet_readout(torch.arange(4096))
    assert int(full["n_fixed"][0]) == 64
    assert int(full["n_offdiagonal_pairs"][0]) == 2016


# ---------------------------------------------------------------------------
# 2.5 32-bit lift
# ---------------------------------------------------------------------------


def test_lift32_readout() -> None:
    l = lift32_readout()
    assert l["intron"].shape == (256,)
    assert l["shadow_intron_parity"].shape == (256,)
    from src import api

    # Shadow-invariance: intron[b] ^ intron[shadow(b)] is the same value for
    # every byte (a kernel property of the 32-bit lift). The byte-level parity
    # of that constant equals popcount(SHADOW_PARTNER_MASK) mod 2.
    mask = api.SHADOW_PARTNER_MASK
    expected = int(bin(mask).count("1") % 2)
    assert np.all(l["shadow_intron_parity"] == expected)
    for b in range(256):
        partner = int(l["shadow_partner"][b])
        per_byte = bin(int(l["intron"][b]) ^ int(l["intron"][partner])).count("1") % 2
        assert int(l["shadow_intron_parity"][b]) == per_byte


# ---------------------------------------------------------------------------
# 2.6 Code
# ---------------------------------------------------------------------------


def test_code_readout_membership() -> None:
    c = code_readout()
    mask12 = c["mask12"]
    synd = c["syndrome"]
    # syndrome computed via kernel api matches the readout
    from src import api

    for b in range(256):
        assert int(synd[b]) == api.mask12_syndrome(int(mask12[b]))
    assert set(c["c64_membership"].tolist()) <= {0, 1}


# ---------------------------------------------------------------------------
# 2.9 Exact denoiser
# ---------------------------------------------------------------------------


def test_exact_denoiser_multipliers() -> None:
    probs = [0.3] * 6
    mult = exact_denoiser_multipliers(probs)
    assert mult.shape == (64,)
    # isotropic: multiplier for character a is eta^wt(a), eta = 1 - 2*0.3 = 0.4
    eta = 0.4
    for a in range(64):
        expected = eta ** bin(a).count("1")
        assert abs(mult[a] - expected) < 1e-9


# ---------------------------------------------------------------------------
# 2.10 Climate synthesizer
# ---------------------------------------------------------------------------


def test_climate_synthesizer() -> None:
    lam = [0.5, 1.0, 4.0]
    s = climate_synthesizer(lam, n=8192, seed=1)
    assert s["lambda"].shape == (3,)
    assert s["M2_pred"].shape == (3,)
    # sampled shell histogram is a distribution over 7 shells and matches
    # the binomial law within sampling noise
    smp = np.asarray(s["sampled_shell_histogram"][1])
    law = np.asarray(s["law_shell_histogram"][1])
    assert smp.shape == (7,)
    assert abs(smp.sum() - 1.0) < 1e-9
    assert abs(smp.sum() - law.sum()) < 1e-9
    # lambda = 1 is the neutral law: C(6,s)/64
    assert abs(law[6] - 1.0 / 64.0) < 1e-12
    # finite-sample KL is small but nonzero at this budget
    kl = np.asarray(s["kl_divergence"])
    assert np.all(kl < 0.05)
