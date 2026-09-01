"""Evaluation metrics, latent probes, and kernel-exact readouts.

Two parts live in this one module:

- Metrics (section 1): reconstruction / equivariance / transition error
  functions, the Agrawal psi-hat symmetry readout, and the closed-form latent
  probes (1.4b) plus the shadow-invariance metric (1.4d).
- Readouts (section 2): kernel-exact readouts over the autoencoder and the
  kernel - climate, anisotropy, gauge-character, Z2 sheet, 32-bit lift, code,
  the exact/closed-form denoiser, the climate synthesizer, operator-structure
  and genomics scorers, and the spectral diagnostics.

Each readout returns kernel-exact targets. They are pure functions over the
kernel census/constants plus (where relevant) the model's named latent
components; the accompanying test file asserts the closed forms. The
operator-structure analyzer and genomics scorers are built at the adapter
level from public kernel surfaces. Nothing here re-implements the kernel. The
package's authority remains src.api / src.constants / src.family / src.sdk.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch

from src import api, constants
from src.tools.autoencoder.datasets import byte_census_arrays, state_census_arrays
from src.tools.autoencoder.kernel import (
    popcount6,
    signature_inverse_id,
    state24_from_index,
    word_signature_id,
)
from src.tools.autoencoder.models.super import (
    SpectralAutoencoder,
    irrep_block_index,
)
from .evals_datasets import (
    shell_ensemble_labels,
    walsh_multipliers,
)


# ---------------------------------------------------------------------------
# State metrics
# ---------------------------------------------------------------------------


def exact_reconstruction_accuracy(pred_index: np.ndarray, true_index: np.ndarray) -> float:
    return float((pred_index == true_index).mean())


def hamming_error_12(pred_index: np.ndarray, true_index: np.ndarray) -> np.ndarray:
    pu = (pred_index.astype(np.int32) >> 6) & 63
    pv = pred_index.astype(np.int32) & 63
    tu = (true_index.astype(np.int32) >> 6) & 63
    tv = true_index.astype(np.int32) & 63
    du = np.zeros(len(pred_index), dtype=np.int64)
    dv = np.zeros(len(pred_index), dtype=np.int64)
    for bit in range(6):
        du += ((pu >> bit) & 1) != ((tu >> bit) & 1)
        dv += ((pv >> bit) & 1) != ((tv >> bit) & 1)
    return du + dv


def chirality_accuracy(pred_index: np.ndarray, true_index: np.ndarray) -> float:
    pred_chi = ((pred_index >> 6) & 63) ^ (pred_index & 63)
    true_chi = ((true_index >> 6) & 63) ^ (true_index & 63)
    return float((pred_chi == true_chi).mean())


def shell_accuracy(pred_index: np.ndarray, true_index: np.ndarray) -> float:
    pred_shell = popcount6(((pred_index >> 6) & 63) ^ (pred_index & 63))
    true_shell = popcount6(((true_index >> 6) & 63) ^ (true_index & 63))
    return float((pred_shell == true_shell).mean())


def percolation_rank_accuracy(
    pred_rank: np.ndarray, true_rank: np.ndarray
) -> float:
    """Exact-rank fraction on held-out percolation rows.

    ``true_rank`` comes from the kernel's ``restriction_labels`` (Dataset F);
    the learner must predict the GF(2)^6 rank exactly, not merely a correlate.
    """
    return float((np.asarray(pred_rank) == np.asarray(true_rank)).mean())


# ---------------------------------------------------------------------------
# Equivariance metrics
# ---------------------------------------------------------------------------


@torch.inference_mode()
def k4_equivariance_error(
    encoder: torch.nn.Module,
    state_indices: torch.Tensor,
    k4_perm: torch.Tensor,
    latent_action,
) -> dict[str, Any]:
    """Exhaustive encoder equivariance defect over the given states and K4.

    latent_action(gate_i, z) -> rho(g) z. Returns max/mean per gate and total.
    """
    encoder.eval()
    z = encoder(state_indices)
    per_gate_max: list[float] = []
    per_gate_mean: list[float] = []
    for gate_i in range(4):
        transformed = k4_perm[gate_i][state_indices]
        z_g = encoder(transformed)
        rho_z = latent_action(gate_i, z)
        err = (z_g - rho_z).pow(2).sum(dim=-1)
        per_gate_max.append(float(err.max()))
        per_gate_mean.append(float(err.mean()))
    return {
        "max_per_gate": per_gate_max,
        "mean_per_gate": per_gate_mean,
        "max": max(per_gate_max),
        "mean": float(np.mean(per_gate_mean)),
    }


def k4_decoder_equivariance_error(
    model: torch.nn.Module,
    state_indices: torch.Tensor,
    k4_perm: torch.Tensor,
    rho,
) -> dict[str, Any]:
    """Exhaustive decoder equivariance defect over the given states and K4.

    Checks the second half of group equivariance: decoding a latent that has
    been group-rotated must equal permuting the decoded state by the same
    gate, ``D(rho(g) z) = P_g D(z)``. The encoder half is ``k4_equivariance_error``.
    """
    model.eval()
    z = model.encode(state_indices)
    dec = model.decode(z)
    per_gate_max: list[float] = []
    per_gate_mean: list[float] = []
    for gate_i in range(4):
        rho_z = rho(gate_i, z)
        dec_rho = model.decode(rho_z)
        perm = k4_perm[gate_i].long()
        err = (dec_rho.index_select(1, perm) - dec).pow(2).sum(dim=-1)
        per_gate_max.append(float(err.max()))
        per_gate_mean.append(float(err.mean()))
    return {
        "max_per_gate": per_gate_max,
        "mean_per_gate": per_gate_mean,
        "max": max(per_gate_max),
        "mean": float(np.mean(per_gate_mean)),
    }


def generic_full_g_equivariance_error(
    model: torch.nn.Module,
    state_indices: torch.Tensor,
    sig_ids: torch.Tensor,
    apply_signature_index,
) -> dict[str, Any]:
    """Output-permutation full-group equivariance defect for any state autoencoder.

    Used for non-spectral models (mlp, k4, unified-free/k4) that do not expose a
    Walsh carrier. Applies a signature to the input states and to the output
    state logits; equivariance requires the two to match, ``model(g.x) =
    P_g . model(x)``. These models are not equivariant by design, so this
    reports a large defect (the honest symmetry-breaking contrast) instead of
    crashing.
    """
    model.eval()
    x = state_indices.long()
    with torch.inference_mode():
        y_x = model(x)  # invariant under g.x; compute once
        max_err = 0.0
        mean_errs: list[float] = []
        for sig in sig_ids.tolist():
            sig = int(sig)
            transformed = torch.tensor(
                [apply_signature_index(int(i), sig) for i in x.tolist()],
                dtype=torch.long,
            )
            yg = model(transformed)
        # Equivariance is ``model(g.x) = P_g . model(x)``. The canonical spectral
        # check (full_g_equivariance_error) realizes ``P_g . y`` as
        # ``index_add_(1, perm, y)`` which produces ``y[g^{-1}.j]``; we use the
        # same inverse permutation here so the two paths agree. Using the forward
        # permutation ``y[g.j]`` would be correct only for involutions (K4 gates)
        # and would wrongly report a non-zero defect for translations, which are
        # not their own inverse.
        inv_sig = signature_inverse_id(sig)
        perm = torch.tensor(
            [apply_signature_index(j, inv_sig) for j in range(y_x.shape[1])],
            dtype=torch.long,
        )
        err = (yg - y_x.index_select(1, perm)).abs().max(dim=-1).values
        max_err = max(max_err, float(err.max()))
        mean_errs.append(float(err.mean()))
    return {
        "max": max_err,
        "mean": float(np.mean(mean_errs)) if mean_errs else 0.0,
        "forward_max": max_err,
        "forward_mean": float(np.mean(mean_errs)) if mean_errs else 0.0,
    }


# ---------------------------------------------------------------------------
# Transition metrics
# ---------------------------------------------------------------------------


def next_state_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float((pred == true).mean())


def rollout_accuracy(
    step_fn,
    start_indices: np.ndarray,
    bytes_seq: np.ndarray,
    next_state_tables: np.ndarray,
) -> dict[int, float]:
    """Multi-step rollout accuracy at the given sequence lengths."""
    results: dict[int, float] = {}
    for length in (2, 4, 8, 16):
        if length > bytes_seq.shape[1]:
            break
        current = start_indices.copy()
        for t in range(length):
            b = bytes_seq[:, t]
            current = step_fn(current, b)
        truth = start_indices
        for t in range(length):
            truth = next_state_tables[truth, bytes_seq[:, t]]
        results[length] = float((current == truth).mean())
    return results


def transition_k4_equivariance_error(
    model,
    state_indices: torch.Tensor,
    byte: torch.Tensor,
    k4_perm: torch.Tensor,
    byte_k4_perm: torch.Tensor,
    gate_pair: tuple[int, int] = (3, 0),
) -> dict[str, float]:
    """Equivariance of a byte-conditioned transition model under K4.

    The transition law T(state, byte) is equivariant when transforming both
    arguments simultaneously: T(g.x, g.b) = g. T(x, b). On the byte side the
    kernel's exact action is the shadow-partner involution for the F gate
    (verified exhaustively on Omega); pass the corresponding byte permutation
    as byte_k4_perm: [4, 256].

    gate_pair selects the nontrivial gate probed (default F=3 against id=0);
    the model's logits for (g.x, g.b) are compared with the gate-permuted
    logits for (x, b).
    """
    model.eval()
    gate_i, _gate_j = gate_pair
    with torch.inference_mode():
        logits_base = model(state_indices, byte)
        moved_states = k4_perm[gate_i][state_indices]
        moved_bytes = byte_k4_perm[gate_i][byte]
        logits_moved = model(moved_states, moved_bytes)
        logits_permuted = logits_base[:, k4_perm[gate_i].long()]
        err = (logits_moved - logits_permuted).pow(2).sum(dim=-1)
    return {
        "max": float(err.max()),
        "mean": float(err.mean()),
    }


def psi_hat(
    encoder,
    state_indices: np.ndarray,
    signature_perm: dict[int, np.ndarray],
    device: str = "cpu",
    chunk: int = 4096,
) -> dict[int, float]:
    """Agrawal psi-hat: per-generator equivariance similarity readout.

    For each group generator g, psi_hat(g) is the mean cosine similarity
    between encoder outputs on x and on g.x over the corpus:

        psi_hat(g) = mean_x <E(x), E(g.x)> / (|E(x)| |E(g.x)|)

    A perfectly equivariant encoder with orthogonal real representation rho(g)
    yields |psi| = 1. The signed value is the symmetry-breaking order
    parameter for ensemble comparisons. ``signature_perm`` maps each
    generator's signature id to an [N] array of destination indices.
    """
    encoder.eval()

    def _encode_all(idxs: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for start in range(0, idxs.shape[0], chunk):
            parts.append(encoder(idxs[start : start + chunk]))
        return torch.cat(parts, dim=0)

    idx = torch.as_tensor(np.asarray(state_indices, dtype=np.int64), device=device)
    with torch.inference_mode():
        z = torch.nn.functional.normalize(_encode_all(idx), dim=-1)
        z = torch.nan_to_num(z)
        out: dict[int, float] = {}
        for sig_id, perm in signature_perm.items():
            moved = torch.as_tensor(np.asarray(perm, dtype=np.int64), device=device)
            z_g = torch.nn.functional.normalize(_encode_all(moved), dim=-1)
            z_g = torch.nan_to_num(z_g)
            cos = (z * z_g).sum(dim=-1)
            nonzero = z.norm(dim=-1) > 0
            if nonzero.any():
                out[int(sig_id)] = float(cos[nonzero].mean())
            else:
                out[int(sig_id)] = 0.0
    return out


# ---------------------------------------------------------------------------
# Closed-form latent probes (1.4b) and shadow-invariance metric (1.4d)
# ---------------------------------------------------------------------------


def byte_factorization_targets() -> dict[str, np.ndarray]:
    """Kernel-exact census columns that the raw-byte latent should recover."""
    from src.tools.autoencoder.datasets import byte_census_arrays

    census = byte_census_arrays()
    return {
        "family": census["family_u2"].astype(np.int64),
        "micro": census["micro_ref_u6"].astype(np.int64),
        "q6": census["q6"].astype(np.int64),
        "mask12": census["mask12"].astype(np.int64),
        "intron": census["intron_u8"].astype(np.int64),
        "l0_parity": census["l0_parity"].astype(np.int64),
    }


def _bits_of(value: int, width: int) -> list[int]:
    return [int((value >> i) & 1) for i in range(width)]


def factorization_target_matrix() -> np.ndarray:
    """[256, 2+6+6+12+8+1] one-hot-free target vector for closed-form probes."""
    t = byte_factorization_targets()
    rows = []
    for b in range(256):
        rows.append(
            _bits_of(int(t["family"][b]), 2)
            + _bits_of(int(t["micro"][b]), 6)
            + _bits_of(int(t["q6"][b]), 6)
            + _bits_of(int(t["mask12"][b]), 12)
            + _bits_of(int(t["intron"][b]), 8)
            + [int(t["l0_parity"][b])]
        )
    return np.asarray(rows, dtype=np.float64)


def probe_from_latent(latent: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Closed-form least-squares linear probe of ``target`` from ``latent``.

    Returns the predicted target values (no training). Exact on the 256-row
    census when the latent carries the information.
    """
    X = latent.detach().double()
    Y = target.detach().double()
    xtx = X.t() @ X
    xtx = xtx + 1e-8 * torch.eye(xtx.shape[0], dtype=xtx.dtype)
    w = torch.linalg.solve(xtx, X.t() @ Y)
    return (X @ w).float()


def shadow_invariance_error(
    model: torch.nn.Module,
    state_index: torch.Tensor,
    byte: torch.Tensor,
) -> float:
    """Max |model(s, b) - model(s, shadow(b))| under the shadow partner.

    Equals 0 for the exact kernel table and the equivariant transition model,
    nonzero for an untrained MLP.
    """
    from src import api

    shadow = torch.as_tensor(
        [api.shadow_partner_byte(int(b)) for b in byte.tolist()], dtype=byte.dtype
    )
    with torch.no_grad():
        out_b = model(state_index, byte)
        out_s = model(state_index, shadow)
        if out_b.shape[-1] == 4096:
            out_b = torch.nn.functional.softmax(out_b, dim=-1)
            out_s = torch.nn.functional.softmax(out_s, dim=-1)
        return float((out_b - out_s).abs().max())


# ---------------------------------------------------------------------------
# 2.1 Climate readout
# ---------------------------------------------------------------------------


def climate_readout(lambdas: list[float]) -> dict[str, np.ndarray]:
    """rho, eta, M2, shell histograms, Krawtchouk A(r) for the lambda ensemble.

    Closed forms per QuBEC: rho = lambda/(1+lambda), eta = (1-lambda)/(1+lambda),
    M2 = 64 x participation ratio. Two shell histograms are returned:
    ensemble_shell_histogram (the binomial law P(shell=s) = C(6,s) rho^s
    (1-rho)^(6-s)) and census_shell_histogram (the neutral lambda=1 geometry).
    The Krawtchouk A(r) transform is computed against the ensemble histogram.
    """
    labels = shell_ensemble_labels(lambdas)
    out = {k: labels[k] for k in ("lambda", "rho", "eta", "M2")}
    from math import comb

    rho = np.asarray(out["rho"], dtype=np.float64)
    binom = np.array([comb(6, s) for s in range(7)], dtype=np.float64)
    s_axis = np.arange(7, dtype=np.float64)
    terms = binom[None, :] * np.power(rho[:, None], s_axis[None, :]) * np.power(
        1.0 - rho[:, None], (6 - s_axis)[None, :]
    )
    terms = terms / terms.sum(axis=1, keepdims=True)
    out["ensemble_shell_histogram"] = terms
    census = state_census_arrays()
    shell_hist = np.bincount(census["shell_chi"].astype(np.int64), minlength=7).astype(
        np.float64
    )
    shell_hist /= shell_hist.sum()
    out["census_shell_histogram"] = shell_hist
    out["krawtchouk_A"] = np.array(
        [api.shell_krawtchouk_transform_exact(tuple(row.tolist())) for row in terms]
    )
    return out


def plancherel_consistency(shell_hist: np.ndarray) -> float:
    """Plancherel identity for the weighted Krawtchouk transform.

    sum_w C(6,w) f_w^2 == 64 * sum_k C(6,k) A_k^2. Returns the difference.
    """
    from math import comb

    w = np.array([comb(6, i) for i in range(7)], dtype=np.float64)
    f = np.asarray(shell_hist, dtype=np.float64)
    A = np.array(api.shell_krawtchouk_transform_exact(tuple(f.tolist())), dtype=np.float64)
    lhs = float(np.sum(w * f**2))
    rhs = 64.0 * float(np.sum(w * A**2))
    return lhs - rhs


# ---------------------------------------------------------------------------
# 2.2 Anisotropy readout
# ---------------------------------------------------------------------------


def anisotropy_readout(axis_flip_probs: list[float]) -> dict[str, np.ndarray]:
    """Per-axis flip/damping and the radial ``eta^wt`` damping readout.

    The 64-entry ``damping_eta_wt`` is the product over the set bits of
    character ``a`` of the per-axis QuBEC damping parameters ``eta_i``
    (hQVM_QuBEC_Theory.md §9.1): the ``eta^wt`` spectral damping law.
    """
    m = walsh_multipliers(axis_flip_probs)
    eta_vec = m["eta_vec"]
    chars = np.array(
        [[(a >> i) & 1 for i in range(6)] for a in range(64)], dtype=np.int64
    )
    damp = np.ones(64, dtype=np.float64)
    for a in range(64):
        for i in range(6):
            if chars[a, i]:
                damp[a] *= eta_vec[i]
    return {
        "flip_probs": m["flip_probs"],
        "eta_vec": eta_vec,
        "walsh_multiplier": m["walsh_multiplier"],
        "damping_eta_wt": damp,
        "isotropic": m["isotropic"],
    }


# ---------------------------------------------------------------------------
# 2.3 Gauge-character readout
# ---------------------------------------------------------------------------


def gauge_character_readout(state_index: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-state K4 gauge-character data: fixed-point flags and orbit ids."""
    from src.tools.autoencoder.kernel import k4_action_arrays

    action, fixed = k4_action_arrays()
    sel = state_index.detach().cpu().numpy().astype(np.int64)
    fixed_flags = fixed[:, sel]
    orbit_sizes = np.array(
        [len({int(action[g, s]) for g in range(4)}) for s in sel], dtype=np.int64
    )
    return {
        "fixed_flags": torch.as_tensor(fixed_flags, dtype=torch.long),
        "orbit_size": torch.as_tensor(orbit_sizes, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# 2.4 Z2 sheet readout
# ---------------------------------------------------------------------------


def z2_sheet_readout(state_index: torch.Tensor) -> dict[str, np.ndarray]:
    """Z2 sheet structure under the swap gate S (u6 <-> v6)."""
    from src.tools.autoencoder.kernel import k4_action_arrays

    action, _ = k4_action_arrays()
    sel = state_index.detach().cpu().numpy().astype(np.int64)
    census = state_census_arrays()
    swapped = action[1][sel].astype(np.int64)
    fixed_by_S = (swapped == sel).astype(np.int8)
    mask_weight = np.array(
        [bin(int(s >> 6) | int(s & 63)).count("1") for s in sel], dtype=np.int32
    )
    shell = census["shell_chi"][sel].astype(np.int64)
    shell_swapped = census["shell_chi"][swapped].astype(np.int64)
    return {
        "fixed_by_swap": fixed_by_S,
        "mask_weight": mask_weight,
        "shell": shell,
        "shell_swapped": shell_swapped,
        "n_fixed": np.array([int(fixed_by_S.sum())], dtype=np.int64),
        "n_offdiagonal_pairs": np.array(
            [int((fixed_by_S == 0).sum()) // 2], dtype=np.int64
        ),
    }


# ---------------------------------------------------------------------------
# 2.5 32-bit lift readout
# ---------------------------------------------------------------------------


def lift32_readout() -> dict[str, np.ndarray]:
    """Intron and shadow-partner fields plus the byte-level shadow-XOR parity.

    Returns the per-byte intron (``intron``), the kernel's shadow partner
    (``shadow_partner``), and the parity of the byte-level XOR
    ``intron[b] ^ intron[shadow_partner[b]]`` (a kernel-invariance-derived
    scalar per byte).
    """
    census = byte_census_arrays()
    shadow = census["shadow_partner_byte"]
    intron = census["intron_u8"].astype(np.int64)
    shadow_intron_parity = np.array(
        [int(bin(int(intron[b]) ^ int(intron[int(shadow[b])])).count("1") % 2)
         for b in range(256)],
        dtype=np.int8,
    )
    return {
        "intron": intron,
        "shadow_partner": shadow.astype(np.int64),
        "shadow_intron_parity": shadow_intron_parity,
    }


# ---------------------------------------------------------------------------
# 2.6 Code readout
# ---------------------------------------------------------------------------


def code_readout() -> dict[str, np.ndarray]:
    """C64 membership, mask12 syndrome, and the horizon indicator."""
    census = byte_census_arrays()
    mask12 = census["mask12"].astype(np.int64)
    synd = np.array([api.mask12_syndrome(int(m)) for m in mask12], dtype=np.int64)
    c64 = (synd == 0).astype(np.int8)
    return {
        "mask12": mask12,
        "syndrome": synd,
        "c64_membership": c64,
    }


# ---------------------------------------------------------------------------
# 2.9 Exact denoiser
# ---------------------------------------------------------------------------


def exact_denoiser_multipliers(axis_flip_probs: list[float]) -> np.ndarray:
    """Closed-form spectral shrinkage multipliers eta^r under the byte bath."""
    m = walsh_multipliers(axis_flip_probs)
    return m["walsh_multiplier"]


def denoiser_block_multipliers(axis_flip_probs: list[float]) -> np.ndarray:
    """Closed-form per-block codec multipliers for bath-denoising (hQVM_QuBEC_Theory.md §7.2/§9.1).

    The bath flips chirality axes independently with flip probabilities
    ``p_i``, acting on a state as ``(u, v) -> (u ^ d, v ^ d)``. A character
    ``phi_(a, b)`` picks up the sign ``(-1)^((a^b) . d)``, so the posterior
    mean denoiser multiplies coefficient ``(a, b)`` by

        E[(-1)^((a^b) . d)] = prod_i (1 - 2 p_i)^{(a^b)_i}
                            = prod_i eta_i^{(a^b)_i},

    the Walsh multiplier of the carrier frequency ``a ^ b``, with the per-axis
    damping parameters ``eta_i = 1 - 2 p_i`` (hQVM_QuBEC_Theory.md §9.1).
    """
    flip_probs = np.asarray(axis_flip_probs, dtype=np.float64)
    damping = 1.0 - 2.0 * flip_probs
    vals = np.empty(4096, dtype=np.float64)
    for a in range(64):
        carrier = a ^ np.arange(64)
        vals[a * 64 : (a + 1) * 64] = np.array(
            [np.prod([damping[i] for i in range(6) if (c >> i) & 1]) for c in carrier]
        )
    return vals


def denoiser_gain_report(model, axis_flip_probs: list[float]) -> dict[str, float]:
    """Compare a spectral codec's learned gains against the closed form.

    Pairing: for the full ladder the codec has 2080 free block gains, paired
    to the per-block mean closed-form target over the irrep block. For an
    orbit-tied rung (shell_radial, shell_gauge, chirality_gauge) the codec has
    one gain per orbit (indexed by ``model.bottleneck.orbit_index[k]`` for
    coefficient k); the per-orbit target is the mean of the per-block
    closed-form targets over the coefficients belonging to that orbit.

    Returns a machine-checkable pass flag with the same ``tol`` used to gate
    it, so callers (and tests) can certify that "gains track the closed form"
    without reading prose.
    """
    target = denoiser_block_multipliers(axis_flip_probs)  # [4096]
    target_per_coeff = np.asarray(target, dtype=np.float64)
    bid, _ = irrep_block_index()  # bid[a, b] = irrep block id of pair (a, b); [64, 64]
    target_mat = target_per_coeff.reshape(64, 64)
    n_blocks = int(bid.max()) + 1
    block_target = np.array(
        [target_mat[bid == i].mean() for i in range(n_blocks)]
    )
    orbit_index = getattr(model.bottleneck, "orbit_index", None)
    if orbit_index is None:
        # full ladder: one free gain per irrep block; pair by block mean.
        free = np.asarray(model.bottleneck.gain.detach().cpu().numpy(), dtype=np.float64)
        err = free - block_target[: len(free)]
    else:
        # orbit-tied: one free gain per orbit; pair each orbit gain with the
        # mean closed-form target over the coefficients in that orbit.
        free = np.asarray(model.bottleneck.gain.detach().cpu().numpy(), dtype=np.float64)
        orbit_index = np.asarray(orbit_index, dtype=np.int64)
        n_orbits = int(free.shape[0])
        per_orbit_target = np.array(
            [block_target[orbit_index == o].mean() for o in range(n_orbits)]
        )
        err = free - per_orbit_target
    max_abs = float(np.abs(err).max())
    mean_abs = float(np.abs(err).mean())
    tol = 0.2  # gains must track the closed form within 0.2 to be certifiable
    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "n_gains": int(len(free)),
        "pass": bool(np.isfinite(max_abs) and max_abs <= tol),
        "tol": tol,
    }


# ---------------------------------------------------------------------------
# 2.10 Climate synthesizer
# ---------------------------------------------------------------------------


def climate_synthesizer(
    lambdas: list[float], n: int = 2048, seed: int = 0
) -> dict[str, np.ndarray]:
    """Sample lambda-ensemble corpora and compare sampled versus law."""
    from .evals_datasets import (
        corpus_shell_histogram,
        sample_lambda_corpus,
    )

    law = climate_readout(lambdas)
    expected = law["ensemble_shell_histogram"].astype(np.float64)
    rng = np.random.default_rng(seed)
    synt: dict[str, list[float]] = {
        "lambda": [],
        "M2_pred": [],
        "sampled_shell_histogram": [],
        "law_shell_histogram": [],
        "kl_divergence": [],
    }
    for i, lam in enumerate(lambdas):
        states = sample_lambda_corpus(lam, n, seed=int(rng.integers(0, 2**31)))
        hist = corpus_shell_histogram(states).astype(np.float64)
        hist /= hist.sum()
        p = expected[i]
        kl = float(
            np.sum(np.where(p > 1e-12, p * np.log((p + 1e-12) / (hist + 1e-12)), 0.0))
        )
        synt["lambda"].append(lam)
        synt["M2_pred"].append(float(law["M2"][i]))
        synt["sampled_shell_histogram"].append(hist)
        synt["law_shell_histogram"].append(p)
        synt["kl_divergence"].append(kl)
    return {k: np.array(v, dtype=np.float64) for k, v in synt.items()}


# ---------------------------------------------------------------------------
# Operator-structure analyzer and genomics scorers
# ---------------------------------------------------------------------------


def operator_structure() -> dict[str, np.ndarray]:
    """P_Q, D_Q, R_Q per 64x64 block from the spectral model's block index.

    Returns the block-id map and the commutant dimension (exactly 2080).
    """
    bid, pos = irrep_block_index()
    n_blocks = 64 + 2016
    counts = np.bincount(bid.reshape(-1).astype(np.int64), minlength=n_blocks)
    return {
        "block_id": bid,
        "position": pos,
        "block_counts": counts,
        "commutant_dim": np.array([n_blocks], dtype=np.int64),
    }


def genomics_compile(window: Iterable[int]) -> dict[str, np.ndarray]:
    """Deterministic compile of a sequence window into the carrier feature
    record. Maps each byte to its public census columns (family, micro, q,
    mask12, intron)."""
    census = byte_census_arrays()
    arr = np.array(list(window), dtype=np.int64)
    return {
        "family": census["family_u2"][arr],
        "micro": census["micro_ref_u6"][arr],
        "q6": census["q6"][arr],
        "mask12": census["mask12"][arr],
        "intron": census["intron_u8"][arr],
    }


# ---------------------------------------------------------------------------
# Spectral diagnostics
# ---------------------------------------------------------------------------


def walsh_sector_energy(
    model: SpectralAutoencoder, state_index: int
) -> dict[str, float]:
    """Energy per irrep sector of a state's Walsh spectrum.

    Diagonal (1D) sectors and off-diagonal (2D) blocks reported separately.
    """
    bid, _ = irrep_block_index()
    onehot = np.zeros(4096, dtype=np.float32)
    onehot[state_index] = 1.0
    coeff = model.walsh_coefficients(torch.as_tensor(onehot[None]))[0]
    coeff_sq = (coeff**2).detach().numpy().astype(np.float64)
    diag_energy = float(sum(coeff_sq[a * 64 + a] for a in range(64)))
    total = float(coeff_sq.sum())
    return {
        "diag_energy": diag_energy,
        "offdiag_energy": total - diag_energy,
        "total": total,
    }


def shell_distribution_ensemble(
    rng: np.random.Generator, lam: float, n: int
) -> dict[str, Any]:
    """Shell/occupation ensemble P(chi) proportional to lam^wt(chi)."""
    from math import comb

    weights = np.array([comb(6, w) * lam**w for w in range(7)])
    weights /= weights.sum()
    shells = rng.choice(7, size=n, p=weights)
    return {
        "shell": shells,
        "expected_shell": weights @ np.arange(7),
        "shell_variance": float(
            weights @ (np.arange(7) ** 2) - (weights @ np.arange(7)) ** 2
        ),
    }


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


class Readouts:
    """Runs the kernel-exact readouts over a state corpus."""

    def __init__(self, state_index: np.ndarray) -> None:
        self.state_index = np.asarray(state_index)

    def climate(self, lambdas: list[float]) -> dict[str, np.ndarray]:
        return climate_readout(lambdas)

    def anisotropy(self, axis_flip_probs: list[float]) -> dict[str, np.ndarray]:
        return anisotropy_readout(axis_flip_probs)

    def z2_sheet(self) -> dict[str, np.ndarray]:
        return z2_sheet_readout(torch.as_tensor(self.state_index))


class ScaleSuite:
    """Runs the scale items that are exact (not slow)."""

    def operator(self) -> dict[str, np.ndarray]:
        return operator_structure()

    def multicell_equivariance(self, rng: np.random.Generator) -> float:
        from src.tools.autoencoder.models.super import MultiCellSpectral

        return MultiCellSpectral(2).equivariance_check(rng)

    def multicell_product(
        self, cell_states: list[torch.Tensor]
    ) -> dict[str, float]:
        from src.tools.autoencoder.models.super import MultiCellSpectral

        m = MultiCellSpectral(len(cell_states))
        spectrum = m.joint_spectrum(cell_states)
        return {
            "equivariance_max_err": m.product_equivariance_check(cell_states),
            **m.concentration(spectrum),
        }

    def genomics(self, window: Iterable[int]) -> dict[str, np.ndarray]:
        return genomics_compile(window)
