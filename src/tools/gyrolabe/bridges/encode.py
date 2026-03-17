from __future__ import annotations

from dataclasses import dataclass

import torch

from src.sdk import RuntimeOps
from src.tools.gyrolabe import ops as gyrolabe_ops


_POPCOUNT6_LUT = torch.tensor([i.bit_count() for i in range(64)], dtype=torch.long)


def popcount6_tensor(q6: torch.Tensor) -> torch.Tensor:
    """Element-wise popcount for 6-bit values (0..63)."""
    lut = _POPCOUNT6_LUT if q6.device.type == "cpu" else _POPCOUNT6_LUT.to(q6.device)
    return lut[q6.to(torch.long).clamp(0, 63)]


def hidden_geometry_distance6(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Native spectral geometry distance.
    Folds the hidden dimension to 64, applies the exact Walsh-Hadamard transform,
    extracts the sign-bit spectrum, and computes adjacent Hamming distance.
    Zero transcendentals, fully native spectral chart.
    Uses the CPU default WHT path (`wht64`) and skips folding work when H == 64.
    """
    hs = hidden_states.detach().to(torch.float32)

    if hs.ndim != 3:
        raise ValueError(f"hidden_states must be [B,T,H], got shape {tuple(hs.shape)}")

    B, T, H = hs.shape

    if H == 64:
        folded = hs.contiguous()
    elif H > 64:
        usable = (H // 64) * 64
        folded = hs[..., :usable].reshape(B, T, 64, usable // 64).sum(dim=-1)
    else:
        pad = torch.zeros((B, T, 64 - H), dtype=hs.dtype, device=hs.device)
        folded = torch.cat([hs, pad], dim=-1)

    # Move to spectral chart via exact WHT
    spectral = gyrolabe_ops.wht64(folded.contiguous())

    # Extract spectral sign bits (Hamming space)
    sign_bits = spectral < 0
    diff = sign_bits[:, 1:, :] ^ sign_bits[:, :-1, :]
    dist64 = diff.sum(dim=-1).to(torch.int32)

    # exact integer quantization: 0..64 -> 0..6
    dist6 = ((dist64 * 6) + 32) // 64

    out = torch.zeros((B, T), dtype=torch.int32, device=dist6.device)
    out[:, :-1] = torch.clamp(dist6, min=0, max=6)
    return out.cpu()


@dataclass(frozen=True)
class EncodedFields:
    canonical_bytes: torch.Tensor
    valid_mask: torch.Tensor
    boundary_mask: torch.Tensor
    q_class: torch.Tensor
    family: torch.Tensor
    micro_ref: torch.Tensor
    signatures: torch.Tensor
    states: torch.Tensor
    omega12: torch.Tensor
    omega12_valid: torch.Tensor
    chirality6: torch.Tensor
    shell: torch.Tensor
    q_hist64: torch.Tensor
    family_hist4: torch.Tensor
    micro_hist64: torch.Tensor
    shell_hist7: torch.Tensor
    q_weight_hist7: torch.Tensor
    bit_excitation6: torch.Tensor
    boundary_valid_mask: torch.Tensor

    @property
    def valid_count(self) -> int:
        return int(self.valid_mask.sum().item())


@dataclass(frozen=True)
class QBathEstimate:
    """
    Estimated q-bath source parameters from an encoded byte sequence.
    This is nu(q) from CE17, collapsed to radial form nu(j) from CE19.
    """

    q_bath_64: tuple[float, ...]
    q_weight_distribution: tuple[float, ...]
    rho_bath: float
    eta_bath: float
    lam_bath: float
    m_bath: float
    m2_eq_bath: float
    shell_law_bath: tuple[float, ...]
    eta_vec_bath: tuple[float, ...]


def estimate_q_bath(fields: EncodedFields) -> QBathEstimate:
    """
    Estimate the q-bath parameters from the encode-side field extraction.

    The q_weight_hist7 already contains the distribution of q-weights
    across valid positions. This IS the radial bath nu(j).
    The bit_excitation6 gives per-axis excitation rates.
    """
    total = int(fields.q_weight_hist7.sum().item())
    if total == 0:
        return QBathEstimate(
            q_bath_64=(1.0,) + (0.0,) * 63,
            q_weight_distribution=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            rho_bath=0.0,
            eta_bath=1.0,
            lam_bath=0.0,
            m_bath=-1.0,
            m2_eq_bath=64.0,
            shell_law_bath=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            eta_vec_bath=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        )

    total_q = int(fields.q_hist64.sum().item())
    q_dist = tuple((fields.q_hist64.to(torch.float64) / max(1, total_q)).tolist())
    qw = fields.q_weight_hist7.to(torch.float64)
    nu_j = tuple((qw / total).tolist())

    mean_j = sum(j * nu_j[j] for j in range(7))
    rho_bath = mean_j / 6.0
    eta_bath = 1.0 - 2.0 * rho_bath
    m_bath = 2.0 * rho_bath - 1.0
    if rho_bath >= 1.0 - 1e-12:
        lam_bath = 1e12
    else:
        lam_bath = rho_bath / max(1e-12, 1.0 - rho_bath)
    m2_eq_bath = 4096.0 / ((1.0 + eta_bath * eta_bath) ** 6)

    # CE2 shell law from the estimated fugacity.
    # pi_lambda(N) = C(6,N) * lambda^N / (1+lambda)^6
    from math import comb

    denom = (1.0 + lam_bath) ** 6
    shell_law = tuple(
        (comb(6, n) * (lam_bath ** n)) / max(1e-12, denom)
        for n in range(7)
    )

    eta_vec_bath = tuple(
        1.0 - 2.0 * (float(fields.bit_excitation6[i].item()) / total)
        for i in range(6)
    )

    return QBathEstimate(
        q_bath_64=q_dist,
        q_weight_distribution=nu_j,
        rho_bath=rho_bath,
        eta_bath=eta_bath,
        lam_bath=lam_bath,
        m_bath=m_bath,
        m2_eq_bath=m2_eq_bath,
        shell_law_bath=shell_law,
        eta_vec_bath=eta_vec_bath,
    )


def _histogram(values: torch.Tensor, mask: torch.Tensor, bins: int) -> torch.Tensor:
    flat_mask = mask.reshape(-1)
    if flat_mask.numel() == 0 or not bool(flat_mask.any().item()):
        return torch.zeros(bins, dtype=torch.int64, device=values.device)
    flat_vals = values.reshape(-1)[flat_mask]
    return torch.bincount(flat_vals.to(torch.long), minlength=bins)


def patch_lengths_from_boundary_mask(
    boundary_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[int, ...]:
    """
    Only count valid byte positions; collect runs ending at boundary positions.
    If the final valid byte is not marked boundary, include the trailing open run.
    """
    b_mask = boundary_mask.reshape(-1).to(torch.bool)
    v_mask = valid_mask.reshape(-1).to(torch.bool)
    n = b_mask.shape[0]
    lengths: list[int] = []
    run = 0
    for i in range(n):
        if not v_mask[i].item():
            continue
        run += 1
        if b_mask[i].item():
            lengths.append(run)
            run = 0
    if run > 0:
        lengths.append(run)
    return tuple(lengths)


@dataclass(frozen=True)
class ExactBoundaryTrace:
    chi_threshold: int
    chi_distances: list[int]
    boundary_mask: list[bool]
    patch_lengths: tuple[int, ...]


def explain_exact_boundary(
    fields: EncodedFields,
    *,
    chi_threshold: int,
) -> ExactBoundaryTrace:
    states = fields.states.to(torch.int32)
    valid = fields.valid_mask.to(torch.bool)

    if states.ndim == 2:
        states = states[0]
        valid = valid[0]

    states = states.contiguous()
    chi_dist = gyrolabe_ops.chirality_distance_adjacent(states, lookahead=1)
    chi_list = [int(x) for x in chi_dist.tolist()]

    boundary = [(d >= chi_threshold) for d in chi_list]
    if boundary:
        boundary[0] = True

    lengths: list[int] = []
    run = 0
    for i, is_valid in enumerate(valid.tolist()):
        if not is_valid:
            continue
        run += 1
        if i < len(boundary) and boundary[i]:
            lengths.append(run)
            run = 0
    if run > 0:
        lengths.append(run)

    return ExactBoundaryTrace(
        chi_threshold=chi_threshold,
        chi_distances=chi_list,
        boundary_mask=boundary,
        patch_lengths=tuple(lengths),
    )


def extract_encoded_fields(
    canonical_bytes: torch.Tensor,
    valid_mask: torch.Tensor,
    boundary_mask: torch.Tensor,
) -> EncodedFields:
    if (
        canonical_bytes.shape != valid_mask.shape
        or canonical_bytes.shape != boundary_mask.shape
    ):
        raise ValueError(
            "canonical_bytes, valid_mask, boundary_mask must have the same shape, "
            f"got {canonical_bytes.shape}, {valid_mask.shape}, {boundary_mask.shape}"
        )

    q_class, family, micro_ref, signatures, states = gyrolabe_ops.extract_scan(
        canonical_bytes.contiguous().to(torch.uint8)
    )

    omega12, omega12_valid = RuntimeOps.omega12_and_valid_from_states(states.to(torch.int32))
    omega12 = omega12.to(torch.int32)
    omega12_valid = omega12_valid.to(torch.bool)

    chirality6 = torch.bitwise_and(
        torch.bitwise_xor(torch.bitwise_right_shift(omega12, 6), omega12),
        0x3F,
    ).to(torch.long)

    lut = _POPCOUNT6_LUT if chirality6.device.type == "cpu" else _POPCOUNT6_LUT.to(chirality6.device)
    shell = lut[chirality6]

    effective_mask = valid_mask.to(torch.bool) & omega12_valid

    q_hist64 = _histogram(q_class, effective_mask, 64)
    family_hist4 = _histogram(family, effective_mask, 4)
    micro_hist64 = _histogram(micro_ref, effective_mask, 64)
    shell_hist7 = _histogram(shell, effective_mask, 7)

    q6 = (q_class.to(torch.long) & 0x3F).to(q_class.device)
    wt_q = popcount6_tensor(q6).to(torch.long)
    q_weight_hist7 = _histogram(wt_q, effective_mask, 7)

    bit_excitation6 = torch.zeros(6, dtype=torch.int64, device=q_class.device)
    for j in range(6):
        bit_j = (torch.bitwise_right_shift(q6, j) & 1).to(torch.bool)
        bit_excitation6[j] = (effective_mask & bit_j).sum().to(torch.long)
    bit_excitation6 = bit_excitation6.cpu()

    boundary_valid_mask = valid_mask.to(torch.bool) & boundary_mask.to(torch.bool)

    return EncodedFields(
        canonical_bytes=canonical_bytes.to(torch.uint8),
        valid_mask=valid_mask.to(torch.bool),
        boundary_mask=boundary_mask.to(torch.bool),
        q_class=q_class.to(torch.uint8),
        family=family.to(torch.uint8),
        micro_ref=micro_ref.to(torch.uint8),
        signatures=signatures.to(torch.int32),
        states=states.to(torch.int32),
        omega12=omega12,
        omega12_valid=omega12_valid,
        chirality6=chirality6.to(torch.uint8),
        shell=shell.to(torch.uint8),
        q_hist64=q_hist64.cpu(),
        family_hist4=family_hist4.cpu(),
        micro_hist64=micro_hist64.cpu(),
        shell_hist7=shell_hist7.cpu(),
        q_weight_hist7=q_weight_hist7.cpu(),
        bit_excitation6=bit_excitation6,
        boundary_valid_mask=boundary_valid_mask,
    )


