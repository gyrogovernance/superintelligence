"""Super process model."""

from __future__ import annotations

from dataclasses import dataclass

from functools import lru_cache

import numpy as np
import torch
from torch import nn

from src import api
from src.constants import GENE_MAC_REST, step_state_by_byte
from src.family import fold_disagreement_d
from src.tools.autoencoder import kernel

N_STATES = 4096
D = 6


@lru_cache(maxsize=1)
def _byte_tables() -> dict[str, np.ndarray]:
    """Per-byte LUT built from the product kernel/API and asserted at build."""
    intron = np.array(api.INTRON_BY_BYTE, dtype=np.int64)
    micro = np.array(api.MICRO_REF_BY_BYTE, dtype=np.int64)
    i0, i7 = intron & 1, (intron >> 7) & 1
    tau_u, tau_v = i0 * 63, micro ^ (i7 * 63)
    for b in range(256):
        _p, u, v = kernel.sig_id_parts(kernel.word_signature_id([b]))
        assert (u, v) == (int(tau_u[b]), int(tau_v[b]))
    return {
        "intron": intron,
        "family": np.array(api.FAMILY_BY_BYTE, dtype=np.int64),
        "micro": micro,
        "q6": np.array([api.q_word6(b) for b in range(256)], dtype=np.int64),
        "fold": np.array([fold_disagreement_d(b, D) for b in range(256)], dtype=np.int64),
        "eps_a": np.array(api.EPS_A6_BY_BYTE, dtype=np.int64),
        "eps_b": np.array(api.EPS_B6_BY_BYTE, dtype=np.int64),
        "tau_u": tau_u,
        "tau_v": tau_v,
    }


@lru_cache(maxsize=1)
def _state24_by_uv() -> np.ndarray:
    tbl = np.empty(4096, dtype=np.int64)
    for u in range(64):
        for v in range(64):
            tbl[(u << 6) | v] = int(api.omega12_to_state24((u, v)))
    tbl.setflags(write=False)
    return tbl


def prefix_factor_table(ledgers: np.ndarray) -> np.ndarray:
    """[N, T, 3] Theorem-2 signature factors of every prefix (vectorized)."""
    tbl = _byte_tables()
    n, t_len = ledgers.shape
    out = np.zeros((n, t_len, 3), dtype=np.int64)
    p = np.zeros(n, np.int64)
    su = np.zeros(n, np.int64)
    sv = np.zeros(n, np.int64)
    for j in range(t_len):
        b = ledgers[:, j].astype(np.int64)
        su, sv, p = sv ^ tbl["tau_u"][b], su ^ tbl["tau_v"][b], p ^ 1
        out[:, j, 0] = p
        out[:, j, 1] = su
        out[:, j, 2] = sv
    return out


@dataclass
class ScanTrajectory:
    """Exact per-step ledger scan through the hQVM kernel.

    Tensor fields are shaped [B, T] (or [T] when unbatched). Masked steps
    publish zeros for post-state fields so those values never enter inputs.
    """

    bytes: torch.Tensor
    intron: torch.Tensor
    family_phase: torch.Tensor
    micro_ref: torch.Tensor
    q6: torch.Tensor
    state_before: torch.Tensor
    state_after: torch.Tensor
    chi_shell: torch.Tensor
    arch_shell: torch.Tensor
    prefix_sig: torch.Tensor
    fold_disagreement: torch.Tensor
    mask: torch.Tensor | None = None

    def visible_bytes(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return ledger bytes with masked positions zeroed."""
        m = self.mask if mask is None else mask
        out = self.bytes.to(dtype=torch.long).clone()
        if m is None:
            return out
        hidden = m.to(device=out.device).bool()
        out = out.masked_fill(hidden, 0)
        return out

    def validate_decode(self, predicted_bytes: torch.Tensor) -> torch.Tensor:
        """Replay predicted bytes from GENE_MAC_REST; match last state_after."""
        pred = predicted_bytes.to(dtype=torch.long)
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)
            after = (
                self.state_after.unsqueeze(0)
                if self.state_after.ndim == 1
                else self.state_after
            )
            squeeze = True
        else:
            after = self.state_after
            squeeze = False
        batch, steps = pred.shape
        ok = torch.zeros(batch, dtype=torch.bool, device=pred.device)
        for b in range(batch):
            state = int(GENE_MAC_REST)
            for t in range(steps):
                state = step_state_by_byte(state, int(pred[b, t].item()))
            ok[b] = state == int(after[b, -1].item())
        return ok[0] if squeeze else ok


class ExactHQVMScan(nn.Module):
    """Zero-parameter exact scan of a byte ledger through the hQVM kernel.

    For each visible step the carrier is advanced with ``step_state_by_byte``
    and the published fields (post-state, shells, prefix signature) are filled
    from ``src.api`` / ``src.family`` / ``kernel.word_signature_id``. Masked
    steps do not advance the published carrier and publish zeros for those
    post-state fields.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        ledger: torch.Tensor,
        mask: torch.Tensor | None = None,
        start_state: int = GENE_MAC_REST,
    ) -> ScanTrajectory:
        bytes_t = ledger.to(dtype=torch.long)
        squeeze = False
        if bytes_t.ndim == 1:
            bytes_t = bytes_t.unsqueeze(0)
            squeeze = True
        if mask is not None:
            mask_t = mask.to(dtype=torch.bool)
            if mask_t.ndim == 1:
                mask_t = mask_t.unsqueeze(0)
            if squeeze and mask_t.shape[0] != 1:
                raise ValueError("mask batch must match ledger")
        else:
            mask_t = None

        batch, steps = bytes_t.shape
        device = bytes_t.device

        intron = torch.zeros(batch, steps, dtype=torch.long, device=device)
        family_phase = torch.zeros(batch, steps, dtype=torch.long, device=device)
        micro_ref = torch.zeros(batch, steps, dtype=torch.long, device=device)
        q6 = torch.zeros(batch, steps, dtype=torch.long, device=device)
        state_before = torch.zeros(batch, steps, dtype=torch.long, device=device)
        state_after = torch.zeros(batch, steps, dtype=torch.long, device=device)
        chi_shell = torch.zeros(batch, steps, dtype=torch.long, device=device)
        arch_shell = torch.zeros(batch, steps, dtype=torch.long, device=device)
        prefix_sig = torch.zeros(batch, steps, dtype=torch.long, device=device)
        fold_dis = torch.zeros(batch, steps, dtype=torch.long, device=device)

        T = _byte_tables()
        S24 = _state24_by_uv()
        omega0 = api.state24_to_omega12(int(start_state))

        for b in range(batch):
            u = int(omega0.u6)
            v = int(omega0.v6)
            sp = 0
            su = 0
            sv = 0
            state = int(start_state)
            for t in range(steps):
                byte = int(bytes_t[b, t].item())
                hidden = bool(mask_t[b, t].item()) if mask_t is not None else False
                state_before[b, t] = state
                if hidden:
                    continue
                intron[b, t] = int(T["intron"][byte])
                family_phase[b, t] = int(T["family"][byte])
                micro_ref[b, t] = int(T["micro"][byte])
                q6[b, t] = int(T["q6"][byte])
                fold_dis[b, t] = int(T["fold"][byte])
                u, v = (
                    v ^ int(T["eps_a"][byte]),
                    u ^ int(T["micro"][byte]) ^ int(T["eps_b"][byte]),
                )
                su, sv, sp = sv ^ int(T["tau_u"][byte]), su ^ int(T["tau_v"][byte]), sp ^ 1
                state = int(S24[(u << 6) | v])
                chi = u ^ v
                chi_s = chi.bit_count()
                state_after[b, t] = state
                chi_shell[b, t] = chi_s
                arch_shell[b, t] = 6 - chi_s
                prefix_sig[b, t] = (sp << 12) | (su << 6) | sv

        def _maybe_squeeze(x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(0) if squeeze else x

        mask_out = None
        if mask_t is not None:
            mask_out = _maybe_squeeze(mask_t)

        return ScanTrajectory(
            bytes=_maybe_squeeze(bytes_t),
            intron=_maybe_squeeze(intron),
            family_phase=_maybe_squeeze(family_phase),
            micro_ref=_maybe_squeeze(micro_ref),
            q6=_maybe_squeeze(q6),
            state_before=_maybe_squeeze(state_before),
            state_after=_maybe_squeeze(state_after),
            chi_shell=_maybe_squeeze(chi_shell),
            arch_shell=_maybe_squeeze(arch_shell),
            prefix_sig=_maybe_squeeze(prefix_sig),
            fold_disagreement=_maybe_squeeze(fold_dis),
            mask=mask_out,
        )


def _raw_byte_bits(bytes_t: torch.Tensor) -> torch.Tensor:
    """[B, T] integer bytes -> [B, T, 8] raw bit features (LSB first)."""
    shifts = torch.arange(8, device=bytes_t.device, dtype=bytes_t.dtype)
    return (torch.bitwise_right_shift(bytes_t.unsqueeze(-1), shifts) & 1).to(
        dtype=torch.float32
    )


class MomentAtomEncoder(nn.Module):
    """Encode raw 8-bit byte features plus optional visible scan fields.

    Input ledger bytes [B, T]; optional extra [B, T, extra_dim];
    output atom sequence [B, T, atom_dim]. Masked positions use a learned
    mask token, not the executable rest byte 0x00.
    """

    def __init__(self, atom_dim: int = 32, extra_dim: int = 0) -> None:
        super().__init__()
        self.atom_dim = int(atom_dim)
        self.extra_dim = int(extra_dim)
        self.proj = nn.Linear(8 + self.extra_dim, self.atom_dim)
        self.mask_token = nn.Parameter(torch.zeros(self.atom_dim))

    def forward(
        self,
        bytes_t: torch.Tensor,
        mask: torch.Tensor | None = None,
        extra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bits = _raw_byte_bits(bytes_t.to(dtype=torch.long))
        if extra is not None:
            bits = torch.cat([bits, extra.to(dtype=bits.dtype)], dim=-1)
        elif self.extra_dim:
            bits = torch.cat(
                [
                    bits,
                    torch.zeros(
                        *bits.shape[:-1],
                        self.extra_dim,
                        device=bits.device,
                        dtype=bits.dtype,
                    ),
                ],
                dim=-1,
            )
        atoms = self.proj(bits)
        if mask is not None:
            m = mask.to(dtype=torch.bool, device=atoms.device)
            if m.ndim == 1:
                m = m.unsqueeze(0)
            atoms = torch.where(
                m.unsqueeze(-1),
                self.mask_token.view(1, 1, -1).expand_as(atoms),
                atoms,
            )
        return atoms


class Depth4FrameEncoder(nn.Module):
    """Map consecutive groups of four atoms into frame latents.

    Input [B, T, atom_dim] with T a multiple of 4; output [B, T/4, frame_dim].
    """

    def __init__(self, atom_dim: int = 32, frame_dim: int = 64) -> None:
        super().__init__()
        self.atom_dim = int(atom_dim)
        self.frame_dim = int(frame_dim)
        self.proj = nn.Linear(4 * self.atom_dim, self.frame_dim)

    def forward(self, atoms: torch.Tensor) -> torch.Tensor:
        if atoms.ndim != 3:
            raise ValueError(
                f"atoms must be [B, T, atom_dim], got {tuple(atoms.shape)}"
            )
        batch, steps, dim = atoms.shape
        if steps % 4 != 0:
            raise ValueError(f"T must be a multiple of 4, got T={steps}")
        if dim != self.atom_dim:
            raise ValueError(f"atom_dim mismatch: expected {self.atom_dim}, got {dim}")
        frames = atoms.reshape(batch, steps // 4, 4 * self.atom_dim)
        return self.proj(frames)


class LedgerContext(nn.Module):
    """GRU over a sequence (atoms or frames); returns the final hidden state."""

    def __init__(self, atom_dim: int = 32, context_dim: int = 64) -> None:
        super().__init__()
        self.atom_dim = int(atom_dim)
        self.context_dim = int(context_dim)
        self.gru = nn.GRU(
            input_size=self.atom_dim,
            hidden_size=self.context_dim,
            batch_first=True,
        )

    def forward(self, atoms: torch.Tensor) -> torch.Tensor:
        _out, h_n = self.gru(atoms)
        return h_n.squeeze(0)

    def sequence(self, atoms: torch.Tensor) -> torch.Tensor:
        out, _h_n = self.gru(atoms)
        return out


@dataclass
class SuperCode:
    """Ledger-free Super encoding: exact kernel facts plus learned provenance."""

    provenance: torch.Tensor
    signature: torch.Tensor
    signature_bits: torch.Tensor
    endpoint: torch.Tensor
    q_transport: torch.Tensor
    climate: torch.Tensor
    length: torch.Tensor


class LedgerDecoder(nn.Module):
    """Per-position byte logits from the exact signature and provenance code."""

    def __init__(self, ctx_dim: int, sig_dim: int = 13, pos_dim: int = 16) -> None:
        super().__init__()
        self.pos = nn.Embedding(32, pos_dim)
        self.net = nn.Linear(sig_dim + ctx_dim + pos_dim, 256)

    def forward(
        self,
        sig_bits: torch.Tensor,
        provenance_code: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        p = self.pos(positions.clamp(0, 31))
        code = provenance_code.unsqueeze(1).expand(-1, p.shape[1], -1)
        sig = sig_bits.unsqueeze(1).expand(-1, p.shape[1], -1)
        return self.net(torch.cat([sig, code, p], dim=-1))


class ByteDecoder(nn.Module):
    """Predict byte / family / payload / signature factors from context.

    Inputs are context, frame, position, the visible raw-bit pool, neighbor
    atoms, and an optional corruption marker. Exact prefix signatures are not
    inputs.
    """

    def __init__(
        self,
        context_dim: int = 64,
        frame_dim: int = 64,
        pos_dim: int = 8,
        marker_dim: int = 4,
        raw_dim: int = 16,
        atom_dim: int = 32,
        analytic_grammar: bool = True,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.frame_dim = int(frame_dim)
        self.pos_dim = int(pos_dim)
        self.marker_dim = int(marker_dim)
        self.raw_dim = int(raw_dim)
        self.atom_dim = int(atom_dim)
        self.analytic_grammar = bool(analytic_grammar)
        self.pos_embed = nn.Embedding(32, self.pos_dim)
        self.corrupt_embed = nn.Embedding(2, self.marker_dim)
        in_dim = (
            self.context_dim
            + self.frame_dim
            + self.pos_dim
            + self.marker_dim
            + self.raw_dim
            + 2 * self.atom_dim
        )
        resid_dim = (
            self.context_dim
            + self.frame_dim
            + self.pos_dim
            + self.marker_dim
            + 2 * self.atom_dim
        )
        self.grammar_bits = nn.Linear(8 + 4, 8)
        bias = self.grammar_bits.bias
        assert bias is not None
        with torch.no_grad():
            self.grammar_bits.weight.zero_()
            bias.zero_()
            for j in range(1, 7):
                self.grammar_bits.weight[j, j] = 8.0
                bias[j] = -4.0
            self.grammar_bits.weight[0, 8:12] = torch.tensor([-4.0, 4.0, -4.0, 4.0])
            self.grammar_bits.weight[7, 8:12] = torch.tensor([4.0, 4.0, -4.0, -4.0])
        if self.analytic_grammar:
            self.grammar_bits.weight.requires_grad_(False)
            bias.requires_grad_(False)
        else:
            nn.init.normal_(self.grammar_bits.weight, std=0.5)
            nn.init.zeros_(bias)
        hid = max(64, resid_dim)
        qw = torch.tensor([int(api.Q_WEIGHT_BY_BYTE[b]) for b in range(256)], dtype=torch.float32)
        self.register_buffer("q_weight256", qw)
        micro_bits = torch.zeros(256, 6, dtype=torch.float32)
        for b in range(256):
            m = int(api.MICRO_REF_BY_BYTE[b])
            for i in range(6):
                micro_bits[b, i] = float((m >> i) & 1)
        self.register_buffer("byte_micro_bits", micro_bits)
        # softplus(θ) ≈ log((1-p)/p); θ≈2 → p≈0.12 near the Markov eval p=0.1
        self.markov_flip = nn.Parameter(torch.full((6,), 2.0))
        self.prev_micro_proj = nn.Linear(6, resid_dim)
        nn.init.zeros_(self.prev_micro_proj.weight)
        nn.init.zeros_(self.prev_micro_proj.bias)
        self.climate_head = nn.Linear(7, 256)
        nn.init.zeros_(self.climate_head.weight)
        if self.climate_head.bias is not None:
            nn.init.zeros_(self.climate_head.bias)
        self.seq_mlp = nn.Sequential(
            nn.Linear(resid_dim, hid),
            nn.GELU(),
            nn.Linear(hid, 256),
        )
        last = self.seq_mlp[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)
        self.family_head = nn.Linear(in_dim, 4)
        self.payload_head = nn.Linear(in_dim, 6)
        self.parity_head = nn.Linear(in_dim, 2)
        self.tau_u_head = nn.Linear(in_dim, 64)
        self.tau_v_head = nn.Linear(in_dim, 64)

    def forward(
        self,
        context: torch.Tensor,
        frames: torch.Tensor,
        mask_pos: torch.Tensor | None = None,
        corrupt_flag: torch.Tensor | None = None,
        raw_pool: torch.Tensor | None = None,
        micro_agree: torch.Tensor | None = None,
        nvis: torch.Tensor | None = None,
        prev_atom: torch.Tensor | None = None,
        next_atom: torch.Tensor | None = None,
        coset_mask: torch.Tensor | None = None,
        q_hist: torch.Tensor | None = None,
        prev_micro: torch.Tensor | None = None,
        has_prev: torch.Tensor | None = None,
        apply_markov: bool = False,
    ) -> dict[str, torch.Tensor]:
        last_frame = frames[:, -1, :]
        if mask_pos is not None and frames.shape[1] > 1:
            idx = (mask_pos.to(dtype=torch.long) // 4).clamp(0, frames.shape[1] - 1)
            last_frame = frames[
                torch.arange(frames.shape[0], device=frames.device), idx
            ]
        if mask_pos is None:
            mask_pos = torch.zeros(
                context.shape[0], dtype=torch.long, device=context.device
            )
        else:
            mask_pos = mask_pos.to(dtype=torch.long, device=context.device).view(-1)
        if corrupt_flag is None:
            corrupt_flag = torch.zeros(
                context.shape[0], dtype=torch.long, device=context.device
            )
        else:
            corrupt_flag = corrupt_flag.to(
                dtype=torch.long, device=context.device
            ).view(-1)
        if raw_pool is None:
            raw_pool = torch.zeros(
                context.shape[0],
                self.raw_dim,
                device=context.device,
                dtype=context.dtype,
            )
        pos_e = self.pos_embed(mask_pos.clamp(0, 31))
        if prev_atom is None:
            prev_atom = torch.zeros(
                context.shape[0], self.atom_dim, device=context.device, dtype=context.dtype
            )
        if next_atom is None:
            next_atom = torch.zeros(
                context.shape[0], self.atom_dim, device=context.device, dtype=context.dtype
            )
        feat = torch.cat(
            [
                context,
                last_frame,
                pos_e,
                self.corrupt_embed(corrupt_flag.clamp(0, 1)),
                raw_pool,
                prev_atom,
                next_atom,
            ],
            dim=-1,
        )
        feat_resid = torch.cat(
            [
                context,
                last_frame,
                pos_e,
                self.corrupt_embed(corrupt_flag.clamp(0, 1)),
                prev_atom,
                next_atom,
            ],
            dim=-1,
        )
        if prev_micro is None:
            prev_micro = torch.zeros(
                context.shape[0], 6, device=context.device, dtype=context.dtype
            )
        else:
            prev_micro = prev_micro.to(dtype=context.dtype, device=context.device)
        if has_prev is None:
            has_prev = torch.zeros(
                context.shape[0], 1, device=context.device, dtype=context.dtype
            )
        else:
            has_prev = has_prev.to(dtype=context.dtype, device=context.device).view(-1, 1)
        feat_resid = feat_resid + self.prev_micro_proj(prev_micro) * has_prev * float(
            apply_markov
        )
        pos_oh = torch.nn.functional.one_hot(mask_pos.remainder(4), 4).to(
            dtype=raw_pool.dtype
        )
        src_bits = raw_pool[:, :8]
        if (corrupt_flag > 0).any():
            maj = raw_pool[:, 8:16]
            src_bits = torch.where(corrupt_flag.view(-1, 1).bool(), maj, src_bits)
        bit_logits = self.grammar_bits(torch.cat([src_bits, pos_oh], dim=-1))
        if nvis is None:
            nvis = torch.ones(
                bit_logits.shape[0],
                1,
                device=bit_logits.device,
                dtype=bit_logits.dtype,
            )
        else:
            nvis = nvis.to(dtype=bit_logits.dtype, device=bit_logits.device).view(-1, 1)
        if micro_agree is None:
            micro_agree = torch.ones(
                bit_logits.shape[0], 1, device=bit_logits.device, dtype=bit_logits.dtype
            )
        else:
            micro_agree = micro_agree.to(
                dtype=bit_logits.dtype, device=bit_logits.device
            ).view(-1, 1)
        empty = nvis < 0.5
        family_only = torch.cat(
            [
                bit_logits[:, :1],
                torch.zeros_like(bit_logits[:, 1:7]),
                bit_logits[:, 7:],
            ],
            dim=-1,
        )
        law = bit_logits * micro_agree
        corrupt = corrupt_flag.view(-1, 1).to(dtype=torch.bool)
        bit_logits = torch.where(
            empty, family_only, torch.where(corrupt, bit_logits, law)
        )
        shifts = torch.arange(8, device=bit_logits.device)
        bit_table = (
            torch.bitwise_right_shift(
                torch.arange(256, device=bit_logits.device).unsqueeze(1), shifts
            )
            & 1
        )
        signs = bit_table.to(dtype=bit_logits.dtype) * 2.0 - 1.0
        grammar = bit_logits @ signs.T
        if coset_mask is not None:
            grammar = grammar.masked_fill(~coset_mask.to(dtype=torch.bool), float("-inf"))
        seq = self.seq_mlp(feat_resid)
        # Independent-bit flip filter on previous micro (family remains free = +2 bits).
        # Opt-in only: must stay off for G6/G7 certificates on uniform/canonical ledgers.
        if apply_markov:
            pen = torch.nn.functional.softplus(self.markov_flip).view(1, 1, 6)
            diff = (prev_micro.unsqueeze(1) - self.byte_micro_bits.unsqueeze(0)).abs()
            markov = -(diff * pen).sum(dim=-1) * has_prev
        else:
            markov = torch.zeros_like(seq)
        if q_hist is None:
            q_hist = torch.zeros(
                seq.shape[0], 7, device=seq.device, dtype=seq.dtype
            )
        j = torch.arange(7, device=q_hist.device, dtype=q_hist.dtype)
        e_j = (q_hist * j).sum(dim=-1).clamp(0.05, 5.95)
        lam_hat = e_j / (6.0 - e_j).clamp(min=0.05)
        analytic = self.q_weight256.to(device=seq.device, dtype=seq.dtype).unsqueeze(0) * torch.log(
            lam_hat.clamp(min=1e-3)
        ).unsqueeze(-1)
        climate = analytic + self.climate_head(q_hist.to(dtype=seq.dtype))
        nvis_ok = nvis.to(dtype=climate.dtype).view(-1, 1) >= 2.0
        disagree = 1.0 - micro_agree.to(dtype=climate.dtype).view(-1, 1)
        climate = climate * nvis_ok.to(dtype=climate.dtype) * disagree
        residual = climate + seq + markov
        # seq L2 on canonical; markov is gated by has_prev and trained only on seq arms.
        return {
            "byte_logits": grammar + residual,
            "exact_prior_logits": grammar,
            "residual_logits": residual,
            "climate_logits": climate,
            "seq_logits": seq,
            "markov_logits": markov,
            "family_logits": self.family_head(feat),
            "payload_logits": self.payload_head(feat),
            "parity_logits": self.parity_head(feat),
            "tau_u_logits": self.tau_u_head(feat),
            "tau_v_logits": self.tau_v_head(feat),
        }


class Super(nn.Module):
    """Kernel scan, byte grammar, signature registers, frame GRU context."""

    SCAN_EXTRA_DIM = 4 + 6 + 6 + 1 + 1 + 13

    def __init__(
        self,
        context_dim: int = 64,
        atom_dim: int = 32,
        frame_dim: int = 64,
        signature_mode: str = "exact",
        analytic_grammar: bool = True,
    ) -> None:
        super().__init__()
        if signature_mode not in ("exact", "learned"):
            raise ValueError("signature_mode must be 'exact' or 'learned'")
        self.context_dim = int(context_dim)
        self.atom_dim = int(atom_dim)
        self.frame_dim = int(frame_dim)
        self.signature_mode = signature_mode
        self.analytic_grammar = bool(analytic_grammar)
        self.scan = ExactHQVMScan()
        self.atom = MomentAtomEncoder(
            atom_dim=self.atom_dim, extra_dim=self.SCAN_EXTRA_DIM
        )
        self.frame = Depth4FrameEncoder(
            atom_dim=self.atom_dim, frame_dim=self.frame_dim
        )
        self.context = LedgerContext(
            atom_dim=self.frame_dim, context_dim=self.context_dim
        )
        self.prefix_gru = LedgerContext(
            atom_dim=self.atom_dim, context_dim=self.context_dim
        )
        self.byte_head = ByteDecoder(
            context_dim=self.context_dim * 2,
            frame_dim=self.frame_dim,
            atom_dim=self.atom_dim,
            analytic_grammar=self.analytic_grammar,
        )
        self.ledger_decoder = LedgerDecoder(ctx_dim=self.context_dim * 2)

    def get_config(self) -> dict:
        return {
            "context_dim": self.context_dim,
            "atom_dim": self.atom_dim,
            "frame_dim": self.frame_dim,
            "signature_mode": self.signature_mode,
            "analytic_grammar": self.analytic_grammar,
        }

    def encode(self, ledger: torch.Tensor, mask: torch.Tensor | None = None) -> SuperCode:
        """Exact kernel facts plus learned provenance code (no source ledger)."""
        out = self.forward(ledger, mask=mask, hide_masked=True, compute_dynamics=False)
        traj = out["traj"]
        assert isinstance(traj, ScanTrajectory)
        last = traj.prefix_sig[..., -1]
        if last.ndim == 0:
            last = last.unsqueeze(0)
        endpoint = traj.state_after[..., -1]
        if endpoint.ndim == 0:
            endpoint = endpoint.unsqueeze(0)
        q6 = traj.q6
        if q6.ndim == 1:
            q6 = q6.unsqueeze(0)
        shells = traj.chi_shell
        if shells.ndim == 1:
            shells = shells.unsqueeze(0)
        climate = torch.nn.functional.one_hot(shells.clamp(0, 6).long(), 7).to(
            dtype=torch.float32
        ).mean(dim=1)
        length = torch.full(
            (last.shape[0],),
            int(ledger.shape[-1]),
            dtype=torch.long,
            device=last.device,
        )
        shifts = torch.arange(13, device=last.device)
        sig_bits = (
            torch.bitwise_right_shift(last.unsqueeze(-1).to(dtype=torch.long), shifts) & 1
        ).to(dtype=torch.float32)
        prov = out["context"]
        if not isinstance(prov, torch.Tensor):
            raise TypeError("encode expected Tensor context")
        if prov.ndim == 1:
            prov = prov.unsqueeze(0)
        return SuperCode(
            provenance=prov,
            signature=last.to(dtype=torch.long),
            signature_bits=sig_bits,
            endpoint=endpoint.to(dtype=torch.long),
            q_transport=q6.to(dtype=torch.long),
            climate=climate,
            length=length,
        )

    def decode(
        self,
        encoded: SuperCode | dict,
        length: int | None = None,
        mask: torch.Tensor | None = None,
        observed_ledger: torch.Tensor | None = None,
        observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-position logits from SuperCode; optional exact signature coset mask."""
        if isinstance(encoded, dict):
            provenance = encoded.get("provenance_code", encoded.get("provenance"))
            signature = encoded.get("exact_signature", encoded.get("signature"))
            endpoint = encoded["endpoint"]
            q_transport = encoded["q_transport"]
            climate = encoded["climate"]
            if not isinstance(provenance, torch.Tensor):
                raise TypeError("provenance must be a Tensor")
            if not isinstance(signature, torch.Tensor):
                raise TypeError("signature must be a Tensor")
            if not isinstance(endpoint, torch.Tensor):
                raise TypeError("endpoint must be a Tensor")
            if not isinstance(q_transport, torch.Tensor):
                raise TypeError("q_transport must be a Tensor")
            if not isinstance(climate, torch.Tensor):
                raise TypeError("climate must be a Tensor")
            signature_bits = encoded.get("signature_bits")
            if not isinstance(signature_bits, torch.Tensor):
                shifts = torch.arange(13, device=signature.device)
                signature_bits = (
                    torch.bitwise_right_shift(
                        signature.unsqueeze(-1).to(dtype=torch.long), shifts
                    )
                    & 1
                ).to(dtype=torch.float32)
            length_t = encoded.get("length")
            if not isinstance(length_t, torch.Tensor):
                if length is None:
                    raise ValueError("decode dict requires length or length tensor")
                length_t = torch.full(
                    (provenance.shape[0],),
                    int(length),
                    dtype=torch.long,
                    device=provenance.device,
                )
            encoded = SuperCode(
                provenance=provenance,
                signature=signature,
                signature_bits=signature_bits,
                endpoint=endpoint,
                q_transport=q_transport,
                climate=climate,
                length=length_t,
            )
        t_len = int(length) if length is not None else int(encoded.length[0].item())
        batch = encoded.provenance.shape[0]
        device = encoded.provenance.device
        positions = torch.arange(t_len, device=device).unsqueeze(0).expand(batch, -1)
        sig_bits = encoded.signature_bits
        logits = self.ledger_decoder(sig_bits, encoded.provenance, positions)
        obs = observed_ledger
        obs_mask = observed_mask if observed_mask is not None else mask
        if obs is not None and obs_mask is not None:
            led = obs.to(dtype=torch.long)
            if led.ndim == 1:
                led = led.unsqueeze(0)
            mb = obs_mask.to(dtype=torch.bool)
            if mb.ndim == 1:
                mb = mb.unsqueeze(0)
            for bi in range(led.shape[0]):
                idxs = torch.where(mb[bi, :t_len])[0]
                if len(idxs) != 1:
                    continue
                pos = int(idxs[0].item())
                word = [int(x) for x in led[bi, :t_len].tolist()]
                target = int(encoded.signature[bi].item())
                valid = torch.zeros(256, dtype=torch.bool, device=device)
                prefix = word[:pos]
                suffix = word[pos + 1 : t_len]
                for b in range(256):
                    valid[b] = kernel.word_signature_id(prefix + [b] + suffix) == target
                logits[bi, pos] = logits[bi, pos].masked_fill(~valid, float("-inf"))
        return logits.squeeze(0) if batch == 1 and obs is not None and obs.ndim == 1 else logits

    def forward(
        self,
        ledger: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        hide_masked: bool = True,
        corruption_mask: torch.Tensor | None = None,
        compute_dynamics: bool = False,
        compute_signature_posterior: bool = False,
        coset_mask: torch.Tensor | None = None,
        apply_markov: bool = False,
    ) -> dict[str, torch.Tensor | ScanTrajectory]:
        """Forward pass.

        ``mask`` marks prediction sites. When ``hide_masked`` is True (default),
        those sites are zeroed for the VisibleScan / atom path (masked grammar).
        When False (corruption recovery), the ledger bytes stay visible and
        ``corruption_mask`` (or ``mask``) only marks the predict position.
        """
        predict = corruption_mask if corruption_mask is not None else mask
        scan_mask = mask if hide_masked else None
        traj = self.scan(ledger, mask=scan_mask)
        raw_ledger = ledger.to(dtype=torch.long)
        if raw_ledger.ndim == 1:
            raw_ledger = raw_ledger.unsqueeze(0)
        mask_b = None
        if predict is not None:
            mask_b = predict.to(dtype=torch.bool, device=raw_ledger.device)
            if mask_b.ndim == 1:
                mask_b = mask_b.unsqueeze(0)

        atom_mask = mask_b if hide_masked else None
        t_len = raw_ledger.shape[1]
        extra = _scan_atom_features(traj, raw_ledger.shape[0], t_len, raw_ledger.device)
        atoms = self.atom(raw_ledger, mask=atom_mask, extra=extra)
        pad = (4 - (t_len % 4)) % 4
        if pad:
            pad_tok = self.atom.mask_token.view(1, 1, -1).expand(atoms.shape[0], pad, -1)
            atoms = torch.cat([atoms, pad_tok], dim=1)
        n_real_frames = (t_len + 3) // 4
        frames = self.frame(atoms)
        frame_ctx = self.context(frames[:, :n_real_frames, :])
        mask_pos = None
        corrupt_flag = None
        if mask_b is not None:
            pos = []
            flags = []
            for bi in range(mask_b.shape[0]):
                idxs = torch.where(mask_b[bi])[0]
                pos.append(int(idxs[0].item()) if len(idxs) else 0)
                flags.append(1 if corruption_mask is not None else 0)
            mask_pos = torch.as_tensor(pos, dtype=torch.long, device=raw_ledger.device)
            corrupt_flag = torch.as_tensor(
                flags, dtype=torch.long, device=raw_ledger.device
            )
        ar = torch.arange(atoms.shape[0], device=atoms.device)
        t_hi = max(t_len - 1, 0)
        if mask_pos is None:
            pos_idx = torch.full(
                (atoms.shape[0],), t_hi, dtype=torch.long, device=atoms.device
            )
        else:
            pos_idx = mask_pos.clamp(0, t_hi)
        prefix_out = self.prefix_gru.sequence(atoms[:, :t_len, :])
        causal_idx = (pos_idx - 1).clamp(min=0)
        prefix_ctx = prefix_out[ar, causal_idx]
        context = torch.cat([frame_ctx, prefix_ctx], dim=-1)
        prev_atom = atoms[ar, causal_idx]
        next_idx = (pos_idx + 1).clamp(max=t_hi)
        next_atom = atoms[ar, next_idx]
        if mask_b is not None:
            next_blocked = mask_b[ar, next_idx] | (pos_idx >= t_hi)
            next_atom = torch.where(
                next_blocked.unsqueeze(-1), torch.zeros_like(next_atom), next_atom
            )
            start_site = pos_idx <= 0
            prev_atom = torch.where(
                start_site.unsqueeze(-1), torch.zeros_like(prev_atom), prev_atom
            )
        aa = torch.as_tensor(0xAA, device=raw_ledger.device, dtype=raw_ledger.dtype)
        intron = torch.bitwise_xor(raw_ledger, aa)
        micro6 = torch.bitwise_right_shift(intron, 1) & 63
        shifts6 = torch.arange(6, device=raw_ledger.device)
        prev_micro = (
            torch.bitwise_right_shift(micro6[ar, causal_idx].unsqueeze(-1), shifts6) & 1
        ).to(dtype=torch.float32)
        has_prev = (pos_idx > 0).to(dtype=torch.float32).unsqueeze(-1)
        if mask_b is not None:
            prev_vis = (~mask_b[ar, causal_idx]).to(dtype=torch.float32).unsqueeze(-1)
            has_prev = has_prev * prev_vis
            prev_micro = prev_micro * has_prev
        bits = _raw_byte_bits(raw_ledger)
        if mask_b is not None:
            vis = (~mask_b).unsqueeze(-1).to(dtype=bits.dtype)
            vis_count = vis.sum(dim=1)
            nvis = vis_count.clamp(min=1.0)
            counts = (bits * vis).sum(dim=1)
            raw_mean = counts / nvis
            raw_maj = (counts > (nvis * 0.5)).to(dtype=bits.dtype)
            agree = ((counts == 0) | (counts == vis_count)) & (vis_count > 0)
            micro_agree = agree[:, 1:7].all(dim=-1, keepdim=True).to(dtype=bits.dtype)
        else:
            vis_count = torch.full(
                (bits.shape[0], 1),
                float(bits.shape[1]),
                device=bits.device,
                dtype=bits.dtype,
            )
            raw_mean = bits.mean(dim=1)
            raw_maj = (raw_mean > 0.5).to(dtype=bits.dtype)
            agree = (raw_mean * (1.0 - raw_mean) < 1e-6)
            micro_agree = agree[:, 1:7].all(dim=-1, keepdim=True).to(dtype=bits.dtype)
        raw_pool = torch.cat([raw_mean, raw_maj], dim=-1)
        q6 = _bt(traj.q6, raw_ledger.shape[0], t_len).to(device=raw_ledger.device)
        q_wt = _popcount6(q6).clamp(0, 6)
        q_oh = torch.nn.functional.one_hot(q_wt.long(), 7).to(dtype=bits.dtype)
        if mask_b is not None:
            vis_m = (~mask_b).unsqueeze(-1).to(dtype=q_oh.dtype)
            q_hist = (q_oh * vis_m).sum(dim=1) / vis_m.sum(dim=1).clamp(min=1.0)
        else:
            q_hist = q_oh.mean(dim=1)
        heads = self.byte_head(
            context,
            frames,
            mask_pos=mask_pos,
            corrupt_flag=corrupt_flag,
            raw_pool=raw_pool,
            micro_agree=micro_agree,
            nvis=vis_count,
            prev_atom=prev_atom,
            next_atom=next_atom,
            coset_mask=coset_mask,
            q_hist=q_hist,
            prev_micro=prev_micro,
            has_prev=has_prev,
            apply_markov=apply_markov,
        )
        if hide_masked and mask_b is not None:
            sig_vis = ~mask_b
        else:
            sig_vis = torch.ones(
                raw_ledger.shape[0],
                t_len,
                dtype=torch.bool,
                device=raw_ledger.device,
            )
        exact_sig = _visible_signature_logits(raw_ledger[:, :t_len], sig_vis)
        heads["exact_parity_logits"] = exact_sig["parity_logits"]
        heads["exact_tau_u_logits"] = exact_sig["tau_u_logits"]
        heads["exact_tau_v_logits"] = exact_sig["tau_v_logits"]
        if self.signature_mode == "exact":
            heads["parity_logits"] = exact_sig["parity_logits"]
            heads["tau_u_logits"] = exact_sig["tau_u_logits"]
            heads["tau_v_logits"] = exact_sig["tau_v_logits"]
        if compute_signature_posterior and mask_b is not None and mask_b.any():
            if int(mask_b.sum(dim=1).max().item()) > 1:
                raise ValueError("signature posterior is single-hidden only")
            byte_probs = torch.softmax(
                heads["byte_logits"].to(dtype=torch.float32), dim=-1
            )
            heads["signature_posterior"] = signature_posterior_uv(
                raw_ledger[:, :t_len], mask_b, byte_probs
            )
        pred_bytes = heads["byte_logits"].argmax(dim=-1)
        recon = raw_ledger[:, :t_len].clone()
        if mask_b is not None:
            for bi in range(recon.shape[0]):
                idxs = torch.where(mask_b[bi])[0]
                pos = int(idxs[0].item()) if len(idxs) else recon.shape[1] - 1
                recon[bi, pos] = pred_bytes[bi]
        else:
            recon[:, -1] = pred_bytes

        out: dict[str, torch.Tensor | ScanTrajectory] = {
            **heads,
            "context": context,
            "frames": frames,
            "atoms": atoms[:, :t_len, :],
            "traj": traj,
            "raw_pool": raw_pool,
        }
        if compute_dynamics:
            replay_valid = _replay_matches(raw_ledger[:, :t_len], recon)
            batch = pred_bytes.shape[0]
            state_before_idx = torch.zeros(
                batch, dtype=torch.long, device=pred_bytes.device
            )
            for bi in range(batch):
                p = int(mask_pos[bi].item()) if mask_pos is not None else (t_len - 1)
                if hide_masked and mask_b is not None:
                    sb = int(GENE_MAC_REST)
                    for t in range(p):
                        if not bool(mask_b[bi, t]):
                            sb = step_state_by_byte(sb, int(raw_ledger[bi, t].item()))
                else:
                    sb = _ledger_state_before(raw_ledger[bi], p)
                state_before_idx[bi] = kernel.state_index(sb)
            probs = torch.softmax(heads["byte_logits"].to(dtype=torch.float32), dim=-1)
            occupation = _next_state_occupation(probs, state_before_idx)
            out["replay_valid"] = replay_valid
            out["next_state_occupation"] = occupation
            out["shell_distribution"] = _shell_distribution(occupation)
        return out


def _ledger_state_before(ledger: torch.Tensor, pos: int) -> int:
    state = int(GENE_MAC_REST)
    for t in range(pos):
        state = step_state_by_byte(state, int(ledger[t].item()))
    return state


def _bt(x: torch.Tensor, batch: int, steps: int) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.shape[0] == 1 and batch > 1:
        x = x.expand(batch, -1)
    return x[:, :steps]


def _scan_atom_features(
    traj: ScanTrajectory, batch: int, steps: int, device: torch.device
) -> torch.Tensor:
    """Visible scan fields as concatenated bits/scalars for the atom encoder."""
    fam = _bt(traj.family_phase, batch, steps).to(device=device)
    micro = _bt(traj.micro_ref, batch, steps).to(device=device)
    q6 = _bt(traj.q6, batch, steps).to(device=device)
    shell = _bt(traj.chi_shell, batch, steps).to(device=device, dtype=torch.float32)
    fold = _bt(traj.fold_disagreement, batch, steps).to(
        device=device, dtype=torch.float32
    )
    psig = _bt(traj.prefix_sig, batch, steps).to(device=device)
    fam_oh = torch.nn.functional.one_hot(fam.clamp(0, 3), 4).to(dtype=torch.float32)
    micro_bits = _raw_byte_bits(micro)[..., :6]
    q_bits = _raw_byte_bits(q6)[..., :6]
    shifts = torch.arange(13, device=device)
    sig_bits = (
        torch.bitwise_right_shift(psig.unsqueeze(-1), shifts) & 1
    ).to(dtype=torch.float32)
    return torch.cat(
        [
            fam_oh,
            micro_bits,
            q_bits,
            (shell / 6.0).unsqueeze(-1),
            (fold / 4.0).unsqueeze(-1),
            sig_bits,
        ],
        dim=-1,
    )


def signature_posterior_uv(
    bytes_t: torch.Tensor,
    mask_b: torch.Tensor,
    byte_probs: torch.Tensor,
) -> torch.Tensor:
    """Posterior over (tau_u, tau_v); parity is length-mod-2.

    ``byte_probs`` is ``[B, 256]`` (single hidden site) or ``[B, T, 256]``.
    Visible bytes are point masses under the exact semidirect register law.
    """
    if byte_probs.ndim == 3 and int(mask_b.sum(dim=1).max().item()) > 1:
        # Per-position potentials are defined; keep the loop site-local below.
        pass
    elif byte_probs.ndim == 2 and int(mask_b.to(dtype=torch.long).sum(dim=1).max().item()) > 1:
        raise ValueError("signature posterior with [B,256] probs is single-hidden only")
    tbl = _byte_tables()
    tu = torch.as_tensor(tbl["tau_u"], device=bytes_t.device, dtype=torch.long)
    tv = torch.as_tensor(tbl["tau_v"], device=bytes_t.device, dtype=torch.long)
    batch, t_len = bytes_t.shape
    idx = torch.arange(4096, device=bytes_t.device)
    u0 = torch.bitwise_right_shift(idx, 6)
    v0 = idx & 63
    dest = torch.bitwise_left_shift(v0.unsqueeze(0) ^ tu.unsqueeze(1), 6) | (
        u0.unsqueeze(0) ^ tv.unsqueeze(1)
    )
    dist = torch.zeros(batch, 4096, device=bytes_t.device, dtype=torch.float32)
    dist[:, 0] = 1.0
    hidden_pos = mask_b.to(dtype=torch.bool)
    for t in range(t_len):
        nxt = torch.zeros_like(dist)
        vis = ~hidden_pos[:, t]
        if vis.any():
            vis_idx = vis.nonzero(as_tuple=False).view(-1)
            drow = dest[bytes_t[vis_idx, t].clamp(0, 255)]
            tmp = torch.zeros(vis_idx.shape[0], 4096, device=dist.device, dtype=dist.dtype)
            tmp.scatter_add_(1, drow, dist[vis_idx])
            nxt[vis_idx] = tmp
        hid = hidden_pos[:, t]
        if hid.any():
            hid_idx = hid.nonzero(as_tuple=False).view(-1)
            h = int(hid_idx.numel())
            mixed = torch.zeros(h, 4096, device=dist.device, dtype=dist.dtype)
            site_p = byte_probs[:, t] if byte_probs.ndim == 3 else byte_probs
            src = (dist[hid_idx].unsqueeze(1) * site_p[hid_idx].unsqueeze(2)).reshape(
                h, -1
            )
            idx = dest.unsqueeze(0).expand(h, -1, -1).reshape(h, -1)
            mixed.scatter_add_(1, idx, src)
            nxt[hid_idx] = mixed
        dist = nxt
    return dist.reshape(batch, 64, 64)


def _class_logits_from_int(
    values: torch.Tensor, n_bits: int, scale: float = 8.0
) -> torch.Tensor:
    """Lift integer bit-fields to a 2^n_bits-way logit table (LSB first)."""
    shifts = torch.arange(n_bits, device=values.device)
    bits = torch.bitwise_right_shift(values.unsqueeze(-1), shifts) & 1
    bit_logits = (bits.to(dtype=torch.float32) * 2.0 - 1.0) * scale
    table = torch.arange(1 << n_bits, device=values.device)
    signs = (
        torch.bitwise_right_shift(table.unsqueeze(1), shifts) & 1
    ).to(dtype=torch.float32) * 2.0 - 1.0
    return bit_logits @ signs.T


def _visible_signature_logits(
    bytes_t: torch.Tensor, visible: torch.Tensor
) -> dict[str, torch.Tensor]:
    """XOR-register readout of the visible word (Theorem 2).

    Per-byte factors are affine in the intron bits; composition is a 1-bit
    parity register plus two 6-bit XOR registers with a parity-indexed swap.
    Masked / padded sites are skipped. No kernel lookup.
    """
    aa = torch.as_tensor(0xAA, device=bytes_t.device, dtype=bytes_t.dtype)
    intron = torch.bitwise_xor(bytes_t, aa)
    i0 = intron & 1
    i7 = torch.bitwise_right_shift(intron, 7) & 1
    micro = torch.bitwise_right_shift(intron, 1) & 63
    tau_u_b = i0 * 63
    tau_v_b = torch.bitwise_xor(micro, i7 * 63)
    batch = bytes_t.shape[0]
    p = torch.zeros(batch, dtype=torch.long, device=bytes_t.device)
    u = torch.zeros(batch, dtype=torch.long, device=bytes_t.device)
    v = torch.zeros(batch, dtype=torch.long, device=bytes_t.device)
    vis = visible.to(dtype=torch.bool, device=bytes_t.device)
    for t in range(bytes_t.shape[1]):
        # compose(byte_sig, running): new byte is left, prefix is right.
        ru = v
        rv = u
        np_ = p ^ 1
        nu = ru ^ tau_u_b[:, t]
        nv = rv ^ tau_v_b[:, t]
        take = vis[:, t]
        p = torch.where(take, np_, p)
        u = torch.where(take, nu, u)
        v = torch.where(take, nv, v)
    return {
        "parity_logits": _class_logits_from_int(p, 1),
        "tau_u_logits": _class_logits_from_int(u, 6),
        "tau_v_logits": _class_logits_from_int(v, 6),
    }


_TRANSITION_TABLE_T: dict[str, torch.Tensor] = {}


def _transition_table_tensor(device: torch.device) -> torch.Tensor:
    key = str(device)
    cached = _TRANSITION_TABLE_T.get(key)
    if cached is None:
        from src.tools.autoencoder.datasets import transition_table

        cached = torch.as_tensor(
            np.array(transition_table(), copy=True), device=device, dtype=torch.long
        )
        _TRANSITION_TABLE_T[key] = cached
    return cached


def _next_state_occupation(
    probs: torch.Tensor, state_before_idx: torch.Tensor
) -> torch.Tensor:
    """Scatter softmax(byte) through the dense kernel transition table."""
    table = _transition_table_tensor(probs.device)
    dest = table[state_before_idx.clamp(0, N_STATES - 1)]
    occ = torch.zeros(probs.shape[0], N_STATES, device=probs.device, dtype=probs.dtype)
    occ.scatter_add_(1, dest, probs)
    return occ


def _shell_distribution(occupation: torch.Tensor) -> torch.Tensor:
    """Sum occupation mass by chirality shell (0..6)."""
    idx = torch.arange(N_STATES, device=occupation.device)
    chi = torch.bitwise_xor(torch.bitwise_right_shift(idx, 6), idx & 63)
    shells = _popcount6(chi)
    dist = torch.zeros(
        occupation.shape[0],
        7,
        device=occupation.device,
        dtype=occupation.dtype,
    )
    dist.scatter_add_(
        1,
        shells.unsqueeze(0).expand(occupation.shape[0], -1),
        occupation,
    )
    return dist


def _popcount6(x: torch.Tensor) -> torch.Tensor:
    n = torch.zeros_like(x)
    v = x
    for _ in range(6):
        n = n + (v & 1)
        v = torch.bitwise_right_shift(v, 1)
    return n


def _replay_matches(
    true_ledger: torch.Tensor, pred_ledger: torch.Tensor
) -> torch.Tensor:
    """True when replaying pred_ledger reaches the same final state as true_ledger."""
    batch = true_ledger.shape[0]
    ok = torch.zeros(batch, dtype=torch.bool, device=true_ledger.device)
    for b in range(batch):
        state_t = int(GENE_MAC_REST)
        state_p = int(GENE_MAC_REST)
        for t in range(true_ledger.shape[1]):
            state_t = step_state_by_byte(state_t, int(true_ledger[b, t].item()))
            state_p = step_state_by_byte(state_p, int(pred_ledger[b, t].item()))
        ok[b] = state_t == state_p
    return ok
