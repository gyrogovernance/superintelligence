"""Models that build in no structure: raw bits in, prediction out.

This is the "narrow" tier. These models do not enforce the machine's group
symmetry by construction - they are the null baselines, the byte-mechanism
predictors, the reference encoding tables, and the percolation rank learner.
They are the controls against which the symmetry-constrained tiers
(models.general, models.super) are measured.

Class groups:
- Reference codecs: deterministic encoding tables (no learning).
- MLP baselines: plain feed-forward autoencoders (the null model).
- Task models: byte-conditioned predictors (next state, raw byte, word, frame).
- PercolationLearner: reads a byte-alphabet mask and predicts the rank.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src import api

from src.tools.autoencoder.kernel import (
    apply_k4_index,
    sig_id_parts,
    word_signature_id,
)

N_STATES = 4096
N_BITS = 12
N_BYTES = 256
N_MASK_BITS = 256


# ---------------------------------------------------------------------------
# Reference codecs: deterministic encoding tables (no learning)
# ---------------------------------------------------------------------------


class ExactUVCodec:
    """Lossless 12-bit chart codec: state <-> (u6, v6)."""

    name = "exact_uv"

    def encode(self, index: int) -> tuple[int, int]:
        from src.tools.autoencoder.kernel import state24_from_index

        omega = api.state24_to_omega12(state24_from_index(index))
        return omega.u6, omega.v6

    def decode(self, code: tuple[int, int]) -> int:
        u6, v6 = code
        return (u6 << 6) | v6  # canonical index IS the chart

    def reconstruct_all(self) -> np.ndarray:
        return np.arange(N_STATES, dtype=np.uint16)


class BoundaryChiralityCodec:
    """Lossless alternative chart: (u6, chi6); v6 = u6 XOR chi6. Also 12 bits."""

    name = "boundary_chirality"

    def encode(self, index: int) -> tuple[int, int]:
        from src.tools.autoencoder.kernel import state24_from_index

        omega = api.state24_to_omega12(state24_from_index(index))
        return omega.u6, omega.chirality6

    def decode(self, code: tuple[int, int]) -> int:
        u6, chi6 = code
        return (u6 << 6) | (u6 ^ chi6)

    def reconstruct_all(self) -> np.ndarray:
        idx = np.arange(N_STATES, dtype=np.uint16)
        u6 = (idx >> 6) & 63
        chi6 = u6 ^ (idx & 63)
        return ((u6 << 6) | (u6 ^ chi6)).astype(np.uint16)


class ChiralityOnlyCodec:
    """Lossy 6-bit codec: keeps chi6, decodes to a uniform distribution over
    the 64 states sharing that chirality."""

    name = "chirality_only"

    def encode(self, index: int) -> int:
        from src.tools.autoencoder.kernel import state24_from_index

        omega = api.state24_to_omega12(state24_from_index(index))
        return omega.chirality6

    def fiber(self, chi6: int) -> np.ndarray:
        u6 = np.arange(64, dtype=np.uint16)
        v6 = u6 ^ np.uint16(chi6)
        return (u6 << 6) | v6

    def reconstruct_distribution(self, index: int) -> np.ndarray:
        dist = np.zeros(N_STATES, dtype=np.float32)
        dist[self.fiber(self.encode(index))] = 1.0 / 64
        return dist


class ShellOnlyCodec:
    """Lossy 3-bit codec: keeps shell_chi only; uniform over the shell states."""

    name = "shell_only"

    def encode(self, index: int) -> int:
        from src.tools.autoencoder.kernel import state24_from_index

        omega = api.state24_to_omega12(state24_from_index(index))
        return omega.chirality6.bit_count()

    def shell_states(self, shell_chi: int) -> np.ndarray:
        idx = np.arange(N_STATES, dtype=np.uint16)
        u6 = (idx >> 6) & 63
        v6 = idx & 63
        chi6 = u6 ^ v6
        shell = np.zeros(N_STATES, dtype=np.uint8)
        for bit in range(6):
            shell += (chi6 >> bit) & 1
        return idx[shell == shell_chi]

    def reconstruct_distribution(self, index: int) -> np.ndarray:
        dist = np.zeros(N_STATES, dtype=np.float32)
        members = self.shell_states(self.encode(index))
        dist[members] = 1.0 / len(members)
        return dist


# ---------------------------------------------------------------------------
# MLP baselines: plain feed-forward autoencoders (the null model)
# ---------------------------------------------------------------------------


def state_bits(state_index: torch.Tensor) -> torch.Tensor:
    """[B] -> [B, 12] one-hot-free bit features of a 12-bit Omega state."""
    u6 = torch.bitwise_right_shift(state_index, 6) & 63
    v6 = state_index & 63
    bits: list[torch.Tensor] = []
    for i in range(6):
        bits.append((torch.bitwise_right_shift(u6, i) & 1).float().unsqueeze(-1))
    for i in range(6):
        bits.append((torch.bitwise_right_shift(v6, i) & 1).float().unsqueeze(-1))
    return torch.cat(bits, dim=-1)


def state_to_bits(index: torch.Tensor) -> torch.Tensor:
    """[B] state indices -> [B, 12] binary Omega coordinates."""
    return state_bits(index)


class MLPEncoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_BITS, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.output_dim = latent_dim

    def forward(self, state_index: torch.Tensor) -> torch.Tensor:
        return self.net(state_to_bits(state_index))


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, N_STATES),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MLPAutoencoder(nn.Module):
    """Ordinary MLP baseline: no equivariance enforcement (null model)."""

    def __init__(self, latent_dim: int = 8, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = MLPEncoder(latent_dim, hidden_dim)
        self.decoder = MLPDecoder(latent_dim, hidden_dim)

    def forward(self, state_index: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(state_index))

    def encode(self, state_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(state_index)

    def get_config(self) -> dict:
        return {
            "latent_dim": self.encoder.output_dim,
            "hidden_dim": self.encoder.net[0].out_features,
        }

    def encoder_fn(self):
        model = self

        class _Fn:
            def eval(self) -> None:
                model.eval()

            def __call__(self, state_index: torch.Tensor) -> torch.Tensor:
                return model.encoder(state_index)

        return _Fn()


class SymmetryRegularizedAutoencoder(nn.Module):
    """Symmetry-regularized baseline: a plain autoencoder whose training may
    add a soft equivariance penalty over exact K4 generators."""

    def __init__(self, latent_dim: int = 8, hidden_dim: int = 128) -> None:
        super().__init__()
        self.base = MLPAutoencoder(latent_dim, hidden_dim)
        self.latent_dim = latent_dim

    @property
    def encoder(self) -> nn.Module:
        return self.base.encoder

    @property
    def decoder(self) -> nn.Module:
        return self.base.decoder

    def forward(self, state_index: torch.Tensor) -> torch.Tensor:
        return self.base(state_index)


def k4_generator_batch(
    state_index: torch.Tensor, k4_perm: torch.Tensor
) -> list[torch.Tensor]:
    """Exact K4 transforms of a state batch via the kernel action table.

    k4_perm: [4, 4096] long tensor; returns the 4 transformed index batches
    (id, S, C, F)."""
    return [k4_perm[gate_i][state_index] for gate_i in range(4)]


def soft_equivariance_loss(
    encoder: nn.Module,
    state_index: torch.Tensor,
    k4_perm: torch.Tensor,
    latent_action=None,
) -> torch.Tensor:
    """Mean per-sample encoder equivariance defect over K4 generators.

    The equivariance target is the declared latent representation rho(g):
    latent_action(gate_i, z) must return rho(g) z for the chosen latent
    layout. A scalar latent that transforms as z -> -z under a gate is a valid
    sign representation: the defect is ||E(g.x) - rho(g) E(x)||. Without a
    declared latent_action the target is undefined and this raises."""
    from src.tools.autoencoder.helpers.training_losses import equivariance_defect

    if latent_action is None:
        raise ValueError(
            "soft_equivariance_loss requires a declared latent_action(gate_i, z) "
            "returning rho(g) z; without a latent representation the "
            "equivariance target is undefined."
        )
    return equivariance_defect(
        encoder, state_index, k4_perm, latent_action
    )


# ---------------------------------------------------------------------------
# Task models: byte-conditioned predictors
# ---------------------------------------------------------------------------


_BYTE_FEATURE_CACHE: torch.Tensor | None = None


def byte_features(byte: torch.Tensor) -> torch.Tensor:
    """Structured byte encoding: [family bits (2), micro bits (6), q bits (6)].

    Uses exact kernel census tables, cached at module level. Returns [B, 14]."""
    global _BYTE_FEATURE_CACHE
    if _BYTE_FEATURE_CACHE is None:
        from src.tools.autoencoder.datasets import byte_census_arrays

        census = byte_census_arrays()
        fam = torch.as_tensor(census["family_u2"].astype(np.int64))
        micro = torch.as_tensor(census["micro_ref_u6"].astype(np.int64))
        q = torch.as_tensor(census["q6"].astype(np.int64))
        rows = []
        for b in range(N_BYTES):
            fb = [(int(fam[b]) >> 1) & 1, int(fam[b]) & 1]
            mb = [(int(micro[b]) >> i) & 1 for i in range(6)]
            qb = [(int(q[b]) >> i) & 1 for i in range(6)]
            rows.append(fb + mb + qb)
        _BYTE_FEATURE_CACHE = torch.tensor(rows, dtype=torch.float32)
    return _BYTE_FEATURE_CACHE[byte]


def _bits_of_tensor(value: torch.Tensor, width: int) -> torch.Tensor:
    """[B] -> [B, width] binary features via broadcast (no Python loop)."""
    shifts = torch.arange(width, device=value.device).unsqueeze(0)
    bits = (torch.bitwise_right_shift(value.unsqueeze(-1), shifts) & 1).float()
    return bits


class TransitionModel(nn.Module):
    """(state bits, structured byte) -> next-state logits over 4096."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(12 + 14, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, N_STATES),
        )

    def get_config(self) -> dict:
        return {"hidden_dim": self._hidden_dim}

    def forward(self, state_index: torch.Tensor, byte: torch.Tensor) -> torch.Tensor:
        state_feats = state_bits(state_index)
        byte_feats = byte_features(byte)
        return self.net(torch.cat([state_feats, byte_feats], dim=-1))


class RawByteTransitionModel(nn.Module):
    """(raw 8-bit byte, raw 12-bit state index) -> next-state logits (4096).

    No hand-decomposed family/micro/q features; the model must learn the byte
    mechanism from raw coordinates. Shares state_bits with TransitionModel.
    """

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(12 + 8, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, N_STATES),
        )

    def get_config(self) -> dict:
        return {"hidden_dim": self._hidden_dim}

    def forward(self, state_index: torch.Tensor, byte: torch.Tensor) -> torch.Tensor:
        byte_feats = _bits_of_tensor(byte, 8)
        return self.net(torch.cat([state_bits(state_index), byte_feats], dim=-1))


class WordActionModel(nn.Module):
    """word -> OmegaSignature12 as an algebra-aware accumulator.

    Per-byte heads predict the byte's translation part (tau_u6, tau_v6) from an
    exact kernel-derived feature basis; the word signature is composed EXACTLY
    with the kernel group law, so model(w1 + w2) == compose(model(w1), model(w2))
    holds structurally.
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self.byte_embedding = nn.Embedding(256, 16)
        self.tau_head = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 128),  # 64 tau_u + 64 tau_v logits
        )

    def get_config(self) -> dict:
        return {"hidden_dim": self._hidden_dim}

    def byte_logits(self, bytes_seq: torch.Tensor) -> torch.Tensor:
        """[B] bytes -> [B, 128] logits for (tau_u6, tau_v6)."""
        return self.tau_head(self.byte_embedding(bytes_seq))

    def forward(self, words: list[bytes]) -> dict[str, torch.Tensor]:
        """Eval-only composition path (non-differentiable by design)."""
        batch_u = torch.zeros(len(words), 64)
        batch_v = torch.zeros(len(words), 64)
        parity = torch.zeros(len(words), dtype=torch.long)
        for row, word in enumerate(words):
            parity[row] = len(word) & 1
            running = api.OmegaSignature12(0, 0, 0)  # identity
            for byte in word:
                logits = self.byte_logits(torch.tensor([byte]))[0]
                byte_sig = api.OmegaSignature12(
                    1, int(logits[:64].argmax()), int(logits[64:].argmax())
                )
                running = api.compose_omega_signatures(byte_sig, running)
            batch_u[row, running.tau_u6] = 1.0
            batch_v[row, running.tau_v6] = 1.0
        return {
            "parity_logits": torch.stack(
                [1.0 - parity.float(), parity.float()], dim=-1
            )
            * 20.0,
            "tau_u_logits": batch_u * 20.0,
            "tau_v_logits": batch_v * 20.0,
        }


def compositional_consistency(
    model: WordActionModel, words: list[tuple[bytes, bytes]]
) -> float:
    """Structural compositionality of the model's own predictions.

    model(w1 + w2) must equal compose(model(w1), model(w2)) using the model's
    argmax per-byte signatures. Returns the fraction of exact agreements."""
    lefts = [left for left, _ in words]
    rights = [right for _, right in words]
    concats = [left + right for left, right in words]
    pred_left = model(lefts)
    pred_right = model(rights)
    pred_concat = model(concats)
    ok = 0
    for row in range(len(words)):
        sig_concat = _argmax_signature(pred_concat, row)
        sig_left = _argmax_signature(pred_left, row)
        sig_right = _argmax_signature(pred_right, row)
        left_sig = api.OmegaSignature12(
            (sig_left >> 12) & 1, (sig_left >> 6) & 63, sig_left & 63
        )
        right_sig = api.OmegaSignature12(
            (sig_right >> 12) & 1, (sig_right >> 6) & 63, sig_right & 63
        )
        composed = api.compose_omega_signatures(right_sig, left_sig)
        packed = (composed.parity << 12) | (composed.tau_u6 << 6) | composed.tau_v6
        if sig_concat == packed:
            ok += 1
    return ok / max(1, len(words))


def _argmax_signature(preds: dict[str, torch.Tensor], row: int) -> int:
    parity = int(preds["parity_logits"][row].argmax())
    tau_u = int(preds["tau_u_logits"][row].argmax())
    tau_v = int(preds["tau_v_logits"][row].argmax())
    return (parity << 12) | (tau_u << 6) | tau_v


def signature_to_bits(sig_id: int) -> list[int]:
    """Pack a 13-bit Omega signature (parity, tau_u6, tau_v6) into a bit list."""
    parity = (sig_id >> 12) & 1
    tau_u6 = (sig_id >> 6) & 63
    tau_v6 = sig_id & 63
    return [int(parity)] + _bits_of(tau_u6, 6) + _bits_of(tau_v6, 6)


def _bits_of(value: int, width: int) -> list[int]:
    return [int((value >> i) & 1) for i in range(width)]


class FrameHead(nn.Module):
    """Frame head: from the 32-bit intron sequence to the compiled frame
    signature and the staged final state.

    The four introns determine both the signature and the final state. The head
    predicts the 13-bit packed signature (parity, tau_u6, tau_v6) as independent
    bits and the 4096-way final state.
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.sig_head = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 13),
        )
        self.state_head = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, N_STATES),
        )

    def forward(self, intron_seq32: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = _bits_of_tensor(intron_seq32, 32)
        return {
            "signature_logits": self.sig_head(feats),
            "state_logits": self.state_head(feats),
        }


# ---------------------------------------------------------------------------
# PercolationLearner: reads a byte-alphabet mask and predicts the rank
# ---------------------------------------------------------------------------


class PercolationLearner(nn.Module):
    """Small supervised head over the packed 256-bit allowed mask.

    Input: a packed 256-bit allowed byte mask (32 uint8 entries per row).
    Targets (all supervised from the kernel's exact labels):
    - transport_rank: the GF(2)^6 rank of the q-class span (7 classes, rank 0..6);
    - reach_size: number of states reachable from the identity (a count);
    - full_reachability / horizon_spanning / giant: binary flags.

    Small and dependency-free (input 256, hidden 128, no batch-norm) so the
    source of an accuracy signal stays visible.
    """

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(N_MASK_BITS, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.rank_head = nn.Linear(hidden_dim, 7)
        self.reach_head = nn.Linear(hidden_dim, 1)
        self.full_head = nn.Linear(hidden_dim, 1)
        self.horizon_head = nn.Linear(hidden_dim, 1)
        self.giant_head = nn.Linear(hidden_dim, 1)

    def get_config(self) -> dict:
        return {"hidden_dim": self._hidden_dim}

    def forward(self, allowed_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """allowed_mask [B, 32] uint8 packed-bits -> named logits."""
        masks = allowed_mask.long()
        shifts = torch.arange(8, device=masks.device)
        bits = torch.bitwise_right_shift(masks.unsqueeze(-1), shifts)
        bits = (bits & 1).float()
        feats = bits.reshape(masks.shape[0], N_MASK_BITS)
        h = self.net(feats)
        return {
            "rank_logits": self.rank_head(h),
            "reach_logits": self.reach_head(h).squeeze(-1),
            "full_logits": self.full_head(h).squeeze(-1),
            "horizon_logits": self.horizon_head(h).squeeze(-1),
            "giant_logits": self.giant_head(h).squeeze(-1),
        }

    def predict_rank(self, allowed_mask: torch.Tensor) -> torch.Tensor:
        """[B] argmax rank labels for the given masks."""
        logits = self.forward(allowed_mask)["rank_logits"]
        return logits.argmax(dim=-1)
