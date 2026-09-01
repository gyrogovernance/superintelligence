"""Model registry and plug.

This file is the single entry point for constructing any model. It owns:

- ``UnifiedAutoencoder`` (the selector over the symmetry ladder) plus its
  optional ``MultiTaskHeads``; the unified model IS the selector, so it lives
  here rather than in a tier file;
- ``build_model``, the constructor the CLI and tests use.

The three tier files hold the model classes:

- ``narrow``  - no structure built in (codecs, MLP, byte-mechanism predictors,
  percolation learner);
- ``general``  - builds in the K4 gate symmetry;
- ``super``    - builds in the full group / multi-register structure.

A new model joins the tier whose symmetry it builds in. A new *file* is only
created for a new tier - never for a new naming idea. Flags are frozen; the
tier names and ``all`` are added as selectors below.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import general as _general
from . import super as _super

N_STATES = 4096
N_BLOCKS = 2080


class MultiTaskHeads(nn.Module):
    """Optional task heads over the spectral carrier latent (the "super" code).

    Heads are opt-in: an autoencoder constructed without them is bit-identical
    to the plain unified model. The heads read, for each task, features built
    from the gated Walsh spectrum PLUS the raw state identity, so they can
    learn state-conditioned maps (next state, word) rather than only
    invariant statistics:

    - a signed per-block pooling of the gated spectrum (state-dependent: the
      coefficient signs differ per state, unlike the abs-pooled vector which
      is identical for every one-hot state);
    - a small learned embedding of the raw state index (full state identity);
    - the structured byte features (for the byte-conditioned tasks).

    Reading the equivariant code plus the state identity leaves the codec's
    exact equivariance untouched - the heads consume features, they do not
    mutate the codec.

    - transition_head: block features + state embedding + byte -> next-state logits;
    - word_head: block features + state embedding + byte -> (tau_u6, tau_v6) logits;
    - rank_head: packed 256-bit allowed mask -> 7-class transport rank.
    """

    def __init__(
        self,
        n_blocks: int = N_BLOCKS,
        hidden_dim: int = 128,
        state_embed_dim: int = 32,
    ) -> None:
        nn.Module.__init__(self)
        self.n_blocks = n_blocks
        self.state_embed_dim = state_embed_dim
        # Raw state index carries full state identity for the byte-conditioned
        # tasks; the heads need it to learn a state-conditioned next-state map.
        self.state_embed = nn.Embedding(N_STATES, state_embed_dim)
        head_in = n_blocks + state_embed_dim + 14
        self.transition_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, N_STATES),
        )
        self.word_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 128),
        )
        self.rank_head = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 7),
        )

    def _feats(self, block_feats: torch.Tensor, state_index: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [block_feats, self.state_embed(state_index.long())], dim=-1
        )

    def transition_logits(
        self, block_feats: torch.Tensor, byte: torch.Tensor, state_index: torch.Tensor
    ) -> torch.Tensor:
        from .narrow import byte_features

        feats = self._feats(block_feats, state_index)
        return self.transition_head(torch.cat([feats, byte_features(byte)], dim=-1))

    def word_logits(
        self, block_feats: torch.Tensor, byte: torch.Tensor, state_index: torch.Tensor
    ) -> torch.Tensor:
        from .narrow import byte_features

        feats = self._feats(block_feats, state_index)
        return self.word_head(torch.cat([feats, byte_features(byte)], dim=-1))

    def rank_logits(self, allowed_mask: torch.Tensor) -> torch.Tensor:
        """allowed_mask [B, 32] uint8 packed -> [B, 7] rank logits."""
        masks = allowed_mask.long()
        shifts = torch.arange(8, device=masks.device)
        bits = torch.bitwise_right_shift(masks.unsqueeze(-1), shifts)
        bits = (bits & 1).float()
        return self.rank_head(bits.reshape(masks.shape[0], 256))


_BLOCK_MEMBERSHIP_CACHE: dict[tuple, torch.Tensor] = {}


def _block_membership(block_id: torch.Tensor, n_blocks: int, device, dtype) -> torch.Tensor:
    """Cached [4096, n_blocks] block-membership matrix (one-hot over block id)."""
    # Key on the tensor's Python object id (not data_ptr): the kernel
    # block_id buffer is registered once and is the same object for the
    # whole process, so the cache hit-rate is exactly 1. data_ptr is
    # unreliable because a freed-and-reallocated tensor can reuse the
    # same address and serve a stale cached entry.
    key = (id(block_id), block_id.shape[0], n_blocks, device, dtype)
    m = _BLOCK_MEMBERSHIP_CACHE.get(key)
    if m is not None:
        return m
    blk = block_id.to(device)
    m = torch.zeros((blk.numel(), n_blocks), device=device, dtype=dtype)
    m.scatter_(1, blk.unsqueeze(1), 1.0)
    _BLOCK_MEMBERSHIP_CACHE[key] = m
    return m


def block_features(
    spectrum: torch.Tensor,
    block_id: torch.Tensor,
    n_blocks: int,
    signed: bool = False,
) -> torch.Tensor:
    """Per-block pooled features of a gated Walsh spectrum.

    spectrum [B, 4096] Walsh coefficients; block_id [4096] maps each
    coefficient to its irrep block. With ``signed=False`` (the default) it
    returns the mean-abs pooling inside each block, which is invariant to the
    within-block sign/permutation action and therefore equivariance-safe. With
    ``signed=True`` it returns the mean signed pooling, which preserves the
    per-state coefficient sign pattern and so is state-dependent (every one-hot
    state has the same |coefficient| but a different sign arrangement).

    The unified task heads use ``signed=True`` so they can learn state-
    conditioned maps; the invariant ``signed=False`` form remains available for
    tasks that must be equivariant by construction.
    """
    membership = _block_membership(block_id, n_blocks, spectrum.device, spectrum.dtype)
    inside = spectrum if signed else spectrum.abs()
    sums = inside @ membership
    counts = membership.sum(dim=0).clamp_min(1.0)
    return sums / counts


class UnifiedAutoencoder(nn.Module):
    """Single autoencoder with symmetry level ``free`` | ``k4`` | ``full``."""

    def __init__(
        self,
        symmetry: str = "full",
        hidden_dim: int = 128,
        latent_dim: int = 32,
        ladder: str = "full",
        init_gain: float = 1.0,
        heads: tuple[str, ...] | list[str] | None = None,
        sector_mask: np.ndarray | None = None,
        orbit_index: np.ndarray | None = None,
    ) -> None:
        nn.Module.__init__(self)
        if symmetry not in ("free", "k4", "full"):
            raise ValueError(f"unknown symmetry level: {symmetry}")
        self.symmetry = symmetry
        self.ladder = ladder if symmetry == "full" else None
        self.heads_kind: tuple[str, ...] = tuple(heads or ())
        self.n_sign = latent_dim // 4
        self.n_trivial = latent_dim - 3 * self.n_sign
        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim

        self.task_heads = None
        if self.heads_kind and symmetry == "full":
            self.task_heads = MultiTaskHeads(N_BLOCKS, hidden_dim)
        elif self.heads_kind:
            raise ValueError("multi-task heads require symmetry='full'")

        if symmetry == "full":
            self.encoder = None
            self.decoder = None
            self.spectral = _super.SpectralAutoencoder(
                init_gain=init_gain,
                ladder=ladder,
                sector_mask=sector_mask,
                orbit_index=orbit_index,
            )
        else:
            base_in = 12
            self.encoder = nn.Sequential(
                nn.Linear(base_in, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, N_STATES),
            )
            self.spectral = None

        if symmetry in ("k4",):
            self.register_buffer("k4_action", torch.as_tensor(_general.k4_action_matrix()))

    def encode(self, state_index: torch.Tensor) -> torch.Tensor:
        if self.symmetry == "full":
            assert self.spectral is not None
            onehot = torch.zeros(
                (state_index.shape[0], N_STATES),
                device=state_index.device,
                dtype=torch.get_default_dtype(),
            )
            onehot[torch.arange(state_index.shape[0], device=state_index.device), state_index] = 1.0
            coeff = self.spectral.walsh_coefficients(onehot)
            return self.spectral.bottleneck(coeff, self.spectral.block_id)
        assert self.encoder is not None
        bits = self._bits_from_index(state_index)
        if self.symmetry == "k4":
            z = torch.zeros(
                (state_index.shape[0], self.encoder[-1].out_features),
                device=bits.device,
                dtype=bits.dtype,
            )
            characters = [
                (1, 1, 1, 1),
                (1, -1, 1, -1),
                (1, 1, -1, -1),
                (1, -1, -1, 1),
            ]
            for gate_i in range(4):
                g_bits = self._bits_after_gate(state_index, gate_i)
                g_out = self.encoder(g_bits)
                signs = characters[gate_i]
                nt = self.n_trivial
                z[:, :nt] += signs[0] * g_out[:, :nt]
                for block in range(3):
                    start = nt + block * self.n_sign
                    end = start + self.n_sign
                    z[:, start:end] += signs[block + 1] * g_out[:, start:end]
            return z / 4.0
        return self.encoder(bits)

    @staticmethod
    def _bits_from_index(index: torch.Tensor) -> torch.Tensor:
        u6 = torch.bitwise_right_shift(index, 6) & 63
        v6 = index & 63
        out = []
        for bit in range(6):
            out.append((torch.bitwise_right_shift(u6, bit) & 1).float())
        for bit in range(6):
            out.append((torch.bitwise_right_shift(v6, bit) & 1).float())
        return torch.stack(out, dim=-1)

    def _bits_after_gate(self, index: torch.Tensor, gate_i: int) -> torch.Tensor:
        assert self.k4_action is not None
        dest = self.k4_action[gate_i][index]
        return self._bits_from_index(dest)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.symmetry == "full":
            assert self.spectral is not None
            return self.spectral.inverse_walsh(latent)
        assert self.decoder is not None
        return self.decoder(latent)

    def forward(self, state_index: torch.Tensor) -> torch.Tensor:
        if self.symmetry == "full":
            assert self.spectral is not None
            onehot = torch.zeros(
                (state_index.shape[0], N_STATES),
                device=state_index.device,
                dtype=torch.get_default_dtype(),
            )
            onehot[torch.arange(state_index.shape[0], device=state_index.device), state_index] = 1.0
            coeff = self.spectral.walsh_coefficients(onehot)
            gated = self.spectral.bottleneck(coeff, self.spectral.block_id)
            return self.spectral.inverse_walsh(gated)
        z = self.encode(state_index)
        if self.symmetry == "k4":
            return self._decode_sym_k4(z)
        return self.decode(z)

    def _decode_sym_k4(self, z: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((z.shape[0], N_STATES), device=z.device, dtype=z.dtype)
        characters = [
            (1, 1, 1, 1),
            (1, -1, 1, -1),
            (1, 1, -1, -1),
            (1, -1, -1, 1),
        ]
        nt = self.n_trivial
        blocks = [nt] + [self.n_sign] * 3
        offsets = [0, nt, nt + self.n_sign, nt + 2 * self.n_sign]
        for gate_i, char in enumerate(characters):
            zg = z.clone()
            for c in range(4):
                zg[..., offsets[c] : offsets[c] + blocks[c]] *= char[c]
            d0 = self.decode(zg)
            perm = self.k4_action[gate_i].long()
            out.index_add_(1, perm, d0)
        return out / 4.0

    @torch.no_grad()
    def load_from(self, standalone: nn.Module) -> None:
        """Initialize so that unified(symmetry) matches a standalone model.

        For k4 copy the encoder/decoder linear weights of K4Autoencoder; for
        full copy the spectral bottleneck gain and sector mask. Enables exact-
        equality equivalence tests."""
        if self.symmetry == "k4" and isinstance(standalone, _general.K4Autoencoder):
            self._copy_linear(self.encoder[0], standalone.base_encoder[0])
            self._copy_linear(self.encoder[2], standalone.base_encoder[2])
            self._copy_linear(self.decoder[0], standalone.base_decoder[0])
            self._copy_linear(self.decoder[2], standalone.base_decoder[2])
        elif self.symmetry == "full" and isinstance(standalone, _super.SpectralAutoencoder):
            assert self.spectral is not None
            self.spectral.bottleneck.gain.copy_(standalone.bottleneck.gain)
            self.spectral.bottleneck.sector_mask.copy_(standalone.bottleneck.sector_mask)

    @staticmethod
    def _copy_linear(dst: nn.Linear, src: nn.Linear) -> None:
        dst.weight.copy_(src.weight)
        if dst.bias is not None and src.bias is not None:
            dst.bias.copy_(src.bias)

    def get_config(self) -> dict:
        return {
            "symmetry": self.symmetry,
            "hidden_dim": self._hidden_dim,
            "latent_dim": self._latent_dim,
            "ladder": self.ladder,
            "heads": list(self.heads_kind),
        }

    def gated_spectrum(self, state_index: torch.Tensor) -> torch.Tensor:
        assert self.spectral is not None, "gated_spectrum requires symmetry='full'"
        onehot = torch.zeros(
            (state_index.shape[0], N_STATES),
            device=state_index.device,
            dtype=torch.get_default_dtype(),
        )
        onehot[torch.arange(state_index.shape[0], device=state_index.device), state_index] = 1.0
        coeff = self.spectral.walsh_coefficients(onehot)
        return self.spectral.bottleneck(coeff, self.spectral.block_id)

    def per_block_features(self, state_index: torch.Tensor) -> torch.Tensor:
        """[B, N_BLOCKS] signed per-block pooled features of the gated spectrum.

        Uses signed pooling (not abs) so the per-state coefficient sign pattern
        is preserved - the unified task heads then see a state-dependent vector
        and can learn state-conditioned maps (next state, word) instead of only
        invariant statistics. The raw state identity is added separately by the
        heads via their state embedding.
        """
        assert self.spectral is not None
        gated = self.gated_spectrum(state_index)
        return block_features(gated, self.spectral.block_id, N_BLOCKS, signed=True)


# ---------------------------------------------------------------------------
# Registry + constructor
# ---------------------------------------------------------------------------

# Individual model kinds (flags, frozen).
MODEL_KINDS = (
    "mlp",
    "k4",
    "spectral",
    "transition",
    "rawbyte",
    "word",
    "unified",
    "percolation",
)

# Tier selectors: a tier name builds every model in that tier (eval/benchmark
# use). ``all`` builds everything.
TIER_MEMBERS = {
    "narrow": ("mlp", "transition", "rawbyte", "word", "percolation"),
    "general": ("k4",),
    "super": ("spectral",),
}
TIER_MEMBERS["all"] = tuple(MODEL_KINDS)


# Per-kind tuned hidden widths. These are the single source of truth for the
# constructor defaults; the CLI no longer keeps a parallel copy.
_HIDDEN_DEFAULTS = {
    "mlp": 64,
    "k4": 64,
    "transition": 128,
    "rawbyte": 256,
    "word": 64,
    "percolation": 128,
    "unified": 128,
}


def build_model(
    kind: str,
    symmetry: str | None = None,
    ladder: str | None = None,
    hidden_dim: int | None = None,
    latent_dim: int = 8,
    heads: tuple[str, ...] | list[str] | None = None,
    n_trivial: int | None = None,
    n_sign: int | None = None,
    sector_mask: np.ndarray | None = None,
    orbit_index: np.ndarray | None = None,
) -> nn.Module:
    """Construct a model instance by kind.

    ``kind`` is one of ``MODEL_KINDS``, a ``spectral:<ladder>`` rung select, or
    a tier selector (``narrow`` / ``general`` / ``super`` / ``all``). For a
    tier selector this returns the first member model of that tier (the CLI
    trains one model at a time; tiers drive sweeps via the list in
    ``TIER_MEMBERS``). ``unified`` is the symmetry selector and accepts
    ``symmetry`` / ``ladder`` / ``heads``. ``hidden_dim`` overrides the
    per-kind default from ``_HIDDEN_DEFAULTS`` when given. ``n_trivial`` and
    ``n_sign`` (when given) override the K4 latent layout."""
    from .narrow import (
        MLPAutoencoder,
        PercolationLearner,
        RawByteTransitionModel,
        TransitionModel,
        WordActionModel,
    )

    if kind in TIER_MEMBERS:
        kind = TIER_MEMBERS[kind][0]

    h = hidden_dim if hidden_dim is not None else _HIDDEN_DEFAULTS.get(kind, 128)
    if kind == "mlp":
        return MLPAutoencoder(latent_dim=latent_dim, hidden_dim=h)
    if kind == "k4":
        return _general.K4Autoencoder(
            n_trivial=n_trivial if n_trivial is not None else 2,
            n_sign=n_sign if n_sign is not None else 2,
            hidden_dim=h,
        )
    if kind == "spectral":
        return _super.SpectralAutoencoder(
            init_gain=1.0,
            ladder=ladder or "full",
            sector_mask=sector_mask,
            orbit_index=orbit_index,
        )
    if kind == "transition":
        return TransitionModel(hidden_dim=h)
    if kind == "rawbyte":
        return RawByteTransitionModel(hidden_dim=h)
    if kind == "word":
        return WordActionModel(hidden_dim=h)
    if kind == "percolation":
        return PercolationLearner(hidden_dim=h)
    if kind == "unified":
        return UnifiedAutoencoder(
            symmetry=symmetry or "full",
            ladder=ladder or "full",
            hidden_dim=h,
            latent_dim=latent_dim,
            heads=heads,
            sector_mask=sector_mask,
            orbit_index=orbit_index,
        )
    if kind.startswith("spectral:"):
        kind_ladder = kind.split(":", 1)[1]
        if symmetry is not None:
            return UnifiedAutoencoder(
                symmetry=symmetry,
                ladder=kind_ladder,
                hidden_dim=h,
                sector_mask=sector_mask,
                orbit_index=orbit_index,
            )
        return _super.SpectralAutoencoder(
            ladder=kind_ladder,
            sector_mask=sector_mask,
            orbit_index=orbit_index,
        )
    raise ValueError(f"unknown model kind {kind!r}")
