"""The general tier: K4Autoencoder and AffineSpectralCodec.

K4Autoencoder is exactly equivariant under the Klein four-group {id, S, C, F}
by Reynolds symmetrization. AffineSpectralCodec is the Walsh occupation codec
with exact full affine-group equivariance.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.tools.autoencoder.kernel import apply_k4_index

K4_GATES = ("id", "S", "C", "F")

# Characters of V4 as sign patterns (+1/-1) per gate (id, S, C, F):
#   trivial:  + + + +
#   chi_S:    + - + -   (kernel of chi_S = {id, C})
#   chi_C:    + + - -
#   chi_S*chi_C = chi_F: + - - +
K4_CHARACTERS: tuple[tuple[int, ...], ...] = (
    (1, 1, 1, 1),
    (1, -1, 1, -1),
    (1, 1, -1, -1),
    (1, -1, -1, 1),
)


def k4_action_matrix() -> np.ndarray:
    """[4, 4096] int64 permutation matrix of the K4 action on state indices."""
    action = np.empty((4, 4096), dtype=np.int64)
    for gate_i, gate in enumerate(K4_GATES):
        for index in range(4096):
            action[gate_i, index] = apply_k4_index(index, gate)
    return action


class K4Autoencoder(nn.Module):
    """Exactly K4-equivariant autoencoder over the 4096-state simplex.

    Latent layout: n_trivial trivial-character channels followed by n_sign
    channels for each nontrivial character, i.e.
    z = [z_invariant (n_trivial), z_char_S (n_sign), z_char_C (n_sign), z_char_F (n_sign)].
    rho(g) acts as: z_invariant -> +z_invariant; each sign block scaled by its character.
    """

    def __init__(
        self,
        n_trivial: int = 2,
        n_sign: int = 2,
        hidden_dim: int = 64,
        k4_perm: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.n_trivial = n_trivial
        self.n_sign = n_sign
        self.latent_dim = n_trivial + 3 * n_sign
        perm = k4_action_matrix() if k4_perm is None else k4_perm
        self.register_buffer("k4_perm", torch.as_tensor(perm))
        chars = torch.tensor(K4_CHARACTERS, dtype=torch.float32)  # [4, 4]
        self.register_buffer("characters", chars)

        self.base_encoder = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )
        self.base_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 4096),
        )
        self._hidden_dim = hidden_dim

    def get_config(self) -> dict:
        return {
            "n_trivial": self.n_trivial,
            "n_sign": self.n_sign,
            "hidden_dim": self._hidden_dim,
        }

    @property
    def z_slices(self) -> dict[str, slice]:
        """Named latent components by K4 character sector.

        - z_invariant: trivial-character block, invariant under K4;
        - z_char_S: sign character of S (inverts under S and F);
        - z_char_C: sign character of C (inverts under C and F);
        - z_char_F: sign character of F (inverts under S and C).
        """
        s = self.n_sign
        t0 = 0
        t1 = self.n_trivial
        return {
            "z_invariant": slice(t0, t1),
            "z_char_S": slice(t1, t1 + s),
            "z_char_C": slice(t1 + s, t1 + 2 * s),
            "z_char_F": slice(t1 + 2 * s, t1 + 3 * s),
        }

    def named_components(self, state_index: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode and split the latent into named components."""
        z = self.encode(state_index)
        return {name: z[:, sl] for name, sl in self.z_slices.items()}

    def rho(self, gate_i: int, z: torch.Tensor) -> torch.Tensor:
        """Exact latent representation: signed scaling per character block."""
        signs = self.characters[:, gate_i]  # [4] values chi_block(gate_i)
        out = z.clone()
        out[:, : self.n_trivial] = signs[0] * z[:, : self.n_trivial]
        for block in range(3):
            start = self.n_trivial + block * self.n_sign
            end = start + self.n_sign
            out[:, start:end] = signs[block + 1] * z[:, start:end]
        return out

    def bits_from_index(self, index: torch.Tensor) -> torch.Tensor:
        u6 = torch.bitwise_right_shift(index, 6) & 63
        v6 = index & 63
        bits = []
        for bit in range(6):
            bits.append((torch.bitwise_right_shift(u6, bit) & 1).float())
        for bit in range(6):
            bits.append((torch.bitwise_right_shift(v6, bit) & 1).float())
        return torch.stack(bits, dim=-1)

    def bits_after_gate(self, index: torch.Tensor, gate_i: int) -> torch.Tensor:
        """12-bit chart of g.x without touching the base network."""
        dest = self.k4_perm[gate_i][index]
        return self.bits_from_index(dest)

    def encode(self, state_index: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(
            (state_index.shape[0], self.latent_dim),
            device=state_index.device,
            dtype=torch.get_default_dtype(),
        )
        for gate_i in range(4):
            g_bits = self.bits_after_gate(state_index, gate_i)
            g_out = self.base_encoder(g_bits)
            signs = self.characters[:, gate_i].to(g_out.dtype)
            z[:, : self.n_trivial] += signs[0] * g_out[:, : self.n_trivial]
            for block in range(3):
                start = self.n_trivial + block * self.n_sign
                end = start + self.n_sign
                z[:, start:end] += signs[block + 1] * g_out[:, start:end]
        return z / 4.0

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Symmetrize decoder output over the group with exact permutation
        # action on the 4096-simplex:
        #   D(z) = (1/4) sum_g P_g^-1 . D0(rho(g) z)
        # Every K4 element is an involution (P_g == P_g^-1), so we index with
        # perm directly.
        out = torch.zeros((z.shape[0], 4096), device=z.device, dtype=z.dtype)
        for gate_i in range(4):
            rho_z = self.rho(gate_i, z)
            d0 = self.base_decoder(rho_z)
            perm = self.k4_perm[gate_i].long()
            out.index_add_(1, perm, d0)
        return out / 4.0

    def forward(self, state_index: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(state_index))

    def encoder_eval(self):
        return _EncoderFn(self)

    def predict_index(self, state_index: torch.Tensor) -> torch.Tensor:
        logits = self.forward(state_index)
        return logits.argmax(dim=-1)


class _EncoderFn:
    def __init__(self, model: K4Autoencoder) -> None:
        self.model = model

    def __call__(self, state_index: torch.Tensor) -> torch.Tensor:
        return self.model.encode(state_index)

    def eval(self) -> None:
        self.model.eval()


N_STATES = 4096


def walsh_matrix_64() -> np.ndarray:
    """Unnormalized 64x64 Walsh-Hadamard matrix W[a, x] = (-1)^(dot(a,x))."""
    w = np.empty((64, 64), dtype=np.float32)
    for a in range(64):
        for x in range(64):
            bit = 1
            parity = 0
            while bit <= 63:
                parity ^= ((a & bit) != 0) and ((x & bit) != 0)
                bit <<= 1
            w[a, x] = -1.0 if parity else 1.0
    return w


def irrep_block_index() -> tuple[np.ndarray, np.ndarray]:
    """Block assignment over frequency pairs (a, b).

    Returns (block_id, position_in_block):
    - diagonal pairs (a, a): block_id = a, position 0  -> 64 one-dim sectors
    - off-diagonal {a, b}, a < b: block_id = 64 + linear index, position p
      so that (a, b) and (b, a) share a block with positions 0/1.
    Total: 64 + 2016 = 2080 blocks.
    """
    block_id = np.zeros((64, 64), dtype=np.int32)
    position = np.zeros((64, 64), dtype=np.int8)
    for a in range(64):
        block_id[a, a] = a
        position[a, a] = 0
    linear = 0
    for a in range(64):
        for b in range(a + 1, 64):
            bid = 64 + linear
            block_id[a, b] = bid
            position[a, b] = 0
            block_id[b, a] = bid
            position[b, a] = 1
            linear += 1
    assert linear == 2016
    return block_id, position


def translation_signs(a: int, b: int, tau_u: int, tau_v: int) -> int:
    """(-1)^(dot(a, tau_u) xor dot(b, tau_v)) as +1/-1."""
    parity = (
        bin(a & tau_u).count("1") + bin(b & tau_v).count("1")
    ) & 1
    return -1 if parity else 1


class SpectralBlockFilter(nn.Module):
    """Gates Walsh coefficients inside each irrep block.

    A learnable per-block scalar gain; gain == 0 removes the sector, defining
    the lossy bottleneck with exact equivariance preserved (each block is a
    G-subrepresentation). A frozen sector mask hard-zeros a subset of blocks,
    turning the model into a lossy codec.
    """

    def __init__(
        self,
        n_blocks: int = 2080,
        init_gain: float = 1.0,
        sector_mask: np.ndarray | None = None,
        orbit_index: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if orbit_index is not None:
            orbit = np.asarray(orbit_index, dtype=np.int64)
            assert orbit.shape == (n_blocks,)
            self.gain = nn.Parameter(
                torch.full((int(orbit.max()) + 1,), float(init_gain))
            )
            self.register_buffer("orbit_index", torch.as_tensor(orbit))
        else:
            self.orbit_index = None
            self.gain = nn.Parameter(torch.full((n_blocks,), float(init_gain)))
        if sector_mask is None:
            mask = torch.ones(n_blocks, dtype=torch.float32)
        else:
            mask = torch.as_tensor(np.asarray(sector_mask, dtype=np.float32))
            assert mask.shape == (n_blocks,)
        self.register_buffer("sector_mask", mask)

    def block_gains(self) -> torch.Tensor:
        if self.orbit_index is not None:
            idx = self.orbit_index
            out = torch.zeros(
                idx.shape[0], device=self.gain.device, dtype=self.gain.dtype
            )
            valid = idx >= 0
            out[valid] = self.gain[idx[valid]]
            return out
        return self.gain

    def forward(self, coeff: torch.Tensor, block_id: torch.Tensor) -> torch.Tensor:
        """coeff [B, 4096] ordered by (a, b) flat; block_id [4096]."""
        gains = self.block_gains()[block_id] * self.sector_mask[block_id]
        return coeff * gains.unsqueeze(0)

    def active_blocks(self) -> int:
        return int(self.sector_mask.sum().item())

    def rate_penalty(self) -> torch.Tensor:
        """L1 penalty on the free gains; a rate term for learned bottlenecks."""
        return (self.block_gains().abs() * self.sector_mask).sum()


LADDER_ALIASES = {
    "diagonal": "chirality",
    "shell_radial": "diagonal_translation_radial",
    "shell": "w2_invariant",
}


def resolve_ladder(ladder: str) -> str:
    """Map legacy ladder names onto the canonical rung ids."""
    return LADDER_ALIASES.get(ladder, ladder)


def codec_ladder(ladder: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Frozen sector mask and optional gain-orbit partition for a ladder rung.

    Sector rungs (mask, one free gain per kept block):
    - "full": all 2080 blocks (identity codec);
    - "chirality" (alias "diagonal"): the 64 diagonal sectors (a = b);
    - "w2_invariant" (alias "shell"): even-weight diagonal sectors;
    - "trivial": only the constant sector;
    - "offdiagonal": the 2016 two-dimensional sectors only.

    Tied rungs (mask + orbit_index; orbit_index == -1 for masked blocks):
    - "diagonal_translation_radial" (alias "shell_radial"): all blocks, gain
      tied by wt(a ⊕ b);
    - "shell_climate": diagonal only, gains tied by popcount(a);
    - "shell_gauge": gain tied by the unordered shell pair;
    - "chirality_gauge": gain tied by (wt(a), wt(b), parity of wt(a & b)).
    """
    canonical = resolve_ladder(ladder)
    bid, _ = irrep_block_index()
    n_blocks = 64 + 2016
    mask = np.zeros(n_blocks, dtype=np.float32)
    orbit: np.ndarray | None = None
    if canonical == "full":
        mask[:] = 1.0
    elif canonical == "chirality":
        mask[:64] = 1.0
    elif canonical == "w2_invariant":
        for a in range(64):
            if a.bit_count() % 2 == 0:
                mask[a] = 1.0
    elif canonical == "trivial":
        mask[0] = 1.0
    elif canonical == "offdiagonal":
        mask[64:] = 1.0
    elif canonical == "shell_climate":
        orbit = np.full(n_blocks, -1, dtype=np.int64)
        for a in range(64):
            block = int(bid[a, a])
            mask[block] = 1.0
            orbit[block] = a.bit_count()
    elif canonical in (
        "diagonal_translation_radial",
        "shell_gauge",
        "chirality_gauge",
    ):
        mask[:] = 1.0
        orbit = np.full(n_blocks, -1, dtype=np.int64)
        key_to_orbit: dict[tuple, int] = {}
        for a in range(64):
            for b in range(a, 64):
                wa, wb = a.bit_count(), b.bit_count()
                key: tuple
                if canonical == "diagonal_translation_radial":
                    key = ((a ^ b).bit_count(),)
                elif canonical == "chirality_gauge":
                    key = (wa, wb, (a & b).bit_count() % 2)
                else:
                    key = (min(wa, wb), max(wa, wb))
                oid = key_to_orbit.get(key)
                if oid is None:
                    oid = len(key_to_orbit)
                    key_to_orbit[key] = oid
                orbit[int(bid[a, b])] = oid
    else:
        raise ValueError(f"unknown codec ladder rung: {ladder}")
    return mask, orbit


def codec_ladder_mask(ladder: str) -> np.ndarray:
    """The frozen sector mask of a ladder rung."""
    return codec_ladder(ladder)[0]


class AffineSpectralCodec(nn.Module):
    """one-hot -> factored Walsh -> block gains -> inverse Walsh -> softmax.

    Exact full-G equivariance: translations act by per-coefficient signs, the
    swap by per-block 2x2 permutations; both commute with scalar gains.
    """

    def __init__(
        self,
        init_gain: float = 1.0,
        ladder: str | None = None,
        sector_mask: np.ndarray | None = None,
        orbit_index: np.ndarray | None = None,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        w = walsh_matrix_64()
        self.register_buffer("W", torch.as_tensor(w))  # [64, 64]
        bid, pos = irrep_block_index()
        self.register_buffer(
            "block_id", torch.as_tensor(bid.reshape(-1).astype(np.int64))
        )
        self.register_buffer(
            "position", torch.as_tensor(pos.reshape(-1).astype(np.int64))
        )
        sign_table = np.array(
            [-1.0 if bin(i).count("1") & 1 else 1.0 for i in range(64)],
            dtype=np.float32,
        )
        self.register_buffer("translation_sign_table", torch.as_tensor(sign_table))
        if sector_mask is None and ladder is not None:
            sector_mask, orbit_from_ladder = codec_ladder(ladder)
            if orbit_index is None:
                orbit_index = orbit_from_ladder
        self.bottleneck = SpectralBlockFilter(
            2080, init_gain, sector_mask, orbit_index=orbit_index
        )
        self.ladder = ladder
        if ladder is not None:
            self.ladder = resolve_ladder(ladder)
        self._init_gain = init_gain
        self.frozen = bool(frozen)
        if self.frozen:
            for p in self.parameters():
                p.requires_grad_(False)

    def get_config(self) -> dict:
        return {
            "init_gain": self._init_gain,
            "ladder": self.ladder,
            "frozen": self.frozen,
            "sector_mask": np.asarray(self.bottleneck.sector_mask.cpu()),
            "orbit_index": (
                None
                if self.bottleneck.orbit_index is None
                else np.asarray(self.bottleneck.orbit_index.cpu())
            ),
        }

    def walsh_coefficients(self, onehot: torch.Tensor) -> torch.Tensor:
        """[B, 4096] one-hot -> [B, 4096] Walsh coeffs (flat (a,b) order)."""
        f = onehot.reshape(-1, 64, 64)
        ca = torch.einsum("au,buv->bav", self.W, f)
        coeff = torch.einsum("bav,cv->bac", ca, self.W)
        return coeff.reshape(-1, 4096)

    def inverse_walsh(self, coeff: torch.Tensor) -> torch.Tensor:
        """[B, 4096] coeffs -> [B, 4096] function values (self-inverse / 4096)."""
        c = coeff.reshape(-1, 64, 64)
        f1 = torch.einsum("au,bac->buc", self.W, c)
        f = torch.einsum("cv,buc->buv", self.W, f1)
        return (f / 4096.0).reshape(-1, 4096)

    def apply_pq_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply P_Q: x -> inverse_walsh(bottleneck(WHT(x))).

        Input  : [B, 4096] activations (continuous, not one-hot).
        Output : [B, 4096] = P_Q(gains) . x in the original basis.
        """
        coeff = self.walsh_coefficients(x)
        gated = self.bottleneck(coeff, self.block_id)
        return self.inverse_walsh(gated)

    def encode(self, state_index: torch.Tensor) -> torch.Tensor:
        onehot = torch.zeros(
            (state_index.shape[0], N_STATES),
            device=state_index.device,
            dtype=self.W.dtype,
        )
        rows = torch.arange(state_index.shape[0], device=state_index.device)
        onehot[rows, state_index] = 1.0
        return self.walsh_coefficients(onehot)

    def forward(self, state_index: torch.Tensor) -> torch.Tensor:
        onehot = torch.zeros(
            (state_index.shape[0], N_STATES),
            device=state_index.device,
            dtype=self.W.dtype,
        )
        rows = torch.arange(state_index.shape[0], device=state_index.device)
        onehot[rows, state_index] = 1.0
        coeff = self.walsh_coefficients(onehot)
        gated = self.bottleneck(coeff, self.block_id)
        recon = self.inverse_walsh(gated)
        return recon

    def spectral_action(
        self, sig_id: int, coeff: torch.Tensor
    ) -> torch.Tensor:
        """Exact rho(g) on Walsh coefficients for g = (parity, tau_u, tau_v)."""
        parity, tau_u, tau_v = (sig_id >> 12) & 1, (sig_id >> 6) & 63, sig_id & 63
        device = coeff.device
        a = torch.arange(64, device=device)
        su = self.translation_sign_table.to(device, dtype=coeff.dtype)[a & tau_u]
        sv = self.translation_sign_table.to(device, dtype=coeff.dtype)[a & tau_v]
        sign = su[None, :, None] * sv[None, None, :]
        c = coeff.reshape(-1, 64, 64)
        if parity == 0:
            out = c * sign
        else:
            out = c.transpose(1, 2) * sign
        return out.reshape(-1, 4096)


def full_g_equivariance_error(
    model: nn.Module,
    state_indices: torch.Tensor,
    sig_ids: torch.Tensor,
) -> dict[str, float]:
    """Equivariance defect of the spectral map over sampled states/signatures.

    Reports coefficient-level (max/mean) and end-to-end forward (forward_max/
    forward_mean) identities."""
    from src.tools.autoencoder.kernel import apply_signature_index

    with torch.no_grad():
        max_err = 0.0
        mean_errs = []
        fwd_max = 0.0
        fwd_errs = []
        for sig_id in sig_ids.tolist():
            sig = int(sig_id)
            x = state_indices
            transformed = torch.tensor(
                [apply_signature_index(int(i), sig) for i in x.tolist()],
                dtype=torch.long,
            )
            onehot_x = torch.zeros((len(x), N_STATES), device=x.device)
            onehot_x[torch.arange(len(x), device=x.device), x] = 1.0
            onehot_g = torch.zeros((len(x), N_STATES), device=x.device)
            onehot_g[torch.arange(len(x), device=x.device), transformed] = 1.0
            c_x = model.walsh_coefficients(onehot_x)
            c_g = model.walsh_coefficients(onehot_g)
            rho_c_x = model.spectral_action(sig, c_x)
            err = (c_g - rho_c_x).abs().max().item()
            max_err = max(max_err, err)
            mean_errs.append(err)

            fwd_x = model(x)
            fwd_g = model(transformed)
            perm = torch.tensor(
                [apply_signature_index(int(i), sig) for i in range(N_STATES)],
                dtype=torch.long,
                device=fwd_x.device,
            )
            permuted = torch.zeros_like(fwd_x)
            permuted.index_add_(1, perm, fwd_x)
            ferr = (fwd_g - permuted).abs().max().item()
            fwd_max = max(fwd_max, ferr)
            fwd_errs.append(ferr)
        return {
            "max": max_err,
            "mean": float(np.mean(mean_errs)),
            "forward_max": max(fwd_max, 0.0),
            "forward_mean": float(np.mean(fwd_errs)),
        }


