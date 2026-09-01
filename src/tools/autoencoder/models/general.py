"""The "general" tier: models that build in the machine's K4 gate symmetry.

K4 = the Klein four-group {id, S, C, F}. These models are exactly
equivariant under that group by construction (Reynolds symmetrization over its
four one-dimensional characters), with no loss constraint.
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
    z = [z_inv (n_trivial), z_S (n_sign), z_C (n_sign), z_F (n_sign)].
    rho(g) acts as: z_inv -> +z_inv; each sign block scaled by its character.
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
        """Named latent components.

        - z_inv: trivial-character block, invariant under K4;
        - z_chi: chirality-sensitive block (inverts under S and F);
        - z_shell: shell-sensitive block (inverts under C);
        - z_irrep: the fourth sign block completing the K4 irrep content.
        """
        s = self.n_sign
        t0 = 0
        t1 = self.n_trivial
        return {
            "z_inv": slice(t0, t1),
            "z_chi": slice(t1, t1 + s),
            "z_shell": slice(t1 + s, t1 + 2 * s),
            "z_irrep": slice(t1 + 2 * s, t1 + 3 * s),
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
