# === secret_lab_ignore/agent/runtime.py ===
# COMPLETE REPLACEMENT — implements the full Gyroscopic ASI spec
# including token binding and embedding lookup

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.router.kernel import RouterKernel

from .adaptor import GyroAdaptor
from .inference_core import InferenceEgress, InferenceRoles, InferenceState


@dataclass
class GyroASIConfig:
    atlas_dir: str
    adaptor_path: str
    embedding_path: str              # path to saved embedding table
    K: int
    eta: float
    vocab_size: int = 100278
    bytes_per_token: int = 4         # L=4 for OLMo
    temperature: float = 0.7
    weight_penalty: float = 0.02
    seed: int = 0


def allowed_max_next_byte(prefix: list[int], vocab_size: int, L: int) -> int:
    """
    Given a partially-built token prefix, what is the maximum allowed
    value for the next byte such that the final token_id < vocab_size?

    For big-endian encoding: token_id = prefix[0]<<(8*(L-1)) + ... + prefix[-1]

    Returns 255 if unconstrained, or a smaller value if the prefix
    constrains the next byte.
    """
    pos = len(prefix)  # 0-indexed position we're filling next
    remaining = L - pos - 1  # bytes still to come after this one

    if pos == 0 and L == 4:
        # First byte: token_id < vocab_size
        # max first byte = vocab_size >> (8 * 3)
        max_first = (vocab_size - 1) >> (8 * remaining)
        return min(255, max_first)

    # Compute the value so far from prefix
    val_so_far = 0
    for b in prefix:
        val_so_far = (val_so_far << 8) | b

    # If we set next byte to X and all remaining to 255:
    # max_token = (val_so_far << 8*(remaining+1)) | (X << 8*remaining) | (2^(8*remaining) - 1)
    # We need max_token < vocab_size

    # Simpler: the prefix already determines the high bits.
    # If val_so_far already exceeds what's possible, constrain to 0
    high_shift = 8 * (remaining + 1)
    base = val_so_far << high_shift

    if base >= vocab_size:
        return 0  # this prefix is already too large

    # How much room is left?
    room = vocab_size - 1 - base
    max_next = room >> (8 * remaining)
    return min(255, max_next)


@dataclass
class GyroASI:
    cfg: GyroASIConfig

    kernel: RouterKernel = field(init=False)
    adaptor: GyroAdaptor = field(init=False)
    embedding_table: NDArray[np.float32] = field(init=False)  # [vocab, D]
    sigma: InferenceState = field(init=False)
    roles: InferenceRoles = field(init=False)
    egress: InferenceEgress = field(init=False)

    genealogy: list[int] = field(default_factory=list)
    token_history: list[int] = field(default_factory=list)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.kernel = RouterKernel(atlas_dir=Path(self.cfg.atlas_dir))
        self.adaptor = GyroAdaptor.load(self.cfg.adaptor_path)

        # Load embedding table (static, no forward pass)
        self.embedding_table = np.load(self.cfg.embedding_path).astype(np.float32)
        assert self.embedding_table.shape[0] >= self.cfg.vocab_size
        assert self.embedding_table.shape[1] == 256 * self.cfg.K

        self.sigma = InferenceState.create(K=self.cfg.K)

        atlas_path = Path(self.cfg.atlas_dir) / "phenomenology.npz"
        with np.load(str(atlas_path), allow_pickle=False) as phen:
            features = phen[f"features_K{self.cfg.K}"].astype(np.float32)
            byte_weight = phen["byte_weight"].astype(np.uint8)

        self.roles = InferenceRoles(K=self.cfg.K, eta=self.cfg.eta, mode="td")

        self.egress = InferenceEgress(
            K=self.cfg.K,
            temperature=self.cfg.temperature,
            weight_penalty=self.cfg.weight_penalty,
            features=features,
            byte_weight=byte_weight,
        )

        self.rng = np.random.default_rng(self.cfg.seed)

    def reset(self) -> None:
        self.kernel.reset()
        self.genealogy.clear()
        self.token_history.clear()
        self.sigma = InferenceState.create(K=self.cfg.K)

    def get_context_embedding(self) -> NDArray[np.float32]:
        """
        Build context embedding from token history using static
        embedding table. Exponential moving average over recent tokens.

        Returns: [256 * K] vector
        """
        D = 256 * self.cfg.K
        if not self.token_history:
            return np.zeros(D, dtype=np.float32)

        alpha = 0.9
        ctx = np.zeros(D, dtype=np.float32)
        for tid in self.token_history:
            if tid < self.embedding_table.shape[0]:
                e = self.embedding_table[tid]
                ctx = alpha * ctx + (1.0 - alpha) * e

        # Normalize
        norm = np.linalg.norm(ctx)
        if norm > 1e-8:
            ctx = (ctx / norm).astype(np.float32)
        return ctx

    def step_input(self, token_id: int) -> None:
        """Process an input token (Egress in the spec)."""
        L = self.cfg.bytes_per_token
        bs = list(token_id.to_bytes(L, "big"))
        for b in bs:
            i = int(self.kernel.state_index[0])
            last_b = int(self.kernel.last_byte[0]) & 0xFF
            h = int(self.kernel.state_horizon[i])
            chi = int(self.kernel.state_vertex[i])
            p = int(self.kernel.phase[i, last_b])

            O_field = self._build_combined_field()
            self.roles.L_inf(self.sigma, h=h, chi=chi, p=p,
                             o_local=O_field[h, :])

            self.kernel.step_byte(b)
            self.genealogy.append(b)

            i1 = int(self.kernel.state_index[0])
            h1 = int(self.kernel.state_horizon[i1])
            chi1 = int(self.kernel.state_vertex[i1])
            p1 = int(self.kernel.phase[i1, b])
            w1 = int(self.kernel.byte_weight[b])

            self.roles.R_inf(
                self.sigma, b=b, h_next=h1, chi_next=chi1,
                p_next=p1, byte_weight=w1, o_next=O_field[h1, :],
            )

        self.token_history.append(token_id)

    def step_output(self) -> int:
        """
        Generate one token (Ingress in the spec).

        Sequential prefix-peek: select L bytes one at a time,
        constrained by vocab_size, then decode to token_id.
        """
        L = self.cfg.bytes_per_token
        planned: list[int] = []

        # Save kernel state for peek
        saved_state = int(self.kernel.state_index[0])
        saved_byte = int(self.kernel.last_byte[0])
        saved_step = self.kernel.step

        # Build combined field from adaptor + context embedding
        O_field = self._build_combined_field()

        for pos in range(L):
            i = int(self.kernel.state_index[0])
            last_b = int(self.kernel.last_byte[0]) & 0xFF
            h = int(self.kernel.state_horizon[i])
            chi = int(self.kernel.state_vertex[i])
            p = int(self.kernel.phase[i, last_b])

            # L_inf at current position
            self.roles.L_inf(self.sigma, h=h, chi=chi, p=p,
                             o_local=O_field[h, :])

            # Vocab constraint for this byte position
            max_byte = allowed_max_next_byte(planned, self.cfg.vocab_size, L)

            # Get policy distribution
            cf_chi = self.kernel.next_vertex[i, :].astype(np.uint8, copy=False)
            pi = self.egress.policy(
                h=h, chi=chi, p=p,
                sigma=self.sigma, O_field=O_field, cf_chi=cf_chi,
            )

            # Zero out bytes above vocab constraint
            if max_byte < 255:
                pi[max_byte + 1:] = 0.0
                total = pi.sum()
                if total > 1e-10:
                    pi = pi / total
                else:
                    # All valid bytes got zero prob — fall back to uniform
                    pi[:max_byte + 1] = 1.0 / (max_byte + 1)
                    pi[max_byte + 1:] = 0.0

            b = self.egress.sample(pi, self.rng)
            planned.append(b)

            # Advance kernel (ephemeral during planning)
            self.kernel.step_byte(b)
            self.genealogy.append(b)

            # R_inf at new position
            i1 = int(self.kernel.state_index[0])
            h1 = int(self.kernel.state_horizon[i1])
            chi1 = int(self.kernel.state_vertex[i1])
            p1 = int(self.kernel.phase[i1, b])
            w1 = int(self.kernel.byte_weight[b])

            self.roles.R_inf(
                self.sigma, b=b, h_next=h1, chi_next=chi1,
                p_next=p1, byte_weight=w1, o_next=O_field[h1, :],
            )

        # Decode planned bytes to token_id
        token_id = int.from_bytes(bytes(planned), "big")

        # Clamp to vocab (safety)
        if token_id >= self.cfg.vocab_size:
            token_id = 0  # fallback to padding token

        self.token_history.append(token_id)
        return token_id

    def _build_combined_field(self) -> NDArray[np.float32]:
        """
        Combine adaptor navigation with context embedding.

        Returns O_field [256, K].
        """
        K = self.cfg.K
        i = int(self.kernel.state_index[0])
        last_b = int(self.kernel.last_byte[0]) & 0xFF
        h = int(self.kernel.state_horizon[i])
        chi = int(self.kernel.state_vertex[i])
        p = int(self.kernel.phase[i, last_b])
        # Navigation component from adaptor
        sim = self.adaptor.build_similarity(h)                     # [256]
        nav_field = (sim.reshape(256, 1) * self.egress.features)   # [256,K]
        nav_field = nav_field.astype(np.float32, copy=False)

        # Semantic component from context embedding
        ctx = self.get_context_embedding()  # [256*K]
        sem_field = ctx.reshape(256, K)

        # Combine: navigation provides structure, semantics provides content
        # Weight navigation by CGM aperture scale (~0.02) relative to semantics
        nav_weight = np.float32(0.02)
        sem_weight = np.float32(1.0) - nav_weight

        O_field = nav_weight * nav_field + sem_weight * sem_field

        return O_field.astype(np.float32)
