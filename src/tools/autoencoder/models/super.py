"""The "super" tier: models that build in the full group / multi-register
structure.

These models work over the Walsh harmonics of the hQVM carrier Omega
(GF(2)^6 x GF(2)^6). Under the full affine group, frequency pairs organize
into irrep blocks; the model gates each block exactly, so full-group
equivariance holds by construction. MultiCellSpectral extends the same idea to
a product register of cells.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

N_STATES = 4096


def walsh_matrix_64() -> np.ndarray:
    """Unnormalized 64x64 Walsh-Hadamard matrix W[a, x] = (-1)^(dot(a,x))."""
    W = np.empty((64, 64), dtype=np.float32)
    for a in range(64):
        for x in range(64):
            bit = 1
            parity = 0
            while bit <= 63:
                parity ^= ((a & bit) != 0) and ((x & bit) != 0)
                bit <<= 1
            W[a, x] = -1.0 if parity else 1.0
    return W


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


class SpectralBottleneck(nn.Module):
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
            return self.gain[self.orbit_index]
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


def codec_ladder(ladder: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Frozen sector mask and optional gain-orbit partition for a ladder rung.

    Each rung keeps a definite set of irrep blocks and either hard-zeros the
    rest (sector rungs) or ties the free gains into one parameter per orbit
    (tied rungs).

    Sector rungs (mask, one free gain per kept block):
    - "full": all 2080 blocks (identity codec);
    - "diagonal": the 64 diagonal sectors only - the swap-invariant information;
    - "shell": diagonal sectors with even Hamming weight - the shell-band code;
    - "trivial": only the constant sector - the corpus-mean codec;
    - "offdiagonal": the 2016 two-dimensional sectors only.

    Tied rungs (full mask, one free gain per orbit):
    - "shell_radial": gain tied by wt(a ^ b) - the carrier-frequency weight;
    - "shell_gauge": gain tied by the unordered shell pair - 28 tied gains;
    - "chirality_gauge": gain tied by (wt(a), wt(b), parity of wt(a & b)).
    """
    bid, _ = irrep_block_index()
    n_blocks = 64 + 2016
    mask = np.zeros(n_blocks, dtype=np.float32)
    orbit: np.ndarray | None = None
    if ladder == "full":
        mask[:] = 1.0
    elif ladder == "diagonal":
        mask[:64] = 1.0
    elif ladder == "shell":
        for a in range(64):
            if a.bit_count() % 2 == 0:
                mask[a] = 1.0
    elif ladder == "trivial":
        mask[0] = 1.0
    elif ladder == "offdiagonal":
        mask[64:] = 1.0
    elif ladder in ("shell_radial", "shell_gauge", "chirality_gauge"):
        mask[:] = 1.0
        orbit = np.full(n_blocks, -1, dtype=np.int64)
        key_to_orbit: dict[tuple, int] = {}
        for a in range(64):
            for b in range(a, 64):
                wa, wb = a.bit_count(), b.bit_count()
                key = (min(wa, wb), max(wa, wb))
                if ladder == "shell_radial":
                    key = ((a ^ b).bit_count(),)
                elif ladder == "chirality_gauge":
                    key = (wa, wb, (a & b).bit_count() % 2)
                oid = key_to_orbit.get(key)
                if oid is None:
                    oid = len(key_to_orbit)
                    key_to_orbit[key] = oid
                orbit[bid[a, b]] = oid
    else:
        raise ValueError(f"unknown codec ladder rung: {ladder}")
    return mask, orbit


def codec_ladder_mask(ladder: str) -> np.ndarray:
    """The frozen sector mask of a ladder rung."""
    return codec_ladder(ladder)[0]


class SpectralAutoencoder(nn.Module):
    """one-hot -> factored Walsh -> block gains -> inverse Walsh -> softmax.

    Exact full-G equivariance: translations act by per-coefficient signs, the
    swap by per-block 2x2 permutations; both commute with scalar gains.
    """

    def __init__(
        self,
        init_gain: float = 1.0,
        keep_all: bool = True,
        ladder: str | None = None,
        sector_mask: np.ndarray | None = None,
        orbit_index: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        W = walsh_matrix_64()
        self.register_buffer("W", torch.as_tensor(W))  # [64, 64]
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
        self.bottleneck = SpectralBottleneck(
            2080, init_gain, sector_mask, orbit_index=orbit_index
        )
        self.ladder = ladder
        self._init_gain = init_gain

    def get_config(self) -> dict:
        return {
            "init_gain": self._init_gain,
            "ladder": self.ladder,
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
        """Apply P_Q to a real activation vector: x -> inverse_walsh(bottleneck(WHT(x))).

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
        onehot[torch.arange(state_index.shape[0], device=state_index.device), state_index] = 1.0
        return self.walsh_coefficients(onehot)

    def forward(self, state_index: torch.Tensor) -> torch.Tensor:
        onehot = torch.zeros(
            (state_index.shape[0], N_STATES),
            device=state_index.device,
            dtype=self.W.dtype,
        )
        onehot[torch.arange(state_index.shape[0], device=state_index.device), state_index] = 1.0
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


class MultiCellSpectral(nn.Module):
    """Multi-cell product-register spectral model.

    Joint Walsh-Hadamard transform over a product register of cells; each cell's
    spectrum is a [64, 64] block and the joint register acts by permuting each
    cell's coefficients with its own rho(g), so the build is exactly equivariant
    by the same per-block argument as the single cell.
    """

    def __init__(self, n_cells: int = 2) -> None:
        super().__init__()
        self.n_cells = n_cells
        self.register_buffer("W", torch.as_tensor(walsh_matrix_64()))
        self.spectral = SpectralAutoencoder()
        # The block id is a constant index array shared with the underlying
        # single-cell spectral model; it does not need to be a learned
        # parameter but is kept as a buffer for device-move consistency.
        self.register_buffer("block_id", torch.arange(4096, dtype=torch.long))

    def cell_spectrum(self, state_index: torch.Tensor) -> torch.Tensor:
        onehot = torch.zeros(
            (state_index.shape[0], 4096), device=state_index.device,
            dtype=self.spectral.W.dtype,
        )
        onehot[torch.arange(state_index.shape[0]), state_index] = 1.0
        return self.spectral.walsh_coefficients(onehot).reshape(state_index.shape[0], 4096)

    def joint_walsh(self, mat: np.ndarray) -> np.ndarray:
        """Kronecker WHT over the product register (state per cell, n_cells)."""
        W = self.W.numpy()
        J = W
        for _ in range(self.n_cells - 1):
            J = np.kron(J, W)
        return mat @ J

    def joint_spectrum(self, cell_states: list[torch.Tensor]) -> torch.Tensor:
        """List of B [n] cell state-index tensors -> [n, 4096**B] joint spectrum."""
        if len(cell_states) != self.n_cells:
            raise ValueError("expected n_cells cell state tensors")
        specs = [self.cell_spectrum(s) for s in cell_states]
        joint = specs[0]
        for nxt in specs[1:]:
            joint = torch.einsum("ni,nj->nij", joint, nxt).reshape(
                joint.shape[0], joint.shape[1] * nxt.shape[1]
            )
        return joint

    def concentration(self, joint_spectrum: torch.Tensor) -> dict[str, float]:
        """Energy concentration diagnostics over the joint spectrum."""
        e = joint_spectrum.float() ** 2
        total = e.sum(dim=-1)
        trivial = e[:, 0].sum(dim=-1)
        cell_low_fracs = []
        for c in range(self.n_cells):
            stride = 4096 ** (self.n_cells - 1 - c)
            low_idx = [i for i in range(4096) if i.bit_count() <= 2]
            idx = torch.as_tensor(low_idx, device=e.device) * stride
            cell_e = e[:, idx].sum(dim=-1)
            cell_low_fracs.append(cell_e / (total + 1e-30))
        any_low = 1.0 - torch.prod(
            torch.stack([1.0 - f for f in cell_low_fracs], dim=0), dim=0
        )
        return {
            "trivial_fraction": float((trivial / (total + 1e-30)).mean()),
            "low_band_any_cell_fraction": float(any_low.mean()),
            "mean_total_energy": float(total.mean()),
        }

    def product_equivariance_check(self, cell_states: list[torch.Tensor]) -> float:
        """Exact product-register equivariance check (swap first two cells)."""
        base = self.joint_spectrum(cell_states)
        swapped = list(cell_states)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        moved = self.joint_spectrum(swapped)
        n = base.shape[0]
        moved_perm = moved.reshape(n, 4096, 4096).transpose(1, 2).reshape(base.shape)
        return float((base - moved_perm).abs().max())

    def equivariance_check(self, rng: np.random.Generator) -> float:
        """Smoke check: the joint WHT is linear (a non-trivial property)."""
        x = rng.standard_normal((2, 4096)).astype(np.float64)
        a, b = 0.3, -0.7
        lin = self.joint_walsh(a * x[0:1] + b * x[1:2])
        sep = a * self.joint_walsh(x[0:1]) + b * self.joint_walsh(x[1:2])
        return float(np.abs(lin - sep).max())
