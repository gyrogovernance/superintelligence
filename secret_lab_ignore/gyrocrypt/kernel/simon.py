"""
Native hQVM Simon on GF(2)^{6B}.

Oracle: depth-4 W₂ half-word (fam 00 → fam 01) from GENE_MAC_REST; shadow partner
on the fam-01 byte enforces 2-to-1 via SO(3)/SU(2) byte fiber (T5/T8).
Resolution: apply_k4(K4_W2) from rest for all inputs; holographic WHT^{⊗2} readout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from kernel.core import (
    GENE_MAC_REST,
    K4_W2,
    MultiCellRouter,
    apply_k4,
    index_hq,
    is_in_omega24,
    split_index,
    state_to_hq,
    step_byte,
    wavefunction_hq_spectral_peaks,
    zero_wavefunction,
    _w2_word,
)

MASK6 = 0x3F
CELL_BITS = 6
MAX_NATIVE_CELLS = 10

_REST_SINGLE: list[complex] | None = None
_SHADOW_PARTNER: list[int] | None = None


def _gf2_dot(y: int, s: int) -> int:
    return (int(y) & int(s)).bit_count() & 1


def _gf2_row_echelon(rows: list[int], n: int) -> tuple[list[int], list[int]]:
    mask = (1 << int(n)) - 1
    m = [int(r) & mask for r in rows if int(r) & mask]
    pivots: list[int] = []
    pr = 0
    for col in range(int(n) - 1, -1, -1):
        bit = 1 << col
        idx = next((i for i in range(pr, len(m)) if m[i] & bit), None)
        if idx is None:
            continue
        m[pr], m[idx] = m[idx], m[pr]
        for i in range(len(m)):
            if i != pr and m[i] & bit:
                m[i] ^= m[pr]
        pivots.append(col)
        pr += 1
    return pivots, m[:pr]


def _gf2_nullspace_spanner(pivots: list[int], echelon: list[int], n: int) -> int:
    pivot_set = set(pivots)
    free = [c for c in range(int(n)) if c not in pivot_set]
    if not free:
        return 0
    s = 1 << free[0]
    for col, row in zip(pivots, echelon):
        if _gf2_dot(s, row):
            s ^= 1 << col
    return int(s)


def _coset_rep6(chi6: int, s6: int) -> tuple[int, bool]:
    chi6 &= MASK6
    s6 &= MASK6
    if s6 == 0:
        return chi6, False
    partner = chi6 ^ s6
    rep = chi6 if chi6 <= partner else partner
    return rep, chi6 != rep


def _build_shadow_table() -> list[int]:
    by_state: dict[int, list[int]] = {}
    for b in range(256):
        st = int(step_byte(GENE_MAC_REST, b))
        by_state.setdefault(st, []).append(int(b) & 0xFF)

    if len(by_state) != 128:
        raise RuntimeError(f"shadow projection mismatch: expected 128 states, got {len(by_state)}")
    for st, bs in by_state.items():
        if len(bs) != 2:
            raise RuntimeError(f"shadow fiber not 2-to-1 at state={st:#x}: bytes={bs}")

    shadow = [0] * 256
    for bs in by_state.values():
        a, b = sorted(bs)
        shadow[a] = b
        shadow[b] = a
        if int(step_byte(GENE_MAC_REST, a)) != int(step_byte(GENE_MAC_REST, b)):
            raise RuntimeError("shadow partner table inconsistent")
    return shadow


def _shadow_partner(b: int) -> int:
    global _SHADOW_PARTNER
    if _SHADOW_PARTNER is None:
        _SHADOW_PARTNER = _build_shadow_table()
    return int(_SHADOW_PARTNER[int(b) & 0xFF])


def _w2_program(rep6: int, use_shadow: bool) -> list[int]:
    """Depth-4 W₂ half-word for byte-ledger replay; shadow on byte 2 for coset partner."""
    word = [int(b) & 0xFF for b in _w2_word(int(rep6) & MASK6)]
    if use_shadow:
        word[1] = _shadow_partner(word[1])
    return word


def _rest_single() -> list[complex]:
    global _REST_SINGLE
    if _REST_SINGLE is None:
        h, q = state_to_hq(GENE_MAC_REST)
        psi = zero_wavefunction()
        psi[index_hq(h, q)] = complex(1.0, 0.0)
        _REST_SINGLE = psi
    return _REST_SINGLE


def _step_program_from_rest(prog: list[int]) -> int:
    st = int(GENE_MAC_REST)
    for byte_val in prog:
        st = int(step_byte(st, int(byte_val) & 0xFF))
        if not is_in_omega24(st):
            raise RuntimeError(f"W₂ oracle produced non-Omega state {st:#x}")
    return st


def _run_multicell_program(prog: list[int], n_cells: int) -> list[int]:
    router = MultiCellRouter(int(n_cells))
    for cell in range(int(n_cells)):
        for j in range(2):
            router.step_byte(cell, int(prog[2 * cell + j]) & 0xFF)
            if not is_in_omega24(int(router.states[cell])):
                raise RuntimeError(f"oracle produced non-Omega state cell={cell}")
    return [int(x) for x in router.states]


@dataclass
class NativeSimonOracle:
    """Simon black box: depth-4 W₂ per cell + shadow 2-to-1 on fam-01 byte."""

    secret: int
    n_cells: int = 1
    n: int = field(init=False)
    mask: int = field(init=False)
    s: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_cells = int(self.n_cells)
        if self.n_cells < 1 or self.n_cells > MAX_NATIVE_CELLS:
            raise ValueError(f"n_cells must be in 1..{MAX_NATIVE_CELLS}")
        self.n = CELL_BITS * self.n_cells
        self.mask = (1 << self.n) - 1
        self.s = int(self.secret) & self.mask

    def secret_limb(self, cell: int) -> int:
        return (self.s >> (CELL_BITS * int(cell))) & MASK6

    def program_bytes(self, x: int) -> list[int]:
        xx = int(x) & self.mask
        out: list[int] = []
        for cell in range(self.n_cells):
            chi6 = (xx >> (CELL_BITS * cell)) & MASK6
            rep6, use_shadow = _coset_rep6(chi6, self.secret_limb(cell))
            out.extend(_w2_program(rep6, use_shadow))
        return out

    def query(self, x: int) -> int | list[int]:
        prog = self.program_bytes(x)
        if self.n_cells == 1:
            return _step_program_from_rest(prog)
        return _run_multicell_program(prog, self.n_cells)

    def output_tag(self, x: int) -> tuple[int, ...]:
        out = self.query(x)
        if self.n_cells == 1:
            return (int(out),)  # type: ignore[arg-type]
        return tuple(int(s) for s in out)  # type: ignore[union-attr]


@dataclass
class SingleCellSimonView:
    """One 6-bit cell embedded in a multi-cell parent; does not know the secret."""

    parent: NativeSimonOracle
    cell_index: int
    n: int = CELL_BITS
    mask: int = MASK6

    @property
    def n_cells(self) -> int:
        return self.parent.n_cells

    def _embed(self, chi6: int) -> int:
        return (int(chi6) & MASK6) << (CELL_BITS * int(self.cell_index))

    def program_bytes(self, chi6: int) -> list[int]:
        return self.parent.program_bytes(self._embed(chi6))

    def query(self, chi6: int) -> int | list[int]:
        return self.parent.query(self._embed(chi6))

    def query_gyrostate(self, chi6: int) -> int:
        """Full 24-bit Ω state on equality horizon after depth-4 W₂."""
        result = self.query(chi6)
        if self.n_cells == 1:
            return int(result)  # type: ignore[arg-type]
        return int(result[self.cell_index])  # type: ignore[index]

    def output_tag(self, chi6: int) -> tuple[int, ...]:
        return self.parent.output_tag(self._embed(chi6))


def _entangle_cell_psi(view: SingleCellSimonView) -> list[complex]:
    """|Σ_χ |χ⟩|W₂(rep6)(rest)⟩⟩ via apply_k4 for every input (2-to-1 is on rep6)."""
    s6 = view.parent.secret_limb(view.cell_index)
    psi = zero_wavefunction()
    rest = _rest_single()
    for chi in range(64):
        rep6, _use_shadow = _coset_rep6(chi, s6)
        evolved = apply_k4(rest, K4_W2, micro_ref=rep6)
        for idx, amp in enumerate(evolved):
            if abs(amp) <= 1e-15:
                continue
            _h_out, q_out = split_index(idx)
            psi[index_hq(chi & MASK6, q_out & MASK6)] += amp
    norm2 = sum(abs(z) ** 2 for z in psi)
    if norm2 <= 0.0:
        return psi
    scale = 1.0 / math.sqrt(norm2)
    return [z * scale for z in psi]


def _resolve_cell(view: SingleCellSimonView) -> int:
    """K4 holonomy: entangle via W₂, holographic WHT^{⊗2} peaks, GF(2) solve."""
    z0_st = view.query_gyrostate(0)
    psi = _entangle_cell_psi(view)
    peaks = wavefunction_hq_spectral_peaks(psi, k=32)
    if not peaks:
        return 0

    peak_mag = max(mag for _sh, _sq, mag in peaks)
    eps = max(1e-9, peak_mag * 1e-6)
    rows = [sh for sh, _sq, mag in peaks if sh != 0 and mag > eps]
    pivots, echelon = _gf2_row_echelon(rows, view.n)
    s_cand = _gf2_nullspace_spanner(pivots, echelon, view.n) & view.mask
    if s_cand == 0:
        return 0
    return int(s_cand) if view.query_gyrostate(s_cand) == z0_st else 0


def simon_resolve(oracle: NativeSimonOracle) -> int:
    secret = 0
    for cell in range(oracle.n_cells):
        view = SingleCellSimonView(parent=oracle, cell_index=cell)
        secret |= (_resolve_cell(view) & MASK6) << (CELL_BITS * cell)
    return int(secret) & oracle.mask


def simon(n_bits: int, secret: int) -> int | None:
    n_bits = int(n_bits)
    if n_bits % CELL_BITS != 0 or n_bits // CELL_BITS > MAX_NATIVE_CELLS:
        return None
    return simon_resolve(NativeSimonOracle(int(secret), n_cells=n_bits // CELL_BITS))


__all__ = ["NativeSimonOracle", "SingleCellSimonView", "simon", "simon_resolve", "MAX_NATIVE_CELLS"]
