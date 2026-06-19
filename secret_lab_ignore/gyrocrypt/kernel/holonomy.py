"""
Gyroscopic Holonomy Relation Finding — OPEN research path.

Compile (N,a) into byte operators on GENE_Mac; detect period via K4/horizon/Z₂
spectral closure on Ω. Production period finding uses kernel.shor (native spectral).

See docs/notes/Intelligence/45/3.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .core import (
    GENE_MAC_REST,
    K4_F,
    OMEGA_SIZE,
    MultiCellRouter,
    apply_k4,
    apply_word,
    byte_from_family_and_payload6,
    chirality_word,
    index_hq,
    q_word6,
    state_to_hq,
    step_byte,
    top_nontrivial_peaks,
    wht64_dc_free,
    zero_wavefunction,
)
from .bindings import exp_mod_ladder, mul_mod_ladder

from src.api import (
    OmegaSignature12,
    chirality_word6,
    compose_omega_signatures,
    omega_word_signature,
    shell_index_from_chirality6,
    state24_to_omega12,
)
from src.constants import (
    GENE_MIC_S,
    EPSILON_6,
    apply_gate_S,
    intron_family,
    intron_micro_ref,
    is_on_equality_horizon,
    is_on_horizon,
)

IDENTITY_SIG = OmegaSignature12(0, 0, 0)
MASK6 = 0x3F


# ── Horizon / affine algebra (compile-time analysis) ─────────────────────────


@dataclass(frozen=True)
class HorizonAlgebraReport:
    """Orders achievable on chirality vs full Ω affine signatures."""

    distinct_q6: int
    max_chirality_order: int
    max_depth1_omega_order: int
    max_depth4_omega_order: int
    note: str


def analyze_horizon_algebra(*, sample_depth4: int = 4096) -> HorizonAlgebraReport:
    """
    Chirality transport is XOR on GF(2)^6 (every nonzero q has order 2).
    Full Ω words carry OmegaSignature12 (parity swap + τ translations); orders can exceed 2.
    """
    from itertools import product

    from .core import q_word6

    q_vals = {q_word6(b) for b in range(256)}
    max_chi = 2 if any(q != 0 for q in q_vals) else 1

    def _omega_order(sig: OmegaSignature12, cap: int = 128) -> int:
        cur = sig
        for d in range(1, cap + 1):
            if cur.parity == 0 and cur.tau_u6 == 0 and cur.tau_v6 == 0:
                return d
            cur = compose_omega_signatures(cur, sig)
        return cap

    d1_max = 1
    for b in range(256):
        d1_max = max(d1_max, _omega_order(omega_word_signature([b])))

    d4_max = 1
    count = 0
    for payloads in product(range(64), repeat=4):
        word = [byte_from_family_and_payload6(f, p) for f, p in enumerate(payloads)]
        d4_max = max(d4_max, _omega_order(omega_word_signature(word)))
        count += 1
        if count >= sample_depth4:
            break

    return HorizonAlgebraReport(
        distinct_q6=len(q_vals),
        max_chirality_order=max_chi,
        max_depth1_omega_order=d1_max,
        max_depth4_omega_order=d4_max,
        note=(
            "Chirality-only holonomy cannot carry odd orders >2; period readout uses "
            "full Ω affine holonomy + wavefunction trajectory."
        ),
    )


# ── Register embed (GENE_Mic-relative, phase-linked) ─────────────────────────


def native_cell_count(n: int) -> int:
    bits = max(1, int(n).bit_length())
    return max(1, (bits + 5) // 6)


def uv_ir_phase_link(cell: int, modulus: int) -> int:
    """Per-cell limb phase from modulus radix slices (matches native gyro_kern_phase_link)."""
    slice_r = (int(modulus) >> (6 * int(cell))) & MASK6
    byte = (slice_r & 0xFF) ^ GENE_MIC_S
    state = step_byte(GENE_MAC_REST, byte)
    return int(chirality_word6(state)) & MASK6


def phase_links(modulus: int, n_cells: int) -> List[int]:
    return [uv_ir_phase_link(c, modulus) for c in range(int(n_cells))]


def cell_byte_for_residue(y: int, cell: int, phases: List[int]) -> int:
    limb = (int(y) >> (6 * int(cell))) & MASK6
    payload = (limb ^ int(phases[cell])) & MASK6
    return byte_from_family_and_payload6(int(cell) & 3, payload)


def inject_residue_state(y: int, modulus: int) -> int:
    """Legacy single-chain embed: one state24 after sequential cell bytes."""
    nc = native_cell_count(modulus)
    ph = phase_links(modulus, nc)
    state = int(GENE_MAC_REST)
    for c in range(nc):
        state = int(step_byte(state, cell_byte_for_residue(y, c, ph)))
    return state


def inject_residue_multicell(y: int, modulus: int, n_cells: int | None = None) -> List[int]:
    """Encode y ∈ ℤ_N as B independent GENE_Mac cell states (QuBEC register)."""
    nc = int(n_cells) if n_cells is not None else native_cell_count(modulus)
    ph = phase_links(modulus, nc)
    router = MultiCellRouter(nc)
    for c in range(nc):
        router.step_byte(c, cell_byte_for_residue(y, c, ph))
    return router.snapshot()


def multicell_omega_key(states: List[int]) -> Tuple[int, ...]:
    return tuple(omega_index_from_state(s) for s in states)


def omega_index_from_state(state24: int) -> int:
    o = state24_to_omega12(int(state24) & 0xFFFFFF)
    return int(o.u6) * 64 + int(o.v6)


def apply_bytes_state(state24: int, word: List[int]) -> int:
    s = int(state24) & 0xFFFFFF
    for b in word:
        s = int(step_byte(s, int(b) & 0xFF))
    return s


# ── Gyroscopic oracle compiler ───────────────────────────────────────────────


@dataclass
class GyroOperator:
    """Compiled byte/intron operator on GENE_Mac."""

    kind: str
    N: int
    param_a: int
    param_b: int = 0
    words: List[List[int]] = field(default_factory=list)
    bytes_flat: List[int] = field(default_factory=list)
    introns: List[int] = field(default_factory=list)
    families: List[int] = field(default_factory=list)
    micro_refs: List[int] = field(default_factory=list)
    omega_signature: OmegaSignature12 = field(default_factory=lambda: IDENTITY_SIG)
    compile_method: str = ""
    compiled: bool = False

    def __post_init__(self) -> None:
        if not self.bytes_flat and self.words:
            flat: List[int] = []
            for w in self.words:
                flat.extend(int(b) & 0xFF for b in w)
            object.__setattr__(self, "bytes_flat", flat)
        if not self.introns and self.bytes_flat:
            object.__setattr__(
                self,
                "introns",
                [int(b) ^ GENE_MIC_S for b in self.bytes_flat],
            )
        if not self.families and self.introns:
            object.__setattr__(
                self,
                "families",
                [int(intron_family(i)) for i in self.introns],
            )
        if not self.micro_refs and self.introns:
            object.__setattr__(
                self,
                "micro_refs",
                [int(intron_micro_ref(i)) for i in self.introns],
            )
        if self.omega_signature == IDENTITY_SIG and self.bytes_flat:
            object.__setattr__(
                self,
                "omega_signature",
                omega_word_signature(self.bytes_flat),
            )

    def apply_state24(self, state24: int) -> int:
        return apply_bytes_state(state24, self.bytes_flat)

    def apply_multicell(self, states: List[int]) -> List[int]:
        """Apply compiled word per cell. Fail-closed stub until compiler lands."""
        if not self.compiled or not self.bytes_flat:
            return [int(s) for s in states]
        return [apply_bytes_state(int(s), self.bytes_flat) for s in states]

    def apply_psi(self, psi: List[complex]) -> List[complex]:
        return apply_word(psi, self.bytes_flat)

    def signature_power(self, exponent: int) -> OmegaSignature12:
        e = max(0, int(exponent))
        if e == 0:
            return IDENTITY_SIG
        acc, base, k = IDENTITY_SIG, self.omega_signature, e
        while k > 0:
            if k & 1:
                acc = compose_omega_signatures(acc, base)
            base = compose_omega_signatures(base, base)
            k >>= 1
        return acc


_COMPILE_CACHE: Dict[Tuple[int, int], GyroOperator] = {}


def compile_factor_operator(N: int, base: int) -> GyroOperator:
    """
    Compile (N,a) → U_{N,a}: multi-cell byte word with holonomy y ↦ a·y (mod N)
    on phase-linked QuBEC register. Fail-closed until carry-coupled compiler lands.

    Target: inject_residue_multicell(y) → inject_residue_multicell(a·y mod N)
    via MultiCellRouter byte-ledger (no classical pow() in the hot path).
    """
    key = (int(N), int(base) % max(1, int(N)))
    if key in _COMPILE_CACHE:
        return _COMPILE_CACHE[key]

    nn, aa = key[0], key[1]
    if nn <= 1 or aa == 0:
        raise ValueError("invalid factor compile inputs")

    nc = native_cell_count(nn)
    op = GyroOperator(
        kind="factor_mul_OPEN",
        N=nn,
        param_a=aa,
        words=[],
        bytes_flat=[],
        compile_method=f"MULTICELL_OPEN_B={nc}",
        compiled=False,
    )
    _COMPILE_CACHE[key] = op
    return op


def compile_dlp_operator(N: int, g: int, h: int) -> Tuple[GyroOperator, GyroOperator]:
    ug = compile_factor_operator(N, g)
    hh = int(h) % int(N)
    if hh == 0:
        raise ValueError("h must be invertible mod N")
    h_inv = pow(hh, -1, int(N))
    uh = compile_factor_operator(N, h_inv)
    uh = GyroOperator(
        kind="dlp_h_inv",
        N=ug.N,
        param_a=h_inv,
        param_b=hh,
        words=uh.words,
        bytes_flat=uh.bytes_flat,
        omega_signature=uh.omega_signature,
        compile_method=uh.compile_method,
        compiled=uh.compiled,
    )
    return ug, uh


# ── Holonomy spectrum readout ────────────────────────────────────────────────


@dataclass
class HolonomySpectrum:
    path: str
    closure_depth: Optional[int]
    half_depth: Optional[int]
    signature_order: Optional[int]
    k4_sector: int
    shell_trajectory: List[int]
    horizon_hits: int
    z2_sheet_flips: int
    wht_peaks: List[Tuple[int, float]]
    eigenspace_balance: float
    candidate_period: Optional[int]
    register_matches_mul: bool
    notes: str = ""


def _signature_is_identity(sig: OmegaSignature12) -> bool:
    return int(sig.parity) == 0 and int(sig.tau_u6) == 0 and int(sig.tau_v6) == 0


def _rest_psi() -> List[complex]:
    h, q = state_to_hq(GENE_MAC_REST)
    psi = zero_wavefunction()
    psi[index_hq(h, q)] = complex(1.0, 0.0)
    return psi


def _eigenspace_balance(psi: List[complex]) -> float:
    psi_f = apply_k4(psi, K4_F, micro_ref=1)
    num = sum(abs(psi_f[i] - psi[i]) for i in range(OMEGA_SIZE))
    den = sum(abs(psi[i]) for i in range(OMEGA_SIZE)) + 1e-30
    return float(num / den)


def _chirality_histogram(psi: List[complex]) -> List[float]:
    hist = [0.0] * 64
    for i, z in enumerate(psi):
        if abs(z) < 1e-30:
            continue
        hist[i // 64] += abs(z) ** 2
    return hist


def _z2_sheet(state24: int) -> int:
    rest = int(GENE_MAC_REST)
    swap = int(apply_gate_S(rest))
    s = int(state24) & 0xFFFFFF
    d_rest = (s ^ rest).bit_count()
    d_swap = (s ^ swap).bit_count()
    return 0 if d_rest <= d_swap else 1


def holonomy_spectrum(
    op: GyroOperator,
    *,
    max_depth: int = 256,
) -> HolonomySpectrum:
    """
    Drive holonomy on Ω: compose operator on ψ and track K4/horizon/Z₂/signature closure.
    No ℤ_Q enumeration.
    """
    if not op.compiled or not op.bytes_flat:
        return HolonomySpectrum(
            path="HOLONOMY_FAIL_CLOSED",
            closure_depth=None,
            half_depth=None,
            signature_order=None,
            k4_sector=0,
            shell_trajectory=[],
            horizon_hits=0,
            z2_sheet_flips=0,
            wht_peaks=[],
            eigenspace_balance=0.0,
            candidate_period=None,
            register_matches_mul=False,
            notes=f"operator not compiled (method={op.compile_method})",
        )

    max_d = max(2, int(max_depth))
    sig_order: Optional[int] = None
    for d in range(1, max_d + 1):
        if _signature_is_identity(op.signature_power(d)):
            sig_order = d
            break

    nn = int(op.N)
    inject1 = inject_residue_state(1, nn)
    reg_state = inject1
    closure_depth: Optional[int] = None
    shell_traj: List[int] = []
    horizon_hits = 0
    z2_flips = 0
    prev_z2 = _z2_sheet(reg_state)

    psi = _rest_psi()
    for depth in range(1, max_d + 1):
        psi = op.apply_psi(psi)
        reg_state = op.apply_state24(reg_state)

        chi = int(chirality_word(reg_state)) & MASK6
        shell_traj.append(int(shell_index_from_chirality6(chi)))
        if (
            is_on_horizon(reg_state)
            or is_on_equality_horizon(reg_state)
            or (int(chirality_word(reg_state)) & MASK6) == EPSILON_6
        ):
            horizon_hits += 1

        z2 = _z2_sheet(reg_state)
        if z2 != prev_z2:
            z2_flips += 1
        prev_z2 = z2

        if reg_state == inject1 and depth > 0 and closure_depth is None:
            closure_depth = depth

    hist = _chirality_histogram(psi)
    peaks = top_nontrivial_peaks(wht64_dc_free(hist), k=8)
    balance = _eigenspace_balance(psi)

    half = closure_depth // 2 if closure_depth and closure_depth % 2 == 0 else None

    reg_ok = True
    for y in range(min(nn, 64)):
        got = op.apply_state24(inject_residue_state(y, nn))
        exp = inject_residue_state(int(mul_mod_ladder(y, op.param_a, nn)), nn)
        if got != exp:
            reg_ok = False
            break

    candidate = sig_order if sig_order else closure_depth
    notes: List[str] = []
    if sig_order and closure_depth and sig_order != closure_depth:
        notes.append(f"sig_order={sig_order} reg_closure={closure_depth}")

    return HolonomySpectrum(
        path="HOLONOMY_SPECTRUM",
        closure_depth=closure_depth,
        half_depth=half,
        signature_order=sig_order,
        k4_sector=int(op.omega_signature.parity),
        shell_trajectory=shell_traj[:32],
        horizon_hits=horizon_hits,
        z2_sheet_flips=z2_flips,
        wht_peaks=peaks,
        eigenspace_balance=balance,
        candidate_period=candidate,
        register_matches_mul=reg_ok,
        notes="; ".join(notes),
    )


def _verify_period(n: int, base: int, r: int) -> bool:
    if r <= 1:
        return False
    return int(exp_mod_ladder(int(base) % int(n), int(r), int(n))) == 1


def gyro_period(N: int, base: int, *, max_depth: int = 512) -> Optional[int]:
    """ord_N(base) from holonomy closure; single exp_mod_ladder verify at end."""
    nn, aa = int(N), int(base) % int(N)
    if nn <= 1 or math.gcd(aa, nn) != 1:
        return None

    op = compile_factor_operator(nn, aa)
    if not op.compiled:
        return None

    spec = holonomy_spectrum(op, max_depth=int(max_depth))
    for cand in (spec.signature_order, spec.closure_depth):
        if cand is not None and cand > 1 and _verify_period(nn, aa, int(cand)):
            return int(cand)
    return None


def gyro_factor(N: int, base: int | None = None) -> Optional[Tuple[int, int]]:
    nn = int(N)
    if nn <= 2 or nn % 2 == 0:
        return None

    bb = 2 if base is None else int(base) % nn
    if math.gcd(bb, nn) != 1:
        p = math.gcd(bb, nn)
        return (p, nn // p) if 1 < p < nn else None

    r = gyro_period(nn, bb)
    if r is None or r % 2 != 0:
        return None

    x = int(exp_mod_ladder(bb, r // 2, nn))
    if x in (1, nn - 1):
        return None

    p = math.gcd(x - 1, nn)
    q = math.gcd(x + 1, nn)
    if 1 < p < nn and 1 < q < nn:
        return (min(p, q), max(p, q))
    return None


def gyro_dlp(N: int, g: int, h: int) -> Optional[int]:
    """Dual-flow holonomy DLP — fail closed until relation readout lands."""
    _ = (N, g, h)
    return None


__all__ = [
    "HorizonAlgebraReport",
    "GyroOperator",
    "HolonomySpectrum",
    "analyze_horizon_algebra",
    "compile_factor_operator",
    "compile_dlp_operator",
    "holonomy_spectrum",
    "gyro_period",
    "gyro_factor",
    "gyro_dlp",
    "inject_residue_state",
    "inject_residue_multicell",
    "multicell_omega_key",
    "native_cell_count",
    "uv_ir_phase_link",
]
