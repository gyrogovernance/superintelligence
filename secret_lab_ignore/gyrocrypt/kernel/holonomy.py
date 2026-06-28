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


def decode_limb_from_state(
    cell: int, state24: int, modulus: int, phases: List[int]
) -> int:
    """Invert phase-linked inject for one cell (limb is NOT raw chirality)."""
    target = int(state24) & 0xFFFFFF
    c = int(cell)
    for limb in range(64):
        probe_y = limb << (6 * c)
        b = cell_byte_for_residue(probe_y, c, phases)
        if int(step_byte(GENE_MAC_REST, b)) == target:
            return limb
    raise ValueError(f"decode_limb_from_state: no limb for cell {c}")


def decode_residue_multicell(
    states: List[int], modulus: int, n_cells: int | None = None
) -> int:
    """Decode y from multi-cell QuBEC register (inverse of inject_residue_multicell)."""
    nn = int(modulus)
    nc = int(n_cells) if n_cells is not None else native_cell_count(nn)
    if len(states) != nc:
        raise ValueError("states length must equal n_cells")
    ph = phase_links(nn, nc)
    y = 0
    for c in range(nc):
        limb = decode_limb_from_state(c, states[c], nn, ph)
        y |= int(limb) << (6 * c)
    return y % nn


def _chi_as_limb_decode(states: List[int], n_cells: int) -> int:
    """WRONG readback used in proposed fake ALU — chi ≠ encoded limb (K17)."""
    y = 0
    for c in range(int(n_cells)):
        chi = int(chirality_word6(states[c])) & MASK6
        y |= chi << (6 * c)
    return y


# ── Multi-cell ALU program (byte-ledger target; no classical mul in hot path) ──


@dataclass(frozen=True)
class RouterStep:
    """One reversible step on MultiCellRouter (audit/replay ledger)."""

    kind: str  # "byte" | "cnot" | "gate"
    cell: int = 0
    byte: int = 0
    control_cell: int = 0
    target_cell: int = 0
    bit: int = 0
    family: int = 0
    gate: str = ""


@dataclass
class MulticellALUProgram:
    """
    Compiled carry-coupled oracle as a byte/CNOT ledger on MultiCellRouter.

    Real multiply-by-a mod N requires reversible ripple-carry across B 6-bit limbs
    via chirality_cnot_meas_ctrl — not (decode → int mul → reinject).
    """

    N: int
    param_a: int
    n_cells: int
    steps: List[RouterStep] = field(default_factory=list)

    @property
    def compiled(self) -> bool:
        return len(self.steps) > 0


def replay_router_program(states: List[int], program: MulticellALUProgram) -> List[int]:
    """Replay a compiled ALU ledger from an initial multi-cell snapshot."""
    router = MultiCellRouter(program.n_cells)
    router.states = [int(s) for s in states]
    for st in program.steps:
        if st.kind == "byte":
            router.step_byte(st.cell, st.byte)
        elif st.kind == "cnot":
            router.chirality_cnot_meas_ctrl(
                st.control_cell, st.target_cell, st.bit, family=st.family
            )
        elif st.kind == "gate":
            router.apply_gate_all(st.gate)
        else:
            raise ValueError(f"unknown RouterStep kind {st.kind!r}")
    return router.snapshot()


def verify_multicell_mul_reference(
    op: GyroOperator, *, sample: int | None = None
) -> bool:
    """
    Check apply_multicell against inject reference (offline audit only).
    mul_mod_ladder states the target — not used in apply_multicell hot path.
    """
    if not op.compiled:
        return False
    nn, aa = int(op.N), int(op.param_a)
    nc = int(op.n_cells)
    cap = nn if sample is None else min(nn, int(sample))
    for y in range(cap):
        st = inject_residue_multicell(y, nn, nc)
        got = op.apply_multicell(st)
        exp_y = int(mul_mod_ladder(y, aa, nn))
        exp = inject_residue_multicell(exp_y, nn, nc)
        if got != exp:
            return False
    return True


def _compile_inject_tabulation(nn: int, aa: int, nc: int) -> GyroOperator:
    """Wiring-only tabulation (K6): compile-time table; not production oracle."""
    lookup = [int(mul_mod_ladder(y, aa, nn)) for y in range(nn)]
    return GyroOperator(
        kind="factor_mul",
        N=nn,
        param_a=aa,
        n_cells=nc,
        lookup_z=lookup,
        compile_method=f"WIRE_TABULATE_B={nc}",
        compiled=True,
    )


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
    """Compiled multi-cell operator on GENE_Mac (byte word and/or inject tabulation)."""

    kind: str
    N: int
    param_a: int
    param_b: int = 0
    n_cells: int = 1
    lookup_z: List[int] = field(default_factory=list)
    alu_program: Optional[MulticellALUProgram] = None
    words: List[List[int]] = field(default_factory=list)
    bytes_flat: List[int] = field(default_factory=list)
    introns: List[int] = field(default_factory=list)
    families: List[int] = field(default_factory=list)
    micro_refs: List[int] = field(default_factory=list)
    omega_signature: OmegaSignature12 = field(default_factory=lambda: IDENTITY_SIG)
    compile_method: str = ""
    compiled: bool = False

    def __post_init__(self) -> None:
        if self.n_cells <= 0:
            object.__setattr__(self, "n_cells", native_cell_count(int(self.N)))
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
        """Apply compiled oracle: inject tabulation, ALU ledger, or byte word."""
        if not self.compiled:
            return [int(s) for s in states]
        nn, nc = int(self.N), int(self.n_cells)
        if self.lookup_z:
            y = decode_residue_multicell(states, nn, nc)
            z = int(self.lookup_z[int(y) % len(self.lookup_z)])
            return inject_residue_multicell(z, nn, nc)
        if self.alu_program is not None and self.alu_program.compiled:
            return replay_router_program(states, self.alu_program)
        if self.bytes_flat:
            return [apply_bytes_state(int(s), self.bytes_flat) for s in states]
        return [int(s) for s in states]

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


def search_multiply_word_bfs(
    n: int,
    base: int,
    *,
    n_cells: int = 1,
    max_depth: int = 6,
    max_nodes: int = 500_000,
) -> Optional[List[int]]:
    """
    Offline search: byte word W with parallel action
      apply(W, inject_residue_multicell(y)) = inject_residue_multicell(a·y mod N)
    for all y. mul_mod_ladder states the target only (audit).

    K16/K18: no word at depth≤6 for N=15,a=7 (single-cell). apply_word on ℂ^4096
    is the same state map — wavefunction lift does not add carry structure.
    """
    from collections import deque

    nn, aa = int(n), int(base) % int(n)
    nc = max(1, int(n_cells))

    def embed(y: int) -> Tuple[int, ...]:
        return tuple(int(s) for s in inject_residue_multicell(y, nn, nc))

    start = tuple(embed(y) for y in range(nn))
    target = tuple(embed(int(mul_mod_ladder(y, aa, nn))) for y in range(nn))

    def apply_b(
        sts: Tuple[Tuple[int, ...], ...], byte: int
    ) -> Tuple[Tuple[int, ...], ...]:
        b = int(byte) & 0xFF
        return tuple(
            tuple(apply_bytes_state(int(s), [b]) for s in row) for row in sts
        )

    if start == target:
        return []

    seen: Dict[Tuple[Tuple[int, ...], ...], List[int]] = {start: []}
    q: deque[Tuple[Tuple[int, ...], ...]] = deque([start])

    while q and len(seen) < max_nodes:
        sts = q.popleft()
        word = seen[sts]
        if len(word) >= int(max_depth):
            continue
        for b in range(256):
            nsts = apply_b(sts, b)
            if nsts == target:
                return word + [b]
            if nsts not in seen:
                seen[nsts] = word + [b]
                q.append(nsts)
    return None


def compile_factor_operator(N: int, base: int) -> GyroOperator:
    """
    Compile (N,a) → U_{N,a} on phase-linked QuBEC register.

    Current compiler: WIRE_TABULATE (K6 wiring harness — compile-time table only).
    CNOT carry ledger (MulticellALUProgram) is the open milestone.
    """
    key = (int(N), int(base) % max(1, int(N)))
    if key in _COMPILE_CACHE:
        return _COMPILE_CACHE[key]

    nn, aa = key[0], key[1]
    if nn <= 1 or aa == 0:
        raise ValueError("invalid factor compile inputs")

    nc = native_cell_count(nn)
    if nn >= (1 << 63):
        raise ValueError(f"N={nn} exceeds uint64 holonomy tabulation")

    op = _compile_inject_tabulation(nn, aa, nc)
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
    Drive holonomy: compose operator on register / ψ; track closure and mul reference.
    """
    if not op.compiled:
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

    nn, nc = int(op.N), int(op.n_cells)

    if op.lookup_z or op.alu_program is not None:
        inject1 = inject_residue_multicell(1, nn, nc)
        reg_state = list(inject1)
        closure_depth: Optional[int] = None
        shell_traj: List[int] = []
        horizon_hits = 0
        z2_flips = 0
        prev_z2 = _z2_sheet(reg_state[0])

        for depth in range(1, max(2, int(max_depth)) + 1):
            reg_state = op.apply_multicell(reg_state)
            chi = sum(int(chirality_word6(s)) & MASK6 for s in reg_state)
            shell_traj.append(chi & MASK6)
            if any(is_on_horizon(s) or is_on_equality_horizon(s) for s in reg_state):
                horizon_hits += 1
            z2 = _z2_sheet(reg_state[0])
            if z2 != prev_z2:
                z2_flips += 1
            prev_z2 = z2
            if reg_state == inject1 and depth > 0 and closure_depth is None:
                closure_depth = depth

        reg_ok = verify_multicell_mul_reference(op, sample=min(nn, 64))
        return HolonomySpectrum(
            path="HOLONOMY_MULTICELL",
            closure_depth=closure_depth,
            half_depth=closure_depth // 2
            if closure_depth and closure_depth % 2 == 0
            else None,
            signature_order=None,
            k4_sector=0,
            shell_trajectory=shell_traj[:32],
            horizon_hits=horizon_hits,
            z2_sheet_flips=z2_flips,
            wht_peaks=[],
            eigenspace_balance=0.0,
            candidate_period=closure_depth,
            register_matches_mul=reg_ok,
            notes=f"method={op.compile_method}",
        )

    if not op.bytes_flat:
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
            notes="no byte word or tabulation",
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
    reg_closure_depth: Optional[int] = None
    reg_shell_traj: List[int] = []
    horizon_hits = 0
    z2_flips = 0
    prev_z2 = _z2_sheet(reg_state)

    psi = _rest_psi()
    for depth in range(1, max_d + 1):
        psi = op.apply_psi(psi)
        reg_state = op.apply_state24(reg_state)

        chi = int(chirality_word(reg_state)) & MASK6
        reg_shell_traj.append(int(shell_index_from_chirality6(chi)))
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

        if reg_state == inject1 and depth > 0 and reg_closure_depth is None:
            reg_closure_depth = depth

    hist = _chirality_histogram(psi)
    peaks = top_nontrivial_peaks(wht64_dc_free(hist), k=8)
    balance = _eigenspace_balance(psi)

    half = reg_closure_depth // 2 if reg_closure_depth and reg_closure_depth % 2 == 0 else None

    reg_ok = True
    for y in range(min(nn, 64)):
        got = op.apply_state24(inject_residue_state(y, nn))
        exp = inject_residue_state(int(mul_mod_ladder(y, op.param_a, nn)), nn)
        if got != exp:
            reg_ok = False
            break

    candidate = sig_order if sig_order else reg_closure_depth
    notes: List[str] = []
    if sig_order and reg_closure_depth and sig_order != reg_closure_depth:
        notes.append(f"sig_order={sig_order} reg_closure={reg_closure_depth}")

    return HolonomySpectrum(
        path="HOLONOMY_SPECTRUM",
        closure_depth=reg_closure_depth,
        half_depth=half,
        signature_order=sig_order,
        k4_sector=int(op.omega_signature.parity),
        shell_trajectory=reg_shell_traj[:32],
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


def holonomy_closure_period(
    op: GyroOperator, *, max_depth: int = 50_000
) -> Optional[int]:
    """Period from iterating apply_multicell until register returns to inject(1)."""
    if not op.compiled:
        return None
    nn, nc = int(op.N), int(op.n_cells)
    one = inject_residue_multicell(1, nn, nc)
    st = list(one)
    cap = max(2, int(max_depth))
    for d in range(1, cap + 1):
        st = op.apply_multicell(st)
        if st == one:
            return int(d) if d > 1 else None
    return None


def holonomy_suffix_period(N: int, base: int, Q: int | None = None) -> Optional[int]:
    """F_{G_X} suffix/beam readout via native audit scorer (readout leg)."""
    from kernel.audit import period_reference

    return period_reference(int(N), int(base), Q)


@dataclass
class HolonomyE2EReport:
    N: int
    base: int
    n_cells: int
    compile_method: str
    oracle_ok: bool
    oracle_checked: int
    closure_period: Optional[int]
    suffix_period: Optional[int]
    suffix_path: str
    periods_match: bool
    notes: str = ""


def holonomy_e2e(
    N: int,
    base: int,
    Q: int | None = None,
    *,
    verify_all_oracle: bool = True,
    max_closure_depth: int = 50_000,
) -> HolonomyE2EReport:
    """
    End-to-end holonomy pipeline:
      compile → oracle verify → closure (small r) / suffix readout → audit parity.
    """
    from kernel.bindings import shor_last_path_tag

    nn, aa = int(N), int(base) % int(N)
    if nn <= 1 or math.gcd(aa, nn) != 1:
        raise ValueError("invalid holonomy_e2e inputs")

    op = compile_factor_operator(nn, aa)
    nc = int(op.n_cells)
    checked = nn if verify_all_oracle else min(nn, 256)
    oracle_ok = verify_multicell_mul_reference(op, sample=checked)

    closure: Optional[int] = None
    if oracle_ok and max_closure_depth > 0:
        closure = holonomy_closure_period(op, max_depth=max_closure_depth)
        if closure is not None and not _verify_period(nn, aa, closure):
            closure = None

    suffix = holonomy_suffix_period(nn, aa, Q) if oracle_ok else None
    path = shor_last_path_tag() if suffix else "NONE"

    notes: List[str] = []
    if op.lookup_z:
        notes.append("oracle=WIRE_TABULATE (K6 wiring; CNOT ALU open)")
    if closure and suffix and closure != suffix:
        notes.append(f"closure={closure} suffix={suffix}")

    return HolonomyE2EReport(
        N=nn,
        base=aa,
        n_cells=nc,
        compile_method=op.compile_method,
        oracle_ok=oracle_ok,
        oracle_checked=checked,
        closure_period=closure,
        suffix_period=suffix,
        suffix_path=path,
        periods_match=closure == suffix if closure is not None else suffix is not None,
        notes="; ".join(notes),
    )


def gyro_period(N: int, base: int, *, max_depth: int = 512) -> Optional[int]:
    """ord_N(base): closure when cheap; else suffix readout after oracle verify."""
    nn, aa = int(N), int(base) % int(N)
    if nn <= 1 or math.gcd(aa, nn) != 1:
        return None

    op = compile_factor_operator(nn, aa)
    if not op.compiled or not verify_multicell_mul_reference(op, sample=min(nn, 256)):
        return None

    if nn <= 4096:
        r = holonomy_closure_period(op, max_depth=max(int(max_depth), 50_000))
        if r is not None and _verify_period(nn, aa, r):
            return int(r)

    r = holonomy_suffix_period(nn, aa)
    return int(r) if r is not None and _verify_period(nn, aa, int(r)) else None


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
    "search_multiply_word_bfs",
    "holonomy_spectrum",
    "holonomy_closure_period",
    "holonomy_suffix_period",
    "holonomy_e2e",
    "HolonomyE2EReport",
    "gyro_period",
    "gyro_factor",
    "gyro_dlp",
    "inject_residue_state",
    "inject_residue_multicell",
    "decode_residue_multicell",
    "decode_limb_from_state",
    "MulticellALUProgram",
    "RouterStep",
    "replay_router_program",
    "verify_multicell_mul_reference",
    "_chi_as_limb_decode",
    "multicell_omega_key",
    "native_cell_count",
    "uv_ir_phase_link",
]
