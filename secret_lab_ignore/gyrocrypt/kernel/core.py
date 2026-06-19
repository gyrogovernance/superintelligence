"""
kernel/core.py — aQPU kernel: byte bridge, wavefunction holonomy, WHT, MultiCellRouter.

This module is the Python realization of the aQPU computational primitives specified
in the Gyroscopic ASI SDK (docs/Gyroscopic_ASI_SDK_Quantum_Computing.md). It provides
four overlapping computational surfaces over the same 4096-state manifold Ω:

─────────────────────────────────────────────────────────────────────────────────
I. BYTE-LEDGER BRIDGE (SDK §5.1)
─────────────────────────────────────────────────────────────────────────────────
Wraps src.api and src.constants to expose the kernel transition law as Python
functions: step_byte, step_byte_inverse, q6 (q-class), chirality_word, shell_index,
horizon_distance, ab_distance, rest_distance, commutator_defect. WHT via wht64_fast;
wht64_dc_free removes DC before spectral analysis.

─────────────────────────────────────────────────────────────────────────────────
II. WAVEFUNCTION HOLONOMY — QuBEC-native (Analysis_aQPU_Wavefunction.md, T1–T10)
─────────────────────────────────────────────────────────────────────────────────
QuBEC = occupied Shared Moment on Ω. Wavefunction holonomy is the Hilbert lift
on that manifold — an advanced kernel path that bypasses the byte stream while
staying on Ω. ψ ∈ ℂ⁴⁰⁹⁶ is acted on by K4 {id, W₂, W₂', F} via apply_k4.

Full holographic readout: wavefunction_hq_spectral_peaks measures p(h,q)=|ψ|²
and applies WHT^{⊗2} on the native 64×64 chart (dual coordinates sh, sq).

─────────────────────────────────────────────────────────────────────────────────
III. RESIDUE ↔ CHIRALITY BRIDGE (SDK §5.1, F58/F59)
─────────────────────────────────────────────────────────────────────────────────
residue_to_chirality(r) routes through one kernel byte step on GENE_MAC_REST
(chirality_word6). uv_ir_phase_link grounds per-cell limb phases for compile tables.

─────────────────────────────────────────────────────────────────────────────────
IV. MULTI-CELL TENSOR PRODUCTS (SDK §11.4)
─────────────────────────────────────────────────────────────────────────────────
wht_tensor2 provides the 4096-point WHT^⊗2 for two-cell joint spectral
analysis via row-column decomposition using wht64. Holographic readout
transform for the full (h,q) chart (Section II).

MultiCellRouter, per-cell gate application (apply_F/S/C_each_cell),
chirality-conditioned classical control (chirality_cnot_meas_ctrl),
depth-4 cycle verification (depth4_cycle), and byte_for_q6 implement
QuBEC-native multi-cell execution (byte ledger on Ω per cell).

Imports: src.api, src.constants. Re-exported via kernel/__init__.py.
"""

from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

from typing import Dict, List, Optional, Tuple


def _find_repo_root(start: Path) -> Path:
    for p in (start.parent, *start.parents):
        if (p / "src").is_dir():
            return p
    raise RuntimeError("gyrocrypt: could not locate repo root containing ./src")


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.api import (  # noqa: E402
    OmegaSignature12,
    OmegaState12,
    apply_omega_signature,
    chirality_word6,
    compose_omega_signatures,
    is_in_omega24,
    omega_word_signature,
    q_word6,
    state24_to_omega12,
    word_signature,
)
from src.constants import (  # noqa: E402
    COMPLEMENT_MASK_12,
    DELTA_BU,
    GATE_C_BYTES,
    GATE_S_BYTES,
    GENE_MAC_REST,
    GENE_MIC_S,
    LAYER_MASK_12,
    M_A,
    MASK_STATE24,
    RHO,
    apply_gate_C,
    apply_gate_F,
    apply_gate_S,
    apply_gate,
    byte_micro_ref,
    byte_to_intron,
    component_density,
    intron_family,
    intron_micro_ref,
    inverse_step_by_byte,
    pack_state,
    popcount,
    step_state_by_byte,
    unpack_state,
    single_step_trace,
)

step_byte = step_state_by_byte
step_byte_inverse = inverse_step_by_byte
q6 = q_word6
chirality_word = chirality_word6


def step_byte_stages(state24: int, byte: int) -> Dict[str, int]:
    """
    CGM stage trace for one byte transition (diagnostics / experiments only).

    Returns cs (intron), una (A_mut), ona (A_next), bu (B_next), state24 (next).
    """
    trace = single_step_trace(int(state24) & MASK_STATE24, int(byte) & 0xFF)
    return {
        "cs": trace["cs"],
        "una": trace["una"],
        "ona": trace["ona"],
        "bu": trace["bu"],
        "state24": trace["state24"],
    }


def horizon_distance(state24: int) -> int:
    a12, b12 = unpack_state(state24)
    return popcount(int(a12) ^ (int(b12) ^ COMPLEMENT_MASK_12))


def ab_distance(state24: int) -> int:
    a12, b12 = unpack_state(state24)
    return popcount(int(a12) ^ int(b12))


def intron_to_byte(intron: int) -> int:
    return (int(intron) & 0xFF) ^ GENE_MIC_S


def wht64_fast(v: list[float]) -> list[float]:
    """Orthonormal 64-point WHT via butterfly (no matrix multiply)."""
    if len(v) != 64:
        raise ValueError("wht64_fast requires length 64")
    a = [float(x) for x in v]
    h = 1
    n = 64
    while h < n:
        step = h << 1
        for i in range(0, n, step):
            j_end = i + h
            for j in range(i, j_end):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h = step
    inv = 1.0 / 8.0
    return [x * inv for x in a]


def wht64(v: list[float]) -> list[float]:
    """Orthonormal 64-point Walsh-Hadamard transform (src.api matrix)."""
    if len(v) != 64:
        raise ValueError("wht64 requires length 64")
    return wht64_fast(v)


def wht64_dc_free(hist: list[float]) -> list[float]:
    """Remove the trivial character (DC) before WHT."""
    if len(hist) != 64:
        raise ValueError("wht64_dc_free requires length 64")
    mean = sum(hist) / 64.0
    return wht64([x - mean for x in hist])


def top_nontrivial_peaks(spec: list[float], k: int = 8) -> List[Tuple[int, float]]:
    """Top-k peaks excluding index 0 (trivial DC character)."""
    ranked = sorted(
        ((i, abs(spec[i])) for i in range(1, 64)),
        key=lambda t: t[1],
        reverse=True,
    )
    return ranked[:k]


# --- Wavefunction holonomy (L0 psi over Omega) ---

HORIZON_SIZE = 64
OMEGA_SIZE = 4096
CHIRALITY_MASK_6 = 0x3F

K4_ID = 0
K4_W2 = 1
K4_W2P = 2
K4_F = 3


def zero_wavefunction() -> List[complex]:
    return [0j] * OMEGA_SIZE


def index_hq(h: int, q: int) -> int:
    return (h & CHIRALITY_MASK_6) * HORIZON_SIZE + (q & CHIRALITY_MASK_6)


def split_index(i: int) -> Tuple[int, int]:
    return (i // HORIZON_SIZE, i % HORIZON_SIZE)


def _byte_from_family_micro(family: int, micro_ref: int) -> int:
    fam = int(family) & 0x3
    m = int(micro_ref) & 0x3F
    intron = ((fam >> 1) & 1) << 7 | (m << 1) | (fam & 1)
    return intron ^ GENE_MIC_S


def byte_from_family_and_payload6(family: int, payload6: int) -> int:
    """Byte from (family, payload6): intron[7,0]=family, intron[1..6]=micro_ref."""
    return _byte_from_family_micro(int(family) & 0x3, int(payload6) & 0x3F)


def _w2_word(micro_ref: int) -> List[int]:
    m = int(micro_ref) & 0x3F
    return [_byte_from_family_micro(0, m), _byte_from_family_micro(1, m)]


def _w2p_word(micro_ref: int) -> List[int]:
    m = int(micro_ref) & 0x3F
    return [_byte_from_family_micro(2, m), _byte_from_family_micro(3, m)]


def _k4_omega_signature(gate: int, micro_ref: int) -> OmegaSignature12:
    m = int(micro_ref) & 0x3F
    if gate == K4_W2:
        return omega_word_signature(_w2_word(m))
    if gate == K4_W2P:
        return omega_word_signature(_w2p_word(m))
    if gate == K4_F:
        return compose_omega_signatures(
            omega_word_signature(_w2_word(m)),
            omega_word_signature(_w2p_word(m)),
        )
    return OmegaSignature12(0, 0, 0)


_K4_PERM_CACHE: Dict[Tuple[int, int], List[int]] = {}
_WORD_PERM_CACHE: Dict[Tuple[int, ...], List[int]] = {}


def word_index_permutation(word: List[int]) -> List[int]:
    """Index permutation on Ω for an arbitrary byte ledger (compiled Ω12 signature)."""
    key = tuple(int(b) & 0xFF for b in word)
    cached = _WORD_PERM_CACHE.get(key)
    if cached is not None:
        return cached
    sig = omega_word_signature(list(key))
    perm = [0] * OMEGA_SIZE
    for h in range(HORIZON_SIZE):
        for q in range(HORIZON_SIZE):
            src = index_hq(h, q)
            dst_omega = apply_omega_signature(OmegaState12(h, q), sig)
            perm[src] = index_hq(dst_omega.u6, dst_omega.v6)
    seen = bytearray(OMEGA_SIZE)
    for dst in perm:
        if dst < 0 or dst >= OMEGA_SIZE or seen[dst]:
            raise RuntimeError(f"word permutation is not bijective for {len(key)} bytes")
        seen[dst] = 1
    _WORD_PERM_CACHE[key] = perm
    return perm


def apply_word(psi: List[complex], word: List[int]) -> List[complex]:
    """Hilbert lift of an arbitrary byte ledger on Ω."""
    perm = word_index_permutation(word)
    out = [0j] * OMEGA_SIZE
    for src, dst in enumerate(perm):
        amp = psi[src]
        if amp != 0j:
            out[dst] = amp
    return out


def _k4_index_permutation(gate: int, micro_ref: int) -> List[int]:
    key = (int(gate), int(micro_ref) & 0x3F)
    cached = _K4_PERM_CACHE.get(key)
    if cached is not None:
        return cached
    sig = _k4_omega_signature(key[0], key[1])
    perm = [0] * OMEGA_SIZE
    for h in range(HORIZON_SIZE):
        for q in range(HORIZON_SIZE):
            src = index_hq(h, q)
            dst_omega = apply_omega_signature(OmegaState12(h, q), sig)
            perm[src] = index_hq(dst_omega.u6, dst_omega.v6)
    seen = bytearray(OMEGA_SIZE)
    for dst in perm:
        if dst < 0 or dst >= OMEGA_SIZE or seen[dst]:
            raise RuntimeError(f"K4 permutation is not bijective for key={key}")
        seen[dst] = 1
    _K4_PERM_CACHE[key] = perm
    return perm


def apply_k4(
    psi: List[complex],
    gate: int,
    *,
    micro_ref: int = 1,
) -> List[complex]:
    """Apply W₂(m), W₂'(m), or F(m)=W₂∘W₂' via canonical Ω12 signatures (T1–T10)."""
    if gate == K4_ID:
        return list(psi)
    perm = _k4_index_permutation(gate, micro_ref)
    out = [0j] * OMEGA_SIZE
    for src, dst in enumerate(perm):
        amp = psi[src]
        if amp != 0j:
            out[dst] = amp
    return out


def to_holographic(psi: List[complex]) -> List[List[complex]]:
    """Reshape the 4096-vector to the 64x64 horizon representation."""
    return [
        [psi[h * HORIZON_SIZE + q] for q in range(HORIZON_SIZE)]
        for h in range(HORIZON_SIZE)
    ]


def from_holographic(holo: List[List[complex]]) -> List[complex]:
    psi = zero_wavefunction()
    for h in range(HORIZON_SIZE):
        for q in range(HORIZON_SIZE):
            psi[h * HORIZON_SIZE + q] = holo[h][q]
    return psi


# ── Residue ↔ Chirality bridge ──
# Projects modular-arithmetic residues into the aQPU's 6-bit chirality space.


def state_to_hq(state24: int) -> Tuple[int, int]:
    """Holographic (h, q) = (u6, v6) on Ω — correct chart for apply_k4 / WHT⊗WHT."""
    omega = state24_to_omega12(int(state24) & MASK_STATE24)
    return (omega.u6, omega.v6)


def residue_to_byte(r: int) -> int:
    """Compile the low byte of residue r into a kernel transition byte."""
    return intron_to_byte(r & 0xFF)


def residue_to_chirality(r: int) -> int:
    """Production chirality: one QuBEC kernel byte step from compiled residue."""
    return residue_to_chirality_kernel(r)


def residue_to_chirality_kernel(r: int) -> int:
    """Chirality after one kernel byte step from compiled residue byte."""
    state = step_state_by_byte(GENE_MAC_REST, residue_to_byte(r))
    return chirality_word6(state)


def wavefunction_hq_spectral_peaks(
    psi: List[complex],
    k: int = 16,
) -> List[Tuple[int, int, float]]:
    """Top (sh, sq) dual peaks from DC-free WHT^{⊗2} on p(h,q)=|ψ|²."""
    p = [0.0] * OMEGA_SIZE
    total = 0.0
    for i, z in enumerate(psi):
        amp2 = abs(z) ** 2
        if amp2 <= 1e-30:
            continue
        p[i] = amp2
        total += amp2
    if total <= 0.0:
        return []

    inv = 1.0 / total
    mean = 1.0 / float(OMEGA_SIZE)
    p0 = [x * inv - mean for x in p]
    spec = wht_tensor2(p0)

    ranked = sorted(
        ((idx, abs(spec[idx])) for idx in range(1, OMEGA_SIZE)),
        key=lambda item: item[1],
        reverse=True,
    )[:k]

    out: List[Tuple[int, int, float]] = []
    for idx, mag in ranked:
        if mag <= 0.0:
            continue
        sh = (idx // HORIZON_SIZE) & CHIRALITY_MASK_6
        sq = (idx % HORIZON_SIZE) & CHIRALITY_MASK_6
        out.append((sh, sq, float(mag)))
    return out


def wavefunction_from_hq_weights(
    weights: List[Tuple[int, int, float]],
) -> List[complex]:
    """Amplitude-weighted superposition over holographic (h, q) basis states."""
    psi = zero_wavefunction()
    for h, q, weight in weights:
        if weight <= 0.0:
            continue
        psi[index_hq(h, q)] += complex(math.sqrt(float(weight)), 0.0)
    norm2 = sum(abs(z) ** 2 for z in psi)
    if norm2 <= 0.0:
        return psi
    scale = 1.0 / math.sqrt(norm2)
    return [z * scale for z in psi]


def wavefunction_from_hq_weights_signed(
    weights: List[Tuple[int, int, float]],
) -> List[complex]:
    """Signed amplitude superposition: weight may be ± mass (interference-carrying)."""
    psi = zero_wavefunction()
    for h, q, weight in weights:
        if abs(weight) <= 1e-30:
            continue
        amp = math.copysign(math.sqrt(abs(weight)), weight)
        psi[index_hq(h, q)] += complex(amp, 0.0)
    norm2 = sum(abs(z) ** 2 for z in psi)
    if norm2 <= 0.0:
        return psi
    scale = 1.0 / math.sqrt(norm2)
    return [z * scale for z in psi]


def wht_tensor2(v: list[float]) -> list[float]:
    """4096-point WHT^{⊗2} = WHT x WHT (row-column butterflies; no NumPy)."""
    if len(v) != 4096:
        raise ValueError("wht_tensor2 requires length 4096")
    mat: list[list[float]] = []
    it = iter(v)
    for _ in range(HORIZON_SIZE):
        row = [float(next(it)) for _ in range(HORIZON_SIZE)]
        mat.append(wht64_fast(row))
    out = [[0.0] * HORIZON_SIZE for _ in range(HORIZON_SIZE)]
    for c in range(HORIZON_SIZE):
        col = [mat[r][c] for r in range(HORIZON_SIZE)]
        col_t = wht64_fast(col)
        for r in range(HORIZON_SIZE):
            out[r][c] = col_t[r]
    flat: list[float] = []
    for r in range(HORIZON_SIZE):
        flat.extend(out[r])
    return flat


# ── Cyclic QFT primitives (operator/tensor class, SDK §11.2) ──
# Fourier object is the cyclic character of Z_N; δ_BU is operator-layer synthesis.

_CQFT64_TWIDDLES: list[complex] | None = None


def _delta_bu_twiddle(k: int, N: int) -> complex:
    """Cyclic character of Z_N: χ_k(n) = exp(2πi · k · n / N).

    Semantic Fourier object for strict order-finding: χ(1)^N = 1 and
    χ(a+b) = χ(a)χ(b). δ_BU is the non-Clifford monodromy resource used
    to synthesize these phases in the aQPU operator layer — it does not
    replace the 2π cyclic closure of the exponent group.

    Shor peak condition: k/Q ≈ m/r (continued fractions on k/Q).
    """
    if N <= 0:
        return complex(1.0, 0.0)
    kk = int(k) % int(N)
    angle = (2.0 * math.pi * float(kk)) / float(N)
    return complex(math.cos(angle), math.sin(angle))


def _ensure_cqft64_twiddles() -> list[complex]:
    global _CQFT64_TWIDDLES
    if _CQFT64_TWIDDLES is None:
        _CQFT64_TWIDDLES = [
            _delta_bu_twiddle(k * n, 64) for k in range(64) for n in range(64)
        ]
    return _CQFT64_TWIDDLES


def cqft64_fast(v: list[complex]) -> list[complex]:
    """Cyclic QFT on Z_64 via cyclic characters (δ_BU synthesis at operator layer)."""
    if len(v) != 64:
        raise ValueError("cqft64_fast requires length 64")
    tw = _ensure_cqft64_twiddles()
    out = [0j] * 64
    for k in range(64):
        base = k * 64
        out[k] = sum(v[n] * tw[base + n] for n in range(64))
    inv = 1.0 / 8.0
    return [z * inv for z in out]


def cqft(v: list[complex]) -> list[complex]:
    """Cyclic QFT on Z_64 only. Larger Q uses native sparse_cqft_peaks (native.c)."""
    if len(v) != 64:
        raise ValueError("Python cqft supports N=64 only; use bindings.sparse_cqft_peaks for scale")
    return cqft64_fast(v)


# ============================
# Multi-cell primitives
# ============================

_Q6_TO_BYTE_BY_FAMILY: list[list[int]] = [[0] * 64 for _ in range(4)]
_Q6_TO_BYTE_BY_FAMILY_VALID: list[list[bool]] = [[False] * 64 for _ in range(4)]
for _b in range(256):
    _q = int(q6(_b)) & 0x3F
    _fam = int(intron_family(byte_to_intron(_b))) & 0x3
    if not _Q6_TO_BYTE_BY_FAMILY_VALID[_fam][_q]:
        _Q6_TO_BYTE_BY_FAMILY[_fam][_q] = _b
        _Q6_TO_BYTE_BY_FAMILY_VALID[_fam][_q] = True


def byte_for_q6(q: int, family: int = 0) -> int:
    """Representative byte with given q6 and intron family (deterministic)."""
    qq = int(q) & 0x3F
    fam = int(family) & 0x3
    if not _Q6_TO_BYTE_BY_FAMILY_VALID[fam][qq]:
        raise RuntimeError(f"no byte for q6={qq} family={fam}")
    return int(_Q6_TO_BYTE_BY_FAMILY[fam][qq]) & 0xFF


def apply_S_each_cell(states: list[int]) -> list[int]:
    return [apply_gate_S(int(s)) for s in states]


def apply_C_each_cell(states: list[int]) -> list[int]:
    return [apply_gate_C(int(s)) for s in states]


def apply_F_each_cell(states: list[int]) -> list[int]:
    return [apply_gate_F(int(s)) for s in states]


def depth4_cycle(state24: int, byte: int) -> int:
    """Apply byte four times (b^4 = id) and return final state."""
    s = int(state24) & MASK_STATE24
    b = int(byte) & 0xFF
    for _ in range(4):
        s = step_state_by_byte(s, b)
    return s


@dataclass
class MultiCellRouter:
    """Multicell execution harness: per-cell stepping, K4 gates, χ-controlled bytes."""

    n_cells: int
    states: list[int] = field(default_factory=list)
    ledgers: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_cells <= 0:
            raise ValueError("n_cells must be >= 1")
        if not self.states:
            self.states = [int(GENE_MAC_REST)] * self.n_cells
        if len(self.states) != self.n_cells:
            raise ValueError("states length must equal n_cells")
        if not self.ledgers:
            self.ledgers = [[] for _ in range(self.n_cells)]
        if len(self.ledgers) != self.n_cells:
            raise ValueError("ledgers length must equal n_cells")

    def snapshot(self) -> list[int]:
        return [int(s) for s in self.states]

    def chirality_words(self) -> list[int]:
        return [int(chirality_word6(s)) & 0x3F for s in self.states]

    def shell_indices(self) -> list[int]:
        return [int(popcount(int(chirality_word6(s)) & 0x3F)) for s in self.states]

    def step_byte(self, cell: int, byte: int) -> int:
        c = int(cell)
        if not (0 <= c < self.n_cells):
            raise IndexError("cell out of range")
        b = int(byte) & 0xFF
        s0 = int(self.states[c]) & MASK_STATE24
        s1 = step_state_by_byte(s0, b)
        self.ledgers[c].append(b)
        self.states[c] = int(s1)
        return int(s1)

    def step_bytes(self, cell: int, payload: list[int]) -> int:
        s = int(self.states[cell])
        for b in payload:
            s = self.step_byte(cell, int(b))
        return int(s)

    def step_byte_all(self, bytes_by_cell: list[int]) -> list[int]:
        if len(bytes_by_cell) != self.n_cells:
            raise ValueError("bytes_by_cell length must equal n_cells")
        out = []
        for i, b in enumerate(bytes_by_cell):
            out.append(self.step_byte(i, int(b)))
        return out

    def apply_gate_all(self, gate: str) -> list[int]:
        g = str(gate).upper()
        if g == "S":
            self.states = apply_S_each_cell(self.states)
        elif g == "C":
            self.states = apply_C_each_cell(self.states)
        elif g == "F":
            self.states = apply_F_each_cell(self.states)
        elif g in ("ID", "I"):
            self.states = [int(s) for s in self.states]
        else:
            raise ValueError(f"unknown gate {gate!r}")
        return self.snapshot()

    def chirality_cnot_meas_ctrl(
        self,
        control_cell: int,
        target_cell: int,
        bit: int,
        *,
        family: int = 0,
    ) -> tuple[int, int]:
        """Measure χ bit on control; if 1, apply representative byte on target."""
        cc = int(control_cell)
        tc = int(target_cell)
        k = int(bit)
        if not (0 <= cc < self.n_cells and 0 <= tc < self.n_cells):
            raise IndexError("cell out of range")
        if not (0 <= k < 6):
            raise ValueError("bit must be in 0..5")
        chi_c = int(chirality_word6(self.states[cc])) & 0x3F
        ctrl = (chi_c >> k) & 1
        if ctrl:
            b = byte_for_q6(1 << k, family=family)
            self.step_byte(tc, b)
        return ctrl, int(self.states[tc])


__all__ = [
    "GENE_MAC_REST",
    "is_in_omega24",
    "step_byte",
    "step_byte_inverse",
    "q6",
    "chirality_word",
    "step_byte_stages",
    "horizon_distance",
    "ab_distance",
    "intron_to_byte",
    "byte_from_family_and_payload6",
    "residue_to_byte",
    "state_to_hq",
    "residue_to_chirality",
    "residue_to_chirality_kernel",
    "wht64",
    "wht64_dc_free",
    "wht_tensor2",
    "_delta_bu_twiddle",
    "DELTA_BU",
    "cqft64_fast",
    "cqft",
    "top_nontrivial_peaks",
    "zero_wavefunction",
    "index_hq",
    "split_index",
    "apply_k4",
    "apply_word",
    "word_index_permutation",
    "to_holographic",
    "from_holographic",
    "wavefunction_hq_spectral_peaks",
    "wavefunction_from_hq_weights",
    "wavefunction_from_hq_weights_signed",
    "byte_for_q6",
    "apply_S_each_cell",
    "apply_C_each_cell",
    "apply_F_each_cell",
    "depth4_cycle",
    "MultiCellRouter",
    "HORIZON_SIZE",
    "OMEGA_SIZE",
    "CHIRALITY_MASK_6",
    "K4_ID",
    "K4_W2",
    "K4_W2P",
    "K4_F",
]
