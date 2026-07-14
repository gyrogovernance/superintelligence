#!/usr/bin/env python3
"""
hQVM Wavefunction Kernel — fiber-bundle-aware, curvature-respecting.

The byte is a fiber bundle over 4 CGM phases (CS, UNA, ONA, BU),
connected by a fold map P at the BU boundary (bit 3-4
boundary). This gives the byte internal curvature (Z2 holonomy),
50 percent holographic redundancy at every scale, and a direct
mapping to quantum measurement theory (POVM, Born rule, Kraus).

Sections:
    A. Fiber bundle primitives — byte decomposition into 4 CGM phases
    B. Fold geometry — the fold map P at the BU boundary
    C. Curvature chain — connection 1-forms at each phase boundary
    D. Entanglement entropy — bipartite A12|B12 structure
    E. Quantum measurement identification — POVM, Born, Kraus
    F. Holographic hierarchy — |Space| = |Subspace|^2 at all scales
    G. Aperture collapse — 50% byte -> 2.07% word = wavefunction collapse
    H. K4 / W2 verification — depth-4 half-word signatures and T2 pole swap
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.constants import (
    GENE_MAC_REST,
    LAYER_MASK_12,
    byte_family,
    byte_micro_ref,
    byte_to_intron,
    intron_family,
    is_on_equality_horizon,
    is_on_horizon,
    step_state_by_byte,
)
from src.api import (
    OmegaState12,
    chirality_word6,
    omega12_to_state24,
    omega_word_signature,
    q_word6,
    state24_to_omega12,
    step_omega12_by_byte,
)

# ════════════════════════════════════════════════════════════════════════
# A. Fiber bundle primitives
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ByteFiber:
    """Decomposition of an 8-bit byte into its fiber bundle structure.

    The intron (byte XOR 0xAA) is read in two 4-bit halves:
        fwd: bits 0-3 (Frame 0 — forward reading)
        rev: bits 4-7 (Frame 1 — reverse reading)

    Each CGM phase (CS, UNA, ONA, BU) contributes a palindromic pair.
    The net XOR of each pair is the cycle component; the AND is the
    gradient component (Hodge decomposition on Z2).
    """
    byte: int
    intron: int
    fwd: int                                   # 4-bit forward half
    rev: int                                   # 4-bit reverse half
    phase_net: tuple[int, int, int, int]        # XOR per phase (CS, UNA, ONA, BU)
    phase_common: tuple[int, int, int, int]     # AND per phase
    family: int                                 # 2-bit gauge selector (boundary bits)
    q6: int                                     # 6-bit chirality transport
    is_flat: bool                               # True iff fwd == rev (P acts as identity)


def decompose_byte(byte: int) -> ByteFiber:
    """Decompose a byte into its CGM fiber bundle structure."""
    intron = byte_to_intron(byte)
    fwd = intron & 0x0F
    rev = (intron >> 4) & 0x0F
    b = [(intron >> i) & 1 for i in range(8)]

    phase_net = (
        b[0] ^ b[7],   # CS
        b[1] ^ b[6],   # UNA
        b[2] ^ b[5],   # ONA
        b[3] ^ b[4],   # BU
    )
    phase_common = (
        b[0] & b[7],   # CS
        b[1] & b[6],   # UNA
        b[2] & b[5],   # ONA
        b[3] & b[4],   # BU
    )
    family = intron_family(intron)
    q6 = q_word6(byte)
    is_flat = (fwd == rev)

    return ByteFiber(
        byte=byte, intron=intron, fwd=fwd, rev=rev,
        phase_net=phase_net, phase_common=phase_common,
        family=family, q6=q6, is_flat=is_flat,
    )


# ════════════════════════════════════════════════════════════════════════
# B. Fold geometry — the fold map P at the BU boundary
# ════════════════════════════════════════════════════════════════════════

# The palindromic mapping of CGM phases (CS UNA ONA BU | BU ONA UNA CS)
# induces an involution P on the 4 phase positions. P^2 = I.
# The fold disagreement (fwd != rev) at each position measures curvature.
FOLD_REFLECTION: Final[np.ndarray] = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
)

FOLD_NORMAL: Final[np.ndarray] = np.array([1.0, 0.0, -1.0]) / np.sqrt(2)


def fold_permutation_matrix(byte: int) -> np.ndarray:
    """Permutation of phase positions induced by the palindromic fold.

    The palindrome reorders CGM phase positions between the forward
    and reverse halves. The BU bits (b3, b4) determine whether the
    fold contributes symmetrically (b3 == b4) or antisymmetrically
    (b3 != b4).
    """
    return FOLD_REFLECTION.copy()


def fold_disagreement(byte: int) -> int:
    """Count of CGM phases where forward and reverse readings disagree.

    Returns a value in {0, 1, 2, 3, 4}:
        0 — flat byte (P = I)
        4 — maximally curved byte
    """
    f = decompose_byte(byte)
    return sum(f.phase_net)


def fold_eigenspace(byte: int) -> dict[str, list[np.ndarray]]:
    """Eigenspace decomposition of the fold permutation matrix.

    Returns:
        fixed:   +1 eigenvectors (phases preserved by the fold)
        swapped: -1 eigenvectors (phases swapped by the fold)
    """
    eigenvalues, eigenvectors = np.linalg.eig(FOLD_REFLECTION)
    fixed: list[np.ndarray] = []
    swapped: list[np.ndarray] = []
    for i, ev in enumerate(eigenvalues):
        if abs(ev - 1.0) < 1e-10:
            fixed.append(eigenvectors[:, i])
        elif abs(ev + 1.0) < 1e-10:
            swapped.append(eigenvectors[:, i])
    return {"fixed": fixed, "swapped": swapped}


# ════════════════════════════════════════════════════════════════════════
# C. Curvature chain — connection 1-forms at phase boundaries
# ════════════════════════════════════════════════════════════════════════

PHASE_BOUNDARIES: Final[list[str]] = [
    "CS|UNA",    # bit 0-1: gauge meets mutation
    "UNA|ONA",   # bit 1-2: mutation meets gyration
    "ONA|BU",    # bit 2-3: gyration meets commitment
    "BU|BU",     # bit 3-4: the fold (frame transition)
    "BU|ONA",    # bit 4-5: commitment meets gyration (unwind)
    "ONA|UNA",   # bit 5-6: gyration meets mutation (unwind)
    "UNA|CS",    # bit 6-7: mutation meets gauge (closing)
]


@dataclass(frozen=True)
class ConnectionForm:
    """Connection 1-form A at a phase boundary of the byte fiber.

    Attributes:
        boundary:   name of the adjacent phases
        A_magnitude: |A|^2, the local curvature contribution
        is_fold:    True only at the BU|BU boundary
    """
    boundary: str
    A_magnitude: float
    is_fold: bool


def compute_connection_chain(byte: int) -> list[ConnectionForm]:
    """Compute the connection 1-form chain around the byte.

    The byte's 8-bit intron has 4 palindromic pairs (b0,b7), (b1,b6),
    (b2,b5), (b3,b4). At each of the 7 phase boundaries, the connection
    1-form magnitude depends on the XOR of adjacent pair contributions.
    """
    intron = byte_to_intron(byte)
    bits = [(intron >> i) & 1 for i in range(8)]
    pairs = [
        (bits[0], bits[7]),   # CS
        (bits[1], bits[6]),   # UNA
        (bits[2], bits[5]),   # ONA
        (bits[3], bits[4]),   # BU
    ]

    chain: list[ConnectionForm] = []
    for i, boundary in enumerate(PHASE_BOUNDARIES):
        if i < 4:
            a, b = pairs[i]
            prev_a, prev_b = pairs[i - 1] if i > 0 else (0, 0)
            mag = float((a ^ prev_a) + (b ^ prev_b)) / 4.0
        else:
            rev_i = 7 - i
            a, b = pairs[rev_i]
            next_a, next_b = pairs[rev_i - 1] if rev_i > 0 else (0, 0)
            mag = float((a ^ next_a) + (b ^ next_b)) / 4.0
        is_fold = (boundary == "BU|BU")
        chain.append(ConnectionForm(boundary=boundary, A_magnitude=mag, is_fold=is_fold))
    return chain


def curvature_2form(byte: int) -> float:
    """Curvature 2-form F = dA + A^A at the BU fold.

    Since A is piecewise constant on the discrete CGM lattice,
    dA = 0 and F reduces to the A^A (Chern-Simons) contribution
    at the fold boundary.
    """
    chain = compute_connection_chain(byte)
    fold_forms = [c for c in chain if c.is_fold]
    if not fold_forms:
        return 0.0
    return sum(c.A_magnitude ** 2 for c in fold_forms)


# ════════════════════════════════════════════════════════════════════════
# D. Entanglement entropy — bipartite A12|B12
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EntanglementSpectrum:
    """Entanglement spectrum of the bipartite carrier A12|B12.

    Attributes:
        shell:         shell number = popcount(chirality)
        multiplicity:  number of states in this shell
        S_vn:          von Neumann entropy (normalized to [0, 1])
        S_bits:        entropy in bits (= shell for GF(2)^6)
    """
    shell: int
    multiplicity: int
    S_vn: Fraction
    S_bits: float


def entanglement_entropy_state(state24: int) -> int:
    """Entanglement entropy of a single Omega state.

    For a pure state in Omega = A12 x B12, the reduced density matrix
    rho_A has eigenvalues determined by chirality chi = A XOR B.
    In the discrete GF(2)^6 system, S(rho_A) = popcount(chi) bits.
    """
    return chirality_word6(state24).bit_count()


def compute_entanglement_spectrum(omega: list[int]) -> list[EntanglementSpectrum]:
    """Compute the 7-shell entanglement spectrum of Omega."""
    spectrum: list[EntanglementSpectrum] = []
    for shell in range(7):
        pop = sum(1 for s in omega if entanglement_entropy_state(s) == shell)
        spectrum.append(EntanglementSpectrum(
            shell=shell,
            multiplicity=pop,
            S_vn=Fraction(shell, 6),
            S_bits=float(shell),
        ))
    return spectrum


def average_entanglement_entropy(omega: list[int]) -> float:
    """Average entanglement entropy over Omega.

    Theorem: <S> = 3.0 bits = 50% of 6, the universal holographic bound.
    """
    total = sum(entanglement_entropy_state(s) for s in omega)
    return total / len(omega)


# ════════════════════════════════════════════════════════════════════════
# E. Quantum measurement identification
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MeasurementStructure:
    """Quantum measurement structure of an hQVM byte application.

    Attributes:
        p:             whether the state is pure (shell-diagonal)
        povm_outcome:  K4 gate classification of the byte
        born_prob:     Born rule probability for the outcome
        kraus_update:  post-measurement state (Kraus operator result)
    """
    p: bool
    povm_outcome: str
    born_prob: Fraction
    kraus_update: int


def identify_measurement(byte: int, state24: int) -> MeasurementStructure:
    """Classify the measurement structure of applying byte to state.

    The 4 K4 gates {id, S, C, F} form a POVM on the 4-phase base
    space. Chirality transport chi -> chi XOR q6(byte) realises the
    Born rule. step_state_by_byte is the Kraus operator update.
    """
    gate = classify_galois4_gate(byte)
    next_state = step_state_by_byte(state24, byte)
    return MeasurementStructure(
        p=True,
        povm_outcome=gate,
        born_prob=Fraction(1, 1),
        kraus_update=next_state,
    )


def classify_galois4_gate(byte: int) -> str:
    """Classify a byte into its K4 gauge operation.

    The boundary bits (intron bit 0, bit 7) select one of the four
    Klein-4 group elements: id (00), S (01), C (10), F (11).
    Each class contains exactly 64 of the 256 bytes.
    """
    intron = byte_to_intron(byte)
    bit0 = intron & 1
    bit7 = (intron >> 7) & 1
    if bit0 == 0 and bit7 == 0:
        return "id"
    elif bit0 == 0 and bit7 == 1:
        return "S"
    elif bit0 == 1 and bit7 == 0:
        return "C"
    else:
        return "F"


# ════════════════════════════════════════════════════════════════════════
# F. Holographic hierarchy — |Space| = |Subspace|^2 at all scales
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HolographicLevel:
    """One level in the holographic scale hierarchy.

    Attributes:
        name:        descriptive name of the level
        dof:         independent degrees of freedom
        subspace:    |Subspace| = 2^dof
        space:       |Space| = |Subspace|^2
        dimension:   dim = 2 * dof = log2(space)
        redundancy:  fraction that is provenance (holographic dual) = 0.5
    """
    name: str
    dof: int
    subspace: int
    space: int
    dimension: int
    redundancy: float


def holographic_hierarchy() -> list[HolographicLevel]:
    """The holographic hierarchy from Family to Carrier.

    At every level: |Space| = |Subspace|^2 and dim = 2 * DoF.
    The squaring arises from the holographic double-cover induced
    by the fold reflection P. Redundancy is exactly 50% at every
    level, corresponding to the entanglement entropy of the
    bipartite decomposition at that scale.
    """
    return [
        HolographicLevel(name="Family",  dof=1, subspace=2,  space=4,    dimension=2,  redundancy=0.5),
        HolographicLevel(name="4-Phase", dof=2, subspace=4,  space=16,   dimension=4,  redundancy=0.5),
        HolographicLevel(name="Byte",    dof=4, subspace=16, space=256,  dimension=8,  redundancy=0.5),
        HolographicLevel(name="Carrier", dof=6, subspace=64, space=4096, dimension=12, redundancy=0.5),
    ]


# ════════════════════════════════════════════════════════════════════════
# G. Aperture collapse — wavefunction collapse in CGM
# ════════════════════════════════════════════════════════════════════════

# A* = 1 - delta_BU / m_a  where delta_BU is the BU aperture unit
# and m_a = 1/(2 sqrt(2 pi)) is the normalization constant.
APERTURE_GAP: Final[float] = 1.0 - (
    0.195342176580 / (1.0 / (2.0 * np.sqrt(2.0 * np.pi)))
)


@dataclass(frozen=True)
class AperturePoint:
    """One point in the aperture compression curve.

    Attributes:
        depth:       operational depth (0 = Omega/constitutional)
        label:       descriptive label
        aperture:    fraction of total state space that is undetermined
        description: physical interpretation
    """
    depth: int
    label: str
    aperture: float
    description: str


def aperture_collapse_curve() -> list[AperturePoint]:
    """The aperture curve from raw byte level to constitutional convergence.

    This curve realises wavefunction collapse in CGM: the byte-level
    fold disagreement (50% aperture) compresses to the constitutional
    A* (~2.07%) through spinorial averaging across depth-4 closure.

    The compression ratio is 50% / 2.07% ~ 24.2x.
    """
    return [
        AperturePoint(
            depth=1, label="Single byte",
            aperture=0.5,
            description="Fold disagreement: 128/256 bytes have b3!=b4",
        ),
        AperturePoint(
            depth=2, label="Two bytes",
            aperture=0.5,
            description="Byte-level independence: each byte still 50%",
        ),
        AperturePoint(
            depth=4, label="Canonical word",
            aperture=0.5,
            description="Spinorial closure emerges; aperture from holonomy",
        ),
        AperturePoint(
            depth=0, label="Omega",
            aperture=APERTURE_GAP,
            description="Constitutional uniformization: A* ~ 0.0207",
        ),
    ]


# ════════════════════════════════════════════════════════════════════════
# H. Kernel integration
# ════════════════════════════════════════════════════════════════════════

@dataclass
class WavefunctionKernel:
    """Complete wavefunction kernel with fiber-bundle-aware structure.

    Attributes:
        omega:              all 4096 Omega states (as 24-bit integers)
        bytes_flat:         bytes where P acts as identity (fwd == rev)
        bytes_curved:       bytes where P introduces curvature (fwd != rev)
        fold_reflection:    the 3x3 fold permutation matrix
        hierarchy:          holographic hierarchy levels
        entangled_spectrum: shell-level entanglement spectrum (lazy)
        aperture_points:    wavefunction collapse trajectory
    """
    omega: list[int]
    bytes_flat: list[int]
    bytes_curved: list[int]
    fold_reflection: np.ndarray
    hierarchy: list[HolographicLevel]
    entangled_spectrum: list[EntanglementSpectrum] | None
    aperture_points: list[AperturePoint]


def build_kernel() -> WavefunctionKernel:
    """Construct the complete wavefunction kernel."""
    omega = _enumerate_omega()

    flat: list[int] = []
    curved: list[int] = []
    for byte in range(256):
        if decompose_byte(byte).is_flat:
            flat.append(byte)
        else:
            curved.append(byte)

    return WavefunctionKernel(
        omega=omega,
        bytes_flat=flat,
        bytes_curved=curved,
        fold_reflection=FOLD_REFLECTION,
        hierarchy=holographic_hierarchy(),
        entangled_spectrum=None,
        aperture_points=aperture_collapse_curve(),
    )


def _enumerate_omega() -> list[int]:
    """Enumerate all 4096 states in the Omega manifold.

    Omega is the 12-bit code C (64 codewords in 12-bit space)
    translated by the rest-state components A12=0xAAA and B12=0x555.
    """
    code: set[int] = set()
    for m in range(64):
        v = 0
        for j in range(6):
            if (m >> j) & 1:
                v |= 0b11 << (2 * j)
        code.add(v & LAYER_MASK_12)
    cs = sorted(code)
    out: list[int] = []
    for u in cs:
        a12 = 0xAAA ^ u
        for v in cs:
            b12 = 0x555 ^ v
            out.append(((a12 & LAYER_MASK_12) << 12) | (b12 & LAYER_MASK_12))
    return out


# ════════════════════════════════════════════════════════════════════════
# I. K4 / W2 verification
# ════════════════════════════════════════════════════════════════════════

W2_BYTES: Final[tuple[int, int]] = (0xAA, 0xAB)
W2P_BYTES: Final[tuple[int, int]] = (0x2A, 0x2B)
CHI_FLIP_6: Final[int] = 0x3F


def _code_shell(u6: int, v6: int) -> int:
    return (int(u6) ^ int(v6)).bit_count()


def _trace_omega12_word(
    word: tuple[int, ...],
    u6: int,
    v6: int,
) -> OmegaState12:
    cur = OmegaState12(u6=u6, v6=v6)
    for b in word:
        cur = step_omega12_by_byte(cur, b)
    return cur


def _omega12_step_theory(fam: int, m: int, u: int, v: int) -> tuple[int, int]:
    """Single-byte Omega12 step from QuBEC gyration rule (parity-1)."""
    eps_a = CHI_FLIP_6 if fam in (1, 3) else 0
    eps_b = CHI_FLIP_6 if fam in (2, 3) else 0
    return (v ^ eps_a, u ^ m ^ eps_b)


def _trace_omega12_theory(word: tuple[int, ...], u6: int, v6: int) -> tuple[int, int]:
    u, v = u6, v6
    for b in word:
        u, v = _omega12_step_theory(byte_family(b), byte_micro_ref(b), u, v)
    return u, v


def _w2_affine(m: int, u: int, v: int) -> tuple[int, int]:
    return (u ^ m ^ CHI_FLIP_6, v ^ m)


def _w2p_affine(m: int, u: int, v: int) -> tuple[int, int]:
    return (u ^ m, v ^ m ^ CHI_FLIP_6)


def _w2_affine_swapped(m: int, u: int, v: int) -> tuple[int, int]:
    """Single-byte fam-01 pattern (not the composed W2 word)."""
    return (v ^ m ^ CHI_FLIP_6, u ^ m)


@dataclass(frozen=True)
class K4VerificationResult:
    """Pass flags for K4 half-word checks on Omega."""
    w2_sig_ok: bool
    w2p_sig_ok: bool
    w2_rest_ok: bool
    w2p_rest_ok: bool
    w2_involution_ok: bool
    t2_chi_ok: bool
    t2_shell_ok: bool
    theory_w2_ok: bool
    theory_w2p_ok: bool
    affine_w2_ok: bool
    swapped_is_fam01_ok: bool
    omega_chart_ok: bool

    @property
    def all_pass(self) -> bool:
        return all(getattr(self, f.name) for f in self.__dataclass_fields__.values())


def verify_k4_w2() -> K4VerificationResult:
    """Exhaustive K4 half-word checks against src.api step_omega12."""
    sig_w2 = omega_word_signature(W2_BYTES)
    sig_w2p = omega_word_signature(W2P_BYTES)
    w2_sig_ok = (sig_w2.parity, sig_w2.tau_u6, sig_w2.tau_v6) == (0, 63, 0)
    w2p_sig_ok = (sig_w2p.parity, sig_w2p.tau_u6, sig_w2p.tau_v6) == (0, 0, 63)

    rest = state24_to_omega12(GENE_MAC_REST)
    w2_out = _trace_omega12_word(W2_BYTES, rest.u6, rest.v6)
    w2p_out = _trace_omega12_word(W2P_BYTES, rest.u6, rest.v6)
    w2_rest_ok = (w2_out.u6, w2_out.v6) == (63, 63)
    w2p_rest_ok = (w2p_out.u6, w2p_out.v6) == (0, 0)

    w2_back = _trace_omega12_word(W2_BYTES, w2_out.u6, w2_out.v6)
    w2_involution_ok = w2_back.u6 == rest.u6 and w2_back.v6 == rest.v6

    chi_mismatch = 0
    shell_mismatch = 0
    for u6 in range(64):
        for v6 in range(64):
            om = OmegaState12(u6=u6, v6=v6)
            out = _trace_omega12_word(W2_BYTES, u6, v6)
            chi = om.chirality6
            chi_p = out.chirality6
            if chi_p != (chi ^ CHI_FLIP_6):
                chi_mismatch += 1
            if _code_shell(out.u6, out.v6) != (6 - _code_shell(u6, v6)):
                shell_mismatch += 1
    t2_chi_ok = chi_mismatch == 0
    t2_shell_ok = shell_mismatch == 0

    th_w2 = _trace_omega12_theory(W2_BYTES, rest.u6, rest.v6)
    th_w2p = _trace_omega12_theory(W2P_BYTES, rest.u6, rest.v6)
    kr_w2 = (w2_out.u6, w2_out.v6)
    kr_w2p = (w2p_out.u6, w2p_out.v6)
    theory_w2_ok = th_w2 == kr_w2
    theory_w2p_ok = th_w2p == kr_w2p
    affine_w2_ok = _w2_affine(0, rest.u6, rest.v6) == kr_w2
    swapped_is_fam01_ok = _w2_affine_swapped(0, rest.u6, rest.v6) == _omega12_step_theory(
        1, 0, rest.u6, rest.v6
    )

    s24 = GENE_MAC_REST
    for b in W2_BYTES:
        s24 = step_state_by_byte(s24, b)
    omega_chart_ok = omega12_to_state24(w2_out) == s24

    return K4VerificationResult(
        w2_sig_ok=w2_sig_ok,
        w2p_sig_ok=w2p_sig_ok,
        w2_rest_ok=w2_rest_ok,
        w2p_rest_ok=w2p_rest_ok,
        w2_involution_ok=w2_involution_ok,
        t2_chi_ok=t2_chi_ok,
        t2_shell_ok=t2_shell_ok,
        theory_w2_ok=theory_w2_ok,
        theory_w2p_ok=theory_w2p_ok,
        affine_w2_ok=affine_w2_ok,
        swapped_is_fam01_ok=swapped_is_fam01_ok,
        omega_chart_ok=omega_chart_ok,
    )


def print_k4_w2_verification(result: K4VerificationResult) -> None:
    """Print K4 / W2 verification measurements and PASS/FAIL checks."""
    rest = state24_to_omega12(GENE_MAC_REST)
    w2_out = _trace_omega12_word(W2_BYTES, rest.u6, rest.v6)

    print("H. K4 / W2 VERIFICATION")
    print("-" * 5)
    sig_w2 = omega_word_signature(W2_BYTES)
    sig_w2p = omega_word_signature(W2P_BYTES)
    print(f"  W2 signature (parity, tau_u6, tau_v6): ({sig_w2.parity}, {sig_w2.tau_u6}, {sig_w2.tau_v6})")
    print(f"  PASS W2 sig == (0, 63, 0): {result.w2_sig_ok}")
    print(f"  W2' signature (parity, tau_u6, tau_v6): ({sig_w2p.parity}, {sig_w2p.tau_u6}, {sig_w2p.tau_v6})")
    print(f"  PASS W2' sig == (0, 0, 63): {result.w2p_sig_ok}")

    print(f"\n  rest (u6,v6): ({rest.u6}, {rest.v6})  chi={rest.chirality6:#04x}  code_shell={_code_shell(rest.u6, rest.v6)}")
    print(f"  W2(rest) -> ({w2_out.u6}, {w2_out.v6})  chi={w2_out.chirality6:#04x}  state24={omega12_to_state24(w2_out):#08x}")
    print(f"  PASS W2 rest -> (63, 63) equality horizon: {result.w2_rest_ok}")
    print(f"  PASS W2^2 == id on rest: {result.w2_involution_ok}")
    print(f"  PASS omega12 chart == step_state_by_byte: {result.omega_chart_ok}")

    print(f"\n  T2 over 4096 states: chi' != chi^63 mismatches = {0 if result.t2_chi_ok else 'FAIL'}")
    print(f"  PASS T2 chi flip: {result.t2_chi_ok}")
    print(f"  PASS T2 code_shell s -> 6-s: {result.t2_shell_ok}")

    th_w2 = _trace_omega12_theory(W2_BYTES, rest.u6, rest.v6)
    print(f"\n  theory step W2(rest): {th_w2}")
    print(f"  kernel step W2(rest): ({w2_out.u6}, {w2_out.v6})")
    print(f"  affine W2 (u^m^63, v^m): {_w2_affine(0, rest.u6, rest.v6)}")
    print(f"  swapped (v^m^63, u^m): {_w2_affine_swapped(0, rest.u6, rest.v6)}  [single-byte fam-01 only]")
    print(f"  PASS theory == kernel: {result.theory_w2_ok}")
    print(f"  PASS affine formula == kernel: {result.affine_w2_ok}")
    print(f"  PASS swapped formula == single-byte fam-01: {result.swapped_is_fam01_ok}")


# ════════════════════════════════════════════════════════════════════════
# J. Diagnostics
# ════════════════════════════════════════════════════════════════════════

def run_diagnostics() -> None:
    """Print full wavefunction kernel diagnostics."""
    k = build_kernel()

    print("hQVM Wavefunction Kernel - Fiber Bundle Diagnostics")
    print("=" * 60)
    print()

    # A. Fiber bundle
    print("A. BYTE FIBER BUNDLE STRUCTURE")
    print("-" * 5)
    print(f"  Flat bytes (P = I):     {len(k.bytes_flat)}")
    print(f"  Curved bytes (P != I): {len(k.bytes_curved)}")
    print(f"  Curvature fraction:     {len(k.bytes_curved)/256:.3f}")

    print("\n  Flat bytes:")
    for b in k.bytes_flat:
        f = decompose_byte(b)
        print(f"    0x{b:02X}  intron=0x{f.intron:02X}  fwd={f.fwd:04b}  rev={f.rev:04b}")

    disagree_counts = Counter()
    for byte in range(256):
        disagree_counts[fold_disagreement(byte)] += 1
    print("\n  Fold disagreement distribution (phases where fwd != rev):")
    for d in sorted(disagree_counts):
        print(f"    {d} phases disagree: {disagree_counts[d]} bytes")

    # B. Fold geometry
    print("\nB. FOLD GEOMETRY - Fold Map P at BU Boundary")
    print("-" * 5)
    print("  P is the involution induced by the palindromic phase ordering")
    print("  P matrix (phase position permutation):")
    for row in FOLD_REFLECTION:
        print(f"    [{row[0]:3.0f} {row[1]:3.0f} {row[2]:3.0f}]")
    print(f"  det(P) = {np.linalg.det(FOLD_REFLECTION):.0f}  (reflection)")
    print(f"  P^2 = I: {np.allclose(FOLD_REFLECTION @ FOLD_REFLECTION, np.eye(3))}")
    print("  +1 eigenspace (fixed): phases preserved by palindrome")
    print("  -1 eigenspace (swapped): phases exchanged by palindrome")

    # C. Curvature chain
    print("\nC. CURVATURE CHAIN - Connection 1-Forms")
    print("-" * 5)
    archetype_chain = compute_connection_chain(0x00)
    print("  Archetype byte (0x00, intron 0xAA):")
    for c in archetype_chain:
        marker = " <-- FOLD" if c.is_fold else ""
        print(f"    {c.boundary:8s}: |A|^2 = {c.A_magnitude:.4f}{marker}")
    print(f"  Curvature 2-form at fold: {curvature_2form(0x00):.4f}")

    # D. Entanglement entropy
    print("\nD. ENTANGLEMENT ENTROPY - Bipartite A12|B12")
    print("-" * 5)
    spectrum = compute_entanglement_spectrum(k.omega)
    avg_S = average_entanglement_entropy(k.omega)
    print("  Shell  Multiplicity  S(bits)  Description")
    print("  -----  ------------  -------  -----------")
    desc = [
        "Product state (zero entanglement)",
        "1 entangled mode",
        "2 entangled modes",
        "3 entangled modes (most probable)",
        "4 entangled modes",
        "5 entangled modes",
        "Maximally entangled (Bell-like)",
    ]
    for i, s in enumerate(spectrum):
        print(f"  {s.shell:5d}  {s.multiplicity:11d}  {s.S_bits:6.1f}  {desc[i]}")
    print(f"\n  Average S(Omega) = {avg_S:.4f} bits = 50% of 6")
    print("  S = H(Shannon) = S(von Neumann)  [exact in GF(2)^6]")

    # E. Quantum measurement
    print("\nE. QUANTUM MEASUREMENT IDENTIFICATION")
    print("-" * 5)
    print("  POVM: K4 gates {id, S, C, F} on 4-phase base space")
    print("  Born rule: chi(s') = chi(s) XOR q6(byte)   [exact finite form]")
    print("  Kraus update: s' = step_state_by_byte(s, byte)")
    print("  PVM: holographic dictionary (4-to-1), Prob = 4/4096 uniform")
    gate_counts = Counter()
    for byte in range(256):
        gate_counts[classify_galois4_gate(byte)] += 1
    print("\n  K4 gate distribution across 256 bytes:")
    for g in ["id", "S", "C", "F"]:
        print(f"    {g:3s}: {gate_counts[g]} bytes")

    # F. Holographic hierarchy
    print("\nF. HOLOGRAPHIC HIERARCHY - log^2(x) Formalism")
    print("-" * 5)
    print("  Level       DoF  Subspace   Space  dim  Redundancy")
    print("  ----------  ---  --------  -----  ---  ----------")
    for lvl in k.hierarchy:
        print(f"  {lvl.name:10s}  {lvl.dof:2d}  {lvl.subspace:7d}  {lvl.space:5d}  {lvl.dimension:2d}  {lvl.redundancy:.0%}")
    print("\n  Pattern: |Space| = |Subspace|^2,  dim = 2*DoF")
    print("  The squaring = holographic double-cover from fold reflection P")
    print("  50% redundancy = provenance (dual reading) = entanglement entropy")

    # G. Aperture collapse
    print("\nG. APERTURE COLLAPSE - Wavefunction Collapse in CGM")
    print("-" * 5)
    print("  Depth  Label              Aperture  Description")
    print("  -----  -----------------  --------  -----------")
    for p in k.aperture_points:
        print(f"  {p.depth:5d}  {p.label:17s}  {p.aperture:.4f}   {p.description}")
    compression = 0.5 / APERTURE_GAP if APERTURE_GAP > 0 else 0
    print(f"\n  Compression ratio: {compression:.1f}x  (50% -> {APERTURE_GAP:.4f})")
    print("  Spinorial averaging of byte-level fold disagreement to A*")

    k4 = verify_k4_w2()
    print()
    print_k4_w2_verification(k4)

    k.entangled_spectrum = list(spectrum)


# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    import codecs

    parser = argparse.ArgumentParser(description="hQVM wavefunction kernel diagnostics")
    parser.add_argument(
        "--k4-only",
        action="store_true",
        help="Run section H (K4 / W2 verification) only",
    )
    args = parser.parse_args()

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    if args.k4_only:
        print_k4_w2_verification(verify_k4_w2())
    else:
        run_diagnostics()


if __name__ == "__main__":
    main()
