# src/api.py
"""
Gyroscopic ASI aQPU Kernel API: precomputed tables, integrity checks, and derived functions.

Built from theoretical constants in src.constants. No HTTP or REST;
this module provides mask lookup, dual code and syndromes, trajectory
parity commitments, depth-4 projections, and derived measurements.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Final, Union

import numpy as np

from src.constants import (
    CHIRALITY_MASK_6,
    CHIRALITY_QUBITS_6,
    EPSILON_6,
    GENE_MAC_A12,
    LAYER_MASK_12,
    apply_gate,
    byte_to_intron,
    dot12,
    expand_intron_to_mask12,
    intron_family,
    intron_micro_ref,
    pack_state,
    step_state_by_byte,
    unpack_state,
)

# ----------------------------------------
# Precomputed per-byte tables
# ----------------------------------------

INTRON_BY_BYTE: tuple[int, ...] = tuple(
    byte_to_intron(b) for b in range(256)
)
FAMILY_BY_BYTE: tuple[int, ...] = tuple(
    intron_family(i) for i in INTRON_BY_BYTE
)
MICRO_REF_BY_BYTE: tuple[int, ...] = tuple(
    intron_micro_ref(i) for i in INTRON_BY_BYTE
)
MASK12_BY_BYTE: tuple[int, ...] = tuple(
    expand_intron_to_mask12(i) for i in INTRON_BY_BYTE
)
EPS_A6_BY_BYTE: tuple[int, ...] = tuple(
    EPSILON_6 if (INTRON_BY_BYTE[b] & 0x01) else 0
    for b in range(256)
)
EPS_B6_BY_BYTE: tuple[int, ...] = tuple(
    EPSILON_6 if (INTRON_BY_BYTE[b] & 0x80) else 0
    for b in range(256)
)


def mask12_for_byte(byte: int) -> int:
    """12-bit Type-A mask for a given byte."""
    return MASK12_BY_BYTE[int(byte) & 0xFF]


# ----------------------------------------
# Canonical 6-qubit / pair-diagonal helpers
# ----------------------------------------


def is_pair_diagonal12(word12: int) -> bool:
    """True iff each 2-bit pair is 00 or 11."""
    x = int(word12) & LAYER_MASK_12
    for i in range(CHIRALITY_QUBITS_6):
        pair = (x >> (2 * i)) & 0x3
        if pair not in (0x0, 0x3):
            return False
    return True


def pairdiag12_to_word6(word12: int) -> int:
    """
    Collapse pair-diagonal 12-bit word to 6-bit word.
    Each pair 00 -> 0, 11 -> 1.
    """
    x = int(word12) & LAYER_MASK_12
    out = 0
    for i in range(CHIRALITY_QUBITS_6):
        pair = (x >> (2 * i)) & 0x3
        if pair == 0x3:
            out |= 1 << i
        elif pair != 0x0:
            raise ValueError(f"Not pair-diagonal: {word12:#05x}")
    return out & CHIRALITY_MASK_6


def word6_to_pairdiag12(word6: int) -> int:
    """Expand 6-bit word to pair-diagonal 12-bit word: bit i -> pair (2i,2i+1)."""
    x = int(word6) & CHIRALITY_MASK_6
    out = 0
    for i in range(CHIRALITY_QUBITS_6):
        if (x >> i) & 1:
            out |= 0x3 << (2 * i)
    return out & LAYER_MASK_12


def component12_to_spin6(component12: int) -> tuple[int, ...]:
    """
    Convert a 12-bit component in Omega-style pair encoding to 6 spins in {-1,+1}.
    Pair 10 -> +1, pair 01 -> -1.
    Raises on non-spin-pair states.
    """
    x = int(component12) & LAYER_MASK_12
    spins: list[int] = []
    for i in range(CHIRALITY_QUBITS_6):
        pair = (x >> (2 * i)) & 0x3
        if pair == 0x2:
            spins.append(+1)
        elif pair == 0x1:
            spins.append(-1)
        else:
            raise ValueError(
                f"Component is not in spin-pair form: {component12:#05x}"
            )
    return tuple(spins)


def spin6_to_component12(spins: Iterable[int]) -> int:
    """
    Convert 6 spins in {-1,+1} to 12-bit component.
    +1 -> pair 10, -1 -> pair 01.
    """
    vals = tuple(int(v) for v in spins)
    if len(vals) != CHIRALITY_QUBITS_6:
        raise ValueError(f"Expected 6 spins, got {len(vals)}")
    out = 0
    for i, v in enumerate(vals):
        if v == +1:
            out |= 0x2 << (2 * i)
        elif v == -1:
            out |= 0x1 << (2 * i)
        else:
            raise ValueError(f"Spin must be +/-1, got {v}")
    return out & LAYER_MASK_12


def state24_to_spin6_pair(
    state24: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return (spin_A6, spin_B6) for a 24-bit state."""
    a12, b12 = unpack_state(state24)
    return component12_to_spin6(a12), component12_to_spin6(b12)


# ----------------------------------------
# Dual code and syndromes
# ----------------------------------------


def _compute_c_perp() -> tuple[int, ...]:
    """Dual code: all 12-bit vectors orthogonal to every mask codeword."""
    masks = set(MASK12_BY_BYTE)
    return tuple(
        s
        for s in range(1 << 12)
        if all(dot12(s, c) == 0 for c in masks)
    )


C_PERP_12: tuple[int, ...] = _compute_c_perp()
assert len(C_PERP_12) == 64, (
    f"Expected |C_PERP_12|=64, got {len(C_PERP_12)}"
)


def syndrome_is_valid_mask(m12: int) -> bool:
    """True if m12 has zero syndrome with respect to C_perp."""
    return all(dot12(m12, v) == 0 for v in C_PERP_12)


def syndrome_detects_corruption(m12: int) -> bool:
    """True if m12 has non-zero syndrome."""
    return any(dot12(m12, v) == 1 for v in C_PERP_12)


def mask12_syndrome(m12: int) -> int:
    """Syndrome bitmap for m12 with respect to C_perp."""
    syndrome = 0
    for i, v in enumerate(C_PERP_12):
        if dot12(m12, v) == 1:
            syndrome |= 1 << i
    return syndrome


# ----------------------------------------
# Trajectory parity commitments
# ----------------------------------------

ByteItem = Union[int, bytes, bytearray, memoryview]


def trajectory_parity_commitment(
    items: Iterable[ByteItem],
) -> tuple[int, int, int]:
    """
    Algebraic integrity check for trajectory histories.
    Returns (O, E, parity): XOR of masks at even/odd positions.
    """
    O = E = 0
    idx = 0
    for item in items:
        if isinstance(item, (bytes, bytearray, memoryview)):
            for b in item:
                m = MASK12_BY_BYTE[int(b) & 0xFF]
                if (idx & 1) == 0:
                    O ^= m
                else:
                    E ^= m
                idx += 1
        else:
            m = MASK12_BY_BYTE[int(item) & 0xFF]
            if (idx & 1) == 0:
                O ^= m
            else:
                E ^= m
            idx += 1
    return (O & LAYER_MASK_12, E & LAYER_MASK_12, idx & 1)


def trajectory_commitment_bytes(O: int, E: int, parity: int) -> bytes:
    """Encode parity commitment as 5-byte representation."""
    return (
        (O & LAYER_MASK_12).to_bytes(2, "big")
        + (E & LAYER_MASK_12).to_bytes(2, "big")
        + bytes([parity & 1])
    )


# ----------------------------------------
# Depth-4 projections
# ----------------------------------------


def depth4_mask_projection48(b0: int, b1: int, b2: int, b3: int) -> int:
    """48-bit: 4 x 12-bit masks packed big-endian."""
    return (
        (mask12_for_byte(b0) << 36)
        | (mask12_for_byte(b1) << 24)
        | (mask12_for_byte(b2) << 12)
        | mask12_for_byte(b3)
    )


def depth4_intron_sequence32(b0: int, b1: int, b2: int, b3: int) -> int:
    """32-bit: 4 x 8-bit introns packed big-endian (bijective)."""
    return (
        (byte_to_intron(b0) << 24)
        | (byte_to_intron(b1) << 16)
        | (byte_to_intron(b2) << 8)
        | byte_to_intron(b3)
    )


# ----------------------------------------
# Derived measurements
# ----------------------------------------


def so3_shadow_count(state24: int) -> int:
    """
    Count distinct 24-bit next states reachable by all 256 bytes.
    Should be 128 from any state (SO(3)/SU(2) 2-to-1 projection).
    """
    seen = set()
    for b in range(256):
        seen.add(step_state_by_byte(state24, b))
    return len(seen)


# ----------------------------------------
# Chirality register (6-bit) and q-word
# ----------------------------------------


def chirality_word6(state24: int) -> int:
    """
    6-bit chirality register: one bit per dipole from (A^B).
    Pair 00 -> 0, pair 11 -> 1. Spectrum has antipodal poles and binomial latitude.
    """
    a12, b12 = unpack_state(state24)
    diff = (a12 ^ b12) & LAYER_MASK_12
    out = 0
    for i in range(6):
        pair = (diff >> (2 * i)) & 0x3
        if pair == 0x3:
            out |= 1 << i
    return out


def q_word6(byte: int) -> int:
    """
    6-bit q-word from byte: C64 codeword (mask or mask^0xFFF by L0).
    Satisfies chirality_word6(step(s,b)) == chirality_word6(s) ^ q_word6(b) on Omega.
    """
    b = int(byte) & 0xFF
    intron = byte_to_intron(b)
    l0 = (intron & 1) ^ ((intron >> 7) & 1)
    q12 = mask12_for_byte(b) ^ (LAYER_MASK_12 if l0 else 0)
    q12 &= LAYER_MASK_12
    out = 0
    for i in range(6):
        pair = (q12 >> (2 * i)) & 0x3
        if pair == 0x3:
            out |= 1 << i
    return out


def q_word12(byte: int) -> int:
    """
    12-bit q-word before 6-bit collapse:
      q12 = mask12(byte) XOR (0xFFF if L0 parity odd else 0)
    This is the commutation-class representative in C64.
    """
    b = int(byte) & 0xFF
    intron = byte_to_intron(b)
    l0 = (intron & 1) ^ ((intron >> 7) & 1)
    q12 = mask12_for_byte(b) ^ (LAYER_MASK_12 if l0 else 0)
    return q12 & LAYER_MASK_12


BYTES_BY_Q6: tuple[tuple[int, ...], ...] = tuple(
    tuple(b for b in range(256) if q_word6(b) == q6) for q6 in range(64)
)
Q_KERNEL_BYTES: tuple[int, ...] = BYTES_BY_Q6[0]
Q_WEIGHT_BY_BYTE: tuple[int, ...] = tuple(
    q_word6(b).bit_count() for b in range(256)
)


def q_word6_for_items(items: Iterable[ByteItem]) -> int:
    """
    Accumulated chirality transport for a byte word.
    For a word b1...bn:
      chi_out = chi_in XOR q_word6_for_items(word)
    on Omega.
    """
    q = 0
    for item in items:
        if isinstance(item, (bytes, bytearray, memoryview)):
            for b in item:
                q ^= q_word6(b)
        else:
            q ^= q_word6(int(item) & 0xFF)
    return q & CHIRALITY_MASK_6


def chirality_distance6(s1: int, s2: int) -> int:
    """Hamming distance between 6-bit chirality words of two states."""
    return (chirality_word6(s1) ^ chirality_word6(s2)).bit_count()


# ----------------------------------------
# Word operator signatures
# ----------------------------------------


@dataclass(frozen=True)
class WordSignature:
    """
    Canonical affine signature of a byte word on the 24-bit carrier.

    parity = 0 -> identity linear part
    parity = 1 -> swap linear part

    tau_a12, tau_b12 are the translation parts.
    """

    parity: int
    tau_a12: int
    tau_b12: int

    @property
    def tau_a6(self) -> int:
        return pairdiag12_to_word6(self.tau_a12)

    @property
    def tau_b6(self) -> int:
        return pairdiag12_to_word6(self.tau_b12)


def word_signature(items: Iterable[ByteItem]) -> WordSignature:
    """
    Signature of a byte word.
    Since each byte has swap linear part, parity is word length mod 2.
    Translation is the image of (0,0) under the word.
    """
    s = pack_state(0, 0)
    n = 0
    for item in items:
        if isinstance(item, (bytes, bytearray, memoryview)):
            for b in item:
                s = step_state_by_byte(s, b)
                n += 1
        else:
            s = step_state_by_byte(s, int(item) & 0xFF)
            n += 1
    tau_a12, tau_b12 = unpack_state(s)
    return WordSignature(
        parity=n & 1,
        tau_a12=tau_a12,
        tau_b12=tau_b12,
    )


def apply_word_signature(state24: int, sig: WordSignature) -> int:
    """Apply a signature directly without replaying bytes."""
    a12, b12 = unpack_state(state24)
    if sig.parity == 0:
        return pack_state(a12 ^ sig.tau_a12, b12 ^ sig.tau_b12)
    return pack_state(b12 ^ sig.tau_a12, a12 ^ sig.tau_b12)


def compose_word_signatures(
    left: WordSignature,
    right: WordSignature,
) -> WordSignature:
    """
    Composition law:
      f_(p1,t1) o f_(p2,t2) = f_(p1 xor p2, L^p1(t2) xor t1)
    where L is the A/B swap on translation pairs.
    """
    if left.parity == 0:
        ra, rb = right.tau_a12, right.tau_b12
    else:
        ra, rb = right.tau_b12, right.tau_a12

    return WordSignature(
        parity=left.parity ^ right.parity,
        tau_a12=(ra ^ left.tau_a12) & LAYER_MASK_12,
        tau_b12=(rb ^ left.tau_b12) & LAYER_MASK_12,
    )


# ----------------------------------------
# Omega state chart
# ----------------------------------------


@dataclass(frozen=True)
class OmegaState12:
    """
    Exact compact chart on Omega.
    u6, v6 are 6-bit GF(2)^6 coordinates such that:
      A12 = GENE_MAC_A12 ^ word6_to_pairdiag12(u6)
      B12 = GENE_MAC_A12 ^ word6_to_pairdiag12(v6)
    """
    u6: int
    v6: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "u6", int(self.u6) & CHIRALITY_MASK_6)
        object.__setattr__(self, "v6", int(self.v6) & CHIRALITY_MASK_6)

    @property
    def chirality6(self) -> int:
        return (self.u6 ^ self.v6) & CHIRALITY_MASK_6

    @property
    def shell(self) -> int:
        return self.chirality6.bit_count()

    @property
    def is_on_equality_horizon(self) -> bool:
        return self.u6 == self.v6

    @property
    def is_on_complement_horizon(self) -> bool:
        return self.chirality6 == EPSILON_6

    @property
    def optical_eq(self) -> Fraction:
        return Fraction(self.shell, 6)

    @property
    def optical_comp(self) -> Fraction:
        return Fraction(6 - self.shell, 6)

    @property
    def optical_mu(self) -> Fraction:
        return Fraction(2 * self.shell - 6, 6)


def is_in_omega24(state24: int) -> bool:
    """
    Exact structural Omega-membership check.
    A state is in Omega iff both components lie on the trace-1 affine shell:
      A12 ^ 0xAAA is pair-diagonal
      B12 ^ 0xAAA is pair-diagonal
    """
    a12, b12 = unpack_state(state24)
    return is_pair_diagonal12(a12 ^ GENE_MAC_A12) and is_pair_diagonal12(
        b12 ^ GENE_MAC_A12
    )


def try_state24_to_omega12(state24: int) -> OmegaState12 | None:
    s = int(state24) & 0xFFFFFF
    a12, b12 = unpack_state(s)

    ua12 = a12 ^ GENE_MAC_A12
    vb12 = b12 ^ GENE_MAC_A12

    if not is_pair_diagonal12(ua12):
        return None
    if not is_pair_diagonal12(vb12):
        return None

    return OmegaState12(
        u6=pairdiag12_to_word6(ua12),
        v6=pairdiag12_to_word6(vb12),
    )


def state24_to_omega12(state24: int) -> OmegaState12:
    omega = try_state24_to_omega12(state24)
    if omega is None:
        raise ValueError(f"State {int(state24) & 0xFFFFFF:#08x} is not in Omega")
    return omega


def omega12_to_state24(omega: OmegaState12 | tuple[int, int]) -> int:
    if isinstance(omega, OmegaState12):
        u6, v6 = omega.u6, omega.v6
    else:
        u6, v6 = omega

    a12 = GENE_MAC_A12 ^ word6_to_pairdiag12(u6)
    b12 = GENE_MAC_A12 ^ word6_to_pairdiag12(v6)
    return pack_state(a12, b12)


# ----------------------------------------
# GF(4) mode helpers
# ----------------------------------------


def frobenius_pair(pair_bits: int) -> int:
    """
    GF(4) Frobenius on one 2-bit pair.
      00 -> 00
      11 -> 11
      10 -> 01
      01 -> 10
    """
    p = int(pair_bits) & 0x3
    return ((p & 0x1) << 1) | ((p >> 1) & 0x1)


def gf4_trace(pair_bits: int) -> int:
    """
    GF(4) trace to GF(2) on one pair.
      00, 11 -> 0
      10, 01 -> 1
    """
    p = int(pair_bits) & 0x3
    return 0 if p in (0x0, 0x3) else 1


def gf4_norm(pair_bits: int) -> int:
    """
    GF(4) norm to GF(2) on one pair.
      00 -> 0
      11, 10, 01 -> 1
    """
    p = int(pair_bits) & 0x3
    return 0 if p == 0x0 else 1


def is_trace1_pair(pair_bits: int) -> bool:
    """
    True exactly for the physical/trace-1 pair states:
      10, 01
    """
    p = int(pair_bits) & 0x3
    return p in (0x2, 0x1)


def frobenius_component12(component12: int) -> int:
    """
    Apply pairwise Frobenius to a 12-bit component.
    """
    x = int(component12) & LAYER_MASK_12
    out = 0
    for i in range(6):
        pair = (x >> (2 * i)) & 0x3
        out |= frobenius_pair(pair) << (2 * i)
    return out & LAYER_MASK_12


def is_reachable_component(component12: int) -> bool:
    """
    True iff all 6 pairs are trace-1, i.e. each pair is 10 or 01.
    """
    x = int(component12) & LAYER_MASK_12
    for i in range(6):
        pair = (x >> (2 * i)) & 0x3
        if not is_trace1_pair(pair):
            return False
    return True


def step_omega12_by_byte(
    omega: OmegaState12 | tuple[int, int],
    byte: int,
) -> OmegaState12:
    if isinstance(omega, OmegaState12):
        u6, v6 = omega.u6, omega.v6
    else:
        u6, v6 = omega

    b = int(byte) & 0xFF
    u_next = v6 ^ EPS_A6_BY_BYTE[b]
    v_next = u6 ^ MICRO_REF_BY_BYTE[b] ^ EPS_B6_BY_BYTE[b]
    return OmegaState12(u6=u_next, v6=v_next)


def step_omega12_by_items(
    omega: OmegaState12,
    items: Iterable[ByteItem],
) -> OmegaState12:
    current = omega
    for item in items:
        if isinstance(item, (bytes, bytearray, memoryview)):
            for b in item:
                current = step_omega12_by_byte(current, b)
        else:
            current = step_omega12_by_byte(current, int(item) & 0xFF)
    return current


def apply_omega_gate_S(
    omega: OmegaState12 | tuple[int, int],
) -> OmegaState12:
    if isinstance(omega, OmegaState12):
        u6, v6 = omega.u6, omega.v6
    else:
        u6, v6 = omega
    return OmegaState12(u6=v6, v6=u6)


def apply_omega_gate_F(
    omega: OmegaState12 | tuple[int, int],
) -> OmegaState12:
    if isinstance(omega, OmegaState12):
        u6, v6 = omega.u6, omega.v6
    else:
        u6, v6 = omega
    return OmegaState12(
        u6=u6 ^ EPSILON_6,
        v6=v6 ^ EPSILON_6,
    )


def apply_omega_gate_C(
    omega: OmegaState12 | tuple[int, int],
) -> OmegaState12:
    if isinstance(omega, OmegaState12):
        u6, v6 = omega.u6, omega.v6
    else:
        u6, v6 = omega
    return OmegaState12(
        u6=v6 ^ EPSILON_6,
        v6=u6 ^ EPSILON_6,
    )


def apply_omega_gate(
    omega: OmegaState12 | tuple[int, int],
    name: str,
) -> OmegaState12:
    if name == "id":
        if isinstance(omega, OmegaState12):
            return omega
        return OmegaState12(u6=omega[0], v6=omega[1])
    if name == "S":
        return apply_omega_gate_S(omega)
    if name == "F":
        return apply_omega_gate_F(omega)
    if name == "C":
        return apply_omega_gate_C(omega)
    raise ValueError(f"Unknown Omega gate: {name!r}")


OMEGA_STATES_4096: tuple[int, ...] = tuple(
    omega12_to_state24(OmegaState12(u6=u6, v6=v6))
    for u6 in range(64)
    for v6 in range(64)
)


def state_conjugate_f(state24: int) -> int:
    """
    State-level conjugation by global complement.
    Equivalent to gate F on the 24-bit carrier.
    """
    a12, b12 = unpack_state(state24)
    return pack_state(a12 ^ LAYER_MASK_12, b12 ^ LAYER_MASK_12)


def k4_orbit(state24: int) -> frozenset[int]:
    """
    Full K4 orbit of a state under {id, S, C, F}.
    """
    s = int(state24) & 0xFFFFFF
    return frozenset((
        s,
        apply_gate(s, "S"),
        apply_gate(s, "C"),
        apply_gate(s, "F"),
    ))


def k4_stabilizer(state24: int) -> frozenset[str]:
    """
    Non-trivial K4 stabilizer of a state: subset of {"S","C","F"}.
    """
    s = int(state24) & 0xFFFFFF
    out: list[str] = []
    for name in ("S", "C", "F"):
        if apply_gate(s, name) == s:
            out.append(name)
    return frozenset(out)


def fixed_locus(gate_name: str) -> frozenset[int]:
    """
    Pointwise fixed locus of a K4 gate on Omega.
    gate_name in {"id","S","C","F"}.
    """
    name = str(gate_name)
    if name not in ("id", "S", "C", "F"):
        raise ValueError(f"Unknown gate name: {gate_name!r}")
    return frozenset(
        s for s in OMEGA_STATES_4096
        if apply_gate(s, name) == s
    )


def fixed_states_of_gate(gate_name: str) -> frozenset[int]:
    """
    Alias of fixed_locus.
    """
    return fixed_locus(gate_name)


SHADOW_PARTNER_BY_BYTE: tuple[int, ...] = tuple(
    (b ^ 0xFE) & 0xFF for b in range(256)
)


def shadow_partner_byte(byte: int) -> int:
    """
    Universal shadow partner of a byte.
    The partner produces the same 24-bit affine action on Omega.
    """
    return SHADOW_PARTNER_BY_BYTE[int(byte) & 0xFF]


def shadow_partner_map() -> dict[int, int]:
    """
    Full byte -> partner involution over all 256 bytes.
    """
    return {b: SHADOW_PARTNER_BY_BYTE[b] for b in range(256)}


def pack_omega12(omega: OmegaState12 | tuple[int, int]) -> int:
    if isinstance(omega, OmegaState12):
        u6, v6 = omega.u6, omega.v6
    else:
        u6, v6 = omega
    return ((u6 & CHIRALITY_MASK_6) << 6) | (v6 & CHIRALITY_MASK_6)


def unpack_omega12(packed: int) -> OmegaState12:
    x = int(packed) & 0xFFF
    return OmegaState12(
        u6=(x >> 6) & CHIRALITY_MASK_6,
        v6=x & CHIRALITY_MASK_6,
    )


def pack_omega_signature12(sig: OmegaSignature12) -> int:
    return (
        ((sig.parity & 1) << 12)
        | ((sig.tau_u6 & CHIRALITY_MASK_6) << 6)
        | (sig.tau_v6 & CHIRALITY_MASK_6)
    )


def unpack_omega_signature12(packed: int) -> OmegaSignature12:
    x = int(packed) & 0x1FFF
    return OmegaSignature12(
        parity=(x >> 12) & 1,
        tau_u6=(x >> 6) & CHIRALITY_MASK_6,
        tau_v6=x & CHIRALITY_MASK_6,
    )


def optical_coordinates_from_omega12(
    omega: OmegaState12,
) -> tuple[Fraction, Fraction, Fraction]:
    return (omega.optical_eq, omega.optical_comp, omega.optical_mu)


def optical_coordinates_from_state24(state24: int) -> tuple[Fraction, Fraction, Fraction]:
    return optical_coordinates_from_omega12(state24_to_omega12(state24))


def stabilizer_type_from_omega12(omega: OmegaState12) -> str:
    if omega.is_on_equality_horizon:
        return "equality"
    if omega.is_on_complement_horizon:
        return "complement"
    return "bulk"


def stabilizer_type_from_state24(state24: int) -> str:
    return stabilizer_type_from_omega12(state24_to_omega12(state24))


def verify_optical_conjugacy(
    states: Iterable[int],
    obs_plus: Callable[[int], Union[int, float, Fraction]],
    obs_minus: Callable[[int], Union[int, float, Fraction]],
) -> bool:
    """
    Verify obs_plus(s) + obs_minus(s) is constant over the given states.
    """
    baseline = None
    seen = False

    for s in states:
        x = obs_plus(int(s))
        y = obs_minus(int(s))
        total = x + y
        if not seen:
            baseline = total
            seen = True
        elif total != baseline:
            return False

    return True


# ----------------------------------------
# Omega signatures
# ----------------------------------------


@dataclass(frozen=True)
class OmegaSignature12:
    """
    Exact affine signature of a word on Omega.
    parity = 0 -> identity linear part
    parity = 1 -> swap linear part
    """
    parity: int
    tau_u6: int
    tau_v6: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "parity", int(self.parity) & 1)
        object.__setattr__(self, "tau_u6", int(self.tau_u6) & CHIRALITY_MASK_6)
        object.__setattr__(self, "tau_v6", int(self.tau_v6) & CHIRALITY_MASK_6)


def omega_signature_from_word_signature(sig: WordSignature) -> OmegaSignature12:
    if not is_pair_diagonal12(sig.tau_a12):
        raise ValueError(f"tau_a12 is not pair-diagonal: {sig.tau_a12:#05x}")
    if not is_pair_diagonal12(sig.tau_b12):
        raise ValueError(f"tau_b12 is not pair-diagonal: {sig.tau_b12:#05x}")

    return OmegaSignature12(
        parity=sig.parity,
        tau_u6=pairdiag12_to_word6(sig.tau_a12),
        tau_v6=pairdiag12_to_word6(sig.tau_b12),
    )


def omega_word_signature(items: Iterable[ByteItem]) -> OmegaSignature12:
    return omega_signature_from_word_signature(word_signature(items))


def apply_omega_signature(
    omega: OmegaState12 | tuple[int, int],
    sig: OmegaSignature12,
) -> OmegaState12:
    if isinstance(omega, OmegaState12):
        u6, v6 = omega.u6, omega.v6
    else:
        u6, v6 = omega

    if sig.parity == 0:
        return OmegaState12(u6=u6 ^ sig.tau_u6, v6=v6 ^ sig.tau_v6)
    return OmegaState12(u6=v6 ^ sig.tau_u6, v6=u6 ^ sig.tau_v6)


def compose_omega_signatures(
    left: OmegaSignature12,
    right: OmegaSignature12,
) -> OmegaSignature12:
    if left.parity == 0:
        ru, rv = right.tau_u6, right.tau_v6
    else:
        ru, rv = right.tau_v6, right.tau_u6

    return OmegaSignature12(
        parity=left.parity ^ right.parity,
        tau_u6=ru ^ left.tau_u6,
        tau_v6=rv ^ left.tau_v6,
    )


# ----------------------------------------
# Shell algebra primitives
# ----------------------------------------


def shell_index_from_chirality6(chirality6: int) -> int:
    return (int(chirality6) & CHIRALITY_MASK_6).bit_count()


def shell_index_from_omega12(omega: OmegaState12) -> int:
    return omega.shell


def shell_population(shell: int) -> int:
    w = int(shell)
    if w < 0 or w > 6:
        raise ValueError(f"shell must be in 0..6, got {shell}")
    return comb(6, w) * 64


SHELL_POPULATIONS_7: tuple[int, ...] = tuple(
    shell_population(w) for w in range(7)
)


def shell_transition_probability(
    w_src: int,
    q_weight: int,
    w_dst: int,
) -> Fraction:
    """
    Exact shell transition probability for adding a q-vector of Hamming weight q_weight.
    Formula:
      w' = w + j - 2t
      P = C(w,t) C(6-w, j-t) / C(6,j)
    where t = (w + j - w') / 2
    """
    w = int(w_src)
    j = int(q_weight)
    wp = int(w_dst)

    if not (0 <= w <= 6 and 0 <= j <= 6 and 0 <= wp <= 6):
        raise ValueError("w_src, q_weight, w_dst must all be in 0..6")

    delta = w + j - wp
    if delta < 0 or (delta & 1):
        return Fraction(0, 1)

    t = delta // 2
    if t < 0 or t > min(w, j):
        return Fraction(0, 1)
    if (j - t) < 0 or (j - t) > (6 - w):
        return Fraction(0, 1)

    return Fraction(comb(w, t) * comb(6 - w, j - t), comb(6, j))


def shell_transition_matrix_for_q_weight(
    q_weight: int,
) -> tuple[tuple[Fraction, ...], ...]:
    j = int(q_weight)
    if not (0 <= j <= 6):
        raise ValueError(f"q_weight must be in 0..6, got {q_weight}")
    return tuple(
        tuple(shell_transition_probability(w, j, wp) for wp in range(7))
        for w in range(7)
    )


def shell_markov_step(
    distribution: Iterable[int | Fraction],
    q_weight: int,
) -> tuple[Fraction, ...]:
    """
    Push a shell distribution forward by one exact q-weight shell kernel.
    Input distribution is over source shells 0..6.
    Output distribution is over destination shells 0..6.
    """
    d = tuple(Fraction(x) for x in distribution)
    if len(d) != 7:
        raise ValueError(f"Expected length-7 shell distribution, got {len(d)}")

    T = shell_transition_matrix_for_q_weight(int(q_weight))
    return tuple(
        Fraction(sum(d[w] * T[w][wp] for w in range(7)))
        for wp in range(7)
    )


FULL_BYTE_SHELL_DISTRIBUTION: tuple[Fraction, ...] = tuple(
    Fraction(comb(6, w), 64) for w in range(7)
)


# ----------------------------------------
# Krawtchouk shell spectral tools
# ----------------------------------------


def _krawtchouk_6() -> tuple[tuple[int, ...], ...]:
    rows = []
    for w in range(7):
        row = []
        for k in range(7):
            val = 0
            for j in range(k + 1):
                if j <= w and (k - j) <= (6 - w):
                    val += ((-1) ** j) * comb(w, j) * comb(6 - w, k - j)
            row.append(val)
        rows.append(tuple(row))
    return tuple(rows)


KRAWTCHOUK_7: tuple[tuple[int, ...], ...] = _krawtchouk_6()


def shell_krawtchouk_transform_exact(
    f_shell: Iterable[int | Fraction],
) -> tuple[Fraction, ...]:
    f = tuple(Fraction(x) for x in f_shell)
    if len(f) != 7:
        raise ValueError(f"Expected length-7 shell vector, got {len(f)}")

    out = []
    for k in range(7):
        numer = sum(
            Fraction(comb(6, w) * KRAWTCHOUK_7[w][k], 1) * f[w]
            for w in range(7)
        )
        denom = 64 * comb(6, k)
        out.append(numer / denom)
    return tuple(out)


def shell_krawtchouk_inverse_exact(
    coeffs: Iterable[int | Fraction],
) -> tuple[Fraction, ...]:
    c = tuple(Fraction(x) for x in coeffs)
    if len(c) != 7:
        raise ValueError(f"Expected length-7 spectral vector, got {len(c)}")

    return tuple(
        Fraction(sum(Fraction(KRAWTCHOUK_7[w][k], 1) * c[k] for k in range(7)))
        for w in range(7)
    )


def shell_krawtchouk_transform_float(f_shell: Iterable[float]) -> tuple[float, ...]:
    f = tuple(float(x) for x in f_shell)
    if len(f) != 7:
        raise ValueError(f"Expected length-7 shell vector, got {len(f)}")

    out: list[float] = []
    for k in range(7):
        numer = 0.0
        for w in range(7):
            numer += comb(6, w) * KRAWTCHOUK_7[w][k] * f[w]
        denom = 64.0 * comb(6, k)
        out.append(numer / denom)
    return tuple(out)


# ----------------------------------------
# Walsh / character primitives (DJ, BV, HSP)
# ----------------------------------------


def walsh_sign6(q: int, r: int) -> int:
    """Character (-1)^(q*r) on GF(2)^6."""
    return -1 if ((int(q) & int(r)).bit_count() & 1) else 1


def bv_phase6(secret6: int) -> tuple[int, ...]:
    """
    Phase oracle values for Bernstein-Vazirani:
      chi -> (-1)^(secret * chi)
    """
    s = int(secret6) & CHIRALITY_MASK_6
    return tuple(walsh_sign6(s, chi) for chi in range(64))


def dj_balanced_phase6(mask6: int = 0x20) -> tuple[int, ...]:
    """
    A canonical balanced phase function:
      chi -> (-1)^(mask * chi)
    """
    m = int(mask6) & CHIRALITY_MASK_6
    return tuple(walsh_sign6(m, chi) for chi in range(64))


_WALSH_HADAMARD64: Final[np.ndarray] = np.array(
    [[walsh_sign6(q, r) / 8.0 for r in range(64)] for q in range(64)],
    dtype=np.float64,
)


def walsh_hadamard64() -> np.ndarray:
    """Return the canonical 64x64 Walsh-Hadamard matrix on the chirality register."""
    return _WALSH_HADAMARD64.copy()


# ----------------------------------------
# Import-time verification
# ----------------------------------------


def _verify_mask_structure() -> None:
    """Verify 64 distinct masks and 256 distinct (family, mask) pairs."""
    masks: set[int] = set()
    pairs: set[tuple[int, int]] = set()
    for b in range(256):
        intron = byte_to_intron(b)
        m = mask12_for_byte(b)
        f = intron_family(intron)
        masks.add(m)
        pairs.add((f, m))
    assert len(masks) == 64, (
        f"Expected 64 distinct masks, got {len(masks)}"
    )
    assert len(pairs) == 256, (
        f"Expected 256 (family, mask) pairs, got {len(pairs)}"
    )


_verify_mask_structure()
