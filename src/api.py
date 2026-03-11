# src/api.py
"""
Gyroscopic ASI aQPU Kernel API: precomputed tables, integrity checks, and derived functions.

Built from theoretical constants in src.constants. No HTTP or REST;
this module provides mask lookup, dual code and syndromes, trajectory
parity commitments, depth-4 projections, and derived measurements.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Union

import numpy as np

from src.constants import (
    CHIRALITY_MASK_6,
    CHIRALITY_QUBITS_6,
    EPSILON_6,
    LAYER_MASK_12,
    PAIR_MASKS_12,
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
