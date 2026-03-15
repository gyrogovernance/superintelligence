from __future__ import annotations

from enum import IntEnum

from src.api import q_word6_for_items


class ResonanceProfile(IntEnum):
    """
    Stable profile identifiers for persistence and runtime selection.
    """

    CHIRALITY = 1
    SHELL = 2
    HORIZON_CLASS = 3
    OMEGA_COINCIDENCE = 4
    SIGNATURE = 5
    Q_TRANSPORT = 6


def chi6_from_omega12(omega12: int) -> int:
    """Packed Ω helper: chi6 = u6 XOR v6."""
    x = int(omega12) & 0xFFF
    return ((x >> 6) ^ x) & 0x3F


def shell_from_omega12(omega12: int) -> int:
    """Packed Ω helper: shell = popcount(chi6)."""
    return chi6_from_omega12(omega12).bit_count()


def horizon_class_from_omega12(omega12: int) -> int:
    """
    3-way horizon partition:
      0 = equality horizon
      1 = complement horizon
      2 = bulk
    """
    chi6 = chi6_from_omega12(omega12)
    if chi6 == 0:
        return 0
    if chi6 == 0x3F:
        return 1
    return 2


def bucket_count(profile: ResonanceProfile) -> int:
    """Number of resonance buckets for the given profile."""
    if profile == ResonanceProfile.CHIRALITY:
        return 64
    if profile == ResonanceProfile.SHELL:
        return 7
    if profile == ResonanceProfile.HORIZON_CLASS:
        return 3
    if profile == ResonanceProfile.OMEGA_COINCIDENCE:
        return 4096
    if profile == ResonanceProfile.SIGNATURE:
        return 8192  # packed OmegaSignature12 occupies 13 bits
    if profile == ResonanceProfile.Q_TRANSPORT:
        return 64
    raise ValueError(f"Unknown profile: {profile!r}")


def key_for_closed_word(
    profile: ResonanceProfile,
    *,
    omega12: int,
    word4: bytes,
    omega_sig: int,
) -> int:
    """
    Compute the resonance key at word closure.

    This is intentionally closure-boundary semantics.
    """
    if profile == ResonanceProfile.CHIRALITY:
        return chi6_from_omega12(omega12)

    if profile == ResonanceProfile.SHELL:
        return shell_from_omega12(omega12)

    if profile == ResonanceProfile.HORIZON_CLASS:
        return horizon_class_from_omega12(omega12)

    if profile == ResonanceProfile.OMEGA_COINCIDENCE:
        return int(omega12) & 0xFFF

    if profile == ResonanceProfile.SIGNATURE:
        return int(omega_sig) & 0x1FFF

    if profile == ResonanceProfile.Q_TRANSPORT:
        return q_word6_for_items(word4)

    raise ValueError(f"Unknown profile: {profile!r}")