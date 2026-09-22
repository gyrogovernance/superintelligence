"""Exact path field of a nucleic-acid sequence.

The kernel is the physics authority. This module compiles a DNA window into
the finite objects the signature group carries:

* the genomic byte ledger;
* the open-path word signature of the whole walk and of declared blocks;
* the group commutator ``C(L,R) = L R L^{-1} R^{-1}`` of those blocks;
* multiscale sliding-window commutators (depth-4, depth-8, half-split);
* fold disagreement and family-sheet occupation from the certified compile;
* shell climate (eta, mean_shell) as a secondary thermodynamic dial.

A ``WordSignature`` is a finite affine residual. A nonzero commutator is a
finite path-order defect. ``delta_BU``, ``rho``, and ``APERTURE_GAP_Q256`` are
constitutional scale anchors from ``src.constants``.

Theory: Analysis_Holonomy.md Sec. 15-17; Analysis_hQVM_CGM_Genomics.md;
kernel APIs ``word_signature``, ``compose_word_signatures``, ``apply_word_signature``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np

from src.api import WordSignature, apply_word_signature, compose_word_signatures, word_signature
from src.constants import (
    APERTURE_GAP_Q256,
    BU_APERTURE_GAP,
    BU_CLOSURE_RATIO,
    BU_HOLONOMY_ANGLE,
    LAYER_MASK_12,
    pack_state,
)
from src.family import fold_disagreement_d
from src.tools.autoencoder.programs.genomics.genomics import (
    NucleotideEncoding,
    all_nucleotide_encodings,
    clean_acgt,
    compile_interval,
    genomic_byte_stream,
    translation_table,
)

IDENTITY = WordSignature(parity=0, tau_a12=0, tau_b12=0)

#: Constitutional scale anchors. Printed for reference; never optimised as targets.
CONSTITUTIONAL_SCALE: Dict[str, float] = {
    "delta_bu": float(BU_HOLONOMY_ANGLE),
    "closure_ratio_rho": float(BU_CLOSURE_RATIO),
    "aperture_gap": float(BU_APERTURE_GAP),
    "aperture_gap_q256": float(APERTURE_GAP_Q256),
}


# ---------------------------------------------------------------------------
# Signature algebra
# ---------------------------------------------------------------------------


def inverse_word_signature(sig: WordSignature) -> WordSignature:
    """Group inverse of an affine word signature.

    Even parity: translation is self-inverse under XOR.
    Odd parity: the A/B swap requires exchanging the translation components.
    """
    if int(sig.parity) & 1 == 0:
        return WordSignature(
            parity=0,
            tau_a12=int(sig.tau_a12) & LAYER_MASK_12,
            tau_b12=int(sig.tau_b12) & LAYER_MASK_12,
        )
    return WordSignature(
        parity=1,
        tau_a12=int(sig.tau_b12) & LAYER_MASK_12,
        tau_b12=int(sig.tau_a12) & LAYER_MASK_12,
    )


def multiply_signatures(*sigs: WordSignature) -> WordSignature:
    """Left-to-right path product: first segment, then second, then ...

    Matches the repository convention that ``compose(later, earlier)`` equals
    the signature of ``earlier`` followed by ``later``.
    """
    if not sigs:
        return IDENTITY
    acc = sigs[0]
    for nxt in sigs[1:]:
        acc = compose_word_signatures(nxt, acc)
    return acc


def commutator(left: WordSignature, right: WordSignature) -> WordSignature:
    """Exact group commutator ``C(L,R) = L R L^{-1} R^{-1}``."""
    return multiply_signatures(
        left, right, inverse_word_signature(left), inverse_word_signature(right)
    )


def is_identity(sig: WordSignature) -> bool:
    return (
        int(sig.parity) & 1 == 0
        and int(sig.tau_a12) & LAYER_MASK_12 == 0
        and int(sig.tau_b12) & LAYER_MASK_12 == 0
    )


def signature_key(sig: WordSignature) -> tuple[int, int, int]:
    """Exact finite identity of a signature: (parity, tau_a12, tau_b12)."""
    return (int(sig.parity) & 1, int(sig.tau_a12) & LAYER_MASK_12, int(sig.tau_b12) & LAYER_MASK_12)


def translation_chirality(sig: WordSignature) -> int:
    """``tau_a XOR tau_b`` projected to the 6-bit chirality register when possible."""
    try:
        return int(sig.tau_a6) ^ int(sig.tau_b6)
    except ValueError:
        # Non pair-diagonal translation: fall back to low 6 bits of each layer.
        return ((int(sig.tau_a12) & 0x3F) ^ (int(sig.tau_b12) & 0x3F)) & 0x3F


def conjugacy_invariant(sig: WordSignature) -> tuple[int, int]:
    """Coordinate-free finite invariants of a signature class.

    Returns ``(parity, popcount(translation_chirality))``. Equal values are
    necessary for conjugacy in the affine signature group; they are not a
    continuous rotation angle.
    """
    chi = translation_chirality(sig)
    return (int(sig.parity) & 1, int(chi).bit_count())


def carrier_displacement(sig: WordSignature, base: int = 0) -> int:
    """Image of a base state under the signature, as a 24-bit state."""
    return int(apply_word_signature(int(base) & 0xFFFFFF, sig))


def signature_order(sig: WordSignature, *, max_order: int = 16) -> int:
    """Smallest positive k with sig^k = identity, or 0 if none within max_order."""
    if is_identity(sig):
        return 1
    acc = sig
    for k in range(1, max_order + 1):
        if is_identity(acc):
            return k
        acc = multiply_signatures(acc, sig)
    return 0


# ---------------------------------------------------------------------------
# Commutator report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommutatorReport:
    """Exact finite commutator of two path blocks."""

    left: WordSignature
    right: WordSignature
    commutator: WordSignature
    is_identity: bool
    order: int
    conjugacy: tuple[int, int]
    translation_chirality: int
    displacement_of_rest: int
    weight: int  # popcount of both translation layers; a coordinate descriptor only

    def as_dict(self) -> Dict[str, float]:
        c = self.commutator
        return {
            "commutator_is_identity": float(self.is_identity),
            "commutator_order": float(self.order),
            "commutator_parity": float(c.parity),
            "commutator_tau_a": float(c.tau_a12),
            "commutator_tau_b": float(c.tau_b12),
            "commutator_conjugacy_parity": float(self.conjugacy[0]),
            "commutator_conjugacy_shell": float(self.conjugacy[1]),
            "commutator_translation_chirality": float(self.translation_chirality),
            "commutator_displacement": float(self.displacement_of_rest),
            "commutator_weight": float(self.weight),
        }


def report_commutator(left: WordSignature, right: WordSignature) -> CommutatorReport:
    c = commutator(left, right)
    return CommutatorReport(
        left=left,
        right=right,
        commutator=c,
        is_identity=is_identity(c),
        order=signature_order(c),
        conjugacy=conjugacy_invariant(c),
        translation_chirality=translation_chirality(c),
        displacement_of_rest=carrier_displacement(c, pack_state(0, 0)),
        weight=int(c.tau_a12).bit_count() + int(c.tau_b12).bit_count(),
    )


# ---------------------------------------------------------------------------
# Multiscale path field
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathField:
    """Compiled finite path field of one coding sequence under one encoding."""

    sequence: str
    encoding_index: int
    bytes: tuple[int, ...]
    whole_signature: WordSignature
    half_commutator: CommutatorReport
    depth3_nonidentity_fraction: float
    depth5_nonidentity_fraction: float
    depth3_mean_weight: float
    depth5_mean_weight: float
    mean_fold_disagreement: float
    family_l1: float
    family_0_frac: float
    mean_shell: float
    eta: float
    m2: float
    observables: Dict[str, float]

    @property
    def n_bytes(self) -> int:
        return len(self.bytes)

    @property
    def n_codons(self) -> int:
        return len(self.sequence) // 3


def _sliding_commutators(
    stream: Sequence[int], depth: int
) -> list[CommutatorReport]:
    """Commutators of consecutive half-blocks of length ``depth`` bytes each."""
    if depth < 1 or len(stream) < 2 * depth:
        return []
    out: list[CommutatorReport] = []
    step = max(1, depth // 2)
    for i in range(0, len(stream) - 2 * depth + 1, step):
        left = word_signature(stream[i : i + depth])
        right = word_signature(stream[i + depth : i + 2 * depth])
        out.append(report_commutator(left, right))
    return out


def _census(reports: Sequence[CommutatorReport]) -> tuple[float, float]:
    if not reports:
        return (float("nan"), float("nan"))
    non_id = sum(1 for r in reports if not r.is_identity) / len(reports)
    mean_w = float(np.mean([r.weight for r in reports]))
    return (float(non_id), mean_w)


def compile_path_field(
    seq: str,
    *,
    enc: NucleotideEncoding | None = None,
    encoding_index: int = 0,
    ncbi_id: int = 11,
    label: str = "path",
    certificates: bool = True,
) -> PathField:
    """Compile a coding sequence into its exact finite path field."""
    encoding = enc or all_nucleotide_encodings()[encoding_index]
    s = clean_acgt(seq)
    if len(s) % 3:
        raise ValueError(f"sequence length {len(s)} is not a multiple of three")
    stream = tuple(int(b) & 0xFF for b in genomic_byte_stream(s, encoding))
    whole = word_signature(stream) if stream else IDENTITY

    # Half commutator at a matched odd split. An even/even split makes both
    # half words pure translations, which commute, so the feature would be a
    # CDS-length mod 6 dummy.
    mid = len(stream) // 2
    if mid % 2 == 0:
        mid -= 1
    if mid >= 1 and len(stream) - mid >= 1:
        half = report_commutator(
            word_signature(stream[:mid]), word_signature(stream[mid:])
        )
    else:
        half = report_commutator(IDENTITY, IDENTITY)

    d3 = _sliding_commutators(stream, 3)
    d5 = _sliding_commutators(stream, 5)
    d3_frac, d3_w = _census(d3)
    d5_frac, d5_w = _census(d5)

    compiled = compile_interval(
        s, encoding, label=label, ncbi_id=ncbi_id, certificates=certificates
    )

    def _v(layer: str, key: str) -> float:
        v = compiled.value(layer, key)
        return float("nan") if v is None else float(v)

    fold = (
        float(sum(fold_disagreement_d(b, 6) for b in stream) / len(stream))
        if stream
        else float("nan")
    )
    family_l1 = _v("family_sheet", "l1_uniform")
    family_0 = _v("family_sheet", "mu_0")
    mean_shell = _v("chi_shells", "mean_shell")
    eta = _v("qubec_order", "eta")
    m2 = _v("qubec_order", "M2")

    observables: Dict[str, float] = {
        "whole_parity": float(whole.parity),
        "whole_tau_a": float(whole.tau_a12),
        "whole_tau_b": float(whole.tau_b12),
        "whole_conjugacy_parity": float(conjugacy_invariant(whole)[0]),
        "whole_conjugacy_shell": float(conjugacy_invariant(whole)[1]),
        "half_commutator_is_identity": float(half.is_identity),
        "half_commutator_order": float(half.order),
        "half_commutator_weight": float(half.weight),
        "half_commutator_conjugacy_shell": float(half.conjugacy[1]),
        "depth3_nonidentity_fraction": d3_frac,
        "depth5_nonidentity_fraction": d5_frac,
        "depth3_mean_weight": d3_w,
        "depth5_mean_weight": d5_w,
        "mean_fold_disagreement": fold,
        "family_l1": family_l1,
        "family_0_frac": family_0,
        "mean_shell": mean_shell,
        "eta": eta,
        "m2": m2,
        "n_bytes": float(len(stream)),
    }
    observables.update({f"half_{k}": v for k, v in half.as_dict().items()})

    return PathField(
        sequence=s,
        encoding_index=encoding_index,
        bytes=stream,
        whole_signature=whole,
        half_commutator=half,
        depth3_nonidentity_fraction=d3_frac,
        depth5_nonidentity_fraction=d5_frac,
        depth3_mean_weight=d3_w,
        depth5_mean_weight=d5_w,
        mean_fold_disagreement=fold,
        family_l1=family_l1,
        family_0_frac=family_0,
        mean_shell=mean_shell,
        eta=eta,
        m2=m2,
        observables=observables,
    )


def field_delta(a: PathField, b: PathField) -> Dict[str, float]:
    """Signed change ``a - b`` on every shared scalar observable."""
    return {
        name: float(a.observables[name] - b.observables[name])
        for name in a.observables
        if name in b.observables
        and np.isfinite(a.observables[name])
        and np.isfinite(b.observables[name])
    }


def protein_of(seq: str, ncbi_id: int = 11) -> str:
    code = translation_table(ncbi_id)
    s = clean_acgt(seq)
    return "".join(code.get(s[i : i + 3], "?") for i in range(0, len(s) - 2, 3))


def synonymous_alternates(codon: str, ncbi_id: int = 11) -> tuple[str, ...]:
    code = translation_table(ncbi_id)
    amino = code.get(codon)
    if amino is None or amino == "*":
        return (codon,)
    return tuple(sorted(c for c, a in code.items() if a == amino and c != codon))


#: Observables that are designable finite path-order dials.
PATH_OBSERVABLES: tuple[str, ...] = (
    "half_commutator_weight",
    "half_commutator_conjugacy_shell",
    "depth3_nonidentity_fraction",
    "depth5_nonidentity_fraction",
    "depth3_mean_weight",
    "mean_fold_disagreement",
    "whole_conjugacy_shell",
)

#: Observables that are thermodynamic climate dials.
CLIMATE_OBSERVABLES: tuple[str, ...] = (
    "mean_shell",
    "eta",
    "m2",
    "family_l1",
    "family_0_frac",
)

#: Theorem invariants: never design targets.
INVARIANT_OBSERVABLES: tuple[str, ...] = (
    "ab_plus_horizon",
    "depth4_parity_zero_fraction",
)

FIELD_SCALES: Dict[str, float] = {
    "half_commutator_weight": 2.0,
    "half_commutator_conjugacy_shell": 1.0,
    "half_commutator_order": 1.0,
    "depth3_nonidentity_fraction": 0.10,
    "depth5_nonidentity_fraction": 0.10,
    "depth3_mean_weight": 1.0,
    "depth5_mean_weight": 1.0,
    "mean_fold_disagreement": 0.10,
    "whole_conjugacy_shell": 1.0,
    "mean_shell": 0.10,
    "eta": 0.010,
    "m2": 20.0,
    "family_l1": 0.05,
    "family_0_frac": 0.04,
}


def scale_of(name: str) -> float:
    return float(FIELD_SCALES.get(name, 1.0))
