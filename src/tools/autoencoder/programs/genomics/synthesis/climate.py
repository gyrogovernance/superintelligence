"""Carrier climate and path memory of a nucleic-acid sequence.

A sequence is a path on the hQVM carrier manifold. Two classes of number are
read from it:

* **Path memory** (primary). The finite residual of the byte walk: the Omega
  word signature, its weight, the path-order noncommutativity of the two
  halves, the byte-fold disagreement, and the family-sheet occupation. These
  are the quantities CGM says carry order-dependent displacement. Theory:
  Analysis_Holonomy.md Sec. 15-17; Analysis_hQVM_CGM_Genomics.md Sec. 2.3,
  11; kernel APIs ``word_signature``, ``compose_word_signatures``,
  ``omega_word_signature``, ``fold_disagreement_d``.

* **Climate** (secondary). Shell census, radial modes, character spectrum,
  QuBEC order. Averages of the chirality register. Useful as a report and as
  a secondary dial, but blind to reordering — a mean is invariant under the
  synonymous rearrangements that are this lab's only design freedom.

Every field is either a certified compile read or an exact transform of one,
or an exact kernel signature of the genomic byte stream. Nothing is fitted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Dict, Iterable, Literal, Sequence

import numpy as np

from src.api import compose_word_signatures, omega_word_signature, word_signature
from src.family import fold_disagreement_d
from src.tools.autoencoder.programs.genomics.genomics import (
    genomic_byte_stream,

    CHIRALITY_D,
    N_CODONS,
    NucleotideEncoding,
    all_nucleotide_encodings,
    clean_acgt,
    compile_interval,
    pack_codon_bits,
    pair_inversion_encoding,
    translation_table,
)

# ---------------------------------------------------------------------------
# Basis transforms. Exact, no fitting.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _krawtchouk_table(n: int) -> tuple[tuple[int, ...], ...]:
    """``K_k(x; n)`` for ``0 <= k, x <= n``, exact integers.

    ``K_k(x) = sum_j (-1)^j C(x, j) C(n - x, k - j)``.
    """
    table: list[tuple[int, ...]] = []
    for k in range(n + 1):
        row = []
        for x in range(n + 1):
            total = 0
            for j in range(k + 1):
                if j > x or (k - j) > (n - x):
                    continue
                total += (-1) ** j * math.comb(x, j) * math.comb(n - x, k - j)
            row.append(total)
        table.append(tuple(row))
    return tuple(table)


def radial_modes(shell_census: np.ndarray, n: int = None) -> np.ndarray:  # type: ignore[assignment]
    """Krawtchouk radial coefficients of a shell histogram.

    Mode 0 is the total mass, mode 1 is the signed mean-shell displacement,
    higher modes read the shape of the radial distribution. These are the
    certified radial channel of Sec. 8.4.
    """
    n = CHIRALITY_D if n is None else n
    table = _krawtchouk_table(int(n))
    p = np.asarray(shell_census, dtype=np.float64)
    if p.shape != (int(n) + 1,):
        raise ValueError(f"shell census must have {int(n) + 1} entries")
    return np.asarray(
        [float(np.dot(p, np.asarray(table[k], dtype=np.float64))) for k in range(int(n) + 1)],
        dtype=np.float64,
    )


@lru_cache(maxsize=1)
def _character_signs() -> np.ndarray:
    """Sign matrix ``S[j, x] = (-1)^popcount(j AND x)`` on ``GF(2)^6``."""
    idx = np.arange(N_CODONS, dtype=np.int64)
    j = idx[:, None]
    x = idx[None, :]
    parity = np.zeros((N_CODONS, N_CODONS), dtype=np.int64)
    bit = np.bitwise_and(j, x)
    for k in range(CHIRALITY_D):
        parity += (bit >> k) & 1
    return np.where(parity % 2 == 0, 1.0, -1.0)


@lru_cache(maxsize=1)
def _character_degree() -> np.ndarray:
    """Hamming degree of each character index ``j``."""
    idx = np.arange(N_CODONS, dtype=np.int64)
    deg = np.zeros(N_CODONS, dtype=np.int64)
    for k in range(CHIRALITY_D):
        deg += (idx >> k) & 1
    return deg


def character_spectrum(chi_occupation: np.ndarray) -> np.ndarray:
    """Walsh character transform of a chirality-register occupation measure."""
    p = np.asarray(chi_occupation, dtype=np.float64)
    if p.shape != (N_CODONS,):
        raise ValueError(f"chirality occupation must have {N_CODONS} entries")
    return _character_signs() @ p


def spectral_summary(spectrum: np.ndarray) -> tuple[float, float]:
    """``(spectral_centroid, high_mode_fraction)`` of a character spectrum.

    Both are normalised by total character power, so they are invariants of the
    *shape* of the spectrum rather than of its magnitude. Low-degree characters
    are the coarse transport modes; high-degree characters are the fine ones.
    """
    power = np.asarray(spectrum, dtype=np.float64) ** 2
    total = float(power.sum())
    if total <= 0.0:
        return float("nan"), float("nan")
    deg = _character_degree()
    centroid = float(np.dot(deg, power) / total)
    high = float(power[deg >= 4].sum() / total)
    return centroid, high


def local_character_centroid(chi_values: np.ndarray, window: int = 15) -> np.ndarray:
    """Sliding-window character centroid of a chirality register.

    One value per position. The centroid is the mean transport degree of the
    local character power spectrum, so it measures how fine-grained the local
    transport climate is at that position.

    This exists because the natural-looking alternative does not work: the Walsh
    transform of a *single* carrier state has every coefficient at magnitude one,
    so its squared norm is the same constant for every possible state and can
    never distinguish one sequence from another. Only the transform of an
    *occupation measure* over a window carries sequence information.
    """
    chi = np.asarray(chi_values, dtype=np.int64) & (N_CODONS - 1)
    n = int(chi.size)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    w = int(max(1, min(int(window), n)))
    if w < 3:
        w = n
    if w < 1:
        return np.zeros(n, dtype=np.float32)
    pad = w // 2
    padded = np.pad(chi, (pad, pad), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, w)[:n]
    counts = np.zeros((win.shape[0], N_CODONS), dtype=np.float64)
    rows = np.repeat(np.arange(win.shape[0]), w)
    np.add.at(counts, (rows, win.reshape(-1)), 1.0)
    p = counts / counts.sum(axis=1, keepdims=True)
    spectrum = p @ _character_signs().T
    power = spectrum**2
    total = power.sum(axis=1)
    total[total <= 0.0] = 1.0
    deg = _character_degree().astype(np.float64)
    return (power @ deg / total).astype(np.float32)


# ---------------------------------------------------------------------------
# Path memory. Exact residual of the genomic byte walk.
# ---------------------------------------------------------------------------


def _popcount(x: int) -> int:
    return int(x).bit_count()


def path_memory_of_stream(stream: Sequence[int]) -> Dict[str, float]:
    """Finite path memory of a genomic byte stream.

    Returns the Omega signature of the whole walk, the Hamming weight of its
    translation residual, the noncommutativity of the two halves under the
    word-signature composition law, and the mean byte-fold disagreement.
    All are exact kernel reads.
    """
    items = [int(b) & 0xFF for b in stream]
    n = len(items)
    if n == 0:
        nan = float("nan")
        return {
            "signature_parity": nan,
            "signature_tau_u": nan,
            "signature_tau_v": nan,
            "signature_weight": nan,
            "signature_axis_u": nan,
            "signature_axis_v": nan,
            "path_noncommutativity": nan,
            "mean_fold_disagreement": nan,
            "n_bytes": 0.0,
        }
    sig = omega_word_signature(items)
    weight = float(_popcount(sig.tau_u6) + _popcount(sig.tau_v6))
    axis_u = float(_popcount(sig.tau_u6))
    axis_v = float(_popcount(sig.tau_v6))
    mid = n // 2
    # Split at an odd boundary: an even/even split makes both half words pure
    # translations, which commute, so the defect would measure sequence
    # length mod 4 rather than path order.
    if mid % 2 == 0:
        mid -= 1
    if mid >= 1 and n - mid >= 1:
        left = word_signature(items[:mid])
        right = word_signature(items[mid:])
        lr = compose_word_signatures(left, right)
        rl = compose_word_signatures(right, left)
        noncommute = float(
            _popcount(lr.tau_a12 ^ rl.tau_a12)
            + _popcount(lr.tau_b12 ^ rl.tau_b12)
            + (lr.parity ^ rl.parity)
        )
    else:
        noncommute = 0.0
    fold_mean = float(sum(fold_disagreement_d(b, 6) for b in items) / n)
    return {
        "signature_parity": float(sig.parity),
        "signature_tau_u": float(sig.tau_u6),
        "signature_tau_v": float(sig.tau_v6),
        "signature_weight": weight,
        "signature_axis_u": axis_u,
        "signature_axis_v": axis_v,
        "path_noncommutativity": noncommute,
        "mean_fold_disagreement": fold_mean,
        "n_bytes": float(n),
    }


def path_memory_of_sequence(
    seq: str, *, enc: NucleotideEncoding | None = None
) -> Dict[str, float]:
    """Path memory of a coding sequence under one nucleotide encoding."""
    encoding = enc or all_nucleotide_encodings()[0]
    stream = list(genomic_byte_stream(clean_acgt(seq), encoding))
    return path_memory_of_stream(stream)



# ---------------------------------------------------------------------------
# The climate vector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CarrierClimate:
    """Measured physical climate of one nucleic-acid window.

    All fields are either a certified compile read or an exact transform of one.
    ``observables`` is the flat scalar view used by targets and losses.
    """

    label: str
    n_codons: int
    n_pairs: int
    mean_shell: float
    low_shell_fraction: float
    shell_census: np.ndarray
    radial_modes: np.ndarray
    chi_occupation: np.ndarray
    character_spectrum: np.ndarray
    spectral_centroid: float
    high_mode_fraction: float
    eta: float
    m2: float
    bit_occupation: np.ndarray
    anisotropy_spread: float
    family_occupation: np.ndarray
    family_l1: float
    depth4_parity_zero_fraction: float
    ab_plus_horizon: float
    mean_fold_disagreement: float
    signature_parity: float
    signature_tau_u: float
    signature_tau_v: float
    signature_weight: float
    signature_axis_u: float
    signature_axis_v: float
    path_noncommutativity: float
    observables: Dict[str, float]

    def scalar(self, name: str) -> float:
        value = self.observables.get(name)
        if value is None:
            raise KeyError(f"unknown climate observable: {name}")
        return float(value)

    def delta(self, other: CarrierClimate) -> Dict[str, float]:
        """Signed change of every shared scalar observable, ``self - other``."""
        return {
            name: float(self.observables[name] - other.observables[name])
            for name in self.observables
            if name in other.observables
        }

    def as_reference(self, label: str | None = None) -> ClimateReference:
        return ClimateReference(
            label=label or self.label,
            values=dict(self.observables),
            provenance=f"measured on one molecule of {self.n_codons} codons",
        )


def _flat_observables(
    *,
    mean_shell: float,
    low_shell_fraction: float,
    shell_census: np.ndarray,
    radial: np.ndarray,
    spectrum: np.ndarray,
    spectral_centroid: float,
    high_mode_fraction: float,
    eta: float,
    m2: float,
    bit_occupation: np.ndarray,
    anisotropy_spread: float,
    family_occupation: np.ndarray,
    family_l1: float,
    certificates_computed: bool = True,
) -> Dict[str, float]:
    out: Dict[str, float] = {
        "mean_shell": mean_shell,
        "low_shell_fraction": low_shell_fraction,
        "spectral_centroid": spectral_centroid,
        "high_mode_fraction": high_mode_fraction,
        "eta": eta,
        "m2": m2,
        "family_l1": family_l1,
        "anisotropy_spread": anisotropy_spread,
    }
    # The path certificates live on the dataclass too, but they belong in the
    # flat view so that a program can target them and a report can print them
    # next to every other observable.
    for name in CERTIFICATE_OBSERVABLES:
        out[name] = float("nan")
    for k in range(len(shell_census)):
        out[f"shell_{k}_frac"] = float(shell_census[k])
    for k in range(len(radial)):
        out[f"radial_{k}"] = float(radial[k])
    for k in range(len(bit_occupation)):
        out[f"bit_{k}_occupation"] = float(bit_occupation[k])
        out[f"bit_{k}_anisotropy"] = float(0.5 - bit_occupation[k])
    for k in range(len(family_occupation)):
        out[f"family_{k}_frac"] = float(family_occupation[k])
    out["character_power_low"] = float(
        np.dot(_character_degree() <= 1, np.asarray(spectrum, dtype=np.float64) ** 2)
    )
    # 1.0 means the path-certificate layers were actually computed for this
    # molecule; 0.0 means they were skipped inside a search loop.
    out["certificates_computed"] = float(certificates_computed)
    return out


def climate_of_sequence(
    seq: str,
    *,
    ncbi_id: int = 1,
    enc: NucleotideEncoding | None = None,
    label: str = "sequence",
    certificates: bool = True,
) -> CarrierClimate:
    """Measure the carrier climate of one coding window.

    The window is read in frame 0. ``chi`` is taken on adjacent *codon pairs*,
    which is the carrier of translation (Sec. 2.3), not on nucleotide bytes.

    ``certificates=False`` skips the path-certificate layers, which no target
    reads. It is only ever used inside a search loop; every molecule that is
    returned is measured again with ``certificates=True``.
    """
    encoding = enc or all_nucleotide_encodings()[0]
    s = clean_acgt(seq)
    compiled = compile_interval(
        s, encoding, label=label, ncbi_id=ncbi_id, certificates=certificates
    )

    def _value(layer: str, key: str) -> float:
        v = compiled.value(layer, key)
        return float("nan") if v is None else float(v)

    n_codons = len(s) // 3
    codons = [s[3 * i : 3 * i + 3] for i in range(n_codons)]

    shell_census = np.asarray(
        [_value("chi_shell_census", f"shell_{k}_frac") for k in range(7)],
        dtype=np.float64,
    )
    bit_occupation = np.asarray(
        [_value("chi_bit_occupation", f"bit_{k}_frac") for k in range(CHIRALITY_D)],
        dtype=np.float64,
    )
    family_occupation = np.asarray(
        [_value("family_sheet", f"mu_{k}") for k in range(4)], dtype=np.float64
    )
    chi_occupation = _chi_occupation(codons, encoding)
    spectrum = character_spectrum(chi_occupation)
    centroid, high = spectral_summary(spectrum)
    finite = bit_occupation[np.isfinite(bit_occupation)]
    anisotropy_spread = (
        float(np.sqrt(np.mean((finite - 0.5) ** 2))) if finite.size else float("nan")
    )
    mean_shell = _value("chi_shells", "mean_shell")
    radial = (
        radial_modes(shell_census) if np.isfinite(shell_census).all() else np.full(7, np.nan)
    )
    eta = _value("qubec_order", "eta")
    m2 = _value("qubec_order", "M2")
    family_l1 = _value("family_sheet", "l1_uniform")
    low_fraction = _value("chi_shells", "low_shell_fraction")
    n_pairs = int(_value("chi_shells", "n_pairs"))
    ab_horizon = _value("ab_horizon", "ab_plus_horizon")
    # NOTE: this is the mean fold disagreement of the byte layer, as named on
    # the certified compile. It is NOT the plaquette curvature census of
    # Sec. 8.2, which this compile does not compute; the two must not be
    # relabelled into each other.
    fold_disagreement = _value("byte_fold_w", "mean_fold_disagreement")

    observables = _flat_observables(
        mean_shell=mean_shell,
        low_shell_fraction=low_fraction,
        shell_census=shell_census,
        radial=radial,
        spectrum=spectrum,
        spectral_centroid=centroid,
        high_mode_fraction=high,
        eta=eta,
        m2=m2,
        bit_occupation=bit_occupation,
        anisotropy_spread=anisotropy_spread,
        family_occupation=family_occupation,
        family_l1=family_l1,
        certificates_computed=certificates,
    )
    memory = path_memory_of_sequence(s, enc=encoding)
    observables["ab_plus_horizon"] = ab_horizon
    # Prefer the kernel fold read; fall back to the compile layer.
    fold_value = memory["mean_fold_disagreement"]
    if not math.isfinite(fold_value):
        fold_value = fold_disagreement
    observables["mean_fold_disagreement"] = fold_value
    for key in (
        "signature_parity",
        "signature_tau_u",
        "signature_tau_v",
        "signature_weight",
        "signature_axis_u",
        "signature_axis_v",
        "path_noncommutativity",
    ):
        observables[key] = float(memory[key])

    return CarrierClimate(
        label=label,
        n_codons=len(codons),
        n_pairs=n_pairs,
        mean_shell=mean_shell,
        low_shell_fraction=low_fraction,
        shell_census=shell_census,
        radial_modes=radial,
        chi_occupation=chi_occupation,
        character_spectrum=spectrum,
        spectral_centroid=centroid,
        high_mode_fraction=high,
        eta=eta,
        m2=m2,
        bit_occupation=bit_occupation,
        anisotropy_spread=anisotropy_spread,
        family_occupation=family_occupation,
        family_l1=family_l1,
        depth4_parity_zero_fraction=_value("depth4_parity", "parity_zero_frac"),
        ab_plus_horizon=_value("ab_horizon", "ab_plus_horizon"),
        mean_fold_disagreement=fold_value,
        signature_parity=float(memory["signature_parity"]),
        signature_tau_u=float(memory["signature_tau_u"]),
        signature_tau_v=float(memory["signature_tau_v"]),
        signature_weight=float(memory["signature_weight"]),
        signature_axis_u=float(memory["signature_axis_u"]),
        signature_axis_v=float(memory["signature_axis_v"]),
        path_noncommutativity=float(memory["path_noncommutativity"]),
        observables=observables,
    )


def _chi_occupation(codons: Sequence[str], enc: NucleotideEncoding) -> np.ndarray:
    """Occupation measure of the 64 chirality values over a codon window."""
    if len(codons) < 2:
        return np.full(N_CODONS, np.nan)
    packed = np.asarray([pack_codon_bits(c, enc) for c in codons], dtype=np.int64)
    chi = np.bitwise_xor(packed[:-1], packed[1:]) & 0x3F
    return np.bincount(chi, minlength=N_CODONS).astype(np.float64) / chi.size


def window_climate(
    cds: str,
    start_codon: int,
    end_codon: int,
    *,
    ncbi_id: int = 1,
    enc: NucleotideEncoding | None = None,
    label: str = "window",
    certificates: bool = True,
) -> CarrierClimate:
    """Climate of a codon-addressed sub-window, frame preserved.

    This is what makes the physics *addressable*: a program can name an interval
    and a target climate for that interval alone.
    """
    s = clean_acgt(cds)
    if start_codon < 0 or end_codon > len(s) // 3 or end_codon <= start_codon:
        raise ValueError("codon window out of range")
    return climate_of_sequence(
        s[3 * start_codon : 3 * end_codon],
        ncbi_id=ncbi_id,
        enc=enc,
        label=label,
        certificates=certificates,
    )


# ---------------------------------------------------------------------------
# Reference climates (measured, with uncertainty)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceClimate:
    """A climate measured over a population of sequences, with uncertainty."""

    label: str
    n_sequences: int
    mean: Dict[str, float]
    stderr: Dict[str, float]
    provenance: str

    def interval(self, observable: str, k: float = 2.0) -> tuple[float, float]:
        m = self.mean.get(observable)
        s = self.stderr.get(observable)
        if m is None or s is None:
            return (float("nan"), float("nan"))
        return (m - k * s, m + k * s)

    def as_reference(self) -> "ClimateReference":
        return ClimateReference(
            label=self.label,
            values=dict(self.mean),
            stderr=dict(self.stderr),
            provenance=self.provenance or f"population mean over {self.n_sequences} sequences",
        )


@dataclass(frozen=True)
class ClimateReference:
    """A named climate vector, used as a design target.

    A program reads named values off this object and nothing else, so a target
    may come either from a measured population or from a single molecule without
    the program knowing the difference.
    """

    label: str
    values: Dict[str, float]
    stderr: Dict[str, float] = field(default_factory=dict)
    provenance: str = ""

    def get(self, name: str) -> float:
        return float(self.values.get(name, float("nan")))

    def interval(self, name: str, k: float = 2.0) -> tuple[float, float]:
        value = self.values.get(name)
        error = self.stderr.get(name)
        if value is None or error is None:
            return (float("nan"), float("nan"))
        return (value - k * error, value + k * error)


#: Published values from Sec. 8.2 of the genomics analysis. Carried for
#: comparison only; the runner re-measures them from the ingested catalogs and
#: reports both, so a drift between the two is visible rather than hidden.
DOCUMENTED_CLIMATES: Dict[str, Dict[str, float]] = {
    "ecoli_k12": {"mean_shell": 2.854, "eta": 0.0487, "m2": 4038.0},
    "yeast_s288c": {"mean_shell": 2.890, "eta": 0.0368, "m2": 4063.0},
    "human_chr22": {"mean_shell": 2.897, "eta": 0.0344, "m2": 4067.0},
}

#: The uniform (thermal) carrier register. A sequence whose climate sits here
#: carries no transport structure at this layer.
THERMAL_CLIMATE: Dict[str, float] = {"mean_shell": 3.0, "eta": 0.0, "m2": 4096.0}


def measure_reference_climate(
    sequences: Iterable[str],
    *,
    label: str,
    ncbi_id: int = 1,
    provenance: str = "",
) -> ReferenceClimate:
    """Measure a climate distribution over a population of coding sequences."""
    rows: list[CarrierClimate] = []
    for seq in sequences:
        s = clean_acgt(seq)
        if len(s) < 60 or len(s) % 3:
            continue
        try:
            rows.append(climate_of_sequence(s, ncbi_id=ncbi_id, label=label))
        except (ValueError, KeyError):
            continue
    if not rows:
        raise ValueError(f"no usable sequences for reference climate {label!r}")
    names = sorted(rows[0].observables)
    stacked = np.asarray([[r.observables[n] for n in names] for r in rows], dtype=np.float64)
    mean = {n: float(np.nanmean(stacked[:, i])) for i, n in enumerate(names)}
    stderr = {
        n: float(np.nanstd(stacked[:, i], ddof=1) / math.sqrt(len(rows)))
        if len(rows) > 1
        else float("nan")
        for i, n in enumerate(names)
    }
    return ReferenceClimate(
        label=label,
        n_sequences=len(rows),
        mean=mean,
        stderr=stderr,
        provenance=provenance,
    )


def compare_climates(
    design: CarrierClimate, reference: ReferenceClimate, observables: Sequence[str]
) -> Dict[str, tuple[float, float, bool]]:
    """``{observable: (design, reference mean, inside 2 sigma)}``."""
    out: Dict[str, tuple[float, float, bool]] = {}
    for name in observables:
        ref = reference.mean.get(name, float("nan"))
        lo, hi = reference.interval(name)
        value = design.observables.get(name, float("nan"))
        out[name] = (value, ref, bool(np.isfinite(lo) and lo <= value <= hi))
    return out


# ---------------------------------------------------------------------------
# Null models (Sec. 12). Preserve the layer whose effect is already explained.
# ---------------------------------------------------------------------------


def codons_of(seq: str) -> list[str]:
    """Split a cleaned ACGT sequence into in-frame codon triplets."""
    cleaned = clean_acgt(seq)
    return [
        cleaned[i : i + 3]
        for i in range(0, len(cleaned) - 2, 3)
        if len(cleaned[i : i + 3]) == 3
    ]


def mean_adjacent_shell(
    codons: Sequence[str],
    *,
    enc: NucleotideEncoding | None = None,
) -> float:
    """Mean adjacent codon-pair shell (kernel control meter)."""
    if len(codons) < 2:
        return float("nan")
    encoding = enc or pair_inversion_encoding()
    payloads = np.fromiter(
        (pack_codon_bits(codon, encoding) for codon in codons),
        dtype=np.int64,
        count=len(codons),
    )
    xor = np.bitwise_xor(payloads[:-1], payloads[1:])
    if hasattr(np, "bitwise_count"):
        return float(np.bitwise_count(xor).mean())
    return float(sum(int(value).bit_count() for value in xor) / len(xor))


@lru_cache(maxsize=32)
def _codon_fiber(ncbi_id: int) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """``((codon, synonyms), ...)`` for the sense codons of one translation table."""
    code = translation_table(int(ncbi_id))
    fibers: Dict[str, list[str]] = {}
    for codon, amino in code.items():
        if amino == "*":
            continue
        fibers.setdefault(amino, []).append(codon)
    return tuple(
        (codon, tuple(sorted(fibers[amino])))
        for codon, amino in sorted(code.items())
        if amino != "*"
    )


@lru_cache(maxsize=32)
def _fiber_lookup(ncbi_id: int) -> Dict[str, tuple[str, ...]]:
    return dict(_codon_fiber(int(ncbi_id)))


def protein_fixed_resample(
    cds: str, *, ncbi_id: int = 1, rng: np.random.Generator
) -> str:
    """Resample every codon uniformly inside its synonymous fiber.

    Preserves the protein exactly. This is the null that decides whether a
    designed climate is *achievable* or whether plain recoding reaches it too.
    """
    s = clean_acgt(cds)
    fibers = _fiber_lookup(ncbi_id)
    out: list[str] = []
    for i in range(0, len(s) - 2, 3):
        codon = s[i : i + 3]
        choices = fibers.get(codon)
        out.append(codon if not choices else str(choices[rng.integers(len(choices))]))
    return "".join(out) + s[3 * (len(s) // 3) :]


def gc_matched_shuffle(cds: str, *, rng: np.random.Generator) -> str:
    """Permute the nucleotide multiset, preserving composition and length."""
    s = clean_acgt(cds)
    bases = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
    permuted = bases.copy()
    rng.shuffle(permuted)
    return permuted.tobytes().decode("ascii")


def null_distribution(
    cds: str,
    observable: str,
    *,
    ncbi_id: int = 1,
    replicates: int = 64,
    seed: int = 20260909,
    kind: Literal["protein_fixed", "gc_shuffle"] = "protein_fixed",
    region_codons: tuple[int, int] | None = None,
) -> np.ndarray:
    """Null sample of one observable under a named Sec. 12 null model.

    When ``region_codons`` is given the observable is measured on that codon
    window alone, so a regionally addressed program is tested against a null on
    the region it actually claims to control.
    """
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(int(replicates)):
        s = (
            protein_fixed_resample(cds, ncbi_id=ncbi_id, rng=rng)
            if kind == "protein_fixed"
            else gc_matched_shuffle(cds, rng=rng)
        )
        try:
            if region_codons is None:
                climate = climate_of_sequence(s, ncbi_id=ncbi_id)
            else:
                climate = window_climate(
                    s, region_codons[0], region_codons[1], ncbi_id=ncbi_id
                )
            values.append(climate.observables[observable])
        except (ValueError, KeyError):
            continue
    return np.asarray(values, dtype=np.float64)


def observable_value(
    cds: str,
    observable: str,
    *,
    ncbi_id: int = 1,
    region_codons: tuple[int, int] | None = None,
) -> float:
    """Measure one observable, globally or on one addressed codon window."""
    if region_codons is None:
        return float(climate_of_sequence(cds, ncbi_id=ncbi_id).observables[observable])
    return float(
        window_climate(cds, region_codons[0], region_codons[1], ncbi_id=ncbi_id).observables[
            observable
        ]
    )


def null_percentile(value: float, null: np.ndarray) -> float:
    """Where a design sits inside a null sample, in ``[0, 1]``."""
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(value):
        return float("nan")
    return float((null < value).mean())


# ---------------------------------------------------------------------------
# Reference thermodynamics (standard method, reported not fitted)
# ---------------------------------------------------------------------------
# The hQVM climate above is the design object. The functions below are an
# independent, externally standardised readout so that a design's physical
# plausibility can be checked without trusting our own algebra. They are
# reported in the audit and are never used to steer a design.
#
# Nearest-neighbour parameters: SantaLucia 1998 unified DNA parameters, as
# tabulated in the standard literature. Units: kcal/mol, cal/(mol K).

_NN_DH = {
    "AA": -7.9, "TT": -7.9, "AT": -7.2, "TA": -7.2,
    "CA": -8.5, "TG": -8.5, "GT": -8.4, "AC": -8.4,
    "CT": -7.8, "AG": -7.8, "GA": -8.2, "TC": -8.2,
    "CG": -10.6, "GC": -9.8, "GG": -8.0, "CC": -8.0,
}
_NN_DS = {
    "AA": -22.2, "TT": -22.2, "AT": -20.4, "TA": -21.3,
    "CA": -22.7, "TG": -22.7, "GT": -22.4, "AC": -22.4,
    "CT": -21.0, "AG": -21.0, "GA": -22.2, "TC": -22.2,
    "CG": -27.2, "GC": -24.4, "GG": -19.9, "CC": -19.9,
}
_R_CAL = 1.98720425864083  # cal / (mol K)
_INIT_DH = 0.0
_INIT_DS = -5.7
_TERMINAL_AT_DH = 2.2
_TERMINAL_AT_DS = 6.9


def duplex_thermodynamics(
    seq_a: str,
    seq_b: str,
    *,
    celsius: float = 37.0,
    strand_m: float = 1e-6,
    sodium_m: float = 1.0,
) -> Dict[str, float]:
    """Nearest-neighbour duplex free energy and melting temperature.

    Standard method with the unified SantaLucia (1998) parameters, including the
    initiation term and the terminal AT penalty. The entropy carries the usual
    monovalent-salt correction ``0.368 (n-1) ln[Na+]``, so the melting
    temperature refers to the sodium concentration that is actually declared in
    the design environment. Reported as an external sanity column next to the
    hQVM climate, never used as a design objective.
    """
    a = clean_acgt(seq_a)
    b = clean_acgt(seq_b)
    # Perfect-duplex NN sums are labeled by one strand. seq_b must be that
    # strand's reverse complement (or the same sequence, for legacy callers).
    # Heteroduplex / mismatch stacks are not modeled here.
    rc = a.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    if b and b != rc and b != a:
        raise ValueError(
            "duplex_thermodynamics models a perfect duplex; "
            "seq_b must be reverse_complement(seq_a) or equal to seq_a"
        )
    n = len(a)
    if n < 2:
        nan = float("nan")
        return {"duplex_dh": nan, "duplex_ds": nan, "duplex_dg": nan, "tm_c": nan}
    dh = _INIT_DH
    ds = _INIT_DS
    for i in range(n - 1):
        key = a[i : i + 2]
        if key not in _NN_DH:
            continue
        dh += _NN_DH[key]
        ds += _NN_DS[key]
    for terminal in (a[0], a[-1]):
        if terminal in ("A", "T"):
            dh += _TERMINAL_AT_DH
            ds += _TERMINAL_AT_DS
    if sodium_m > 0.0:
        ds += 0.368 * (n - 1) * math.log(sodium_m)
    tk = celsius + 273.15
    dg = dh - tk * ds / 1000.0
    denominator = ds + _R_CAL * math.log(strand_m / 4.0)
    tm = (dh * 1000.0) / denominator - 273.15 if denominator else float("nan")
    return {
        "duplex_dh": float(dh),
        "duplex_ds": float(ds),
        "duplex_dg": float(dg),
        "tm_c": float(tm),
    }


#: Jacobson-Stockmayer loop initiation free energy at 37 C, in kcal/mol, for
#: the tabulated small loop sizes. Larger loops continue with the standard
#: 1.75-RT-per-base logarithmic growth.
_HAIRPIN_LOOP_INIT = {3: 5.4, 4: 5.6, 5: 5.7, 6: 5.4, 7: 5.6, 8: 5.8, 9: 6.4}
_LOOP_GROWTH = 1.75 * (_R_CAL * 310.15 / 1000.0)  # kcal per ln(base)


def hairpin_opening_energy(seq: str, window: tuple[int, int]) -> float:
    """Most stable local hairpin free energy inside a nucleotide window.

    For every complementary pair that can close a loop of three or more bases,
    the stem is grown outwards for as long as the pairs hold, and the free
    energy is summed over the whole stem rather than over the closing pair
    alone. Loop initiation uses the tabulated small-loop values and the
    logarithmic growth above them; the helix pays the duplex initiation term
    and the terminal AT penalty.

    Returns the most negative value found, or NaN when the window contains no
    complementary stem that reaches a negative free energy. NaN means "nothing
    found" and must not be read as zero stability.

    This is a first-order estimate with a generic loop model. It is an external
    sanity column and is never used as a design objective.
    """
    s = clean_acgt(seq)
    lo = max(0, int(window[0]))
    hi = min(len(s), int(window[1]))
    if hi - lo < 8:
        return float("nan")
    best = float("nan")
    for i in range(lo, hi - 3):
        for j in range(i + 3, hi):
            if not _pairs(s[i], s[j]):
                continue
            loop = j - i - 1
            if loop < 3:
                continue
            dg = _loop_initiation_dg(loop)
            depth = 0
            while i - depth - 1 >= lo and j + depth + 1 < hi:
                outer = _pairs(s[i - depth - 1], s[j + depth + 1])
                if not outer:
                    break
                dg += _stack_dg(s[i - depth - 1], s[i - depth])
                depth += 1
            if depth == 0:
                continue
            if s[i - depth] in ("A", "T"):
                dg += _TERMINAL_AT_DH - 310.15 * _TERMINAL_AT_DS / 1000.0
            dg += _INIT_DH - 310.15 * _INIT_DS / 1000.0
            if not math.isfinite(best) or dg < best:
                best = dg
    return float(best)


def _loop_initiation_dg(loop: int) -> float:
    """Loop-closure cost at 37 C in kcal/mol for a hairpin loop of this size."""
    if loop < 3:
        return float("inf")
    if loop <= 9:
        return float(_HAIRPIN_LOOP_INIT[loop])
    return float(_HAIRPIN_LOOP_INIT[9] + _LOOP_GROWTH * math.log(loop / 9.0))


def _stack_dg(x: str, y: str) -> float:
    """Free energy at 37 C of one base-pair step, read on the left strand.

    ``x`` and ``y`` are consecutive bases of the 5'->3' strand; the step is the
    dinucleotide ``xy``.
    """
    key = f"{x}{y}"
    if key not in _NN_DH:
        return 0.0
    return float(_NN_DH[key] - 310.15 * _NN_DS[key] / 1000.0)


def _pairs(x: str, y: str) -> bool:
    return {x, y} in ({"A", "T"}, {"G", "C"})


# ---------------------------------------------------------------------------
# Preregistered scales
# ---------------------------------------------------------------------------
# Every loss in this program divides a physical deviation by one of these
# numbers. They are fixed here, before any design is run, and they are never a
# property of the candidate being scored. A candidate-relative scale would make
# the objective move while the sequence moves.

CLIMATE_SCALES: Dict[str, float] = {
    "mean_shell": 0.10,
    "low_shell_fraction": 0.02,
    "eta": 0.010,
    "m2": 20.0,
    "spectral_centroid": 0.25,
    "high_mode_fraction": 0.05,
    "family_l1": 0.05,
    "anisotropy_spread": 0.02,
    "character_power_low": 2.0,
    "ab_plus_horizon": 0.30,
    "mean_fold_disagreement": 0.10,
    "signature_weight": 1.0,
    "signature_axis_u": 0.5,
    "signature_axis_v": 0.5,
    "signature_parity": 1.0,
    "path_noncommutativity": 2.0,
    "signature_tau_u": 8.0,
    "signature_tau_v": 8.0,
}
for _k in range(7):
    CLIMATE_SCALES[f"shell_{_k}_frac"] = 0.02
    CLIMATE_SCALES[f"radial_{_k}"] = 0.30
for _k in range(6):
    CLIMATE_SCALES[f"bit_{_k}_occupation"] = 0.03
    CLIMATE_SCALES[f"bit_{_k}_anisotropy"] = 0.03
for _k in range(4):
    CLIMATE_SCALES[f"family_{_k}_frac"] = 0.04

#: Observables that are structural certificates rather than design targets.
CERTIFICATE_OBSERVABLES: tuple[str, ...] = (
    "ab_plus_horizon",
    "depth4_parity_zero_fraction",
)

#: Primary design dials: finite path memory of the genomic byte walk.
#: Shell climate stays measurable and reportable, but is not the objective.
DESIGN_OBSERVABLES: tuple[str, ...] = (
    "mean_fold_disagreement",
    "path_noncommutativity",
    "signature_weight",
    "signature_axis_u",
    "signature_axis_v",
    "family_l1",
    "family_0_frac",
)


def scale_of(observable: str) -> float:
    return float(CLIMATE_SCALES.get(observable, 1.0))


def with_observables(climate: CarrierClimate, **updates: float) -> CarrierClimate:
    """Return a copy of a climate with named scalars replaced."""
    merged = dict(climate.observables)
    merged.update({k: float(v) for k, v in updates.items()})
    return replace(climate, observables=merged)
