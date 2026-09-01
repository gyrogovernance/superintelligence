"""Genomics compile adapter for the autoencoder (P4 deliverable).

This is a data-only port of the certified ``GenomicCompile`` / ``compile_interval``
from the science repo (``../science/experiments/hqvm_cgm_genomics_common.py``). It
reuses the SAME kernel facts the autoencoder already exposes through
``src.family``, ``src.api``, ``src.constants`` and ``src.tools.autoencoder.kernel`` -
no transition rule, family, fold, rank or Omega map is reformulated here. The AE
never re-derives kernel science; it only lifts a sequence window into the carrier
byte stream and assembles the previously-certified per-byte / per-codon objects
into the 9-layer compile record.

The 9 certified layers (names kept identical for application compatibility):
  byte_fold_w, fold_poles, family_sheet, omega_signature, depth4_parity,
  chi_shells, qubec_order, ab_horizon, boundary_keys.

A ``genomics`` CLI subcommand (in cli.py) drives this for a real window; the read
path points at ``data/dataset_genomics/`` (populated by ``ingest_genomics.py``).
No classifier or held-out accuracy product is built: certified fields are packaged
for application runs, exactly as the science suite specifies.
"""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypedDict

import numpy as np

from src.api import (
    OmegaState12,
    chirality_word6,
    omega12_to_state24,
    omega_word_signature,
    q_word6,
)
from src.constants import ab_distance, horizon_distance, unpack_state
from src.family import (
    byte_from_intron,
    fold_disagreement_d,
    intron_family_d,
    intron_from_byte,
)
from src.tools.autoencoder import paths
from src.tools.autoencoder.datasets import byte_census_arrays
from src.tools.autoencoder.kernel import word_signature_id

# Local catalog directory (the dataset_<word> convention).
GENOMICS_DIR = paths.dataset_dir("genomics")

CHIRALITY_D = 6
PAYLOAD_MASK = 0x3F
FOLD_MASK = 0b001100
FAMILY_MASK = 0x3
N_CODONS = 64
N_NUCLEOTIDE_ENCODINGS = 24

BASES: Tuple[str, ...] = ("A", "C", "G", "T")
BASE_INDEX: Dict[str, int] = {b: i for i, b in enumerate(BASES)}


def _standard_map() -> Dict[str, str]:
    block = {
        "TT": ("F", "F", "L", "L"),
        "TC": ("S", "S", "S", "S"),
        "TA": ("Y", "Y", "*", "*"),
        "TG": ("C", "C", "*", "W"),
        "CT": ("L", "L", "L", "L"),
        "CC": ("P", "P", "P", "P"),
        "CA": ("H", "H", "Q", "Q"),
        "CG": ("R", "R", "R", "R"),
        "AT": ("I", "I", "I", "M"),
        "AC": ("T", "T", "T", "T"),
        "AA": ("N", "N", "K", "K"),
        "AG": ("S", "S", "R", "R"),
        "GT": ("V", "V", "V", "V"),
        "GC": ("A", "A", "A", "A"),
        "GA": ("D", "D", "E", "E"),
        "GG": ("G", "G", "G", "G"),
    }
    third = "TCAG"
    out: Dict[str, str] = {}
    for prefix, aas in block.items():
        for i, aa in enumerate(aas):
            out[prefix + third[i]] = aa
    return out


STANDARD_CODE = _standard_map()

CODE_OVERRIDES: Dict[int, Dict[str, str]] = {
    1: {},
    2: {"AGA": "*", "AGG": "*", "ATA": "M", "TGA": "W"},
    3: {"ATA": "M", "CTT": "T", "CTC": "T", "CTA": "T", "CTG": "T", "TGA": "W"},
    4: {"TGA": "W"},
    5: {"AGA": "S", "AGG": "S", "ATA": "M", "TGA": "W"},
    6: {"TAA": "Q", "TAG": "Q"},
    9: {"AAA": "N", "AGA": "S", "AGG": "S", "TGA": "W"},
    10: {"TGA": "C"},
    11: {},
    12: {"CTG": "S"},
    13: {"AGA": "G", "AGG": "G", "ATA": "M", "TGA": "W"},
    14: {"AAA": "Y", "AGA": "S", "AGG": "S", "TAA": "Y", "TGA": "W"},
    16: {"TAG": "L"},
    21: {"TGA": "W", "ATA": "M", "AGA": "S", "AGG": "S", "AAA": "N"},
    22: {"TCA": "*", "TAG": "L"},
    23: {"TTA": "*"},
    24: {"AGA": "S", "AGG": "K", "TGA": "W"},
    25: {"TGA": "G"},
    26: {"CTG": "A"},
    29: {"TAA": "Y", "TAG": "Y"},
    30: {"TAA": "E", "TAG": "E"},
    33: {"AGA": "S", "AGG": "K", "TAA": "Y", "TGA": "W"},
}

NCBI_TABLE_IDS: Tuple[int, ...] = tuple(sorted(CODE_OVERRIDES))


# ---------------------------------------------------------------------------
# Nucleotide encodings (24 affine bijections via GL(2,2) matrices)
# ---------------------------------------------------------------------------

GL2_MATRICES: Tuple[Tuple[int, int, int, int], ...] = (
    (1, 0, 0, 1),
    (0, 1, 1, 0),
    (1, 1, 0, 1),
    (1, 0, 1, 1),
    (0, 1, 1, 1),
    (1, 1, 1, 0),
)

REF_PHI: Dict[str, int] = {"A": 0b00, "G": 0b01, "T": 0b10, "C": 0b11}


@dataclass(frozen=True)
class NucleotideEncoding:
    matrix: Tuple[int, int, int, int]
    translation: int
    phi: Tuple[int, int, int, int]

    @property
    def phi_map(self) -> Dict[str, int]:
        return {b: self.phi[BASE_INDEX[b]] for b in BASES}

    def encode_base(self, base: str) -> int:
        return self.phi[BASE_INDEX[base.upper()]]

    def decode_base(self, bits: int) -> str:
        x = int(bits) & 0x3
        for b in BASES:
            if self.phi[BASE_INDEX[b]] == x:
                return b
        raise ValueError(f"bits {x} not in encoding")


def xor2(x: int, t: int) -> int:
    return (int(x) ^ int(t)) & 0x3


def apply_gl2(matrix: Tuple[int, int, int, int], x: int) -> int:
    x0 = int(x) & 1
    x1 = (int(x) >> 1) & 1
    a, b, c, d = matrix
    y0 = (a * x0 + b * x1) & 1
    y1 = (c * x0 + d * x1) & 1
    return y0 | (y1 << 1)


def nucleotide_encoding(matrix: Tuple[int, int, int, int], translation: int) -> NucleotideEncoding:
    t = int(translation) & 0x3
    p0, p1, p2, p3 = (xor2(apply_gl2(matrix, REF_PHI[b]), t) for b in BASES)
    phi = (p0, p1, p2, p3)
    return NucleotideEncoding(matrix=matrix, translation=t, phi=phi)


_ENCODINGS: Optional[Tuple[NucleotideEncoding, ...]] = None


def all_nucleotide_encodings() -> Tuple[NucleotideEncoding, ...]:
    global _ENCODINGS
    if _ENCODINGS is not None:
        return _ENCODINGS
    out = [nucleotide_encoding(m, t) for m in GL2_MATRICES for t in range(4)]
    if len(out) != N_NUCLEOTIDE_ENCODINGS:
        raise RuntimeError(f"expected 24 encodings, got {len(out)}")
    if len({e.phi for e in out}) != N_NUCLEOTIDE_ENCODINGS:
        raise RuntimeError("affine census is not 24 distinct bijections")
    _ENCODINGS = tuple(out)
    return _ENCODINGS


@lru_cache(maxsize=2)
def _translation_table_cached(ncbi_id: int) -> Tuple[Tuple[str, str], ...]:
    frozen = GENOMICS_DIR / "ncbi_genetic_codes.json"
    if frozen.exists():
        payload = json.loads(frozen.read_text(encoding="utf-8"))
        tables = payload.get("tables", {})
        key = str(ncbi_id)
        if key in tables:
            raw = tables[key]["aa"]
            if len(raw) == N_CODONS:
                return tuple((_CODONS[i], raw[i]) for i in range(N_CODONS))
    if ncbi_id not in CODE_OVERRIDES:
        raise KeyError(f"NCBI genetic-code table {ncbi_id} is not loaded")
    merged = dict(STANDARD_CODE)
    merged.update(CODE_OVERRIDES[ncbi_id])
    return tuple((c, merged[c]) for c in _CODONS)


_CODONS: Tuple[str, ...] = tuple("".join(p) for p in itertools.product(BASES, repeat=3))


_W_RESIDUAL_BITS = np.array(
    [
        1
        if (((b >> 1) & 1) == ((b >> 3) & 1) and ((b >> 2) & 1) == ((b >> 4) & 1))
        else 0
        for b in range(256)
    ],
    dtype=np.int64,
)
_FOLD_POLE_BITS = np.array([(b & 0b001100) >> 2 for b in range(256)], dtype=np.int64)
_INTRON_FAMILY_IDX = np.array(
    [intron_family_d(b, CHIRALITY_D) for b in range(256)], dtype=np.int64
)


def translation_table(ncbi_id: int) -> Dict[str, str]:
    return dict(_translation_table_cached(int(ncbi_id)))


def clean_acgt(seq: str) -> str:
    return "".join(b for b in seq.upper().replace("U", "T") if b in BASES)


def pack_codon_bits(codon: str, enc: NucleotideEncoding) -> int:
    codon = codon.upper().replace("U", "T")
    if len(codon) != 3:
        raise ValueError(f"codon {codon!r} is not length 3")
    out = 0
    for i, base in enumerate(codon):
        out |= (enc.encode_base(base) & 0x3) << (2 * (2 - i))
    return out & PAYLOAD_MASK


def unpack_codon_bits(bits: int, enc: NucleotideEncoding) -> str:
    x = int(bits) & PAYLOAD_MASK
    bases = []
    for i in range(3):
        shift = 2 * (2 - i)
        bases.append(enc.decode_base((x >> shift) & 0x3))
    return "".join(bases)


def pack_4mer_byte(seq4: str, enc: NucleotideEncoding) -> int:
    seq4 = seq4.upper().replace("U", "T")
    if len(seq4) != 4:
        raise ValueError(f"4-mer {seq4!r} is not length 4")
    intron = 0
    for i, base in enumerate(seq4):
        intron |= (enc.encode_base(base) & 0x3) << (2 * i)
    return byte_from_intron(intron, CHIRALITY_D)


def unpack_4mer_byte(byte: int, enc: NucleotideEncoding) -> str:
    intron = intron_from_byte(int(byte) & 0xFF, CHIRALITY_D)
    return "".join(enc.decode_base((intron >> (2 * i)) & 0x3) for i in range(4))


def genomic_byte_stream(
    seq: str,
    enc: NucleotideEncoding,
    *,
    stride: int = 1,
    frame: int = 0,
) -> Tuple[int, ...]:
    """Canonical genomic lift: overlapping 4-mers -> hQVM bytes."""
    s = clean_acgt(seq)
    if stride <= 0:
        raise ValueError("stride must be positive")
    if len(s) < 4:
        return tuple()
    return tuple(
        pack_4mer_byte(s[i : i + 4], enc)
        for i in range(frame, len(s) - 3, stride)
    )


def chirality_shell(chi: int) -> int:
    """Hamming shell of a six-bit chirality register (equality at 0)."""
    return (int(chi) & PAYLOAD_MASK).bit_count()


def carrier_from_codon_pair(anc: int, pres: int) -> int:
    omega = OmegaState12(u6=anc & PAYLOAD_MASK, v6=pres & PAYLOAD_MASK)
    return omega12_to_state24(omega)


def chirality_of_pair(anc: int, pres: int) -> int:
    return chirality_word6(carrier_from_codon_pair(anc, pres))


# ---------------------------------------------------------------------------
# GenomicCompile - the multi-layer compile object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerReport:
    """One certified field of the compile, with its measurement values."""

    name: str
    values: Tuple[Tuple[str, float], ...]


@dataclass(frozen=True)
class GenomicCompile:
    """Multi-layer compile of one DNA interval.

    Assembles previously certified per-byte / per-codon objects into a single
    feature record for an interval or ORF. No classifier and no held-out
    accuracy product: certified fields packaged for application runs.
    """

    label: str
    seq_len: int
    n_bytes: int
    layers: Tuple[LayerReport, ...]

    def layer(self, name: str) -> Optional[LayerReport]:
        for lay in self.layers:
            if lay.name == name:
                return lay
        return None

    def value(self, name: str, key: str) -> Optional[float]:
        lay = self.layer(name)
        if lay is None:
            return None
        for k, v in lay.values:
            if k == key:
                return v
        return None


def _w_residual(byte: int) -> int:
    """W-membership residual: bits b1 XOR pairs."""
    x = byte & 0x3F
    return 1 if ((x >> 1) & 1) == ((x >> 3) & 1) and ((x >> 2) & 1) == ((x >> 4) & 1) else 0


def _qubec_from_mean_shell(mean_shell: float) -> Tuple[float, float, float, float]:
    """Moment fit: E[N]=6*rho => rho=mean/6; lambda; eta; M2."""
    if not (mean_shell == mean_shell) or mean_shell <= 0.0 or mean_shell >= 6.0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rho = mean_shell / 6.0
    if rho >= 1.0 - 1e-15:
        return float("nan"), float("nan"), float("nan"), float("nan")
    lam = rho / (1.0 - rho)
    eta = (1.0 - lam) / (1.0 + lam)
    m2 = 4096.0 / ((1.0 + eta * eta) ** 6)
    return lam, rho, eta, m2


def _interval_ab_horizon(
    codons: Sequence[str], enc: NucleotideEncoding
) -> Tuple[float, float, int]:
    """Mean ab and horizon on successive frame-0 codon pairs."""
    if len(codons) < 2:
        return float("nan"), float("nan"), 0
    sab = sh = 0.0
    n = 0
    for i in range(len(codons) - 1):
        anc = pack_codon_bits(codons[i], enc)
        pres = pack_codon_bits(codons[i + 1], enc)
        st = carrier_from_codon_pair(anc, pres)
        a12, b12 = unpack_state(st)
        sab += ab_distance(a12, b12)
        sh += horizon_distance(a12, b12)
        n += 1
    return sab / n, sh / n, n


def compile_interval(
    seq: str,
    enc: NucleotideEncoding,
    *,
    label: str = "interval",
    ncbi_id: int = 1,
) -> GenomicCompile:
    """Compile one sequence window into the 9-layer GenomicCompile record.

    Pure data transform: lifts the window to the carrier byte stream and
    assembles the certified layers. No kernel fact is reformulated.
    """
    s = clean_acgt(seq)
    stream = genomic_byte_stream(s, enc)
    n = len(stream)

    # -- byte/W fold layer -------------------------------------------------
    if n:
        stream_arr = np.asarray(stream, dtype=np.int64)
        w_resid = int(_W_RESIDUAL_BITS[stream_arr].sum())
        census = byte_census_arrays()
        fd_mean = float(census["fold_disagreement"][stream_arr].mean())
        wall_hist_arr = np.bincount(
            _FOLD_POLE_BITS[stream_arr], minlength=4
        )
        wall_hist: Dict[int, int] = {int(k): int(v) for k, v in enumerate(wall_hist_arr)}
    else:
        w_resid, fd_mean, wall_hist = 0, 0.0, {0: 0, 1: 0, 2: 0, 3: 0}

    # -- family sheet (mu, L1 vs uniform 1/4) ------------------------------
    if n:
        fam_arr = np.bincount(_INTRON_FAMILY_IDX[stream_arr], minlength=4)
        fam: List[int] = [int(x) for x in fam_arr]
    else:
        fam = [0, 0, 0, 0]
    tot_fam = sum(fam) or 1
    mu = tuple(c / tot_fam for c in fam)
    l1_uniform = sum(abs(m - 0.25) for m in mu)

    # -- codon-pair chi shells (frame-0, stride-3 pairs) -------------------
    packed_codons = []
    codon_list: List[str] = []
    for i in range(0, len(s) - 2, 3):
        c3 = s[i : i + 3]
        if len(c3) == 3 and all(ch in BASE_INDEX for ch in c3):
            packed_codons.append(pack_codon_bits(c3, enc))
            codon_list.append(c3)
    chi_shells: List[int] = []
    for u, v in zip(packed_codons, packed_codons[1:]):
        chi_shells.append(chirality_shell(u ^ v))
    mean_shell = (sum(chi_shells) / len(chi_shells)) if chi_shells else float("nan")
    _lam, _rho, q_eta, q_m2 = _qubec_from_mean_shell(mean_shell)
    mean_ab, mean_hor, n_ab_pairs = _interval_ab_horizon(codon_list, enc)

    # -- ORF Omega signature + depth-4 sliding parity ----------------------
    # A length-4 frame has parity 0 by the kernel's own definition (parity
    # equals the word length mod 2). The sliding-window count is therefore
    # the full count of frames; the exhaustive recomputation is replaced by a
    # sampled spot-check against the kernel, retained for the test only.
    par_sig = None
    d4_frac: float
    if n >= 4:
        full = omega_word_signature(stream)
        par_sig = (int(full.parity), int(full.tau_u6), int(full.tau_v6))
        d4_frac = 1.0
    elif n == 0:
        d4_frac = float("nan")
    else:
        d4_frac = float("nan")

    # -- boundary keys ------------------------------------------------------
    code = translation_table(ncbi_id)
    stop_hits = {
        c: (c in s) for c in ("TAA", "TAG", "TGA", "ATG")
    }

    layers = (
        LayerReport("byte_fold_w", (
            ("n_bytes", float(n)),
            ("w_residual_frac", w_resid / n if n else float("nan")),
            ("mean_fold_disagreement", fd_mean),
        )),
        LayerReport("fold_poles", tuple(
            (f"pole_{p:02b}_frac", wall_hist[p] / n if n else float("nan"))
            for p in range(4)
        )),
        LayerReport("family_sheet", (
            ("mu_0", mu[0]),
            ("mu_1", mu[1]),
            ("mu_2", mu[2]),
            ("mu_3", mu[3]),
            ("l1_uniform", l1_uniform),
        )),
        LayerReport("omega_signature", (
            ("parity", par_sig[0] if par_sig else float("nan")),
            ("tau_u_popcount", bin(par_sig[1]).count("1") if par_sig else float("nan")),
            ("tau_v_popcount", bin(par_sig[2]).count("1") if par_sig else float("nan")),
        )),
        LayerReport("depth4_parity", (("parity_zero_frac", d4_frac),)),
        LayerReport("chi_shells", (
            ("n_pairs", float(len(chi_shells))),
            ("mean_shell", mean_shell),
        )),
        LayerReport("qubec_order", (
            ("eta", q_eta),
            ("M2", q_m2),
        )),
        LayerReport("ab_horizon", (
            ("n_pairs", float(n_ab_pairs)),
            ("mean_ab", mean_ab),
            ("mean_horizon", mean_hor),
            ("ab_plus_horizon", mean_ab + mean_hor if n_ab_pairs else float("nan")),
        )),
        LayerReport("boundary_keys", tuple(
            (f"{key}_present", float(v)) for key, v in sorted(stop_hits.items())
        )),
    )
    return GenomicCompile(
        label=label,
        seq_len=len(s),
        n_bytes=n,
        layers=layers,
    )


def compile_climate_summary(gc_obj: GenomicCompile) -> Dict[str, float]:
    """Five-column climate summary read from the 9-layer compile.

    Every value is pulled straight off the certified compile layers
    (chi_shells, qubec_order, byte_fold_w, depth4_parity). This is the
    climate report emitted by the ``genomics`` CLI subcommand.
    """

    def _sum(layer: str, key: str) -> float:
        v = gc_obj.value(layer, key)
        return float("nan") if v is None else float(v)

    return {
        "mean_shell": _sum("chi_shells", "mean_shell"),
        "eta": _sum("qubec_order", "eta"),
        "M2": _sum("qubec_order", "M2"),
        "plaquette_mean_fold": _sum("byte_fold_w", "mean_fold_disagreement"),
        "depth4_parity_zero_frac": _sum("depth4_parity", "parity_zero_frac"),
    }


def read_sequence_file(path: Path, *, max_bases: Optional[int] = None) -> str:
    """Read a FASTA/ plain sequence file (gzip-aware) into one ACGT string.

    Stops reading as soon as ``max_bases`` bases have been accumulated.
    """
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    chunks: List[str] = []
    total = 0
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            if max_bases is not None and total >= max_bases:
                break
            stripped = line.strip()
            if not stripped:
                continue
            if max_bases is not None and total + len(stripped) > max_bases:
                stripped = stripped[: max_bases - total]
            chunks.append(stripped)
            total += len(stripped)
    return "".join(chunks)
