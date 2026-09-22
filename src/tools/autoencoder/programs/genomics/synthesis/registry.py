"""FEATURE_REGISTRY: CGM experiment census sections to suite dispositions.

Keys are experiment census section IDs from hqvm_cgm_genomics_run.py.
Duplicate IDs across scripts 6/7/8 are script-qualified (e.g. 29_s6, 29_s8).
Key 15 is script 5 Theta_kin; the run script CHECK TALLY is not keyed.
Gate-prefix notes point into the flat hqvm_cgm_genomics_gates.json name map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Disposition = Literal["included", "excluded", "infrastructure"]


@dataclass(frozen=True)
class FeatureEntry:
    key: str
    title: str
    disposition: Disposition
    script: int | None
    census_keys: tuple[str, ...]
    note: str
    gate_prefixes: tuple[str, ...] = ()


def _e(
    key: str,
    title: str,
    disposition: Disposition,
    *,
    script: int | None = None,
    census_keys: tuple[str, ...] = (),
    note: str = "",
    gate_prefixes: tuple[str, ...] = (),
) -> FeatureEntry:
    return FeatureEntry(
        key=key,
        title=title,
        disposition=disposition,
        script=script,
        census_keys=census_keys,
        note=note,
        gate_prefixes=gate_prefixes,
    )


FEATURE_REGISTRY: dict[str, FeatureEntry] = {
    "0": _e(
        "0",
        "Kernel dependency",
        "infrastructure",
        script=0,
        note="alignment context only",
        gate_prefixes=("11_kernel",),
    ),
    "1": _e(
        "1",
        "Nucleotide chart / translation quotient",
        "excluded",
        script=1,
        note="code-space / table-space",
    ),
    "2": _e(
        "2",
        "Kernel layer (packing, Omega, signatures)",
        "included",
        script=2,
        census_keys=("align", "g4"),
        note="General K4 equivariance; Super provenance capacity",
    ),
    "3": _e(
        "3",
        "Sequence windows on real DNA",
        "included",
        script=2,
        census_keys=("splice",),
        note="junction motif floors (GT/AG); Super scan floor; architecture-visible alignment",
    ),
    "4": _e(
        "4",
        "Spectrum, nulls, bundle",
        "excluded",
        script=3,
        note="code-space",
    ),
    "5": _e(
        "5",
        "Polarity, dual RC, stop tree",
        "excluded",
        script=3,
        note="code-space",
    ),
    "6": _e(
        "6",
        "W, payload-RC, box beta1",
        "excluded",
        script=5,
        note="code-space",
    ),
    "7": _e(
        "7",
        "Theta = S o F o R_block; chirality pole map",
        "excluded",
        script=4,
        note="code-space",
    ),
    "8": _e(
        "8",
        "Nested wall-breach; stage anatomy; serine antipode",
        "excluded",
        script=4,
        note="code-space",
    ),
    "9": _e(
        "9",
        "Identity across the fold; two axes; stop square",
        "excluded",
        script=4,
        note="code-space",
    ),
    "10": _e(
        "10",
        "Theta_kin; cycle intertwiners; substitution cost",
        "excluded",
        script=4,
        note="code-space",
    ),
    "11": _e(
        "11",
        "QuBEC walk; gauge selection; shell velocity",
        "included",
        script=4,
        census_keys=("climate",),
        note="Super climate_logits vs GC-shuffle; random-init is analytic channel",
    ),
    "12": _e(
        "12",
        "Splice hinge; surgery; restriction; skew",
        "included",
        script=5,
        census_keys=("splice",),
        note="junction motif floors on §20 8-bp flanks; Super scan floor",
    ),
    "13": _e(
        "13",
        "NCBI moduli; wall closure; degeneracy profile",
        "excluded",
        script=5,
        note="table-space",
    ),
    "15": _e(
        "15",
        "Theta_kin signature law (script 5)",
        "excluded",
        script=5,
        note="code-space; CHECK TALLY is not keyed",
    ),
    "16": _e(
        "16",
        "Omega walk vs GC-matched shuffles",
        "included",
        script=5,
        census_keys=("shuffle",),
        note="Super specialization label-null",
    ),
    "17": _e(
        "17",
        "Splice RC-mirror; Chargaff parity; skew walk",
        "included",
        script=5,
        census_keys=("splice",),
        note="same junction lane as key 12",
    ),
    "18": _e(
        "18",
        "Surgery moduli neighborhood",
        "excluded",
        script=5,
        note="code-space",
    ),
    "19": _e(
        "19",
        "Cycle-to-J2 ker/coker bases",
        "excluded",
        script=1,
        note="code-space",
    ),
    "20": _e(
        "20",
        "Family-sheet mu on K4",
        "included",
        script=2,
        census_keys=("splice",),
        note="AE junction reads on Analysis splice objects",
    ),
    "20b": _e(
        "20b",
        "Genealogy ingress (open transcript)",
        "excluded",
        script=2,
        note=(
            "K4 -35 vs base-shuffle n=400 AUC_trained=0.7074 AUC_random=0.6915 "
            "T-R +0.016; Super T-R +0.044; architecture-visible, not weight-carried"
        ),
    ),
    "21": _e(
        "21",
        "Constitutional poles on codon pairs",
        "included",
        script=2,
        census_keys=("climate",),
        note="Super climate_logits vs GC-shuffle; random-init is analytic channel",
    ),
    "22": _e(
        "22",
        "QuBEC order parameters on ORFs",
        "included",
        script=4,
        census_keys=("climate",),
        note="Super climate_logits vs GC-shuffle; random-init is analytic channel",
    ),
    "23": _e(
        "23",
        "Compiled ORF signatures vs GC shuffle",
        "included",
        script=6,
        census_keys=("g4", "prov", "align"),
        note=(
            "Super permutation-pair provenance on ecoli/yeast/sars/chr22; "
            "zero-shot carrier prior; domain_gap + depth_loc preflight"
        ),
    ),
    "24": _e(
        "24",
        "Plaquette defect weights on ORFs",
        "excluded",
        script=6,
        note=(
            "K4 windowed defect decode: ecoli |rho|_trained=0.1417 |rho|_random=0.1325 "
            "T-R +0.009; yeast |rho|_trained=0.0434 |rho|_random=0.0419; "
            "latents see the field, weights do not add"
        ),
    ),
    "25": _e(
        "25",
        "Flat-byte (trivial-connection) frequency",
        "excluded",
        script=6,
        note="16 fd=0 bytes are code-space; CDS vs GC flat-frac is a kernel census",
    ),
    "26": _e(
        "26",
        "Depth-4 closure on genomic frames",
        "included",
        script=6,
        census_keys=("align",),
        note=(
            "depth_loc preflight: even-depth commutators identity; "
            "odd-depth non-identity; path memory confined to odd depth"
        ),
    ),
    "27": _e(
        "27",
        "Synonymous recoding separation",
        "included",
        script=6,
        census_keys=("recode",),
        note="Super specialization label-null",
    ),
    "28": _e(
        "28",
        "Three-sector skew at ori/ter",
        "excluded",
        script=6,
        note="replicon-coordinate; see worknotes",
    ),
    "28c": _e(
        "28c",
        "Path-ordered replichore holonomy",
        "excluded",
        script=6,
        note="replicon-coordinate; see worknotes",
    ),
    "29_s6": _e(
        "29_s6",
        "REBASE Type II palindromic length parity",
        "excluded",
        script=6,
        note=(
            "palindrome-vs-shuffle Narrow T-R +0.009; length mod-4 decode is "
            "length-visible (trained AUC 0.98, random 0.99); not a weight read"
        ),
    ),
    "29_s8": _e(
        "29_s8",
        "S6 edge-character covariance",
        "excluded",
        script=8,
        note="code-space",
    ),
    "30_s6": _e(
        "30_s6",
        "Shell-walk vs hydropathy by cellular location",
        "excluded",
        script=6,
        note="no local location catalog; see worknotes",
    ),
    "30_s8": _e(
        "30_s8",
        "Stop-boundary moduli under Iso(H(6,2))",
        "excluded",
        script=8,
        note="code-space",
    ),
    "31_s7": _e(
        "31_s7",
        "Code-classification constraints on standard code",
        "excluded",
        script=7,
        note="code-space",
    ),
    "31_s8": _e(
        "31_s8",
        "NCBI wall breaches as BU positions",
        "excluded",
        script=8,
        note="table-space",
    ),
    "32_s7": _e(
        "32_s7",
        "Local moduli under full code constraints",
        "excluded",
        script=7,
        note="code-space",
    ),
    "32_s8": _e(
        "32_s8",
        "Genomic compile print",
        "infrastructure",
        script=8,
        note="GenomicCompile via genomics.py",
    ),
    "33_s7": _e(
        "33_s7",
        "Structured slices (stop square; ser bridge)",
        "excluded",
        script=7,
        note="code-space",
    ),
    "33_s8": _e(
        "33_s8",
        "Iso(H(6,2)) orbit of standard code",
        "excluded",
        script=8,
        note="code-space",
    ),
    "34": _e(
        "34",
        "tRNA identity elements vs anticodon / fold",
        "excluded",
        script=7,
        note="see worknotes; not in local AE suite",
    ),
    "35": _e(
        "35",
        "Local moduli: Met expansions; wall-aut probe",
        "excluded",
        script=7,
        note="code-space / table-space",
    ),
    "36": _e(
        "36",
        "Structured two-move generators",
        "excluded",
        script=7,
        note="code-space",
    ),
    "37": _e(
        "37",
        "NCBI tables under full code constraints",
        "excluded",
        script=7,
        note="table-space",
    ),
    "38": _e(
        "38",
        "AUT reconciliation",
        "excluded",
        script=7,
        note="code-space",
    ),
    "39": _e(
        "39",
        "Constitutional fiber side conditions",
        "excluded",
        script=7,
        note="code-space",
    ),
    "40_s7": _e(
        "40_s7",
        "Wall direct sum: H = L_sense + P_fold",
        "excluded",
        script=7,
        note="code-space",
    ),
    "40_s8": _e(
        "40_s8",
        "Singular-sector NCBI avoidance",
        "excluded",
        script=8,
        note="table-space",
    ),
    "41_s7": _e(
        "41_s7",
        "Kernel percolation ladder",
        "excluded",
        script=7,
        note="§6.1 generator alphabets are code-space; not a catalog AE read",
    ),
    "41_s8": _e(
        "41_s8",
        "Serine matched synthetase check",
        "excluded",
        script=8,
        note="external / code check",
    ),
    "42_s7": _e(
        "42_s7",
        "BU projection of commutator / difference defects",
        "excluded",
        script=7,
        note="code-space",
    ),
    "42_s8": _e(
        "42_s8",
        "Codon-pair radial channel",
        "included",
        script=8,
        census_keys=("radial",),
        note="kernel radial shell dial + AE climate; Super climate_logits vs synonymous recode remains the suite specialization read",
        gate_prefixes=("radial_super_",),
    ),
    "43": _e(
        "43",
        "Fold as Weyl reflection of BU",
        "excluded",
        script=7,
        note="code-space",
    ),
    "44": _e(
        "44",
        "Genomes through the wall decomposition",
        "excluded",
        script=7,
        note="wall algebra; covered by climate_align",
    ),
}


INCLUDED_CENSUS_KEYS: tuple[str, ...] = (
    "align",
    "g4",
    "climate",
    "prov",
    "shuffle",
    "recode",
    "radial",
    "splice",
)



def coverage_rows() -> list[tuple[str, str, str, str]]:
    """Return (key, disposition, census_keys, note) rows in registry order."""
    rows: list[tuple[str, str, str, str]] = []
    for key, entry in FEATURE_REGISTRY.items():
        cens = ",".join(entry.census_keys) if entry.census_keys else "-"
        rows.append((key, entry.disposition, cens, entry.note))
    return rows


def coverage_table_text() -> str:
    lines = [
        "registry coverage",
        "key\tdisposition\tcensus\tnote",
    ]
    for key, disp, cens, note in coverage_rows():
        lines.append(f"{key}\t{disp}\t{cens}\t{note}")
    counts: dict[str, int] = {}
    for e in FEATURE_REGISTRY.values():
        counts[e.disposition] = counts.get(e.disposition, 0) + 1
    lines.append(
        "counts\t"
        + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    return "\n".join(lines)


def assert_registry_complete() -> None:
    """Every dispositioned key resolves; included rows name known census keys.

    Also requires every battery census key to be owned by at least one included
    registry row (except coverage infrastructure).
    """
    known = set(INCLUDED_CENSUS_KEYS)
    owned: set[str] = set()
    for entry in FEATURE_REGISTRY.values():
        if entry.disposition == "included":
            for ck in entry.census_keys:
                if ck not in known:
                    raise AssertionError(
                        f"registry {entry.key}: unknown census key {ck!r}"
                    )
                owned.add(ck)
    orphan = set(INCLUDED_CENSUS_KEYS) - owned - {"annex", "coverage"}
    # preflight is extraction infrastructure owned implicitly
    orphan -= {"preflight"}
    if orphan:
        raise AssertionError(f"census keys with no included registry owner: {sorted(orphan)}")
