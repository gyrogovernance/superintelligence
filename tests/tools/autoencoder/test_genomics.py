"""Genomics compile tests (P4).

The compile is a data-only adapter: it lifts a sequence window through the
carrier byte stream and packages the certified per-byte/per-codon fields into
the 9-layer ``GenomicCompile``. It never re-derives kernel science. These
tests therefore assert only what the AE read path itself offers:

- the nine certified layers are present with their documented keys,
- measurements are finite on a real nondegenerate window,
- the one kernel-grounded invariant the adapter is responsible for wiring: the
  depth-4 ``parity_zero_frac`` equals the kernel-exact fraction of sliding
  4-byte frames whose Omega signature has parity 0 (the certified pure-
  translation law), recomputed via ``omega_word_signature``.

The tests skip when the local catalog is absent so no external dependency is
built into the suite; run ``python -m src.tools.autoencoder.helpers.ingest_genomics``
to populate ``data/dataset_genomics/``.
"""

from __future__ import annotations

import math

import pytest

from src.api import omega_word_signature
from src.tools.autoencoder.helpers.genomics import (
    GENOMICS_DIR,
    all_nucleotide_encodings,
    compile_interval,
    genomic_byte_stream,
    read_sequence_file,
)

CDS = GENOMICS_DIR / "ecoli_k12_cds.fna.gz"

pytestmark = pytest.mark.skipif(
    not CDS.exists(),
    reason="genomics catalog absent (run helpers.ingest_genomics)",
)

EXPECTED_LAYERS = {
    "byte_fold_w",
    "fold_poles",
    "family_sheet",
    "omega_signature",
    "depth4_parity",
    "chi_shells",
    "qubec_order",
    "ab_horizon",
    "boundary_keys",
}

LAYER_KEYS = {
    "byte_fold_w": ("n_bytes", "w_residual_frac", "mean_fold_disagreement"),
    "fold_poles": ("pole_00_frac", "pole_01_frac", "pole_10_frac", "pole_11_frac"),
    "family_sheet": ("mu_0", "mu_1", "mu_2", "mu_3", "l1_uniform"),
    "omega_signature": ("parity", "tau_u_popcount", "tau_v_popcount"),
    "depth4_parity": ("parity_zero_frac",),
    "chi_shells": ("n_pairs", "mean_shell"),
    "qubec_order": ("eta", "M2"),
    "ab_horizon": ("n_pairs", "mean_ab", "mean_horizon", "ab_plus_horizon"),
    "boundary_keys": ("ATG_present", "TAA_present", "TAG_present", "TGA_present"),
}


def _compile_window(max_bases=200000):
    enc = all_nucleotide_encodings()[0]
    seq = read_sequence_file(CDS, max_bases=max_bases)
    return compile_interval(seq, enc, label="ecoli")


def _val(gc, layer, key):
    v = gc.value(layer, key)
    assert v is not None, (layer, key)
    return v


def test_layers_present_with_expected_keys():
    gc = _compile_window()
    assert set(lay.name for lay in gc.layers) == EXPECTED_LAYERS
    assert gc.n_bytes > 0 and gc.seq_len > 0
    for name, keys in LAYER_KEYS.items():
        lay = gc.layer(name)
        assert lay is not None
        assert [k for k, _ in lay.values] == list(keys), name


def test_finite_measurements_when_nondegenerate():
    gc = _compile_window()
    for layer_name, key in (
        ("byte_fold_w", "mean_fold_disagreement"),
        ("family_sheet", "l1_uniform"),
        ("chi_shells", "mean_shell"),
        ("qubec_order", "eta"),
        ("qubec_order", "M2"),
        ("ab_horizon", "mean_ab"),
        ("ab_horizon", "mean_horizon"),
    ):
        v = _val(gc, layer_name, key)
        assert not math.isnan(v), (layer_name, key)


def test_depth4_parity_is_kernel_exact_parity_zero_fraction():
    """The compile wires the kernel law: on a real window, every sliding
    4-byte frame is a pure translation (Omega signature parity 0), so the
    reported parity_zero_frac equals 1, matching an independent recomputation
    through the kernel's own ``omega_word_signature``."""
    enc = all_nucleotide_encodings()[0]
    seq = read_sequence_file(CDS, max_bases=200000)
    stream = genomic_byte_stream(seq, enc)
    assert len(stream) >= 4
    recomputed = sum(
        1 for i in range(0, len(stream) - 3)
        if omega_word_signature(stream[i : i + 4]).parity == 0
    ) / (len(stream) - 3)

    gc = compile_interval(seq, enc)
    compiled = _val(gc, "depth4_parity", "parity_zero_frac")
    assert pytest.approx(compiled, abs=1e-12) == recomputed
    assert compiled == 1.0


def test_ecoli_cds_template_matches_genomics_report() -> None:
    """The compile of the full E. coli K-12 CDS file matches the genomics
    report (mean_shell ~ 2.85, eta ~ 0.05, M2 ~ 4.0e3, depth-4 parity 1.0)."""
    from src.tools.autoencoder.helpers.genomics import compile_climate_summary

    enc = all_nucleotide_encodings()[0]
    seq = read_sequence_file(CDS, max_bases=10_000_000)
    gc = compile_interval(seq, enc, label="ecoli_cds")
    s = compile_climate_summary(gc)
    assert s["mean_shell"] == pytest.approx(2.85, abs=0.10)
    assert s["eta"] == pytest.approx(0.05, abs=0.03)
    assert s["M2"] == pytest.approx(4000.0, abs=80.0)
    assert s["depth4_parity_zero_frac"] == 1.0