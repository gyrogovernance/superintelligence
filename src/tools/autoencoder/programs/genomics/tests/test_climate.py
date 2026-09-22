"""Contracts for climate measurement helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def test_hairpin_scan_finds_a_designed_stem():
    from src.tools.autoencoder.programs.genomics.synthesis.climate import (
        hairpin_opening_energy,
    )

    stem = "GCGCGC" + "TTT" + "GCGCGC"
    seq = "A" * 20 + stem + "A" * 20
    dg = hairpin_opening_energy(seq, (0, len(seq)))
    assert np.isfinite(dg) and dg < -3.5, dg
    assert not np.isfinite(hairpin_opening_energy("AAAA" * 30, (0, 120)))


def test_duplex_thermodynamics_matches_published_scale():
    from src.tools.autoencoder.programs.genomics.synthesis.climate import (
        duplex_thermodynamics,
    )

    seq = ("GC" * 30)[:60]
    out = duplex_thermodynamics(seq, seq)
    assert -140.0 < out["duplex_dg"] < -110.0, out["duplex_dg"]
    dilute = duplex_thermodynamics(seq, seq, sodium_m=0.15)
    assert dilute["tm_c"] < out["tm_c"] < 115.0, (dilute["tm_c"], out["tm_c"])


def test_fasta_header_is_not_compiled_as_sequence(tmp_path: Path):
    from src.tools.autoencoder.programs.genomics.genomics import read_sequence_file

    path = tmp_path / "gene.fna"
    path.write_text(">CG-rich header CTGAC\nACGTACGT\n", encoding="ascii")
    cleaned = read_sequence_file(path)
    assert cleaned == "ACGTACGT", cleaned


def test_local_character_centroid_is_not_constant():
    from src.tools.autoencoder.programs.genomics.synthesis.climate import (
        local_character_centroid,
    )

    flat = np.zeros(64, dtype=np.int64)
    varied = np.asarray([i % 7 for i in range(64)], dtype=np.int64)
    assert float(np.ptp(local_character_centroid(flat))) == 0.0
    assert float(np.ptp(local_character_centroid(varied))) > 0.0


def test_measure_reference_climate_reports_uncertainty_and_mean():
    from src.tools.autoencoder.programs.genomics.synthesis.climate import (
        measure_reference_climate,
    )

    sequences = [
        "ATG" + "CTGAAA" * 10 + "TAA",
        "ATG" + "TTGAAA" * 10 + "TAA",
    ]
    reference = measure_reference_climate(
        sequences, label="test", ncbi_id=11, provenance="two toy coding sequences"
    )
    assert reference.n_sequences == 2
    assert np.isfinite(reference.mean["mean_shell"])
    assert reference.stderr["mean_shell"] >= 0.0
    assert reference.provenance


def test_radial_modes_track_the_shell_histogram():
    from src.tools.autoencoder.programs.genomics.synthesis.climate import radial_modes

    narrow = np.zeros(7)
    narrow[3] = 1.0
    wide = np.full(7, 1.0 / 7.0)
    assert not np.allclose(radial_modes(narrow), radial_modes(wide))
    assert float(np.ptp(radial_modes(narrow))) > 0.0
