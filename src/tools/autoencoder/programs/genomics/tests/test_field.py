"""Contracts for the path-field compiler."""
from __future__ import annotations

from src.api import WordSignature, compose_word_signatures
from src.tools.autoencoder.programs.genomics.synthesis.field import (
    commutator,
    compile_path_field,
    inverse_word_signature,
    is_identity,
    multiply_signatures,
)


def test_signature_inverse_is_group_inverse():
    for parity, a, b in ((0, 5, 9), (1, 5, 9), (0, 0, 0), (1, 0, 7), (1, 12, 3)):
        sig = WordSignature(parity, a, b)
        assert is_identity(multiply_signatures(sig, inverse_word_signature(sig)))


def test_commutator_of_equal_blocks_is_identity():
    sig = WordSignature(0, 3, 4)
    assert is_identity(commutator(sig, sig))


def test_commutator_composition_matches_path_product():
    left = WordSignature(1, 2, 5)
    right = WordSignature(0, 7, 1)
    path = multiply_signatures(left, right)
    assert path == compose_word_signatures(right, left)


def test_path_field_exposes_exact_signatures_not_axes():
    field = compile_path_field("ATG" + ("CTGAAA" * 16) + "TAA")
    assert "whole_tau_a" in field.observables
    assert "whole_tau_b" in field.observables
    assert "signature_axis_u" not in field.observables
    assert "axis" not in field.observables
    assert isinstance(field.whole_signature, WordSignature)
    assert isinstance(field.half_commutator.commutator, WordSignature)


def test_path_memory_responds_to_synonymous_edit():
    seq = "ATG" + ("CTGAAA" * 12) + "TAA"
    before = compile_path_field(seq)
    edited = seq[:3] + "TTG" + seq[6:]
    after = compile_path_field(edited)
    moved = (
        before.whole_signature != after.whole_signature
        or before.half_commutator.commutator != after.half_commutator.commutator
        or before.mean_fold_disagreement != after.mean_fold_disagreement
    )
    assert moved, (before.observables, after.observables)
