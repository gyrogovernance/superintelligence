"""Verify physics matches src.router.constants."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from secret_lab_ignore.autobots import physics
from src.router.constants import (
    GENE_MIC_S,
    mask12_for_byte,
    vertex_charge_from_mask,
)


def test_intron_matches():
    for b in [0, 0xAA, 0x55, 255]:
        assert physics.intron(b) == ((b & 0xFF) ^ GENE_MIC_S)


def test_mask12_matches():
    for b in range(256):
        m_phys = physics.expand_intron_to_mask12(physics.intron(b))
        m_router = mask12_for_byte(b)
        assert m_phys == m_router, f"byte {b}: {m_phys} != {m_router}"


def test_vertex_charge_matches():
    for b in range(256):
        m = physics.expand_intron_to_mask12(physics.intron(b))
        v_phys = physics.vertex_charge(m)
        v_router = vertex_charge_from_mask(m)
        assert v_phys == v_router


def test_mask12_table_shape():
    tab = physics.compute_mask12_table()
    assert tab.shape == (256,)
    assert tab.dtype == torch.int32


def test_l1_trajectory():
    introns = torch.tensor([[1, 2, 3]], dtype=torch.uint8)
    out = physics.compute_l1_trajectory(introns)
    assert out.shape == (1, 3)


if __name__ == "__main__":
    test_intron_matches()
    test_mask12_matches()
    test_vertex_charge_matches()
    test_mask12_table_shape()
    test_l1_trajectory()
    print("test_physics OK")
