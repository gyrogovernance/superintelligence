"""Verify projection exactness on Bolmo weights."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from secret_lab_ignore.autobots.config import HGTConfig
from secret_lab_ignore.autobots.model import HGTForCausalLM
from secret_lab_ignore.autobots.convert import StructuralProjector


def test_structural_basis_shape():
    config = HGTConfig()
    model = HGTForCausalLM(config)
    proj = StructuralProjector(model)
    basis = proj.compute_structural_basis()
    assert basis.shape == (256, config.resolution_dims[0])


def test_project_decomposition():
    config = HGTConfig()
    model = HGTForCausalLM(config)
    proj = StructuralProjector(model)
    dim = config.resolution_dims[0]
    target = torch.randn(100, dim)
    E_struct, E_res = proj.project_token_embeddings(target, None)
    assert E_struct.shape == target.shape
    assert E_res.shape == target.shape
    recon = E_struct + E_res
    assert torch.allclose(recon, target, atol=1e-5)
    E_struct2, _ = proj.project_token_embeddings(E_struct, None)
    assert torch.allclose(E_struct2, E_struct, atol=1e-5)


if __name__ == "__main__":
    test_structural_basis_shape()
    test_project_decomposition()
    print("test_convert OK")
