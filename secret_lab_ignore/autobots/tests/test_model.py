"""Forward pass, gradient flow, output shape."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from secret_lab_ignore.autobots.config import HGTConfig
from secret_lab_ignore.autobots.model import HGTForCausalLM


def test_forward_shape():
    config = HGTConfig()
    model = HGTForCausalLM(config)
    batch, seq = 2, 16
    input_ids = torch.randint(0, 256, (batch, seq))
    out = model(input_ids=input_ids)
    assert out.logits.shape == (batch, seq, 256)


def test_forward_with_labels():
    config = HGTConfig()
    model = HGTForCausalLM(config)
    batch, seq = 2, 16
    input_ids = torch.randint(0, 256, (batch, seq))
    labels = input_ids.clone()
    out = model(input_ids=input_ids, labels=labels)
    assert out.loss is not None
    assert out.loss.dim() == 0


def test_generate():
    config = HGTConfig()
    model = HGTForCausalLM(config)
    model.eval()
    inp = torch.tensor([[1, 2, 3, 4, 5]])
    out = model.generate(inp, max_new_tokens=5, do_sample=False)
    assert out.shape == (1, 10)


def test_gradient_flow():
    config = HGTConfig()
    model = HGTForCausalLM(config)
    batch, seq = 2, 8
    input_ids = torch.randint(0, 256, (batch, seq))
    labels = input_ids.clone()
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()


if __name__ == "__main__":
    test_forward_shape()
    test_forward_with_labels()
    test_gradient_flow()
    print("test_model OK")
