"""Verify physics survives training unchanged."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from secret_lab_ignore.autobots import physics
from secret_lab_ignore.autobots.config import HGTConfig
from secret_lab_ignore.autobots.model import HGTForCausalLM


def test_mask12_buffer_unchanged_after_forward():
    config = HGTConfig()
    model: HGTForCausalLM = HGTForCausalLM(config)
    mask12: torch.Tensor = getattr(model, "mask12_table")
    expected = mask12.clone()
    _ = model(torch.randint(0, 256, (2, 8)))
    assert torch.equal(getattr(model, "mask12_table"), expected)


def test_mask12_buffer_unchanged_after_backward():
    config = HGTConfig()
    model: HGTForCausalLM = HGTForCausalLM(config)
    mask12: torch.Tensor = getattr(model, "mask12_table")
    expected = mask12.clone()
    out = model(
        input_ids=torch.randint(0, 256, (2, 8)),
        labels=torch.randint(0, 256, (2, 8)),
    )
    out.loss.backward()
    assert torch.equal(getattr(model, "mask12_table"), expected)


if __name__ == "__main__":
    test_mask12_buffer_unchanged_after_forward()
    test_mask12_buffer_unchanged_after_backward()
    print("test_lossless OK")
