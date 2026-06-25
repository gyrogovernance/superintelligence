#!/usr/bin/env python3
"""GGUF reader and tile projection tests."""

from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    for candidate in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("repo root not found")


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.gyroscopic.helpers.diagnostics import (
    GGML_TYPE_Q1_0,
    Q1_0_BLOCK_BYTES,
    decompose_tile,
    open_gguf,
    project_chi,
    project_shell,
    random_tile_reference,
)


def _write_minimal_q1_gguf(path: Path, *, n: int = 128) -> None:
    """One Q1_0 tensor shape [128] = one block, for parser smoke test."""
    blocks = 1
    data = bytearray()
    scale = struct.pack("<e", 1.0)
    qs = bytes([0xAA] * 16)
    data.extend(scale + qs)

    kv_blob = bytearray()
    key = b"general.alignment"
    kv_blob.extend(struct.pack("<Q", len(key)))
    kv_blob.extend(key)
    kv_blob.extend(struct.pack("<i", 4))  # UINT32
    kv_blob.extend(struct.pack("<I", 32))

    tensor_blob = bytearray()
    tname = b"test.weight"
    tensor_blob.extend(struct.pack("<Q", len(tname)))
    tensor_blob.extend(tname)
    tensor_blob.extend(struct.pack("<I", 1))  # n_dims
    tensor_blob.extend(struct.pack("<q", n))
    tensor_blob.extend(struct.pack("<i", GGML_TYPE_Q1_0))
    tensor_blob.extend(struct.pack("<Q", 0))

    header = bytearray(b"GGUF")
    header.extend(struct.pack("<Iqq", 3, 1, 1))
    body = bytes(kv_blob + tensor_blob)
    pad = (32 - (len(header) + len(body)) % 32) % 32
    out = header + body + bytes(pad) + bytes(data)
    path.write_bytes(out)


def test_open_gguf_minimal() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "tiny.gguf"
        _write_minimal_q1_gguf(p)
        gf = open_gguf(p)
        assert len(gf.tensors) == 1
        assert gf.tensors[0].ggml_type == GGML_TYPE_Q1_0
        assert gf.tensors[0].nbytes == Q1_0_BLOCK_BYTES


def test_circulant_projection_is_idempotent() -> None:
    rng = np.random.default_rng(1)
    W = rng.normal(size=(64, 64))
    p = project_chi(W)
    p2 = project_chi(p)
    assert np.allclose(p, p2, atol=1e-10)


def test_shell_matrix_high_r_shell() -> None:
    idx = np.arange(64, dtype=np.uint8)
    shell = np.bitwise_xor(idx[:, None], idx[None, :]).astype(np.uint8)
    pop = np.zeros(shell.shape, dtype=np.float64)
    for r in range(8):
        pop += ((shell >> r) & 1).astype(np.float64)
    W = pop
    d = decompose_tile(W)
    assert d["r_shell"] > 0.99
    assert d["r_defect"] < 0.01


def test_random_reference_has_defect() -> None:
    ref = random_tile_reference(64)
    assert ref["r_defect"]["mean"] > 0.5


def test_real_gguf_if_present() -> None:
    try:
        from src.tools.gyroscopic.config import get_gyroscopic_llm_config, resolve_gguf_path
    except Exception:
        return
    path = resolve_gguf_path(get_gyroscopic_llm_config())
    if not path.is_file():
        return
    gf = open_gguf(path)
    q1 = [t for t in gf.tensors if t.ggml_type == GGML_TYPE_Q1_0]
    assert q1, "expected at least one Q1_0 tensor"
    assert gf.data_base + q1[0].offset + q1[0].nbytes <= len(gf.blob)
