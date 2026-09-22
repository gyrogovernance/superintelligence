"""Load Bonsai Q1_0 rows as 4096-d vectors for the interpretability convertor.

Each hidden row is 64 tiles of 64 coordinates. Chirality and Ω reads live in
converter.py; this module only dequantizes GGUF tensors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.tools.autoencoder import paths

HIDDEN = 4096
TILE = 64
Q1_BLOCK = 128
Q1_BLOCK_BYTES = 18
Q1_ROW_BYTES = (HIDDEN // Q1_BLOCK) * Q1_BLOCK_BYTES

DEFAULT_GGUF = (
    paths.DATA_HOME.parents[3] / "data" / "models" / "Bonsai-8B-gguf" / "Bonsai-8B-Q1_0.gguf"
)


def _fp16_bits_to_f32(bits: int) -> float:
    return float(np.array([bits], dtype=np.uint16).view(np.float16)[0])


def dequant_q1_row(raw: np.ndarray) -> np.ndarray:
    """Dequantize one packed Q1_0 row. Width must be a multiple of 128."""
    out = dequant_q1_rows(np.asarray(raw, dtype=np.uint8).reshape(1, -1))
    return out[0]


def dequant_q1_rows(packed: np.ndarray) -> np.ndarray:
    """Vectorized Q1_0 dequant. packed [N, B*18] -> [N, B*128] float32."""
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    if packed.ndim != 2:
        raise ValueError(f"expected [N, nbytes], got {packed.shape}")
    n, nbytes = packed.shape
    if nbytes % Q1_BLOCK_BYTES:
        raise ValueError(f"nbytes={nbytes} not divisible by {Q1_BLOCK_BYTES}")
    n_blocks = nbytes // Q1_BLOCK_BYTES
    blocks = packed.reshape(n, n_blocks, Q1_BLOCK_BYTES)
    d_u16 = blocks[:, :, 0].astype(np.uint16) | (blocks[:, :, 1].astype(np.uint16) << 8)
    scales = d_u16.view(np.float16).astype(np.float32)
    qs = blocks[:, :, 2:].reshape(-1, 16)
    bits = np.unpackbits(qs, axis=1, bitorder="little").reshape(n, n_blocks, Q1_BLOCK)
    signed = np.where(bits > 0, scales[:, :, None], -scales[:, :, None])
    return signed.reshape(n, n_blocks * Q1_BLOCK).astype(np.float32, copy=False)


def tensor_meta(gguf_path: Path, tensor_name: str) -> dict:
    """Row count and packed width for a Q1_0 (or dense) GGUF tensor."""
    from gguf import GGMLQuantizationType, GGUFReader

    reader = GGUFReader(str(gguf_path))
    tensor = next((t for t in reader.tensors if t.name == tensor_name), None)
    if tensor is None:
        raise KeyError(tensor_name)
    data = np.asarray(tensor.data)
    qtype = GGMLQuantizationType(tensor.tensor_type)
    if qtype == GGMLQuantizationType.Q1_0:
        if data.ndim != 2:
            raise ValueError(f"{tensor_name}: expected 2-d Q1 packed, got {data.shape}")
        n_rows, row_bytes = int(data.shape[0]), int(data.shape[1])
        width = (row_bytes // Q1_BLOCK_BYTES) * Q1_BLOCK
        return {
            "name": tensor_name,
            "n_rows": n_rows,
            "row_bytes": row_bytes,
            "width": width,
            "qtype": "Q1_0",
            "segments": max(1, width // HIDDEN),
        }
    n_rows = int(tensor.shape[1]) if len(tensor.shape) == 2 else int(tensor.shape[0])
    return {
        "name": tensor_name,
        "n_rows": n_rows,
        "row_bytes": None,
        "width": HIDDEN,
        "qtype": str(qtype),
        "segments": 1,
    }


def iter_tensor_rows(
    gguf_path: Path,
    tensor_name: str,
    *,
    batch: int = 256,
    max_rows: int | None = None,
    row_offset: int = 0,
):
    """Yield batches of [B, 4096] float32 rows.

    Wide Q1 rows (width = k*4096, e.g. ffn_down at 12288) are split into k
    consecutive 4096-d segments so every convertor input is an Ω chart.
    """
    from gguf import GGMLQuantizationType, GGUFReader

    reader = GGUFReader(str(gguf_path))
    tensor = next((t for t in reader.tensors if t.name == tensor_name), None)
    if tensor is None:
        raise KeyError(tensor_name)
    data = np.asarray(tensor.data)
    qtype = GGMLQuantizationType(tensor.tensor_type)

    if qtype != GGMLQuantizationType.Q1_0:
        n_rows = int(tensor.shape[1]) if len(tensor.shape) == 2 else int(tensor.shape[0])
        flat = data.reshape(n_rows, -1).astype(np.float32)
        if flat.shape[1] != HIDDEN:
            raise ValueError(f"{tensor_name}: expected ncols={HIDDEN}, got {flat.shape}")
        end = n_rows if max_rows is None else min(n_rows, row_offset + int(max_rows))
        for i in range(row_offset, end, batch):
            yield flat[i : min(i + batch, end)]
        return

    if data.ndim != 2:
        raise ValueError(f"{tensor_name}: expected 2-d Q1 packed, got {data.shape}")
    n_rows, row_bytes = int(data.shape[0]), int(data.shape[1])
    width = (row_bytes // Q1_BLOCK_BYTES) * Q1_BLOCK
    if width % HIDDEN:
        raise ValueError(f"{tensor_name}: width {width} not divisible by {HIDDEN}")
    segments = width // HIDDEN
    end = n_rows if max_rows is None else min(n_rows, row_offset + int(max_rows))
    for i in range(row_offset, end, batch):
        packed = data[i : min(i + batch, end)]
        wide = dequant_q1_rows(packed)
        if segments == 1:
            yield wide
        else:
            # [B, k*4096] -> emit [B*k, 4096] in segment-major order per row.
            b = wide.shape[0]
            yield wide.reshape(b, segments, HIDDEN).reshape(b * segments, HIDDEN)


def load_tensor_rows(
    gguf_path: Path, tensor_name: str, max_rows: int | None
) -> np.ndarray:
    """Load up to max_rows convertor rows (4096-d). Prefer iter_tensor_rows for census."""
    chunks = list(iter_tensor_rows(gguf_path, tensor_name, batch=512, max_rows=max_rows))
    if not chunks:
        return np.empty((0, HIDDEN), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def list_q1_matrix_tensors(gguf_path: Path) -> list[str]:
    """All Q1_0 matrices whose packed width is a multiple of 4096."""
    from gguf import GGMLQuantizationType, GGUFReader

    reader = GGUFReader(str(gguf_path))
    out = []
    for t in reader.tensors:
        if GGMLQuantizationType(t.tensor_type) != GGMLQuantizationType.Q1_0:
            continue
        data = np.asarray(t.data)
        if data.ndim != 2:
            continue
        width = (int(data.shape[1]) // Q1_BLOCK_BYTES) * Q1_BLOCK
        if width % HIDDEN == 0 and width >= HIDDEN:
            out.append(t.name)
    return out
