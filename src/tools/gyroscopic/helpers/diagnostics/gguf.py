"""Minimal GGUF reader for Q1_0 tensors (bench --diag scans)."""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GGML_TYPE_Q1_0 = 41
Q1_0_BLOCK_BYTES = 18
Q1_0_BLOCK_ELEMS = 128
TILE = 64

_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

GGUF_DEFAULT_ALIGNMENT = 32

_GGML_TYPE_INFO: dict[int, tuple[int, int]] = {
    41: (18, 128),  # Q1_0
}


@dataclass(frozen=True)
class GgufTensor:
    name: str
    ggml_type: int
    dims: tuple[int, ...]
    offset: int
    nbytes: int
    padded_nbytes: int


@dataclass(frozen=True)
class GgufFile:
    alignment: int
    data_base: int
    blob: memoryview
    tensors: tuple[GgufTensor, ...]


def _pad(n: int, align: int) -> int:
    if align <= 0:
        return n
    return (n + align - 1) // align * align


def _read_string(data: memoryview, offset: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    raw = bytes(data[offset : offset + length])
    return raw.decode("utf-8", errors="replace"), offset + length


def _skip_gguf_value(data: memoryview, offset: int, value_type: int) -> int:
    if value_type == _GGUF_TYPE_STRING:
        _, offset = _read_string(data, offset)
        return offset
    if value_type in (_GGUF_TYPE_UINT8, _GGUF_TYPE_BOOL):
        return offset + 1
    if value_type in (_GGUF_TYPE_INT8, _GGUF_TYPE_INT16, _GGUF_TYPE_UINT16):
        return offset + 2
    if value_type in (_GGUF_TYPE_UINT32, _GGUF_TYPE_INT32, _GGUF_TYPE_FLOAT32):
        return offset + 4
    if value_type in (_GGUF_TYPE_UINT64, _GGUF_TYPE_INT64):
        return offset + 8
    if value_type == _GGUF_TYPE_FLOAT64:
        return offset + 8
    if value_type == _GGUF_TYPE_ARRAY:
        (elem_type,) = struct.unpack_from("<i", data, offset)
        offset += 4
        (count,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        for _ in range(count):
            offset = _skip_gguf_value(data, offset, elem_type)
        return offset
    raise ValueError(f"unsupported GGUF value type {value_type}")


def _read_gguf_value(data: memoryview, offset: int, value_type: int) -> tuple[object, int]:
    if value_type == _GGUF_TYPE_UINT32:
        (v,) = struct.unpack_from("<I", data, offset)
        return v, offset + 4
    if value_type == _GGUF_TYPE_STRING:
        return _read_string(data, offset)
    if value_type == _GGUF_TYPE_ARRAY:
        (elem_type,) = struct.unpack_from("<i", data, offset)
        offset += 4
        (count,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        items = []
        for _ in range(count):
            v, offset = _read_gguf_value(data, offset, elem_type)
            items.append(v)
        return items, offset
    _, offset = _skip_gguf_value(data, offset, value_type), offset
    return None, offset


def _ggml_nbytes(ggml_type: int, dims: tuple[int, ...]) -> int:
    type_size, blck = _GGML_TYPE_INFO[ggml_type]
    n_elem = 1
    for d in dims:
        n_elem *= int(d)
    return type_size * (n_elem // blck)


def open_gguf(gguf_path: Path) -> GgufFile:
    path = Path(gguf_path)
    raw = path.read_bytes()
    blob = memoryview(raw)
    if blob[:4].tobytes() != b"GGUF":
        raise ValueError(f"not a GGUF file: {path}")

    offset = 4
    (version, n_tensors, n_kv) = struct.unpack_from("<Iqq", blob, offset)
    offset += 20
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")

    alignment = GGUF_DEFAULT_ALIGNMENT
    for _ in range(n_kv):
        key, offset = _read_string(blob, offset)
        (value_type,) = struct.unpack_from("<i", blob, offset)
        offset += 4
        if key == "general.alignment":
            (align_val, offset) = _read_gguf_value(blob, offset, value_type)
            if isinstance(align_val, int) and align_val > 0:
                alignment = align_val
            continue
        offset = _skip_gguf_value(blob, offset, value_type)

    tensors: list[GgufTensor] = []
    for _ in range(n_tensors):
        name, offset = _read_string(blob, offset)
        (n_dims,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        dims_list: list[int] = []
        for _d in range(n_dims):
            (dim,) = struct.unpack_from("<q", blob, offset)
            offset += 8
            dims_list.append(int(dim))
        dims = tuple(dims_list)
        (ggml_type,) = struct.unpack_from("<i", blob, offset)
        offset += 4
        (tensor_offset,) = struct.unpack_from("<Q", blob, offset)
        offset += 8
        try:
            nbytes = _ggml_nbytes(ggml_type, dims)
        except (KeyError, ZeroDivisionError):
            nbytes = 0
        padded = _pad(nbytes, alignment) if nbytes else 0
        tensors.append(
            GgufTensor(
                name=name,
                ggml_type=ggml_type,
                dims=dims,
                offset=int(tensor_offset),
                nbytes=nbytes,
                padded_nbytes=padded,
            )
        )

    data_base = _pad(offset, alignment)
    return GgufFile(alignment=alignment, data_base=data_base, blob=blob, tensors=tuple(tensors))


def _dequantize_q1_blob(data: memoryview, *, n_elems: int, signs_only: bool = False) -> np.ndarray:
    n_blocks = n_elems // Q1_0_BLOCK_ELEMS
    out = np.empty(n_elems, dtype=np.float32)
    pos = 0
    for b in range(n_blocks):
        block = data[b * Q1_0_BLOCK_BYTES : (b + 1) * Q1_0_BLOCK_BYTES]
        scale = 1.0 if signs_only else float(struct.unpack_from("<e", block, 0)[0])
        qs = block[2:18]
        for i in range(Q1_0_BLOCK_ELEMS):
            byte = qs[i >> 3]
            bit = (byte >> (i & 7)) & 1
            out[pos + i] = scale if bit else -scale
        pos += Q1_0_BLOCK_ELEMS
    return out


def tensor_to_numpy(
    gf: GgufFile, tensor: GgufTensor, *, signs_only: bool = False
) -> np.ndarray:
    if tensor.ggml_type != GGML_TYPE_Q1_0:
        raise ValueError(f"tensor {tensor.name}: not Q1_0")
    start = gf.data_base + tensor.offset
    end = start + tensor.nbytes
    flat = _dequantize_q1_blob(
        gf.blob[start:end], n_elems=int(np.prod(tensor.dims)), signs_only=signs_only
    )
    if len(tensor.dims) == 1:
        return flat.reshape(tensor.dims[0])
    return flat.reshape(tensor.dims[0], tensor.dims[1])


def iter_q1_sign_bytes(gguf_path: Path, *, max_groups: int | None = None) -> Iterator[bytes]:
    """Yield 16-byte Q1_0 sign payloads (qs only) from every Q1_0 tensor."""
    gf = open_gguf(gguf_path)
    yielded = 0
    tensors = sorted(gf.tensors, key=lambda t: (0 if "blk." in t.name else 1, t.name))
    for tensor in tensors:
        if tensor.ggml_type != GGML_TYPE_Q1_0:
            continue
        start = gf.data_base + tensor.offset
        pos = start
        limit = start + tensor.nbytes
        while pos + Q1_0_BLOCK_BYTES <= limit:
            block = bytes(gf.blob[pos : pos + Q1_0_BLOCK_BYTES])
            yield block[2:18]
            pos += Q1_0_BLOCK_BYTES
            yielded += 1
            if max_groups is not None and yielded >= max_groups:
                return


def iter_q1_tiles(
    gguf_path: Path,
    *,
    max_tiles: int | None = None,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (tensor_name, 64x64 float tile) from each Q1_0 weight matrix."""
    gf = open_gguf(gguf_path)
    count = 0
    for tensor in gf.tensors:
        if tensor.ggml_type != GGML_TYPE_Q1_0:
            continue
        ne0 = tensor.dims[0]
        ne1 = tensor.dims[1] if len(tensor.dims) > 1 else 1
        if ne0 < TILE or ne1 < TILE:
            continue
        if ne0 % TILE != 0 or ne1 % TILE != 0:
            continue
        mat = tensor_to_numpy(gf, tensor, signs_only=True)
        for i1 in range(0, ne1, TILE):
            for i0 in range(0, ne0, TILE):
                tile = mat[i0 : i0 + TILE, i1 : i1 + TILE]
                if tile.shape != (TILE, TILE):
                    continue
                yield tensor.name, np.asarray(tile, dtype=np.float64)
                count += 1
                if max_tiles is not None and count >= max_tiles:
                    return
