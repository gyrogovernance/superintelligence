from __future__ import annotations

"""
GYRG on-disk layout (internal):
  magic "GYRG", header IIII (format_tag, n_cells, flags, reserved), 32-byte
  kernel-law digest, 32-byte kernel digest, then n_cells * _CELL_V2_SIZE
  bytes per cell (see _pack_cell_v2). Format is fixed.
"""

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_MAGIC = b"GYRG"
_FORMAT_TAG = 1
_CELL_V2_SIZE = 184
# Explicit v2 padding (little-endian wire); keep sizes in sync with _CELL_V2_SIZE.
_PAD_AFTER_LAST_BYTE = 7
_PAD_AFTER_PARITY3 = 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def compute_kernel_digest() -> bytes:
    root = _repo_root()
    k = root / "src" / "kernel.py"
    c = root / "src" / "constants.py"
    data = k.read_bytes() + b"\n|||\n" + c.read_bytes()
    return hashlib.sha256(data).digest()


def compute_kernel_law_digest() -> bytes:
    root = _repo_root()
    k = root / "src" / "kernel.py"
    c = root / "src" / "constants.py"
    data = k.read_bytes() + b"\n|||\n" + c.read_bytes()
    return hashlib.sha256(data).digest()


def _u16_row(src: Any, n: int) -> tuple[int, ...]:
    if src is None:
        return (0,) * n
    if hasattr(src, "__iter__") and not isinstance(src, (bytes, str)):
        seq = list(src)
    else:
        return (0,) * n
    if len(seq) < n:
        seq = list(seq) + [0] * (n - len(seq))
    out = []
    for i in range(n):
        out.append(int(seq[i]) & 0xFFFF)
    return tuple(out)


def _u8_row(src: Any, n: int) -> tuple[int, ...]:
    if src is None:
        return (0,) * n
    seq = list(src) if hasattr(src, "__iter__") and not isinstance(src, (bytes, str)) else []
    if len(seq) < n:
        seq = list(seq) + [0] * (n - len(seq))
    return tuple(int(seq[i]) & 0xFF for i in range(n))


def _pack_cell_v2(c: Any) -> bytes:
    omega = int(getattr(c, "omega12", 0))
    step = int(getattr(c, "step", 0)) & 0xFFFFFFFFFFFFFFFF
    lb = int(getattr(c, "last_byte", 0)) & 0xFF
    chi = _u16_row(getattr(c, "chi_hist64", None), 64)
    shell = _u16_row(getattr(c, "shell_hist7", None), 7)
    fam = _u8_row(getattr(c, "family_hist4", None), 4)
    osig = int(getattr(c, "omega_sig", 0))
    pO = int(getattr(c, "parity_O12", 0)) & 0xFFFF
    pE = int(getattr(c, "parity_E12", 0)) & 0xFFFF
    pb = int(getattr(c, "parity_bit", 0)) & 0xFF
    rk = int(getattr(c, "resonance_key", 0)) & 0xFFFFFFFF

    buf = bytearray()
    buf.extend(struct.pack("<qQ", omega, step))
    buf.extend(struct.pack("<B", lb))
    buf.extend(b"\x00" * _PAD_AFTER_LAST_BYTE)
    buf.extend(struct.pack("<" + "H" * 64, *chi))
    buf.extend(struct.pack("<" + "H" * 7, *shell))
    buf.extend(struct.pack("<" + "B" * 4, *fam))
    buf.extend(struct.pack("<i", osig))
    buf.extend(struct.pack("<HHB", pO, pE, pb))
    buf.extend(b"\x00" * _PAD_AFTER_PARITY3)
    buf.extend(struct.pack("<I", rk))
    return bytes(buf)


def _unpack_cell_v2(raw: bytes, offset: int) -> tuple[dict[str, Any], int]:
    end = offset + _CELL_V2_SIZE
    chunk = raw[offset:end]
    if len(chunk) != _CELL_V2_SIZE:
        raise ValueError("truncated cell record")
    o = 0
    omega, step = struct.unpack_from("<qQ", chunk, o)
    o += 16
    lb = struct.unpack_from("<B", chunk, o)[0]
    o += 1 + _PAD_AFTER_LAST_BYTE
    chi = struct.unpack_from("<" + "H" * 64, chunk, o)
    o += 128
    shell = struct.unpack_from("<" + "H" * 7, chunk, o)
    o += 14
    fam = struct.unpack_from("<" + "B" * 4, chunk, o)
    o += 4
    osig = struct.unpack_from("<i", chunk, o)[0]
    o += 4
    pO, pE, pb = struct.unpack_from("<HHB", chunk, o)
    o += 5 + _PAD_AFTER_PARITY3
    rk = struct.unpack_from("<I", chunk, o)[0]
    cell = {
        "omega12": int(omega),
        "step": int(step),
        "last_byte": int(lb),
        "chi_hist64": list(chi),
        "shell_hist7": list(shell),
        "family_hist4": list(fam),
        "omega_sig": int(osig),
        "parity_O12": int(pO),
        "parity_E12": int(pE),
        "parity_bit": int(pb),
        "resonance_key": int(rk),
    }
    return cell, end


def snapshot(path: str, cells: list[Any]) -> None:
    digest = compute_kernel_digest()
    law_digest = compute_kernel_law_digest()
    n = len(cells)
    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<IIII", _FORMAT_TAG, n, 0, 0))
        f.write(law_digest)
        f.write(digest)
        for c in cells:
            f.write(_pack_cell_v2(c))


@dataclass(frozen=True)
class GyrographSnapshot:
    version: int
    kernel_digest: bytes
    cells: tuple[dict[str, Any], ...]


def restore(
    path: str,
    *,
    verify_kernel: bool = True,
) -> GyrographSnapshot:
    raw = Path(path).read_bytes()
    if len(raw) < 4 + 16:
        raise ValueError("file too small")
    if raw[:4] != _MAGIC:
        raise ValueError("bad magic")
    fmt, n, _flags, _rsv = struct.unpack_from("<IIII", raw, 4)
    if fmt != _FORMAT_TAG:
        raise ValueError(f"unsupported snapshot format tag {fmt}")

    need = 84 + n * _CELL_V2_SIZE
    if len(raw) < need:
        raise ValueError("truncated v3 snapshot")
    law_digest = raw[20:52]
    digest = raw[52:84]
    if verify_kernel and law_digest != compute_kernel_law_digest():
        raise ValueError("kernel_law_digest mismatch")
    if verify_kernel and digest != compute_kernel_digest():
        raise ValueError("kernel_digest mismatch")
    cells = []
    off = 84
    for _ in range(n):
        cell, off = _unpack_cell_v2(raw, off)
        cells.append(cell)
    if off != len(raw):
        raise ValueError("trailing bytes in snapshot")
    return GyrographSnapshot(_FORMAT_TAG, bytes(digest), tuple(cells))


__all__ = ["GyrographSnapshot", "compute_kernel_digest", "compute_kernel_law_digest", "restore", "snapshot"]
