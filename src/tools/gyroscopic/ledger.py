"""Python side of the thin HQVMLEDS production ledger (companion to ledger.c).

The ledger file holds shared temporal-inference extras only: byte table, bin
edges, allowlist. Q1_0 signs/scales stay in the GGUF / ggml RAM.

On-disk default: data/models/Bonsai-8B-gguf/hqvm_sidecar.bin (HQVMLEDS v1).
Runtime loads it via GYRO_LEDGER_PATH (see ledger.c hqvm_sidecar_load).

Call ensure_ledger() from the run path — do not treat this as a manual export.
"""

from __future__ import annotations

import argparse
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.family import shell_uv, step_uv

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEDGER_PATH = (
    _REPO_ROOT / "data" / "models" / "Bonsai-8B-gguf" / "hqvm_sidecar.bin"
)
D = 6
N_BIN = 7

DEFAULT_ALLOW = (
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    # Arc 1 compile: embd GET_ROWS + logits MUL_MAT (NavPad §3 / Arc 4)
    "token_embd.weight",
    "output.weight",
)
ATTN_ALLOW = (
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
)


def default_ledger_path() -> Path:
    return DEFAULT_LEDGER_PATH


def build_byte_table(n_bin: int = N_BIN) -> np.ndarray:
    table = np.zeros((64, n_bin), dtype=np.int16)
    for chi0 in range(64):
        for tgt in range(n_bin):
            if shell_uv(chi0, 0, D) == tgt:
                table[chi0, tgt] = -1
                continue
            found = 0
            for bb in range(256):
                u, v = step_uv(chi0, 0, bb, D)
                if shell_uv(u, v, D) == tgt:
                    found = bb
                    break
            table[chi0, tgt] = found
    return table


def write_ledger(
    path: Path | str | None = None,
    *,
    allows: Sequence[str] | None = None,
    attn_only: bool = False,
) -> Path:
    """Write HQVMLEDS v1 ledger. Returns output path."""
    out = Path(path) if path is not None else DEFAULT_LEDGER_PATH
    if allows is None:
        allows = list(ATTN_ALLOW if attn_only else DEFAULT_ALLOW)

    edges = np.array([0.0, 26.0, 28.0, 31.0, 33.0, 36.0, 38.0, 65.0], dtype=np.float32)
    byte_table = build_byte_table(N_BIN)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"HQVMLEDS")
        f.write(struct.pack("<III", 1, N_BIN, len(allows)))
        f.write(byte_table.astype(np.int16).tobytes())
        f.write(edges.astype(np.float32).tobytes())
        for a in allows:
            b = a.encode("utf-8")
            if len(b) >= 96:
                raise ValueError(f"allow too long: {a!r}")
            f.write(struct.pack("<H", len(b)))
            f.write(b)
    return out


def ensure_ledger(
    path: Path | str | None = None,
    *,
    force: bool = False,
    allows: Sequence[str] | None = None,
    attn_only: bool = False,
) -> Path:
    """Return ledger path, writing HQVMLEDS if missing (or if force=True)."""
    out = Path(path) if path is not None else DEFAULT_LEDGER_PATH
    if force or not out.is_file():
        write_ledger(out, allows=allows, attn_only=attn_only)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Write / rebuild the thin HQVMLEDS production ledger",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_LEDGER_PATH)
    ap.add_argument("--allow", action="append", default=[])
    ap.add_argument("--attn-only", action="store_true")
    args = ap.parse_args(argv)

    if args.allow:
        allows = args.allow
    elif args.attn_only:
        allows = list(ATTN_ALLOW)
    else:
        allows = list(DEFAULT_ALLOW)

    print("hQVM LEDGER (HQVMLEDS v1)")
    print("=" * 5)
    print(f"  allows: {allows}")
    path = write_ledger(args.out, allows=allows)
    sz = path.stat().st_size
    print(f"  wrote {path}  ({sz} bytes)")
    print(f"  thin ledger (no weight copy)  {'PASS' if sz < 10000 else 'FAIL'}")
    print(f"  byte_table 64x{N_BIN} + edges + {len(allows)} allows")
    print("DONE")
    print("=" * 5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
