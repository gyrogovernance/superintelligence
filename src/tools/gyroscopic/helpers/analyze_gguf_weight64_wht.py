#!/usr/bin/env python3
"""
Phase A: scan Q8_0 tensors in a GGUF, form 64-wide vectors from consecutive
Q8_0 pairs along the quantized (last) axis, run WHT64, report energy
concentration and top-k reconstruction error.

Uses vendored llama.cpp gguf-py (no extra pip package).
"""

# pyright: reportMissingImports=false
# gguf: loaded from external/llama.cpp/gguf-py (sys.path); external/ is pyright-excluded.

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
GGUF_PY = REPO_ROOT / "external" / "llama.cpp" / "gguf-py"
if GGUF_PY.is_dir() and str(GGUF_PY) not in sys.path:
    sys.path.insert(0, str(GGUF_PY))

from gguf.constants import GGMLQuantizationType  # noqa: E402
from gguf.gguf_reader import GGUFReader  # noqa: E402
from gguf.quants import Q8_0  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from src.tools.gyroscopic.helpers.weight64_wht import (  # noqa: E402
    WHT_SIZE,
    wht_64_batch,
    topk_energy_fractions,
    topk_reconstruction_rel_l2,
)


KS = (1, 4, 8, 16)


@dataclass
class TensorAgg:
    name: str
    layer_key: str
    n_blocks: int = 0
    sum_top: dict[int, float] = field(default_factory=dict)
    sum_err: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for k in KS:
            self.sum_top[k] = 0.0
            self.sum_err[k] = 0.0

    def add_blocks(self, blocks64: np.ndarray, chunk_size: int | None = None) -> None:
        """blocks64: (n, 64) float. If chunk_size is set, process in slices (bounded RAM)."""
        if blocks64.size == 0:
            return
        n = int(blocks64.shape[0])
        if chunk_size is None or n <= chunk_size:
            self._add_blocks_once(blocks64)
            return
        for start in range(0, n, chunk_size):
            self._add_blocks_once(blocks64[start : start + chunk_size])

    def _add_blocks_once(self, blocks64: np.ndarray) -> None:
        n = int(blocks64.shape[0])
        c = wht_64_batch(blocks64)
        fr = topk_energy_fractions(c, KS)
        er = topk_reconstruction_rel_l2(blocks64, c, KS)
        self.n_blocks += n
        for k in KS:
            self.sum_top[k] += fr[k] * n
            self.sum_err[k] += er[k] * n

    def means(self) -> tuple[dict[int, float], dict[int, float]]:
        if self.n_blocks == 0:
            z = {k: float("nan") for k in KS}
            return z, z
        inv = 1.0 / float(self.n_blocks)
        return (
            {k: self.sum_top[k] * inv for k in KS},
            {k: self.sum_err[k] * inv for k in KS},
        )


def layer_key_from_tensor_name(name: str) -> str:
    m = re.match(r"^blk\.(\d+)\.", name)
    if m:
        return f"L{m.group(1)}"
    return "non-blk"


def row_uint8_to_blocks64(row: np.ndarray) -> np.ndarray:
    """One row of Q8_0 bytes (last storage dim); return (n_pair, 64) float64."""
    bsz = Q8_0.type_size  # 34
    if row.size % bsz != 0:
        return np.zeros((0, WHT_SIZE), dtype=np.float64)
    n_b = row.size // bsz
    n_pair = n_b // 2
    if n_pair == 0:
        return np.zeros((0, WHT_SIZE), dtype=np.float64)
    blocks = row[: n_pair * 2 * bsz].reshape(n_pair * 2, bsz)
    f32 = Q8_0.dequantize_blocks(blocks)
    return f32.reshape(n_pair, WHT_SIZE).astype(np.float64, copy=False)


def analyze_tensor_q8_0(t, chunk_blocks: int) -> TensorAgg | None:
    if t.tensor_type != GGMLQuantizationType.Q8_0:
        return None
    raw = t.data
    if raw.ndim < 1:
        return None
    layer = layer_key_from_tensor_name(t.name)
    lead = int(np.prod(raw.shape[:-1])) if raw.ndim else 0
    row_len = raw.shape[-1]
    flat = raw.reshape(lead, row_len)
    agg = TensorAgg(name=t.name, layer_key=layer)
    buf: list[np.ndarray] = []
    buf_n = 0

    def flush_buf() -> None:
        nonlocal buf, buf_n
        if not buf:
            return
        merged = np.concatenate(buf, axis=0) if len(buf) > 1 else buf[0]
        agg.add_blocks(merged, chunk_size=chunk_blocks)
        buf.clear()
        buf_n = 0

    for ri in range(lead):
        blk = row_uint8_to_blocks64(flat[ri])
        if blk.shape[0] == 0:
            continue
        bn = int(blk.shape[0])
        if bn >= chunk_blocks:
            flush_buf()
            agg.add_blocks(blk, chunk_size=chunk_blocks)
            continue
        if buf_n + bn > chunk_blocks:
            flush_buf()
        buf.append(blk)
        buf_n += bn
    flush_buf()
    if agg.n_blocks == 0:
        return None
    return agg


def write_report(path: Path, gguf_path: Path, per_tensor: list[TensorAgg]) -> None:
    by_layer: dict[str, list[TensorAgg]] = defaultdict(list)
    for a in per_tensor:
        by_layer[a.layer_key].append(a)

    lines: list[str] = [
        "# Weight 64-wide WHT analysis (Phase A)",
        "",
        f"Source GGUF: `{gguf_path}`",
        "",
        "Method: for each Q8_0 tensor, scan storage rows along the packed last "
        "dimension. Each consecutive pair of Q8_0 blocks (2 x 32 quants) forms "
        "one 64-float vector. Apply orthonormal WHT64; report mean top-k spectral "
        "energy fraction and mean relative L2 reconstruction error when keeping "
        "only top-k coefficients (by magnitude in WHT domain). "
        "Large tensors are processed in chunks (see script `--chunk-blocks`) so RAM "
        "stays bounded.",
        "",
        "## Global",
        "",
    ]
    total_blocks = sum(a.n_blocks for a in per_tensor)
    lines.append(f"- Q8_0 tensors analyzed: {len(per_tensor)}")
    lines.append(f"- Total 64-wide blocks: {total_blocks}")
    lines.append("")

    lines.append("## Per-layer (aggregated over tensors in layer bucket)")
    lines.append("")
    lines.append(
        "| layer | blocks | top1 E | top4 E | top8 E | top16 E | "
        "relL2 k=1 | k=4 | k=8 | k=16 |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    layer_keys = sorted(by_layer.keys(), key=lambda s: (0, int(s[1:])) if s.startswith("L") else (1, s))
    for lk in layer_keys:
        aggs = by_layer[lk]
        nb = sum(a.n_blocks for a in aggs)
        if nb == 0:
            continue
        sum_top = {k: 0.0 for k in KS}
        sum_err = {k: 0.0 for k in KS}
        for a in aggs:
            mt, me = a.means()
            w = a.n_blocks
            for k in KS:
                sum_top[k] += mt[k] * w
                sum_err[k] += me[k] * w
        inv = 1.0 / float(nb)
        t1, t4, t8, t16 = (sum_top[k] * inv for k in KS)
        e1, e4, e8, e16 = (sum_err[k] * inv for k in KS)
        lines.append(
            f"| {lk} | {nb} | {t1:.4f} | {t4:.4f} | {t8:.4f} | {t16:.4f} | "
            f"{e1:.4f} | {e4:.4f} | {e8:.4f} | {e16:.4f} |"
        )
    lines.append("")

    lines.append("## Per tensor (mean over 64-blocks in that tensor)")
    lines.append("")
    lines.append(
        "| tensor | layer | blocks | top1 | top4 | top8 | top16 | "
        "err1 | err4 | err8 | err16 |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for a in sorted(per_tensor, key=lambda x: x.name):
        mt, me = a.means()
        lines.append(
            f"| {a.name} | {a.layer_key} | {a.n_blocks} | "
            f"{mt[1]:.4f} | {mt[4]:.4f} | {mt[8]:.4f} | {mt[16]:.4f} | "
            f"{me[1]:.4f} | {me[4]:.4f} | {me[8]:.4f} | {me[16]:.4f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="WHT64 weight analysis for Q8_0 GGUF tensors")
    ap.add_argument(
        "--gguf",
        type=Path,
        default=None,
        help="Path to .gguf (default: config/gyroscopic_llm.yaml or env GYROSCOPIC_GGUF_PATH)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT
        / "src/tools/gyroscopic/workflow/products/weight64_analysis.md",
        help="Output markdown path",
    )
    ap.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="No progress on stderr (only final line)",
    )
    ap.add_argument(
        "--chunk-blocks",
        type=int,
        default=8192,
        metavar="N",
        help="Max 64-wide blocks per WHT batch (lower uses less RAM; default 8192)",
    )
    args = ap.parse_args()
    if args.chunk_blocks < 1:
        print("--chunk-blocks must be >= 1", file=sys.stderr)
        return 2
    gguf = args.gguf
    if gguf is None:
        try:
            from src.tools.gyroscopic.config import get_gyroscopic_llm_config, resolve_gguf_path

            cfg = get_gyroscopic_llm_config()
            gguf = resolve_gguf_path(cfg)
        except Exception:
            gguf = None

    if gguf is None or not gguf.is_file():
        stub = REPO_ROOT / "src/tools/gyroscopic/workflow/products/weight64_analysis.md"
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text(
            "# Weight 64-wide WHT analysis (Phase A)\n\n"
            "No GGUF path provided or file missing. Run:\n\n"
            "    python scripts/analyze_gguf_weight64_wht.py --gguf path/to/model-Q8_0.gguf\n\n"
            "Or:\n\n"
            "    python -m src.tools.gyroscopic.helpers.analyze_gguf_weight64_wht "
            "--gguf path/to/model-Q8_0.gguf\n\n"
            "Or set `gguf_path` in config/gyroscopic_llm.yaml or env GYROSCOPIC_GGUF_PATH.\n",
            encoding="utf-8",
        )
        print("GGUF not found; wrote stub report.", file=sys.stderr)
        return 1

    if not GGUF_PY.is_dir():
        print(f"Missing gguf-py at {GGUF_PY}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    if not args.quiet:
        nbytes = gguf.stat().st_size
        print(
            f"GGUF open: {gguf.name} ({nbytes / (1024**3):.2f} GiB on disk)",
            file=sys.stderr,
            flush=True,
        )

    t_reader = time.perf_counter()
    reader = GGUFReader(str(gguf))
    q8_list = [t for t in reader.tensors if t.tensor_type == GGMLQuantizationType.Q8_0]
    n_all = len(reader.tensors)
    n_q8 = len(q8_list)
    if not args.quiet:
        print(
            f"GGUFReader: {n_all} tensor headers, {n_q8} Q8_0 (mmap/metadata "
            f"{time.perf_counter() - t_reader:.2f}s).",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"Settings: chunk_blocks={args.chunk_blocks}. "
            "Same GGUF + same script version -> same numbers; rerun only to change code, "
            "chunk size, or add new metrics.",
            file=sys.stderr,
            flush=True,
        )
        print(
            "Rough wall time: often minutes on a 4B Q8_0 CPU run; large rows (e.g. embeddings) "
            "cost more per line below.",
            file=sys.stderr,
            flush=True,
        )

    per_tensor: list[TensorAgg] = []
    t_loop0 = time.perf_counter()
    for i, t in enumerate(q8_list, start=1):
        t_one = time.perf_counter()
        agg = analyze_tensor_q8_0(t, args.chunk_blocks)
        dt = time.perf_counter() - t_one
        if not args.quiet:
            loop_elapsed = time.perf_counter() - t_loop0
            left = n_q8 - i
            eta_s = (loop_elapsed / float(i)) * float(left) if i else 0.0
            eta_m = eta_s / 60.0
            nblk = agg.n_blocks if agg is not None else 0
            print(
                f"[{i}/{n_q8}] {t.name}  +{dt:.1f}s  blocks={nblk}  "
                f"loop={loop_elapsed:.0f}s  ETA~{eta_m:.1f}m",
                file=sys.stderr,
                flush=True,
            )
        if agg is None:
            continue
        per_tensor.append(agg)

    write_report(args.out, gguf, per_tensor)
    elapsed = time.perf_counter() - t0
    print(f"Wrote {args.out} ({len(per_tensor)} tensors) in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
