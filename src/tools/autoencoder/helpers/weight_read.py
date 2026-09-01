"""Read external weight blocks into Super Walsh operative form.

Maps a dense 64×64 tile (or 4096×4096 matrix) to 2080 sector gains P_Q(W)
plus defect D_Q(W) = W − P_Q(W), using the same factored Walsh geometry as
``SpectralAutoencoder``.

Usage::

    from src.tools.autoencoder.helpers.weight_read import (
        read_weight_matrix,
        apply_operative,
        verify_read,
    )
    rep = read_weight_matrix(W)
    y = apply_operative(rep, x)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.tools.autoencoder.models.super import irrep_block_index, walsh_matrix_64

HIDDEN = 4096
N_BLOCKS = 2080
TILE = 64
Q1_ROW_BYTES = 576  # Bonsai Q1_0: 32 blocks × 18 bytes (128 weights/block)


@dataclass(frozen=True)
class OperativeRead:
    """P_Q sector gains + dense defect for one weight operator."""

    gains: np.ndarray  # [2080] float64
    pq: np.ndarray  # P_Q(W), shape (n_out, n_in)
    defect: np.ndarray  # D_Q(W) = W - P_Q(W)
    block_id: np.ndarray  # [4096] int32 flat (for n_in == 4096)
    proj_energy_ratio: float
    defect_ratio: float
    n_out: int
    n_in: int

    def apply(self, x: np.ndarray) -> np.ndarray:
        return apply_operative(self, x)


def _w64() -> np.ndarray:
    return walsh_matrix_64().astype(np.float64)


def walsh_forward_vec(x: np.ndarray) -> np.ndarray:
    """Match ``SpectralAutoencoder.walsh_coefficients`` on a dense vector."""
    W = _w64()
    f = np.asarray(x, dtype=np.float64).reshape(64, 64)
    coeff = (W @ f) @ W.T
    return coeff.reshape(-1)


def walsh_inverse_vec(coeff: np.ndarray) -> np.ndarray:
    """Match ``SpectralAutoencoder.inverse_walsh``."""
    W = _w64()
    c = np.asarray(coeff, dtype=np.float64).reshape(64, 64)
    f = (W @ c) @ W.T
    return (f / 4096.0).reshape(-1)


def block_id_flat() -> np.ndarray:
    bid, _ = irrep_block_index()
    return bid.reshape(-1).astype(np.int32)


@lru_cache(maxsize=1)
def walsh_matrix_4096() -> np.ndarray:
    """Forward Walsh on flattened 4096 vectors (column basis)."""
    W = _w64()
    return np.kron(W, W)


@lru_cache(maxsize=1)
def walsh_inverse_matrix_4096() -> np.ndarray:
    """Inverse Walsh (matches ``walsh_inverse_vec``)."""
    return walsh_matrix_4096() / 4096.0


def embed_tile_4096(tile64: np.ndarray, row0: int = 0, col0: int = 0) -> np.ndarray:
    """Place a 64×64 tile into a 4096×4096 matrix."""
    W = np.zeros((HIDDEN, HIDDEN), dtype=np.float64)
    W[row0 : row0 + TILE, col0 : col0 + TILE] = np.asarray(tile64, dtype=np.float64)
    return W


def project_equivariant_gains(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frobenius projection of W onto block-scalar operators in Walsh basis.

    Returns (gains2080, P_Q_matrix).
    """
    W = np.asarray(W, dtype=np.float64)
    if W.shape != (HIDDEN, HIDDEN):
        raise ValueError(f"expected ({HIDDEN},{HIDDEN}) weight, got {W.shape}")
    T = walsh_matrix_4096()
    G = walsh_inverse_matrix_4096()
    Wc = T @ W @ G
    bid = block_id_flat()
    gains = np.zeros(N_BLOCKS, dtype=np.float64)
    Pc = np.zeros_like(Wc)
    for k in range(N_BLOCKS):
        idx = np.nonzero(bid == k)[0]
        sub = Wc[np.ix_(idx, idx)]
        lam = float(np.trace(sub) / len(idx))
        gains[k] = lam
        Pc[np.ix_(idx, idx)] = lam * np.eye(len(idx))
    Pq = G @ Pc @ T
    return gains, Pq


def _gains_from_weight(W: np.ndarray) -> np.ndarray:
    """Sector gains for W with shape (m, 4096)."""
    W = np.asarray(W, dtype=np.float64)
    m, n = W.shape
    if n != HIDDEN:
        raise ValueError(f"expected n_in={HIDDEN}, got {n}")
    bid = block_id_flat()
    if m == HIDDEN:
        return project_equivariant_gains(W)[0]
    G = walsh_inverse_matrix_4096()
    Wg = W @ G
    gains = np.zeros(N_BLOCKS, dtype=np.float64)
    for k in range(N_BLOCKS):
        idx = np.nonzero(bid == k)[0]
        gains[k] = float(Wg[:, idx].mean())
    return gains


def pq_matrix_from_gains(W: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Build P_Q(W) matrix from sector gains (m × 4096)."""
    W = np.asarray(W, dtype=np.float64)
    m, n = W.shape
    if n != HIDDEN:
        raise ValueError(f"expected n_in={HIDDEN}, got {n}")
    bid = block_id_flat()
    G = walsh_inverse_matrix_4096()
    T = walsh_matrix_4096()
    if m == HIDDEN:
        Pc = np.zeros((HIDDEN, HIDDEN), dtype=np.float64)
        for k in range(N_BLOCKS):
            idx = np.nonzero(bid == k)[0]
            Pc[np.ix_(idx, idx)] = gains[k] * np.eye(len(idx))
        return G @ Pc @ T
    d = gains[bid]
    return W @ G @ (d * T)


def read_weight(
    W: np.ndarray,
    *,
    sector_mask: np.ndarray | None = None,
    gains: np.ndarray | None = None,
) -> OperativeRead:
    """Read any (m, 4096) weight into operative gains + defect."""
    W = np.asarray(W, dtype=np.float64)
    m, n = W.shape
    if n != HIDDEN:
        raise ValueError(f"expected n_in={HIDDEN}, got {n}")
    if gains is None:
        gains = _gains_from_weight(W)
    else:
        gains = np.asarray(gains, dtype=np.float64).reshape(-1)
        if gains.shape != (N_BLOCKS,):
            raise ValueError(f"gains must be ({N_BLOCKS},)")
    if sector_mask is not None:
        mask = np.asarray(sector_mask, dtype=np.float64).reshape(-1)
        if mask.shape != (N_BLOCKS,):
            raise ValueError(f"sector_mask must be ({N_BLOCKS},)")
        gains = gains * mask
    Pq = pq_matrix_from_gains(W, gains)
    defect = W - Pq
    w_norm = float(np.linalg.norm(W, ord="fro")) + 1e-30
    return OperativeRead(
        gains=gains,
        pq=Pq,
        defect=defect,
        block_id=block_id_flat(),
        proj_energy_ratio=float(np.linalg.norm(Pq, ord="fro") ** 2 / (w_norm**2)),
        defect_ratio=float(np.linalg.norm(defect, ord="fro") / w_norm),
        n_out=m,
        n_in=HIDDEN,
    )


def read_weight_matrix(W: np.ndarray, **kwargs) -> OperativeRead:
    """Read a 4096×4096 operator into operative gains + defect."""
    W = np.asarray(W, dtype=np.float64)
    if W.shape != (HIDDEN, HIDDEN):
        raise ValueError(f"expected ({HIDDEN},{HIDDEN}), got {W.shape}")
    return read_weight(W, **kwargs)


def read_tile_embedded(tile64: np.ndarray, row0: int = 0, col0: int = 0) -> OperativeRead:
    """Read one 64×64 Bonsai tile embedded in 4096×4096."""
    return read_weight_matrix(embed_tile_4096(tile64, row0, col0))


def apply_pq_weight(
    W: np.ndarray,
    gains: np.ndarray,
    x: np.ndarray,
    block_id: np.ndarray | None = None,
) -> np.ndarray:
    """Apply P_Q(W)·x for W shape (m, 4096) without storing defect."""
    W = np.asarray(W, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.shape[0] != HIDDEN:
        raise ValueError(f"expected length {HIDDEN}, got {x.shape[0]}")
    bid = block_id if block_id is not None else block_id_flat()
    coeff = walsh_forward_vec(x) * gains[bid]
    return W @ walsh_inverse_vec(coeff)


def apply_pq(gains: np.ndarray, x: np.ndarray, block_id: np.ndarray | None = None) -> np.ndarray:
    """Apply P_Q(W) to vector x using sector gains."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.shape[0] != HIDDEN:
        raise ValueError(f"expected length {HIDDEN}, got {x.shape[0]}")
    bid = block_id if block_id is not None else block_id_flat()
    coeff = walsh_forward_vec(x)
    coeff = coeff * gains[bid]
    return walsh_inverse_vec(coeff)


def apply_operative(rep: OperativeRead, x: np.ndarray) -> np.ndarray:
    """y = P_Q(W)·x + D_Q(W)·x (exact when defect is stored)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if rep.n_in != HIDDEN:
        raise ValueError(f"expected n_in={HIDDEN}")
    return rep.pq @ x + rep.defect @ x


def apply_pq_only(rep: OperativeRead, x: np.ndarray) -> np.ndarray:
    """P_Q(W)·x only (no defect) — approximate fast path."""
    return rep.pq @ x


def verify_read(
    rep: OperativeRead,
    W: np.ndarray,
    *,
    n_random: int = 256,
    seed: int = 0,
) -> dict[str, float]:
    """Check P_Q + D_Q equals dense W on random activations."""
    W = np.asarray(W, dtype=np.float64)
    rng = np.random.default_rng(seed)
    max_err = 0.0
    rel_err = 0.0
    for _ in range(n_random):
        x = rng.standard_normal(HIDDEN)
        y_dense = W @ x
        y_read = apply_operative(rep, x)
        err = float(np.max(np.abs(y_dense - y_read)))
        denom = float(np.max(np.abs(y_dense))) + 1e-12
        max_err = max(max_err, err)
        rel_err = max(rel_err, err / denom)
    return {
        "max_abs_err": max_err,
        "max_rel_err": rel_err,
        "proj_energy_ratio": rep.proj_energy_ratio,
        "defect_ratio": rep.defect_ratio,
    }


# ---------------------------------------------------------------------------
# Bonsai Q1_0 GGUF
# ---------------------------------------------------------------------------


def _fp16_to_f32(bits: int) -> float:
    return float(np.frombuffer(np.uint16(bits).tobytes(), dtype=np.float16)[0])


def dequant_q1_row_cols(row: bytes | np.ndarray, col0: int) -> np.ndarray:
    """Dequantize 64 columns at col0 from one Q1_0 row (Bonsai block width 128)."""
    if col0 % 64 != 0:
        raise ValueError("col0 must be 64-aligned")
    buf = memoryview(row) if isinstance(row, (bytes, bytearray)) else memoryview(
        np.asarray(row, dtype=np.uint8).tobytes()
    )
    blk_idx = col0 // 128
    half = (col0 // 64) & 1
    off = blk_idx * 18
    if off + 18 > len(buf):
        raise ValueError("row buffer too short for column block")
    scale = _fp16_to_f32(int.from_bytes(buf[off : off + 2], "little"))
    qs = buf[off + 2 : off + 18]
    out = np.empty(64, dtype=np.float32)
    base = half * 64
    for i in range(64):
        qi = base + i
        bit = (qs[qi >> 3] >> (qi & 7)) & 1
        out[i] = scale if bit else -scale
    return out


def dequant_q1_tile64(
    data: np.ndarray,
    *,
    row_stride: int = Q1_ROW_BYTES,
    n_rows: int = 4096,
    n_cols: int = 4096,
    row0: int = 0,
    col0: int = 0,
) -> np.ndarray:
    """Dequantize a 64×64 float tile from raw Q1_0 row-major storage."""
    if row0 % 64 != 0 or col0 % 64 != 0:
        raise ValueError("row0/col0 must be 64-aligned")
    tile = np.empty((64, 64), dtype=np.float32)
    for r in range(64):
        row_ptr = data[row0 + r]
        if isinstance(row_ptr, np.ndarray):
            row_bytes = row_ptr.tobytes()
        else:
            row_bytes = bytes(data[row0 + r])
        tile[r] = dequant_q1_row_cols(row_bytes, col0)
    return tile.astype(np.float64)


def _import_gguf():
    try:
        import gguf
    except ImportError as e:
        raise ImportError(
            "The 'gguf' package is required to read Bonsai GGUF weights. "
            "Install with: pip install -r requirements-gyroscopic.txt"
        ) from e
    return gguf


def load_bonsai_q1_tensor(
    gguf_path: Path | str,
    tensor_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load raw Q1_0 bytes for a named Bonsai tensor."""
    gguf = _import_gguf()

    path = Path(gguf_path)
    reader = gguf.GGUFReader(str(path))
    for t in reader.tensors:
        if t.name == tensor_name:
            data = np.array(t.data, copy=True)
            meta = {
                "name": t.name,
                "shape": tuple(int(x) for x in t.shape),
                "tensor_type": int(t.tensor_type),
                "row_stride": int(data.shape[1]) if data.ndim == 2 else 0,
            }
            return data, meta
    raise KeyError(f"tensor not found: {tensor_name}")


def bonsai_tile(
    gguf_path: Path | str,
    tensor_name: str,
    row0: int = 0,
    col0: int = 0,
) -> np.ndarray:
    """Dequantize one 64×64 tile from a Bonsai GGUF weight tensor."""
    data, meta = load_bonsai_q1_tensor(gguf_path, tensor_name)
    n_rows, n_cols = meta["shape"]
    return dequant_q1_tile64(
        data,
        row_stride=meta["row_stride"],
        n_rows=n_rows,
        n_cols=n_cols,
        row0=row0,
        col0=col0,
    )


def dequant_q1_matrix(
    data: np.ndarray,
    n_rows: int,
    n_cols: int,
    *,
    row_stride: int = Q1_ROW_BYTES,
    max_rows: int | None = None,
) -> np.ndarray:
    """Dequantize Q1_0 to dense float (optional row cap for smoke tests)."""
    n_use = n_rows if max_rows is None else min(n_rows, max_rows)
    W = np.zeros((n_use, n_cols), dtype=np.float64)
    for r in range(n_use):
        row = data[r]
        row_bytes = row.tobytes() if isinstance(row, np.ndarray) else bytes(row)
        for c0 in range(0, n_cols, 64):
            W[r, c0 : c0 + 64] = dequant_q1_row_cols(row_bytes, c0)
    return W


def tier_ratios_tile(tile64: np.ndarray) -> dict[str, float]:
    """Narrow/general routing: chi/shell/defect ratios on a 64×64 tile."""
    from src.tools.gyroscopic import ops

    flat = np.asarray(tile64, dtype=np.float64).reshape(-1).tolist()
    rep = ops.tile_decompose_ratios(flat)
    return rep


def export_gains_npy(path: Path | str, rep: OperativeRead) -> Path:
    """Write 2080 sector gains for native / layer.c consumption."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, rep.gains.astype(np.float32))
    return out


def read_bonsai_tensor_tile(
    gguf_path: Path | str,
    tensor_name: str,
    row0: int = 0,
    col0: int = 0,
) -> OperativeRead:
    """Load + read one embedded Bonsai tile."""
    tile = bonsai_tile(gguf_path, tensor_name, row0, col0)
    return read_tile_embedded(tile, row0, col0)


def bonsai_dequant_tensor(
    gguf_path: Path | str,
    tensor_name: str,
    *,
    max_rows: int | None = None,
    cache_npz: Path | str | None = None,
) -> np.ndarray:
    """Dequantize a Bonsai Q1_0 tensor to dense float."""
    cache = Path(cache_npz) if cache_npz else None
    if cache and cache.is_file():
        return np.load(cache)["W"]
    data, meta = load_bonsai_q1_tensor(gguf_path, tensor_name)
    shape = meta["shape"]
    row_stride = meta["row_stride"]
    # GGUF may list logical [out,in] while storage is row-major on the larger dim.
    if data.shape[0] == shape[0]:
        n_rows, n_cols = shape[0], shape[1]
    else:
        n_rows, n_cols = shape[1], shape[0]
    W = dequant_q1_matrix(
        data, n_rows, n_cols, row_stride=row_stride, max_rows=max_rows
    )
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, W=W.astype(np.float32), name=tensor_name)
    return W


def q1_hidden_dim(row_stride: int) -> int:
    """Hidden width implied by a Bonsai Q1_0 row (128 weights per 18-byte block)."""
    return (row_stride // 18) * 128


def dequant_q1_rows(
    data: np.ndarray,
    row_indices: np.ndarray | list[int],
    n_cols: int | None = None,
    *,
    row_stride: int = Q1_ROW_BYTES,
) -> np.ndarray:
    """Dequantize selected rows (e.g. token embedding rows)."""
    if n_cols is None:
        n_cols = q1_hidden_dim(row_stride)
    rows = np.asarray(row_indices, dtype=np.int64)
    W = np.zeros((len(rows), n_cols), dtype=np.float64)
    for i, r in enumerate(rows):
        row_bytes = data[int(r)].tobytes()
        for c0 in range(0, n_cols, 64):
            W[i, c0 : c0 + 64] = dequant_q1_row_cols(row_bytes, c0)
    return W


def embedding_corpus(
    gguf_path: Path | str,
    row_indices: np.ndarray | list[int],
    *,
    cache_npz: Path | str | None = None,
) -> np.ndarray:
    """Activation corpus from token_embd rows [n, 4096]."""
    cache = Path(cache_npz) if cache_npz else None
    if cache and cache.is_file():
        data = np.load(cache)
        return np.asarray(data["corpus"], dtype=np.float64)
    data, meta = load_bonsai_q1_tensor(gguf_path, "token_embd.weight")
    n_cols = q1_hidden_dim(meta["row_stride"])
    rows = np.asarray(row_indices, dtype=np.int64)
    corpus = dequant_q1_rows(data, rows, n_cols, row_stride=meta["row_stride"])
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, corpus=corpus.astype(np.float32), row_indices=rows)
    return corpus


def evaluate_on_corpus(
    rep: OperativeRead,
    W: np.ndarray,
    corpus: np.ndarray,
    *,
    pq_only: bool = False,
) -> dict[str, float]:
    """Matmul error stats on activation corpus [n, 4096]."""
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(corpus, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != HIDDEN:
        raise ValueError(f"corpus must be [n, {HIDDEN}]")
    max_err = 0.0
    rel_err = 0.0
    pq_rel = 0.0
    for i in range(X.shape[0]):
        x = X[i]
        y = W @ x
        y_read = apply_pq_only(rep, x) if pq_only else apply_operative(rep, x)
        err = float(np.max(np.abs(y - y_read)))
        denom = float(np.max(np.abs(y))) + 1e-12
        max_err = max(max_err, err)
        rel_err = max(rel_err, err / denom)
        if not pq_only:
            y_pq = apply_pq_only(rep, x)
            pq_rel = max(pq_rel, float(np.max(np.abs(y - y_pq))) / denom)
    out = {
        "max_abs_err": max_err,
        "max_rel_err": rel_err,
        "proj_energy_ratio": rep.proj_energy_ratio,
        "defect_ratio": rep.defect_ratio,
        "n_samples": float(X.shape[0]),
    }
    if not pq_only:
        out["pq_only_max_rel_err"] = pq_rel
    return out


def sector_ladder_masks() -> dict[str, np.ndarray]:
    """Named codec-ladder sector masks (2080)."""
    from src.tools.autoencoder.models.super import codec_ladder_mask

    return {
        "full": codec_ladder_mask("full"),
        "diagonal": codec_ladder_mask("diagonal"),
        "shell": codec_ladder_mask("shell"),
        "trivial": codec_ladder_mask("trivial"),
        "offdiagonal": codec_ladder_mask("offdiagonal"),
    }


def export_operative_bundle(
    path: Path | str,
    rep: OperativeRead,
    *,
    tensor_name: str = "",
) -> Path:
    """Export gains + defect for native consumption (NPZ)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        gains=rep.gains.astype(np.float32),
        pq=rep.pq.astype(np.float32),
        defect=rep.defect.astype(np.float32),
        block_id=rep.block_id.astype(np.int32),
        proj_energy_ratio=np.float32(rep.proj_energy_ratio),
        defect_ratio=np.float32(rep.defect_ratio),
        tensor_name=np.array(tensor_name),
    )
    return out


def search_sector_ladders(
    W: np.ndarray,
    corpus: np.ndarray,
) -> list[dict[str, float | str]]:
    """Compare codec ladder masks on activation reconstruction (P_Q only)."""
    rows: list[dict[str, float | str]] = []
    for name, mask in sector_ladder_masks().items():
        rep = read_weight(W, sector_mask=mask)
        stats = evaluate_on_corpus(rep, W, corpus, pq_only=True)
        rows.append(
            {
                "ladder": name,
                "active_blocks": float(mask.sum()),
                "proj_energy_ratio": rep.proj_energy_ratio,
                "pq_only_max_rel_err": stats["max_rel_err"],
            }
        )
    return sorted(rows, key=lambda r: float(r["pq_only_max_rel_err"]))
