"""
SPC job: Structural Probing Compilation.
Probes -> Bolmo oracle -> Router features -> ridge solve -> report -> artifact.
No external corpus; probes from Router physics + Bolmo vocabulary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", message=".*rope_config_validation.*")

BOLMO_VOCAB_OFFSET = 4
BOLMO_VOCAB_SIZE = 520
BOLMO_BOS_ID = 1


def _load_bolmo(model_dir: Path, device: torch.device) -> tuple[Any, Any]:
    """Load Bolmo model and tokenizer via blomo_port common (same wiring as lab.py)."""
    _blomo_port = Path(__file__).resolve().parent.parent.parent.parent / "secret_lab_ignore" / "blomo_port"
    if str(_blomo_port) not in sys.path:
        sys.path.insert(0, str(_blomo_port))
    from common import load_bolmo, maybe_patch_expand_byte_ids  # type: ignore[reportMissingImports]
    model, tokenizer = load_bolmo(model_dir, device)
    maybe_patch_expand_byte_ids(tokenizer)
    return model, tokenizer


def _bytes_to_token_ids(seq: list[int], add_bos: bool = True) -> list[int]:
    """Convert byte sequence to Bolmo token IDs (base tokens only)."""
    ids = [BOLMO_BOS_ID] if add_bos else []
    for byte_val in seq:
        tid = BOLMO_VOCAB_OFFSET + (int(byte_val) & 0xFF)
        ids.append(tid)
    return ids


def build_probes(
    pairs_max: int = 65536,
    separator: bool = True,
    alternation_size: int = 4096,
    vocab_count: int = 0,
    tokenizer: Any | None = None,
) -> list[tuple[bytes, str]]:
    """
    Build probe set: (byte_sequence, family).
    No corpus; finite deterministic probes.
    """
    probes: list[tuple[bytes, str]] = []

    for b in range(256):
        probes.append((bytes([b]), "byte1"))

    pairs_added = 0
    for b1 in range(256):
        for b2 in range(256):
            probes.append((bytes([b1, b2]), "pair"))
            pairs_added += 1
            if pairs_added >= pairs_max:
                break
        if pairs_added >= pairs_max:
            break

    if separator:
        for x in range(256):
            probes.append((bytes([x, 0xAA]), "sep_ax"))
        for x in range(256):
            probes.append((bytes([0xAA, x]), "sep_xa"))

    if alternation_size > 0:
        step = max(1, (256 * 256) // alternation_size)
        for i in range(0, 256 * 256, step):
            x, y = i // 256, i % 256
            probes.append((bytes([x, y, x, y]), "alt4"))

    if vocab_count > 0 and tokenizer is not None:
        for idx in range(min(vocab_count, tokenizer.vocab_size)):
            try:
                tok = tokenizer.convert_ids_to_tokens(idx)
                if tok is None:
                    continue
                text = tok if isinstance(tok, str) else str(tok)
                enc = tokenizer.encode(text, add_special_tokens=False)
                if enc:
                    raw = tokenizer.decode(enc)
                    byts = raw.encode("utf-8", errors="replace")[:8]
                    if byts:
                        probes.append((bytes(byts), "vocab"))
            except Exception:
                pass

    return probes


def _run_oracle_batch(
    model: Any,
    tokenizer: Any,
    batch_input_ids: list[list[int]],
    V: int,
    dev: torch.device,
) -> np.ndarray:
    """Run Bolmo oracle on batch; deterministic boundary_mask."""
    inp = torch.tensor(batch_input_ids, dtype=torch.long, device=dev)
    boundary_mask = torch.ones_like(inp, dtype=torch.bool, device=dev)
    expanded_list = []
    for row in batch_input_ids:
        expanded_list.append(tokenizer.expand_byte_ids(row))
    expanded_input_ids = torch.tensor(expanded_list, dtype=torch.long, device=dev)
    with torch.no_grad():
        out = model(
            input_ids=inp,
            expanded_input_ids=expanded_input_ids,
            boundary_mask=boundary_mask,
            logits_to_keep=1,
        )
    logits = out.logits[:, -1, :].float().cpu().numpy()
    if logits.shape[1] > V:
        logits = logits[:, :V]
    elif logits.shape[1] < V:
        logits = np.pad(logits, ((0, 0), (0, V - logits.shape[1])))
    return logits.astype(np.float64)


_CKPT_FNAME = "spc_pass1_ckpt.npz"


def _ridge_solve(
    XtX: np.ndarray,
    YtX: np.ndarray,
    sum_phi: np.ndarray,
    sum_y: np.ndarray,
    N: int,
    lambda_reg: float,
    D: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Centered ridge regression. Returns (W, bias, condition_number)."""
    phi_mean = sum_phi / N
    y_mean = sum_y / N
    XtX_c = XtX - N * np.outer(phi_mean, phi_mean)
    YtX_c = YtX - np.outer(y_mean, N * phi_mean)
    lam = lambda_reg * N
    W = np.linalg.solve(XtX_c + lam * np.eye(D), YtX_c.T).T
    bias = (y_mean - W @ phi_mean).astype(np.float32)
    try:
        s = np.linalg.svd(XtX_c + lam * np.eye(D), compute_uv=False)
        cond = float(s[0] / max(s[-1], 1e-12))
    except Exception:
        cond = float("nan")
    return W, bias, cond


def _load_pass1_ckpt(output_dir: Path) -> dict[str, Any] | None:
    ckpt_path = output_dir / _CKPT_FNAME
    if not ckpt_path.exists():
        return None
    data = np.load(str(ckpt_path), allow_pickle=False)
    return {
        "XtX": data["XtX"],
        "YtX": data["YtX"],
        "sum_phi": data["sum_phi"],
        "sum_y": data["sum_y"],
        "N": int(data["N"]),
        "completed_lengths": set(int(x) for x in data["completed_lengths"]),
    }


def _save_pass1_ckpt(
    output_dir: Path,
    XtX: np.ndarray,
    YtX: np.ndarray,
    sum_phi: np.ndarray,
    sum_y: np.ndarray,
    N: int,
    completed_lengths: list[int],
) -> None:
    dest = output_dir / _CKPT_FNAME
    tmp = output_dir / "spc_pass1_ckpt.tmp.npz"
    np.savez(
        str(tmp),
        XtX=XtX,
        YtX=YtX,
        sum_phi=sum_phi,
        sum_y=sum_y,
        N=np.int64(N),
        completed_lengths=np.array(completed_lengths, dtype=np.int64),
    )
    os.replace(str(tmp), str(dest))


def run_spc(
    model_dir: Path,
    output_dir: Path,
    l3_path: Path,
    *,
    pairs_max: int = 65536,
    lambda_reg: float = 1e-3,
    feature_dim: int = 2048,
    batch_size: int = 256,
    device: str = "cpu",
    resume: bool = False,
) -> dict[str, Any]:
    """Single SPC run: probes -> solve W -> pass 2 metrics -> save."""
    from src.tools.adaptors import RouterFeatureBuilder
    from src.tools.layers import create_default_four_layers

    dev = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Bolmo oracle...")
    model, tokenizer = _load_bolmo(model_dir, dev)

    print("Building Router (FourLayers)...")
    four = create_default_four_layers(l3_path=l3_path, build_l3_if_missing=False)

    probes = build_probes(pairs_max=pairs_max, vocab_count=0)
    N_total = len(probes)
    by_len: dict[int, list[tuple[bytes, str]]] = {}
    for p in probes:
        L = len(p[0])
        by_len.setdefault(L, []).append(p)

    Y_mem_path = output_dir / "oracle_logits.mmap"
    if not resume:
        if Y_mem_path.exists():
            Y_mem_path.unlink()
        ckpt_path = output_dir / _CKPT_FNAME
        if ckpt_path.exists():
            ckpt_path.unlink()

    pairs_added = sum(1 for _, fam in probes if fam == "pair")
    probe_manifest = {
        "pairs_max": pairs_max,
        "pairs_added": pairs_added,
        "separator": True,
        "alternation_size": 4096,
        "byte1": True,
        "total": len(probes),
        "hash": hashlib.sha256(str(probes[:100]).encode()).hexdigest()[:16],
    }
    print(f"Probes: {len(probes)}")

    D = feature_dim
    V = BOLMO_VOCAB_SIZE
    completed_lengths: list[int] = []
    if resume:
        ckpt = _load_pass1_ckpt(output_dir)
        if ckpt:
            XtX = ckpt["XtX"].copy()
            YtX = ckpt["YtX"].copy()
            sum_phi = ckpt["sum_phi"].copy()
            sum_y = ckpt["sum_y"].copy()
            N = ckpt["N"]
            completed_lengths = sorted(ckpt["completed_lengths"])
            print(f"Resumed: N={N}, completed len={completed_lengths}")
        else:
            XtX = np.zeros((D, D), dtype=np.float64)
            YtX = np.zeros((V, D), dtype=np.float64)
            sum_phi = np.zeros(D, dtype=np.float64)
            sum_y = np.zeros(V, dtype=np.float64)
            N = 0
    else:
        XtX = np.zeros((D, D), dtype=np.float64)
        YtX = np.zeros((V, D), dtype=np.float64)
        sum_phi = np.zeros(D, dtype=np.float64)
        sum_y = np.zeros(V, dtype=np.float64)
        N = 0

    done_lens = set(completed_lengths)
    row_idx = sum(len(by_len[s]) for s in completed_lengths)
    Y_mem = np.memmap(
        str(Y_mem_path),
        mode="r+" if resume and Y_mem_path.exists() else "w+",
        dtype=np.float32,
        shape=(N_total, V),
    )

    print("Pass 1: accumulate XtX, YtX, write oracle logits...")
    for seq_len, group in sorted(by_len.items()):
        if seq_len in done_lens:
            print(f"  len={seq_len}: {len(group)} (skip, resumed)")
            continue
        for i in range(0, len(group), batch_size):
            batch = group[i : i + batch_size]
            phis_arr: list[np.ndarray] = []
            batch_input_ids: list[list[int]] = []
            for seq_bytes, _ in batch:
                seq = list(seq_bytes)
                four.reset()
                for byte_val in seq:
                    four.ingest_byte(byte_val)
                last_byte = seq[-1] if seq else 0xAA
                phi = RouterFeatureBuilder.build_raw(
                    four.regs.l1_state8,
                    four.regs.l2_state16,
                    four.regs.l3_state24,
                    four.regs.l4.O,
                    four.regs.l4.E,
                    four.regs.l4.parity,
                    last_byte,
                )
                phi_exp = RouterFeatureBuilder.walsh_expand(phi, D)
                phis_arr.append(phi_exp.detach().numpy())
                batch_input_ids.append(_bytes_to_token_ids(seq))

            Phi = np.stack(phis_arr, axis=0)
            Y = _run_oracle_batch(model, tokenizer, batch_input_ids, V, dev)
            XtX += Phi.T @ Phi
            YtX += Y.T @ Phi
            sum_phi += Phi.sum(axis=0)
            sum_y += Y.sum(axis=0)
            N += Phi.shape[0]
            B = Phi.shape[0]
            Y_mem[row_idx : row_idx + B, :] = Y.astype(np.float32)
            row_idx += B

        completed_lengths.append(seq_len)
        _save_pass1_ckpt(output_dir, XtX, YtX, sum_phi, sum_y, N, completed_lengths)
        print(f"  len={seq_len}: {len(group)} (checkpointed)")

    phi_mean = sum_phi / N
    y_mean = sum_y / N
    XtX_c = XtX - N * np.outer(phi_mean, phi_mean)
    YtX_c = YtX - np.outer(y_mean, N * phi_mean)

    print("Solving ridge regression (centered)...")
    lam = lambda_reg * N
    W = np.linalg.solve(XtX_c + lam * np.eye(D), YtX_c.T)
    W = W.T
    bias = (y_mean - W @ phi_mean).astype(np.float32)

    try:
        s = np.linalg.svd(XtX_c + lam * np.eye(D), compute_uv=False)
        cond = float(s[0] / max(s[-1], 1e-12))
    except Exception:
        cond = float("nan")

    feature_spec = {
        "D": D,
        "core_features": RouterFeatureBuilder.NUM_CORE_FEATURES,
        "expansion": "fwht",
    }

    def _to_bytes_arr(obj: dict) -> np.ndarray:
        return np.frombuffer(json.dumps(obj).encode("utf-8"), dtype=np.uint8)

    out_path = output_dir / "router_operator.npz"
    metrics_partial = {"N": int(N), "cond": float(cond), "pass2_pending": True}
    np.savez(
        out_path,
        W=W.astype(np.float32),
        b=bias,
        feature_spec=_to_bytes_arr(feature_spec),
        probe_manifest=_to_bytes_arr(probe_manifest),
        metrics=_to_bytes_arr(metrics_partial),
    )
    print(f"Saved {out_path} (operator ready; Pass 2 pending)")

    print("Pass 2: streaming metrics (reuse oracle logits from Pass 1)...")
    n_top1, n_top5, n_top10 = 0, 0, 0
    ce_sum, kl_sum = 0.0, 0.0
    base_top1, base_count = 0, 0
    family_counts: dict[str, int] = {}
    family_top1: dict[str, int] = {}

    def _log_softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=1, keepdims=True)
        return x - np.log(np.exp(x).sum(axis=1, keepdims=True) + 1e-12)

    row_idx = 0
    for seq_len, group in sorted(by_len.items()):
        for i in range(0, len(group), batch_size):
            batch = group[i : i + batch_size]
            phis_arr = []
            families = []
            for seq_bytes, fam in batch:
                seq = list(seq_bytes)
                four.reset()
                for byte_val in seq:
                    four.ingest_byte(byte_val)
                last_byte = seq[-1] if seq else 0xAA
                phi = RouterFeatureBuilder.build_raw(
                    four.regs.l1_state8,
                    four.regs.l2_state16,
                    four.regs.l3_state24,
                    four.regs.l4.O,
                    four.regs.l4.E,
                    four.regs.l4.parity,
                    last_byte,
                )
                phi_exp = RouterFeatureBuilder.walsh_expand(phi, D)
                phis_arr.append(phi_exp.detach().numpy())
                families.append(fam)

            Phi = np.stack(phis_arr, axis=0)
            B = Phi.shape[0]
            Y = Y_mem[row_idx : row_idx + B, :].astype(np.float64)
            row_idx += B
            Y_pred = (Phi @ W.T) + bias

            top1_bolmo = np.argmax(Y, axis=1)
            top1_pred = np.argmax(Y_pred, axis=1)
            topk5 = np.argsort(-Y, axis=1)[:, :5]
            topk10 = np.argsort(-Y, axis=1)[:, :10]

            for j in range(len(batch)):
                n_top1 += int(top1_bolmo[j] == top1_pred[j])
                n_top5 += int(top1_pred[j] in topk5[j])
                n_top10 += int(top1_pred[j] in topk10[j])
                fam = families[j]
                family_counts[fam] = family_counts.get(fam, 0) + 1
                family_top1[fam] = family_top1.get(fam, 0) + int(top1_bolmo[j] == top1_pred[j])
                if BOLMO_VOCAB_OFFSET <= top1_bolmo[j] < BOLMO_VOCAB_OFFSET + 256:
                    base_count += 1
                    base_top1 += int(top1_bolmo[j] == top1_pred[j])

            log_probs_pred = _log_softmax(Y_pred)
            log_probs_bolmo = _log_softmax(Y)
            probs_bolmo = np.exp(log_probs_bolmo)
            ce_sum += float(-np.sum(probs_bolmo * log_probs_pred))
            kl_sum += float(np.sum(probs_bolmo * (log_probs_bolmo - log_probs_pred)))

    top1_match = n_top1 / N
    top5_match = n_top5 / N
    top10_match = n_top10 / N
    ce = ce_sum / N
    kl = kl_sum / N
    top1_base_rate = base_top1 / max(1, base_count)
    family_breakdown = {
        fam: family_top1[fam] / max(1, family_counts[fam])
        for fam in family_counts
    }

    metrics = {
        "top1_match": float(top1_match),
        "top5_match": float(top5_match),
        "top10_match": float(top10_match),
        "top1_base_rate": float(top1_base_rate),
        "ce": float(ce),
        "kl": float(kl),
        "cond": float(cond),
        "N": int(N),
        "D": int(D),
        "V": int(V),
        "family_breakdown": family_breakdown,
    }

    np.savez(
        out_path,
        W=W.astype(np.float32),
        b=bias,
        feature_spec=_to_bytes_arr(feature_spec),
        probe_manifest=_to_bytes_arr(probe_manifest),
        metrics=_to_bytes_arr(metrics),
    )
    print(f"Saved {out_path}")

    report_path = output_dir / "spc_report.json"
    report = {"metrics": metrics, "probe_manifest": probe_manifest}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {report_path}")

    try:
        if hasattr(Y_mem, "_mmap") and Y_mem._mmap is not None:
            Y_mem._mmap.close()
    except Exception:
        pass
    del Y_mem
    try:
        if Y_mem_path.exists():
            Y_mem_path.unlink()
    except OSError:
        pass
    ckpt_path = output_dir / _CKPT_FNAME
    if ckpt_path.exists():
        ckpt_path.unlink()
    print("Removed checkpoint and oracle logits (run complete)")

    print("--- SPC Report ---")
    print(f"top1:  {metrics['top1_match']:.4f}")
    print(f"top5:  {metrics['top5_match']:.4f}")
    print(f"top10: {metrics['top10_match']:.4f}")
    print(f"top1_base: {metrics['top1_base_rate']:.4f}")
    print(f"CE: {metrics['ce']:.4f}  KL: {metrics['kl']:.4f}")
    print(f"cond: {metrics['cond']:.2e}")
    print("family:", metrics["family_breakdown"])

    return metrics


def run_multirate_spc(
    model_dir: Path,
    output_dir: Path,
    l3_path: Path,
    *,
    pairs_max: int = 65536,
    lambda_reg: float = 1e-3,
    fast_dim: int = 256,
    slow_dim: int = 512,
    batch_size: int = 256,
    device: str = "cpu",
    vocab_count: int = 0,
    boundary_threshold: float = 0.25,
) -> dict[str, Any]:
    """
    Multi-rate SPC: compile FastOperator and SlowOperator from Bolmo.
    Fast: fit to teacher Y. Slow: fit to RESIDUAL R = Y - Y_fast.
    Optionally train boundary head and restrict slow to boundary samples.
    """
    from src.tools.adaptors.state_encoder import (
        FastFeatureBuilder,
        SlowFeatureBuilder,
        walsh_expand,
    )
    from src.tools.layers import create_default_four_layers
    from src.tools.tuning.inference_utils import boundary_mass_from_logits, boundary_prob_given_byte
    from src.tools.tuning.operators import CompiledOperator

    dev = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Bolmo oracle...")
    model, tokenizer = _load_bolmo(model_dir, dev)

    print("Building Router (FourLayers)...")
    four = create_default_four_layers(l3_path=l3_path, build_l3_if_missing=False)

    probes = build_probes(
        pairs_max=pairs_max, vocab_count=vocab_count, tokenizer=tokenizer
    )
    N_total = len(probes)
    by_len: dict[int, list[tuple[bytes, str]]] = {}
    for p in probes:
        by_len.setdefault(len(p[0]), []).append(p)

    V = BOLMO_VOCAB_SIZE
    D_fast = fast_dim
    D_slow = slow_dim

    XtX_f = np.zeros((D_fast, D_fast), dtype=np.float64)
    YtX_f = np.zeros((V, D_fast), dtype=np.float64)
    sum_phi_f = np.zeros(D_fast, dtype=np.float64)
    sum_y = np.zeros(V, dtype=np.float64)
    N = 0

    print(f"Probes: {N_total}")
    print(f"Fast features: {FastFeatureBuilder.NUM_FEATURES} -> Walsh dim {D_fast}")
    print(f"Slow features: {SlowFeatureBuilder.NUM_FEATURES} -> Walsh dim {D_slow}")
    print("Pass 1: accumulate fast, solve fast...")

    for seq_len, group in sorted(by_len.items()):
        for i in range(0, len(group), batch_size):
            batch = group[i : i + batch_size]
            fast_phis: list[np.ndarray] = []
            batch_input_ids: list[list[int]] = []

            for seq_bytes, _ in batch:
                seq = list(seq_bytes)
                four.reset()
                for b in seq:
                    four.ingest_byte(b)
                last_byte = seq[-1] if seq else 0xAA
                regs = four.regs
                phi_f_raw = FastFeatureBuilder.build(regs.l2_state16, last_byte)
                phi_f = walsh_expand(phi_f_raw, D_fast).detach().numpy()
                fast_phis.append(phi_f)
                batch_input_ids.append(_bytes_to_token_ids(seq))

            Phi_f = np.stack(fast_phis, axis=0)
            Y = _run_oracle_batch(model, tokenizer, batch_input_ids, V, dev)

            XtX_f += Phi_f.T @ Phi_f
            YtX_f += Y.T @ Phi_f
            sum_phi_f += Phi_f.sum(axis=0)
            sum_y += Y.sum(axis=0)
            N += Phi_f.shape[0]

        print(f"  len={seq_len}: {len(group)} probes done")

    print(f"Total samples: {N}")

    print("Solving FastOperator (ridge)...")
    W_f, b_f, cond_f = _ridge_solve(
        XtX_f, YtX_f, sum_phi_f, sum_y, N, lambda_reg, D_fast
    )
    print(f"  condition: {cond_f:.2e}")

    print("Pass 2: accumulate slow on RESIDUAL R = Y - Y_fast...")
    XtX_s = np.zeros((D_slow, D_slow), dtype=np.float64)
    RtX_s = np.zeros((V, D_slow), dtype=np.float64)
    sum_phi_s = np.zeros(D_slow, dtype=np.float64)
    sum_r = np.zeros(V, dtype=np.float64)
    N_s = 0

    for seq_len, group in sorted(by_len.items()):
        for i in range(0, len(group), batch_size):
            batch = group[i : i + batch_size]
            fast_phis = []
            slow_phis = []
            batch_input_ids = []

            for seq_bytes, _ in batch:
                seq = list(seq_bytes)
                four.reset()
                for b in seq:
                    four.ingest_byte(b)
                last_byte = seq[-1] if seq else 0xAA
                regs = four.regs
                phi_f = walsh_expand(
                    FastFeatureBuilder.build(regs.l2_state16, last_byte), D_fast
                ).detach().numpy()
                phi_s = walsh_expand(
                    SlowFeatureBuilder.build(
                        regs.l3_state24, regs.l4.O, regs.l4.E, regs.l4.parity,
                    ),
                    D_slow,
                ).detach().numpy()
                fast_phis.append(phi_f)
                slow_phis.append(phi_s)
                batch_input_ids.append(_bytes_to_token_ids(seq))

            Phi_f = np.stack(fast_phis, axis=0)
            Phi_s = np.stack(slow_phis, axis=0)
            Y = _run_oracle_batch(model, tokenizer, batch_input_ids, V, dev)

            Y_fast = (Phi_f @ W_f.T) + b_f
            R = Y - Y_fast

            base = Y[:, BOLMO_VOCAB_OFFSET : BOLMO_VOCAB_OFFSET + 256]
            fused = Y[:, BOLMO_VOCAB_OFFSET + 256 : BOLMO_VOCAB_OFFSET + 512]
            stack = np.stack([base, fused], axis=-1)
            m = stack.max(axis=-1, keepdims=True)
            byte_logits = np.log((np.exp(stack - m)).sum(axis=-1) + 1e-12) + m.squeeze(-1)
            b_star = np.argmax(byte_logits, axis=1)
            p_bnd = np.array([
                boundary_prob_given_byte(base[i], fused[i], int(b_star[i]))
                for i in range(len(b_star))
            ])
            mask = (p_bnd >= boundary_threshold).astype(bool)
            Phi_s2 = Phi_s[mask]
            R2 = R[mask]
            if Phi_s2.shape[0] == 0:
                continue

            XtX_s += Phi_s2.T @ Phi_s2
            RtX_s += R2.T @ Phi_s2
            sum_phi_s += Phi_s2.sum(axis=0)
            sum_r += R2.sum(axis=0)
            N_s += Phi_s2.shape[0]

        print(f"  len={seq_len}: {len(group)} residual samples")

    if N_s < 10:
        print("Warning: very few boundary samples; using all samples for slow residual")
        N_s = 0
        XtX_s = np.zeros((D_slow, D_slow), dtype=np.float64)
        RtX_s = np.zeros((V, D_slow), dtype=np.float64)
        sum_phi_s = np.zeros(D_slow, dtype=np.float64)
        sum_r = np.zeros(V, dtype=np.float64)
        for seq_len, group in sorted(by_len.items()):
            for i in range(0, len(group), batch_size):
                batch = group[i : i + batch_size]
                fast_phis = []
                slow_phis = []
                batch_input_ids = []
                for seq_bytes, _ in batch:
                    seq = list(seq_bytes)
                    four.reset()
                    for b in seq:
                        four.ingest_byte(b)
                    last_byte = seq[-1] if seq else 0xAA
                    regs = four.regs
                    phi_f = walsh_expand(
                        FastFeatureBuilder.build(regs.l2_state16, last_byte), D_fast
                    ).detach().numpy()
                    phi_s = walsh_expand(
                        SlowFeatureBuilder.build(
                            regs.l3_state24, regs.l4.O, regs.l4.E, regs.l4.parity,
                        ),
                        D_slow,
                    ).detach().numpy()
                    fast_phis.append(phi_f)
                    slow_phis.append(phi_s)
                    batch_input_ids.append(_bytes_to_token_ids(seq))
                Phi_f = np.stack(fast_phis, axis=0)
                Phi_s = np.stack(slow_phis, axis=0)
                Y = _run_oracle_batch(model, tokenizer, batch_input_ids, V, dev)
                R = Y - (Phi_f @ W_f.T + b_f)
                XtX_s += Phi_s.T @ Phi_s
                RtX_s += R.T @ Phi_s
                sum_phi_s += Phi_s.sum(axis=0)
                sum_r += R.sum(axis=0)
                N_s += Phi_s.shape[0]

    print("Solving SlowOperator on residual (ridge)...")
    W_s, b_s, cond_s = _ridge_solve(
        XtX_s, RtX_s, sum_phi_s, sum_r, N_s, lambda_reg, D_slow
    )
    print(f"  condition: {cond_s:.2e} (N_slow={N_s})")

    fast_op = CompiledOperator(
        W=torch.from_numpy(W_f.astype(np.float32)),
        b=torch.from_numpy(b_f),
        name="fast",
        feature_dim=D_fast,
        vocab_size=V,
    )
    slow_op = CompiledOperator(
        W=torch.from_numpy(W_s.astype(np.float32)),
        b=torch.from_numpy(b_s),
        name="slow",
        feature_dim=D_slow,
        vocab_size=V,
    )

    fast_op.save(output_dir / "fast_operator.npz")
    slow_op.save(output_dir / "slow_operator.npz")
    print(f"Saved fast_operator.npz and slow_operator.npz to {output_dir}")

    print("Training boundary head (p_boundary from phi_f)...")
    XtX_b = np.zeros((D_fast, D_fast), dtype=np.float64)
    ptX_b = np.zeros(D_fast, dtype=np.float64)
    sum_p = 0.0
    N_b = 0
    for seq_len, group in sorted(by_len.items()):
        for i in range(0, len(group), batch_size):
            batch = group[i : i + batch_size]
            fast_phis = []
            batch_input_ids = []
            for seq_bytes, _ in batch:
                seq = list(seq_bytes)
                four.reset()
                for b in seq:
                    four.ingest_byte(b)
                last_byte = seq[-1] if seq else 0xAA
                regs = four.regs
                phi_f = walsh_expand(
                    FastFeatureBuilder.build(regs.l2_state16, last_byte), D_fast
                ).detach().numpy()
                fast_phis.append(phi_f)
                batch_input_ids.append(_bytes_to_token_ids(seq))
            Phi_f = np.stack(fast_phis, axis=0)
            Y = _run_oracle_batch(model, tokenizer, batch_input_ids, V, dev)
            p = boundary_mass_from_logits(Y)
            XtX_b += Phi_f.T @ Phi_f
            ptX_b += p.T @ Phi_f
            sum_p += p.sum()
            N_b += Phi_f.shape[0]
    sum_phi_b = sum_phi_f
    W_b, b_b, _ = _ridge_solve(
        XtX_b,
        ptX_b.reshape(1, -1),
        sum_phi_b,
        np.array([sum_p]),
        N_b,
        lambda_reg,
        D_fast,
    )
    np.savez(
        output_dir / "boundary_head.npz",
        W=W_b.astype(np.float32),
        b=np.float32(b_b[0]),
    )
    print(f"Saved boundary_head.npz")

    print("Pass 2: streaming metrics (combined = fast + slow)...")
    n_top1, n_top5, n_top10 = 0, 0, 0
    base_top1, base_count = 0, 0
    family_counts: dict[str, int] = {}
    family_top1: dict[str, int] = {}

    for seq_len, group in sorted(by_len.items()):
        for i in range(0, len(group), batch_size):
            batch = group[i : i + batch_size]
            fast_phis = []
            slow_phis = []
            families = []
            batch_input_ids = []

            for seq_bytes, fam in batch:
                seq = list(seq_bytes)
                four.reset()
                for b in seq:
                    four.ingest_byte(b)
                last_byte = seq[-1] if seq else 0xAA
                regs = four.regs

                phi_f = walsh_expand(
                    FastFeatureBuilder.build(regs.l2_state16, last_byte), D_fast
                ).detach().numpy()
                phi_s = walsh_expand(
                    SlowFeatureBuilder.build(
                        regs.l3_state24, regs.l4.O, regs.l4.E, regs.l4.parity,
                    ),
                    D_slow,
                ).detach().numpy()

                fast_phis.append(phi_f)
                slow_phis.append(phi_s)
                families.append(fam)
                batch_input_ids.append(_bytes_to_token_ids(seq))

            Phi_f = np.stack(fast_phis, axis=0)
            Phi_s = np.stack(slow_phis, axis=0)
            Y = _run_oracle_batch(model, tokenizer, batch_input_ids, V, dev)

            Y_pred = (Phi_f @ W_f.T + b_f) + (Phi_s @ W_s.T + b_s)

            top1_bolmo = np.argmax(Y, axis=1)
            top1_pred = np.argmax(Y_pred, axis=1)
            topk5 = np.argsort(-Y, axis=1)[:, :5]
            topk10 = np.argsort(-Y, axis=1)[:, :10]

            for j in range(len(batch)):
                n_top1 += int(top1_bolmo[j] == top1_pred[j])
                n_top5 += int(top1_pred[j] in topk5[j])
                n_top10 += int(top1_pred[j] in topk10[j])
                fam = families[j]
                family_counts[fam] = family_counts.get(fam, 0) + 1
                family_top1[fam] = family_top1.get(fam, 0) + int(
                    top1_bolmo[j] == top1_pred[j]
                )
                if BOLMO_VOCAB_OFFSET <= top1_bolmo[j] < BOLMO_VOCAB_OFFSET + 256:
                    base_count += 1
                    base_top1 += int(top1_bolmo[j] == top1_pred[j])

    metrics = {
        "top1": n_top1 / N,
        "top5": n_top5 / N,
        "top10": n_top10 / N,
        "top1_base": base_top1 / max(1, base_count),
        "cond_fast": float(cond_f),
        "cond_slow": float(cond_s),
        "N": int(N),
        "D_fast": D_fast,
        "D_slow": D_slow,
        "V": V,
        "family_breakdown": {
            fam: family_top1[fam] / max(1, family_counts[fam])
            for fam in family_counts
        },
    }

    report_path = output_dir / "spc_report.json"
    report_path.write_text(json.dumps({"metrics": metrics}, indent=2))
    print(f"Saved {report_path}")

    print("--- Multi-Rate SPC Report ---")
    print(f"top1:  {metrics['top1']:.4f}")
    print(f"top5:  {metrics['top5']:.4f}")
    print(f"top10: {metrics['top10']:.4f}")
    print(f"top1_base: {metrics['top1_base']:.4f}")
    print(f"cond_fast: {metrics['cond_fast']:.2e}")
    print(f"cond_slow: {metrics['cond_slow']:.2e}")
    print("family:", metrics["family_breakdown"])

    return metrics


def run_candidate_scorer_spc(
    model_dir: Path,
    output_dir: Path,
    l3_path: Path,
    *,
    pairs_max: int = 4096,
    lambda_reg: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Compile candidate scorer: theta s.t. score(b) = theta @ psi(state, b).
    Router-native 256-way routing. Teacher = merged log(exp(base)+exp(fused)).
    """
    from src.tools.adaptors.candidate_features import candidate_features, candidate_features_dim
    from src.tools.layers import create_default_four_layers

    dev = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Bolmo oracle...")
    model, tokenizer = _load_bolmo(model_dir, dev)

    print("Building Router (FourLayers)...")
    four = create_default_four_layers(l3_path=l3_path, build_l3_if_missing=False)

    probes = build_probes(pairs_max=pairs_max, vocab_count=0)
    D = candidate_features_dim()
    V = 256

    XtX = np.zeros((D, D), dtype=np.float64)
    XtY = np.zeros(D, dtype=np.float64)
    sum_x = np.zeros(D, dtype=np.float64)
    sum_y = 0.0
    N = 0

    print(f"Candidate features: {D} dims")
    print("Accumulating psi(state,b) -> teacher byte logits...")

    for idx, (seq_bytes, _) in enumerate(probes):
        seq = list(seq_bytes)
        four.reset()
        for b in seq:
            four.ingest_byte(b)
        regs = four.regs
        state24 = regs.l3_state24
        O, E, parity = regs.l4.O, regs.l4.E, regs.l4.parity

        batch_ids = [_bytes_to_token_ids(seq)]
        Y = _run_oracle_batch(model, tokenizer, batch_ids, BOLMO_VOCAB_SIZE, dev)
        base = Y[0, BOLMO_VOCAB_OFFSET : BOLMO_VOCAB_OFFSET + 256]
        fused = Y[0, BOLMO_VOCAB_OFFSET + 256 : BOLMO_VOCAB_OFFSET + 512]
        m = np.maximum(base, fused)
        byte_logits = np.log(np.exp(base - m) + np.exp(fused - m) + 1e-12) + m

        for b in range(256):
            psi = candidate_features(state24, O, E, parity, b)
            y = float(byte_logits[b])
            XtX += np.outer(psi, psi)
            XtY += psi * y
            sum_x += psi
            sum_y += y
            N += 1

        if len(probes) > 100 and idx % 500 == 0:
            print(f"  {idx} probes...")

    print(f"Total samples: {N}")

    phi_mean = sum_x / N
    y_mean = sum_y / N
    XtX_c = XtX - N * np.outer(phi_mean, phi_mean)
    XtY_c = XtY - y_mean * sum_x
    lam = lambda_reg * N
    theta = np.linalg.solve(XtX_c + lam * np.eye(D), XtY_c)

    np.savez(
        output_dir / "candidate_scorer.npz",
        theta=theta.astype(np.float32),
        feature_dim=np.int32(D),
    )
    print(f"Saved candidate_scorer.npz to {output_dir}")

    return {"N": N, "D": D}


def main() -> None:
    parser = argparse.ArgumentParser(description="SPC: Structural Probing Compilation")
    parser.add_argument("--model-dir", type=Path, default=Path("data/models/Bolmo-1B"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/router_operator"))
    parser.add_argument("--l3-path", type=Path, default=Path("data/layers/l3_packed_u24.bin"))
    parser.add_argument("--pairs-max", type=int, default=65536)
    parser.add_argument("--lambda", dest="lambda_reg", type=float, default=1e-3)
    parser.add_argument("--feature-dim", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true", help="Resume Pass 1 from checkpoint if present")
    parser.add_argument("--multirate", action="store_true", help="Multi-rate SPC: compile fast + slow operators")
    parser.add_argument("--fast-dim", type=int, default=256, help="Walsh dim for fast operator (multirate only)")
    parser.add_argument("--slow-dim", type=int, default=512, help="Walsh dim for slow operator (multirate only)")
    parser.add_argument("--vocab-count", type=int, default=0, help="Extra vocab probes (multirate only)")
    parser.add_argument("--boundary-threshold", type=float, default=0.25, help="Min p(boundary|byte) for slow training (multirate)")
    parser.add_argument("--candidate-scorer", action="store_true", help="Compile candidate scorer for 256-way routing")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if args.candidate_scorer:
        run_candidate_scorer_spc(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            l3_path=args.l3_path,
            pairs_max=args.pairs_max,
            lambda_reg=args.lambda_reg,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.multirate:
        run_multirate_spc(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            l3_path=args.l3_path,
            pairs_max=args.pairs_max,
            lambda_reg=args.lambda_reg,
            fast_dim=args.fast_dim,
            slow_dim=args.slow_dim,
            batch_size=args.batch_size,
            device=args.device,
            vocab_count=args.vocab_count,
            boundary_threshold=args.boundary_threshold,
        )
    else:
        run_spc(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            l3_path=args.l3_path,
            pairs_max=args.pairs_max,
            lambda_reg=args.lambda_reg,
            feature_dim=args.feature_dim,
            batch_size=args.batch_size,
            device=args.device,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
