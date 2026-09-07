"""Super training: tasks, gates, and production regeneration."""

from __future__ import annotations

from typing import Any, cast

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src import api
from src.constants import GENE_MAC_A12, GENE_MAC_B12, GENE_MAC_REST, step_state_by_byte
from src.family import intron_family_d, intron_from_byte, intron_micro_ref_d
from src.tools.autoencoder import kernel, paths
from src.tools.autoencoder.datasets import NullCorpus
from src.tools.autoencoder.helpers.training_losses import LossWeights, weighted_total
from src.tools.autoencoder.models.super import Super, prefix_factor_table

D = 6
GENE_MAC_SWAPPED = (GENE_MAC_B12 << 12) | GENE_MAC_A12
G7_BITS_MAX = 2.5  # authored threshold: ~1.5 optimum + 1 bit margin
G6_UNIFORM_TOL = 0.5
SUPER_TASKS = (
    "masked_frame",
    "holonomy",
    "provenance",
    "next_byte",
    "corrupt",
    "sigword",
    "measure_lambda",
    "markov",
    "boundary",
    "super_all",
)
RESIDUAL_ZERO_WEIGHT = 0.1
COMPLETION_ARMS = (
    "masked_frame",
    "super_all",
    "corrupt",
    "next_byte",
    "measure_lambda",
    "markov",
    "boundary",
)


def _device(device: str | torch.device) -> torch.device:
    return torch.device(device) if not isinstance(device, torch.device) else device


def _byte_targets(bytes_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = bytes_np.reshape(-1)
    fam = np.empty(len(flat), dtype=np.int64)
    payload = np.empty((len(flat), 6), dtype=np.float32)
    for i, b in enumerate(flat):
        intron = intron_from_byte(int(b), D)
        fam[i] = intron_family_d(intron, D)
        micro = intron_micro_ref_d(intron, D)
        for bit in range(6):
            payload[i, bit] = float((micro >> bit) & 1)
    return fam, payload, flat.astype(np.int64)


def _prefix_sig_factors(ledger: np.ndarray, upto: int) -> tuple[int, int, int]:
    word = [int(x) for x in ledger[:upto]]
    sid = kernel.word_signature_id(word)
    return kernel.signature_from_id(sid)


def _sig_table_last(ledgers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tbl = prefix_factor_table(np.asarray(ledgers, dtype=np.int64))
    last = tbl[:, -1, :]
    return last[:, 0], last[:, 1], last[:, 2]


class _EMA:
    def __init__(self, model: Super, decay: float = 0.999) -> None:
        self.decay = decay
        self.n = 0
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.is_floating_point()
        }

    def update(self, model: Super) -> None:
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    def copy_to(self, model: Super) -> None:
        cur = model.state_dict()
        for k, v in self.shadow.items():
            if (
                "seq_mlp" in k
                or "climate_head" in k
                or "prefix_gru" in k
                or "ledger_decoder" in k
                or "markov_flip" in k
                or "prev_micro_proj" in k
            ):
                continue
            cur[k] = v
        model.load_state_dict(cur, strict=False)


def _contrastive_margin_loss(
    z_a: torch.Tensor, z_b: torch.Tensor, margin: float = 0.1
) -> torch.Tensor:
    a = F.normalize(z_a, dim=-1)
    b = F.normalize(z_b, dim=-1)
    cos = (a * b).sum(dim=-1)
    return F.relu(margin - (1.0 - cos)).mean()


def _collision_ledgers_masked(
    row: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    la = int(row["length_a"])
    lb = int(row["length_b"])
    a = np.array(
        [int(row["a0"]), int(row["a1"]), int(row["a2"]), int(row["a3"])],
        dtype=np.uint8,
    )
    b = np.array(
        [int(row["b0"]), int(row["b1"]), int(row["b2"]), int(row["b3"])],
        dtype=np.uint8,
    )
    ma = np.zeros(4, dtype=np.bool_)
    mb = np.zeros(4, dtype=np.bool_)
    ma[la:] = True
    mb[lb:] = True
    return a, b, ma, mb


def _corrupt_ledger(ledger: np.ndarray, n_corrupt: int, rng: np.random.Generator) -> np.ndarray:
    out = ledger.copy()
    positions = rng.choice(len(out), size=min(n_corrupt, len(out)), replace=False)
    for p in positions:
        out[p] = int(rng.integers(0, 256))
        while out[p] == ledger[p]:
            out[p] = int(rng.integers(0, 256))
    return out


def _loss_from_heads(
    out: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: LossWeights,
    label_smoothing: float = 0.0,
    arm: str = "canonical",
) -> tuple[torch.Tensor, dict[str, float]]:
    components: dict[str, torch.Tensor] = {}
    if weights.byte_ce > 0:
        prior = out["exact_prior_logits"].detach()
        climate = out.get("climate_logits")
        seq = out.get("seq_logits")
        if arm == "climate" and climate is not None:
            logits = prior + climate
            if "coset_mask" in out and out["coset_mask"] is not None:
                logits = logits.masked_fill(~out["coset_mask"], float("-inf"))
            components["byte_ce"] = F.cross_entropy(
                logits, targets["byte"], label_smoothing=label_smoothing
            )
        elif arm == "seq" and seq is not None:
            markov = out.get("markov_logits")
            logits = prior + seq + (markov if markov is not None else 0.0)
            if "coset_mask" in out and out["coset_mask"] is not None:
                logits = logits.masked_fill(~out["coset_mask"], float("-inf"))
            components["byte_ce"] = F.cross_entropy(
                logits, targets["byte"], label_smoothing=label_smoothing
            )
        elif arm == "completion":
            logits = prior + out["residual_logits"]
            if "coset_mask" in out and out["coset_mask"] is not None:
                logits = logits.masked_fill(~out["coset_mask"], float("-inf"))
            components["byte_ce"] = F.cross_entropy(
                logits, targets["byte"], label_smoothing=label_smoothing
            )
        else:
            seq_t = seq if seq is not None else out["residual_logits"]
            components["residual_zero"] = seq_t.square().mean()
    if weights.family_ce > 0:
        components["family_ce"] = F.cross_entropy(
            out["family_logits"], targets["family"], label_smoothing=label_smoothing
        )
    if weights.payload_ce > 0:
        components["payload_ce"] = F.binary_cross_entropy_with_logits(
            out["payload_logits"], targets["payload"]
        )
    if weights.signature > 0:
        pl = out.get("parity_logits")
        if pl is not None and bool(pl.requires_grad):
            components["signature"] = (
                F.cross_entropy(
                    out["parity_logits"], targets["parity"], label_smoothing=label_smoothing
                )
                + F.cross_entropy(
                    out["tau_u_logits"], targets["tau_u"], label_smoothing=label_smoothing
                )
                + F.cross_entropy(
                    out["tau_v_logits"], targets["tau_v"], label_smoothing=label_smoothing
                )
            ) / 3.0
    return weighted_total(components, weights)


def _sig_row_ledger(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = int(row["length"])
    led = np.array(
        [int(row["b0"]), int(row["b1"]), int(row["b2"]), int(row["b3"])],
        dtype=np.uint8,
    )
    mask = np.zeros(4, dtype=np.bool_)
    mask[length:] = True
    return led, mask


def train_super(
    model: Super,
    corpus: NullCorpus,
    task: str = "masked_frame",
    epochs: int = 10,
    device: str = "cpu",
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 0,
    margin: float = 0.1,
    loss_weights: LossWeights | None = None,
    label_smoothing: float = 0.05,
) -> dict[str, Any]:
    """Train Super on one task or ``super_all`` (tasks 1–5)."""
    if task not in SUPER_TASKS:
        raise ValueError(f"unknown super task {task!r}; expected one of {SUPER_TASKS}")
    dev = _device(device)
    model = cast(Super, model.to(dev))
    model.train()
    rng = np.random.default_rng(seed)
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        bucket = decay if p.ndim >= 2 else no_decay
        bucket.append(cast(torch.nn.Parameter, p))
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 1e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )
    warmup_epochs = max(1, epochs // 10) if epochs > 1 else 0

    def _lr_lambda(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        span = max(1, epochs - warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(span)
        return 0.5 * (1.0 + float(np.cos(np.pi * min(1.0, max(0.0, progress)))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
    weights = loss_weights or LossWeights(
        state_ce=0.0,
        byte_ce=1.0,
        family_ce=0.5,
        payload_ce=0.5,
        signature=0.25,
        provenance=1.0,
        residual_zero=RESIDUAL_ZERO_WEIGHT,
    )
    tasks = list(SUPER_TASKS[:-1]) if task == "super_all" else [task]
    if getattr(model, "signature_mode", "exact") == "exact":
        tasks = [t for t in tasks if t != "sigword"]
    history: list[dict[str, float]] = []
    ema = _EMA(model)
    component_sums: dict[str, float] = {}
    epoch_loss = 0.0
    n_steps = 0

    def _step(total: torch.Tensor, logs: dict[str, float] | None = None) -> None:
        nonlocal epoch_loss, n_steps
        if logs:
            for k, v in logs.items():
                component_sums[k] = component_sums.get(k, 0.0) + float(v)
        if not torch.is_tensor(total) or not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {n_steps}")
        opt.zero_grad(set_to_none=True)
        if total.requires_grad:
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            ema.update(model)
        epoch_loss += float(total.detach())
        n_steps += 1

    train_frames = corpus.canonical_frames("train")
    train_cycles = corpus.canonical_cycles("train")
    train_pairs = corpus.collision_pairs("train")
    train_sigs = corpus.signature_words_split("train")

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_steps = 0
        component_sums = {}
        for task_name in tasks:
            if task_name == "masked_frame":
                pairs = [
                    (fi, pos)
                    for fi in range(len(train_frames))
                    for pos in range(4)
                ]
                order = rng.permutation(len(pairs))
                for start in range(0, len(order), batch_size):
                    chunk = [pairs[i] for i in order[start : start + batch_size]]
                    n = len(chunk)
                    ledgers = np.empty((n, 4), dtype=np.uint8)
                    masks = np.zeros((n, 4), dtype=np.bool_)
                    targets_b = np.empty(n, dtype=np.uint8)
                    for i, (fi, pos) in enumerate(chunk):
                        ledgers[i] = train_frames[fi]
                        masks[i, pos] = True
                        targets_b[i] = train_frames[fi, pos]
                    fam, payload, byte_t = _byte_targets(targets_b)
                    sig_parity, sig_tau_u, sig_tau_v = _sig_table_last(ledgers)
                    ledgers_t = torch.as_tensor(ledgers, dtype=torch.long, device=dev)
                    mask_t = torch.as_tensor(masks, dtype=torch.bool, device=dev)
                    targets = {
                        "byte": torch.as_tensor(byte_t, dtype=torch.long, device=dev),
                        "family": torch.as_tensor(fam, dtype=torch.long, device=dev),
                        "payload": torch.as_tensor(payload, dtype=torch.float32, device=dev),
                        "parity": torch.as_tensor(sig_parity, dtype=torch.long, device=dev),
                        "tau_u": torch.as_tensor(sig_tau_u, dtype=torch.long, device=dev),
                        "tau_v": torch.as_tensor(sig_tau_v, dtype=torch.long, device=dev),
                    }
                    w = LossWeights(
                        state_ce=0.0,
                        byte_ce=weights.byte_ce,
                        family_ce=weights.family_ce,
                        payload_ce=0.0,
                        signature=0.0,
                        residual_zero=RESIDUAL_ZERO_WEIGHT,
                    )
                    out = model(ledgers_t, mask=mask_t, hide_masked=True)
                    total, _logs = _loss_from_heads(
                        out, targets, w, label_smoothing=label_smoothing, arm="canonical"
                    )
                    _step(total, _logs)

            elif task_name == "holonomy":
                order = rng.permutation(len(train_cycles))
                for start in range(0, len(order), batch_size):
                    idx = order[start : start + batch_size]
                    batch = train_cycles[idx]
                    n = len(batch)
                    masks = np.zeros((n, 8), dtype=np.bool_)
                    targets_b = np.empty(n, dtype=np.uint8)
                    for i in range(n):
                        pos = int(rng.integers(0, 8))
                        masks[i, pos] = True
                        targets_b[i] = batch[i, pos]
                    fam, payload, byte_t = _byte_targets(targets_b)
                    sig_p, sig_u, sig_v = _sig_table_last(batch)
                    ledgers = torch.as_tensor(batch, dtype=torch.long, device=dev)
                    mask_t = torch.as_tensor(masks, dtype=torch.bool, device=dev)
                    targets = {
                        "byte": torch.as_tensor(byte_t, dtype=torch.long, device=dev),
                        "family": torch.as_tensor(fam, dtype=torch.long, device=dev),
                        "payload": torch.as_tensor(payload, dtype=torch.float32, device=dev),
                        "parity": torch.as_tensor(sig_p, dtype=torch.long, device=dev),
                        "tau_u": torch.as_tensor(sig_u, dtype=torch.long, device=dev),
                        "tau_v": torch.as_tensor(sig_v, dtype=torch.long, device=dev),
                    }
                    w = LossWeights(
                        state_ce=0.0,
                        byte_ce=weights.byte_ce,
                        family_ce=weights.family_ce,
                        payload_ce=weights.payload_ce,
                        signature=0.0,
                        residual_zero=RESIDUAL_ZERO_WEIGHT,
                    )
                    out = model(ledgers, mask=mask_t, hide_masked=True)
                    total, _logs = _loss_from_heads(
                        out, targets, w, label_smoothing=label_smoothing, arm="canonical"
                    )
                    _step(total, _logs)

            elif task_name == "provenance":
                if len(train_pairs) == 0:
                    continue
                order = rng.permutation(len(train_pairs))
                for start in range(0, len(order), batch_size):
                    rows = train_pairs[order[start : start + batch_size]]
                    la, lb, ma, mb = [], [], [], []
                    for row in rows:
                        a, b, mska, mskb = _collision_ledgers_masked(row)
                        la.append(a)
                        lb.append(b)
                        ma.append(mska)
                        mb.append(mskb)
                    ta = torch.as_tensor(np.stack(la), dtype=torch.long, device=dev)
                    tb = torch.as_tensor(np.stack(lb), dtype=torch.long, device=dev)
                    mta = torch.as_tensor(np.stack(ma), dtype=torch.bool, device=dev)
                    mtb = torch.as_tensor(np.stack(mb), dtype=torch.bool, device=dev)
                    za = model(ta, mask=mta, hide_masked=True)["context"]
                    zb = model(tb, mask=mtb, hide_masked=True)["context"]
                    prov = _contrastive_margin_loss(za, zb, margin=margin)
                    components = {"provenance": prov}
                    w = LossWeights(state_ce=0.0, provenance=weights.provenance)
                    total, _logs = weighted_total(components, w)
                    _step(total, _logs)

            elif task_name == "next_byte":
                frame_samples = [
                    (frame, t) for frame in train_frames for t in range(4)
                ]
                sig_samples = []
                for row in train_sigs[:: max(1, len(train_sigs) // 512)]:
                    led, _pad = _sig_row_ledger(row)
                    length = int(row["length"])
                    for t in range(max(length, 1)):
                        sig_samples.append((led, t))

                def _next_byte_epoch(samples, w):
                    nonlocal epoch_loss, n_steps
                    if not samples:
                        return
                    order_nb = rng.permutation(len(samples))
                    for start in range(0, len(order_nb), batch_size):
                        chunk = [samples[i] for i in order_nb[start : start + batch_size]]
                        n = len(chunk)
                        ledgers = np.zeros((n, 4), dtype=np.uint8)
                        masks = np.zeros((n, 4), dtype=np.bool_)
                        targets_b = np.empty(n, dtype=np.uint8)
                        for i, (frame, t) in enumerate(chunk):
                            ledgers[i] = frame
                            masks[i, t] = True
                            if t + 1 < 4:
                                ledgers[i, t + 1 :] = 0
                                masks[i, t + 1 :] = True
                            targets_b[i] = frame[t]
                        fam, payload, byte_t = _byte_targets(targets_b)
                        tbl = prefix_factor_table(ledgers.astype(np.int64))
                        ts = np.array([t for _, t in chunk], dtype=np.int64)
                        ts = np.clip(ts, 0, tbl.shape[1] - 1)
                        rows = np.arange(n)
                        sig_parity = tbl[rows, ts, 0]
                        sig_tau_u = tbl[rows, ts, 1]
                        sig_tau_v = tbl[rows, ts, 2]
                        ledgers_t = torch.as_tensor(ledgers, dtype=torch.long, device=dev)
                        mask_t = torch.as_tensor(masks, dtype=torch.bool, device=dev)
                        targets = {
                            "byte": torch.as_tensor(byte_t, dtype=torch.long, device=dev),
                            "family": torch.as_tensor(fam, dtype=torch.long, device=dev),
                            "payload": torch.as_tensor(payload, dtype=torch.float32, device=dev),
                            "parity": torch.as_tensor(sig_parity, dtype=torch.long, device=dev),
                            "tau_u": torch.as_tensor(sig_tau_u, dtype=torch.long, device=dev),
                            "tau_v": torch.as_tensor(sig_tau_v, dtype=torch.long, device=dev),
                        }
                        out = model(ledgers_t, mask=mask_t, hide_masked=True)
                        total, _logs = _loss_from_heads(
                            out, targets, w, label_smoothing=label_smoothing, arm="canonical"
                        )
                        _step(total, _logs)

                _next_byte_epoch(
                    frame_samples,
                    LossWeights(
                        state_ce=0.0,
                        byte_ce=weights.byte_ce,
                        family_ce=weights.family_ce,
                        payload_ce=weights.payload_ce,
                        signature=weights.signature,
                    ),
                )
                _next_byte_epoch(
                    sig_samples,
                    LossWeights(
                        state_ce=0.0,
                        byte_ce=0.0,
                        family_ce=0.0,
                        payload_ce=0.0,
                        signature=weights.signature,
                    ),
                )

            elif task_name == "corrupt":
                n_corrupt = corpus.corruption_n_bytes("train")
                order = rng.permutation(len(train_cycles))
                for start in range(0, len(order), batch_size):
                    idx = order[start : start + batch_size]
                    clean = train_cycles[idx]
                    n = len(clean)
                    corrupted = np.stack(
                        [_corrupt_ledger(clean[i], n_corrupt, rng) for i in range(n)]
                    )
                    masks = corrupted != clean
                    for i in range(n):
                        if not masks[i].any():
                            masks[i, int(rng.integers(0, 8))] = True
                    # Predict the first corrupted site; model sees corrupted bytes.
                    targets_b = np.array(
                        [clean[i, int(np.where(masks[i])[0][0])] for i in range(n)],
                        dtype=np.uint8,
                    )
                    fam, payload, byte_t = _byte_targets(targets_b)
                    # Single-site predict mask (first corrupt index).
                    pred_mask = np.zeros((n, 8), dtype=np.bool_)
                    for i in range(n):
                        pred_mask[i, int(np.where(masks[i])[0][0])] = True
                    ledgers_t = torch.as_tensor(corrupted, dtype=torch.long, device=dev)
                    mask_t = torch.as_tensor(pred_mask, dtype=torch.bool, device=dev)
                    tbl = prefix_factor_table(np.asarray(clean, dtype=np.int64))
                    pos = np.array(
                        [int(np.where(pred_mask[i])[0][0]) for i in range(n)],
                        dtype=np.int64,
                    )
                    pos = np.clip(pos, 0, tbl.shape[1] - 1)
                    rows = np.arange(n)
                    sig_parity = tbl[rows, pos, 0]
                    sig_tau_u = tbl[rows, pos, 1]
                    sig_tau_v = tbl[rows, pos, 2]
                    targets = {
                        "byte": torch.as_tensor(byte_t, dtype=torch.long, device=dev),
                        "family": torch.as_tensor(fam, dtype=torch.long, device=dev),
                        "payload": torch.as_tensor(payload, dtype=torch.float32, device=dev),
                        "parity": torch.as_tensor(sig_parity, dtype=torch.long, device=dev),
                        "tau_u": torch.as_tensor(sig_tau_u, dtype=torch.long, device=dev),
                        "tau_v": torch.as_tensor(sig_tau_v, dtype=torch.long, device=dev),
                    }
                    out = model(
                        ledgers_t,
                        mask=mask_t,
                        hide_masked=False,
                        corruption_mask=mask_t,
                    )
                    w = LossWeights(
                        state_ce=0.0,
                        byte_ce=weights.byte_ce,
                        family_ce=weights.family_ce,
                        payload_ce=weights.payload_ce,
                        signature=weights.signature,
                    )
                    total, _logs = _loss_from_heads(
                        out, targets, w, label_smoothing=label_smoothing, arm="seq"
                    )
                    _step(total, _logs)

            elif task_name == "sigword":
                order = rng.permutation(len(train_sigs))
                for start in range(0, len(order), batch_size):
                    rows = train_sigs[order[start : start + batch_size]]
                    ledgers = []
                    masks = []
                    sig_p = []
                    sig_u = []
                    sig_v = []
                    for row in rows:
                        led, msk = _sig_row_ledger(row)
                        ledgers.append(led)
                        masks.append(msk)
                        sig_p.append(int(row["parity"]))
                        sig_u.append(int(row["tau_u6"]))
                        sig_v.append(int(row["tau_v6"]))
                    ledgers_t = torch.as_tensor(np.stack(ledgers), dtype=torch.long, device=dev)
                    mask_t = torch.as_tensor(np.stack(masks), dtype=torch.bool, device=dev)
                    targets = {
                        "byte": torch.zeros(len(rows), dtype=torch.long, device=dev),
                        "family": torch.zeros(len(rows), dtype=torch.long, device=dev),
                        "payload": torch.zeros((len(rows), 6), dtype=torch.float32, device=dev),
                        "parity": torch.as_tensor(sig_p, dtype=torch.long, device=dev),
                        "tau_u": torch.as_tensor(sig_u, dtype=torch.long, device=dev),
                        "tau_v": torch.as_tensor(sig_v, dtype=torch.long, device=dev),
                    }
                    w = LossWeights(
                        state_ce=0.0,
                        byte_ce=0.0,
                        family_ce=0.0,
                        payload_ce=0.0,
                        signature=weights.signature,
                    )
                    out = model(ledgers_t, mask=mask_t, hide_masked=True)
                    total, _logs = _loss_from_heads(
                        out, targets, w, label_smoothing=label_smoothing, arm="canonical"
                    )
                    _step(total, _logs)

            elif task_name == "measure_lambda":
                from src.tools.autoencoder.helpers.evals_datasets import lambda_byte_corpus

                n_lam = max(batch_size * 12, 384)
                for lam in (0.5, 2.0, 4.0, 8.0):
                    ledgers_np = lambda_byte_corpus(
                        lam=lam,
                        n=n_lam,
                        length=8,
                        seed=seed + epoch * 10 + int(lam * 8),
                    )
                    _step_measure_ledgers(
                        model,
                        ledgers_np,
                        rng,
                        batch_size,
                        dev,
                        weights,
                        label_smoothing,
                        causal=False,
                        _step=_step,
                        arm="climate",
                    )

            elif task_name == "markov":
                from src.tools.autoencoder.helpers.evals_datasets import markov_ledgers

                n_mk = max(batch_size * 24, 768)
                for p_flip, extra in ((0.1, 0), (0.25, 101), (0.4, 202)):
                    ledgers_np = markov_ledgers(
                        n=n_mk,
                        length=16,
                        p_flip=p_flip,
                        seed=seed + epoch * 19 + 17 + extra,
                    )
                    _step_measure_ledgers(
                        model,
                        ledgers_np,
                        rng,
                        batch_size,
                        dev,
                        weights,
                        label_smoothing,
                        causal=True,
                        _step=_step,
                        arm="seq",
                    )

            elif task_name == "boundary":
                from src.tools.autoencoder.helpers.evals_datasets import boundary_rows

                rows_b = boundary_rows(n=max(batch_size * 4, 64), length=8, seed=seed + epoch + 31)
                order_b = rng.permutation(len(rows_b))
                for start in range(0, len(order_b), batch_size):
                    chunk = [rows_b[i] for i in order_b[start : start + batch_size]]
                    n = len(chunk)
                    t_len = len(chunk[0]["ledger"])
                    ledgers = np.stack([r["ledger"] for r in chunk])
                    masks = np.zeros((n, t_len), dtype=np.bool_)
                    targets_b = np.empty(n, dtype=np.uint8)
                    coset = np.zeros((n, 256), dtype=np.bool_)
                    for i, r in enumerate(chunk):
                        pos = int(r["pos"])
                        masks[i, pos] = True
                        targets_b[i] = r["ledger"][pos]
                        coset[i] = r["valid_mask"]
                    fam, payload, byte_t = _byte_targets(targets_b)
                    dummy_p = np.zeros(n, dtype=np.int64)
                    ledgers_t = torch.as_tensor(ledgers, dtype=torch.long, device=dev)
                    mask_t = torch.as_tensor(masks, dtype=torch.bool, device=dev)
                    coset_t = torch.as_tensor(coset, dtype=torch.bool, device=dev)
                    targets = {
                        "byte": torch.as_tensor(byte_t, dtype=torch.long, device=dev),
                        "family": torch.as_tensor(fam, dtype=torch.long, device=dev),
                        "payload": torch.as_tensor(payload, dtype=torch.float32, device=dev),
                        "parity": torch.as_tensor(dummy_p, dtype=torch.long, device=dev),
                        "tau_u": torch.as_tensor(dummy_p, dtype=torch.long, device=dev),
                        "tau_v": torch.as_tensor(dummy_p, dtype=torch.long, device=dev),
                    }
                    out = model(
                        ledgers_t, mask=mask_t, hide_masked=True, coset_mask=coset_t
                    )
                    out["coset_mask"] = coset_t
                    w = LossWeights(
                        state_ce=0.0,
                        byte_ce=weights.byte_ce,
                        family_ce=0.0,
                        payload_ce=0.0,
                        signature=0.0,
                    )
                    total, _logs = _loss_from_heads(
                        out, targets, w, label_smoothing=0.0, arm="seq"
                    )
                    shifts = torch.arange(13, device=dev)
                    sig_ids = torch.as_tensor(
                        [int(r["target_sig"]) for r in chunk],
                        dtype=torch.long,
                        device=dev,
                    )
                    sig_bits = (
                        torch.bitwise_right_shift(sig_ids.unsqueeze(-1), shifts) & 1
                    ).to(dtype=torch.float32)
                    positions = torch.arange(t_len, device=dev).unsqueeze(0).expand(n, -1)
                    dec = model.ledger_decoder(
                        sig_bits, out["context"].detach(), positions
                    )
                    pos_t = torch.as_tensor(
                        [int(r["pos"]) for r in chunk], dtype=torch.long, device=dev
                    )
                    pick = dec[torch.arange(n, device=dev), pos_t]
                    pick = pick.masked_fill(~coset_t, float("-inf"))
                    dec_loss = F.cross_entropy(pick, targets["byte"])
                    total = total + dec_loss
                    _logs["decode_ce"] = float(dec_loss.detach())
                    _step(total, _logs)

        if task in ("masked_frame", "super_all", "corrupt", "next_byte"):
            incomplete = corpus.incomplete_prior_frames(
                n=max(batch_size, 32), seed=seed + epoch, split="train"
            )
            order = rng.permutation(len(incomplete))
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                ledgers = incomplete[idx]
                n = len(ledgers)
                pos = rng.integers(0, 4, size=n)
                masks = np.zeros((n, 4), dtype=np.bool_)
                targets_b = np.empty(n, dtype=np.uint8)
                for i in range(n):
                    masks[i, int(pos[i])] = True
                    targets_b[i] = ledgers[i, int(pos[i])]
                fam, payload, byte_t = _byte_targets(targets_b)
                dummy_p = np.zeros(n, dtype=np.int64)
                ledgers_t = torch.as_tensor(ledgers, dtype=torch.long, device=dev)
                mask_t = torch.as_tensor(masks, dtype=torch.bool, device=dev)
                targets = {
                    "byte": torch.as_tensor(byte_t, dtype=torch.long, device=dev),
                    "family": torch.as_tensor(fam, dtype=torch.long, device=dev),
                    "payload": torch.as_tensor(payload, dtype=torch.float32, device=dev),
                    "parity": torch.as_tensor(dummy_p, dtype=torch.long, device=dev),
                    "tau_u": torch.as_tensor(dummy_p, dtype=torch.long, device=dev),
                    "tau_v": torch.as_tensor(dummy_p, dtype=torch.long, device=dev),
                }
                out = model(ledgers_t, mask=mask_t, hide_masked=True)
                w = LossWeights(
                    state_ce=0.0,
                    byte_ce=weights.byte_ce,
                    family_ce=0.0,
                    payload_ce=0.0,
                    signature=0.0,
                )
                total, _logs = _loss_from_heads(
                    out, targets, w, label_smoothing=0.0, arm="climate"
                )
                _step(total, _logs)

        mean_loss = epoch_loss / max(n_steps, 1)
        row = {"epoch": float(epoch), "loss": mean_loss}
        row.update({f"component_{k}": v / max(n_steps, 1) for k, v in component_sums.items()})
        history.append(row)
        sched.step()

    ema.copy_to(model)
    last_steps = max(n_steps, 1)
    return {
        "epochs_run": epochs,
        "task": task,
        "history": history,
        "final_loss": history[-1]["loss"] if history else 0.0,
        "component_loss": {
            k: float(v / last_steps) for k, v in component_sums.items()
        },
    }


def _step_measure_ledgers(
    model: Super,
    ledgers_np: np.ndarray,
    rng: np.random.Generator,
    batch_size: int,
    dev: torch.device,
    weights: LossWeights,
    label_smoothing: float,
    *,
    causal: bool,
    _step,
    arm: str = "completion",
) -> None:
    """Completion training on generated ledgers (λ or Markov)."""
    n_all, t_len = ledgers_np.shape
    order = rng.permutation(n_all)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        ledgers = ledgers_np[idx]
        n = len(ledgers)
        if causal:
            pos = rng.integers(1, t_len, size=n)
        else:
            pos = rng.integers(0, t_len, size=n)
        masks = np.zeros((n, t_len), dtype=np.bool_)
        cols = np.arange(t_len)
        if causal:
            masks = cols[None, :] >= pos[:, None]
        else:
            masks[np.arange(n), pos] = True
        targets_b = ledgers[np.arange(n), pos]
        fam, payload, byte_t = _byte_targets(targets_b)
        dummy_p = np.zeros(n, dtype=np.int64)
        ledgers_t = torch.as_tensor(ledgers, dtype=torch.long, device=dev)
        mask_t = torch.as_tensor(masks, dtype=torch.bool, device=dev)
        targets = {
            "byte": torch.as_tensor(byte_t, dtype=torch.long, device=dev),
            "family": torch.as_tensor(fam, dtype=torch.long, device=dev),
            "payload": torch.as_tensor(payload, dtype=torch.float32, device=dev),
            "parity": torch.as_tensor(dummy_p, dtype=torch.long, device=dev),
            "tau_u": torch.as_tensor(dummy_p, dtype=torch.long, device=dev),
            "tau_v": torch.as_tensor(dummy_p, dtype=torch.long, device=dev),
        }
        out = model(
            ledgers_t,
            mask=mask_t,
            hide_masked=True,
            apply_markov=bool(causal and arm == "seq"),
        )
        w = LossWeights(
            state_ce=0.0,
            byte_ce=weights.byte_ce,
            family_ce=0.0,
            payload_ce=0.0,
            signature=0.0,
            residual_zero=0.0,
        )
        total, _logs = _loss_from_heads(
            out, targets, w, label_smoothing=0.0, arm=arm
        )
        _step(total, _logs)


@torch.inference_mode()
def eval_g1(model: Super, corpus: NullCorpus, device: str = "cpu") -> float:
    """Masked-byte accuracy on held-out micro frames via the 256-way byte head."""
    model.eval()
    dev = _device(device)
    model = cast(Super, model.to(dev))
    frames = corpus.canonical_frames("holdout")
    n = len(frames)
    if n == 0:
        return 0.0
    correct = 0
    frames_t = torch.as_tensor(frames, dtype=torch.long, device=dev)
    for pos in range(4):
        mask = torch.zeros(n, 4, dtype=torch.bool, device=dev)
        mask[:, pos] = True
        out = model(frames_t, mask=mask, hide_masked=True)
        pred = out["byte_logits"].argmax(dim=-1).cpu().numpy()
        correct += int((pred == frames[:, pos]).sum())
    return float(correct) / float(4 * n)


@torch.inference_mode()
def eval_g2_replay(model: Super, corpus: NullCorpus, device: str = "cpu") -> float:
    """One-site fill with true context (not autoregressive). Exact byte + holonomy."""
    model.eval()
    dev = _device(device)
    model = cast(Super, model.to(dev))
    cycles = corpus.canonical_cycles("holdout")
    n = len(cycles)
    if n == 0:
        return 0.0
    recon = np.empty((n, 8), dtype=np.uint8)
    cycles_t = torch.as_tensor(cycles, dtype=torch.long, device=dev)
    for pos in range(8):
        mask = torch.zeros(n, 8, dtype=torch.bool, device=dev)
        mask[:, pos] = True
        out = model(cycles_t, mask=mask, hide_masked=True)
        recon[:, pos] = out["byte_logits"].argmax(dim=-1).cpu().numpy().astype(np.uint8)
    ok = 0
    for i, cycle in enumerate(cycles):
        st = int(GENE_MAC_REST)
        mid_ok = True
        for t in range(8):
            st = step_state_by_byte(st, int(recon[i, t]))
            if t == 3 and st != GENE_MAC_SWAPPED:
                mid_ok = False
        ok += int(mid_ok and st == GENE_MAC_REST and np.array_equal(recon[i], cycle))
    return float(ok) / float(n)


@torch.inference_mode()
def eval_residual_improvement(
    model: Super, corpus: NullCorpus, device: str = "cpu", n: int = 64
) -> float:
    """Prior NLL minus residual-augmented NLL on held-out incomplete frames (bits)."""
    model.eval()
    dev = _device(device)
    model = cast(Super, model.to(dev))
    frames = corpus.incomplete_prior_frames(n=n, seed=0, split="holdout")
    rng = np.random.default_rng(4242)
    prior_bits = []
    full_bits = []
    for frame in frames:
        pos = int(rng.integers(0, 4))
        mask = np.zeros(4, dtype=np.bool_)
        mask[pos] = True
        out = model(
            torch.as_tensor(frame, dtype=torch.long, device=dev).unsqueeze(0),
            mask=torch.as_tensor(mask, dtype=torch.bool, device=dev).unsqueeze(0),
            hide_masked=True,
        )
        tgt = int(frame[pos])
        prior = F.log_softmax(out["exact_prior_logits"], dim=-1)[0, tgt]
        full = F.log_softmax(out["byte_logits"], dim=-1)[0, tgt]
        prior_bits.append(float(-prior.item()) / np.log(2.0))
        full_bits.append(float(-full.item()) / np.log(2.0))
    return float(np.mean(prior_bits) - np.mean(full_bits))


def _masked_nll_bits(
    model: Super,
    ledgers: np.ndarray,
    device: str,
    *,
    causal: bool = False,
    batch_size: int = 32,
    head: str = "full",
) -> tuple[float, float]:
    dev = _device(device)
    n, t_len = ledgers.shape
    prior_b: list[float] = []
    full_b: list[float] = []
    use_markov = bool(causal and head == "seq")
    for start in range(0, n, batch_size):
        chunk = ledgers[start : start + batch_size]
        bsz = len(chunk)
        if causal:
            pos = np.full(bsz, t_len - 1, dtype=np.int64)
            masks = np.zeros((bsz, t_len), dtype=np.bool_)
            masks[:, t_len - 1] = True
        else:
            pos = np.array([int((start + i) % t_len) for i in range(bsz)], dtype=np.int64)
            masks = np.zeros((bsz, t_len), dtype=np.bool_)
            for i, p in enumerate(pos):
                masks[i, int(p)] = True
        out = model(
            torch.as_tensor(chunk, dtype=torch.long, device=dev),
            mask=torch.as_tensor(masks, dtype=torch.bool, device=dev),
            hide_masked=True,
            apply_markov=use_markov,
        )
        prior_logits = out["exact_prior_logits"]
        if head == "climate":
            pred_logits = prior_logits + out["climate_logits"]
        elif head == "seq":
            markov = out.get("markov_logits")
            pred_logits = prior_logits + out["seq_logits"] + (
                markov if markov is not None else 0.0
            )
        else:
            pred_logits = out["byte_logits"]
        log_prior = F.log_softmax(prior_logits, dim=-1)
        log_full = F.log_softmax(pred_logits, dim=-1)
        tgt = torch.as_tensor(
            chunk[np.arange(bsz), pos], dtype=torch.long, device=dev
        )
        prior_nll = -log_prior[torch.arange(bsz, device=dev), tgt] / np.log(2.0)
        full_nll = -log_full[torch.arange(bsz, device=dev), tgt] / np.log(2.0)
        prior_b.extend(prior_nll.detach().cpu().tolist())
        full_b.extend(full_nll.detach().cpu().tolist())
    return float(np.mean(prior_b)), float(np.mean(full_b))


@torch.inference_mode()
def eval_lambda_residual(
    model: Super, device: str = "cpu", lam: float = 4.0, n: int = 256
) -> dict[str, float]:
    from src.tools.autoencoder.helpers.evals_datasets import (
        lambda_byte_corpus,
        lambda_ceiling_bits,
    )

    model.eval()
    model = cast(Super, model.to(_device(device)))
    ledgers = lambda_byte_corpus(lam, n, length=8, seed=9973)
    prior_b, full_b = _masked_nll_bits(model, ledgers, device, causal=False, head="climate")
    ceiling = lambda_ceiling_bits(lam)
    improvement = prior_b - full_b
    return {
        "lam": float(lam),
        "prior_bits": prior_b,
        "model_bits": full_b,
        "improvement_bits": improvement,
        "ceiling_bits": ceiling,
        "fraction_of_ceiling": improvement / max(ceiling, 1e-9),
        "gap_to_bound_bits": full_b - (8.0 - ceiling),
    }


@torch.inference_mode()
def eval_markov_residual(
    model: Super, device: str = "cpu", p_flip: float = 0.1, n: int = 128
) -> dict[str, float]:
    from src.tools.autoencoder.helpers.evals_datasets import (
        markov_causal_bound_bits,
        markov_ledgers,
    )

    model.eval()
    model = cast(Super, model.to(_device(device)))
    ledgers = markov_ledgers(n, length=16, p_flip=p_flip, seed=9973)
    prior_b, full_b = _masked_nll_bits(model, ledgers, device, causal=True, head="seq")
    bound = markov_causal_bound_bits(p_flip)
    return {
        "p_flip": float(p_flip),
        "prior_bits": prior_b,
        "model_bits": full_b,
        "improvement_bits": prior_b - full_b,
        "bound_bits": bound,
        "gap_to_bound_bits": full_b - bound,
    }


@torch.inference_mode()
def eval_boundary_completion(
    model: Super, device: str = "cpu", n: int = 64
) -> dict[str, float]:
    from src.tools.autoencoder.helpers.evals_datasets import boundary_rows

    model.eval()
    dev = _device(device)
    model = cast(Super, model.to(dev))
    rows = boundary_rows(n=n, length=8, seed=9973)
    exact = 0
    valid = 0
    mass = []
    amb = []
    for r in rows:
        led = r["ledger"]
        pos = int(r["pos"])
        t_len = len(led)
        mask = np.zeros(t_len, dtype=np.bool_)
        mask[pos] = True
        coset = torch.as_tensor(r["valid_mask"], dtype=torch.bool, device=dev).unsqueeze(0)
        out = model(
            torch.as_tensor(led, dtype=torch.long, device=dev).unsqueeze(0),
            mask=torch.as_tensor(mask, dtype=torch.bool, device=dev).unsqueeze(0),
            hide_masked=True,
            coset_mask=coset,
        )
        logits = out["byte_logits"][0]
        pred = int(logits.argmax().item())
        tgt = int(led[pos])
        exact += int(pred == tgt)
        valid += int(bool(r["valid_mask"][pred]))
        probs = torch.softmax(logits, dim=-1)
        mass.append(float(probs[torch.as_tensor(r["valid_mask"], device=dev)].sum().item()))
        amb.append(float(r["ambiguity"]))
    return {
        "exact_byte_accuracy": float(exact) / float(len(rows)),
        "valid_completion_accuracy": float(valid) / float(len(rows)),
        "predicted_mass_on_valid_set": float(np.mean(mass)),
        "mean_ambiguity": float(np.mean(amb)),
    }


def eval_free_grammar_ablation(
    corpus: NullCorpus,
    context_dim: int,
    atom_dim: int,
    epochs: int = 4,
    device: str = "cpu",
    seed: int = 0,
    lr: float = 2e-3,
) -> dict[str, float]:
    """Train analytic vs free grammar twins; report G1/G7 for the report table."""
    rows = {}
    for name, analytic in (("analytic", True), ("free", False)):
        model = Super(
            context_dim=context_dim,
            atom_dim=atom_dim,
            frame_dim=max(context_dim, 32),
            analytic_grammar=analytic,
        )
        train_super(
            model,
            corpus,
            task="masked_frame",
            epochs=epochs,
            device=device,
            seed=seed,
            lr=lr,
            batch_size=32,
        )
        rows[f"{name}_g1"] = eval_g1(model, corpus, device=device)
        bits = eval_bits_per_byte(model, corpus, policy="canonical", device=device)
        rows[f"{name}_g7"] = float(bits["mean"])
    return rows


@torch.inference_mode()
def eval_g2_iterative(model: Super, corpus: NullCorpus, device: str = "cpu") -> float:
    """Iterative reconstruct: feed predicted bytes forward."""
    model.eval()
    dev = _device(device)
    model = cast(Super, model.to(dev))
    cycles = corpus.canonical_cycles("holdout")
    ok = 0
    for cycle in cycles:
        recon = cycle.copy()
        for pos in range(8):
            mask = np.zeros(8, dtype=np.bool_)
            mask[pos] = True
            out = model(
                torch.as_tensor(recon, dtype=torch.long, device=dev).unsqueeze(0),
                mask=torch.as_tensor(mask, dtype=torch.bool, device=dev).unsqueeze(0),
                hide_masked=True,
            )
            recon[pos] = int(out["byte_logits"].argmax(dim=-1).item())
        st = int(GENE_MAC_REST)
        mid_ok = True
        for t in range(8):
            st = step_state_by_byte(st, int(recon[t]))
            if t == 3 and st != GENE_MAC_SWAPPED:
                mid_ok = False
        ok += int(mid_ok and st == GENE_MAC_REST and np.array_equal(recon, cycle))
    return float(ok) / float(len(cycles)) if len(cycles) else 0.0


def _joint_word_ledger(
    wa: list[int], wb: list[int]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Concatenate two words; pad to 4 or 8. Skip pairs that exceed T=8."""
    joint = list(wa) + list(wb)
    if len(joint) > 8:
        return None
    cap = 8 if len(joint) > 4 else 4
    lj = np.zeros(cap, dtype=np.uint8)
    mj = np.ones(cap, dtype=np.bool_)
    for t, b in enumerate(joint):
        lj[t] = b
        mj[t] = False
    return lj, mj


def _pred_sig_factors(model: Super, ledger: np.ndarray, mask: np.ndarray, device: str):
    led_t = torch.as_tensor(ledger, dtype=torch.long, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
    if bool(mask_t.any()):
        out = model(led_t, mask=mask_t, hide_masked=True)
    else:
        out = model(led_t, mask=None, hide_masked=True)
    p = int(out["parity_logits"].argmax(dim=-1).item())
    u = int(out["tau_u_logits"].argmax(dim=-1).item())
    v = int(out["tau_v_logits"].argmax(dim=-1).item())
    return p, u, v


@torch.inference_mode()
def eval_g3_composition(
    model: Super, corpus: NullCorpus, device: str = "cpu", n_pairs: int = 256
) -> float:
    """Fraction of holdout word pairs with exact signature-head homomorphism."""
    model.eval()
    dev = _device(device)
    model = cast(Super, model.to(dev))
    words = corpus.signature_words_split("holdout")
    if len(words) < 2:
        return 0.0
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(words), size=min(n_pairs, len(words) // 2 * 2), replace=False)
    good = 0
    total = 0
    for i in range(0, len(idxs) - 1, 2):
        ra, rb = words[int(idxs[i])], words[int(idxs[i + 1])]
        la, ma = _sig_row_ledger(ra)
        lb, mb = _sig_row_ledger(rb)
        # Concatenate the full joint word; frame machinery already pads T to a multiple of 4.
        wa = [int(x) for x in la[: int(ra["length"])]]
        wb = [int(x) for x in lb[: int(rb["length"])]]
        packed = _joint_word_ledger(wa, wb)
        if packed is None:
            continue
        lj, mj = packed
        p1, u1, v1 = _pred_sig_factors(model, la, ma, str(dev))
        p2, u2, v2 = _pred_sig_factors(model, lb, mb, str(dev))
        pj, uj, vj = _pred_sig_factors(model, lj, mj, str(dev))
        s1 = api.OmegaSignature12(p1, u1, v1)
        s2 = api.OmegaSignature12(p2, u2, v2)
        pred_comp = api.compose_omega_signatures(s2, s1)
        true1 = api.OmegaSignature12(int(ra["parity"]), int(ra["tau_u6"]), int(ra["tau_v6"]))
        true2 = api.OmegaSignature12(int(rb["parity"]), int(rb["tau_u6"]), int(rb["tau_v6"]))
        true_comp = api.compose_omega_signatures(true2, true1)
        pred_joint = (pj, uj, vj)
        good += int(
            pred_joint == (true_comp.parity, true_comp.tau_u6, true_comp.tau_v6)
            and (pred_comp.parity, pred_comp.tau_u6, pred_comp.tau_v6)
            == (true_comp.parity, true_comp.tau_u6, true_comp.tau_v6)
        )
        total += 1
    return float(good) / float(total) if total else 0.0


@torch.inference_mode()
def eval_g4_provenance(
    model: Super,
    corpus: NullCorpus,
    margin: float = 0.1,
    device: str = "cpu",
) -> dict[str, float]:
    """Provenance separation vs an untrained-model floor on holdout components."""
    model.eval()
    dev = _device(device)
    model = cast(Super, model.to(dev))
    pairs = corpus.collision_pairs("holdout")
    if len(pairs) == 0:
        return {
            "rate": 0.0,
            "untrained_floor": 0.0,
            "margin_over_floor": 0.0,
            "rate_margin_0p3": 0.0,
            "mean_cosine_distance": 0.0,
        }

    def _distances(m: Super) -> np.ndarray:
        dists = []
        for row in pairs:
            a, b, ma, mb = _collision_ledgers_masked(row)
            za = m(
                torch.as_tensor(a, dtype=torch.long, device=dev).unsqueeze(0),
                mask=torch.as_tensor(ma, dtype=torch.bool, device=dev).unsqueeze(0),
                hide_masked=True,
            )["context"]
            zb = m(
                torch.as_tensor(b, dtype=torch.long, device=dev).unsqueeze(0),
                mask=torch.as_tensor(mb, dtype=torch.bool, device=dev).unsqueeze(0),
                hide_masked=True,
            )["context"]
            cos = float(F.cosine_similarity(za, zb, dim=-1).item())
            dists.append(1.0 - cos)
        return np.asarray(dists, dtype=np.float64)

    trained_d = _distances(model)
    fresh = cast(
        Super,
        Super(
            context_dim=model.context_dim,
            atom_dim=model.atom_dim,
            frame_dim=model.frame_dim,
        ).to(dev),
    )
    fresh.eval()
    floor_d = _distances(fresh)
    trained = float((trained_d > margin).mean())
    floor = float((floor_d > margin).mean())
    return {
        "rate": trained,
        "untrained_floor": floor,
        "margin_over_floor": float(trained - floor),
        "rate_margin_0p3": float((trained_d > 0.3).mean()),
        "mean_cosine_distance": float(trained_d.mean()),
    }


@torch.inference_mode()
def eval_bits_per_byte(
    model: Super,
    corpus: NullCorpus,
    policy: str = "canonical",
    device: str = "cpu",
) -> dict[str, float]:
    """Mean next-byte NLL in bits from the 256-way head; also per-position."""
    model.eval()
    dev = _device(device)
    model = cast(Super, model.to(dev))

    def _nll_bits(out: dict[str, torch.Tensor], byte: int) -> float:
        log_p = F.log_softmax(out["byte_logits"], dim=-1)[0, byte]
        return float(-log_p.item()) / np.log(2.0)

    if policy == "canonical":
        frames = corpus.canonical_frames("holdout")
        by_pos: dict[int, list[float]] = {0: [], 1: [], 2: [], 3: []}
        for frame in frames:
            for t in range(4):
                ledger = frame.copy()
                mask = np.zeros(4, dtype=np.bool_)
                mask[t] = True
                if t + 1 < 4:
                    ledger[t + 1 :] = 0
                    mask[t + 1 :] = True
                out = model(
                    torch.as_tensor(ledger, dtype=torch.long, device=dev).unsqueeze(0),
                    mask=torch.as_tensor(mask, dtype=torch.bool, device=dev).unsqueeze(0),
                    hide_masked=True,
                )
                by_pos[t].append(_nll_bits(out, int(frame[t])))
        means = {p: float(np.mean(v)) if v else 8.0 for p, v in by_pos.items()}
        all_v = [x for v in by_pos.values() for x in v]
        return {
            "mean": float(np.mean(all_v)) if all_v else 8.0,
            **{f"pos_{p}": means[p] for p in range(4)},
        }

    if policy == "uniform":
        rng = np.random.default_rng(0)
        nlls = []
        for _ in range(512):
            frame = rng.integers(0, 256, size=4, dtype=np.uint8)
            t = int(rng.integers(0, 4))
            ledger = frame.copy()
            mask = np.zeros(4, dtype=np.bool_)
            mask[t] = True
            out = model(
                torch.as_tensor(ledger, dtype=torch.long, device=dev).unsqueeze(0),
                mask=torch.as_tensor(mask, dtype=torch.bool, device=dev).unsqueeze(0),
                hide_masked=True,
            )
            nlls.append(_nll_bits(out, int(frame[t])))
        return {"mean": float(np.mean(nlls)) if nlls else 8.0}

    raise ValueError(f"unknown policy {policy!r}")


@torch.inference_mode()
def eval_corrupt_replay(
    model: Super, corpus: NullCorpus, device: str = "cpu"
) -> float:
    """Holdout 2-byte corruption: recover until kernel holonomy rest→swapped→rest."""
    model.eval()
    dev = _device(device)
    model = cast(Super, model.to(dev))
    cycles = corpus.canonical_cycles("holdout")
    n_corrupt = corpus.corruption_n_bytes("holdout")
    rng = np.random.default_rng(1)
    ok = 0
    for cycle in cycles:
        corrupted = _corrupt_ledger(cycle, n_corrupt, rng)
        recon = corrupted.copy()
        sites = np.where(corrupted != cycle)[0]
        for pos in sites:
            mask = np.zeros(8, dtype=np.bool_)
            mask[pos] = True
            out = model(
                torch.as_tensor(recon, dtype=torch.long, device=dev).unsqueeze(0),
                mask=torch.as_tensor(mask, dtype=torch.bool, device=dev).unsqueeze(0),
                hide_masked=False,
                corruption_mask=torch.as_tensor(mask, dtype=torch.bool, device=dev).unsqueeze(0),
            )
            recon[pos] = int(out["byte_logits"].argmax(dim=-1).item())
        st = int(GENE_MAC_REST)
        mid_ok = True
        for t in range(8):
            st = step_state_by_byte(st, int(recon[t]))
            if t == 3 and st != GENE_MAC_SWAPPED:
                mid_ok = False
        ok += int(mid_ok and st == GENE_MAC_REST)
    return float(ok) / float(len(cycles)) if len(cycles) else 0.0


@torch.inference_mode()
def eval_g5_provenance_beats_null(
    model: Super,
    corpus: NullCorpus,
    device: str = "cpu",
    margin: float = 0.1,
) -> float:
    """Super provenance rate minus a path-blind Ω-endpoint baseline."""
    g4 = eval_g4_provenance(model, corpus, margin=margin, device=device)
    pairs = corpus.collision_pairs("holdout", kind="same_end")
    if len(pairs) == 0:
        pairs = corpus.collision_pairs("holdout")
    if len(pairs) == 0:
        return 0.0
    # State-only: same endpoint => identical key => separation rate = 0 by construction.
    null_rate = 0.0
    for row in pairs:
        if str(row["kind"]) == "same_end":
            if int(row["endpoint_a"]) != int(row["endpoint_b"]):
                null_rate = 1.0  # should never happen
                break
    return float(g4["rate"] - null_rate)


def latent_factorization_probe(
    model: Super, corpus: NullCorpus, device: str = "cpu"
) -> dict[str, float]:
    """Linear probe from context to (parity, τu, τv) on signature_words holdout."""
    model.eval()
    dev = _device(device)
    model = cast(Super, model.to(dev))
    train = corpus.signature_words_split("train")
    hold = corpus.signature_words_split("holdout")

    def _embed(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ctx, yp, yu, yv = [], [], [], []
        with torch.inference_mode():
            for row in rows:
                led, mask = _sig_row_ledger(row)
                out = model(
                    torch.as_tensor(led, dtype=torch.long, device=dev).unsqueeze(0),
                    mask=torch.as_tensor(mask, dtype=torch.bool, device=dev).unsqueeze(0),
                    hide_masked=True,
                )
                ctx.append(out["context"].cpu().numpy().reshape(-1))
                yp.append(int(row["parity"]))
                yu.append(int(row["tau_u6"]))
                yv.append(int(row["tau_v6"]))
        return (
            np.stack(ctx).astype(np.float64),
            np.asarray(yp, dtype=np.int64),
            np.asarray(yu, dtype=np.int64),
            np.asarray(yv, dtype=np.int64),
        )

    # Subsample train for speed.
    rng = np.random.default_rng(0)
    if len(train) > 1024:
        train = train[rng.choice(len(train), size=1024, replace=False)]
    if len(hold) > 512:
        hold = hold[rng.choice(len(hold), size=512, replace=False)]

    Xtr, yptr, yutr, yvtr = _embed(train)
    Xte, ypte, yute, yvte = _embed(hold)
    Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1))], axis=1)
    Xte = np.concatenate([Xte, np.ones((len(Xte), 1))], axis=1)

    def _fit_acc(
        X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, n_classes: int
    ) -> float:
        Y = np.eye(n_classes, dtype=np.float64)[y_tr]
        xtx = X_tr.T @ X_tr + 1e-6 * np.eye(X_tr.shape[1])
        W = np.linalg.solve(xtx, X_tr.T @ Y)
        pred = (X_te @ W).argmax(axis=1)
        return float((pred == y_te).mean())

    # Exact 13-bit: pack predicted factors.
    def _exact() -> float:
        # Probe each factor separately then pack.
        Yp = np.eye(2, dtype=np.float64)[yptr]
        Yu = np.eye(64, dtype=np.float64)[yutr]
        Yv = np.eye(64, dtype=np.float64)[yvtr]
        xtx = Xtr.T @ Xtr + 1e-6 * np.eye(Xtr.shape[1])
        Wp = np.linalg.solve(xtx, Xtr.T @ Yp)
        Wu = np.linalg.solve(xtx, Xtr.T @ Yu)
        Wv = np.linalg.solve(xtx, Xtr.T @ Yv)
        pp = (Xte @ Wp).argmax(axis=1)
        pu = (Xte @ Wu).argmax(axis=1)
        pv = (Xte @ Wv).argmax(axis=1)
        pred_id = (pp << 12) | (pu << 6) | pv
        true_id = (ypte << 12) | (yute << 6) | yvte
        return float((pred_id == true_id).mean())

    def _mlp_tau_u() -> float:
        xt = torch.as_tensor(Xtr, dtype=torch.float32)
        yt = torch.as_tensor(yutr, dtype=torch.long)
        xe = torch.as_tensor(Xte, dtype=torch.float32)
        ye = torch.as_tensor(yute, dtype=torch.long)
        probe = nn.Sequential(
            nn.Linear(xt.shape[1], 32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )
        opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
        probe.train()
        for _ in range(40):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(xt), yt)
            loss.backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            pred = probe(xe).argmax(dim=-1)
        return float((pred == ye).to(dtype=torch.float32).mean().item())

    return {
        "parity_accuracy": _fit_acc(Xtr, yptr, Xte, ypte, 2),
        "tau_u_accuracy": _fit_acc(Xtr, yutr, Xte, yute, 64),
        "tau_v_accuracy": _fit_acc(Xtr, yvtr, Xte, yvte, 64),
        "exact_13bit_accuracy": _exact(),
        "tau_u_mlp_accuracy": _mlp_tau_u(),
        "holdout_parity_nunique": int(len(np.unique(ypte))),
        "holdout_tau_u_nunique": int(len(np.unique(yute))),
        "holdout_tau_v_nunique": int(len(np.unique(yvte))),
    }


def evaluate_gates(
    model: Super,
    corpus: NullCorpus,
    device: str = "cpu",
    margin: float = 0.1,
    *,
    include_probe: bool = True,
    lite: bool = False,
) -> dict[str, float]:
    """Run the Super gate suite and return a report dict.

    ``lite=True`` still runs the cheap measure gates (λ residual, Markov
    residual, boundary completion) because the capacity sweep ranks on them.
    It only skips iterative G2, corrupt replay, residual probe, and the
    latent-factorization probe.
    """
    bits_c = eval_bits_per_byte(model, corpus, policy="canonical", device=device)
    bits_u = eval_bits_per_byte(model, corpus, policy="uniform", device=device)
    g1 = eval_g1(model, corpus, device=device)
    g2 = eval_g2_replay(model, corpus, device=device)
    g3 = eval_g3_composition(model, corpus, device=device)
    g4 = eval_g4_provenance(model, corpus, margin=margin, device=device)
    g5 = eval_g5_provenance_beats_null(model, corpus, device=device, margin=margin)
    # Measure gates always run: capacity_sweep ranks on λ fraction and Markov Δ.
    lam = eval_lambda_residual(model, device=device)
    mk = eval_markov_residual(model, device=device)
    bd = eval_boundary_completion(model, device=device)
    out = {
        "G1_masked_holdout": g1,
        "G2_replay": g2,
        "G3_composition": g3,
        "G4_provenance": g4["rate"],
        "G4_untrained_floor": g4["untrained_floor"],
        "G4_margin_over_floor": g4["margin_over_floor"],
        "G4_rate_margin_0p3": g4.get("rate_margin_0p3", g4["rate"]),
        "G4_mean_cosine_distance": g4.get("mean_cosine_distance", 0.0),
        "G5_provenance_margin_over_null": g5,
        "G6_uniform_bits_per_byte": bits_u["mean"],
        "G7_canonical_bits_per_byte": bits_c["mean"],
        "G7_pos_0": bits_c.get("pos_0", bits_c["mean"]),
        "G7_pos_1": bits_c.get("pos_1", bits_c["mean"]),
        "G7_pos_2": bits_c.get("pos_2", bits_c["mean"]),
        "G7_pos_3": bits_c.get("pos_3", bits_c["mean"]),
        "bits_per_byte_canonical": bits_c["mean"],
        "bits_per_byte_uniform": bits_u["mean"],
        "G2_iterative": 0.0,
        "corrupt_replay_holdout": 0.0,
        "residual_improvement_bits": 0.0,
        "lambda_improvement_bits": lam["improvement_bits"],
        "lambda_ceiling_bits": lam["ceiling_bits"],
        "lambda_fraction_of_ceiling": lam["fraction_of_ceiling"],
        "lambda_prior_bits": lam["prior_bits"],
        "lambda_model_bits": lam["model_bits"],
        "markov_improvement_bits": mk["improvement_bits"],
        "markov_gap_to_bound_bits": mk["gap_to_bound_bits"],
        "markov_model_bits": mk["model_bits"],
        "markov_bound_bits": mk["bound_bits"],
        "boundary_valid_completion": bd["valid_completion_accuracy"],
        "boundary_exact_byte": bd["exact_byte_accuracy"],
        "boundary_mass_on_valid": bd["predicted_mass_on_valid_set"],
        "boundary_mean_ambiguity": bd["mean_ambiguity"],
        "context_probe_parity_accuracy": 0.0,
        "context_probe_tau_u_accuracy": 0.0,
        "context_probe_tau_v_accuracy": 0.0,
        "context_probe_exact_13bit_accuracy": 0.0,
        "context_probe_tau_u_mlp_accuracy": 0.0,
        "context_probe_holdout_parity_nunique": 2.0,
        "context_probe_holdout_tau_u_nunique": 2.0,
        "context_probe_holdout_tau_v_nunique": 2.0,
    }
    if lite:
        return out
    out["G2_iterative"] = eval_g2_iterative(model, corpus, device=device)
    out["corrupt_replay_holdout"] = eval_corrupt_replay(model, corpus, device=device)
    out["residual_improvement_bits"] = eval_residual_improvement(
        model, corpus, device=device
    )
    if include_probe:
        probe = latent_factorization_probe(model, corpus, device=device)
        out["context_probe_parity_accuracy"] = probe["parity_accuracy"]
        out["context_probe_tau_u_accuracy"] = probe["tau_u_accuracy"]
        out["context_probe_tau_v_accuracy"] = probe["tau_v_accuracy"]
        out["context_probe_exact_13bit_accuracy"] = probe["exact_13bit_accuracy"]
        out["context_probe_tau_u_mlp_accuracy"] = probe.get("tau_u_mlp_accuracy", 0.0)
        out["context_probe_holdout_parity_nunique"] = float(probe["holdout_parity_nunique"])
        out["context_probe_holdout_tau_u_nunique"] = float(probe["holdout_tau_u_nunique"])
        out["context_probe_holdout_tau_v_nunique"] = float(probe["holdout_tau_v_nunique"])
    return out


def gate_pass_flags(gates: dict[str, float]) -> dict[str, bool]:
    return {
        "g1_pass": gates["G1_masked_holdout"] >= 1.0 - 1e-12,
        "g2_pass": gates["G2_replay"] >= 1.0 - 1e-12,
        "g3_pass": gates["G3_composition"] >= 1.0 - 1e-12,
        "g4_pass": gates["G4_margin_over_floor"] > 0.0,
        "g5_pass": gates["G5_provenance_margin_over_null"] > 0.0,
        # Two-sided uniformity check: below 8 means hallucinated structure;
        # far above means degenerate saturation.
        "g6_pass": abs(gates["G6_uniform_bits_per_byte"] - 8.0) <= G6_UNIFORM_TOL,
        "g7_pass": gates["G7_canonical_bits_per_byte"] <= G7_BITS_MAX,
        "g9_pass": (
            gates["context_probe_holdout_parity_nunique"] >= 2
            and gates["context_probe_holdout_tau_u_nunique"] >= 2
            and gates["context_probe_holdout_tau_v_nunique"] >= 2
        ),
    }


def capacity_sweep(
    corpus: NullCorpus,
    context_dims: tuple[int, ...] = (16, 32, 64),
    atom_dims: tuple[int, ...] = (8, 16, 32),
    epochs: int = 8,
    device: str = "cpu",
    seed: int = 0,
    task: str = "super_all",
    lr: float = 2e-3,
) -> list[dict[str, Any]]:
    """Grid over GRU sizes. Each candidate is trained then scored on G1–G7
    plus the measure gates (λ / Markov / boundary) via ``evaluate_gates(..., lite=True)``.
    """
    results = []
    for cd in context_dims:
        for ad in atom_dims:
            model = Super(context_dim=cd, atom_dim=ad, frame_dim=max(cd, 32))
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            stats = train_super(
                model,
                corpus,
                task=task,
                epochs=epochs,
                device=device,
                seed=seed,
                lr=lr,
            )
            gates = evaluate_gates(model, corpus, device=device, lite=True)
            flags = gate_pass_flags(gates)
            mandatory = ("g1_pass", "g2_pass", "g3_pass", "g4_pass", "g5_pass", "g6_pass", "g7_pass")
            results.append(
                {
                    "context_dim": cd,
                    "atom_dim": ad,
                    "n_params": int(n_params),
                    "final_loss": stats["final_loss"],
                    "gates": gates,
                    "flags": flags,
                    "all_mandatory_pass": all(flags[k] for k in mandatory),
                    "model": model,
                }
            )
    return results


# ---------------------------------------------------------------------------
# Gate runner
# ---------------------------------------------------------------------------

def main_gates() -> int:
    p = argparse.ArgumentParser(
        description="Train Super, sweep capacity, write super.pt and super_gates.json."
    )
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--sweep-epochs", type=int, default=8)
    p.add_argument("--task", default="super_all")
    p.add_argument("--context-dim", type=int, default=64)
    p.add_argument("--atom-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument(
        "--context-dims",
        default="16,32",
        help="comma-separated context dims for the capacity sweep",
    )
    p.add_argument(
        "--atom-dims",
        default="8",
        help="comma-separated atom dims for the capacity sweep",
    )
    p.add_argument(
        "--gate-version",
        type=int,
        default=3,
        help="schema version written into super_gates.json (v1 is the pre-frame-GRU record)",
    )
    p.add_argument(
        "--ablate-grammar",
        action="store_true",
        default=True,
        help="train analytic vs free grammar twins for the report ablation table",
    )
    p.add_argument(
        "--no-ablate-grammar",
        action="store_false",
        dest="ablate_grammar",
    )
    p.add_argument(
        "--continue-epochs",
        type=int,
        default=6,
        help="extra super_all epochs on the sweep winner before final gates (0 disables)",
    )
    p.add_argument(
        "--continue-lr",
        type=float,
        default=1e-3,
        help="learning rate for continue-train on the sweep winner",
    )
    p.add_argument(
        "--markov-path-epochs",
        type=int,
        default=2,
        help="opt-in Markov-path polish epochs after continue-train (0 disables)",
    )
    p.add_argument(
        "--no-continue-train",
        action="store_true",
        help="skip continue-train and Markov-path polish; ship the sweep winner as-is",
    )
    p.add_argument(
        "--out",
        default=str(paths.reports_dir() / "super_gates.json"),
    )
    p.add_argument(
        "--checkpoint",
        default=str(paths.checkpoints_dir() / "production" / "super.pt"),
    )
    args = p.parse_args()

    corpus = NullCorpus()
    weights = LossWeights(
        state_ce=0.0,
        byte_ce=1.0,
        family_ce=0.5,
        payload_ce=0.5,
        signature=0.25,
        provenance=1.0,
        residual_zero=RESIDUAL_ZERO_WEIGHT,
    )
    context_dims = tuple(int(x) for x in str(args.context_dims).split(",") if x.strip())
    atom_dims = tuple(int(x) for x in str(args.atom_dims).split(",") if x.strip())

    sweep = capacity_sweep(
        corpus,
        context_dims=context_dims,
        atom_dims=atom_dims,
        epochs=args.sweep_epochs,
        device=args.device,
        seed=args.seed,
        task=args.task,
        lr=args.lr,
    )
    def _sweep_score(r: dict[str, Any]) -> tuple:
        g = r["gates"]
        return (
            int(r["all_mandatory_pass"]),
            float(g.get("lambda_fraction_of_ceiling", 0.0)),
            float(g.get("markov_improvement_bits", 0.0)),
            float(g.get("G4_margin_over_floor", 0.0)),
            -int(r["n_params"]),
        )

    minimal = max(sweep, key=_sweep_score)

    # Promote the sweep winner (same lr/epochs/seed); do not retrain under different hparams.
    winner = minimal.get("model")
    if isinstance(winner, Super):
        model = winner
        epochs_run = int(args.sweep_epochs)
        fit = {"final_loss": float(minimal.get("final_loss", 0.0)), "epochs_run": epochs_run}
    else:
        model = Super(
            context_dim=int(minimal["context_dim"]),
            atom_dim=int(minimal["atom_dim"]),
            frame_dim=max(int(minimal["context_dim"]), 32),
        )
        fit = train_super(
            model,
            corpus,
            task=args.task,
            epochs=args.epochs,
            device=args.device,
            lr=args.lr,
            seed=args.seed,
            loss_weights=weights,
        )
        epochs_run = int(fit.get("epochs_run", args.epochs))

    continue_train: dict[str, Any] = {}
    if not args.no_continue_train and int(args.continue_epochs) > 0:
        baseline = evaluate_gates(model, corpus, device=args.device, lite=True)
        cont = train_super(
            model,
            corpus,
            task=args.task,
            epochs=int(args.continue_epochs),
            device=args.device,
            lr=float(args.continue_lr),
            seed=args.seed + 17,
            loss_weights=weights,
        )
        epochs_run += int(cont.get("epochs_run", args.continue_epochs))
        continue_train = {
            "epochs": int(args.continue_epochs),
            "lr": float(args.continue_lr),
            "baseline_markov": float(baseline.get("markov_improvement_bits", 0.0)),
            "baseline_lambda": float(baseline.get("lambda_fraction_of_ceiling", 0.0)),
            "markov_path_epochs": 0,
        }
        if int(args.markov_path_epochs) > 0:
            # Polish only the opt-in Markov path so G6/G7 grammar stays calibrated.
            frozen: list[tuple[str, bool]] = []
            for name, param in model.named_parameters():
                allow = (
                    "markov_flip" in name
                    or "prev_micro_proj" in name
                    or name.startswith("byte_head.seq_mlp")
                    or ".seq_mlp." in name
                )
                frozen.append((name, bool(param.requires_grad)))
                param.requires_grad_(allow)
            try:
                mk_fit = train_super(
                    model,
                    corpus,
                    task="markov",
                    epochs=int(args.markov_path_epochs),
                    device=args.device,
                    lr=float(args.continue_lr),
                    seed=args.seed + 23,
                    loss_weights=weights,
                )
            finally:
                by_name = dict(frozen)
                for name, param in model.named_parameters():
                    param.requires_grad_(by_name.get(name, True))
            epochs_run += int(mk_fit.get("epochs_run", args.markov_path_epochs))
            continue_train["markov_path_epochs"] = int(args.markov_path_epochs)
        fit = {
            "final_loss": float(cont.get("final_loss", fit.get("final_loss", 0.0))),
            "epochs_run": int(epochs_run),
        }

    gates = evaluate_gates(model, corpus, device=args.device)
    flags = gate_pass_flags(gates)
    if continue_train:
        continue_train["markov_gap"] = float(gates.get("markov_gap_to_bound_bits", 0.0))
        continue_train["markov_improvement"] = float(
            gates.get("markov_improvement_bits", 0.0)
        )
        continue_train["lambda_fraction"] = float(
            gates.get("lambda_fraction_of_ceiling", 0.0)
        )

    multi_seed = []
    if int(args.n_seeds) >= 3:
        for s in range(int(args.n_seeds)):
            seeded = Super(
                context_dim=int(minimal["context_dim"]),
                atom_dim=int(minimal["atom_dim"]),
                frame_dim=max(int(minimal["context_dim"]), 32),
            )
            train_super(
                seeded,
                corpus,
                task=args.task,
                epochs=args.epochs,
                device=args.device,
                lr=args.lr,
                seed=args.seed + s,
                loss_weights=weights,
            )
            sg = evaluate_gates(seeded, corpus, device=args.device, lite=True)
            multi_seed.append(
                {
                    "seed": args.seed + s,
                    "G1": sg["G1_masked_holdout"],
                    "G4_margin_over_floor": sg["G4_margin_over_floor"],
                    "G7": sg["G7_canonical_bits_per_byte"],
                }
            )

    grammar_ablation = {}
    if getattr(args, "ablate_grammar", False):
        grammar_ablation = eval_free_grammar_ablation(
            corpus,
            context_dim=int(minimal["context_dim"]),
            atom_dim=int(minimal["atom_dim"]),
            epochs=min(4, int(args.epochs)),
            device=args.device,
            seed=args.seed,
            lr=args.lr,
        )

    ckpt = Path(args.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": {},
            "extra": {
                "model_kind": "super",
                "model_config": model.get_config(),
                "fit": fit,
                "gates": gates,
                "gate_flags": flags,
            },
            "model_state": model.state_dict(),
        },
        ckpt,
    )

    try:
        ckpt_rel = ckpt.resolve().relative_to(Path(__file__).resolve().parents[4]).as_posix()
    except ValueError:
        ckpt_rel = ckpt.as_posix().replace("\\", "/")

    report = {
        "checkpoint": ckpt_rel,
        "gate_version": int(getattr(args, "gate_version", 2)),
        "fit": fit,
        "gates": gates,
        "g7_bits_max": G7_BITS_MAX,
        "g6_uniform_tol": G6_UNIFORM_TOL,
        "residual_improvement_bits": float(gates.get("residual_improvement_bits", 0.0)),
        "lambda_fraction_of_ceiling": float(gates.get("lambda_fraction_of_ceiling", 0.0)),
        "boundary_valid_completion": float(gates.get("boundary_valid_completion", 0.0)),
        "claim": (
            "exact_grammar_plus_provenance_and_residual"
            if float(gates.get("lambda_fraction_of_ceiling", 0.0)) >= 0.5
            or float(gates.get("markov_improvement_bits", 0.0)) > 0.05
            or float(gates.get("residual_improvement_bits", 0.0)) > 0.05
            else "exact_grammar_plus_provenance"
        ),
        **flags,
        "grid": {
            "context_dims": list(context_dims),
            "atom_dims": list(atom_dims),
            "sweep_epochs": int(args.sweep_epochs),
            "epochs": int(args.epochs),
            "task": args.task,
            "capacity_eval": "measure",
            "continue_epochs": int(args.continue_epochs) if not args.no_continue_train else 0,
            "markov_path_epochs": (
                int(args.markov_path_epochs) if not args.no_continue_train else 0
            ),
        },
        "g8_minimal": {
            "context_dim": minimal["context_dim"],
            "atom_dim": minimal["atom_dim"],
            "n_params": minimal["n_params"],
            "all_mandatory_pass": minimal["all_mandatory_pass"],
            "G4_margin_full": gates["G4_margin_over_floor"],
            "G4_margin_lite": minimal["gates"]["G4_margin_over_floor"],
        },
        "g9_probe": {
            "parity": gates["context_probe_parity_accuracy"],
            "tau_u": gates["context_probe_tau_u_accuracy"],
            "tau_v": gates["context_probe_tau_v_accuracy"],
            "tau_u_mlp": gates.get("context_probe_tau_u_mlp_accuracy", 0.0),
            "exact_13bit": gates["context_probe_exact_13bit_accuracy"],
            "note": "G9 is a linear/MLP readout of GRU context; tau_u non-recovery is documented, not a fail.",
        },
        "grammar_ablation": grammar_ablation,
        "continue_train": continue_train,
        "capacity_sweep": [
            {
                "context_dim": r["context_dim"],
                "atom_dim": r["atom_dim"],
                "n_params": r["n_params"],
                "all_mandatory_pass": r["all_mandatory_pass"],
                "G1": r["gates"]["G1_masked_holdout"],
                "G3": r["gates"]["G3_composition"],
                "G4_margin": r["gates"]["G4_margin_over_floor"],
                "G4_rate_0p3": r["gates"].get("G4_rate_margin_0p3"),
                "lambda_frac": r["gates"].get("lambda_fraction_of_ceiling"),
                "markov_improvement": r["gates"].get("markov_improvement_bits"),
                "G6": r["gates"]["G6_uniform_bits_per_byte"],
                "G7": r["gates"]["G7_canonical_bits_per_byte"],
            }
            for r in sweep
        ],
        "multi_seed": multi_seed,
        "g4_margin_min": (
            min(float(r["G4_margin_over_floor"]) for r in multi_seed)
            if multi_seed
            else float(gates["G4_margin_over_floor"])
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out, report)
    print(json.dumps({k: report[k] for k in report if k != "capacity_sweep"}, indent=2))

    mandatory = ("g1_pass", "g2_pass", "g3_pass", "g4_pass", "g5_pass", "g6_pass", "g7_pass")
    failed = [k for k in mandatory if not flags[k]]
    if failed:
        print(f"mandatory gates failed: {failed}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Production regeneration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = REPO_ROOT / "src" / "tools" / "autoencoder"
DATA_HOME = PACKAGE_ROOT / "data"
CHECKPOINTS = DATA_HOME / "checkpoints" / "production"
REPORTS = DATA_HOME / "reports"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _rel(p: Path) -> str:
    return p.resolve().relative_to(REPO_ROOT).as_posix()


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        print(
            f"command failed (rc={proc.returncode}): {' '.join(cmd)}",
            file=sys.stderr,
        )
        sys.exit(proc.returncode)


def _train(flag: str, *, epochs: int) -> Path:
    model = "mlp" if flag.startswith("mlp") else "k4"
    out = CHECKPOINTS / f"{flag}.pt"
    cmd = [
        sys.executable,
        "-m",
        "src.tools.autoencoder.cli",
        "train",
        "--model",
        model,
        "--output-dir",
        str(CHECKPOINTS),
        "--run-name",
        flag,
        "--epochs",
        str(epochs),
        "--learning-rate",
        "1e-3",
        "--val-fraction",
        "0.15",
        "--patience",
        "5",
    ]
    _run(cmd)
    return out


def _evaluate(ckpt: Path, *, suffix: str) -> Path:
    out = REPORTS / f"{suffix}_eval.json"
    _run(
        [
            sys.executable,
            "-m",
            "src.tools.autoencoder.cli",
            "evaluate",
            "--checkpoint",
            str(ckpt),
            "--report-file",
            str(out),
        ]
    )
    return out


def _verify(ckpt: Path, *, suffix: str) -> Path:
    out = REPORTS / f"{suffix}_eq.json"
    _run(
        [
            sys.executable,
            "-m",
            "src.tools.autoencoder.cli",
            "verify-equivariance",
            "--checkpoint",
            str(ckpt),
            "--seed",
            "0",
            "--report-file",
            str(out),
        ]
    )
    return out


def _ensure_super(*, skip_train: bool) -> tuple[Path, Path]:
    ckpt = CHECKPOINTS / "super.pt"
    gates = REPORTS / "super_gates.json"
    if skip_train and ckpt.exists() and gates.exists():
        prev_path = REPORTS / "production_summary.json"
        if prev_path.exists():
            prev = json.loads(prev_path.read_text(encoding="utf-8"))
            expected = (prev.get("super") or {}).get("sha256")
            if expected and _sha256_file(ckpt) != expected:
                print(
                    "super.pt sha256 does not match production_summary; "
                    "refusing --skip-train",
                    file=sys.stderr,
                )
                sys.exit(2)
        return ckpt, gates
    if not skip_train or not ckpt.exists() or not gates.exists():
        _run(
            [
                sys.executable,
                "-m",
                "src.tools.autoencoder.helpers.training_super",
                "gates",
                "--task",
                "super_all",
                "--epochs",
                "8",
                "--sweep-epochs",
                "8",
                "--out",
                str(gates),
                "--checkpoint",
                str(ckpt),
            ]
        )
    return ckpt, gates


def main_production(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="reuse existing checkpoints; regenerate reports where needed",
    )
    args = parser.parse_args(argv)

    REPORTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, str]] = {}

    if args.skip_train:
        checkpoints = {
            "k4_full": CHECKPOINTS / "k4_full.pt",
            "mlp_full": CHECKPOINTS / "mlp_full.pt",
        }
        prev_path = REPORTS / "production_summary.json"
        prev = (
            json.loads(prev_path.read_text(encoding="utf-8"))
            if prev_path.exists()
            else {}
        )
        for name, ckpt in checkpoints.items():
            expected = (prev.get(name) or {}).get("sha256")
            if expected and ckpt.exists() and _sha256_file(ckpt) != expected:
                print(
                    f"{name} sha256 does not match production_summary; "
                    "refusing --skip-train",
                    file=sys.stderr,
                )
                return 2
    else:
        checkpoints = {
            "k4_full": _train("k4_full", epochs=700),
            "mlp_full": _train("mlp_full", epochs=2000),
        }

    for name, ckpt in checkpoints.items():
        if not ckpt.exists():
            print(f"missing checkpoint: {ckpt}", file=sys.stderr)
            return 2
        summary[name] = {
            "checkpoint": _rel(ckpt),
            "eval": _rel(_evaluate(ckpt, suffix=name)),
            "equivariance": _rel(_verify(ckpt, suffix=name)),
            "sha256": _sha256_file(ckpt),
        }

    super_ckpt, super_gates = _ensure_super(skip_train=args.skip_train)
    if not super_ckpt.exists() or not super_gates.exists():
        print("missing Super checkpoint or gate report", file=sys.stderr)
        return 2
    gates_doc = json.loads(super_gates.read_text(encoding="utf-8"))
    mandatory = ("g1_pass", "g2_pass", "g3_pass", "g4_pass", "g5_pass", "g6_pass", "g7_pass")
    failed = [k for k in mandatory if not gates_doc.get(k)]
    if failed:
        print(
            f"Super gate report failed mandatory gates {failed}; "
            "refusing to publish production_summary Super entry",
            file=sys.stderr,
        )
        return 1
    residual_bits = float(
        (gates_doc.get("gates") or {}).get("residual_improvement_bits", 0.0)
    )
    lam_frac = float(
        (gates_doc.get("gates") or {}).get("lambda_fraction_of_ceiling", 0.0)
    )
    summary["super"] = {
        "checkpoint": _rel(super_ckpt),
        "gates": _rel(super_gates),
        "sha256": _sha256_file(super_ckpt),
        "claim": gates_doc.get("claim")
        or (
            "exact_grammar_plus_provenance_and_residual"
            if lam_frac >= 0.5
            or residual_bits > 0.05
            or float((gates_doc.get("gates") or {}).get("markov_improvement_bits", 0.0))
            > 0.05
            else "exact_grammar_plus_provenance"
        ),
    }

    out = REPORTS / "production_summary.json"
    _atomic_write_json(out, summary)
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI: ``production`` or Super gates (default, including bare flags)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print(
            "usage: python -m src.tools.autoencoder.helpers.training_super "
            "[gates|production] ...\n"
            "  gates       sweep/retrain Super and write super_gates.json (default)\n"
            "  production  regenerate production checkpoints and summary"
        )
        return 0
    if argv and argv[0] == "production":
        return main_production(argv[1:])
    if argv and argv[0] == "gates":
        argv = argv[1:]
    old = sys.argv
    try:
        sys.argv = [old[0], *argv]
        return main_gates()
    finally:
        sys.argv = old


if __name__ == "__main__":
    raise SystemExit(main())
