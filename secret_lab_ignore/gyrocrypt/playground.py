#!/usr/bin/env python3
"""Robust playground for SHA-256 structure decoders in one place.

This module runs both:
- Test 1: Chirality projection and WHT spectral structure on SHA-256 digests.
- Test 2: One-round decomposition-like probes over message schedules and depth-4 frame signatures.
"""

from __future__ import annotations

import gzip
import hashlib
import math
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if not (ROOT / "src").is_dir():
    for _depth in (2, 1, 0):
        _candidate = Path(__file__).resolve().parents[_depth]
        if (_candidate / "src").is_dir():
            ROOT = _candidate
            break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import sha256_structure as _sha_module

chirality_histogram64 = _sha_module.chirality_histogram64
chirality_words_histogram64 = _sha_module.chirality_words_histogram64
sha256_block_words_from_messages = _sha_module.sha256_block_words_from_messages
random_block_words = _sha_module.random_block_words
random_messages = _sha_module.random_messages
digest_to_chirality6_words = _sha_module.digest_to_chirality6_words
schedule_depth4_frame_signatures = _sha_module.schedule_depth4_frame_signatures
schedule_depth4_projection_lift = _sha_module.schedule_depth4_projection_lift
sha256_round_chirality_trajectory = _sha_module.sha256_round_chirality_trajectory
sha256_digests = _sha_module.sha256_digests
sha256_round_profiles = _sha_module.sha256_round_profiles
schedule_omega_trajectory = _sha_module.schedule_omega_trajectory
topk_energy_fractions = _sha_module.topk_energy_fractions
wht64 = _sha_module.wht64
from src.api import byte_to_intron, intron_family, intron_micro_ref, q_word6

WORD_COUNT = 42
TOP_KS = (1, 2, 4, 8, 16, 32)
DEFAULT_RUN = {
    "samples": 8192,
    "round_blocks": 12000,
    "reps": 3,
    "seed": 42,
    "out": Path("secret_lab_ignore/gyrocrypt/benchmarks/gyrocrypt_stress_report.json"),
}

MAX_TEST2_BLOCKS = {
    "profile": 12000,
    "depth": 12000,
    "projection": 12000,
    "trajectory": 12000,
    "omega": 12000,
}


def _entropy_bits(p: np.ndarray) -> float:
    mask = p > 0.0
    if not np.any(mask):
        return 0.0
    x = p[mask]
    return float(-np.sum(x * np.log2(x)))


def _random_digest_bytes(sample_count: int, seed: int) -> list[bytes]:
    rng = np.random.default_rng(seed)
    return [bytes(rng.integers(0, 256, size=32, dtype=np.uint8)) for _ in range(sample_count)]


def _blake2s_digests(messages: list[bytes]) -> list[bytes]:
    return [hashlib.blake2s(m, digest_size=32).digest() for m in messages]


def _blake2s_chain_block_words(messages: list[bytes]) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for msg in messages:
        d1 = hashlib.blake2s(msg, digest_size=32).digest()
        d2 = hashlib.blake2s(d1, digest_size=32).digest()
        ext = d1 + d2
        out.append(tuple(int.from_bytes(ext[i : i + 4], "big") for i in range(0, 64, 4)))
    return out


def _aqpu_native_digest_features(digest_list: list[bytes]) -> dict[str, Any]:
    """Build digest-stream features in aQPU-native byte coordinates."""
    if not digest_list:
        return {
            "family_entropy": 0.0,
            "family_bit0": 0.0,
            "family_bit7": 0.0,
            "payload_entropy": 0.0,
            "payload_bit_entropy": 0.0,
            "q_entropy": 0.0,
            "commutativity_collision_rate": 0.0,
            "q_topk": {k: 0.0 for k in TOP_KS},
            "payload_topk": {k: 0.0 for k in TOP_KS},
        }

    family_counts = np.zeros(4, dtype=np.float64)
    payload_counts = np.zeros(64, dtype=np.float64)
    q_counts = np.zeros(64, dtype=np.float64)
    payload_bit_counts = np.zeros(6, dtype=np.float64)
    family_bit0_sum = 0.0
    family_bit7_sum = 0.0
    collision_rates: list[float] = []
    byte_total = 0.0

    for digest in digest_list:
        q_seq = []
        for b in digest:
            intron = byte_to_intron(b)
            family = intron_family(intron)
            payload = intron_micro_ref(intron)
            qv = q_word6(int(b))
            byte_total += 1.0
            family_counts[family] += 1.0
            payload_counts[payload] += 1.0
            q_counts[qv] += 1.0
            q_seq.append(int(qv))
            family_bit0_sum += float(intron & 1)
            family_bit7_sum += float((intron >> 7) & 1)
            for bit_idx in range(6):
                payload_bit_counts[bit_idx] += float((payload >> bit_idx) & 1)

        if len(q_seq) >= 2:
            q_hist = np.bincount(np.asarray(q_seq, dtype=np.int64), minlength=64).astype(np.float64)
            total_pairs = float(len(q_seq) * (len(q_seq) - 1) // 2)
            if total_pairs > 0.0:
                dup_pairs = float(np.sum(q_hist * (q_hist - 1.0)) / 2.0)
                collision_rates.append(dup_pairs / total_pairs)

    family_prob = family_counts / float(np.sum(family_counts))
    payload_prob = payload_counts / float(np.sum(payload_counts))
    q_prob = q_counts / float(np.sum(q_counts))

    payload_bits_entropy = []
    for bit_idx in range(6):
        ones = payload_bit_counts[bit_idx]
        p = ones / byte_total if byte_total > 0.0 else 0.0
        payload_bits_entropy.append(_entropy_bits(np.array([p, 1.0 - p], dtype=np.float64)))

    family_bit0 = family_bit0_sum / byte_total if byte_total > 0.0 else 0.0
    family_bit7 = family_bit7_sum / byte_total if byte_total > 0.0 else 0.0

    q_topk = _wht_centered_topk(q_counts, TOP_KS)
    payload_topk = _wht_centered_topk(payload_counts, TOP_KS)

    return {
        "family_entropy": float(_entropy_bits(family_prob)),
        "family_bit0": float(family_bit0),
        "family_bit7": float(family_bit7),
        "payload_entropy": float(_entropy_bits(payload_prob)),
        "payload_bit_entropy": float(np.mean(payload_bits_entropy)),
        "q_entropy": float(_entropy_bits(q_prob)),
        "commutativity_collision_rate": float(np.mean(collision_rates)) if collision_rates else 0.0,
        "q_topk": {k: float(v) for k, v in q_topk.items()},
        "payload_topk": {k: float(v) for k, v in payload_topk.items()},
    }


def _wht_centered_topk(hist: np.ndarray, ks: tuple[int, ...]) -> dict[int, float]:
    centered = np.asarray(hist, dtype=np.float64) - float(np.mean(hist))
    return topk_energy_fractions(wht64(centered), ks)


def _hamming_u32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xor_vals = np.bitwise_xor(a.astype(np.uint64), b.astype(np.uint64)).astype(np.uint64)
    return np.fromiter((int(v).bit_count() for v in xor_vals), dtype=np.float64)


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=0))


def _env_int(name: str, default: int) -> int:
    """Optional environment override for run knobs."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        _progress(f"invalid env {name}={value!r}; using {default}")
        return default


def _env_mode_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower()


def _cap_block_rows(blocks: list[tuple[int, ...]], cap: int) -> list[tuple[int, ...]]:
    if cap <= 0 or len(blocks) <= cap:
        return blocks
    return blocks[:cap]


def _progress(message: str) -> None:
    print(message, flush=True)


def _run_mode() -> dict[str, int | str]:
    mode = _env_mode_str("GYROCRYPT_MODE", "").strip().lower()
    if mode in {"full", "max", "high_power", "hq"}:
        rounds = _env_int("GYROCRYPT_ROUND_BLOCKS", 50000)
        reps = _env_int("GYROCRYPT_REPS", 10)
    elif mode in {"fast", "quick", "smoke"}:
        rounds = _env_int("GYROCRYPT_ROUND_BLOCKS", 4000)
        reps = _env_int("GYROCRYPT_REPS", 3)
    elif mode in {"normal", "default", ""}:
        rounds = _env_int("GYROCRYPT_ROUND_BLOCKS", DEFAULT_RUN["round_blocks"])
        reps = _env_int("GYROCRYPT_REPS", DEFAULT_RUN["reps"])
    else:
        _progress(f"unknown GYROCRYPT_MODE={mode!r}; using default mode")
        rounds = _env_int("GYROCRYPT_ROUND_BLOCKS", DEFAULT_RUN["round_blocks"])
        reps = _env_int("GYROCRYPT_REPS", DEFAULT_RUN["reps"])
    return {"mode": mode, "round_blocks": rounds, "reps": reps}


def _resolve_metric_caps(mode: str, block_count: int) -> dict[str, int]:
    if mode in {"full", "max", "high_power", "hq"}:
        return {key: block_count for key in MAX_TEST2_BLOCKS}
    if mode in {"fast", "quick", "smoke"}:
        fast_cap = _env_int("GYROCRYPT_BLOCK_CAP", 4000)
        return {key: min(block_count, fast_cap) for key in MAX_TEST2_BLOCKS}
    return {key: min(block_count, cap) for key, cap in MAX_TEST2_BLOCKS.items()}


def _fmt_ms(ms: float) -> str:
    if ms >= 1000.0:
        return f"{ms / 1000.0:.2f}s"
    return f"{ms:.0f}ms"


def _run_test1_once(sample_count: int, seed: int) -> dict[str, Any]:
    _progress(f"    _run_test1_once sample_count={sample_count} seed={seed}")
    msgs = random_messages(sample_count=sample_count, seed=seed, min_len=1, max_len=64)
    _progress("    building sha digests")
    digest_list = sha256_digests(msgs)
    _progress("    building control digest sets")
    uniform_random = _random_digest_bytes(sample_count=sample_count, seed=seed + 11)
    uniform_random_alt = _random_digest_bytes(sample_count=sample_count, seed=seed + 12)
    null_msgs = random_messages(sample_count=sample_count, seed=seed + 13, min_len=1, max_len=64)
    null_sha = sha256_digests(null_msgs)
    blake_digests = _blake2s_digests(msgs)

    primary = _run_test1_compare(
        digests=digest_list,
        random_digests=uniform_random,
        seed=seed,
        sample_count=sample_count,
    )
    null_controls = {
        "sha_random": _run_test1_compare(
            digests=digest_list,
            random_digests=null_sha,
            seed=seed,
            sample_count=sample_count,
        ),
        "uniform_random": _run_test1_compare(
            digests=digest_list,
            random_digests=uniform_random_alt,
            seed=seed,
            sample_count=sample_count,
        ),
        "blake2s": _run_test1_compare(
            digests=digest_list,
            random_digests=blake_digests,
            seed=seed,
            sample_count=sample_count,
        ),
    }
    primary["null_controls"] = null_controls
    return primary

def _run_test1_compare(
    digests: list[bytes],
    random_digests: list[bytes],
    seed: int,
    sample_count: int,
) -> dict[str, Any]:
    sha_hist = chirality_histogram64(digests).astype(np.float64)
    rnd_hist = chirality_histogram64(random_digests).astype(np.float64)
    sha_prob = sha_hist / float(np.sum(sha_hist))
    rnd_prob = rnd_hist / float(np.sum(rnd_hist))

    sha_spec = wht64(sha_hist - np.mean(sha_hist))
    rnd_spec = wht64(rnd_hist - np.mean(rnd_hist))
    sha_top = topk_energy_fractions(sha_spec, (1, 2, 4, 8, 16, 32))
    rnd_top = topk_energy_fractions(rnd_spec, (1, 2, 4, 8, 16, 32))

    topk_residual = {k: float(abs(sha_top[k] - rnd_top[k])) for k in sha_top}
    multi = _multi_chirality_summary(digests, random_digests)
    sha_native = _aqpu_native_digest_features(digests)
    rnd_native = _aqpu_native_digest_features(random_digests)
    byte_native_residual = {
        "family_entropy": float(abs(sha_native["family_entropy"] - rnd_native["family_entropy"])),
        "family_bit0": float(abs(sha_native["family_bit0"] - rnd_native["family_bit0"])),
        "family_bit7": float(abs(sha_native["family_bit7"] - rnd_native["family_bit7"])),
        "payload_entropy": float(abs(sha_native["payload_entropy"] - rnd_native["payload_entropy"])),
        "payload_bit_entropy": float(abs(sha_native["payload_bit_entropy"] - rnd_native["payload_bit_entropy"])),
        "q_entropy": float(abs(sha_native["q_entropy"] - rnd_native["q_entropy"])),
        "commutativity_collision_rate": float(
            abs(sha_native["commutativity_collision_rate"] - rnd_native["commutativity_collision_rate"])
        ),
        "q_topk_residual": {k: float(abs(sha_native["q_topk"][k] - rnd_native["q_topk"][k])) for k in TOP_KS},
        "payload_topk_residual": {
            k: float(abs(sha_native["payload_topk"][k] - rnd_native["payload_topk"][k])) for k in TOP_KS
        },
    }
    return {
        "seed": seed,
        "sample_count": sample_count,
        "hist_entropy_bits": _entropy_bits(sha_prob),
        "max_abs_vs_random": float(np.max(np.abs(sha_prob - rnd_prob))),
        "topk_sha": sha_top,
        "topk_random": rnd_top,
        "topk_residual": topk_residual,
        "multi": {
            "entropy_mean": multi["entropy_mean"],
            "entropy_std": multi["entropy_std"],
            "max_abs_vs_random": multi["max_abs_vs_random"],
            "transition_entropy": multi["transition_entropy"],
            "topk_sha": multi["topk_sha"],
            "topk_random": multi["topk_random"],
            "topk_residual": multi["topk_residual"],
        },
        "byte_native": {
            "family_entropy": sha_native["family_entropy"],
            "family_bit0": sha_native["family_bit0"],
            "family_bit7": sha_native["family_bit7"],
            "payload_entropy": sha_native["payload_entropy"],
            "payload_bit_entropy": sha_native["payload_bit_entropy"],
            "q_entropy": sha_native["q_entropy"],
            "commutativity_collision_rate": sha_native["commutativity_collision_rate"],
            "q_topk": sha_native["q_topk"],
            "payload_topk": sha_native["payload_topk"],
            "family_random_entropy": rnd_native["family_entropy"],
            "family_random_bit0": rnd_native["family_bit0"],
            "family_random_bit7": rnd_native["family_bit7"],
            "payload_random_entropy": rnd_native["payload_entropy"],
            "payload_random_bit_entropy": rnd_native["payload_bit_entropy"],
            "q_random_entropy": rnd_native["q_entropy"],
            "random_commutativity_collision_rate": rnd_native["commutativity_collision_rate"],
            "q_topk_random": rnd_native["q_topk"],
            "payload_topk_random": rnd_native["payload_topk"],
            "residual": byte_native_residual,
        },
    }


def _multi_chirality_summary(
    digest_list: list[bytes],
    random_digest_list: list[bytes],
    word_count: int = WORD_COUNT,
) -> dict[str, Any]:
    sha_hist = chirality_words_histogram64(digest_list, word_count=word_count).astype(np.float64)
    rnd_hist = chirality_words_histogram64(random_digest_list, word_count=word_count).astype(np.float64)

    sha_denom = sha_hist.sum(axis=1, keepdims=True)
    rnd_denom = rnd_hist.sum(axis=1, keepdims=True)
    sha_prob = sha_hist / sha_denom
    rnd_prob = rnd_hist / rnd_denom

    entropy_by_pos = [_entropy_bits(p) for p in sha_prob]
    trans_counts = np.zeros(7, dtype=np.float64)
    random_trans_counts = np.zeros(7, dtype=np.float64)

    for digest, random_digest in zip(digest_list, random_digest_list):
        seq = digest_to_chirality6_words(digest, word_count=word_count)
        seq2 = digest_to_chirality6_words(random_digest, word_count=word_count)
        for a, b in zip(seq[:-1], seq[1:]):
            trans_counts[int((a ^ b).bit_count())] += 1
        for a, b in zip(seq2[:-1], seq2[1:]):
            random_trans_counts[int((a ^ b).bit_count())] += 1

    total = float(len(digest_list) * max(word_count - 1, 1))
    trans_prob = trans_counts / total if total > 0 else np.zeros(7, dtype=np.float64)
    random_trans_prob = random_trans_counts / total if total > 0 else np.zeros(7, dtype=np.float64)

    topk_sha = {k: [] for k in TOP_KS}
    topk_rand = {k: [] for k in TOP_KS}
    for pos in range(word_count):
        t_sha = _wht_centered_topk(sha_hist[pos], TOP_KS)
        t_rand = _wht_centered_topk(rnd_hist[pos], TOP_KS)
        for k in TOP_KS:
            topk_sha[k].append(float(t_sha[k]))
            topk_rand[k].append(float(t_rand[k]))

    return {
        "entropy_mean": float(float(np.mean(entropy_by_pos)) if entropy_by_pos else 0.0),
        "entropy_std": float(float(np.std(entropy_by_pos)) if entropy_by_pos else 0.0),
        "max_abs_vs_random": float(float(np.max(np.abs(sha_prob - rnd_prob)))),
        "transition_entropy": float(_entropy_bits(trans_prob)),
        "topk_sha": {k: float(np.mean(v)) for k, v in topk_sha.items()},
        "topk_random": {k: float(np.mean(v)) for k, v in topk_rand.items()},
        "topk_residual": {
            k: float(abs(float(np.mean(topk_sha[k])) - float(np.mean(topk_rand[k]))))
            for k in TOP_KS
        },
    }


def _round_profile_mean(blocks: list[tuple[int, ...]]) -> dict[str, float]:
    accum: dict[str, float] = {}
    n = float(len(blocks))
    for block in blocks:
        profile = sha256_round_profiles(block)
        for key, val in profile.items():
            accum[key] = accum.get(key, 0.0) + float(val)
    for key in accum:
        accum[key] /= n
    return accum


def _depth4_signature_summary(blocks: list[tuple[int, ...]]) -> dict[str, float]:
    all_sigs = np.array([schedule_depth4_frame_signatures(b) for b in blocks], dtype=np.uint64)
    low8 = (all_sigs & 0xFF).astype(np.float64)
    ent_cols = []
    for col_idx in range(low8.shape[1]):
        counts = np.bincount(low8[:, col_idx].astype(np.int64), minlength=256).astype(np.float64)
        p = counts / float(np.sum(counts))
        ent_cols.append(_entropy_bits(p))
    hams = []
    for idx in range(all_sigs.shape[1] - 1):
        h = _hamming_u32(all_sigs[:, idx], all_sigs[:, idx + 1])
        hams.append(float(np.mean(h)))
    return {
        "entropy_mean": float(np.mean(ent_cols)),
        "entropy_min": float(np.min(ent_cols)),
        "entropy_max": float(np.max(ent_cols)),
        "hamming_mean": float(np.mean(hams)),
        "hamming_min": float(np.min(hams)),
        "hamming_max": float(np.max(hams)),
    }


def _projection_lift_summary(blocks: list[tuple[int, ...]]) -> dict[str, float]:
    projection_counts: dict[int, int] = {}
    lift_sets: dict[int, set[int]] = {}
    pair_set: set[tuple[int, int]] = set()
    omega_mask_counts = np.zeros(16, dtype=np.float64)
    total_frames = 0

    for block in blocks:
        for frame in schedule_depth4_projection_lift(block):
            projection = int(frame["projection48"])
            lift = int(frame["lift32"])
            omega_mask4 = int(frame["omega_mask4"]) & 0xF
            total_frames += 1
            projection_counts[projection] = projection_counts.get(projection, 0) + 1
            lift_sets.setdefault(projection, set()).add(lift)
            pair_set.add((projection, lift))
            omega_mask_counts[omega_mask4] += 1.0

    if total_frames <= 0:
        return {
            "projection_collision_fraction": 0.0,
            "pair_collision_fraction": 0.0,
            "mean_projection_multiplicity": 0.0,
            "mean_lifts_per_projection": 0.0,
            "max_lifts_per_projection": 0.0,
            "lift_resolution_fraction": 0.0,
            "omega_mask_entropy": 0.0,
            "omega_mask_support": 0.0,
        }

    distinct_projection = float(len(projection_counts))
    distinct_pairs = float(len(pair_set))
    multiplicities = np.array(list(projection_counts.values()), dtype=np.float64)
    lifts_per_projection = np.array([len(v) for v in lift_sets.values()], dtype=np.float64)
    lost_at_projection = float(total_frames) - distinct_projection
    recovered_by_lift = max(0.0, distinct_pairs - distinct_projection)
    omega_prob = omega_mask_counts / float(np.sum(omega_mask_counts))

    return {
        "projection_collision_fraction": float(1.0 - distinct_projection / float(total_frames)),
        "pair_collision_fraction": float(1.0 - distinct_pairs / float(total_frames)),
        "mean_projection_multiplicity": float(np.mean(multiplicities)),
        "mean_lifts_per_projection": float(np.mean(lifts_per_projection)),
        "max_lifts_per_projection": float(np.max(lifts_per_projection)),
        "lift_resolution_fraction": (
            float(recovered_by_lift / lost_at_projection) if lost_at_projection > 0.0 else 0.0
        ),
        "omega_mask_entropy": float(_entropy_bits(omega_prob)),
        "omega_mask_support": float(np.count_nonzero(omega_mask_counts)),
    }


def _round_trajectory_summary(blocks: list[tuple[int, ...]]) -> dict[str, Any]:
    if not blocks:
        return {
            "entropy": 0.0,
            "split_entropy_0_15": 0.0,
            "split_entropy_16_63": 0.0,
            "hop_mean": 0.0,
            "hop_mean_0_15": 0.0,
            "hop_mean_16_63": 0.0,
            "topk": {k: 0.0 for k in TOP_KS},
        }

    trajectories = [sha256_round_chirality_trajectory(block) for block in blocks]
    traj = np.array(trajectories, dtype=np.uint8)

    all_hist = np.bincount(traj.reshape(-1), minlength=64).astype(np.float64)
    all_prob = all_hist / float(np.sum(all_hist))

    first = traj[:, :16].reshape(-1)
    last = traj[:, 16:].reshape(-1)
    first_hist = np.bincount(first, minlength=64).astype(np.float64)
    last_hist = np.bincount(last, minlength=64).astype(np.float64)

    hops = []
    hops_first = []
    hops_last = []
    for row in traj:
        d = np.fromiter(
            (int((int(a) ^ int(b)).bit_count()) for a, b in zip(row[:-1], row[1:])),
            dtype=np.float64,
        )
        if d.size > 0:
            hops.append(float(np.mean(d)))
            hops_first.append(float(np.mean(d[:15])))
            hops_last.append(float(np.mean(d[15:])))

    topk = {k: [] for k in TOP_KS}
    for row in traj:
        counts = np.bincount(row, minlength=64).astype(np.float64)
        kspec = _wht_centered_topk(counts, TOP_KS)
        for k in TOP_KS:
            topk[k].append(float(kspec[k]))

    return {
        "entropy": float(_entropy_bits(all_prob)),
        "split_entropy_0_15": float(_entropy_bits(first_hist / float(np.sum(first_hist)))),
        "split_entropy_16_63": float(_entropy_bits(last_hist / float(np.sum(last_hist)))),
        "hop_mean": float(np.mean(hops)),
        "hop_mean_0_15": float(np.mean(hops_first)),
        "hop_mean_16_63": float(np.mean(hops_last)),
        "topk": {k: float(np.mean(v)) for k, v in topk.items()},
    }


def _omega_summary(blocks: list[tuple[int, ...]]) -> dict[str, float]:
    in_omega = []
    horizon = []
    ab = []
    horizon_hops = []
    ab_hops = []
    for block in blocks:
        traj = schedule_omega_trajectory(block)
        in_omega_values = traj.get("in_omega")
        horizon_values = traj.get("horizon_distance")
        ab_values = traj.get("ab_distance")
        if not isinstance(in_omega_values, list):
            in_omega_values = []
        if not isinstance(horizon_values, list):
            horizon_values = []
        if not isinstance(ab_values, list):
            ab_values = []

        in_omega.extend([float(v) for v in in_omega_values])
        h = horizon_values
        a = ab_values
        horizon.extend([float(v) for v in h])
        ab.extend([float(v) for v in a])
        if len(h) >= 2:
            horizon_hops.append(float(np.mean([abs(h[i + 1] - h[i]) for i in range(len(h) - 1)])))
            ab_hops.append(float(np.mean([abs(a[i + 1] - a[i]) for i in range(len(a) - 1)])))

    if not in_omega or not horizon or not ab:
        return {
            "in_omega_ratio": 0.0,
            "horizon_mean": 0.0,
            "ab_mean": 0.0,
            "horizon_hop_mean": 0.0,
            "ab_hop_mean": 0.0,
        }

    return {
        "in_omega_ratio": float(np.mean(np.array(in_omega, dtype=np.float64))),
        "horizon_mean": float(np.mean(np.array(horizon, dtype=np.float64))),
        "ab_mean": float(np.mean(np.array(ab, dtype=np.float64))),
        "horizon_hop_mean": float(np.mean(horizon_hops)) if horizon_hops else 0.0,
        "ab_hop_mean": float(np.mean(ab_hops)) if ab_hops else 0.0,
    }


def _sha_blocks_from_messages(sample_count: int, seed: int) -> list[tuple[int, ...]]:
    msgs = random_messages(sample_count=sample_count, seed=seed, min_len=1, max_len=64)
    return sha256_block_words_from_messages(msgs, block_index=0)


def _run_test2_once(
    block_count: int, 
    seed: int, 
    block_caps: dict[str, int] | None = None,
) -> dict[str, Any]:
    _progress(f"    _run_test2_once block_count={block_count} seed={seed}")
    if block_caps is None:
        block_caps = {key: cap for key, cap in MAX_TEST2_BLOCKS.items()}
    else:
        block_caps = {
            key: int(value) for key, value in block_caps.items() if key in MAX_TEST2_BLOCKS
        }
        for key, value in MAX_TEST2_BLOCKS.items():
            block_caps.setdefault(key, value)
    msgs = random_messages(sample_count=block_count, seed=seed, min_len=1, max_len=64)
    _progress("    sha blocks: build padded blocks")
    sha_blocks = sha256_block_words_from_messages(msgs, block_index=0)
    _progress("    random controls: uniform random words")
    random_blocks = random_block_words(block_count, seed=seed + 9_000)
    _progress("    random controls: alternate words")
    random_blocks_alt = random_block_words(block_count, seed=seed + 9_001)
    _progress("    null control messages")
    null_msgs = random_messages(sample_count=block_count, seed=seed + 13, min_len=1, max_len=64)
    _progress("    null control blocks")
    null_sha_blocks = sha256_block_words_from_messages(null_msgs, block_index=0)
    _progress("    blake2s chain blocks")
    blake_blocks = _blake2s_chain_block_words(msgs)

    primary = _run_test2_compare(
        sha_blocks=sha_blocks,
        random_blocks=random_blocks,
        seed=seed,
        block_count=block_count,
        block_caps=block_caps,
    )
    primary["null_controls"] = {
        "sha_random": _run_test2_compare(
            sha_blocks=sha_blocks,
            random_blocks=null_sha_blocks,
            seed=seed,
            block_count=block_count,
            block_caps=block_caps,
        ),
        "uniform_random": _run_test2_compare(
            sha_blocks=sha_blocks,
            random_blocks=random_blocks_alt,
            seed=seed,
            block_count=block_count,
            block_caps=block_caps,
        ),
        "blake2s": _run_test2_compare(
            sha_blocks=sha_blocks,
            random_blocks=blake_blocks,
            seed=seed,
            block_count=block_count,
            block_caps=block_caps,
        ),
    }
    return primary


def _normal_two_sided_p(z: float) -> float:
    if not np.isfinite(z):
        return 0.0 if z != 0.0 else 1.0
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _metric_significance(
    runs: list[dict[str, Any]],
    metric_keys: list[str],
    control_keys: list[str],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for metric in metric_keys:
        observed: list[float] = [float(_extract_nested(r, metric)) for r in runs]
        obs_mean = float(np.mean(observed)) if observed else 0.0
        obs_std = float(np.std(observed, ddof=0)) if observed else 0.0
        null_values: list[float] = []
        for run in runs:
            controls = run.get("null_controls", {})
            for control_key in control_keys:
                control_run = controls.get(control_key)
                if not control_run:
                    continue
                null_values.append(float(_extract_nested(control_run, metric)))
        if null_values:
            null_arr = np.array(null_values, dtype=np.float64)
            null_mean = float(np.mean(null_arr))
            null_std = float(np.std(null_arr, ddof=0))
            if null_std > 0.0:
                z = (obs_mean - null_mean) / null_std
                p = _normal_two_sided_p(z)
            else:
                z = float("inf") if obs_mean != null_mean else 0.0
                p = 0.0 if obs_mean != null_mean else 1.0
        else:
            z = 0.0
            p = 1.0
            null_mean = 0.0
            null_std = 0.0
        out[metric] = {
            "observed_mean": obs_mean,
            "observed_std": obs_std,
            "null_mean": null_mean,
            "null_std": null_std,
            "z": float(z),
            "p": float(p),
            "samples": len(observed),
        }
    if out:
        total = len(out)
        p_by_metric = {metric: stats["p"] for metric, stats in out.items()}
        ranked = sorted(p_by_metric.items(), key=lambda item: item[1])
        fdr: dict[str, float] = {}
        running_min = 1.0
        for idx in range(len(ranked) - 1, -1, -1):
            metric, pval = ranked[idx]
            rank = idx + 1
            candidate = pval * total / rank
            running_min = min(running_min, float(candidate))
            fdr[metric] = min(1.0, running_min)
        for metric, stats in out.items():
            pval = float(stats["p"])
            stats["p_bonferroni"] = min(1.0, pval * total)
            stats["p_bh"] = min(1.0, fdr[metric])
    return out


def _interpetation(report: dict[str, Any]) -> dict[str, str]:
    summary = report.get("summary", {})
    projection = summary.get("projection_lift", {})
    test1_sig = summary.get("test1_significance", {})
    test2_sig = summary.get("test2_significance", {})
    omega_metric = "omega_residual.in_omega_ratio"
    traj_metric = "trajectory_residual.topk.1"
    byte_metric = "byte_native.residual.q_entropy"

    omega_p = 1.0
    traj_p = 1.0
    byte_p = 1.0
    if omega_metric in test2_sig:
        omega_p = float(test2_sig[omega_metric].get("p_bh", 1.0))
    if traj_metric in test2_sig:
        traj_p = float(test2_sig[traj_metric].get("p_bh", 1.0))
    if byte_metric in test1_sig:
        byte_p = float(test1_sig[byte_metric].get("p_bh", 1.0))

    quotient = (
        "quotient collapse likely present"
        if float(projection.get("projection_collision_fraction", 0.0)) > 0.05
        else "no clear quotient collapse yet"
    )
    lift = (
        "lift partially recovering collisions"
        if float(projection.get("lift_resolution_fraction", 0.0)) > 0.05
        else "lift recovery not yet visible"
    )
    omega = "omega sticking signal seen" if omega_p < 0.05 else "omega sticking not significant"
    trajectory = "round trajectory anomalies stand out" if traj_p < 0.05 else "no strong trajectory anomalies"
    byte_features = "aQPU byte-features are distinct" if byte_p < 0.05 else "aQPU byte-features are not distinct yet"
    return {
        "quotient_collapse": quotient,
        "lift_recovery": lift,
        "omega_sticking": omega,
        "round_trajectory": trajectory,
        "byte_native_features": byte_features,
    }

def _run_test2_compare(
    sha_blocks: list[tuple[int, ...]],
    random_blocks: list[tuple[int, ...]],
    seed: int,
    block_count: int,
    block_caps: dict[str, int] | None = None,
) -> dict[str, Any]:
    if block_caps is None:
        block_caps = {key: cap for key, cap in MAX_TEST2_BLOCKS.items()}
    else:
        block_caps = {
            key: int(value) for key, value in block_caps.items() if key in MAX_TEST2_BLOCKS
        }
        for key, value in MAX_TEST2_BLOCKS.items():
            block_caps.setdefault(key, value)
    _progress(
        f"    _run_test2_compare seed={seed}"
        f" blocks={len(sha_blocks)}"
    )
    sha_profile_blocks = _cap_block_rows(sha_blocks, block_caps["profile"])
    random_profile_blocks = _cap_block_rows(random_blocks, block_caps["profile"])
    _progress(
        f"      profile: cap={len(sha_profile_blocks)}/{len(sha_blocks)}"
        f" random_cap={len(random_profile_blocks)}/{len(random_blocks)}"
    )
    sha_profile = _round_profile_mean(sha_profile_blocks)
    random_profile = _round_profile_mean(random_profile_blocks)
    residual = {
        key: float(abs(sha_profile[key] - random_profile[key])) for key in sha_profile
    }

    sha_depth_blocks = _cap_block_rows(sha_blocks, block_caps["depth"])
    rnd_depth_blocks = _cap_block_rows(random_blocks, block_caps["depth"])
    _progress(
        f"      depth: cap={len(sha_depth_blocks)}/{len(sha_blocks)}"
        f" random_cap={len(rnd_depth_blocks)}/{len(random_blocks)}"
    )
    sha_depth = _depth4_signature_summary(sha_depth_blocks)
    rnd_depth = _depth4_signature_summary(rnd_depth_blocks)

    sha_projection_blocks = _cap_block_rows(sha_blocks, block_caps["projection"])
    rnd_projection_blocks = _cap_block_rows(random_blocks, block_caps["projection"])
    _progress(
        f"      projection: cap={len(sha_projection_blocks)}/{len(sha_blocks)}"
        f" random_cap={len(rnd_projection_blocks)}/{len(random_blocks)}"
    )
    sha_projection_lift = _projection_lift_summary(sha_projection_blocks)
    rnd_projection_lift = _projection_lift_summary(rnd_projection_blocks)

    sha_round_blocks = _cap_block_rows(sha_blocks, block_caps["trajectory"])
    rnd_round_blocks = _cap_block_rows(random_blocks, block_caps["trajectory"])
    _progress(
        f"      trajectory: cap={len(sha_round_blocks)}/{len(sha_blocks)}"
        f" random_cap={len(rnd_round_blocks)}/{len(random_blocks)}"
    )
    sha_round = _round_trajectory_summary(sha_round_blocks)
    rnd_round = _round_trajectory_summary(rnd_round_blocks)

    sha_omega_blocks = _cap_block_rows(sha_blocks, block_caps["omega"])
    rnd_omega_blocks = _cap_block_rows(random_blocks, block_caps["omega"])
    _progress(
        f"      omega: cap={len(sha_omega_blocks)}/{len(sha_blocks)}"
        f" random_cap={len(rnd_omega_blocks)}/{len(random_blocks)}"
    )
    sha_omega = _omega_summary(sha_omega_blocks)
    rnd_omega = _omega_summary(rnd_omega_blocks)
    depth_residual = {
        "entropy": float(abs(sha_depth["entropy_mean"] - rnd_depth["entropy_mean"])),
        "hamming": float(abs(sha_depth["hamming_mean"] - rnd_depth["hamming_mean"])),
    }
    projection_lift_residual = {
        key: float(abs(sha_projection_lift[key] - rnd_projection_lift[key]))
        for key in sha_projection_lift
    }
    trajectory_residual = {
        "entropy": float(abs(sha_round["entropy"] - rnd_round["entropy"])),
        "split_0_15": float(abs(sha_round["split_entropy_0_15"] - rnd_round["split_entropy_0_15"])),
        "split_16_63": float(abs(sha_round["split_entropy_16_63"] - rnd_round["split_entropy_16_63"])),
        "hop_mean": float(abs(sha_round["hop_mean"] - rnd_round["hop_mean"])),
        "hop_0_15": float(abs(sha_round["hop_mean_0_15"] - rnd_round["hop_mean_0_15"])),
        "hop_16_63": float(abs(sha_round["hop_mean_16_63"] - rnd_round["hop_mean_16_63"])),
        "topk": {},
    }
    for k in TOP_KS:
        trajectory_residual["topk"][k] = float(abs(sha_round["topk"][k] - rnd_round["topk"][k]))
    omega_residual = {
        "in_omega_ratio": float(abs(sha_omega["in_omega_ratio"] - rnd_omega["in_omega_ratio"])),
        "horizon_mean": float(abs(sha_omega["horizon_mean"] - rnd_omega["horizon_mean"])),
        "ab_mean": float(abs(sha_omega["ab_mean"] - rnd_omega["ab_mean"])),
        "horizon_hop_mean": float(abs(sha_omega["horizon_hop_mean"] - rnd_omega["horizon_hop_mean"])),
        "ab_hop_mean": float(abs(sha_omega["ab_hop_mean"] - rnd_omega["ab_hop_mean"])),
    }

    return {
        "seed": seed,
        "block_count": block_count,
        "sha_profile": sha_profile,
        "random_profile": random_profile,
        "profile_residual": residual,
        "sha_depth": sha_depth,
        "random_depth": rnd_depth,
        "depth_residual": depth_residual,
        "sha_projection_lift": sha_projection_lift,
        "random_projection_lift": rnd_projection_lift,
        "projection_lift_residual": projection_lift_residual,
        "sha_round_trajectory": sha_round,
        "random_round_trajectory": rnd_round,
        "trajectory_residual": trajectory_residual,
        "sha_omega": sha_omega,
        "random_omega": rnd_omega,
        "omega_residual": omega_residual,
    }


def _extract_nested(payload: dict[str, Any], path: str) -> Any:
    out: Any = payload
    for key in path.split("."):
        if isinstance(out, dict) and key.isdigit():
            out = out[int(key)]
        else:
            out = out[key]
    return out


def _to_report_entry(results: list[dict[str, Any]], key_selector: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in key_selector:
        values = [_extract_nested(r, key) for r in results]
        mean, std = _mean_std([float(v) for v in values])
        out[key] = {"mean": mean, "std": std, "samples": len(values)}
    return out


def _top_mean_metrics(summary: dict[str, dict[str, float]], top_n: int = 3) -> list[dict[str, float]]:
    rows = []
    for metric, stats in summary.items():
        rows.append({"metric": metric, "mean": float(stats["mean"])})
    rows.sort(key=lambda item: item["mean"], reverse=True)
    return rows[:top_n]


def _print_structural_summary(report: dict[str, Any]) -> None:
    print("\nSTRUCTURAL SUMMARY")
    print("test1 strongest:")
    for row in report["summary"]["test1_top_residuals"]:
        print(f"  {row['metric']}: {row['mean']:.6f}")
    print("test2 strongest:")
    for row in report["summary"]["test2_top_residuals"]:
        print(f"  {row['metric']}: {row['mean']:.6f}")
    lift = report["summary"]["projection_lift"]
    print(
        "projection/lift:"
        f" mean_lifts={lift['mean_lifts_per_projection']:.6f}"
        f" max_lifts={lift['max_lifts_per_projection']:.6f}"
        f" lift_resolution={lift['lift_resolution_fraction']:.6f}"
    )
    interpretation = report["summary"]["interpretation"]
    print("interpretation:")
    for key, value in interpretation.items():
        print(f"  {key}: {value}")


def _topk_variance_ratio(test_runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for k in TOP_KS:
        sha_vals = [float(r["topk_sha"][k]) for r in test_runs]
        rnd_vals = [float(r["topk_random"][k]) for r in test_runs]
        sha_var = float(np.var(sha_vals)) if sha_vals else 0.0
        rnd_var = float(np.var(rnd_vals)) if rnd_vals else 0.0
        out[str(k)] = {
            "sha_var": sha_var,
            "random_var": rnd_var,
            "ratio": sha_var / rnd_var if rnd_var > 0 else float("inf"),
            "samples": len(sha_vals),
        }
    return out


TEST1_METRICS = [
    "max_abs_vs_random",
    "multi.entropy_mean",
    "multi.entropy_std",
    "multi.max_abs_vs_random",
    "multi.transition_entropy",
    "multi.topk_residual.1",
    "multi.topk_residual.2",
    "multi.topk_residual.4",
    "multi.topk_residual.8",
    "topk_residual.1",
    "topk_residual.2",
    "topk_residual.4",
    "byte_native.residual.family_entropy",
    "byte_native.residual.family_bit0",
    "byte_native.residual.family_bit7",
    "byte_native.residual.payload_entropy",
    "byte_native.residual.payload_bit_entropy",
    "byte_native.residual.q_entropy",
    "byte_native.residual.commutativity_collision_rate",
    "byte_native.residual.q_topk_residual.1",
    "byte_native.residual.q_topk_residual.2",
    "byte_native.residual.q_topk_residual.4",
    "byte_native.residual.payload_topk_residual.1",
    "byte_native.residual.payload_topk_residual.2",
    "byte_native.residual.payload_topk_residual.4",
]


TEST2_METRICS = [
    "profile_residual.low_carry_fraction",
    "profile_residual.high_carry_fraction",
    "profile_residual.temp1_k4_d00",
    "profile_residual.temp1_k4_d01",
    "profile_residual.temp1_k4_d10",
    "profile_residual.temp1_k4_d11",
    "profile_residual.temp2_k4_d00",
    "profile_residual.temp2_k4_d01",
    "profile_residual.temp2_k4_d10",
    "profile_residual.temp2_k4_d11",
    "profile_residual.temp1_pair_diag_mean",
    "profile_residual.temp2_pair_diag_mean",
    "projection_lift_residual.projection_collision_fraction",
    "projection_lift_residual.pair_collision_fraction",
    "projection_lift_residual.mean_projection_multiplicity",
    "projection_lift_residual.mean_lifts_per_projection",
    "projection_lift_residual.max_lifts_per_projection",
    "projection_lift_residual.lift_resolution_fraction",
    "projection_lift_residual.omega_mask_entropy",
    "projection_lift_residual.omega_mask_support",
    "trajectory_residual.entropy",
    "trajectory_residual.split_0_15",
    "trajectory_residual.split_16_63",
    "trajectory_residual.hop_mean",
    "trajectory_residual.hop_0_15",
    "trajectory_residual.hop_16_63",
    "trajectory_residual.topk.1",
    "trajectory_residual.topk.2",
    "trajectory_residual.topk.4",
    "trajectory_residual.topk.8",
    "trajectory_residual.topk.16",
    "trajectory_residual.topk.32",
    "omega_residual.in_omega_ratio",
    "omega_residual.horizon_mean",
    "omega_residual.ab_mean",
    "omega_residual.horizon_hop_mean",
    "omega_residual.ab_hop_mean",
    "depth_residual.entropy",
    "depth_residual.hamming",
]


def run_default() -> int:
    """Run with the default preset. No CLI flags required."""
    mode_cfg = _run_mode()
    mode = str(mode_cfg.get("mode", "normal"))
    metric_caps = _resolve_metric_caps(mode, _env_int("GYROCRYPT_ROUND_BLOCKS", DEFAULT_RUN["round_blocks"]))
    _progress(
        "run defaults"
        f" | mode={_env_mode_str('GYROCRYPT_MODE', 'normal') or 'normal'}"
        f" | samples={_env_int('GYROCRYPT_SAMPLES', DEFAULT_RUN['samples'])}"
        f" | round_blocks={mode_cfg['round_blocks']}"
        f" | reps={mode_cfg['reps']}"
        f" | seed={_env_int('GYROCRYPT_SEED', DEFAULT_RUN['seed'])}"
    )
    return run_stress(
        samples=_env_int("GYROCRYPT_SAMPLES", DEFAULT_RUN["samples"]),
        round_blocks=int(mode_cfg["round_blocks"]),
        reps=int(mode_cfg["reps"]),
        seed=_env_int("GYROCRYPT_SEED", DEFAULT_RUN["seed"]),
        metric_caps=metric_caps,
        out_path=DEFAULT_RUN["out"],
    )


def run_stress(
    samples: int,
    round_blocks: int,
    reps: int,
    seed: int,
    out_path: Path,
    metric_caps: dict[str, int] | None = None,
) -> int:
    if metric_caps is None:
        metric_caps = {key: cap for key, cap in MAX_TEST2_BLOCKS.items()}
    else:
        metric_caps = {
            key: int(value) for key, value in metric_caps.items() if key in MAX_TEST2_BLOCKS
        }
        for key, value in MAX_TEST2_BLOCKS.items():
            metric_caps.setdefault(key, value)
    test1_runs: list[dict[str, Any]] = []
    test2_runs: list[dict[str, Any]] = []
    jsonl_path = out_path.with_suffix(".jsonl")
    jsonl_gz_path = out_path.with_suffix(".jsonl.gz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_gz_path.parent.mkdir(parents=True, exist_ok=True)
    _progress(
        "running stress test"
        f" | reps={reps}"
        f" samples={samples}"
        f" round_blocks={round_blocks}"
        f" seed={seed}"
        f" out={out_path}"
    )

    with (
        jsonl_path.open("w", encoding="utf-8") as jsonl_file,
        gzip.open(jsonl_gz_path, "wt", encoding="utf-8") as jsonl_gz_file,
    ):
        for rep in range(reps):
            rep_seed = seed + rep * 101
            _progress(f"rep={rep+1}/{reps} start (seed={rep_seed})")
            _progress("  test1: generating sha hashes and digest controls")
            test1_start = time.perf_counter()
            t1 = _run_test1_once(sample_count=samples, seed=rep_seed)
            test1_ms = (time.perf_counter() - test1_start) * 1000.0
            _progress(f"  test1 done in {_fmt_ms(test1_ms)}")
            _progress("  test2: generating sha message blocks and round probes")
            test2_start = time.perf_counter()
            t2 = _run_test2_once(
                block_count=round_blocks,
                seed=rep_seed + 1000,
                block_caps=metric_caps,
            )
            test2_ms = (time.perf_counter() - test2_start) * 1000.0
            _progress(f"  test2 done in {_fmt_ms(test2_ms)}")
            test1_runs.append(t1)
            test2_runs.append(t2)
            row = {
                "rep": rep + 1,
                "seed": rep_seed,
                "test1": t1,
                "test2": t2,
            }
            line = json.dumps(row, separators=(",", ":"), ensure_ascii=False)
            jsonl_file.write(line + "\n")
            jsonl_gz_file.write(line + "\n")
            print(
                f"rep={rep+1}/{reps}"
                f" test1_top1={t1['topk_residual'][1]:.5f}"
                f" test1_multi_top4={t1['multi']['topk_residual'][4]:.5f}"
                f" test2_lift={t2['projection_lift_residual']['lift_resolution_fraction']:.5f}",
                flush=True,
            )

    test1_summary = _to_report_entry(
        test1_runs,
        TEST1_METRICS,
    )
    test2_summary = _to_report_entry(
        test2_runs,
        TEST2_METRICS,
    )
    test1_significance = _metric_significance(
        test1_runs,
        TEST1_METRICS,
        ["sha_random", "uniform_random", "blake2s"],
    )
    test2_significance = _metric_significance(
        test2_runs,
        TEST2_METRICS,
        ["sha_random", "uniform_random", "blake2s"],
    )

    report: dict[str, Any] = {
        "meta": {
            "samples": samples,
            "round_blocks": round_blocks,
            "repetitions": reps,
            "seed": seed,
            "mode": _env_mode_str("GYROCRYPT_MODE", "normal") or "normal",
        },
        "test1_runs": test1_runs,
        "test2_runs": test2_runs,
        "test1": test1_summary,
        "test2": test2_summary,
        "test1_significance": test1_significance,
        "test2_significance": test2_significance,
        "summary": {
            "top_significant_test1": [
                {
                    "metric": metric,
                    "p_bh": stats["p_bh"],
                    "z": stats["z"],
                }
                for metric, stats in sorted(
                    test1_significance.items(),
                    key=lambda item: item[1].get("p_bh", 1.0),
                )[:5]
            ],
            "top_significant_test2": [
                {
                    "metric": metric,
                    "p_bh": stats["p_bh"],
                    "z": stats["z"],
                }
                for metric, stats in sorted(
                    test2_significance.items(),
                    key=lambda item: item[1].get("p_bh", 1.0),
                )[:5]
            ],
            "test1_top_residuals": _top_mean_metrics(test1_summary),
            "test2_top_residuals": _top_mean_metrics(test2_summary),
            "test1_topk_variance_ratio": _topk_variance_ratio(test1_runs),
            "projection_lift": {
                "mean_lifts_per_projection": float(
                    np.mean([run["sha_projection_lift"]["mean_lifts_per_projection"] for run in test2_runs])
                ),
                "max_lifts_per_projection": float(
                    np.mean([run["sha_projection_lift"]["max_lifts_per_projection"] for run in test2_runs])
                ),
                "lift_resolution_fraction": float(
                    np.mean([run["sha_projection_lift"]["lift_resolution_fraction"] for run in test2_runs])
                ),
            },
            "artifacts": {
                "json": str(out_path),
                "jsonl": str(jsonl_path),
                "jsonl_gz": str(jsonl_gz_path),
            },
        },
    }
    report["summary"]["interpretation"] = _interpetation(report)

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_structural_summary(report)
    print(f"wrote: {out_path}")
    print(f"wrote: {jsonl_path}")
    print(f"wrote: {jsonl_gz_path}")
    return 0


def main() -> int:
    return run_default()


if __name__ == "__main__":
    raise SystemExit(main())
