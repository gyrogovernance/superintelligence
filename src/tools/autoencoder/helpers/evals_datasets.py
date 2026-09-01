"""Eval dataset builders: ensembles, percolation, words, byte-mechanism.

One flat module for everything that *constructs an eval corpus or dataset*.
Section headers keep each builder domain readable:

- Ensembles: lambda-ensemble corpora, the never-broken symmetry group, the
  controlled symmetry-breaking / anomaly corpora, and hQVM(d) generalization.
- Percolation: the restricted-generator dataset and QuBEC ensemble labels.
- Words: the canonical K4/BU word dataset and word / signature builders.
- Byte mechanism: fold targets, the [L]/[R] factorization audit, and depth-4
  frames.

All labels are exact kernel facts; models/readouts consume these builders but
nothing here re-implements the kernel.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from typing import Any, Iterable

import numpy as np

from src import api, constants
from src.family import (
    bfs_reach,
    build_hqvm_d,
    byte_from_family_micro,
    enumerate_omega_d,
    fiber_complete,
    fold_disagreement_d,
    gf2_rank,
    predicted_cluster_size,
    q_word_d,
    step_uv,
)

from ..datasets import transition_table
from ..kernel import (
    enumerate_signature_ids,
    popcount6,
    sig_id_parts,
    signature_id,
    word_signature_id,
)


# ---------------------------------------------------------------------------
# Ensembles: lambda-ensemble corpora
# ---------------------------------------------------------------------------


def lambda_chirality_distribution(lam: float) -> np.ndarray:
    """Exact P(chi) proportional to lambda^wt(chi) over the 64 words."""
    weights = np.array([chi.bit_count() for chi in range(64)], dtype=np.float64)
    p = np.power(lam, weights)
    return p / p.sum()


def chi_fiber(chi: int, d: int = 6) -> list[tuple[int, int]]:
    """All states (u, v) with u ^ v == chi; exactly 2^d of them."""
    return [(u, u ^ chi) for u in range(1 << d)]


def sample_lambda_corpus(
    lam: float, n: int, seed: int = 0, d: int = 6
) -> np.ndarray:
    """Sample n state indices from the lambda ensemble."""
    rng = np.random.default_rng(seed)
    p = lambda_chirality_distribution(lam)
    chis = rng.choice(64, size=n, p=p)
    us = rng.integers(0, 1 << d, size=n)
    return (us.astype(np.int64) << d) | (us ^ chis).astype(np.int64)


def corpus_shell_histogram(state_indices: np.ndarray, d: int = 6) -> np.ndarray:
    """Shell histogram (popcount of chirality) of a corpus."""
    chi = (state_indices >> d) ^ (state_indices & ((1 << d) - 1))
    shells = np.zeros_like(chi)
    for bit in range(d):
        shells += (chi >> bit) & 1
    hist = np.zeros(d + 1, dtype=np.int64)
    for s in range(d + 1):
        hist[s] = int((shells == s).sum())
    return hist


def signature_shell_action(sig_id: int, d: int = 6) -> np.ndarray:
    """Exact [2^d x (d+1)] matrix: how g maps the shell histogram."""
    _parity, tau_u, tau_v = sig_id_parts(sig_id)
    delta = tau_u ^ tau_v
    counts = np.zeros((1 << d, d + 1), dtype=np.int64)
    for chi in range(1 << d):
        chi_next = (chi ^ delta) & ((1 << d) - 1)
        counts[chi, chi_next.bit_count()] += 1
    return counts


def is_stabilizer_of_lambda_ensemble(
    sig_id: int, lam_grid: np.ndarray | None = None, d: int = 6
) -> bool:
    """g stabilizes P(chi) ~ lambda^wt(chi) for every lambda in the grid iff
    it preserves the Hamming weight of every chirality word."""
    action = signature_shell_action(sig_id, d)
    for chi in range(1 << d):
        row = action[chi]
        if row[chi.bit_count()] != 1 or row.sum() != 1:
            return False
    return True


def never_broken_group(d: int = 6, lam_grid: np.ndarray | None = None) -> dict:
    """Brute-force the lambda-ensemble stabilizer over all 2^(2d+1) signatures.

    Returns the subgroup H found, its order, and the exhaustive check that
    H = {(0, t, t), (1, t, t)}.
    """
    if lam_grid is None:
        lam_grid = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    stabilizer: list[int] = []
    broken: list[int] = []
    for sig_id in enumerate_signature_ids():
        if is_stabilizer_of_lambda_ensemble(sig_id, lam_grid, d):
            stabilizer.append(sig_id)
        else:
            broken.append(sig_id)
    expected = set()
    for t in range(1 << d):
        expected.add((0 << 12) | (t << 6) | t)
        expected.add((1 << 12) | (t << 6) | t)
    return {
        "order": len(stabilizer),
        "stabilizer_ids": stabilizer,
        "n_broken": len(broken),
        "matches_diagonal_form": set(stabilizer) == expected,
        "is_group": _closed_under_composition(stabilizer, d),
        "n_signatures": len(stabilizer) + len(broken),
    }


def _closed_under_composition(ids: list[int], d: int) -> bool:
    """Check group closure of a signature set via kernel composition."""
    id_set = set(ids)
    for a in ids:
        pa, ta, va = sig_id_parts(a)
        sa = api.OmegaSignature12(pa, ta, va)
        for b in ids:
            pb, tb, vb = sig_id_parts(b)
            sb = api.OmegaSignature12(pb, tb, vb)
            comp = api.compose_omega_signatures(sb, sa)
            cid = (comp.parity << 12) | (comp.tau_u6 << 6) | comp.tau_v6
            if cid not in id_set:
                return False
    return True


def w2_word_signature_ids() -> list[int]:
    """Signature ids of the canonical W2(m) words for all m; the broken
    generators of the experiment."""
    ids = []
    for m in range(64):
        word = [byte_from_family_micro(0, m, 6), byte_from_family_micro(1, m, 6)]
        sig = api.omega_word_signature(word)
        ids.append((sig.parity << 12) | (sig.tau_u6 << 6) | sig.tau_v6)
    return ids


def w2_not_in_never_broken(d: int = 6) -> bool:
    """The pre-registered brokenness: no W2(m) stabilizes the lambda family."""
    lam_grid = np.array([0.25, 0.5, 2.0, 4.0])
    for sig_id in w2_word_signature_ids():
        if is_stabilizer_of_lambda_ensemble(sig_id, lam_grid, d):
            return False
    return True


def ensemble_transport_summary(allowed: list[int], d: int = 6) -> dict[str, int]:
    """Rank/fiber summary of a byte restriction (used by the sweep arms)."""
    qset = sorted({int(q_word_d(int(b), d)) for b in allowed})
    return {
        "n_bytes": len(set(allowed)),
        "transport_rank": gf2_rank(qset, d),
        "n_q_classes": len(qset),
    }


def lambda_grid(default: str = "log") -> np.ndarray:
    """Standard lambda grid spanning equality-horizon to complement-horizon."""
    if default == "log":
        return np.geomspace(0.125, 8.0, 7)
    return np.linspace(0.125, 8.0, 7)


# ---------------------------------------------------------------------------
# Ensembles: controlled / anomaly ensembles
# ---------------------------------------------------------------------------


def _byte_family(byte: int) -> int:
    return constants.byte_family(byte)


ANOMALY_TYPES = (
    "biased_family",
    "biased_q_weight",
    "missing_q_classes",
    "rank_restricted",
    "even_parity_only",
    "corrupted_mask",
    "invalid_syndrome",
    "byte_substitution",
    "adjacent_swap",
    "deletion",
    "shadow_substitution",
)


def biased_family_bytes(rng: np.random.Generator, family: int, n: int) -> np.ndarray:
    """Sample bytes from a single K4 family (gauge-biased ensemble)."""
    pool = [b for b in range(256) if _byte_family(b) == family]
    return rng.choice(pool, size=n)


def biased_q_weight_bytes(rng: np.random.Generator, weight: int, n: int) -> np.ndarray:
    pool = [b for b in range(256) if api.Q_WEIGHT_BY_BYTE[b] == weight]
    return rng.choice(pool, size=n)


def missing_q_class_alphabet(
    rng: np.random.Generator | None = None, n_missing: int = 0, seed: int = 0
):
    """Remove whole q-classes from the alphabet; exact rank labels included."""
    inner = np.random.default_rng(seed)
    removed = inner.choice(64, size=n_missing, replace=False)
    allowed = np.array([b for b in range(256) if api.q_word6(b) not in set(removed.tolist())])
    q_vectors = [api.q_word6(b) for b in allowed.tolist()]
    return allowed, q_vectors, removed


def corrupted_mask_dataset(rng: np.random.Generator, n: int) -> list[dict]:
    """Valid masks vs corrupted masks with exact syndrome labels."""
    rows = []
    for _ in range(n):
        byte = int(rng.integers(0, 256))
        clean = api.MASK12_BY_BYTE[byte]
        corrupt = clean ^ (1 << int(rng.integers(0, 12)))
        rows.append(
            {
                "mask": clean,
                "is_valid": 1,
                "syndrome": int(api.mask12_syndrome(clean)),
                "expected": "syndrome_zero",
            }
        )
        rows.append(
            {
                "mask": corrupt,
                "is_valid": 0,
                "syndrome": int(api.mask12_syndrome(corrupt)),
                "expected": "syndrome_nonzero",
            }
        )
    return rows


def byte_perturbation_dataset(rng: np.random.Generator, n: int) -> list[dict]:
    """Byte substitutions, adjacent swaps, deletions on short words.

    Exact labels record how each perturbation changes the Omega signature.
    """
    rows = []
    for _ in range(n):
        length = int(rng.integers(2, 5))
        word = [int(rng.integers(0, 256)) for _ in range(length)]
        base_sig = word_signature_id(word)

        pos = int(rng.integers(0, length))
        shadowed = list(word)
        shadowed[pos] = api.shadow_partner_byte(word[pos])
        shadow_sig = word_signature_id(shadowed)
        rows.append(
            {
                "kind": "shadow_substitution",
                "word": bytes(shadowed),
                "signature_preserved": int(shadow_sig == base_sig),
                "base_sig": base_sig,
                "new_sig": shadow_sig,
            }
        )

        swapped = list(word)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        swap_sig = word_signature_id(swapped)
        rows.append(
            {
                "kind": "adjacent_swap",
                "word": bytes(swapped),
                "signature_preserved": int(swap_sig == base_sig),
                "base_sig": base_sig,
                "new_sig": swap_sig,
            }
        )

        deleted = word[1:]
        del_sig = word_signature_id(deleted)
        rows.append(
            {
                "kind": "deletion",
                "word": bytes(deleted),
                "signature_preserved": int(del_sig == base_sig),
                "base_sig": base_sig,
                "new_sig": del_sig,
            }
        )

        sub = list(word)
        sub[int(rng.integers(0, length))] = int(rng.integers(0, 256))
        sub_sig = word_signature_id(sub)
        rows.append(
            {
                "kind": "byte_substitution",
                "word": bytes(sub),
                "signature_preserved": int(sub_sig == base_sig),
                "base_sig": base_sig,
                "new_sig": sub_sig,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Ensembles: hQVM(d) generalization datasets
# ---------------------------------------------------------------------------


def hqvm_d_transition_dataset(d: int) -> dict[str, np.ndarray]:
    """Exhaustive transitions for hQVM(d) with 2^(2d) states (procedural)."""
    omega = enumerate_omega_d(d)
    index_of = {uv: i for i, uv in enumerate(omega)}
    alphabet = _alphabet_d(d)
    n = len(omega)
    next_state = np.empty((n, len(alphabet)), dtype=np.uint32)
    q_by_byte = np.empty(len(alphabet), dtype=np.uint8)
    for j, byte in enumerate(alphabet):
        q_by_byte[j] = q_word_d(byte, d)
        for i, (u, v) in enumerate(omega):
            u2, v2 = step_uv(u, v, byte, d)
            next_state[i, j] = index_of[(u2, v2)]
    return {
        "next_state": next_state,
        "q_by_byte": q_by_byte,
        "u": np.array([u for u, _ in omega], dtype=np.uint32),
        "v": np.array([v for _, v in omega], dtype=np.uint32),
    }


def _alphabet_d(d: int) -> list[int]:
    return list(range(1 << (d + 2)))


def dimension_transfer_probe(d_train: int, d_test: int) -> dict[str, object]:
    """Verify the structural facts that should transfer across d."""
    alph_train = _alphabet_d(d_train)
    qs_train = [q_word_d(b, d_train) for b in alph_train]
    alph_test = _alphabet_d(d_test)
    qs_test = [q_word_d(b, d_test) for b in alph_test]
    from collections import Counter

    counts_train = Counter(qs_train)
    counts_test = Counter(qs_test)
    return {
        "d_train": d_train,
        "d_test": d_test,
        "q_class_size_train": sorted(set(counts_train.values())),
        "q_class_size_test": sorted(set(counts_test.values())),
        "rank_train": gf2_rank(qs_train, d_train),
        "rank_test": gf2_rank(qs_test, d_test),
        "expected_rank": d_train,
    }


# ---------------------------------------------------------------------------
# Percolation: restricted-generator dataset and ensemble labels
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _eng() -> Any:
    """Lazily build the hQVM(6) graph; imported modules pay the cost only on
    first use of a restriction label, not at import time."""
    return build_hqvm_d(6)


def restriction_labels(allowed: list[int]) -> dict[str, Any]:
    """Exact labels for one allowed-byte restriction."""
    eng = _eng()
    qs = sorted({int(q_word_d(int(b), 6)) for b in allowed})
    reach, spans, giant, full = bfs_reach(eng, allowed)
    return {
        "n_allowed": len(set(allowed)),
        "transport_rank": gf2_rank(qs, 6),
        "n_q_classes": len(qs),
        "reach_size": reach,
        "reach_fraction": reach / eng.n_omega,
        "horizon_spanning": int(spans),
        "giant": int(giant),
        "full_reachability": int(full),
        "fiber_complete": int(fiber_complete(allowed, eng)),
        "predicted_cluster": predicted_cluster_size(gf2_rank(qs, 6)),
    }


def _random_rank_k_alphabet(rng: np.random.Generator, k: int) -> list[int]:
    """Byte alphabet whose q-classes span exactly rank k in GF(2)^6."""
    basis: list[int] = []
    while len(basis) < k:
        v = int(rng.integers(0, 64))
        if v and gf2_rank(basis + [v], 6) > gf2_rank(basis, 6):
            basis.append(v)
    bytes_out = []
    for q in basis:
        candidates = [b for b in range(256) if int(q_word_d(b, 6)) == q]
        bytes_out.append(int(rng.choice(candidates)))
    return bytes_out


def percolation_dataset(
    n_singletons: int = 64,
    n_rank_samples: int = 6,
    n_random: int = 128,
    seed: int = 7,
) -> dict[str, np.ndarray]:
    """Dataset F rows: singleton alphabets, rank-controlled subsets, random ones."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    if n_singletons < 256:
        singleton_bytes = rng.choice(256, size=n_singletons, replace=False).tolist()
    else:
        singleton_bytes = list(range(256))
    for b in singleton_bytes:
        rows.append({"allowed": [int(b)], **restriction_labels([int(b)])})

    for k in range(1, 7):
        for _ in range(n_rank_samples):
            allowed = _random_rank_k_alphabet(rng, k)
            rows.append({"allowed": allowed, **restriction_labels(allowed)})

    for _ in range(n_random):
        n = int(rng.integers(1, 9))
        allowed = list(rng.choice(256, size=n, replace=False))
        rows.append({"allowed": [int(b) for b in allowed], **restriction_labels(allowed)})

    return {
        "allowed_mask": np.array(
            [
                np.packbits(
                    np.isin(np.arange(256), np.array(r["allowed"], dtype=np.int64)).astype(np.uint8)
                )
                for r in rows
            ],
            dtype=np.uint8,
        ),
        "n_allowed": np.array([r["n_allowed"] for r in rows], dtype=np.uint16),
        "transport_rank": np.array([r["transport_rank"] for r in rows], dtype=np.uint8),
        "n_q_classes": np.array([r["n_q_classes"] for r in rows], dtype=np.uint8),
        "reach_size": np.array([r["reach_size"] for r in rows], dtype=np.uint16),
        "reach_fraction": np.array([r["reach_fraction"] for r in rows], dtype=np.float32),
        "horizon_spanning": np.array([r["horizon_spanning"] for r in rows], dtype=np.uint8),
        "giant": np.array([r["giant"] for r in rows], dtype=np.uint8),
        "full_reachability": np.array([r["full_reachability"] for r in rows], dtype=np.uint8),
        "fiber_complete": np.array([r["fiber_complete"] for r in rows], dtype=np.uint8),
        "predicted_cluster": np.array([r["predicted_cluster"] for r in rows], dtype=np.uint16),
    }


def shell_ensemble_labels(lambdas: list[float]) -> dict[str, np.ndarray]:
    """Dataset E shell/occupation ensemble exact labels (corpus conventions).

    rho = lambda/(1+lambda); eta = (1-lambda)/(1+lambda); M2 = 4096/(1+eta^2)^6.
    """
    weights = np.array([(chi).bit_count() for chi in range(64)], dtype=np.float64)
    out: dict[str, list[float]] = {
        "lambda": [],
        "rho": [],
        "eta": [],
        "M2": [],
        "expected_shell": [],
        "shell_variance": [],
        "wt_var_norm": [],
        "M2_chi": [],
    }
    for lam in lambdas:
        p_reg = np.power(lam, weights)
        p_reg /= p_reg.sum()
        mean_w = float((p_reg * weights).sum())
        var_w = float((p_reg * (weights - mean_w) ** 2).sum())
        rho = mean_w / 6.0
        eta = (1.0 - lam) / (1.0 + lam)
        m2_chi = 1.0 / float((p_reg * p_reg).sum())
        out["lambda"].append(lam)
        out["rho"].append(rho)
        out["eta"].append(eta)
        out["M2"].append(64.0 * m2_chi)
        out["expected_shell"].append(mean_w)
        out["shell_variance"].append(var_w)
        out["wt_var_norm"].append(var_w / 6.0)
        out["M2_chi"].append(m2_chi)
    return {k: np.array(v, dtype=np.float64) for k, v in out.items()}


def walsh_multipliers(axis_flip_probs: list[float]) -> dict[str, np.ndarray]:
    """Dataset E q-sector ensemble labels: exact Walsh multipliers.

    Returns the per-axis flip probabilities (``flip_probs``) and the
    per-axis QuBEC damping parameters (``eta_vec = 1 - 2 * flip_probs``,
    hQVM_QuBEC_Theory.md §9.1). The 64-entry ``walsh_multiplier`` is the product over
    set bits: ``prod_i (1 - 2 p_i)^{a_i}``.
    """
    flip_probs = np.asarray(axis_flip_probs, dtype=np.float64)
    eta_vec = 1.0 - 2.0 * flip_probs
    chars = np.array(
        [[((a >> i) & 1) for i in range(6)] for a in range(64)], dtype=np.int64
    )
    vals = np.ones(64, dtype=np.float64)
    for a in range(64):
        for i in range(6):
            if chars[a, i]:
                vals[a] *= eta_vec[i]
    isotropic = bool(np.allclose(flip_probs, flip_probs[0]))
    eta_iso = float(eta_vec[0]) if isotropic else float("nan")
    return {
        "flip_probs": flip_probs,
        "eta_vec": eta_vec,
        "eta_isotropic": np.array(eta_iso),
        "walsh_multiplier": vals,
        "isotropic": np.array(float(isotropic)),
    }


def q_word6_for_indices(items: Iterable[int]) -> int:
    return api.q_word6_for_items(int(b) for b in items)


# ---------------------------------------------------------------------------
# Words: canonical K4/BU word dataset
# ---------------------------------------------------------------------------

WORD_TYPES = ("identity", "W2", "W2p", "F", "same_family", "reversed_family", "shuffled_family")


def canonical_words() -> list[dict]:
    """One row per (word_type, micro_ref) with exact kernel labels."""
    rows: list[dict] = []
    for micro in range(64):
        words: dict[str, list[int]] = {
            "identity": [],
            "W2": [byte_from_family_micro(0, micro, 6), byte_from_family_micro(1, micro, 6)],
            "W2p": [byte_from_family_micro(2, micro, 6), byte_from_family_micro(3, micro, 6)],
            "F": [
                byte_from_family_micro(0, micro, 6),
                byte_from_family_micro(1, micro, 6),
                byte_from_family_micro(2, micro, 6),
                byte_from_family_micro(3, micro, 6),
            ],
            "same_family": [byte_from_family_micro(0, micro, 6)] * 4,
            "reversed_family": list(reversed(
                [byte_from_family_micro(0, micro, 6), byte_from_family_micro(1, micro, 6),
                 byte_from_family_micro(2, micro, 6), byte_from_family_micro(3, micro, 6)]
            )),
            "shuffled_family": [
                byte_from_family_micro(2, micro, 6),
                byte_from_family_micro(0, micro, 6),
                byte_from_family_micro(3, micro, 6),
                byte_from_family_micro(1, micro, 6),
            ],
        }
        for word_type, word in words.items():
            sig_id = word_signature_id(word)
            rows.append(
                {
                    "word_type": WORD_TYPES.index(word_type),
                    "micro_ref": micro,
                    "word": bytes(word),
                    "sig_id": sig_id,
                    "parity": (sig_id >> 12) & 1,
                    "tau_u6": (sig_id >> 6) & 63,
                    "tau_v6": sig_id & 63,
                }
            )
    return rows


def word_replay_effects(word: list[int]) -> dict[str, Any]:
    """Replay a word from all 4096 states; return exact effect labels."""
    T = transition_table().astype(np.int64)  # [4096, 256]
    dest = np.arange(4096, dtype=np.int64)
    for byte in word:
        dest = T[dest, int(byte)]
    dest = dest.astype(np.uint16)
    idx = np.arange(4096, dtype=np.int64)
    chi_before = ((idx >> 6) & 63) ^ (idx & 63)
    dest64 = dest.astype(np.int64)
    chi_after = ((dest64 >> 6) & 63) ^ (dest64 & 63)

    shell_before = popcount6(chi_before)
    shell_after = popcount6(chi_after)
    return {
        "dest": dest,
        "chi_before": chi_before.astype(np.uint8),
        "chi_after": chi_after.astype(np.uint8),
        "shell_before": shell_before.astype(np.uint8),
        "shell_after": shell_after.astype(np.uint8),
        "chirality_preserved": (chi_after == chi_before).all(),
        "chirality_inverted": bool((chi_after == (chi_before ^ 63)).all()),
        "shell_reflected": bool((shell_after == 6 - shell_before).all()),
        "shell_preserved": bool((shell_after == shell_before).all()),
        "is_involution": bool((dest[dest] == idx).all()),
    }


def canonical_word_dataset() -> dict[str, np.ndarray]:
    """Dataset D: canonical words x replay effects with exact labels."""
    rows = canonical_words()
    effects = [word_replay_effects(list(r["word"])) for r in rows]
    return {
        "word_type": np.array([r["word_type"] for r in rows], dtype=np.uint8),
        "micro_ref": np.array([r["micro_ref"] for r in rows], dtype=np.uint8),
        "sig_id": np.array([r["sig_id"] for r in rows], dtype=np.uint16),
        "parity": np.array([r["parity"] for r in rows], dtype=np.uint8),
        "tau_u6": np.array([r["tau_u6"] for r in rows], dtype=np.uint8),
        "tau_v6": np.array([r["tau_v6"] for r in rows], dtype=np.uint8),
        "dest_mean": np.array([float(e["dest"].mean()) for e in effects], dtype=np.float32),
        "is_involution": np.array([int(e["is_involution"]) for e in effects], dtype=np.uint8),
        "chirality_preserved": np.array([int(e["chirality_preserved"]) for e in effects], dtype=np.uint8),
        "chirality_inverted": np.array([int(e["chirality_inverted"]) for e in effects], dtype=np.uint8),
        "shell_reflected": np.array([int(e["shell_reflected"]) for e in effects], dtype=np.uint8),
        "shell_preserved": np.array([int(e["shell_preserved"]) for e in effects], dtype=np.uint8),
    }


# ---------------------------------------------------------------------------
# Words: word / signature dataset builders and splits
# ---------------------------------------------------------------------------


def all_signature_ids() -> np.ndarray:
    return np.arange(8192, dtype=np.uint16)


def minimal_representative_words(max_len: int = 4) -> dict[int, bytes]:
    """BFS over byte words to find a minimal representative word for every
    reachable Omega signature."""
    representatives: dict[int, bytes] = {}
    queue: deque[tuple[bytes, int]] = deque()
    byte_sigs = [
        word_signature_id([b]) for b in range(256)
    ]
    sig0 = word_signature_id(())
    representatives[sig0] = b""
    queue.append((b"", 0))
    while queue:
        word, sig_id = queue.popleft()
        if len(word) >= max_len:
            continue
        for byte in range(256):
            child = bytes([byte]) + word
            child_sig = api.compose_omega_signatures(
                api.OmegaSignature12(*sig_id_parts(sig_id)),
                api.OmegaSignature12(*sig_id_parts(byte_sigs[byte])),
            )
            child_id = int(signature_id(child_sig.parity, child_sig.tau_u6, child_sig.tau_v6))
            if child_id not in representatives:
                representatives[child_id] = child
                queue.append((child, child_id))
                if len(representatives) >= 8192:
                    return representatives
    return representatives


def _replay_end_index(word: bytes, start_index: int) -> int:
    """Replay a word from a canonical start index via the kernel."""
    u6, v6 = (start_index >> 6) & 63, start_index & 63
    omega = api.OmegaState12(u6=u6, v6=v6)
    for byte in word:
        omega = api.step_omega12_by_byte(omega, int(byte))
    return (omega.u6 << 6) | omega.v6


def stratified_word_sample(
    n_words: int = 4096,
    lengths: tuple[int, ...] = (1, 2, 3, 4),
    seed: int = 0,
) -> list[dict]:
    """Deterministic stratified word sample with exact labels."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    n_lens = len(lengths)
    counts = [n_words // n_lens] * n_lens
    for i in range(n_words % n_lens):
        counts[i] += 1
    seen_sig_counts: dict[int, int] = {}
    for length, count in zip(lengths, counts):
        for _ in range(count):
            word = bytes(rng.integers(0, 256, size=length, dtype=np.uint8))
            sig_id = word_signature_id(list(word))
            parity, tau_u6, tau_v6 = sig_id_parts(sig_id)
            O, E, tp = api.trajectory_parity_commitment(list(word))
            n_seen = seen_sig_counts.get(sig_id, 0)
            seen_sig_counts[sig_id] = n_seen + 1
            rows.append(
                {
                    "word": word,
                    "length": length,
                    "sig_id": sig_id,
                    "parity": parity,
                    "tau_u6": tau_u6,
                    "tau_v6": tau_v6,
                    "q_total": int(api.q_word6_for_items(list(word))),
                    "commitment_O": int(O),
                    "commitment_E": int(E),
                    "commitment_parity": int(tp),
                    "same_signature_different_ledger": int(n_seen > 0),
                    "provenance_needed": 1,
                    "start_index": 0,
                    "end_index": _replay_end_index(word, 0),
                }
            )
    return rows


def same_signature_different_ledger_pairs(
    n_pairs: int = 256, seed: int = 0
) -> list[dict]:
    """Word pairs with identical Omega signature but different exact bytes."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for _ in range(n_pairs):
        length = int(rng.integers(1, 5))
        word = [int(rng.integers(0, 256)) for _ in range(length)]
        pos = int(rng.integers(0, length))
        twin = list(word)
        twin[pos] = int(api.shadow_partner_byte(word[pos]))
        left_id = word_signature_id(word)
        right_id = word_signature_id(twin)
        if left_id != right_id:
            continue
        O_l, E_l, p_l = api.trajectory_parity_commitment(word)
        O_r, E_r, p_r = api.trajectory_parity_commitment(twin)
        rows.append(
            {
                "word_left": bytes(word),
                "word_right": bytes(twin),
                "length": length,
                "sig_id": left_id,
                "signature_equal": 1,
                "commitment_left": (int(O_l), int(E_l), int(p_l)),
                "commitment_right": (int(O_r), int(E_r), int(p_r)),
                "commitment_differs": int((O_l, E_l, p_l) != (O_r, E_r, p_r)),
                "provenance_needed": 1,
            }
        )
    return rows


def shadow_pair_split(byte_rows: dict[str, np.ndarray], seed: int = 0) -> dict[str, np.ndarray]:
    """Shadow-pair regime: both members of each shadow pair share a split."""
    rng = np.random.default_rng(seed)
    pair_ids = byte_rows["shadow_pair_id"]
    unique_pairs = np.unique(pair_ids)
    shuffled = rng.permutation(unique_pairs)
    n_val = len(shuffled) // 5
    split_of_pair = np.zeros(128, dtype=np.int64)
    split_of_pair[shuffled[:n_val]] = 1
    split_of_pair[shuffled[n_val : 2 * n_val]] = 2
    split = split_of_pair[pair_ids]
    return {
        "train": np.nonzero(split == 0)[0],
        "val": np.nonzero(split == 1)[0],
        "test": np.nonzero(split == 2)[0],
    }


def held_out_q_class_split(n_holdout: int = 8, seed: int = 0) -> dict[str, np.ndarray]:
    """Held-out q-class regime: hold out whole q-classes (4 bytes each)."""
    rng = np.random.default_rng(seed)
    q6 = np.array([api.q_word6(b) for b in range(256)])
    heldout = rng.choice(64, size=n_holdout, replace=False)
    is_test = np.isin(q6, heldout)
    return {
        "train": np.nonzero(~is_test)[0],
        "test": np.nonzero(is_test)[0],
    }


# ---------------------------------------------------------------------------
# Byte mechanism: fold targets, factorization audit, depth-4 frames
# ---------------------------------------------------------------------------


def fold_targets() -> dict[str, Any]:
    """Fold-disagreement grade and the flat-byte flag for all 256 bytes."""
    grades = np.array([fold_disagreement_d(b, 6) for b in range(256)], dtype=np.int64)
    flat_flag = (grades == 0).astype(np.int8)
    return {"grade": grades, "flat_flag": flat_flag, "n_flat": int(flat_flag.sum())}


def lr_factorization_audit(byte: int) -> dict[str, bool]:
    """Check the [12,6,2] code structure for a single byte.

    The byte's generative grammar factors exactly as a 12-bit pair-diagonal
    mask (payload via pair expansion), a 6-bit micro reference, and a 2-bit
    family. This audit verifies that structure (the [L]/[R] factorization).
    Exact kernel facts; no re-implementation.
    """
    m12 = api.mask12_for_byte(byte)
    family = int(api.FAMILY_BY_BYTE[byte])
    micro = int(api.MICRO_REF_BY_BYTE[byte])
    intron = int(api.byte_to_intron(byte))
    return {
        "mask_is_pair_diagonal": api.is_pair_diagonal12(m12),
        "family_has_two_bits": 0 <= family <= 3,
        "micro_has_six_bits": 0 <= micro <= 63,
        "intron_has_eight_bits": 0 <= intron <= 255,
        "decomposition_width_20": (
            api.is_pair_diagonal12(m12) and 0 <= family <= 3 and 0 <= micro <= 63
        ),
    }


def depth4_frame_dataset(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Build ``n`` random 4-byte frame records.

    Columns: mask_proj48, intron_seq32 (the identifying record, Formalism 6.3),
    frame_signature, final_state.
    """
    rng = np.random.default_rng(seed)
    bytes_mat = rng.integers(0, 256, size=(n, 4), dtype=np.uint8)
    mask_proj = np.zeros(n, dtype=np.int64)
    intron_seq = np.zeros(n, dtype=np.int64)
    sig_ids = np.zeros(n, dtype=np.int64)
    final_state = np.zeros(n, dtype=np.int64)
    for row in range(n):
        b = [int(x) for x in bytes_mat[row]]
        mask_proj[row] = int(api.depth4_mask_projection48(*b))
        intron_seq[row] = int(api.depth4_intron_sequence32(*b))
        sig_ids[row] = word_signature_id(b)
        omega = api.OmegaState12(u6=0, v6=0)
        for byte in b:
            omega = api.step_omega12_by_byte(omega, byte)
        final_state[row] = (omega.u6 << 6) | omega.v6
    return {
        "bytes": bytes_mat.astype(np.int64),
        "mask_proj48": mask_proj,
        "intron_seq32": intron_seq,
        "frame_signature": sig_ids,
        "final_state": final_state,
    }


def frame_parity_zero(frames: dict[str, np.ndarray]) -> bool:
    """Every sliding 4-byte frame compiles to a pure translation (exact)."""
    for row in range(len(frames["bytes"])):
        word = [int(x) for x in frames["bytes"][row]]
        sig_id = word_signature_id(word)
        parity, _tau_u, _tau_v = sig_id_parts(sig_id)
        if parity != 0:
            return False
    return True


def frame_masks_pair_diagonal(frames: dict[str, np.ndarray]) -> bool:
    """Every per-byte mask projection inside each frame is pair diagonal."""
    for row in range(len(frames["bytes"])):
        b = [int(x) for x in frames["bytes"][row]]
        m48 = api.depth4_mask_projection48(*b)
        for k in range(4):
            m12 = (m48 >> (k * 12)) & 0xFFF
            if not api.is_pair_diagonal12(m12):
                return False
    return True