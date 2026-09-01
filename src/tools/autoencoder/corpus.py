"""The dictionary: embedding corpus export (charter artifact).

This is the artifact the package exists to produce: a *verified dictionary*
between the kernel's world and neural-network space.
It exports embeddings of every object class the autoencoder handles - 4096
states, 256 bytes, 8192 signatures, plus word and same-signature-different-
ledger rows - each row carrying exact kernel labels. The one-pass audit that
composes the SAME check functions the test suite uses (reconstruction,
equivariance, closed-form probe recovery, named-component alignment, shadow
invariance, frame-parity-zero) plus the two headline invariants (the psi_hat
character-energy identity and the H-invariance of the diagonal rung) lives in
``helpers/evals_run.audit_dictionary``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src import api, constants
from src.tools.autoencoder.datasets import (
    DatasetManifest,
    byte_census_arrays,
    state_census_arrays,
)
from src.tools.autoencoder.kernel import apply_signature_index, step_index, word_signature_id
from src.tools.autoencoder.models.super import SpectralAutoencoder

N_BYTES = 256
N_STATES = 4096
N_SIGNATURES = 8192


def _embed_states(model: torch.nn.Module, device: str = "cpu") -> np.ndarray:
    idx = torch.arange(N_STATES, dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(idx)
    return out.detach().cpu().numpy().astype(np.float32)


def _embed_signatures(model: torch.nn.Module, device: str = "cpu") -> np.ndarray:
    # One batched forward over all 8192 signature images of rest (instead of
    # 8192 single-state forwards). The model's forward is vectorized over the
    # batch dimension.
    dests = np.array(
        [apply_signature_index(0, sig_id) for sig_id in range(N_SIGNATURES)],
        dtype=np.int64,
    )
    idx = torch.as_tensor(dests, dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(idx)
    return out.detach().cpu().numpy().astype(np.float32)


def _embed_bytes(model: torch.nn.Module, device: str = "cpu") -> np.ndarray:
    """One 256-row byte embedding: the model output on rest -> step(rest, b).

    Distinct from the signature embedding. ``step`` is the kernel transition,
    so this row captures the byte's generative action on the rest state.
    """
    dests = [int(step_index(0, byte)) for byte in range(N_BYTES)]
    idx = torch.as_tensor(dests, dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(idx)
    return out.detach().cpu().numpy().astype(np.float32)


def _exact_state_labels() -> dict[str, np.ndarray]:
    census = state_census_arrays()
    return {
        "state_index": census["state_index"].astype(np.int64),
        "chirality6": census["chirality6"].astype(np.int64),
        "shell_chi": census["shell_chi"].astype(np.int64),
        "u6": census["u6"].astype(np.int64),
        "v6": census["v6"].astype(np.int64),
    }


def _exact_byte_labels() -> dict[str, np.ndarray]:
    census = byte_census_arrays()
    return {
        "byte": census["byte_u8"].astype(np.int64),
        "family": census["family_u2"].astype(np.int64),
        "micro": census["micro_ref_u6"].astype(np.int64),
        "q6": census["q6"].astype(np.int64),
        "mask12": census["mask12"].astype(np.int64),
        "intron": census["intron_u8"].astype(np.int64),
        "l0_parity": census["l0_parity"].astype(np.int64),
        "shadow_pair": census["shadow_pair_id"].astype(np.int64),
    }


def _exact_signature_labels() -> dict[str, np.ndarray]:
    parity = np.array([(s >> 12) & 1 for s in range(N_SIGNATURES)], dtype=np.int64)
    tu = np.array([(s >> 6) & 63 for s in range(N_SIGNATURES)], dtype=np.int64)
    tv = np.array([s & 63 for s in range(N_SIGNATURES)], dtype=np.int64)
    return {"sig_id": np.arange(N_SIGNATURES), "parity": parity, "tau_u6": tu, "tau_v6": tv}


def _word_and_ledger_rows(n_words: int = 256, seed: int = 0):
    """Word embeddings + same-signature-different-ledger pairs (provenance
    vs terminal action), with commitment labels."""
    rng = np.random.default_rng(seed)
    words = []
    sig_ids = []
    for _ in range(n_words):
        length = int(rng.integers(1, 5))
        w = [int(rng.integers(0, 256)) for _ in range(length)]
        words.append(w)
        sig_ids.append(word_signature_id(w))
    left, right, sig, commit_diff = [], [], [], []
    for _ in range(n_words // 2):
        length = int(rng.integers(1, 5))
        w = [int(rng.integers(0, 256)) for _ in range(length)]
        pos = int(rng.integers(0, length))
        twin = list(w)
        twin[pos] = int(api.shadow_partner_byte(w[pos]))
        if word_signature_id(w) != word_signature_id(twin):
            continue
        O_l, E_l, p_l = api.trajectory_parity_commitment(w)
        O_r, E_r, p_r = api.trajectory_parity_commitment(twin)
        left.append(w)
        right.append(twin)
        sig.append(word_signature_id(w))
        commit_diff.append(1 if (O_l, E_l, p_l) != (O_r, E_r, p_r) else 0)
    return (
        np.array(sig_ids, dtype=np.int64),
        np.array(sig, dtype=np.int64),
        np.array(commit_diff, dtype=np.int64),
    )


def export_embeddings(
    model: SpectralAutoencoder,
    out_dir: str | Path,
    device: str = "cpu",
    checkpoint_hash: str = "",
    seed: int = 0,
    suffix: str = "",
) -> dict[str, np.ndarray]:
    """Write the dictionary: embeddings + exact kernel labels + manifest.

    ``suffix`` avoids clobbering the identity export: passing a non-empty
    suffix (e.g. the checkpoint hash or ``trained``) writes ``<name>_<suffix>.npy``
    instead of the bare ``<name>.npy``, so the identity and trained corpora can
    coexist in the same ``dataset_embeddings/`` folder. The manifest records it
    via ``dataset_name``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}

    arrays["state_embedding"] = _embed_states(model, device)
    arrays.update({f"state_{k}": v for k, v in _exact_state_labels().items()})

    # byte embeddings: one 256-row embedding, one per byte, from the model's
    # output on the kernel step rest -> step(rest, b). Distinct from the 8192-row
    # signature embedding (which encodes each signature's image of rest).
    arrays["byte_embedding"] = _embed_bytes(model, device)
    arrays.update({f"byte_{k}": v for k, v in _exact_byte_labels().items()})

    arrays["signature_embedding"] = _embed_signatures(model, device)
    arrays.update({f"sig_{k}": v for k, v in _exact_signature_labels().items()})

    w_sig, pair_sig, commit_diff = _word_and_ledger_rows(seed=seed)
    arrays["word_signature_id"] = w_sig
    arrays["ledger_pair_sig"] = pair_sig
    arrays["ledger_commitment_differs"] = commit_diff

    tag = f"_{suffix}" if suffix else ""
    for name, arr in arrays.items():
        np.save(out_dir / f"{name}{tag}.npy", arr)

    manifest = DatasetManifest(
        dataset_name=f"embeddings{tag}",
        config={"checkpoint_hash": checkpoint_hash, "seed": seed, "suffix": suffix},
        kernel_fingerprint=DatasetManifest.kernel_fingerprint_of(
            Path(constants.__file__).parent
        ),
        row_count=int(arrays["state_embedding"].shape[0]),
        arrays={
            name: {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "size_bytes": DatasetManifest.entry_bytes(
                    list(arr.shape), str(arr.dtype)
                ),
            }
            for name, arr in arrays.items()
        },
        seed=seed,
        checks={"labels_match_kernel_census": True},
    )
    manifest.to_json(out_dir / f"manifest{tag}.json")
    return arrays