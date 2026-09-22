"""Frozen production AE constellation: Narrow, General (K4 + spectral), Super."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.tools.autoencoder import paths
from src.tools.autoencoder.helpers.evals_run import load_any_checkpoint
from src.tools.autoencoder.kernel import step_index
from src.tools.autoencoder.models.general import AffineSpectralCodec, K4Autoencoder
from src.tools.autoencoder.models.narrow import MLPAutoencoder
from src.tools.autoencoder.models.super import Super
from src.tools.autoencoder.programs.genomics.genomics import (
    NucleotideEncoding,
    clean_acgt,
    genomic_byte_stream,
    pair_inversion_encoding,
)
from src.tools.autoencoder.programs.genomics.synthesis.climate import (
    codons_of,
    mean_adjacent_shell,
)

_ENC = pair_inversion_encoding()
_WALK_CAP = 512
_N_WINDOWS = 4


@dataclass(frozen=True)
class FrozenConstellation:
    k4: K4Autoencoder
    mlp: MLPAutoencoder
    spectral: AffineSpectralCodec
    super_model: Super
    device: str


def production_checkpoint_paths() -> dict[str, Path]:
    root = paths.checkpoints_dir() / "production"
    return {
        "k4": root / "k4_full.pt",
        "mlp": root / "mlp_full.pt",
        "spectral": root / "spectral_bottleneck.pt",
        "super": root / "super.pt",
    }


def _load_spectral(path: Path, device: str) -> AffineSpectralCodec:
    payload = torch.load(path, map_location=device, weights_only=False)
    extra_raw = payload.get("extra") if isinstance(payload, dict) else None
    extra: dict = extra_raw if isinstance(extra_raw, dict) else {}
    cfg_raw = extra.get("model_config")
    cfg: dict = cfg_raw if isinstance(cfg_raw, dict) else {}
    model = AffineSpectralCodec(
        init_gain=float(cfg.get("init_gain", 1.0)),
        ladder=cfg.get("ladder", extra.get("ladder")),
        sector_mask=cfg.get("sector_mask"),
        orbit_index=cfg.get("orbit_index"),
        frozen=bool(cfg.get("frozen", True)),
    )
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _freeze(model: torch.nn.Module) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def load_frozen_constellation(device: str = "cpu") -> FrozenConstellation:
    needed = production_checkpoint_paths()
    missing = [str(p) for p in needed.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing production checkpoints: " + ", ".join(missing))

    k4, _ = load_any_checkpoint(needed["k4"], device=device)
    mlp, _ = load_any_checkpoint(needed["mlp"], device=device)
    super_model, _ = load_any_checkpoint(needed["super"], device=device)
    spectral = _load_spectral(needed["spectral"], device)

    if not isinstance(k4, K4Autoencoder):
        raise TypeError(f"k4_full is {type(k4)}")
    if not isinstance(mlp, MLPAutoencoder):
        raise TypeError(f"mlp_full is {type(mlp)}")
    if not isinstance(super_model, Super):
        raise TypeError(f"super is {type(super_model)}")

    for model in (k4, mlp, spectral, super_model):
        model.to(device)
        _freeze(model)

    return FrozenConstellation(
        k4=k4,
        mlp=mlp,
        spectral=spectral,
        super_model=super_model,
        device=device,
    )


def _spectral_config(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    extra_raw = payload.get("extra") if isinstance(payload, dict) else None
    extra: dict = extra_raw if isinstance(extra_raw, dict) else {}
    cfg_raw = extra.get("model_config")
    cfg: dict = cfg_raw if isinstance(cfg_raw, dict) else {}
    if "ladder" not in cfg and "ladder" in extra:
        cfg = {**cfg, "ladder": extra.get("ladder")}
    return cfg


def random_constellation(seed: int = 0, device: str = "cpu") -> FrozenConstellation:
    """Same architectures as production, untrained weights.

    Reads constructor configs from the production checkpoints so the random
    control matches Narrow / K4 / spectral / Super exactly, then leaves the
    weights at their default initialization. Shared analytic layers (ExactHQVMScan,
    XOR signature registers) remain part of Super by construction.
    """
    needed = production_checkpoint_paths()
    missing = [str(p) for p in needed.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing production checkpoints: " + ", ".join(missing))

    k4_prod, _ = load_any_checkpoint(needed["k4"], device="cpu")
    mlp_prod, _ = load_any_checkpoint(needed["mlp"], device="cpu")
    super_prod, _ = load_any_checkpoint(needed["super"], device="cpu")
    if not isinstance(k4_prod, K4Autoencoder):
        raise TypeError(f"k4_full is {type(k4_prod)}")
    if not isinstance(mlp_prod, MLPAutoencoder):
        raise TypeError(f"mlp_full is {type(mlp_prod)}")
    if not isinstance(super_prod, Super):
        raise TypeError(f"super is {type(super_prod)}")

    cfg = _spectral_config(needed["spectral"])
    torch.manual_seed(int(seed))
    k4 = K4Autoencoder(**k4_prod.get_config())
    mlp = MLPAutoencoder(**mlp_prod.get_config())
    super_model = Super(**super_prod.get_config())
    spectral = AffineSpectralCodec(
        init_gain=float(cfg.get("init_gain", 1.0)),
        ladder=cfg.get("ladder"),
        sector_mask=cfg.get("sector_mask"),
        orbit_index=cfg.get("orbit_index"),
        frozen=bool(cfg.get("frozen", True)),
    )
    for model in (k4, mlp, spectral, super_model):
        model.to(device)
        _freeze(model)
    return FrozenConstellation(
        k4=k4,
        mlp=mlp,
        spectral=spectral,
        super_model=super_model,
        device=device,
    )


@torch.inference_mode()
def state_latents(
    constellation: FrozenConstellation,
    states: np.ndarray,
) -> dict[str, np.ndarray]:
    if len(states) == 0:
        return {
            "k4": np.zeros((0, 1), dtype=np.float32),
            "mlp": np.zeros((0, 1), dtype=np.float32),
            "spectral": np.zeros((0, 1), dtype=np.float32),
        }
    x = torch.as_tensor(states, dtype=torch.long, device=constellation.device)
    return {
        "k4": constellation.k4.encode(x).detach().cpu().numpy(),
        "mlp": constellation.mlp.encode(x).detach().cpu().numpy(),
        "spectral": constellation.spectral.encode(x).detach().cpu().numpy(),
    }


@torch.inference_mode()
def super_forward(
    constellation: FrozenConstellation,
    byte_ledger: Sequence[int],
) -> dict[str, np.ndarray]:
    """Run Super on a genomic byte ledger; return context and selected heads."""
    if not byte_ledger:
        return {
            "context": np.zeros(64, dtype=np.float32),
            "family_logits": np.zeros(4, dtype=np.float32),
            "climate_logits": np.zeros(256, dtype=np.float32),
            "raw_pool": np.zeros(16, dtype=np.float32),
        }
    ledger = torch.as_tensor(
        [list(byte_ledger)],
        dtype=torch.long,
        device=constellation.device,
    )
    out = constellation.super_model(
        ledger,
        mask=None,
        hide_masked=True,
        apply_markov=False,
    )
    result: dict[str, np.ndarray] = {}
    for key in ("context", "family_logits", "climate_logits", "raw_pool"):
        tensor = out[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Super {key} must be a Tensor")
        arr = tensor[0].detach().cpu().numpy().astype(np.float32, copy=False)
        result[key] = arr
    return result


def ledger_bytes(cds: str, enc: NucleotideEncoding | None = None, cap: int = _WALK_CAP) -> list[int]:
    """Genomic byte ledger for one CDS, truncated to the AE walk cap."""
    encoding = enc or _ENC
    return list(genomic_byte_stream(clean_acgt(cds), encoding))[:cap]


def walk_states(stream: Sequence[int], cap: int = _WALK_CAP) -> np.ndarray:
    """Omega walk states visited by a byte stream."""
    states: list[int] = []
    cur = 0
    for byte in list(stream)[:cap]:
        states.append(cur)
        cur = int(step_index(cur, int(byte)))
    if not states:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(states, dtype=np.int64)


def window_heterogeneity(latents: np.ndarray) -> float:
    """Mean per-window latent std along an Omega walk."""
    if len(latents) < _N_WINDOWS:
        return float(np.std(latents)) if len(latents) else float("nan")
    bounds = np.linspace(0, len(latents), _N_WINDOWS + 1).astype(np.int64)
    stds: list[float] = []
    for k in range(_N_WINDOWS):
        segment = latents[bounds[k] : bounds[k + 1]]
        if len(segment) == 0:
            continue
        stds.append(float(segment.std(axis=0).mean()))
    return float(np.mean(stds)) if stds else float("nan")


def _spectral_participation(spectrum: np.ndarray) -> float:
    values = np.abs(np.asarray(spectrum, dtype=np.float64).ravel())
    sum_sq = float((values**2).sum())
    if sum_sq <= 0:
        return float("nan")
    return float((values.sum() ** 2) / (len(values) * sum_sq))


@dataclass(frozen=True)
class ConstellationRead:
    """Frozen Narrow + General + Super read of one CDS."""

    narrow_het: float
    k4_het: float
    spectral_pr: float
    super_climate_mean: float
    super_context: tuple[float, ...]
    kernel_mean_shell: float


def read_constellation(
    holder: FrozenConstellation,
    cds: str,
    *,
    enc: NucleotideEncoding | None = None,
) -> ConstellationRead:
    """Score one CDS through the frozen production constellation."""
    encoding = enc or _ENC
    stream = ledger_bytes(cds, encoding)
    states = walk_states(stream)
    kernel_mean_shell = float(mean_adjacent_shell(codons_of(cds)))
    if len(states) == 0:
        return ConstellationRead(
            narrow_het=float("nan"),
            k4_het=float("nan"),
            spectral_pr=float("nan"),
            super_climate_mean=float("nan"),
            super_context=tuple(0.0 for _ in range(64)),
            kernel_mean_shell=kernel_mean_shell,
        )
    latents = state_latents(holder, states)
    mlp = np.asarray(latents["mlp"], dtype=np.float64)
    k4 = np.asarray(latents["k4"], dtype=np.float64)
    spectral = np.asarray(latents["spectral"], dtype=np.float64)
    super_out = super_forward(holder, stream)
    climate = np.asarray(super_out["climate_logits"], dtype=np.float64).ravel()
    context = np.asarray(super_out["context"], dtype=np.float64).ravel()
    norm = float(np.linalg.norm(context))
    if norm > 0:
        context = context / norm
    return ConstellationRead(
        narrow_het=window_heterogeneity(mlp),
        k4_het=window_heterogeneity(k4),
        spectral_pr=_spectral_participation(
            spectral.mean(axis=0) if spectral.ndim == 2 else spectral
        ),
        super_climate_mean=float(climate.mean()) if climate.size else float("nan"),
        super_context=tuple(float(x) for x in context),
        kernel_mean_shell=kernel_mean_shell,
    )
