"""Paths and llama.cpp (C backend) options for local GGUF models.

Defaults are baked in here. Override via environment only:

  GYROSCOPIC_GGUF_PATH
  GYROSCOPIC_LLAMA_CLI
  GYROSCOPIC_LLAMA_PERPLEXITY
  GYROSCOPIC_N_CTX
  GYROSCOPIC_N_THREADS
  GYROSCOPIC_N_GPU_LAYERS
  GYROSCOPIC_VERBOSE
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_GGUF_REL = "data/models/Bonsai-8B-gguf/Bonsai-8B-Q1_0.gguf"
_DEFAULT_N_CTX = 4096
_DEFAULT_N_GPU_LAYERS = 0


def _env_opt_path(name: str) -> str | None:
    v = os.environ.get(name)
    if not v:
        return None
    v = v.strip()
    return v or None


def _resolve_maybe_relative(root: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else root / p


def repo_root() -> Path:
    """Repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parent.parent.parent.parent


def _defaults_dict() -> dict[str, Any]:
    return {
        "gguf_path": _DEFAULT_GGUF_REL,
        "llama_cli_path": None,
        "llama_perplexity_path": None,
        "n_ctx": _DEFAULT_N_CTX,
        "n_threads": None,
        "n_gpu_layers": _DEFAULT_N_GPU_LAYERS,
        "verbose": False,
    }


def _default_llama_cli_candidates(*, backend: str = "gyroscopic") -> list[Path]:
    """Typical CMake output locations for ``llama-cli`` (build from ``external/llama.cpp``)."""
    root = repo_root()
    build_dir = "build-stock" if backend == "stock" else "build"
    if sys.platform == "win32":
        return [
            root / "external" / "llama.cpp" / build_dir / "bin" / "Release" / "llama-cli.exe",
            root / "external" / "llama.cpp" / build_dir / "bin" / "Debug" / "llama-cli.exe",
            root / "external" / "llama.cpp" / build_dir / "bin" / "llama-cli.exe",
        ]
    return [root / "external" / "llama.cpp" / build_dir / "bin" / "llama-cli"]


def _default_llama_perplexity_candidates() -> list[Path]:
    root = repo_root()
    if sys.platform == "win32":
        return [
            root / "external" / "llama.cpp" / "build" / "bin" / "Release" / "llama-perplexity.exe",
            root / "external" / "llama.cpp" / "build" / "bin" / "Debug" / "llama-perplexity.exe",
            root / "external" / "llama.cpp" / "build" / "bin" / "llama-perplexity.exe",
        ]
    return [root / "external" / "llama.cpp" / "build" / "bin" / "llama-perplexity"]


@dataclass(frozen=True)
class GyroscopicLLMConfig:
    """Settings for the native ``llama-cli`` binary (llama.cpp C backend)."""

    gguf_path: str
    llama_cli_path: str | None
    llama_perplexity_path: str | None
    n_ctx: int
    n_threads: int | None
    n_gpu_layers: int
    verbose: bool


def _parse_config(data: dict[str, Any]) -> GyroscopicLLMConfig:
    gguf = data.get("gguf_path", _DEFAULT_GGUF_REL)
    if not isinstance(gguf, str):
        raise TypeError("gyroscopic_llm: gguf_path must be a string")
    raw_cli = data.get("llama_cli_path", None)
    llama_cli_path: str | None
    if raw_cli is None or raw_cli == "":
        llama_cli_path = None
    elif isinstance(raw_cli, str):
        llama_cli_path = raw_cli.strip() or None
    else:
        raise TypeError("gyroscopic_llm: llama_cli_path must be a string or null")
    raw_ppl = data.get("llama_perplexity_path", None)
    llama_perplexity_path: str | None
    if raw_ppl is None or raw_ppl == "":
        llama_perplexity_path = None
    elif isinstance(raw_ppl, str):
        llama_perplexity_path = raw_ppl.strip() or None
    else:
        raise TypeError("gyroscopic_llm: llama_perplexity_path must be a string or null")
    n_ctx = int(data.get("n_ctx", _DEFAULT_N_CTX))
    if n_ctx <= 0:
        raise ValueError(f"gyroscopic_llm: n_ctx must be positive, got {n_ctx}")

    raw_threads = data.get("n_threads", None)
    n_threads: int | None
    if raw_threads is None:
        n_threads = None
    else:
        n_threads = int(raw_threads)
        if n_threads <= 0:
            raise ValueError(
                f"gyroscopic_llm: n_threads must be positive if set, got {n_threads}"
            )

    n_gpu_layers = int(data.get("n_gpu_layers", _DEFAULT_N_GPU_LAYERS))
    if n_gpu_layers < 0:
        raise ValueError(
            f"gyroscopic_llm: n_gpu_layers must be >= 0, got {n_gpu_layers}"
        )

    verbose = bool(data.get("verbose", False))
    return GyroscopicLLMConfig(
        gguf_path=gguf,
        llama_cli_path=llama_cli_path,
        llama_perplexity_path=llama_perplexity_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )


def _apply_env(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    if os.environ.get("GYROSCOPIC_GGUF_PATH"):
        out["gguf_path"] = os.environ["GYROSCOPIC_GGUF_PATH"].strip()
    if os.environ.get("GYROSCOPIC_LLAMA_CLI"):
        out["llama_cli_path"] = _env_opt_path("GYROSCOPIC_LLAMA_CLI")
    if os.environ.get("GYROSCOPIC_LLAMA_PERPLEXITY"):
        out["llama_perplexity_path"] = _env_opt_path("GYROSCOPIC_LLAMA_PERPLEXITY")
    if os.environ.get("GYROSCOPIC_N_CTX"):
        out["n_ctx"] = int(os.environ["GYROSCOPIC_N_CTX"].strip())
    if os.environ.get("GYROSCOPIC_N_THREADS"):
        v = os.environ["GYROSCOPIC_N_THREADS"].strip().lower()
        out["n_threads"] = None if v in ("", "null", "none") else int(v)
    if os.environ.get("GYROSCOPIC_N_GPU_LAYERS"):
        out["n_gpu_layers"] = int(os.environ["GYROSCOPIC_N_GPU_LAYERS"].strip())
    if os.environ.get("GYROSCOPIC_VERBOSE"):
        out["verbose"] = os.environ["GYROSCOPIC_VERBOSE"].strip().lower() in (
            "1",
            "true",
            "yes",
        )
    return out


def get_gyroscopic_llm_config() -> GyroscopicLLMConfig:
    """Return defaults with ``GYROSCOPIC_*`` environment overrides applied."""
    return _parse_config(_apply_env(_defaults_dict()))


def resolve_gguf_path(cfg: GyroscopicLLMConfig) -> Path:
    """GGUF model path (CS anchor for the llama backend).

    Resolve ``cfg.gguf_path`` relative to the repo root when not absolute.
    """
    p = Path(cfg.gguf_path)
    if p.is_absolute():
        return p
    return repo_root() / p


def resolve_llama_cli_path(cfg: GyroscopicLLMConfig, *, backend: str = "gyroscopic") -> Path:
    """Resolve the ``llama-cli`` executable (native C backend).

    ``backend`` is ``"stock"`` (vanilla ``build-stock``) or ``"gyroscopic"`` (``build``).
    """
    root = repo_root()
    if cfg.llama_cli_path and backend == "gyroscopic":
        p = _resolve_maybe_relative(root, cfg.llama_cli_path)
        assert p is not None
        if p.is_file():
            return p
        raise FileNotFoundError(
            "gyroscopic_llm: llama_cli_path is set but file not found: " + str(p)
        )
    tried: list[str] = []
    for c in _default_llama_cli_candidates(backend=backend):
        tried.append(str(c))
        if c.is_file():
            return c
    build_hint = (
        "build-stock (stock)" if backend == "stock" else "build (gyroscopic)"
    )
    raise FileNotFoundError(
        "gyroscopic_llm: llama-cli not found for "
        f"{backend} backend. Build external/llama.cpp/{build_hint} "
        "or set GYROSCOPIC_LLAMA_CLI. Tried:\n  " + "\n  ".join(tried)
    )


def resolve_llama_perplexity_path(cfg: GyroscopicLLMConfig) -> Path:
    root = repo_root()
    if cfg.llama_perplexity_path:
        p = _resolve_maybe_relative(root, cfg.llama_perplexity_path)
        assert p is not None
        if p.is_file():
            return p
        raise FileNotFoundError(
            "gyroscopic_llm: llama_perplexity_path is set but file not found: " + str(p)
        )
    tried: list[str] = []
    for c in _default_llama_perplexity_candidates():
        tried.append(str(c))
        if c.is_file():
            return c
    try:
        return resolve_llama_cli_path(cfg)
    except FileNotFoundError:
        pass
    raise FileNotFoundError(
        "gyroscopic_llm: llama-perplexity not found. Build llama.cpp tools or set "
        "GYROSCOPIC_LLAMA_PERPLEXITY. Tried:\n  " + "\n  ".join(tried)
    )
