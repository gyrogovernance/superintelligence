"""Load paths and llama.cpp (C backend) options for local GGUF models."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

_CONFIG_FILENAME = "gyroscopic_llm.yaml"
_DEFAULT_GGUF_REL = "data/models/Bonsai-8B-gguf/Bonsai-8B-Q1_0.gguf"
_DEFAULT_LLAMA_PPL_REL = None


def repo_root() -> Path:
    """Repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parent.parent.parent.parent


def _config_path() -> Path:
    return repo_root() / "config" / _CONFIG_FILENAME


def _defaults_dict() -> dict[str, Any]:
    return {
        "gguf_path": _DEFAULT_GGUF_REL,
        "llama_cli_path": None,
        "llama_perplexity_path": _DEFAULT_LLAMA_PPL_REL,
        "n_ctx": 4096,
        "n_threads": None,
        "n_gpu_layers": 0,
        "verbose": False,
    }


def _load_yaml_raw() -> dict[str, Any]:
    path = _config_path()
    if not path.is_file():
        return _defaults_dict()
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load gyroscopic_llm.yaml. "
            "Install with: pip install pyyaml"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return _defaults_dict()
    out = _defaults_dict()
    out.update(data)
    return out


def _default_llama_cli_candidates() -> list[Path]:
    """Typical CMake output locations for ``llama-cli`` (build from ``external/llama.cpp``)."""
    root = repo_root()
    if sys.platform == "win32":
        return [
            root / "external" / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe",
            root / "external" / "llama.cpp" / "build" / "bin" / "Debug" / "llama-cli.exe",
            root / "external" / "llama.cpp" / "build" / "bin" / "llama-cli.exe",
        ]
    return [root / "external" / "llama.cpp" / "build" / "bin" / "llama-cli"]


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
    n_ctx = int(data.get("n_ctx", 4096))
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

    n_gpu_layers = int(data.get("n_gpu_layers", 0))
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
        v = os.environ["GYROSCOPIC_LLAMA_CLI"].strip()
        out["llama_cli_path"] = v if v else None
    if os.environ.get("GYROSCOPIC_LLAMA_PERPLEXITY"):
        v = os.environ["GYROSCOPIC_LLAMA_PERPLEXITY"].strip()
        out["llama_perplexity_path"] = v if v else None
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
    """Load ``config/gyroscopic_llm.yaml`` if present, else defaults; then apply env overrides."""
    raw = _apply_env(_load_yaml_raw())
    return _parse_config(raw)


def resolve_gguf_path(cfg: GyroscopicLLMConfig) -> Path:
    """Resolve ``cfg.gguf_path`` relative to the repo root when not absolute."""
    p = Path(cfg.gguf_path)
    if p.is_absolute():
        return p
    return repo_root() / p


def resolve_llama_cli_path(cfg: GyroscopicLLMConfig) -> Path:
    """Resolve the ``llama-cli`` executable (native C backend)."""
    root = repo_root()
    if cfg.llama_cli_path:
        p = Path(cfg.llama_cli_path)
        if not p.is_absolute():
            p = root / p
        if p.is_file():
            return p
        raise FileNotFoundError(
            "gyroscopic_llm: llama_cli_path is set but file not found: " + str(p)
        )
    tried: list[str] = []
    for c in _default_llama_cli_candidates():
        tried.append(str(c))
        if c.is_file():
            return c
    raise FileNotFoundError(
        "gyroscopic_llm: llama-cli not found. Build llama.cpp under external/llama.cpp "
        "(see external/llama.cpp/docs/build.md) or set llama_cli_path in config/gyroscopic_llm.yaml "
        "or GYROSCOPIC_LLAMA_CLI. Tried:\n  " + "\n  ".join(tried)
    )


def resolve_llama_perplexity_path(cfg: GyroscopicLLMConfig) -> Path:
    root = repo_root()
    if cfg.llama_perplexity_path:
        p = Path(cfg.llama_perplexity_path)
        if not p.is_absolute():
            p = root / p
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
        "llama_perplexity_path in config/gyroscopic_llm.yaml or GYROSCOPIC_LLAMA_PERPLEXITY. Tried:\n  "
        + "\n  ".join(tried)
    )
