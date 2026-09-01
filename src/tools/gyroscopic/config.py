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
    """CMake output under ``src/tools/gyroscopic/_build`` (never repo-root ``_build``)."""
    root = repo_root()
    build_name = "llama-cpp-stock" if backend == "stock" else "llama-cpp"
    base = root / "src" / "tools" / "gyroscopic" / "_build" / build_name / "bin"
    if sys.platform == "win32":
        return [
            base / "Release" / "llama-cli.exe",
            base / "Debug" / "llama-cli.exe",
            base / "llama-cli.exe",
        ]
    return [base / "llama-cli"]


def _default_llama_perplexity_candidates() -> list[Path]:
    root = repo_root()
    base = root / "src" / "tools" / "gyroscopic" / "_build" / "llama-cpp" / "bin"
    if sys.platform == "win32":
        return [
            base / "Release" / "llama-perplexity.exe",
            base / "Debug" / "llama-perplexity.exe",
            base / "llama-perplexity.exe",
        ]
    return [base / "llama-perplexity"]


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

    ``backend`` is ``"stock"`` (``_build/llama-cpp-stock``) or ``"gyroscopic"`` (``_build/llama-cpp``).
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
        "src/tools/gyroscopic/_build/llama-cpp-stock"
        if backend == "stock"
        else "src/tools/gyroscopic/_build/llama-cpp"
    )
    raise FileNotFoundError(
        "gyroscopic_llm: llama-cli not found for "
        f"{backend} backend. Build into {build_hint} "
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


def production_gyroscopic_env(
    *,
    stats: bool = False,
    ledger_path: str | None = None,
    holonomic_kv: bool = False,
    incomplete_forward: bool = False,
) -> dict[str, str]:
    """Environment for gyroscopic Bonsai-8B-Q1_0.

    Always sets GYRO_LEDGER_* and ensures the thin HQVMLEDS ledger file exists
    (see ``ledger.ensure_ledger``). holonomic_kv enables Q8 K/V + holonomic attn.

    With holonomic_kv: native driver + owned site laws (Attn shell+QK, FFN shell gate,
    Norm commit, RoPE codec, residual Δ-law, CGM lift).
    incomplete_forward: unfinished stress flags only (not production law).
    """
    from .ledger import ensure_ledger

    env: dict[str, str] = {
        "GYRO_LEDGER_STRICT": "1",
        "GYRO_LEDGER_ALLOW": (
            "attn_q.weight,attn_k.weight,attn_v.weight,attn_output.weight,"
            "ffn_gate.weight,ffn_up.weight,ffn_down.weight,"
            "token_embd.weight,output.weight"
        ),
    }
    path = ensure_ledger(ledger_path) if ledger_path else ensure_ledger()
    env["GYRO_LEDGER_PATH"] = str(path)
    if stats:
        env["GYRO_LEDGER_VERBOSE"] = "1"
    if holonomic_kv or incomplete_forward:
        env["GYRO_KV_KQ8"] = "1"
        env["GYRO_KV_V"] = "1"
        env["GYRO_HOLONOMIC_ATTN"] = "1"
    if holonomic_kv and not incomplete_forward:
        env["GYRO_NATIVE_FORWARD"] = "1"
        env["GYRO_NATIVE_EMISSION"] = "1"
        env["GYRO_ATTN_SHELL_QK"] = "1"
        # H5/H6/H7: dyad×Q8 scores, dyad×Q8 V-reduce, FFN L2 joint law (Paris held).
        env["GYRO_NATIVE_ATTN_SCORES"] = "1"
        # Documented FFN L2 joint law (Theory_Drop §4.1.2); Paris held with opt-in.
        env["GYRO_FFN_NATIVE"] = "1"
        # Analysis §7.3 dyad×Q8 V-reduce; Paris held.
        env["GYRO_NATIVE_VREDUCE"] = "1"
        env["GYRO_NORM_COMMIT"] = "1"
        env["GYRO_ROPE_CODEC"] = "1"
        env["GYRO_CGM_LIFT"] = "1"
        env["GYRO_RESIDUAL_LAW"] = "1"
        env["GYRO_PI_FROM_EMBD"] = "1"
        # Bonsai-8B-Q1_0 is Qwen3 (GGUF qwen3.rope.freq_base=1e6, yarn×4 → freq_scale=0.25).
        env["GYRO_ROPE_FREQ_BASE"] = "1000000"
        env["GYRO_ROPE_FREQ_SCALE"] = "0.25"
    if incomplete_forward:
        # Causal-gate stress path (no native forward); not product holonomic.
        env["GYRO_ROPE_CODEC"] = "1"
        env["GYRO_CGM_LIFT"] = "1"
        env["GYRO_RESIDUAL_LAW"] = "1"
        # Do not set GYRO_RESIDUAL_HYBRID (deprecated alias of GYRO_RESIDUAL_LAW).
    return env