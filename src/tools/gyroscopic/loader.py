"""Run the native ``llama-cli`` (llama.cpp C backend) from :class:`GyroscopicLLMConfig`."""
from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .config import (
    GyroscopicLLMConfig,
    resolve_gguf_path,
    resolve_llama_cli_path,
    resolve_llama_perplexity_path,
)

def _run_command(
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout_sec: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with stdout/stderr captured as text."""
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=env,
        timeout=timeout_sec,
        check=False,
    )

def _thread_count(cfg: GyroscopicLLMConfig) -> int:
    """Thread count for llama-cli (-t); defaults to CPU count."""
    t = cfg.n_threads
    if t is None:
        return max(1, (os.cpu_count() or 4))
    return int(t)

def _require_gguf_path(cfg: GyroscopicLLMConfig) -> Path:
    """Resolve and verify the configured GGUF exists on disk."""
    gguf = resolve_gguf_path(cfg)
    if not gguf.is_file():
        raise FileNotFoundError(f"GGUF not found: {gguf}")
    return gguf


def _gyro_env(base: dict[str, str] | None = None, *, strict: bool = False) -> dict[str, str]:
    """Prepare env for gyroscopic llama-cli runs (GGML_GYROSCOPIC=1)."""
    env = dict(os.environ) if base is None else dict(base)
    env["GGML_GYROSCOPIC"] = "1"
    if strict:
        env["GGML_GYROSCOPIC_STRICT"] = "1"
    return env


def _llama_engine_prefix(exe: Path, gguf: Path, cfg: GyroscopicLLMConfig) -> list[str]:
    return [
        str(exe),
        "-m", str(gguf),
        "-c", str(cfg.n_ctx),
        "-t", str(_thread_count(cfg)),
        "-ngl", str(cfg.n_gpu_layers),
    ]

def build_llama_cli_command(
    cfg: GyroscopicLLMConfig,
    *,
    prompt: str,
    n_predict: int,
    extra_args: Sequence[str] | None = None,
    backend: str = "gyroscopic",
    llama_cli_exe: Path | None = None,
) -> list[str]:
    """Build argv for ``llama-cli``.

    ``backend`` is ``stock`` or ``gyroscopic``. Pass ``llama_cli_exe`` to pin the binary
    (used by the compare benchmark after dual builds).
    """
    exe = llama_cli_exe if llama_cli_exe is not None else resolve_llama_cli_path(cfg, backend=backend)
    gguf = _require_gguf_path(cfg)
    args: list[str] = [
        *_llama_engine_prefix(exe, gguf, cfg),
        "-n", str(n_predict),
        "-p", prompt,
        "--no-display-prompt",
        "--single-turn",
    ]
    if cfg.verbose:
        args.append("-v")
    if extra_args:
        args.extend(extra_args)
    return args

def run_llama_cli(
    cfg: GyroscopicLLMConfig,
    *,
    prompt: str,
    n_predict: int,
    extra_args: Sequence[str] | None = None,
    timeout_sec: float | None = 900.0,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``llama-cli`` once; returns completed process."""
    argv = build_llama_cli_command(cfg, prompt=prompt, n_predict=n_predict, extra_args=extra_args)
    run_env = dict(os.environ) if env is None else env
    return _run_command(argv, env=run_env, timeout_sec=timeout_sec)

def run_llama_cli_smoke(
    cfg: GyroscopicLLMConfig,
    *,
    timeout_sec: float | None = 900.0,
) -> subprocess.CompletedProcess[str]:
    """Load the GGUF once with the gyroscopic backend (strict mode) as a sanity check."""
    return run_llama_cli(
        cfg,
        prompt=".",
        n_predict=1,
        extra_args=[],
        timeout_sec=timeout_sec,
        env=_gyro_env(strict=True),
    )

def run_llama_cli_version(cfg: GyroscopicLLMConfig) -> subprocess.CompletedProcess[str]:
    """Run ``llama-cli --version``."""
    exe = resolve_llama_cli_path(cfg)
    return _run_command([str(exe), "--version"], timeout_sec=30.0)

def build_llama_perplexity_command(
    cfg: GyroscopicLLMConfig,
    *,
    corpus_path: str,
    extra_args: Sequence[str] | None = None,
) -> list[str]:
    exe = resolve_llama_perplexity_path(cfg)
    gguf = _require_gguf_path(cfg)
    if not os.path.isfile(corpus_path):
        raise FileNotFoundError(f"Perplexity corpus not found: {corpus_path}")

    args: list[str] = [*_llama_engine_prefix(exe, gguf, cfg), "-f", corpus_path]
    if cfg.verbose:
        args.append("-v")
    if extra_args:
        args.extend(extra_args)
    return args

def parse_llama_perplexity_output(stdout: str, stderr: str) -> float | None:
    """Extract perplexity from standard llama-perplexity output."""
    combined = (stdout or "") + "\n" + (stderr or "")

    # 1. Try JSON format first
    for line in combined.splitlines():
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            m = re.search(r'"(?:ppl|perplexity|final_ppl)"\s*:\s*([-+]?\d+(?:\.\d+)?)', s, re.I)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass

    # 2. Standard text format
    patterns = [
        r"Final\s+estimate.*?PPL\s*=\s*([\d.]+)",
        r"^\s*perplexity\s*[:=]\s*([\d.]+)",
        r"PPL\s*=\s*([\d.]+)",
    ]
    for line in combined.splitlines():
        for pattern in patterns:
            m = re.search(pattern, line, re.I)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
    return None

def run_llama_perplexity(
    cfg: GyroscopicLLMConfig,
    *,
    corpus_path: str,
    extra_args: Sequence[str] | None = None,
    env: dict[str, str] | None = None,
    timeout_sec: float | None = 1800.0,
) -> dict[str, object]:
    argv = build_llama_perplexity_command(cfg, corpus_path=corpus_path, extra_args=extra_args)
    cp = _run_command(argv, env=env, timeout_sec=timeout_sec)
    ppl = parse_llama_perplexity_output(cp.stdout or "", cp.stderr or "")
    return {
        "rc": int(cp.returncode),
        "ppl": ppl,
        "stdout": cp.stdout or "",
        "stderr": cp.stderr or "",
        "argv": argv,
    }
