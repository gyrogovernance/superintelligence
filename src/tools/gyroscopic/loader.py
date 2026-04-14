"""Run the native ``llama-cli`` (llama.cpp C backend) from :class:`GyroscopicLLMConfig`."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence

from .config import GyroscopicLLMConfig, resolve_gguf_path, resolve_llama_cli_path


def build_llama_cli_command(
    cfg: GyroscopicLLMConfig,
    *,
    prompt: str,
    n_predict: int,
    extra_args: Sequence[str] | None = None,
) -> list[str]:
    """Build argv for ``llama-cli`` (model path, ctx, threads, GPU offload, prompt)."""
    exe = resolve_llama_cli_path(cfg)
    gguf = resolve_gguf_path(cfg)
    if not gguf.is_file():
        raise FileNotFoundError(f"GGUF not found: {gguf}")

    threads = cfg.n_threads
    if threads is None:
        threads = max(1, (os.cpu_count() or 4))

    args: list[str] = [
        str(exe),
        "-m",
        str(gguf),
        "-c",
        str(cfg.n_ctx),
        "-t",
        str(threads),
        "-ngl",
        str(cfg.n_gpu_layers),
        "-n",
        str(n_predict),
        "-p",
        prompt,
        "--no-display-prompt",
        # Without this, llama-cli clears -p after one turn and blocks on readline (hangs under pytest).
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
) -> subprocess.CompletedProcess[str]:
    """Run ``llama-cli`` once; returns completed process (stdout/stderr captured)."""
    argv = build_llama_cli_command(
        cfg, prompt=prompt, n_predict=n_predict, extra_args=extra_args
    )
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=timeout_sec,
        check=False,
    )


def run_llama_cli_smoke(
    cfg: GyroscopicLLMConfig,
    *,
    timeout_sec: float | None = 900.0,
) -> subprocess.CompletedProcess[str]:
    """Minimal generation to verify the GGUF loads under the C backend."""
    return run_llama_cli(
        cfg,
        prompt=".",
        n_predict=1,
        timeout_sec=timeout_sec,
    )


def run_llama_cli_version(cfg: GyroscopicLLMConfig) -> subprocess.CompletedProcess[str]:
    """Run ``llama-cli --version`` (no model load)."""
    exe = resolve_llama_cli_path(cfg)
    return subprocess.run(
        [str(exe), "--version"],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
