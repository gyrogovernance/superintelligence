"""Run Bonsai-8B-Q1_0 with gyroscopic llama.cpp (ledger + holonomic KV by default).

Examples:
  python -m src.tools.gyroscopic.helpers.run_bonsai
  python -m src.tools.gyroscopic.helpers.run_bonsai --incomplete-forward
  python -m src.tools.gyroscopic.helpers.run_bonsai -p "Hello" -n 64
  python -m src.tools.gyroscopic.helpers.run_bonsai --verify
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

try:
    from src.tools.gyroscopic.config import (
        get_gyroscopic_llm_config,
        production_gyroscopic_env,
        resolve_llama_cli_path,
    )
    from src.tools.gyroscopic.loader import _llama_engine_prefix
    from src.tools.gyroscopic.build import build_llama_cpp_if_needed
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from src.tools.gyroscopic.config import (
        get_gyroscopic_llm_config,
        production_gyroscopic_env,
        resolve_llama_cli_path,
    )
    from src.tools.gyroscopic.loader import _llama_engine_prefix
    from src.tools.gyroscopic.build import build_llama_cpp_if_needed

from src.tools.gyroscopic.loader import _require_gguf_path


def _chat_argv(
    exe: Path,
    gguf: Path,
    cfg,
    *,
    prompt: str,
    n_predict: int,
) -> list[str]:
    return [
        *_llama_engine_prefix(exe, gguf, cfg),
        "-n",
        str(n_predict),
        "-p",
        prompt,
        "--no-display-prompt",
        "--seed",
        "42",
        "--temp",
        "0.5",
        "--top-p",
        "0.85",
        "--top-k",
        "20",
        "--reasoning",
        "off",
        "--single-turn",
        "--n-gpu-layers",
        "0",
        "--no-context-shift",
        "--flash-attn",
        "on",
    ]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Chat with Bonsai-8B using gyroscopic ledger llama.cpp.",
    )
    p.add_argument("-p", "--prompt", default="Tell me about the Sun.")
    p.add_argument("-n", "--n-predict", type=int, default=128)
    p.add_argument("-c", "--n-ctx", type=int, default=None)
    p.add_argument("--stats", action="store_true", help="Verbose ledger displace logs.")
    p.add_argument(
        "--incomplete-forward",
        action="store_true",
        help="Stress unfinished forward-site flags (NavPad §8). Not a product mode.",
    )
    p.add_argument("--skip-build", action="store_true")
    p.add_argument(
        "--verify",
        action="store_true",
        help="After build, run smoke benchmark (stock vs gyro) and exit.",
    )
    args = p.parse_args()

    if not args.skip_build:
        build_llama_cpp_if_needed(mode="gyroscopic")

    if args.verify:
        cmd = [
            sys.executable,
            "-m",
            "src.tools.gyroscopic.helpers.bench_gyroscopic_llama",
            "--suite",
            "smoke",
            "--n-ctx",
            "512",
            "--n-predict",
            "32",
            "--skip-build",
        ]
        return int(subprocess.run(cmd).returncode or 0)

    cfg = get_gyroscopic_llm_config()
    if args.n_ctx is not None:
        cfg = replace(cfg, n_ctx=args.n_ctx)
    gguf = _require_gguf_path(cfg)
    exe = resolve_llama_cli_path(cfg, backend="gyroscopic")

    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("GGML_GYROSCOPIC", "GYROSCOPIC_", "GYRO_")):
            env.pop(key, None)
    env.update(
        production_gyroscopic_env(
            stats=args.stats,
            holonomic_kv=True,
            incomplete_forward=bool(args.incomplete_forward),
        )
    )
    if "GYRO_LEDGER_PATH" not in env:
        print("[bonsai] ledger missing after ensure — abort", flush=True)
        return 1

    argv = _chat_argv(exe, gguf, cfg, prompt=args.prompt, n_predict=args.n_predict)
    print("[bonsai] gyroscopic chat", flush=True)
    print(f"[bonsai] model: {gguf.name}", flush=True)
    print(f"[bonsai] ledger: {env['GYRO_LEDGER_PATH']}", flush=True)
    proc = subprocess.run(argv, env=env)
    return int(proc.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
