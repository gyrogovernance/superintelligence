"""
Ablation: sweep GGML_GYROSCOPIC_M2_WHT and run a short llama-cli generation.

Usage (repo root):

    python scripts/analyze_m2_wht.py

Or:

    python -m src.tools.gyroscopic.helpers.analyze_m2_wht

Requires a built gyroscopic llama-cli (same as bench_gyroscopic_llama).
Parses GyroMatMul trace lines from stderr for wht_bypass and graph_m2.

Does not pick an optimal threshold automatically; inspect printed stderr summaries.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.tools.gyroscopic.config import get_gyroscopic_llm_config
from src.tools.gyroscopic.loader import build_llama_cli_command
from src.tools.gyroscopic.ops_build import build_llama_cpp_if_needed

THRESHOLDS = (64, 128, 256, 384, 512)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep GGML_GYROSCOPIC_M2_WHT and print GyroMatMul trace lines")
    ap.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        metavar="SEC",
        help="Per-run wall limit for llama-cli (0 = no limit). Default 1800. Large Q8_0 CPU runs can exceed 300s.",
    )
    args = ap.parse_args()
    timeout_sec: float | None = None if args.timeout <= 0 else args.timeout

    build_llama_cpp_if_needed()
    cfg = get_gyroscopic_llm_config()
    argv = build_llama_cli_command(
        cfg,
        prompt="Hello",
        n_predict=16,
        extra_args=["--seed", "123", "--temp", "0", "--top-k", "1", "--top-p", "1.0"],
    )
    base = {
        "GGML_GYROSCOPIC": "1",
        "GGML_GYROSCOPIC_TRACE": "1",
        "GGML_GYROSCOPIC_STRICT": "1",
        "GGML_GYROSCOPIC_KERNEL": "avx2",
    }
    print("M2_WHT sweep (stderr excerpts):\n")
    if timeout_sec is not None:
        print(f"(per-run timeout {timeout_sec:.0f}s; use --timeout 0 for no limit)\n")
    had_timeout = False
    for thr in THRESHOLDS:
        env = dict(os.environ)
        env.update(base)
        env["GGML_GYROSCOPIC_M2_WHT"] = str(thr)
        try:
            p = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                env=env,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as e:
            had_timeout = True
            lim = f"{timeout_sec:.0f}s" if timeout_sec is not None else "none"
            print(f"--- M2_WHT={thr} TIMEOUT (limit {lim}) ---", file=sys.stderr)
            tail = ""
            err_bytes = getattr(e, "stderr", None)
            if err_bytes:
                tail = err_bytes[-4000:] if len(err_bytes) > 4000 else err_bytes
            elif getattr(e, "output", None):
                tail = str(e.output)[-4000:]
            if tail:
                print("Last stderr/output tail:", file=sys.stderr)
                print(tail, file=sys.stderr)
            print()
            continue
        err = p.stderr or ""
        print(f"--- M2_WHT={thr} rc={p.returncode} ---")
        for line in err.splitlines():
            if line.startswith("GyroMatMul"):
                print(line)
        print()
    return 1 if had_timeout else 0


if __name__ == "__main__":
    raise SystemExit(main())
