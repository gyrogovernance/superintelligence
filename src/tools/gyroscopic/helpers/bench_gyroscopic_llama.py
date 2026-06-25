"""Gyroscopic LLM benchmark.

Compares stock llama.cpp Q1_0 against the gyroscopic gravity-scale backend.

Usage:
  python -m src.tools.gyroscopic.helpers.bench_gyroscopic_llama
  python -m src.tools.gyroscopic.helpers.bench_gyroscopic_llama --gyro-only --skip-build
  python -m src.tools.gyroscopic.helpers.bench_gyroscopic_llama --diag
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

try:
    from src.tools.gyroscopic.config import (
        GyroscopicLLMConfig,
        get_gyroscopic_llm_config,
        repo_root,
        resolve_llama_cli_path,
    )
    from src.tools.gyroscopic.loader import build_llama_cli_command
    from src.tools.gyroscopic.ops_build import (
        LlamaBuildMode,
        build_llama_cpp_if_needed,
        resolve_llama_cli_out,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from src.tools.gyroscopic.config import (
        GyroscopicLLMConfig,
        get_gyroscopic_llm_config,
        repo_root,
        resolve_llama_cli_path,
    )
    from src.tools.gyroscopic.loader import build_llama_cli_command
    from src.tools.gyroscopic.ops_build import (
        LlamaBuildMode,
        build_llama_cpp_if_needed,
        resolve_llama_cli_out,
    )

# =========================================================================
# Defaults
# =========================================================================

PROMPT_SUITES: dict[str, list[str]] = {
    "smoke": [
        "Tell me about the Sun.",
    ],
    "diverse": [
        "Hello",
        "What is 17 multiplied by 19?",
        "Write a short Python function that reverses a linked list.",
        "Explain gravity-scaled Q1 matmul in one sentence.",
    ],
}

DEFAULT_N_PREDICT = 32
DEFAULT_TOTAL_LAYERS = 36         # Bonsai-8B
DEFAULT_DIAG_MAX_GROUPS = 8192  # full Bonsai-8B Q1 scan is ~64M groups (~days)
TIMEOUT_DEFAULT = 900.0           # cold GGUF load + CPU eval can take minutes
SILENT_KILL_SEC = 900.0          # kill only if NO output for this long (not wall-clock)
BENCH_N_CTX = 512

# CPU-safe sampling flags. --no-context-shift avoids large CPU reallocations.
# Bonsai-8B (Qwen3 base): temp 0.5, top_k 20, top_p 0.85.
LLAMA_EXTRA_ARGS = [
    "--seed", "42", "--temp", "0.5", "--top-p", "0.85", "--top-k", "20",
    "--reasoning", "off", "--single-turn",
    "--n-gpu-layers", "0", "--no-context-shift", "--flash-attn", "off",
    "--perf",
]

# Lines worth echoing live when --live-log (load milestones only).
_LIVE_LOG_RE = re.compile(r"(load_tensors|print_timings)", re.I)

# Env var prefixes scrubbed so a stale shell can't leak tuning into stock runs.
_GYRO_ENV_PREFIXES = ("GGML_GYROSCOPIC", "GYROSCOPIC_", "GYRO_")

DATA_DIR = repo_root() / "data" / "benchmarks" / "gyroscopic_llama"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = DATA_DIR / "bench.json"

# =========================================================================
# Subprocess environment
# =========================================================================


def _clean_llama_env(mode: str) -> dict[str, str]:
    """Hermetic env: scrub stale gyro vars, then set only what this mode needs."""
    env = os.environ.copy()
    for key in list(env):
        if any(key.startswith(p) for p in _GYRO_ENV_PREFIXES):
            env.pop(key, None)
    if mode == "gyroscopic":
        env["GGML_GYROSCOPIC"] = "1"
    else:
        env["GGML_GYROSCOPIC"] = "0"
    return env


def _assert_stock_exe(exe: Path) -> None:
    if "build-stock" not in str(exe.resolve()).replace("\\", "/").lower():
        raise RuntimeError(f"stock bench must use build-stock llama-cli, got: {exe}")


# =========================================================================
# llama-cli perf parsing
# =========================================================================

_THROUGHPUT_BRACKET_RE = re.compile(
    r"\[\s*Prompt:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*t/s\s*\|\s*"
    r"Generation:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*t/s\s*\]"
)
_PERF_PROMPT_MS_RE = re.compile(
    r"(?:^|[^\w])prompt\s+eval\s+time\s*[=:]?\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*"
    r"(?:ms|milliseconds)\s*/\s*(\d+)\s+(?:tokens|runs)\b",
    re.I,
)
_PERF_GEN_MS_RE = re.compile(
    r"(?:^|[^\w])eval\s+time\s*[=:]?\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*"
    r"(?:ms|milliseconds)\s*/\s*(\d+)\s+(?:runs|tokens)\b",
    re.I,
)


@dataclass
class LlamaPerf:
    prompt_tps: float | None = None
    gen_tps: float | None = None
    prompt_eval_ms: float | None = None
    prompt_eval_tokens: int | None = None
    gen_eval_ms: float | None = None
    gen_eval_tokens: int | None = None


def _norm(text: str) -> str:
    if not text:
        return ""
    if text.startswith("\ufeff"):
        text = text[1:]
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_llama_perf(stdout: str, stderr: str) -> LlamaPerf:
    """Extract prompt/generation throughput from llama-cli perf output."""
    combined = _norm(stdout) + "\n" + _norm(stderr)
    perf = LlamaPerf()

    m = _THROUGHPUT_BRACKET_RE.search(combined)
    if m:
        try:
            perf.prompt_tps = float(m.group(1))
            perf.gen_tps = float(m.group(2))
        except ValueError:
            pass

    for line in combined.splitlines():
        mp = _PERF_PROMPT_MS_RE.search(line)
        if mp:
            ms, n = float(mp.group(1)), max(int(mp.group(2)), 1)
            if ms > 0:
                perf.prompt_tps = 1000.0 * n / ms
                perf.prompt_eval_ms = ms
                perf.prompt_eval_tokens = n
        mg = _PERF_GEN_MS_RE.search(line)
        if mg:
            ms, n = float(mg.group(1)), max(int(mg.group(2)), 1)
            if ms > 0:
                perf.gen_tps = 1000.0 * n / ms
                perf.gen_eval_ms = ms
                perf.gen_eval_tokens = n
    return perf


def _extract_generation(stdout_lines: list[str]) -> str:
    collecting, gen = False, []
    for line in stdout_lines:
        s = line.strip()
        if not collecting and s.startswith(">"):
            collecting = True
            continue
        if collecting and (s.startswith("[") or "t/s" in s.lower()):
            break
        if collecting:
            gen.append(s)
    return "\n".join(gen).strip()


# =========================================================================
# Run result
# =========================================================================


@dataclass
class RunResult:
    mode: str
    prompt_idx: int
    prompt: str
    elapsed: float
    timed_out: bool
    silent_kill: bool
    rc: int | None
    prompt_tps: float | None
    gen_tps: float | None
    wall_gen_tps: float | None
    gen_text: str
    gen_hash: str
    stderr_text: str = ""

    @property
    def ok(self) -> bool:
        return not self.timed_out and not self.silent_kill and self.rc == 0

    @property
    def status(self) -> str:
        if self.timed_out:
            return "TIMEOUT"
        if self.silent_kill:
            return "KILLED"
        if self.rc not in (0, None):
            return f"FAIL({self.rc})"
        return "OK"


# =========================================================================
# Streaming runner: wall-clock + silent-kill timeout for cold GGUF loads
# =========================================================================


def _pipe_reader(
    pipe: TextIO,
    out: list[str],
    lock: threading.Lock,
    last_read: list[float],
    *,
    echo: bool,
    prefix: str,
) -> None:
    try:
        for raw in iter(pipe.readline, ""):
            if raw == "":
                break
            s = raw.rstrip("\r\n")
            with lock:
                out.append(s)
            last_read[0] = time.perf_counter()
            if echo and s and _LIVE_LOG_RE.search(s):
                print(f"  [{prefix}] {s}", flush=True)
    except Exception:
        pass


def run_llama(
    mode: str,
    prompt: str,
    n_predict: int,
    timeout: float,
    idx: int,
    *,
    live_log: bool = False,
    verbose: bool = False,
) -> RunResult:
    from dataclasses import replace

    cfg = replace(get_gyroscopic_llm_config(), n_ctx=BENCH_N_CTX)
    backend = "stock" if mode == "stock" else "gyroscopic"
    env = _clean_llama_env(backend if mode != "stock" else "stock")

    exe = resolve_llama_cli_out(mode=backend) or resolve_llama_cli_path(cfg, backend=backend)
    if mode == "stock":
        _assert_stock_exe(exe)
        live_log = False

    args = build_llama_cli_command(
        cfg, prompt=prompt, n_predict=n_predict,
        extra_args=LLAMA_EXTRA_ARGS, backend=backend, llama_cli_exe=exe,
    )

    if verbose:
        print(
            f"[bench] {mode}: n_predict={n_predict} n_ctx={cfg.n_ctx} "
            f"timeout={timeout:.0f}s silent_kill={SILENT_KILL_SEC:.0f}s",
            flush=True,
        )
        print(f"[bench]   exe: {args[0]}", flush=True)
    else:
        print(f"[bench] {mode}...", flush=True)

    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        env=env, stdin=subprocess.DEVNULL,
    )
    if proc.stdout is None or proc.stderr is None:
        proc.kill()
        raise RuntimeError("bench: subprocess must capture stdout and stderr")

    start = time.perf_counter()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    lock = threading.Lock()
    last_read = [start]

    th_out = threading.Thread(
        target=_pipe_reader, args=(proc.stdout, stdout_lines, lock, last_read),
        kwargs={"echo": live_log, "prefix": "out"}, daemon=True,
    )
    th_err = threading.Thread(
        target=_pipe_reader, args=(proc.stderr, stderr_lines, lock, last_read),
        kwargs={"echo": live_log, "prefix": "err"}, daemon=True,
    )
    th_out.start()
    th_err.start()

    timed_out = silent_kill = False
    last_beat = start
    while True:
        elapsed = time.perf_counter() - start
        if proc.poll() is not None:
            break
        if elapsed > timeout:
            timed_out = True
            break
        idle = time.perf_counter() - last_read[0]
        if idle > SILENT_KILL_SEC:
            silent_kill = True
            break
        if live_log and time.perf_counter() - last_beat >= 20.0:
            print(f"[bench]   ... {elapsed:.0f}s, last output {idle:.0f}s ago", flush=True)
            last_beat = time.perf_counter()
        time.sleep(0.05)

    if timed_out or silent_kill:
        try:
            proc.terminate()
            proc.wait(timeout=8)
        except Exception:
            proc.kill()
    try:
        proc.wait(timeout=5)
    except Exception:
        pass
    th_out.join(timeout=12.0)
    th_err.join(timeout=12.0)

    perf = parse_llama_perf("\n".join(stdout_lines), "\n".join(stderr_lines))
    gen_text = _extract_generation(stdout_lines)
    elapsed = time.perf_counter() - start
    wall_gen_tps = None
    if perf.gen_eval_tokens and elapsed > 0:
        wall_gen_tps = perf.gen_eval_tokens / elapsed

    stderr_text = "\n".join(stderr_lines)

    return RunResult(
        mode=mode, prompt_idx=idx, prompt=prompt, elapsed=elapsed,
        timed_out=timed_out, silent_kill=silent_kill, rc=proc.returncode,
        prompt_tps=perf.prompt_tps, gen_tps=perf.gen_tps, wall_gen_tps=wall_gen_tps,
        gen_text=gen_text, gen_hash=hashlib.sha256(gen_text.encode()).hexdigest()[:12],
        stderr_text=stderr_text,
    )


# =========================================================================
# Kernel structure probe
# =========================================================================


def run_kernel_structure() -> dict[str, Any]:
    """Native kernel probe: gravity g1 and per-layer scale table."""
    from src.tools.gyroscopic import ops

    g1 = ops.gravity_g1()
    curve = [
        {"layer": L, "psi": round(L / DEFAULT_TOTAL_LAYERS, 4),
         "scale": round(ops.gravity_scale(L, DEFAULT_TOTAL_LAYERS), 6)}
        for L in range(0, DEFAULT_TOTAL_LAYERS + 1, 6)
    ]

    min_scale = min(
        ops.gravity_scale(L, DEFAULT_TOTAL_LAYERS, k4, sh)
        for L in range(DEFAULT_TOTAL_LAYERS + 1)
        for k4 in range(4)
        for sh in range(7)
    )

    return {
        "gravity_g1": round(g1, 6),
        "attenuation_curve": curve,
        "min_scale_all_groups": round(min_scale, 6),
        "min_scale_positive": min_scale > 0.0,
    }


# =========================================================================
# Diagnostics (--diag): route scan + kernel summary
# =========================================================================


def run_route_diagnostic(*, max_weight_groups: int | None = None) -> dict[str, Any]:
    from src.tools.gyroscopic.config import get_gyroscopic_llm_config, resolve_gguf_path
    from src.tools.gyroscopic.helpers.diagnostics import run_route_diagnostic as _diag

    cfg = get_gyroscopic_llm_config()
    return _diag(gguf_path=resolve_gguf_path(cfg), max_groups=max_weight_groups)


def run_diagnostics(*, max_groups: int | None = DEFAULT_DIAG_MAX_GROUPS) -> dict[str, Any]:
    """Run static GGUF route-structure diagnostic."""
    route = run_route_diagnostic(max_weight_groups=max_groups)
    return {"route": route}


def _print_diagnostics(
    diagnostics: dict[str, Any] | None,
    kernel: dict[str, Any] | None,
    *,
    verbose: bool = False,
) -> None:
    if not diagnostics and not kernel:
        return
    print("\n--- Diagnostics ---\n")

    if kernel:
        print(
            f"  kernel  g1={kernel.get('gravity_g1')}  "
            f"min_scale={kernel.get('min_scale_all_groups')}"
        )
        if verbose:
            for row in kernel.get("attenuation_curve", []):
                print(f"          L={row['layer']:>2} scale={row['scale']:.4f}")

    route = diagnostics.get("route") if diagnostics else None
    if isinstance(route, dict):
        block = route.get("weights")
        if isinstance(block, dict) and block.get("groups", 0) > 0:
            route_cmp = block.get("route_path", {}).get("compare", {})
            print(
                f"  route   groups={block['groups']}  "
                f"max|delta|={route_cmp.get('max_abs_deviation', '--')}"
            )
        elif block and block.get("error"):
            print(f"  route   {block['error']}")

    print()


# =========================================================================
# Report
# =========================================================================


def _fmt_tps(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "--"


def print_report(
    results: list[RunResult],
    kernel: dict[str, Any] | None,
    diagnostics: dict[str, Any] | None = None,
    *,
    verbose: bool = False,
    show_diag: bool = False,
) -> None:
    print("\n" + "=" * 72)
    print("GYROSCOPIC BENCHMARK")
    print("=" * 72)

    if kernel and verbose and not show_diag:
        print("\n--- Kernel structure ---\n")
        print(f"  gravity g1                {kernel['gravity_g1']}")
        print(f"  min scale (all layers)      {kernel['min_scale_all_groups']}")
        print("  depth attenuation exp(g1 * L/N):")
        for row in kernel["attenuation_curve"]:
            bar = "#" * max(1, int(row["scale"] * 40))
            print(f"    L={row['layer']:>2} psi={row['psi']:.2f}  {row['scale']:.4f}  {bar}")

    if results:
        print("\n--- Generation (stock vs gyroscopic) ---\n")
        by_prompt: dict[int, dict[str, RunResult]] = {}
        for r in results:
            by_prompt.setdefault(r.prompt_idx, {})[r.mode] = r
        for idx in sorted(by_prompt):
            runs = by_prompt[idx]
            sample = next(iter(runs.values()))
            print(f"  Prompt {idx + 1}: {sample.prompt!r}")
            print(f"    {'mode':<12} {'status':<10} {'wall_s':>8} {'prompt_tps':>11} {'gen_tps':>9} {'wall_tps':>9}")
            for mode in ("stock", "gyroscopic"):
                r = runs.get(mode)
                if r is None:
                    continue
                print(f"    {mode:<12} {r.status:<10} {r.elapsed:>8.1f} "
                      f"{_fmt_tps(r.prompt_tps):>11} {_fmt_tps(r.gen_tps):>9} {_fmt_tps(r.wall_gen_tps):>9}")
            for mode in ("stock", "gyroscopic"):
                r = runs.get(mode)
                if r is not None and r.gen_text and verbose:
                    snippet = r.gen_text.replace("\n", " ")[:120]
                    print(f"    {mode} text: {snippet}")
            # Speed comparison.
            s, g = runs.get("stock"), runs.get("gyroscopic")
            if s and g and s.gen_tps and g.gen_tps:
                ratio = g.gen_tps / s.gen_tps
                print(f"    gyro/stock gen_tps ratio: {ratio:.3f}")
            print()

    if show_diag:
        _print_diagnostics(diagnostics, kernel, verbose=verbose)

    print("=" * 72)
    print(f"Results written to {OUT_JSON}")
    print("=" * 72 + "\n")


def write_json(
    results: list[RunResult],
    kernel: dict[str, Any] | None,
    meta: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        "kernel_structure": kernel,
        "runs": [{k: v for k, v in asdict(r).items() if k != "stderr_text"} for r in results],
        "diagnostics": diagnostics,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# =========================================================================
# Main
# =========================================================================


def main() -> int:
    p = argparse.ArgumentParser(
        description="Stock vs gyroscopic Q1_0 benchmark; shows kernel structure. "
                    f"Writes {OUT_JSON}.",
    )
    p.add_argument("--stock-only", action="store_true", help="Run only the stock backend.")
    p.add_argument("--gyro-only", action="store_true", help="Run only the gyroscopic backend.")
    p.add_argument("--suite", choices=sorted(PROMPT_SUITES), default="smoke",
                   help="Prompt suite (default: smoke = one prompt).")
    p.add_argument("--n-predict", type=int, default=DEFAULT_N_PREDICT,
                   help=f"Tokens to generate (default {DEFAULT_N_PREDICT}).")
    p.add_argument("--timeout", type=float, default=TIMEOUT_DEFAULT,
                   help=f"Per-run wall-clock timeout in seconds (default {TIMEOUT_DEFAULT:.0f}).")
    p.add_argument("--no-kernel", action="store_true", help="Skip the kernel structure probe.")
    p.add_argument("--skip-build", action="store_true", help="Use existing llama-cli; do not rebuild.")
    p.add_argument("--force-build", action="store_true", help="Rebuild llama-cli even if present.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Verbose bench output (gen text, build log, extra diag detail).")
    p.add_argument("--live-log", action="store_true",
                   help="Stream llama-cli load milestones during gyroscopic runs.")
    p.add_argument("--diag", action="store_true",
                   help="Run diagnostics: kernel summary and Q1 route scan.")
    p.add_argument("--diag-max-groups", type=int, default=DEFAULT_DIAG_MAX_GROUPS,
                   help="Cap GGUF groups scanned with --diag (default "
                        f"{DEFAULT_DIAG_MAX_GROUPS}; use 0 for unlimited).")
    args = p.parse_args()

    run_stock = not args.gyro_only
    run_gyro = not args.stock_only
    run_kernel_probe = not args.no_kernel
    prompts = PROMPT_SUITES[args.suite]
    live_log = args.live_log

    print(f"[bench] {'stock+gyro' if run_stock and run_gyro else ('gyro' if run_gyro else 'stock')} "
          f"| suite={args.suite} ({len(prompts)} prompt) | n_predict={args.n_predict}")

    # Kernel structure probe (fast, native, no subprocess).
    kernel = None
    if run_kernel_probe:
        try:
            kernel = run_kernel_structure()
        except Exception as e:
            print(f"[bench] kernel probe failed: {e}")

    # Build the needed backends.
    built: set[str] = set()

    def ensure(backend: LlamaBuildMode) -> None:
        if args.skip_build or backend in built:
            return
        build_llama_cpp_if_needed(mode=backend, force=args.force_build)
        built.add(backend)

    # Generation runs.
    results: list[RunResult] = []
    for i, prompt in enumerate(prompts):
        if run_stock:
            ensure("stock")
            results.append(run_llama("stock", prompt, args.n_predict, args.timeout, i,
                                     verbose=args.verbose))
        if run_gyro:
            ensure("gyroscopic")
            results.append(run_llama("gyroscopic", prompt, args.n_predict, args.timeout, i,
                                     live_log=live_log, verbose=args.verbose))

    # Static diagnostics (route scan).
    diagnostics = None
    if args.diag:
        max_groups = args.diag_max_groups
        if max_groups is not None and max_groups <= 0:
            max_groups = None
        cap = "all" if max_groups is None else str(max_groups)
        print(f"[bench] diagnostics (max={cap})...", flush=True)
        diagnostics = run_diagnostics(max_groups=max_groups)

    meta = {
        "suite": args.suite,
        "n_predict": args.n_predict,
        "timeout": args.timeout,
        "run_stock": run_stock,
        "run_gyro": run_gyro,
        "total_layers": DEFAULT_TOTAL_LAYERS,
        "diag": args.diag,
        "diag_max_groups": args.diag_max_groups,
    }
    write_json(results, kernel, meta, diagnostics)
    print_report(
        results, kernel, diagnostics,
        verbose=args.verbose, show_diag=args.diag,
    )

    # Exit non-zero if any requested run failed.
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
