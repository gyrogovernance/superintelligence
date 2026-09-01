"""Gyroscopic LLM benchmark  stock vs gyroscopic backend at KV-relevant scale.

Measures generation throughput (ms/token), prompt/decode timing, and confirms
the gyroscopic backend (ggml-gyroscopic copy + kernel.c linked) builds and
runs Bonsai generation identically to the stock (vanilla ggml-cpu) backend.

The retired additive-blend hook (GYROSCOPIC_INJECT / gyro_carrier) is gone.
Current integration is a separate, contract-correct design (analysis_NavPAD
7.2 / Runtime Spec 17) and is NOT exercised here. This bench only
proves the two backends are at parity for the forward pass.

  stock      = _build/llama-cpp-stock (vanilla ggml-cpu)
  gyroscopic = _build/llama-cpp (ggml-gyroscopic + kernel.c, no blend hook)

Usage:
  python -m src.tools.gyroscopic.helpers.bench_gyroscopic_llama
  python -m src.tools.gyroscopic.helpers.bench_gyroscopic_llama --suite smoke --n-ctx 512
  python -m src.tools.gyroscopic.helpers.bench_gyroscopic_llama --gyro-only --skip-build
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
from dataclasses import asdict, dataclass, replace
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
    from src.tools.gyroscopic.build import (
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
    from src.tools.gyroscopic.build import (
        LlamaBuildMode,
        build_llama_cpp_if_needed,
        resolve_llama_cli_out,
    )

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_N_CTX = 4096
DEFAULT_N_PREDICT = 128
TIMEOUT_DEFAULT = 1800.0
SILENT_KILL_SEC = 900.0

DATA_DIR = repo_root() / "data" / "benchmarks" / "gyroscopic_llama"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = DATA_DIR / "bench.json"

LLAMA_EXTRA_ARGS = [
    "--seed", "42", "--temp", "0.0", "--top-p", "0.85", "--top-k", "20",
    "--single-turn",
    "--n-gpu-layers", "0", "--no-context-shift", "--flash-attn", "on",
    "--perf",
]

_KV_PREFILL_PARA = (
    "The Sun is a G-type main-sequence star at the center of the solar system. "
    "Nuclear fusion in its core converts hydrogen into helium and releases the "
    "radiation that powers climate and life on Earth. "
)

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
_MEMORY_BREAKDOWN_RE = re.compile(
    r"memory breakdown\s*\[MiB\].*?=\s*(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)",
    re.I,
)


@dataclass(frozen=True)
class BenchCase:
    name: str
    prompt: str | None = None
    prompt_file: Path | None = None
    description: str = ""


def _suite_table() -> dict[str, BenchCase]:
    return {
        "smoke": BenchCase(
            name="smoke",
            prompt="The quantum algorithm computes",
            description="Quick sanity (short context; hook path exercised).",
        ),
        "scale": BenchCase(
            name="scale",
            prompt_file=_ensure_kv_prefill(),
            description="Long prefill file + sustained decode (KV/at-scale path).",
        ),
    }


def _ensure_kv_prefill(*, target_chars: int = 14_000) -> Path:
    path = DATA_DIR / "kv_prefill.txt"
    if not path.is_file() or path.stat().st_size < target_chars // 2:
        n = max(1, target_chars // len(_KV_PREFILL_PARA))
        path.write_text((_KV_PREFILL_PARA * n)[:target_chars], encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def _clean_llama_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("GYROSCOPIC_") or key.startswith("GGML_GYROSCOPIC"):
            env.pop(key, None)
    return env


def _gyro_env() -> dict[str, str]:
    env = _clean_llama_env()
    env["GGML_GYROSCOPIC"] = "1"
    return env


def _assert_gyro_exe(exe: Path) -> None:
    path = str(exe.resolve()).replace("\\", "/").lower()
    if "llama-cpp-stock" in path or path.endswith("/build-stock/") or "/build-stock/" in path:
        raise RuntimeError(f"gyroscopic bench must not use stock llama-cli, got: {exe}")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


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


def parse_memory_mib(stderr: str) -> dict[str, int] | None:
    m = _MEMORY_BREAKDOWN_RE.search(_norm(stderr))
    if not m:
        return None
    total, model, context, compute = (int(m.group(i)) for i in range(1, 5))
    return {"total": total, "model": model, "context": context, "compute": compute}


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


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    mode: str
    case: str
    prompt_label: str
    n_ctx: int
    elapsed: float
    timed_out: bool
    silent_kill: bool
    rc: int | None
    prompt_tps: float | None
    gen_tps: float | None
    ms_per_gen_token: float | None
    prompt_tokens: int | None
    gen_tokens: int | None
    memory_mib: dict[str, int] | None
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


def _build_argv(
    cfg: GyroscopicLLMConfig,
    exe: Path,
    case: BenchCase,
    n_predict: int,
) -> list[str]:
    from src.tools.gyroscopic.loader import _llama_engine_prefix, _require_gguf_path

    gguf = _require_gguf_path(cfg)
    args: list[str] = [*_llama_engine_prefix(exe, gguf, cfg), "-n", str(n_predict)]
    if case.prompt_file is not None:
        args.extend(["-f", str(case.prompt_file)])
    else:
        args.extend(["-p", case.prompt or "."])
    args.append("--no-display-prompt")
    args.extend(LLAMA_EXTRA_ARGS)
    return args


def _pipe_reader(pipe, out, lock, last_read, *, echo, prefix):
    try:
        for raw in iter(pipe.readline, ""):
            if raw == "":
                break
            s = raw.rstrip("\r\n")
            with lock:
                out.append(s)
            last_read[0] = time.perf_counter()
            if echo and s and ("load_tensors" in s or "print_timings" in s):
                print(f"  [{prefix}] {s}", flush=True)
    except Exception:
        pass


def run_llama(
    mode: str,
    case: BenchCase,
    n_ctx: int,
    n_predict: int,
    timeout: float,
    *,
    live_log: bool = False,
    verbose: bool = False,
) -> RunResult:
    cfg = replace(get_gyroscopic_llm_config(), n_ctx=n_ctx)
    backend = "stock" if mode == "stock" else "gyroscopic"
    env = _clean_llama_env() if mode == "stock" else _gyro_env()

    exe = resolve_llama_cli_out(mode=backend) or resolve_llama_cli_path(cfg, backend=backend)
    if mode == "gyroscopic":
        _assert_gyro_exe(exe)

    label = case.prompt if case.prompt else str(case.prompt_file)
    args = _build_argv(cfg, exe, case, n_predict)

    if verbose:
        print(f"[bench] {mode} case={case.name} n_ctx={n_ctx} "
              f"n_predict={n_predict} timeout={timeout:.0f}s", flush=True)
        print(f"[bench]   exe: {args[0]}", flush=True)
    else:
        print(f"[bench] {mode} ({case.name})...", flush=True)

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
        kwargs={"echo": live_log, "prefix": "out"}, daemon=True)
    th_err = threading.Thread(
        target=_pipe_reader, args=(proc.stderr, stderr_lines, lock, last_read),
        kwargs={"echo": live_log, "prefix": "err"}, daemon=True)
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
        silent_limit = max(SILENT_KILL_SEC, timeout)
        if idle > silent_limit:
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

    stdout = "\n".join(stdout_lines)
    stderr_text = "\n".join(stderr_lines)
    perf = parse_llama_perf(stdout, stderr_text)
    gen_text = _extract_generation(stdout_lines)
    elapsed = time.perf_counter() - start

    ms_per_gen = None
    if perf.gen_eval_ms and perf.gen_eval_tokens:
        ms_per_gen = perf.gen_eval_ms / perf.gen_eval_tokens
    elif perf.gen_tps and perf.gen_tps > 0:
        ms_per_gen = 1000.0 / perf.gen_tps

    return RunResult(
        mode=mode,
        case=case.name,
        prompt_label=label or "",
        n_ctx=n_ctx,
        elapsed=elapsed,
        timed_out=timed_out,
        silent_kill=silent_kill,
        rc=proc.returncode,
        prompt_tps=perf.prompt_tps,
        gen_tps=perf.gen_tps,
        ms_per_gen_token=ms_per_gen,
        prompt_tokens=perf.prompt_eval_tokens,
        gen_tokens=perf.gen_eval_tokens,
        memory_mib=parse_memory_mib(stderr_text),
        gen_text=gen_text,
        gen_hash=hashlib.sha256(gen_text.encode()).hexdigest()[:12],
        stderr_text=stderr_text,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(v, *, prec=1, suffix=""):
    if v is None:
        return "--"
    return f"{v:.{prec}f}{suffix}"


def print_report(results, meta, *, verbose):
    print("\n" + "=" * 5)
    print("GYROSCOPIC BENCHMARK")
    print("=" * 5)
    print(f"  suite={meta['suite']}  n_ctx={meta['n_ctx']}  n_predict={meta['n_predict']}  "
          f"flash_attn=on (both modes)")

    if not results:
        print("  (no runs)")
    else:
        print("\n--- Throughput (stock vs gyroscopic) ---\n")
        by_case: dict[str, list[RunResult]] = {}
        for r in results:
            by_case.setdefault(r.case, []).append(r)

        for case_name, runs in by_case.items():
            sample = runs[0]
            print(f"  Case: {case_name} - {sample.prompt_label[:72]}...")
            print(f"    {'mode':<12} {'status':<8} {'wall_s':>7} "
                  f"{'ms/tok':>8} {'gen_tps':>8}")
            for r in runs:
                print(f"    {r.mode:<12} {r.status:<8} "
                      f"{r.elapsed:>7.1f} {_fmt(r.ms_per_gen_token, prec=2):>8} {_fmt(r.gen_tps):>8}")
            stock_runs = [r for r in runs if r.mode == "stock"]
            gyro_runs = [r for r in runs if r.mode == "gyroscopic"]
            if stock_runs and gyro_runs:
                s = stock_runs[0]
                for g in gyro_runs:
                    if s.ms_per_gen_token and g.ms_per_gen_token:
                        ratio = s.ms_per_gen_token / g.ms_per_gen_token
                        print(f"    gyro/stock decode speed ratio: {ratio:.3f}x")
            if verbose:
                for r in runs:
                    if r.gen_text:
                        snip = r.gen_text.replace("\n", " ")[:120]
                        print(f"    {r.mode} text: {snip}")
            print()

    _print_plain_summary(results, meta)

    print("=" * 5)
    print(f"Results: {OUT_JSON}")
    print("=" * 5 + "\n")


def _print_plain_summary(results, meta):
    print("\n--- What this means ---")
    print("  Stock = vanilla ggml-cpu (_build/llama-cpp-stock).")
    print("  Gyroscopic = ggml-gyroscopic backend (_build/llama-cpp; kernel.c linked, no blend hook).")
    print("  Parity check: both run the same forward pass; generation + speed must match.")
    stock = next((r for r in results if r.mode == "stock" and r.ok), None)
    gyro = next((r for r in results if r.mode == "gyroscopic" and r.ok), None)
    if stock and gyro:
        if stock.gen_hash == gyro.gen_hash:
            print(f"  generation HASH MATCHES stock (backends at parity).")
        else:
            print(f"  generation HASH DIFFERS from stock "
                  f"({stock.gen_hash} vs {gyro.gen_hash}) -- investigate backend divergence.")
        if stock.gen_tps and gyro.gen_tps:
            pct = 100.0 * gyro.gen_tps / stock.gen_tps
            print(f"  decode speed: gyroscopic is {pct:.0f}% of stock "
                  f"({gyro.gen_tps:.1f} vs {stock.gen_tps:.1f} tok/s).")


def write_json(results, meta):
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        "runs": [{k: v for k, v in asdict(r).items() if k != "stderr_text"} for r in results],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    suites = _suite_table()
    p = argparse.ArgumentParser(
        description="Stock vs gyroscopic-backend benchmark (parity check).",
    )
    p.add_argument("--stock-only", action="store_true")
    p.add_argument("--gyro-only", action="store_true")
    p.add_argument("--suite", choices=sorted(suites), default="scale")
    p.add_argument("--n-ctx", type=int, default=DEFAULT_N_CTX)
    p.add_argument("--n-predict", type=int, default=DEFAULT_N_PREDICT)
    p.add_argument("--timeout", type=float, default=TIMEOUT_DEFAULT)
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--force-build", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--live-log", action="store_true")
    args = p.parse_args()

    run_stock = not args.gyro_only
    run_gyro = not args.stock_only
    case = suites[args.suite]

    print(f"[bench] {'stock+gyro' if run_stock and run_gyro else ('gyro' if run_gyro else 'stock')} "
          f"| {case.description} | n_ctx={args.n_ctx} n_predict={args.n_predict}", flush=True)

    built: set[str] = set()

    def ensure(backend: LlamaBuildMode) -> None:
        if args.skip_build or backend in built:
            return
        build_llama_cpp_if_needed(mode=backend, force=args.force_build)
        built.add(backend)

    results: list[RunResult] = []
    if run_stock:
        ensure("stock")
        results.append(run_llama(
            "stock", case, args.n_ctx, args.n_predict, args.timeout, verbose=args.verbose))
    if run_gyro:
        ensure("gyroscopic")
        results.append(run_llama(
            "gyroscopic", case, args.n_ctx, args.n_predict, args.timeout,
            live_log=args.live_log, verbose=args.verbose))

    meta = {
        "suite": args.suite,
        "suite_description": case.description,
        "n_ctx": args.n_ctx,
        "n_predict": args.n_predict,
        "timeout": args.timeout,
        "run_stock": run_stock,
        "run_gyro": run_gyro,
        "flash_attn": "on",
        "hook": "none (retired GYROSCOPIC_INJECT blend); backends compared for parity",
    }
    write_json(results, meta)
    print_report(results, meta, verbose=args.verbose)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
