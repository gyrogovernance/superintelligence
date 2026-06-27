"""Gyroscopic LLM benchmark — stock vs gyroscopic at KV-relevant scale.

Measures generation throughput (ms/token), host memory breakdown, and
gyroscopic KV-chirality path counters.  Default suite prefills a long context
then decodes many tokens so attention/KV work dominates.

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
from dataclasses import asdict, dataclass, field
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
    from src.tools.gyroscopic.ops_build import (
        LlamaBuildMode,
        build_llama_cpp_if_needed,
        resolve_llama_cli_out,
    )

# ---------------------------------------------------------------------------
# Defaults — scale regime (KV + attention matter; not a one-line smoke)
# ---------------------------------------------------------------------------

DEFAULT_N_CTX = 4096
DEFAULT_N_PREDICT = 128
DEFAULT_TOTAL_LAYERS = 36
TIMEOUT_DEFAULT = 1800.0
SILENT_KILL_SEC = 900.0

DATA_DIR = repo_root() / "data" / "benchmarks" / "gyroscopic_llama"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = DATA_DIR / "bench.json"
KV_PREFILL_PATH = DATA_DIR / "kv_prefill.txt"

_KV_PREFILL_PARA = (
    "The Sun is a G-type main-sequence star at the center of the solar system. "
    "Nuclear fusion in its core converts hydrogen into helium and releases the "
    "radiation that powers climate and life on Earth. "
)

LLAMA_EXTRA_ARGS = [
    "--seed", "42", "--temp", "0.5", "--top-p", "0.85", "--top-k", "20",
    "--reasoning", "off", "--single-turn",
    "--n-gpu-layers", "0", "--no-context-shift", "--flash-attn", "on",
    "--perf",
]

_GYRO_ENV_PREFIXES = ("GGML_GYROSCOPIC", "GYROSCOPIC_", "GYRO_")
_LIVE_LOG_RE = re.compile(r"(load_tensors|print_timings)", re.I)

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
_KV_CHI_STATS_RE = re.compile(r"gyro_kv_chi_stats:\s*(\{[^}]+\})")
_HYBRID_STATS_RE = re.compile(r"gyro_hybrid_stats:\s*(\{[^}]+\})")
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
            prompt="Tell me about the Sun.",
            description="Quick sanity (short context; KV not stressed).",
        ),
        "scale": BenchCase(
            name="scale",
            prompt_file=_ensure_kv_prefill(),
            description="Long prefill file + sustained decode (KV/at-scale path).",
        ),
    }


def _ensure_kv_prefill(*, target_chars: int = 14_000) -> Path:
    """~3k+ tokens of prefill text (rule-of-thumb ~4 chars/token)."""
    if not KV_PREFILL_PATH.is_file() or KV_PREFILL_PATH.stat().st_size < target_chars // 2:
        n = max(1, target_chars // len(_KV_PREFILL_PARA))
        KV_PREFILL_PATH.write_text((_KV_PREFILL_PARA * n)[:target_chars], encoding="utf-8")
    return KV_PREFILL_PATH


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def _clean_llama_env(mode: str) -> dict[str, str]:
    from src.tools.gyroscopic.config import production_gyroscopic_env

    env = os.environ.copy()
    for key in list(env):
        if any(key.startswith(p) for p in _GYRO_ENV_PREFIXES):
            env.pop(key, None)
    if mode == "gyroscopic":
        env.update(production_gyroscopic_env(stats=True))
    else:
        env["GGML_GYROSCOPIC"] = "0"
    return env


def _assert_stock_exe(exe: Path) -> None:
    if "build-stock" not in str(exe.resolve()).replace("\\", "/").lower():
        raise RuntimeError(f"stock bench must use build-stock llama-cli, got: {exe}")


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


def parse_kv_chi_stats(stdout: str, stderr: str, *, gen_tokens: int | None) -> dict[str, Any] | None:
    combined = _norm(stdout) + "\n" + _norm(stderr)
    m = _KV_CHI_STATS_RE.search(combined)
    if not m:
        return None
    try:
        raw = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None

    checks = int(raw.get("attn_checks") or 0)
    filtered = int(raw.get("attn_filtered") or 0)
    empty_bypass = int(raw.get("attn_empty_bypass") or 0)
    kv_writes = int(raw.get("kv_writes") or 0)
    active = max(checks - empty_bypass, 0)

    out: dict[str, Any] = {
        "kv_writes": kv_writes,
        "q_wht_rows": int(raw.get("q_wht_rows") or 0),
        "attn_checks": checks,
        "attn_filtered": filtered,
        "attn_empty_bypass": empty_bypass,
        "dmax": int(raw.get("dmax") or 0),
        "m2": float(raw.get("m2") or 0.0),
        "eta": float(raw.get("eta") or 0.0),
        "index": int(raw.get("index") or 0),
        "index_builds": int(raw.get("index_builds") or 0),
        "index_keep": int(raw.get("index_keep") or 0),
        "index_skip": int(raw.get("index_skip") or 0),
        "filter_rate": round(filtered / checks, 4) if checks else None,
        "empty_bypass_rate": round(empty_bypass / checks, 4) if checks else None,
        "active_filter_rate": round(filtered / active, 4) if active else None,
    }
    index_keep = out["index_keep"]
    index_skip = out["index_skip"]
    denom = index_keep + index_skip
    if denom > 0:
        out["index_skip_rate"] = round(index_skip / denom, 4)
    if gen_tokens and gen_tokens > 0:
        out["kv_writes_per_gen_token"] = round(kv_writes / gen_tokens, 2)
        out["attn_checks_per_gen_token"] = round(checks / gen_tokens, 1)
    if kv_writes > 0:
        out["attn_checks_per_kv_write"] = round(checks / kv_writes, 1)
    return out


def parse_hybrid_stats(stdout: str, stderr: str) -> dict[str, Any] | None:
    combined = _norm(stdout) + "\n" + _norm(stderr)
    m = _HYBRID_STATS_RE.search(combined)
    if not m:
        return None
    try:
        raw = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    builds = int(raw.get("tile_builds") or 0)
    hits = int(raw.get("cache_hits") or 0)
    misses = int(raw.get("cache_misses") or 0)
    preload_tensors = int(raw.get("preload_tensors") or 0)
    preload_tiles = int(raw.get("preload_tiles") or 0)
    preload_hits = int(raw.get("preload_hits") or 0)
    lookups = hits + misses
    return {
        "tile_builds": builds,
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_hit_rate": round(hits / lookups, 4) if lookups else None,
        "preload_tensors": preload_tensors,
        "preload_tiles": preload_tiles,
        "preload_hits": preload_hits,
    }


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
    kv_chi_stats: dict[str, Any] | None = None
    hybrid_stats: dict[str, Any] | None = None
    stderr_text: str = field(default="", repr=False)

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
    case: BenchCase,
    n_ctx: int,
    n_predict: int,
    timeout: float,
    *,
    live_log: bool = False,
    verbose: bool = False,
) -> RunResult:
    from dataclasses import replace

    cfg = replace(get_gyroscopic_llm_config(), n_ctx=n_ctx)
    backend = "stock" if mode == "stock" else "gyroscopic"
    env = _clean_llama_env(backend)

    exe = resolve_llama_cli_out(mode=backend) or resolve_llama_cli_path(cfg, backend=backend)
    if mode == "stock":
        _assert_stock_exe(exe)
        live_log = False

    label = case.prompt if case.prompt else str(case.prompt_file)
    args = _build_argv(cfg, exe, case, n_predict)

    if verbose:
        print(
            f"[bench] {mode} case={case.name} n_ctx={n_ctx} n_predict={n_predict} "
            f"timeout={timeout:.0f}s",
            flush=True,
        )
        print(f"[bench]   exe: {args[0]}", flush=True)
        print(f"[bench]   input: {label}", flush=True)
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

    kv_chi_stats = None
    hybrid_stats = None
    if mode == "gyroscopic":
        kv_chi_stats = parse_kv_chi_stats(stdout, stderr_text, gen_tokens=perf.gen_eval_tokens)
        hybrid_stats = parse_hybrid_stats(stdout, stderr_text)

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
        kv_chi_stats=kv_chi_stats,
        hybrid_stats=hybrid_stats,
        stderr_text=stderr_text,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(v: float | None, *, prec: int = 1, suffix: str = "") -> str:
    if v is None:
        return "--"
    return f"{v:.{prec}f}{suffix}"


def _fmt_rate(v: float | None) -> str:
    return f"{v:.1%}" if v is not None else "--"


def _print_kv_path(r: RunResult) -> None:
    mem = r.memory_mib
    if mem:
        print(
            f"    memory  total={mem['total']} MiB  model={mem['model']}  "
            f"context={mem['context']}  compute={mem['compute']}"
        )
    print(
        f"    prefill {r.prompt_tokens or '--'} tok @ {_fmt(r.prompt_tps)} t/s  |  "
        f"decode {r.gen_tokens or '--'} tok @ {_fmt(r.ms_per_gen_token, prec=2)} ms/tok "
        f"({_fmt(r.gen_tps)} t/s)"
    )
    stats = r.kv_chi_stats
    if stats is None:
        print("    kv_chi  (no stats — rebuild gyroscopic backend)")
        return
    print(
        f"    kv_chi  writes={stats['kv_writes']}  "
        f"writes/gen_tok={stats.get('kv_writes_per_gen_token', '--')}  "
        f"checks/gen_tok={stats.get('attn_checks_per_gen_token', '--')}  "
        f"m2={stats.get('m2', '--')}  eta={stats.get('eta', '--')}"
    )
    print(
        f"            filtered={stats['attn_filtered']}  "
        f"filter_rate={_fmt_rate(stats.get('filter_rate'))}  "
        f"empty_bypass={_fmt_rate(stats.get('empty_bypass_rate'))}"
    )
    if stats.get("index"):
        print(
            f"            index  builds={stats.get('index_builds', '--')}  "
            f"skip_rate={_fmt_rate(stats.get('index_skip_rate'))}"
        )
    hybrid = r.hybrid_stats
    if hybrid is not None:
        print(
            f"    hybrid  tile_builds={hybrid['tile_builds']}  "
            f"preload_tiles={hybrid.get('preload_tiles', '--')}  "
            f"cache_hit={_fmt_rate(hybrid.get('cache_hit_rate'))}  "
            f"preload_hits={hybrid.get('preload_hits', '--')}"
        )


def print_report(results: list[RunResult], meta: dict[str, Any], *, verbose: bool) -> None:
    print("\n" + "=" * 72)
    print("GYROSCOPIC BENCHMARK")
    print("=" * 72)
    print(
        f"  suite={meta['suite']}  n_ctx={meta['n_ctx']}  n_predict={meta['n_predict']}  "
        f"flash_attn=on (both modes)"
    )

    if not results:
        print("  (no runs)")
    else:
        print("\n--- Throughput (stock vs gyroscopic) ---\n")
        by_case: dict[str, dict[str, RunResult]] = {}
        for r in results:
            by_case.setdefault(r.case, {})[r.mode] = r

        for case_name, runs in by_case.items():
            sample = next(iter(runs.values()))
            print(f"  Case: {case_name} - {sample.prompt_label[:72]}...")
            print(f"    {'mode':<12} {'status':<8} {'wall_s':>7} {'ms/tok':>8} {'gen_tps':>8}")
            for mode in ("stock", "gyroscopic"):
                r = runs.get(mode)
                if r is None:
                    continue
                print(
                    f"    {mode:<12} {r.status:<8} {r.elapsed:>7.1f} "
                    f"{_fmt(r.ms_per_gen_token, prec=2):>8} {_fmt(r.gen_tps):>8}"
                )
            s, g = runs.get("stock"), runs.get("gyroscopic")
            if s and g and s.ms_per_gen_token and g.ms_per_gen_token:
                ratio = s.ms_per_gen_token / g.ms_per_gen_token
                print(f"    gyro/stock decode speed ratio: {ratio:.3f}x")
            if g is not None:
                print()
                _print_kv_path(g)
            if verbose:
                for mode in ("stock", "gyroscopic"):
                    r = runs.get(mode)
                    if r and r.gen_text:
                        snip = r.gen_text.replace("\n", " ")[:100]
                        print(f"    {mode} text: {snip}")
            print()

    _print_plain_summary(results, meta)

    print("=" * 72)
    print(f"Results: {OUT_JSON}")
    print("=" * 72 + "\n")


def _print_plain_summary(results: list[RunResult], meta: dict[str, Any]) -> None:
    print("\n--- What this means ---")
    print("  Stock = normal llama.cpp on your mini-PC.")
    print("  Gyroscopic = same model with physics-guided attention:")
    print("    - KV chirality prefilter uses live M2/eta (entanglement entropy), not static LSH.")
    by_case: dict[str, dict[str, RunResult]] = {}
    for r in results:
        by_case.setdefault(r.case, {})[r.mode] = r
    for runs in by_case.values():
        s, g = runs.get("stock"), runs.get("gyroscopic")
        if s and g and s.status == "OK" and g.status == "OK":
            if s.gen_tps and g.gen_tps:
                pct = 100.0 * g.gen_tps / s.gen_tps
                print(f"  This run: gyroscopic decode is {pct:.0f}% of stock ({g.gen_tps:.1f} vs {s.gen_tps:.1f} tok/s).")
            break
    print("  Chat:  python -m src.tools.gyroscopic.helpers.run_bonsai")
    print("  Verify: python -m src.tools.gyroscopic.helpers.run_bonsai --verify")


def write_json(results: list[RunResult], meta: dict[str, Any]) -> None:
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
        description="Stock vs gyroscopic benchmark at KV-relevant scale.",
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

    print(
        f"[bench] {'stock+gyro' if run_stock and run_gyro else ('gyro' if run_gyro else 'stock')} "
        f"| {case.description} | n_ctx={args.n_ctx} n_predict={args.n_predict}",
        flush=True,
    )

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
            "stock", case, args.n_ctx, args.n_predict, args.timeout, verbose=args.verbose,
        ))
    if run_gyro:
        ensure("gyroscopic")
        results.append(run_llama(
            "gyroscopic", case, args.n_ctx, args.n_predict, args.timeout,
            live_log=args.live_log, verbose=args.verbose,
        ))

    meta = {
        "suite": args.suite,
        "suite_description": case.description,
        "n_ctx": args.n_ctx,
        "n_predict": args.n_predict,
        "timeout": args.timeout,
        "run_stock": run_stock,
        "run_gyro": run_gyro,
        "flash_attn": "on",
    }
    write_json(results, meta)
    print_report(results, meta, verbose=args.verbose)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
