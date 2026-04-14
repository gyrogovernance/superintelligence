"""Gyroscopic LLM & QuBEC Climate Benchmark (Hardened Streaming Runner).

Native GyroLabe notes (for maintainers):
- gyrolabe_block_info_t.dq_lattice_empty: set in pack_DQ_lattice when defect D=B-P is all
  zeros; gyrolabe_qubec_matmul_q8_0 skips k4_gemv64_avx2 in that case (see gyrolabe_registry.h).
- GyroPulse / GGML_GYROSCOPIC_TRACE_SNAPSHOT_EVERY: mid-run stderr snapshots when trace is on;
  footer GyroMatMul stats may be missing if the process is killed on timeout (Windows).
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO
import statistics

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

from src.tools.gyroscopic.config import get_gyroscopic_llm_config, repo_root
from src.tools.gyroscopic.loader import build_llama_cli_command
from src.tools.gyroscopic.ops_build import build_llama_cpp_if_needed

# === Robust Defaults (no CLI flags required) ===
# One llama prompt by default so a broken gyro path does not multiply wasted wall time.
PROMPTS = ["Hello"]
MULTI_CELL_MAX_CELLS = 8
# Distinct strings for native word4 batch only (not passed to llama-cli).
MULTI_CELL_NATIVE_LABELS = [f"cell-{i}" for i in range(MULTI_CELL_MAX_CELLS)]
DEFAULT_N_PREDICT = 4
# Cold GGUF load + CPU prompt eval often exceeds 120s on large models; override with --timeout.
TIMEOUT_STOCK = 600.0
TIMEOUT_GYRO = 1200.0
SILENT_KILL_SEC = 120.0  # Kill only if no stdout/stderr for this long (not wall-clock)

# Force CPU-safe flags. --no-context-shift prevents massive CPU reallocations.
# --reasoning off is required for Qwen 3.5 as noted.
EXTRA_ARGS = [
    "--seed", "42", "--temp", "0", "--top-p", "1.0", "--top-k", "1",
    "--reasoning", "off", "--single-turn",
    "--n-gpu-layers", "0", "--no-context-shift", "--flash-attn", "off"
]

DATA_DIR = repo_root() / "data" / "benchmarks" / "gyroscopic_llama"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = DATA_DIR / "bench.json"

# === Trace Parsing ===
_TRACE_STATS_RE = re.compile(
    r"^(GyroMatMul stats|GyroRows|GyroDispatch|GyroGraph):\s*(.*)$"
)
_GYRO_REG_RE = re.compile(
    r"GYRO_REG:\s*tensors_scanned=(\d+)\s+q8_0_tensors=(\d+)\s+registry_entries=(\d+)"
)
_GYRO_PULSE_RE = re.compile(
    r"^GyroPulse:\s*attempts=(\d+)\s+qubec_calls=(\d+)\s+dense_calls=(\d+)\s+scanned_blocks=(\d+)\s+reg=(\d+)\s*$"
)

@dataclass
class TraceStats:
    qubec_calls: int = 0
    dense_calls: int = 0
    structured_rows: int = 0
    dense_rows: int = 0
    attempt_rows: int = 0
    exact_witness: int = 0
    parity_mismatch: int = 0
    max_abs_row_error: float = 0.0
    dispatch_scanned: int = 0
    dispatch_k64_miss: int = 0
    registry_entries: int = 0
    graph_m2_mean: float | None = None
    graph_cells: int = 0
    # Emitted at end of model load (stderr); present even if GyroDispatch footer never runs (timeout).
    reg_load_tensors_scanned: int = 0
    reg_load_q8_0_tensors: int = 0
    reg_load_registry_entries: int = 0
    # Last GyroPulse line (stderr); written every N attempts if GGML_GYROSCOPIC_TRACE_SNAPSHOT_EVERY is set.
    gyro_pulse_attempts: int = 0
    gyro_pulse_qubec_calls: int = 0
    gyro_pulse_dense_calls: int = 0
    gyro_pulse_scanned_blocks: int = 0
    gyro_pulse_reg: int = 0

@dataclass
class RunResult:
    mode: str
    prompt_idx: int
    prompt: str
    elapsed: float
    timed_out: bool
    rc: int | None
    prompt_tps: float | None
    gen_tps: float | None
    gen_text: str
    trace: TraceStats
    stdout_hash: str
    silent_kill: bool = False

    @property
    def ok(self) -> bool:
        return not self.timed_out and not self.silent_kill and self.rc == 0

def _parse_trace(stderr_lines: list[str]) -> TraceStats:
    tr = TraceStats()
    for line in stderr_lines:
        s = line.strip()
        mreg = _GYRO_REG_RE.search(s)
        if mreg:
            tr.reg_load_tensors_scanned = int(mreg.group(1))
            tr.reg_load_q8_0_tensors = int(mreg.group(2))
            tr.reg_load_registry_entries = int(mreg.group(3))
            continue
        mp = _GYRO_PULSE_RE.match(s)
        if mp:
            tr.gyro_pulse_attempts = int(mp.group(1))
            tr.gyro_pulse_qubec_calls = int(mp.group(2))
            tr.gyro_pulse_dense_calls = int(mp.group(3))
            tr.gyro_pulse_scanned_blocks = int(mp.group(4))
            tr.gyro_pulse_reg = int(mp.group(5))
            continue
        m = _TRACE_STATS_RE.match(s)
        if not m: continue
        section, rest = m.group(1), m.group(2)
        kv = {p.split("=")[0]: p.split("=")[1].strip(",") for p in rest.split() if "=" in p}
        if section == "GyroMatMul stats":
            tr.qubec_calls = int(kv.get("qubec_calls", 0))
            tr.dense_calls = int(kv.get("dense_calls", 0))
        elif section == "GyroRows":
            tr.structured_rows = int(kv.get("structured_rows", 0))
            tr.dense_rows = int(kv.get("dense_rows", 0))
            tr.attempt_rows = int(kv.get("structured_attempt_rows", 0))
            tr.exact_witness = int(kv.get("exact_witness_rows", 0))
            tr.parity_mismatch = int(kv.get("parity_mismatch_rows", 0))
            tr.max_abs_row_error = float(kv.get("max_abs_row_error", 0.0))
        elif section == "GyroDispatch":
            tr.dispatch_scanned = int(kv.get("scanned_blocks", 0))
            tr.dispatch_k64_miss = int(kv.get("no_k64_blocks", 0))
            tr.registry_entries = int(kv.get("dispatch_entries", 0))
        elif section == "GyroGraph":
            tr.graph_m2_mean = float(kv.get("m2_mean", 0.0))
            tr.graph_cells = int(kv.get("cells", 0))
    return tr


_THROUGHPUT_BRACKET_RE = re.compile(
    r"\[\s*Prompt:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*t/s\s*\|\s*"
    r"Generation:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*t/s\s*\]"
)
_TPS_TAIL_RE = re.compile(
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+tokens per second"
)
_PERF_PROMPT_MS_RE = re.compile(
    r":\s+prompt\s+eval\s+time\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*ms\s*/\s*(\d+)\s+tokens\b",
    re.I,
)
_PERF_GEN_MS_RE = re.compile(
    r":\s+eval\s+time\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*ms\s*/\s*(\d+)\s+(?:runs|tokens)\b",
    re.I,
)


def _normalize_cli_log_text(text: str) -> str:
    if not text:
        return ""
    if text.startswith("\ufeff"):
        text = text[1:]
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_llama_throughput(stdout: str, stderr: str) -> tuple[float | None, float | None]:
    """
    llama_perf_context_print uses stderr (tokens per second). Some builds print
    a one-line [ Prompt: x t/s | Generation: y t/s ] summary on stdout.
    """
    combined = _normalize_cli_log_text(stdout) + "\n" + _normalize_cli_log_text(stderr)
    prompt_tps: float | None = None
    gen_tps: float | None = None

    m = _THROUGHPUT_BRACKET_RE.search(combined)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            pass

    for line in combined.splitlines():
        mp = _PERF_PROMPT_MS_RE.search(line)
        if mp:
            ms, n = float(mp.group(1)), max(int(mp.group(2)), 1)
            if ms > 0:
                prompt_tps = 1000.0 * n / ms
        mg = _PERF_GEN_MS_RE.search(line)
        if mg:
            ms, n = float(mg.group(1)), max(int(mg.group(2)), 1)
            if ms > 0:
                gen_tps = 1000.0 * n / ms

    if prompt_tps is None or gen_tps is None:
        for line in combined.splitlines():
            low = line.lower()
            if "prompt eval time" in low and "tokens per second" in low:
                m2 = _TPS_TAIL_RE.search(line)
                if m2:
                    try:
                        prompt_tps = float(m2.group(1))
                    except ValueError:
                        pass
            elif (
                "prompt eval time" not in low
                and "tokens per second" in low
                and _PERF_GEN_MS_RE.search(line)
            ):
                m2 = _TPS_TAIL_RE.search(line)
                if m2:
                    try:
                        gen_tps = float(m2.group(1))
                    except ValueError:
                        pass

    if prompt_tps is None and gen_tps is None:
        for line in combined.splitlines():
            if "Prompt:" in line and "t/s" in line and "Generation:" in line:
                try:
                    parts = line.replace("[", "").replace("]", "").split("|")
                    prompt_tps = float(parts[0].split()[-1])
                    gen_tps = float(parts[1].split()[-1])
                except Exception:
                    prompt_tps, gen_tps = None, None
                break

    return prompt_tps, gen_tps

def _extract_generation(stdout_lines: list[str]) -> str:
    collecting, gen = False, []
    for line in stdout_lines:
        s = line.strip()
        if not collecting and s.startswith(">"):
            collecting = True; continue
        if collecting and (s.startswith("[") or "t/s" in s.lower()):
            break
        if collecting: gen.append(s)
    return "\n".join(gen).strip()

def _pipe_line_reader(
    pipe: TextIO,
    lines_out: list[str],
    lock: threading.Lock,
    last_read: list[float],
) -> None:
    try:
        for raw in iter(pipe.readline, ""):
            if raw == "":
                break
            s = raw.rstrip("\r\n")
            with lock:
                lines_out.append(s)
            last_read[0] = time.perf_counter()
    except Exception:
        pass


# === Streaming Runner (wall-clock timeout even when readline would block) ===
def run_llama_streaming(
    mode: str,
    prompt: str,
    n_predict: int,
    timeout: float,
    idx: int,
    *,
    trace: bool = True,
) -> RunResult:
    cfg = get_gyroscopic_llm_config()
    env = os.environ.copy()
    env["GGML_GYROSCOPIC"] = "1" if mode == "gyroscopic" else "0"
    env["GGML_GYROSCOPIC_TRACE"] = "1" if (mode == "gyroscopic" and trace) else "0"
    
    args = build_llama_cli_command(cfg, prompt=prompt, n_predict=n_predict, extra_args=EXTRA_ARGS)
    
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        stdin=subprocess.DEVNULL,
    )
    stdout_io = proc.stdout
    stderr_io = proc.stderr
    if stdout_io is None or stderr_io is None:
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        raise RuntimeError("bench: subprocess must capture stdout and stderr")

    start = time.perf_counter()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    line_lock = threading.Lock()
    last_read = [start]
    timed_out, silent_kill = False, False

    th_out = threading.Thread(
        target=_pipe_line_reader,
        args=(stdout_io, stdout_lines, line_lock, last_read),
        daemon=True,
        name="bench-llama-stdout",
    )
    th_err = threading.Thread(
        target=_pipe_line_reader,
        args=(stderr_io, stderr_lines, line_lock, last_read),
        daemon=True,
        name="bench-llama-stderr",
    )
    th_out.start()
    th_err.start()

    while True:
        elapsed = time.perf_counter() - start
        if proc.poll() is not None:
            break
        if elapsed > timeout:
            timed_out = True
            break
        if elapsed - last_read[0] > SILENT_KILL_SEC:
            silent_kill = True
            break
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
    else:
        try:
            proc.wait(timeout=5)
        except Exception:
            pass

    th_out.join(timeout=12.0)
    th_err.join(timeout=12.0)

    out_text = "\n".join(stdout_lines)
    err_text = "\n".join(stderr_lines)
    prompt_tps, gen_tps = parse_llama_throughput(out_text, err_text)
    gen_text = _extract_generation(stdout_lines)

    return RunResult(
        mode=mode,
        prompt_idx=idx,
        prompt=prompt,
        elapsed=time.perf_counter() - start,
        timed_out=timed_out,
        rc=proc.returncode,
        prompt_tps=prompt_tps,
        gen_tps=gen_tps,
        gen_text=gen_text,
        trace=_parse_trace(stderr_lines),
        stdout_hash=hashlib.sha256(gen_text.encode()).hexdigest()[:12],
        silent_kill=silent_kill,
    )

# === Native Climate Probe ===
def run_kv_priority_illustrative() -> dict[str, Any]:
    """
    Illustrative climate ranking only (not wired to llama KV yet).
    Lower score => more eviction pressure in this toy ordering.
    """
    from src.tools.gyroscopic.climate import cell_climate_from_histograms

    pole = cell_climate_from_histograms(
        [64] + [0] * 63,
        [64] + [0] * 6,
        [16, 16, 16, 16],
    )
    equator = cell_climate_from_histograms(
        [1] * 64,
        [4, 6, 8, 12, 12, 12, 10],
        [16, 16, 16, 16],
    )

    def priority_row(c: dict[str, Any]) -> float:
        m2 = float(c["M2_empirical"])
        n_mean = float(c["N_mean"])
        return m2 * (1.0 + 2.0 * abs(n_mean - 3.0) / 3.0)

    pp = priority_row(pole)
    pe = priority_row(equator)
    return {
        "pole_priority": pp,
        "equator_priority": pe,
        "illustrative_eviction_pole_first": pp < pe,
    }


def run_multi_cell_word4_benchmark(prompts: list[str], profile: int = 0) -> dict[str, Any]:
    """
    Native multi-cell: one word4 ingest per prompt, then SLCP emit (no llama subprocess).
    Exercises batched GyroGraph buffers; not llama.cpp resonance batching.
    """
    try:
        from src.tools.gyroscopic.ops import (
            gyrograph_compute_m2_empirical,
            gyrograph_emit_slcp_batch,
            gyrograph_ingest_word4_batch_indexed,
            gyromatmul_runtime_caps,
        )
    except Exception as e:
        return {"error": str(e), "skipped": True}

    try:
        gyromatmul_runtime_caps()
    except Exception as e:
        return {"error": str(e), "skipped": True}

    n = len(prompts)
    if n == 0:
        return {"error": "empty prompts"}

    cell_ids = (ct.c_int64 * n)(*range(n))
    omega12 = (ct.c_int32 * n)(*(0 for _ in range(n)))
    step = (ct.c_uint64 * n)(*(0 for _ in range(n)))
    last_byte = (ct.c_uint8 * n)(*(0 for _ in range(n)))
    has_closed = (ct.c_uint8 * n)(*(0 for _ in range(n)))
    word4 = (ct.c_uint8 * (4 * n))(*(0 for _ in range(4 * n)))
    chi_ring = (ct.c_uint8 * (64 * n))(*(0 for _ in range(64 * n)))
    chi_pos = (ct.c_uint8 * n)(*(0 for _ in range(n)))
    chi_valid = (ct.c_uint8 * n)(*(0 for _ in range(n)))
    chi_hist = (ct.c_uint16 * (64 * n))(*(0 for _ in range(64 * n)))
    shell_hist = (ct.c_uint16 * (7 * n))(*(0 for _ in range(7 * n)))
    fam_ring = (ct.c_uint8 * (64 * n))(*(0 for _ in range(64 * n)))
    fam_hist = (ct.c_uint16 * (4 * n))(*(0 for _ in range(4 * n)))
    omega_sig = (ct.c_int32 * n)(*(0 for _ in range(n)))
    p_o = (ct.c_uint16 * n)(*(0 for _ in range(n)))
    p_e = (ct.c_uint16 * n)(*(0 for _ in range(n)))
    pbit = (ct.c_uint8 * n)(*(0 for _ in range(n)))
    words_in = (ct.c_uint8 * (4 * n))()
    for i, text in enumerate(prompts):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        for b in range(4):
            words_in[4 * i + b] = digest[b]
    rkey = (ct.c_uint32 * n)(*(0 for _ in range(n)))

    t0 = time.perf_counter()
    gyrograph_ingest_word4_batch_indexed(
        cell_ids,
        omega12,
        step,
        last_byte,
        has_closed,
        word4,
        chi_ring,
        chi_pos,
        chi_valid,
        chi_hist,
        shell_hist,
        fam_ring,
        fam_hist,
        omega_sig,
        p_o,
        p_e,
        pbit,
        words_in,
        rkey,
        profile & 0xFF,
        n,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    fam_u8 = (ct.c_uint8 * (4 * n))()
    for c in range(n):
        for g in range(4):
            v = int(fam_hist[4 * c + g])
            fam_u8[4 * c + g] = v if v < 256 else 255

    batch = gyrograph_emit_slcp_batch(
        n,
        cell_ids,
        omega12,
        step,
        last_byte,
        word4,
        chi_hist,
        shell_hist,
        fam_u8,
        omega_sig,
        p_o,
        p_e,
        pbit,
        rkey,
    )

    chi6_vals = [int(s.chi6) for s in batch]
    m2_vals: list[float] = []
    for c in range(n):
        row = (ct.c_uint16 * 64)()
        for i in range(64):
            row[i] = chi_hist[c * 64 + i]
        tot = sum(int(row[i]) for i in range(64))
        if tot > 0:
            m2_vals.append(float(gyrograph_compute_m2_empirical(row, tot)))

    uniq_chi6 = len(set(chi6_vals))
    grouping = (1.0 - uniq_chi6 / float(n)) if n else 0.0

    return {
        "cells": n,
        "ingest_elapsed_ms": elapsed_ms,
        "unique_chi6": uniq_chi6,
        "grouping_metric": grouping,
        "m2_empirical_mean": statistics.mean(m2_vals) if m2_vals else None,
        "m2_empirical_stdev": statistics.stdev(m2_vals) if len(m2_vals) > 1 else None,
        "steps_ok": all(int(batch[i].step) == 4 for i in range(n)),
    }


def run_qubec_climate_probe() -> dict[str, Any]:
    out: dict[str, Any] = {"status": "pass", "details": {}}
    try:
        from src.tools.gyroscopic.ops import (
            gyrolabe_analyze_operator_64, gyrolabe_chirality_evolve_n,
            gyrograph_emit_slcp_batch, gyromatmul_runtime_caps
        )
        gyromatmul_runtime_caps()
        hist, ens = [0]*64, [1]*64; hist[0] = 64
        chi = gyrolabe_chirality_evolve_n(hist, ens, 2)
        out["details"]["chirality_2step_uniform"] = sum(chi)==64 and max(chi)-min(chi)<=1
        
        import numpy as np
        kernel = np.array([(3 * i + 1) % 17 for i in range(64)], dtype=np.float32)
        wr = np.stack([np.roll(kernel, i) for i in range(64)])
        rep = gyrolabe_analyze_operator_64(wr, threshold=0.01)
        out["details"]["hybrid_routing_ready"] = int(rep.op_class) != 0
        out["details"]["scr"] = float(rep.scr)

        N = 16
        ids = (ct.c_int64 * N)(*range(N))
        omega12 = (ct.c_int32 * N)(*(0 for _ in range(N)))
        step = (ct.c_uint64 * N)(*(10+c for c in range(N)))
        last_byte = (ct.c_uint8 * N)(*(0xAA for _ in range(N)))
        word4 = (ct.c_uint8 * (4*N))(*(0 for _ in range(4*N)))
        chi_h = (ct.c_uint16 * (64*N))(*(0 for _ in range(64*N)))
        shell_h = (ct.c_uint16 * (7*N))(*(0 for _ in range(7*N)))
        fam_h = (ct.c_uint8 * (4*N))(*(0 for _ in range(4*N)))
        osig = (ct.c_int32 * N)(*(0 for _ in range(N)))
        pO = (ct.c_uint16 * N)(*(0 for _ in range(N)))
        pE = (ct.c_uint16 * N)(*(0 for _ in range(N)))
        pbit = (ct.c_uint8 * N)(*(0 for _ in range(N)))
        rkey = (ct.c_uint32 * N)(*(0 for _ in range(N)))
        for c in range(N):
            for j in range(64): chi_h[c*64+j] = (c*100+j)&0xFFFF
            shell_h[c*7+(c%7)] = 3+(c%4); fam_h[c*4+(c%4)] = 11+c
            
        batch = gyrograph_emit_slcp_batch(N, ids, omega12, step, last_byte, word4, chi_h, shell_h, fam_h, osig, pO, pE, pbit, rkey)
        out["details"]["multi_cell_slcp_pass"] = len(batch)==N and len({tuple(b.spectral64) for b in batch})>1
    except Exception as e:
        out["status"] = "fail"; out["error"] = str(e)
    return out

def print_full_report(
    results: list[RunResult],
    climate: dict[str, Any],
    kv_pri: dict[str, Any],
    multi_cell: dict[str, Any],
) -> None:
    """Single stdout report: llama runs, native probes, hybrid trace (no extra log files)."""
    w = 65
    print("\n" + "=" * w)
    print("GYROSCOPIC BENCHMARK REPORT")
    print("=" * w)

    print("\n--- llama-cli runs (stock vs gyroscopic) ---\n")
    print(f"{'Mode':<10} {'Prompt':<35} {'Time':>6} {'TPS':>6} {'Status':<8}")
    print("-" * w)
    stock_tps, gyro_tps = [], []
    for r in results:
        status = "OK" if r.ok else ("TIMEOUT" if r.timed_out else "SILENT_KILL" if r.silent_kill else "FAIL")
        if r.gen_tps is not None:
            tps_str = f"{r.gen_tps:.1f}"
        elif r.timed_out or r.silent_kill:
            tps_str = "--"
        else:
            tps_str = "n/a"
        print(f"{r.mode:<10} {r.prompt[:33]:<35} {r.elapsed:>5.1f}s {tps_str:>6} {status:<8}")
        if r.mode == "stock" and r.gen_tps is not None:
            stock_tps.append(r.gen_tps)
        if r.mode == "gyroscopic" and r.gen_tps is not None:
            gyro_tps.append(r.gen_tps)

    print("-" * w)
    if stock_tps and gyro_tps:
        ratio = statistics.mean(gyro_tps) / statistics.mean(stock_tps)
        print(f"Throughput Ratio (Gyro/Stock): {ratio:.3f}x")
    else:
        print("Throughput Ratio (Gyro/Stock): n/a (no completed run with parsed TPS)")

    g = next((r.trace for r in reversed(results) if r.mode == "gyroscopic"), None)
    if g and g.structured_rows + g.dense_rows > 0:
        total = g.structured_rows + g.dense_rows
        print(f"Hybrid Routing: {g.structured_rows}/{total} structured ({g.structured_rows/total*100:.1f}%)")
        print(f"Parity Mismatches: {g.parity_mismatch} (max err: {g.max_abs_row_error:.6f})")
        est_struct = g.structured_rows * 512
        est_dense = g.dense_rows * 16384
        est_tot = est_struct + est_dense
        dense_full = total * 16384
        if dense_full > 0:
            savings = 100.0 * (1.0 - est_tot / float(dense_full))
            print(
                f"Est. block read bytes (rough): {est_tot/1e6:.2f} MB vs {dense_full/1e6:.2f} MB all-dense "
                f"(~{savings:.1f}% vs upper-bound dense)"
            )

    g_load = next((r.trace for r in reversed(results) if r.mode == "gyroscopic"), None)
    if g_load is not None:
        print("\n--- gyro load registration (stderr GYRO_REG, survives timeout) ---\n")
        if g_load.reg_load_tensors_scanned > 0 or g_load.reg_load_registry_entries > 0:
            print(f"  tensors_scanned: {g_load.reg_load_tensors_scanned}")
            print(f"  q8_0_tensors: {g_load.reg_load_q8_0_tensors}")
            print(f"  registry_entries: {g_load.reg_load_registry_entries}")
        else:
            print("  (no GYRO_REG line parsed; rebuild llama.cpp after gyro patch or check stderr capture)")

    g_pulse = next((r.trace for r in reversed(results) if r.mode == "gyroscopic"), None)
    if g_pulse is not None and g_pulse.gyro_pulse_attempts > 0:
        print("\n--- last GyroPulse (mid-run trace; survives timeout) ---\n")
        print(
            f"  attempts={g_pulse.gyro_pulse_attempts} qubec_calls={g_pulse.gyro_pulse_qubec_calls} "
            f"dense_calls={g_pulse.gyro_pulse_dense_calls} scanned_blocks={g_pulse.gyro_pulse_scanned_blocks} "
            f"reg={g_pulse.gyro_pulse_reg}"
        )

    print("\n--- native climate probe ---\n")
    print(f"status: {climate.get('status', '?')}")
    if climate.get("error"):
        print(f"error: {climate['error']}")
    det = climate.get("details") or {}
    for k in sorted(det.keys()):
        print(f"  {k}: {det[k]}")

    print("\n--- KV priority (illustrative, not llama KV) ---\n")
    for k in sorted(kv_pri.keys()):
        print(f"  {k}: {kv_pri[k]}")

    print("\n--- multi-cell GyroGraph (word4 ingest + SLCP, no llama) ---\n")
    if multi_cell.get("skipped"):
        print(f"  skipped: {multi_cell.get('error', 'unknown')}")
    else:
        for k in (
            "cells",
            "ingest_elapsed_ms",
            "unique_chi6",
            "grouping_metric",
            "m2_empirical_mean",
            "m2_empirical_stdev",
            "steps_ok",
        ):
            if k in multi_cell:
                print(f"  {k}: {multi_cell[k]}")
        if multi_cell.get("error") and not multi_cell.get("skipped"):
            print(f"  error: {multi_cell['error']}")

    print("\n" + "=" * w)

def main() -> int:
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Default: one llama prompt, no warmup, native climate + multi-cell probes; "
            f"writes only {OUT_JSON.relative_to(_REPO_ROOT)}."
        )
    )
    dbg = p.add_argument_group("debug only (optional)")
    dbg.add_argument(
        "--stock-only",
        action="store_true",
        help="Stock llama-cli only (skip gyroscopic passes)",
    )
    dbg.add_argument("--timeout", type=float, default=None, help="Override TIMEOUT_STOCK and TIMEOUT_GYRO")
    dbg.add_argument(
        "--n-predict",
        type=int,
        default=None,
        metavar="N",
        help=f"Tokens to generate (default {DEFAULT_N_PREDICT})",
    )
    dbg.add_argument(
        "--warmup",
        action="store_true",
        help="Run short llama-cli warmup (stock + gyro) before measured runs",
    )
    dbg.add_argument(
        "--extra-prompts",
        action="store_true",
        help="Add two longer prompts after the default (3 llama-cli pairs total)",
    )
    dbg.add_argument(
        "--diag",
        action="store_true",
        help=(
            "Shorter run + periodic GyroPulse on stderr: sets GGML_GYROSCOPIC_TRACE_SNAPSHOT_EVERY=25, "
            "caps n_predict and timeouts unless you pass --n-predict / --timeout explicitly."
        ),
    )
    dbg.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable gyroscopic trace counters (faster, but no hot-path counter visibility).",
    )
    dbg.add_argument(
        "--omp1",
        action="store_true",
        help="Set OMP_NUM_THREADS=1 for llama-cli (test OpenMP oversubscription vs ggml thread pool).",
    )
    args = p.parse_args()

    print("[bench] Building native surfaces if needed...")
    build_llama_cpp_if_needed()

    prompts = list(PROMPTS)
    if args.extra_prompts:
        prompts.extend(
            [
                "Explain exact hybrid preservation in one sentence.",
                "List three properties of the QuBEC climate manifold.",
            ]
        )
    timeout_stock = float(args.timeout) if args.timeout is not None else TIMEOUT_STOCK
    timeout_gyro = float(args.timeout) if args.timeout is not None else TIMEOUT_GYRO
    n_predict = int(args.n_predict) if args.n_predict is not None else DEFAULT_N_PREDICT
    trace_enabled = not args.no_trace
    if args.diag or trace_enabled:
        os.environ.setdefault("GGML_GYROSCOPIC_TRACE_SNAPSHOT_EVERY", "25")
    if args.diag:
        os.environ["GGML_GYROSCOPIC_TRACE_SNAPSHOT_EVERY"] = "25"
        if args.n_predict is None:
            n_predict = min(n_predict, 4)
    if args.omp1:
        os.environ["OMP_NUM_THREADS"] = "1"
    results: list[RunResult] = []

    run_gyro = not args.stock_only
    mode = "stock + gyro" if run_gyro else "stock only"
    print(
        f"[bench] {mode} | {len(prompts)} prompt(s) | n_predict={n_predict} | "
        f"timeouts stock={timeout_stock:.0f}s gyro={timeout_gyro:.0f}s | "
        f"silent_kill={SILENT_KILL_SEC:.0f}s | warmup={'on' if args.warmup else 'off'}"
        f"{' | diag=on (GyroPulse every 25 attempts)' if args.diag else ''}"
        f"{' | trace=on' if trace_enabled else ' | trace=off'}"
        f"{' | OMP=1' if args.omp1 else ''}"
    )

    if args.warmup and prompts:
        pr0 = prompts[0]
        w_n = max(1, min(8, n_predict))
        w_to = min(300.0, timeout_stock, timeout_gyro)
        print(f"[bench] Warmup stock ({w_n} tok, timeout {w_to:.0f}s)...")
        _ = run_llama_streaming("stock", pr0, w_n, w_to, -1, trace=trace_enabled)
        if run_gyro:
            print(f"[bench] Warmup gyroscopic ({w_n} tok, timeout {w_to:.0f}s)...")
            _ = run_llama_streaming("gyroscopic", pr0, w_n, w_to, -1, trace=trace_enabled)
        time.sleep(0.05)

    for i, pr in enumerate(prompts):
        r_s = run_llama_streaming("stock", pr, n_predict, timeout_stock, i, trace=trace_enabled)
        results.append(r_s)
        tps_s = f"{r_s.gen_tps:.1f}" if r_s.gen_tps is not None else "--"
        print(f"  Stock {i+1}/{len(prompts)}: {r_s.elapsed:.1f}s | TPS:{tps_s} | {r_s.stdout_hash}")

        if run_gyro:
            r_g = run_llama_streaming("gyroscopic", pr, n_predict, timeout_gyro, i, trace=trace_enabled)
            results.append(r_g)
            status = "OK" if r_g.ok else ("TIMEOUT" if r_g.timed_out else "KILLED")
            tps_g = f"{r_g.gen_tps:.1f}" if r_g.gen_tps is not None else "--"
            print(f"  Gyro  {i+1}/{len(prompts)}: {r_g.elapsed:.1f}s | TPS:{tps_g} | {status} | {r_g.stdout_hash}")
            if r_g.silent_kill:
                print(
                    "  [diag] No stdout/stderr progress: possible hang or blocked pipe; "
                    "check llama-cli / Gyro backend."
                )
            elif r_g.timed_out:
                print(
                    "  [diag] Wall-clock timeout (not silent I/O stall). "
                    "Raise TIMEOUT_STOCK/TIMEOUT_GYRO in this script or use debug --timeout."
                )
            if r_s.stdout_hash != r_g.stdout_hash:
                print(
                    f"  [bench] Parity: stdout hash differs "
                    f"(stock={r_s.stdout_hash} gyro={r_g.stdout_hash})"
                )

    if args.stock_only:
        print("[bench] Gyroscopic passes skipped (--stock-only).")

    climate = run_qubec_climate_probe()
    kv_pri = run_kv_priority_illustrative()
    multi_cell = run_multi_cell_word4_benchmark(MULTI_CELL_NATIVE_LABELS)

    print_full_report(results, climate, kv_pri, multi_cell)
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runs": [asdict(r) for r in results],
        "climate_probe": climate,
        "kv_priority_illustrative": kv_pri,
        "multi_cell_word4": multi_cell,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[bench] Exported: {OUT_JSON}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())